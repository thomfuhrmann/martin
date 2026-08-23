use std::collections::HashMap;
use std::fmt::Debug;
use std::path::PathBuf;
use std::str::FromStr as _;
use std::sync::Arc;
use std::time::Duration;

use aws_credential_types::provider::{ProvideCredentials as _, SharedCredentialsProvider};
use martin_config_macros::CollectUnrecognizedKeys;
use martin_core::tiles::BoxedSource;
use martin_core::tiles::zarr::cache::WarpCache;
use martin_core::tiles::zarr::source::ZarrSource;
use object_store::aws::{AmazonS3Builder, AwsCredential, AwsCredentialProvider};
use object_store::{CredentialProvider, ObjectStore, ObjectStoreScheme};
use serde::{Deserialize, Serialize};
use tracing::{trace, warn};
use url::Url;

use crate::MartinResult;
use crate::config::file::{
    CachePolicy, CacheSizeConfig, ConfigFileError, ConfigFileResult, ConfigurationLivecycleHooks,
    TileSourceConfiguration, UnrecognizedValues,
};

pub const DEFAULT_RELOAD_INTERVAL: Duration = Duration::from_mins(10);

fn default_reload_interval() -> Duration {
    DEFAULT_RELOAD_INTERVAL
}

fn is_default_reload_interval(v: &Duration) -> bool {
    *v == DEFAULT_RELOAD_INTERVAL
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, CollectUnrecognizedKeys)]
#[cfg_attr(feature = "unstable-schemas", derive(schemars::JsonSchema))]
pub struct ZarrConfig {
    /// Size of the warp-cache cache (in MB).
    /// Defaults to `cache.size_mb` / 4
    ///
    /// Note:
    /// Tile and warp-cache caching are complementary.
    /// For good performance, you want
    /// - warp-cache caching (to not calculate the warp-grid on each request) and
    /// - tile caching (for high access tiles)
    ///
    /// Use `warp_cache: disable` to disable
    #[serde(default, skip_serializing_if = "CacheSizeConfig::is_empty")]
    #[cfg_attr(
        feature = "unstable-schemas",
        schemars(with = "crate::config::file::CacheSizeConfigShape")
    )]
    pub warp_cache_config: CacheSizeConfig,

    /// How often remote URL prefixes (`s3://bucket/`, `gs://bucket/`, etc.) re-`LIST` for source discovery.
    /// Has no effect on local directories, which are watched via filesystem events.
    ///
    /// Supports human-readable formats: "10m", "1h", "30s".
    /// Defaults to "10m". Set to "0s" to disable remote polling.
    #[serde(
        default = "default_reload_interval",
        skip_serializing_if = "is_default_reload_interval",
        with = "humantime_serde"
    )]
    #[cfg_attr(
        feature = "unstable-schemas",
        schemars(with = "String", example = &"10m")
    )]
    pub reload_interval: Duration,

    /// AWS SDK profile used for S3 credentials and region resolution.
    #[serde(
        default,
        alias = "aws_profile",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "unstable-schemas", schemars(skip))]
    pub profile: Option<String>,

    // if the key is in the allowed set, we assume it is there for a purpose
    // settings and unreconginsed values are partitioned from each other in the init_parsing step
    #[serde(skip)]
    #[cfg_attr(feature = "unstable-schemas", schemars(skip))]
    pub options: HashMap<String, String>,

    #[serde(flatten, skip_serializing)]
    #[cfg_attr(feature = "unstable-schemas", schemars(skip))]
    pub unrecognized: UnrecognizedValues,

    /// `Zarr` warp-grid cache (internal state, not serialized)
    #[serde(skip)]
    #[cfg_attr(feature = "unstable-schemas", schemars(skip))]
    pub warp_cache: WarpCache,

    #[serde(skip)]
    #[cfg_attr(feature = "unstable-schemas", schemars(skip))]
    pub aws_credentials: Option<AwsCredentialProvider>,

    #[cfg(test)]
    #[serde(skip)]
    #[cfg_attr(feature = "unstable-schemas", schemars(skip))]
    pub(crate) aws_profile_files: Option<EnvConfigFiles>,
}

impl Default for ZarrConfig {
    fn default() -> Self {
        Self {
            warp_cache_config: CacheSizeConfig::default(),
            reload_interval: DEFAULT_RELOAD_INTERVAL,
            profile: None,
            options: HashMap::default(),
            unrecognized: UnrecognizedValues::default(),
            warp_cache: WarpCache::default(),
            aws_credentials: None,
            #[cfg(test)]
            aws_profile_files: None,
        }
    }
}

impl PartialEq for ZarrConfig {
    fn eq(&self, other: &Self) -> bool {
        self.warp_cache_config == other.warp_cache_config
            && self.reload_interval == other.reload_interval
            && self.profile == other.profile
            && self.options == other.options
            && self.unrecognized == other.unrecognized
    }
}

impl ConfigurationLivecycleHooks for ZarrConfig {
    async fn finalize(&mut self) -> ConfigFileResult<()> {
        // if the key is in the allowed set, we assume it is there for a purpose
        // because of how serde(flatten) works, we need to collect all in one place and then
        // partition them into options and unrecognized keys
        //
        // If we don't do this, the error message is not clear enough
        self.partition_options_and_unrecognized();
        self.load_aws_profile().await;

        Ok(())
    }
}

impl ZarrConfig {
    async fn load_aws_profile(&mut self) {
        let Some(profile) = self.profile.clone() else {
            return;
        };

        let loader = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .profile_name(profile.clone());

        #[cfg(test)]
        let loader = if let Some(files) = &self.aws_profile_files {
            let region_provider = ProfileFileRegionProvider::builder()
                .profile_name(profile)
                .profile_files(files.clone())
                .build();
            loader.profile_files(files.clone()).region(region_provider)
        } else {
            loader
        };

        let sdk_config = loader.load().await;

        self.apply_aws_config(&sdk_config);
    }

    fn apply_aws_config(&mut self, sdk_config: &aws_config::SdkConfig) {
        let region_specified_by_config = [
            "region",
            "aws_region",
            "default_region",
            "aws_default_region",
        ]
        .iter()
        .any(|key| self.options.contains_key(*key));

        if region_specified_by_config {
            warn!("Region from zarr.profile is ignored in favor of explicit region configuration.");
        } else if let Some(region) = sdk_config.region() {
            self.options
                .insert("region".to_owned(), region.as_ref().to_owned());
        }

        let has_explicit_credentials = [
            "access_key_id",
            "aws_access_key_id",
            "secret_access_key",
            "aws_secret_access_key",
            "session_token",
            "aws_session_token",
            "token",
            "aws_token",
            "web_identity_token_file",
            "aws_web_identity_token_file",
            "role_arn",
            "aws_role_arn",
            "role_session_name",
            "aws_role_session_name",
            "container_credentials_relative_uri",
            "aws_container_credentials_relative_uri",
            "container_credentials_full_uri",
            "aws_container_credentials_full_uri",
            "container_authorization_token_file",
            "aws_container_authorization_token_file",
            "metadata_endpoint",
            "aws_metadata_endpoint",
            "imdsv1_fallback",
            "aws_imdsv1_fallback",
            "endpoint_url_sts",
            "aws_endpoint_url_sts",
        ]
        .iter()
        .any(|key| self.options.contains_key(*key));

        let skips_signature = ["skip_signature", "aws_skip_signature"].iter().any(|key| {
            self.options
                .get(*key)
                .is_some_and(|value| value.eq_ignore_ascii_case("true") || value == "1")
        });

        if has_explicit_credentials {
            warn!(
                "Credentials from zarr.profile are ignored in favor of explicit credential-provider configuration."
            );
        } else if skips_signature {
            warn!("Credentials from zarr.profile are ignored because request signing is disabled.");
        } else if let Some(provider) = sdk_config.credentials_provider() {
            self.aws_credentials = Some(Arc::new(AwsSdkCredentialProvider {
                provider: provider.clone(),
            }));
        }
    }

    pub(crate) fn parse_url_opts(
        &self,
        url: &Url,
    ) -> object_store::Result<(Box<dyn ObjectStore>, object_store::path::Path)> {
        let (scheme, path) = ObjectStoreScheme::parse(url)?;

        if scheme != ObjectStoreScheme::AmazonS3 {
            return object_store::parse_url_opts(url, &self.options);
        }

        let mut builder = self.options.iter().fold(
            AmazonS3Builder::new().with_url(url.to_string()),
            |builder, (key, value)| match key.parse() {
                Ok(key) => builder.with_config(key, value),
                Err(_) => builder,
            },
        );

        if let Some(credentials) = &self.aws_credentials {
            builder = builder.with_credentials(Arc::clone(credentials));
        }

        Ok((Box::new(builder.build()?), path))
    }

    /// Partition options and unrecognized keys
    fn partition_options_and_unrecognized(&mut self) {
        for (key, value) in self.unrecognized.clone() {
            let key_could_configure_object_store =
                object_store::aws::AmazonS3ConfigKey::from_str(key.as_str()).is_ok()
                    || object_store::gcp::GoogleConfigKey::from_str(key.as_str()).is_ok()
                    || object_store::azure::AzureConfigKey::from_str(key.as_str()).is_ok()
                    || object_store::client::ClientConfigKey::from_str(key.as_str()).is_ok();
            if key_could_configure_object_store {
                self.unrecognized
                    .remove(&key)
                    .expect("key should exist in the hashmap");
                // a hashmap cannot contain duplicate keys => ignore the replaced value
                let _ = match value {
                    serde_json::Value::Bool(b) => self.options.insert(key.clone(), b.to_string()),
                    serde_json::Value::Number(n) => self.options.insert(key.clone(), n.to_string()),
                    serde_json::Value::String(s) => self.options.insert(key.clone(), s.clone()),
                    v => {
                        // warn early with better context
                        warn!(
                            "Ignoring unrecognized configuration key '{key}': {v:?}. Only boolean, string or number values are allowed here. Please check your configuration file for typos."
                        );
                        None
                    }
                };
            }
        }
    }
}

impl TileSourceConfiguration for ZarrConfig {
    fn parse_urls() -> bool {
        true
    }

    async fn new_sources(
        &self,
        id: String,
        path: PathBuf,
        cache: CachePolicy,
    ) -> MartinResult<BoxedSource> {
        // canonicalize to resolve symlinks
        let path = path
            .canonicalize()
            .map_err(|e| ConfigFileError::IoError(e, path))?;
        // path->url conversion requires absolute path, otherwise it errors
        let path = std::path::absolute(&path).map_err(|e| ConfigFileError::IoError(e, path))?;
        // windows needs unix style paths, I.e. replace backslashes with forward slashes
        // a simple "add file://" does not work on windows
        // example: C:\Users\martin\Documents\pmtiles -> file://C:/Users/martin/Documents/pmtiles
        let url = Url::from_file_path(&path)
            .or(Err(ConfigFileError::PathNotConvertibleToUrl(path.clone())))?;
        trace!(
            "Zarr source {id} ({}) will be loaded as {url}",
            path.display()
        );
        self.new_sources_url(id, url, cache).await
    }

    async fn new_sources_url(
        &self,
        id: String,
        url: Url,
        cache: CachePolicy,
    ) -> MartinResult<BoxedSource> {
        let (store, _) = self
            .parse_url_opts(&url)
            .map_err(|e| ConfigFileError::ObjectStoreUrlParsing(e, id.clone()))?;
        let warp_cache = self.warp_cache.clone();
        let store: Arc<dyn ObjectStore> = store.into();
        let source = ZarrSource::new(id, store, warp_cache, cache.zoom()).await?;
        Ok(Box::new(source))
    }
}

#[derive(Debug)]
pub struct AwsSdkCredentialProvider {
    provider: SharedCredentialsProvider,
}

#[async_trait::async_trait]
impl CredentialProvider for AwsSdkCredentialProvider {
    type Credential = AwsCredential;

    async fn get_credential(&self) -> object_store::Result<Arc<Self::Credential>> {
        let credentials = self
            .provider
            .provide_credentials()
            .await
            .map_err(|source| object_store::Error::Generic {
                store: "S3",
                source: Box::new(source),
            })?;

        Ok(Arc::new(AwsCredential {
            key_id: credentials.access_key_id().to_owned(),
            secret_key: credentials.secret_access_key().to_owned(),
            token: credentials.session_token().map(str::to_owned),
        }))
    }
}
