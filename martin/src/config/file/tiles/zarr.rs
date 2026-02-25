use std::fmt::Debug;
use std::path::PathBuf;

use martin_core::config::IdResolver;
use martin_core::tiles::BoxedSource;
use martin_core::tiles::zarr::source::ZarrSource;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::config::file::{
    ConfigFileError, ConfigurationLivecycleHooks, FileConfigEnum, TileSourceConfiguration,
    TileSourceWarning, UnrecognizedKeys, UnrecognizedValues,
};
use crate::{MartinError, MartinResult};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ZarrConfig {
    #[serde(flatten, skip_serializing)]
    pub unrecognized: UnrecognizedValues,
}

impl ZarrConfig {
    pub async fn resolve(
        config: &mut FileConfigEnum<ZarrConfig>,
        idr: IdResolver,
    ) -> Result<(Vec<BoxedSource>, Vec<TileSourceWarning>), MartinError> {
        let Some(cfg) = config.extract_file_config() else {
            return Ok((vec![], vec![]));
        };
        // Return base directory only
        let mut sources = vec![];
        let warnings = vec![];
        for path in cfg.paths {
            let can = path
                .canonicalize()
                .map_err(|e| ConfigFileError::IoError(e, path.clone()))?;
            let id = path.file_stem().map_or_else(
                || "_unknown".to_string(),
                |s| s.to_string_lossy().to_string(),
            );
            let id = idr.resolve(&id, can.to_string_lossy().to_string());
            let source = cfg.custom.new_sources(id, path).await?;
            sources.push(source);
        }

        Ok((sources, warnings))
    }
}

impl ConfigurationLivecycleHooks for ZarrConfig {
    fn get_unrecognized_keys(&self) -> UnrecognizedKeys {
        self.unrecognized.keys().cloned().collect()
    }
}

impl TileSourceConfiguration for ZarrConfig {
    fn parse_urls() -> bool {
        false
    }

    async fn new_sources(&self, id: String, path: PathBuf) -> MartinResult<BoxedSource> {
        let zarr_source = ZarrSource::new(id, path)?;
        Ok(Box::new(zarr_source))
    }

    async fn new_sources_url(&self, _id: String, _url: Url) -> MartinResult<BoxedSource> {
        unreachable!()
    }
}
