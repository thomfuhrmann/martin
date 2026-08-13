//! Source for Zarr data
use core::fmt;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::vec;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use martin_tile_utils::{Format, TileCoord, TileData, TileInfo};
use moka::future::Cache;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use tilejson::{TileJSON, tilejson};
use tracing::info;
use zarrs::array::Array;
use zarrs_object_store::AsyncObjectStore;

use crate::CacheZoomRange;
use crate::tiles::zarr::error::ZarrError;
use crate::tiles::zarr::utils::{
    data_variables, get_bbox, get_proj_code, get_spatial_transform, sample_data_var, time_coords,
};
use crate::tiles::{MartinCoreError, MartinCoreResult, Source, UrlQuery};

/// Currently no pyramids - therefore full zoom range
const MIN_ZOOM: u8 = 0;
const MAX_ZOOM: u8 = 23;

#[derive(Debug, PartialEq, Eq, Hash)]
struct WarpCacheKey(TileCoord);

/// Warp cache that stores transformation from tile pixel grid to source pixel grid
#[derive(Debug, Clone)]
pub struct WarpCache {
    cache: Cache<WarpCacheKey, [u32; 1024]>,
}

impl WarpCache {
    /// Creates a new warp cache
    pub fn new(
        max_size_bytes: u64,
        expiry: Option<Duration>,
        idle_timeout: Option<Duration>,
    ) -> Self {
        let mut builder = Cache::builder()
            .name("zarr_warp_cache")
            .weigher(|_key: &WarpCacheKey, value: &[u32; 1024]| {
                value
                    .len()
                    .saturating_mul(size_of::<u32>())
                    .try_into()
                    .unwrap_or(u32::MAX)
            })
            .max_capacity(max_size_bytes);

        if let Some(ttl) = expiry {
            builder = builder.time_to_live(ttl);
        }

        if let Some(tti) = idle_timeout {
            builder = builder.time_to_idle(tti);
        }

        Self {
            cache: builder.build(),
        }
    }
}

/// Tile source that reads from `Zarr` stores
#[derive(Clone)]
pub struct ZarrSource<T: ObjectStore> {
    id: String,
    tilejson: TileJSON,
    tileinfo: TileInfo,
    min_zoom: u8,
    max_zoom: u8,
    time_coords: Option<Arc<Array<AsyncObjectStore<T>>>>,
    data_vars: Arc<HashMap<String, Array<AsyncObjectStore<T>>>>,
    src_crs: String,
    src_bbox: [f64; 4],
    src_transform: [f64; 6],
    cache_zoom: CacheZoomRange,
    warp_cache: WarpCache,
}

impl<T: ObjectStore> Debug for ZarrSource<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ZarrTileSource")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl ZarrSource<LocalFileSystem> {
    /// Creates a new Zarr tile source from a file path
    pub async fn async_new(
        id: String,
        path: PathBuf,
        cache_zoom: CacheZoomRange,
    ) -> Result<Self, ZarrError> {
        let tileinfo = TileInfo::new(
            Format::OctetStream,
            martin_tile_utils::Encoding::Uncompressed,
        );

        let local_store = LocalFileSystem::new_with_prefix(path.clone())
            .map_err(|e| ZarrError::OjbectStoreError(e, path))?;
        let zarr_store = Arc::new(AsyncObjectStore::new(local_store));

        let time_coords = time_coords(Arc::clone(&zarr_store)).await?.map(Arc::new);

        let mut data_vars = HashMap::new();
        let node_paths = data_variables(Arc::clone(&zarr_store)).await?;
        for node_path in node_paths {
            let array = Array::async_open(Arc::clone(&zarr_store), node_path.as_str())
                .await
                .map_err(ZarrError::ArrayCreateError)?;
            data_vars.insert(node_path.as_str().into(), array);
        }

        let src_crs = get_proj_code(Arc::clone(&zarr_store)).await?;
        let src_transform = get_spatial_transform(Arc::clone(&zarr_store)).await?;
        let src_bbox = get_bbox(Arc::clone(&zarr_store)).await?;

        let tilejson = tilejson! {
            tiles: vec![],
            minzoom: MIN_ZOOM,
            maxzoom: MAX_ZOOM,
        };

        let warp_cache = WarpCache::new(0, None, None);

        Ok(Self {
            id,
            tilejson,
            tileinfo,
            min_zoom: MIN_ZOOM,
            max_zoom: MAX_ZOOM,
            time_coords,
            data_vars: data_vars.into(),
            src_crs,
            src_bbox,
            src_transform,
            cache_zoom,
            warp_cache,
        })
    }
}

#[derive(Debug, Default)]
struct QueryParams<'a> {
    date_time: Option<DateTime<Utc>>,
    data_var: Option<&'a str>,
}

#[async_trait]
impl<T: ObjectStore + Clone> Source for ZarrSource<T> {
    fn get_id(&self) -> &str {
        &self.id
    }

    fn get_tilejson(&self) -> &TileJSON {
        &self.tilejson
    }

    fn get_tile_info(&self) -> TileInfo {
        self.tileinfo
    }

    fn clone_source(&self) -> Box<dyn Source> {
        Box::new(self.clone())
    }

    // Query parameters are required to pass time and data variable values
    fn support_url_query(&self) -> bool {
        true
    }

    /// Zoom-level bounds for tile caching.
    fn cache_zoom(&self) -> CacheZoomRange {
        self.cache_zoom
    }

    async fn get_tile(
        &self,
        xyz: TileCoord,
        url_query: Option<&UrlQuery>,
    ) -> MartinCoreResult<TileData> {
        let mut query_params = QueryParams::default();
        if let Some(url_query) = url_query {
            for (key, value) in url_query {
                match key.as_str() {
                    "time" => {
                        let date_time = DateTime::parse_from_rfc3339(value)
                            .map(|dt| dt.with_timezone(&Utc))
                            .map_err(ZarrError::ParseError)
                            .map_err(MartinCoreError::ZarrError)?;
                        query_params.date_time = Some(date_time);
                    }
                    "data_var" => query_params.data_var = Some(value.as_str()),
                    _ => info!("Query string not supported"),
                }
            }
        }

        let time_str = "2026-08-13T00:00:00.000Z";
        let data_var = "/snow_depth";
        if xyz.z >= self.min_zoom && xyz.z <= self.max_zoom {
            let data_var = self.data_vars.get(data_var).expect("msg");
            let datetime = DateTime::parse_from_rfc3339(time_str).expect("msg");
            let datetime_utc: DateTime<Utc> = datetime.with_timezone(&Utc);
            let data = sample_data_var(
                &xyz,
                &self.src_crs,
                self.src_transform,
                self.src_bbox,
                self.time_coords.clone(),
                datetime_utc,
                data_var,
            )
            .await
            .expect("could not sample data");
            return Ok(data);
        }

        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {}
