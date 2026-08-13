//! Source for Zarr data
use core::fmt;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::Arc;
use std::vec;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use martin_tile_utils::{Format, TileCoord, TileData, TileInfo};
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
use crate::tiles::{MartinCoreResult, Source, UrlQuery};

/// Currently no pyramids - so full zoom range
const MIN_ZOOM: u8 = 0;
const MAX_ZOOM: u8 = 23;

/// Tile source that reads from `Zarr` stores
#[derive(Clone)]
pub struct ZarrSource<T: ObjectStore> {
    id: String,
    tilejson: TileJSON,
    tileinfo: TileInfo,
    min_zoom: u8,
    max_zoom: u8,
    x_coords: Arc<Array<AsyncObjectStore<T>>>,
    y_coords: Arc<Array<AsyncObjectStore<T>>>,
    time_coords: Option<Arc<Array<AsyncObjectStore<T>>>>,
    data_vars: Arc<HashMap<String, Array<AsyncObjectStore<T>>>>,
    src_crs: String,
    src_transform: [f64; 6],
    cache_zoom: CacheZoomRange,
}

impl<T: ObjectStore> Debug for ZarrSource<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ZarrTileSource")
            .field("id", &self.id)
            .finish()
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

        let local_store = LocalFileSystem::new_with_prefix(path)
            .map_err(|e| ZarrError::OjbectStoreError(e, path.clone()))?;
        let zarr_store = Arc::new(AsyncObjectStore::new(local_store));

        let x_coords = Array::async_open(zarr_store.clone(), "/x")
            .await
            .map_err(|e| ZarrError::ArrayCreateError(e))?;
        let y_coords = Array::async_open(zarr_store.clone(), "/y")
            .await
            .map_err(|e| ZarrError::ArrayCreateError(e))?;
        let time_coords = if let Some(coords) = time_coords(zarr_store.clone()).await? {
            Some(Arc::new(coords))
        } else {
            None
        };

        let mut data_vars = HashMap::new();
        let node_paths = data_variables(zarr_store.clone()).await?;
        for node_path in node_paths {
            let array = Array::async_open(zarr_store.clone(), node_path.as_str())
                .await
                .map_err(|e| ZarrError::ArrayCreateError(e))?;
            data_vars.insert(node_path.as_str().into(), array);
        }

        let src_crs = get_proj_code(zarr_store.clone()).await?;
        let src_transform = get_spatial_transform(zarr_store.clone()).await?;
        let src_bbox = get_bbox(zarr_store.clone()).await?;

        let tilejson = tilejson! {
            tiles: vec![],
            minzoom: MIN_ZOOM,
            maxzoom: MAX_ZOOM,
        };

        Ok(ZarrSource {
            id,
            tilejson,
            tileinfo,
            min_zoom: MIN_ZOOM,
            max_zoom: MAX_ZOOM,
            x_coords: Arc::new(x_coords),
            y_coords: Arc::new(y_coords),
            time_coords,
            data_vars: data_vars.into(),
            src_crs,
            src_transform,
            cache_zoom,
        })
    }
}

#[derive(Debug, Default)]
struct QueryParams<'a> {
    date_time: Option<DateTime<Utc>>,
    data_var: Option<&'a str>,
}

#[async_trait]
impl<T: ObjectStore> Source for ZarrSource<T> {
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
                            .unwrap()
                            .with_timezone(&Utc);
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
            let data_var = self.data_vars.get(data_var).unwrap();
            let datetime = DateTime::parse_from_rfc3339(time_str).unwrap();
            let datetime_utc: DateTime<Utc> = datetime.with_timezone(&Utc);
            let data = sample_data_var(
                &xyz,
                &self.x_coords,
                &self.y_coords,
                &self.time_coords,
                data_var,
                &self.src_crs,
                datetime_utc,
            )
            .unwrap();
            return Ok(data);
        }

        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {}
