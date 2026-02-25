//! Source for Zarr data
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::Arc;
use std::vec;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use martin_tile_utils::{Format, TileCoord, TileData, TileInfo};
use tilejson::{TileJSON, tilejson};
use tracing::info;
use zarrs::array::Array;

use crate::tiles::zarr::error::ZarrError;
use crate::tiles::zarr::utils::{enumerate_data_variables, get_wkt_string};
use crate::tiles::{MartinCoreResult, Source, UrlQuery, zarr};
use zarrs::filesystem::FilesystemStore;

/// Tile source that reads from `Zarr` stores
#[derive(Clone, Debug)]
pub struct ZarrSource {
    id: String,
    path: PathBuf,
    tilejson: TileJSON,
    tileinfo: TileInfo,
    min_zoom: u8,
    max_zoom: u8,
    zarr_store: Arc<FilesystemStore>,
    x_coords: Arc<Array<FilesystemStore>>,
    y_coords: Arc<Array<FilesystemStore>>,
    time_coords: Arc<Array<FilesystemStore>>,
    data_vars: Arc<HashMap<String, Array<FilesystemStore>>>,
    wkt_str: String,
}

impl ZarrSource {
    /// Creates a new Zarr tile source from a file path
    pub fn new(id: String, path: PathBuf) -> Result<Self, ZarrError> {
        let tileinfo = TileInfo::new(Format::Png, martin_tile_utils::Encoding::Uncompressed);

        let zarr_store = Arc::new(
            FilesystemStore::new(&path)
                .map_err(|e| ZarrError::FilesystemStoreCreateError(e, path.clone()))?,
        );

        let x_coords = Array::open(zarr_store.clone(), "/x")?;
        let y_coords = Array::open(zarr_store.clone(), "/y")?;
        let time_coords = Array::open(zarr_store.clone(), "/time")?;
        let mut data_vars = HashMap::new();
        let node_paths = enumerate_data_variables(zarr_store.clone())?;
        for node_path in node_paths {
            let array = Array::open(zarr_store.clone(), node_path.as_str())?;
            data_vars.insert(node_path.as_str().into(), array);
        }
        let wkt_str = get_wkt_string(zarr_store.clone())?;

        let min_zoom = 0;
        let max_zoom = 30;
        let tilejson = tilejson! {
            tiles: vec![],
            minzoom: min_zoom,
            maxzoom: max_zoom
        };

        Ok(ZarrSource {
            id,
            path,
            tilejson,
            tileinfo,
            min_zoom,
            max_zoom,
            zarr_store,
            x_coords: Arc::new(x_coords),
            y_coords: Arc::new(y_coords),
            time_coords: Arc::new(time_coords),
            data_vars: data_vars.into(),
            wkt_str,
        })
    }
}

#[derive(Debug, Default)]
struct QueryParams<'a> {
    date_time: Option<DateTime<Utc>>,
    data_var: Option<&'a str>,
}

#[async_trait]
impl Source for ZarrSource {
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

        let time_str = "2026-02-17T00:00:00.000Z";
        let data_var = "/snow_depth";
        if xyz.z < self.min_zoom || xyz.z > self.max_zoom {
            return Ok(Vec::new());
        }
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {}
