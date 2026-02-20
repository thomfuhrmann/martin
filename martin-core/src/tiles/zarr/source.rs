//! Source for Zarr data
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::Arc;
use std::vec;

use async_trait::async_trait;
use martin_tile_utils::{Format, TileCoord, TileData, TileInfo};
use tilejson::{TileJSON, tilejson};

use crate::tiles::zarr::error::ZarrError;
use crate::tiles::{MartinCoreResult, Source, UrlQuery};
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
}

impl ZarrSource {
    /// Creates a new Zarr tile source from a file path
    pub fn new(id: String, path: PathBuf) -> Result<Self, ZarrError> {
        let tileinfo = TileInfo::new(Format::Png, martin_tile_utils::Encoding::Uncompressed);

        let zarr_store = Arc::new(
            FilesystemStore::new(&path)
                .map_err(|e| ZarrError::FilesystemStoreCreateError(e, path.clone()))?,
        );

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
        })
    }
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

    async fn get_tile(
        &self,
        xyz: TileCoord,
        _url_query: Option<&UrlQuery>,
    ) -> MartinCoreResult<TileData> {
        if xyz.z < self.min_zoom || xyz.z > self.max_zoom {
            return Ok(Vec::new());
        }
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {}
