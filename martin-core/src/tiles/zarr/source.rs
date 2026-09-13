//! Source for Zarr data
use core::fmt;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use std::vec;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use martin_tile_utils::{Format, TileCoord, TileData, TileInfo};
use object_store::ObjectStore;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::LazyLock;
use tilejson::{Bounds, TileJSON, tilejson};
use tokio::sync::oneshot;
use zarrs::array::Array;
use zarrs_object_store::AsyncObjectStore;

use crate::CacheZoomRange;
use crate::tiles::zarr::cache::{WarpCache, WarpCacheKey};
use crate::tiles::zarr::error::ZarrError;
use crate::tiles::zarr::utils::{
    TILE_PIXELS, WarpGrid, calculate_warp_grid, data_variables, fill_value_f32, get_bbox,
    get_proj_code, get_spatial_registration, get_spatial_shape, get_spatial_transform,
    retrieve_tile_data, sample_data, time_coords,
};
use crate::tiles::{MartinCoreError, MartinCoreResult, Source, UrlQuery};

/// Currently no pyramids - therefore full zoom range
const MIN_ZOOM: u8 = 0;
const MAX_ZOOM: u8 = 23;

/// Coordinate reference system of Martin tile server
pub(crate) const TARGET_CRS: &str = "EPSG:3857";

#[derive(Debug, Clone)]
pub(crate) enum SpatialRegistration {
    Pixel,
    Node,
}

/// Tile source that reads from `Zarr` stores
#[derive(Clone)]
pub struct ZarrSource<T: ObjectStore + Clone> {
    id: String,
    tilejson: TileJSON,
    tileinfo: TileInfo,
    min_zoom: u8,
    max_zoom: u8,
    time_coords: Option<Arc<Array<AsyncObjectStore<T>>>>,
    data_vars: Arc<HashMap<String, Array<AsyncObjectStore<T>>>>,
    src_crs: String,
    src_transform: [f64; 6],
    src_shape: [u64; 2],
    spatial_registration: SpatialRegistration,
    cache_zoom: CacheZoomRange,
    warp_cache: WarpCache,
}

impl<T: ObjectStore + Clone> Debug for ZarrSource<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ZarrTileSource")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl<T: ObjectStore + Clone> ZarrSource<T> {
    /// Creates a new Zarr tile source from a file path
    pub async fn new(
        id: String,
        object_store: T,
        warp_cache: WarpCache,
        cache_zoom: CacheZoomRange,
    ) -> Result<Self, ZarrError> {
        let tileinfo = TileInfo::new(
            Format::OctetStream,
            martin_tile_utils::Encoding::Uncompressed,
        );

        let zarr_store = Arc::new(AsyncObjectStore::new(object_store));

        let time_coords = time_coords(Arc::clone(&zarr_store)).await?.map(Arc::new);

        // resolve paths to data variables
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
        let src_shape = get_spatial_shape(Arc::clone(&zarr_store)).await?;
        let spatial_registration = get_spatial_registration(Arc::clone(&zarr_store)).await?;

        // bounding box in tilejson needs to be in WGS84 - see <https://github.com/mapbox/tilejson-spec/tree/master/3.0.0#35-bounds>
        let transformer = proj::Proj::try_from((src_crs.as_str(), "EPSG:4326"))
            .map_err(ZarrError::ProjCreateError)?;
        let (x_min, y_min) = transformer
            .convert((src_bbox[0], src_bbox[1]))
            .map_err(ZarrError::ProjError)?;
        let (x_max, y_max) = transformer
            .convert((src_bbox[2], src_bbox[3]))
            .map_err(ZarrError::ProjError)?;
        let bounds = Bounds::new(x_min, y_min, x_max, y_max);

        let tilejson = tilejson! {
            tiles: vec![],
            minzoom: MIN_ZOOM,
            maxzoom: MAX_ZOOM,
            bounds: bounds
        };

        Ok(Self {
            id,
            tilejson,
            tileinfo,
            min_zoom: MIN_ZOOM,
            max_zoom: MAX_ZOOM,
            time_coords,
            data_vars: data_vars.into(),
            src_crs,
            src_transform,
            src_shape,
            spatial_registration,
            cache_zoom,
            warp_cache,
        })
    }

    /// Calculate warp grid
    async fn run_calculate_warp_grid(&self, xyz: TileCoord) -> Result<Option<WarpGrid>, ZarrError> {
        let src_transform = self.src_transform;
        let src_crs = self.src_crs.clone();
        let src_shape = self.src_shape;
        let spatial_registration = self.spatial_registration.clone();

        run_on_rayon(move || {
            calculate_warp_grid(
                xyz,
                src_transform,
                &src_crs,
                src_shape,
                &spatial_registration,
            )
        })
        .await?
    }

    /// Sample data at warp grid points
    async fn run_sample_data(
        &self,
        warp_grid: Box<[i64]>,
        warp_grid_bbox: [u64; 4],
        tile_data: ndarray::Array3<f32>,
        fill_value: f32,
    ) -> Result<Vec<u8>, ZarrError> {
        run_on_rayon(move || {
            sample_data(
                &warp_grid,
                warp_grid_bbox,
                &tile_data,
                TILE_PIXELS,
                fill_value,
            )
        })
        .await?
    }
}

static WARP_POOL: LazyLock<Arc<ThreadPool>> = LazyLock::new(|| {
    let num_threads =
        std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(2).max(1));
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .thread_name(|i| format!("zarr-{i}"))
        .build()
        .expect("Failed to create zarr thread pool");
    Arc::new(pool)
});

async fn run_on_rayon<R, F>(f: F) -> Result<R, ZarrError>
where
    R: Send + 'static,
    F: FnOnce() -> R + Send + 'static,
{
    let (tx, rx) = oneshot::channel();

    WARP_POOL.spawn(move || {
        let result = f();
        let _ = tx.send(result);
    });

    rx.await
        .map_err(|_err_| ZarrError::WarpError("Rayon worker dropped".into()))
}

// #[derive(Debug, Default)]
// struct QueryParams<'a> {
//     date_time: Option<DateTime<Utc>>,
//     data_var: Option<&'a str>,
// }

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
        _url_query: Option<&UrlQuery>,
    ) -> MartinCoreResult<TileData> {
        // let mut query_params = QueryParams::default();
        // if let Some(url_query) = url_query {
        //     for (key, value) in url_query {
        //         match key.as_str() {
        //             "time" => {
        //                 let date_time = DateTime::parse_from_rfc3339(value)
        //                     .map(|dt| dt.with_timezone(&Utc))
        //                     .map_err(ZarrError::ParseError)
        //                     .map_err(MartinCoreError::ZarrError)?;
        //                 query_params.date_time = Some(date_time);
        //             }
        //             "data_var" => query_params.data_var = Some(value.as_str()),
        //             _ => info!("Query string not supported"),
        //         }
        //     }
        // }

        let time_str = "2026-08-13T00:00:00.000Z";
        let data_var = "/snow_temp";

        if xyz.z >= self.min_zoom && xyz.z <= self.max_zoom {
            let data_var = self.data_vars.get(data_var).expect("msg");
            let datetime = DateTime::parse_from_rfc3339(time_str)
                .expect("msg")
                .with_timezone(&Utc);

            let warp_grid = self
                .warp_cache
                .cache
                .get_with(WarpCacheKey(xyz, self.id.clone()), async {
                    self.run_calculate_warp_grid(xyz).await.ok()?
                })
                .await;

            let Some((warp_grid, warp_grid_bbox)) = warp_grid else {
                return Ok(vec![]);
            };

            let tile_data =
                retrieve_tile_data(warp_grid_bbox, self.time_coords.clone(), datetime, data_var)
                    .await?;

            let fill_value = fill_value_f32(data_var.fill_value())?;

            let sampled_data = self
                .run_sample_data(warp_grid, warp_grid_bbox, tile_data, fill_value)
                .await
                .map_err(MartinCoreError::ZarrError)?;

            return Ok(sampled_data);
        }

        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {}
