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
use tracing::info;
use zarrs::array::Array;
use zarrs::node::Node;
use zarrs_object_store::AsyncObjectStore;

use crate::CacheZoomRange;
use crate::tiles::zarr::cache::{WarpCache, WarpCacheKey};
use crate::tiles::zarr::error::ZarrError;
use crate::tiles::zarr::utils::{
    LayoutItem, TILE_PIXELS, WarpGrid, abs_scales, calculate_warp_grid, data_variables,
    fill_value_f32, get_bbox, get_multiscales, get_proj, get_spatial_dims,
    get_spatial_dims_indices, get_spatial_registration, get_spatial_transform, retrieve_tile_data,
    sample_data, spatial_resolutions, tile_res, time_coords,
};
use crate::tiles::{MartinCoreError, MartinCoreResult, Source, UrlQuery};

// TODO: implement multiscales Zarr convention: https://github.com/zarr-conventions/multiscales

/// Currently no pyramids - therefore full zoom range
const MIN_ZOOM: u8 = 0;
const MAX_ZOOM: u8 = 30;

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
    layout_items: Option<HashMap<u64, LayoutItem>>,
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
        // tiles will be served as PNG which uses its own encoding
        let tileinfo = TileInfo::new(Format::Png, martin_tile_utils::Encoding::Internal);

        // wrap object store for zarrs
        let zarr_store = Arc::new(AsyncObjectStore::new(object_store));

        // get time coordinates
        let time_coords = time_coords(Arc::clone(&zarr_store)).await?.map(Arc::new);

        // get spatial metadata
        // TODO: currently it is supposed that the metadata includes the following GeoZarr conventions: proj, spatial
        // multiscales is optional
        let root_node = Node::async_open(Arc::clone(&zarr_store), "/")
            .await
            .map_err(|e| ZarrError::NodeCreateError(e))?;
        let src_crs = get_proj(&root_node).unwrap_or("EPSG:4326".into());
        let src_transform =
            get_spatial_transform(&root_node)?.unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let src_bbox = get_bbox(&root_node)?.unwrap_or([0.0, 0.0, 0.0, 0.0]);
        let spatial_registration =
            get_spatial_registration(&root_node)?.unwrap_or(SpatialRegistration::Pixel);
        let spatial_dims =
            get_spatial_dims(&root_node)?.unwrap_or(vec!["lat".into(), "lon".into()]);
        let multiscales = get_multiscales(&root_node)?;

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

        // resolve paths to data variables
        let mut data_vars: HashMap<String, _> = HashMap::new();
        let nodes = data_variables(Arc::clone(&zarr_store)).await?;
        let mut src_shape = [0, 0];
        let mut dimension_names = None;
        let mut first = true;
        for node in &nodes {
            let array = Array::async_open(Arc::clone(&zarr_store), node.path().as_str())
                .await
                .map_err(ZarrError::ArrayCreateError)?;
            if first {
                // assuming that the dimension names are the same for all arrays of this store
                dimension_names = array.dimension_names().clone();
                first = false;
            }
            data_vars.insert(node.path().as_str().into(), array);
        }

        // get the base layout for multiscales
        let base_layout = multiscales
            .clone()
            .and_then(|multi| {
                multi
                    .layout
                    .iter()
                    .find(|item| item.derived_from.is_none())
                    .cloned()
            })
            .expect("should have base layout item");

        // calculate absolute scales and corresponding layout items for multiscales
        let mut layout_items = None;
        if let Some(multiscales) = &multiscales {
            if let Some(dimension_names) = dimension_names {
                let shape = if let Some(shape) = &base_layout.spatial_shape {
                    shape
                } else {
                    let base_path = base_layout.asset;
                    let (_, arr) = data_vars
                        .iter()
                        .find(|(path, _)| path.starts_with(&base_path))
                        .expect("should be at least one variable at base resolution level");
                    arr.shape()
                };

                let indices = get_spatial_dims_indices(spatial_dims.as_slice(), &dimension_names)?;
                let base_shape: Vec<u64> = indices.iter().map(|&idx| shape[idx]).collect();
                src_shape = [base_shape[0], base_shape[1]];

                let base_res = spatial_resolutions(&base_shape, &src_bbox);
                let abs_scales = abs_scales(base_res, &multiscales, indices.as_slice());

                let mut items = HashMap::new();
                for (key, scale) in abs_scales {
                    let item = multiscales.layout.iter().find(|item| item.asset == key);
                    if let Some(item) = item {
                        items.insert(scale, item.clone());
                    }
                }
                layout_items = Some(items);
            }
        }

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
            layout_items,
        })
    }

    /// Calculate warp grid
    async fn run_calculate_warp_grid(&self, xyz: TileCoord) -> Result<Option<WarpGrid>, ZarrError> {
        let src_crs = self.src_crs.clone();
        let spatial_registration = self.spatial_registration.clone();

        // find matching layout item if it is a multiscales geozarr
        let matching_item = if let Some(layout_items) = &self.layout_items {
            let target_res = tile_res(xyz.z);
            let (_, layout_item) = layout_items
                .iter()
                .filter(|(scale_key, _)| **scale_key <= target_res)
                .max_by_key(|(scale_key, _)| **scale_key)
                .or_else(|| layout_items.iter().max_by_key(|(scale_key, _)| **scale_key))
                .expect("at least one item should match");
            Some(layout_item)
        } else {
            None
        };

        let src_transform = matching_item
            .and_then(|item| item.spatial_transform)
            .unwrap_or(self.src_transform);

        let src_shape = matching_item
            .and_then(|item| item.spatial_shape)
            .unwrap_or(self.src_shape);

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
        min: f32,
        max: f32,
    ) -> Result<Vec<u8>, ZarrError> {
        run_on_rayon(move || {
            sample_data(
                &warp_grid,
                warp_grid_bbox,
                &tile_data,
                TILE_PIXELS,
                fill_value,
                min,
                max,
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

#[derive(Debug, Default)]
struct QueryParams {
    date_time: Option<DateTime<Utc>>,
    data_var: Option<String>,
    min: Option<f32>,
    max: Option<f32>,
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
                    "date" => {
                        let date_time = DateTime::parse_from_rfc3339(value)
                            .map(|dt| dt.with_timezone(&Utc))
                            .map_err(ZarrError::ParseError)
                            .map_err(MartinCoreError::ZarrError)?;
                        query_params.date_time = Some(date_time);
                    }
                    "data_var" => {
                        let path_str = value.as_str();
                        let formatted = if path_str.starts_with('/') {
                            path_str.to_owned()
                        } else {
                            format!("/{path_str}")
                        };
                        query_params.data_var = Some(formatted);
                    }
                    "min" => {
                        let val_f32 = value.parse().map_err(ZarrError::ParseFloatError)?;
                        query_params.min = Some(val_f32);
                    }
                    "max" => {
                        let val_f32 = value.parse().map_err(ZarrError::ParseFloatError)?;
                        query_params.max = Some(val_f32);
                    }
                    _ => info!("Query string not supported"),
                }
            }
        }

        if xyz.z >= self.min_zoom
            && xyz.z <= self.max_zoom
            && let (Some(date_time), Some(data_var_key), Some(min), Some(max)) = (
                query_params.date_time,
                query_params.data_var.as_deref(),
                query_params.min,
                query_params.max,
            )
        {
            let data_var = self.data_vars.get(data_var_key).ok_or_else(|| {
                ZarrError::ParameterError(format!(
                    "Data variable {data_var_key} does not exist in Zarr store"
                ))
            })?;
            let fill_value = fill_value_f32(data_var.fill_value())?;

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

            let tile_data = retrieve_tile_data(
                warp_grid_bbox,
                self.time_coords.clone(),
                date_time,
                data_var,
            )
            .await?;

            let sampled_data = self
                .run_sample_data(warp_grid, warp_grid_bbox, tile_data, fill_value, min, max)
                .await
                .map_err(MartinCoreError::ZarrError)?;

            return Ok(sampled_data);
        }

        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {}
