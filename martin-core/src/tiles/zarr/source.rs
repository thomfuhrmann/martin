//! Source for GeoZarr data
//!
//! It is supposed that the metadata includes at least the following GeoZarr conventions:
//!
//! - [proj](https://github.com/zarr-conventions/proj)
//! - [spatial](https://github.com/zarr-conventions/spatial)
//!
//! [multiscales](https://github.com/zarr-conventions/multiscales) is optional
//!
//! Currently all values are converted to `f32` and the data is delivered as a gray-scale PNG
//!
//! Client side tile requests have to include the following query parameters:
//!
//! - date: in the form "YYYY-MM-DDTHH:MM:SS.MMMZ", for example "2026-08-13T00:04:00.000Z",
//! - data_var: the name of the Zarr variable to be rendered, for example "liquid_water"
//! - min: the min value of the data values to be rendered, for example "5.0"
//! - max: the max value of the data values to be rendered, for example "10.0"
//!
//! Tile data is sample based on a (per tile-coord cached) warp grid which determines the relation between tile and source pixels

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
    Proj, ResolutionLevel, TILE_PIXELS, TemporalIndex, WarpGrid, ZarrFillValue, abs_scales,
    calculate_warp_grid, fill_value, get_bbox, get_dims_indices, get_multiscales, get_proj,
    get_spatial_dims, get_spatial_dims_indices, get_spatial_registration, get_spatial_shape,
    get_spatial_transform, is_data_variable, retrieve_tile_data, sample_data, select_best_level,
    spatial_resolutions, zarr_nodes,
};
use crate::tiles::{MartinCoreError, MartinCoreResult, Source, UrlQuery};

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
    src_crs: String,
    temporal_indices: Arc<HashMap<String, TemporalIndex>>,
    data_vars: Arc<HashMap<String, Array<AsyncObjectStore<T>>>>,
    spatial_transform: [f64; 6],
    spatial_shape: Option<[u64; 2]>,
    spatial_registration: SpatialRegistration,
    cache_zoom: CacheZoomRange,
    warp_cache: WarpCache,
    resolution_levels: Option<Vec<ResolutionLevel>>,
    nodes: Vec<Node>,
    dims_indices: Vec<usize>,
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

        // get spatial metadata
        let root_node = Node::async_open(Arc::clone(&zarr_store), "/")
            .await
            .map_err(|e| ZarrError::NodeCreateError(e))?;
        let src_crs = get_proj(&root_node).unwrap_or(Proj::Code("EPSG:4326".into()));
        let spatial_transform =
            get_spatial_transform(&root_node)?.unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let spatial_bbox = get_bbox(&root_node)?;
        let spatial_registration =
            get_spatial_registration(&root_node)?.unwrap_or(SpatialRegistration::Pixel);
        let mut spatial_dims = get_spatial_dims(&root_node)?;
        let multiscales = get_multiscales(&root_node)?;

        // resolve paths to data variables
        let mut data_vars: HashMap<String, _> = HashMap::new();
        let nodes = zarr_nodes(Arc::clone(&zarr_store)).await?;
        let mut dimension_names = None;
        let mut dims_indices = None;
        let mut first = true;
        for node in &nodes {
            if is_data_variable(node).is_ok_and(|val| val) {
                let array = Array::async_open(Arc::clone(&zarr_store), node.path().as_str())
                    .await
                    .map_err(ZarrError::ArrayCreateError)?;
                if first {
                    // assuming that the dimension names are the same for all arrays of this store
                    dimension_names = array.dimension_names().clone();
                    if spatial_dims.is_none() {
                        spatial_dims = get_spatial_dims(&node)?
                    };

                    if let Some(spatial_dims) = &spatial_dims
                        && let Some(dimension_names) = &dimension_names
                    {
                        dims_indices = Some(get_dims_indices(&spatial_dims, &dimension_names)?);
                    }

                    first = false;
                }
                data_vars.insert(node.path().as_str().into(), array);
            };
        }

        // get the base layout for multiscales
        let base_layout = multiscales.clone().and_then(|multi| {
            multi
                .layout
                .iter()
                .find(|item| item.derived_from.is_none())
                .cloned()
        });

        let mut base_spatial_shape = get_spatial_shape(&root_node)?;

        // calculate absolute scales and corresponding layout items for multiscales
        let resolution_levels = if let Some(multiscales) = &multiscales
            && let Some(base_layout) = base_layout
            && let Some(dimension_names) = dimension_names
            && let Some(spatial_dims) = spatial_dims
        {
            let indices = get_spatial_dims_indices(spatial_dims.as_slice(), &dimension_names)?;
            let spatial_shape = if let Some(spatial_shape) = &base_layout.spatial_shape {
                [spatial_shape[0], spatial_shape[1]]
            } else {
                // get spatial shape from array
                let base_path = base_layout.asset;
                let (_, arr) = data_vars
                    .iter()
                    .find(|(path, _)| path.starts_with(&base_path))
                    .expect("should be at least one variable at base resolution level");
                let shape = arr.shape();
                let spatial_shape = indices.iter().map(|&idx| shape[idx]).collect::<Vec<_>>();
                [spatial_shape[0], spatial_shape[1]]
            };

            if base_spatial_shape.is_none() {
                base_spatial_shape = Some(spatial_shape);
            }

            let spatial_bbox = if let Some(spatial_bbox) = spatial_bbox {
                spatial_bbox
            } else {
                let xmin = spatial_transform[2];
                let xmax = (spatial_shape[1] as f64) * spatial_transform[0]
                    + (spatial_shape[0] as f64) * spatial_transform[1]
                    + spatial_transform[2];
                let ymax = spatial_transform[5];
                let ymin = (spatial_shape[1] as f64) * spatial_transform[3]
                    + (spatial_shape[0] as f64) * spatial_transform[4]
                    + spatial_transform[5];
                [xmin, ymin, xmax, ymax]
            };

            let base_res = spatial_resolutions(&spatial_shape, &spatial_bbox);
            let abs_scales = abs_scales(base_res, &multiscales, indices.as_slice());

            let mut levels: Vec<ResolutionLevel> = multiscales
                .layout
                .iter()
                .filter_map(|item| {
                    abs_scales.get(&item.asset).map(|&scale| ResolutionLevel {
                        scale,
                        item: item.clone(),
                    })
                })
                .collect();

            // sort by scale ascending (finer resolution first)
            levels.sort_by(|a, b| {
                a.scale
                    .partial_cmp(&b.scale)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            Some(levels)
        } else {
            None
        };

        // get time coordinates
        // TODO: extend to any non-spatial dimension
        let mut temporal_indices = HashMap::new();
        if let Some(multiscales) = &multiscales {
            for layout_item in &multiscales.layout {
                let base_path = format!("/{}", layout_item.asset);
                let path = format!("/{}/time", layout_item.asset);
                let array = Array::async_open(Arc::clone(&zarr_store), path.as_str())
                    .await
                    .map_err(ZarrError::ArrayCreateError)?;
                let temporal_index = TemporalIndex::load(&array).await?;
                temporal_indices.insert(base_path, temporal_index);
            }
        } else {
            let array = Array::async_open(Arc::clone(&zarr_store), "/time")
                .await
                .map_err(ZarrError::ArrayCreateError)?;
            let temporal_index = TemporalIndex::load(&array).await?;
            temporal_indices.insert("/".into(), temporal_index);
        }

        let bounds = if let Some(spatial_bbox) = spatial_bbox {
            let transformer =
                proj::Proj::try_from((src_crs.clone().into_string().as_str(), "EPSG:4326"))
                    .map_err(ZarrError::ProjCreateError)?;
            let (x_min, y_min) = transformer
                .convert((spatial_bbox[0], spatial_bbox[1]))
                .map_err(ZarrError::ProjError)?;
            let (x_max, y_max) = transformer
                .convert((spatial_bbox[2], spatial_bbox[3]))
                .map_err(ZarrError::ProjError)?;
            let bounds = Bounds::new(x_min, y_min, x_max, y_max);
            Some(bounds)
        } else {
            None
        };

        let tilejson = if let Some(bounds) = bounds {
            tilejson! {
                tiles: vec![],
                minzoom: MIN_ZOOM,
                maxzoom: MAX_ZOOM,
                bounds: bounds
            }
        } else {
            tilejson! {
                tiles: vec![],
                minzoom: MIN_ZOOM,
                maxzoom: MAX_ZOOM,
            }
        };

        Ok(Self {
            id,
            tilejson,
            tileinfo,
            min_zoom: MIN_ZOOM,
            max_zoom: MAX_ZOOM,
            src_crs: src_crs.into_string(),
            temporal_indices: temporal_indices.into(),
            data_vars: data_vars.into(),
            spatial_transform,
            spatial_shape: base_spatial_shape,
            spatial_registration,
            cache_zoom,
            warp_cache,
            resolution_levels,
            nodes,
            dims_indices: dims_indices.expect("should have dimensions"),
        })
    }

    /// Calculate warp grid
    async fn run_calculate_warp_grid(
        &self,
        xyz: TileCoord,
        src_crs: String,
        src_transform: [f64; 6],
        src_shape: [u64; 2],
        spatial_registration: SpatialRegistration,
    ) -> Result<Option<WarpGrid>, ZarrError> {
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
        fill_value: ZarrFillValue,
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
                        query_params.data_var = Some(value.as_str().into());
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
            // dynamically determine path to data array for multiscales zarr stores
            let matching_item = select_best_level(
                xyz,
                self.resolution_levels.as_deref(),
                TARGET_CRS,
                &self.src_crs,
            )?;

            // path to resolution level
            let level_path = matching_item
                .map(|item| format!("/{}", item.asset.clone()))
                .unwrap_or("/".into());

            let data_path = if let Some(layout_item) = matching_item {
                // all asset paths are relative to the group containing the multiscales metadata
                // current assumption is that multiscales is defined at the root group
                let asset_path = layout_item.asset.as_str();
                format!("/{asset_path}/{data_var_key}")
            } else {
                format!("/{data_var_key}")
            };

            let data_var = self.data_vars.get(&data_path).ok_or_else(|| {
                ZarrError::ParameterError(format!(
                    "Data variable {data_path} does not exist in Zarr store"
                ))
            })?;

            let fill_value = fill_value(data_var.fill_value(), data_var.data_type())?;

            let spatial_transform = matching_item
                .and_then(|item| item.spatial_transform)
                .unwrap_or(self.spatial_transform);

            let spatial_shape = matching_item
                .and_then(|item| item.spatial_shape)
                .or_else(|| {
                    // fallback to array node
                    let node = self
                        .nodes
                        .iter()
                        .find(|node| node.path().as_str() == data_path)
                        .expect("array node should be in hierarchy");
                    get_spatial_shape(node).unwrap_or(None)
                })
                .or_else(|| self.spatial_shape)
                .expect("spatial:shape to be present in hierarchy");

            let warp_grid = self
                .warp_cache
                .cache
                .get_with(WarpCacheKey(xyz, self.id.clone()), async {
                    self.run_calculate_warp_grid(
                        xyz,
                        self.src_crs.clone(),
                        spatial_transform,
                        spatial_shape,
                        self.spatial_registration.clone(),
                    )
                    .await
                    .ok()?
                })
                .await;

            let Some((warp_grid, warp_grid_bbox)) = warp_grid else {
                return Ok(vec![]);
            };

            let time_coords = self.temporal_indices.get(&level_path);

            let tile_data = retrieve_tile_data(
                warp_grid_bbox,
                time_coords,
                date_time,
                data_var,
                &self.dims_indices,
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

#[cfg(test)]
mod tests {}
