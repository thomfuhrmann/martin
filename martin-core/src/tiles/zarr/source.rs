//! Source for `GeoZarr` data
//!
//! It is supposed that the metadata includes at least the following `GeoZarr` conventions:
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
//! - dim:*=value: the name of the dimension and its value to be sliced, time has to be in the form "YYYY-MM-DDTHH:MM:SS.MMMZ", for instance `dim:time=2026-08-13T00:04:00.000Z`,
//! - `data_var`: the name of the Zarr variable to be rendered, for instance `data_var=liquid_water`
//! - min: the min value of the data values to be rendered, for instance "5.0"
//! - max: the max value of the data values to be rendered, for instance "10.0"
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
use ndarray::Array2;
use object_store::ObjectStore;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::LazyLock;
use tilejson::{TileJSON, tilejson};
use tokio::sync::oneshot;
use tracing::info;
use zarrs::array::Array;
use zarrs::node::Node;
use zarrs_object_store::AsyncObjectStore;

use crate::CacheZoomRange;
use crate::tiles::zarr::cache::{WarpCache, WarpCacheKey};
use crate::tiles::zarr::error::ZarrError;
use crate::tiles::zarr::utils::{
    DimIndex, DimMeta, Proj, ResolutionLevel, SpatialAxisIndices, TILE_PIXELS, WarpGrid,
    ZarrFillValue, bounds_from_bbox, calculate_resolution_levels, calculate_warp_grid, fill_value,
    get_bbox, get_multiscales, get_non_spatial_dims, get_proj, get_spatial_dims,
    get_spatial_registration, get_spatial_shape, get_spatial_transform, is_data_variable,
    retrieve_tile_data, sample_data, select_best_level, zarr_nodes,
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
    dims_index: Arc<HashMap<String, DimIndex>>,
    data_vars: Arc<HashMap<String, Array<AsyncObjectStore<T>>>>,
    spatial_dims: Vec<String>,
    spatial_transform: [f64; 6],
    spatial_shape: Option<[u64; 2]>,
    spatial_registration: SpatialRegistration,
    cache_zoom: CacheZoomRange,
    warp_cache: WarpCache,
    resolution_levels: Option<Vec<ResolutionLevel>>,
    nodes: Vec<Node>,
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
            .map_err(ZarrError::NodeCreateError)?;
        let src_crs = get_proj(&root_node).unwrap_or(Proj::Code("EPSG:4326".into()));
        let spatial_transform =
            get_spatial_transform(&root_node)?.unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let spatial_bbox = get_bbox(&root_node)?;
        let spatial_registration =
            get_spatial_registration(&root_node).unwrap_or(SpatialRegistration::Pixel);
        let mut spatial_dims = get_spatial_dims(&root_node)?;
        let multiscales = get_multiscales(&root_node)?;

        // resolve paths to data variables
        let mut data_vars: HashMap<String, _> = HashMap::new();
        let nodes = zarr_nodes(Arc::clone(&zarr_store)).await?;
        let mut dimension_names = None;
        let mut first = true;
        for node in &nodes {
            if is_data_variable(node).is_ok_and(|val| val) {
                let array = Array::async_open(Arc::clone(&zarr_store), node.path().as_str())
                    .await
                    .map_err(ZarrError::ArrayCreateError)?;
                if first {
                    // assuming that the dimension names are the same for all arrays of this store
                    dimension_names.clone_from(array.dimension_names());
                    if spatial_dims.is_none() {
                        spatial_dims = get_spatial_dims(node)?;
                    }

                    first = false;
                }
                data_vars.insert(node.path().as_str().into(), array);
            }
        }

        // get the base layout for multiscales
        let base_layout = multiscales.clone().and_then(|multi| {
            multi
                .layout
                .iter()
                .find(|item| item.derived_from.is_none())
                .cloned()
        });

        // calculate absolute scales and corresponding layout items for multiscales
        let resolutions_and_shape = calculate_resolution_levels(
            &data_vars,
            multiscales.as_ref(),
            base_layout.as_ref(),
            dimension_names.as_deref(),
            spatial_dims.as_deref(),
            spatial_transform,
            spatial_bbox,
        )?;

        let base_spatial_shape = if let Some((_, spatial_shape)) = resolutions_and_shape {
            Some(spatial_shape)
        } else {
            get_spatial_shape(&root_node)?
        };

        let resolution_levels = if let Some((levels, _)) = resolutions_and_shape {
            Some(levels)
        } else {
            None
        };

        // get dimensional coordinates
        let non_spatial_dims =
            get_non_spatial_dims(spatial_dims.as_deref(), dimension_names.as_deref());
        let mut dims_index = HashMap::new();
        if let Some(multiscales) = &multiscales {
            for layout_item in &multiscales.layout {
                for dim in &non_spatial_dims {
                    let path = format!("/{}/{dim}", layout_item.asset);
                    let array = Array::async_open(Arc::clone(&zarr_store), path.as_str())
                        .await
                        .map_err(ZarrError::ArrayCreateError)?;
                    let index = DimIndex::load(&array).await?;
                    dims_index.insert(path, index);
                }
            }
        } else {
            for dim in &non_spatial_dims {
                let path = format!("/{dim}");
                let array = Array::async_open(Arc::clone(&zarr_store), path.as_str())
                    .await
                    .map_err(ZarrError::ArrayCreateError)?;
                let index = DimIndex::load(&array).await?;
                dims_index.insert(path, index);
            }
        }

        let bounds = bounds_from_bbox(spatial_bbox, &src_crs)?;
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
            dims_index: dims_index.into(),
            data_vars: data_vars.into(),
            spatial_dims: spatial_dims.unwrap_or(vec!["y".to_owned(), "x".to_owned()]),
            spatial_transform,
            spatial_shape: base_spatial_shape,
            spatial_registration,
            cache_zoom,
            warp_cache,
            resolution_levels,
            nodes,
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
        tile_data: Array2<f32>,
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
    time: Option<DateTime<Utc>>,
    other_dims: HashMap<String, f64>,
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
                    "dim:time" => {
                        let date_time = DateTime::parse_from_rfc3339(value)
                            .map(|dt| dt.with_timezone(&Utc))
                            .map_err(ZarrError::ParseError)
                            .map_err(MartinCoreError::ZarrError)?;
                        query_params.time = Some(date_time);
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
                    key => {
                        let rest = key.strip_prefix("dim:");
                        let val = value.parse().map_err(ZarrError::ParseFloatError)?;
                        if let Some(rest) = rest {
                            query_params.other_dims.insert(rest.to_owned(), val);
                        } else {
                            info!("query parameter not supported: {key}");
                        }
                    }
                }
            }
        }

        if xyz.z >= self.min_zoom
            && xyz.z <= self.max_zoom
            && let (Some(data_var_key), Some(min), Some(max)) = (
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
            let level_path =
                matching_item.map_or("/".into(), |item| format!("/{}", item.asset.clone()));

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

            let mut dim_coords = HashMap::new();
            let time_path = format!("{level_path}/time");
            let time_index = self.dims_index.get(&time_path);
            if let Some(index) = time_index
                && let Some(date_time) = &query_params.time
            {
                let DimMeta::Time(meta) = &index.meta;
                let val = meta.datetime_to_raw(date_time);
                let coord = index.get_index(val);
                dim_coords.insert("time".to_owned(), coord);
            }

            for (dim_name, val) in query_params.other_dims {
                let dim_path = format!("{level_path}/{dim_name}");
                if let Some((_, index)) =
                    self.dims_index.iter().find(|(path, _)| **path == dim_path)
                {
                    let coord = index.get_index(val);
                    dim_coords.insert(dim_name, coord);
                }
            }

            let Some(dimension_names) = data_var.dimension_names() else {
                return Err(MartinCoreError::ZarrError(ZarrError::DimensionError(
                    "missing array dimensions".to_owned(),
                )));
            };

            if dimension_names.len() != dim_coords.len() + 2 {
                return Err(MartinCoreError::ZarrError(ZarrError::DimensionError(
                    format!(
                        "number of dimension parameters must be 2 less than array dimensions: {}",
                        dimension_names.len()
                    ),
                )));
            }

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
                .or(self.spatial_shape)
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

            let tile_data =
                retrieve_tile_data(warp_grid_bbox, dim_coords, data_var, &self.spatial_dims)
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
