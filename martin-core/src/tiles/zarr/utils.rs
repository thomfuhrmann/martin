use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use core::f64;
use object_store::ObjectStore;
use zarrs::node::async_get_child_nodes;
use zarrs_object_store::AsyncObjectStore;
// use image::{ImageBuffer, LumaA};
use martin_tile_utils::TileCoord;
use std::{error::Error, ops::Range, sync::Arc};
use zarrs::storage::{AsyncListableStorageTraits, AsyncReadableStorageTraits};
use zarrs::{
    array::{Array, ArrayMetadata, DimensionName},
    filesystem::FilesystemStore,
    group::Group,
    node::{Node, NodeMetadata, NodePath},
    plugin::ZarrVersion,
};

use crate::tiles::zarr::error::ZarrError;

/// Retrieve all data variables of this store - arrays that are not dimensions
pub async fn data_variables<T: ObjectStore>(
    store: Arc<AsyncObjectStore<T>>,
) -> Result<Vec<NodePath>, ZarrError> {
    let root_path = NodePath::root();
    let child_nodes = async_get_child_nodes(&store, &root_path, true)
        .await
        .map_err(ZarrError::NodeCreateError)?;
    let filtered_child_nodes = child_nodes
        .into_iter()
        .filter(|n| is_data_variable(n).is_ok_and(|is_data| is_data))
        .map(|n| n.path().clone())
        .collect::<Vec<_>>();
    Ok(filtered_child_nodes)
}

/// Retrieve time variable
pub async fn time_coords<S: AsyncReadableStorageTraits + AsyncListableStorageTraits + 'static>(
    store: Arc<S>,
) -> Result<Option<Array<S>>, ZarrError> {
    let root_path = NodePath::root();
    let child_nodes = async_get_child_nodes(&store, &root_path, true)
        .await
        .map_err(ZarrError::NodeCreateError)?;
    let time_node_path = child_nodes
        .into_iter()
        .find(|n| n.name().as_str() == "time")
        .map(|n| n.path().clone());
    if let Some(node_path) = time_node_path {
        let array = Array::async_open(store, node_path.as_str())
            .await
            .map_err(ZarrError::ArrayCreateError)?;
        return Ok(Some(array));
    }
    Ok(None)
}

/// Check if the array is a data variable
fn is_data_variable(node: &Node) -> Result<bool, ZarrError> {
    let metadata = node.metadata();
    let path = node.path().as_str();

    let dim_names = match metadata {
        NodeMetadata::Array(array_metadata) => match array_metadata {
            ArrayMetadata::V2(metadata_v2) => metadata_v2
                .attributes
                .get("_ARRAY_DIMENSIONS")
                .and_then(|val| serde_json::from_value(val.clone()).ok())
                .ok_or_else(|| {
                    ZarrError::AttributeError("_ARRAY_DIMENSIONS missing or invalid".into())
                })?,
            ArrayMetadata::V3(metadata_v3) => metadata_v3
                .dimension_names
                .clone()
                .and_then(|names| names.into_iter().collect())
                .ok_or_else(|| {
                    ZarrError::AttributeError(
                        "dimension_names missing or contains null elements".into(),
                    )
                })?,
        },
        NodeMetadata::Group(_) => Vec::new(),
    };

    let is_coord = dim_names.iter().any(|dim_name| path.ends_with(dim_name));
    Ok(!is_coord)
}

/// Get coordinate system definition as EPSG code
pub async fn get_proj_code<T: ObjectStore>(
    store: Arc<AsyncObjectStore<T>>,
) -> Result<String, ZarrError> {
    let root_group = Group::async_open(store, NodePath::root().as_str())
        .await
        .map_err(ZarrError::GroupCreateError)?;

    root_group
        .attributes()
        .get("proj:code")
        .and_then(|val| val.as_str())
        .map(Into::into)
        .ok_or(ZarrError::AttributeError("proj:code".into()))
}

/// Get the affine transformation from pixel space to geographic space
pub async fn get_spatial_transform<T: ObjectStore>(
    store: Arc<AsyncObjectStore<T>>,
) -> Result<[f64; 6], ZarrError> {
    let root_group = Group::async_open(store, NodePath::root().as_str())
        .await
        .map_err(ZarrError::GroupCreateError)?;

    let transform_value = root_group
        .attributes()
        .get("spatial:transform")
        .ok_or_else(|| ZarrError::AttributeError("spatial:transform is missing".into()))?;

    serde_json::from_value::<[f64; 6]>(transform_value.clone())
        .map_err(|e| ZarrError::AttributeError(format!("Invalid spatial:transform format: {e}")))
}

/// Get the bounding box
pub async fn get_bbox<T: ObjectStore>(
    store: Arc<AsyncObjectStore<T>>,
) -> Result<[f64; 4], ZarrError> {
    let root_group = Group::async_open(store, NodePath::root().as_str())
        .await
        .map_err(ZarrError::GroupCreateError)?;

    let transform_value = root_group
        .attributes()
        .get("spatial:bbox")
        .ok_or_else(|| ZarrError::AttributeError("spatial:transform is missing".into()))?;

    serde_json::from_value::<[f64; 4]>(transform_value.clone())
        .map_err(|e| ZarrError::AttributeError(format!("Invalid spatial:transform format: {e}")))
}

/// Returns the names of the spatial dimensions
pub fn get_spatial_dims(array: &Array<FilesystemStore>) -> Option<Vec<&str>> {
    array
        .attributes()
        .get("spatial:dimensions")
        .and_then(|val| val.as_array())
        .and_then(|dims| dims.iter().map(|dim| dim.as_str()).collect())
}

pub fn get_non_spatial_dims(
    array: &Array<FilesystemStore>,
    spatial_dims: &Vec<&str>,
) -> Result<Vec<String>, Box<dyn Error>> {
    let mut non_spatial = vec![];

    let dim_names = array
        .dimension_names()
        .as_ref()
        .ok_or("Could not retrieve dimension names")?;

    for name in dim_names.iter().flatten() {
        let spatial = spatial_dims.iter().any(|&spatial_dim| spatial_dim == name);

        if !spatial {
            non_spatial.push(name.clone());
        }
    }

    Ok(non_spatial)
}

const TILE_PIXELS: u32 = 512;
const TARGET_CRS: &str = "EPSG:3857";

/// Sample from array using a projection
pub async fn sample_data_var<T: ObjectStore>(
    tile: &TileCoord,
    src_crs: &str,
    src_transform: [f64; 6],
    src_bbox: [f64; 4],
    time_coords: Option<Arc<Array<AsyncObjectStore<T>>>>,
    datetime: DateTime<Utc>,
    data_var: &Array<AsyncObjectStore<T>>,
) -> Result<Vec<u8>, Box<dyn Error>> {
    return Ok(vec![]);

    // Load data for x-coordinates
    //    let data_type = x_coords.data_type();
    //    let name = data_type.name(ZarrVersion::V3);
    //    let data_x = match name.as_deref() {
    //        Some("int32") => x_coords
    //            .retrieve_array_subset::<ndarray::Array1<i32>>(&[0..2])?
    //            .mapv(|v| v as f64),
    //        Some("int64") => x_coords
    //            .retrieve_array_subset::<ndarray::Array1<i64>>(&[0..2])?
    //            .mapv(|v| v as f64),
    //        Some("float32") => x_coords
    //            .retrieve_array_subset::<ndarray::Array1<f32>>(&[0..2])?
    //            .mapv(|v| v as f64),
    //        Some("float64") => x_coords.retrieve_array_subset::<ndarray::Array1<f64>>(&[0..2])?,
    //        _ => panic!("Data type not implemented yet"),
    //    };

    //    // Load data for y-coordinates
    //    let data_type = y_coords.data_type();
    //    let name = data_type.name(ZarrVersion::V3);
    //    let data_y = match name.as_deref() {
    //        Some("int32") => y_coords
    //            .retrieve_array_subset::<ndarray::Array1<i32>>(&[0..2])?
    //            .mapv(|v| v as f64),
    //        Some("int64") => y_coords
    //            .retrieve_array_subset::<ndarray::Array1<i64>>(&[0..2])?
    //            .mapv(|v| v as f64),
    //        Some("float32") => y_coords
    //            .retrieve_array_subset::<ndarray::Array1<f32>>(&[0..2])?
    //            .mapv(|v| v as f64),
    //        Some("float64") => y_coords.retrieve_array_subset::<ndarray::Array1<f64>>(&[0..2])?,
    //        _ => panic!("Data type not implemented yet"),
    //    };

    //    // Calculate spatial origin and spacing of array data
    //    let x_0 = data_x[0];
    //    let spacing_x = data_x[1] - data_x[0];

    //    let y_0 = data_y[0];
    //    let spacing_y = data_y[1] - data_y[0];

    //    // Initialize coordinate transformation
    //    let transformer = proj::Proj::try_from((SOURCE_CRS, wkt_str))?;

    //    // Get tile bounding box in Web Mercator
    //    let tile = tile_bbox(tile.x, tile.y, tile.z);
    //    let x_min_source = tile[0];
    //    let y_min_source = tile[1];
    //    let x_max_source = tile[2];
    //    let y_max_source = tile[3];

    //    // Calculate grid spacings for source CRS based on tile size
    //    let spacing_x_source = (x_max_source - x_min_source) / f64::from(TILE_PIXELS);
    //    let spacing_y_source = (y_max_source - y_min_source) / f64::from(TILE_PIXELS);

    //    // Transform tile bounding box corners from Web Mercator to target system
    //    let (bl_x_target, bl_y_target) = transformer.convert((x_min_source, y_min_source))?;
    //    let (br_x_target, br_y_target) = transformer.convert((x_max_source, y_min_source))?;
    //    let (tl_x_target, tl_y_target) = transformer.convert((x_min_source, y_max_source))?;
    //    let (tr_x_target, tr_y_target) = transformer.convert((x_max_source, y_max_source))?;

    //    // Calculate array indices of bounding box
    //    let bl_ix = ((bl_x_target - x_0) / spacing_x).round() as u64;
    //    let br_ix = ((br_x_target - x_0) / spacing_x).round() as u64;
    //    let tl_ix = ((tl_x_target - x_0) / spacing_x).round() as u64;
    //    let tr_ix = ((tr_x_target - x_0) / spacing_x).round() as u64;

    //    let mut min_ix = bl_ix.min(br_ix).min(tl_ix).min(tr_ix);
    //    let mut max_ix = bl_ix.max(br_ix).max(tl_ix).max(tr_ix);

    //    let bl_iy = ((bl_y_target - y_0) / spacing_y).round() as u64;
    //    let br_iy = ((br_y_target - y_0) / spacing_y).round() as u64;
    //    let tl_iy = ((tl_y_target - y_0) / spacing_y).round() as u64;
    //    let tr_iy = ((tr_y_target - y_0) / spacing_y).round() as u64;

    //    let mut min_iy = bl_iy.min(br_iy).min(tl_iy).min(tr_iy);
    //    let mut max_iy = bl_iy.max(br_iy).max(tl_iy).max(tr_iy);

    //    // Swap indices if coordinate array is in descending order
    //    if min_ix > max_ix {
    //        let temp = min_ix;
    //        min_ix = max_ix;
    //        max_ix = temp;
    //    }

    //    if min_iy > max_iy {
    //        let temp = min_iy;
    //        min_iy = max_iy;
    //        max_iy = temp;
    //    }

    //    // Add buffer
    //    let buffer = 8;
    //    min_ix = min_ix.saturating_sub(buffer);
    //    min_iy = min_iy.saturating_sub(buffer);
    //    max_ix = max_ix.saturating_add(buffer);
    //    max_iy = max_iy.saturating_add(buffer);

    //    // Set index ranges
    //    let target_width = x_coords.shape()[0];
    //    let x_range_start = min_ix.min(target_width);
    //    let x_range_end = (max_ix + 1).min(target_width);
    //    let x_range = x_range_start..x_range_end;

    //    let target_height = y_coords.shape()[0];
    //    let y_range_start = min_iy.min(target_height);
    //    let y_range_end = (max_iy + 1).min(target_height);
    //    let y_range = y_range_start..y_range_end;

    //    // Load tile data into memory
    //    let temporal_index = get_temporal_index(time_coords, datetime)? as u64;
    //    let temporal_range = temporal_index..temporal_index + 1;

    //    // Permute tile data to have a fixed order: time, x, y
    //    let dimension_names = data_var.dimension_names();
    //    let indices = build_subset(dimension_names, temporal_range, x_range, y_range);
    //    let perm = order_dimensions(dimension_names);
    //    let tile_data = data_var
    //        .retrieve_array_subset::<ndarray::Array3<f32>>(&[
    //            indices[0].clone(),
    //            indices[1].clone(),
    //            indices[2].clone(),
    //        ])?
    //        .permuted_axes(perm);

    //    // Sample from array at tile grid points
    //    let width = x_range_end - x_range_start;
    //    let height = y_range_end - y_range_start;
    //    let mut min = f32::INFINITY;
    //    let mut max = f32::NEG_INFINITY;
    //    let mut sampled_data = [f32::NAN; TILE_PIXELS as usize * TILE_PIXELS as usize];
    //    for i in 0..TILE_PIXELS {
    //        for j in 0..TILE_PIXELS {
    //            let x = x_min_source + (f64::from(i) + 0.5) * spacing_x_source;
    //            let y = y_min_source + (f64::from(j) + 0.5) * spacing_y_source;

    //            // Transform from Web Mercator to target system
    //            let (xt, yt) = transformer.convert((x, y))?;

    //            // Index calculation
    //            // use i64 for intermediate calculations to handle coordinates
    //            // that might fall outside min_ix/min_iy
    //            let abs_ix = ((xt - x_0) / spacing_x).round() as i64;
    //            let abs_iy = ((yt - y_0) / spacing_y).round() as i64;

    //            // Map to relative index
    //            let rel_ix = abs_ix - min_ix as i64;
    //            let rel_iy = abs_iy - min_iy as i64;

    //            // Bounds check and sample
    //            if rel_ix >= 0 && rel_ix < width as i64 && rel_iy >= 0 && rel_iy < height as i64 {
    //                let value = tile_data[[0, rel_ix as usize, rel_iy as usize]];
    //                if value < min {
    //                    min = value;
    //                }
    //                if value > max {
    //                    max = value;
    //                }

    //                sampled_data[(j as usize) * TILE_PIXELS as usize + i as usize] = value;
    //            }
    //        }
    //    }

    //    // Cast to raw bytes
    //    let raw_bytes = bytemuck::cast_slice::<f32, u8>(&sampled_data);

    //    // Compress with Zstd
    //    let compressed_bytes = encode_all(raw_bytes, 3)?;
    //    Ok(compressed_bytes)
}

fn calculate_warp_grid(
    tile: TileCoord,
    src_transform: [f64; 6],
    src_bbox: [f64; 4],
    src_crs: &str,
) -> Result<[u32; 1024], ZarrError> {
    // Calculate spatial origin and spacing of array data
    let spacing_x = src_transform[0];
    let x_0 = src_transform[2];

    let spacing_y = src_transform[4];
    let y_0 = src_transform[5];

    // Initialize coordinate transformation
    let transformer = proj::Proj::try_from((TARGET_CRS, src_crs))?;

    // let x_min_source = src_bbox[0];
    // let y_min_source = src_bbox[1];
    // let x_max_source = src_bbox[2];
    // let y_max_source = src_bbox[3];

    // Get tile bounding box in Web Mercator
    let tile = tile_bbox(tile.x, tile.y, tile.z);
    let x_min = tile[0];
    let y_min = tile[1];
    let x_max = tile[2];
    let y_max = tile[3];

    // Calculate grid spacings for target CRS based on tile size
    let spacing_x_source = (x_max - x_min) / f64::from(TILE_PIXELS);
    let spacing_y_source = (y_max - y_min) / f64::from(TILE_PIXELS);

    // Transform tile bounding box corners from Web Mercator to source system
    let (x_min_src, y_min_src) = transformer
        .convert((x_min, y_min))
        .map_err(ZarrError::ProjError)?;
    let (x_max_src, y_max_src) = transformer
        .convert((x_max, y_max))
        .map_err(ZarrError::ProjError)?;

    // Calculate array indices of bounding box
    let bl_ix = ((x_min_src - x_0) / spacing_x).round() as u64;
    let br_ix = ((br_x_target - x_0) / spacing_x).round() as u64;
    let tl_ix = ((tl_x_target - x_0) / spacing_x).round() as u64;
    let tr_ix = ((tr_x_target - x_0) / spacing_x).round() as u64;

    let mut min_ix = bl_ix.min(br_ix).min(tl_ix).min(tr_ix);
    let mut max_ix = bl_ix.max(br_ix).max(tl_ix).max(tr_ix);

    let bl_iy = ((bl_y_target - y_0) / spacing_y).round() as u64;
    let br_iy = ((br_y_target - y_0) / spacing_y).round() as u64;
    let tl_iy = ((tl_y_target - y_0) / spacing_y).round() as u64;
    let tr_iy = ((tr_y_target - y_0) / spacing_y).round() as u64;

    let mut min_iy = bl_iy.min(br_iy).min(tl_iy).min(tr_iy);
    let mut max_iy = bl_iy.max(br_iy).max(tl_iy).max(tr_iy);

    // Swap indices if coordinate array is in descending order
    if min_ix > max_ix {
        let temp = min_ix;
        min_ix = max_ix;
        max_ix = temp;
    }

    if min_iy > max_iy {
        let temp = min_iy;
        min_iy = max_iy;
        max_iy = temp;
    }

    // Add buffer
    let buffer = 8;
    min_ix = min_ix.saturating_sub(buffer);
    min_iy = min_iy.saturating_sub(buffer);
    max_ix = max_ix.saturating_add(buffer);
    max_iy = max_iy.saturating_add(buffer);

    // Set index ranges
    let target_width = x_coords.shape()[0];
    let x_range_start = min_ix.min(target_width);
    let x_range_end = (max_ix + 1).min(target_width);
    let x_range = x_range_start..x_range_end;

    let target_height = y_coords.shape()[0];
    let y_range_start = min_iy.min(target_height);
    let y_range_end = (max_iy + 1).min(target_height);
    let y_range = y_range_start..y_range_end;

    // Load tile data into memory
    let temporal_index = get_temporal_index(time_coords, datetime)? as u64;
    let temporal_range = temporal_index..temporal_index + 1;

    // Permute tile data to have a fixed order: time, x, y
    let dimension_names = data_var.dimension_names();
    let indices = build_subset(dimension_names, temporal_range, x_range, y_range);
    Ok([0; 1024])
}

fn build_subset(
    dimension_names: &Option<Vec<DimensionName>>,
    time: Range<u64>,
    x: Range<u64>,
    y: Range<u64>,
) -> Vec<Range<u64>> {
    match dimension_names {
        Some(names) => names
            .iter()
            .map(|name| match name.as_deref() {
                Some("time") => time.clone(),
                Some("y") => y.clone(),
                Some("x") => x.clone(),
                _ => panic!("Unknown dimension in metadata: {:?}", name),
            })
            .collect(),
        None => {
            vec![time, x, y]
        }
    }
}

fn order_dimensions(dimension_names: &Option<Vec<DimensionName>>) -> [usize; 3] {
    match dimension_names {
        Some(names) => {
            let t_axis = names
                .iter()
                .position(|s| s.as_ref().is_some_and(|v| v == "time"))
                .unwrap_or(0);
            let x_axis = names
                .iter()
                .position(|s| s.as_ref().is_some_and(|v| v == "x"))
                .unwrap_or(1);
            let y_axis = names
                .iter()
                .position(|s| s.as_ref().is_some_and(|v| v == "y"))
                .unwrap_or(2);
            [t_axis, x_axis, y_axis]
        }
        None => [0, 1, 2],
    }
}

const EARTH_CIRCUMFERENCE: f64 = 40_075_016.685_578_5;

/// Calculates the spatial bounding box of a tile
pub fn tile_bbox(x: u32, y: u32, zoom: u8) -> [f64; 4] {
    let tile_length = EARTH_CIRCUMFERENCE / f64::from(1_u32 << zoom);
    let min_x = EARTH_CIRCUMFERENCE * -0.5 + x as f64 * tile_length;
    let max_y = EARTH_CIRCUMFERENCE * 0.5 - y as f64 * tile_length;

    [min_x, max_y - tile_length, min_x + tile_length, max_y]
}

struct TimeMetadata {
    units: String,
    epoch: DateTime<Utc>,
    _calendar: String,
}

impl TimeMetadata {
    fn datetime_to_raw(&self, val: &DateTime<Utc>) -> f64 {
        let duration = val.signed_duration_since(self.epoch);
        let secs = duration.num_seconds() as f64;
        let nanos = duration.subsec_nanos() as f64;

        let total_seconds = secs + (nanos / 1_000_000_000.0);

        let scale = match self.units.as_str() {
            "days" => 1.0 / 86400.0,
            "hours" => 1.0 / 3600.0,
            "minutes" => 1.0 / 60.0,
            "milliseconds" => 1000.0,
            "microseconds" => 1_000_000.0,
            _ => 1.0, // "seconds"
        };

        total_seconds * scale
    }

    fn parse_zarr_time_attrs(units: &str, calendar: &str) -> Result<Self, Box<dyn Error>> {
        // Split "seconds since 1970-01-01"
        let parts: Vec<&str> = units.split(" since ").collect();
        let unit_type = parts[0].to_lowercase();

        // Parse the date part
        let epoch_date =
            NaiveDate::parse_from_str(parts[1], "%Y-%m-%d").expect("Invalid epoch format");

        // Convert to UTC DateTime at midnight
        let epoch_date_time = epoch_date
            .and_hms_opt(0, 0, 0)
            .ok_or("Can not convert to date time value")?;
        let epoch = Utc.from_utc_datetime(&epoch_date_time);

        Ok(TimeMetadata {
            units: unit_type,
            epoch,
            _calendar: calendar.to_string(),
        })
    }
}

fn get_temporal_index(
    time_coords: &Array<FilesystemStore>,
    datetime: DateTime<Utc>,
) -> Result<usize, Box<dyn Error>> {
    // Load time data
    let data_type = time_coords.data_type();
    let name = data_type.name(ZarrVersion::V3);
    let data_time = match name.as_deref() {
        Some("float32") => time_coords
            .retrieve_array_subset::<ndarray::Array1<f32>>(&time_coords.subset_all())?
            .mapv(|v| v as f64),
        Some("float64") => {
            time_coords.retrieve_array_subset::<ndarray::Array1<f64>>(&time_coords.subset_all())?
        }
        _ => unimplemented!("Data type not implemented yet"),
    };

    let units = time_coords.attributes().get("units").and_then(|v| {
        if v.is_string() {
            return v.as_str();
        }
        None
    });

    let calendar = time_coords.attributes().get("calendar").and_then(|v| {
        if v.is_string() {
            return v.as_str();
        }
        None
    });

    let units = units.ok_or("Could not get time units")?;
    let calendar = calendar.ok_or("Could not get calendar metadata")?;
    let time_meta = TimeMetadata::parse_zarr_time_attrs(units, calendar)?;
    let val = time_meta.datetime_to_raw(&datetime);
    let index = find_closest_binary(&data_time, val)?;

    Ok(index)
}

fn find_closest_binary(
    time_array: &ndarray::Array1<f64>,
    target: f64,
) -> Result<usize, Box<dyn Error>> {
    let slice = time_array.as_slice().ok_or("Could not convert to slice")?;

    match slice.binary_search_by(|probe| probe.partial_cmp(&target).unwrap()) {
        Ok(index) => Ok(index),
        Err(index) => {
            // 'index' is the insertion point
            if index == 0 {
                return Ok(0);
            }
            if index >= slice.len() {
                return Ok(slice.len() - 1);
            }

            let diff_left = (target - slice[index - 1]).abs();
            let diff_right = (slice[index] - target).abs();

            if diff_left < diff_right {
                Ok(index - 1)
            } else {
                Ok(index)
            }
        }
    }
}

// TODO: move blocking CPU bound coord transformation to thread pool

// use rayon::{ThreadPool, ThreadPoolBuilder};
// use std::sync::{Arc, LazyLock};
// use tokio::sync::oneshot;
//
// // Thread pool initializes automatically on first dereference
// static REPROJECT_POOL: LazyLock<Arc<ThreadPool>> = LazyLock::new(|| {
//     let pool = ThreadPoolBuilder::new()
//         .num_threads(reproject_worker_count())
//         .thread_name(|i| format!("lazycogs-reproject-{}", i))
//         .build()
//         .expect("failed to create reproject thread pool");
//     Arc::new(pool)
// });
//
// pub async fn run_reproject<F, R>(f: F) -> R
// where
//     F: FnOnce() -> R + Send + 'static,
//     R: Send + 'static,
// {
//     let (tx, rx) = oneshot::channel();
//
//     // Accessing &*REPROJECT_POOL triggers the lazy initialization on first call
//     REPROJECT_POOL.spawn(move || {
//         let res = f();
//         let _ = tx.send(res);
//     });
//
//     rx.await.expect("worker thread panicked or dropped")
// }

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::LazyLock};

    use object_store::local::LocalFileSystem;

    use super::*;

    static ZARR_STORE: LazyLock<Arc<AsyncObjectStore<LocalFileSystem>>> = LazyLock::new(|| {
        let path = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/fixtures/zarr/latest"
        ));
        let local_store =
            LocalFileSystem::new_with_prefix(path).expect("could not create file system storage");
        Arc::new(AsyncObjectStore::new(local_store))
    });

    fn get_zarr_store() -> Arc<AsyncObjectStore<LocalFileSystem>> {
        ZARR_STORE.clone()
    }

    #[tokio::test]
    async fn test_get_spatial_transform() {
        let store = get_zarr_store();
        let transform = get_spatial_transform(store)
            .await
            .expect("could not get transform");

        assert_eq!(transform, [1000.0, 0.0, 19500.0, 0.0, -1000.0, 620500.0]);
    }

    #[tokio::test]
    async fn test_get_bbox() {
        let store = get_zarr_store();
        let bbox = get_bbox(store).await.expect("could not get bounding box");

        assert_eq!(bbox, [19500.0, 189500.0, 720500.0, 620500.0]);
    }

    // 10/558/356
    // 9/275/177
    // #[test]
    // fn test_sample_tile() {
    //     let wkt = r#"PROJCRS["MGI / Austria Lambert",BASEGEOGCRS["MGI",DATUM["Militar-Geographische Institut",ELLIPSOID["Bessel 1841",6377397.155,299.1528128,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4312]],CONVERSION["unnamed",METHOD["Lambert Conic Conformal (2SP)",ID["EPSG",9802]],PARAMETER["Latitude of false origin",47.5,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8821]],PARAMETER["Longitude of false origin",13.3333333333333,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8822]],PARAMETER["Latitude of 1st standard parallel",49,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude of 2nd standard parallel",46,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8824]],PARAMETER["Easting at false origin",400000,LENGTHUNIT["metre",1],ID["EPSG",8826]],PARAMETER["Northing at false origin",400000,LENGTHUNIT["metre",1],ID["EPSG",8827]]],CS[Cartesian,2],AXIS["northing",north,ORDER[1],LENGTHUNIT["metre",1]],AXIS["easting",east,ORDER[2],LENGTHUNIT["metre",1]],ID["EPSG",31287]]"#;
    //     let path = PathBuf::from(r".\tests\fixtures\latest");
    //     let store = Arc::new(FilesystemStore::new(&path).unwrap());
    //     let tile = TileCoord::new_checked(10, 558, 356).unwrap();
    //     let data_var = Array::open(store.clone(), "/snow_depth").unwrap();

    //     // time
    //     let time_str = "2026-02-17T00:00:00.000Z";
    //     let datetime = DateTime::parse_from_rfc3339(time_str).unwrap();
    //     let datetime_utc: DateTime<Utc> = datetime.with_timezone(&Utc);

    //     let x_coords = Array::open(store.clone(), "/x").unwrap();
    //     let y_coords = Array::open(store.clone(), "/y").unwrap();
    //     let time_coords = Array::open(store.clone(), "/time").unwrap();
    //     let res = sample_data_var(
    //         &tile,
    //         &x_coords,
    //         &y_coords,
    //         &time_coords,
    //         &data_var,
    //         wkt,
    //         datetime_utc,
    //     );
    //     assert!(res.is_ok());
    // }
}
