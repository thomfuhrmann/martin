use chrono::{DateTime, NaiveDate, TimeZone as _, Utc};
use core::f64;
use image::ImageFormat;
use martin_tile_utils::TileCoord;
use object_store::ObjectStore;
use std::io::Cursor;
use std::{error::Error, ops::Range, sync::Arc};
use zarrs::node::async_get_child_nodes;
use zarrs::storage::{AsyncListableStorageTraits, AsyncReadableStorageTraits};
use zarrs::{
    array::{Array, ArrayMetadata, DimensionName},
    filesystem::FilesystemStore,
    group::Group,
    node::{Node, NodeMetadata, NodePath},
    plugin::ZarrVersion,
};
use zarrs_object_store::AsyncObjectStore;
// use zstd::encode_all;

use crate::tiles::zarr::error::ZarrError;
use crate::tiles::zarr::source::{SpatialRegistration, TARGET_CRS};

pub(crate) const TILE_PIXELS: u32 = 512;
const EARTH_CIRCUMFERENCE: f64 = 40_075_016.685_578_5;

/// Calculates the spatial bounding box of a tile
pub(crate) fn tile_bbox(x: u32, y: u32, zoom: u8) -> [f64; 4] {
    let tile_length = EARTH_CIRCUMFERENCE / f64::from(1_u32 << zoom);
    let min_x = EARTH_CIRCUMFERENCE * -0.5 + f64::from(x) * tile_length;
    let max_y = EARTH_CIRCUMFERENCE * 0.5 - f64::from(y) * tile_length;

    [min_x, max_y - tile_length, min_x + tile_length, max_y]
}

/// Retrieve all data variables of this store - arrays that are not dimensions
pub(crate) async fn data_variables<T: ObjectStore>(
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
pub(crate) async fn time_coords<
    S: AsyncReadableStorageTraits + AsyncListableStorageTraits + 'static,
>(
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
pub(crate) async fn get_proj_code<T: ObjectStore>(
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
pub(crate) async fn get_spatial_transform<T: ObjectStore>(
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
pub(crate) async fn get_bbox<T: ObjectStore>(
    store: Arc<AsyncObjectStore<T>>,
) -> Result<[f64; 4], ZarrError> {
    let root_group = Group::async_open(store, NodePath::root().as_str())
        .await
        .map_err(ZarrError::GroupCreateError)?;

    let bbox = root_group
        .attributes()
        .get("spatial:bbox")
        .ok_or_else(|| ZarrError::AttributeError("spatial:transform is missing".into()))?;

    serde_json::from_value::<[f64; 4]>(bbox.clone())
        .map_err(|e| ZarrError::AttributeError(format!("Invalid spatial:transform format: {e}")))
}

/// Get the shape of the source array
pub(crate) async fn get_spatial_shape<T: ObjectStore>(
    store: Arc<AsyncObjectStore<T>>,
) -> Result<[u64; 2], ZarrError> {
    let root_group = Group::async_open(store, NodePath::root().as_str())
        .await
        .map_err(ZarrError::GroupCreateError)?;

    let spatial_shape = root_group
        .attributes()
        .get("spatial:shape")
        .ok_or_else(|| ZarrError::AttributeError("spatial:shape is missing".into()))?;

    serde_json::from_value::<[u64; 2]>(spatial_shape.clone())
        .map_err(|e| ZarrError::AttributeError(format!("Invalid spatial:shape format: {e}")))
}

/// Get the fill value of the source array
pub(crate) async fn get_fill_value<T: ObjectStore>(
    store: Arc<AsyncObjectStore<T>>,
) -> Result<f32, ZarrError> {
    let root_group = Group::async_open(store, NodePath::root().as_str())
        .await
        .map_err(ZarrError::GroupCreateError)?;

    let fill_value = root_group
        .attributes()
        .get("fill_value")
        .ok_or_else(|| ZarrError::AttributeError("fill_value is missing".into()))?;

    serde_json::from_value::<f32>(fill_value.clone())
        .map_err(|e| ZarrError::AttributeError(format!("Missing fill_value: {e}")))
}

/// Get the spatial registration of the source array
pub(crate) async fn get_spatial_registration<T: ObjectStore>(
    store: Arc<AsyncObjectStore<T>>,
) -> Result<SpatialRegistration, ZarrError> {
    let root_group = Group::async_open(store, NodePath::root().as_str())
        .await
        .map_err(ZarrError::GroupCreateError)?;

    let registration = root_group
        .attributes()
        .get("spatial:registration")
        .and_then(|val| val.as_str())
        .ok_or_else(|| ZarrError::AttributeError("spatial:registration is missing".into()))?;

    match registration {
        "node" => Ok(SpatialRegistration::Node),
        "pixel" => Ok(SpatialRegistration::Pixel),
        _ => Err(ZarrError::AttributeError(format!(
            "invalid value for spatial:registration: {registration}"
        ))),
    }
}

/// Returns the names of the spatial dimensions
pub(crate) fn _get_spatial_dims(array: &Array<FilesystemStore>) -> Option<Vec<&str>> {
    array
        .attributes()
        .get("spatial:dimensions")
        .and_then(|val| val.as_array())
        .and_then(|dims| dims.iter().map(|dim| dim.as_str()).collect())
}

/// Returns the names of non-spatial dimensions
pub(crate) fn _get_non_spatial_dims(
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

/// Sample from array using a warp-grid
pub(crate) async fn retrieve_tile_data<T: ObjectStore>(
    warp_grid_bbox: [u64; 4],
    time_coords: Option<Arc<Array<AsyncObjectStore<T>>>>,
    datetime: DateTime<Utc>,
    data_var: &Array<AsyncObjectStore<T>>,
) -> Result<ndarray::Array3<f32>, ZarrError> {
    // get time index
    let mut time_range = None;
    if let Some(arr) = time_coords {
        let temporal_index = TimeMetadata::get_temporal_index(&arr, datetime).await?;
        time_range = Some(temporal_index..temporal_index + 1);
    }

    // permute tile data to have a fixed order: time, y, x
    let dimension_names = data_var
        .dimension_names()
        .as_deref()
        .ok_or(ZarrError::DimensionError("Missing dimension names".into()))?;

    let x_end = warp_grid_bbox[2]
        .checked_add(1)
        .ok_or_else(|| ZarrError::DimensionError("x range overflow".into()))?;
    let y_end = warp_grid_bbox[1]
        .checked_add(1)
        .ok_or_else(|| ZarrError::DimensionError("y range overflow".into()))?;
    let x_range = warp_grid_bbox[0]..x_end;
    let y_range = warp_grid_bbox[3]..y_end;

    let ranges = build_ranges(dimension_names, time_range.as_ref(), y_range, x_range)?;
    let perm = order_dimensions(dimension_names);

    let tile_data = match ranges.len() {
        3 => {
            let data = data_var
                .async_retrieve_array_subset::<ndarray::Array3<f32>>(&ranges.as_slice())
                .await
                .map_err(ZarrError::ArrayError)?;

            data.permuted_axes([perm[0], perm[1], perm[2]])
        }

        2 => {
            let data = data_var
                .async_retrieve_array_subset::<ndarray::Array2<f32>>(&ranges.as_slice())
                .await
                .map_err(ZarrError::ArrayError)?;

            data.permuted_axes([perm[0], perm[1]])
                .insert_axis(ndarray::Axis(0))
        }

        _ => {
            return Err(ZarrError::DimensionError(
                "Expected 2 or 3 dimensions".into(),
            ));
        }
    };

    Ok(tile_data)
}

/// Sample the data at the warp grid points
pub(crate) fn sample_data(
    warp_grid: &[i64],
    warp_grid_bbox: [u64; 4],
    tile_data: &ndarray::Array3<f32>,
    tile_len: u32,
    fill_value: f32,
) -> Result<Vec<u8>, ZarrError> {
    // sample from array at warp grid points
    let (sampled_data, min, max) =
        sample_warp_grid(warp_grid, warp_grid_bbox, tile_data, tile_len, fill_value)?;

    // cast to raw bytes
    // let raw_bytes = bytemuck::cast_slice::<f32, u8>(&sampled_data);

    // // compress with Zstd
    // let compressed_bytes = encode_all(raw_bytes, 3).map_err(ZarrError::EncodeError)?;
    // Ok(compressed_bytes)

    sampled_data_to_png(&sampled_data, TILE_PIXELS, TILE_PIXELS, min, max)
}

fn sample_warp_grid(
    warp_grid: &[i64],
    warp_grid_bbox: [u64; 4],
    tile_data: &ndarray::Array3<f32>,
    tile_len: u32,
    fill_value: f32,
) -> Result<(Box<[f32]>, f32, f32), ZarrError> {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sampled_data =
        vec![fill_value; tile_len as usize * tile_len as usize].into_boxed_slice();

    for i in 0..tile_len {
        for j in 0..tile_len {
            let idx = 2 * (i as usize * tile_len as usize + j as usize);
            let right = warp_grid[idx];
            let down = warp_grid[idx + 1];

            // map to relative indices
            let rel_right = if right == -1 {
                right
            } else {
                right - warp_grid_bbox[0].cast_signed()
            };

            let rel_down = if down == -1 {
                down
            } else {
                down - warp_grid_bbox[3].cast_signed()
            };

            // bounds check and sample
            if rel_right != -1 && rel_down != -1 {
                let rel_down = usize::try_from(rel_down)
                    .map_err(|e| ZarrError::CastError(format!("Could not cast to usize: {e:?}")))?;
                let rel_right = usize::try_from(rel_right)
                    .map_err(|e| ZarrError::CastError(format!("Could not cast to usize: {e:?}")))?;
                let value = tile_data[[0, rel_down, rel_right]];
                if value < min {
                    min = value;
                }
                if value > max {
                    max = value;
                }
                sampled_data[(j as usize) * tile_len as usize + i as usize] = value;
            }
        }
    }

    Ok((sampled_data, min, max))
}

#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_possible_truncation)]
fn sampled_data_to_png(
    sampled_data: &[f32],
    width: u32,
    height: u32,
    min: f32,
    max: f32,
) -> Result<Vec<u8>, ZarrError> {
    let mut image = image::RgbaImage::new(width, height);

    let range = max - min;

    for (i, &value) in sampled_data.iter().enumerate() {
        let pixel = if value.is_nan() {
            [0, 0, 0, 0] // transparent
        } else {
            let v = (((value - min) / range) * 255.0).clamp(0.0, 255.0) as u8;

            [v, v, v, 255]
        };

        let x = (i as u32) % width;
        let y = (i as u32) / width;

        image.put_pixel(x, y, image::Rgba(pixel));
    }

    let mut bytes = Cursor::new(Vec::new());

    image
        .write_to(&mut bytes, ImageFormat::Png)
        .map_err(ZarrError::ImageError)?;

    Ok(bytes.into_inner())
}

pub(crate) type WarpGrid = (Box<[i64]>, [u64; 4]);

pub(crate) fn calculate_warp_grid(
    tile: TileCoord,
    src_transform: [f64; 6],
    src_crs: &str,
    src_shape: [u64; 2],
    spatial_registration: &SpatialRegistration,
) -> Result<Option<WarpGrid>, ZarrError> {
    calculate_warp_grid_for_bbox(
        tile_bbox(tile.x, tile.y, tile.z),
        TARGET_CRS,
        src_transform,
        src_crs,
        src_shape,
        TILE_PIXELS,
        spatial_registration,
    )
}

fn calculate_warp_grid_for_bbox(
    tile_bbox: [f64; 4],
    target_crs: &str,
    src_transform: [f64; 6],
    src_crs: &str,
    src_shape: [u64; 2],
    tile_len: u32,
    spatial_registration: &SpatialRegistration,
) -> Result<Option<WarpGrid>, ZarrError> {
    // inverse affine transformation from spatial coordinate system to pixel grid
    let src_inv_affine = inverse_affine(src_transform);

    // inverse spatial coordinate transformation
    let inv_trafo =
        proj::Proj::try_from((target_crs, src_crs)).map_err(ZarrError::ProjCreateError)?;

    // tile bounding box in spatial coordinates
    let x_min_target = tile_bbox[0];
    let y_min_target = tile_bbox[1];
    let x_max_target = tile_bbox[2];
    let y_max_target = tile_bbox[3];

    // calculate grid scales for target CRS based on tile size
    let x_scale_target = (x_max_target - x_min_target) / f64::from(tile_len);
    let y_scale_target = (y_max_target - y_min_target) / f64::from(tile_len);

    let mut src_indices = vec![-1_i64; (2 * tile_len * tile_len) as usize].into_boxed_slice();

    let mut tile_grid_left = i64::MAX;
    let mut tile_grid_right = i64::MIN;
    let mut tile_grid_bottom = i64::MIN;
    let mut tile_grid_top = i64::MAX;

    #[allow(clippy::cast_precision_loss)]
    let width = src_shape[1] as f64;
    #[allow(clippy::cast_precision_loss)]
    let height = src_shape[0] as f64;

    let mut has_valid_pixel = false;
    for i in 0..tile_len {
        for j in 0..tile_len {
            let x_target = x_min_target + (f64::from(i) + 0.5) * x_scale_target;
            let y_target = y_min_target + (f64::from(j) + 0.5) * y_scale_target;

            let (x_src, y_src) = inv_trafo
                .convert((x_target, y_target))
                .map_err(ZarrError::ProjError)?;

            let right = match spatial_registration {
                SpatialRegistration::Pixel => {
                    (src_inv_affine[0] * x_src + src_inv_affine[1] * y_src + src_inv_affine[2])
                        .floor()
                }
                SpatialRegistration::Node => {
                    (src_inv_affine[0] * x_src + src_inv_affine[1] * y_src + src_inv_affine[2])
                        .round()
                }
            };

            let down = match spatial_registration {
                SpatialRegistration::Pixel => {
                    (src_inv_affine[3] * x_src + src_inv_affine[4] * y_src + src_inv_affine[5])
                        .floor()
                }
                SpatialRegistration::Node => {
                    (src_inv_affine[3] * x_src + src_inv_affine[4] * y_src + src_inv_affine[5])
                        .round()
                }
            };

            #[allow(clippy::cast_possible_truncation)]
            let right = if (0.0..width).contains(&right) {
                right as i64
            } else {
                -1
            };

            #[allow(clippy::cast_possible_truncation)]
            let down = if (0.0..height).contains(&down) {
                down as i64
            } else {
                -1
            };

            if right >= 0 && down >= 0 {
                has_valid_pixel = true;
                tile_grid_left = tile_grid_left.min(right);
                tile_grid_right = tile_grid_right.max(right);
                tile_grid_top = tile_grid_top.min(down);
                tile_grid_bottom = tile_grid_bottom.max(down);
            }

            let idx = 2 * (i as usize * tile_len as usize + j as usize);

            src_indices[idx] = right;
            src_indices[idx + 1] = down;
        }
    }

    // tile grid lies outside source data
    if !has_valid_pixel {
        return Ok(None);
    }

    Ok(Some((
        src_indices,
        [
            tile_grid_left.cast_unsigned(),
            tile_grid_bottom.cast_unsigned(),
            tile_grid_right.cast_unsigned(),
            tile_grid_top.cast_unsigned(),
        ],
    )))
}

/// Calculate inverse affine transformation
fn inverse_affine(transform: [f64; 6]) -> [f64; 6] {
    let src_a = transform[0];
    let src_b = transform[1];
    let src_c = transform[2];
    let src_d = transform[3];
    let src_e = transform[4];
    let src_f = transform[5];
    let det = src_a * src_e - src_b * src_d;
    [
        src_e / det,
        -src_b / det,
        (src_b * src_f - src_e * src_c) / det,
        -src_d / det,
        src_a / det,
        (src_d * src_c - src_a * src_f) / det,
    ]
}

fn build_ranges(
    dimension_names: &[DimensionName],
    time: Option<&Range<u64>>,
    y: Range<u64>,
    x: Range<u64>,
) -> Result<Vec<Range<u64>>, ZarrError> {
    dimension_names
        .iter()
        .map(|name| match name.as_deref() {
            Some("time") => time
                .cloned()
                .ok_or_else(|| ZarrError::TimeError("Missing time array indices".into())),
            Some("y") => Ok(y.clone()),
            Some("x") => Ok(x.clone()),
            _ => Err(ZarrError::DimensionError(format!(
                "Unknown dimension: {name:?}"
            ))),
        })
        .collect()
}

fn order_dimensions(names: &[DimensionName]) -> Vec<usize> {
    ["time", "y", "x"]
        .iter()
        .filter_map(|&dimension| {
            names
                .iter()
                .position(|name| name.as_deref() == Some(dimension))
        })
        .collect()
}

struct TimeMetadata {
    units: String,
    epoch: DateTime<Utc>,
    _calendar: String,
}

impl TimeMetadata {
    fn parse_zarr_time_attrs(units: &str, calendar: &str) -> Result<Self, ZarrError> {
        // split "seconds since 1970-01-01"
        let parts: Vec<&str> = units.split(" since ").collect();
        let unit_type = parts[0].to_lowercase();

        // parse the date part
        let epoch_date =
            NaiveDate::parse_from_str(parts[1].split(' ').collect::<Vec<_>>()[0], "%Y-%m-%d")
                .map_err(ZarrError::ParseError)?;

        // convert to UTC DateTime at midnight
        let epoch_date_time = epoch_date.and_hms_opt(0, 0, 0).ok_or(ZarrError::TimeError(
            "Can not convert to date time value".into(),
        ))?;
        let epoch = Utc.from_utc_datetime(&epoch_date_time);

        Ok(Self {
            units: unit_type,
            epoch,
            _calendar: calendar.to_owned(),
        })
    }

    #[allow(clippy::cast_precision_loss)]
    fn datetime_to_raw(&self, val: &DateTime<Utc>) -> f64 {
        let duration = val.signed_duration_since(self.epoch);
        let secs = duration.num_seconds();
        let nanos = i64::from(duration.subsec_nanos());

        let total_seconds = secs + (nanos / 1_000_000_000);

        let scale = match self.units.as_str() {
            "days" => 1.0 / 86400.0,
            "hours" => 1.0 / 3600.0,
            "minutes" => 1.0 / 60.0,
            "milliseconds" => 1000.0,
            "microseconds" => 1_000_000.0,
            _ => 1.0, // "seconds"
        };

        total_seconds as f64 * scale
    }

    // TODO: parsing
    async fn get_temporal_index<T: ObjectStore>(
        time_coords: &Array<AsyncObjectStore<T>>,
        datetime: DateTime<Utc>,
    ) -> Result<u64, ZarrError> {
        // Load time data
        let data_type = time_coords.data_type();
        let name = data_type.name(ZarrVersion::V3);

        let data_time = match name.as_deref() {
            Some("float32") => time_coords
                .async_retrieve_array_subset::<ndarray::Array1<f32>>(&time_coords.subset_all())
                .await
                .map_err(ZarrError::ArrayError)?
                .mapv(f64::from),
            Some("float64") => time_coords
                .async_retrieve_array_subset::<ndarray::Array1<f64>>(&time_coords.subset_all())
                .await
                .map_err(ZarrError::ArrayError)?,
            _ => return Err(ZarrError::DimensionError("Data type not supported".into())),
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

        let units = units.ok_or(ZarrError::TimeError("Could not get time units".into()))?;
        let calendar = calendar.ok_or(ZarrError::TimeError(
            "Could not get calendar metadata".into(),
        ))?;
        let time_meta = Self::parse_zarr_time_attrs(units, calendar)?;
        let val = time_meta.datetime_to_raw(&datetime);
        let index = find_closest_binary(&data_time, val)? as u64;

        Ok(index)
    }
}

fn find_closest_binary(time_array: &ndarray::Array1<f64>, target: f64) -> Result<usize, ZarrError> {
    let slice = time_array
        .as_slice()
        .ok_or(ZarrError::TimeError("Could not convert to slice".into()))?;

    match slice.binary_search_by(|probe| {
        probe
            .partial_cmp(&target)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
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

    #[test]
    fn test_inverse_affine() {
        let transform = [2.0, 0.0, 10.0, 0.0, 3.0, 20.0];
        let inverse = inverse_affine(transform);

        let x = inverse[0] * 12.0 + inverse[1] * 23.0 + inverse[2];
        let y = inverse[3] * 12.0 + inverse[4] * 23.0 + inverse[5];

        assert!((x - 1.0).abs() < 1e-12);
        assert!((y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_warp_grid_size() {
        let (warp_grid, warp_grid_bbox) = calculate_warp_grid_for_bbox(
            [0.0, 0.0, 4.0, 4.0],
            "EPSG:4326",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "EPSG:4326",
            [4, 4],
            4,
            &SpatialRegistration::Pixel,
        )
        .expect("could not calculate warp grid")
        .expect("should be some");

        assert_eq!(warp_grid.len(), 32);
    }

    #[test]
    fn test_warp_grid_identity() {
        let result = calculate_warp_grid_for_bbox(
            [0.0, 0.0, 4.0, 4.0],
            "EPSG:4326",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "EPSG:4326",
            [4, 4],
            4,
            &SpatialRegistration::Pixel,
        )
        .expect("could not calculate warp grid");

        assert_eq!(
            result,
            Some((
                vec![
                    0, 0, 0, 1, 0, 2, 0, 3, 1, 0, 1, 1, 1, 2, 1, 3, 2, 0, 2, 1, 2, 2, 2, 3, 3, 0,
                    3, 1, 3, 2, 3, 3
                ]
                .into_boxed_slice(),
                [0, 3, 3, 0]
            ))
        );
    }

    #[test]
    fn test_warp_grid_outside() {
        let result = calculate_warp_grid_for_bbox(
            [5.0, 10.0, 7.0, 12.0],
            "EPSG:4326",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "EPSG:4326",
            [4, 4],
            4,
            &SpatialRegistration::Pixel,
        )
        .expect("could not calculate warp grid");

        assert!(result.is_none());
    }

    #[test]
    fn test_warp_grid_partial() {
        let result = calculate_warp_grid_for_bbox(
            [2.0, 2.0, 5.0, 5.0],
            "EPSG:4326",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "EPSG:4326",
            [4, 4],
            4,
            &SpatialRegistration::Pixel,
        )
        .expect("could not calculate warp grid");

        assert_eq!(
            result,
            Some((
                vec![
                    2, 2, 2, 3, 2, 3, 2, -1, 3, 2, 3, 3, 3, 3, 3, -1, 3, 2, 3, 3, 3, 3, 3, -1, -1,
                    2, -1, 3, -1, 3, -1, -1
                ]
                .into_boxed_slice(),
                [2, 3, 3, 2]
            ))
        );
    }

    #[test]
    fn test_sample_warp_grid() {
        let mut data = ndarray::Array3::<f32>::zeros((1, 4, 4));

        for y in 0..3 {
            for x in 0..3 {
                data[[0, y, x]] = (y * 10 + x) as f32;
            }
        }

        let (warp_grid, warp_grid_bbox) = calculate_warp_grid_for_bbox(
            [0.0, 0.0, 4.0, 4.0],
            "EPSG:4326",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "EPSG:4326",
            [4, 4],
            4,
            &SpatialRegistration::Pixel,
        )
        .expect("could not calculate warp grid")
        .expect("should be some");

        let (sampled_data, min, max) = sample_warp_grid(&warp_grid, warp_grid_bbox, &data, 4, 0.0)
            .expect("could not calculate warp grid");

        assert_eq!(sampled_data[6], 12.0);
        assert_eq!(min, 0.0);
        assert_eq!(max, 22.0);
    }

    // 10/558/356
    // 9/275/177
    #[tokio::test]
    async fn test_sample_tile() {
        let store = get_zarr_store();
        let data_var = Array::async_open(Arc::clone(&store), "/snow_depth")
            .await
            .unwrap();

        let tile = TileCoord::new_checked(10, 558, 356).unwrap();

        // time
        let time_str = "2026-08-13T00:00:00.000Z";
        let datetime = DateTime::parse_from_rfc3339(time_str).unwrap();
        let datetime_utc: DateTime<Utc> = datetime.with_timezone(&Utc);

        let time_coords = time_coords(Arc::clone(&store)).await.unwrap().map(Arc::new);
        let src_transform = get_spatial_transform(Arc::clone(&store)).await.unwrap();
        let src_bbox = get_bbox(Arc::clone(&store)).await.unwrap();
        let src_shape = get_spatial_shape(Arc::clone(&store)).await.unwrap();
        let src_crs = get_proj_code(Arc::clone(&store)).await.unwrap();
        let (warp_grid, warp_grid_bbox) = calculate_warp_grid(
            tile,
            src_transform,
            src_crs.as_str(),
            src_shape,
            &SpatialRegistration::Pixel,
        )
        .expect("could not calculate warp grid")
        .expect("should be some");
        let res = sample_data_var(
            warp_grid,
            warp_grid_bbox,
            time_coords,
            datetime_utc,
            &data_var,
            TILE_PIXELS,
            0.0,
        )
        .await;
        assert!(res.is_ok());
    }
}
