use chrono::{DateTime, NaiveDate, TimeZone as _, Utc};
use core::f64;
use object_store::ObjectStore;
use zarrs::node::async_get_child_nodes;
use zarrs_object_store::AsyncObjectStore;
use zstd::encode_all;
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
use crate::tiles::zarr::source::TARGET_CRS;

const TILE_PIXELS: u32 = 512;
const GRID_SIZE: usize = (2 * TILE_PIXELS * TILE_PIXELS) as usize;
const EARTH_CIRCUMFERENCE: f64 = 40_075_016.685_578_5;

/// Calculates the spatial bounding box of a tile
pub fn tile_bbox(x: u32, y: u32, zoom: u8) -> [f64; 4] {
    let tile_length = EARTH_CIRCUMFERENCE / f64::from(1_u32 << zoom);
    let min_x = EARTH_CIRCUMFERENCE * -0.5 + f64::from(x) * tile_length;
    let max_y = EARTH_CIRCUMFERENCE * 0.5 - f64::from(y) * tile_length;

    [min_x, max_y - tile_length, min_x + tile_length, max_y]
}

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

/// Get the shape of the source array
pub async fn get_spatial_shape<T: ObjectStore>(
    store: Arc<AsyncObjectStore<T>>,
) -> Result<[u64; 2], ZarrError> {
    let root_group = Group::async_open(store, NodePath::root().as_str())
        .await
        .map_err(ZarrError::GroupCreateError)?;

    let transform_value = root_group
        .attributes()
        .get("spatial:shape")
        .ok_or_else(|| ZarrError::AttributeError("spatial:shape is missing".into()))?;

    serde_json::from_value::<[u64; 2]>(transform_value.clone())
        .map_err(|e| ZarrError::AttributeError(format!("Invalid spatial:shape format: {e}")))
}

/// Returns the names of the spatial dimensions
pub fn _get_spatial_dims(array: &Array<FilesystemStore>) -> Option<Vec<&str>> {
    array
        .attributes()
        .get("spatial:dimensions")
        .and_then(|val| val.as_array())
        .and_then(|dims| dims.iter().map(|dim| dim.as_str()).collect())
}

/// Returns the names of non-spatial dimensions
pub fn _get_non_spatial_dims(
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
pub async fn sample_data_var<T: ObjectStore>(
    warp_grid: Box<[i64]>,
    warp_grid_bbox: [u64; 4],
    time_coords: Option<Arc<Array<AsyncObjectStore<T>>>>,
    datetime: DateTime<Utc>,
    data_var: &Array<AsyncObjectStore<T>>,
) -> Result<Vec<u8>, ZarrError> {
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
    let x_range = warp_grid_bbox[0]..warp_grid_bbox[2];
    let y_range = warp_grid_bbox[3]..warp_grid_bbox[1];
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

    // sample from array at tile grid points
    // TODO: use no-data value
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sampled_data =
        vec![f32::NAN; TILE_PIXELS as usize * TILE_PIXELS as usize].into_boxed_slice();
    for i in 0..TILE_PIXELS {
        for j in 0..TILE_PIXELS {
            let idx = 2 * (i as usize * TILE_PIXELS as usize + j as usize);
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
                sampled_data[(j as usize) * TILE_PIXELS as usize + i as usize] = value;
            }
        }
    }

    // Cast to raw bytes
    let raw_bytes = bytemuck::cast_slice::<f32, u8>(&sampled_data);

    // Compress with Zstd
    let compressed_bytes = encode_all(raw_bytes, 3).map_err(ZarrError::EncodeError)?;
    Ok(compressed_bytes)
}

pub(crate) fn calculate_warp_grid(
    tile: TileCoord,
    src_transform: [f64; 6],
    src_crs: &str,
    src_shape: [u64; 2],
) -> Result<Option<(Box<[i64]>, [u64; 4])>, ZarrError> {
    // inverse affine transformation from spatial coordinate system to pixel grid
    let src_a = src_transform[0];
    let src_b = src_transform[1];
    let src_c = src_transform[2];
    let src_d = src_transform[3];
    let src_e = src_transform[4];
    let src_f = src_transform[5];
    let det = src_a * src_e - src_b * src_d;
    let src_inv_trafo = [
        src_e / det,
        -src_b / det,
        (src_b * src_f - src_e * src_c) / det,
        -src_d / det,
        src_a / det,
        (src_d * src_c - src_a * src_f) / det,
    ];

    // inverse spatial coordinate transformation
    let inv_trafo =
        proj::Proj::try_from((TARGET_CRS, src_crs)).map_err(ZarrError::ProjCreateError)?;

    // tile bounding box in Web Mercator
    let tile = tile_bbox(tile.x, tile.y, tile.z);
    let x_min_target = tile[0];
    let y_min_target = tile[1];
    let x_max_target = tile[2];
    let y_max_target = tile[3];

    // calculate grid scales for target CRS based on tile size
    let x_scale_target = (x_max_target - x_min_target) / f64::from(TILE_PIXELS);
    let y_scale_target = (y_max_target - y_min_target) / f64::from(TILE_PIXELS);

    let mut src_indices = vec![-1_i64; GRID_SIZE].into_boxed_slice();

    let mut tile_grid_left = -1_i64;
    let mut tile_grid_right = -1_i64;
    let mut tile_grid_bottom = -1_i64;
    let mut tile_grid_top = -1_i64;

    #[allow(clippy::cast_precision_loss)]
    let width = src_shape[1] as f64;
    #[allow(clippy::cast_precision_loss)]
    let height = src_shape[0] as f64;

    for i in 0..TILE_PIXELS {
        for j in 0..TILE_PIXELS {
            let x_target = x_min_target + (f64::from(i) + 0.5) * x_scale_target;
            let y_target = y_min_target + (f64::from(j) + 0.5) * y_scale_target;

            let (x_src, y_src) = inv_trafo
                .convert((x_target, y_target))
                .map_err(ZarrError::ProjError)?;

            let right =
                (src_inv_trafo[0] * x_src + src_inv_trafo[1] * y_src + src_inv_trafo[2]).round();
            let down =
                (src_inv_trafo[3] * x_src + src_inv_trafo[4] * y_src + src_inv_trafo[5]).round();

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

            if right >= 0 {
                tile_grid_left = if tile_grid_left == -1 {
                    right
                } else {
                    tile_grid_left.min(right)
                };
                tile_grid_right = tile_grid_right.max(right);
            }

            if down >= 0 {
                tile_grid_top = if tile_grid_top == -1 {
                    down
                } else {
                    tile_grid_top.min(down)
                };
                tile_grid_bottom = tile_grid_bottom.max(down);
            }

            let idx = 2 * (i as usize * TILE_PIXELS as usize + j as usize);

            src_indices[idx] = right;
            src_indices[idx + 1] = down;
        }
    }

    // tile grid lies outside source data
    if tile_grid_left == -1
        || tile_grid_bottom == -1
        || tile_grid_right == -1
        || tile_grid_top == -1
    {
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
