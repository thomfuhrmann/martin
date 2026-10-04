use chrono::{DateTime, NaiveDate, TimeZone as _, Utc};
use core::f64;
use image::{GrayAlphaImage, ImageFormat, LumaA};
use martin_tile_utils::TileCoord;
use ndarray::{Array2, ArrayD, Axis, Ix2};
use object_store::ObjectStore;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;
use tilejson::Bounds;
use zarrs::array::{DataType, FillValue};
use zarrs::group::GroupMetadata;
use zarrs::{
    array::{Array, ArrayMetadata, DimensionName},
    node::{Node, NodeMetadata, NodePath},
    plugin::ZarrVersion,
};
use zarrs_object_store::AsyncObjectStore;

use crate::tiles::zarr::error::ZarrError;
use crate::tiles::zarr::source::{SpatialRegistration, TARGET_CRS};

use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub(crate) struct Multiscales {
    pub(crate) layout: Vec<LayoutItem>,
    #[allow(dead_code)]
    pub(crate) resampling_method: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
pub(crate) struct LayoutItem {
    pub(crate) asset: String,
    #[serde(rename = "spatial:transform")]
    pub(crate) spatial_transform: Option<[f64; 6]>,
    #[serde(rename = "spatial:shape")]
    pub(crate) spatial_shape: Option<[u64; 2]>,
    pub(crate) derived_from: Option<String>,
    pub(crate) transform: Option<Transform>,
    #[allow(dead_code)]
    pub(crate) resampling_method: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
pub(crate) struct Transform {
    scale: Option<Vec<f64>>,
    #[allow(dead_code)]
    translation: Option<Vec<f64>>,
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub(crate) enum Proj {
    Code(String),
    Wkt2(String),
    Projjson(Value),
}

impl Proj {
    pub fn into_string(self) -> String {
        match self {
            Self::Code(code) => code,
            Self::Wkt2(wkt) => wkt,
            Self::Projjson(value) => value.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ResolutionLevel {
    pub(crate) scale: f64,
    pub(crate) item: LayoutItem,
}

pub(crate) const TILE_PIXELS: u32 = 512;
const EARTH_RADIUS_M: f64 = 6_371_000.0;
const EARTH_CIRCUMFERENCE: f64 = 40_075_016.685_578_5;

/// Calculate tile length for zoom level
pub(crate) fn tile_len(zoom: u8) -> f64 {
    EARTH_CIRCUMFERENCE / f64::from(1_u32 << zoom)
}

/// Calculates the spatial bounding box of a tile
pub(crate) fn tile_bbox(x: u32, y: u32, zoom: u8) -> [f64; 4] {
    let tile_length = tile_len(zoom);
    let min_x = EARTH_CIRCUMFERENCE * -0.5 + f64::from(x) * tile_length;
    let max_y = EARTH_CIRCUMFERENCE * 0.5 - f64::from(y) * tile_length;

    [min_x, max_y - tile_length, min_x + tile_length, max_y]
}

/// Test if it is a spherial CRS
pub(crate) fn is_spherical_crs(json_value: &Value) -> bool {
    let Some(target_crs) = json_value.get("target_crs") else {
        return false;
    };
    let Some(crs_type) = target_crs.get("type").and_then(|v| v.as_str()) else {
        return false;
    };

    match crs_type {
        "GeographicCRS" | "DerivedGeographicCRS" => true,
        "GeodeticCRS" | "DerivedGeodeticCRS" => {
            // ensure primary axis uses angular units
            is_angular_axis(target_crs)
        }
        "CompoundCRS" => {
            if let Some(components) = target_crs.get("components").and_then(|v| v.as_array()) {
                components.iter().any(is_spherical_crs)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Test if the CRS uses anulgar units
fn is_angular_axis(json: &Value) -> bool {
    if let Some(unit) = json
        .pointer("/coordinate_system/axis/0/unit")
        .or_else(|| json.pointer("/datum/prime_meridian/unit"))
    {
        if let Some(unit_str) = unit.as_str() {
            return unit_str.to_lowercase().contains("degree")
                || unit_str.to_lowercase().contains("radian");
        }
        if let Some(unit_name) = unit.get("name").and_then(|v| v.as_str()) {
            return unit_name.to_lowercase().contains("degree")
                || unit_name.to_lowercase().contains("radian");
        }
    }
    true
}

/// Calculate tile resolution in source coordinate system
pub(crate) fn native_tile_res(
    tile: TileCoord,
    target_crs: &str,
    src_crs: &str,
) -> Result<f64, ZarrError> {
    let bbox = tile_bbox(tile.x, tile.y, tile.z);
    let inv_trafo =
        proj::Proj::try_from((target_crs, src_crs)).map_err(ZarrError::ProjCreateError)?;

    let (x1, _) = inv_trafo
        .convert((bbox[0], bbox[1]))
        .map_err(ZarrError::ProjError)?; // bottom-left
    let (x2, _) = inv_trafo
        .convert((bbox[2], bbox[1]))
        .map_err(ZarrError::ProjError)?; // bottom-right
    let (x3, _) = inv_trafo
        .convert((bbox[0], bbox[3]))
        .map_err(ZarrError::ProjError)?; // top-left
    let (x4, _) = inv_trafo
        .convert((bbox[2], bbox[3]))
        .map_err(ZarrError::ProjError)?; // top-right

    let min_x = x1.min(x2).min(x3).min(x4);
    let max_x = x1.max(x2).max(x3).max(x4);

    let span_x = (max_x - min_x).abs();

    let res_per_pixel = span_x / (f64::from(TILE_PIXELS));

    Ok(res_per_pixel)
}

/// Retrieve bounds in EPSG:4326 from source
pub(crate) fn bounds_from_bbox(
    spatial_bbox: Option<[f64; 4]>,
    src_crs: &Proj,
) -> Result<Option<Bounds>, ZarrError> {
    if let Some(spatial_bbox) = spatial_bbox {
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
        Ok(Some(bounds))
    } else {
        Ok(None)
    }
}

type ResolutionsShape = (Vec<ResolutionLevel>, [u64; 2]);
#[allow(clippy::cast_precision_loss)]
pub(crate) fn calculate_resolution_levels<T: ObjectStore>(
    data_vars: &HashMap<String, Array<AsyncObjectStore<T>>>,
    multiscales: Option<&Multiscales>,
    base_layout: Option<&LayoutItem>,
    dimension_names: Option<&[DimensionName]>,
    spatial_dims: Option<&[String]>,
    spatial_transform: [f64; 6],
    spatial_bbox: Option<[f64; 4]>,
) -> Result<Option<ResolutionsShape>, ZarrError> {
    if let Some(multiscales) = &multiscales
        && let Some(base_layout) = base_layout
        && let Some(dimension_names) = dimension_names
        && let Some(spatial_dims) = spatial_dims
    {
        let indices = get_spatial_dims_indices(spatial_dims, dimension_names)?;
        let spatial_shape = if let Some(spatial_shape) = &base_layout.spatial_shape {
            [spatial_shape[0], spatial_shape[1]]
        } else {
            // get spatial shape from array
            let base_path = &base_layout.asset;
            let (_, arr) = data_vars
                .iter()
                .find(|(path, _)| path.starts_with(base_path))
                .expect("should be at least one variable at base resolution level");
            let shape = arr.shape();
            let spatial_shape = indices.iter().map(|&idx| shape[idx]).collect::<Vec<_>>();
            [spatial_shape[0], spatial_shape[1]]
        };

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
        let abs_scales = abs_scales(base_res, multiscales, indices.as_slice());

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

        Ok(Some((levels, spatial_shape)))
    } else {
        Ok(None)
    }
}

/// Check if the array is a data variable
pub(crate) fn is_data_variable(node: &Node) -> Result<bool, ZarrError> {
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

    if dim_names.is_empty() {
        return Ok(false);
    }

    let is_coord = dim_names.iter().any(|dim_name| path.ends_with(dim_name));
    Ok(!is_coord)
}

/// Helper function for node traversal
fn visit_nodes_recursive(node: &Node, nodes: &mut Vec<Node>) {
    for child in node.children() {
        nodes.push(child.clone());
        visit_nodes_recursive(child, nodes);
    }
}

/// Retrieve all nodes
pub(crate) async fn zarr_nodes<T: ObjectStore>(
    store: Arc<AsyncObjectStore<T>>,
) -> Result<Vec<Node>, ZarrError> {
    let mut nodes = Vec::new();
    let root_path = NodePath::root();
    let root_node = Node::async_open(Arc::clone(&store), root_path.as_str())
        .await
        .map_err(ZarrError::NodeCreateError)?;
    nodes.push(root_node.clone());
    visit_nodes_recursive(&root_node, &mut nodes);
    Ok(nodes)
}

pub(crate) fn node_attributes(node: &Node) -> Map<String, Value> {
    match node.metadata() {
        NodeMetadata::Array(ArrayMetadata::V2(metadata)) => metadata.attributes.clone(),
        NodeMetadata::Array(ArrayMetadata::V3(metadata)) => metadata.attributes.clone(),
        NodeMetadata::Group(GroupMetadata::V2(metadata)) => metadata.attributes.clone(),
        NodeMetadata::Group(GroupMetadata::V3(metadata)) => metadata.attributes.clone(),
    }
}

/// Get coordinate system definition
///
/// At least one of proj:code, proj:wkt2, or proj:projjson MUST be provided.
/// Inherited to direct child arrays of a group. Can be overriden at array level.
pub(crate) fn get_proj(node: &Node) -> Option<Proj> {
    let attributes = node_attributes(node);
    if let Some(code) = attributes.get("proj:code").and_then(|v| v.as_str()) {
        Some(Proj::Code(code.to_owned()))
    } else if let Some(wkt) = attributes.get("proj:wkt2").and_then(|v| v.as_str()) {
        Some(Proj::Wkt2(wkt.to_owned()))
    } else {
        attributes
            .get("proj:projjson")
            .map(|json| Proj::Projjson(json.clone()))
    }
}

/// Get spatial dimension names
///
/// Optional for groups and required for arrays
pub(crate) fn get_spatial_dims(node: &Node) -> Result<Option<Vec<String>>, ZarrError> {
    node_attributes(node)
        .get("spatial:dimensions")
        .map(|val| {
            serde_json::from_value::<Vec<String>>(val.clone())
                .map_err(|e| ZarrError::AttributeError(format!("Invalid spatial:dimensions: {e}")))
        })
        .transpose()
}

/// Get the indices of spatial dimensions
///
/// Each entry in spatial:dimensions MUST match one of the names declared in the array's `dimension_names` metadata field (a top-level field of the Zarr V3 array metadata, not an attribute).
pub(crate) fn get_spatial_dims_indices<T: AsRef<str>>(
    spatial_dims: &[T],
    dimension_names: &[DimensionName],
) -> Result<Vec<usize>, ZarrError> {
    spatial_dims
        .iter()
        .map(|spatial_dim| {
            let dim_str = spatial_dim.as_ref();
            dimension_names
                .iter()
                .position(|name| name.as_deref() == Some(dim_str))
                .ok_or_else(|| {
                    ZarrError::DimensionError(format!("Missing spatial dimension: {dim_str}"))
                })
        })
        .collect()
}

/// Get the names of non-spatial dimensions
pub(crate) fn get_non_spatial_dims<T: AsRef<str>>(
    spatial_dims: Option<&[T]>,
    dimension_names: Option<&[DimensionName]>,
) -> Vec<String> {
    if let Some(spatial_dims) = spatial_dims
        && let Some(dimension_names) = dimension_names
    {
        dimension_names
            .iter()
            .filter_map(|dim| dim.as_deref())
            .filter(|name| !spatial_dims.iter().any(|s| s.as_ref() == *name))
            .map(String::from)
            .collect()
    } else {
        Vec::new()
    }
}

/// Get the affine transformation from pixel space to geographic space
///
/// Required when `spatial:transform_type` is "affine" (which is the default if omitted).
/// The transform operates on array indices where (0, 0) is at the top-left corner of the top-left pixel, and (width, height) is at the bottom-right corner of the bottom-right pixel.
/// The center of the top-left pixel is at (0.5, 0.5).
pub(crate) fn get_spatial_transform(node: &Node) -> Result<Option<[f64; 6]>, ZarrError> {
    node_attributes(node)
        .get("spatial:transform")
        .map(|val| {
            serde_json::from_value::<[f64; 6]>(val.clone()).map_err(|e| {
                ZarrError::AttributeError(format!("Invalid spatial:transform format: {e}"))
            })
        })
        .transpose()
}

/// Get the bounding box
///
/// Optional
pub(crate) fn get_bbox(node: &Node) -> Result<Option<[f64; 4]>, ZarrError> {
    node_attributes(node)
        .get("spatial:bbox")
        .map(|val| {
            serde_json::from_value::<[f64; 4]>(val.clone()).map_err(|e| {
                ZarrError::AttributeError(format!("Invalid spatial:transform format: {e}"))
            })
        })
        .transpose()
}

/// Get the shape of the source array
///
/// Optional
pub(crate) fn get_spatial_shape(node: &Node) -> Result<Option<[u64; 2]>, ZarrError> {
    node_attributes(node)
        .get("spatial:shape")
        .map(|value| {
            serde_json::from_value::<[u64; 2]>(value.clone()).map_err(|e| {
                ZarrError::AttributeError(format!("Invalid spatial:shape format: {e}"))
            })
        })
        .transpose()
}

/// Get the spatial registration of the source array
///
/// Optional
pub(crate) fn get_spatial_registration(node: &Node) -> Option<SpatialRegistration> {
    let attributes = node_attributes(node);
    let registration = attributes
        .get("spatial:registration")
        .and_then(|val| val.as_str());
    match registration {
        Some("node") => Some(SpatialRegistration::Node),
        Some("pixel") => Some(SpatialRegistration::Pixel),
        _ => None,
    }
}

/// Get multiscales info if present
///
/// Optional
pub(crate) fn get_multiscales(node: &Node) -> Result<Option<Multiscales>, ZarrError> {
    node_attributes(node)
        .get("multiscales")
        .map(|val| {
            serde_json::from_value(val.clone())
                .map_err(|e| ZarrError::AttributeError(e.to_string()))
        })
        .transpose()
}

/// Calculate the resolutions in x and y
#[allow(clippy::cast_precision_loss)]
pub(crate) fn spatial_resolutions(shape: &[u64], bbox: &[f64]) -> [f64; 2] {
    let delta_x = bbox[2] - bbox[0];
    let delta_y = bbox[3] - bbox[1];
    let res_x = delta_x / (shape[1] as f64);
    let res_y = delta_y / (shape[0] as f64);
    [res_y, res_x]
}

/// Helper to create absolute scale values
fn insert_scales(
    key: &str,
    scale: &[f64],
    multiscales: &Multiscales,
    scales: &mut HashMap<String, f64>,
    indices: &[usize],
) {
    let curr_levels = multiscales
        .layout
        .iter()
        .filter(|&item| item.derived_from.as_deref() == Some(key));

    for level in curr_levels {
        if scales.get(&level.asset).is_none() {
            let current_key = &level.asset;
            let current_scale = level.transform.clone().and_then(|trans| trans.scale);
            if let Some(current_scale) = current_scale {
                let new_scale = [
                    scale[0] * current_scale[indices[0]],
                    scale[1] * current_scale[indices[1]],
                ];
                scales.insert(current_key.into(), new_scale[0].max(new_scale[1]));
                insert_scales(current_key, &new_scale[..], multiscales, scales, indices);
            }
        }
    }
}

/// Calculate absolute scales from the multiscales DAG
pub(crate) fn abs_scales(
    base_res: [f64; 2],
    multiscales: &Multiscales,
    indices: &[usize],
) -> HashMap<String, f64> {
    let mut scales = HashMap::with_capacity(multiscales.layout.len());

    let root_key = multiscales
        .layout
        .iter()
        .find(|item| item.derived_from.is_none())
        .map_or_else(String::new, |root| root.asset.clone());

    let root_scale_max = base_res[0].max(base_res[1]);

    scales.insert(root_key.clone(), root_scale_max);

    insert_scales(
        root_key.as_str(),
        &base_res[..],
        multiscales,
        &mut scales,
        indices,
    );

    scales
}

/// Find matching layout item for a given tile
pub(crate) fn select_best_level<'a>(
    xyz: TileCoord,
    levels: Option<&'a [ResolutionLevel]>,
    target_crs: &str,
    src_crs: &str,
) -> Result<Option<&'a LayoutItem>, ZarrError> {
    if let Some(levels) = levels {
        if levels.is_empty() {
            return Ok(None);
        }

        let target_res = native_tile_res(xyz, target_crs, src_crs)?;
        Ok(levels
            .iter()
            .min_by(|a, b| {
                let diff_a = (a.scale - target_res).abs();
                let diff_b = (b.scale - target_res).abs();
                diff_a
                    .partial_cmp(&diff_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|lvl| &lvl.item))
    } else {
        Ok(None)
    }
}

/// Convert fill value to `f32`
pub(crate) fn fill_value_f32(fill_value: &FillValue) -> Result<f32, ZarrError> {
    let bytes: [u8; 4] = fill_value
        .as_ne_bytes()
        .try_into()
        .map_err(ZarrError::FillValueError)?;

    Ok(f32::from_ne_bytes(bytes))
}

/// Convert fill value to `f64`
pub(crate) fn fill_value_f64(fill_value: &FillValue) -> Result<f64, ZarrError> {
    let bytes: [u8; 8] = fill_value
        .as_ne_bytes()
        .try_into()
        .map_err(ZarrError::FillValueError)?;

    Ok(f64::from_ne_bytes(bytes))
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ZarrFillValue {
    F32(f32),
    // F64(f64),
}

/// Convert based on data type
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn fill_value(
    fill_value: &FillValue,
    data_type: &DataType,
) -> Result<ZarrFillValue, ZarrError> {
    match data_type.name(ZarrVersion::V3).as_deref() {
        Some("float32") => fill_value_f32(fill_value).map(ZarrFillValue::F32),
        // Some("float64") => fill_value_f64(fill_value).map(|val| ZarrFillValue::F64(val)),
        Some("float64") => fill_value_f64(fill_value).map(|val| ZarrFillValue::F32(val as f32)),
        Some(_) => Err(ZarrError::CastError("Unsupported data type".into())),
        None => Err(ZarrError::CastError("Could not cast fill value".into())),
    }
}

/// Retrieve tile data from array
#[allow(clippy::cast_possible_truncation)]
pub(crate) async fn retrieve_tile_data<T: ObjectStore, S: AsRef<str>>(
    warp_grid_bbox: [u64; 4],
    dim_coords: HashMap<String, u64>,
    data_var: &Array<AsyncObjectStore<T>>,
    spatial_dims: &[S],
) -> Result<Array2<f32>, ZarrError> {
    let Some(dimension_names) = data_var.dimension_names() else {
        return Err(ZarrError::DimensionError(
            "missing dimension names".to_owned(),
        ));
    };
    let spatial_indices = get_spatial_dims_indices(spatial_dims, dimension_names)?;

    let x_end = warp_grid_bbox[2]
        .checked_add(1)
        .ok_or_else(|| ZarrError::DimensionError("x range overflow".into()))?;
    let y_end = warp_grid_bbox[3]
        .checked_add(1)
        .ok_or_else(|| ZarrError::DimensionError("y range overflow".into()))?;
    let x_range = warp_grid_bbox[0]..x_end;
    let y_range = warp_grid_bbox[1]..y_end;

    let ranges: Vec<_> = dimension_names
        .iter()
        .flatten()
        .filter_map(|name| {
            if let Some(&coord) = dim_coords.get(name) {
                return Some(coord..coord + 1);
            }

            match spatial_dims
                .iter()
                .position(|s| s.as_ref() == name.as_str())
            {
                Some(0) => Some(y_range.clone()),
                Some(_) => Some(x_range.clone()),
                None => None,
            }
        })
        .collect();

    let full_subset = match data_var.data_type().name(ZarrVersion::V3).as_deref() {
        Some("float32") => data_var
            .async_retrieve_array_subset::<ArrayD<f32>>(&ranges)
            .await
            .map_err(ZarrError::ArrayError)?,
        Some("float64") => data_var
            .async_retrieve_array_subset::<ArrayD<f64>>(&ranges)
            .await
            .map_err(ZarrError::ArrayError)?
            .mapv(|val| val as f32),
        _ => return Err(ZarrError::CastError("Unimplemented data type".into())),
    };

    let mut squeezed = full_subset;
    for (idx, _) in dimension_names.iter().enumerate() {
        // remove non-spatial axis
        if spatial_indices
            .iter()
            .find(|spatial| **spatial == idx)
            .is_none()
        {
            squeezed = squeezed.remove_axis(Axis(idx));
        }
    }

    let tile_data = if spatial_indices[0] < spatial_indices[1] {
        // already in [Y, X] orientation
        squeezed
            .into_dimensionality::<Ix2>()
            .map_err(|_shape_error| {
                ZarrError::DimensionError("Expected exactly 2 remaining dimensions".to_owned())
            })?
    } else {
        // array is in [X, Y] orientation - transpose into [Y, X]
        squeezed
            .into_dimensionality::<Ix2>()
            .map_err(|_shape_error| {
                ZarrError::DimensionError("Expected exactly 2 remaining dimensions".to_owned())
            })?
            .reversed_axes()
    };

    Ok(tile_data)
}

/// Sample the data at the warp grid points
pub(crate) fn sample_data(
    warp_grid: &[i64],
    warp_grid_bbox: [u64; 4],
    tile_data: &Array2<f32>,
    tile_len: u32,
    fill_value: ZarrFillValue,
    min: f32,
    max: f32,
) -> Result<Vec<u8>, ZarrError> {
    // sample from array at warp grid points
    let (sampled_data, _min, _max) =
        sample_at_warp_grid(warp_grid, warp_grid_bbox, tile_data, tile_len, fill_value)?;

    // cast to raw bytes
    // let raw_bytes = bytemuck::cast_slice::<f32, u8>(&sampled_data);

    // // compress with Zstd
    // let compressed_bytes = encode_all(raw_bytes, 3).map_err(ZarrError::EncodeError)?;
    // Ok(compressed_bytes)

    sampled_data_to_png(
        &sampled_data,
        TILE_PIXELS,
        TILE_PIXELS,
        min,
        max,
        fill_value,
    )
}

fn sample_at_warp_grid(
    warp_grid: &[i64],
    warp_grid_bbox: [u64; 4],
    tile_data: &Array2<f32>,
    tile_len: u32,
    fill_value: ZarrFillValue,
) -> Result<(Box<[f32]>, f32, f32), ZarrError> {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sampled_data = match fill_value {
        ZarrFillValue::F32(fill_value) => {
            vec![fill_value; tile_len as usize * tile_len as usize].into_boxed_slice()
        }
    };

    for i in 0..tile_len {
        for j in 0..tile_len {
            let idx = 2 * (j as usize * tile_len as usize + i as usize);

            let right = warp_grid[idx];
            let down = warp_grid[idx + 1];

            if right == -1 || down == -1 {
                continue;
            }

            let rel_right = right - warp_grid_bbox[0].cast_signed();
            let rel_down = down - warp_grid_bbox[1].cast_signed();

            // bounds check and sample
            // TODO: account for dimension order
            if rel_right >= 0 && rel_down >= 0 {
                let rel_down = usize::try_from(rel_down)
                    .map_err(|e| ZarrError::CastError(format!("Could not cast to usize: {e:?}")))?;
                let rel_right = usize::try_from(rel_right)
                    .map_err(|e| ZarrError::CastError(format!("Could not cast to usize: {e:?}")))?;
                let value = tile_data[[rel_down, rel_right]];
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
    fill_value: ZarrFillValue,
) -> Result<Vec<u8>, ZarrError> {
    let expected_len = (width * height) as usize;
    if sampled_data.len() != expected_len {
        return Err(ZarrError::DimensionError(format!(
            "Buffer size {} does not match dimensions {}x{}",
            sampled_data.len(),
            width,
            height
        )));
    }

    let mut image = GrayAlphaImage::new(width, height);

    let range = if max > min { max - min } else { 1.0 };

    let ZarrFillValue::F32(target_fill) = fill_value;

    for (i, &value) in sampled_data.iter().enumerate() {
        let is_nodata = value.is_nan()
            || (target_fill.is_nan() && value.is_nan())
            || (!target_fill.is_nan()
                && (value - target_fill).abs() <= f32::EPSILON * target_fill.abs().max(1.0));

        let pixel = if is_nodata {
            [0, 0]
        } else {
            // Normalize against min/max bounds
            let normalized = ((value - min) / range).clamp(0.0, 1.0);
            let v = (normalized * 255.0).round() as u8;

            [v, 255]
        };

        let x = (i as u32) % width;
        let y = (i as u32) / width;

        image.put_pixel(x, y, LumaA(pixel));
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
        src_crs,
        src_transform,
        src_shape,
        TILE_PIXELS,
        spatial_registration,
    )
}

#[allow(clippy::cast_precision_loss)]
fn calculate_warp_grid_for_bbox(
    tile_bbox: [f64; 4],
    target_crs: &str,
    src_crs: &str,
    src_transform: [f64; 6],
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

    let src_uses_0_360 = src_transform[2] > 180.0
        || (src_transform[2] + src_transform[0] * src_shape[1] as f64) > 180.0;

    let mut has_valid_pixel = false;
    for i in 0..tile_len {
        for j in 0..tile_len {
            let x_target = x_min_target + (f64::from(i) + 0.5) * x_scale_target;
            let y_target = y_max_target - (f64::from(j) + 0.5) * y_scale_target;

            let (mut x_src, y_src) = inv_trafo
                .convert((x_target, y_target))
                .map_err(ZarrError::ProjError)?;

            if src_uses_0_360 && x_src < 0.0 {
                x_src += 360.0;
            }

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

            let idx = 2 * (j as usize * tile_len as usize + i as usize);

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
            tile_grid_left.cast_unsigned(),   // col_min
            tile_grid_top.cast_unsigned(),    // row_min (top)
            tile_grid_right.cast_unsigned(),  // col_max
            tile_grid_bottom.cast_unsigned(), // row_max (bottom)
        ],
    )))
}

/// Calculates inverse affine transformation
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

/// Metadata for temporal units and epoch
#[derive(Debug, Clone)]
pub(crate) struct TimeMetadata {
    pub(crate) units: String,
    pub(crate) epoch: DateTime<Utc>,
    #[allow(dead_code)]
    pub(crate) calendar: String,
}

/// Metadata for dimensions
#[derive(Debug, Clone)]
pub(crate) enum DimMeta {
    Time(TimeMetadata),
}

/// An index for dimensional coords
#[derive(Debug, Clone)]
pub(crate) struct DimIndex {
    /// In-memory sorted array of timestamps
    pub(crate) values: Vec<f64>,
    pub(crate) meta: DimMeta,
}

impl DimIndex {
    #[allow(clippy::cast_lossless)]
    #[allow(clippy::cast_precision_loss)]
    pub async fn load<T: ObjectStore>(
        coords: &Array<AsyncObjectStore<T>>,
    ) -> Result<Self, ZarrError> {
        let data_type = coords.data_type();
        let name = data_type.name(ZarrVersion::V3);

        let data = match name.as_deref() {
            Some("int32") => coords
                .async_retrieve_array_subset::<ndarray::Array1<i32>>(&coords.subset_all())
                .await
                .map_err(ZarrError::ArrayError)?
                .mapv(|val| val as f64),
            Some("int64") => coords
                .async_retrieve_array_subset::<ndarray::Array1<i64>>(&coords.subset_all())
                .await
                .map_err(ZarrError::ArrayError)?
                .mapv(|val| val as f64),
            Some("float32") => coords
                .async_retrieve_array_subset::<ndarray::Array1<f32>>(&coords.subset_all())
                .await
                .map_err(ZarrError::ArrayError)?
                .mapv(f64::from),
            Some("float64") => coords
                .async_retrieve_array_subset::<ndarray::Array1<f64>>(&coords.subset_all())
                .await
                .map_err(ZarrError::ArrayError)?,
            _ => return Err(ZarrError::DimensionError("Data type not supported".into())),
        };

        // Extract metadata
        let attrs = coords.attributes();
        let units = attrs
            .get("units")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ZarrError::TimeError("Could not get time units".into()))?;

        let calendar = attrs
            .get("calendar")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ZarrError::TimeError("Could not get calendar metadata".into()))?;

        let time_meta = TimeMetadata::parse_zarr_time_attrs(units, calendar)?;
        let (values, _offset) = data.into_raw_vec_and_offset();

        Ok(Self {
            values,
            meta: DimMeta::Time(time_meta),
        })
    }

    /// Get index of datetime along temporal dimension
    pub fn get_index(&self, val: f64) -> u64 {
        self.find_closest_binary(val) as u64
    }

    /// Binary search helper function
    fn find_closest_binary(&self, target: f64) -> usize {
        let array = &self.values;
        match array.binary_search_by(|probe| {
            probe
                .partial_cmp(&target)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(index) => index,
            Err(index) => {
                // 'index' is the insertion point
                if index == 0 {
                    return 0;
                }
                if index >= array.len() {
                    return array.len() - 1;
                }

                let diff_left = (target - array[index - 1]).abs();
                let diff_right = (array[index] - target).abs();

                if diff_left < diff_right {
                    index - 1
                } else {
                    index
                }
            }
        }
    }
}

impl TimeMetadata {
    fn parse_zarr_time_attrs(units: &str, calendar: &str) -> Result<Self, ZarrError> {
        // split "seconds since 1970-01-01"
        let parts: Vec<&str> = units.split(" since ").collect();
        let unit_type = parts[0].to_lowercase();

        // parse the date part
        let epoch_date =
            NaiveDate::parse_from_str(parts[1].split(' ').collect::<Vec<_>>()[0], "%Y-%m-%d")
                .map_err(ZarrError::ChronoError)?;

        // convert to UTC DateTime at midnight
        let epoch_date_time = epoch_date.and_hms_opt(0, 0, 0).ok_or(ZarrError::TimeError(
            "Can not convert to date time value".into(),
        ))?;
        let epoch = Utc.from_utc_datetime(&epoch_date_time);

        Ok(Self {
            units: unit_type,
            epoch,
            calendar: calendar.to_owned(),
        })
    }

    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn datetime_to_raw(&self, val: &DateTime<Utc>) -> f64 {
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
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::LazyLock};

    use object_store::local::LocalFileSystem;

    use super::*;

    static ZARR_STORE: LazyLock<Arc<AsyncObjectStore<LocalFileSystem>>> = LazyLock::new(|| {
        let path = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/fixtures/zarr/pyramid.zarr"
        ));
        let local_store =
            LocalFileSystem::new_with_prefix(path).expect("could not create file system storage");
        Arc::new(AsyncObjectStore::new(local_store))
    });

    fn get_zarr_store() -> Arc<AsyncObjectStore<LocalFileSystem>> {
        ZARR_STORE.clone()
    }

    #[tokio::test]
    async fn test_data_vars() {
        let store = get_zarr_store();
        let nodes = zarr_nodes(store).await.expect("should have data variables");
        let paths = nodes
            .into_iter()
            .map(|node| node.path().clone())
            .collect::<Vec<_>>();
        assert_eq!(
            paths,
            vec![
                NodePath::new("/0/air").expect("should be valid"),
                NodePath::new("/1/air").expect("should be valid")
            ]
        );
    }

    #[tokio::test]
    async fn test_projjson() {
        let crs = proj::Proj::new_known_crs("EPSG:4326", "EPSG:4326", None).expect("crs");
        let json_str = crs.to_projjson(None, None, None).expect("projjson");
        let json_value: Value = serde_json::from_str(&json_str).expect("json value");
        let target_crs = json_value.get("target_crs").expect("target crs");
        let crs_type = target_crs.get("type").expect("type");
        assert_eq!(crs_type, Some("GeographicCRS"));
    }

    #[tokio::test]
    async fn test_proj() {
        let store = get_zarr_store();
        let root_node = Node::async_open(Arc::clone(&store), "/")
            .await
            .expect("should have root group");
        let src_crs = get_proj(&root_node).expect("should have crs");
        assert_eq!(src_crs, Proj::Code("EPSG:4326".into()));
    }

    #[tokio::test]
    async fn test_spatial_res() {
        let store = get_zarr_store();
        let root_node = Node::async_open(Arc::clone(&store), "/")
            .await
            .expect("should have root group");
        let bbox = get_bbox(&root_node)
            .expect("could not get bounding box")
            .expect("should have bounding box");
        let shape = get_spatial_shape(&root_node)
            .expect("could not get spatial shape")
            .expect("could not get spatial shape");
        let res = spatial_resolutions(&shape, &bbox);
        assert_eq!(res, [2.5, 2.5]);
    }

    #[tokio::test]
    async fn test_get_indices() {
        let store = get_zarr_store();
        let data_var = Array::async_open(Arc::clone(&store), "/0/air")
            .await
            .expect("should have array");
        let root_node = Node::async_open(Arc::clone(&store), "/")
            .await
            .expect("should have root group");
        let spatial_dims = get_spatial_dims(&root_node)
            .expect("should have spatial:dimensions")
            .expect("should have spatial:dimensions");
        let dimension_names = data_var
            .dimension_names()
            .as_deref()
            .expect("should have dimension names");
        let indices = get_spatial_dims_indices(spatial_dims.as_slice(), dimension_names)
            .expect("should have indices");
        assert_eq!(indices, vec![1, 2]);
    }

    #[tokio::test]
    async fn test_abs_scales() {
        let store = get_zarr_store();
        let data_var = Array::async_open(Arc::clone(&store), "/0/air")
            .await
            .expect("should have array");
        let root_node = Node::async_open(Arc::clone(&store), "/")
            .await
            .expect("should have root group");
        let spatial_dims = get_spatial_dims(&root_node)
            .expect("should have spatial:dimensions")
            .expect("should have spatial:dimensions");
        let dimension_names = data_var
            .dimension_names()
            .as_deref()
            .expect("should have dimension names");
        let indices = get_spatial_dims_indices(spatial_dims.as_slice(), dimension_names)
            .expect("should have indices");
        let multiscales = get_multiscales(&root_node)
            .expect("could not get transform")
            .expect("should have multiscales");
        let scales = abs_scales([1.0, 1.0], &multiscales, &indices[..]);
        let first = scales.get("1");
        assert_eq!(first, Some(&2.5));
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
        let (warp_grid, _warp_grid_bbox) = calculate_warp_grid_for_bbox(
            [0.0, 0.0, 4.0, 4.0],
            "EPSG:4326",
            "EPSG:4326",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
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
            "EPSG:4326",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
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
            "EPSG:4326",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
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
            "EPSG:4326",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
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
    fn test_sample_at_warp_grid() {
        let mut data = Array2::<f32>::zeros((4, 4));

        for y in 0..3 {
            for x in 0..3 {
                data[[y, x]] = (y * 10 + x) as f32;
            }
        }

        let (warp_grid, warp_grid_bbox) = calculate_warp_grid_for_bbox(
            [0.0, 0.0, 4.0, 4.0],
            "EPSG:4326",
            "EPSG:4326",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [4, 4],
            4,
            &SpatialRegistration::Pixel,
        )
        .expect("could not calculate warp grid")
        .expect("should be some");

        let (sampled_data, min, max) = sample_at_warp_grid(
            &warp_grid,
            warp_grid_bbox,
            &data,
            4,
            ZarrFillValue::F32(0.0),
        )
        .expect("could not calculate warp grid");

        assert_eq!(sampled_data[6], 22.0);
        assert_eq!(min, 0.0);
        assert_eq!(max, 22.0);
    }

    #[tokio::test]
    async fn test_fill_value() {
        let store = get_zarr_store();
        let data_var = Array::async_open(Arc::clone(&store), "/0/air")
            .await
            .unwrap();
        let fill_value = fill_value(data_var.fill_value(), data_var.data_type())
            .expect("should have fill value");
        match fill_value {
            ZarrFillValue::F32(val) => assert!(val.is_nan()),
        }
    }

    #[tokio::test]
    async fn test_calculate_warp_grid() {
        let (warp_grid, _warp_grid_bbox) = calculate_warp_grid(
            TileCoord { z: 5, x: 3, y: 10 },
            [5.0, 0.0, 198.75, 0.0, -5.0, 76.25],
            "EPSG:4326",
            [12, 26],
            &SpatialRegistration::Pixel,
        )
        .expect("could not calculate warp grid")
        .expect("should be some");

        println!("{warp_grid:?}");
    }

    #[tokio::test]
    async fn test_sample_tile() {
        let store = get_zarr_store();
        let data_var = Array::async_open(Arc::clone(&store), "/0/air")
            .await
            .unwrap();

        let tile = TileCoord::new_checked(6, 34, 22).unwrap();

        let root_node = Node::async_open(Arc::clone(&store), "/")
            .await
            .expect("should have root group");

        // time
        let time_str = "2026-08-13T00:04:00.000Z";
        let datetime = DateTime::parse_from_rfc3339(time_str)
            .expect("msg")
            .with_timezone(&Utc);
        let time_array = Arc::new(
            Array::async_open(Arc::clone(&store), "/0/time")
                .await
                .map_err(ZarrError::ArrayCreateError)
                .expect("should open"),
        );
        let temporal_index = DimIndex::load(&time_array).await.expect("should load");

        let src_transform = get_spatial_transform(&root_node)
            .expect("could not get spatial:transform")
            .expect("should have spatial transform");
        let src_shape = get_spatial_shape(&root_node)
            .expect("could not get spatial shpae")
            .expect("could not get spatial shape");
        let src_crs = get_proj(&root_node).expect("should have crs");
        let (warp_grid, warp_grid_bbox) = calculate_warp_grid(
            tile,
            src_transform,
            &src_crs.into_string(),
            src_shape,
            &SpatialRegistration::Pixel,
        )
        .expect("could not calculate warp grid")
        .expect("should be some");
        // let tile_data = retrieve_tile_data(
        //     warp_grid_bbox,
        //     Some(&temporal_index),
        //     datetime,
        //     &data_var,
        //     &[0, 1, 2],
        // )
        // .await
        // .expect("could not retrieve tile data");
        // let res = sample_data(
        //     &warp_grid,
        //     warp_grid_bbox,
        //     &tile_data,
        //     TILE_PIXELS,
        //     ZarrFillValue::F32(0.0),
        //     0.1,
        //     90.7,
        // );

        // assert!(res.is_ok());
    }
}
