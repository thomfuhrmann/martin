use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use core::f64;
use image::{ImageBuffer, LumaA};
use std::{
    error::Error,
    path::{Path, PathBuf},
    sync::Arc,
};
use zarrs::{
    array::Array, filesystem::FilesystemStore, group::Group, node::NodePath, plugin::ZarrVersion,
};

pub fn open_zarr(root_path: PathBuf) -> Result<(), Box<dyn Error>> {
    let store = Arc::new(FilesystemStore::new(&root_path)?);
    let root_path = NodePath::root();
    let root_group = Group::open(store.clone(), root_path.as_str())?;
    let attributes = root_group.attributes();
    let crs_path = attributes
        .get("coordinates")
        .map(|v| v.as_str())
        .flatten()
        .map(|v| NodePath::new(&format!("/{}", v)))
        .transpose()?;

    // Get WKT for projection information
    let crs_array = crs_path
        .clone()
        .map(|p| Array::open(store.clone(), p.as_str()))
        .transpose()?;

    let crs_wkt: Option<String> = crs_array.and_then(|arr| {
        arr.attributes()
            .get("crs_wkt")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    });

    // Filter all non-dimension and non-crs variables
    let children = root_group.child_arrays()?;
    for array in &children {
        let is_data_var = array_is_data_variable(array)?;
        if is_data_var && !array_is_crs(array, &crs_path) {
            println!("{:?}", array.path().as_str());
            let path = array.path().as_str();

            if path == "/airtemp" {
                if let Some(crs_wkt) = crs_wkt {
                    let tile = Tile::new(10, 558, 356);
                    let time_str = "2026-02-03T00:00:00.000Z";
                    let datetime = DateTime::parse_from_rfc3339(time_str)?;
                    let datetime_utc: DateTime<Utc> = datetime.with_timezone(&Utc);
                    sample_data_var(
                        &tile,
                        store,
                        "/x",
                        "/y",
                        "/time",
                        array,
                        &crs_wkt,
                        datetime_utc,
                    )?;
                }
                break;
            }
        }
    }

    Ok(())
}

/// Check if the array is a NetCDF data variable
fn array_is_data_variable(array: &Array<FilesystemStore>) -> Result<bool, Box<dyn Error>> {
    let array_path = array.path().as_path();
    let dim_names = array
        .dimension_names()
        .as_ref()
        .ok_or("Could not retrieve dimension names")?;

    let is_coord = dim_names.iter().any(|dim_name| {
        if let Some(dim) = dim_name
            && array_path.ends_with(dim)
        {
            true
        } else {
            false
        }
    });

    Ok(!is_coord)
}

/// Check if array contains coordinate system information
fn array_is_crs(array: &Array<FilesystemStore>, crs_path: &Option<NodePath>) -> bool {
    let crs_path = crs_path
        .as_ref()
        .map(|p| p.as_path())
        .unwrap_or(Path::new(""));
    let array_path = array.path().as_path();
    crs_path == array_path
}

const TILE_PIXELS: u32 = 256;
const SOURCE_CRS: &str = "EPSG:3857";

#[derive(Debug)]
pub struct Tile {
    zoom: u8,
    x: u32,
    y: u32,
}

impl Tile {
    fn new(zoom: u8, x: u32, y: u32) -> Self {
        Tile { zoom, x, y }
    }
}

/// Sample from array using a coordinate transformation
fn sample_data_var(
    tile: &Tile,
    store: Arc<FilesystemStore>,
    path_x: &str,
    path_y: &str,
    path_time: &str,
    data_var: &Array<FilesystemStore>,
    wkt_str: &str,
    datetime: DateTime<Utc>,
) -> Result<(), Box<dyn Error>> {
    // Load data for x-coordinates
    let array_x = Array::open(store.clone(), path_x)?;
    let data_type = array_x.data_type();
    let name = data_type.name(ZarrVersion::V3);
    let data_x = match name.as_deref() {
        Some("int64") => array_x
            .retrieve_array_subset::<ndarray::Array1<i64>>(&[0..2])?
            .mapv(|v| v as f64),
        _ => unimplemented!("Data type not implemented yet"),
    };

    // Load data for y-coordinates
    let array_y = Array::open(store.clone(), path_y)?;
    let data_type = array_y.data_type();
    let name = data_type.name(ZarrVersion::V3);
    let data_y = match name.as_deref() {
        Some("int64") => array_y
            .retrieve_array_subset::<ndarray::Array1<i64>>(&[0..2])?
            .mapv(|v| v as f64),
        _ => unimplemented!("Data type not implemented yet"),
    };

    // Calculate spatial origin and spacing of array data
    let x_0 = data_x[0];
    let spacing_x = data_x[1] - data_x[0];

    let y_0 = data_y[0];
    let spacing_y = data_y[1] - data_y[0];

    // Initialize coordinate transformation
    let transformer = proj::Proj::try_from((SOURCE_CRS, wkt_str))?;

    // Get tile bounding box in Web Mercator
    let tile = tile_bbox(tile.x, tile.y, tile.zoom);
    let x_min_source = tile[0];
    let y_min_source = tile[1];
    let x_max_source = tile[2];
    let y_max_source = tile[3];

    // Calculate grid spacings for source CRS based on tile size
    let spacing_x_source = (x_max_source - x_min_source) / f64::from(TILE_PIXELS);
    let spacing_y_source = (y_max_source - y_min_source) / f64::from(TILE_PIXELS);

    // Transform tile bounding box corners from Web Mercator to target system
    let (bl_x_target, bl_y_target) = transformer.convert((x_min_source, y_min_source))?;
    let (br_x_target, br_y_target) = transformer.convert((x_max_source, y_min_source))?;
    let (tl_x_target, tl_y_target) = transformer.convert((x_min_source, y_max_source))?;
    let (tr_x_target, tr_y_target) = transformer.convert((x_max_source, y_max_source))?;

    // Calculate array indices of bounding box
    let bl_ix = ((bl_x_target - x_0) / spacing_x).round() as u64;
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

    // Swap indices if y-coordinate array is in descending order
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
    let target_width = array_x.shape()[0];
    let x_range_start = min_ix;
    let x_range_end = (max_ix + 1).min(target_width);
    let x_range = x_range_start..x_range_end;

    let target_height = array_y.shape()[0];
    let y_range_start = min_iy;
    let y_range_end = (max_iy + 1).min(target_height);
    let y_range = y_range_start..y_range_end;

    // Load tile data into memory
    let temporal_index = get_temporal_index(store.clone(), path_time, datetime)? as u64;
    println!("{temporal_index}");
    let tile_data = data_var.retrieve_array_subset::<ndarray::Array3<f32>>(&[
        temporal_index..temporal_index + 1,
        y_range,
        x_range,
    ])?;

    // Sample from array at tile grid points
    let width = x_range_end - x_range_start;
    let height = y_range_end - y_range_start;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sampled_data = [[None; TILE_PIXELS as usize]; TILE_PIXELS as usize];
    for i in 0..TILE_PIXELS {
        for j in 0..TILE_PIXELS {
            let x = x_min_source + (f64::from(i) + 0.5) * spacing_x_source;
            let y = y_min_source + (f64::from(j) + 0.5) * spacing_y_source;

            // Transform from Web Mercator to target system
            let (xt, yt) = transformer.convert((x, y))?;

            // Index calculation:
            // use i64 for intermediate calculations to handle coordinates
            // that might fall outside min_ix/min_iy
            let abs_ix = ((xt - x_0) / spacing_x).round() as i64;
            let abs_iy = ((yt - y_0) / spacing_y).round() as i64;

            // Map to relative index
            let rel_ix = abs_ix - min_ix as i64;
            let rel_iy = abs_iy - min_iy as i64;

            // Bounds check and sample
            if rel_ix >= 0 && rel_ix < width as i64 && rel_iy >= 0 && rel_iy < height as i64 {
                let value = tile_data[[0, rel_iy as usize, rel_ix as usize]];
                if value < min {
                    min = value;
                }
                if value > max {
                    max = value;
                }

                sampled_data[i as usize][j as usize] = Some(value);
            }
        }
    }

    // Create a new ImageBuffer.
    let mut img = ImageBuffer::new(TILE_PIXELS, TILE_PIXELS);

    // Iterate over the array and convert f64 to u8
    for (y, row) in sampled_data.iter().enumerate() {
        for (x, &value) in row.iter().enumerate() {
            // Normalize, scale to 0-255 and clamp
            let pixel_val = value
                .map(|v| (v - min) / (max - min))
                .map(|v| (v * 255.0).clamp(0.0, 255.0) as u8)
                .unwrap_or(0);

            let alpha = if value.is_some() { 255 } else { 0 };
            img.put_pixel(
                x as u32,
                (TILE_PIXELS - 1) - y as u32,
                LumaA([pixel_val, alpha]),
            );
        }
    }

    // Save image
    img.save("output.png")?;

    Ok(())
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
    calendar: String,
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
            calendar: calendar.to_string(),
        })
    }
}

fn get_temporal_index(
    store: Arc<FilesystemStore>,
    time_path: &str,
    datetime: DateTime<Utc>,
) -> Result<usize, Box<dyn Error>> {
    // Load time data
    let array_time = Array::open(store.clone(), time_path)?;
    let data_type = array_time.data_type();
    let name = data_type.name(ZarrVersion::V3);
    let data_time = match name.as_deref() {
        Some("float64") => {
            array_time.retrieve_array_subset::<ndarray::Array1<f64>>(&array_time.subset_all())?
        }
        _ => unimplemented!("Data type not implemented yet"),
    };

    let units = array_time.attributes().get("units").and_then(|v| {
        if v.is_string() {
            return v.as_str();
        } else {
            None
        }
    });

    let calendar = array_time.attributes().get("calendar").and_then(|v| {
        if v.is_string() {
            return v.as_str();
        } else {
            None
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_zarr() {
        let path = PathBuf::from(r".\tests\fixtures\latest");
        let _ = open_zarr(path);
    }

    // 10/558/356
    // 9/275/177
    #[test]
    fn test_sample_tile() {
        let wkt = r#"PROJCRS["MGI / Austria Lambert",BASEGEOGCRS["MGI",DATUM["Militar-Geographische Institut",ELLIPSOID["Bessel 1841",6377397.155,299.1528128,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4312]],CONVERSION["unnamed",METHOD["Lambert Conic Conformal (2SP)",ID["EPSG",9802]],PARAMETER["Latitude of false origin",47.5,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8821]],PARAMETER["Longitude of false origin",13.3333333333333,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8822]],PARAMETER["Latitude of 1st standard parallel",49,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude of 2nd standard parallel",46,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8824]],PARAMETER["Easting at false origin",400000,LENGTHUNIT["metre",1],ID["EPSG",8826]],PARAMETER["Northing at false origin",400000,LENGTHUNIT["metre",1],ID["EPSG",8827]]],CS[Cartesian,2],AXIS["northing",north,ORDER[1],LENGTHUNIT["metre",1]],AXIS["easting",east,ORDER[2],LENGTHUNIT["metre",1]],ID["EPSG",31287]]"#;
        let path = PathBuf::from(r".\tests\fixtures\latest");
        let store = Arc::new(FilesystemStore::new(&path).unwrap());
        let tile = Tile::new(10, 558, 356);
        let data_var = Array::open(store.clone(), "/snow_depth").unwrap();

        // time
        let time_str = "2026-02-17T00:00:00.000Z";
        let datetime = DateTime::parse_from_rfc3339(time_str).unwrap();
        let datetime_utc: DateTime<Utc> = datetime.with_timezone(&Utc);

        let res = sample_data_var(
            &tile,
            store,
            "/x",
            "/y",
            "/time",
            &data_var,
            wkt,
            datetime_utc,
        );
        assert!(res.is_ok());
    }
}
