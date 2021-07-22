use crate::collect_points::ResultCollector;
use anyhow::{anyhow, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap::MmapOptions;
use pasture_core::{
    containers::{InterleavedPointBufferExt, InterleavedVecPointStorage},
    layout::PointType,
    math::AABB,
    nalgebra::{Point3, Vector3},
};
use pasture_io::{
    base::PointReader,
    las::LASReader,
    las_rs::{raw, Header},
};
use readers::Point;
use std::fs::File;
use std::io::SeekFrom;
use std::io::{Cursor, Seek};
use std::ops::Range;
use std::path::Path;

/// Open a file reader to the given file
fn open_file_reader<P: AsRef<Path>>(path: P) -> Result<Cursor<memmap::Mmap>> {
    let file = File::open(path)?;
    unsafe {
        let mmapped_file = MmapOptions::new().map(&file)?;
        let cursor = Cursor::new(mmapped_file);
        Ok(cursor)
    }
}

fn parse_las_header<R: std::io::Read>(reader: R) -> Result<raw::Header> {
    let raw_header = raw::Header::read_from(reader)?;
    Ok(raw_header)
}

fn las_offset_to_color(point_record_format: u8) -> Option<u64> {
    match point_record_format {
        2 => Some(20),
        3 => Some(28),
        5 => Some(28),
        _ => None,
    }
}

/**
 * Each type of search is implemented twice, once by using the pointstream crate (simulating a real-world use-case)
 * and once with a hand-rolled solution (simulating maximum efficiency)
 */

pub fn search_las_file_by_bounds_optimized<P: AsRef<Path>>(
    path: P,
    bounds: &AABB<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    let mut reader = open_file_reader(path.as_ref())?;

    let raw_header = parse_las_header(&mut reader)?;
    let header = Header::from_raw(raw_header.clone())?;
    let file_bounds = AABB::from_min_max(
        Point3::new(
            header.bounds().min.x,
            header.bounds().min.y,
            header.bounds().min.z,
        ),
        Point3::new(
            header.bounds().max.x,
            header.bounds().max.y,
            header.bounds().max.z,
        ),
    );
    println!("Point record size: {}", raw_header.point_data_record_length);
    let color_offset = las_offset_to_color(
        header
            .point_format()
            .to_u8()
            .context("Invalid point record format")?,
    );
    let color_offset_from_classification = color_offset.map(|o| o - 16); //Classification is always 16 bytes (in all non-extended LAS formats, which are the only ones supported in this experiment)

    if !file_bounds.intersects(bounds) {
        return Ok(());
    }

    // Convert bounds of query area into integer coordinates in local space of file. This makes intersection
    // checks very fast because they can be done on integer values
    let query_bounds_local = AABB::<i64>::from_min_max(
        Point3::<i64>::new(
            ((bounds.min().x - raw_header.x_offset) / raw_header.x_scale_factor) as i64,
            ((bounds.min().y - raw_header.y_offset) / raw_header.x_scale_factor) as i64,
            ((bounds.min().z - raw_header.z_offset) / raw_header.x_scale_factor) as i64,
        ),
        Point3::<i64>::new(
            ((bounds.max().x - raw_header.x_offset) / raw_header.x_scale_factor) as i64,
            ((bounds.max().y - raw_header.y_offset) / raw_header.y_scale_factor) as i64,
            ((bounds.max().z - raw_header.z_offset) / raw_header.z_scale_factor) as i64,
        ),
    );

    for point_idx in 0..header.number_of_points() {
        let point_offset: u64 = point_idx as u64 * raw_header.point_data_record_length as u64
            + raw_header.offset_to_point_data as u64;
        reader.seek(SeekFrom::Start(point_offset))?;

        let point_x = reader.read_i32::<LittleEndian>()? as i64;
        if point_x < query_bounds_local.min().x || point_x > query_bounds_local.max().x {
            continue;
        }

        let point_y = reader.read_i32::<LittleEndian>()? as i64;
        if point_y < query_bounds_local.min().y || point_y > query_bounds_local.max().y {
            continue;
        }

        let point_z = reader.read_i32::<LittleEndian>()? as i64;
        if point_z < query_bounds_local.min().z || point_z > query_bounds_local.max().z {
            continue;
        }

        // Seek to classification
        reader.seek(SeekFrom::Current(3))?;

        let class = reader.read_u8()?;

        // Seek to color (if it exists)
        let color = if let Some(color_offset) = color_offset_from_classification {
            reader.seek(SeekFrom::Current(color_offset as i64))?;
            let r = reader.read_u16::<LittleEndian>()?;
            let g = reader.read_u16::<LittleEndian>()?;
            let b = reader.read_u16::<LittleEndian>()?;
            Vector3::new(r, g, b)
        } else {
            Vector3::new(0, 0, 0)
        };

        result_collector.collect_one(Point {
            position: Vector3::new(
                (point_x as f64 * raw_header.x_scale_factor) + raw_header.x_offset,
                (point_y as f64 * raw_header.y_scale_factor) + raw_header.y_offset,
                (point_z as f64 * raw_header.z_scale_factor) + raw_header.z_offset,
            ),
            classification: class,
            color,
        });
    }
    Ok(())
}

pub fn search_las_file_by_bounds<P: AsRef<Path>>(
    path: P,
    bounds: &AABB<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    let mmap = open_file_reader(path)?;

    let mut reader = LASReader::from_read(mmap, false)?;

    let metadata = reader.get_metadata().clone();
    if !metadata
        .bounds()
        .expect("No bounds found in LAS file")
        .intersects(&bounds)
    {
        return Ok(());
    }

    let number_of_points = metadata
        .number_of_points()
        .expect("No number of points found in LAS file");

    // Read in chunks of fixed size
    let chunk_size = 65536; //24 bytes per point ^= ~1.5MiB
    let mut point_buffer = InterleavedVecPointStorage::with_capacity(chunk_size, Point::layout());

    let num_chunks = (number_of_points + chunk_size - 1) / chunk_size;
    for idx in 0..num_chunks {
        let points_in_chunk = usize::min(chunk_size, number_of_points - (idx * chunk_size));
        reader.read_into(&mut point_buffer, points_in_chunk)?;

        point_buffer
            .get_points_ref::<Point>(0..points_in_chunk)
            .iter()
            .filter(|point| bounds.contains(&point.position.into()))
            .for_each(|point| {
                result_collector.collect_one(point.clone());
            });
    }
    Ok(())
}

pub fn search_las_file_by_classification_optimized<P: AsRef<Path>>(
    path: P,
    class: u8,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    let mut reader = open_file_reader(&path)?;

    let raw_header = parse_las_header(&mut reader)?;
    let header = Header::from_raw(raw_header.clone())?;

    let offset_to_classification_in_point = match raw_header.point_data_record_format {
        0..=5 => 15,
        6..=10 => 16,
        _ => {
            return Err(anyhow!(
                "Invalid LAS format {} in file {}",
                raw_header.point_data_record_format,
                path.as_ref().display()
            ))
        }
    };

    let offset_to_color_in_point = match raw_header.point_data_record_format {
        2 => Some(20),
        3 => Some(28),
        5 => Some(28),
        _ => None,
    };

    for point_idx in 0..header.number_of_points() {
        let point_offset: u64 = point_idx as u64 * raw_header.point_data_record_length as u64
            + raw_header.offset_to_point_data as u64;
        reader.seek(SeekFrom::Start(
            point_offset + offset_to_classification_in_point,
        ))?;

        let classification = reader.read_u8()?;
        if classification != class {
            continue;
        }

        // Read position
        reader.seek(SeekFrom::Start(point_offset))?;
        let point_x = reader.read_i32::<LittleEndian>()?;
        let point_y = reader.read_i32::<LittleEndian>()?;
        let point_z = reader.read_i32::<LittleEndian>()?;

        // Try to read color (if supported)
        let color = if let Some(color_offset) = offset_to_color_in_point {
            reader.seek(SeekFrom::Start(point_offset + color_offset))?;
            let r = reader.read_u16::<LittleEndian>()?;
            let g = reader.read_u16::<LittleEndian>()?;
            let b = reader.read_u16::<LittleEndian>()?;
            Vector3::new(r, g, b)
        } else {
            Vector3::new(0, 0, 0)
        };

        result_collector.collect_one(Point {
            position: Vector3::new(
                (point_x as f64 * raw_header.x_scale_factor) + raw_header.x_offset,
                (point_y as f64 * raw_header.y_scale_factor) + raw_header.y_offset,
                (point_z as f64 * raw_header.z_scale_factor) + raw_header.z_offset,
            ),
            classification: classification,
            color,
        });
    }
    Ok(())
}

pub fn search_las_file_by_classification<P: AsRef<Path>>(
    path: P,
    class: u8,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    let mmap = open_file_reader(path)?;

    let mut reader = LASReader::from_read(mmap, false)?;

    let metadata = reader.get_metadata().clone();
    let number_of_points = metadata
        .number_of_points()
        .expect("No number of points found in LAS file");

    // Read in chunks of fixed size
    let chunk_size = 65536; //24 bytes per point ^= ~1.5MiB
    let mut point_buffer = InterleavedVecPointStorage::with_capacity(chunk_size, Point::layout());

    let num_chunks = (number_of_points + chunk_size - 1) / chunk_size;
    for idx in 0..num_chunks {
        let points_in_chunk = usize::min(chunk_size, number_of_points - (idx * chunk_size));
        reader.read_into(&mut point_buffer, points_in_chunk)?;

        point_buffer
            .get_points_ref::<Point>(0..points_in_chunk)
            .iter()
            .filter(|point| point.classification == class)
            .for_each(|point| {
                result_collector.collect_one(point.clone());
            });
    }
    Ok(())
}

pub fn _search_las_file_by_time_range_optimized<P: AsRef<Path>>(
    path: P,
    time_range: Range<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    let mut reader = open_file_reader(&path)?;

    let raw_header = parse_las_header(&mut reader)?;
    let header = Header::from_raw(raw_header.clone())?;

    let offset_to_gps_time_in_point = match raw_header.point_data_record_format {
        0 => {
            return Err(anyhow!(
                "File {} does not contain GPS times!",
                path.as_ref().display()
            ))
        }
        1 => 20,
        2 => {
            return Err(anyhow!(
                "File {} does not contain GPS times!",
                path.as_ref().display()
            ))
        }
        3..=5 => 20,
        6..=10 => 22,
        _ => {
            return Err(anyhow!(
                "Invalid LAS format {} in file {}",
                raw_header.point_data_record_format,
                path.as_ref().display()
            ))
        }
    };

    for point_idx in 0..header.number_of_points() {
        let point_offset: u64 = point_idx as u64 * raw_header.point_data_record_length as u64
            + raw_header.offset_to_point_data as u64;
        reader.seek(SeekFrom::Start(point_offset + offset_to_gps_time_in_point))?;

        let gps_time = reader.read_f64::<LittleEndian>()?;
        if !time_range.contains(&gps_time) {
            continue;
        }

        // Now we read XYZ
        reader.seek(SeekFrom::Start(point_offset))?;
        let point_x = reader.read_i32::<LittleEndian>()?;
        let point_y = reader.read_i32::<LittleEndian>()?;
        let point_z = reader.read_i32::<LittleEndian>()?;

        result_collector.collect_one(Point {
            position: Vector3::new(
                (point_x as f64 * raw_header.x_scale_factor) + raw_header.x_offset,
                (point_y as f64 * raw_header.y_scale_factor) + raw_header.y_offset,
                (point_z as f64 * raw_header.z_scale_factor) + raw_header.z_offset,
            ),
            ..Default::default()
        });
    }
    Ok(())
}

pub fn _search_las_file_by_time_range<P: AsRef<Path>>(
    _path: P,
    _time_range: Range<f64>,
    _result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    // PointStream does not yet support GPS time
    todo!("not implemented")
}
