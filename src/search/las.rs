use crate::collect_points::ResultCollector;
use crate::math::AABB;
use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap::MmapOptions;
use nalgebra::Vector3;
use pointstream::pointcloud::{
    LasSource, LinearPointBuffer, PointAttributes, PointBufferReadable, PointBufferWriteable,
    PointSource,
};
use std::convert::TryInto;
use std::fs::File;
use std::io::SeekFrom;
use std::io::{BufReader, Cursor, Read, Seek};
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

fn parse_las_header<R: std::io::Read>(mut reader: R) -> Result<las::raw::Header> {
    let raw_header = las::raw::Header::read_from(reader)?;
    Ok(raw_header)
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
    let mut reader = open_file_reader(path)?;

    let raw_header = parse_las_header(&mut reader)?;
    let header = las::Header::from_raw(raw_header.clone())?;
    let file_bounds = AABB::new(
        Vector3::new(
            header.bounds().min.x,
            header.bounds().min.y,
            header.bounds().min.z,
        ),
        Vector3::new(
            header.bounds().max.x,
            header.bounds().max.y,
            header.bounds().max.z,
        ),
    );

    if !file_bounds.intersects(bounds) {
        return Ok(());
    }

    // Convert bounds of query area into integer coordinates in local space of file. This makes intersection
    // checks very fast because they can be done on integer values
    let query_bounds_local = AABB::<i64>::new(
        Vector3::<i64>::new(
            ((bounds.min.x - raw_header.x_offset) / raw_header.x_scale_factor) as i64,
            ((bounds.min.y - raw_header.y_offset) / raw_header.x_scale_factor) as i64,
            ((bounds.min.z - raw_header.z_offset) / raw_header.x_scale_factor) as i64,
        ),
        Vector3::<i64>::new(
            ((bounds.max.x - raw_header.x_offset) / raw_header.x_scale_factor) as i64,
            ((bounds.max.y - raw_header.y_offset) / raw_header.y_scale_factor) as i64,
            ((bounds.max.z - raw_header.z_offset) / raw_header.z_scale_factor) as i64,
        ),
    );

    for point_idx in 0..header.number_of_points() {
        let point_offset: u64 = point_idx as u64 * raw_header.point_data_record_length as u64
            + raw_header.offset_to_point_data as u64;
        reader.seek(SeekFrom::Start(point_offset))?;

        let point_x = reader.read_i32::<LittleEndian>()? as i64;
        if point_x < query_bounds_local.min.x || point_x > query_bounds_local.max.x {
            continue;
        }

        let point_y = reader.read_i32::<LittleEndian>()? as i64;
        if point_y < query_bounds_local.min.y || point_y > query_bounds_local.max.y {
            continue;
        }

        let point_z = reader.read_i32::<LittleEndian>()? as i64;
        if point_z < query_bounds_local.min.z || point_z > query_bounds_local.max.z {
            continue;
        }

        result_collector.collect_one(las::point::Point {
            x: (point_x as f64 * raw_header.x_scale_factor) + raw_header.x_offset,
            y: (point_y as f64 * raw_header.y_scale_factor) + raw_header.y_offset,
            z: (point_z as f64 * raw_header.z_scale_factor) + raw_header.z_offset,
            ..Default::default()
        });
    }
    Ok(())
}

pub fn search_las_file_by_bounds<P: AsRef<Path>>(
    path: P,
    bounds: &AABB<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    let reader = open_file_reader(path)?;

    let mut point_source = LasSource::from(reader)?;

    let metadata = point_source.metadata().clone();
    if !metadata.bounds().intersects(&bounds.into()) {
        return Ok(());
    }

    // Read in chunks of fixed size
    let chunk_size = 65536; //24 bytes per point ^= ~1.5MiB
    let mut point_buffer = LinearPointBuffer::new(chunk_size, PointAttributes::Position);

    let num_chunks = (metadata.point_count() + chunk_size - 1) / chunk_size;
    for idx in 0..num_chunks {
        let points_in_chunk = usize::min(chunk_size, metadata.point_count() - (idx * chunk_size));
        point_source.read_into(&mut point_buffer, points_in_chunk)?;

        point_buffer
            .positions()
            .iter()
            .take(points_in_chunk)
            .filter(|pos| bounds.contains(pos))
            .for_each(|pos| {
                let point = las::point::Point {
                    x: pos.x,
                    y: pos.y,
                    z: pos.z,
                    ..Default::default()
                };
                result_collector.collect_one(point);
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
    let header = las::Header::from_raw(raw_header.clone())?;

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

        // Now we read XYZ
        reader.seek(SeekFrom::Start(point_offset))?;
        let point_x = reader.read_i32::<LittleEndian>()?;
        let point_y = reader.read_i32::<LittleEndian>()?;
        let point_z = reader.read_i32::<LittleEndian>()?;

        result_collector.collect_one(las::point::Point {
            x: (point_x as f64 * raw_header.x_scale_factor) + raw_header.x_offset,
            y: (point_y as f64 * raw_header.y_scale_factor) + raw_header.y_offset,
            z: (point_z as f64 * raw_header.z_scale_factor) + raw_header.z_offset,
            classification: las::point::Classification::new(classification)?,
            ..Default::default()
        });
    }
    Ok(())
}

pub fn search_las_file_by_classification<P: AsRef<Path>>(
    path: P,
    class: u8,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    let reader = open_file_reader(path)?;

    let mut point_source = LasSource::from(reader)?;

    let metadata = point_source.metadata().clone();

    // Read in chunks of fixed size
    let chunk_size = 65536; //24 bytes per point ^= ~1.5MiB
    let mut point_buffer = LinearPointBuffer::new(
        chunk_size,
        PointAttributes::Position | PointAttributes::Classification,
    );

    let num_chunks = (metadata.point_count() + chunk_size - 1) / chunk_size;
    for idx in 0..num_chunks {
        let points_in_chunk = usize::min(chunk_size, metadata.point_count() - (idx * chunk_size));
        point_source.read_into(&mut point_buffer, points_in_chunk)?;

        for point_idx in 0..points_in_chunk {
            let point_class = point_buffer.classifications().unwrap()[point_idx];
            if point_class != class {
                continue;
            }

            let point_position = point_buffer.positions()[point_idx];
            let point = las::point::Point {
                x: point_position.x,
                y: point_position.y,
                z: point_position.z,
                classification: las::point::Classification::new(point_class)?,
                ..Default::default()
            };
            result_collector.collect_one(point);
        }
    }
    Ok(())
}

pub fn search_las_file_by_time_range_optimized<P: AsRef<Path>>(
    path: P,
    time_range: Range<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    let mut reader = open_file_reader(&path)?;

    let raw_header = parse_las_header(&mut reader)?;
    let header = las::Header::from_raw(raw_header.clone())?;

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

        result_collector.collect_one(las::point::Point {
            x: (point_x as f64 * raw_header.x_scale_factor) + raw_header.x_offset,
            y: (point_y as f64 * raw_header.y_scale_factor) + raw_header.y_offset,
            z: (point_z as f64 * raw_header.z_scale_factor) + raw_header.z_offset,
            gps_time: Some(gps_time),
            ..Default::default()
        });
    }
    Ok(())
}

pub fn search_las_file_by_time_range<P: AsRef<Path>>(
    path: P,
    time_range: Range<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    // PointStream does not yet support GPS time
    todo!("not implemented")
}
