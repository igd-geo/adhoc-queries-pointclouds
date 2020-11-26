use crate::collect_points::ResultCollector;
use crate::math::AABB;
use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap::MmapOptions;
use nalgebra::Vector3;
use pointstream::pointcloud::{
    LaserSource, LinearPointBuffer, PointAttributes, PointBufferReadable, PointBufferWriteable,
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

pub fn search_laser_file_by_bounds_optimized<P: AsRef<Path>>(
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

    // Parse block size and block offsets
    reader.seek(SeekFrom::Start(raw_header.offset_to_point_data as u64))?;
    let block_size = reader.read_u64::<LittleEndian>()?;
    let num_blocks = f64::ceil((header.number_of_points() as f64) / (block_size as f64)) as u64;

    let block_offsets = (0..num_blocks)
        .map(|_| reader.read_u64::<LittleEndian>())
        .collect::<Result<Vec<_>, _>>()?;

    let mut seek_in_block = |block_idx: usize| -> Result<()> {
        let block_offset = block_offsets[block_idx];

        reader.seek(SeekFrom::Start(block_offset))?;

        let num_points_in_block = u64::min(
            block_size,
            header.number_of_points() as u64 - (block_idx as u64 * block_size),
        );

        for _ in 0..num_points_in_block {
            let point_x = reader.read_i32::<LittleEndian>()? as i64;
            if point_x < query_bounds_local.min.x || point_x > query_bounds_local.max.x {
                reader.seek(SeekFrom::Current(8))?;
                continue;
            }

            let point_y = reader.read_i32::<LittleEndian>()? as i64;
            if point_y < query_bounds_local.min.y || point_y > query_bounds_local.max.y {
                reader.seek(SeekFrom::Current(4))?;
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
    };

    for idx in 0..num_blocks {
        seek_in_block(idx as usize)?;
    }

    Ok(())
}

pub fn search_laser_file_by_bounds<P: AsRef<Path>>(
    path: P,
    bounds: &AABB<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    let reader = open_file_reader(path)?;

    let mut point_source = LaserSource::from(reader)?;

    let metadata = point_source.metadata().clone();
    if !metadata.bounds().intersects(&bounds.into()) {
        return Ok(());
    }

    // Read in chunks equal to the block size
    let chunk_size = point_source.block_size() as usize;
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

pub fn search_laser_file_by_classification_optimized<P: AsRef<Path>>(
    path: P,
    class: u8,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    todo!("not implemented")
}

pub fn search_laser_file_by_classification<P: AsRef<Path>>(
    path: P,
    class: u8,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    todo!("not implemented")
}

pub fn search_laser_file_by_time_range_optimized<P: AsRef<Path>>(
    path: P,
    time_range: Range<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    todo!("not implemented")
}

pub fn search_laser_file_by_time_range<P: AsRef<Path>>(
    path: P,
    time_range: Range<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    // PointStream does not yet support GPS time
    todo!("not implemented")
}
