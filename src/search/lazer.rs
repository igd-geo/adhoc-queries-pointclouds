use crate::collect_points::ResultCollector;
use crate::math::AABB;
use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap::MmapOptions;
use nalgebra::Vector3;
use pointstream::pointcloud::{
    LazerSource, LinearPointBuffer, PointAttributes, PointBufferReadable, PointBufferWriteable,
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

pub fn search_lazer_file_by_bounds<P: AsRef<Path>>(
    path: P,
    bounds: &AABB<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    let reader = open_file_reader(path)?;

    let mut point_source = LazerSource::from(reader)?;

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

pub fn search_lazer_file_by_classification<P: AsRef<Path>>(
    path: P,
    class: u8,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    let reader = open_file_reader(path)?;

    let mut point_source = LazerSource::from(reader)?;

    let metadata = point_source.metadata().clone();

    // Read in chunks equal to the block size
    let chunk_size = point_source.block_size() as usize;
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

pub fn search_lazer_file_by_time_range<P: AsRef<Path>>(
    path: P,
    time_range: Range<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    // PointStream does not yet support GPS time
    todo!("not implemented")
}
