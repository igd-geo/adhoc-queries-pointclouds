use crate::collect_points::ResultCollector;
use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap::MmapOptions;
use pasture_core::{
    containers::{
        PerAttributePointBufferExt, PerAttributeVecPointStorage, PointBufferExt,
        PointBufferWriteable,
    },
    layout::{
        attributes::{CLASSIFICATION, POSITION_3D},
        PointType,
    },
    math::AABB,
    nalgebra::{Point3, Vector3},
};
use pasture_io::{base::PointReader, las_rs::raw};
use readers::{LAZERSource, Point};
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

fn parse_las_header<R: std::io::Read>(mut reader: R) -> Result<raw::Header> {
    let raw_header = raw::Header::read_from(reader)?;
    Ok(raw_header)
}

pub fn search_lazer_file_by_bounds<P: AsRef<Path>>(
    path: P,
    bounds: &AABB<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    let mmap = open_file_reader(path)?;

    let mut reader = LAZERSource::from(mmap)?;

    let metadata = reader.get_metadata();
    let num_points = metadata
        .number_of_points()
        .expect("No number of points found in LAZER file");
    if !metadata
        .bounds()
        .expect("No bounds found in LAZER file")
        .intersects(&bounds)
    {
        return Ok(());
    }

    // Read in chunks equal to the block size
    let chunk_size = reader.block_size() as usize;
    let mut point_buffer = PerAttributeVecPointStorage::with_capacity(chunk_size, Point::layout());

    let num_chunks = (num_points + chunk_size - 1) / chunk_size;
    for idx in 0..num_chunks {
        let points_in_chunk = usize::min(chunk_size, num_points - (idx * chunk_size));
        reader.read_into(&mut point_buffer, points_in_chunk)?;

        point_buffer
            .get_attribute_range_ref::<Vector3<f64>>(0..points_in_chunk, &POSITION_3D)
            .iter()
            .enumerate()
            .filter(|(_, pos)| bounds.contains(&(**pos).into()))
            .for_each(|(idx, _)| {
                let point = point_buffer.get_point::<Point>(idx);
                result_collector.collect_one(point);
            });

        point_buffer.clear();
    }
    Ok(())
}

pub fn search_lazer_file_by_classification<P: AsRef<Path>>(
    path: P,
    class: u8,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    let mmap = open_file_reader(path)?;

    let mut reader = LAZERSource::from(mmap)?;

    let metadata = reader.get_metadata();
    let num_points = metadata
        .number_of_points()
        .expect("No number of points found in LAZER file");

    // Read in chunks equal to the block size
    let chunk_size = reader.block_size() as usize;
    let mut point_buffer = PerAttributeVecPointStorage::with_capacity(chunk_size, Point::layout());

    let num_chunks = (num_points + chunk_size - 1) / chunk_size;
    for idx in 0..num_chunks {
        let points_in_chunk = usize::min(chunk_size, num_points - (idx * chunk_size));
        reader.read_into(&mut point_buffer, points_in_chunk)?;

        point_buffer
            .get_attribute_range_ref::<u8>(0..points_in_chunk, &CLASSIFICATION)
            .iter()
            .enumerate()
            .filter(|(_, &classification)| class == classification)
            .for_each(|(idx, _)| {
                let point = point_buffer.get_point::<Point>(idx);
                result_collector.collect_one(point);
            });
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
