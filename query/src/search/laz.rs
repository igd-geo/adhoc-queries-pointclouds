use crate::collect_points::ResultCollector;
use crate::search::{search_las_file_by_bounds, search_las_file_by_classification};
use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap::MmapOptions;
use pasture_core::{
    containers::{InterleavedPointBufferExt, InterleavedVecPointStorage, PointBufferWriteable},
    layout::PointType,
    math::AABB,
};
use pasture_io::{base::PointReader, las::LASReader};
use readers::Point;
use std::convert::TryInto;
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

/**
 * For LAZ, we can't run the same optimizations that we do for LAS (memmap and raw inspection of the bytes)
 * Here, the 'optimized' version uses the las-rs crate directly, the regular version uses the pointstream crate and
 * thus is equivalent to the 'search_las_file...' versions
 */

pub fn search_laz_file_by_bounds_optimized<P: AsRef<Path>>(
    path: P,
    bounds: &AABB<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    let mmap = open_file_reader(path)?;

    let mut reader = LASReader::from_read(mmap, true)?;

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

        eprintln!("Read chunk {}/{}", idx, num_chunks);

        point_buffer
            .get_points_ref::<Point>(0..points_in_chunk)
            .iter()
            .filter(|point| bounds.contains(&point.position.into()))
            .for_each(|point| {
                result_collector.collect_one(point.clone());
            });

        point_buffer.clear();
    }

    Ok(())
}

pub fn search_laz_file_by_bounds<P: AsRef<Path>>(
    path: P,
    bounds: &AABB<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    search_laz_file_by_bounds_optimized(path, bounds, result_collector)
}

pub fn search_laz_file_by_classification_optimized<P: AsRef<Path>>(
    path: P,
    class: u8,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    let mmap = open_file_reader(path)?;

    let mut reader = LASReader::from_read(mmap, true)?;

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

pub fn search_laz_file_by_classification<P: AsRef<Path>>(
    path: P,
    class: u8,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    search_las_file_by_classification(path, class, result_collector)
}

pub fn search_laz_file_by_time_range_optimized<P: AsRef<Path>>(
    path: P,
    time_range: Range<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    todo!("Not implemented")
}

pub fn search_laz_file_by_time_range<P: AsRef<Path>>(
    path: P,
    time_range: Range<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    // PointStream does not yet support GPS time
    todo!("not implemented")
}
