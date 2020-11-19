use crate::collect_points::ResultCollector;
use crate::math::AABB;
use crate::search::{search_las_file_by_bounds, search_las_file_by_classification};
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
use std::io::{Cursor, Seek};
use std::ops::Range;
use std::path::Path;

use ::las::{Read, Reader};

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
    let reader = open_file_reader(path)?;
    let mut las_reader = ::las::Reader::new(reader)?;

    let header = las_reader.header();
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

    for wrapped_point in las_reader.points() {
        let point = wrapped_point?;
        if bounds.contains(&Vector3::new(point.x, point.y, point.z)) {
            result_collector.collect_one(point);
        }
    }

    Ok(())
}

pub fn search_laz_file_by_bounds<P: AsRef<Path>>(
    path: P,
    bounds: &AABB<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    search_las_file_by_bounds(path, bounds, result_collector)
}

pub fn search_laz_file_by_classification_optimized<P: AsRef<Path>>(
    path: P,
    class: u8,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    let reader = open_file_reader(path)?;
    let mut las_reader = ::las::Reader::new(reader)?;

    let target_class = las::point::Classification::new(class)?;

    for wrapped_point in las_reader.points() {
        let point = wrapped_point?;

        if point.classification == target_class {
            result_collector.collect_one(point);
        }
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
    let reader = open_file_reader(path)?;
    let mut las_reader = ::las::Reader::new(reader)?;

    for wrapped_point in las_reader.points() {
        let point = wrapped_point?;

        if let Some(gps_time) = point.gps_time {
            if time_range.contains(&gps_time) {
                result_collector.collect_one(point);
            }
        }
    }

    Ok(())
}

pub fn search_laz_file_by_time_range<P: AsRef<Path>>(
    path: P,
    time_range: Range<f64>,
    result_collector: &mut dyn ResultCollector,
) -> Result<()> {
    // PointStream does not yet support GPS time
    todo!("not implemented")
}
