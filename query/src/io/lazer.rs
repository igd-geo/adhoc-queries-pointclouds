use std::{
    fs::File,
    io::{BufReader, Cursor, SeekFrom},
    ops::Range,
    path::Path,
    sync::Mutex,
};

use anyhow::{bail, Context, Result};
use io::lazer::LazerReader;
use pasture_core::{
    containers::{HashMapBuffer, MakeBufferFromLayout, OwningBuffer},
    layout::{attributes::POSITION_3D, PointLayout},
};
use pasture_io::{
    base::{PointReader, SeekToPoint},
    las::{point_layout_from_las_metadata, LASReader},
};

use crate::io::PointData;

use super::PointDataLoader;

/// Data loader for points in the LAZER custom format using `mmap`
pub(crate) struct LAZERPointDataLoaderMmap {
    mmap: memmap::Mmap,
    default_point_layout: PointLayout,
}

impl LAZERPointDataLoaderMmap {
    pub(crate) fn new(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("Failed to open LAZ file {}", path.display()))?;
        let mmap = unsafe {
            memmap::Mmap::map(&file)
                .with_context(|| format!("Failed to mmap LAZ file {}", path.display()))?
        };

        let las_metadata = LASReader::from_read(Cursor::new(&mmap), false, false)
            .with_context(|| format!("Failed to get metadata from LAZER file {}", path.display()))?
            .las_metadata()
            .clone();

        let default_point_layout = point_layout_from_las_metadata(&las_metadata, true)
            .with_context(|| {
                format!(
                    "No matching PointLayout found for LAZ file {}",
                    path.display()
                )
            })?;

        Ok(Self {
            mmap,
            default_point_layout,
        })
    }
}

impl PointDataLoader for LAZERPointDataLoaderMmap {
    fn get_point_data(
        &self,
        point_range: Range<usize>,
        target_layout: &PointLayout,
        positions_in_world_space: bool,
    ) -> Result<super::PointData> {
        let _span = tracy_client::span!("LAZER::get_point_data");

        if point_range.is_empty() {
            let empty_buffer = HashMapBuffer::new_from_layout(self.default_point_layout.clone());
            return Ok(PointData::OwnedColumnar(empty_buffer));
        }

        let mut lazer_reader =
            LazerReader::new(Cursor::new(&self.mmap)).context("Failed to open LAZER reader")?;
        // if point_range.start > 0 {
        lazer_reader.seek_point(SeekFrom::Start(point_range.start as u64))?;
        // }

        if positions_in_world_space {
            if !target_layout.has_attribute(&POSITION_3D) {
                bail!("When `positions_in_world_space` is set to `true`, the target layout must contain a `POSITION_3D` attribute with `Vec3f64` as datatype!")
            }
        }

        let mut points = HashMapBuffer::with_capacity(point_range.len(), target_layout.clone());
        points.resize(point_range.len());
        lazer_reader
            .read_into(&mut points, point_range.len())
            .context("Failed to read points")?;
        Ok(PointData::OwnedColumnar(points))
    }

    fn mem_size(&self) -> usize {
        self.mmap.len()
    }

    fn default_point_layout(&self) -> &PointLayout {
        &self.default_point_layout
    }

    fn has_positions_in_world_space(&self) -> bool {
        false
    }

    fn supports_borrowed_data(&self) -> bool {
        false
    }
}

pub(crate) struct LAZERPointDataLoaderFile {
    default_point_layout: PointLayout,
    lazer_reader: Mutex<LazerReader<BufReader<File>>>,
}

impl LAZERPointDataLoaderFile {
    pub(crate) fn new(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("Could not open file {}", path.display()))?;
        let reader = LazerReader::new(BufReader::new(file))
            .with_context(|| format!("Could not open reader to LAZER file {}", path.display()))?;
        let default_point_layout = reader.get_default_point_layout().clone();

        Ok(Self {
            default_point_layout,
            lazer_reader: Mutex::new(reader),
        })
    }
}

impl PointDataLoader for LAZERPointDataLoaderFile {
    fn get_point_data(
        &self,
        point_range: Range<usize>,
        target_layout: &PointLayout,
        positions_in_world_space: bool,
    ) -> Result<super::PointData> {
        let _span = tracy_client::span!("LAZER::get_point_data");

        if point_range.is_empty() {
            let empty_buffer = HashMapBuffer::new_from_layout(self.default_point_layout.clone());
            return Ok(PointData::OwnedColumnar(empty_buffer));
        }

        let mut lazer_reader = self.lazer_reader.lock().expect("Lock was poisoned");
        lazer_reader.seek_point(SeekFrom::Start(point_range.start as u64))?;

        if positions_in_world_space {
            if !target_layout.has_attribute(&POSITION_3D) {
                bail!("When `positions_in_world_space` is set to `true`, the target layout must contain a `POSITION_3D` attribute with `Vec3f64` as datatype!")
            }
        }

        let mut points = HashMapBuffer::with_capacity(point_range.len(), target_layout.clone());
        points.resize(point_range.len());
        lazer_reader
            .read_into(&mut points, point_range.len())
            .context("Failed to read points")?;
        Ok(PointData::OwnedColumnar(points))
    }

    fn mem_size(&self) -> usize {
        0
    }

    fn default_point_layout(&self) -> &PointLayout {
        &self.default_point_layout
    }

    fn has_positions_in_world_space(&self) -> bool {
        false
    }

    fn supports_borrowed_data(&self) -> bool {
        false
    }
}
