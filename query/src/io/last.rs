use anyhow::{bail, Context, Result};
use io::last::LASTReader;
use pasture_core::{
    containers::{HashMapBuffer, OwningBuffer},
    layout::{attributes::POSITION_3D, PointLayout},
};
use pasture_io::{
    base::{PointReader, SeekToPoint},
    las::{point_layout_from_las_metadata, LASReader},
};
use std::{
    fs::File,
    io::{BufReader, Cursor, SeekFrom},
    path::Path,
    sync::Mutex,
};

use crate::io::PointData;

use super::PointDataLoader;

pub(crate) struct LASTPointDataReaderMmap {
    mmap: memmap::Mmap,
    default_point_layout: PointLayout,
}

impl LASTPointDataReaderMmap {
    pub(crate) fn new(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("Failed to open LAST file {}", path.display()))?;
        let mmap = unsafe {
            memmap::Mmap::map(&file)
                .with_context(|| format!("Failed to mmap LAST file {}", path.display()))?
        };

        let las_metadata = LASReader::from_read(Cursor::new(&mmap), false, false)
            .with_context(|| format!("Failed to get metadata from LAST file {}", path.display()))?
            .las_metadata()
            .clone();

        let default_point_layout = point_layout_from_las_metadata(&las_metadata, true)
            .with_context(|| {
                format!(
                    "No matching PointLayout found for LAST file {}",
                    path.display()
                )
            })?;

        Ok(Self {
            mmap,
            default_point_layout,
        })
    }
}

impl PointDataLoader for LASTPointDataReaderMmap {
    fn get_point_data(
        &self,
        point_range: std::ops::Range<usize>,
        target_layout: &pasture_core::layout::PointLayout,
        positions_in_world_space: bool,
    ) -> Result<PointData> {
        let _span = tracy_client::span!("LAST::get_point_data");
        if positions_in_world_space && !target_layout.has_attribute(&POSITION_3D) {
            bail!("If positions_in_world_space is set, the target PointLayout must have the default POSITION_3D attribute");
        }

        let mut last_reader = LASTReader::from_read(Cursor::new(&self.mmap[..]))?;
        last_reader.seek_point(SeekFrom::Start(point_range.start as u64))?;

        // TODO Support for borrowed columnar data
        let mut buffer = HashMapBuffer::with_capacity(point_range.len(), target_layout.clone());
        buffer.resize(point_range.len());
        last_reader.read_into(&mut buffer, point_range.len())?;
        Ok(PointData::OwnedColumnar(buffer))
    }

    fn mem_size(&self) -> usize {
        self.mmap.len()
    }

    fn default_point_layout(&self) -> &pasture_core::layout::PointLayout {
        &self.default_point_layout
    }

    fn has_positions_in_world_space(&self) -> bool {
        false
    }

    fn supports_borrowed_data(&self) -> bool {
        // TODO It could support borrowed data, as soon as we have an `ExternalMemoryColumnarBuffer` type in pasture
        false
    }
}

pub(crate) struct LASTPointDataReaderFile {
    default_point_layout: PointLayout,
    last_reader: Mutex<LASTReader<BufReader<File>>>,
}

impl LASTPointDataReaderFile {
    pub(crate) fn new(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("Could not open file {}", path.display()))?;
        let reader = LASTReader::from_read(BufReader::new(file))
            .with_context(|| format!("Could not open reader to LAST file {}", path.display()))?;
        let default_point_layout = reader.get_default_point_layout().clone();

        Ok(Self {
            default_point_layout,
            last_reader: Mutex::new(reader),
        })
    }
}

impl PointDataLoader for LASTPointDataReaderFile {
    fn get_point_data(
        &self,
        point_range: std::ops::Range<usize>,
        target_layout: &PointLayout,
        positions_in_world_space: bool,
    ) -> Result<PointData> {
        let _span = tracy_client::span!("LAST::get_point_data");
        if positions_in_world_space && !target_layout.has_attribute(&POSITION_3D) {
            bail!("If positions_in_world_space is set, the target PointLayout must have the default POSITION_3D attribute");
        }

        let mut last_reader = self.last_reader.lock().expect("Lock was poisoned");
        last_reader.seek_point(SeekFrom::Start(point_range.start as u64))?;

        let mut buffer = HashMapBuffer::with_capacity(point_range.len(), target_layout.clone());
        buffer.resize(point_range.len());
        last_reader.read_into(&mut buffer, point_range.len())?;
        Ok(PointData::OwnedColumnar(buffer))
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
