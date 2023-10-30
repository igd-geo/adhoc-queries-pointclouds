use anyhow::{anyhow, bail, Context, Result};
use io::last::LASTReader;
use pasture_core::{
    containers::{HashMapBuffer, OwningBuffer, VectorBuffer},
    layout::{attributes::POSITION_3D, PointLayout},
};
use pasture_io::{
    base::{PointReader, SeekToPoint},
    las::{point_layout_from_las_metadata, LASMetadata, LASReader},
};
use std::{
    cell::RefCell,
    fs::File,
    io::{BufReader, Cursor, SeekFrom},
    ops::Range,
    path::{Path, PathBuf},
    time::Duration,
};
use thread_local::ThreadLocal;

use crate::{index::ValueType, io::PointData};

use super::{
    FileFormat, IOMethod, IOStats, IOStatsParameters, PointDataLoader, PointDataMemoryLayout,
};

pub(crate) struct LASTPointDataReaderMmap {
    mmap: memmap::Mmap,
    default_point_layout: PointLayout,
    las_metadata: LASMetadata,
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
            las_metadata,
        })
    }
}

impl PointDataLoader for LASTPointDataReaderMmap {
    fn get_point_data(
        &self,
        point_range: std::ops::Range<usize>,
        target_layout: &pasture_core::layout::PointLayout,
        positions_in_world_space: bool,
        desired_memory_layout: PointDataMemoryLayout,
    ) -> Result<PointData> {
        let _span = tracy_client::span!("LAST::get_point_data");
        if positions_in_world_space && !target_layout.has_attribute(&POSITION_3D) {
            bail!("If positions_in_world_space is set, the target PointLayout must have the default POSITION_3D attribute");
        }

        let mut last_reader = LASTReader::from_read(Cursor::new(&self.mmap[..]))?;
        last_reader.seek_point(SeekFrom::Start(point_range.start as u64))?;

        match desired_memory_layout {
            PointDataMemoryLayout::Interleaved => {
                let mut buffer =
                    VectorBuffer::with_capacity(point_range.len(), target_layout.clone());
                buffer.resize(point_range.len());
                last_reader.read_into(&mut buffer, point_range.len())?;
                Ok(PointData::OwnedInterleaved(buffer))
            }
            PointDataMemoryLayout::Columnar => {
                let mut buffer =
                    HashMapBuffer::with_capacity(point_range.len(), target_layout.clone());
                buffer.resize(point_range.len());
                last_reader.read_into(&mut buffer, point_range.len())?;
                Ok(PointData::OwnedColumnar(buffer))
            }
        }
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

    fn estimate_io_time_for_point_range(
        &self,
        point_range: &Range<usize>,
        value_type: ValueType,
    ) -> Result<Duration> {
        let io_stats =
            IOStats::global().ok_or_else(|| anyhow!("Could not get global I/O stats"))?;
        let point_record_format = self
            .las_metadata
            .point_format()
            .to_u8()
            .context("Unsupported point format")?;
        let million_points_per_second = io_stats.throughputs_mpts().get(&IOStatsParameters {
            file_format: FileFormat::LAST,
            io_method: IOMethod::Mmap,
            point_record_format,
        }).ok_or_else(|| anyhow!("No statistics for point record format {point_record_format} of mmapped LAST file found"))?;
        let points_per_second = million_points_per_second * 1e6;
        let bytes_in_point = self.default_point_layout.size_of_point_entry() as f64;
        // Estimate a speedup factor by dividing the size of the attribute from the ValueType by the size of all attributes in a point
        let value_type_percentage = match value_type {
            ValueType::Classification => 1.0 / bytes_in_point,
            ValueType::GpsTime => 8.0 / bytes_in_point,
            ValueType::NumberOfReturns => 1.0 / bytes_in_point,
            ValueType::Position3D => 12.0 / bytes_in_point,
            ValueType::ReturnNumber => 1.0 / bytes_in_point,
        };
        let expected_time_seconds =
            (point_range.len() as f64 / points_per_second) * value_type_percentage;
        Ok(Duration::from_secs_f64(expected_time_seconds))
    }

    fn preferred_memory_layout(&self) -> PointDataMemoryLayout {
        PointDataMemoryLayout::Columnar
    }
}

pub(crate) struct LASTPointDataReaderFile {
    default_point_layout: PointLayout,
    path: PathBuf,
    last_reader: ThreadLocal<RefCell<LASTReader<BufReader<File>>>>,
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
            path: path.to_path_buf(),
            last_reader: Default::default(),
        })
    }

    fn get_thread_local_reader(&self) -> Result<&RefCell<LASTReader<BufReader<File>>>> {
        self.last_reader.get_or_try(|| {
            let file = File::open(&self.path)
                .with_context(|| format!("Failed to open LAST file {}", self.path.display()))?;
            let reader = LASTReader::from_read(BufReader::new(file))
                .context("Failed to create LAST reader")?;
            Ok(RefCell::new(reader))
        })
    }
}

impl PointDataLoader for LASTPointDataReaderFile {
    fn get_point_data(
        &self,
        point_range: std::ops::Range<usize>,
        target_layout: &PointLayout,
        positions_in_world_space: bool,
        desired_memory_layout: PointDataMemoryLayout,
    ) -> Result<PointData> {
        let _span = tracy_client::span!("LAST::get_point_data");
        if positions_in_world_space && !target_layout.has_attribute(&POSITION_3D) {
            bail!("If positions_in_world_space is set, the target PointLayout must have the default POSITION_3D attribute");
        }

        let last_reader = self.get_thread_local_reader()?;
        let mut last_reader_mut = last_reader.borrow_mut();
        last_reader_mut.seek_point(SeekFrom::Start(point_range.start as u64))?;

        match desired_memory_layout {
            PointDataMemoryLayout::Interleaved => {
                let mut buffer =
                    VectorBuffer::with_capacity(point_range.len(), target_layout.clone());
                buffer.resize(point_range.len());
                last_reader_mut.read_into(&mut buffer, point_range.len())?;
                Ok(PointData::OwnedInterleaved(buffer))
            }
            PointDataMemoryLayout::Columnar => {
                let mut buffer =
                    HashMapBuffer::with_capacity(point_range.len(), target_layout.clone());
                buffer.resize(point_range.len());
                last_reader_mut.read_into(&mut buffer, point_range.len())?;
                Ok(PointData::OwnedColumnar(buffer))
            }
        }
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

    fn estimate_io_time_for_point_range(
        &self,
        point_range: &Range<usize>,
        value_type: crate::index::ValueType,
    ) -> Result<std::time::Duration> {
        let io_stats =
            IOStats::global().ok_or_else(|| anyhow!("Could not get global I/O stats"))?;
        let last_reader = self.get_thread_local_reader()?;
        let point_record_format = last_reader
            .borrow()
            .las_metadata()
            .point_format()
            .to_u8()
            .context("Unsupported point format")?;
        let million_points_per_second = io_stats.throughputs_mpts().get(&IOStatsParameters {
            file_format: FileFormat::LAST,
            io_method: IOMethod::File,
            point_record_format,
        }).ok_or_else(|| anyhow!("No statistics for point record format {point_record_format} of LAST file found"))?;
        let points_per_second = million_points_per_second * 1e6;
        let bytes_in_point = self.default_point_layout.size_of_point_entry() as f64;
        // Estimate a speedup factor by dividing the size of the attribute from the ValueType by the size of all attributes in a point
        let value_type_percentage = match value_type {
            ValueType::Classification => 1.0 / bytes_in_point,
            ValueType::GpsTime => 8.0 / bytes_in_point,
            ValueType::NumberOfReturns => 1.0 / bytes_in_point,
            ValueType::Position3D => 12.0 / bytes_in_point,
            ValueType::ReturnNumber => 1.0 / bytes_in_point,
        };
        let expected_time_seconds =
            (point_range.len() as f64 / points_per_second) * value_type_percentage;
        Ok(Duration::from_secs_f64(expected_time_seconds))
    }

    fn preferred_memory_layout(&self) -> PointDataMemoryLayout {
        PointDataMemoryLayout::Columnar
    }
}
