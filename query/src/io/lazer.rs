use std::{
    cell::RefCell,
    fs::File,
    io::{BufReader, Cursor, SeekFrom},
    ops::Range,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{anyhow, bail, Context, Result};
use io::lazer::LazerReader;
use pasture_core::{
    containers::{HashMapBuffer, MakeBufferFromLayout, OwningBuffer, VectorBuffer},
    layout::{attributes::POSITION_3D, PointLayout},
};
use pasture_io::{
    base::{PointReader, SeekToPoint},
    las::{point_layout_from_las_metadata, LASMetadata, LASReader},
};
use thread_local::ThreadLocal;

use crate::{index::ValueType, io::PointData};

use super::{
    FileFormat, IOMethod, IOStats, IOStatsParameters, PointDataLoader, PointDataMemoryLayout,
};

/// Data loader for points in the LAZER custom format using `mmap`
pub(crate) struct LAZERPointDataLoaderMmap {
    mmap: memmap::Mmap,
    default_point_layout: PointLayout,
    las_metadata: LASMetadata,
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
            las_metadata,
        })
    }
}

impl PointDataLoader for LAZERPointDataLoaderMmap {
    fn get_point_data(
        &self,
        point_range: Range<usize>,
        target_layout: &PointLayout,
        positions_in_world_space: bool,
        desired_memory_layout: PointDataMemoryLayout,
    ) -> Result<super::PointData> {
        let _span = tracy_client::span!("LAZER::get_point_data");

        if point_range.is_empty() {
            match desired_memory_layout {
                PointDataMemoryLayout::Interleaved => {
                    Ok(VectorBuffer::new_from_layout(self.default_point_layout.clone()).into())
                }
                PointDataMemoryLayout::Columnar => {
                    Ok(HashMapBuffer::new_from_layout(self.default_point_layout.clone()).into())
                }
            }
        } else {
            let mut lazer_reader =
                LazerReader::new(Cursor::new(&self.mmap)).context("Failed to open LAZER reader")?;
            lazer_reader.seek_point(SeekFrom::Start(point_range.start as u64))?;

            if positions_in_world_space {
                if !target_layout.has_attribute(&POSITION_3D) {
                    bail!("When `positions_in_world_space` is set to `true`, the target layout must contain a `POSITION_3D` attribute with `Vec3f64` as datatype!")
                }
            }

            match desired_memory_layout {
                PointDataMemoryLayout::Interleaved => {
                    let mut points =
                        VectorBuffer::with_capacity(point_range.len(), target_layout.clone());
                    points.resize(point_range.len());
                    lazer_reader
                        .read_into(&mut points, point_range.len())
                        .context("Failed to read points")?;
                    Ok(PointData::OwnedInterleaved(points))
                }
                PointDataMemoryLayout::Columnar => {
                    let mut points =
                        HashMapBuffer::with_capacity(point_range.len(), target_layout.clone());
                    points.resize(point_range.len());
                    lazer_reader
                        .read_into(&mut points, point_range.len())
                        .context("Failed to read points")?;
                    Ok(PointData::OwnedColumnar(points))
                }
            }
        }
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

    fn estimate_io_time_for_point_range(
        &self,
        point_range: &Range<usize>,
        value_type: crate::index::ValueType,
    ) -> Result<std::time::Duration> {
        let io_stats =
            IOStats::global().ok_or_else(|| anyhow!("Could not get global I/O stats"))?;
        let point_record_format = self
            .las_metadata
            .point_format()
            .to_u8()
            .context("Unsupported point format")?;
        let million_points_per_second = io_stats.throughputs_mpts().get(&IOStatsParameters {
            file_format: FileFormat::LAZER,
            io_method: IOMethod::Mmap,
            point_record_format,
        }).ok_or_else(|| anyhow!("No statistics for point record format {point_record_format} of mmapped LAZER file found"))?;
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

pub(crate) struct LAZERPointDataLoaderFile {
    default_point_layout: PointLayout,
    path: PathBuf,
    lazer_reader: ThreadLocal<RefCell<LazerReader<BufReader<File>>>>,
}

impl LAZERPointDataLoaderFile {
    pub(crate) fn new(path: &Path) -> Result<Self> {
        let _span = tracy_client::span!("LAZERPointDataLoaderFile::new");
        let file =
            File::open(path).with_context(|| format!("Could not open file {}", path.display()))?;
        let reader = LazerReader::new(BufReader::new(file))
            .with_context(|| format!("Could not open reader to LAZER file {}", path.display()))?;
        let default_point_layout = reader.get_default_point_layout().clone();

        Ok(Self {
            default_point_layout,
            path: path.to_path_buf(),
            lazer_reader: Default::default(),
        })
    }

    fn get_thread_local_reader(&self) -> Result<&RefCell<LazerReader<BufReader<File>>>> {
        self.lazer_reader.get_or_try(|| {
            let file = File::open(&self.path)
                .with_context(|| format!("Failed to open LAZER file {}", self.path.display()))?;
            let reader =
                LazerReader::new(BufReader::new(file)).context("Failed to create LAZER reader")?;
            Ok(RefCell::new(reader))
        })
    }
}

impl PointDataLoader for LAZERPointDataLoaderFile {
    fn get_point_data(
        &self,
        point_range: Range<usize>,
        target_layout: &PointLayout,
        positions_in_world_space: bool,
        desired_memory_layout: PointDataMemoryLayout,
    ) -> Result<super::PointData> {
        let _span = tracy_client::span!("LAZER::get_point_data");

        if point_range.is_empty() {
            match desired_memory_layout {
                PointDataMemoryLayout::Interleaved => {
                    Ok(VectorBuffer::new_from_layout(self.default_point_layout.clone()).into())
                }
                PointDataMemoryLayout::Columnar => {
                    Ok(HashMapBuffer::new_from_layout(self.default_point_layout.clone()).into())
                }
            }
        } else {
            let lazer_reader = self.get_thread_local_reader()?;
            let mut lazer_reader_mut = lazer_reader.borrow_mut();
            lazer_reader_mut.seek_point(SeekFrom::Start(point_range.start as u64))?;

            if positions_in_world_space {
                if !target_layout.has_attribute(&POSITION_3D) {
                    bail!("When `positions_in_world_space` is set to `true`, the target layout must contain a `POSITION_3D` attribute with `Vec3f64` as datatype!")
                }
            }

            match desired_memory_layout {
                PointDataMemoryLayout::Interleaved => {
                    let mut points =
                        VectorBuffer::with_capacity(point_range.len(), target_layout.clone());
                    points.resize(point_range.len());
                    lazer_reader_mut
                        .read_into(&mut points, point_range.len())
                        .context("Failed to read points")?;
                    Ok(PointData::OwnedInterleaved(points))
                }
                PointDataMemoryLayout::Columnar => {
                    let mut points =
                        HashMapBuffer::with_capacity(point_range.len(), target_layout.clone());
                    points.resize(point_range.len());
                    lazer_reader_mut
                        .read_into(&mut points, point_range.len())
                        .context("Failed to read points")?;
                    Ok(PointData::OwnedColumnar(points))
                }
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
        let reader = self.get_thread_local_reader()?;
        let point_record_format = reader
            .borrow()
            .las_metadata()
            .point_format()
            .to_u8()
            .context("Unsupported point format")?;
        let million_points_per_second = io_stats.throughputs_mpts().get(&IOStatsParameters {
            file_format: FileFormat::LAZER,
            io_method: IOMethod::File,
            point_record_format,
        }).ok_or_else(|| anyhow!("No statistics for point record format {point_record_format} of mmapped LAZER file found"))?;
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
