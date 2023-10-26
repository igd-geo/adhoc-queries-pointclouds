use anyhow::{anyhow, bail, Context, Result};
use memmap::Mmap;
use ouroboros::self_referencing;
use pasture_core::{
    containers::{
        BorrowedBuffer, BufferSliceInterleaved, ExternalMemoryBuffer, SliceBuffer, VectorBuffer,
    },
    layout::{
        attributes::POSITION_3D, conversion::BufferLayoutConverter, PointAttributeDataType,
        PointLayout,
    },
    nalgebra::Vector3,
};
use pasture_io::{
    base::{PointReader, SeekToPoint},
    las::{point_layout_from_las_metadata, LASMetadata, LASReader, ATTRIBUTE_LOCAL_LAS_POSITION},
};
use std::{
    cell::RefCell,
    fs::File,
    io::{BufReader, Cursor, SeekFrom},
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use thread_local::ThreadLocal;

use crate::{
    index::ValueType,
    io::{FileFormat, IOMethod, IOStats, IOStatsParameters},
};

use super::{PointData, PointDataLoader};

#[self_referencing]
pub struct MappedLASFile {
    pub metadata: LASMetadata,
    mmap: memmap::Mmap,
    #[covariant]
    #[borrows(mmap)]
    pub las_points_buffer: ExternalMemoryBuffer<&'this [u8]>,
}

impl MappedLASFile {
    /// Map the file at `path` into memory
    pub(crate) fn map_path(path: &Path) -> Result<Self> {
        let file =
            File::open(path).context(format!("Failed to open LAS file {}", path.display()))?;
        let mmap = unsafe { memmap::Mmap::map(&file).context("Failed to mmap LAS file")? };

        let las_metadata = LASReader::from_read(Cursor::new(&mmap), false, false)
            .context("Failed to read LAS file")?
            .las_metadata()
            .clone();

        let point_layout = point_layout_from_las_metadata(&las_metadata, true).context(format!(
            "No matching PointLayout found for LAS file {}",
            path.display()
        ))?;

        let las_header = las_metadata
            .raw_las_header()
            .ok_or(anyhow!("No LAS header found"))?;
        let raw_las_header = las_header
            .clone()
            .into_raw()
            .context("Can't convert LAS header into raw format")?;
        let point_record_byte_range = (raw_las_header.offset_to_point_data as usize)
            ..(raw_las_header.offset_to_end_of_points() as usize);

        Ok(MappedLASFileBuilder {
            metadata: las_metadata,
            mmap,
            las_points_buffer_builder: |mmap: &Mmap| {
                ExternalMemoryBuffer::new(&mmap[point_record_byte_range], point_layout)
            },
        }
        .build())
    }

    pub(crate) fn get_buffer_for_points(
        &self,
        point_range: Range<usize>,
    ) -> BufferSliceInterleaved<'_, ExternalMemoryBuffer<&[u8]>> {
        self.borrow_las_points_buffer().slice(point_range)
    }

    pub(crate) fn point_layout(&self) -> &PointLayout {
        self.borrow_las_points_buffer().point_layout()
    }

    /// Number of bytes of the mapped file
    pub(crate) fn mem_size(&self) -> usize {
        self.borrow_mmap().len()
    }
}

/// Efficient reader for LAS data that uses `mmap`
pub(crate) struct LASPointDataReaderMmap {
    mapped_file: Arc<MappedLASFile>,
}

impl LASPointDataReaderMmap {
    pub(crate) fn new(path: &Path) -> Result<Self> {
        let mapped_file = MappedLASFile::map_path(path)
            .with_context(|| format!("Could not mmap LAS file {}", path.display()))?;
        Ok(Self {
            mapped_file: Arc::new(mapped_file),
        })
    }
}

impl PointDataLoader for LASPointDataReaderMmap {
    fn get_point_data(
        &self,
        point_range: Range<usize>,
        target_layout: &PointLayout,
        positions_in_world_space: bool,
    ) -> Result<PointData> {
        let _span = tracy_client::span!("LASPointDataReaderMmap::get_point_data");

        let source_layout = self.mapped_file.point_layout();
        if target_layout == source_layout && !positions_in_world_space {
            Ok(PointData::MmappedLas(
                BorrowedLasPointData::from_file_and_range(self.mapped_file.clone(), point_range),
            ))
        } else {
            let _span = tracy_client::span!("LAS::get_point_data_with_conversion");
            // Read the points in target layout!
            let mut converter =
                BufferLayoutConverter::for_layouts_with_default(source_layout, target_layout);
            // By default, LAS positions are in local space, but we might want them in world space, especially if the
            // dataset contains multiple files. This requires the output PointLayout to have a POSITION_3D attribute!
            if positions_in_world_space {
                let source_positions_attribute = ATTRIBUTE_LOCAL_LAS_POSITION;
                let target_positions_attribute = target_layout
                    .get_attribute_by_name(POSITION_3D.name())
                    .ok_or(anyhow!("No POSITION_3D attribute found in target layout"))?;
                if target_positions_attribute.datatype() != PointAttributeDataType::Vec3f64 {
                    bail!("Outputting positions in world space is only possible if the POSITION_3D attribute of the output layout has datatype Vec3f64!");
                }

                let transforms = *self
                    .mapped_file
                    .borrow_metadata()
                    .raw_las_header()
                    .ok_or(anyhow!("Could not get LAS header"))?
                    .transforms();

                converter.set_custom_mapping_with_transformation(
                    &source_positions_attribute,
                    target_positions_attribute.attribute_definition(),
                    move |position: Vector3<f64>| -> Vector3<f64> {
                        Vector3::new(
                            position.x * transforms.x.scale + transforms.x.offset,
                            position.y * transforms.y.scale + transforms.y.offset,
                            position.z * transforms.z.scale + transforms.z.offset,
                        )
                    },
                    false,
                );
            }
            let source_slice = self.mapped_file.get_buffer_for_points(point_range);
            let converted_points = converter.convert::<VectorBuffer, _>(&source_slice);
            Ok(PointData::OwnedInterleaved(converted_points))
        }
    }

    fn mem_size(&self) -> usize {
        self.mapped_file.mem_size()
    }

    fn default_point_layout(&self) -> &PointLayout {
        self.mapped_file.point_layout()
    }

    fn has_positions_in_world_space(&self) -> bool {
        false
    }

    fn supports_borrowed_data(&self) -> bool {
        true
    }

    fn estimate_io_time_for_point_range(
        &self,
        point_range: &Range<usize>,
        _value_type: ValueType,
    ) -> Result<Duration> {
        // ValueType is irrelevant, this is LAS, we don't get a big speedup from reading less data
        let io_stats =
            IOStats::global().ok_or_else(|| anyhow!("Could not get global I/O stats"))?;
        let point_record_format = self
            .mapped_file
            .borrow_metadata()
            .point_format()
            .to_u8()
            .context("Unsupported point format")?;
        let million_points_per_second = io_stats.throughputs_mpts().get(&IOStatsParameters {
            file_format: FileFormat::LAS,
            io_method: IOMethod::Mmap,
            point_record_format,
        }).ok_or_else(|| anyhow!("No statistics for point record format {point_record_format} of mmapped LAS file found"))?;
        let points_per_second = million_points_per_second * 1e6;
        let expected_time_seconds = point_range.len() as f64 / points_per_second;
        Ok(Duration::from_secs_f64(expected_time_seconds))
    }
}

#[self_referencing]
pub struct BorrowedLasPointData {
    pub mapped_file: Arc<MappedLASFile>,
    #[covariant]
    #[borrows(mapped_file)]
    pub buffer_slice: BufferSliceInterleaved<'this, ExternalMemoryBuffer<&'this [u8]>>,
}

impl BorrowedLasPointData {
    pub(crate) fn from_file_and_range(
        mapped_file: Arc<MappedLASFile>,
        point_range: Range<usize>,
    ) -> Self {
        BorrowedLasPointDataBuilder {
            mapped_file,
            buffer_slice_builder: |borrowed_file: &Arc<MappedLASFile>| {
                borrowed_file.get_buffer_for_points(point_range)
            },
        }
        .build()
    }
}

/// LAS point reader that uses a (buffered) file instead of `mmap`, since this is faster on macOS
pub(crate) struct LASPointDataReaderFile {
    metadata: LASMetadata,
    default_point_layout: PointLayout,
    path: PathBuf,
    las_reader: ThreadLocal<RefCell<LASReader<'static, BufReader<File>>>>,
}

impl LASPointDataReaderFile {
    pub(crate) fn new(path: &Path) -> Result<Self> {
        let las_reader = LASReader::from_path(path, true)?;
        let metadata = las_reader.las_metadata().clone();
        Ok(Self {
            metadata,
            default_point_layout: las_reader.get_default_point_layout().clone(),
            path: path.to_path_buf(),
            las_reader: Default::default(),
            // las_reader: Mutex::new(las_reader),
        })
    }

    fn get_thread_local_reader(&self) -> Result<&RefCell<LASReader<'static, BufReader<File>>>> {
        let _span = tracy_client::span!("LASPointDataReaderFile::get_thread_local_reader");
        self.las_reader.get_or_try(|| {
            let reader = LASReader::from_path(&self.path, true).with_context(|| {
                format!("Failed to open reader to LAS file {}", self.path.display())
            })?;
            Ok(RefCell::new(reader))
        })
    }
}

impl PointDataLoader for LASPointDataReaderFile {
    fn get_point_data(
        &self,
        point_range: Range<usize>,
        target_layout: &PointLayout,
        positions_in_world_space: bool,
    ) -> Result<PointData> {
        let _span = tracy_client::span!("LASPointDataReaderFile::get_point_data");

        let points = {
            let reader = self.get_thread_local_reader()?;
            let mut reader_mut = reader.borrow_mut();
            {
                let _span = tracy_client::span!("LASPointDataReaderFile::seek_point");
                reader_mut.seek_point(SeekFrom::Start(point_range.start as u64))?;
            }
            reader_mut.read::<VectorBuffer>(point_range.len())?
        };

        let source_layout = &self.default_point_layout;
        if target_layout == source_layout && !positions_in_world_space {
            Ok(PointData::OwnedInterleaved(points))
        } else {
            let _span =
                tracy_client::span!("LASPointDataReaderFile::get_point_data_with_conversion");
            // Read the points in target layout!
            let mut converter =
                BufferLayoutConverter::for_layouts_with_default(source_layout, target_layout);
            // By default, LAS positions are in local space, but we might want them in world space, especially if the
            // dataset contains multiple files. This requires the output PointLayout to have a POSITION_3D attribute!
            if positions_in_world_space {
                let source_positions_attribute = ATTRIBUTE_LOCAL_LAS_POSITION;
                let target_positions_attribute = target_layout
                    .get_attribute_by_name(POSITION_3D.name())
                    .ok_or(anyhow!("No POSITION_3D attribute found in target layout"))?;
                if target_positions_attribute.datatype() != PointAttributeDataType::Vec3f64 {
                    bail!("Outputting positions in world space is only possible if the POSITION_3D attribute of the output layout has datatype Vec3f64!");
                }

                let transforms = *self
                    .metadata
                    .raw_las_header()
                    .ok_or(anyhow!("Could not get LAS header"))?
                    .transforms();

                converter.set_custom_mapping_with_transformation(
                    &source_positions_attribute,
                    target_positions_attribute.attribute_definition(),
                    move |position: Vector3<f64>| -> Vector3<f64> {
                        Vector3::new(
                            position.x * transforms.x.scale + transforms.x.offset,
                            position.y * transforms.y.scale + transforms.y.offset,
                            position.z * transforms.z.scale + transforms.z.offset,
                        )
                    },
                    false,
                );
            }
            let converted_points = converter.convert::<VectorBuffer, _>(&points);
            Ok(PointData::OwnedInterleaved(converted_points))
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
        _value_type: crate::index::ValueType,
    ) -> Result<Duration> {
        let io_stats =
            IOStats::global().ok_or_else(|| anyhow!("Could not get global I/O stats"))?;
        let point_record_format = self
            .metadata
            .point_format()
            .to_u8()
            .context("Unsupported point record format")?;
        let million_points_per_second = io_stats
            .throughputs_mpts()
            .get(&IOStatsParameters {
                file_format: FileFormat::LAS,
                io_method: IOMethod::File,
                point_record_format,
            })
            .ok_or_else(|| {
                anyhow!(
                    "No statistics for point record format {point_record_format} of LAS file found"
                )
            })?;
        let points_per_second = million_points_per_second * 1e6;
        let expected_time_seconds = point_range.len() as f64 / points_per_second;
        Ok(Duration::from_secs_f64(expected_time_seconds))
    }
}
