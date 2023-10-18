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
    fs::File,
    io::{BufReader, Cursor, SeekFrom},
    ops::Range,
    path::Path,
    sync::{Arc, Mutex},
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
        let _span = tracy_client::span!("LAS::get_point_data");

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
    las_reader: Mutex<LASReader<'static, BufReader<File>>>,
}

impl LASPointDataReaderFile {
    pub(crate) fn new(path: &Path) -> Result<Self> {
        let las_reader = LASReader::from_path(path, true)?;
        let metadata = las_reader.las_metadata().clone();
        Ok(Self {
            metadata,
            default_point_layout: las_reader.get_default_point_layout().clone(),
            las_reader: Mutex::new(las_reader),
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
        let _span = tracy_client::span!("LAS::get_point_data");

        let points = {
            let mut reader = self.las_reader.lock().expect("Lock was poisoned");
            reader.seek_point(SeekFrom::Start(point_range.start as u64))?;
            reader.read::<VectorBuffer>(point_range.len())?
        };

        let source_layout = &self.default_point_layout;
        if target_layout == source_layout && !positions_in_world_space {
            Ok(PointData::OwnedInterleaved(points))
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
}
