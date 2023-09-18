use std::{
    fs::File,
    io::{Cursor, Seek, SeekFrom},
    path::Path,
};

use anyhow::{anyhow, bail, Context, Result};
use laz::{LasZipDecompressor, LazVlr};
use pasture_core::{
    containers::{InterleavedBufferMut, MakeBufferFromLayout, OwningBuffer, VectorBuffer},
    layout::{
        attributes::POSITION_3D, conversion::BufferLayoutConverter, PointAttributeDataType,
        PointLayout,
    },
    nalgebra::Vector3,
};
use pasture_io::{
    las::{point_layout_from_las_metadata, LASMetadata, LASReader},
    las_rs::Vlr,
};

use crate::io::PointData;

use super::PointDataLoader;

fn is_laszip_vlr(vlr: &Vlr) -> bool {
    vlr.user_id == laz::LazVlr::USER_ID && vlr.record_id == laz::LazVlr::RECORD_ID
}

pub(crate) struct LAZPointDataReader {
    mmap: memmap::Mmap,
    las_metadata: LASMetadata,
    laz_vlr: LazVlr,
    default_point_layout: PointLayout,
    offset_to_point_records: usize,
}

impl LAZPointDataReader {
    pub(crate) fn new(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("Failed to open LAZ file {}", path.display()))?;
        let mmap = unsafe {
            memmap::Mmap::map(&file)
                .with_context(|| format!("Failed to mmap LAZ file {}", path.display()))?
        };

        let las_metadata = LASReader::from_read(Cursor::new(&mmap), true)
            .with_context(|| format!("Failed to get metadata from LAZ file {}", path.display()))?
            .las_metadata()
            .clone();

        let header = las_metadata
            .raw_las_header()
            .ok_or(anyhow!("No LAZ header found"))?;
        let laz_vlr = match header.vlrs().iter().find(|vlr| is_laszip_vlr(vlr)) {
            None => Err(anyhow!("LAZ variable length record not found")),
            Some(vlr) => {
                let laz_record = laz::las::laszip::LazVlr::from_buffer(&vlr.data)
                    .map_err(|e| anyhow!("Could not parse laszip VLR: {e}"))?;
                Ok(laz_record)
            }
        }?;

        let raw_header = header.clone().into_raw()?;

        let default_point_layout = point_layout_from_las_metadata(&las_metadata, true)
            .with_context(|| {
                format!(
                    "No matching PointLayout found for LAZ file {}",
                    path.display()
                )
            })?;

        Ok(Self {
            mmap,
            las_metadata,
            laz_vlr,
            default_point_layout,
            offset_to_point_records: raw_header.offset_to_point_data as usize,
        })
    }
}

impl PointDataLoader for LAZPointDataReader {
    fn get_point_data(
        &self,
        point_range: std::ops::Range<usize>,
        target_layout: &PointLayout,
        positions_in_world_space: bool,
    ) -> Result<super::PointData> {
        let _span = tracy_client::span!("LAZ::get_point_data");

        if point_range.is_empty() {
            let empty_buffer = VectorBuffer::new_from_layout(self.default_point_layout.clone());
            return Ok(PointData::OwnedInterleaved(empty_buffer));
        }

        // The LAZ decompressor requires ALL of the files memory, but it must be at the start of the point data records
        // when calling `new`
        let mut point_data_reader = Cursor::new(&self.mmap[..]);
        point_data_reader.seek(SeekFrom::Start(self.offset_to_point_records as u64))?;

        let mut reader = LasZipDecompressor::new(point_data_reader, self.laz_vlr.clone())
            .map_err(|e| anyhow!("Could not create LAZ decompressor: {}", e))?;

        reader.seek(point_range.start as u64)?;

        // Regardless of the target layout, we first have to decompress all points in their default layout
        let mut point_buffer =
            VectorBuffer::with_capacity(point_range.len(), self.default_point_layout.clone());
        point_buffer.resize(point_range.len());
        reader
            .decompress_many(point_buffer.get_point_range_mut(0..point_range.len()))
            .context("Decompressing LAZ points failed")?;

        if *target_layout == self.default_point_layout {
            // If the target layout matches the default binary layout of the LAZ file, we can simply return the
            // decompressed buffer
            Ok(PointData::OwnedInterleaved(point_buffer))
        } else {
            let _span = tracy_client::span!("LAZ::get_point_data_with_conversion");
            // TODO Newer LAZ versions seem to support selective decompression, but only for point record formats 6-10
            // Might be worth to explore this, but certainly not for this prototype here...

            // For now, we simply convert to the target layout
            let mut converter = BufferLayoutConverter::for_layouts_with_default(
                &self.default_point_layout,
                target_layout,
            );
            // By default, LAS positions are in local space, but we might want them in world space, especially if the
            // dataset contains multiple files. This requires the output PointLayout to have a POSITION_3D attribute!
            if positions_in_world_space {
                let source_positions_attribute =
                    POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3i32);
                let target_positions_attribute = target_layout
                    .get_attribute_by_name(POSITION_3D.name())
                    .ok_or(anyhow!("No POSITION_3D attribute found in target layout"))?;
                if target_positions_attribute.datatype() != PointAttributeDataType::Vec3f64 {
                    bail!("Outputting positions in world space is only possible if the POSITION_3D attribute of the output layout has datatype Vec3f64!");
                }

                let transforms = *self
                    .las_metadata
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
                );
            }
            let converted_points = converter.convert::<VectorBuffer, _>(&point_buffer);
            Ok(PointData::OwnedInterleaved(converted_points))
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
}
