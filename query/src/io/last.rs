use anyhow::{bail, Context, Result};
use io::last::LASTReader;
use pasture_core::{
    containers::HashMapBuffer,
    layout::{attributes::POSITION_3D, PointLayout},
};
use pasture_io::{
    base::{PointReader, SeekToPoint},
    las::{point_layout_from_las_metadata, LASMetadata, LASReader},
};
use std::{
    fs::File,
    io::{Cursor, SeekFrom},
    path::Path,
};

use crate::io::PointData;

use super::PointDataLoader;

pub(crate) struct LASTPointDataReader {
    mmap: memmap::Mmap,
    las_metadata: LASMetadata,
    default_point_layout: PointLayout,
}

impl LASTPointDataReader {
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
            las_metadata,
            default_point_layout,
        })
    }
}

impl PointDataLoader for LASTPointDataReader {
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
        last_reader.read_into(&mut buffer, point_range.len())?;
        Ok(PointData::OwnedColumnar(buffer))

        // if *target_layout == self.default_point_layout && !positions_in_world_space {
        //     let points = last_reader.read::<HashMapBuffer>(point_range.len())?;
        //     Ok(PointData::OwnedColumnar(points))
        // } else {
        //     let mut converter = BufferLayoutConverter::for_layouts_with_default(
        //         self.default_point_layout(),
        //         target_layout,
        //     );
        //     // By default, LAS positions are in local space, but we might want them in world space, especially if the
        //     // dataset contains multiple files. This requires the output PointLayout to have a POSITION_3D attribute!
        //     if positions_in_world_space {
        //         let source_positions_attribute =
        //             POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3i32);
        //         let target_positions_attribute = target_layout
        //             .get_attribute_by_name(POSITION_3D.name())
        //             .ok_or(anyhow!("No POSITION_3D attribute found in target layout"))?;
        //         if target_positions_attribute.datatype() != PointAttributeDataType::Vec3f64 {
        //             bail!("Outputting positions in world space is only possible if the POSITION_3D attribute of the output layout has datatype Vec3f64!");
        //         }

        //         let transforms = *self
        //             .las_metadata
        //             .raw_las_header()
        //             .ok_or(anyhow!("Could not get LAS header"))?
        //             .transforms();

        //         converter.set_custom_mapping_with_transformation(
        //             &source_positions_attribute,
        //             target_positions_attribute.attribute_definition(),
        //             move |position: Vector3<f64>| -> Vector3<f64> {
        //                 Vector3::new(
        //                     position.x * transforms.x.scale + transforms.x.offset,
        //                     position.y * transforms.y.scale + transforms.y.offset,
        //                     position.z * transforms.z.scale + transforms.z.offset,
        //                 )
        //             },
        //         );
        //     }

        //     let source_points = last_reader.read::<HashMapBuffer>(point_range.len())?;
        //     let converted_points = converter.convert::<HashMapBuffer, _>(&source_points);
        //     Ok(PointData::OwnedColumnar(converted_points))
        // }
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
