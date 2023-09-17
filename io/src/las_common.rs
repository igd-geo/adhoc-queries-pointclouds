use anyhow::{bail, Result};
use pasture_core::{
    layout::{
        attributes::{
            EDGE_OF_FLIGHT_LINE, NUMBER_OF_RETURNS, POSITION_3D, RETURN_NUMBER, SCAN_DIRECTION_FLAG,
        },
        conversion::BufferLayoutConverter,
        FieldAlignment, PointAttributeDataType, PointLayout,
    },
    nalgebra::Vector3,
};
use pasture_io::{
    las::{ATTRIBUTE_BASIC_FLAGS, ATTRIBUTE_EXTENDED_FLAGS},
    las_rs::raw,
};

pub fn get_minimum_layout_for_las_conversion(
    source_layout: &PointLayout,
    target_layout: &PointLayout,
) -> Result<PointLayout> {
    if source_layout.has_attribute(&ATTRIBUTE_EXTENDED_FLAGS) {
        bail!("Extended LAS-like formats (point records >= 6) are currently unsupported");
    }

    let mut common_attributes = target_layout
        .attributes()
        .filter_map(|a| {
            source_layout
                .get_attribute_by_name(a.name())
                .map(|a| a.attribute_definition().clone())
        })
        .collect::<PointLayout>();

    if target_layout.has_attribute_with_name(RETURN_NUMBER.name())
        || target_layout.has_attribute_with_name(NUMBER_OF_RETURNS.name())
        || target_layout.has_attribute_with_name(SCAN_DIRECTION_FLAG.name())
        || target_layout.has_attribute_with_name(EDGE_OF_FLIGHT_LINE.name())
    {
        common_attributes.add_attribute(ATTRIBUTE_BASIC_FLAGS, FieldAlignment::Default);
    }

    Ok(common_attributes)
}

/// Returns a `BufferLayoutConverter` for parsing LAS-like formats. It handles local-to-world-space position transformation
/// and extraction of the flag attributes into higher-level attributes like `RETURN_NUMBER` etc.
pub fn get_default_las_converter<'a>(
    source_layout: &'a PointLayout,
    target_layout: &'a PointLayout,
    raw_las_header: raw::Header,
) -> Result<BufferLayoutConverter<'a>> {
    if source_layout.has_attribute(&ATTRIBUTE_EXTENDED_FLAGS) {
        bail!("Extended LAS-like formats (point records >= 6) are currently unsupported");
    }

    let mut converter =
        BufferLayoutConverter::for_layouts_with_default(source_layout, target_layout);
    // Some custom parsing things:
    // - If target layout has positions as Vector3<f64>, we assume this means 'world space' and convert accordingly
    // - For all high-level flag attributes in target layout, we convert accordingly from the low level attributes.
    //   For example, RETURN_NUMBER in target will be converted from either ATTRIBUTE_BASIC_FLAGS or
    //   ATTRIBUTE_EXTENDED_FLAGS, depending on the underlying point format
    if target_layout.has_attribute(&POSITION_3D) {
        converter.set_custom_mapping_with_transformation(
            &POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3i32),
            &POSITION_3D,
            move |pos: Vector3<f64>| -> Vector3<f64> {
                Vector3::new(
                    (pos.x * raw_las_header.x_scale_factor) + raw_las_header.x_offset,
                    (pos.y * raw_las_header.y_scale_factor) + raw_las_header.y_offset,
                    (pos.z * raw_las_header.z_scale_factor) + raw_las_header.z_offset,
                )
            },
        );
    }
    if target_layout.has_attribute(&RETURN_NUMBER) {
        converter.set_custom_mapping_with_transformation(
            &ATTRIBUTE_BASIC_FLAGS,
            &RETURN_NUMBER,
            |flags: u8| -> u8 { flags & 0b111 },
        );
    }
    if target_layout.has_attribute(&NUMBER_OF_RETURNS) {
        converter.set_custom_mapping_with_transformation(
            &ATTRIBUTE_BASIC_FLAGS,
            &NUMBER_OF_RETURNS,
            |flags: u8| -> u8 { (flags >> 3) & 0b111 },
        );
    }
    if target_layout.has_attribute(&SCAN_DIRECTION_FLAG) {
        converter.set_custom_mapping_with_transformation(
            &ATTRIBUTE_BASIC_FLAGS,
            &SCAN_DIRECTION_FLAG,
            |flags: u8| -> u8 { (flags >> 6) & 1 },
        );
    }
    if target_layout.has_attribute(&EDGE_OF_FLIGHT_LINE) {
        converter.set_custom_mapping_with_transformation(
            &ATTRIBUTE_BASIC_FLAGS,
            &EDGE_OF_FLIGHT_LINE,
            |flags: u8| -> u8 { (flags >> 7) & 1 },
        );
    }

    Ok(converter)
}
