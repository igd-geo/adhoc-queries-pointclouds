use anyhow::{bail, Result};
use pasture_core::{
    layout::{
        attributes::{
            CLASSIFICATION_FLAGS, EDGE_OF_FLIGHT_LINE, NUMBER_OF_RETURNS, POSITION_3D,
            RETURN_NUMBER, SCANNER_CHANNEL, SCAN_DIRECTION_FLAG,
        },
        conversion::BufferLayoutConverter,
        FieldAlignment, PointAttributeDataType, PointLayout,
    },
    nalgebra::Vector3,
};
use pasture_io::{
    las::{ATTRIBUTE_BASIC_FLAGS, ATTRIBUTE_EXTENDED_FLAGS, ATTRIBUTE_LOCAL_LAS_POSITION},
    las_rs::raw,
};

pub fn get_minimum_layout_for_las_conversion(
    source_layout: &PointLayout,
    target_layout: &PointLayout,
) -> Result<PointLayout> {
    let is_extended = source_layout.has_attribute(&ATTRIBUTE_EXTENDED_FLAGS);

    let mut common_attributes = target_layout
        .attributes()
        .filter_map(|a| {
            source_layout
                .get_attribute_by_name(a.name())
                .map(|a| a.attribute_definition().clone())
        })
        .collect::<PointLayout>();

    if target_layout.has_attribute_with_name(POSITION_3D.name()) {
        common_attributes.add_attribute(ATTRIBUTE_LOCAL_LAS_POSITION, FieldAlignment::Default);
    }

    if is_extended {
        if target_layout.has_attribute_with_name(RETURN_NUMBER.name())
            || target_layout.has_attribute_with_name(NUMBER_OF_RETURNS.name())
            || target_layout.has_attribute_with_name(CLASSIFICATION_FLAGS.name())
            || target_layout.has_attribute_with_name(SCANNER_CHANNEL.name())
            || target_layout.has_attribute_with_name(SCAN_DIRECTION_FLAG.name())
            || target_layout.has_attribute_with_name(EDGE_OF_FLIGHT_LINE.name())
        {
            common_attributes.add_attribute(ATTRIBUTE_EXTENDED_FLAGS, FieldAlignment::Default);
        }
    } else if target_layout.has_attribute_with_name(RETURN_NUMBER.name())
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
    let is_extended = source_layout.has_attribute(&ATTRIBUTE_EXTENDED_FLAGS);

    let mut converter =
        BufferLayoutConverter::for_layouts_with_default(source_layout, target_layout);
    if let Some(position_attribute) = target_layout.get_attribute_by_name(POSITION_3D.name()) {
        match position_attribute.datatype() {
                PointAttributeDataType::Vec3f64 => converter.set_custom_mapping_with_transformation(&ATTRIBUTE_LOCAL_LAS_POSITION, position_attribute.attribute_definition(), move |pos: Vector3<f64>| -> Vector3<f64> {
                    Vector3::new(
                        (pos.x * raw_las_header.x_scale_factor) + raw_las_header.x_offset,
                        (pos.y * raw_las_header.y_scale_factor) + raw_las_header.y_offset,
                        (pos.z * raw_las_header.z_scale_factor) + raw_las_header.z_offset,
                    )
                }, false),
                PointAttributeDataType::Vec3f32 => converter.set_custom_mapping_with_transformation(&ATTRIBUTE_LOCAL_LAS_POSITION, position_attribute.attribute_definition(), move |pos: Vector3<f32>| -> Vector3<f32> {
                    Vector3::new(
                        ((pos.x as f64 * raw_las_header.x_scale_factor) + raw_las_header.x_offset) as f32,
                        ((pos.y as f64 * raw_las_header.y_scale_factor) + raw_las_header.y_offset) as f32,
                        ((pos.z as f64 * raw_las_header.z_scale_factor) + raw_las_header.z_offset) as f32,
                    )
                }, false),
                other => bail!("Invalid datatype {other} for POSITION_3D attribute. Only Vec3f64 and Vec3f32 are supported!"),
            }
    }
    if target_layout.has_attribute(&RETURN_NUMBER) {
        if is_extended {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_EXTENDED_FLAGS,
                &RETURN_NUMBER,
                |flags: u16| -> u16 { flags & 0b1111 },
                true,
            );
        } else {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_BASIC_FLAGS,
                &RETURN_NUMBER,
                |flags: u8| -> u8 { flags & 0b111 },
                false,
            );
        }
    }
    if target_layout.has_attribute(&NUMBER_OF_RETURNS) {
        if is_extended {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_EXTENDED_FLAGS,
                &NUMBER_OF_RETURNS,
                |flags: u16| -> u16 { (flags >> 4) & 0b1111 },
                true,
            );
        } else {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_BASIC_FLAGS,
                &NUMBER_OF_RETURNS,
                |flags: u8| -> u8 { (flags >> 3) & 0b111 },
                false,
            );
        }
    }
    if target_layout.has_attribute(&CLASSIFICATION_FLAGS) && is_extended {
        converter.set_custom_mapping_with_transformation(
            &ATTRIBUTE_EXTENDED_FLAGS,
            &CLASSIFICATION_FLAGS,
            |flags: u16| -> u16 { (flags >> 8) & 0b1111 },
            true,
        );
    }
    if target_layout.has_attribute(&SCANNER_CHANNEL) && is_extended {
        converter.set_custom_mapping_with_transformation(
            &ATTRIBUTE_EXTENDED_FLAGS,
            &SCANNER_CHANNEL,
            |flags: u16| -> u16 { (flags >> 12) & 0b11 },
            true,
        );
    }
    if target_layout.has_attribute(&SCAN_DIRECTION_FLAG) {
        if is_extended {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_EXTENDED_FLAGS,
                &SCAN_DIRECTION_FLAG,
                |flags: u16| -> u16 { (flags >> 14) & 1 },
                true,
            );
        } else {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_BASIC_FLAGS,
                &SCAN_DIRECTION_FLAG,
                |flags: u8| -> u8 { (flags >> 6) & 1 },
                false,
            );
        }
    }
    if target_layout.has_attribute(&EDGE_OF_FLIGHT_LINE) {
        if is_extended {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_EXTENDED_FLAGS,
                &EDGE_OF_FLIGHT_LINE,
                |flags: u16| -> u16 { (flags >> 15) & 1 },
                true,
            );
        } else {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_BASIC_FLAGS,
                &EDGE_OF_FLIGHT_LINE,
                |flags: u8| -> u8 { (flags >> 7) & 1 },
                false,
            );
        }
    }

    Ok(converter)
}
