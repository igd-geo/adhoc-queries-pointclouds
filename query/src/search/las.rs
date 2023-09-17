use crate::{
    index::{
        Classification, CompareExpression, DatasetID, Geometry, GpsTime, NumberOfReturns,
        PointRange, Position, ReturnNumber,
    },
    io::{FileHandle, InputLayer, PointData},
    stats::BlockQueryRuntimeTracker,
};
use anyhow::{anyhow, bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};

use geo::{coord, Contains, LineString, Polygon};
use pasture_core::{
    containers::{AttributeView, BorrowedBuffer},
    layout::{
        attributes::{CLASSIFICATION, GPS_TIME, NUMBER_OF_RETURNS, POSITION_3D, RETURN_NUMBER},
        PointAttributeDataType, PointAttributeDefinition, PointLayout, PrimitiveType,
    },
    math::AABB,
    nalgebra::{clamp, Point3, Vector3},
};
use pasture_io::{
    las::{ATTRIBUTE_BASIC_FLAGS, ATTRIBUTE_EXTENDED_FLAGS},
    las_rs::{raw, Transform, Vector},
};
use std::io::SeekFrom;
use std::io::{Cursor, Seek};
use std::ops::Range;

use super::{eval_impl, CompiledQueryAtom};

/// Convert a world-space position into the local integer-value space of the given LAS file using the offset and scale parameters in the header.
/// If the position is outside the representable range, i32::MIN/i32::MAX values are returned instead
pub(crate) fn to_local_integer_position(
    position_world: &Vector3<f64>,
    las_transforms: &pasture_io::las_rs::Vector<Transform>,
) -> Vector3<i32> {
    let local_x = (position_world.x / las_transforms.x.scale) - las_transforms.x.offset;
    let local_y = (position_world.y / las_transforms.y.scale) - las_transforms.y.offset;
    let local_z = (position_world.z / las_transforms.z.scale) - las_transforms.z.offset;
    Vector3::new(
        clamp(local_x, i32::MIN as f64, i32::MAX as f64) as i32,
        clamp(local_y, i32::MIN as f64, i32::MAX as f64) as i32,
        clamp(local_z, i32::MIN as f64, i32::MAX as f64) as i32,
    )
}

/// Calculates the world-space bounding box for the given range of points in the given file
fn _get_bounds_of_point_range(
    file: &mut Cursor<&[u8]>,
    file_header: &raw::Header,
    point_range: Range<usize>,
) -> Result<AABB<f64>> {
    let mut local_min = Vector3::<i32>::new(i32::MAX, i32::MAX, i32::MAX);
    let mut local_max = Vector3::<i32>::new(i32::MIN, i32::MIN, i32::MIN);
    for point_index in point_range {
        let point_offset: u64 = point_index as u64 * file_header.point_data_record_length as u64
            + file_header.offset_to_point_data as u64;
        file.seek(SeekFrom::Start(point_offset))?;

        let x = file.read_i32::<LittleEndian>()?;
        let y = file.read_i32::<LittleEndian>()?;
        let z = file.read_i32::<LittleEndian>()?;

        local_min.x = local_min.x.min(x);
        local_min.y = local_min.y.min(y);
        local_min.z = local_min.z.min(z);

        local_max.x = local_max.x.max(x);
        local_max.y = local_max.y.max(y);
        local_max.z = local_max.z.max(z);
    }

    Ok(AABB::from_min_max(
        Point3::new(
            (local_min.x as f64 * file_header.x_scale_factor) + file_header.x_offset,
            (local_min.y as f64 * file_header.y_scale_factor) + file_header.y_offset,
            (local_min.z as f64 * file_header.z_scale_factor) + file_header.z_offset,
        ),
        Point3::new(
            (local_max.x as f64 * file_header.x_scale_factor) + file_header.x_offset,
            (local_max.y as f64 * file_header.y_scale_factor) + file_header.y_offset,
            (local_max.z as f64 * file_header.z_scale_factor) + file_header.z_offset,
        ),
    ))
}

fn line_string_to_local_las_space(
    line_string: &LineString,
    las_transforms: &Vector<Transform>,
) -> LineString {
    let local_coords = line_string
        .0
        .iter()
        .map(|coord| {
            coord! {
                x: (coord.x / las_transforms.x.scale) - las_transforms.x.offset,
                y: (coord.y / las_transforms.y.scale) - las_transforms.y.offset,
            }
        })
        .collect();
    LineString(local_coords)
}

/// Converts the given `geometry` to the local coordinate system of a LAS file, based on the given LAS transforms
fn geometry_to_local_las_space(
    geometry: &Geometry,
    las_transforms: &Vector<Transform>,
) -> Geometry {
    match geometry {
        Geometry::Polygon(world_space_polygon) => {
            let local_exterior =
                line_string_to_local_las_space(world_space_polygon.exterior(), las_transforms);
            let local_interiors = world_space_polygon
                .interiors()
                .iter()
                .map(|line_str| line_string_to_local_las_space(line_str, las_transforms))
                .collect();
            Geometry::Polygon(Polygon::new(local_exterior, local_interiors))
        }
    }
}

fn return_number_from_las_basic_flags(flags: u8) -> u8 {
    flags & 0b111
}

fn return_number_from_las_extended_flags(flags: u16) -> u8 {
    (flags & 0b1111) as u8
}

fn number_of_returns_from_las_basic_flags(flags: u8) -> u8 {
    (flags >> 3) & 0b111
}

fn number_of_returns_from_las_extended_flags(flags: u16) -> u8 {
    ((flags >> 4) & 0b1111) as u8
}

/// Returns data from the input layer for the given point range that contains at least the given `attribute`. The input layer
/// might decide to return point data containing more attributes, if this is more efficient, for example because the underlying
/// file might be mapped into memory as is the case for LAS files
fn get_data_with_at_least_attribute(
    attribute: &PointAttributeDefinition,
    input_layer: &InputLayer,
    dataset_id: DatasetID,
    block: PointRange,
) -> Result<PointData> {
    let default_layout =
        input_layer.get_default_point_layout_of_file(FileHandle(dataset_id, block.file_index))?;
    // Queries might use higher-level attributes such as RETURN_NUMBER that are stored in bitfields inside the LAS family
    // of files, so we have to perform an appropriate mapping
    let is_extended_format = default_layout.has_attribute(&ATTRIBUTE_EXTENDED_FLAGS);
    let native_attribute = if *attribute == RETURN_NUMBER || *attribute == NUMBER_OF_RETURNS {
        if is_extended_format {
            ATTRIBUTE_EXTENDED_FLAGS
        } else {
            ATTRIBUTE_BASIC_FLAGS
        }
    } else {
        attribute.clone()
    };

    if input_layer.can_get_borrowed_point_data(dataset_id, block.clone())? {
        input_layer
            .get_point_data(dataset_id, block.clone())
            .context("Could not access point data")
    } else {
        let custom_layout = PointLayout::from_attributes(&[native_attribute]);
        input_layer.get_point_data_in_layout(
            dataset_id,
            block.clone(),
            &custom_layout,
            false, //LAS family of files never uses positions in world space
        )
    }
}

/// Implementation of the 'Within' query for LAS files
pub(crate) struct LasQueryAtomWithin<T> {
    min: T,
    max: T,
}

impl<T> LasQueryAtomWithin<T> {
    pub(crate) fn new(min: T, max: T) -> Self {
        Self { min, max }
    }
}

/// Implementation of the 'Compare' query for LAS files
pub(crate) struct LasQueryAtomCompare<T> {
    value: T,
    compare_expression: CompareExpression,
}

impl<T> LasQueryAtomCompare<T> {
    pub(crate) fn new(value: T, compare_expression: CompareExpression) -> Self {
        Self {
            value,
            compare_expression,
        }
    }

    fn eval_primitive<U: PrimitiveType + PartialEq + PartialOrd, B: for<'a> BorrowedBuffer<'a>>(
        &self,
        primitive_attributes: AttributeView<'_, '_, B, U>,
        compare_value: U,
        block: PointRange,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        match self.compare_expression {
            CompareExpression::Equals => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let attribute = primitive_attributes.at(point_index);
                    Ok(attribute == compare_value)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::NotEquals => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let attribute = primitive_attributes.at(point_index);
                    Ok(attribute != compare_value)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::LessThan => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let attribute = primitive_attributes.at(point_index);
                    Ok(attribute < compare_value)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::LessThanOrEquals => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let attribute = primitive_attributes.at(point_index);
                    Ok(attribute <= compare_value)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::GreaterThan => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let attribute = primitive_attributes.at(point_index);
                    Ok(attribute > compare_value)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::GreaterThanOrEquals => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let attribute = primitive_attributes.at(point_index);
                    Ok(attribute >= compare_value)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
        }
    }

    fn eval_flags_primitive<
        Flag: PrimitiveType,
        U: PrimitiveType + PartialEq + PartialOrd,
        B: for<'a> BorrowedBuffer<'a>,
    >(
        &self,
        flags: AttributeView<'_, '_, B, Flag>,
        extractor: impl Fn(Flag) -> U,
        compare_value: U,
        block: PointRange,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        match self.compare_expression {
            CompareExpression::Equals => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let attribute = extractor(flags.at(point_index));
                    Ok(attribute == compare_value)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::NotEquals => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let attribute = extractor(flags.at(point_index));
                    Ok(attribute != compare_value)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::LessThan => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let attribute = extractor(flags.at(point_index));
                    Ok(attribute < compare_value)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::LessThanOrEquals => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let attribute = extractor(flags.at(point_index));
                    Ok(attribute <= compare_value)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::GreaterThan => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let attribute = extractor(flags.at(point_index));
                    Ok(attribute > compare_value)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::GreaterThanOrEquals => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let attribute = extractor(flags.at(point_index));
                    Ok(attribute >= compare_value)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
        }
    }
}

impl CompiledQueryAtom for LasQueryAtomWithin<Position> {
    fn eval(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        let las_metadata = input_layer
            .get_las_metadata(FileHandle(dataset_id, block.file_index))
            .context("Could not get LAS metadata for file")?;

        let las_transforms = las_metadata.raw_las_header().unwrap().transforms();

        let local_min = to_local_integer_position(&self.min.0, las_transforms);
        let local_max = to_local_integer_position(&self.max.0, las_transforms);
        let local_bounds = AABB::from_min_max(local_min.into(), local_max.into());

        // We can access the raw position data using a pasture `AttributeView`
        let local_position_attribute =
            POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3i32);
        let point_data = get_data_with_at_least_attribute(
            &local_position_attribute,
            input_layer,
            dataset_id,
            block.clone(),
        )?;
        let positions = point_data.view_attribute::<Vector3<i32>>(
            &POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3i32),
        );

        let test_point = |point_index: usize| -> Result<bool> {
            let position = positions.at(point_index);
            Ok(local_bounds.contains(&position.into()))
        };

        eval_impl(
            block,
            matching_indices,
            which_indices_to_loop_over,
            test_point,
            runtime_tracker,
        )
    }
}

impl CompiledQueryAtom for LasQueryAtomWithin<Classification> {
    fn eval(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        let point_data = get_data_with_at_least_attribute(
            &CLASSIFICATION,
            input_layer,
            dataset_id,
            block.clone(),
        )?;
        let classifications = point_data.view_attribute::<u8>(&CLASSIFICATION);

        let test_point = |point_index: usize| -> Result<bool> {
            let classification = classifications.at(point_index);
            Ok(classification >= self.min.0 && classification < self.max.0)
        };

        eval_impl(
            block,
            matching_indices,
            which_indices_to_loop_over,
            test_point,
            runtime_tracker,
        )
    }
}

impl CompiledQueryAtom for LasQueryAtomWithin<ReturnNumber> {
    fn eval(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        // The bit flag attributes are a bit different, we ask for the high-level attribute (i.e. RETURN_NUMBER)
        // but the `PointData` will still contain the low-level flag attribute (e.g. ATTRIBUTE_BASIC_FLAGS)
        let point_data = get_data_with_at_least_attribute(
            &RETURN_NUMBER,
            input_layer,
            dataset_id,
            block.clone(),
        )?;

        // Since we are working with raw LAS-like files, we can't access the bit-flag attributes separately and instead
        // have to use either `ATTRIBUTE_BASIC_FLAGS` or `ATTRIBUTE_EXTENDED_FLAGS`
        if point_data
            .point_layout()
            .has_attribute(&ATTRIBUTE_BASIC_FLAGS)
        {
            let flags = point_data.view_attribute::<u8>(&ATTRIBUTE_BASIC_FLAGS);
            let test_point = |point_index: usize| -> Result<bool> {
                let flag = flags.at(point_index);
                let return_number = return_number_from_las_basic_flags(flag);
                Ok(return_number >= self.min.0 && return_number < self.max.0)
            };

            eval_impl(
                block,
                matching_indices,
                which_indices_to_loop_over,
                test_point,
                runtime_tracker,
            )
        } else {
            let extended_flags = point_data.view_attribute::<u16>(&ATTRIBUTE_EXTENDED_FLAGS);
            let test_point = |point_index: usize| -> Result<bool> {
                let flag = extended_flags.at(point_index);
                let return_number = return_number_from_las_extended_flags(flag);
                Ok(return_number >= self.min.0 && return_number < self.max.0)
            };

            eval_impl(
                block,
                matching_indices,
                which_indices_to_loop_over,
                test_point,
                runtime_tracker,
            )
        }
    }
}

impl CompiledQueryAtom for LasQueryAtomWithin<NumberOfReturns> {
    fn eval(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        // The bit flag attributes are a bit different, we ask for the high-level attribute (i.e. NUMBER_OF_RETURNS)
        // but the `PointData` will still contain the low-level flag attribute (e.g. ATTRIBUTE_BASIC_FLAGS)
        let point_data = get_data_with_at_least_attribute(
            &NUMBER_OF_RETURNS,
            input_layer,
            dataset_id,
            block.clone(),
        )?;

        if point_data
            .point_layout()
            .has_attribute(&ATTRIBUTE_BASIC_FLAGS)
        {
            let flags = point_data.view_attribute::<u8>(&ATTRIBUTE_BASIC_FLAGS);
            let test_point = |point_index: usize| -> Result<bool> {
                let flag = flags.at(point_index);
                let number_of_returns = number_of_returns_from_las_basic_flags(flag);
                Ok(number_of_returns >= self.min.0 && number_of_returns < self.max.0)
            };

            eval_impl(
                block,
                matching_indices,
                which_indices_to_loop_over,
                test_point,
                runtime_tracker,
            )
        } else {
            let extended_flags = point_data.view_attribute::<u16>(&ATTRIBUTE_EXTENDED_FLAGS);
            let test_point = |point_index: usize| -> Result<bool> {
                let flag = extended_flags.at(point_index);
                let number_of_returns = number_of_returns_from_las_extended_flags(flag);
                Ok(number_of_returns >= self.min.0 && number_of_returns < self.max.0)
            };

            eval_impl(
                block,
                matching_indices,
                which_indices_to_loop_over,
                test_point,
                runtime_tracker,
            )
        }
    }
}

impl CompiledQueryAtom for LasQueryAtomWithin<GpsTime> {
    fn eval(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        let point_data =
            get_data_with_at_least_attribute(&GPS_TIME, input_layer, dataset_id, block.clone())?;
        let gps_times = point_data.view_attribute::<f64>(&GPS_TIME);

        let test_point = |point_index: usize| -> Result<bool> {
            let gps_time = gps_times.at(point_index);
            Ok(gps_time >= self.min.0 && gps_time < self.max.0)
        };

        eval_impl(
            block,
            matching_indices,
            which_indices_to_loop_over,
            test_point,
            runtime_tracker,
        )
    }
}

impl CompiledQueryAtom for LasQueryAtomCompare<Position> {
    fn eval(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        let las_metadata = input_layer
            .get_las_metadata(FileHandle(dataset_id, block.file_index))
            .context("Could not get LAS metadata for file")?;

        let las_transforms = las_metadata.raw_las_header().unwrap().transforms();
        let local_position = to_local_integer_position(&self.value.0, las_transforms);

        // We can access the raw position data using a pasture `AttributeView`
        let local_position_attribute =
            POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3i32);
        let point_data = get_data_with_at_least_attribute(
            &local_position_attribute,
            input_layer,
            dataset_id,
            block.clone(),
        )?;
        let positions = point_data.view_attribute::<Vector3<i32>>(&local_position_attribute);

        match self.compare_expression {
            CompareExpression::Equals => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let position = positions.at(point_index);
                    Ok(position == local_position)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::NotEquals => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let position = positions.at(point_index);
                    Ok(position != local_position)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            other => bail!("Unsupported compare expression {other:#?} for Position attribute"),
        }
    }
}

impl CompiledQueryAtom for LasQueryAtomCompare<Classification> {
    fn eval(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        let point_data = get_data_with_at_least_attribute(
            &CLASSIFICATION,
            input_layer,
            dataset_id,
            block.clone(),
        )?;
        let classifications = point_data.view_attribute::<u8>(&CLASSIFICATION);

        self.eval_primitive(
            classifications,
            self.value.0,
            block,
            matching_indices,
            which_indices_to_loop_over,
            runtime_tracker,
        )
    }
}

impl CompiledQueryAtom for LasQueryAtomCompare<ReturnNumber> {
    fn eval(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        // The bit flag attributes are a bit different, we ask for the high-level attribute (i.e. RETURN_NUMBER)
        // but the `PointData` will still contain the low-level flag attribute (e.g. ATTRIBUTE_BASIC_FLAGS)
        let point_data = get_data_with_at_least_attribute(
            &RETURN_NUMBER,
            input_layer,
            dataset_id,
            block.clone(),
        )?;

        if point_data
            .point_layout()
            .has_attribute(&ATTRIBUTE_BASIC_FLAGS)
        {
            let flags = point_data.view_attribute::<u8>(&ATTRIBUTE_BASIC_FLAGS);
            self.eval_flags_primitive(
                flags,
                return_number_from_las_basic_flags,
                self.value.0,
                block,
                matching_indices,
                which_indices_to_loop_over,
                runtime_tracker,
            )
        } else {
            let extended_flags = point_data.view_attribute::<u16>(&ATTRIBUTE_EXTENDED_FLAGS);
            self.eval_flags_primitive(
                extended_flags,
                return_number_from_las_extended_flags,
                self.value.0,
                block,
                matching_indices,
                which_indices_to_loop_over,
                runtime_tracker,
            )
        }
    }
}

impl CompiledQueryAtom for LasQueryAtomCompare<NumberOfReturns> {
    fn eval(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        // The bit flag attributes are a bit different, we ask for the high-level attribute (i.e. NUMBER_OF_RETURNS)
        // but the `PointData` will still contain the low-level flag attribute (e.g. ATTRIBUTE_BASIC_FLAGS)
        let point_data = get_data_with_at_least_attribute(
            &NUMBER_OF_RETURNS,
            input_layer,
            dataset_id,
            block.clone(),
        )?;

        if point_data
            .point_layout()
            .has_attribute(&ATTRIBUTE_BASIC_FLAGS)
        {
            let flags = point_data.view_attribute::<u8>(&ATTRIBUTE_BASIC_FLAGS);
            self.eval_flags_primitive(
                flags,
                number_of_returns_from_las_basic_flags,
                self.value.0,
                block,
                matching_indices,
                which_indices_to_loop_over,
                runtime_tracker,
            )
        } else {
            let extended_flags = point_data.view_attribute::<u16>(&ATTRIBUTE_EXTENDED_FLAGS);
            self.eval_flags_primitive(
                extended_flags,
                number_of_returns_from_las_extended_flags,
                self.value.0,
                block,
                matching_indices,
                which_indices_to_loop_over,
                runtime_tracker,
            )
        }
    }
}

impl CompiledQueryAtom for LasQueryAtomCompare<GpsTime> {
    fn eval(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        let point_data =
            get_data_with_at_least_attribute(&GPS_TIME, input_layer, dataset_id, block.clone())?;
        let gps_times = point_data.view_attribute::<f64>(&GPS_TIME);

        self.eval_primitive(
            gps_times,
            self.value.0,
            block,
            matching_indices,
            which_indices_to_loop_over,
            runtime_tracker,
        )
    }
}

pub(crate) struct LasQueryAtomIntersects {
    geometry: Geometry,
}

impl LasQueryAtomIntersects {
    pub(crate) fn new(geometry: Geometry) -> Self {
        Self { geometry }
    }
}

impl CompiledQueryAtom for LasQueryAtomIntersects {
    fn eval(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        let las_metadata = input_layer
            .get_las_metadata(FileHandle(dataset_id, block.file_index))
            .ok_or(anyhow!("No LAS metadata found"))?;
        let las_transforms = las_metadata
            .raw_las_header()
            .ok_or(anyhow!("No LAS header found"))?
            .transforms();
        let local_geometry = geometry_to_local_las_space(&self.geometry, las_transforms);

        let local_position_attribute =
            &POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3i32);
        let point_data = get_data_with_at_least_attribute(
            &local_position_attribute,
            input_layer,
            dataset_id,
            block.clone(),
        )?;

        let positions = point_data.view_attribute::<Vector3<i32>>(local_position_attribute);

        match local_geometry {
            Geometry::Polygon(polygon) => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let position = positions.at(point_index);
                    let geo_point = geo::Point(coord! {
                        x: position.x as f64,
                        y: position.y as f64,
                    });
                    Ok(polygon.contains(&geo_point))
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
        }
    }
}
