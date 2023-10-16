use std::time::Instant;

use anyhow::{bail, Result};
use pasture_core::layout::PrimitiveType;

use crate::{
    index::{AtomicExpression, DatasetID, PointRange, QueryExpression, Value},
    io::InputLayer,
    stats::{BlockQueryRuntimeTracker, BlockQueryRuntimeType},
};

use super::{LasQueryAtomCompare, LasQueryAtomIntersects, LasQueryAtomWithin};

/// A single piece of a query compiled into an optimized format so that we can pass it the raw memory of a file (LAS, LAZ, LAST, etc.)
/// together with the file header. Evaluating a query atom sets all matching indices in `matching_indices` to true
pub trait CompiledQueryAtom: Sync + Send {
    fn eval(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize>;
}

/// Evaluating a query atom calculates matching indices. In the base case, we iterate over `block`, which is a range describing the point indices
/// in the current block. If we use the AND combinator, the second expression can be optimized by only checking the matching indices that the
/// first expression produced. This is the `Matching` literal. When using the OR combinator, it goes the other way around, the second expression
/// doesn't have to check the matching indices from the first expression, these are already matches, so it only has to check those indices that
/// did not match previously. This is the `NotMatching` literal
pub enum WhichIndicesToLoopOver {
    All,
    Matching,
    NotMatching,
}

pub enum CompiledQueryExpression {
    Atom(Box<dyn CompiledQueryAtom>),
    And(Box<CompiledQueryExpression>, Box<CompiledQueryExpression>),
    Or(Box<CompiledQueryExpression>, Box<CompiledQueryExpression>),
}

impl CompiledQueryExpression {
    pub fn eval<'a>(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'a mut [bool],
        num_matches: usize,
        which_indices: WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        let _span = tracy_client::span!("eval_compiled_query");

        match self {
            CompiledQueryExpression::Atom(atom) => atom.eval(
                input_layer,
                block,
                dataset_id,
                matching_indices,
                which_indices,
                runtime_tracker,
            ),
            CompiledQueryExpression::And(left, right) => {
                let num_matches_left = left.eval(
                    input_layer,
                    block.clone(),
                    dataset_id,
                    matching_indices,
                    num_matches,
                    which_indices,
                    runtime_tracker,
                )?;
                // second expression doesn't have to consider indices which are already false
                right.eval(
                    input_layer,
                    block,
                    dataset_id,
                    matching_indices,
                    num_matches_left,
                    WhichIndicesToLoopOver::Matching,
                    runtime_tracker,
                )
            }
            CompiledQueryExpression::Or(left, right) => {
                let num_matches_left = left.eval(
                    input_layer,
                    block.clone(),
                    dataset_id,
                    matching_indices,
                    num_matches,
                    which_indices,
                    runtime_tracker,
                )?;
                // second expression doesn't have to consider indices which are already true
                right.eval(
                    input_layer,
                    block,
                    dataset_id,
                    matching_indices,
                    num_matches_left,
                    WhichIndicesToLoopOver::NotMatching,
                    runtime_tracker,
                )
            }
        }
    }
}

/// Query the given block using the given compiled query. Data is extracted using the file-format-specific extractor and sent to result_collector
// pub fn query_block(
//     file: &mut Cursor<&[u8]>,
//     file_header: &Header,
//     block: Range<usize>,
//     matching_indices: &mut [bool],
//     compiled_query: &CompiledQueryExpression,
//     extractor: &dyn Extractor,
//     result_collector: &mut dyn ResultCollector,
// ) -> Result<()> {
//     let num_matches = compiled_query.eval(
//         file,
//         file_header,
//         block.clone(),
//         matching_indices,
//         matching_indices.len(),
//         WhichIndicesToLoopOver::All,
//     )?;
//     if num_matches == 0 {
//         return Ok(());
//     }
//     let points = extractor.extract_data(file, file_header, block, matching_indices, num_matches)?;
//     result_collector.collect(points);
//     Ok(())
// }

pub(crate) fn compile_query(
    query: &QueryExpression,
    file_format: &str,
) -> Result<CompiledQueryExpression> {
    match query {
        QueryExpression::Atomic(atomic_expr) => match file_format {
            // TODO All LAS derivates seem to have identical implementations for their compiled queries? Maybe rename them
            // as such!
            "las" | "last" | "laz" | "lazer" => match atomic_expr {
                AtomicExpression::Within(range) => {
                    let las_expr: Box<dyn CompiledQueryAtom> = match (range.start, range.end) {
                        // TODO How can I match on 'both enums have the same type'?
                        (Value::Classification(min), Value::Classification(max)) => {
                            Box::new(LasQueryAtomWithin::new(min, max))
                        }
                        (Value::Position(min), Value::Position(max)) => {
                            Box::new(LasQueryAtomWithin::new(min, max))
                        }
                        (Value::GpsTime(min), Value::GpsTime(max)) => {
                            Box::new(LasQueryAtomWithin::new(min, max))
                        }
                        (Value::ReturnNumber(min), Value::ReturnNumber(max)) => {
                            Box::new(LasQueryAtomWithin::new(min, max))
                        }
                        (Value::NumberOfReturns(min), Value::NumberOfReturns(max)) => {
                            Box::new(LasQueryAtomWithin::new(min, max))
                        }
                        (min, max) => bail!(
                            "Wrong Value types ({},{}) for Within expression!",
                            min.value_type(),
                            max.value_type()
                        ),
                    };
                    Ok(CompiledQueryExpression::Atom(las_expr))
                }
                AtomicExpression::Compare((compare_expr, value)) => {
                    let las_expr: Box<dyn CompiledQueryAtom> = match value {
                        Value::Position(position) => {
                            Box::new(LasQueryAtomCompare::new(*position, *compare_expr))
                        }
                        Value::Classification(classification) => {
                            Box::new(LasQueryAtomCompare::new(*classification, *compare_expr))
                        }
                        Value::GpsTime(gps_time) => {
                            Box::new(LasQueryAtomCompare::new(*gps_time, *compare_expr))
                        }
                        Value::ReturnNumber(return_number) => {
                            Box::new(LasQueryAtomCompare::new(*return_number, *compare_expr))
                        }
                        Value::NumberOfReturns(number_of_returns) => {
                            Box::new(LasQueryAtomCompare::new(*number_of_returns, *compare_expr))
                        }
                        Value::LOD(discrete_lod) => {
                            Box::new(LasQueryAtomCompare::new(*discrete_lod, *compare_expr))
                        }
                    };
                    Ok(CompiledQueryExpression::Atom(las_expr))
                }
                AtomicExpression::Intersects(geometry) => Ok(CompiledQueryExpression::Atom(
                    Box::new(LasQueryAtomIntersects::new(geometry.clone())),
                )),
            },
            _ => bail!("Unsupported file format {}", file_format),
        },
        QueryExpression::And(l, r) => {
            let combined = CompiledQueryExpression::And(
                Box::new(compile_query(l, file_format)?),
                Box::new(compile_query(r, file_format)?),
            );
            Ok(combined)
        }
        QueryExpression::Or(l, r) => {
            let combined = CompiledQueryExpression::Or(
                Box::new(compile_query(l, file_format)?),
                Box::new(compile_query(r, file_format)?),
            );
            Ok(combined)
        }
    }
}

pub(crate) fn eval_impl<F: FnMut(U) -> Result<bool>, U: PrimitiveType>(
    block: PointRange,
    matching_indices: &'_ mut [bool],
    which_indices_to_loop_over: super::WhichIndicesToLoopOver,
    mut test_point: F,
    point_data: impl Iterator<Item = U>,
    runtime_tracker: &BlockQueryRuntimeTracker,
) -> Result<usize> {
    let timer = Instant::now();

    let mut num_matches = 0;
    match which_indices_to_loop_over {
        super::WhichIndicesToLoopOver::All => {
            assert!(block.points_in_file.len() <= matching_indices.len());
            for (point_data, index_matches) in point_data
                .skip(block.points_in_file.start)
                .zip(matching_indices.iter_mut())
                .take(block.points_in_file.len())
            {
                *index_matches = test_point(point_data)?;
                if *index_matches {
                    num_matches += 1;
                }
            }

            // for point_index in block.points_in_file.clone() {
            //     let local_index = point_index - block.points_in_file.start;
            //     matching_indices[local_index] = test_point(point_index)?;
            //     if matching_indices[local_index] {
            //         num_matches += 1;
            //     }
            // }
        }
        super::WhichIndicesToLoopOver::Matching => {
            for (point_data, index_matches) in point_data
                .skip(block.points_in_file.start)
                .zip(matching_indices.iter_mut())
                .take(block.points_in_file.len())
                .filter(|(_, is_match)| **is_match)
            {
                *index_matches = test_point(point_data)?;
                if *index_matches {
                    num_matches += 1;
                }
            }

            // for (local_index, is_match) in matching_indices
            //     .iter_mut()
            //     .enumerate()
            //     .filter(|(_, is_match)| **is_match)
            // {
            //     let point_index = local_index + block.points_in_file.start;
            //     *is_match = test_point(point_index)?;
            //     if *is_match {
            //         num_matches += 1;
            //     }
            // }
        }
        super::WhichIndicesToLoopOver::NotMatching => {
            for (point_data, index_matches) in point_data
                .skip(block.points_in_file.start)
                .zip(matching_indices.iter_mut())
                .take(block.points_in_file.len())
                .filter(|(_, is_match)| !**is_match)
            {
                *index_matches = test_point(point_data)?;
                if *index_matches {
                    num_matches += 1;
                }
            }
        }
    }

    runtime_tracker.log_runtime(block, BlockQueryRuntimeType::Eval, timer.elapsed());

    Ok(num_matches)
}
