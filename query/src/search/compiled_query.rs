use anyhow::{bail, Result};

use crate::{
    index::{AtomicExpression, DatasetID, PointRange, QueryExpression, Value},
    io::InputLayer,
    stats::BlockQueryRuntimeTracker,
};

use super::{LasQueryAtomCompare, LasQueryAtomWithin};

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
            "las" => match atomic_expr {
                AtomicExpression::Within(range) => {
                    let las_expr: Box<dyn CompiledQueryAtom> = match (range.start, range.end) {
                        (Value::Classification(min), Value::Classification(max)) => {
                            Box::new(LasQueryAtomWithin::new(min, max))
                        }
                        (Value::Position(min), Value::Position(max)) => {
                            Box::new(LasQueryAtomWithin::new(min, max))
                        }
                        _ => bail!("Wrong Value types, min and max Value must have the same type!"),
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
                    };
                    Ok(CompiledQueryExpression::Atom(las_expr))
                }
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
