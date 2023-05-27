use std::io::Cursor;

use anyhow::{bail, Result};
use pasture_io::las_rs::raw::Header;

use crate::{
    collect_points::PointBufferSend,
    index::{PointRange, QueryExpression, Value},
    stats::BlockQueryRuntimeTracker,
};

use super::{LasQueryAtomEquals, LasQueryAtomWithin};

/// A single piece of a query compiled into an optimized format so that we can pass it the raw memory of a file (LAS, LAZ, LAST, etc.)
/// together with the file header. Evaluating a query atom sets all matching indices in `matching_indices` to true
pub trait CompiledQueryAtom: Sync + Send {
    fn eval(
        &self,
        file: &mut Cursor<&[u8]>,
        file_header: &Header,
        block: PointRange,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize>;
}

pub trait Extractor {
    fn extract_data(
        &self,
        file: &mut Cursor<&[u8]>,
        file_header: &Header,
        block: PointRange,
        matching_indices: &[bool],
        num_matches: usize,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<Box<dyn PointBufferSend>>;
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
        file: &mut Cursor<&[u8]>,
        file_header: &Header,
        block: PointRange,
        matching_indices: &'a mut [bool],
        num_matches: usize,
        which_indices: WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        match self {
            CompiledQueryExpression::Atom(atom) => atom.eval(
                file,
                file_header,
                block,
                matching_indices,
                which_indices,
                runtime_tracker,
            ),
            CompiledQueryExpression::And(left, right) => {
                let num_matches_left = left.eval(
                    file,
                    file_header,
                    block.clone(),
                    matching_indices,
                    num_matches,
                    which_indices,
                    runtime_tracker,
                )?;
                // second expression doesn't have to consider indices which are already false
                right.eval(
                    file,
                    file_header,
                    block,
                    matching_indices,
                    num_matches_left,
                    WhichIndicesToLoopOver::Matching,
                    runtime_tracker,
                )
            }
            CompiledQueryExpression::Or(left, right) => {
                let num_matches_left = left.eval(
                    file,
                    file_header,
                    block.clone(),
                    matching_indices,
                    num_matches,
                    which_indices,
                    runtime_tracker,
                )?;
                // second expression doesn't have to consider indices which are already true
                right.eval(
                    file,
                    file_header,
                    block,
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
        QueryExpression::Within(within) => match file_format {
            "las" => match (&within.start, &within.end) {
                (Value::Classification(min), Value::Classification(max)) => Ok(
                    CompiledQueryExpression::Atom(Box::new(LasQueryAtomWithin::new(*min, *max))),
                ),
                (Value::Position(min), Value::Position(max)) => Ok(CompiledQueryExpression::Atom(
                    Box::new(LasQueryAtomWithin::new(*min, *max)),
                )),
                _ => bail!("Wrong Value types, min and max Value must have the same type!"),
            },
            _ => bail!("Unsupported file format {}", file_format),
        },
        QueryExpression::Equals(equals) => match file_format {
            "las" => match equals {
                Value::Classification(classification) => Ok(CompiledQueryExpression::Atom(
                    Box::new(LasQueryAtomEquals::new(*classification)),
                )),
                Value::Position(position) => Ok(CompiledQueryExpression::Atom(Box::new(
                    LasQueryAtomEquals::new(*position),
                ))),
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
