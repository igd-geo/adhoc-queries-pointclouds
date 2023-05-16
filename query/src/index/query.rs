use std::{collections::HashSet, ops::Range};

use itertools::Itertools;
use pasture_core::nalgebra::Vector3;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::util::intersect_ranges;

use super::{Block, BlockIndex, PointRange};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum IndexResult {
    MatchAll,
    MatchSome,
    NoMatch,
}

impl IndexResult {
    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (IndexResult::MatchAll, IndexResult::MatchAll) => IndexResult::MatchAll,
            (IndexResult::MatchAll, IndexResult::MatchSome) => IndexResult::MatchSome,
            (IndexResult::MatchSome, IndexResult::MatchAll) => IndexResult::MatchSome,
            (IndexResult::MatchSome, IndexResult::MatchSome) => IndexResult::MatchSome,
            (IndexResult::NoMatch, _) => IndexResult::NoMatch,
            (_, IndexResult::NoMatch) => IndexResult::NoMatch,
        }
    }

    pub fn or(self, other: Self) -> Self {
        match (self, other) {
            (IndexResult::MatchAll, _) => IndexResult::MatchAll,
            (_, IndexResult::MatchAll) => IndexResult::MatchAll,
            (IndexResult::MatchSome, _) => IndexResult::MatchSome,
            (_, IndexResult::MatchSome) => IndexResult::MatchSome,
            _ => IndexResult::NoMatch,
        }
    }
}

pub trait Index: Send + Sync {
    fn within(&self, range: &Range<Value>, num_points_in_block: usize) -> IndexResult;
    fn equals(&self, data: &Value, num_points_in_block: usize) -> IndexResult;
    fn value_type(&self) -> ValueType;
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub enum ValueType {
    Classification,
    Position3D,
}

#[derive(Copy, Clone, Debug)]
pub struct Position(pub Vector3<f64>);
#[derive(Copy, Clone, Debug)]
pub struct Classification(pub u8);

#[derive(Copy, Clone, Debug)]
pub enum Value {
    Classification(Classification),
    Position(Position),
}

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::Classification(_) => ValueType::Classification,
            Value::Position(_) => ValueType::Position3D,
        }
    }
}

/// Result of a block query, i.e. a query that determines the blocks within file(s) that might containt
/// points that match the query
#[derive(Debug, Default)]
pub struct BlockQueryResult {
    /// The matching blocks of the block query. These are all blocks that might contain points that match
    /// the query. The `IndexResult` determines how good the match is, i.e. whether the whole block matches
    /// the query or only some points of it. Blocks that don't match any points are not returned in a `BlockQueryResult`
    pub matching_blocks: Vec<(PointRange, IndexResult)>,
    /// List of potential blocks that could be refined for each `BlockIndex`. These are blocks where the query result
    /// indicates `MatchSome`, i.e. only some points match. These are good candidates for further refinement
    pub potential_blocks_for_refinement: FxHashMap<ValueType, FxHashSet<PointRange>>,
}

impl BlockQueryResult {
    /// Gets all matching blocks grouped by their file ID
    pub fn matching_blocks_by_file(&self) -> FxHashMap<usize, Vec<(PointRange, IndexResult)>> {
        let mut ret: FxHashMap<usize, Vec<(PointRange, IndexResult)>> = Default::default();
        for (range, index_result) in self.matching_blocks.iter() {
            if let Some(ranges) = ret.get_mut(&range.file_index) {
                ranges.push((range.clone(), *index_result));
            } else {
                ret.insert(range.file_index, vec![(range.clone(), *index_result)]);
            }
        }
        ret
    }
}

/// Combine two ranges of matching blocks into a single range of matching blocks using logical `AND`
fn and_blocks_within_file<
    I: Iterator<Item = (PointRange, IndexResult)>,
    J: Iterator<Item = (PointRange, IndexResult)>,
>(
    mut blocks_a: I,
    mut blocks_b: J,
) -> Vec<(PointRange, IndexResult)> {
    let mut first_block = blocks_a.next();
    let mut second_block = blocks_b.next();
    let mut combined = vec![];

    while first_block.is_some() && second_block.is_some() {
        let first_block_unwrapped = first_block.as_ref().unwrap();
        let second_block_unwrapped = second_block.as_ref().unwrap();
        let intersection = intersect_ranges(
            &first_block_unwrapped.0.points_in_file,
            &second_block_unwrapped.0.points_in_file,
        );
        if !intersection.is_empty() {
            let combined_index_result = first_block_unwrapped.1.and(second_block_unwrapped.1);
            combined.push((
                PointRange::new(first_block_unwrapped.0.file_index, intersection),
                combined_index_result,
            ));
        }

        // This feels wrong. If we e.g. have this pattern:
        // |------|------|
        //     |------|
        //
        // Then the resulting blocks should be
        //     |--|---|
        // I hate this algorithm... Verify it...
        if first_block_unwrapped.0.points_in_file.end <= second_block_unwrapped.0.points_in_file.end
        {
            first_block = blocks_a.next();
        } else if second_block_unwrapped.0.points_in_file.end
            <= first_block_unwrapped.0.points_in_file.end
        {
            second_block = blocks_b.next();
        }
    }

    combined
}

fn combine_potential_blocks_for_refinement(
    left: FxHashMap<ValueType, FxHashSet<PointRange>>,
    mut right: FxHashMap<ValueType, FxHashSet<PointRange>>,
) -> FxHashMap<ValueType, FxHashSet<PointRange>> {
    let mut ret = FxHashMap::default();
    for (key, mut value) in left {
        if let Some(other_value) = right.remove(&key) {
            // If the ValueTypes are equal, we know the PointRanges come from the same BlockIndex, so we can simply
            // merge the two HashSets
            value.extend(other_value.into_iter());
        }
        ret.insert(key, value);
    }
    ret
}

/// Combine the results of two block queries using the `AND` operator
fn and_query_results(first: BlockQueryResult, second: BlockQueryResult) -> BlockQueryResult {
    // Edge cases
    if first.matching_blocks.is_empty() || second.matching_blocks.is_empty() {
        return Default::default();
    }

    let first_blocks_by_file = first.matching_blocks_by_file();
    let mut second_blocks_by_file = second.matching_blocks_by_file();

    let mut combined_blocks = vec![];

    for (file_index, blocks) in first_blocks_by_file {
        if !second_blocks_by_file.contains_key(&file_index) {
            continue;
        }

        let other_blocks = second_blocks_by_file.remove(&file_index).unwrap();
        combined_blocks.append(&mut and_blocks_within_file(
            blocks.into_iter(),
            other_blocks.into_iter(),
        ));
    }

    combined_blocks.sort_by(|a, b| a.0.cmp(&b.0));

    // Refinement blocks are always combined with logical OR, because we care about all blocks that were inspected
    // during the query in any way, not just about those ranges that came out of the combined query. If a range for
    // index A matches, but doesn't for index B, it still might be a range worth refining *for index A*, so we keep it
    let merged_blocks_for_refinement = combine_potential_blocks_for_refinement(
        first.potential_blocks_for_refinement,
        second.potential_blocks_for_refinement,
    );

    BlockQueryResult {
        matching_blocks: combined_blocks,
        potential_blocks_for_refinement: merged_blocks_for_refinement,
    }
}

fn combine_ranges(
    a: (Range<usize>, IndexResult),
    b: (Range<usize>, IndexResult),
    file_index: usize,
) -> Vec<(PointRange, IndexResult)> {
    use std::cmp::{max, min};

    // Check if the ranges intersect
    if a.0.end >= b.0.start && a.0.start <= b.0.end {
        let lower_start = min(a.0.start, b.0.start);
        let lower_end = max(a.0.start, b.0.start);
        let upper_start = min(a.0.end, b.0.end);
        let upper_end = max(a.0.end, b.0.end);

        let mut result = Vec::new();

        if lower_start != lower_end {
            let index_result = if a.0.start < b.0.start { a.1 } else { b.1 };

            result.push((
                PointRange::new(file_index, lower_start..lower_end),
                index_result,
            ));
        }

        result.push((
            PointRange::new(file_index, lower_end..upper_start),
            a.1.or(b.1),
        ));

        if upper_start != upper_end {
            let index_result = if a.0.end < b.0.end { a.1 } else { b.1 };
            result.push((
                PointRange::new(file_index, upper_start..upper_end),
                index_result,
            ));
        }

        result
    } else {
        vec![
            (PointRange::new(file_index, a.0), a.1),
            (PointRange::new(file_index, b.0), b.1),
        ]
    }
}

fn combine_query_ranges(
    a: &[(Range<usize>, IndexResult)],
    b: &[(Range<usize>, IndexResult)],
    file_index: usize,
) -> Vec<(PointRange, IndexResult)> {
    let mut i = 0;
    let mut j = 0;
    let mut result = Vec::new();

    while i < a.len() && j < b.len() {
        let combined = combine_ranges(a[i].clone(), b[j].clone(), file_index);
        match combined.len() {
            1 => {
                result.push(combined[0].clone());
                i += 1;
                j += 1;
            }
            2 => {
                if a[i].0.start < b[j].0.start {
                    result.push(combined[0].clone());
                    i += 1;
                } else {
                    result.push(combined[1].clone());
                    j += 1;
                }
            }
            3 => {
                result.push(combined[0].clone());
                result.push(combined[1].clone());
                i += 1;
                j += 1;
            }
            _ => {}
        }
    }

    while i < a.len() {
        result.push((PointRange::new(file_index, a[i].0.clone()), a[i].1));
        i += 1;
    }

    while j < b.len() {
        result.push((PointRange::new(file_index, b[i].0.clone()), b[i].1));
        j += 1;
    }

    result
}

/// Combine the results of two block queries using the `OR` operator
fn or_query_results(first: BlockQueryResult, second: BlockQueryResult) -> BlockQueryResult {
    todo!()
    // We return all ranges in first AND second, but to be correct, we have to combine two ranges that overlap (within the same file) into a single range

    // build a lookup table to make merging easier...
    // let mut lookup: FxHashMap<
    //     usize,
    //     (
    //         Vec<(Range<usize>, IndexResult)>,
    //         Vec<(Range<usize>, IndexResult)>,
    //     ),
    // > = Default::default();

    // for (point_range, index_result) in first {
    //     if !lookup.contains_key(&point_range.file_index) {
    //         lookup.insert(point_range.file_index, (vec![], vec![]));
    //     }

    //     lookup
    //         .get_mut(&point_range.file_index)
    //         .unwrap()
    //         .0
    //         .push((point_range.points_in_file, index_result));
    // }

    // for (point_range, index_result) in second {
    //     if !lookup.contains_key(&point_range.file_index) {
    //         lookup.insert(point_range.file_index, (vec![], vec![]));
    //     }

    //     lookup
    //         .get_mut(&point_range.file_index)
    //         .unwrap()
    //         .1
    //         .push((point_range.points_in_file, index_result));
    // }

    // let mut combined_ranges = vec![];

    // for (file_index, (first_ranges, second_ranges)) in lookup {
    //     let mut combined = combine_query_ranges(
    //         first_ranges.as_slice(),
    //         second_ranges.as_slice(),
    //         file_index,
    //     );
    //     combined_ranges.append(&mut combined);
    // }

    // combined_ranges
}

/// Query the given index using the given query_fn. Helper function for redundant code when evaluating the
/// atomic query expressions (e.g. `Within` or `Equals`)
fn query_index(
    index: &BlockIndex,
    index_value_type: ValueType,
    query_fn: impl Fn(&dyn Index, &Block) -> IndexResult,
) -> BlockQueryResult {
    let matching_blocks = index
        .blocks()
        .iter()
        .filter_map(|block| {
            if let Some(index_of_block) = block.index() {
                let index_result = query_fn(index_of_block, block);
                match index_result {
                    IndexResult::NoMatch => None,
                    _ => Some((block.point_range(), index_result)),
                }
            } else {
                Some((block.point_range(), IndexResult::MatchSome))
            }
        })
        .collect::<Vec<_>>();
    let refinement_candidates = matching_blocks
        .iter()
        .filter_map(|(block, index_result)| match index_result {
            IndexResult::MatchSome => Some(block.clone()),
            _ => None,
        })
        .collect::<FxHashSet<_>>();

    let mut potential_blocks_for_refinement = FxHashMap::default();
    potential_blocks_for_refinement.insert(index_value_type, refinement_candidates);

    BlockQueryResult {
        matching_blocks,
        potential_blocks_for_refinement,
    }
}

#[derive(Clone, Debug)]
pub enum QueryExpression {
    Within(Range<Value>),
    Equals(Value),
    And(Box<QueryExpression>, Box<QueryExpression>),
    Or(Box<QueryExpression>, Box<QueryExpression>),
}

impl QueryExpression {
    pub fn eval(&self, indices: &FxHashMap<ValueType, BlockIndex>) -> BlockQueryResult {
        match self {
            QueryExpression::Within(range) => {
                if let Some(index) = indices.get(&range.start.value_type()) {
                    query_index(index, range.start.value_type(), |index, block| {
                        index.within(range, block.len())
                    })
                } else {
                    panic!("No index found for value type");
                }
            }
            QueryExpression::Equals(value) => {
                if let Some(index) = indices.get(&value.value_type()) {
                    query_index(index, value.value_type(), |index, block| {
                        index.equals(value, block.len())
                    })
                } else {
                    panic!("No index found for value type");
                }
            }
            QueryExpression::And(l_expr, r_expr) => {
                let left_blocks = l_expr.eval(indices);
                let right_blocks = r_expr.eval(indices);
                and_query_results(left_blocks, right_blocks)
            }
            QueryExpression::Or(l_expr, r_expr) => {
                let left_blocks = l_expr.eval(indices);
                let right_blocks = r_expr.eval(indices);
                or_query_results(left_blocks, right_blocks)
            }
        }
    }

    pub fn required_indices(&self) -> FxHashSet<ValueType> {
        match self {
            QueryExpression::Within(range) => {
                let mut indices: FxHashSet<_> = Default::default();
                indices.insert(range.start.value_type());
                indices
            }
            QueryExpression::Equals(value) => {
                let mut indices: FxHashSet<_> = Default::default();
                indices.insert(value.value_type());
                indices
            }
            QueryExpression::And(l, r) => {
                let l_indices = l.required_indices();
                let r_indices = r.required_indices();
                l_indices.union(&r_indices).copied().collect()
            }
            QueryExpression::Or(l, r) => {
                let l_indices = l.required_indices();
                let r_indices = r.required_indices();
                l_indices.union(&r_indices).copied().collect()
            }
        }
    }
}

pub type Query = QueryExpression;

#[cfg(test)]
mod tests {
    use pasture_core::{math::AABB, nalgebra::Point3};

    use crate::index::{Block, ClassificationIndex, PositionIndex};

    use super::*;

    fn create_default_indices() -> FxHashMap<ValueType, BlockIndex> {
        let mut position_blocks: Vec<Block> = vec![
            Block::new(0..1000, 0),
            Block::new(1000..1500, 0),
            Block::new(0..500, 1),
        ];
        position_blocks[0].set_index(Box::new(PositionIndex::new(AABB::from_min_max(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 3.0),
        ))));
        position_blocks[1].set_index(Box::new(PositionIndex::new(AABB::from_min_max(
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(2.0, 3.0, 4.0),
        ))));
        position_blocks[2].set_index(Box::new(PositionIndex::new(AABB::from_min_max(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
        ))));
        let position_block_index = BlockIndex::new(position_blocks);

        let mut classification_blocks: Vec<Block> =
            vec![Block::new(0..1500, 0), Block::new(0..500, 1)];
        classification_blocks[0].set_index(Box::new(ClassificationIndex::new(
            vec![(0, 1000), (1, 500)].into_iter().collect(),
        )));
        classification_blocks[1].set_index(Box::new(ClassificationIndex::new(
            vec![(0, 200), (2, 300)].into_iter().collect(),
        )));
        let classification_block_index = BlockIndex::new(classification_blocks);

        vec![
            (ValueType::Position3D, position_block_index),
            (ValueType::Classification, classification_block_index),
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn test_simple_queries() {
        let indices = create_default_indices();

        let query1 = QueryExpression::Within(
            Value::Position(Position(Vector3::new(0.5, 0.5, 0.5)))
                ..Value::Position(Position(Vector3::new(1.5, 1.5, 1.5))),
        );

        let query1_results = query1.eval(&indices);
        let expected_query1_results = vec![
            (PointRange::new(0, 0..1000), IndexResult::MatchSome),
            (PointRange::new(1, 0..500), IndexResult::MatchSome),
        ];
        assert_eq!(expected_query1_results, query1_results.matching_blocks);

        let query2 = QueryExpression::Within(
            Value::Position(Position(Vector3::new(0.0, 0.0, 0.0)))
                ..Value::Position(Position(Vector3::new(4.0, 4.0, 4.0))),
        );
        let query2_results = query2.eval(&indices);
        let expected_query2_results = vec![
            (PointRange::new(0, 0..1000), IndexResult::MatchAll),
            (PointRange::new(0, 1000..1500), IndexResult::MatchAll),
            (PointRange::new(1, 0..500), IndexResult::MatchAll),
        ];
        assert_eq!(expected_query2_results, query2_results.matching_blocks);

        let query3 = QueryExpression::Equals(Value::Classification(Classification(0)));
        let query3_results = query3.eval(&indices);
        let expected_query3_results = vec![
            (PointRange::new(0, 0..1500), IndexResult::MatchSome),
            (PointRange::new(1, 0..500), IndexResult::MatchSome),
        ];
        assert_eq!(expected_query3_results, query3_results.matching_blocks);

        let query4 = QueryExpression::Equals(Value::Classification(Classification(1)));
        let query4_results = query4.eval(&indices);
        let expected_query4_results = vec![(PointRange::new(0, 0..1500), IndexResult::MatchSome)];
        assert_eq!(expected_query4_results, query4_results.matching_blocks);

        let query5 = QueryExpression::Within(
            Value::Classification(Classification(1))..Value::Classification(Classification(3)),
        );
        let query5_results = query5.eval(&indices);
        let expected_query5_results = vec![
            (PointRange::new(0, 0..1500), IndexResult::MatchSome),
            (PointRange::new(1, 0..500), IndexResult::MatchSome),
        ];
        assert_eq!(expected_query5_results, query5_results.matching_blocks);
    }

    #[test]
    fn test_combined_queries() {
        let indices = create_default_indices();

        let query1 = QueryExpression::And(
            Box::new(QueryExpression::Within(
                Value::Position(Position(Vector3::new(0.0, 0.0, 0.0)))
                    ..Value::Position(Position(Vector3::new(4.0, 4.0, 4.0))),
            )),
            Box::new(QueryExpression::Equals(Value::Classification(
                Classification(1),
            ))),
        );

        let query1_results = query1.eval(&indices);
        let expected_query1_results = vec![
            (PointRange::new(0, 0..1000), IndexResult::MatchSome),
            (PointRange::new(0, 1000..1500), IndexResult::MatchSome),
        ];
        assert_eq!(expected_query1_results, query1_results.matching_blocks);

        let query2 = QueryExpression::And(
            Box::new(QueryExpression::Within(
                Value::Position(Position(Vector3::new(0.0, 0.0, 0.0)))
                    ..Value::Position(Position(Vector3::new(4.0, 4.0, 4.0))),
            )),
            Box::new(QueryExpression::Equals(Value::Classification(
                Classification(2),
            ))),
        );

        let query2_results = query2.eval(&indices);
        let expected_query2_results = vec![(PointRange::new(1, 0..500), IndexResult::MatchSome)];
        assert_eq!(expected_query2_results, query2_results.matching_blocks);

        let query3 = QueryExpression::Or(
            Box::new(QueryExpression::Within(
                Value::Position(Position(Vector3::new(-2.0, -2.0, -2.0)))
                    ..Value::Position(Position(Vector3::new(-1.0, -1.0, -1.0))),
            )),
            Box::new(QueryExpression::Equals(Value::Classification(
                Classification(0),
            ))),
        );

        let query3_results = query3.eval(&indices);
        let expected_query3_results = vec![
            // Returns the block of the SECOND index! It can't know that the first index has two blocks. Have to see
            // what this means for evaluating the queries later on, this might potentially fail?
            // I think it also might be a stupid idea to have separate blocks per index type, this makes the code super
            // complicated... But it IS realistic, an index over positions will have very different structure than an
            // index over classifications...
            // After looking at it some more, I think I'm allowed to combine ranges/blocks (as long as they are within the same file) because query evaluation looks into the files anyway. This could make the code a bit easier
            (PointRange::new(0, 0..1500), IndexResult::MatchSome),
            (PointRange::new(1, 0..500), IndexResult::MatchSome),
        ];
        assert_eq!(expected_query3_results, query3_results.matching_blocks);
    }

    /// Helper function to create matching blocks (PointRange + IndexResult) from a string in the form
    /// '|----|' or '   |**|  '. The vertical lines indicate start and end positions for a PointRange,
    /// the symbols between the lines indicate whether it is a `MatchAll` range ('*') or a `MatchSome`
    /// range ('-'). Multiple ranges per string are fine. Edge-case '||' always has `MatchAll`
    ///
    /// This might look a bit silly, but helps with testing the matching blocks combination algorithms.
    fn create_matching_blocks_from_str<S: AsRef<str>>(string: S) -> Vec<(PointRange, IndexResult)> {
        let mut ret = vec![];
        let mut string = string.as_ref();
        let mut next_start_index = string.find('|');

        while let Some(start_index) = next_start_index {
            let end_index = start_index
                + 1
                + string[(start_index + 1)..]
                    .find('|')
                    .expect("Found starting | without end |");
            if start_index + 1 == end_index {
                ret.push((
                    PointRange::new(0, start_index..end_index),
                    IndexResult::MatchAll,
                ));
            } else {
                let index_result = match string.chars().nth(start_index + 1).unwrap() {
                    '-' => IndexResult::MatchSome,
                    '*' => IndexResult::MatchAll,
                    _ => panic!("Invalid character"),
                };
                ret.push((PointRange::new(0, start_index..end_index), index_result));
            }

            // Move to the next starting '|'. If there is any of '*', '-', or '|' immediately after the
            // current end position, the current end is the new start, otherwise we use str::find
            if let Some(next_char_after_current_end) = string.chars().nth(end_index + 1) {
                match next_char_after_current_end {
                    '*' | '-' | '|' => next_start_index = Some(end_index),
                    _ => {
                        next_start_index = string[(end_index + 1)..]
                            .find('|')
                            .map(|idx| idx + end_index + 1)
                    }
                }
            } else {
                next_start_index = None;
            }
        }

        ret
    }

    #[test]
    fn test_create_matching_blocks_from_str() {
        assert_eq!(
            vec![(PointRange::new(0, 0..1), IndexResult::MatchAll)],
            create_matching_blocks_from_str("||")
        );
        assert_eq!(
            vec![(PointRange::new(0, 0..2), IndexResult::MatchAll)],
            create_matching_blocks_from_str("|*|")
        );
        assert_eq!(
            vec![(PointRange::new(0, 0..2), IndexResult::MatchSome)],
            create_matching_blocks_from_str("|-|")
        );

        assert_eq!(
            vec![(PointRange::new(0, 1..3), IndexResult::MatchAll)],
            create_matching_blocks_from_str(" |*|")
        );
        assert_eq!(
            vec![(PointRange::new(0, 1..3), IndexResult::MatchAll)],
            create_matching_blocks_from_str(" |*| ")
        );

        assert_eq!(
            vec![(PointRange::new(0, 1..5), IndexResult::MatchAll)],
            create_matching_blocks_from_str(" |***| ")
        );
        assert_eq!(
            vec![(PointRange::new(0, 1..5), IndexResult::MatchSome)],
            create_matching_blocks_from_str(" |---| ")
        );

        assert_eq!(
            vec![
                (PointRange::new(0, 0..2), IndexResult::MatchAll),
                (PointRange::new(0, 4..6), IndexResult::MatchSome)
            ],
            create_matching_blocks_from_str("|*| |-|")
        );

        assert_eq!(
            vec![
                (PointRange::new(0, 0..2), IndexResult::MatchAll),
                (PointRange::new(0, 2..4), IndexResult::MatchSome)
            ],
            create_matching_blocks_from_str("|*|-|")
        );

        assert_eq!(
            vec![
                (PointRange::new(0, 1..5), IndexResult::MatchAll),
                (PointRange::new(0, 5..9), IndexResult::MatchSome)
            ],
            create_matching_blocks_from_str(" |***|---| ")
        );

        assert_eq!(
            vec![
                (PointRange::new(0, 0..2), IndexResult::MatchAll),
                (PointRange::new(0, 4..6), IndexResult::MatchAll),
                (PointRange::new(0, 9..11), IndexResult::MatchAll),
            ],
            create_matching_blocks_from_str("|*| |*|  |*| ")
        );
    }

    #[test]
    fn test_and_blocks_within_file() {
        let verify = |in1: &str, in2: &str, expected_out: &str| {
            let combined_blocks = and_blocks_within_file(
                create_matching_blocks_from_str(in1).into_iter(),
                create_matching_blocks_from_str(in2).into_iter(),
            );
            assert_eq!(
                combined_blocks,
                create_matching_blocks_from_str(expected_out)
            );

            // Swap in1 and in2, result should be the same!
            let combined_blocks = and_blocks_within_file(
                create_matching_blocks_from_str(in2).into_iter(),
                create_matching_blocks_from_str(in1).into_iter(),
            );
            assert_eq!(
                combined_blocks,
                create_matching_blocks_from_str(expected_out)
            );
        };

        {
            let str1 = "|----|";
            let str2 = "|----|";
            let str3 = "|----|";
            verify(str1, str2, str3);
        }

        {
            let str1 = "|****|";
            let str2 = "|****|";
            let str3 = "|****|";
            verify(str1, str2, str3);
        }

        {
            let str1 = "|----|";
            let str2 = "|****|";
            let str3 = "|----|";
            verify(str1, str2, str3);
        }

        {
            let str1 = "|----|";
            let str2 = "  |-| ";
            let str3 = "  |-| ";
            verify(str1, str2, str3);
        }

        {
            let str1 = "|----|   ";
            let str2 = "   |----|";
            let str3 = "   |-|   ";
            verify(str1, str2, str3);
        }

        {
            let str1 = "|----|   ";
            let str2 = "   |****|";
            let str3 = "   |-|   ";
            verify(str1, str2, str3);
        }

        {
            let str1 = "|----|----|";
            let str2 = "   |---|   ";
            let str3 = "   |-|-|   ";
            verify(str1, str2, str3);
        }

        {
            let str1 = "|****|****|";
            let str2 = "   |---|   ";
            let str3 = "   |-|-|   ";
            verify(str1, str2, str3);
        }

        {
            let str1 = "|----|----|  ";
            let str2 = "   |---|----|";
            let str3 = "   |-|-|--|  ";
            verify(str1, str2, str3);
        }
    }
}
