use std::ops::Range;

use pasture_core::nalgebra::Vector3;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::util::intersect_ranges;

use super::{BlockIndex, PointRange};

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

fn and_query_results(
    first: Vec<(PointRange, IndexResult)>,
    second: Vec<(PointRange, IndexResult)>,
) -> Vec<(PointRange, IndexResult)> {
    // Only those point ranges that are in first AND second. For overlaps, we have to combine the two point ranges
    let mut both = vec![];
    let mut first_iter = first.iter();
    let mut second_iter = second.iter();
    let mut current_pair = (first_iter.next(), second_iter.next());

    // TODO What a DISGUSTING algorithm. Surely there is a more elegant way?

    loop {
        match current_pair {
            (Some((first_range, first_result)), Some((second_range, second_result))) => {
                match first_range.file_index.cmp(&second_range.file_index) {
                    std::cmp::Ordering::Less => current_pair.0 = first_iter.next(),
                    std::cmp::Ordering::Greater => current_pair.1 = second_iter.next(),
                    std::cmp::Ordering::Equal => {
                        // Find the overlapping region and then increment the iterator of the lower region (or potentially both, if the new region is outside the other region)
                        let intersection = intersect_ranges(
                            &first_range.points_in_file,
                            &second_range.points_in_file,
                        );
                        if !intersection.is_empty() {
                            let combined_index_result = first_result.and(*second_result);
                            both.push((
                                PointRange::new(first_range.file_index, intersection),
                                combined_index_result,
                            ));

                            if first_range.points_in_file.end <= second_range.points_in_file.end {
                                current_pair.0 = first_iter.next();
                            }
                            if second_range.points_in_file.end <= first_range.points_in_file.end {
                                current_pair.1 = second_iter.next();
                            }
                        } else {
                            current_pair.0 = first_iter.next();
                            current_pair.1 = second_iter.next();
                        }
                    }
                }
            }
            _ => break,
        }
    }

    both
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

fn or_query_results(
    first: Vec<(PointRange, IndexResult)>,
    second: Vec<(PointRange, IndexResult)>,
) -> Vec<(PointRange, IndexResult)> {
    // We return all ranges in first AND second, but to be correct, we have to combine two ranges that overlap (within the same file) into a single range

    // build a lookup table to make merging easier...
    let mut lookup: FxHashMap<
        usize,
        (
            Vec<(Range<usize>, IndexResult)>,
            Vec<(Range<usize>, IndexResult)>,
        ),
    > = Default::default();

    for (point_range, index_result) in first {
        if !lookup.contains_key(&point_range.file_index) {
            lookup.insert(point_range.file_index, (vec![], vec![]));
        }

        lookup
            .get_mut(&point_range.file_index)
            .unwrap()
            .0
            .push((point_range.points_in_file, index_result));
    }

    for (point_range, index_result) in second {
        if !lookup.contains_key(&point_range.file_index) {
            lookup.insert(point_range.file_index, (vec![], vec![]));
        }

        lookup
            .get_mut(&point_range.file_index)
            .unwrap()
            .1
            .push((point_range.points_in_file, index_result));
    }

    let mut combined_ranges = vec![];

    for (file_index, (first_ranges, second_ranges)) in lookup {
        let mut combined = combine_query_ranges(
            first_ranges.as_slice(),
            second_ranges.as_slice(),
            file_index,
        );
        combined_ranges.append(&mut combined);
    }

    combined_ranges
}

#[derive(Clone, Debug)]
pub enum QueryExpression {
    Within(Range<Value>),
    Equals(Value),
    And(Box<QueryExpression>, Box<QueryExpression>),
    Or(Box<QueryExpression>, Box<QueryExpression>),
}

impl QueryExpression {
    pub fn eval(
        &self,
        indices: &FxHashMap<ValueType, BlockIndex>,
    ) -> Vec<(PointRange, IndexResult)> {
        match self {
            QueryExpression::Within(range) => {
                if let Some(index) = indices.get(&range.start.value_type()) {
                    index
                        .blocks()
                        .iter()
                        .filter_map(|block| {
                            if let Some(index_of_block) = block.index() {
                                let index_result = index_of_block.within(range, block.len());
                                match index_result {
                                    IndexResult::NoMatch => None,
                                    _ => Some((block.point_range(), index_result)),
                                }
                            } else {
                                Some((block.point_range(), IndexResult::MatchSome))
                            }
                        })
                        .collect()
                } else {
                    panic!("No index found for value type");
                }
            }
            QueryExpression::Equals(value) => {
                if let Some(index) = indices.get(&value.value_type()) {
                    index
                        .blocks()
                        .iter()
                        .filter_map(|block| {
                            if let Some(index_of_block) = block.index() {
                                let index_result = index_of_block.equals(value, block.len());
                                match index_result {
                                    IndexResult::NoMatch => None,
                                    _ => Some((block.point_range(), index_result)),
                                }
                            } else {
                                Some((block.point_range(), IndexResult::MatchSome))
                            }
                        })
                        .collect()
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
        assert_eq!(expected_query1_results, query1_results);

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
        assert_eq!(expected_query2_results, query2_results);

        let query3 = QueryExpression::Equals(Value::Classification(Classification(0)));
        let query3_results = query3.eval(&indices);
        let expected_query3_results = vec![
            (PointRange::new(0, 0..1500), IndexResult::MatchSome),
            (PointRange::new(1, 0..500), IndexResult::MatchSome),
        ];
        assert_eq!(expected_query3_results, query3_results);

        let query4 = QueryExpression::Equals(Value::Classification(Classification(1)));
        let query4_results = query4.eval(&indices);
        let expected_query4_results = vec![(PointRange::new(0, 0..1500), IndexResult::MatchSome)];
        assert_eq!(expected_query4_results, query4_results);

        let query5 = QueryExpression::Within(
            Value::Classification(Classification(1))..Value::Classification(Classification(3)),
        );
        let query5_results = query5.eval(&indices);
        let expected_query5_results = vec![
            (PointRange::new(0, 0..1500), IndexResult::MatchSome),
            (PointRange::new(1, 0..500), IndexResult::MatchSome),
        ];
        assert_eq!(expected_query5_results, query5_results);
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
        assert_eq!(expected_query1_results, query1_results);

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
        assert_eq!(expected_query2_results, query2_results);

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
        assert_eq!(expected_query3_results, query3_results);
    }
}
