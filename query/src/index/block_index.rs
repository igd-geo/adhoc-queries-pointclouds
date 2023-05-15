use std::{fmt::Display, ops::Range};

use pasture_core::{containers::PointBuffer, math::AABB, nalgebra::Point3};
use rustc_hash::FxHashMap;

use super::{Index, IndexResult, Value, ValueType};

pub struct PositionIndex {
    bounds: AABB<f64>,
}

impl PositionIndex {
    pub fn new(bounds: AABB<f64>) -> Self {
        Self { bounds }
    }
}

impl Index for PositionIndex {
    fn within(&self, range: &Range<Value>, _num_points_in_block: usize) -> IndexResult {
        match (&range.start, &range.end) {
            (Value::Position(min_pos), Value::Position(max_pos)) => {
                let other_bounds =
                    AABB::from_min_max_unchecked(Point3::from(min_pos.0), Point3::from(max_pos.0));
                let intersects = self.bounds.intersects(&other_bounds);
                if !intersects {
                    IndexResult::NoMatch
                } else {
                    let index_is_fully_contained = min_pos.0.x <= self.bounds.min().x
                        && min_pos.0.y <= self.bounds.min().y
                        && min_pos.0.z <= self.bounds.min().z
                        && max_pos.0.x >= self.bounds.max().x
                        && max_pos.0.y >= self.bounds.max().y
                        && max_pos.0.z >= self.bounds.max().z;
                    if index_is_fully_contained {
                        IndexResult::MatchAll
                    } else {
                        IndexResult::MatchSome
                    }
                }
            }
            _ => panic!("invalid value type"),
        }
    }

    fn equals(&self, _data: &Value, _num_points_in_block: usize) -> IndexResult {
        unimplemented!()
    }

    fn value_type(&self) -> ValueType {
        ValueType::Position3D
    }
}

pub struct ClassificationIndex {
    // This is a full histogram, not a sample, i.e. it contains the counts of ALL points within a block!
    histogram: FxHashMap<u8, usize>,
}

impl ClassificationIndex {
    pub fn new(histogram: FxHashMap<u8, usize>) -> Self {
        Self { histogram }
    }
}

impl Index for ClassificationIndex {
    fn within(&self, range: &Range<Value>, num_points_in_block: usize) -> IndexResult {
        match (&range.start, &range.end) {
            (Value::Classification(low), Value::Classification(high)) => {
                let matches_within_range: usize = self
                    .histogram
                    .iter()
                    .filter_map(|(key, val)| {
                        if *key < low.0 || *key >= high.0 {
                            None
                        } else {
                            Some(*val)
                        }
                    })
                    .sum();
                if matches_within_range == 0 {
                    IndexResult::NoMatch
                } else if matches_within_range == num_points_in_block {
                    IndexResult::MatchAll
                } else {
                    IndexResult::MatchSome
                }
            }
            _ => panic!("Invalid value type"),
        }
    }

    fn equals(&self, data: &Value, num_points_in_block: usize) -> IndexResult {
        match data {
            Value::Classification(classification) => {
                let num_matches = *self.histogram.get(&classification.0).unwrap_or(&0);
                if num_matches == 0 {
                    IndexResult::NoMatch
                } else if num_matches == num_points_in_block {
                    IndexResult::MatchAll
                } else {
                    IndexResult::MatchSome
                }
            }
            _ => panic!("Invalid value type"),
        }
    }

    fn value_type(&self) -> ValueType {
        ValueType::Classification
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PointRange {
    pub file_index: usize,
    pub points_in_file: Range<usize>,
}

impl PointRange {
    pub fn new(file_index: usize, points_in_file: Range<usize>) -> Self {
        Self {
            file_index,
            points_in_file,
        }
    }
}

impl Display for PointRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "file {} [{}-{})",
            self.file_index, self.points_in_file.start, self.points_in_file.end
        )
    }
}

pub struct Block {
    point_range: PointRange,
    index: Option<Box<dyn Index>>,
}

impl Block {
    /// Minimum size of a block in a block index. This constant seems reasonable because at some point the
    /// overhead of checking an index outweighs its benefit. e.g. the extreme case where there is one block
    /// per point, that would waste a huge amount of memory and performance
    pub const MIN_BLOCK_SIZE: usize = 1 << 10;

    // TODO Implement block refinement next. Question: What if we have a refined block ONLY for position, what happens to the other indices?
    // i.e. how does the datastructure look like for block refinement? Is it a tree / something like a BVH?

    pub fn new(points_in_file: Range<usize>, file_index: usize) -> Self {
        Self {
            point_range: PointRange {
                file_index,
                points_in_file,
            },
            index: None,
        }
    }

    pub fn point_range(&self) -> PointRange {
        self.point_range.clone()
    }

    pub fn points_in_file(&self) -> Range<usize> {
        self.point_range.points_in_file.clone()
    }

    pub fn len(&self) -> usize {
        self.point_range.points_in_file.len()
    }

    pub fn set_index(&mut self, index: Box<dyn Index>) {
        self.index = Some(index);
    }

    pub fn index(&self) -> Option<&dyn Index> {
        self.index.as_deref()
    }

    /// The file ID within the associated dataset (the file path is stored within the `ProgressiveIndex`!)
    /// Using an ID instead of a &Path makes it a bit easier because the file paths are stored together with the BlockIndex and then we would have
    /// a self-referential structure...
    pub fn file_id(&self) -> usize {
        self.point_range.file_index
    }

    /// Refines this block using the given data at the given indices (all indices that are `true` in `matching_indices`)
    /// Returns one block if no refinement can be done, or multiple blocks after refinement
    pub fn refine(&self, _data: &dyn PointBuffer, _matching_indices: &[bool]) -> Vec<Block> {
        // How to refine?
        // We could look for the longest contiguous sequence of `true` values inside `matching_indices`, and if this sequence exceeds
        // some length, this might warrant a new block. We would need something like a minimum block size (otherwise the index becomes too big)
        // and also an improvement criterion. If the index for the new smaller block is not significantly better than the current index, we
        // should not refine! As an example: Suppose this block has a position index with bounds (0,0,0) (1,1,1) and references 1M points. If
        // we find a sequence of e.g. 50k points, but their bounds would be (0.05, 0.05, 0.05) (1,1,1), then it doesn't make sense to refine, because
        // the new index is only marginally better than the old one
        //
        // Refinement also has to happen for each index, which is a bit weird, because index A might be refineable, but index B not. What to do then?
        // Would it make more sense to have one (block-)index per attribute and combine their results accordingly in the queries?
        todo!()
    }
}

pub struct BlockIndex {
    blocks: Vec<Block>,
}

impl BlockIndex {
    pub fn new(blocks: Vec<Block>) -> Self {
        Self { blocks }
    }

    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    pub fn add_block(&mut self, block: Block) {
        self.blocks.push(block);
    }

    pub fn blocks_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn largest_block(&self) -> Option<&Block> {
        self.blocks.iter().max_by_key(|block| block.len())
    }

    /// Apply the given index refinements to this BlockIndex
    pub fn apply_refinements<I: Iterator<Item = IndexRefinement>>(&mut self, refinements: I) {
        for refinement in refinements {
            let old_point_range = refinement.point_range_before_refinement;
            let new_blocks = refinement
                .refined_indices
                .into_iter()
                .map(|(point_range, index)| Block {
                    index: Some(index),
                    point_range,
                });

            let pos_of_old_block = self
                .blocks
                .iter()
                .position(|block| block.point_range == old_point_range)
                .expect("Original block for refinement not found!");
            self.blocks
                .splice(pos_of_old_block..=pos_of_old_block, new_blocks);
        }
    }
}

/// Result of an index refinement. Contains the new indices for a specific range of points
pub struct IndexRefinement {
    pub point_range_before_refinement: PointRange,
    pub value_type: ValueType,
    pub refined_indices: Vec<(PointRange, Box<dyn Index>)>,
}
