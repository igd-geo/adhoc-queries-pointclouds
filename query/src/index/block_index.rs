use std::ops::Range;

use pasture_core::{math::AABB, nalgebra::Point3};
use rustc_hash::FxHashMap;

use super::{Index, IndexResult, Query, Value, ValueType};

pub struct PositionIndex {
    bounds: AABB<f64>,
}

impl PositionIndex {
    pub fn new(bounds: AABB<f64>) -> Self {
        Self { bounds }
    }
}

impl Index for PositionIndex {
    fn within(&self, range: &Range<Value>, num_points_in_block: usize) -> IndexResult {
        match (&range.start, &range.end) {
            (Value::Position(min_pos), Value::Position(max_pos)) => {
                let other_bounds =
                    AABB::from_min_max_unchecked(Point3::from(min_pos.0), Point3::from(max_pos.0));
                let intersects = self.bounds.intersects(&other_bounds);
                if !intersects {
                    IndexResult::NoMatch
                } else {
                    let fully_contained = min_pos.0.x >= self.bounds.min().x
                        && min_pos.0.y >= self.bounds.min().y
                        && min_pos.0.z >= self.bounds.min().z
                        && max_pos.0.x <= self.bounds.max().x
                        && max_pos.0.y <= self.bounds.max().y
                        && max_pos.0.z <= self.bounds.max().z;
                    if fully_contained {
                        IndexResult::MatchAll
                    } else {
                        IndexResult::MatchSome
                    }
                }
            }
            _ => panic!("invalid value type"),
        }
    }

    fn equals(&self, _data: &Value, num_points_in_block: usize) -> IndexResult {
        unimplemented!()
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
}

pub struct Block {
    point_range: Range<usize>,
    file_id: usize,
    indices: FxHashMap<ValueType, Box<dyn Index>>,
}

impl Block {
    pub fn new(point_range: Range<usize>, file_id: usize) -> Self {
        Self {
            point_range,
            file_id,
            indices: Default::default(),
        }
    }

    pub fn indices(&self) -> &FxHashMap<ValueType, Box<dyn Index>> {
        &self.indices
    }

    pub fn indices_mut(&mut self) -> &mut FxHashMap<ValueType, Box<dyn Index>> {
        &mut self.indices
    }

    pub fn point_range(&self) -> Range<usize> {
        self.point_range.clone()
    }

    pub fn len(&self) -> usize {
        self.point_range.len()
    }

    /// The file ID within the associated dataset (the file path is stored within the `ProgressiveIndex`!)
    /// Using an ID instead of a &Path makes it a bit easier because the file paths are stored together with the BlockIndex and then we would have
    /// a self-referential structure...
    pub fn file_id(&self) -> usize {
        self.file_id
    }
}

pub struct BlockIndex {
    blocks: Vec<Block>,
}

impl BlockIndex {
    pub fn new(blocks: Vec<Block>) -> Self {
        Self { blocks }
    }

    pub fn get_matching_blocks<'b>(
        &'b self,
        query: &'b Query,
    ) -> impl Iterator<Item = (&'b Block, IndexResult)> + 'b {
        self.blocks.iter().filter_map(move |block| {
            let query_result = query.eval(block);
            match query_result {
                IndexResult::NoMatch => None,
                _ => Some((block, query_result)),
            }
        })
    }

    pub fn get_matching_blocks_mut<'b>(
        &'b mut self,
        query: &'b Query,
    ) -> impl Iterator<Item = (&'b mut Block, IndexResult)> + 'b {
        self.blocks.iter_mut().filter_map(move |block| {
            let query_result = query.eval(block);
            match query_result {
                IndexResult::NoMatch => None,
                _ => Some((block, query_result)),
            }
        })
    }

    pub fn add_block(&mut self, block: Block) {
        self.blocks.push(block);
    }
}
