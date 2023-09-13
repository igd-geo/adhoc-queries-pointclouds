use std::{fmt::Display, ops::Range, path::PathBuf};

use anyhow::{Context, Result};
use divide_range::RangeDivisions;
use geo::{coord, Intersects, Contains};
use itertools::Itertools;
use pasture_core::{
    math::AABB,
    nalgebra::{Point3, Vector3},
};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

// use crate::io::{open_reader, PointReader};

use super::{AtomicExpression, CompareExpression, Index, IndexResult, Value, ValueType, Geometry};

// TODO pasture_core does not support serde, maybe implement it?

#[derive(Serialize, Deserialize)]
#[serde(remote = "AABB<f64>")]
struct AABBDeff64 {
    #[serde(getter = "AABB::min")]
    min: Point3<f64>,
    #[serde(getter = "AABB::max")]
    max: Point3<f64>,
}

impl From<AABBDeff64> for AABB<f64> {
    fn from(def: AABBDeff64) -> Self {
        AABB::<f64>::from_min_max_unchecked(def.min, def.max)
    }
}

/// Index over positions using an AABB
#[derive(Serialize, Deserialize)]
pub struct PositionIndex {
    #[serde(with = "AABBDeff64")]
    bounds: AABB<f64>,
}

impl PositionIndex {
    pub fn new(bounds: AABB<f64>) -> Self {
        Self { bounds }
    }

    /// Create a new `PositionIndex` from the given range of positions
    ///
    /// # Panics
    ///
    /// If `positions` is an empty range
    pub fn build_from_positions<I: Iterator<Item = Vector3<f64>>>(mut positions: I) -> Self {
        let first_position = positions
            .next()
            .expect("Can't build a PositionIndex from an empty range of positions!");

        let mut bounds = AABB::from_min_max_unchecked(first_position.into(), first_position.into());
        for position in positions {
            bounds = AABB::extend_with_point(&bounds, &position.into());
        }
        Self { bounds }
    }

    fn within(&self, min_position: &Vector3<f64>, max_position: &Vector3<f64>) -> IndexResult {
        let other_bounds =
            AABB::from_min_max_unchecked(Point3::from(*min_position), Point3::from(*max_position));
        let intersects = self.bounds.intersects(&other_bounds);
        if !intersects {
            IndexResult::NoMatch
        } else {
            let index_is_fully_contained = min_position.x <= self.bounds.min().x
                && min_position.y <= self.bounds.min().y
                && min_position.z <= self.bounds.min().z
                && max_position.x >= self.bounds.max().x
                && max_position.y >= self.bounds.max().y
                && max_position.z >= self.bounds.max().z;
            if index_is_fully_contained {
                IndexResult::MatchAll
            } else {
                IndexResult::MatchSome
            }
        }
    }

    fn intersects(&self, geometry: &Geometry) -> IndexResult {
        match geometry {
            Geometry::Polygon(polygon) => {
                let self_bounds_as_2d_rect = geo::Rect::new(coord! {
                    x: self.bounds.min().x,
                    y: self.bounds.min().y,
                }, coord! {
                    x: self.bounds.max().x,
                    y: self.bounds.max().y,
                });
                let intersects = polygon.intersects(&self_bounds_as_2d_rect);
                if !intersects {
                    IndexResult::NoMatch
                } else {
                    // Is this index fully contained within the polygon?
                    if polygon.contains(&self_bounds_as_2d_rect) {
                        IndexResult::MatchAll
                    } else {
                        IndexResult::MatchSome
                    }
                }
            }
        }
    }
}

#[typetag::serde]
impl Index for PositionIndex {
    fn matches(&self, atomic_expr: &AtomicExpression, _num_points_in_block: usize) -> IndexResult {
        match atomic_expr {
            AtomicExpression::Within(range) => {
                match (range.start, range.end) {
                    (Value::Position(min_pos), Value::Position(max_pos)) => self.within(&min_pos.0, &max_pos.0),
                    (other_start, other_end) => panic!("Encountered invalid values for range of 'Within' expression. Expected (Position, Position) but got ({},{}) instead", other_start.value_type(), other_end.value_type()),
                }
            },
            AtomicExpression::Intersects(geometry) => 
                self.intersects(geometry),
            other => panic!("Unsupported query expression {other:#?} for PositionIndex"),
        }
    }

    fn value_type(&self) -> ValueType {
        ValueType::Position3D
    }
}

/// Histogram-based index over classifications
#[derive(Serialize, Deserialize)]
pub struct ClassificationIndex {
    // This is a full histogram, not a sample, i.e. it contains the counts of ALL points within a block!
    histogram: FxHashMap<u8, usize>,
}

impl ClassificationIndex {
    /// Creates a new `ClassificationIndex` from the given histogram
    pub fn new(histogram: FxHashMap<u8, usize>) -> Self {
        Self { histogram }
    }

    /// Builds a `ClassificationIndex` from the given range of classification values
    pub fn build_from_classifications<I: Iterator<Item = u8>>(classifications: I) -> Self {
        let mut histogram: FxHashMap<u8, usize> = Default::default();
        for class in classifications {
            if let Some(count) = histogram.get_mut(&class) {
                *count += 1;
            } else {
                histogram.insert(class, 1);
            }
        }
        Self { histogram }
    }

    fn get_matches_from_histogram(
        &self,
        comparator: CompareExpression,
        classification: u8,
        num_points_in_block: usize,
    ) -> usize {
        match comparator {
            CompareExpression::Equals => *self.histogram.get(&classification).unwrap_or(&0),
            CompareExpression::NotEquals => {
                num_points_in_block - *self.histogram.get(&classification).unwrap_or(&0)
            }
            CompareExpression::LessThan => (0..classification)
                .map(|c| *self.histogram.get(&c).unwrap_or(&0))
                .sum(),
            CompareExpression::LessThanOrEquals => (0..=classification)
                .map(|c| *self.histogram.get(&c).unwrap_or(&0))
                .sum(),
            CompareExpression::GreaterThan => ((classification + 1)..std::u8::MAX)
                .map(|c| *self.histogram.get(&c).unwrap_or(&0))
                .sum(),
            CompareExpression::GreaterThanOrEquals => (classification..std::u8::MAX)
                .map(|c| *self.histogram.get(&c).unwrap_or(&0))
                .sum(),
        }
    }

    fn within_range(
        &self,
        min_class: u8,
        max_class: u8,
        num_points_in_block: usize,
    ) -> IndexResult {
        let matches_within_range: usize = self
            .histogram
            .iter()
            .filter_map(|(key, val)| {
                if *key < min_class || *key >= max_class {
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
}

#[typetag::serde]
impl Index for ClassificationIndex {
    fn matches(
        &self,
        atomic_expression: &AtomicExpression,
        num_points_in_block: usize,
    ) -> IndexResult {
        match atomic_expression {
            AtomicExpression::Compare((compare_expr, value)) => {
                match value {
                    Value::Classification(classification) => {
                        let num_matches =
                            self.get_matches_from_histogram(*compare_expr, classification.0, num_points_in_block);
                        if num_matches == 0 {
                            IndexResult::NoMatch
                        } else if num_matches == num_points_in_block {
                            IndexResult::MatchAll
                        } else {
                            IndexResult::MatchSome
                        }
                    },
                    other => panic!("Encountered invalid value in 'Compare' expression. Expected Classification but got {} instead", other.value_type()),
                } 
            },
            AtomicExpression::Within(range) => {
                match (range.start, range.end) {
                    (Value::Classification(min_class), Value::Classification(max_class)) => self.within_range(min_class.0, max_class.0, num_points_in_block),
                    (other_min, other_max) => panic!("Encountered invalid values for range of 'Within' expression. Expected (Classification, Classification) but got ({},{}) instead", other_min.value_type(), other_max.value_type()),
                }
            },
            other => panic!("ClassificationIndex does not support query expression {other:#?}"),
        }
    }

    fn value_type(&self) -> ValueType {
        ValueType::Classification
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct PointRange {
    pub file_index: usize,
    pub points_in_file: Range<usize>,
}

impl PartialOrd for PointRange {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PointRange {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.file_index.cmp(&other.file_index) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        self.points_in_file.start.cmp(&other.points_in_file.start)
    }
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

#[derive(Serialize, Deserialize)]
pub struct Block {
    point_range: PointRange,
    index: Option<Box<dyn Index>>,
}

impl Block {
    /// Minimum size of a block in a block index. This constant seems reasonable because at some point the
    /// overhead of checking an index outweighs its benefit. e.g. the extreme case where there is one block
    /// per point, that would waste a huge amount of memory and performance
    pub const MIN_BLOCK_SIZE: usize = 1 << 10;

    pub fn new(points_in_file: Range<usize>, file_index: usize) -> Self {
        Self {
            point_range: PointRange {
                file_index,
                points_in_file,
            },
            index: None,
        }
    }

    pub fn with_index(
        points_in_file: Range<usize>,
        file_index: usize,
        index: Box<dyn Index>,
    ) -> Self {
        Self {
            point_range: PointRange {
                file_index,
                points_in_file,
            },
            index: Some(index),
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
    pub fn refine(
        &self,
        value_type: ValueType,
        // reader: &mut dyn PointReader,
    ) -> Result<Vec<Block>> {
        todo!()
        // So either we have an index, in which case we split it up into smaller indices, or we don't, in which
        // case we still could create the smaller number of indices (because the RefinementStrategy has determined
        // that we have the time to refine this index, which means we have the time to scan through all points of
        // this index)

        // Depending on how large the block is, split it into either 2 or 4 sub-blocks
        // We can experiment with refinement strategies here, this is just a first test
        // let num_splits = match self.point_range.points_in_file.len() / Self::MIN_BLOCK_SIZE {
        //     0..=1 => panic!("Should not refine block that is less than 2x the MIN_BLOCK_SIZE"),
        //     2..=3 => 2,
        //     _ => 4,
        // };

        // let new_block_ranges = self
        //     .point_range
        //     .points_in_file
        //     .clone()
        //     .divide_evenly_into(num_splits);

        // match value_type {
        //     ValueType::Classification => {
        //         let classifications = reader
        //             .read_classifications(self.point_range.points_in_file.clone())
        //             .context("Could not read classifications")?;

        //         let start_index = self.point_range.points_in_file.start;
        //         Ok(new_block_ranges
        //             .map(|new_block_range| {
        //                 let local_range = (new_block_range.start - start_index)
        //                     ..(new_block_range.end - start_index);
        //                 let classification_index = ClassificationIndex::build_from_classifications(
        //                     classifications[local_range].into_iter().copied(),
        //                 );
        //                 Block::with_index(
        //                     new_block_range,
        //                     self.point_range.file_index,
        //                     Box::new(classification_index),
        //                 )
        //             })
        //             .collect())
        //     }
        //     ValueType::Position3D => {
        //         let positions = reader
        //             .read_positions(self.point_range.points_in_file.clone())
        //             .context("Could not read positions")?;

        //         let start_index = self.point_range.points_in_file.start;
        //         Ok(new_block_ranges
        //             .map(|new_block_range| {
        //                 let local_range = (new_block_range.start - start_index)
        //                     ..(new_block_range.end - start_index);
        //                 let positions_index = PositionIndex::build_from_positions(
        //                     positions[local_range].into_iter().copied(),
        //                 );
        //                 Block::with_index(
        //                     new_block_range,
        //                     self.point_range.file_index,
        //                     Box::new(positions_index),
        //                 )
        //             })
        //             .collect())
        //     },
        //     _ => unimplemented!(),
        // }
    }
}

#[derive(Serialize, Deserialize)]
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
    pub fn apply_refinements<I: Iterator<Item = PointRange>>(
        &mut self,
        blocks_to_refine: I,
        value_type: ValueType,
        files: &[PathBuf],
    ) -> Result<()> {
        // We can refine all blocks that are passed to this method because candidate selection happens within the
        // ProgressiveIndex
        
        for (file_index, blocks_in_file) in &blocks_to_refine.group_by(|block| block.file_index) {
            todo!()
        }

        Ok(())

        //     let mut reader = open_reader(&files[file_index]).context(format!(
        //         "Can't open reader to file {}",
        //         files[file_index].display()
        //     ))?;

        //     // Refine all blocks that are large enough to be refined. We can't guarantee that the refinement strategy
        //     // always gives valid blocks, since users might implement their own refinement strategy
        //     for block_in_file in blocks_in_file
        //         .filter(|block| block.points_in_file.len() >= 2 * Block::MIN_BLOCK_SIZE)
        //     {
        //         // TODO Maybe we won't find the exact block, because the input ranges might have been combined? But I think
        //         // I implemented it in a way that we always get exact matches and never combine the 'to-refine' ranges!
        //         let position_of_old_block = self
        //             .blocks
        //             .iter()
        //             .position(|block| block.point_range == block_in_file)
        //             .expect("Original block for refinement not found!");

        //         let block = &self.blocks[position_of_old_block];
        //         let refined_blocks = block
        //             .refine(value_type, reader.as_mut())
        //             .context("Failed to refine block")?;
        //         self.blocks.splice(
        //             position_of_old_block..=position_of_old_block,
        //             refined_blocks,
        //         );
        //     }
        // }

        // Ok(())
    }
}
