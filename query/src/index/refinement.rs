use std::time::Duration;

use itertools::Itertools;
use rustc_hash::FxHashSet;

use crate::io::InputLayer;

use super::{Block, DatasetID, PointRange, ValueType};

fn block_is_too_small(block: &PointRange) -> bool {
    block.points_in_file.len() < 2 * Block::MIN_BLOCK_SIZE
}

/// Trait for an index refinement strategy that defines which parts of the index shall be refined
pub trait RefinementStrategy {
    /// Selects the best candidates for refinement from the given PointRanges
    fn select_best_candidates(
        &self,
        potential_refinements: FxHashSet<PointRange>,
        value_type: ValueType,
        dataset_id: DatasetID,
        input_layer: &InputLayer,
    ) -> Vec<PointRange>;
}

/// Refinement strategy that never refines any index
pub struct NoRefinementStrategy;

impl RefinementStrategy for NoRefinementStrategy {
    fn select_best_candidates(
        &self,
        _potential_refinements: FxHashSet<PointRange>,
        _value_type: ValueType,
        _dataset_id: DatasetID,
        _input_layer: &InputLayer,
    ) -> Vec<PointRange> {
        vec![]
    }
}

/// Refinement strategy that always refines all blocks that can be refined
pub struct AlwaysRefinementStrategy;

impl RefinementStrategy for AlwaysRefinementStrategy {
    fn select_best_candidates(
        &self,
        potential_refinements: FxHashSet<PointRange>,
        _value_type: ValueType,
        _dataset_id: DatasetID,
        _input_layer: &InputLayer,
    ) -> Vec<PointRange> {
        potential_refinements
            .into_iter()
            .filter(|range| range.points_in_file.len() >= 2 * Block::MIN_BLOCK_SIZE)
            .collect_vec()
    }
}

/// Refine for a maximum time per query. Uses heuristics to figure out how many blocks can be refined
/// within that time limit. If all possible block are more work-intensive than the time-limit allows,
/// the smallest of the blocks is refined so that we always make some progress
pub struct TimeBudgetRefinementStrategy {
    max_time: Duration,
}

impl TimeBudgetRefinementStrategy {
    /// How many times the pure I/O duration do we think refinement takes? This is an estimate based on
    /// intuition as well as some measurements. We know I/O is not parallelized, but the actual refinement
    /// is mostly compute and hence is parallelized, so the overhead should be small
    const IO_TO_REFINEMENT_FACTOR_UNCOMPRESSED: f64 = 1.2;
    const IO_TO_REFINEMENT_FACTOR_LAZ: f64 = 0.15;

    pub fn new(max_time: Duration) -> Self {
        Self { max_time }
    }
}

impl RefinementStrategy for TimeBudgetRefinementStrategy {
    fn select_best_candidates(
        &self,
        potential_refinements: FxHashSet<PointRange>,
        value_type: ValueType,
        dataset_id: DatasetID,
        input_layer: &InputLayer,
    ) -> Vec<PointRange> {
        let mut expected_times_for_refinement = potential_refinements
            .into_iter()
            .filter_map(|range| {
                if block_is_too_small(&range) {
                    return None;
                }
                let expected_time_io = input_layer
                    .estimate_io_time_for_point_range(dataset_id, &range, value_type)
                    .unwrap_or(Duration::MAX);
                let expected_refinement_time =
                    expected_time_io.mul_f64(Self::IO_TO_REFINEMENT_FACTOR_UNCOMPRESSED);
                Some((expected_refinement_time, range))
            })
            .collect::<Vec<_>>();
        if expected_times_for_refinement.is_empty() {
            return vec![];
        }

        expected_times_for_refinement.sort_by_key(|v| v.0);

        // If everything will take too long, just refine the block that takes the least amount of time so that
        // we always make some progress
        if expected_times_for_refinement[0].0 > self.max_time {
            return vec![expected_times_for_refinement[0].1.clone()];
        }

        // Take the first N blocks until their cumulative duration exceeds the maximum refinement duration
        // Explanation for this heuristic:
        //   We benchmarked the maximum I/O throughput for point reading and use this as an estimate in the
        //   readers in the InputLayer. I/O itself does not parallelize well, so even though refinement might
        //   be parallelized, we estimate single-threaded runtimes. This is more conservative, but it is an
        //   estimate and doesn't have to be perfect
        expected_times_for_refinement
            .into_iter()
            .scan(self.max_time, |accum, (duration, range)| {
                if *accum < duration {
                    None
                } else {
                    *accum -= duration;
                    Some(range)
                }
            })
            .collect()
    }
}
