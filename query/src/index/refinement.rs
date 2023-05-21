use rustc_hash::FxHashSet;

use super::PointRange;

/// Trait for an index refinement strategy that defines which parts of the index shall be refined
pub trait RefinementStrategy {
    /// Selects the best candidates for refinement from the given PointRanges
    fn select_best_candidates(
        &self,
        potential_refinements: FxHashSet<PointRange>,
    ) -> Vec<PointRange>;
}

/// Refinement strategy that never refines any index
pub struct NoRefinementStrategy;

impl RefinementStrategy for NoRefinementStrategy {
    fn select_best_candidates(
        &self,
        _potential_refinements: FxHashSet<PointRange>,
    ) -> Vec<PointRange> {
        vec![]
    }
}
