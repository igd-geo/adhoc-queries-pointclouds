use crate::index::{CompareExpression, Position};

use super::CompiledQueryAtom;

pub(crate) struct LastQueryAtomWithin<T> {
    min: T,
    max: T,
}

impl<T> LastQueryAtomWithin<T> {
    pub(crate) fn new(min: T, max: T) -> Self {
        Self { min, max }
    }
}

pub(crate) struct LastQueryAtomCompare<T> {
    value: T,
    compare_expression: CompareExpression,
}

impl<T> LastQueryAtomCompare<T> {
    pub(crate) fn new(value: T, compare_expression: CompareExpression) -> Self {
        Self {
            value,
            compare_expression,
        }
    }
}

impl CompiledQueryAtom for LastQueryAtomCompare<Position> {
    fn eval(
        &self,
        input_layer: &crate::io::InputLayer,
        block: crate::index::PointRange,
        dataset_id: crate::index::DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &crate::stats::BlockQueryRuntimeTracker,
    ) -> anyhow::Result<usize> {
        todo!()
    }
}
