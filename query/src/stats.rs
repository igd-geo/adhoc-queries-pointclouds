use std::{fmt::Display, io::Write, sync::Mutex, time::Duration};

use anyhow::Result;
use human_repr::{HumanCount, HumanDuration};
use rustc_hash::FxHashMap;

use crate::index::PointRange;

#[derive(Debug, Copy, Clone)]
pub struct QueryStats {
    pub total_blocks_queried: usize,
    pub full_match_blocks: usize,
    pub partial_match_blocks: usize,
    pub total_points_queried: usize,
    pub matching_points: usize,
    pub runtime: Duration,
    pub refinement_time: Duration,
}

impl Display for QueryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Runtime:               {}",
            self.runtime.human_duration()
        )?;
        writeln!(
            f,
            "Refinement time:       {}",
            self.refinement_time.human_duration()
        )?;
        writeln!(
            f,
            "Total blocks queried:  {}",
            self.total_blocks_queried.human_count_bare()
        )?;
        writeln!(
            f,
            "Full match blocks:     {}",
            self.full_match_blocks.human_count_bare()
        )?;
        writeln!(
            f,
            "Partial match blocks:  {}",
            self.partial_match_blocks.human_count_bare()
        )?;
        writeln!(
            f,
            "Total points queried:  {}",
            self.total_points_queried.human_count("points")
        )?;
        writeln!(
            f,
            "Matching points:       {}",
            self.matching_points.human_count("points")
        )?;
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BlockQueryRuntimeType {
    Eval,
    Refinement,
    Extraction,
}

#[derive(Default, Debug)]
pub struct BlockQueryRuntimeTracker {
    blocks: Mutex<FxHashMap<PointRange, FxHashMap<BlockQueryRuntimeType, Duration>>>,
}

impl BlockQueryRuntimeTracker {
    pub fn log_runtime(
        &self,
        block: PointRange,
        runtime_type: BlockQueryRuntimeType,
        duration: Duration,
    ) {
        let mut blocks = self.blocks.lock().expect("Mutex was poisoned");
        if !blocks.contains_key(&block) {
            blocks.insert(block.clone(), Default::default());
        }
        blocks
            .get_mut(&block)
            .unwrap()
            .insert(runtime_type, duration);
    }

    pub fn to_csv<W: Write>(&self, mut writer: W) -> Result<()> {
        writeln!(
            writer,
            "file;block start;block end;eval;refinement;extraction"
        )?;
        let blocks = self.blocks.lock().expect("Mutex was poisoned");
        for (block, entries) in &*blocks {
            let eval_duration = entries
                .get(&BlockQueryRuntimeType::Eval)
                .copied()
                .unwrap_or(Duration::default());
            let refine_duration = entries
                .get(&BlockQueryRuntimeType::Refinement)
                .copied()
                .unwrap_or(Duration::default());
            let extraction_duration = entries
                .get(&BlockQueryRuntimeType::Extraction)
                .copied()
                .unwrap_or(Duration::default());
            writeln!(
                writer,
                "{};{};{};{};{};{}",
                block.file_index,
                block.points_in_file.start,
                block.points_in_file.end,
                eval_duration.as_millis(),
                refine_duration.as_millis(),
                extraction_duration.as_millis()
            )?;
        }

        Ok(())
    }
}
