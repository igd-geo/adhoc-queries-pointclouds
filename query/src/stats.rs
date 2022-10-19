use std::{fmt::Display, time::Duration};

#[derive(Debug, Copy, Clone)]
pub struct QueryStats {
    pub total_blocks_queried: usize,
    pub full_match_blocks: usize,
    pub partial_match_blocks: usize,
    pub total_points_queried: usize,
    pub matching_points: usize,
    pub runtime: Duration,
}

impl Display for QueryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Runtime: {}ms", self.runtime.as_secs_f64() * 1000.0)?;
        writeln!(f, "Total blocks queried: {}", self.total_blocks_queried)?;
        writeln!(f, "Full match blocks: {}", self.full_match_blocks)?;
        writeln!(f, "Partial match blocks: {}", self.partial_match_blocks)?;
        writeln!(f, "Total points queried: {}", self.total_points_queried)?;
        writeln!(f, "Matching points: {}", self.matching_points)?;
        Ok(())
    }
}
