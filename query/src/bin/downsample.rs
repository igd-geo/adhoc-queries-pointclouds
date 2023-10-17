use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use query::{
    index::{
        AtomicExpression, CompareExpression, DiscreteLod, NoRefinementStrategy, ProgressiveIndex,
        QueryExpression, Value,
    },
    io::{get_point_files_in_path, LASOutput},
};

#[derive(Parser)]
struct Args {
    input: PathBuf,
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let files = if args.input.is_file() {
        vec![args.input.clone()]
    } else {
        get_point_files_in_path(&args.input)
    };

    let mut index = ProgressiveIndex::new();
    let dataset_id = index.add_dataset(&files)?;

    let stats = index.dataset_stats(dataset_id);

    let output = LASOutput::new(&args.output, stats.point_layout()).with_context(|| {
        format!(
            "Could not create LAS output for file {}",
            args.output.display()
        )
    })?;
    let query = QueryExpression::Atomic(AtomicExpression::Compare((
        CompareExpression::Equals,
        Value::LOD(DiscreteLod(3)),
    )));
    index
        .query(dataset_id, query, &NoRefinementStrategy, &output)
        .context("Failed to query for subsample of point cloud")?;

    Ok(())
}
