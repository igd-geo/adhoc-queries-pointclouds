use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use query::io::get_point_files_in_path;

#[derive(Parser)]
struct Args {
    input: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let files = if args.input.is_file() {
        vec![args.input.clone()]
    } else {
        get_point_files_in_path(&args.input)
    };

    todo!()
}
