use std::{fs::File, hint::black_box, io::BufReader, time::Instant};

use anyhow::Result;
use io::lazer::LazerReader;
use statrs::statistics::{Data, Distribution, Median};

fn main() -> Result<()> {
    let vals = [];
    let data = Data::new(vals);
    println!(
        "Median: {} Mean: {:#?} Stddev: {:#?}",
        data.median(),
        data.mean(),
        data.std_dev()
    );
    return Ok(());

    let _client = tracy_client::Client::start();

    let timer = Instant::now();
    let reader = LazerReader::new(BufReader::new(File::open("/Users/pbormann/data/projects/progressive_indexing/experiment_data/ahn4s/lazer/C_25GN1.lazer")?))?;
    black_box(reader);
    eprintln!("Creating reader took {:.3}s", timer.elapsed().as_secs());
    Ok(())
}
