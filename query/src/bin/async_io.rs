use std::{
    hint::black_box,
    process::Command,
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};
use human_repr::HumanThroughput;
use rand::{distributions::Uniform, thread_rng, Rng};
use statrs::statistics::{Data, Distribution, Median};
use tabled::{builder::Builder, settings::Style};

const TEST_FILE_PATH: &str = "testdata.bin";
const TEST_FILE_SIZE: usize = 1 << 25; //32MiB
const NUM_SAMPLES: usize = 10;

struct Sample {
    name: &'static str,
    throughput_mean: f64,
    throughput_median: f64,
    throughput_stddev: f64,
}

fn flush_disk_cache() -> Result<()> {
    let sync_output = Command::new("sync")
        .output()
        .context("Could not execute sync command")?;
    if !sync_output.status.success() {
        bail!("Sync command failed with exit code {}", sync_output.status);
    }

    if std::env::consts::OS == "macos" {
        let purge_output = Command::new("purge")
            .output()
            .context("Could not execute purge command")?;
        if !purge_output.status.success() {
            bail!(
                "Purge command failed with exit code {}",
                purge_output.status
            );
        }
    }

    Ok(())
}

fn gen_test_file() -> Result<()> {
    let rnd_data = thread_rng()
        .sample_iter::<u8, _>(Uniform::from(0..255))
        .take(TEST_FILE_SIZE)
        .collect::<Vec<_>>();
    std::fs::write(TEST_FILE_PATH, rnd_data)?;
    Ok(())
}

fn baseline_with_alloc() -> Result<Duration> {
    let timer = Instant::now();
    let data = std::fs::read(TEST_FILE_PATH)?;
    black_box(data);
    Ok(timer.elapsed())
}

fn exec(name: &'static str, mut func: impl FnMut() -> Result<Duration>) -> Result<Sample> {
    let runtimes = (0..NUM_SAMPLES)
        .map(|_| -> Result<Duration> {
            flush_disk_cache()?;
            func()
        })
        .collect::<Result<Vec<_>>>()?;

    let throughputs = runtimes
        .iter()
        .map(|runtime| TEST_FILE_SIZE as f64 / runtime.as_secs_f64())
        .collect::<Vec<_>>();
    let throughputs_data = Data::new(throughputs);

    let throughput_mean = throughputs_data.mean().unwrap();
    let throughput_median = throughputs_data.median();
    let throughput_stddev = throughputs_data.std_dev().unwrap();

    Ok(Sample {
        name,
        throughput_mean,
        throughput_median,
        throughput_stddev,
    })
}

fn main() -> Result<()> {
    gen_test_file()?;
    scopeguard::defer! {
        std::fs::remove_file(TEST_FILE_PATH).expect("Failed to remove test file");
    }

    let samples = vec![exec("Baseline (alloc)", baseline_with_alloc)?];

    let mut table_builder = Builder::default();
    table_builder.set_header(["Name", "Throughput (mean)", "Throughput (median)", "Error"]);
    for sample in samples {
        table_builder.push_record([
            sample.name.to_string(),
            sample.throughput_mean.human_throughput_bytes().to_string(),
            sample
                .throughput_median
                .human_throughput_bytes()
                .to_string(),
            sample
                .throughput_stddev
                .human_throughput_bytes()
                .to_string(),
        ]);
    }

    let mut table = table_builder.build();
    table.with(Style::modern());
    eprintln!("{table}");

    Ok(())
}
