use std::{
    fs::File,
    hint::black_box,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};
use memmap2::Advice;
use rand::{distributions::Uniform, thread_rng, Rng};

fn gen_test_files() -> Result<Vec<PathBuf>> {
    eprintln!("Generating test files...");

    const SIZES: [usize; 5] = [1 << 12, 1 << 16, 1 << 20, 1 << 24, 1 << 28];
    const NAMES: [&str; 5] = ["4K", "64K", "1M", "16M", "128M"];

    SIZES
        .iter()
        .zip(NAMES.iter())
        .map(|(size, name)| -> Result<PathBuf> {
            let rnd_data = thread_rng()
                .sample_iter::<u8, _>(Uniform::from(0..255))
                .take(*size)
                .collect::<Vec<_>>();
            let path = PathBuf::from(format!("test_file_{name}.bin"));
            std::fs::write(path.as_path(), rnd_data)?;
            Ok(path)
        })
        .collect()
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

fn read_file(file: &Path, count: usize) -> Result<Duration> {
    let timer = Instant::now();
    let data = std::fs::read(file)?;
    for _ in 0..count {
        for b in data.iter() {
            black_box(*b);
        }
    }
    Ok(timer.elapsed())
}

fn read_mmap(file: &Path, count: usize) -> Result<Duration> {
    let timer = Instant::now();
    let file = File::open(file)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    mmap.advise(Advice::sequential())?;
    for _ in 0..count {
        for b in mmap.iter() {
            black_box(*b);
        }
    }
    Ok(timer.elapsed())
}

fn bench(files: &[PathBuf], flush_cache: bool) -> Result<()> {
    let flush_str = if flush_cache { "with" } else { "without" };
    eprintln!("Benchmarking {flush_str} flushing disk cache");

    let profile_files_with_count = |count: usize| -> Result<Vec<Duration>> {
        files
            .iter()
            .map(|file| {
                if flush_cache {
                    flush_disk_cache()?;
                }
                read_file(file, count)
            })
            .collect()
    };
    let profile_mmap_with_count = |count: usize| -> Result<Vec<Duration>> {
        files
            .iter()
            .map(|file| {
                if flush_cache {
                    flush_disk_cache()?;
                }
                read_mmap(file, count)
            })
            .collect()
    };

    let runtimes_file = [
        ("1x", profile_files_with_count(1)?),
        ("2x", profile_files_with_count(2)?),
        ("4x", profile_files_with_count(4)?),
        ("8x", profile_files_with_count(8)?),
    ];
    let runtimes_mmap = [
        ("1x", profile_mmap_with_count(1)?),
        ("2x", profile_mmap_with_count(2)?),
        ("4x", profile_mmap_with_count(4)?),
        ("8x", profile_mmap_with_count(8)?),
    ];

    println!("method,4k,64k,1M,16M,128M");
    for (method, file_runtimes) in runtimes_file {
        println!(
            "File ({method}),{}",
            file_runtimes
                .iter()
                .map(|runtime| runtime.as_millis().to_string())
                .collect::<Vec<String>>()
                .join(",")
        );
    }

    for (method, mmap_runtimes) in runtimes_mmap {
        println!(
            "mmap ({method}),{}",
            mmap_runtimes
                .iter()
                .map(|runtime| runtime.as_millis().to_string())
                .collect::<Vec<String>>()
                .join(",")
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    let os = std::env::consts::OS;
    if os != "linux" && os != "macos" {
        bail!("Can only run this benchmark on Linux or MacOS");
    }

    let files = gen_test_files().context("Failed to create test files")?;
    scopeguard::defer! {
        for file in files.iter() {
            match std::fs::remove_file(file.as_path()) {
                Ok(_) => (),
                Err(why) => eprintln!("Failed to remove file {} ({})", file.display(), why),
            }
        }
    }

    bench(&files, false).context("Benchmarks without flushing file cache failed")?;
    bench(&files, true).context("Benchmarks with flushing file cache failed")?;

    Ok(())
}
