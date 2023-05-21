use std::{fs::File, path::PathBuf};

use anyhow::{Context, Result};
use log::warn;
use query::index::ProgressiveIndex;

/// Name of the `PROG_INDEX_ROOT` environment variable
const ENV_PROG_INDEX_ROOT: &str = "PROG_INDEX_ROOT";

/// Returns the default path to the data directory
fn default_data_dir() -> PathBuf {
    match std::env::consts::OS {
        "macos" => {
            let mut home = std::env::home_dir().expect("Can't get path to home directory");
            home.push("/Library/Application Support/Progressive Indexer");
            home
        }
        _ => panic!("Unsupported OS"),
    }
}

/// Returns the common data directory where things like the ProgressiveIndex are stored. This can be set through
/// the `PROG_INDEX_ROOT` environment variable
pub(crate) fn data_dir() -> PathBuf {
    match std::env::var(ENV_PROG_INDEX_ROOT) {
        Ok(var) => var.into(),
        _ => {
            let default_data_dir = default_data_dir();
            // Make sure data dir exists
            std::fs::create_dir_all(&default_data_dir)
                .expect("Failed to create default data directory");

            std::env::set_var(ENV_PROG_INDEX_ROOT, default_data_dir.display().to_string());
            default_data_dir
        }
    }
}

/// Persist the ProgressiveIndex to disk, using the default output directory given by the `PROG_INDEX_ROOT` env variable
fn persist_index(index: &ProgressiveIndex) -> Result<()> {
    let root = data_dir();
    let datasets_dir = root.join("datasets");
    for (id, dataset) in index.datasets() {
        let file = datasets_dir.join(format!("{id}"));
        serde_json::to_writer(
            File::create(file).context("Failed to create file for dataset")?,
            dataset,
        )
        .context("Failed to serialize dataset")?;
    }
    Ok(())
}

/// Try to load the ProgressiveIndex from disk. If there is no index on disk, an empty ProgressiveIndex is returned
fn load_index_from_disk() -> Result<ProgressiveIndex> {
    let root = data_dir();
    let datasets_dir = root.join("datasets");
    let datasets = std::fs::read_dir(datasets_dir)?
        .filter_map(|entry_in_dataset_dir| match entry_in_dataset_dir {
            Ok(entry) => {
                let dataset_id = entry.path().file_stem()?.to_string_lossy().parse().ok()?;
                let dataset = File::open(entry.path()).and_then(|file| {
                    let dataset = serde_json::from_reader(file)?;
                    Ok(dataset)
                });
                match dataset {
                    Ok(dataset) => Some((dataset_id, dataset)),
                    Err(why) => {
                        warn!(
                            "Failed to load index from file {} with error {}",
                            entry.path().display(),
                            why
                        );
                        None
                    }
                }
            }
            Err(why) => {
                warn!(
                    "Invalid file found while trying to load ProgressiveIndex from disk: {}",
                    why
                );
                None
            }
        })
        .collect();

    Ok(ProgressiveIndex::with_datasets(datasets))
}

fn main() {
    println!("Hello, world!");
}
