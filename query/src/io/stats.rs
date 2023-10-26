use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Context, Result};
use itertools::Itertools;
use lazy_static::lazy_static;
use log::warn;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileFormat {
    LAS,
    LAZ,
    LAST,
    LAZER,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IOMethod {
    File,
    Mmap,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IOStatsParameters {
    pub file_format: FileFormat,
    pub io_method: IOMethod,
    pub point_record_format: u8,
}

/// Statistics about the I/O performance of a the current machine
#[derive(Debug, Serialize, Deserialize)]
pub struct IOStats {
    pub throughputs_mpts: HashMap<IOStatsParameters, f64>,
}

lazy_static! {
    static ref GLOBAL_IO_STATS: Option<IOStats> = {
        let stats_from_config = match IOStats::load_from_config() {
            Ok(stats) => stats,
            Err(why) => {
                warn!("Could not load default IOStats ({why})");
                None
            }
        };
        stats_from_config
    };
}

impl IOStats {
    pub fn new(throughputs_mpts: HashMap<IOStatsParameters, f64>) -> Self {
        Self { throughputs_mpts }
    }

    pub(crate) fn load_from_config() -> Result<Option<Self>> {
        let path = Self::default_path().context("Could not get default path for IOStats config")?;
        if !path.exists() {
            return Ok(None);
        }
        let stats_helper: Vec<(IOStatsParameters, f64)> =
            serde_json::from_reader(BufReader::new(File::open(&path).with_context(|| {
                format!("Could not open IOStats config file {}", path.display())
            })?))
            .context("Could not parse IOStats from default config file")?;
        let stats = stats_helper.into_iter().collect();
        Ok(Some(Self {
            throughputs_mpts: stats,
        }))
    }

    pub fn store_to_config(&self) -> Result<()> {
        let path = Self::default_path().context("Could not get default path for IOStats config")?;
        std::fs::create_dir_all(
            path.parent()
                .expect("Could not get parent directory from default path"),
        )
        .with_context(|| {
            format!(
                "Failed to create directories for IOStats config file at {}",
                path.display()
            )
        })?;

        // Write to a local file and move, to prevent overwriting file if something goes wrong
        let local_path = Path::new("./io_stats.config.tmp");
        let serialize_helper = self
            .throughputs_mpts
            .iter()
            .map(|(a, b)| (a.clone(), b.clone()))
            .collect_vec();
        serde_json::to_writer_pretty(
            BufWriter::new(File::create(local_path).with_context(|| {
                format!("Failed to create temporary file {}", local_path.display())
            })?),
            &serialize_helper,
        )
        .context("Failed to serialize IOStats")?;

        std::fs::rename(local_path, &path).with_context(|| {
            format!(
                "Failed to overwrite existing IOStats config file {}",
                path.display()
            )
        })?;

        Ok(())
    }

    pub fn global() -> Option<&'static IOStats> {
        GLOBAL_IO_STATS.as_ref()
    }

    pub fn throughputs_mpts(&self) -> &HashMap<IOStatsParameters, f64> {
        &self.throughputs_mpts
    }

    pub fn default_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir().ok_or(anyhow!("Could not determine config dir"))?;
        let stats_path = config_dir.join("adhoc_queries").join("io_stats.json");
        Ok(stats_path)
    }
}
