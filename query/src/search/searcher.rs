use crate::collect_points::ResultCollector;
use crate::search::search_last_file_by_bounds;
use crate::search::search_last_file_by_bounds_optimized;
use crate::search::search_last_file_by_classification;
use crate::search::search_last_file_by_classification_optimized;
use crate::search::search_lazer_file_by_bounds;
use crate::search::search_lazer_file_by_classification;
use crate::search::{
    search_las_file_by_bounds, search_las_file_by_bounds_optimized,
    search_las_file_by_classification, search_las_file_by_classification_optimized,
    search_laz_file_by_bounds, search_laz_file_by_bounds_optimized,
    search_laz_file_by_classification, search_laz_file_by_classification_optimized,
};
use anyhow::{anyhow, Result};
use pasture_core::math::AABB;
use std::ffi::OsStr;
use std::path::Path;

pub enum SearchImplementation {
    Regular,
    Optimized,
}

pub trait Searcher {
    fn search_file(
        &self,
        path: &Path,
        search_impl: &SearchImplementation,
        collector: &mut dyn ResultCollector,
    ) -> Result<()>;
}

pub struct BoundsSearcher {
    bounds: AABB<f64>,
}

impl BoundsSearcher {
    pub fn new(bounds: AABB<f64>) -> Self {
        Self { bounds: bounds }
    }
}

impl Searcher for BoundsSearcher {
    fn search_file(
        &self,
        path: &Path,
        search_impl: &SearchImplementation,
        collector: &mut dyn ResultCollector,
    ) -> Result<()> {
        match path.extension().and_then(OsStr::to_str) {
            Some("las") => match search_impl {
                SearchImplementation::Regular => {
                    search_las_file_by_bounds(path, &self.bounds, collector)
                }
                SearchImplementation::Optimized => {
                    search_las_file_by_bounds_optimized(path, &self.bounds, collector)
                }
            },
            Some("laz") => match search_impl {
                SearchImplementation::Regular => {
                    search_laz_file_by_bounds(path, &self.bounds, collector)
                }
                SearchImplementation::Optimized => {
                    search_laz_file_by_bounds_optimized(path, &self.bounds, collector)
                }
            },
            Some("last") => match search_impl {
                SearchImplementation::Regular => {
                    search_last_file_by_bounds(path, &self.bounds, collector)
                }
                SearchImplementation::Optimized => {
                    search_last_file_by_bounds_optimized(path, &self.bounds, collector)
                }
            },
            // Some("laser") => match search_impl {
            //     SearchImplementation::Regular => {
            //         search_laser_file_by_bounds(path, &self.bounds, collector)
            //     }
            //     SearchImplementation::Optimized => {
            //         search_laser_file_by_bounds_optimized(path, &self.bounds, collector)
            //     }
            // },
            Some("lazer") => search_lazer_file_by_bounds(path, &self.bounds, collector),
            Some(_) => Err(anyhow!(
                "Unsupported file extension in file {}",
                path.display()
            )),
            None => Err(anyhow!("Invalid extension on file {}", path.display())),
        }
    }
}

/// Searcher for searching points by class
pub struct ClassSearcher {
    class: u8,
}

impl ClassSearcher {
    pub fn new(class: u8) -> Self {
        Self { class: class }
    }
}

impl Searcher for ClassSearcher {
    fn search_file(
        &self,
        path: &Path,
        search_impl: &SearchImplementation,
        collector: &mut dyn ResultCollector,
    ) -> Result<()> {
        match path.extension().and_then(OsStr::to_str) {
            Some("las") => match search_impl {
                SearchImplementation::Regular => {
                    search_las_file_by_classification(path, self.class, collector)
                }
                SearchImplementation::Optimized => {
                    search_las_file_by_classification_optimized(path, self.class, collector)
                }
            },
            Some("laz") => match search_impl {
                SearchImplementation::Regular => {
                    search_laz_file_by_classification(path, self.class, collector)
                }
                SearchImplementation::Optimized => {
                    search_laz_file_by_classification_optimized(path, self.class, collector)
                }
            },
            Some("last") => match search_impl {
                SearchImplementation::Regular => {
                    search_last_file_by_classification(path, self.class, collector)
                }
                SearchImplementation::Optimized => {
                    search_last_file_by_classification_optimized(path, self.class, collector)
                }
            },
            // Some("laser") => match search_impl {
            //     SearchImplementation::Regular => {
            //         search_laser_file_by_classification(path, self.class, collector)
            //     }
            //     SearchImplementation::Optimized => {
            //         search_laser_file_by_classification_optimized(path, self.class, collector)
            //     }
            // },
            Some("lazer") => search_lazer_file_by_classification(path, self.class, collector),
            Some(_) => Err(anyhow!(
                "Unsupported file extension in file {}",
                path.display()
            )),
            None => Err(anyhow!("Invalid extension on file {}", path.display())),
        }
    }
}
