use anyhow::{anyhow, Result};
use bincode;
use pasture_core::{math::AABB, nalgebra::{Point3, Vector3}};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct LASHeader {
    pub signature: [i8; 4],
    pub file_source_id: u16,
    pub global_encoding: u16,
    pub guid1: u32,
    pub guid2: u16,
    pub guid3: u16,
    pub guid4: [u8; 8],
    pub version_major: u8,
    pub version_minor: u8,
    pub system_identifier: [i8; 32],
    pub creating_software: [i8; 32],
    pub file_creation_day_of_year: u16,
    pub file_creation_year: u16,
    pub header_size: u16,
    pub offset_to_point_data: u32,
    pub num_vlrs: u32,
    pub point_format: u8,
    pub point_record_length: u16,
    pub num_point_records: u32,
    pub num_points_by_returns: [u32; 5],
    pub x_scale: f64,
    pub y_scale: f64,
    pub z_scale: f64,
    pub x_offset: f64,
    pub y_offset: f64,
    pub z_offset: f64,
    pub max_x: f64,
    pub min_x: f64,
    pub max_y: f64,
    pub min_y: f64,
    pub max_z: f64,
    pub min_z: f64,
}

impl LASHeader {
    pub fn bounds(&self) -> AABB<f64> {
        AABB::from_min_max(
            Point3::new(self.min_x, self.min_y, self.min_z),
            Point3::new(self.max_x, self.max_y, self.max_z),
        )
    }
}

/**
 * Try parsing a LAS header from the given binary blob
 */
pub fn try_parse_las_header(data: &[u8]) -> Result<LASHeader> {
    if data.len() < 227 {
        return Err(anyhow!(
            "Could not parse LAS header, buffer has to be at least 227 bytes large!"
        ));
    }
    bincode::deserialize(data).map_err(|e| anyhow!("Could not deserialize LAS header: {}", e))
}
