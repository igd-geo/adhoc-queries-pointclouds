mod last_reader;
mod lazer_reader;

pub use self::last_reader::*;
pub use self::lazer_reader::*;

use pasture_core::nalgebra::Vector3;
use pasture_derive::PointType;

#[repr(C, packed)]
#[derive(PointType, Copy, Clone, Debug, Default)]
pub struct Point {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_COLOR_RGB)]
    pub color: Vector3<u16>,
    #[pasture(BUILTIN_CLASSIFICATION)]
    pub classification: u8,
}
