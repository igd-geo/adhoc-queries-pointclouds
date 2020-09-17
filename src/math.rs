use nalgebra::Vector3;

/**
 * 3D axis-aligned bounding box
 */
#[derive(Debug)]
pub struct AABB<T: nalgebra::Scalar + PartialOrd> {
    pub min: Vector3<T>,
    pub max: Vector3<T>,
}

impl<T: nalgebra::Scalar + PartialOrd> AABB<T> {
    pub fn new(min: Vector3<T>, max: Vector3<T>) -> Self {
        Self { min: min, max: max }
    }

    pub fn intersects(&self, other: &AABB<T>) -> bool {
        (self.min.x <= other.max.x && self.max.x >= other.min.x)
            && (self.min.y <= other.max.y && self.max.y >= other.min.y)
            && (self.min.z <= other.max.z && self.max.z >= other.min.z)
    }
}
