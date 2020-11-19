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

    pub fn contains(&self, point: &Vector3<T>) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }
}

impl From<AABB<f64>> for pointstream::math::AABB {
    fn from(bounds: AABB<f64>) -> Self {
        pointstream::math::AABB::from_min_max(bounds.min, bounds.max)
    }
}

impl From<&AABB<f64>> for pointstream::math::AABB {
    fn from(bounds: &AABB<f64>) -> Self {
        pointstream::math::AABB::from_min_max(bounds.min, bounds.max)
    }
}
