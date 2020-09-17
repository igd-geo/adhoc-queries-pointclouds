pub trait ResultCollector {
    fn collect_one(&mut self, point: &[u8]);
}

pub struct BufferCollector {
    buffer: Vec<u8>,
}

impl BufferCollector {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn buffer(&self) -> &[u8] {
        &self.buffer[..]
    }
}

impl ResultCollector for BufferCollector {
    fn collect_one(&mut self, point: &[u8]) {
        self.buffer.extend_from_slice(point);
    }
}

pub struct StdOutCollector {}

impl StdOutCollector {
    pub fn new() -> Self {
        Self {}
    }
}

impl ResultCollector for StdOutCollector {
    fn collect_one(&mut self, point: &[u8]) {
        // LAS is little-endian
        let x = u32::from_le_bytes([point[0], point[1], point[2], point[3]]);
        let y = u32::from_le_bytes([point[4], point[5], point[6], point[7]]);
        let z = u32::from_le_bytes([point[8], point[9], point[10], point[11]]);
        println!("Found point: ({}, {}, {})", x, y, z);
    }
}
