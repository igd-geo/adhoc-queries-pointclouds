use las::point::Point;

pub trait ResultCollector {
    fn collect_one(&mut self, point: Point);
}

pub struct BufferCollector {
    buffer: Vec<Point>,
}

impl BufferCollector {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn buffer(&self) -> &[Point] {
        &self.buffer[..]
    }
}

impl ResultCollector for BufferCollector {
    fn collect_one(&mut self, point: Point) {
        self.buffer.push(point);
    }
}

pub struct StdOutCollector {}

impl StdOutCollector {
    pub fn new() -> Self {
        Self {}
    }
}

impl ResultCollector for StdOutCollector {
    fn collect_one(&mut self, point: Point) {
        println!("Found point: {:#?}", point);
    }
}
