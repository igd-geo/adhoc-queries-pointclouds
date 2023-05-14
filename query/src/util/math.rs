use std::ops::Range;

/// Computes the intersection of two ranges. If the two ranges do not intersect, an empty range is returned
/// with an implementation-defined value for start and end being equal to start
pub fn intersect_ranges(range1: &Range<usize>, range2: &Range<usize>) -> Range<usize> {
    let start = range1.start.max(range2.start);
    let end = range1.end.min(range2.end);
    if start <= end {
        start..end
    } else {
        let min = start.min(end);
        min..min
    }
}