//! Bounds of ordinary tessellated geometry.
pub fn positions_bounds(positions: &[f32]) -> Option<[f64; 4]> {
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for p in positions.chunks_exact(2) {
        let (x, y) = (p[0] as f64, p[1] as f64);
        if !(x.is_finite() && y.is_finite()) {
            continue;
        }
        b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
    }
    (b[0] <= b[2] && b[1] <= b[3]).then_some(b)
}
