//! Fractional depth ordering of paint layers and arrowheads.
pub const LAYER_Z_SPAN: f64 = 0.45;
thread_local! {
    static Z_SPAN: std::cell::Cell<f64> = const { std::cell::Cell::new(LAYER_Z_SPAN) };
}
pub fn with_z_span<R>(span: f64, f: impl FnOnce() -> R) -> R {
    let prev = Z_SPAN.with(|c| c.replace(span));
    let out = f();
    Z_SPAN.with(|c| c.set(prev));
    out
}
pub fn item_z(z: f64, k: usize, n: usize) -> f64 {
    let span = Z_SPAN.with(std::cell::Cell::get);
    if span == LAYER_Z_SPAN {
        return z + 0.05 + 0.4 * k as f64 / n.max(1) as f64;
    }
    z + span * (0.05 + 0.4 * k as f64 / n.max(1) as f64) / LAYER_Z_SPAN
}
