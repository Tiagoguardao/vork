/// A length in output pixels (`len × scale`) rounded up to whole pixels, after rounding to 1e-6 so float error
/// doesn't add one: 960 × (300 / 72) is 4000.0000000000005, and a plain `ceil` made a 960-wide page's 300 dpi PNG
/// 4001 px wide (a 1920-wide slide 8001). Camera and
/// page plans use the same rounding.
#[inline]
pub fn px_ceil(v: f64) -> f64 {
    ((v * 1e6).round() / 1e6).ceil()
}

#[cfg(test)]
mod tests {
    use super::px_ceil;

    #[test]
    fn output_pixels_ignore_float_error() {
        let dpi300 = 300.0 / 72.0;
        assert_eq!(px_ceil(960.0 * dpi300), 4000.0);
        assert_eq!(px_ceil(1920.0 * dpi300), 8000.0);
        assert_eq!(px_ceil(595.2756 * dpi300), 2481.0);
        assert_eq!(px_ceil(841.8898 * dpi300), 3508.0);
        assert_eq!(px_ceil(100.5), 101.0);
    }
}
