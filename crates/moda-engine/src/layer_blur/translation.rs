//! Capture-local solid geometry. The key and the draw use the same normalized
//! inputs; a whole-pixel move therefore cannot change the capture's pixels.
use crate::seams::{BodyPaint, DrawItem, LayerBlurGroup, PaintInstance};

/// Only a unit linear transform and whole canvas-pixel placement. Other
/// transforms keep the original rasterization, including during scale tracks.
pub(crate) fn pixel_translation(m: &[f32; 16]) -> bool {
    m[0] == 1.0 && m[1] == 0.0 && m[4] == 0.0 && m[5] == 1.0
        && [m[12], m[13]].iter().all(|v| v.is_finite() && v.fract() == 0.0)
}

// The planner snaps edges within 1e-3 target pixels (plan::SNAP_PX).
// Remove the residual f32 projection error using that same grid convention.
pub(crate) fn eligible(g: &LayerBlurGroup) -> bool {
    let [x, y, w, h] = g.region_canvas;
    g.filters.source_raster.is_none() && [x, y].iter().all(|v| v.is_finite() && (v - v.round()).abs() < 1e-3 && v.abs() < 8_388_608.0)
        && (w - g.width_px as f64).abs() < 1e-3 && (h - g.height_px as f64).abs() < 1e-3
        && matches!(g.filters.blur, Some(crate::seams::BlurFilter::Uniform { .. }))
        && g.masks.is_empty() && g.coverage.is_empty()
        && g.filters.color_matrices.is_empty() && g.filters.texture.is_none() && g.filters.noise.is_empty()
        && g.filters.feather.is_none() && g.filters.shadow.is_none()
        && g.filters.drop_shadows.is_empty()
        && g.items.iter().all(|i| matches!(i, DrawItem::Body { paint: BodyPaint::Solid, mask: None, instance, .. }
            if pixel_translation(&instance.model_matrix)))
}

pub(crate) fn instance(p: &PaintInstance, region: [f64; 4]) -> PaintInstance {
    let region = [region[0].round(), region[1].round(), region[2].round(), region[3].round()];
    let mut p = *p;
    for axis in 0..2 {
        p.model_matrix[12 + axis] = (p.model_matrix[12 + axis] as f64 - region[axis]) as f32;
        // Outside the capture no fragment exists. Clamping makes an unchanged
        // enclosing page clip independent of the capture's canvas position.
        let lo = p.page_clip_bounds[axis] as f64;
        let hi = lo + p.page_clip_bounds[axis + 2] as f64;
        let a = (lo - region[axis]).clamp(0.0, region[axis + 2]);
        let b = (hi - region[axis]).clamp(0.0, region[axis + 2]);
        p.page_clip_bounds[axis] = a as f32;
        p.page_clip_bounds[axis + 2] = (b - a) as f32;
    }
    p
}
