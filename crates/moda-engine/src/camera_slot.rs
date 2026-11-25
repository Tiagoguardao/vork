//! Mask and depth remapping shared by page plans and prepared rows.

use std::sync::Arc;

use crate::seams::DrawItem;

pub(crate) const CANVAS_LIST_KEY: &str = "\u{0}list:canvas-root";

pub(crate) fn item_mask_id(item: &DrawItem) -> Option<u64> {
    match item {
        DrawItem::Body { mask, .. } => mask.as_ref().map(|m| m.mask_id),
        // vork: a text batch carries no mask (moda-text-draw's TextQuadBatch; see PROVENANCE.md).
        DrawItem::Text(_) => None,
        DrawItem::LayerBlurComposite { mask, .. } => mask.as_ref().map(|m| m.mask_id),
        _ => None,
    }
}

pub(crate) fn set_item_mask_id(item: &mut DrawItem, id: u64) {
    match item {
        DrawItem::Body { mask: Some(m), .. } => m.mask_id = id,
        DrawItem::LayerBlurComposite { mask: Some(m), .. } => m.mask_id = id,
        // vork: a text batch carries no mask (moda-text-draw's TextQuadBatch; see PROVENANCE.md).
        _ => {}
    }
}

pub(crate) fn z_shiftable(item: &DrawItem, dz: i64) -> bool {
    let z_ok = |z: f64| {
        let a = z.floor();
        if (z * 4.0).fract() == 0.0 {
            return true;
        }
        let b = a + dz as f64;
        (a >= 1.0 && b >= 1.0 && (a as u64).ilog2() == (b as u64).ilog2()) || (a == 0.0 && dz > 0)
    };
    let depth_ok = |z: f64, d: f32| crate::depth::linear_depth(z) as f32 == d;
    match item {
        DrawItem::Body { z, instance, paint, .. } => {
            z_ok(*z)
                && depth_ok(*z, instance.path_info[3])
                && match paint {
                    crate::seams::BodyPaint::Pattern(p) => depth_ok(*z, p.instance.path_info[3]),
                    crate::seams::BodyPaint::Gradient(g) => {
                        depth_ok(*z, g.path_info[2])
                    }
                    _ => true,
                }
        }
        DrawItem::Shadow { z, instance, .. } => z_ok(*z) && depth_ok(*z, instance.shape_data[3]),
        DrawItem::SolidShapeSdf { z, instance, .. } => z_ok(*z) && depth_ok(*z, instance.dimensions[2]),
        DrawItem::GradientShapeSdf { z, instance, .. } => {
            z_ok(*z) && depth_ok(*z, instance.dimensions[2])
        }
        // Text bakes its depth at present, from `z`.
        DrawItem::Text(b) => z_ok(b.z),
        DrawItem::LayerBlurComposite { z, group, .. } => {
            z_ok(*z)
                && group.items.iter().all(|i| z_shiftable(i, dz))
                && group.coverage.iter().all(|i| z_shiftable(i, dz))
        }
        DrawItem::Vacant | DrawItem::BackdropSplit(_) | DrawItem::PageFill { .. } => true,
    }
}

pub(crate) fn shift_item_z(item: &mut DrawItem, dz: i64) {
    let d = dz as f64;
    let depth = |z: f64| crate::depth::linear_depth(z) as f32;
    match item {
        DrawItem::Body { z, instance, paint, .. } => {
            *z += d;
            instance.path_info[3] = depth(*z);
            // A pattern or gradient body draws from its paint's own instance, which bakes the depth too (an image
            // fill kept its old depth after a z shift: found by the raster spike's paint layers in the fuzz).
            match paint {
                crate::seams::BodyPaint::Pattern(p) => p.instance.path_info[3] = depth(*z),
                crate::seams::BodyPaint::Gradient(g) => g.path_info[2] = depth(*z),
                _ => {}
            }
        }
        DrawItem::Shadow { z, instance, .. } => {
            *z += d;
            instance.shape_data[3] = depth(*z);
        }
        DrawItem::SolidShapeSdf { z, instance, .. } => {
            *z += d;
            instance.dimensions[2] = depth(*z);
        }
        DrawItem::GradientShapeSdf { z, instance, .. } => {
            *z += d;
            instance.dimensions[2] = depth(*z);
        }
        DrawItem::Text(b) if dz != 0 => std::sync::Arc::make_mut(b).z += d,
        DrawItem::Text(_) => {}
        DrawItem::LayerBlurComposite { z, group, .. } => {
            *z += d;
            let g = Arc::make_mut(group);
            // The noise's silhouette items too (they bake the depth as the body's do): a noise-filtered layer
            // shifted in render-z kept its silhouette at the old z (found by the slot fuzz's shader backgrounds).
            for i in g.items.iter_mut().chain(g.coverage.iter_mut()) {
                shift_item_z(i, dz);
            }
        }
        DrawItem::Vacant | DrawItem::BackdropSplit(_) | DrawItem::PageFill { .. } => {}
    }
}
