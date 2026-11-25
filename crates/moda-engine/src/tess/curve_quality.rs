//! Scale and flattening quality shared by ordinary canvas curves.
use super::TrimParams;
use moda_scene::types::RawNode;
use std::cell::Cell;
/// Outline fit + flattening tolerance at band 0 (1 px per canvas unit), in
/// canvas units — i.e. a fifth of a device pixel at every band.

pub const OUTLINE_TOLERANCE_PX: f64 = 0.2;
/// Zoom bands: `round(log2(px per unit))`, clamped. Below 1/16 the outline
/// stops getting coarser (tolerance ≤ 3.2 units); above 512 it stops getting
/// finer (≤ 0.0004 units). 512 device px per unit is the editor's top zoom,
/// 25600%, on a 2× display, so curves stay within a fifth of a device pixel
/// across the whole zoom range.

pub const MIN_BAND: i32 = -4;
pub const MAX_BAND: i32 = 9;
/// True when any node strokes or fills curves (ellipses, curved or rounded
/// paths, rounded shapes, connectors, curved or rounded line routes). Curve
/// flattening follows the zoom band
/// (`tessellator::flatness_tolerance`), so such scenes key camera plans on the
/// band too (quantised: one plan per band, `MIN_BAND..=MAX_BAND`);
/// straight-only scenes stay zoom-invariant.

pub fn scene_has_curves(scene: &moda_scene::Scene) -> bool {
    scene
        .node_order()
        .iter()
        .filter_map(|id| scene.effective_node(id))
        .chain(
            scene
                .transient_by_id()
                .keys()
                .filter_map(|id| scene.effective_node(id)),
        )
        .any(node_has_curves)
}
/// Whether `n`'s stroke draws round ends: round caps or joins, or a circle arrowhead. Their arcs flatten
/// per zoom band (`tessellator::round_arc_segments`), so a straight line or polyline with them is curved.

pub fn node_has_round_ends(n: &RawNode) -> bool {
    let style = moda_scene::paint::node_stroke_style(n);
    let stroked = n.node_type() == "line" || style.width.is_some_and(|w| w.is_finite() && w > 0.0);
    stroked
        && (style.cap.as_deref() == Some("round")
            || style.join.as_deref() == Some("round")
            || style.start_arrow.as_deref() == Some("circle")
            || style.end_arrow.as_deref() == Some("circle"))
}
/// The per-node term of [`scene_has_curves`].

pub fn node_has_curves(n: &RawNode) -> bool {
    // Illustrator's Appearance (E8.2): a layer's own geometry effects, and a group's merged outline, may curve
    // whatever the node's own outline is; counted as curved (the plan keys on the band).
    if crate::paint::layers::has_entry_effects(n) || moda_scene::appearance::group_has_appearance(n)
    {
        return true;
    }
    let curved = |n: &RawNode| -> bool {
        if node_has_round_ends(n) {
            return true;
        }
        // A line's body is curved on a curved route or a rounded elbow
        // (`line_to_path`); its stroke defaults on, so no stroke check.
        if n.node_type() == "line" {
            let routed = n.array("waypoints").is_some_and(|w| !w.is_empty());
            return routed
                && match n.str("routing") {
                    Some("curved") => true,
                    Some("elbowed") => n
                        .f64("cornerRadius")
                        .is_some_and(|r| r.is_finite() && r > 0.0),
                    _ => false,
                };
        }
        // Stroked or filled: both flatten per band.
        let stroked = moda_scene::paint::node_stroke_style(n)
            .width
            .is_some_and(|w| w.is_finite() && w > 0.0);
        if !stroked && !crate::paint::fill_paints(n) {
            return false;
        }
        let rounded = n
            .f64("cornerRadius")
            .is_some_and(|r| r.is_finite() && r > 0.0);
        match n.node_type() {
            "ellipse" | "connector" => true,
            "path" => {
                rounded
                    || n.points().is_some_and(|pts| {
                        pts.iter()
                            .any(|p| p.kind().is_some_and(|t| t != "move" && t != "line"))
                    })
            }
            "rectangle" | "container" | "star" | "polygon" => rounded,
            _ => false,
        }
    };
    curved(n)
}
/// The camera scale a plan build enters: the real one for scenes with
/// outlines or curves (both refit per band, above and below 1×), none
/// otherwise (straight-only scenes keep one zoom-invariant plan).

pub fn camera_scale_for(has_curves: bool, px_per_unit: f64) -> Option<f64> {
    has_curves.then_some(px_per_unit)
}
/// Zoom band for `px_per_unit` device pixels per canvas unit.

pub fn band_for_scale(px_per_unit: f64) -> i32 {
    if !(px_per_unit.is_finite() && px_per_unit > 0.0) {
        return 0;
    }
    (px_per_unit.log2().round() as i32).clamp(MIN_BAND, MAX_BAND)
}
/// Fit / flattening tolerance (canvas units) at `band`.

pub fn tolerance_for_band(band: i32) -> f64 {
    OUTLINE_TOLERANCE_PX / 2f64.powi(band.clamp(MIN_BAND, MAX_BAND))
}
thread_local! {
    /// Camera frames: device px per canvas unit (zoom × dpr), set for the
    /// duration of a camera plan build. Export / page plans leave it unset.
    static CAMERA_SCALE: Cell<Option<f64>> = const { Cell::new(None) };
    /// Page plans: `RenderOptions::scale`, set by the coverage plan builder
    /// when no camera scale is active.
    static PLAN_SCALE: Cell<Option<f64>> = const { Cell::new(None) };
}
/// Restores the previous scale on drop.

pub(crate) struct ScaleGuard {
    camera: bool,
    prev: Option<f64>,
}
impl Drop for ScaleGuard {
    fn drop(&mut self) {
        if self.camera {
            CAMERA_SCALE.with(|c| c.set(self.prev));
        } else {
            PLAN_SCALE.with(|c| c.set(self.prev));
        }
    }
}
/// Enter a camera plan build at `zoom × dpr` (only when the scene has
/// outlines; `None` keeps the camera plan zoom-invariant).

pub(crate) fn enter_camera(px_per_unit: Option<f64>) -> ScaleGuard {
    let prev = CAMERA_SCALE.with(|c| c.replace(px_per_unit));
    ScaleGuard { camera: true, prev }
}
/// Enter a page plan build at `scale` (the export pixel ratio).

pub(crate) fn enter_plan(scale: f64) -> ScaleGuard {
    let prev = PLAN_SCALE.with(|c| c.replace(Some(scale)));
    ScaleGuard {
        camera: false,
        prev,
    }
}
/// The band the current build draws outlines and curves at: the view's band
/// (camera or export scale), raised by the scale of the node being
/// tessellated ([`enter_node_scale`], [`node_band`]).

pub fn current_band() -> i32 {
    node_band(view_band(), NODE_SCALE.with(Cell::get))
}
/// The band of the view alone (camera `zoom × dpr`, or the export pixel
/// ratio), without the node's own scale.

pub fn view_band() -> i32 {
    let s = CAMERA_SCALE
        .with(Cell::get)
        .or_else(|| PLAN_SCALE.with(Cell::get))
        .unwrap_or(1.0);
    band_for_scale(s)
}
/// Highest band a node's own scale takes it to: 2^16 device px per node-local
/// unit. Outline fitting stays capped at [`MAX_BAND`] ([`tolerance_for_band`]
/// clamps); curve flattening and round caps / joins follow the band up to
/// here.

pub const MAX_NODE_BAND: i32 = 16;
thread_local! {
    /// The largest scale of the node being tessellated: its canvas affine's
    /// linear part (its own scale × its parents', animated values included),
    /// as [`affine_max_scale`] measures it. 1 outside a node.
    static NODE_SCALE: Cell<f64> = const { Cell::new(1.0) };
}
/// Restores the previous node scale on drop.

pub(crate) struct NodeScaleGuard(f64);
impl Drop for NodeScaleGuard {
    fn drop(&mut self) {
        NODE_SCALE.with(|c| c.set(self.0));
    }
}
/// Tessellate the next node at `scale` canvas units per node-local unit
/// ([`affine_max_scale`] of its canvas affine): its curves flatten for the
/// size they reach the screen at, not their source size.

pub(crate) fn enter_node_scale(scale: f64) -> NodeScaleGuard {
    NodeScaleGuard(NODE_SCALE.with(|c| c.replace(scale)))
}
/// The largest singular value of `affine`'s linear part `[a c; b d]`: how far
/// a node-local unit can stretch on the canvas in any direction (the larger
/// axis of a non-uniform scale, skew included). 1 for a degenerate or
/// non-finite affine.

pub fn affine_max_scale(affine: &moda_scene::types::Affine) -> f64 {
    let (a, b, c, d) = (affine.a, affine.b, affine.c, affine.d);
    let sum = 0.5 * (a * a + b * b + c * c + d * d);
    let diff = 0.5 * (a * a + b * b - c * c - d * d);
    let cross = a * c + b * d;
    let s = (sum + diff.hypot(cross)).sqrt();
    if s.is_finite() && s > 0.0 {
        s
    } else {
        1.0
    }
}
/// The band a node draws its curves at: the view's band raised by the node's
/// own scale (`node_scale` canvas units per node-local unit) in the same
/// power-of-two steps, `round(view_band + log2(node_scale))`, up to
/// [`MAX_NODE_BAND`]. A small shape scaled up flattens for its on-screen
/// size (within a device pixel), and a scale change within the same octave
/// keeps its band and so its cached tessellation. A scale below 1 never lowers
/// the view's band.

pub fn node_band(view_band: i32, node_scale: f64) -> i32 {
    if !(node_scale.is_finite() && node_scale > 1.0) {
        return view_band;
    }
    let band = (f64::from(view_band) + node_scale.log2()).round();
    (band as i32).clamp(view_band, MAX_NODE_BAND.max(view_band))
}
/// The camera band folded into plan keys (`None` outside a camera build or
/// for a scene without outlines).

pub fn camera_key_band() -> Option<i32> {
    CAMERA_SCALE.with(Cell::get).map(band_for_scale)
}
/// Trim window for an ordinary stroked path.
pub fn trim_params_from_node(node: &RawNode) -> Option<TrimParams> {
    super::trim_params_from_fields(
        |k| node.f64(k),
        node.str("trimPathMode"),
        node.str("trimPathBy"),
    )
}

pub fn curve_cache_salt(band: i32) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    b"stroke-outline".hash(&mut h);
    band.hash(&mut h);
    // Preserve the salt used by ordinary curves before brush removal.
    None::<u64>.hash(&mut h);
    h.finish()
}
