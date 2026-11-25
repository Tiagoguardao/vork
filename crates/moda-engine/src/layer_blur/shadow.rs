//! Plan side of the capture shadows: drop and inner shadows for every shape
//! the SDF shadow lane (`effects::shadow`, rect / ellipse silhouettes only)
//! can't draw — paths (vector networks, booleans, outlined text), stars,
//! polygons, lines, partial ellipses, distorted nodes — and rich text's inner
//! shadows (its drop shadows are the text lane's own glyph passes).
//!
//! Figma draws a layer's shadows from the layer's shape: a drop shadow is the
//! shape moved by the offset and blurred (σ = blur / 2), knocked out under
//! the layer unless "show behind"; an inner shadow is the shape's outside
//! moved and blurred, clipped to the shape. Spread applies to rectangles,
//! ellipses and frames only (Figma's `DropShadowEffect.spread`; the SDF lane,
//! `effects::shadow`), so it plays no part here: a path, star, polygon, line,
//! text or fill-less frame casts its plain silhouette whatever the spread, as
//! in Figma, and a shadow that only spreads draws nothing.
//!
//! The shape is the node's SILHOUETTE ([`silhouette_node`]): every visible
//! fill and its widest stroke, painted opaque black, at opacity 1. A fill's
//! own opacity doesn't weaken the shadow (Figma: fill opacity keeps the shadow,
//! layer opacity fades it) — the node's effective opacity multiplies each
//! shadow's alpha instead, like the SDF lane.
//!
//! One capture group per shadow kind (`LayerBlurGroup` with
//! `LayerFilters::shadow`), rendered in the layer-filter pre-pass; its result
//! holds the shadows alone, and the main pass composites it at the body's z
//! (drop, drawn before the body in its bucket) or z + `INNER_SHADOW_Z_OFFSET`
//! (inner, over the body), clipped to the node's effective clip.
//!
//! Glows (Illustrator's Outer and Inner Glow, `glow` / `inner-glow`) are the
//! same passes with no offset and blur = the glow's radius, on every shape
//! and on text: an outer glow shown behind, an inner glow clipped to the
//! shape, from its outside (edge) or the shape itself (center). Each glow
//! mode is its own group, composited with that mode (`CaptureKind::Glow` /
//! `InnerGlow`, `seams::CaptureBlend`).
//!
//! The capture region is the silhouette's drawn bounds grown by the shadows'
//! reach, cut to the part of the target the shadows can reach (the viewport
//! grown by the same reach), so it stays bounded at any zoom and the capture
//! is at full target resolution: the knockout and the inner-shadow clip stay
//! as crisp as the body.

use moda_scene::effect::{DropShadow, InnerShadow};
use moda_scene::types::RawNode;
use moda_scene::Scene;

use crate::backdrop::plan::{PlanCtx, Rect};
use crate::seams::{CaptureBlend, LayerBlurGroup, LayerFilters, ShadowFilter, ShadowPass};

/// wgpu texture dimension ceiling (as the layer blur).
const MAX_TEXTURE_DIM: f64 = 8192.0;
/// Capture area budget (as the layer blur): beyond it the capture scale drops.
const MAX_CAPTURE_AREA_PX: f64 = 4096.0 * 4096.0;
/// Empty border kept around the capture, in capture px: samples past the
/// capture clamp to its edge, which must read "no shape".
const EDGE_PX: f64 = 2.0;
/// How close to a whole target pixel a capture edge must be to count as on it.
const SNAP_PX: f64 = 1e-3;
/// How far off the target's pixel grid a full-scale capture sits (see its use).
const CAPTURE_NUDGE_PX: f64 = 1.0 / 64.0;

/// Node types whose shadows the capture lane may draw.
fn is_capture_type(node_type: &str) -> bool {
    matches!(
        node_type,
        "rectangle"
            | "container"
            | "ellipse"
            | "star"
            | "polygon"
            | "path"
            | "line"
            | "richtext"
            | "group"
    )
}

/// Node types that contribute a silhouette to a subtree capture (a group's
/// or an unfilled frame's shadows, [`is_subtree_capture_node`]).
pub(crate) fn is_silhouette_type(node_type: &str) -> bool {
    is_capture_type(node_type) && node_type != "group"
}

/// Whether the node's shadows (and layer filters) come from its whole
/// subtree, as in Figma: a group, or a frame without a visible fill that has
/// children. Its capture is the composited silhouette of every visible
/// descendant (plus the frame's own stroke) instead of its own shape. A frame
/// WITH a fill keeps its rectangle shadow (the SDF lane).
pub fn is_subtree_capture_node(node: &RawNode) -> bool {
    let has_children = node.array("children").is_some_and(|c| !c.is_empty());
    match node.node_type() {
        "group" => true,
        "container" => {
            has_children
                && !moda_scene::paint::node_fills_shared(node)
                    .iter()
                    .any(|f| moda_scene::paint::shown(f.visible, f.opacity))
        }
        _ => false,
    }
}

/// The opacity `id` paints with inside `root`'s subtree: its own opacity
/// times every parent node's between it and `root` (neither `root`'s nor
/// above it — those fade the root's shadow as a whole). `None` when `root`
/// is not an ancestor of `id` (or `id` is `root`).
pub(crate) fn opacity_within(scene: &Scene, id: &str, root: &str) -> Option<f64> {
    let own = scene
        .effective_node(id)
        .and_then(|n| n.f64("opacity"))
        .unwrap_or(1.0);
    let parent_of = |c: &str| {
        scene
            .parent_map()
            .get(c)
            .or_else(|| scene.transient_parent_of().get(c))
            .cloned()
    };
    let mut product = own;
    let mut cursor = parent_of(id);
    // Bounded: corrupt parent data (a cycle) must not hang the planner. A chain of parents visits each link at
    // most once, so no valid chain is longer than the links the scene holds, however deep its nodes nest.
    let links = scene.parent_map().len() + scene.transient_parent_of().len();
    for _ in 0..=links {
        let pid = cursor?;
        if pid == root {
            return Some(product);
        }
        if let Some(p) = scene.effective_node(&pid) {
            if moda_scene::scene::is_parent_node(p) {
                product *= p.f64("opacity").unwrap_or(1.0);
            }
        }
        cursor = parent_of(&pid);
    }
    None
}

/// Whether `root` is an ancestor of `id` (through groups, frames and
/// transient hosts).
pub(crate) fn is_within(scene: &Scene, id: &str, root: &str) -> bool {
    opacity_within(scene, id, root).is_some()
}

/// The drop shadows the capture lane draws for `node`, back to front: those of
/// a shape without an SDF silhouette. Rich text draws its own (text lane),
/// except text in a complex script, which the degraded raster tier draws
/// without effects: its shadows come from its silhouette here.
/// The spread is ignored, as Figma does on these shapes, so a shadow that
/// only spreads draws nothing here.
pub fn capture_drop_shadows(node: &RawNode) -> Vec<DropShadow> {
    let t = node.node_type();
    if !is_capture_type(t)
        || (t == "richtext" && !richtext_draws_degraded(node))
        || crate::effects::shadow::has_sdf_silhouette(node)
    {
        return Vec::new();
    }

    moda_scene::effect::drop_shadows(node)
        .into_iter()
        .filter(|s| s.blur > 0.0 || !s.offset.is_zero())
        .collect()
}

/// Whether rich text draws through the degraded raster tier, which paints its
/// glyphs' fill only: text with a letter of a complex script (Arabic, Hebrew,
/// Indic, Thai, …; `moda_text::complex_script`), as the vector lane declines it.
pub(crate) fn richtext_draws_degraded(node: &RawNode) -> bool {
    node.str("htmlContent")
        .is_some_and(moda_text::complex_script::text_needs_complex_shaping)
}

/// The inner shadows the capture lane draws for `node`, back to front: those
/// of a shape without an SDF silhouette, and rich text's. A shadow that only
/// spreads draws nothing here (spread is rectangles', ellipses' and frames').
pub fn capture_inner_shadows(node: &RawNode) -> Vec<InnerShadow> {
    if !is_capture_type(node.node_type()) || crate::effects::shadow::has_sdf_silhouette(node) {
        return Vec::new();
    }
    moda_scene::effect::inner_shadows(node)
        .into_iter()
        .filter(|s| s.blur > 0.0 || !s.offset.is_zero())
        .collect()
}

/// One glow of a shape or a text, as the capture lane draws it: Illustrator's
/// Outer Glow (`glow`) or Inner Glow (`inner-glow`).
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureGlow {
    pub color: moda_scene::value::Value,
    /// Halo radius (the Blur), node-local px: σ = radius / 2.
    pub radius: f64,
    /// `opacity` (0-1) × `intensity` (≥ 0, may exceed 1): the halo's weight.
    pub weight: f64,
    /// `mode`: how the halo composites (absent = screen, as in Illustrator).
    pub blend: CaptureBlend,
    /// Inner glows only: `source: 'center'` (glow from the middle outward)
    /// rather than the default `edge` (from the edges inward).
    pub center: bool,
}

/// Every visible entry of `effect_type` (`glow` / `inner-glow`) of a capture
/// shape or text that has a radius and a weight, back to front.
fn glows_of(node: &RawNode, effect_type: &str) -> Vec<CaptureGlow> {
    if !is_capture_type(node.node_type()) {
        return Vec::new();
    }
    moda_scene::effect::visible_effects_json(node, effect_type)
        .filter_map(|o| {
            let f = |k: &str| o.get(k).and_then(moda_scene::value::Value::as_f64);
            let radius = f("radiusPx").filter(|r| r.is_finite() && *r > 0.0)?;
            let weight = f("opacity").unwrap_or(1.0).clamp(0.0, 1.0)
                * f("intensity").unwrap_or(1.0).max(0.0);
            (weight.is_finite() && weight > 0.0).then(|| CaptureGlow {
                color: o
                    .get("color")
                    .filter(|c| c.is_string())
                    .cloned()
                    .unwrap_or_else(|| moda_scene::value::Value::from("#000000")),
                radius,
                weight,
                blend: CaptureBlend::from_glow_mode(
                    o.get("mode").and_then(moda_scene::value::Value::as_str),
                ),
                center: o.get("source").and_then(moda_scene::value::Value::as_str) == Some("center"),
            })
        })
        .collect()
}

/// The outer glows the capture lane draws for `node` (every shape, rich text
/// included), back to front.
pub fn capture_glows(node: &RawNode) -> Vec<CaptureGlow> {
    glows_of(node, "glow")
}

/// The inner glows the capture lane draws for `node`, back to front.
pub fn capture_inner_glows(node: &RawNode) -> Vec<CaptureGlow> {
    glows_of(node, "inner-glow")
}

/// Whether the node draws any capture shadow or glow.
pub fn has_capture_shadow(node: &RawNode) -> bool {
    !capture_drop_shadows(node).is_empty()
        || !capture_inner_shadows(node).is_empty()
        || !capture_glows(node).is_empty()
        || !capture_inner_glows(node).is_empty()
}

/// Which of a node's capture effects one group draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureKind {
    /// Drop shadows: at the body's z, before it.
    Drop,
    /// The outer glows of one mode: the silhouette blurred in the glow
    /// colour, not knocked out, composited with the mode over what lies below;
    /// at the body's z, before it and after the drop shadows.
    Glow(CaptureBlend),
    /// Inner shadows: at z + `INNER_SHADOW_Z_OFFSET`, over the body.
    Inner,
    /// The inner glows of one mode: the silhouette's outside (edge) or the
    /// silhouette itself (center) blurred, clipped to the silhouette and
    /// composited with the mode; over the body, after the inner shadows.
    InnerGlow(CaptureBlend),
}

impl CaptureKind {
    /// The groups `node` plans, in plan order: drop shadows, the outer glows
    /// (one group per mode, in the order the modes first appear), inner
    /// shadows, the inner glows (likewise). A kind with nothing to draw plans
    /// no group ([`plan_shadow_group`] declines it).
    pub fn for_node(node: &RawNode) -> Vec<CaptureKind> {
        let modes = |glows: Vec<CaptureGlow>| {
            let mut out: Vec<CaptureBlend> = Vec::new();
            for g in glows {
                if !out.contains(&g.blend) {
                    out.push(g.blend);
                }
            }
            out
        };
        let mut kinds = vec![CaptureKind::Drop];
        kinds.extend(
            modes(capture_glows(node))
                .into_iter()
                .map(CaptureKind::Glow),
        );
        kinds.push(CaptureKind::Inner);
        kinds.extend(
            modes(capture_inner_glows(node))
                .into_iter()
                .map(CaptureKind::InnerGlow),
        );
        kinds
    }

    /// Drawn over the body, clipped to the silhouette.
    pub fn is_inner(self) -> bool {
        matches!(self, CaptureKind::Inner | CaptureKind::InnerGlow(_))
    }

    /// The composite's node-id suffix: `glow` / `innerglow` for the default
    /// (screen) mode, `glow-<mode>` / `innerglow-<mode>` for the others.
    pub fn suffix(self) -> String {
        let glow = |base: &str, blend: CaptureBlend| {
            if blend == CaptureBlend::Screen {
                base.to_string()
            } else {
                format!("{base}-{}", blend.glow_mode())
            }
        };
        match self {
            CaptureKind::Drop => "dropshadow".into(),
            CaptureKind::Glow(b) => glow("glow", b),
            CaptureKind::Inner => "innershadow".into(),
            CaptureKind::InnerGlow(b) => glow("innerglow", b),
        }
    }
}

/// Whether the node's visible fills are all images: its pixels' alpha is its
/// shape (a cut-out PNG casts the cut-out's shadow and glow, not its box's),
/// as Figma shadows a layer by what it draws. No visible fill: false.
pub fn fills_are_images(node: &RawNode) -> bool {
    use moda_scene::paint as sp;
    let fills = sp::node_fills_shared(node);
    let mut shown = fills.iter().filter(|f| sp::shown(f.visible, f.opacity)).peekable();
    shown.peek().is_some() && shown.all(|f| matches!(f.paint, sp::Paint::Image(_)))
}

/// The node's silhouette: the node with its visible fills folded to one opaque
/// black fill (image fills keep their pixels, at full opacity, for their
/// alpha: the shadow passes read only the capture's alpha), its widest visible
/// stroke painted opaque black, opacity 1, normal blend, and no effects but its
/// live geometry ones (which reshape the outline). Rich text also paints its
/// glyphs black (`textFills`).
pub fn silhouette_node(node: &RawNode) -> RawNode {
    use moda_scene::paint as sp;
    let black = || moda_scene::value::Value::from("#000000");
    let mut m = node.fields().clone();
    m.insert("opacity".into(), moda_scene::value::Value::from(1.0));
    m.remove("blendMode");

    m.remove("effects");
    let has_fill = sp::node_fills_shared(node)
        .iter()
        .any(|f| sp::shown(f.visible, f.opacity));
    sp::set_fills(
        &mut m,
        if fills_are_images(node) {
            sp::node_fills(node)
                .into_iter()
                .filter(|f| sp::shown(f.visible, f.opacity))
                .map(|mut f| {
                    f.opacity = None;
                    f.blend_mode = None;
                    f.effects = None;
                    f
                })
                .collect()
        } else if has_fill {
            vec![sp::Fill::solid("fill-0", black())]
        } else {
            Vec::new()
        },
    );
    let stroke = crate::paint::layers::widest_stroke(node).map(|mut s| {
        s.paint = sp::Paint::Solid(sp::SolidPaint::new(black()));
        s.opacity = None;
        s.visible = None;
        s.blend_mode = None;
        s
    });
    sp::set_strokes(&mut m, stroke.into_iter().collect());
    if node.node_type() == "richtext" {
        m.insert(
            "textFills".into(),
            moda_scene::value::json!([{"id": "text-fill-0", "paint": {"type": "solid", "color": "#000000"}}]),
        );
    }
    node.with_fields(m)
}

/// The shape a noise effect covers (`seams::LayerBlurGroup::coverage`): the
/// node's silhouette with its fill area always filled, at the node's opacity,
/// or `None` when the node's own paint already covers its shape (a visible,
/// opaque solid fill) or it has no fill area (a line, text). Figma grains a
/// shape with a transparent or missing fill too.
pub fn noise_coverage_node(node: &RawNode) -> Option<RawNode> {
    use moda_scene::paint as sp;
    if matches!(node.node_type(), "line" | "richtext") {
        return None;
    }
    let covered = sp::node_fills_shared(node).iter().any(|f| {
        f.visible != Some(false)
            && f.opacity.unwrap_or(1.0) >= 1.0
            && f.blend_mode.as_deref().is_none_or(|b| b == "normal")
            && matches!(&f.paint, sp::Paint::Solid(s)
                if s.rgba.is_some_and(|c| c.a >= 1.0))
    });
    if covered {
        return None;
    }
    let sil = silhouette_node(node);
    let mut m = sil.fields().clone();
    m.insert(
        "opacity".into(),
        moda_scene::value::Value::from(node.f64("opacity").unwrap_or(1.0)),
    );
    sp::set_fills(
        &mut m,
        vec![sp::Fill::solid("fill-0", moda_scene::value::Value::from("#000000"))],
    );
    Some(sil.with_fields(m))
}

/// Local → canvas scale of the node's blur: the shorter axis of its canvas
/// affine (the SDF lane blurs in node-local units, so a scaled node's shadow
/// blurs with it; the offset is canvas-space).
fn node_blur_scale(scene: &Scene, node: &RawNode) -> f64 {
    scene
        .node_canvas_affine(moda_scene::scene::paint_transform_id(node))
        .map(|a| a.a.hypot(a.b).min(a.c.hypot(a.d)))
        .filter(|s| s.is_finite() && *s > 0.0)
        .unwrap_or(1.0)
}

/// The canvas rect the target shows (the projection is axis-aligned).
fn target_canvas_rect(ctx: &PlanCtx) -> Option<Rect> {
    let (x0, y0) = ctx.project_point(0.0, 0.0)?;
    let (x1, y1) = ctx.project_point(1.0, 1.0)?;
    let (sx, sy) = (x1 - x0, y1 - y0);
    if !(sx.abs() > 1e-12 && sy.abs() > 1e-12) {
        return None;
    }
    let ax = (0.0 - x0) / sx;
    let bx = (ctx.width_px as f64 - x0) / sx;
    let ay = (0.0 - y0) / sy;
    let by = (ctx.height_px as f64 - y0) / sy;
    let r = Rect {
        x: ax.min(bx),
        y: ay.min(by),
        w: (bx - ax).abs(),
        h: (by - ay).abs(),
    };
    r.finite().then_some(r)
}

/// One shadow in canvas units, before the capture scale.
struct CanvasShadow {
    color: [f32; 4],
    offset: (f64, f64),
    /// 3σ in canvas units.
    reach: f64,
    show_behind: bool,
    /// Inner glow from the centre (`ShadowPass::center`).
    center: bool,
}

/// The capture group of one kind of the node's shadows, or `None` when it has
/// none, or none of it can reach the target. `items` are the silhouette's
/// draw items (canvas-absolute, unclipped, unmasked); `bounds` their canvas
/// AABB; `node_opacity` the node's effective opacity.
pub(crate) fn plan_shadow_group(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
    kind: CaptureKind,
    items: Vec<crate::seams::DrawItem>,
    bounds: Rect,
    node_opacity: f64,
) -> Option<LayerBlurGroup> {
    let blur_scale = node_blur_scale(scene, node);
    let shadow = |color: &moda_scene::value::Value, opacity: f64, blur: f64, ox: f64, oy: f64, behind| {
        let rgba = crate::effects::shadow::shadow_color_rgba(color);
        CanvasShadow {
            color: [
                rgba[0] as f32,
                rgba[1] as f32,
                rgba[2] as f32,
                (rgba[3] * opacity * node_opacity).clamp(0.0, 1.0) as f32,
            ],
            offset: (ox, oy),
            reach: 1.5 * blur.max(0.0) * blur_scale,
            show_behind: behind,
            center: false,
        }
    };
    let inner = kind.is_inner();
    // A glow is a shadow with no offset and blur = radius (σ = radius / 2),
    // whose weight may exceed 1 (intensity): the halo is min(1, weight ×
    // blurred source), so a stronger glow stays bright further out. The
    // colour's own alpha scales it.
    let glow = |g: &CaptureGlow| {
        let mut s = shadow(&g.color, 1.0, g.radius, 0.0, 0.0, true);
        let a = crate::effects::shadow::shadow_color_rgba(&g.color)[3].clamp(0.0, 1.0);
        s.color[3] = (g.weight * a * node_opacity.clamp(0.0, 1.0)) as f32;
        s.center = g.center;
        s
    };
    let shadows: Vec<CanvasShadow> = match kind {
        CaptureKind::Inner => capture_inner_shadows(node)
            .iter()
            .map(|s| shadow(&s.color, s.opacity(), s.blur, s.offset.x, s.offset.y, false))
            .collect(),
        // An outer glow shows behind the node (not knocked out).
        CaptureKind::Glow(blend) => capture_glows(node)
            .iter()
            .filter(|g| g.blend == blend)
            .map(glow)
            .collect(),
        // Edge: the outside blurred into the shape (an offset-free inner
        // shadow); center: the shape itself blurred, clipped to it.
        CaptureKind::InnerGlow(blend) => capture_inner_glows(node)
            .iter()
            .filter(|g| g.blend == blend)
            .map(glow)
            .collect(),
        CaptureKind::Drop => capture_drop_shadows(node)
            .iter()
            .map(|s| {
                shadow(
                    &s.color,
                    s.opacity(),
                    s.blur,
                    s.offset.x,
                    s.offset.y,
                    s.shows_behind_node(),
                )
            })
            .collect(),
    };
    let shadows: Vec<CanvasShadow> = shadows.into_iter().filter(|s| s.color[3] > 0.0).collect();
    if shadows.is_empty() || !bounds.finite() || !(bounds.w >= 0.0 && bounds.h >= 0.0) {
        return None;
    }
    // How far a shadow pixel reads from its source (offset + 3σ).
    let reach = shadows
        .iter()
        .map(|s| s.offset.0.abs().max(s.offset.1.abs()) + s.reach)
        .fold(0.0, f64::max);
    if !reach.is_finite() {
        return None;
    }
    let canvas_scale = ctx.canvas_target_scale().max(1e-4);
    // Where the shadows paint: a drop shadow around the shape by its reach, an
    // inner shadow inside the shape. One target px of antialiasing either way.
    let aa = 1.0 / canvas_scale;
    let paint_pad = if inner { aa } else { reach + aa };
    let mut region = Rect {
        x: bounds.x - paint_pad,
        y: bounds.y - paint_pad,
        w: bounds.w + 2.0 * paint_pad,
        h: bounds.h + 2.0 * paint_pad,
    };
    // Only what the target shows matters, and a shown pixel reads at most
    // `reach` away from itself.
    if let Some(view) = target_canvas_rect(ctx) {
        let pad = reach + aa;
        let reachable = Rect {
            x: view.x - pad,
            y: view.y - pad,
            w: view.w + 2.0 * pad,
            h: view.h + 2.0 * pad,
        };
        region = region.intersect(&reachable)?;
    }
    // Full target resolution, reduced only to fit the texture limits.
    let mut zoom = canvas_scale;
    let edge = |zoom: f64| EDGE_PX / zoom;
    let dims = |zoom: f64| {
        (
            (region.w + 2.0 * edge(zoom)) * zoom,
            (region.h + 2.0 * edge(zoom)) * zoom,
        )
    };
    let (w, h) = dims(zoom);
    if w.max(h) > MAX_TEXTURE_DIM {
        zoom *= MAX_TEXTURE_DIM / w.max(h);
    }
    let (w, h) = dims(zoom);
    if w * h > MAX_CAPTURE_AREA_PX {
        zoom *= (MAX_CAPTURE_AREA_PX / (w * h)).sqrt();
    }
    let e = edge(zoom);
    region = Rect {
        x: region.x - e,
        y: region.y - e,
        w: region.w + 2.0 * e,
        h: region.h + 2.0 * e,
    };
    // A full-scale capture lands its texels on target pixels (the composite
    // then reads one texel per pixel).
    if zoom == canvas_scale {
        if let Some((px, py)) = ctx.project_point(region.x, region.y) {
            let off = |v: f64| {
                if (v - v.round()).abs() < SNAP_PX {
                    v - v.round()
                } else {
                    v - v.floor()
                }
            };
            // Less a 1/64 px nudge: texel centres sit a hair off the pixel
            // centres (the composite's bilinear read moves 1.6% of a
            // neighbour in), so a centre that would land EXACTLY on a glyph's
            // horizontal extremum doesn't — there the vector text lane's
            // winding ray grazes the curve and leaks a hairline of coverage,
            // which the inner shadow would paint.
            let nudge = CAPTURE_NUDGE_PX / canvas_scale;
            let (dx, dy) = (
                off(px) / canvas_scale + nudge,
                off(py) / canvas_scale + nudge,
            );
            region = Rect {
                x: region.x - dx,
                y: region.y - dy,
                w: region.w + dx,
                h: region.h + dy,
            };
        }
    }
    let whole = |v: f64| {
        if (v - v.round()).abs() < SNAP_PX {
            v.round()
        } else {
            v.ceil()
        }
    };
    // A wide blur runs at 1/2^n resolution (`exec::shadow_capture`): whole
    // 2^n-texel blocks keep each halving an exact 2×2 average, so pad the
    // capture (right and bottom, with more empty border) to a multiple.
    let block = {
        let mut r = shadows.iter().map(|s| s.reach * zoom).fold(0.0, f64::max);
        let mut b = 1.0;
        while r > crate::layer_blur::exec::PROGRESSIVE_BLUR_MAX_TAPS as f64 {
            r /= 2.0;
            b *= 2.0;
        }
        b
    };
    let whole_blocks = |v: f64| {
        let v = whole(v);
        let padded = (v / block).ceil() * block;
        if padded <= MAX_TEXTURE_DIM {
            padded
        } else {
            v
        }
    };
    let width_px = whole_blocks(region.w * zoom).clamp(1.0, MAX_TEXTURE_DIM) as u32;
    let height_px = whole_blocks(region.h * zoom).clamp(1.0, MAX_TEXTURE_DIM) as u32;
    region.w = width_px as f64 / zoom;
    region.h = height_px as f64 / zoom;
    if width_px < 2 || height_px < 2 {
        return None;
    }
    let passes = shadows
        .iter()
        .map(|s| ShadowPass {
            color: s.color,
            offset_px: [(s.offset.0 * zoom) as f32, (s.offset.1 * zoom) as f32],
            radius_px: (s.reach * zoom) as f32,
            show_behind: s.show_behind,
            center: s.center,
        })
        .collect();
    Some(LayerBlurGroup {
        items,
        region_canvas: [region.x, region.y, region.w, region.h],
        extent_canvas: [region.x, region.y, region.w, region.h],
        width_px,
        height_px,
        // The offset and blur, and the halved blur's blocks (each read whole).
        reach_canvas: reach + aa + block / zoom,
        filters: LayerFilters {
            shadow: Some(ShadowFilter {
                inner,
                blend: match kind {
                    CaptureKind::Glow(b) | CaptureKind::InnerGlow(b) => b,
                    _ => CaptureBlend::Normal,
                },
                shadows: passes,
            }),
            ..Default::default()
        },
        cache_key: None,
        coverage: Vec::new(),
        masks: Default::default(),
    })
}

/// The canvas AABB of drawn items (tessellated bodies' geometry bounds, text
/// glyph and op vertices), through each item's model matrix. `None` when
/// nothing measurable draws.
/// Unrounded extrema of the ordinary body/text bounds walk. Keeping extrema until the final Rect avoids
/// reconstructing a maximum with x + width at intermediate retained nodes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ItemExtents { min: (f64, f64), max: (f64, f64) }
impl ItemExtents {
    pub(crate) fn union(self, other: Self) -> Self {
        Self { min: (self.min.0.min(other.min.0), self.min.1.min(other.min.1)),
            max: (self.max.0.max(other.max.0), self.max.1.max(other.max.1)) }
    }
    pub(crate) fn rect(self) -> Rect {
        Rect { x: self.min.0, y: self.min.1, w: self.max.0 - self.min.0, h: self.max.1 - self.min.1 }
    }
}
pub(crate) fn items_canvas_bounds(items: &[crate::seams::DrawItem]) -> Option<Rect> {
    item_extents(items).map(ItemExtents::rect)
}
pub(crate) fn item_extents(items: &[crate::seams::DrawItem]) -> Option<ItemExtents> {
    geometry_extents(items, false)
}

/// Canvas-space geometry used by host read-back. Transform each drawn body vertex
/// before taking its extrema so a rotated nonrectangular path stays tight.
pub fn draw_bounds(item: &crate::seams::DrawItem) -> Option<[f64; 4]> {
    let items = std::slice::from_ref(item);
    if let Some(e) = geometry_extents(items, true) {
        return Some([e.min.0, e.min.1, e.max.0, e.max.1]);
    }
    if matches!(item, crate::seams::DrawItem::Text(_) | crate::seams::DrawItem::Body { .. }) { return None; }
    all_items_canvas_bounds(items).map(|r| [r.x, r.y, r.x + r.w, r.y + r.h])
}

/// Endpoints of each nonempty alpha row. Their four pixel corners suffice for
/// the extrema under any affine placement, including a rotated colour glyph.
fn alpha_row_corners(rgba: &[u8], width: u32, mut point: impl FnMut(f32, f32)) {
    for (y, row) in rgba.chunks_exact(width as usize * 4).enumerate() {
        let mut ink = row.chunks_exact(4).enumerate().filter(|(_, p)| p[3] > 0);
        if let Some((left, _)) = ink.next() {
            let right = ink.next_back().map_or(left, |(x, _)| x) + 1;
            for x in [left, right] {
                for y in [y, y + 1] { point(x as f32, y as f32); }
            }
        }
    }
}

fn geometry_extents(items: &[crate::seams::DrawItem], vertices: bool) -> Option<ItemExtents> {
    use crate::seams::DrawItem;
    let mut min = (f64::INFINITY, f64::INFINITY);
    let mut max = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut add = |m: &[f32; 16], x: f32, y: f32| {
        let (x, y) = (x as f64, y as f64);
        let cx = m[0] as f64 * x + m[4] as f64 * y + m[12] as f64;
        let cy = m[1] as f64 * x + m[5] as f64 * y + m[13] as f64;
        if cx.is_finite() && cy.is_finite() {
            min = (min.0.min(cx), min.1.min(cy));
            max = (max.0.max(cx), max.1.max(cy));
        }
    };
    for item in items {
        match item {
            DrawItem::Body { geometry, instance, .. } if vertices => {
                for p in geometry.positions.chunks_exact(2) {
                    add(&instance.model_matrix, p[0], p[1]);
                }
            }
            DrawItem::Body { instance, .. } => {
                let m = &instance.model_matrix;
                let [x, y, w, h] = instance.bounds;
                for (px, py) in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)] {
                    add(m, px, py);
                }
            }
            DrawItem::Text(batch) if batch.distort.is_some() && batch.vector.is_some() => {
                use moda_text_draw::seams::TextBounds;
                match batch.painted_local_bounds() {
                    TextBounds::Finite([l, t, r, b]) => {
                        for (x, y) in [(l, t), (l, b), (r, t), (r, b)] {
                            add(&batch.model_matrix, x as f32, y as f32);
                        }
                    }
                    TextBounds::CrossesHorizon => {
                        // Glyphs can overflow the node's box, where the homography
                        // can cross the horizon. The shader's canvas clip still holds.
                        // Host read-back asks for finite geometry. An unclipped
                        // horizon has no finite bounds in canvas space.
                        if vertices && batch.page_clip_bounds == crate::seams::INFINITE_CLIP {
                            return None;
                        }
                        let [x, y, w, h] = batch.page_clip_bounds;
                        let identity = [1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.];
                        add(&identity, x, y);
                        add(&identity, x + w, y + h);
                    }
                    TextBounds::Empty => {}
                }
            }
            DrawItem::Text(batch) => {
                let m = &batch.model_matrix;
                for op in &batch.ops {
                    if vertices {
                        if let moda_text::ops::TextOpPipeline::Emoji { emoji } = op.pipeline {
                            let glyph = batch.vector.as_ref().and_then(|v| v.emojis.get(emoji as usize));
                            if let (Some(glyph), Some(engine)) = (glyph, &batch.emoji_engine) {
                                if let Some(img) = glyph.image(engine).filter(|_| op.opacity > 0.) {
                                    // Use the two triangles' collected f32 corners. Round the projected
                                    // texel edge once, as for CPU-collected glyph bounds.
                                    let [x0, y0, x1, y1, x2, y2, x3, y3] = glyph.corners.map(f64::from);
                                    alpha_row_corners(&img.rgba, img.width, |x, y| {
                                        let u = f64::from(x) / f64::from(img.width);
                                        let v = f64::from(y) / f64::from(img.height);
                                        let (x, y) = if u >= v {
                                            ((1. - u) * x0 + (u - v) * x1 + v * x2, (1. - u) * y0 + (u - v) * y1 + v * y2)
                                        } else {
                                            ((1. - v) * x0 + u * x2 + (v - u) * x3, (1. - v) * y0 + u * y2 + (v - u) * y3)
                                        };
                                        add(m, x as f32, y as f32);
                                    });
                                }
                                continue;
                            }
                        }
                    }
                    for v in op.verts.chunks_exact(4) {
                        add(m, v[0], v[1]);
                    }
                }
                // The degraded tier (complex scripts): one raster quad, placed as the executor draws it.
                if let Some(dq) = &batch.degraded {
                    let [x0, y0] = dq.offset;
                    let (x1, y1) = (
                        x0 + dq.raster.logical_w as f32,
                        y0 + dq.raster.logical_h as f32,
                    );
                    if vertices {
                        // The fallback draw has already rasterized its glyphs. Its quad
                        // includes transparent padding, which is outside the ink.
                        let raster = &dq.raster;
                        let sx = raster.logical_w as f32 / raster.width as f32;
                        let sy = raster.logical_h as f32 / raster.height as f32;
                        alpha_row_corners(&raster.rgba, raster.width, |x, y| add(m, x0 + x*sx, y0 + y*sy));
                        continue;
                    }
                    for (px, py) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
                        add(m, px, py);
                    }
                }
                // The vector lane: tessellated triangles, and analytic glyph
                // boxes placed as `kineticGlyphLocal` places them.
                let pad = batch.quad_padding();
                for op in batch.vector.iter().flat_map(|v| &v.ops) {
                    for v in op.tris.chunks_exact(2) {
                        add(m, v[0], v[1]);
                    }
                    for r in op.records.iter() {
                        let [x0, y0, x1, y1] = crate::seams::TextQuadBatch::padded_box(r.em_box, pad);
                        let s = r.font_size_px;
                        let center = (
                            r.pen_x + (x0 + x1) * 0.5 * s,
                            r.baseline_y + (y0 + y1) * 0.5 * s,
                        );
                        let [a, c, b, d] = r.glyph_lin;
                        if vertices {
                            if r.glyph_opacity <= 0. { continue; }
                            // Measure the curves the analytic shader consumes, after its
                            // glyph and node transforms. em_box includes shader AA padding.
                            let place = |ex: f32, ey: f32| {
                                let dx = r.pen_x + ex * s - center.0;
                                let dy = r.baseline_y + ey * s - center.1;
                                let x = center.0 + a * dx + c * dy + r.glyph_off[0];
                                let y = center.1 + b * dx + d * dy + r.glyph_off[1];
                                (f64::from(m[0]*x + m[4]*y + m[12]), f64::from(m[1]*x + m[5]*y + m[13]))
                            };
                            let radius = op.stroke_px * 0.5;
                            let rx = radius * (m[0]*a+m[4]*b).hypot(m[0]*c+m[4]*d);
                            let ry = radius * (m[1]*a+m[5]*b).hypot(m[1]*c+m[5]*d);
                            let first = r.first_curve as usize * 6;
                            let end = first + r.curve_count as usize * 6;
                            let vector = batch.vector.as_ref().unwrap();
                            use crate::tess::moda_engine_path_types::PathCmd;
                            for q in vector.curve_points[first..end].chunks_exact(6) {
                                let (x0,y0) = place(q[0],q[1]);
                                let (x,y) = place(q[4],q[5]);
                                let cmds = [PathCmd::Move{x:x0,y:y0}, PathCmd::Quadratic{x,y,cp:place(q[2],q[3])}];
                                if let Some([l,t,r,b]) = crate::tess::path_bounds(&cmds) {
                                    const ID: [f32;16] = [1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.];
                                    add(&ID,l as f32-rx,t as f32-ry);
                                    add(&ID,r as f32+rx,b as f32+ry);
                                }
                            }
                            continue;
                        }
                        for [x, y] in r.quad_corners([x0, y0, x1, y1]) {
                            add(m, x, y);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    (min.0 <= max.0 && min.1 <= max.1).then_some(ItemExtents { min, max })
}

/// The ordered rectangle union used by capture and capture-local isolation bounds.
/// Keep its arithmetic order: repeated x+w reconstruction is observable at floating-point boundaries.
pub(crate) fn union_canvas_bounds(a: Option<Rect>, b: Option<Rect>) -> Option<Rect> {
    match (a, b) {
        (Some(a), Some(b)) => {
            let (x0, y0) = (a.x.min(b.x), a.y.min(b.y));
            let (x1, y1) = ((a.x + a.w).max(b.x + b.w), (a.y + a.h).max(b.y + b.h));
            Some(Rect { x: x0, y: y0, w: x1 - x0, h: y1 - y0 })
        }
        (a, b) => a.or(b),
    }
}

/// Quad emitted by the analytic shape and shadow vertex shaders. Coordinates are
/// local endpoints; a drop shadow applies its canvas-space offset after the affine.
pub(crate) struct PaintedQuad<'a> {
    pub model: &'a [f32; 16],
    pub edges: [f32; 4],
    pub offset: [f32; 2],
}

impl PaintedQuad<'_> {
    pub fn corners(&self) -> [f32; 8] {
        let [l, t, r, b] = self.edges;
        [l, t, r, t, l, b, r, b]
    }

    fn canvas_bounds(&self) -> Rect {
        let m = self.model;
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for xy in self.corners().chunks_exact(2) {
            for axis in 0..2 {
                let v = f64::from(m[axis]) * f64::from(xy[0])
                    + f64::from(m[4 + axis]) * f64::from(xy[1])
                    + f64::from(m[12 + axis]) + f64::from(self.offset[axis]);
                lo[axis] = lo[axis].min(v);
                hi[axis] = hi[axis].max(v);
            }
        }
        Rect { x: lo[0], y: lo[1], w: hi[0] - lo[0], h: hi[1] - lo[1] }
    }
}

/// The capture, native Gaussian tiles and fade scheduler must agree on the quad
/// before projecting it to their respective pixel grids.
pub(crate) fn painted_quad(item: &crate::seams::DrawItem) -> Option<PaintedQuad<'_>> {
    use crate::seams::DrawItem;
    let (model, dimensions, stroke) = match item {
        DrawItem::Shadow { instance, .. } => {
            let [w, h, blur, _] = instance.shape_data;
            let inner = instance.inner_params[0] > 0.5;
            let spread = if inner { 0. } else { instance.inner_params[1].max(0.) };
            // path-shadow.wgsl: three sigmas per side, positive spread, and one AA pixel.
            let expansion = blur * 3. + 2. * spread + 2.;
            let (width, height) = (w + expansion, h + expansion);
            let edges = if instance.corner_radii[3] < 0. {
                [-width * 0.5, -height * 0.5, width * 0.5, height * 0.5]
            } else {
                let pad = expansion * 0.5;
                [-pad, -pad, width - pad, height - pad]
            };
            let offset = if inner { [0.; 2] } else { [instance.shadow_offset[0], instance.shadow_offset[1]] };
            return Some(PaintedQuad { model: &instance.model_matrix, edges, offset });
        }
        DrawItem::SolidShapeSdf { instance, .. } => (&instance.model_matrix, instance.dimensions, instance.stroke_params),
        DrawItem::GradientShapeSdf { instance, .. } => (&instance.model_matrix, instance.dimensions, instance.stroke_params),
        _ => return None,
    };
    // The full stroke width covers every alignment, including an outside stroke.
    let (x, y) = (dimensions[0] * 0.5, dimensions[1] * 0.5);
    let grow = stroke[0].abs();
    let (rx, ry) = (x.abs() + grow, y.abs() + grow);
    Some(PaintedQuad { model, edges: [x - rx, y - ry, x + rx, y + ry], offset: [0.; 2] })
}

/// Canvas extent of a painted subtree. Preserve the ordered reduction of the
/// ordinary body/text bounds followed by analytic quads and nested captures.
pub(crate) fn all_items_canvas_bounds(items: &[crate::seams::DrawItem]) -> Option<Rect> {
    let mut acc = items_canvas_bounds(items);
    for item in items {
        let r = if let Some(quad) = painted_quad(item) {
            quad.canvas_bounds()
        } else if let crate::seams::DrawItem::LayerBlurComposite { group, .. } = item {
            // Clipping storage to its readers leaves the nested picture's extent intact.
            let [x, y, w, h] = group.extent_canvas;
            Rect { x, y, w, h }
        } else {
            continue;
        };
        if r.finite() { acc = union_canvas_bounds(acc, Some(r)); }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(v: moda_scene::value::Value) -> RawNode {
        RawNode::from_value(v).unwrap()
    }

    #[test]
    fn capture_lane_takes_every_shape_without_an_sdf_silhouette() {
        let fx = moda_scene::value::json!([
            {"id": "d", "type": "drop-shadow", "color": "#000", "blur": 8},
            {"id": "i", "type": "inner-shadow", "color": "#000", "blur": 8}
        ]);
        for t in ["star", "polygon", "path", "line"] {
            let n = raw(moda_scene::value::json!({"id": "a", "type": t, "effects": fx}));
            assert_eq!(capture_drop_shadows(&n).len(), 1, "{t} drop");
            assert_eq!(capture_inner_shadows(&n).len(), 1, "{t} inner");
        }
        // The SDF lane keeps rectangles and full ellipses.
        let rect = raw(
            moda_scene::value::json!({"id": "a", "type": "rectangle", "width": 10, "height": 10, "effects": fx}),
        );
        assert!(!has_capture_shadow(&rect));
        // A wedge has no silhouette: the capture lane draws it.
        let wedge = raw(
            moda_scene::value::json!({"id": "a", "type": "ellipse", "width": 10, "height": 10,
            "sweepAngle": 180, "effects": fx}),
        );
        assert!(has_capture_shadow(&wedge));
        // Rich text: inner shadows only (its drop shadows are the text lane's).
        let text = raw(moda_scene::value::json!({"id": "a", "type": "richtext", "effects": fx}));
        assert!(capture_drop_shadows(&text).is_empty());
        assert_eq!(capture_inner_shadows(&text).len(), 1);
        // Groups cast their subtree's shadows; so does a frame without a fill
        // that has children. A filled frame keeps the SDF lane.
        let group =
            raw(moda_scene::value::json!({"id": "a", "type": "group", "children": [], "effects": fx}));
        assert!(has_capture_shadow(&group) && is_subtree_capture_node(&group));
        let frame = |fills: moda_scene::value::Value| {
            raw(
                moda_scene::value::json!({"id": "a", "type": "container", "width": 10, "height": 10,
                "children": ["c"], "fills": fills, "effects": fx}),
            )
        };
        assert!(is_subtree_capture_node(&frame(moda_scene::value::json!([]))));
        assert!(has_capture_shadow(&frame(moda_scene::value::json!([]))));
        let filled =
            frame(moda_scene::value::json!([{"id": "f", "paint": {"type": "solid", "color": "#fff"}}]));
        assert!(!is_subtree_capture_node(&filled) && !has_capture_shadow(&filled));
    }

    #[test]
    fn an_image_filled_shape_shadows_and_glows_from_its_pixels() {
        // A cut-out PNG (`<img>`, a rectangle with an image fill) casts the
        // cut-out's shadow and glow, not its box's: the capture lane draws it,
        // and its silhouette keeps the image (its alpha) at full opacity.
        let image = moda_scene::value::json!([{"id": "f", "opacity": 0.5, "blendMode": "multiply",
            "paint": {"type": "image", "src": "cutout.png"}}]);
        let fx = moda_scene::value::json!([
            {"id": "d", "type": "drop-shadow", "color": "#000", "blur": 8},
            {"id": "g", "type": "glow", "color": "#ffd36b", "radiusPx": 20}
        ]);
        for t in ["rectangle", "ellipse"] {
            let n = raw(moda_scene::value::json!({"id": "a", "type": t, "width": 10, "height": 10,
                "fills": image, "effects": fx}));
            assert!(fills_are_images(&n), "{t}");
            assert!(!crate::effects::shadow::has_sdf_silhouette(&n), "{t}");
            assert_eq!(capture_drop_shadows(&n).len(), 1, "{t} drop");
            assert_eq!(capture_glows(&n).len(), 1, "{t} glow");
            let fills = moda_scene::paint::node_fills(&silhouette_node(&n));
            assert_eq!(fills.len(), 1);
            assert!(matches!(fills[0].paint, moda_scene::paint::Paint::Image(_)));
            assert_eq!(fills[0].opacity, None);
            assert_eq!(fills[0].blend_mode, None);
        }
        // A solid fill under the image fills the box: the box is the shape.
        let mixed = raw(moda_scene::value::json!({"id": "a", "type": "rectangle", "width": 10, "height": 10,
            "fills": [{"id": "s", "paint": {"type": "solid", "color": "#fff"}},
                      {"id": "f", "paint": {"type": "image", "src": "cutout.png"}}],
            "effects": fx}));
        assert!(!fills_are_images(&mixed));
        assert!(crate::effects::shadow::has_sdf_silhouette(&mixed));
    }

    #[test]
    fn spread_only_drop_shadow_draws_nothing_on_a_path() {
        let n = raw(moda_scene::value::json!({"id": "a", "type": "path", "effects": [
            {"id": "d", "type": "drop-shadow", "color": "#000", "blur": 0, "spread": 4}]}));
        assert!(capture_drop_shadows(&n).is_empty());
        // With a blur it draws, spread ignored (the capture has no spread input).
        let n = raw(moda_scene::value::json!({"id": "a", "type": "path", "effects": [
            {"id": "d", "type": "drop-shadow", "color": "#000", "blur": 6, "spread": -4}]}));
        assert_eq!(capture_drop_shadows(&n).len(), 1);
    }

    #[test]
    fn spread_only_inner_shadow_draws_nothing_on_a_path() {
        let n = raw(moda_scene::value::json!({"id": "a", "type": "path", "effects": [
            {"id": "i", "type": "inner-shadow", "color": "#000", "blur": 0, "spread": 4}]}));
        assert!(capture_inner_shadows(&n).is_empty());
    }

    #[test]
    fn silhouette_is_opaque_black_fill_and_widest_stroke() {
        let n = raw(moda_scene::value::json!({
            "id": "a", "type": "star", "opacity": 0.4, "blendMode": "multiply",
            "fills": [
                {"id": "f0", "paint": {"type": "solid", "color": "#ff000080"}, "opacity": 0.5},
                {"id": "f1", "paint": {"type": "linear", "start": {"x": 0, "y": 0}, "end": {"x": 1, "y": 0},
                    "stops": [{"offset": 0, "color": "#00000000"}, {"offset": 1, "color": "#fff"}]}}
            ],
            "strokes": [
                {"id": "s0", "paint": {"type": "solid", "color": "#0f0"}, "width": 2},
                {"id": "s1", "paint": {"type": "solid", "color": "#00f"}, "width": 6, "align": "outside", "opacity": 0.3}
            ],
            "effects": [
                {"id": "d", "type": "drop-shadow", "color": "#000", "blur": 8},
            ]
        }));
        let sil = silhouette_node(&n);
        assert_eq!(sil.f64("opacity"), Some(1.0));
        assert!(sil.json_field("blendMode").is_none());
        let fills = moda_scene::paint::node_fills(&sil);
        assert_eq!(fills.len(), 1);
        assert_eq!(
            moda_scene::paint::solid_color(&fills[0].paint),
            Some("#000000")
        );
        let strokes = moda_scene::paint::node_strokes(&sil);
        assert_eq!(strokes.len(), 1);
        assert_eq!(strokes[0].width, 6.0);
        assert_eq!(strokes[0].align.as_deref(), Some("outside"));
        assert_eq!(strokes[0].opacity, None);
        assert!(sil.json_field("effects").is_none());
    }

}
