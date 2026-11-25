//! Plan side of the layer filters (layer blur, noise, texture, feather): node
//! predicates, page/canvas-root predicates (residency capture views +
//! plan-build gating), the capture-region / capture-scale math, and each
//! filter's parameters in capture pixels. GPU-free.

use moda_scene::types::RawNode;
use moda_scene::Scene;

use crate::backdrop::plan::{node_canvas_aabb, node_local_rect, PlanCtx, Rect};
use crate::seams::{BlurFilter, FeatherFilter, LayerFilters, NoiseFilter, ShadowPass, TextureFilter};

/// Maximum texture-effect sampling radius in target pixels.
const MAX_TEXTURE_RADIUS_PX: f64 = 300.0;
/// How close to a whole target pixel a capture edge must be to count as on
/// it (the f32 view-projection's rounding).
const SNAP_PX: f64 = 1e-3;
/// wgpu texture dimension ceiling (mirrors backdrop's `MAX_TEXTURE_DIM`).
const MAX_TEXTURE_DIM: f64 = 8192.0;
/// Capture area budget: caps the worst-case transient GPU memory per blurred
/// node (capture + temp + result ≈ 3 × area × 4 B ≈ 200 MB at this cap) since
/// the MVP capture region is not viewport-intersected. Exceeding it reduces
/// the capture scale (softer blur, never a failure).
const MAX_CAPTURE_AREA_PX: f64 = 4096.0 * 4096.0;
/// How much a capture's own rounding out can add to a cut region, per side, in capture px: whole texels, the
/// pixel-grid snap, and the blocks a wide filter shadow's blur halves by (`readable_window`).
const CUT_ROUNDING_PX: f64 = 64.0;

/// Node kinds that can carry a layer filter — mirrors
/// `LAYER_BLUR_CAPABLE_NODE_TYPES` (`core/scene/nodes.ts`). Text captures its
/// glyphs: its capture region covers what they draw
/// ([`compute_layer_blur_params_over`]).
fn is_layer_blur_capable(node_type: &str) -> bool {
    matches!(
        node_type,
        "rectangle" | "container" | "ellipse" | "star" | "polygon" | "path" | "group" | "richtext"
    )
}

/// `hasActiveLayerFilter` (`core/scene/nodes.ts`): a capable node with an
/// active layer blur (uniform or progressive), noise, texture or feather
/// (`effect::has_active_layer_filter`). A zero radius / density is a valid
/// stored-but-inert state; a hidden entry (`visible: false`) renders as if
/// absent.
pub fn has_active_layer_filter(node: &RawNode) -> bool {
    is_layer_blur_capable(node.node_type()) && moda_scene::effect::has_active_layer_filter(node)
}

/// Whether the node renders through a layer capture: an active layer filter,
/// or a shadow of a shape the SDF shadow lane can't draw (`layer_blur::shadow`).
pub fn needs_layer_capture(node: &RawNode) -> bool {
    has_active_layer_filter(node)
        || crate::layer_blur::shadow::has_capture_shadow(node)
        // A fill's or stroke's own drawing effects (E8.2), on a shape or on a group's appearance.
        || (crate::paint::layers::has_entry_effects(node)
            && moda_scene::appearance::has_entry_drawing_effects(node))
}

/// A text the clip-mask groups above can't clip in its own draw (#2148): the
/// text pipelines bind no R8 mask, so only the effective page clip clips text,
/// and it holds a mask's box alone (`effects::has_non_box_clip_mask`). Such a
/// text renders through an identity capture whose composite samples the
/// group's mask ([`compute_mask_capture_params`]), as a filtered node's does.
pub fn needs_mask_capture(scene: &Scene, node: &RawNode) -> bool {
    node.node_type() == "richtext"
        && crate::paint::node_distort(node).is_none()
        && crate::effects::has_non_box_clip_mask(scene, node.id())
}

/// [`needs_layer_capture`] or [`needs_mask_capture`]: whether the node renders
/// through a layer capture in `scene`.
pub fn needs_layer_capture_in(scene: &Scene, node: &RawNode) -> bool {
    needs_layer_capture(node) || needs_mask_capture(scene, node)
}

/// Whether a node's layer filters are of the kind a cut is made for ([`readable_window`]): a uniform blur and
/// filter shadows, whose result at a pixel depends only on the picture within a known distance of it, sized by
/// the capture's scale alone. A progressive blur, noise, a texture and a feather are placed by the node's own
/// box; an agent shader reads its source at any coordinate (its `pad` is how far it may draw, not how far it
/// reads); a motion capture is cut to its own window.
fn cuts_cleanly(node: &RawNode) -> bool {
    use moda_scene::effect;
    effect::layer_blur(node).filter(|b| b.is_active()).is_none_or(|b| b.progressive.is_none())
        && effect::texture(node).is_none()
        && effect::noise_effects(node).is_empty()
        && effect::feather(node).is_none()

}

/// How far past a pixel it writes a node's uniform blur and filter shadows read, in canvas units
/// ([`cuts_cleanly`] nodes only). The kernels are measured as `params_for_bounds` sizes them, with the same
/// projected scales (`node_projected_scales`): a blur by the larger principal stretch, a filter shadow's blur by
/// the smaller projected axis. Those are radii in capture pixels, and a capture at full scale has
/// `canvas_target_scale` pixels per canvas unit on both axes, so a radius is that many pixels over that scale
/// in canvas units, the same along x and y; a capture reduced for a large blur scales its kernel with it. (A
/// capture reduced to fit the size limits does not: a kernel has a smallest size in texels. `readable_window`
/// makes no cut where one could be.) A filter shadow's offset is canvas-space already. A kernel counts twice
/// its nominal 3σ reach: the pass ladder's taps go past 3σ with small weights, and past 6σ a kernel's weight is
/// under 2^-24 of the whole, far below what an 8- or 16-bit target can show. Two capture pixels are added for
/// antialiasing and resampling. `None` when a figure is not a number.
fn filter_read_reach(scene: &Scene, node: &RawNode, ctx: &PlanCtx) -> Option<f64> {
    use moda_scene::effect;
    let canvas_scale = ctx.canvas_target_scale().max(0.0001);
    // The projection is affine wherever a cut is made (`target_canvas_rect`), so where it is measured does
    // not matter.
    let at = Rect { x: 0.0, y: 0.0, w: 0.0, h: 0.0 };
    let scales = node_projected_scales(scene, node, ctx, &at)
        .unwrap_or(ProjectedScales { major: [1.0, 0.0], axis_min: canvas_scale, min: canvas_scale, max: canvas_scale });
    let kernel = |radius: f64, scale: f64| 2.0 * radius.abs() * effect::BLUR_REACH_PER_RADIUS * scale;
    let mut px: f64 = 0.0;
    if let Some(b) = effect::layer_blur(node).filter(|b| b.is_active()) {
        px += kernel(b.radius, scales.max);
    }
    let shadows = effect::filter_shadows(node);
    px += shadows.iter().map(|s| kernel(s.blur, scales.axis_min)).fold(0.0, f64::max);
    let offset = shadows.iter().map(|s| s.offset.x.abs().max(s.offset.y.abs())).fold(0.0, f64::max);
    let reach = (px + 2.0) / canvas_scale + offset;
    reach.is_finite().then_some(reach)
}

/// The part of the canvas a capture of `node` can be read from, when a cut to it is sound: the plan's target,
/// grown by the read reach of every layer filter between the node's picture and the target, its ancestors' and
/// (unless the capture is an identity one) its own. Nothing outside it reaches a pixel of the target, through
/// any chain of captures: each ancestor reads this picture only inside the target grown by the reaches above
/// and including its own, which this window contains. The reach is the same canvas distance along x and y; it
/// is rounded out to whole target pixels on each axis separately, so a cut edge stays on the target's pixel
/// grid whatever the two axis scales.
///
/// `None` (no cut) unless all of these hold:
/// - the target has an axis-aligned canvas rect (`target_canvas_rect`);
/// - every filter in the chain cuts cleanly ([`cuts_cleanly`]): the reach above is then all any of them reads;
/// - the window itself fits the capture limits at full scale. Then this capture, cut, is not reduced to fit
///   them, and neither is any ancestor's (an oversize ancestor is cut to its own window, which is inside this
///   one, or is not captured at all when its region misses that window: `params_for_bounds`; a text behind a
///   mask is the exception there, and is a leaf), so every kernel in the chain runs at the size the reach
///   measures. Where the window does not fit (a
///   target near the limits), captures are reduced as before and a reduced kernel's read distance is not this
///   one, so nothing is cut.
fn readable_window(scene: &Scene, node: &RawNode, ctx: &PlanCtx, identity: bool) -> Option<Rect> {
    let parent = |id: &str| {
        scene
            .parent_map()
            .get(id)
            .or_else(|| scene.transient_parent_of().get(id))
            .map(String::as_str)
    };
    let (view, sx, sy) = target_canvas_rect(ctx)?;
    // Canvas units, summed over the chain.
    let mut reach = if identity { 0.0 } else { filter_read_reach(scene, node, ctx)? };
    let mut seen = moda_scene::scene::SeenIds::default();
    let mut cur = parent(node.id());
    while let Some(pid) = cur {
        if !seen.insert(pid) {
            break;
        }
        if let Some(p) = scene.effective_node(pid).filter(|p| has_active_layer_filter(p)) {
            if !cuts_cleanly(p) {
                return None;
            }
            reach += filter_read_reach(scene, p, ctx)?;
        }
        cur = parent(pid);
    }
    // Whole target pixels on each axis: the two scales differ when a target's sides are rounded apart.
    let (rx, ry) = ((reach * sx).ceil() / sx, (reach * sy).ceil() / sy);
    let window = Rect { x: view.x - rx, y: view.y - ry, w: view.w + 2.0 * rx, h: view.h + 2.0 * ry };
    // In capture pixels at full scale, with room for a capture's own rounding out (to whole texels, and to the
    // blocks a wide shadow's blur halves by).
    let canvas_scale = ctx.canvas_target_scale().max(0.0001);
    let (w_px, h_px) = (window.w * canvas_scale + CUT_ROUNDING_PX, window.h * canvas_scale + CUT_ROUNDING_PX);
    let fits = w_px.max(h_px) <= MAX_TEXTURE_DIM && w_px * h_px <= MAX_CAPTURE_AREA_PX;
    (window.finite() && fits).then_some(window)
}

/// Keep the source pixels that can contribute to the output and the output pixels
/// that can be read. The missing transparent Gaussian surround can be supplied
/// after reduction if both this raster and the padded filter grid fit.
fn uniform_capture_window(region: Rect, body: Rect, window: Rect, scale: f64, pad: f64, reach: f64) -> Option<(Rect, u32)> {
    let expand = |r: Rect, d: f64| Rect {
        x: r.x - d, y: r.y - d, w: r.w + 2.0 * d, h: r.h + 2.0 * d,
    };
    let output = region.intersect(&window)?;
    let source = expand(body, (pad - reach / scale).max(1.0));
    let input = source.intersect(&expand(window, (2.0 * reach + 2.0) / scale))?;
    let x = output.x.min(input.x);
    let y = output.y.min(input.y);
    let compact = Rect { x, y, w: (output.x + output.w).max(input.x + input.w) - x,
        h: (output.y + output.h).max(input.y + input.h) - y };
    let block = crate::layer_blur::exec::uniform_blur_block(reach as f32) as f64;
    // Allow one pixel for snapping, then round to complete pyramid blocks.
    let size = [compact.w, compact.h].map(|v| ((v * scale + 1.0) / block).ceil() * block);
    let fits = |[w, h]: [f64; 2]| w.max(h) <= MAX_TEXTURE_DIM && w * h <= MAX_CAPTURE_AREA_PX;
    let coarse_reach = reach / block;
    if !fits(size) || coarse_reach > MAX_TEXTURE_DIM {
        return None;
    }
    let passes = crate::backdrop::kernel::backdrop_blur_passes(coarse_reach);
    let border = 1.0 + passes.iter().map(|p| p.offsets.last().copied().unwrap_or(0.0).ceil() as f64).sum::<f64>();
    fits(size.map(|v| v / block + 2.0 * border)).then_some((compact, border as u32))
}

/// The canvas rect the plan's target shows, with the target pixels per canvas unit on each axis, read from the
/// projection itself: the two axes are inverted separately (a target's sides are rounded independently, so its
/// scales can differ, and a page can sit far from the origin). `None` when the view is rotated, sheared or
/// degenerate: there is then no canvas rect to cut to.
fn target_canvas_rect(ctx: &PlanCtx) -> Option<(Rect, f64, f64)> {
    let (x0, y0) = ctx.project_point(0.0, 0.0)?;
    let (x1, y1) = ctx.project_point(1.0, 0.0)?;
    let (x2, y2) = ctx.project_point(0.0, 1.0)?;
    let (sx, sy) = (x1 - x0, y2 - y0);
    let aligned = (y1 - y0).abs() <= 1e-9 * sx.abs().max(1.0) && (x2 - x0).abs() <= 1e-9 * sy.abs().max(1.0);
    if !aligned || !(sx.abs() > 1e-12 && sy.abs() > 1e-12) {
        return None;
    }
    let (ax, bx) = ((0.0 - x0) / sx, (ctx.width_px as f64 - x0) / sx);
    let (ay, by) = ((0.0 - y0) / sy, (ctx.height_px as f64 - y0) / sy);
    let r = Rect { x: ax.min(bx), y: ay.min(by), w: (bx - ax).abs(), h: (by - ay).abs() };
    r.finite().then_some((r, sx.abs(), sy.abs()))
}

/// Twin of `backdrop::page_has_active_backdrop` for the layer captures (layer
/// filters and capture shadows, [`needs_layer_capture`]). Drives the plan's
/// capture context, the residency camera lane's capture view and the
/// incremental-page-slot exclusion — deliberately NOT the frame.rs
/// sample_count/transparent forcing (captures keep MSAA and transparent
/// exports).
pub fn page_has_active_layer_filter(scene: &Scene, page_id: &str) -> bool {
    scene
        .flattened_z_order_for_page(page_id)
        .iter()
        .filter_map(|id| scene.effective_node(id))
        .any(|n| needs_layer_capture_in(scene, n))
}

/// Whether the page has an active layer FILTER (blur, noise, texture) — what a
/// tiled region render refuses. Capture shadows tile fine: they read only
/// their own capture.
pub fn page_has_layer_filter_effect(scene: &Scene, page_id: &str) -> bool {
    scene
        .flattened_z_order_for_page(page_id)
        .iter()
        .filter_map(|id| scene.effective_node(id))
        .any(has_active_layer_filter)
}

/// Twin of `backdrop::canvas_root_has_active_backdrop` (freeform floating
/// nodes).
pub fn canvas_root_has_active_layer_filter(scene: &Scene) -> bool {
    scene
        .flattened_z_order_for_canvas()
        .iter()
        .filter_map(|id| scene.effective_node(id))
        .any(|n| needs_layer_capture_in(scene, n))
}

/// Node-level eligibility for the MVP isolation path. Mirrors the TS export
/// gate `isNodeLayerFilterRenderable` (`utils/export/layerBlurEligibility.ts`)
/// — keep the two in lockstep or export gating trips on nodes the engine
/// renders plainly (or vice versa).
///
/// Unlike backdrop blur there are NO rotation/shear/partial-ellipse/opacity
/// rejections: the isolation capture renders the node's own draw items under
/// any affine, and opacity is baked into the captured items. The failure mode
/// of every rejection is "renders normally without blur".
fn is_node_layer_blur_eligible(node: &RawNode) -> bool {
    if !has_active_layer_filter(node) {
        return false;
    }
    // A container with children filters its whole subtree instead (`frame.rs`
    // `divert_subtree_filter`): its children live in the display list AFTER
    // its own surface, so capturing just the surface here would blur the
    // frame but not its content.
    if node
        .array("children")
        .map(|c| !c.is_empty())
        .unwrap_or(false)
    {
        return false;
    }
    // A non-normal blend mode rides a blend split around the composite: the
    // capture draws the body in plain source-over, and the split blends the
    // filtered result over the scene (`backdrop::blend::composite_blend_split`,
    // frame.rs). A plan without a split context leaves such a node unfiltered.
    // A node under a clip-mask group is captured unmasked and its composite
    // samples the group's mask (`DrawItem::LayerBlurComposite::mask`). A
    // clipping CONTAINER (every Design board clips) is checked against the
    // padded capture region in `compute_layer_blur_params`: when the region
    // lies inside the container's clip, the clip changes nothing and the
    // composite needs no mask.
    // Distort-warped nodes decline (mirrors backdrop's distort posture):
    // `node_canvas_aabb` measures the UNWARPED geometry, so a corner-pin warp
    // could extend past the padded capture and come back hard-clipped-then-
    // blurred. Mirrored in the TS gate (`isNodeLayerFilterRenderable`).
    if crate::paint::node_distort(node).is_some() {
        return false;
    }
    true
}

/// Whether the composite quad can be clipped like the node's body by every
/// clipping container (`clipContent`) above it: an axis-aligned container's
/// box is the composite's effective clip, and the innermost rounded one's
/// corners its rounded clip (`effects::clip_rounded_rect`). A rotated or
/// sheared container declines (its AABB is not its clip), and an outer
/// rounded one (not the innermost) must not have its corners reached by the
/// capture `region` (conservatively: its box inset by its largest corner
/// radius); either way the node renders un-filtered, as before.
pub(crate) fn clipping_containers_clip_the_composite(
    scene: &Scene,
    node_id: &str,
    region: &Rect,
) -> bool {
    const EPS: f64 = 1e-6;
    let rounded = crate::effects::clip_rounded_rect(scene, node_id);
    for pid in scene.transform_parent_chain(node_id) {
        let Some(p) = scene.effective_node(&pid) else {
            continue;
        };
        if p.node_type() != "container" || !p.truthy("clipContent") {
            continue;
        }
        let Some(a) = scene.node_canvas_affine(&pid) else {
            return false;
        };
        let Some(clip) = node_canvas_aabb(scene, p) else {
            return false;
        };
        if a.b.abs() > EPS || a.c.abs() > EPS {
            return false;
        }
        let radius = moda_scene::shape_outline::expand_corner_radii(p.json_field("cornerRadius"))
            .into_iter()
            .fold(0.0_f64, f64::max)
            .max(0.0)
            * a.a.abs().max(a.d.abs());
        if radius <= EPS {
            continue;
        }
        let near = |u: f32, v: f64| (u as f64 - v).abs() <= 1e-3;
        if rounded.is_some_and(|r| {
            near(r.rect[0], clip.x)
                && near(r.rect[1], clip.y)
                && near(r.rect[2], clip.w)
                && near(r.rect[3], clip.h)
        }) {
            continue;
        }
        let inside = region.x >= clip.x + radius - EPS
            && region.y >= clip.y + radius - EPS
            && region.x + region.w <= clip.x + clip.w - radius + EPS
            && region.y + region.h <= clip.y + clip.h - radius + EPS;
        if !inside {
            return false;
        }
    }
    true
}

/// Capture parameters for one layer-blurred node, computed at plan time (the
/// projection — and therefore the region — is camera-dependent, which is why
/// the residency camera lane plans them under a zoom-bucketed capture view,
/// `frame::CaptureView`).
#[derive(Debug, Clone, PartialEq)]
pub struct LayerBlurParams {
    /// Padded capture rect in canvas-absolute units [x, y, w, h].
    pub region_canvas: [f64; 4],
    /// What the picture occupies (`LayerBlurGroup::extent_canvas`): the region, or the region before a cut.
    pub extent_canvas: [f64; 4],
    /// Offscreen capture size in physical px.
    pub width_px: u32,
    pub height_px: u32,
    /// How far a result texel reads the capture, in canvas units (the region's padding).
    pub reach_canvas: f64,
    /// What runs over the capture, in capture pixels.
    pub filters: LayerFilters,
}

/// Region + scale math for an eligible node; `None` = render unfiltered.
/// Callers must preserve `diagnostics`, including refusals for unsupported blur reaches.
///
/// - Each filter is measured in TARGET px first (canvas px × the node's
///   effective scale). A filter too small
///   to see is dropped (a blur ≤ 0.5 px, a texture moving ≤ 0.5 px); with no
///   filter left the node renders plainly.
/// - The padded region is the node's canvas AABB expanded by the blur bleed
///   (3σ = the target-px radius, back-converted to canvas units), the
///   texture's displacement, the drop-shadow reach (`offset + 2×blur`, the
///   shadow quad's own vertex expansion) and the stroke width.
/// - Rasterize at target resolution before Gaussian reduction, subject to device limits.
pub fn compute_layer_blur_params(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
    diagnostics: &mut Vec<String>,
) -> Option<LayerBlurParams> {
    compute_layer_blur_params_over(scene, node, ctx, || None, diagnostics)
}

/// Include the leaf's drawn canvas bounds in its capture, once eligible. A stroke or a glyph
/// can extend beyond the geometry box, and a flat stroked path has area only in its draw.
pub(crate) fn compute_layer_blur_params_over(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
    drawn: impl FnOnce() -> Option<Rect>,
    diagnostics: &mut Vec<String>,
) -> Option<LayerBlurParams> {
    if !is_node_layer_blur_eligible(node) {
        return None;
    }

    // A shape without a box (a horizontal or vertical line as a path: no width or no height) still draws its
    // stroke: its drawn items are then its bounds.
    let use_drawn = node.node_type() == "richtext" || moda_scene::paint::strokes_visibly(node)
        || !moda_scene::effect::filter_shadows(node).is_empty();
    let drawn = use_drawn.then(drawn).flatten().filter(Rect::finite);
    let geometry = node_canvas_aabb(scene, node);
    let node_bounds = super::shadow::union_canvas_bounds(geometry, drawn)?;
    if !node_bounds.finite() {
        return None;
    }
    let source = if let Some(drawn) = drawn {
        CaptureSource::DrawnLeaf { geometry: geometry.unwrap_or(drawn), drawn }
    } else {
        CaptureSource::Geometry
    };
    params_for_bounds(scene, node, ctx, node_bounds, source, false, diagnostics)
}

/// Layer-filter parameters for a SUBTREE capture (a group's, or a frame's
/// with children): `bounds` is the canvas AABB of the subtree's drawn items
/// (their strokes and shadows included), so only the filters' own reach pads
/// it. The composite takes the root's effective clip, so a clipping frame
/// above doesn't decline it. `None` = the subtree renders unfiltered.
pub(crate) fn compute_subtree_layer_blur_params(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
    bounds: Rect,
    diagnostics: &mut Vec<String>,
) -> Option<LayerBlurParams> {
    // Under a clip-mask group the composite takes the group's mask.
    if !has_active_layer_filter(node) || !bounds.finite() { return None; }
    params_for_bounds(scene, node, ctx, bounds, CaptureSource::Subtree, false, diagnostics)
}

/// Capture parameters for a text under a mask it can't honour
/// ([`needs_mask_capture`]): no filter, the capture is the result, over
/// `bounds` (the canvas AABB of the text's drawn items). `None` = the text
/// renders plainly, clipped by the mask's box only.
pub(crate) fn compute_mask_capture_params(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
    bounds: Rect,
    diagnostics: &mut Vec<String>,
) -> Option<LayerBlurParams> {
    if !needs_mask_capture(scene, node) || !bounds.finite() {
        return None;
    }
    params_for_bounds(scene, node, ctx, bounds, CaptureSource::Subtree, true, diagnostics)
}

#[derive(Clone, Copy)]
enum CaptureSource {
    /// No draw bounds available. Estimate stroke and shadow reach from the node.
    Geometry,
    /// The draw already includes strokes and shadows. Grain stays on the geometry.
    DrawnLeaf { geometry: Rect, drawn: Rect },
    Subtree,
}

/// The region, scale and filters of the picture over `node_bounds`.
/// `identity`: capture with no filter (a mask capture), the capture is the result.
fn params_for_bounds(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
    node_bounds: Rect,
    source: CaptureSource,
    identity: bool,
    diagnostics: &mut Vec<String>,
) -> Option<LayerBlurParams> {
    // Project the accumulated transform, including ancestor scale and shear.
    // Keep the existing minimum-axis scale for other effects; layer blur
    // needs both principal stretches before it can choose an isotropic kernel.
    let canvas_scale = ctx.canvas_target_scale().max(0.0001);
    let scales = node_projected_scales(scene, node, ctx, &node_bounds)
        .unwrap_or(ProjectedScales { major: [1.0, 0.0], axis_min: canvas_scale, min: canvas_scale, max: canvas_scale });
    let effective_scale = scales.axis_min;
    let to_target = |canvas_px: f64| (canvas_px * effective_scale).min(MAX_TEXTURE_RADIUS_PX);
    // A blur radius (σ = R / 2) as the kernels' 3σ reach, in target px.
    let blur_to_target =
        |radius: f64| radius * moda_scene::effect::BLUR_REACH_PER_RADIUS * scales.max;

    // Check the requested reach before any capture sizing or kernel construction. The
    // projection uses f32 matrices; SNAP_PX admits their rounding at the exact limit.
    // The synthetic mask-content group carries a single identity capture effect.
    let effects = moda_scene::effect::node_effects_shared(node);
    let identity = identity || effects.iter().any(|e| e.is_visible()
        && matches!(e.kind, moda_scene::effect::EffectKind::MaskCapture));
    if !identity {
        if let Some(b) = moda_scene::effect::layer_blur(node).filter(|b| b.is_active()) {
            let radius = b.progressive.map_or(b.radius, |p| b.radius.max(p.start_radius));
            let reach = blur_to_target(radius);
            if !reach.is_finite() {
                diagnostics.push(format!("{}: layer blur reach is non-finite after transforms", node.id()));
                return None;
            }
            // The capture kernel is isotropic. A minimum-axis scalar can hide
            // a visible blur on the other axis; maximum alone would render the
            // wrong isotropic blur. Only projection-rounding differences fit.
            let min_reach = radius * moda_scene::effect::BLUR_REACH_PER_RADIUS * scales.min;
            // The projection is f32. At very large sigma a rounding error
            // between equal stretches can exceed the absolute pixel tolerance.
            let tolerance = SNAP_PX.max(reach * f32::EPSILON as f64);
            if b.progressive.is_some() && reach > 0.5 && reach - min_reach > tolerance {
                diagnostics.push(format!("{}: layer blur requires uniform projected scale; use uniform scale on the filtered node and its ancestors", node.id()));
                return None;
            }
        }
    }

    // The blur's reach (3σ) in target px: (start, end), equal for a uniform blur.
    let blur = moda_scene::effect::layer_blur(node)
        .filter(|b| b.is_active())
        .and_then(|b| {
            let end = blur_to_target(b.radius.max(0.0));
            let start = b
                .progressive
                .map_or(end, |p| blur_to_target(p.start_radius.max(0.0)));
            (start.max(end).is_finite() && start.max(end) > 0.5).then_some((b, start, end))
        });
    let texture = moda_scene::effect::texture(node).and_then(|t| {
        let radius = to_target(t.radius);
        let size = t.size * effective_scale;
        (radius.is_finite() && radius > 0.5 && size.is_finite() && size > 0.0)
            .then_some((t, size, radius))
    });
    let noises: Vec<(moda_scene::effect::Noise, f64)> = moda_scene::effect::noise_effects(node)
        .into_iter()
        .map(|n| {
            // Figma keeps its grain at output size: `size` is target px, not
            // canvas px (its 2x export draws a 1.5 grain as 1.5 export px).
            let size = n.size;
            (n, size)
        })
        .filter(|(_, size)| size.is_finite() && *size > 0.0)
        .collect();
    // Feather: its blur's reach (3σ, σ = radius / 2) in target px. It fades
    // the edge INSIDE the outline, but its blur reads that far outside: the
    // capture is padded by it, or the blur's clamp-to-edge sampling would
    // repeat the outline where it touches the capture's border.
    let feather = moda_scene::effect::feather(node)
        .map(|f| f.radius * moda_scene::effect::BLUR_REACH_PER_RADIUS * effective_scale)
        .filter(|r| r.is_finite() && *r > 0.5);

    // Filter shadows (CSS `filter: drop-shadow()`): the capture's own alpha, moved, blurred and tinted under it.
    let shadows = moda_scene::effect::filter_shadows(node);
    // An identity (mask) capture runs none of them: its node's own filters take the layer capture.
    let (blur, texture, noises, feather, shadows) = if identity {
        (None, None, Vec::new(), None, Vec::new())
    } else {
        (blur, texture, noises, feather, shadows)
    };
    use moda_scene::effect::EffectKind;
    let color_matrices: Vec<_> = effects.iter().filter(|e| e.is_visible())
        .filter_map(|e| match e.kind {
            EffectKind::ColorMatrix { values } => Some(values),
            _ => None,
        }).collect();
    if color_matrices.is_empty() && blur.is_none() && texture.is_none() && noises.is_empty() && feather.is_none() && shadows.is_empty() && !identity {
        return None;
    }
    // A filter shadow's blur reach (3σ, σ = blur / 2) in target px: node-local, so it scales with the node, as a
    // drop shadow's does (`shadow::node_blur_scale`); its offset is canvas-space.
    let shadow_blur_reach_px =
        |s: &moda_scene::effect::FilterShadow| s.blur.max(0.0) * moda_scene::effect::BLUR_REACH_PER_RADIUS * effective_scale;
    // How far a filter shadow paints past the picture, in target px: the farthest one's offset and reach.
    let filter_shadow_px = shadows
        .iter()
        .map(|s| s.offset.x.abs().max(s.offset.y.abs()) * canvas_scale + shadow_blur_reach_px(s))
        .fold(0.0, f64::max);
    if !filter_shadow_px.is_finite() {
        return None;
    }
    let blur_reach_px = blur.map_or(0.0, |(_, s, e)| s.max(e));
    let texture_reach_px = texture.map_or(0.0, |(_, _, r)| r);

    let paint_reach_px = match source {
        CaptureSource::Geometry => estimated_paint_reach(node, effective_scale, canvas_scale),
        CaptureSource::DrawnLeaf { .. } | CaptureSource::Subtree => 0.,
    };
    let pad_canvas = (blur_reach_px + texture_reach_px + feather.unwrap_or(0.) + filter_shadow_px
        + paint_reach_px) / canvas_scale + 1.0;

    let mut region = Rect {
        x: node_bounds.x - pad_canvas,
        y: node_bounds.y - pad_canvas,
        w: node_bounds.w + pad_canvas * 2.0,
        h: node_bounds.h + pad_canvas * 2.0,
    };
    if !region.finite() || !(region.w > 0.0) || !(region.h > 0.0) {
        return None;
    }
    // Inside clipping containers: only when the composite can be clipped as
    // the body is (see the fn doc).
    if !matches!(source, CaptureSource::Subtree) && !clipping_containers_clip_the_composite(scene, node.id(), &region) {
        return None;
    }
    let uniform_blur = blur.is_some_and(|(b, _, _)| b.progressive.is_none());
    // Resolving an elliptical filter at a scale chosen from its long axis
    // would discard detail along the short axis before the convolution.
    let elliptical = uniform_blur && blur_reach_px * (1.0 - scales.min / scales.max) > SNAP_PX.max(blur_reach_px * f32::EPSILON as f64);
    let mut capture_zoom = canvas_scale;
    // A capture over the size limits first leaves out what nothing can read (`readable_window`): content far
    // off the target (a credit roll, a pan across a wide picture) then costs no resolution. A capture under the
    // limits is not cut: its region, and so its pixels and its content key, do not depend on where the target
    // is. The cut is made only where it is sound (`readable_window`); elsewhere the whole capture is taken at a
    // reduced scale, below, as before.
    let cuttable = cuts_cleanly(node);
    let over_limits = |r: &Rect| {
        r.w.max(r.h) * capture_zoom > MAX_TEXTURE_DIM || (r.w * capture_zoom) * (r.h * capture_zoom) > MAX_CAPTURE_AREA_PX
    };
    // The picture's extent when the region is cut: what encloses this capture measures it by that.
    let mut uncut: Option<Rect> = None;
    if cuttable && over_limits(&region) {
        if let Some(window) = readable_window(scene, node, ctx, identity) {
            match region.intersect(&window) {
                // A window that holds the whole region cuts nothing: the capture is then an uncut one, and
                // its extent is its final region.
                Some(cut) => {
                    if cut.w < region.w || cut.h < region.h {
                        uncut = Some(region);
                        region = cut;
                    }
                }
                // Nothing of this layer can be read: no pixel of the target, and no capture above, reaches
                // into its region. It is not captured at all (its items, which lie inside the region, draw
                // where nothing shows them). Kept whole it would be reduced to fit the limits after all, and
                // a reduced capture's texels are large enough to spill onto the target and to read past what
                // a descendant's cut keeps: with this, no capture in a chain that holds a cut is reduced.
                // Not for a text that renders through a mask capture (`needs_mask_capture`), filtered or as
                // the identity capture itself: declining its filter does not leave it undrawn, the caller
                // falls back to that identity capture (or to a text clipped by the mask's box, with a
                // warning). Such a text keeps the capture it had, whole and reduced as before; it has no
                // descendants, so it is never the reduced ancestor of a cut.
                None if !identity && !needs_mask_capture(scene, node) => return None,
                None => {}
            }
        }
    }
    // The capture and the convolution need different bounds. Empty Gaussian
    // padding must not lower the rasterization scale. When it would, capture
    // the contributing source and the readable output. Supply the omitted
    // transparent surround on the reduced grid, where the Gaussian reads it.
    let mut blur_border = 0;
    if over_limits(&region) && uniform_blur && !elliptical && cuttable && shadows.is_empty() {
        if let Some((compact, border)) = readable_window(scene, node, ctx, true).and_then(|window|
            uniform_capture_window(region, node_bounds, window, canvas_scale, pad_canvas, blur_reach_px)) {
            uncut.get_or_insert(region);
            region = compact;
            blur_border = border;
        }
    }
    let max_dim_canvas = region.w.max(region.h);
    if max_dim_canvas * capture_zoom > MAX_TEXTURE_DIM {
        capture_zoom = MAX_TEXTURE_DIM / max_dim_canvas;
    }
    let area_px = (region.w * capture_zoom) * (region.h * capture_zoom);
    if area_px > MAX_CAPTURE_AREA_PX {
        capture_zoom *= (MAX_CAPTURE_AREA_PX / area_px).sqrt();
    }
    let k = capture_zoom / canvas_scale;
    // A full-scale capture lands its texels exactly on target pixels: the
    // region's origin moves back onto the target's pixel grid, so the
    // composite's bilinear sample reads one texel per pixel (a half-texel
    // offset would average neighbouring noise cells and soften a sharp
    // edge). `canvas_scale` carries the f32 view-projection's rounding, hence
    // the snap tolerance.
    if k == 1.0 {
        if let Some((px, py)) = ctx.project_point(region.x, region.y) {
            let off = |v: f64| {
                if (v - v.round()).abs() < SNAP_PX {
                    v - v.round()
                } else {
                    v - v.floor()
                }
            };
            let (dx, dy) = (off(px) / canvas_scale, off(py) / canvas_scale);
            region = Rect {
                x: region.x - dx,
                y: region.y - dy,
                w: region.w + dx,
                h: region.h + dy,
            };
        }
    }
    // The texture holds whole texels, and the region is widened to exactly
    // cover them (region px = texture px), so the composite never stretches
    // the capture. `.min(MAX_TEXTURE_DIM)` guards the 1-ulp ceil overshoot of
    // `x × (MAX/x)` — texture creation must never exceed the device limit.
    let whole = |v: f64| {
        if (v - v.round()).abs() < SNAP_PX {
            v.round()
        } else {
            v.ceil()
        }
    };
    // A wide filter shadow blurs at 1/2^n resolution (`exec::blur_alpha`): whole 2^n-texel blocks keep each
    // halving an exact 2×2 average, so the capture is padded (right and bottom) to a multiple, as a capture
    // shadow's is (`shadow::plan_shadow_group`).
    let block = {
        let mut reach = shadows.iter().map(|s| shadow_blur_reach_px(s) * k).fold(0.0, f64::max);
        let mut block: f64 = 1.0;
        while reach > crate::layer_blur::exec::PROGRESSIVE_BLUR_MAX_TAPS as f64 {
            reach /= 2.0;
            block *= 2.0;
        }
        if uniform_blur && !elliptical {
            block = block.max(crate::layer_blur::exec::uniform_blur_block((blur_reach_px * k) as f32) as f64);
        }
        block
    };
    let whole = |v: f64| {
        let v = whole(v);
        let padded = (v / block).ceil() * block;
        if padded <= MAX_TEXTURE_DIM {
            padded
        } else {
            v
        }
    };
    let width_px = whole(region.w * capture_zoom).clamp(1.0, MAX_TEXTURE_DIM) as u32;
    let height_px = whole(region.h * capture_zoom).clamp(1.0, MAX_TEXTURE_DIM) as u32;
    region.w = width_px as f64 / capture_zoom;
    region.h = height_px as f64 / capture_zoom;
    // Identity captures are mandatory for exact masks, even for subpixel coverage.
    // Degenerate capture (e.g. an extreme zoom-out collapses the region to a
    // texel) — blurring 1px of data is indistinguishable from the node's own
    // render; decline. A filter shadow never declines for size: declining would
    // draw the node without its shadow, and a hairline casts one.
    let must_capture = identity || uniform_blur || !shadows.is_empty();
    if !must_capture && (width_px < 2 || height_px < 2) {
        return None;
    }
    // Footprint guard: the decline contract is "renders normally WITHOUT
    // blur", never "renders as nothing". When the capture scale has been
    // crushed (huge pad from legitimate effect reach, extreme aspect ratios,
    // the texture/area clamps) to the point where the NODE's own body drops
    // below one capture px on its THINNER axis, the capture holds ~no alpha
    // and the composite paints an effectively empty quad over a skipped node
    // — decline instead so the main pass draws the node plainly. Min-axis at
    // 1px keeps legitimate hairlines blurring at normal scales (a 1px strip
    // at capture_zoom 1 is exactly 1 capture px).
    if !must_capture && node_bounds.w.min(node_bounds.h) * capture_zoom < 1.0 {
        return None;
    }

    // Canvas → capture px, and the node's origin there (the grain lattice
    // anchor, so noise and texture stick to the node under any camera).
    let to_capture = |x: f64, y: f64| {
        [
            ((x - region.x) * capture_zoom) as f32,
            ((y - region.y) * capture_zoom) as f32,
        ]
    };
    let anchor = if texture.is_some() || !noises.is_empty() {
        grain_anchor(scene, node, source, node_bounds, !shadows.is_empty())
    } else {
        node_bounds
    };
    let origin_px = to_capture(anchor.x, anchor.y);

    let blur = blur.map(|(b, start, end)| match b.progressive {
        Some(p) => {
            // The ramp's points are fractions of the node's LOCAL bounds,
            // mapped through its canvas affine (a rotated node's ramp turns
            // with it).
            let (start_px, end_px) = scene
                .node_canvas_affine(node.id())
                .zip(node_local_rect(node))
                .map(|(a, local)| {
                    let at = |f: moda_scene::effect::EffectOffset| {
                        let (lx, ly) = (local.x + f.x * local.w, local.y + f.y * local.h);
                        to_capture(a.a * lx + a.c * ly + a.e, a.b * lx + a.d * ly + a.f)
                    };
                    (at(p.start), at(p.end))
                })
                .unwrap_or((origin_px, origin_px));
            BlurFilter::Progressive {
                start_radius_px: (start * k) as f32,
                end_radius_px: (end * k) as f32,
                start_px,
                end_px,
            }
        }
        // With noise the capture stays at full scale (see the fn doc); the
        // separable ladder takes any radius there, sampling every texel.
        None if elliptical => BlurFilter::Elliptical {
            major: scales.major,
            reaches: [(end * k) as f32, (end * k * scales.min / scales.max) as f32],
        },
        None => BlurFilter::Uniform {
            radius_px: (end * k) as f32,
        },
    });
    // The node's opacity is in the captured items already, so a pass carries the shadow's own alpha only.
    let drop_shadows = shadows
        .iter()
        .map(|s| {
            let rgba = crate::effects::shadow::shadow_color_rgba(&s.color);
            ShadowPass {
                color: [
                    rgba[0] as f32,
                    rgba[1] as f32,
                    rgba[2] as f32,
                    (rgba[3] * s.opacity()).clamp(0.0, 1.0) as f32,
                ],
                offset_px: [(s.offset.x * capture_zoom) as f32, (s.offset.y * capture_zoom) as f32],
                radius_px: (shadow_blur_reach_px(s) * k) as f32,
                show_behind: true,
                center: false,
            }
        })
        .collect();
    // Only shadows: they composite before the layer's own items, which draw directly (`LayerFilters::shadows_alone`).
    let shadows_alone = color_matrices.is_empty() && !shadows.is_empty()
        && blur.is_none()
        && texture.is_none()
        && noises.is_empty()
        && feather.is_none()

       ;
    let texture = texture.map(|(t, size, radius)| TextureFilter {
        size_px: (size * k) as f32,
        radius_px: (radius * k) as f32,
        clip_to_shape: t.clips_to_shape(),
    });
    let noise = noises
        .into_iter()
        .enumerate()
        .map(|(seed, (n, size))| NoiseFilter {
            mode: match n.mode.as_str() {
                "duo" => 1,
                "multi" => 2,
                _ => 0,
            },
            color: effect_color(&n.color),
            secondary: n
                .secondary_color
                .as_ref()
                .filter(|_| n.mode == "duo")
                .map(effect_color),
            size_px: (size * k) as f32,
            density: n.density.clamp(0.0, 1.0) as f32,
            opacity: n.opacity().clamp(0.0, 1.0) as f32,
            seed: seed as u32,
        })
        .collect();

    Some(LayerBlurParams {
        region_canvas: [region.x, region.y, region.w, region.h],
        extent_canvas: uncut.map_or([region.x, region.y, region.w, region.h], |r| [r.x, r.y, r.w, r.h]),
        width_px,
        height_px,
        reach_canvas: pad_canvas,
        filters: LayerFilters {
            color_matrices,
            origin_px,
            drop_shadows,
            shadows_alone,
            texture,
            blur,
            blur_border,
            source_raster: (uniform_blur && k < 1.0).then(|| {
                let (x, y) = ctx.project_point(region.x, region.y).unwrap_or((0., 0.));
                let phase = |p: f64| if (p - p.round()).abs() < SNAP_PX { 0.0 } else { p - p.floor() };
                Box::new(crate::seams::CaptureRaster { scale: canvas_scale, phase: [phase(x), phase(y)] })
            }),
            noise,
            feather: feather.map(|r| FeatherFilter {
                radius_px: (r * k) as f32,
            }),
            shadow: None,
            composite_blend: crate::seams::CaptureBlend::Normal,
        },
    })
}

/// Grain is attached to the original geometry. Text and filter shadows already
/// used the union with the ink before ordinary strokes got bigger captures.
fn grain_anchor(scene: &Scene, node: &RawNode, source: CaptureSource, bounds: Rect, filter_shadow: bool) -> Rect {
    let (mut geometry, drawn) = match source {
        CaptureSource::Subtree => return bounds,
        CaptureSource::Geometry => (bounds, None),
        CaptureSource::DrawnLeaf { geometry, drawn } => (geometry, Some(drawn)),
    };
    if node.node_type() == "path" && moda_scene::path_geom::round_corners::PathCorners::read(node).rounds_anything() {
        // The path lattice was placed by the authored anchors even when the paint rounded its corners.
        if let Some(raw) = node.points().and_then(|p| moda_scene::path_geom::bounds::drawn_path_bounds(p.draw_commands())) {
            let rect = Rect { x: raw.min_x, y: raw.min_y, w: raw.max_x - raw.min_x, h: raw.max_y - raw.min_y };
            geometry = crate::backdrop::plan::local_rect_canvas_aabb(scene, node, rect).unwrap_or(geometry);
        }
    }
    if !node.truthy(moda_scene::appearance::ENTRY_PROBE_KEY) && (node.node_type() == "richtext" || filter_shadow) {
        super::shadow::union_canvas_bounds(Some(geometry), drawn).unwrap_or(geometry)
    } else {
        geometry
    }
}

/// Paint reach in target pixels when only the leaf geometry is available.
fn estimated_paint_reach(node: &RawNode, effective_scale: f64, canvas_scale: f64) -> f64 {
    let (shadow_blur_px, shadow_offset_px) = crate::effects::shadow::drop_shadow_quads(node)
        .into_iter()
        .chain(crate::layer_blur::shadow::capture_drop_shadows(node))
        .collect::<Vec<_>>()
        .iter()
        .map(|shadow| {
            (
                // The quad grows by 1.5 × blur and a positive spread per side
                // (`path-shadow.wgsl`); 2 × blur covers both for the blur.
                (shadow.blur.max(0.0) * 2.0 + shadow.spread().max(0.0)) * effective_scale,
                shadow_offset_reach(shadow) * canvas_scale,
            )
        })
        .fold((0.0, 0.0), |widest: (f64, f64), s| {
            if s.0 + s.1 > widest.0 + widest.1 {
                s
            } else {
                widest
            }
        });
    // Capture glows reach as far as a shadow of blur = radius.
    let glow_px = crate::layer_blur::shadow::capture_glows(node)
        .iter()
        .map(|g| g.radius * 2.0 * effective_scale)
        .fold(0.0, f64::max);
    let (shadow_blur_px, shadow_offset_px) = if glow_px > shadow_blur_px + shadow_offset_px {
        (glow_px, 0.0)
    } else {
        (shadow_blur_px, shadow_offset_px)
    };
    let stroke_px = moda_scene::paint::node_stroke_style(node)
        .width
        .unwrap_or(0.0)
        .max(0.0)
        * effective_scale;
    shadow_blur_px + shadow_offset_px + stroke_px
}

/// An effect colour (a CSS string; variable references are resolved before
/// rendering) as straight RGB in the frame's encoding. Unparseable = black.
fn effect_color(v: &moda_scene::value::Value) -> [f32; 3] {
    crate::color::parse_color(v.as_str())
        .map(|c| [c.r as f32, c.g as f32, c.b as f32])
        .unwrap_or([0.0; 3])
}

/// Drop-shadow OFFSET reach in canvas/world units — the offset is added to
/// worldPos after the model transform (`path-shadow.wgsl`), so it is
/// world-space and does NOT scale with the node's own transform.
fn shadow_offset_reach(shadow: &moda_scene::effect::DropShadow) -> f64 {
    shadow.offset.x.abs().max(shadow.offset.y.abs())
}

struct ProjectedScales {
    major: [f32; 2],
    /// Preserve the existing scale for effects other than layer blur.
    axis_min: f64,
    /// Principal stretches of the blur, including nonorthogonal (sheared) axes.
    min: f64,
    max: f64,
}

fn node_projected_scales(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
    bounds: &Rect,
) -> Option<ProjectedScales> {
    let affine = scene.node_canvas_affine(node.id())?;
    let cx = bounds.x + bounds.w * 0.5;
    let cy = bounds.y + bounds.h * 0.5;
    let center = ctx.project_point(cx, cy)?;
    let ax = ctx.project_point(cx + affine.a, cy + affine.b)?;
    let ay = ctx.project_point(cx + affine.c, cy + affine.d)?;
    let (a, b) = (ax.0 - center.0, ax.1 - center.1);
    let (c, d) = (ay.0 - center.0, ay.1 - center.1);
    let axis_min = a.hypot(b).min(c.hypot(d));
    // Singular values of the projected 2x2 linear transform. Equal column
    // lengths alone do not detect shear. det/max avoids cancellation of the
    // small singular value for extremely unequal scales; rank one stays zero.
    let max = ((a + d).hypot(b - c) + (a - d).hypot(b + c)) * 0.5;
    let min = (a * d - b * c).abs() / max;
    // The principal eigenvector of A Aᵀ is the long axis of the projected Gaussian.
    let angle = 0.5 * (2.0 * (a*b + c*d)).atan2(a*a + c*c - b*b - d*d);
    (max.is_finite() && max > 0.0 && min.is_finite()).then_some(ProjectedScales {
        major: [angle.cos() as f32, angle.sin() as f32],
        axis_min: if axis_min > 0.0 { axis_min } else { ctx.canvas_target_scale().max(0.0001) },
        min,
        max,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(json: moda_scene::value::Value) -> RawNode {
        RawNode::new(json.as_object().unwrap().clone())
    }

    fn blurred(node_type: &str, blur: moda_scene::value::Value) -> RawNode {
        let mut entry = moda_scene::value::json!({"id": "layer-blur-0", "type": "layer-blur"});
        entry.as_object_mut().unwrap().extend(blur.as_object().unwrap().clone());
        raw(moda_scene::value::json!({"id": "a", "type": node_type, "effects": [entry]}))
    }

    #[test]
    fn identity_mask_captures_keep_subpixel_coverage() {
        let scene = Scene::load(&moda_scene::value::json!({"nodes":[
            {"id":"clip", "type":"group", "children":["mask", "text"], "clipMaskNodeId":"mask"},
            {"id":"mask", "type":"ellipse", "width":100, "height":100},
            {"id":"text", "type":"richtext", "width":100, "height":100}],
            "pages":[{"id":"p", "width":100, "height":100, "nodeIds":["clip"]}], "config":{}})).unwrap();
        let node = scene.effective_node("text").unwrap();
        for size in [(0.25, 20.0), (20.0, 0.25), (0.25, 0.25)] {
            for pixels in [1, 100] {
                let ctx = PlanCtx::new(&scene, pixels, pixels, 0., 0., 100., 100., [0., 0., 100., 100.]);
                let mut diagnostics = Vec::new();
                let params = compute_mask_capture_params(&scene, node, &ctx,
                    Rect { x: 0., y: 0., w: size.0, h: size.1 }, &mut diagnostics)
                    .expect("visible mask coverage must survive blur-only footprint guards");
                assert!(params.width_px >= 1 && params.height_px >= 1);
                assert!(diagnostics.is_empty());
            }
        }
    }

    #[test]
    fn active_predicate_requires_positive_radius_on_capable_type() {
        assert!(has_active_layer_filter(&blurred("rectangle", moda_scene::value::json!({"radius": 8}))));
        // radius 0 = stored-but-inert
        assert!(!has_active_layer_filter(&blurred("rectangle", moda_scene::value::json!({"radius": 0}))));
        // text carries the layer filters too
        assert!(has_active_layer_filter(&blurred("richtext", moda_scene::value::json!({"radius": 8}))));
        // incapable node kind
        assert!(!has_active_layer_filter(&blurred("line", moda_scene::value::json!({"radius": 8}))));
        // absent
        assert!(!has_active_layer_filter(&raw(moda_scene::value::json!({
            "id": "a", "type": "rectangle"
        }))));
        // hidden via the Effects eye toggle
        assert!(!has_active_layer_filter(&blurred("rectangle", moda_scene::value::json!({"radius": 8, "visible": false}))));
        assert!(has_active_layer_filter(&blurred("rectangle", moda_scene::value::json!({"radius": 8, "visible": true}))));
    }

    #[test]
    fn distorted_nodes_decline() {
        // A corner-pin warp would escape the unwarped capture AABB — the
        // eligibility gate declines it (renders un-blurred), mirroring the
        // backdrop distort posture and the TS export gate.
        let doc = moda_scene::value::json!({
            "nodes": [{
                "id": "d1", "type": "rectangle", "x": 10, "y": 10,
                "width": 100, "height": 80, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}],
                "effects": [{"id": "layer-blur-0", "type": "layer-blur", "radius": 10}],
                "distort": {"type": "matrix",
                            "matrix": [1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0002, 0.0, 1.0]}
            }],
            "pages": [{"id": "p1", "width": 640, "height": 480, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}],
                       "nodeIds": ["d1"]}],
            "config": {},
        });
        let scene = Scene::load(&doc).unwrap();
        let node = scene.effective_node("d1").unwrap();
        assert!(has_active_layer_filter(node), "property itself is active");
        assert!(
            !is_node_layer_blur_eligible(node),
            "distort-warped nodes must render un-blurred"
        );
    }

    #[test]
    fn arc_and_bezier_paths_blur_over_their_drawn_bounds() {
        // Both curves bulge beyond their endpoints. The capture follows the segments used by tessellation.
        let blur = moda_scene::value::json!([{"id": "layer-blur-0", "type": "layer-blur", "radius": 10}]);
        let doc = moda_scene::value::json!({
            "nodes": [
                {"id": "arced", "type": "path", "x": 0, "y": 0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}],
                 "effects": blur,
                 "points": [
                     {"type": "move", "x": 0, "y": 0},
                     {"type": "arc", "x": 100, "y": 0, "rx": 50, "ry": 80,
                      "xAxisRotation": 0, "largeArc": true, "sweep": true}
                 ]},
                {"id": "curved", "type": "path", "x": 0, "y": 120, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}],
                 "effects": blur,
                 "points": [
                     {"type": "move", "x": 0, "y": 0},
                     {"type": "curve", "x": 100, "y": 0,
                      "cp1": {"x": 30, "y": -60}, "cp2": {"x": 70, "y": -60}}
                 ]}
            ],
            "pages": [{"id": "p1", "width": 640, "height": 480, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}],
                       "nodeIds": ["arced", "curved"]}],
            "config": {},
        });
        let scene = Scene::load(&doc).unwrap();
        let arced = scene.effective_node("arced").unwrap();
        assert!(is_node_layer_blur_eligible(arced), "arc-segment paths blur");
        let r = crate::backdrop::plan::node_local_rect(arced).unwrap();
        assert!(
            (r.w - 100.0).abs() < 1e-6 && (r.h - 80.0).abs() < 1e-3,
            "the capture holds the arc's bulge: {r:?}"
        );
        assert!(
            is_node_layer_blur_eligible(scene.effective_node("curved").unwrap()),
            "bezier paths have true extrema bounds and stay eligible"
        );
    }

    #[test]
    fn capture_includes_the_cubic_outside_an_ideal_circle() {
        let radius = 1920.;
        let at = |degrees: f64| (radius * degrees.to_radians().cos(), radius * degrees.to_radians().sin());
        let (x0, y0) = at(-19.);
        let (x1, y1) = at(70.99);
        let n = raw(moda_scene::value::json!({"id":"p","type":"path","points":[
            {"type":"move","x":x0,"y":y0},
            {"type":"arc","rx":radius,"ry":radius,"sweep":true,"x":x1,"y":y1}
        ]}));
        let r = crate::backdrop::plan::node_local_rect(&n).unwrap();
        assert!(r.x + r.w > radius + 0.52, "right edge {}", r.x + r.w);
    }

    #[test]
    fn fill_rect_preserves_subpixel_height_and_flat_strokes_supply_their_own_area() {
        let capture = |h: f64| {
            let node = raw(moda_scene::value::json!({"id":"p", "type":"path", "points":[
                {"type":"move", "x":0, "y":0},
                {"type":"quadratic", "x":2, "y":0, "cp":{"x":1, "y":h}}
            ]}));
            crate::backdrop::plan::node_local_rect(&node)
        };
        assert!(capture(0.0).is_none());
        assert_eq!(capture(1e-11).unwrap().h, 5e-12);
    }

    #[test]
    fn shadow_offset_reach_is_world_space_max_axis() {
        let shadow = |offset: moda_scene::value::Value| {
            let n = raw(moda_scene::value::json!({"id": "a", "type": "rectangle", "effects": [
                {"id": "drop-shadow-0", "type": "drop-shadow", "color": "#000", "blur": 5, "offset": offset}]}));
            moda_scene::effect::drop_shadow(&n).unwrap()
        };
        assert_eq!(shadow_offset_reach(&shadow(moda_scene::value::json!({"x": 3, "y": -7}))), 7.0);
        assert_eq!(shadow_offset_reach(&shadow(moda_scene::value::json!({"x": 0, "y": 0}))), 0.0);
    }
}
