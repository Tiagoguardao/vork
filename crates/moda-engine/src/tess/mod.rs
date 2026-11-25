//! tess/ — path tessellation. OWNED BY M2 (PLAN §0 rule 4).
//!
//! M2 replaced the M1 placeholder WHOLESALE with the faithful port of
//! `renderers/webgpu/geometry/WebGPUPathTessellator.ts`, gated by the vertex
//! oracle (`fixtures/oracle/*/tessellation.json`, tol 1e-4 via
//! `tools/oracle/diff_vertices.py` — stroke vertices; fills are lyon's, not
//! the TS ear clipper's, and are pinned by area/coverage tests + goldens):
//!   - `tessellator.rs` — De Casteljau flattening, lyon sweep-line fills
//!     (evenodd/nonzero on self-intersecting outlines, holes from winding),
//!     hard caps WITH truncation behavior, stroke bands
//!     (align/caps/joins/miter-4→bevel/round 22.5°), dash routing.
//!     Deliberate departures from the TS port (Illustrator comparison,
//!     2026-09-24; mirrored in `WebGPUPathTessellator.ts` except the bands):
//!       - Strokes and fills flatten to half a device pixel (F4.1): 0.5 units
//!         at 1× (as before), halved per zoom band above and doubled per band
//!         below (zoom × dpr on the canvas, the export scale for exports;
//!         bands clamped to 1/16×..64×), so curved fills — placed art-brush
//!         artwork included — keep their curves at 2× and zoomed-out curves
//!         cost fewer vertices. Tessellations are cached per node and band.
//!         A curved stroke or fill that would hit a cap at its band is redone
//!         coarser, band by band, and last with the pre-band flattening
//!         (`stroke_within_caps`, `fill_path`), so zoom never truncates or
//!         drops anything main drew.
//!       - Every stroke piece also obeys a turn limit derived from the stroke's
//!         outer offset (half the width centred, the whole width inside /
//!         outside), so wide curved strokes have no visible facets. This one
//!         changes 1× output for wide curved strokes (a few more edges; golden
//!         deltas in #87).
//!       - Dashes walk the flattened curve (each curve's pieces keep its arc
//!         length) instead of chords between dash ends.
//!       - Inner joins fill to the legs' offset crossing (the miter limit only
//!         clips the outer tip); past a short leg they pivot through the vertex.
//!     `fixtures/oracle/strokes-dashes.json/tessellation.json` was regenerated
//!     from `examples/dump_tess` for the dash change; the other oracle fixtures
//!     are unchanged.
//!   - `arc.rs` — arcs → cubics per `core/svg/arcUtils.ts`.
//!   - `dash.rs` — arc-length walk (`utils/lineToPath.ts`), strokeWidth/2
//!     pattern scaling, odd-pattern doubling; dash kills joins/align/closure.
//!   - `shape_paths.rs` — star/polygon inscribed paths + the distort outline warp.
//!   - `line_to_path.rs` — line bodies + arrowheads
//!     (`WebGPULineToPath.ts`, `core/scene/arrowheads.ts`).
//!
//! INTERFACE (frozen with seams.rs): `tessellate_combined` keeps the M1
//! signature and returns `seams::TessellatedGeometry` (fill vertices first,
//! then stroke; vertex_types 0.0 = fill / 1.0 = stroke).
//! `tessellate_combined_with_dash` is the M2 superset (additive — see the
//! SEAMS.md change log).

mod arc;
mod bounds;
pub mod connector_text;
mod dash;
pub mod line_to_path;
pub mod curve_quality;
pub(crate) mod paint_order;
pub mod path_arrows;
pub mod shape_paths;
pub mod stroke_param;
mod tessellator;
pub mod trim_axis;

pub use bounds::path_bounds;
pub(crate) use bounds::drawn_node_path_bounds;
pub use trim_axis::TrimBy;
pub use tessellator::{
    tessellate_combined_cached, Contour, PathClosure, PathContours,
    is_trim_identity, last_fill_cell_count, last_fill_vertex_peak,
    tessellate_combined, tessellate_combined_oracle, tessellate_combined_with_dash,
    tessellate_combined_with_dash_trim, tessellate_path,
    dash_alignment_now, subpath_corners, trim_params_from_fields, with_dash_alignment,
    with_fill_sweep_budget, with_fill_vertex_cap, with_miter_limit,
    OracleGeometry, PathEnd, TrimParams, MAX_FILL_VERTICES, axis_trim_arc_windows,
};

/// Build a node's visible fill and stroke geometry. Direct path masks use the
/// same settings and command cache as the body lane.
pub(crate) fn tessellate_body(
    node: &moda_scene::types::RawNode,
    cmds: &[moda_engine_path_types::PathCmd],
    closure: PathClosure<'_>,
    stroke_width: f64,
    fill_rule: moda_engine_path_types::FillRule,
    style: &moda_scene::paint::StrokeStyle,
) -> std::sync::Arc<crate::seams::TessellatedGeometry> {
    use moda_engine_path_types as pt;
    let trim = curve_quality::trim_params_from_node(node);
    with_dash_alignment(style.dash_aligned(), || {
        with_miter_limit(node.miter_limit(), || {
            tessellate_combined_cached(
                cmds,
                stroke_width,
                closure,
                trim,
                style.dash.as_deref().filter(|dash| !dash.is_empty()),
                style.dash_offset.unwrap_or(0.0),
                pt::line_cap_from_str(style.cap.as_deref()),
                pt::line_join_from_str(style.join.as_deref()),
                fill_rule,
                pt::stroke_align_from_str(style.align.as_deref()),
            )
        })
    })
}

/// The single derive point for a `path` node's drawn geometry — never read
/// `points` directly in a render lane.
///
/// `path` has an editable uniform `cornerRadius`; the drawn outline is
/// therefore no longer the authored `points` array. Every lane that draws, masks,
/// or measures a path's body (body tessellation, backdrop coverage, mask
/// geometry, the tess dump) must come through here so the rounding cannot be
/// applied in one place and skipped in another.
///
/// `radius <= 0` (the state of EVERY document authored before path rounding, and of
/// every path the user has not rounded) returns `cmds_from_node_points` on the
/// raw array — the same commands, produced by the same code, so existing render
/// plans stay byte-exact.
pub fn path_cmds_for_node(
    node: &moda_scene::types::RawNode,
) -> Vec<moda_engine_path_types::PathCmd> {
    // A vector network (E7.9) is authoritative: its region loops, then the
    // region-less runs — what the editor derives into `points`, recomputed
    // here so a scene that carries only the network still draws.

    let points = node
        .points()
        .unwrap_or(moda_scene::path_points::PathPoints::Json(&[]));
    // `path.cornerRadius || 0` (JS falsy: 0/NaN/missing → 0), matching star/polygon; an anchor's own
    // `cornerRadius` / `cornerStyle` override it (live corners).
    let corners = moda_scene::path_geom::round_corners::PathCorners::read(node);
    if !corners.rounds_anything() {
        return moda_engine_path_types::cmds_from_path_points(points);
    }

    let parsed: Vec<moda_scene::blend::BlendPathPoint> = points
        .iter()
        .filter_map(moda_scene::blend::BlendPathPoint::from_ref)
        .collect();
    moda_scene::path_geom::round_corners::round_path_corners_with(
        &parsed,
        node.path_closure(),
        &corners,
    )
}

/// Path-command model the tessellator consumes — mirrors
/// `core/scene/nodes.ts` PathPoint variants (move/line/curve/quadratic/arc).
pub mod moda_engine_path_types {
    pub use moda_scene::blend::BlendPathPoint as PathCmd;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FillRule {
        NonZero,
        EvenOdd,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum LineCap {
        Butt,
        Round,
        Square,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum LineJoin {
        Miter,
        Round,
        Bevel,
    }

    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub enum StrokeAlign {
        #[default]
        Center,
        Inside,
        Outside,
    }

    /// JSON `points` array of a path node → commands. Unknown types skipped.
    pub fn cmds_from_node_points(points: &[moda_scene::value::Value]) -> Vec<PathCmd> {
        cmds_from_path_points(moda_scene::path_points::PathPoints::Json(points))
    }

    /// A node's points ([`moda_scene::types::RawNode::points`]) → commands; exactly
    /// [`cmds_from_node_points`] of their JSON.
    pub fn cmds_from_path_points(points: moda_scene::path_points::PathPoints<'_>) -> Vec<PathCmd> {
        points.draw_commands().collect()
    }

    /// `node.dash` → dash pattern. Non-array values (the WEB-NEXTJS-13T
    /// regression: a string at runtime) are silently skipped, exactly like
    /// the TS `Array.isArray(dash)` guard.
    pub fn dash_from_node_value(v: Option<&moda_scene::value::Value>) -> Option<Vec<f64>> {
        let arr = v?.as_array()?;
        Some(
            arr.iter()
                .filter_map(moda_scene::value::Value::as_f64)
                .collect::<Vec<f64>>(),
        )
    }

    pub fn fill_rule_from_str(s: Option<&str>) -> FillRule {
        match s {
            Some("evenodd") => FillRule::EvenOdd,
            _ => FillRule::NonZero,
        }
    }

    pub fn line_cap_from_str(s: Option<&str>) -> LineCap {
        match s {
            Some("round") => LineCap::Round,
            Some("square") => LineCap::Square,
            _ => LineCap::Butt,
        }
    }

    pub fn line_join_from_str(s: Option<&str>) -> LineJoin {
        match s {
            Some("round") => LineJoin::Round,
            Some("bevel") => LineJoin::Bevel,
            _ => LineJoin::Miter,
        }
    }

    pub fn stroke_align_from_str(s: Option<&str>) -> StrokeAlign {
        match s {
            Some("inside") => StrokeAlign::Inside,
            Some("outside") => StrokeAlign::Outside,
            _ => StrokeAlign::Center,
        }
    }
}

#[cfg(test)]
mod contour_tests {
    use super::*;
    use moda_scene::types::RawNode;
    use moda_scene::value::{Value, json};

    #[test]
    fn rounded_path_rounds_the_closed_seam_and_preserves_open_ends() {
        let open = json!([
            {"type":"move","x":0,"y":0}, {"type":"line","x":10,"y":0},
            {"type":"line","x":10,"y":10}
        ]);
        let closed = json!([
            {"type":"move","x":20,"y":0}, {"type":"line","x":30,"y":0},
            {"type":"line","x":30,"y":10}
        ]);
        let node = |points, closure| RawNode::new(json!({
            "id":"p","type":"path","points":points,"closed":closure,"cornerRadius":1
        }).as_object().unwrap().clone());
        let points = open.as_array().unwrap().iter().chain(closed.as_array().unwrap()).cloned().collect();
        let compound = path_cmds_for_node(&node(Value::Array(points), json!([false, true])));
        let mut expected = path_cmds_for_node(&node(open, json!(false)));
        expected.extend(path_cmds_for_node(&node(closed.clone(), json!(true))));
        assert_eq!(compound, expected);
        assert_ne!(expected, path_cmds_for_node(&node(closed, json!(false))));
    }
}
