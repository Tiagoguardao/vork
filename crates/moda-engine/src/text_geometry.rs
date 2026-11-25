//! Read-only richtext → vector geometry, for exporters that write vectors (SVG export, F8.1).
//!
//! Runs the SAME vector layout and draw-op builder the renderer draws with
//! (`text_draw::node_to_text_spec` → `moda_text::vector::build_vector_node_ops`, `VectorParity`
//! placement) and turns its ops into path data in node-local px:
//!
//! - analytic glyph ops → the placed glyphs' quadratic outlines (the curve buffer the GPU
//!   coverage shader reads, placed with the shader's own `pen + lin·(em·size − C) + C + off`
//!   math, synthetic oblique included), one path per op, nonzero;
//! - tess ops (list bullets, underline / strike-through rects) → their
//!   triangles as one nonzero path per op (one path, so no anti-aliasing seams).
//!
//! Outlined text therefore lands on the pixels the canvas draws. The runs (text, family,
//! weight, style, size, baseline, colour, per-glyph pen x) come from the text-layer extractor
//! over the same layout, for exporters that keep text live.
//!
//! Declines (returns `Err(reason)`) when the node would not draw through the vector lane, or
//! when its ops carry passes a flat fill can't express (shadow / glow blur, text stroke,
//! additive glow, gradient or shader fills) — the exporter rasterizes those nodes instead.

use moda_scene::Scene;
use moda_text::node_layout::RichTextNodeSpec;
use moda_text::vector::extract::{extract_vector_text_runs, ExtractedRun};
use moda_text::vector::{VectorEngineDecision, VectorOpPipeline, VectorPlacement};

/// One filled path of a text node, node-local px.
#[derive(Debug, Clone)]
pub struct TextGeometryFill {
    /// "glyphs" (outlined glyphs) or "shape" (bullets, decorations); `node_fill_outlines` also writes "effect" (an
    /// echo copy or extrude wall, in its own colour).
    pub kind: &'static str,
    /// Path data: one entry per glyph for "glyphs" (a line of glyphs per fill), one for "shape".
    pub paths: Vec<String>,
    /// Straight RGBA 0..1.
    pub color: [f32; 4],
    pub opacity: f32,
    /// A text range's gradient (`--nf-text-fill`) painting these glyphs instead of `color`, as a Moda paint over
    /// the node box (JSON: `{ type: 'linear' | 'radial', … }`).
    pub gradient: Option<moda_scene::value::Value>,
}

#[derive(Debug, Clone)]
pub struct TextGeometry {
    pub node_id: String,
    pub fills: Vec<TextGeometryFill>,
    /// `None` when the extractor could not map the node's glyphs back to text.
    pub runs: Option<Vec<ExtractedRun>>,
}

fn fmt(v: f32) -> String {
    let r = (v as f64 * 1000.0).round() / 1000.0;
    if r == 0.0 {
        "0".to_string()
    } else {
        format!("{r}")
    }
}

/// Triangles `[x0,y0,x1,y1,x2,y2, …]` → path data.
fn triangles_to_d(tris: &[f32]) -> String {
    let mut d = String::with_capacity(tris.len() * 6);
    for t in tris.chunks_exact(6) {
        d.push_str(&format!(
            "M{} {}L{} {}L{} {}Z",
            fmt(t[0]),
            fmt(t[1]),
            fmt(t[2]),
            fmt(t[3]),
            fmt(t[4]),
            fmt(t[5])
        ));
    }
    d
}

/// Compact SVG path data for glyph outlines, at the same 0.001 px precision as [`fmt`]: the first move absolute,
/// every later point relative to the one before, a command letter only where it changes (`h` / `v` for axis-aligned
/// lines), no leading zero and no separator before a minus sign. Coordinates are rounded to thousandths first and the
/// deltas taken between the rounded values, so the points read back are exactly the rounded absolute ones (no error
/// accumulates along a path). A text-heavy page's outlines take about 60% of the characters absolute data took.
/// [`crate::boolean_operands::parse_glyph_d`] reads it back.
struct GlyphPathData {
    d: String,
    /// Current point and subpath start, in thousandths of a px.
    at: (i64, i64),
    start: (i64, i64),
    /// The last command letter written (0: none yet).
    cmd: u8,
}

impl GlyphPathData {
    fn new() -> Self {
        Self { d: String::new(), at: (0, 0), start: (0, 0), cmd: 0 }
    }

    fn milli(v: f32) -> i64 {
        (v as f64 * 1000.0).round() as i64
    }

    /// `cmd` unless it is the last one written (then its arguments continue it).
    fn command(&mut self, cmd: u8) {
        if self.cmd != cmd {
            self.d.push(cmd as char);
        }
        self.cmd = cmd;
    }

    /// One number in thousandths, after a separator when it follows another number and doesn't start with `-`.
    fn num(&mut self, k: i64, first: bool) {
        use std::fmt::Write as _;
        if k < 0 {
            self.d.push('-');
        } else if !first {
            self.d.push(' ');
        }
        let a = k.unsigned_abs();
        let (int, frac) = (a / 1000, a % 1000);
        if frac == 0 {
            let _ = write!(self.d, "{int}");
            return;
        }
        if int > 0 {
            let _ = write!(self.d, "{int}");
        }
        let (mut frac, mut digits) = (frac, 3);
        while frac % 10 == 0 {
            frac /= 10;
            digits -= 1;
        }
        let _ = write!(self.d, ".{frac:0digits$}");
    }

    /// The arguments of the command just written: the first needs a separator only when the command letter was
    /// left out (it continues the previous command's numbers).
    fn args(&mut self, continued: bool, nums: &[i64]) {
        for (i, &k) in nums.iter().enumerate() {
            self.num(k, i == 0 && !continued);
        }
    }

    fn move_to(&mut self, x: f32, y: f32) {
        let p = (Self::milli(x), Self::milli(y));
        if self.d.is_empty() {
            // Absolute, and a space even before a minus: readers of a glyph's first point split on letters and spaces.
            self.d.push('M');
            self.num(p.0, true);
            self.d.push(' ');
            self.num(p.1, true);
        } else {
            self.d.push('m');
            self.args(false, &[p.0 - self.at.0, p.1 - self.at.1]);
        }
        // After a move, the next command always writes its letter (implicit arguments of `m` are lines).
        self.cmd = b'm';
        self.at = p;
        self.start = p;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p = (Self::milli(x), Self::milli(y));
        let (dx, dy) = (p.0 - self.at.0, p.1 - self.at.1);
        let (cmd, nums): (u8, &[i64]) = match (dx, dy) {
            (0, 0) => (b'l', &[0, 0]),
            (_, 0) => (b'h', &[dx]),
            (0, _) => (b'v', &[dy]),
            _ => (b'l', &[dx, dy]),
        };
        let continued = self.cmd == cmd;
        self.command(cmd);
        self.args(continued, nums);
        self.at = p;
    }

    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let (c, p) = ((Self::milli(cx), Self::milli(cy)), (Self::milli(x), Self::milli(y)));
        let continued = self.cmd == b'q';
        self.command(b'q');
        self.args(continued, &[c.0 - self.at.0, c.1 - self.at.1, p.0 - self.at.0, p.1 - self.at.1]);
        self.at = p;
    }

    fn close(&mut self) {
        self.d.push('Z');
        self.cmd = b'Z';
        self.at = self.start;
    }
}

/// Glyph records of an analytic op → path data (quadratics, contiguous segments joined; [`GlyphPathData`]).
fn glyphs_to_d(records: &[moda_text::vector::ops::VectorAnalyticRecord], curves: &[f32]) -> String {
    let mut d = GlyphPathData::new();
    for r in records {
        let size = r.font_size_px;
        let cx = (r.em_box[0] + r.em_box[2]) * 0.5 * size;
        let cy = (r.em_box[1] + r.em_box[3]) * 0.5 * size;
        let [a, c, b, dd] = r.glyph_lin;
        // Vertex shader placement: pen + lin·(em·size − C) + C + off.
        let place = |ex: f32, ey: f32| -> (f32, f32) {
            let (px, py) = (ex * size - cx, ey * size - cy);
            (
                r.pen_x + a * px + c * py + cx + r.glyph_off[0],
                r.baseline_y + b * px + dd * py + cy + r.glyph_off[1],
            )
        };
        let start = r.first_curve as usize * 6;
        let end = start + r.curve_count as usize * 6;
        let Some(segs) = curves.get(start..end) else {
            continue;
        };
        let mut open: Option<((f32, f32), (f32, f32))> = None; // (subpath start, current point)
        let close_eps = 1e-3;
        for s in segs.chunks_exact(6) {
            let p0 = place(s[0], s[1]);
            let q = place(s[2], s[3]);
            let p1 = place(s[4], s[5]);
            let continues = matches!(open, Some((_, cur)) if (cur.0 - p0.0).abs() < close_eps && (cur.1 - p0.1).abs() < close_eps);
            if !continues {
                if let Some((st, cur)) = open {
                    if (st.0 - cur.0).abs() < close_eps && (st.1 - cur.1).abs() < close_eps {
                        d.close();
                    }
                }
                d.move_to(p0.0, p0.1);
                open = Some((p0, p0));
            }
            // Lines are stored as quadratics with the control point at the midpoint.
            let mid = ((p0.0 + p1.0) * 0.5, (p0.1 + p1.1) * 0.5);
            if (mid.0 - q.0).abs() < 1e-3 && (mid.1 - q.1).abs() < 1e-3 {
                d.line_to(p1.0, p1.1);
            } else {
                d.quad_to(q.0, q.1, p1.0, p1.1);
            }
            if let Some((st, _)) = open {
                open = Some((st, p1));
            }
        }
        if let Some((st, cur)) = open {
            if (st.0 - cur.0).abs() < close_eps && (st.1 - cur.1).abs() < close_eps {
                d.close();
            }
        }
    }
    d.d
}

/// A text gradient's shader uniforms as the Moda paint they came from (node 0-100 box space; stop colours `rgba()`):
/// `{ type: 'linear', start, end, stops }` or `{ type: 'radial', center, radiusX, radiusY, rotation, stops }`.
fn gradient_paint_json(g: &moda_text::ops::TextGradientUniforms) -> moda_scene::value::Value {
    use moda_scene::value::json;
    let count = (g.config[0].max(0.0) as usize).min(g.stops.len());
    let stops: Vec<moda_scene::value::Value> = g.stops[..count]
        .iter()
        .map(|&[offset, r, gr, packed]| {
            // `floor(b·255) + a·0.99999` (packTextGradientUniformFloats).
            let b = packed.floor();
            let a = ((packed - b) / 0.99999).clamp(0.0, 1.0);
            let a = (a * 1000.0).round() / 1000.0;
            let c = |v: f32| (v * 255.0).round().clamp(0.0, 255.0) as u8;
            json!({
                "offset": offset,
                "color": format!("rgba({}, {}, {}, {})", c(r), c(gr), b.clamp(0.0, 255.0) as u8, a),
            })
        })
        .collect();
    let [p0, p1, p2, p3] = g.params;
    if g.config[1] == 2.0 {
        json!({
            "type": "radial",
            "center": { "x": p0, "y": p1 },
            "radiusX": p2,
            "radiusY": p3,
            "rotation": g.config[2].to_degrees(),
            "stops": stops,
        })
    } else {
        json!({ "type": "linear", "start": { "x": p0, "y": p1 }, "end": { "x": p2, "y": p3 }, "stops": stops })
    }
}

/// The vector geometry of one richtext node, or why it can't be expressed as flat fills.
pub(crate) fn node_text_geometry(scene: &Scene, node_id: &str) -> Result<TextGeometry, String> {
    let node = scene
        .effective_node(node_id)
        .ok_or_else(|| format!("no node {node_id}"))?;
    if node.node_type() != "richtext" {
        return Err(format!("{node_id} is a {}, not richtext", node.node_type()));
    }
    let mut spec = crate::text_draw::node_to_text_spec_in(scene, node);
    crate::text_area::apply_to_spec(scene, node, &mut spec);
    if spec.gradient.is_some() {
        return Err("gradient text".into());
    }
    let (store, host) = crate::text_draw::vector_store_and_host().map_err(|e| format!("{e:#}"))?;
    let ops = match moda_text::vector::build_vector_node_ops(
        &spec,
        &store,
        &host.catalog,
        1.0,
        VectorPlacement::VectorParity,
    ) {
        VectorEngineDecision::Vector(ops) => ops,
        VectorEngineDecision::Fallback { reason, .. } => {
            return Err(format!("not vector text: {reason}"))
        }
    };
    if !ops.emojis.is_empty() {
        // Colour emoji are images, not outlines: the exporter rasterizes the node.
        return Err("colour emoji".into());
    }
    let fills = fills_from_ops(&ops, FillMode::Strict)?;
    // Type on a path (F7.2): the runs' straight positions don't say where the glyphs are; exporters that keep text
    // live write the path's own text instead (SVG `<textPath>`), or the outlines.
    let runs = if spec.text_path.is_some() {
        None
    } else {
        extract_vector_text_runs(
            &RichTextNodeSpec {
                node_id: node.id().to_string(),
                html_content: spec.html.to_string(),
                width: spec.width,
                height: spec.height,
                paragraph_spacing: spec.paragraph_spacing,
                line_height_policy: spec.line_height_policy,
                vertical_align: spec.vertical_align.clone(),
                truncation: spec.truncation,
                auto_width: spec.auto_width,
                area: spec.area.clone(),
            },
            &store,
            &host.catalog,
        )
    };
    Ok(TextGeometry {
        node_id: node_id.to_string(),
        fills,
        runs,
    })
}

/// How [`fills_from_ops`] treats what flat fills can't express.
#[derive(Clone, Copy, PartialEq)]
enum FillMode {
    /// Decline effect passes, shader fills and per-glyph opacity.
    Strict,
    /// Leave the effect passes out and draw the rest as flat fills, ignoring per-glyph opacity.
    Lenient,
    /// As `Lenient`, but a glyph's own opacity (a `text.units` unit's) splits the line's fill and scales its
    /// opacity; a glyph at opacity 0 is left out.
    GlyphOpacity,
}

/// The ops' fills as path data, per [`FillMode`].
fn fills_from_ops(
    ops: &moda_text::vector::ops::VectorNodeOps,
    mode: FillMode,
) -> Result<Vec<TextGeometryFill>, String> {
    let strict = mode == FillMode::Strict;
    let mut fills = Vec::new();
    for op in &ops.ops {
        if op.blur_px > 0.0 || op.stroke_px > 0.0 || op.additive {
            if strict {
                return Err("text effect pass".into());
            }
            continue;
        }
        match op.pipeline {
            VectorOpPipeline::Tess => {
                if op.tris.is_empty() {
                    continue;
                }
                fills.push(TextGeometryFill {
                    kind: "shape",
                    paths: vec![triangles_to_d(&op.tris)],
                    color: op.color,
                    opacity: op.opacity,
                    gradient: None,
                });
            }

            VectorOpPipeline::Analytic
            | VectorOpPipeline::AnalyticGradient
            => {
                // A gradient op here is a text range's own gradient (a node gradient declined above).
                let gradient = match (op.pipeline, &op.gradient) {
                    (VectorOpPipeline::AnalyticGradient, Some(g)) => Some(gradient_paint_json(g)),
                    (VectorOpPipeline::AnalyticGradient, None) => {
                        return Err("gradient text fill".into())
                    }
                    _ => None,
                };
                if strict && op.records.iter().any(|r| r.glyph_opacity < 1.0) {
                    return Err("per-glyph opacity".into());
                }
                // A path per glyph, a fill per line (like Illustrator's Create Outlines): one compound path for a
                // whole paragraph is heavy to edit and to re-import.
                let by_opacity = mode == FillMode::GlyphOpacity;
                for line in op.records.chunk_by(|a, b| {
                    a.baseline_y == b.baseline_y && (!by_opacity || a.glyph_opacity == b.glyph_opacity)
                }) {
                    let glyph_opacity = if by_opacity { line[0].glyph_opacity.clamp(0.0, 1.0) } else { 1.0 };
                    if glyph_opacity <= 0.0 {
                        continue;
                    }
                    let paths: Vec<String> = line
                        .iter()
                        .map(|r| glyphs_to_d(std::slice::from_ref(r), &ops.curve_points))
                        .filter(|d| !d.is_empty())
                        .collect();
                    if paths.is_empty() {
                        continue;
                    }
                    fills.push(TextGeometryFill {
                        kind: "glyphs",
                        paths,
                        color: op.color,
                        opacity: op.opacity * glyph_opacity,
                        gradient: gradient.clone(),
                    });
                }
            }
        }
    }
    Ok(fills)
}

/// The glyph fills a solid sweeps (`solid::outline`, `shape: 'extrude'`): node-local px, each glyph where its
/// `text.units` transform puts it, a fill per line and glyph opacity (a unit fading in extrudes at its opacity; one
/// at opacity 0 is left out). Colour emoji (images) and the effect passes (shadows, glows, strokes) have no outline
/// to sweep and are left out, so the rest of the text still extrudes; a gradient fill declines.
pub(crate) fn node_solid_fills(scene: &Scene, node_id: &str) -> Result<Vec<TextGeometryFill>, String> {
    let node = scene
        .effective_node(node_id)
        .ok_or_else(|| format!("no node {node_id}"))?;
    if node.node_type() != "richtext" {
        return Err(format!("{node_id} is a {}, not richtext", node.node_type()));
    }
    let mut spec = crate::text_draw::node_to_text_spec_in(scene, node);
    crate::text_area::apply_to_spec(scene, node, &mut spec);
    if spec.gradient.is_some() {
        return Err("gradient text".into());
    }
    let (store, host) = crate::text_draw::vector_store_and_host().map_err(|e| format!("{e:#}"))?;
    match moda_text::vector::build_vector_node_ops(&spec, &store, &host.catalog, 1.0, VectorPlacement::VectorParity) {
        VectorEngineDecision::Vector(ops) => fills_from_ops(&ops, FillMode::GlyphOpacity),
        VectorEngineDecision::Fallback { reason, .. } => Err(format!("not vector text: {reason}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    fn scene(nodes: moda_scene::value::Value, ids: Vec<&str>) -> Scene {
        Scene::load(&json!({
            "nodes": nodes,
            "pages": [{"id":"p1","name":"P","width":800.0,"height":600.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}],"nodeIds": ids}]
        })).unwrap()
    }

    /// A solid (extrusion) sweeps what a unit shows: a unit fading in extrudes at its opacity, a hidden one is left
    /// out, and emoji (images, no outline) are skipped while the rest of the text still extrudes. Before, any unit
    /// under opacity 1 or any emoji made the whole layer's extrusion draw nothing.
    #[test]
    fn solid_fills_follow_unit_opacity_and_skip_emoji() {
        let text = |html: &str, units: moda_scene::value::Value| {
            scene(
                json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":400.0,"height":100.0,
                    "htmlContent": html, "unitTransforms": units}]),
                vec!["t"],
            )
        };
        let html = "<p><span style=\"font-family: Inter; font-size: 40px; color: #ff8800;\">OPAC</span></p>";
        let s = text(html, json!([{"opacity": 0.5}, {"opacity": 0.0}, {}, {}]));
        assert!(node_text_geometry(&s, "t").is_err(), "flat outlines still decline per-glyph opacity");
        let fills = node_solid_fills(&s, "t").expect("a solid takes per-glyph opacity");
        let glyphs: Vec<(usize, f32)> = fills.iter().filter(|f| f.kind == "glyphs").map(|f| (f.paths.len(), f.opacity)).collect();
        assert_eq!(glyphs, vec![(1, 0.5), (2, 1.0)], "O at half, P left out, AC whole");
        let at_rest = node_solid_fills(&text(html, json!([])), "t").unwrap();
        assert_eq!(at_rest.iter().map(|f| f.paths.len()).sum::<usize>(), 4);

        let emoji = "<p><span style=\"font-family: Inter; font-size: 40px; color: #ff8800;\">GO 🚀</span></p>";
        let s = text(emoji, json!([]));
        assert!(node_text_geometry(&s, "t").is_err(), "flat outlines still decline emoji");
        let fills = node_solid_fills(&s, "t").expect("a solid skips emoji");
        assert_eq!(fills.iter().map(|f| f.paths.len()).sum::<usize>(), 2, "G and O");
    }

    /// The SVG / PDF exports' decorations come from here: a styled underline is the shape its style draws (a wave,
    /// dots, dashes), in its own colour, not a plain bar in the text's.
    #[test]
    fn decoration_styles_reach_the_shape_fills() {
        let fill_for = |css: &str| {
            let html = format!(
                "<p><span style=\"font-family: Inter; font-size: 24px; color: #112233; {css}\">Hello world</span></p>"
            );
            let s = scene(
                json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":400.0,"height":60.0,"htmlContent": html}]),
                vec!["t"],
            );
            let g = node_text_geometry(&s, "t").expect("vector text");
            let shapes: Vec<_> = g.fills.into_iter().filter(|f| f.kind == "shape").collect();
            assert_eq!(shapes.len(), 1, "{css}: one decoration colour");
            shapes.into_iter().next().unwrap()
        };
        let subpaths = |d: &str| d.matches('M').count();
        let plain = fill_for("text-decoration: underline");
        assert_eq!(subpaths(&plain.paths[0]), 2, "a solid underline is one quad (two triangles)");
        assert_eq!(plain.color, [0x11 as f32 / 255.0, 0x22 as f32 / 255.0, 0x33 as f32 / 255.0, 1.0]);
        let wavy = fill_for("text-decoration: underline wavy #ff0000 2px");
        assert_eq!(wavy.color, [1.0, 0.0, 0.0, 1.0]);
        assert!(subpaths(&wavy.paths[0]) > 40, "a wave is a strip of many quads");
        let dotted = fill_for("text-decoration: underline dotted");
        assert_eq!(
            subpaths(&dotted.paths[0]) % moda_text::decorations::DOT_SEGMENTS,
            0,
            "dots are fans of {} triangles",
            moda_text::decorations::DOT_SEGMENTS
        );
        let dashed = fill_for("text-decoration: line-through dashed");
        assert!(subpaths(&dashed.paths[0]) >= 2 * 8, "dashes: a quad each");
    }

    #[test]
    fn compact_glyph_path_data_form() {
        let mut d = GlyphPathData::new();
        d.move_to(1.0, 2.0);
        d.line_to(3.5, -4.0);
        d.line_to(4.5, -3.0);
        d.line_to(4.0, -3.0);
        d.line_to(4.0, -1.0);
        d.quad_to(5.0, 0.0, 6.0, -1.0);
        d.quad_to(7.0, -2.0, 8.0, -1.0);
        d.close();
        d.move_to(2.0, 3.0);
        d.line_to(4.0, 5.0);
        d.move_to(-0.0004, -2.25);
        assert_eq!(d.d, "M1 2l2.5-6 1 1h-.5v2q1 1 2 0 1-1 2 0Zm1 1l2 2m-4-7.25");
        let mut first = GlyphPathData::new();
        first.move_to(0.25, -1.5);
        assert_eq!(first.d, "M.25 -1.5");
    }

    /// Compact glyph path data reads back as exactly the absolute points rounded to 0.001 px, however long the path.
    #[test]
    fn compact_glyph_path_data_round_trips_to_the_rounded_points() {
        use moda_scene::blend::BlendPathPoint as P;
        let mut d = GlyphPathData::new();
        let mut want = Vec::new();
        let r = |v: f32| (v as f64 * 1000.0).round() / 1000.0;
        let mut x = 12.3456f32;
        for k in 0..400 {
            let (px, py) = (x, 40.0 - (k % 7) as f32 * 0.37);
            if k % 50 == 0 {
                if k > 0 {
                    d.close();
                }
                d.move_to(px, py);
                want.push(format!("M{},{}", r(px), r(py)));
            } else if k % 3 == 0 {
                d.quad_to(px - 0.4, py + 1.0, px, py);
                want.push(format!("Q{},{} {},{}", r(px - 0.4), r(py + 1.0), r(px), r(py)));
            } else {
                d.line_to(px, if k % 5 == 0 { 40.0 } else { py });
                want.push(format!("L{},{}", r(px), r(if k % 5 == 0 { 40.0 } else { py })));
            }
            x += if k % 4 == 0 { 0.0 } else { 0.1234 };
        }
        d.close();
        assert!(d.d.starts_with("M12.346 40"));
        let mut out = Vec::new();
        crate::boolean_operands::parse_glyph_d(&d.d, &mut out);
        let got: Vec<String> = out
            .iter()
            .map(|p| {
                let r3 = |v: f64| (v * 1000.0).round() / 1000.0;
                match p {
                    P::Move { x, y } => format!("M{},{}", r3(*x), r3(*y)),
                    P::Line { x, y } => format!("L{},{}", r3(*x), r3(*y)),
                    P::Quadratic { x, y, cp } => format!("Q{},{} {},{}", r3(cp.0), r3(cp.1), r3(*x), r3(*y)),
                    _ => "?".into(),
                }
            })
            .collect();
        assert_eq!(got, want);
        // Every point within float noise of its rounded value: no drift along the path.
        for p in &out {
            if let P::Line { x, y } | P::Move { x, y } = p {
                assert!((x * 1000.0 - (x * 1000.0).round()).abs() < 1e-6 && (y * 1000.0 - (y * 1000.0).round()).abs() < 1e-6);
            }
        }
    }

    /// F7.2 type on a path: the glyphs run along the path (here a closed circle), not along one baseline in the box.
    #[test]
    fn path_text_glyphs_follow_the_path() {
        let k = 0.552_284_75 * 90.0;
        let pts = json!([
            {"x":180.0,"y":90.0,"type":"move"},
            {"x":90.0,"y":180.0,"type":"curve","cp1":{"x":180.0,"y":90.0+k},"cp2":{"x":90.0+k,"y":180.0}},
            {"x":0.0,"y":90.0,"type":"curve","cp1":{"x":90.0-k,"y":180.0},"cp2":{"x":0.0,"y":90.0+k}},
            {"x":90.0,"y":0.0,"type":"curve","cp1":{"x":0.0,"y":90.0-k},"cp2":{"x":90.0-k,"y":0.0}},
            {"x":180.0,"y":90.0,"type":"curve","cp1":{"x":90.0+k,"y":0.0},"cp2":{"x":180.0,"y":90.0-k}}
        ]);
        let s = scene(
            json!([{"id":"t","type":"richtext","x":40.0,"y":30.0,"width":180.0,"height":180.0,
                "rotation":0,"scale":1,"resizeMode":"no-resize",
                "textPath":{"points":pts,"closed":true,"boxWidth":180.0,"boxHeight":180.0},
                "htmlContent":"<p><span style=\"font-family: Inter; font-size: 24px; color: #111111;\">Hello design world</span></p>"}]),
            vec!["t"],
        );
        let g = node_text_geometry(&s, "t").expect("vector text");
        let glyphs: Vec<&String> = g
            .fills
            .iter()
            .filter(|f| f.kind == "glyphs")
            .flat_map(|f| f.paths.iter())
            .collect();
        assert_eq!(glyphs.len(), 16, "a path per visible glyph");
        // Each glyph's first point: along a circle they spread over the box, not along one line of text.
        let first = |d: &str| -> (f32, f32) {
            let n: Vec<f32> = d[1..]
                .split(|c: char| c.is_ascii_alphabetic() || c == ' ')
                .filter_map(|t| t.parse().ok())
                .take(2)
                .collect();
            (n[0], n[1])
        };
        let ys: Vec<f32> = glyphs.iter().map(|d| first(d).1).collect();
        let span = ys.iter().cloned().fold(f32::MIN, f32::max)
            - ys.iter().cloned().fold(f32::MAX, f32::min);
        assert!(span > 60.0, "glyphs on one baseline: {ys:?}");
        assert!(g.runs.is_none(), "path text has no straight runs");
    }

}
