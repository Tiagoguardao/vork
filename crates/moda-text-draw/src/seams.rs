//! seams.rs — the text half of moda-engine's cross-track interface contract (`TextQuadBatch`, `DegradedQuad`),
//! excerpted; see PROVENANCE.md.

pub use moda_text::ops::{TextOp, TextOpPipeline};

/// The draws of one richtext/shapeText/connectorText item: the vector glyph ops (`vector`), or the degraded
/// raster (`degraded`), plus the decoration and emoji ops (`ops`). `--dump-text-quads` serializes them as JSON;
/// `moda_engine::text_draw` renders them.
#[derive(Clone)]
pub struct TextQuadBatch {
    pub node_id: String,
    /// Render z (fractional: node z, or shape z + SHAPE_TEXT_Z_OFFSET).
    pub z: f64,
    /// Column-major model matrix (same construction as shapes).
    pub model_matrix: [f32; 16],
    pub page_clip_bounds: [f32; 4],
    /// The ordered non-glyph op stream (emoji quads, decorations). Draw order == blend order; every op shares
    /// the batch depth.
    pub ops: Vec<TextOp>,
    pub emoji_engine: Option<std::sync::Arc<moda_text::degraded::DegradedEngine>>,
    /// Raster quad for fallback text. When set, `ops` is empty and the
    /// executor draws ONE textured quad via the extracted `text-emoji`
    /// image module (the production Canvas2D display pipeline,
    /// `richtext/shaders/richtext.wgsl.ts`, is not part of the extracted
    /// export surface — its edge-alpha-gamma compensation is a recorded
    /// Degraded-tier divergence inside the 0.90–0.97 band).
    pub degraded: Option<DegradedQuad>,
    /// The vector text engine ops. When set, the executor draws the ordered
    /// `VectorNodeOps` stream through the extracted vector pipelines
    /// (`vector-text-tess` / `vector-text-analytic` /
    /// `vector-text-analytic-additive` (glow, task #142) /
    /// `vector-text-analytic-gradient`). `VectorNodeOps::gradient`
    /// carries the gradient uniforms (contiguous 176-byte binding 3).
    pub vector: Option<moda_text::vector::VectorNodeOps>,
    /// Additive: the node's `distort` warp as the local-px 3×3
    /// homography rows packed for the vector `DrawUniforms` distort rows —
    /// row-major `[h11, h12, h13, h21, h22, h23, h31, h32, h33]`. None ⇒
    /// identity rows with the enabled flag 0. Vector-engine draws only (the
    /// Degraded tier stays unwarped — the accepted fallback gap).
    pub distort: Option<[f32; 9]>,
    /// vork: what this batch draws, as one number: the content key its plan cache keeps it under (text hash,
    /// opacity, render scale, model scale band, font host generation; moda-engine `text_draw::text_cache_key`).
    /// Two batches with the same key and the same placement (z, model, clip) draw the same pixels, whichever
    /// allocation holds them, so a consumer that must know "the same text as before" names a batch by this and
    /// its placement, not by its address (a layer capture's content key, moda-engine
    /// `residency::fold_item_geometry`). `None`: built outside a plan cache; such a batch has only its address.
    ///
    /// It is a name for the fields above, not one more of them: two batches are equal, and print alike, whether
    /// or not either carries it (a plan built through a cache equals the same plan built without one).
    pub content_key: Option<u64>,
    /// Host-certified identity of unchanged vector geometry, for uniform-only paint writes.
    pub paint_geometry_key: Option<[u64; 4]>,
}

impl PartialEq for TextQuadBatch {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
            && self.z == other.z
            && self.model_matrix == other.model_matrix
            && self.page_clip_bounds == other.page_clip_bounds
            && self.ops == other.ops
            && self.emoji_engine.as_ref().map(|e| e.id()) == other.emoji_engine.as_ref().map(|e| e.id())
            && self.degraded == other.degraded
            && self.vector == other.vector
            && self.distort == other.distort
    }
}

impl std::fmt::Debug for TextQuadBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextQuadBatch")
            .field("node_id", &self.node_id)
            .field("z", &self.z)
            .field("model_matrix", &self.model_matrix)
            .field("page_clip_bounds", &self.page_clip_bounds)
            .field("ops", &self.ops)
            .field("degraded", &self.degraded)
            .field("vector", &self.vector)
            .field("distort", &self.distort)
            .finish()
    }
}

/// Placement + raster for one Degraded-tier text quad (M4d). Mirrors the
/// Canvas2D display-quad math (`richtext/transforms.ts:77-117`): quad size
/// is the texture's logical size; `offset` is the alignment offset
/// ((node − texture) × {0,½,1} per detected horizontal align /
/// `verticalAlign`), node-local; `opacity` = node opacity × the raster's
/// `fill_alpha` (uniform-alpha quirk).
#[derive(Debug, Clone, PartialEq)]
pub struct DegradedQuad {
    pub raster: std::sync::Arc<moda_text::degraded::DegradedRaster>,
    pub offset: [f32; 2],
    pub opacity: f32,
}

/// A projective warp can send part of a visible glyph quad through infinity.
pub enum TextBounds {
    Empty,
    Finite([f64; 4]),
    CrossesHorizon,
}

impl TextQuadBatch {
    pub fn quad_padding(&self) -> f64 {
        self.vector.as_ref().and_then(|v| v.uniform_padding).map_or(0., |params| Self::matrix_padding(&self.model_matrix, params))
    }

    pub fn matrix_padding(matrix: &[f32; 16], (font, scale): (f64, f64)) -> f64 {
        let axis = crate::text_draw::model_axis_scales(matrix)[0];
        (moda_text::vector::ops::ANALYTIC_AA_PAD_DEVICE_PX / (font.max(1e-3) * (axis * scale).max(1e-3))).min(1.)
    }

    pub fn padded_box(bounds: [f32; 4], pad: f64) -> [f32; 4] {
        // Match paddedBound in vector-text-bounds.wgsl, including its midpoint
        // decision on the two binary32 words actually delivered to the GPU.
        fn displace(value: f32, high: f32, tail: f32) -> f32 {
            let (large, small) = if value.abs() < high.abs() { (high, value) } else { (value, high) };
            let sum = large + small;
            let rounded = sum - large;
            let error = small - rounded;
            let correction = error + tail;
            let result = sum + correction;
            if correction == error && tail != 0.0 && error != 0.0 && sum == result {
                let neighbor = if error > 0.0 { result.next_up() } else { result.next_down() };
                if error.abs() == (neighbor - result).abs() * 0.5 && error.is_sign_positive() == tail.is_sign_positive() {
                    return neighbor;
                }
            }
            result
        }
        let [high, tail] = crate::text_draw::padding_words(pad);
        let [a, b, c, d] = bounds;
        [displace(a, -high, -tail), displace(b, -high, -tail),
         displace(c, high, tail), displace(d, high, tail)]
    }

    /// The local-space quads this batch draws, including glyph transforms, strokes and raster padding.
    pub fn local_bounds(&self) -> Option<[f64; 4]> {
        match self.painted_local_bounds() {
            TextBounds::Finite(rect) => Some(rect),
            TextBounds::Empty | TextBounds::CrossesHorizon => None,
        }
    }

    pub fn painted_local_bounds(&self) -> TextBounds {
        let mut bounds: Option<[f64; 4]> = None;
        let mut point = |x: f64, y: f64| {
            bounds = Some(match bounds {
                None => [x, y, x, y],
                Some(b) => [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)],
            });
        };
        if let Some(vector) = &self.vector {
            // Vector shaders warp the placed glyph or decoration before applying
            // the node's model matrix. Bitmap emoji and degraded quads use their
            // ordinary placement. A quad crossing w=0 is bounded by the draw clip.
            let mut crosses_horizon = false;
            let mut vector_point = |x: f64, y: f64| {
                if let Some(h) = self.distort {
                    let (x, y) = (x as f32, y as f32);
                    let w = h[6] * x + h[7] * y + h[8];
                    let u = (h[0] * x + h[1] * y + h[2]) / w;
                    let v = (h[3] * x + h[4] * y + h[5]) / w;
                    if w > 0. && u.is_finite() && v.is_finite() {
                        point(f64::from(u), f64::from(v));
                    } else {
                        crosses_horizon = true;
                    }
                } else {
                    point(x, y);
                }
            };
            let pad = self.quad_padding();
            for op in &vector.ops {
                for xy in op.tris.chunks_exact(2) { vector_point(xy[0].into(), xy[1].into()); }
                for glyph in op.records.iter() {
                    if glyph.glyph_opacity <= 0. { continue; }
                    let em_box = Self::padded_box(glyph.em_box, pad);
                    if glyph.local_aa_px > 0. {
                        for [x, y] in glyph.quad_corners(em_box) {
                            vector_point(f64::from(x), f64::from(y));
                        }
                        continue;
                    }
                    let [x0, y0, x1, y1] = em_box.map(f64::from);
                    let size = f64::from(glyph.font_size_px);
                    let cx = f64::from(glyph.pen_x) + (x0 + x1) * size * 0.5;
                    let cy = f64::from(glyph.baseline_y) + (y0 + y1) * size * 0.5;
                    let [a, c, b, d] = glyph.glyph_lin.map(f64::from);
                    for (x, y) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)] {
                        let dx = (x - (x0 + x1) * 0.5) * size;
                        let dy = (y - (y0 + y1) * 0.5) * size;
                        vector_point(cx + a * dx + c * dy + f64::from(glyph.glyph_off[0]),
                            cy + b * dx + d * dy + f64::from(glyph.glyph_off[1]));
                    }
                }
            }
            if crosses_horizon { return TextBounds::CrossesHorizon; }
            for emoji in &vector.emojis {
                for xy in emoji.corners.chunks_exact(2) { point(xy[0].into(), xy[1].into()); }
            }
        }
        for op in &self.ops {
            for xyuv in op.verts.chunks_exact(4) { point(xyuv[0].into(), xyuv[1].into()); }
        }
        if let Some(quad) = &self.degraded {
            let [x, y] = quad.offset.map(f64::from);
            point(x, y);
            point(x + quad.raster.logical_w, y + quad.raster.logical_h);
        }
        bounds.map_or(TextBounds::Empty, TextBounds::Finite)
    }
}
