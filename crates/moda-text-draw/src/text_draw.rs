//! text_draw.rs — text rendering (consumes `seams::TextQuadBatch`).
//!
//! Two halves:
//!
//! 1. **Plan side** (no GPU): `build_richtext_item` / `build_shape_text_item` / `plan_connector_text` lay a text
//!    node out on the vector lane (`moda_text::vector::build_vector_node_ops`: glyph outlines drawn analytically)
//!    and emit `DrawItem::Text(TextQuadBatch)`. A node the vector lane declines (no usable font for a run, a
//!    shaping failure) falls to the degraded tier: one cosmic-text raster quad (`build_degraded_batch`).
//!
//! 2. **GPU side**: `TextDraw` builds the extracted text pipelines (`vector-text-*`, `<id>.vector-text-fill`,
//!    `text-decoration`, `text-emoji` — blend/depth/vertex state from the manifest descriptors, never hand-coded),
//!    uploads glyph curves / emoji cells / degraded rasters, packs a 256-aligned uniform arena (model/color/
//!    params/clip per draw — the production dynamic-offset ring), and draws batches inside the transparent bucket
//!    loop AFTER paths (`renderTransparentItemsWithText`).
//!
//! Fonts come from a `moda_text::TextHost` — vendored bytes under `fixtures/oracle/fonts` natively (env
//! `MODA_FONTS_DIR` overrides), the installed VFS on wasm.

use crate::records::{Column, PendingRecords};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use moda_text::lru::ByteLruCache;
use moda_text::ops::{TextNodeSpec, TextOp, TextOpPipeline};
use moda_text::TextHost;

use crate::device::GpuContext;
use crate::seams::TextQuadBatch;

/// Local-to-screen scales shared with the prepared-batch cache. Outline AA uses
/// the shorter axis and scalable glyph rasters sample at the longer axis.
pub fn model_axis_scales(model_matrix: &[f32; 16]) -> [f64; 2] {
    let sx = f64::from(model_matrix[0]).hypot(f64::from(model_matrix[1]));
    let sy = f64::from(model_matrix[4]).hypot(f64::from(model_matrix[5]));
    [sx.min(sy), sx.max(sy)]
}

/// Size of the vector-text `DrawUniforms` WGSL struct: model(64) + color(16)
/// + params(16) + pageClip(16) + distort0/1/2(48, the warp rows)
/// = 160 bytes. The per-draw packer and the bind-group binding size must both
/// match the struct or wgpu rejects every vector-text draw.
const VECTOR_DRAW_UNIFORM_BYTES: u64 = 160;

/// Each image op indexes the placed glyph that owns its face, raster scale and effect.
fn vector_emoji_ops(emojis: &[moda_text::vector::VectorEmoji]) -> Vec<TextOp> {
    let mut ops = Vec::with_capacity(emojis.len());
    for (index, e) in emojis.iter().enumerate() {
        let c = e.corners;
        ops.push(TextOp {
            pipeline: TextOpPipeline::Emoji { emoji: index as u32 },
            color: [1.0, 1.0, 1.0, 1.0],
            opacity: e.opacity,
            blur_px: 0.0,
            stroke_px: 0.0,
            #[rustfmt::skip]
            verts: vec![
                c[0], c[1], 0.0, 0.0,
                c[2], c[3], 1.0, 0.0,
                c[4], c[5], 1.0, 1.0,
                c[6], c[7], 0.0, 1.0,
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
        });
    }
    ops
}

fn emoji_module(b: &TextQuadBatch, emoji: u32) -> &'static str {
    let additive = b.vector.as_ref().and_then(|v| v.emojis.get(emoji as usize))
        .and_then(|e| e.effect.as_ref()).is_some_and(|fx| fx.additive);
    if additive {
        "text-emoji-additive"
    } else {
        "text-emoji"
    }
}

/// Vector lane for one planned text node (`--text-engine vector`): the
/// browser's default `renderText` engine routing with vector text enabled
/// (`WebGPUTextSystem.ts:385-399`) — try the vector path, and on ANY of its
/// per-node fallbacks (selectTextEngine ≠ vector, emoji runs, unresolved
/// variants, missing TTFs, uncovered chunks, shaping/outline throws) return
/// `None` so the caller re-plans through the MSDF lane, exactly like
/// `renderWithVector` returning false → `renderWithMsdf`.
///
/// `render_scale` feeds `analyticPadScale` (WebGPUTextSystem.ts:712-715):
/// min real local→screen axis scale — min(model sx, sy) × export scale
/// (viewZoom = 1, DPR = export pixelRatio on this path).
/// `fall_label` names the lane the caller falls back to on a vector decline,
/// for the resource-report event: `"vector→msdf"` on the page-global vector
/// lane (`renderWithVector` → false → `renderWithMsdf`), `"hybrid→degraded"`
/// on the hybrid fallback arm (task #128 — the decline lands on the cosmic
/// Degraded tier, exactly as without the hybrid flag).
///
/// `placement` (FIX r2, Q-120): the page-global vector lane keeps
/// `VectorParity` (browser vector-text truth); the hybrid arm passes
/// `Canvas2dParity` so middle/bottom-aligned fallback-class nodes sit where
/// the default-config golden's Canvas2D raster sits.
#[allow(clippy::too_many_arguments)]
pub fn try_build_vector_batch(
    store: &moda_text::vector::VectorFontStore,
    host: &TextHost,
    spec: &TextNodeSpec,
    model_matrix: [f32; 16],
    z: f64,
    page_clip: [f32; 4],
    render_scale: f64,
    placement: moda_text::vector::VectorPlacement,
) -> Result<std::result::Result<TextBuildOutcome, String>> {
    let layout = match moda_text::vector::accept_vector_layout(spec, store, &host.catalog, placement, false, Some(host)) {
        Ok(layout) => layout,
        // The caller draws the node through the degraded tier (`vector_or_degraded`) or refuses it with this reason.
        Err(moda_text::vector::VectorDecline { reason, cause, families, .. }) => {
            return Ok(Err(format!("{reason} ({cause}; families {families:?})")))
        }
    };
    build_vector_batch(&layout, store, host, spec, model_matrix, z, page_clip, render_scale)
}

/// One draw of a text node from its accepted layout (`moda_text::vector::VectorLayout`): `spec` gives what the
/// draw adds to the layout (opacity, paint, stroke, effects, KineticText units), the model matrix and
/// `render_scale` its antialiasing pad. The inner error is the vector lane's decline of this draw.
#[allow(clippy::too_many_arguments)]
pub fn build_vector_batch(
    layout: &moda_text::vector::VectorLayout,
    store: &moda_text::vector::VectorFontStore,
    host: &TextHost,
    spec: &TextNodeSpec,
    model_matrix: [f32; 16],
    z: f64,
    page_clip: [f32; 4],
    render_scale: f64,
) -> Result<std::result::Result<TextBuildOutcome, String>> {
    let [min, max] = model_axis_scales(&model_matrix);
    let font = layout.uniform_padding_font(spec);
    let pad_scale = if font.is_some() { f64::INFINITY } else { (min * render_scale).max(1e-6) };
    let mut ops = match layout.draw_ops(spec, store, &host.catalog, pad_scale, max * render_scale, Some(host)) {
        Ok(ops) => ops,
        Err(e) => {
            let families: Vec<String> = Vec::new();
            return Ok(Err(format!("vector-error: {} (vector-error; families {families:?})", e.message())));
        }
    };
    ops.uniform_padding = font.map(|f| (f, render_scale));
    if ops.ops.is_empty() && ops.emojis.is_empty() {
        // "nothing visible, but handled (don't fall back)"
        // (drawRunsAnalytic:582) — consume the node, draw nothing.
        return Ok(Ok(TextBuildOutcome::Empty));
    }
    let emoji_ops = vector_emoji_ops(&ops.emojis);
    Ok(Ok(TextBuildOutcome::Item(std::sync::Arc::new(TextQuadBatch {
        emoji_engine: (!ops.emojis.is_empty()).then(|| host.canvas_engine().clone()),
        node_id: spec.node_id.clone(),
        z,
        model_matrix,
        page_clip_bounds: page_clip,
        ops: emoji_ops,
        degraded: None,
        vector: Some(ops),
        distort: spec.distort,
        content_key: None, paint_geometry_key: None,
    }))))
}

/// A text node's draw: the vector lane (glyph outlines), or, when it declines (complex script, a glyph no
/// fallback face covers), the degraded raster tier with the bundled fallback set.
#[allow(clippy::too_many_arguments)]
pub fn vector_or_degraded(
    store: &moda_text::vector::VectorFontStore,
    host: &TextHost,
    engine: &moda_text::degraded::DegradedEngine,
    spec: &TextNodeSpec,
    model: [f32; 16],
    z: f64,
    clip: [f32; 4],
    render_scale: f64,
) -> Result<TextBuildOutcome> {
    if let Ok(out) = try_build_vector_batch(
        store,
        host,
        spec,
        model,
        z,
        clip,
        render_scale,
        moda_text::vector::VectorPlacement::VectorParity,
    )? {
        return Ok(out);
    }
    build_degraded_batch(engine, spec, model, z, clip, render_scale).or_else(|e| {
        Ok(TextBuildOutcome::Degraded {
            reason: format!("vector declined; degraded raster failed: {e}"),
        })
    })
}

/// Build the raster batch for fallback text. Placement mirrors the Canvas2D display quad
/// (`richtext/transforms.ts:77-117`): quad = texture logical size, alignment
/// offsets against the node box, opacity × the raster's uniform-alpha
/// `fill_alpha`.
///
/// The raster is as fine as the output ([`moda_text::degraded::output_raster_scale`]): `render_scale` px per
/// canvas unit × the model's MIN axis scale.
pub fn build_degraded_batch(
    engine: &moda_text::degraded::DegradedEngine,
    spec: &TextNodeSpec,
    model_matrix: [f32; 16],
    z: f64,
    page_clip: [f32; 4],
    render_scale: f64,
) -> Result<TextBuildOutcome> {
    let raster = engine
        .render_node_at(spec, render_scale * model_axis_scales(&model_matrix)[0])
        .map_err(|e| anyhow!("degraded raster: {e}"))?;
    let (offset_x, offset_y) = moda_text::degraded::raster_quad_offset(spec, &raster);
    let opacity = (spec.opacity as f32) * raster.fill_alpha;
    Ok(TextBuildOutcome::Item(std::sync::Arc::new(TextQuadBatch {
        emoji_engine: None,
        node_id: spec.node_id.clone(),
        z,
        model_matrix,
        page_clip_bounds: page_clip,
        ops: Vec::new(),
        degraded: Some(crate::seams::DegradedQuad {
            raster: Arc::new(raster),
            offset: [offset_x as f32, offset_y as f32],
            opacity,
        }),
        vector: None,
        // The Degraded raster quad stays unwarped (only the
        // vector lane consumes the rows — the accepted fallback gap).
        distort: None,
        content_key: None, paint_geometry_key: None,
    })))
}

// ---------------------------------------------------------------------------
// Plan side — node → TextQuadBatch
// ---------------------------------------------------------------------------

/// Outcome of planning one text item.
pub enum TextBuildOutcome {
    Item(std::sync::Arc<TextQuadBatch>),
    /// Browser renders this node via Canvas2D — fallback class; the page must
    /// report Degraded/unsupported in the manifest, never silently differ.
    Degraded {
        reason: String,
    },
    /// Nothing to draw (no content / fully empty layout).
    Empty,
}

// ---------------------------------------------------------------------------
// GPU side
// ---------------------------------------------------------------------------

pub(crate) const UNIFORM_ALIGN: u64 = 256;

/// Per-Engine text renderer state (pipelines + texture caches).
struct TextLayouts {
    sampler: wgpu::Sampler,
    bgl0: wgpu::BindGroupLayout,
    /// Group 1 of the textured modules (`text-emoji`, the degraded raster): sampler + texture.
    texture_bgl: wgpu::BindGroupLayout,
    layout_textured: wgpu::PipelineLayout,
    layout_solid: wgpu::PipelineLayout,
    // Track C — vector text engine (`--text-engine vector`). Group-0 BGLs
    // mirror the production factories in vectorTextShaders.ts: tess
    // (createVectorTessPipeline: frame + DrawUniforms), analytic
    // (+ curvePoints storage), gradient (+ GradientUniforms); the per-draw
    // DrawUniforms/GradientUniforms bind with dynamic offsets into the
    // prepare() arena (the production code creates per-draw bind groups at
    // explicit offsets — same windows, same 256 alignment).
    vec_tess_bgl: wgpu::BindGroupLayout,
    vec_analytic_bgl: wgpu::BindGroupLayout,
    vec_gradient_bgl: wgpu::BindGroupLayout,
    layout_vec_tess: wgpu::PipelineLayout,
    layout_vec_analytic: wgpu::PipelineLayout,
    layout_vec_gradient: wgpu::PipelineLayout,
}

pub struct TextDraw {
    layouts: Arc<TextLayouts>,
    /// (module id, sample_count) → pipeline. String keys: the per-shader
    /// `<id>.vector-text-fill` modules are dynamic.
    pipelines: RefCell<Arc<HashMap<(String, u32), wgpu::RenderPipeline>>>,
    /// Shaped run, resolution and optional silhouette effect → texture bind group.
    emoji_bind_groups: RefCell<ByteLruCache<(u64, u32, Option<[u32; 9]>), Arc<wgpu::BindGroup>>>,
    /// T9 (perf): content-addressed GPU buffer cache for the node-LOCAL text
    /// arenas (`verts` / `indices` / per-node curve storage). Those are
    /// byte-identical across a transform-only drag — only the per-batch model
    /// matrix in the SMALL `uniforms` arena changes — so caching them by content
    /// lets a re-`prepare` skip the ~1.5 MB/frame re-upload (the intermittent
    /// drag-jank spikes). Tracks retained mode (synced via the engine's
    /// `set_retained` / `set_vec_arena_cache_enabled`): ON in the wasm editor lane,
    /// OFF in CLI/export and other non-retained lanes.
    vec_arena_cache: RefCell<crate::vec_arena_cache::VecArenaCache<Arc<wgpu::Buffer>>>,
    /// The owning engine's immutable frame color format — every text
    /// pipeline's color target (text draws straight into the main-pass frame
    /// attachment). Emoji/raster SOURCE textures stay `Rgba8Unorm` regardless
    /// (sampled, channel-order neutral).
    frame_format: wgpu::TextureFormat,
    /// The owning engine's working colour (`moda_shaders::working_color`): text and decoration colours are
    /// authored sRGB and converted to it where they are drawn, and a text raster's or an emoji's texture is
    /// uploaded in it. `None` is sRGB.
    working: Option<moda_shaders::WorkingColor>,
}

// T9: cache role discriminators folded into the content key so the same bytes
// used as VERTEX vs INDEX vs STORAGE never alias to one entry (different `wgpu`
// usage flags).
const VEC_ROLE_VERTS: u8 = 0;
const VEC_ROLE_INDICES: u8 = 1;
const VEC_ROLE_CURVES: u8 = 2;

impl TextDraw {
    /// `frame_format` is the owning engine's immutable frame color format and `working` its working colour
    /// — see the field docs.
    pub fn new(gpu: &GpuContext, frame_format: wgpu::TextureFormat, working: Option<moda_shaders::WorkingColor>) -> Result<TextDraw> {
        let device = &gpu.device;
        // Group 0 (manifest BGL 295): binding 0 = frame uniform (static),
        // 1-4 = dynamic-offset uniforms (model / color / params / pageClip).
        let uniform_entry = |binding: u32, dynamic: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Text Uniforms BGL (solid/emoji)"),
            entries: &[
                uniform_entry(0, false),
                uniform_entry(1, true),
                uniform_entry(2, true),
                uniform_entry(3, true),
                uniform_entry(4, true),
            ],
        });
        // Group 1 (manifest BGLs 298/302/306): sampler + texture.
        let texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Text Texture BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let layout_textured = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Text Textured Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl0), Some(&texture_bgl)],
            ..Default::default()
        });
        let layout_solid = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Text Solid Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl0)],
            ..Default::default()
        });
        // Track C — vector text BGLs (vectorTextShaders.ts pipeline
        // factories). binding 0: frame projection and pixel transform (VERTEX); binding 1:
        // DrawUniforms (VERTEX|FRAGMENT, 160 B, dynamic — model + color +
        // params + pageClip + the distort rows); binding 2: glyph
        // curves (FRAGMENT read-only storage); binding 3: GradientUniforms
        // (FRAGMENT uniform, 176 B, dynamic).
        let frame_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let draw_uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        };
        let curve_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let vec_tess_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Vector Text Tess BGL"),
            entries: &[frame_entry, draw_uniform_entry],
        });
        let vec_analytic_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Vector Text Analytic BGL"),
            entries: &[frame_entry, draw_uniform_entry, curve_entry(2)],
        });
        let vec_gradient_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Vector Text Analytic Gradient BGL"),
            entries: &[
                frame_entry,
                draw_uniform_entry,
                curve_entry(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout_vec_tess = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vector Text Tess Pipeline Layout"),
            bind_group_layouts: &[Some(&vec_tess_bgl)],
            ..Default::default()
        });
        let layout_vec_analytic = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vector Text Analytic Pipeline Layout"),
            bind_group_layouts: &[Some(&vec_analytic_bgl)],
            ..Default::default()
        });
        let layout_vec_gradient = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vector Text Gradient Pipeline Layout"),
            bind_group_layouts: &[Some(&vec_gradient_bgl)],
            ..Default::default()
        });
        // The ONE sampler the textured text pipelines share: linear mag/min,
        // clamp-to-edge, no mipmaps.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("text sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let layouts = Arc::new(TextLayouts {
            sampler, bgl0, texture_bgl, layout_textured, layout_solid, vec_tess_bgl,
            vec_analytic_bgl, vec_gradient_bgl, layout_vec_tess, layout_vec_analytic, layout_vec_gradient,
        });
        Ok(Self::from_layouts(frame_format, working, layouts, Default::default()))
    }

    pub fn replica(&self) -> Self {
        Self::from_layouts(self.frame_format, self.working, self.layouts.clone(), self.pipelines.borrow().clone())
    }

    fn from_layouts(frame_format: wgpu::TextureFormat, working: Option<moda_shaders::WorkingColor>,
        layouts: Arc<TextLayouts>, pipelines: Arc<HashMap<(String, u32), wgpu::RenderPipeline>>) -> Self {
        Self { frame_format, working, layouts, pipelines: RefCell::new(pipelines),
            emoji_bind_groups: RefCell::new(ByteLruCache::new(
                moda_text::host::emoji_cap_bytes(),
                moda_text::host::EMOJI_CACHE_MAX_ENTRIES,
            )),
            vec_arena_cache: RefCell::new(crate::vec_arena_cache::VecArenaCache::new(
                crate::vec_arena_cache::VEC_ARENA_BUDGET_BYTES,
            )),
        }
    }

    /// T9 cache gate: enable/disable the content-addressed node-local text arena
    /// cache. Driven by the engine's `set_retained` (ON with retained mode — the
    /// wasm editor lane; OFF in CLI/export). Disabling drops cached buffers and
    /// forces the legacy create-and-upload-every-`prepare` path.
    pub fn set_vec_arena_cache_enabled(&self, on: bool) {
        self.vec_arena_cache.borrow_mut().set_enabled(on);
    }

    /// T9: drop the node-local text arena cache's GPU buffers (keeps it enabled).
    /// Called from `Engine::clear_retained_gpu` so the memory-guard ALARM /
    /// retained-off teardown frees this cache's residency alongside the rest of the
    /// retained GPU state; it repopulates lazily on the next `prepare`.
    pub fn clear_vec_arena_cache(&self) {
        self.vec_arena_cache.borrow_mut().clear();
    }

    /// T9: snapshot of the vector-text arena cache counters, surfaced in the
    /// residency stats JSON as `vec_arena_*`.
    pub fn vec_arena_stats(&self) -> crate::vec_arena_cache::VecArenaStats {
        self.vec_arena_cache.borrow().stats()
    }

    /// Bytes charged to the resident emoji texture cache (`gpu_profile` memory report).
    pub fn emoji_texture_bytes(&self) -> u64 {
        self.emoji_bind_groups.borrow().current_bytes()
    }

    /// T9: get-or-create a content-addressed GPU buffer for a node-local text
    /// arena. Returns `(buffer, uploaded_bytes)` where `uploaded_bytes` is the
    /// data actually written to the GPU this call (0 on a cache hit — the win).
    /// On a miss (or when the cache is disabled) it creates + uploads the buffer;
    /// on a miss with the cache enabled it also inserts it. Byte-exact: a hit
    /// reuses a buffer whose bytes equal this `prepare`'s arena (the content key
    /// + length guarantee it), and every draw offset is recomputed identically
    /// from the same byte-identical batches.
    fn cache_buffer(
        &self,
        gpu: &GpuContext,
        role: u8,
        label: &str,
        data: &[u8],
        usage: wgpu::BufferUsages,
    ) -> (Arc<wgpu::Buffer>, u64) {
        let upload_len = data.len() as u64;
        let mut cache = self.vec_arena_cache.borrow_mut();
        if !cache.is_enabled() {
            return (
                Arc::new(create_filled_buffer(gpu, label, data, usage)),
                upload_len,
            );
        }
        let key = crate::vec_arena_cache::content_key(role, data);
        if let Some(buf) = cache.get(key, data.len()) {
            return (buf, 0);
        }
        let buf = Arc::new(create_filled_buffer(gpu, label, data, usage));
        let bytes = buf.size() as usize;
        cache.put(key, buf.clone(), data.len(), bytes);
        (buf, upload_len)
    }

    fn ensure_pipeline(&self, gpu: &GpuContext, module_id: &str, sample_count: u32) -> Result<()> {
        if self
            .pipelines
            .borrow()
            .contains_key(&(module_id.to_string(), sample_count))
        {
            return Ok(());
        }
        let layout = match module_id {
            "text-emoji" | "text-emoji-additive" => &self.layouts.layout_textured,
            "text-decoration" => &self.layouts.layout_solid,
            // Track C — vector text engine modules (manifest pipeline
            // layouts 313/317/321; vectorTextShaders.ts factories).
            "vector-text-tess" => &self.layouts.layout_vec_tess,
            // The additive (glow) variant shares the analytic BGL — only the
            // blend state differs, and that comes from its manifest
            // descriptor (TEXT_ADDITIVE_BLEND, vectorTextShaders.ts:92-95).
            "vector-text-analytic" | "vector-text-analytic-additive" => &self.layouts.layout_vec_analytic,
            "vector-text-analytic-gradient" => &self.layouts.layout_vec_gradient,
            other => bail!("unknown text module '{other}'"),
        };
        let manifest = moda_shaders::manifest();
        let meta = manifest
            .modules
            .get(module_id)
            .ok_or_else(|| anyhow!("module '{module_id}' missing from manifest"))?;
        let desc = meta
            .pipelines
            .first()
            .ok_or_else(|| anyhow!("module '{module_id}' has no pipeline descriptor"))?;
        let source: std::borrow::Cow<'static, str> = moda_shaders::module_source(module_id)
            .ok_or_else(|| anyhow!("no extracted WGSL module '{module_id}'"))?
            .source
            .into();
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(module_id),
                source: wgpu::ShaderSource::Wgsl(moda_shaders::working_color::specialize(self.working.as_ref(), source)),
            });

        // Vertex layout from the manifest descriptor (one buffer,
        // arrayStride 16, two float32x2 attributes).
        let empty: Vec<moda_shaders::VertexBufferDesc> = Vec::new();
        let buffer_descs = desc
            .vertex
            .as_ref()
            .and_then(|v| v.buffers.as_ref())
            .unwrap_or(&empty);
        let attr_storage: Vec<Vec<wgpu::VertexAttribute>> = buffer_descs
            .iter()
            .map(|b| {
                b.attributes
                    .iter()
                    .map(|a| wgpu::VertexAttribute {
                        format: match a.format.as_str() {
                            "float32" => wgpu::VertexFormat::Float32,
                            "float32x2" => wgpu::VertexFormat::Float32x2,
                            "float32x4" => wgpu::VertexFormat::Float32x4,
                            // vector-text analytic instance curve range
                            "uint32x2" => wgpu::VertexFormat::Uint32x2,
                            other => panic!("unmapped text vertex format '{other}'"),
                        },
                        offset: a.offset,
                        shader_location: a.shader_location,
                    })
                    .collect()
            })
            .collect();
        let buffers: Vec<wgpu::VertexBufferLayout> = buffer_descs
            .iter()
            .zip(attr_storage.iter())
            .map(|(b, attrs)| wgpu::VertexBufferLayout {
                array_stride: b.array_stride,
                // stepMode from the manifest descriptor (the vector-text
                // analytic instance buffer is "instance"; absent = vertex).
                step_mode: match b.step_mode.as_deref() {
                    Some("instance") => wgpu::VertexStepMode::Instance,
                    _ => wgpu::VertexStepMode::Vertex,
                },
                attributes: attrs,
            })
            .collect();

        let fragment = desc
            .fragment
            .as_ref()
            .ok_or_else(|| anyhow!("text pipeline lacks fragment state"))?;
        // Blend state from the extracted manifest (text pipelines blend
        // src-alpha / one-minus-src-alpha on color — straight-alpha output).
        let targets: Vec<Option<wgpu::ColorTargetState>> = fragment
            .targets
            .iter()
            .map(|t| {
                t.as_ref().map(|t| wgpu::ColorTargetState {
                    // Text draws into the main-pass frame attachment,
                    // whose format is per-engine (the manifest's recorded
                    // rgba8unorm/bgra8unorm both name this same attachment).
                    format: self.frame_format,
                    blend: t.blend.as_ref().map(|b| wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: text_blend_factor(&b.color.src_factor),
                            dst_factor: text_blend_factor(&b.color.dst_factor),
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: text_blend_factor(&b.alpha.src_factor),
                            dst_factor: text_blend_factor(&b.alpha.dst_factor),
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })
            })
            .collect();
        let depth_stencil = desc
            .depth_stencil
            .as_ref()
            .map(|d| wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: Some(d.depth_write_enabled),
                depth_compare: Some(match d.depth_compare.as_str() {
                    "less" => wgpu::CompareFunction::Less,
                    "less-equal" => wgpu::CompareFunction::LessEqual,
                    "always" => wgpu::CompareFunction::Always,
                    other => panic!("unmapped text depth compare '{other}'"),
                }),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            });

        // M5b (docs/productionization.md): env-gated pipeline-compile stats.
        if std::env::var_os("MODA_PIPELINE_STATS").is_some() {
            eprintln!("moda-pipeline-stat: {module_id} desc=text msaa={sample_count} blend=-");
        }
        let pipeline = crate::device::create_render_pipeline(&gpu.device, &wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{module_id} s{sample_count}")),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(
                        desc.vertex
                            .as_ref()
                            .map(|v| v.entry_point.as_str())
                            .unwrap_or("vs"),
                    ),
                    compilation_options: Default::default(),
                    buffers: &buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(&fragment.entry_point),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil,
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            });
        Arc::make_mut(&mut self.pipelines.borrow_mut()).insert((module_id.to_string(), sample_count), pipeline);
        Ok(())
    }

    fn vector_emoji_bind_group(&self, gpu: &GpuContext, batch: &TextQuadBatch, index: u32) -> Result<Arc<wgpu::BindGroup>> {
        let emoji = batch.vector.as_ref().and_then(|v| v.emojis.get(index as usize))
            .ok_or_else(|| anyhow!("emoji index {index} out of range"))?;
        let run = emoji.shaped.as_ref().ok_or_else(|| anyhow!("unresolved colour emoji {:?}", emoji.cluster))?;
        let engine = batch.emoji_engine.as_ref().ok_or_else(|| anyhow!("missing colour font engine"))?;
        let key = (run.id, emoji.ppem, emoji.effect.as_ref().map(|fx| fx.key()));
        if let Some(hit) = self.emoji_bind_groups.borrow_mut().get(&key) { return Ok(hit); }
        let img = emoji.image(engine).ok_or_else(|| anyhow!("cannot rasterize colour emoji {:?}", emoji.cluster))?;
        let bytes = img.width as u64 * img.height as u64 * if self.working.is_some() { 8 } else { 4 };
        let made = Arc::new(self.create_texture_bind_group(gpu, &img.rgba, img.width, img.height, "emoji"));
        self.emoji_bind_groups.borrow_mut().insert(key, made.clone(), bytes);
        Ok(made)
    }

    /// The texture of a text raster or an emoji cell (straight-alpha sRGB bytes, one level) and its bind
    /// group. In an engine that blends in another encoding (`working`) the texture is half floats in that
    /// encoding, each texel's colour converted here and its alpha kept straight, as the text shaders take
    /// it: the sampler then filters colour in the encoding the engine blends in.
    fn create_texture_bind_group(
        &self,
        gpu: &GpuContext,
        rgba: &[u8],
        width: u32,
        height: u32,
        label: &str,
    ) -> wgpu::BindGroup {
        let device = &gpu.device;
        let converted: Option<Vec<u8>> = self.working.map(|working| {
            let mut texels = Vec::with_capacity(rgba.len() * 2);
            for texel in rgba.chunks_exact(4) {
                let colour = working.from_srgb.apply([texel[0], texel[1], texel[2]].map(|c| f32::from(c) / 255.0));
                for value in [colour[0], colour[1], colour[2], f32::from(texel[3]) / 255.0] {
                    texels.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
                }
            }
            texels
        });
        let (texels, texel_bytes, format) = match &converted {
            Some(texels) => (texels.as_slice(), 8, wgpu::TextureFormat::Rgba16Float),
            // rgba8unorm, never -srgb (render-semantics §7).
            None => (rgba, 4, wgpu::TextureFormat::Rgba8Unorm),
        };
        let tex = gpu.admission.texture(device, &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.admission.charge(0, texels.len() as u64);
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * texel_bytes),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.layouts.texture_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.layouts.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        })
    }

    /// Build all GPU resources for the plan's text batches (call BEFORE the
    /// render pass; only immutable borrows happen at draw time).
    pub fn prepare(
        &self,
        gpu: &GpuContext,
        frame_uniform: &wgpu::Buffer,
        batches: &[&TextQuadBatch],
        sample_count: u32,
    ) -> Result<PreparedText> {
        if batches.is_empty() {
            return Ok(PreparedText::default());
        }
        // T9: advance the arena cache's LRU clock once per `prepare` (its render
        // sweep) so entries untouched by recent prepares age out under budget.
        self.vec_arena_cache.borrow_mut().begin_frame();

        // Pipelines used by the plan.
        let mut want: Vec<String> = Vec::new();
        for b in batches {
            for op in &b.ops {
                let m: String = match op.pipeline {
                    TextOpPipeline::Solid => "text-decoration".into(),
                    TextOpPipeline::Emoji { emoji } => emoji_module(b, emoji).into(),
                };
                if !want.contains(&m) {
                    want.push(m);
                }
            }
            // M4d: Degraded raster quads draw via the text-emoji image
            // module (sampled texture, color uniform ignored).
            if b.degraded.is_some() && !want.iter().any(|m| m == "text-emoji") {
                want.push("text-emoji".into());
            }
            // Track C: vector-engine modules used by this batch.
            if let Some(v) = &b.vector {
                for op in &v.ops {
                    let m: String = match op.pipeline {
                        moda_text::vector::VectorOpPipeline::Tess => "vector-text-tess".into(),
                        // Glow passes (additive blend) compile their own
                        // pipeline over the same analytic module.
                        moda_text::vector::VectorOpPipeline::Analytic if op.additive => {
                            "vector-text-analytic-additive".into()
                        }
                        moda_text::vector::VectorOpPipeline::Analytic => {
                            "vector-text-analytic".into()
                        }
                        moda_text::vector::VectorOpPipeline::AnalyticGradient => {
                            "vector-text-analytic-gradient".into()
                        }

                    };
                    if !want.contains(&m) {
                        want.push(m);
                    }
                }
            }
        }
        for m in &want {
            self.ensure_pipeline(gpu, m, sample_count)?;
        }

        // Uniform arena (256-aligned slots) + vertex/index arenas.
        let mut uniforms: Vec<UniformSlot> = Vec::new();
        let mut verts: Vec<u8> = Vec::new();
        let mut indices: Vec<u8> = Vec::new();
        let alloc = |data: &[u8], uniforms: &mut Vec<UniformSlot>| -> u32 {
            let off = (uniforms.len() as u64 * UNIFORM_ALIGN) as u32;
            for chunk in data.chunks(UNIFORM_ALIGN as usize) {
                let mut slot = [0; UNIFORM_ALIGN as usize];
                slot[..chunk.len()].copy_from_slice(chunk);
                uniforms.push(slot);
            }
            off
        };

        let mut prepared_batches: Vec<PreparedBatch> = Vec::new();
        let mut padding = Vec::new();
        let mut textures: Vec<Arc<wgpu::BindGroup>> = Vec::new();
        // Track C: vector-engine resources. Curve buffers are created
        // immediately (no arena dependency); group-0 bind groups are
        // recorded as (flavor, curve buffer index) and created once the
        // uniform arena buffer exists.
        // T9: curve buffers are `Arc` so a cache hit shares the handle (an Arc
        // pointer-bump, no re-upload) and `PreparedText` still keeps them alive.
        let mut curve_bufs: Vec<Arc<wgpu::Buffer>> = Vec::new();
        // T9: bytes actually uploaded for per-node curve storage this `prepare`
        // (0 for each cache hit) — folded into `uploaded_bytes`/`pf_text_bytes`.
        let mut curve_uploaded_bytes: u64 = 0;
        let mut pending_vec_groups: Vec<(usize, Option<usize>)> = Vec::new();
        // The frame and uniform arena are shared by this chunk. A pipeline layout
        // and its actual curve buffer identify the complete group; draws select
        // their own uniform records through the existing dynamic offsets.
        let mut group_indices = HashMap::new();
        let any_vector = batches.iter().any(|b| b.vector.is_some());
        // Size the vertex arena for the vector ops up front (their triangles and per-glyph
        // analytic instances are nearly all of it): grown by doubling, a frame with every
        // text node of a large document in view held up to three times its final size at
        // once (the old and new blocks of each realloc) — one of the biggest transients of
        // opening one.
        verts.reserve_exact(
            batches
                .iter()
                .filter_map(|b| b.vector.as_ref())
                .flat_map(|v| v.ops.iter())
                .map(|op| op.tris.len() * 4 + op.records.len() * std::mem::size_of::<moda_text::vector::ops::VectorAnalyticRecord>())
                .sum(),
        );
        for b in batches {
            // Vector operations carry their own model and clip in DrawUniforms. Only
            // quad draws bind separate dynamic ranges for these two fields.
            let mut quad_uniforms = None;
            let mut quad_offsets = |uniforms: &mut Vec<UniformSlot>, color, params| {
                let (model, clip) = *quad_uniforms.get_or_insert_with(|| (
                    alloc(bytemuck::cast_slice(&b.model_matrix), uniforms),
                    alloc(bytemuck::cast_slice(&b.page_clip_bounds), uniforms),
                ));
                vec![model, color, params, clip]
            };
            let depth = crate::depth::linear_depth(b.z) as f32;

            let mut draws: Vec<PreparedDraw> = Vec::new();

            // M4d: one textured quad per Degraded-tier batch — the Canvas2D
            // display-quad stand-in (`richtext/transforms.ts` placement; the
            // 2x raster minifies through the linear sampler like the
            // production qualityScale-2 texture).
            if let Some(dq) = &b.degraded {
                let r = &dq.raster;
                let bg = Arc::new(self.create_texture_bind_group(
                    gpu,
                    &r.rgba,
                    r.width,
                    r.height,
                    "degraded text raster",
                ));
                textures.push(bg);
                let tex = textures.len() - 1;
                let (x0, y0) = (dq.offset[0], dq.offset[1]);
                let (x1, y1) = (x0 + r.logical_w as f32, y0 + r.logical_h as f32);
                #[rustfmt::skip]
                let quad: [f32; 16] = [
                    x0, y0, 0.0, 0.0,
                    x1, y0, 1.0, 0.0,
                    x1, y1, 1.0, 1.0,
                    x0, y1, 0.0, 1.0,
                ];
                let quad_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
                let color_off = alloc(
                    bytemuck::cast_slice(&[1.0f32, 1.0, 1.0, 1.0]),
                    &mut uniforms,
                );
                let params: [f32; 4] = [depth, dq.opacity, 0.0, 0.0];
                let params_off = alloc(bytemuck::cast_slice(&params), &mut uniforms);
                let v_off = verts.len() as u64;
                verts.extend_from_slice(bytemuck::cast_slice(&quad));
                let i_off = indices.len() as u64;
                indices.extend_from_slice(bytemuck::cast_slice(&quad_indices));
                draws.push(PreparedDraw {
                    module: "text-emoji".into(),
                    offsets: quad_offsets(&mut uniforms, color_off, params_off),
                    tex: Some(tex),
                    v_off,
                    v_len: (quad.len() * 4) as u64,
                    i_off,
                    i_len: (quad_indices.len() * 4) as u64,
                    index_count: quad_indices.len() as u32,
                });
            }

            for op in &b.ops {
                // Texture bind group (emoji).
                let tex: Option<usize> = match op.pipeline {
                    TextOpPipeline::Emoji { emoji } => {
                        let bg = self.vector_emoji_bind_group(gpu, b, emoji)
                            .map_err(|e| e.context(format!("text node {}", b.node_id)))?;
                        textures.push(bg);
                        Some(textures.len() - 1)
                    }
                    TextOpPipeline::Solid => None,
                };
                let color_off = alloc(bytemuck::cast_slice(&op.color), &mut uniforms);
                let params: [f32; 4] = [depth, op.opacity, op.blur_px, op.stroke_px];
                let params_off = alloc(bytemuck::cast_slice(&params), &mut uniforms);

                let v_off = verts.len() as u64;
                verts.extend_from_slice(bytemuck::cast_slice(&op.verts));
                let i_off = indices.len() as u64;
                indices.extend_from_slice(bytemuck::cast_slice(&op.indices));

                let offsets: Vec<u32> = quad_offsets(&mut uniforms, color_off, params_off);
                let module: String = match op.pipeline {
                    TextOpPipeline::Solid => "text-decoration".into(),
                    TextOpPipeline::Emoji { emoji } => emoji_module(b, emoji).into(),
                };
                draws.push(PreparedDraw {
                    module,
                    offsets,
                    tex,
                    v_off,
                    v_len: (op.verts.len() * 4) as u64,
                    i_off,
                    i_len: (op.indices.len() * 4) as u64,
                    index_count: op.indices.len() as u32,
                });
            }
            // ── Track C: vector-engine draws (drawRunsAnalytic op stream) ──
            let mut vector_draws: Vec<PreparedVectorDraw> = Vec::new();
            if let Some(v) = &b.vector {
                let depth = crate::depth::linear_depth(b.z) as f32;
                // Per-node curve buffer (the GlyphCurveStore upload).
                let curve_buffer = if v.curve_points.is_empty() {
                    None
                } else {
                    // T9: content-addressed — a transform-only drag reuses the
                    // already-uploaded curve buffer (skips the per-node upload).
                    let (buf, up) = self.cache_buffer(
                        gpu,
                        VEC_ROLE_CURVES,
                        // Static label — a `format!` here would allocate per node
                        // every prepare, including on cache hits where the GPU
                        // buffer (and its label) is never created.
                        "vector text curves",
                        bytemuck::cast_slice(&v.curve_points[..]),
                        wgpu::BufferUsages::STORAGE,
                    );
                    curve_uploaded_bytes += up;
                    Some(buf)
                };
                // Gradient uniforms: one contiguous 176-byte record
                // (packTextGradientUniformFloats layout) per node.
                let gradient_floats = |g: &moda_text::ops::TextGradientUniforms| -> Vec<f32> {
                    let mut floats: Vec<f32> = Vec::with_capacity(44);
                    floats.extend_from_slice(&g.text_bounds);
                    floats.extend_from_slice(&g.params);
                    floats.extend_from_slice(&g.config);
                    for stop in &g.stops {
                        floats.extend_from_slice(stop);
                    }
                    floats
                };
                let vec_grad_off: Option<u32> = v
                    .gradient
                    .as_ref()
                    .map(|g| alloc(bytemuck::cast_slice(&gradient_floats(g)), &mut uniforms));
                let mut group_for = |flavor: usize| -> usize {
                    let curve = if flavor == VEC_FLAVOR_TESS { None } else { curve_buffer.as_ref() };
                    *group_indices.entry((flavor, curve.map(Arc::as_ptr))).or_insert_with(|| {
                        let curve_idx = curve.map(|buffer| {
                            curve_bufs.push(buffer.clone());
                            curve_bufs.len() - 1
                        });
                        let index = pending_vec_groups.len();
                        pending_vec_groups.push((flavor, curve_idx));
                        index
                    })
                };

                for op in &v.ops {
                    // DrawUniforms record: model + color + params + pageClip
                    // + distort0/1/2 (160 B; the warp rows —
                    // distort0.w is the enabled flag, disabled draws carry
                    // identity rows).
                    let uniform = vector_uniform(b, op, depth);
                    let draw_off = alloc(bytemuck::bytes_of(&uniform), &mut uniforms);

                    use moda_text::vector::VectorOpPipeline as VP;
                    match op.pipeline {
                        VP::Tess => {
                            if op.tris.len() < 6 {
                                continue;
                            }
                            let v_off = verts.len() as u64;
                            verts.extend_from_slice(bytemuck::cast_slice(&op.tris));
                            vector_draws.push(PreparedVectorDraw {
                                module: "vector-text-tess".into(),
                                kind: VectorDrawKind::Tess {
                                    vertex_count: (op.tris.len() / 2) as u32,
                                },
                                group: Some(group_for(VEC_FLAVOR_TESS)),
                                dynamic_offsets: vec![draw_off],
                                v_off,
                                v_len: (op.tris.len() * 4) as u64,
                            });
                        }

                        VP::Analytic | VP::AnalyticGradient => {
                            if op.records.is_empty() {
                                continue;
                            }
                            if curve_buffer.is_none() {
                                bail!("vector analytic op without curve points ({})", b.node_id);
                            };
                            // Instance attribute order matches the analytic layouts in the shader manifest.
                            let v_off = verts.len() as u64;
                            for r in op.records.iter() {
                                verts.extend_from_slice(&r.pen_x.to_le_bytes());
                                verts.extend_from_slice(&r.baseline_y.to_le_bytes());
                                verts.extend_from_slice(&r.font_size_px.to_le_bytes());
                                for c in r.em_box {
                                    verts.extend_from_slice(&c.to_le_bytes());
                                }
                                verts.extend_from_slice(&r.first_curve.to_le_bytes());
                                verts.extend_from_slice(&r.curve_count.to_le_bytes());
                                for c in r.glyph_lin {
                                    verts.extend_from_slice(&c.to_le_bytes());
                                }
                                for c in r.glyph_off {
                                    verts.extend_from_slice(&c.to_le_bytes());
                                }
                                verts.extend_from_slice(&r.glyph_opacity.to_le_bytes());
                                verts.extend_from_slice(&r.local_aa_px.to_le_bytes());
                            }
                            let v_len = verts.len() as u64 - v_off;
                            let kind = VectorDrawKind::Analytic {
                                instance_count: op.records.len() as u32,
                            };
                            match op.pipeline {
                                // Glow passes ride the additive (emitted-
                                // light) blend variant of the same analytic
                                // module — createVectorAnalyticAdditive-
                                // Pipeline (vectorTextShaders.ts:886-906);
                                // identical BGL, so the group-0 flavor is
                                // shared.
                                VP::Analytic => vector_draws.push(PreparedVectorDraw {
                                    module: if op.additive {
                                        "vector-text-analytic-additive".into()
                                    } else {
                                        "vector-text-analytic".into()
                                    },
                                    kind,
                                    group: Some(group_for(VEC_FLAVOR_ANALYTIC)),
                                    dynamic_offsets: vec![draw_off],
                                    v_off,
                                    v_len,
                                }),
                                VP::AnalyticGradient => {
                                    // A text range's own gradient (`--nf-text-fill`), else the node's.
                                    let op_grad_off = op.gradient.as_ref().map(|g| {
                                        alloc(bytemuck::cast_slice(&gradient_floats(g)), &mut uniforms)
                                    });
                                    let g = op_grad_off.or(vec_grad_off).ok_or_else(|| {
                                        anyhow!(
                                            "vector gradient op without gradient uniforms ({})",
                                            b.node_id
                                        )
                                    })?;
                                    vector_draws.push(PreparedVectorDraw {
                                        module: "vector-text-analytic-gradient".into(),
                                        kind,
                                        group: Some(group_for(VEC_FLAVOR_GRADIENT)),
                                        // dynamic offsets in binding order: 1 (draw), 3 (gradient)
                                        dynamic_offsets: vec![draw_off, g],
                                        v_off,
                                        v_len,
                                    });
                                }
                                VP::Tess => unreachable!(),
                            }
                        }
                    }
                }
            }
            let params = b.vector.as_ref().and_then(|v| v.uniform_padding);
            if padding.last().and_then(|&(_, previous)| previous) != params {
                padding.push((prepared_batches.len(), params));
            }
            prepared_batches.push(PreparedBatch {
                geometry: uniform_geometry(b), draws,
                vector_draws,
            });
        }

        if uniforms.is_empty() {
            return Ok(PreparedText { batches: prepared_batches, padding, sample_count, ..Default::default() });
        }
        let device = &gpu.device;
        let mk = |label: &str, data: &[u8], usage: wgpu::BufferUsages| -> wgpu::Buffer {
            create_filled_buffer(gpu, label, data, usage)
        };
        // The `uniforms` arena carries the per-batch MODEL MATRICES (and per-draw
        // depth/opacity/gradient), which change every pointermove — so it is
        // ALWAYS rebuilt + uploaded, never cached.
        let uniform_buf = mk("text uniforms", bytemuck::cast_slice(&uniforms), wgpu::BufferUsages::UNIFORM);
        // T9: the `verts`/`indices` arenas are node-LOCAL (position-independent —
        // the model matrix lives in `uniforms`), so they are byte-identical across
        // a transform-only drag and content-addressable. A hit reuses the
        // already-uploaded buffer; the draw offsets are recomputed identically
        // from the same byte-identical batches, so the slices stay valid.
        let (vbuf, verts_uploaded) = self.cache_buffer(
            gpu,
            VEC_ROLE_VERTS,
            "text vertices",
            &verts,
            wgpu::BufferUsages::VERTEX,
        );
        let (ibuf, indices_uploaded) = self.cache_buffer(
            gpu,
            VEC_ROLE_INDICES,
            "text indices",
            &indices,
            wgpu::BufferUsages::INDEX,
        );

        // Track C: the static unit quad (6 verts, 48 B — `VectorText Quad`,
        // VectorTextRenderer.ts:330-336) + the recorded group-0 bind groups.
        let quad_buf: Option<wgpu::Buffer> = any_vector.then(|| {
            let quad: [f32; 12] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
            mk(
                "vector text quad",
                bytemuck::cast_slice(&quad),
                wgpu::BufferUsages::VERTEX,
            )
        });
        let draw_uniform_size = std::num::NonZeroU64::new(VECTOR_DRAW_UNIFORM_BYTES);
        let gradient_uniform_size = std::num::NonZeroU64::new(176);
        let mut vector_binds: Vec<wgpu::BindGroup> = Vec::new();
        for (flavor, curve_idx) in &pending_vec_groups {
            let draw_binding = wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buf,
                    offset: 0,
                    size: draw_uniform_size,
                }),
            };
            let frame_binding = wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_uniform.as_entire_binding(),
            };
            let bg = match flavor {
                &VEC_FLAVOR_TESS => device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("vector text tess bind0"),
                    layout: &self.layouts.vec_tess_bgl,
                    entries: &[frame_binding, draw_binding],
                }),
                &VEC_FLAVOR_ANALYTIC => {
                    let curve = &curve_bufs[curve_idx.expect("analytic group needs curves")];
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("vector text analytic bind0"),
                        layout: &self.layouts.vec_analytic_bgl,
                        entries: &[
                            frame_binding,
                            draw_binding,
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: curve.as_entire_binding(),
                            },
                        ],
                    })
                }
                _ => {
                    let curve = &curve_bufs[curve_idx.expect("gradient group needs curves")];
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("vector text gradient bind0"),
                        layout: &self.layouts.vec_gradient_bgl,
                        entries: &[
                            frame_binding,
                            draw_binding,
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: curve.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &uniform_buf,
                                    offset: 0,
                                    size: gradient_uniform_size,
                                }),
                            },
                        ],
                    })
                }
            };
            vector_binds.push(bg);
        }

        // Group-0 bind groups against the arena (production binds the same
        // ring buffers with per-draw dynamic offsets).
        let vec4 = std::num::NonZeroU64::new(16);
        let mat4 = std::num::NonZeroU64::new(64);
        let arena = |size: Option<std::num::NonZeroU64>| {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &uniform_buf,
                offset: 0,
                size,
            })
        };
        let bind0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("text bind0"),
            layout: &self.layouts.bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: arena(mat4),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: arena(vec4),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: arena(vec4),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: arena(vec4),
                },
            ],
        });
        // The pipelines the draws use, resolved once: a draw looks its module up here (a handful of entries)
        // instead of cloning its name into a hashed key into the shared map every frame.
        let pipes: Vec<(String, wgpu::RenderPipeline)> = {
            let pipelines = self.pipelines.borrow();
            let mut modules: Vec<&str> = Vec::new();
            for b in &prepared_batches {
                for m in b.draws.iter().map(|d| d.module.as_str()).chain(b.vector_draws.iter().map(|d| d.module.as_str())) {
                    if !modules.contains(&m) {
                        modules.push(m);
                    }
                }
            }
            modules
                .into_iter()
                .filter_map(|m| {
                    pipelines
                        .get(&(m.to_string(), sample_count))
                        .map(|p| (m.to_string(), p.clone()))
                })
                .collect()
        };
        Ok(PreparedText {
            pipes,
            _uniform_buf: Some(uniform_buf),
            vbuf: Some(vbuf),
            ibuf: Some(ibuf),
            bind0: Some(bind0),
            textures,
            batches: prepared_batches, padding,
            sample_count,
            quad_buf,
            vector_binds,
            _curve_bufs: curve_bufs,
            // T9: bytes ACTUALLY uploaded this `prepare` — the always-uploaded
            // `uniforms` plus only the node-local arenas that MISSED the cache
            // (0 each on a hit). Now includes the per-node curve storage
            // (`curve_uploaded_bytes`), which was previously omitted, so
            // `pf_text_bytes` reflects reality and plateaus to ~uniforms-only
            // once a transform-only drag hits the arena cache every frame.
            uploaded_bytes: uniforms.len() as u64 * UNIFORM_ALIGN
                + verts_uploaded
                + indices_uploaded
                + curve_uploaded_bytes,
            uniforms: std::cell::RefCell::new(Column::of(uniforms)),
        })
    }

    /// Draw one prepared batch inside the transparent bucket loop (AFTER the
    /// bucket's paths — `renderTransparentItemsWithText`). The caller must
    /// re-bind its own pipeline/bind groups afterwards.
    pub fn draw_batch<'p>(
        &self,
        pass: &mut impl wgpu::util::RenderEncoder<'p>,
        prepared: &'p PreparedText,
        batch_index: usize,
    ) {
        let Some(batch) = prepared.batches.get(batch_index) else {
            return;
        };
        let (Some(vbuf), Some(ibuf), Some(bind0)) =
            (&prepared.vbuf, &prepared.ibuf, &prepared.bind0)
        else {
            return;
        };
        let mut current_module: Option<&str> = None;
        for d in &batch.draws {
            if current_module != Some(d.module.as_str()) {
                let Some(p) = prepared.pipe(&d.module) else { continue };
                pass.set_pipeline(p);
                current_module = Some(d.module.as_str());
            }
            pass.set_bind_group(0, Some(bind0), &d.offsets);
            if let Some(tex) = d.tex {
                pass.set_bind_group(1, Some(prepared.textures[tex].as_ref()), &[]);
            }
            pass.set_vertex_buffer(0, vbuf.slice(d.v_off..d.v_off + d.v_len));
            pass.set_index_buffer(
                ibuf.slice(d.i_off..d.i_off + d.i_len),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed(0..d.index_count, 0, 0..1);
        }

        // ── Track C: vector-engine draws (drawRunsAnalytic submission
        // order is the op order — same depth, no depth write, order layers).
        for d in &batch.vector_draws {
            if current_module != Some(d.module.as_str()) {
                let Some(p) = prepared.pipe(&d.module) else { continue };
                pass.set_pipeline(p);
                current_module = Some(d.module.as_str());
            }
            let group = &prepared.vector_binds[d.group.expect("vector group")];
            pass.set_bind_group(0, Some(group), &d.dynamic_offsets);
            match d.kind {
                VectorDrawKind::Tess { vertex_count } => {
                    pass.set_vertex_buffer(0, vbuf.slice(d.v_off..d.v_off + d.v_len));
                    pass.draw(0..vertex_count, 0..1);
                }
                VectorDrawKind::Analytic { instance_count } => {
                    let quad = prepared.quad_buf.as_ref().expect("vector quad buffer");
                    pass.set_vertex_buffer(0, quad.slice(..));
                    pass.set_vertex_buffer(1, vbuf.slice(d.v_off..d.v_off + d.v_len));
                    pass.draw(0..6, 0..instance_count);
                }
            }
        }
    }
}

/// Create a 4-byte-rounded buffer with `usage | COPY_DST` and fill it (skipping
/// the write for empty data). Shared by the `prepare` `mk` closure and the T9
/// content-addressed arena cache so a cached buffer is byte-/size-/usage-
/// identical to a freshly created one.
fn create_filled_buffer(
    gpu: &GpuContext,
    label: &str,
    data: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let size = (data.len().max(4) as u64).next_multiple_of(4);
    gpu.admission.charge(size, data.len() as u64);
    let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        gpu.queue.write_buffer(&buf, 0, data);
    }
    buf
}

/// Blend factors used by the four text pipelines (extracted manifest data).
fn text_blend_factor(s: &str) -> wgpu::BlendFactor {
    match s {
        "zero" => wgpu::BlendFactor::Zero,
        "one" => wgpu::BlendFactor::One,
        "src-alpha" => wgpu::BlendFactor::SrcAlpha,
        "one-minus-src-alpha" => wgpu::BlendFactor::OneMinusSrcAlpha,
        other => panic!("unmapped text blend factor '{other}'"),
    }
}

struct PreparedDraw {
    module: String,
    offsets: Vec<u32>,
    tex: Option<usize>,
    v_off: u64,
    v_len: u64,
    i_off: u64,
    i_len: u64,
    index_count: u32,
}

// Track C — vector-engine draw records (group-0 flavor indexes the pending
// list built in prepare()).
const VEC_FLAVOR_TESS: usize = 0;
const VEC_FLAVOR_ANALYTIC: usize = 1;
const VEC_FLAVOR_GRADIENT: usize = 2;

enum VectorDrawKind {
    /// `drawSolidTriangles`: slot 0 = flat xy triangles, non-indexed.
    Tess { vertex_count: u32 },
    /// Instanced glyph quads: slot 0 = unit quad, slot 1 = packed VectorAnalyticRecord instances.
    Analytic { instance_count: u32 },
}

struct PreparedVectorDraw {
    module: String,
    kind: VectorDrawKind,
    /// Index into `PreparedText::vector_binds`.
    group: Option<usize>,
    /// Dynamic offsets in binding order (DrawUniforms[, GradientUniforms]).
    dynamic_offsets: Vec<u32>,
    v_off: u64,
    v_len: u64,
}

// A packet needs a proof for the immutable part of the draw, even after the planner
// evicts the source batch. The host key identifies node-local geometry. A scale change can
// affect baked glyph padding. Batches with a padding uniform keep their local geometry at any affine.
fn uniform_geometry(batch: &TextQuadBatch) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    let key = batch.paint_geometry_key?;
    let vector = batch.vector.as_ref()?;
    if !batch.ops.is_empty() || batch.degraded.is_some()
        || !vector.emojis.is_empty() || vector.gradient.is_some()
        || vector.ops.iter().any(|op| op.gradient.is_some()) { return None; }
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    let mut model = batch.model_matrix;
    if vector.uniform_padding.is_some() {
        for i in [0, 1, 4, 5] { model[i] = 0.; }
    }
    model[12] = 0.;
    model[13] = 0.;
    vector.baked_padding_scale.map(f64::to_bits).hash(&mut hash);
    (key, vector.uniform_padding.map(|(size, scale)| (size.to_bits(), scale.to_bits())), &batch.node_id, model.map(f32::to_bits), batch.z.to_bits(),
        batch.page_clip_bounds.map(f32::to_bits), batch.distort.map(|h| h.map(f32::to_bits))).hash(&mut hash);
    for op in &vector.ops {
        (op.pipeline as u8, op.additive, op.stroke_px.to_bits(), op.blur_px.to_bits()).hash(&mut hash);
    }
    Some(hash.finish())
}

type UniformSlot = [u8; UNIFORM_ALIGN as usize];

pub(crate) fn padding_words(pad: f64) -> [f32; 2] {
    let hi = pad as f32;
    [hi, (pad - f64::from(hi)) as f32]
}

/// The same DrawUniforms words for collection and for updates of the retained slot.
fn vector_uniform(batch: &TextQuadBatch, op: &moda_text::vector::VectorOp, depth: f32) -> [f32; 40] {
    let mut u = [0.; 40];
    u[..16].copy_from_slice(&batch.model_matrix);
    u[16..20].copy_from_slice(&op.color);
    u[20..24].copy_from_slice(&[depth, op.opacity, op.blur_px, op.stroke_px]);
    u[24..28].copy_from_slice(&batch.page_clip_bounds);
    if let Some(h) = &batch.distort {
        u[28..31].copy_from_slice(&h[..3]);
        u[31] = 1.;
        u[32..35].copy_from_slice(&h[3..6]);
        u[36..39].copy_from_slice(&h[6..]);
    } else {
        u[28] = 1.;
        u[33] = 1.;
        u[38] = 1.;
    }
    let words = if batch.vector.as_ref().is_some_and(|v| v.uniform_padding.is_some()) {
        padding_words(batch.quad_padding())
    } else { [-1., 0.] };
    u[35] = words[0];
    u[39] = words[1];
    u
}

struct PreparedBatch {
    geometry: Option<u64>,
    draws: Vec<PreparedDraw>,
    /// Track C: vector-engine draws (set ⇔ the batch carries `vector` ops).
    vector_draws: Vec<PreparedVectorDraw>,
}

/// All GPU resources for a frame's text draws (built by `prepare`). The
/// prepared-frame fast path (M2) stores this behind an `Arc<PreparedText>` on its
/// packet, so a hit reuses it with an O(1) refcount bump — never a deep clone of
/// the inner draw metadata (hence no `Clone` derive: the packet shares the Arc).
#[derive(Default)]
pub struct PreparedText {
    /// The pipelines this text's draws use, by module name (see `prepare`).
    pipes: Vec<(String, wgpu::RenderPipeline)>,
    /// Kept for ownership clarity; the bind groups hold their own refs.
    _uniform_buf: Option<wgpu::Buffer>,
    uniforms: std::cell::RefCell<Column<UniformSlot>>,
    /// T9: `Arc` so a content-cache hit shares the already-uploaded buffer with
    /// the cache (an `Arc` pointer-bump, not a re-upload) while this
    /// `PreparedText` still keeps it alive for the frame.
    vbuf: Option<Arc<wgpu::Buffer>>,
    ibuf: Option<Arc<wgpu::Buffer>>,
    bind0: Option<wgpu::BindGroup>,
    textures: Vec<Arc<wgpu::BindGroup>>,
    batches: Vec<PreparedBatch>,
    /// Run starts in batch order. Fonts and render scales usually agree across a text chunk.
    padding: Vec<(usize, Option<(f64, f64)>)>,
    sample_count: u32,
    /// Track C — vector engine: static unit quad, group-0 bind groups
    /// (tess/analytic/gradient flavors), shader-fill curve groups, and the
    /// per-node curve storage buffers (kept alive for the frame).
    quad_buf: Option<wgpu::Buffer>,
    vector_binds: Vec<wgpu::BindGroup>,
    /// T9: `Arc` per-node curve storage (shared with the content cache on a hit).
    _curve_bufs: Vec<Arc<wgpu::Buffer>>,
    /// M2 prepared-frame counter: bytes ACTUALLY uploaded to the text arenas this
    /// `prepare` — uniform + vertex + index + per-node curve storage (T9). The
    /// always-rebuilt `uniforms` are counted every prepare; `verts`/`indices`/curve
    /// bytes count only on a content-cache MISS (0 on a hit). Surfaced as
    /// `pf_text_bytes`, so it plateaus to ~uniforms-only once a transform-only drag
    /// hits the arena cache every frame.
    uploaded_bytes: u64,
}

impl PreparedText {
    /// A placement recipe preserves this batch's geometry and draw order. A
    /// replacement also compares the host's content and immutable draw fields.
    pub fn can_update_uniforms(&self, at: usize, new: &TextQuadBatch, placement_only: bool) -> bool {
        // The vector uniform writer cannot install the selected face's texture binding.
        // Check both sides, including a source re-emitted into a certified placement slot.
        self.batches.get(at).is_some_and(|b| b.geometry.is_some()
            && new.vector.as_ref().is_some_and(|v| v.emojis.is_empty() && v.ops.len() == b.vector_draws.len())
            && (placement_only || b.geometry == uniform_geometry(new)))
    }

    pub fn retain_uniform_tails(&self, budget: &mut usize) {
        self.uniforms.borrow_mut().retain_tails(budget);
    }

    pub fn padding_bytes(&self) -> usize {
        self.padding.capacity() * std::mem::size_of::<(usize, Option<(f64, f64)>)>()
    }

    fn padding_at(&self, at: usize) -> Option<(f64, f64)> {
        let end = self.padding.partition_point(|&(start, _)| start <= at);
        self.padding.get(end.checked_sub(1)?)?.1
    }

    pub fn admits_matrix(&self, at: usize) -> bool {
        self.batches.get(at).is_some_and(|batch| batch.geometry.is_some()) && self.padding_at(at).is_some()
    }

    pub fn stage_matrix(&self, at: usize, matrix: &[f32; 16]) {
        let params = self.padding_at(at).expect("certified uniform padding");
        let padding = padding_words(TextQuadBatch::matrix_padding(matrix, params));
        let mut column = self.uniforms.borrow_mut();
        for draw in &self.batches[at].vector_draws {
            let index = draw.dynamic_offsets[0] / UNIFORM_ALIGN as u32;
            column.put_matrix(index, matrix, Some(padding)).expect("prepared text uniform");
        }
    }

    pub fn fit_uniforms(&self, bytes: usize) -> usize {
        self.uniforms.borrow_mut().fit_budget(bytes)
    }

    /// Stage values by batch address; commit once for this retained uniform column.
    pub fn stage_paint(&self, at: usize, batch: &TextQuadBatch, translated: bool) {
        let mut column = self.uniforms.borrow_mut();
        let depth = crate::depth::linear_depth(batch.z) as f32;
        for (draw, op) in self.batches[at].vector_draws.iter().zip(&batch.vector.as_ref().unwrap().ops) {
            let index = draw.dynamic_offsets[0] / UNIFORM_ALIGN as u32;
            let mut slot = [0; UNIFORM_ALIGN as usize];
            let uniform = vector_uniform(batch, op, depth);
            let data = bytemuck::bytes_of(&uniform);
            slot[..data.len()].copy_from_slice(data);
            let range = if translated { 48..56 } else { 0..data.len() };
            column.put_words(index, &slot, range).expect("prepared text uniform");
        }
    }

    pub fn commit_paint(&self, mut write: impl FnMut(&wgpu::Buffer, u64, &[u8])) -> u64 {
        let mut written = 0;
        self.uniforms.borrow_mut().commit(&mut |offset, bytes| {
            write(self._uniform_buf.as_ref().unwrap(), offset, bytes);
            written += bytes.len() as u64;
        });
        written
    }

    fn pipe(&self, module: &str) -> Option<&wgpu::RenderPipeline> {
        self.pipes.iter().find(|(m, _)| m == module).map(|(_, p)| p)
    }

    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    /// Bytes uploaded to the text arenas by the `prepare` that built this (0 on
    /// the default/empty value). Read by the engine's `pf_text_bytes` counter.
    pub fn uploaded_bytes(&self) -> u64 {
        self.uploaded_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_updates_reject_recollected_baked_padding_at_another_scale() {
        let batch = |scale| TextQuadBatch {
            node_id: "stroke".into(), model_matrix: [0.; 16], z: 1., page_clip_bounds: [0.; 4],
            ops: Vec::new(), degraded: None, emoji_engine: None, distort: None,
            vector: Some(moda_text::vector::VectorNodeOps { baked_padding_scale: Some(scale), ..Default::default() }),
            paint_geometry_key: Some([7; 4]), content_key: None,
        };
        let original = batch(1.);
        let collected = PreparedText { batches: vec![PreparedBatch { geometry: uniform_geometry(&original),
            draws: vec![], vector_draws: vec![] }], ..Default::default() };
        assert!(collected.can_update_uniforms(0, &batch(1.), false));
        assert!(!collected.can_update_uniforms(0, &batch(2.), false));

        // An image in the restored source needs a texture binding even if the
        // node key, baked padding and number of analytic draws still match.
        let mut with_image = batch(1.);
        with_image.vector.as_mut().unwrap().emojis.push(moda_text::vector::VectorEmoji {
            cluster: "✅".into(), shaped: None, ppem: 64, corners: [0.; 8], opacity: 1., effect: None,
        });
        for placement_only in [false, true] {
            assert!(!collected.can_update_uniforms(0, &with_image, placement_only));
        }
    }
}
