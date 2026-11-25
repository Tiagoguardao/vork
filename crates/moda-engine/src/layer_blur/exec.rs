//! Exec side of the layer filters (layer blur, noise, texture): the
//! isolation-capture mini plan, the filter passes over the capture, and the
//! engine-owned composite pipeline the main pass draws at the node's z.
//!
//! GPU flow per `LayerBlurGroup` (driven from `render_plan_to_texture_impl`'s
//! pre-pass, BEFORE the main scene pass — the mask-atlas slot):
//!   1. `build_capture_plan` → a mini `CoveragePlan` holding just the node's
//!      diverted items; the caller renders it through the ordinary plan
//!      executor, which handles pattern/text/shader
//!      resources exactly like any page render.
//!   2. `filter_capture` → the group's `LayerFilters` in order, each pass
//!      reading the previous one's output: texture (`layer-filter-texture`),
//!      blur (`backdrop-blur` passes — backdrop's ladder past σ = 5 — or two
//!      `progressive-blur` passes, H then V),
//!      then every noise (`layer-filter-noise`), then the feather
//!      (`layer-filter-shadow` `fs_feather`). The frame encoder records these
//!      passes after the source draw and before the composite.
//!   3. The main pass binds the result through `composite_bind_group` and
//!      draws one quad via `composite_pipeline` in the node's z bucket.

use anyhow::{anyhow, Result};

use crate::backdrop::kernel::{backdrop_blur_passes, blur_pass_uniform, MAX_BLUR_KERNEL_SIZE};
use crate::scratch::Bound;
use super::batch::{FilterDraw, FilterPipeline, FilterSurface};
use std::sync::Arc;
use crate::seams::{
    BlurFilter, CaptureBlend, ClipRRect, CoveragePlan, LayerBlurGroup, NoiseFilter,
    ShadowPass, TextureFilter,
};

/// Composite uniform: rect(4) + clip(4) + params(4) + rounded clip rect(4) +
/// its corner radii(4) floats — see `layer-blur-composite.wgsl`.
pub const COMPOSITE_UNIFORM_FLOATS: usize = 20;

/// The uniform blur's pass targets: filterable, renderable everywhere, and
/// free of 8-bit steps (the result may be magnified several times).
const BLUR_FLOAT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// `MAX_TAPS` of `progressive-blur.wgsl`: taps per side, so a reach (3σ) up to
/// this many px samples every texel.
pub(crate) const PROGRESSIVE_BLUR_MAX_TAPS: f32 = 32.0;

/// Keep the reduced Gaussian well sampled (at least 12.5 texels per sigma
/// after a halving). Each level integrates a whole 2×2 source footprint.
pub(super) fn uniform_blur_block(mut reach: f32) -> u32 {
    let mut block = 1;
    while reach > 75.0 && block < 8192 {
        reach *= 0.5;
        block *= 2;
    }
    block
}


fn shader(device: &wgpu::Device, id: &str) -> Result<wgpu::ShaderModule> {
    let src = moda_shaders::module_source(id)
        .ok_or_else(|| anyhow!("missing extracted WGSL module '{id}'"))?;
    Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(id),
        source: wgpu::ShaderSource::Wgsl(src.source.into()),
    }))
}

thread_local! {
    static SUPERSAMPLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Restores the previous [`enter_supersampled_capture`] state on drop.
pub struct SupersampledGuard(bool);

impl Drop for SupersampledGuard {
    fn drop(&mut self) {
        SUPERSAMPLED.with(|c| c.set(self.0));
    }
}

/// Renders on this thread until the guard drops are supersampled captures: a target drawn at least 2× sharper than
/// it is shown and filtered down (moda-cli's 3D units), single-sampled. Their capture shadows take the body's one
/// sample too, rather than MSAA ×4: the downsampling already antialiases the knockout at the body's edge.
pub fn enter_supersampled_capture() -> SupersampledGuard {
    SupersampledGuard(SUPERSAMPLED.with(|c| c.replace(true)))
}

/// Whether [`enter_supersampled_capture`] is in force on this thread.
pub fn supersampled_capture() -> bool {
    SUPERSAMPLED.with(std::cell::Cell::get)
}

/// Unfiltered solid pixels depend on the actual capture projection and draw inputs.
pub(crate) fn solid_capture_source_key(
    group: &LayerBlurGroup,
    plan: &CoveragePlan,
    source_offset: [u32; 2],
    clip: Option<[u32; 4]>,
) -> Option<u64> {
    use crate::seams::{BodyPaint, DrawItem};
    use std::hash::{Hash, Hasher};

    if !group.masks.is_empty() || !group.coverage.is_empty() {
        return None;
    }
    let local = super::translation::eligible(group);
    let mut h = std::collections::hash_map::DefaultHasher::new();
    b"unfiltered-solid-source".hash(&mut h);
    (group.width_px, group.height_px, plan.sample_count, clip, local, source_offset,
        plan.width_px, plan.height_px, plan.page_size.map(f64::to_bits), plan.camera.map(f64::to_bits)).hash(&mut h);
    if !local {
        group.region_canvas.map(f64::to_bits).hash(&mut h);
    }
    for item in &group.items {
        let DrawItem::Body {
            geometry,
            instance,
            paint: BodyPaint::Solid,
            phase,
            blend_mode,
            z,
            mask: None,
            ..
        } = item
        else {
            return None;
        };
        let mut instance = if local {
            super::translation::instance(instance, group.region_canvas)
        } else {
            **instance
        };
        // A lone, fill-only path has no depth competitor. Every triangle has
        // the same depth. Keep both clipping planes and the clear depth well
        // outside the range where Depth24Plus quantization could change a test.
        let independent_depth = group.items.len() == 1
            && (0.01..0.99).contains(&instance.path_info[3])
            && geometry.vertex_types.iter().all(|&kind| kind == 0.);
        if independent_depth {
            instance.path_info[3] = 0.5;
        }
        let order = if independent_depth { 0 } else { z.to_bits() };
        h.write(bytemuck::bytes_of(&instance));
        (order, *phase as u8, *blend_mode as u8, geometry.vertex_count).hash(&mut h);
        bytemuck::cast_slice::<f32, u8>(&geometry.positions).hash(&mut h);
        bytemuck::cast_slice::<f32, u8>(&geometry.vertex_types).hash(&mut h);
    }
    Some(h.finish())
}

fn capture_samples(group: &LayerBlurGroup) -> u32 {
    let gaussian_coverage = matches!(group.filters.blur, Some(BlurFilter::Uniform { .. } | BlurFilter::Elliptical { .. }))
        && group.items.iter().any(|item| !matches!(item, crate::seams::DrawItem::Vacant
            | crate::seams::DrawItem::LayerBlurComposite { clip_rrect: None, mask: None, .. }));
    if gaussian_coverage || group.filters.shadow.is_some() || !group.filters.drop_shadows.is_empty() { 4 } else { 1 }
}

/// Crop only the source raster. The first Gaussian or reduction pass restores
/// its placement on the original grid. A clear guard texel makes the existing
/// clamp sampler equivalent to transparent extension at each new edge.
pub(crate) fn crop_gaussian_source(plan: &mut CoveragePlan, group: &LayerBlurGroup) -> [u32; 2] {
    let f = &group.filters;
    if !matches!(f.blur, Some(BlurFilter::Uniform { .. })) || f.blur_border != 0
        || f.texture.is_some() || !f.drop_shadows.is_empty() || f.shadow.is_some()
        || !plan.items.iter().all(|item| match item {
            crate::seams::DrawItem::Body { mask: None, paint, .. } => match paint {
                crate::seams::BodyPaint::Solid => true,
                // Dither hashes the framebuffer position. Preserve its origin.
                crate::seams::BodyPaint::Gradient(g) => {
                    let stroke_flags = (g.stroke_gradient_meta[3].round() as u32) / 4;
                    (g.gradient_flags.round() as u32 | stroke_flags) & 8 == 0
                }
                crate::seams::BodyPaint::Pattern(_) => false,
            },
            _ => false,
        })
    {
        return [0, 0];
    }
    let Some(bounds) = super::shadow::items_canvas_bounds(&plan.items) else { return [0, 0] };
    let mut offset = [0; 2];
    let mut size = [plan.width_px, plan.height_px];
    for (axis, (lo, extent)) in [(bounds.x, bounds.w), (bounds.y, bounds.h)].into_iter().enumerate() {
        let px = plan.page_size[axis] / f64::from(size[axis]);
        let start = ((lo - plan.camera[axis]) / px).floor() - 1.0;
        let end = ((lo + extent - plan.camera[axis]) / px).ceil() + 1.0;
        offset[axis] = start.clamp(0.0, f64::from(size[axis] - 1)) as u32;
        let end = end.clamp(f64::from(offset[axis] + 1), f64::from(size[axis])) as u32;
        size[axis] = end - offset[axis];
        plan.camera[axis] += f64::from(offset[axis]) * px;
        plan.page_size[axis] = f64::from(size[axis]) * px;
    }
    [plan.width_px, plan.height_px] = size;
    offset
}


/// Build the isolation-capture mini plan for one group. The plan maps the
/// padded canvas region 1:1 onto a `width_px × height_px` transparent target
/// via the ordinary page-local projection (camera = region origin, page_size
/// = region canvas size — the physical scale comes solely from the target
/// dimensions, exactly like an export plan).
pub fn build_capture_plan(node_id: &str, group: &LayerBlurGroup) -> CoveragePlan {
    // Nested filtered descendants are rendered by the same executor before this
    // group's filter. The owned capture tree follows the scene hierarchy.
    let [rx, ry, rw, rh] = group.region_canvas;
    let local = super::translation::eligible(group);
    let mut items = group.items.clone();
    if local {
        for item in &mut items {
            if let crate::seams::DrawItem::Body { instance, .. } = item {
                **instance = super::translation::instance(instance, group.region_canvas);
            }
        }
    }
    CoveragePlan {
        page_id: format!("layer-blur:{node_id}"),
        page_index: 0,
        width_px: group.width_px,
        height_px: group.height_px,
        page_size: if local { [group.width_px as f64, group.height_px as f64] } else { [rw, rh] },
        // The capture's canvas→px scale (informational; the projection derives
        // solely from page_size × target dims).
        scale: group.width_px as f64 / rw.max(1e-6),
        transparent: true,
        // A shadow capture's coverage also knocks out / clips the shadows at
        // the body's edge, so it is antialiased like the body (MSAA 4; the
        // executor clamps to what the device takes). Single-sample coverage
        // also let the vector text lane's ray-cast leak a hairline where a
        // pixel centre sits exactly on a glyph's horizontal extremum.
        // A motion capture stands in for the layer's own antialiased draw.
        // A filter shadow's capture is coverage too: the shadows' edges are the layer's antialiased edges, and
        // under another filter the capture is drawn back as the layer itself.
        sample_count: capture_samples(group),
        camera: if local { [0.0, 0.0] } else { [rx, ry] },
        items,
        item_order: None,
        masks: group.masks.clone(),
        camera_frame: None,
        clear_rgba: Some([0.0, 0.0, 0.0, 0.0]),
        region_viewport: None,
    }
}

/// Per-render layer-blur GPU resources (blur + composite pipelines). Built
/// once per `render_plan_to_texture_impl` call that has layer-blur items —
/// the same per-call posture as `BackdropRenderer`.
pub struct LayerBlurRenderer {
    /// The uniform blur's passes, into [`BLUR_FLOAT_FORMAT`] targets.
    blur: Arc<FilterPipeline>,
    border: Arc<FilterPipeline>,
    integrate: Arc<FilterPipeline>,
    color_matrix: Arc<FilterPipeline>,
    progressive_blur: Arc<FilterPipeline>,
    texture: Arc<FilterPipeline>,
    noise: Arc<FilterPipeline>,
    shadow_prepare: Arc<FilterPipeline>,
    shadow_composite: Arc<FilterPipeline>,
    /// The capture drawn source-over on its filter shadows (`layer-filter-shadow` `fs_source`, the filter
    /// layout).
    shadow_source: Arc<FilterPipeline>,
    /// Feather: the layer's hard alpha (`layer-filter-shadow`
    /// `fs_feather_source`, the filter layout), then the layer scaled by its
    /// blur (`fs_feather`, the shadow layout: the blur at binding 2, the layer
    /// at 3).
    feather_source: Arc<FilterPipeline>,
    feather: Arc<FilterPipeline>,
    device: wgpu::Device,
    // Capture and glow blends share a layout. Compile only blends a page actually draws.
    composites: [std::sync::OnceLock<wgpu::RenderPipeline>; 4],
    composite_layout: wgpu::PipelineLayout,
    frame_bgl: wgpu::BindGroupLayout,
    mask_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,
    composite_module: wgpu::ShaderModule,
    sample_count: u32,
    layer_composite: crate::device::OptionalPipeline<(wgpu::BindGroupLayout, wgpu::RenderPipeline)>,
    sampler_linear: wgpu::Sampler,
    /// The owning engine's immutable frame color format — the filter passes'
    /// textures and color targets use it, except the uniform blur's (float,
    /// [`BLUR_FLOAT_FORMAT`]); the capture source is the shared frame target,
    /// and the composite draws into the main-pass frame attachment.
    frame_format: wgpu::TextureFormat,

}

fn premultiplied_blend() -> wgpu::BlendState {
    wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        }
}

fn make_composite_pipeline(device: &wgpu::Device, module: &wgpu::ShaderModule, layout: &wgpu::PipelineLayout,
    frame_format: wgpu::TextureFormat, sample_count: u32, blend: wgpu::BlendState,
    vertex: &str, fragment: &str, label: &str, creation: moda_text_draw::device::PipelineCreation) -> wgpu::RenderPipeline {
    creation.render(device, &wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{label} s{sample_count}")),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: module,
                    entry_point: Some(vertex),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: module,
                    entry_point: Some(fragment),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: frame_format,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            })
}

impl LayerBlurRenderer {
    /// `frame_bgl` is the engine's shared group-0 frame-uniform layout;
    /// `mask_bgl` the group-2 clip-mask layout every path pipeline binds (the
    /// composite takes the node's clip mask there, or the inactive group);
    /// `sample_count` / `clear_is_opaque` describe the MAIN pass the composite
    /// quad draws into (the blur passes themselves are always single-sample).
    /// `frame_format` is the owning engine's immutable frame color format
    /// — see the field doc.
    pub fn new(device: &wgpu::Device, frame_bgl: &wgpu::BindGroupLayout, mask_bgl: &wgpu::BindGroupLayout,
        sample_count: u32, frame_format: wgpu::TextureFormat, working: Option<&moda_shaders::WorkingColor>) -> Result<Self> {
        Self::with_variants(device, frame_bgl, mask_bgl, sample_count, frame_format, working, &Default::default())
    }

    pub(crate) fn with_variants(
        device: &wgpu::Device,
        frame_bgl: &wgpu::BindGroupLayout,
        mask_bgl: &wgpu::BindGroupLayout,
        sample_count: u32,
        frame_format: wgpu::TextureFormat,
        working: Option<&moda_shaders::WorkingColor>,
        library: &super::batch::FilterLibrary,
    ) -> Result<LayerBlurRenderer> {
        let frag = wgpu::ShaderStages::FRAGMENT;
        let uniform_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: frag,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: frag,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };

        // Blur pass resources — same shape as backdrop's effect BGL
        // (uniform / sampler / texture) driving the shared `backdrop-blur`
        // module at sample_count 1.
        let effect_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Layer Blur Effect BGL"),
            entries: &[uniform_entry(0, frag), sampler_entry(1), texture_entry(2)],
        });
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Layer Shadow BGL"),
            entries: &[
                uniform_entry(0, frag),
                sampler_entry(1),
                texture_entry(2),
                texture_entry(3),
            ],
        });
        let filter = |id: &'static str, entry, auxiliary: Option<&'static str>, blend, format| -> Result<_> {
            let source = if id == "css-color-matrix" {
                include_str!("color_matrix.wgsl")
            } else if id == "capture-coverage" {
                include_str!("coverage.wgsl")
            } else {
                moda_shaders::module_source(id).ok_or_else(|| anyhow!("missing filter {id}"))?.source
            };
            let source = moda_shaders::working_color::specialize(working, source.into());
            Ok(FilterPipeline::new(id, entry, source, if auxiliary.is_some() { &shadow_bgl } else { &effect_bgl }, format, auxiliary, blend, library))
        };
        let blur = filter("backdrop-blur", "fs", None, None, BLUR_FLOAT_FORMAT)?;
        let add = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add };
        let integrate = filter("capture-coverage", "fs", None, Some(wgpu::BlendState { color: add, alpha: add }), BLUR_FLOAT_FORMAT)?;
        let border = filter("backdrop-blur", "fs_border", None, None, BLUR_FLOAT_FORMAT)?;
        let progressive_blur = filter("progressive-blur", "fs", None, None, frame_format)?;
        let texture = filter("layer-filter-texture", "fs", None, None, frame_format)?;
        let noise = filter("layer-filter-noise", "fs", Some("coverageTexture"), None, frame_format)?;
        let color_matrix = filter("css-color-matrix", "fs", None, None, frame_format)?;
        let shadow = |entry, auxiliary, blend| filter("layer-filter-shadow", entry, auxiliary, blend, frame_format);
        let over = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        let shadow_prepare = shadow("fs_prepare", None, None)?;
        let shadow_composite = shadow("fs_composite", Some("captureTexture"), over)?;
        let shadow_source = shadow("fs_source", None, over)?;
        let feather_source = shadow("fs_feather_source", None, None)?;
        let feather = shadow("fs_feather", Some("captureTexture"), None)?;

        // Composite pipeline — engine-owned (manifest entry
        // `layer-blur-composite` carries `pipelines: []` like `shape-solid`):
        // premultiplied blend, transparent-phase depth (write OFF, compare
        // Less), MSAA count matching the main pass.
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Layer Blur Composite BGL"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                sampler_entry(1),
                texture_entry(2),
            ],
        });
        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Layer Blur Composite Pipeline Layout"),
            bind_group_layouts: &[Some(frame_bgl), Some(&composite_bgl), Some(mask_bgl)],
            ..Default::default()
        });
        let composite_mod = shader(device, "layer-blur-composite")?;
        let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Layer Blur Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Ok(LayerBlurRenderer {
            frame_format,

            blur,
            integrate,
            border,
            color_matrix,
            progressive_blur,
            texture,
            noise,
            shadow_prepare,
            shadow_composite,
            shadow_source,
            feather_source,
            feather,
            device: device.clone(),
            composites: Default::default(),
            composite_layout,
            frame_bgl: frame_bgl.clone(),
            mask_bgl: mask_bgl.clone(),
            composite_bgl,
            composite_module: composite_mod,
            sample_count,
            layer_composite: Default::default(),
            sampler_linear,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn prepare_filters(self: &Arc<Self>, device: &wgpu::Device, compiler: &moda_text_draw::device::OptionalCompiler) {
        if device.limits().max_storage_textures_per_shader_stage > 0 {
            for filter in [&self.blur, &self.progressive_blur, &self.shadow_prepare, &self.shadow_composite,
                &self.shadow_source, &self.feather_source, &self.feather] {
                if filter.seen(compiler, 0) { filter.request_array(device); }
            }
        }
        let max = FilterPipeline::max_targets(device);
        for width in (2..=max).rev() {
            for filter in [&self.blur, &self.border, &self.color_matrix, &self.progressive_blur, &self.texture,
                &self.noise, &self.shadow_prepare, &self.shadow_composite, &self.shadow_source, &self.feather_source, &self.feather] {
                if filter.seen(compiler, width) { filter.raster_width(device, width, max); }
            }
        }
    }

    pub(crate) fn layer_composite(self: &Arc<Self>, device: &wgpu::Device) -> Option<&(wgpu::BindGroupLayout, wgpu::RenderPipeline)> {
        let renderer = self.clone();
        let samples = self.sample_count;
        self.layer_composite.request(device, "packed composite", move || (samples != 4).then(|| format!("canvas/composite/{samples}")), move |device, cache| {
            let entry = |binding, visibility, ty| wgpu::BindGroupLayoutEntry { binding, visibility, ty, count: None };
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("layer array composites"),
                entries: &[
                    entry(1, wgpu::ShaderStages::FRAGMENT, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                    entry(3, wgpu::ShaderStages::VERTEX_FRAGMENT, wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }),
                    entry(4, wgpu::ShaderStages::FRAGMENT, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false }),
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("layer array composites"), bind_group_layouts: &[Some(&renderer.frame_bgl),
                    Some(&layout), Some(&renderer.mask_bgl)], ..Default::default()
            });
            let pipeline = make_composite_pipeline(device, &renderer.composite_module, &pipeline_layout, renderer.frame_format,
                renderer.sample_count, premultiplied_blend(), "vs_layers", "fs_layers", "layer array composites",
                moda_text_draw::device::PipelineCreation::Optional(cache));
            Ok((layout, pipeline))
        });
        self.layer_composite.ready()
    }

    pub(crate) fn layer_composite_group(&self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout, records: &Bound, layers: &wgpu::TextureView) -> wgpu::BindGroup {
        let buffer = match records {
            Bound::Range(buffer, offset, size) => wgpu::BufferBinding { buffer, offset: *offset, size: Some(*size) },
            Bound::Buffer(buffer) => buffer.as_entire_buffer_binding(),
            _ => unreachable!(),
        };
        device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("layer array composites"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_linear) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(buffer) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(layers) },
            ],
        })
    }

    pub fn composite_pipeline(&self) -> &wgpu::RenderPipeline {
        self.blend_composite_pipeline(CaptureBlend::Normal)
    }

    /// The composite of a capture group with this blend (`ShadowFilter::blend`;
    /// every non-shadow group composites source-over).
    pub fn blend_composite_pipeline(&self, blend: CaptureBlend) -> &wgpu::RenderPipeline {
        use wgpu::BlendFactor as F;
        // All modes union alpha like source-over. Only their colour factors differ.
        let (slot, label, src, dst) = match blend {
            CaptureBlend::Normal => (0, "Layer Blur Composite Pipeline", F::One, F::OneMinusSrcAlpha),
            CaptureBlend::Screen => (1, "Layer Glow Screen Composite Pipeline", F::One, F::OneMinusSrc),
            CaptureBlend::Multiply => (2, "Layer Glow Multiply Composite Pipeline", F::Dst, F::OneMinusSrcAlpha),
            CaptureBlend::Add => (3, "Layer Glow Add Composite Pipeline", F::One, F::One),
        };
        self.composites[slot].get_or_init(|| {
            let mut state = premultiplied_blend();
            state.color.src_factor = src;
            state.color.dst_factor = dst;
            make_composite_pipeline(&self.device, &self.composite_module, &self.composite_layout,
                self.frame_format, self.sample_count, state, "vs", "fs", label, moda_text_draw::device::PipelineCreation::Required)
        })
    }

    /// Add only this tile's intersection with each reduced texel. A separate
    /// recording per tile lets the next raster reuse its resolved attachment.
    pub(crate) fn integrate_tile(&self, queue: &wgpu::Queue, scratch: &crate::scratch::ScratchPool,
        source: wgpu::TextureView, target: &FilterSurface, tile: &super::coverage::Tile, load: bool) {
        let source = FilterSurface::external(source);
        let mut draws = Vec::new();
        self.run_pass(Some(tile.scissor), &mut draws, &self.integrate,
            &[tile.first_edge[0], tile.first_edge[1], tile.step[0], tile.step[1],
              tile.scissor[0] as f32, tile.scissor[1] as f32, 0., 0.], &source, target);
        draws[0].load = load;
        let mut batch = super::batch::FilterBatch::default();
        batch.add(draws);
        batch.record(&self.device, queue, scratch, std::slice::from_ref(&source),
            std::slice::from_ref(target), || 0, Default::default());
    }

    /// Append the filter stages that consume this source. The batch selects their storage before recording.
    pub(crate) fn filter_capture(
        &self,
        scissor: Option<[u32; 4]>,
        draws: &mut Vec<FilterDraw>,
        capture: &FilterSurface,
        group: &LayerBlurGroup,
        source_offset: [u32; 2],
        coverage: Option<&FilterSurface>,
    ) -> Result<FilterSurface> {
        let _filter = crate::prof::span("fx.filter");
        let (width, height) = (group.width_px, group.height_px);
        if let Some(shadow) = &group.filters.shadow {
            return Ok(self.shadows_of(
                scissor,
                draws,
                capture,
                [width, height],
                &shadow.shadows,
                shadow.inner,
                false,
            ));
        }
        let filters = &group.filters;
        // Filter shadows alone: the result is the shadows, and the layer's own items draw over the composite.
        if filters.shadows_alone {
            return Ok(self.shadows_of(
                scissor,
                draws,
                capture,
                [width, height],
                &filters.drop_shadows,
                false,
                false,
            ));
        }
        // Motion captures sample the resolved source, which belongs to this frame generation.
        if filters.color_matrices.is_empty()
            && filters.drop_shadows.is_empty()
            && filters.texture.is_none()
            && filters.blur.is_none()
            && filters.noise.is_empty()
            && filters.feather.is_none()
        {
            return Ok(capture.clone());
        }
        let mk_tex = || FilterSurface::new([width, height], self.frame_format);
        let mut current = capture.clone();

        // Filter shadows first, under another filter (their own draws, submitted before the passes below):
        // the capture's alpha casts them and the capture is drawn over them, so every later filter runs over
        // the picture with its shadows.
        if !filters.drop_shadows.is_empty() {
            current = self.shadows_of(
                scissor,
                draws,
                &current,
                [width, height],
                &filters.drop_shadows,
                false,
                true,
            );
        }
        // Each pass writes a fresh texture: the final one outlives this call
        // as the composite's source, and no pass reads what it writes.
        let mut step_as = |extent: [u32; 2], format: wgpu::TextureFormat,
                           pipeline: &Arc<FilterPipeline>,
                           uniform: &[f32],
                           input: &FilterSurface| {
            let out = FilterSurface::new(extent, format);
            self.run_pass(scaled(scissor, [width, height], extent), draws, pipeline, uniform, input, &out);
            out
        };
        let frame_format = self.frame_format;
        let size = [width as f32, height as f32];

        if let Some(t) = &filters.texture {
            current = step_as([width, height], frame_format,
                &self.texture,
                &texture_uniform(t, size, filters.origin_px),
                &current,
            );
        }
        match filters.blur {
            Some(BlurFilter::Uniform { radius_px }) => {
                let mut cropped = (capture.width(), capture.height()) != (width, height);
                let mut place_source = |mut uniform: Vec<f32>| {
                    if std::mem::take(&mut cropped) {
                        uniform[5] = 1.0;
                        uniform[6..8].copy_from_slice(&source_offset.map(|x| x as f32));
                    }
                    uniform
                };
                let mut extent = [width, height];
                let mut block = uniform_blur_block(radius_px);
                let mut reach = radius_px;
                while block > 1 && extent.iter().all(|n| n % 2 == 0) {
                    extent = extent.map(|n| n / 2);
                    let uniform = place_source(bilinear_copy_uniform(extent));
                    current = step_as(extent, BLUR_FLOAT_FORMAT, &self.blur, &uniform, &current);
                    reach *= 0.5;
                    block /= 2;
                }
                let border = filters.blur_border;
                let padded = extent.map(|n| n + 2 * border);
                if border > 0 {
                    current = step_as(padded, BLUR_FLOAT_FORMAT,
                        &self.border, &border_uniform(-(border as f32)), &current);
                }
                let size = padded.map(|n| n as f32);
                let passes = backdrop_blur_passes((reach as f64).max(0.5));
                for direction in 0..2 {
                    for pass in &passes {
                        let uniform = place_source(blur_pass_uniform(pass, direction, size));
                        current = step_as(padded,
                            BLUR_FLOAT_FORMAT,
                            &self.blur,
                            &uniform,
                            &current,
                        );
                    }
                }
                if border > 0 {
                    current = step_as(extent, BLUR_FLOAT_FORMAT,
                        &self.border, &border_uniform(border as f32), &current);
                }
                // Following effects address the source in full capture pixels.
                if extent != [width, height] && (!filters.color_matrices.is_empty()
                    || !filters.noise.is_empty() || filters.feather.is_some()) {
                    current = step_as([width, height], BLUR_FLOAT_FORMAT,
                        &self.blur, &bilinear_copy_uniform([width, height]), &current);
                }
            }
            Some(BlurFilter::Elliptical { major, reaches }) => {
                for (axis, reach) in [major, [-major[1], major[0]]].into_iter().zip(reaches) {
                    if reach <= 0.5 {
                        continue;
                    }
                    for pass in backdrop_blur_passes(reach as f64) {
                        let mut uniform = blur_pass_uniform(&pass, 2, size);
                        uniform[6..8].copy_from_slice(&axis);
                        current = step_as([width, height], BLUR_FLOAT_FORMAT, &self.blur, &uniform, &current);
                    }
                }
            }
            Some(blur @ BlurFilter::Progressive { .. }) => {
                for direction in 0..2 {
                    current = step_as([width, height], frame_format,
                        &self.progressive_blur,
                        &progressive_blur_uniform(&blur, direction, size),
                        &current,
                    );
                }
            }
            None => {}
        }
        for matrix in &filters.color_matrices {
            current = step_as([width, height], frame_format, &self.color_matrix, matrix, &current);
        }
        for n in &filters.noise {
            let out = mk_tex();
            self.run_pass_with(
                scissor,
                draws,
                &self.noise,
                &noise_uniform(n, size, filters.origin_px),
                &current,
                Some(coverage.unwrap_or(&current)),
                &out,
            );
            current = out;
        }
        if let Some(f) = &filters.feather {
            // The layer's hard alpha, blurred, then the layer scaled by the
            // remapped blur.
            let source = mk_tex();
            self.run_pass(
                scissor,
                draws,
                &self.feather_source,
                &feather_uniform(size),
                &current,
                &source,
            );
            let blurred = self.blur_alpha(scissor, draws, source, [width, height], f.radius_px);
            let out = mk_tex();
            self.run_pass_with(
                scissor,
                draws,
                &self.feather,
                &feather_uniform(size),
                &blurred,
                Some(&current),
                &out,
            );
            current = out;
        }

        Ok(current)
    }

    /// The shadows of a capture, drawn back to front into a fresh texture.
    /// Per shadow: the capture's coverage (drop) or outside (`inner`) moved
    /// by the offset (`fs_prepare`), blurred with two `progressive-blur`
    /// passes at a constant radius (any size at full resolution), then
    /// tinted and knocked out by / clipped to the coverage (`fs_composite`,
    /// source-over into the result). A shadow group's result is its shadows
    /// alone; `under_source` (filter shadows) draws the capture itself over
    /// them (`fs_source`), the picture with its shadows.
    #[allow(clippy::too_many_arguments)]
    fn shadows_of(
        &self,
        scissor: Option<[u32; 4]>,
        draws: &mut Vec<FilterDraw>,
        capture_view: &FilterSurface,
        [width, height]: [u32; 2],
        shadows: &[ShadowPass],
        inner: bool,
        under_source: bool,
    ) -> FilterSurface {
        let size = [width as f32, height as f32];
        let mk_tex = || FilterSurface::new([width, height], self.frame_format);
        let result_view = mk_tex();
        for (i, pass) in shadows.iter().enumerate() {
            let uniform = shadow_uniform(pass, inner, size);
            let prepared = mk_tex();
            self.run_pass(
                scissor,
                draws,
                &self.shadow_prepare,
                &uniform,
                capture_view,
                &prepared,
            );
            let source = self.blur_alpha(scissor, draws, prepared, [width, height], pass.radius_px);
            draws.push(FilterDraw {
                pipeline: self.shadow_composite.clone(),
                parameters: uniform.to_vec(),
                sampler: self.sampler_linear.clone(),
                input: source,
                second: Some(capture_view.clone()),
                scissor,
                target: result_view.clone(),
                load: i > 0,
            });
        }
        if under_source {
            let uniform = feather_uniform(size);
            draws.push(FilterDraw {
                pipeline: self.shadow_source.clone(),
                parameters: uniform.to_vec(),
                sampler: self.sampler_linear.clone(),
                input: capture_view.clone(),
                second: None,
                scissor,
                target: result_view.clone(),
                load: !shadows.is_empty(),
            });
        }
        result_view
    }

    /// `source` blurred at a constant `radius_px` reach (3σ) with two
    /// `progressive-blur` passes (any size at full resolution); under 0.6 px
    /// it is returned as is (σ below 0.2 px is a pass-through). The result
    /// may be smaller than `size` (see below): read it with normalised uv.
    #[allow(clippy::too_many_arguments)]
    fn blur_alpha(
        &self,
        scissor: Option<[u32; 4]>,
        draws: &mut Vec<FilterDraw>,
        mut source: FilterSurface,
        size: [u32; 2],
        radius_px: f32,
    ) -> FilterSurface {
        if radius_px < 0.6 {
            return source;
        }
        let [width, height] = size;
        // Past `PROGRESSIVE_BLUR_MAX_TAPS` px of reach the blur's taps
        // spread wider than a texel, and a hard edge (every shadow's
        // source) blurs into visible steps. Blur at 1/2^n resolution
        // instead, so the taps stay a texel apart: the source is first
        // halved n times (a pass-through `progressive-blur` read at the
        // corner of each 2×2 block: their exact average, so the halved
        // source keeps the edge's sub-texel position), and the
        // composite's bilinear read (normalised uv) brings the smooth
        // result back up.
        let (mut bw, mut bh) = (width, height);
        let mut radius_px = radius_px;
        let mk_blur_tex = |w, h| FilterSurface::new([w, h], self.frame_format);
        while radius_px > PROGRESSIVE_BLUR_MAX_TAPS && bw > 2 && bh > 2 {
            let (hw, hh) = (bw.div_ceil(2), bh.div_ceil(2));
            radius_px *= hw as f32 / bw as f32;
            (bw, bh) = (hw, hh);
            let out = mk_blur_tex(bw, bh);
            let copy = BlurFilter::Uniform { radius_px: 0.0 };
            self.run_pass(
                scaled(scissor, [width, height], [bw, bh]),
                draws,
                &self.progressive_blur,
                &progressive_blur_uniform(&copy, 0, [bw as f32, bh as f32]),
                &source,
                &out,
            );
            source = out;
        }
        let blur = BlurFilter::Progressive {
            start_radius_px: radius_px,
            end_radius_px: radius_px,
            start_px: [0.0; 2],
            end_px: [0.0; 2],
        };
        for direction in 0..2 {
            let out = mk_blur_tex(bw, bh);
            self.run_pass(
                scaled(scissor, [width, height], [bw, bh]),
                draws,
                &self.progressive_blur,
                &progressive_blur_uniform(&blur, direction, [bw as f32, bh as f32]),
                &source,
                &out,
            );
            source = out;
        }
        source
    }

    /// One fullscreen filter pass: `uniform` at binding 0, the linear sampler
    /// at 1, `input` at 2, drawn into `output` (cleared).
    #[allow(clippy::too_many_arguments)]
    fn run_pass(
        &self,
        scissor: Option<[u32; 4]>,
        draws: &mut Vec<FilterDraw>,
        pipeline: &Arc<FilterPipeline>,
        uniform: &[f32],
        input: &FilterSurface,
        output: &FilterSurface,
    ) {
        self.run_pass_with(scissor, draws, pipeline, uniform, input, None, output);
    }

    /// [`Self::run_pass`] with a second input at binding 3 (the shadow layout,
    /// which noise shares): `second` = Some.
    #[allow(clippy::too_many_arguments)]
    fn run_pass_with(
        &self,
        scissor: Option<[u32; 4]>,
        draws: &mut Vec<FilterDraw>,
        pipeline: &Arc<FilterPipeline>,
        uniform: &[f32],
        input: &FilterSurface,
        second: Option<&FilterSurface>,
        output: &FilterSurface,
    ) {
        let out_view = output.clone();
        draws.push(FilterDraw {
            pipeline: pipeline.clone(),
            parameters: uniform.to_vec(),
            sampler: self.sampler_linear.clone(),
            input: input.clone(),
            second: second.cloned(),
            scissor,
            target: out_view,
            load: false,
        });
    }

    /// Uniform data for one composite quad — must match
    /// `LayerBlurCompositeUniforms` in `layer-blur-composite.wgsl`.
    pub fn composite_uniform_data(
        group: &LayerBlurGroup,
        page_clip: [f32; 4],
        clip_rrect: Option<ClipRRect>,
        depth: f32,
        opacity: f32,
    ) -> [f32; COMPOSITE_UNIFORM_FLOATS] {
        let [rx, ry, rw, rh] = group.region_canvas;
        // No rounded clip: a zero-size rect, which the shader skips.
        let rr = clip_rrect.unwrap_or(ClipRRect {
            rect: [0.0; 4],
            radii: [0.0; 4],
        });
        [
            rx as f32,
            ry as f32,
            rw as f32,
            rh as f32,
            page_clip[0],
            page_clip[1],
            page_clip[2],
            page_clip[3],
            depth,
            opacity,
            0.0,
            0.0,
            rr.rect[0],
            rr.rect[1],
            rr.rect[2],
            rr.rect[3],
            rr.radii[0],
            rr.radii[1],
            rr.radii[2],
            rr.radii[3],
        ]
    }

    /// The bind group of one composite draw: its uniform and the capture result it reads. `lasting`: the frame
    /// may keep the group past its render (a prepared frame's recorded pass binds it on later frames), so its
    /// uniform buffer is its own and not the pool's.
    pub fn composite_bind_group(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scratch: &crate::scratch::ScratchPool,
        lasting: bool,
        uniform_data: &[f32; COMPOSITE_UNIFORM_FLOATS],
        capture: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let uniform_binding = scratch.uniform(device, queue, "Layer Blur Composite Uniform", uniform_data, lasting);
        scratch.bind(device, "Layer Blur Composite Bind Group", &self.composite_bgl, &[
            uniform_binding,
            Bound::Sampler(self.sampler_linear.clone()),
            Bound::View(capture.clone()),
        ])
    }
}

/// Copy on the texel grid, with transparent extension outside the input.
fn border_uniform(offset: f32) -> Vec<f32> {
    let mut uniform = vec![0.0; 4 + MAX_BLUR_KERNEL_SIZE * 4];
    uniform[1] = offset;
    uniform
}

/// A unit centre tap addressed by destination pixel position. On halving,
/// bilinear sampling integrates the four source texels with equal weights.
fn bilinear_copy_uniform(extent: [u32; 2]) -> Vec<f32> {
    let mut uniform = vec![0.0; 4 + MAX_BLUR_KERNEL_SIZE * 4];
    uniform[..5].copy_from_slice(&[2.0, 1.0, extent[0] as f32, extent[1] as f32, 1.0]);
    uniform
}

/// `ProgressiveBlurUniforms` of `progressive-blur.wgsl` for one direction.
pub(crate) fn progressive_blur_uniform(
    blur: &BlurFilter,
    direction: u32,
    size: [f32; 2],
) -> [f32; 12] {
    let (start_r, end_r, start, end) = match *blur {
        BlurFilter::Progressive {
            start_radius_px,
            end_radius_px,
            start_px,
            end_px,
        } => (start_radius_px, end_radius_px, start_px, end_px),
        BlurFilter::Elliptical { .. } => unreachable!("elliptical blur uses the Gaussian pass ladder"),
        BlurFilter::Uniform { radius_px } => (radius_px, radius_px, [0.0; 2], [0.0; 2]),
    };
    [
        direction as f32,
        0.0,
        size[0],
        size[1],
        start[0],
        start[1],
        end[0],
        end[1],
        start_r,
        end_r,
        0.0,
        0.0,
    ]
}

/// `ShadowUniforms` of `layer-filter-shadow.wgsl`.
fn shadow_uniform(pass: &ShadowPass, inner: bool, size: [f32; 2]) -> [f32; 12] {
    [
        size[0],
        size[1],
        pass.offset_px[0],
        pass.offset_px[1],
        pass.color[0],
        pass.color[1],
        pass.color[2],
        pass.color[3],
        if inner { 1.0 } else { 0.0 },
        if pass.show_behind { 1.0 } else { 0.0 },
        if pass.center { 1.0 } else { 0.0 },
        0.0,
    ]
}

/// `ShadowUniforms` of `layer-filter-shadow.wgsl` for `fs_feather_source` /
/// `fs_feather` (they read only the texture size).
fn feather_uniform(size: [f32; 2]) -> [f32; 12] {
    let mut u = [0.0; 12];
    u[0] = size[0];
    u[1] = size[1];
    u
}

/// `TextureUniforms` of `layer-filter-texture.wgsl`.
fn texture_uniform(t: &TextureFilter, size: [f32; 2], origin: [f32; 2]) -> [f32; 8] {
    [
        size[0],
        size[1],
        origin[0],
        origin[1],
        t.size_px,
        t.radius_px,
        if t.clip_to_shape { 1.0 } else { 0.0 },
        0.0,
    ]
}

/// `NoiseUniforms` of `layer-filter-noise.wgsl`.
fn noise_uniform(n: &NoiseFilter, size: [f32; 2], origin: [f32; 2]) -> [f32; 20] {
    let secondary = n.secondary.unwrap_or([0.0; 3]);
    [
        size[0],
        size[1],
        origin[0],
        origin[1],
        n.size_px,
        n.density,
        n.opacity,
        n.mode as f32,
        n.color[0],
        n.color[1],
        n.color[2],
        1.0,
        secondary[0],
        secondary[1],
        secondary[2],
        if n.secondary.is_some() { 1.0 } else { 0.0 },
        n.seed as f32,
        noise_threshold(n.density),
        0.0,
        0.0,
    ]
}

/// The share of cells Figma's noise paints at `density` (0..1), each at a
/// ramp of its random value (`layer-filter-noise.wgsl`). Fitted to Figma's
/// own export (mono, density 0.6: 28 % of cells paint, the card darkens 5.5 %
/// on average; docs/qa/paste-fidelity.md §2.1): none at 0, every cell at 1.
pub fn noise_on_share(density: f32) -> f32 {
    density.clamp(0.0, 1.0).powf(2.5)
}

/// The cell-value threshold of [`noise_on_share`]: cells above it paint.
pub fn noise_threshold(density: f32) -> f32 {
    1.0 - noise_on_share(density)
}

/// CPU twin of `cellHash` in `layer-filter-noise.wgsl` /
/// `layer-filter-texture.wgsl`: bit-for-bit the same u32 arithmetic, so tests
/// can predict exactly which noise cells are on.
pub fn noise_cell_hash(x: i32, y: i32, salt: u32) -> f32 {
    fn hash_u32(x: u32) -> u32 {
        let mut h = x;
        h ^= h >> 16;
        h = h.wrapping_mul(0x7feb_352d);
        h ^= h >> 15;
        h = h.wrapping_mul(0x846c_a68b);
        h ^= h >> 16;
        h
    }
    let h = hash_u32(x as u32 ^ hash_u32(y as u32 ^ hash_u32(salt)));
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// `r` (texels of a `w × h` texture) cut to the texture; `None` when nothing is left.
pub(super) fn fit(r: [u32; 4], w: u32, h: u32) -> Option<[u32; 4]> {
    let (x0, y0) = (r[0].min(w), r[1].min(h));
    let (x1, y1) = (r[0].saturating_add(r[2]).min(w), r[1].saturating_add(r[3]).min(h));
    (x1 > x0 && y1 > y0).then_some([x0, y0, x1 - x0, y1 - y0])
}

/// `r` in the texels of a `from`-sized texture, as the texels of a `to`-sized one covering the same area (rounded
/// out).
fn scaled(r: Option<[u32; 4]>, from: [u32; 2], to: [u32; 2]) -> Option<[u32; 4]> {
    let r = r?;
    let (fx, fy) = (f64::from(to[0]) / f64::from(from[0].max(1)), f64::from(to[1]) / f64::from(from[1].max(1)));
    let x0 = (f64::from(r[0]) * fx).floor() as u32;
    let y0 = (f64::from(r[1]) * fy).floor() as u32;
    let x1 = (f64::from(r[0] + r[2]) * fx).ceil() as u32;
    let y1 = (f64::from(r[1] + r[3]) * fy).ceil() as u32;
    Some([x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)])
}
