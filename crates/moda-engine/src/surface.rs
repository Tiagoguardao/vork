//! Mip generation for pattern textures and decoder frames.

/// Levels of a full `w`×`h` mip chain (down to 1×1).
pub fn full_mip_levels(w: u32, h: u32) -> u32 {
    32 - w.max(h).max(1).leading_zeros()
}

/// GPU bytes of a `w`×`h` RGBA8 texture with `levels` mip levels.
pub fn texture_bytes(w: u32, h: u32, levels: u32) -> u64 {
    (0..levels)
        .map(|k| u64::from((w >> k).max(1)) * u64::from((h >> k).max(1)) * 4)
        .sum()
}

/// The mip-regeneration pipeline for textures of one format (one per format per pattern cache, created with its
/// first texture that has mips).
pub struct MipGen {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
}

/// Level k+1 from level k: an exact premultiplied 2×2 box filter on the stored bytes. Odd sizes drop the last
/// row / column (level k+1 is `max(1, size >> 1)`); reads clamp for the 1-texel levels.
const MIP_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn texel(at: vec2<i32>, last: vec2<i32>) -> vec4<u32> {
  return vec4<u32>(round(clamp(textureLoad(src, min(at, last), 0), vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0));
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let last = vec2<i32>(textureDimensions(src)) - vec2<i32>(1);
  let b = vec2<i32>(floor(pos.xy)) * 2;
  let sum = texel(b, last) + texel(b + vec2<i32>(1, 0), last) + texel(b + vec2<i32>(0, 1), last)
    + texel(b + vec2<i32>(1, 1), last);
  return vec4<f32>((sum + vec4<u32>(2u)) / vec4<u32>(4u)) / 255.0;
}
"#;

/// The box filter of a floating-point texture (an image held in an engine's working colour), for any extent.
/// Level k+1 is `max(1, size >> 1)` texels: each of them covers `size / that` texels of level k per axis, two,
/// or a little more where the size is odd, and is their mean weighted by how much of each it covers, so every
/// texel of level k counts once.
const MIP_FLOAT_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let size = vec2<f32>(textureDimensions(src));
  let scale = size / max(floor(size * 0.5), vec2<f32>(1.0));
  // This texel's footprint in level k, [lo, hi) per axis: at most four texels across.
  let lo = floor(pos.xy) * scale;
  let hi = lo + scale;
  let first = vec2<i32>(floor(lo));
  let last = vec2<i32>(size) - vec2<i32>(1);
  var sum = vec4<f32>(0.0);
  for (var j = 0; j < 4; j++) {
    let y = f32(first.y + j);
    let wy = max(min(hi.y, y + 1.0) - max(lo.y, y), 0.0);
    for (var i = 0; i < 4; i++) {
      let x = f32(first.x + i);
      let wx = max(min(hi.x, x + 1.0) - max(lo.x, x), 0.0);
      sum += textureLoad(src, min(first + vec2<i32>(i, j), last), 0) * (wx * wy);
    }
  }
  return sum / (scale.x * scale.y);
}
"#;

impl MipGen {
    /// `format`: the textures' format, `Rgba8Unorm` or `Rgba16Float`.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let float = format == wgpu::TextureFormat::Rgba16Float;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pattern texture mips"),
            source: wgpu::ShaderSource::Wgsl((if float { MIP_FLOAT_WGSL } else { MIP_WGSL }).into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pattern texture mips"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pattern texture mips"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
            label: Some("pattern texture mips"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        MipGen { pipeline, bgl }
    }
}

/// The GPU side of one surface texture's mip chain: a single-level view per level and a bind group reading each
/// level. Owned by the pattern cache that created the texture.
pub struct SurfaceMips {
    views: Vec<wgpu::TextureView>,
    /// `bind_groups[k]` reads level k (the source of level k + 1).
    bind_groups: Vec<wgpu::BindGroup>,
}

impl SurfaceMips {
    pub fn new(device: &wgpu::Device, gen: &MipGen, texture: &wgpu::Texture) -> Self {
        let views: Vec<wgpu::TextureView> = (0..texture.mip_level_count())
            .map(|k| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("pattern texture mip view"),
                    base_mip_level: k,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let bind_groups = views[..views.len().saturating_sub(1)]
            .iter()
            .map(|v| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pattern texture mip source"),
                    layout: &gen.bgl,
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(v) }],
                })
            })
            .collect();
        SurfaceMips { views, bind_groups }
    }

    pub fn regen(&self, device: &wgpu::Device, queue: &wgpu::Queue, gen: &MipGen, texture: &wgpu::Texture) {
        self.update(device, queue, gen, texture, None);
    }

    pub(crate) fn copy_frame(&self, device: &wgpu::Device, queue: &wgpu::Queue, gen: &MipGen,
        source: &wgpu::Texture, texture: &wgpu::Texture) {
        self.update(device, queue, gen, texture, Some(source));
    }

    fn draw_level(&self, encoder: &mut wgpu::CommandEncoder, gen: &MipGen, k: u32) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pattern texture mip level"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.views[k as usize],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&gen.pipeline);
        pass.set_bind_group(0, &self.bind_groups[k as usize - 1], &[]);
        pass.draw(0..3, 0..1);
    }

    fn update(&self, device: &wgpu::Device, queue: &wgpu::Queue, gen: &MipGen, texture: &wgpu::Texture,
        frame: Option<&wgpu::Texture>) {
        if frame.is_none() && self.views.len() < 2 { return; }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pattern texture update"),
        });
        if let Some(source) = frame {
            encoder.copy_texture_to_texture(source.as_image_copy(), texture.as_image_copy(), source.size());
        }
        for k in 1..self.views.len() as u32 {
            self.draw_level(&mut encoder, gen, k);
        }
        crate::scratch::submit(queue, [encoder.finish()]);
    }
}
