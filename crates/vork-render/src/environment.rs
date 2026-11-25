//! Install-time environment convolution.
use super::*;
mod cache;
mod evaluated;
mod brdf;
#[cfg(test)]
mod brdf_tests;
pub(crate) use evaluated::Validation;
// Preserve float32 bits without a float-sampling capability or precision loss.
#[cfg(test)]
const SOURCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;
const FILTER_SHADER: &str = include_str!("../shaders/environment.slang");
const REDUCE_SHADER: &str = include_str!("../shaders/environment_reduce.slang");

pub(crate) const REDUCE_LAYOUT: [wgpu::BindGroupLayoutEntry; 2] = [
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Uint,
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    },
    wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba32Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    },
];
pub(super) struct Environment {
    pub settings: environment_node::Settings,
    pub primary: Option<environment_program::Primary>,
    pub mean: [f32; 3],
    pub lighting_is_black: bool,
    animated: Option<Animated>,
    initial_filter_scale:f32,
    pub node: Option<usize>,
    pub source: wgpu::TextureView,
    pub distribution: Option<wgpu::Buffer>,
    pub distribution_size: [u32; 2],
    pub specular: wgpu::TextureView,
    pub diffuse: wgpu::TextureView,
    pub brdf: wgpu::TextureView,
    pub ltc: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub image_sampler: wgpu::Sampler,
}
struct Animated {
    program: environment_program::Program,
    evaluation:evaluated::Evaluation,
    integrals: Vec<Integral>,
}
struct Integral {
    bind: wgpu::BindGroup,
    params: wgpu::Buffer,
    groups: [u32; 2],
}
impl Integral {
    fn encode(&self, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline) {
        self.encode_timed(encoder, pipeline, None);
    }
    fn encode_timed(&self, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline, timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label: Some("environment integral"), timestamp_writes});
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.dispatch_workgroups(self.groups[0], self.groups[1], 1);
    }
}
impl Environment {
    pub(crate) fn has_media(&self)->bool {self.animated.as_ref().is_some_and(|a|a.program.has_media())}
    /// Whether the environment program reads `time`.
    pub(crate) fn uses_time(&self) -> bool {
        self.primary.as_ref().is_some_and(|p| p.uses_time) || self.animated.as_ref().is_some_and(|a| a.program.uses_time())
    }
    pub(crate) fn animated(&self) -> bool {
        self.animated.is_some()
    }
    /// `time` is the scene instant (scene-owned tracks and the program's `time` read it);
    /// `media` the instant its movies show: the light's node clock (rule 7), see `media_request`.
    pub(crate) fn sample_key(&self,time:impl Into<vork_time::Request>,media:impl Into<vork_time::Request>)->Result<Vec<u8>> {
        self.animated.as_ref().map_or_else(||Ok(Vec::new()), |a|a.program.sample_key(time,media))
    }
    /// The instant the environment's movies show at scene `request`: its node's local time,
    /// or `None` while the node is absent (rule 1).
    pub(crate) fn media_request(&self,animation:&crate::animation::Plan,request:&vork_time::Request)->Result<Option<vork_time::Request>> {
        let Some(node)=self.node else {return Ok(Some(request.clone()))};
        Ok(animation.node_instant(request,node)?.map(vork_time::Request::Exact))
    }
    /// Only values change: source radiance, proposal, moments and the retained
    /// filter passes. Every consumer keeps the same installed texture views.
    pub(crate) fn update(&self, r: &Renderer, time: impl Into<vork_time::Request>, media: impl Into<vork_time::Request>, changes: &mut values::LightingTracker) -> Result<()> {
        let Some(animated) = &self.animated else {
            return Ok(());
        };
        let (time,media)=(time.into(),media.into());
        let key=animated.program.sample_key(&time,&media)?;
        if !changes.changed(values::LightingKey::EnvironmentRadiance,&key) {return Ok(());}
        animated.program.evaluate_source(r,time,media)?;
        let mut encoder = r.device.create_command_encoder(&Default::default());
        animated.evaluation.encode(r,&mut encoder);
        for integral in &animated.integrals {
            animated.evaluation.publish_filter(&mut encoder,&integral.params);
            integral.encode(&mut encoder, Self::pipeline(r));
        }
        crate::command_finish::Segments::submit(&r.queue, encoder);
        Ok(())
    }
    pub(crate) fn initial_filter_scale(&self) -> f32 {self.initial_filter_scale}
    pub(crate) fn initial_mean(&self) -> [f32;3] {self.mean}
    #[cfg(test)]
    pub(crate) fn sampled_mean(&self,r:&Renderer)->Result<[f32;3]> {
        if let Some(animated)=&self.animated {Ok(animated.evaluation.read_metadata(r,r.device.create_command_encoder(&Default::default()))?.0[0].as_vec3().to_array())} else {Ok(self.mean)}
    }
    pub(crate) fn publish_metadata(&self,encoder:&mut wgpu::CommandEncoder,uniform:&wgpu::Buffer) {
        if let Some(animated)=&self.animated {animated.evaluation.publish_frame(encoder,uniform);}
    }
    pub(crate) fn diagnostics(&self,r:&Renderer)->Option<Validation> {self.animated.as_ref().map(|a|a.evaluation.diagnostics(r))}
    pub(crate) fn pipeline(r: &Renderer) -> &wgpu::ComputePipeline {
        let d = &r.device;
        r.pipelines.environment.get_or_init(|| {
            let module = crate::native::shader_module(d, &crate::native::ENVIRONMENT_FILTER_MAIN);
            // Native modules require an explicit layout: wgpu cannot reflect them.
            let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("environment integrals"),
                entries: &layout_entries(),
            });
            let pipeline_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("environment integrals"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("environment install"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("filter_main"),
                compilation_options: Default::default(),
                cache: r.cache.as_ref(),
            })
        })
    }
    /// Shared binding layouts still need typed placeholders when no requested
    /// output reads scene lighting. Do not bake environments or initialize LUTs.
    pub(crate) fn unused(r:&Renderer)->Self {
        let d=&r.device;
        let image=crate::texture(d,"unread environment",1,1,1,wgpu::TextureFormat::Rgba16Float,wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let source=environment_program::source_texture(r,[1,1]).create_view(&Default::default());
        let sampler=d.create_sampler(&Default::default());
        Self {settings:environment_node::Settings {source:None,material:None,medium:None},
            primary:None,mean:[0.;3],lighting_is_black:false,animated:None,initial_filter_scale:1.,node:None,
            source,distribution:None,distribution_size:[1,1],specular:image.clone(),diffuse:image.clone(),
            brdf:image.clone(),ltc:image,sampler:sampler.clone(),image_sampler:sampler}
    }
    #[cfg(test)]
    pub fn new(
        r: &Renderer,
        config: environment_node::Settings,
        scene: &vork_scene::v2::Scene,
        base: &Path,
    ) -> Result<Self> {
        Self::new_in(r, config, scene, base, &Default::default())
    }
    /// [`Self::new`] within a document open, reading media through the sources that open read.
    pub(crate) fn new_in(
        r: &Renderer,
        config: environment_node::Settings,
        scene: &vork_scene::v2::Scene,
        base: &Path,
        opened: &crate::footage::Opened,
    ) -> Result<Self> {
        let settings = config.has_radiance().then_some(&config);
        let d = &r.device;
        let mut program=None;
        let mut primary=None;
        let mut filtered=None;
        let mut filtered_path=None;
        let source=if let Some(e)=settings {
            let material=e.material.as_ref().context("environment radiance material")?;
            let (baked,installed)=environment_program::install_source_in(r,scene,material,base,opened)?;
            primary=baked.primary;program=installed;
            if !primary.as_ref().is_some_and(|p|p.spatial) {
                let path=cache::path(r,baked.identity);filtered=cache::Filtered::read(&path);filtered_path=Some(path);
            }
            baked.texture
        } else {
            let texture=environment_program::source_texture(r,[1,1]);
            r.queue.write_texture(texture.as_image_copy(),bytemuck::cast_slice(&[0f32;8]),wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(16),rows_per_image:Some(1)},texture.size());texture
        };
        let source_transport=!primary.as_ref().is_some_and(|p|p.spatial) && scene.render.settings.integrator.is_path();
        let proposal_needed=filtered.is_none() || program.is_some() || source_transport;
        let evaluation=(settings.is_some() && !primary.as_ref().is_some_and(|p|p.spatial)).then(||if proposal_needed {evaluated::Evaluation::new(r,&source)} else {evaluated::Evaluation::validation(r,&source)});
        // Release preparation only constructs shaders/resources. The no-op
        // device cannot evaluate radiance or produce readback metadata.
        let (mut sh,filter_scale)=if r.release_preparation {([glam::DVec3::ZERO;4],1.)} else {
            evaluation.as_ref().map(|e|e.update(r)).transpose()?.unwrap_or(([glam::DVec3::ZERO;4],1.))
        };
        if !proposal_needed {sh=filtered.as_ref().unwrap().sh.map(glam::DVec3::from_array);}
        let [dw,dh]=evaluation.as_ref().map_or([1,1],|e|e.size);
        let distribution=std::cell::OnceCell::new();
        if let Some(buffer)=evaluation.as_ref().and_then(|e|e.distribution.clone()) {let _=distribution.set(buffer);}
        // The shared filter layout needs a proposal even for first-device LUT
        // initialization. A warm raster install dispatches neither, so creates none.
        let proposal=||distribution.get_or_init(||init(d,"uniform environment proposal",bytemuck::cast_slice(&[[1f32,1./(4.*std::f32::consts::PI)]]),wgpu::BufferUsages::STORAGE));
        let baked_source=source;
        let source=baked_source.create_view(&Default::default());
        let texture = |label, width, height, mips, format, usage| {
            crate::gpu_memory::texture(
                &d,
                &wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: mips,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                },
            )
        };
        let sampler = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lighting integrals linear longitude repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let image_sampler=sampler.clone();
        let usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;
        let size = if settings.is_some() && !primary.as_ref().is_some_and(|p|p.spatial) {
            crate::constants::ENVIRONMENT_WIDTH
        } else {
            1
        };
        let levels = if settings.is_some() && !primary.as_ref().is_some_and(|p|p.spatial) {
            crate::constants::ENVIRONMENT_LEVELS
        } else {
            1
        };
        let specular = texture(
            "GGX environment mips",
            size,
            (size / 2).max(1),
            levels,
            wgpu::TextureFormat::Rgba16Float,
            usage,
        );
        let diffuse = texture(
            "First-order SH irradiance over pi",
            crate::constants::DIFFUSE_SIZE[0],
            crate::constants::DIFFUSE_SIZE[1],
            1,
            wgpu::TextureFormat::Rgba16Float,
            usage,
        );
        let pipeline = Self::pipeline(r);
        let integrate =
            |encoder: &mut wgpu::CommandEncoder, target: &wgpu::Texture, mode: u32, mip: u32, rows: std::ops::Range<u32>, timestamps: Option<wgpu::ComputePassTimestampWrites<'_>>| {
                let view = target.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                });
                let mut parameters = filter_parameters(mip, mode, filter_scale, sh, [dw, dh]);
                parameters[5][3] = rows.start as f32;
                let params = init(
                    d,
                    "environment integral parameters",
                    bytemuck::cast_slice(&parameters),
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                );
                let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("environment filter"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        entry(0, wgpu::BindingResource::TextureView(&source)),
                        entry(1, wgpu::BindingResource::TextureView(&view)),
                        entry(2, params.as_entire_binding()),
                        entry(3, proposal().as_entire_binding()),
                    ],
                });
                let integral = Integral {
                    bind,
                    params,
                    groups: [
                        (target.width() >> mip).max(1).div_ceil(8),
                        (rows.end - rows.start).div_ceil(8),
                    ],
                };
                if filtered.is_none() || mode == 2 {
                    integral.encode_timed(encoder, pipeline, timestamps);
                }
                integral
            };
        // The LUT is device-owned and scene-independent. Serialize fallible GPU
        // calibration and queue publication before another install can use it.
        let brdf = {
            let mut cached = r.brdf.lock().expect("BRDF initialization poisoned");
            if cached.is_none() {
                let lut = texture(
                    "BRDF moments and dielectric directional albedo",
                    crate::constants::BRDF_SIZE,
                    crate::constants::BRDF_HEIGHT,
                    1,
                    wgpu::TextureFormat::Rgba16Float,
                    usage,
                );
                brdf::initialize(r, |encoder, rows, timestamps| {
                    integrate(encoder, &lut, 2, 0, rows, timestamps);
                })?;
                *cached = Some(lut.create_view(&Default::default()));
            }
            cached.as_ref().expect("initialized BRDF lookup").clone()
        };
        let mut encoder = d.create_command_encoder(&Default::default());
        let mut integrals = Vec::new();
        if settings.is_some() && !primary.as_ref().is_some_and(|p|p.spatial) && (filtered.is_none() || program.is_some()) {
            for (target, mode, mip) in (0..levels)
                .map(|m| (&specular, 0u32, m))
                .chain([(&diffuse, 1, 0)])
            {
                integrals.push(integrate(&mut encoder, target, mode, mip, 0..(target.height() >> mip).max(1), None));
            }
        }
        crate::command_finish::Segments::submit(&r.queue, encoder);
        if let Some(cached) = &filtered {
            cached.upload(r, &specular, &diffuse);
        }
        if filtered.is_none() {if let Some(path)=filtered_path {
            cache::Filtered::capture(r,&specular,&diffuse,sh,path);
        }}
        // Immutable LTC fits are uploaded once per device, shared by every scene
        // and frame slot. RGBA16F supports filtering on both Vulkan and Metal.
        let ltc = r.ltc.get_or_init(|| {
            let bytes = include_bytes!("ltc/amplitude.rgba32f");
            {
                let values: Vec<u16> = bytes
                    .chunks_exact(4)
                    .map(|v| {
                        half::f16::from_f32(f32::from_le_bytes(v.try_into().unwrap())).to_bits()
                    })
                    .collect();
                let size = crate::constants::LTC_SIZE;
                let texture = crate::gpu_memory::texture(
                    &d,
                    &wgpu::TextureDescriptor {
                        label: Some("LTC GGX and sheen fit"),
                        size: wgpu::Extent3d {
                            width: size,
                            height: size,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba16Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                );
                r.queue.write_texture(
                    texture.as_image_copy(),
                    bytemuck::cast_slice(&values),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(size * 8),
                        rows_per_image: Some(size),
                    },
                    wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: 1,
                    },
                );
                texture.create_view(&Default::default())
            }
        });
        // Camera rays read the program's compact inputs directly. Static raster
        // lighting is already in its convolution; path transport retains only
        // the lighting layer. Animated evaluations keep their shared bake.
        let retained_source=if settings.is_none() || primary.as_ref().is_some_and(|p|p.spatial) {
            source.clone()
        } else if program.is_some() {
            baked_source.create_view(&wgpu::TextureViewDescriptor {dimension:Some(wgpu::TextureViewDimension::D2Array),base_array_layer:1,array_layer_count:Some(1),..Default::default()})
        } else if source_transport {
            let uniform=evaluation.as_ref().and_then(evaluated::Evaluation::uniform);
            let lighting=environment_program::lighting_texture(r,if uniform.is_some() {[1,1]} else {[baked_source.width(),baked_source.height()]});
            if let Some(rgb)=uniform {
                r.queue.write_texture(lighting.as_image_copy(),bytemuck::cast_slice(&[rgb[0],rgb[1],rgb[2],1f32.to_bits()]),wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(16),rows_per_image:Some(1)},lighting.size());
            } else {
                let mut encoder=d.create_command_encoder(&Default::default());
                encoder.copy_texture_to_texture(wgpu::TexelCopyTextureInfo {origin:wgpu::Origin3d {x:0,y:0,z:1},..baked_source.as_image_copy()},lighting.as_image_copy(),lighting.size());
                crate::command_finish::Segments::submit(&r.queue,encoder);
            }
            lighting.create_view(&wgpu::TextureViewDescriptor {dimension:Some(wgpu::TextureViewDimension::D2Array),..Default::default()})
        } else {
            environment_program::lighting_texture(r,[1,1]).create_view(&wgpu::TextureViewDescriptor {dimension:Some(wgpu::TextureViewDimension::D2Array),..Default::default()})
        };
        // Exact retained lighting bits, not a mean/threshold: underflow in the
        // moments must never classify a dim source as absent. Animated and
        // spatial programs retain their ordinary sampling path.
        let lighting_is_black=program.is_none() && evaluation.as_ref()
            .and_then(evaluated::Evaluation::uniform)
            .is_some_and(|rgb|rgb.into_iter().all(|bits|f32::from_bits(bits)==0.));
        Ok(Self {
            lighting_is_black,
            primary,
            animated: program.map(|program| Animated {
                program,
                evaluation:evaluation.expect("animated environment evaluated state"),
                integrals,
            }),
            initial_filter_scale:filter_scale,
            settings: config,
            mean: sh[0].as_vec3().to_array(),
            node: None,
            source:retained_source,
            distribution: source_transport.then(||proposal().clone()),
            distribution_size: [dw,dh],
            specular: specular.create_view(&Default::default()),
            diffuse: diffuse.create_view(&Default::default()),
            brdf: brdf.clone(),
            ltc: ltc.clone(),
            sampler,
            image_sampler,
        })
    }
}

/// Cache the GPU-filtered importance image, not a full-resolution lookup table.
/// Rotation does not affect this unmodified radiance proposal.
/// Bindings shared by pipeline creation and native shader reflection checks.
pub(crate) fn layout_entries() -> [wgpu::BindGroupLayoutEntry; 4] {
    let binding = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    };
    [
        binding(
            0,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
        ),
        binding(
            1,
            wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
        ),
        binding(
            2,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(96),
            },
        ),
        binding(
            3,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(8),
            },
        ),
    ]
}

// One constructor owns the 96-byte shader ABI for installation and animation.
// Upload the reciprocal: shaders never divide by a runtime buffer field.
fn filter_parameters(mip: u32, mode: u32, scale: f32, sh: [glam::DVec3; 4], size: [u32; 2]) -> [[f32; 4]; 6] {
    assert!(scale.is_finite() && scale >= 1.);
    assert!(size.into_iter().all(|n| n > 0));
    let inverse = scale.recip();
    assert!(inverse.is_finite() && inverse > 0.);
    [
        [mip as f32 / 5., mode as f32, inverse, 0.],
        (sh[0] / scale as f64).as_vec3().extend(0.).to_array(),
        (sh[1] / scale as f64).as_vec3().extend(0.).to_array(),
        (sh[2] / scale as f64).as_vec3().extend(0.).to_array(),
        (sh[3] / scale as f64).as_vec3().extend(0.).to_array(),
        [size[0] as f32, size[1] as f32, 0., 0.],
    ]
}

#[cfg(test)]
fn radiance_scale(image: &image::Rgb32FImage) -> f32 {
    (image.as_raw().iter().copied().fold(0f32,f32::max) / (crate::constants::HALF_FLOAT_MAX*0.5)).max(1.)
}
// Full source RGB remains available to the camera. Alpha records the selected
// core's chromatic residual fraction, applied per texel before interpolation.
#[cfg(test)]
fn source_rgba(camera:&image::Rgb32FImage,lighting:&image::Rgb32FImage)->Vec<f32> {
    camera.pixels().chain(lighting.pixels()).flat_map(|p|[p[0],p[1],p[2],1.]).collect()
}
/// First-order radiance moments; source splitting is an authoring operation.
#[cfg(test)]
fn analyze(image: &image::Rgb32FImage) -> [glam::DVec3;4] {
    let mut sh = [glam::DVec3::ZERO; 4];
    // Solid-angle quadrature over the HDR: isolate the bright lobe's
    // radiometric energy and RGB-magnitude-weighted direction. Uses the same
    // equirectangular integration math as the CDF below.
    let width = image.width();
    let height = image.height();

    let longitude: Vec<_> = (0..width).map(|x| {
        let phi = std::f64::consts::TAU * ((x as f64 + 0.5) / width as f64 - 0.5);
        (phi.cos(), phi.sin())
    }).collect();
    for y in 0..height {
        let omega = std::f64::consts::TAU / width as f64
            * ((std::f64::consts::PI * y as f64 / height as f64).cos()
                - (std::f64::consts::PI * (y + 1) as f64 / height as f64).cos());
        let theta = std::f64::consts::PI * (y as f64 + 0.5) / height as f64;
        let (sin_theta, cos_theta) = (theta.sin(), theta.cos());
        for x in 0..width {
            let pixel = image.get_pixel(x,y);
            let radiance = glam::DVec3::new(pixel[0] as f64, pixel[1] as f64, pixel[2] as f64);
            let (cos_phi,sin_phi) = longitude[x as usize];
            let direction = glam::DVec3::new(sin_theta*cos_phi,cos_theta,sin_theta*sin_phi);
            {
                let residual = radiance;
                // Lookup applies environment rotation; retain local coefficients.
                let local_direction = direction;
                sh[0] += residual * (omega / (4. * std::f64::consts::PI));
                for axis in 0..3 {
                    sh[axis + 1] +=
                        residual * local_direction[axis] * (omega / (2. * std::f64::consts::PI));
                }
            }
        }
    }
    // Dering the first-order reconstruction with one shared factor,
    // preserving hue and keeping every channel nonnegative for all n.
    let mut factor = 1f64;
    for channel in 0..3 {
        let amplitude = glam::DVec3::new(sh[1][channel], sh[2][channel], sh[3][channel]).length();
        factor = factor.min(
            sh[0][channel]
                / (amplitude * (1. + 0.05 / (crate::constants::SH_FIRST_ORDER * (2. / 3.))))
                    .max(crate::constants::POSITIVE_EPSILON_F64),
        );
    }
    for coefficient in &mut sh[1..] {
        *coefficient *= factor;
    }
    sh
}

#[cfg(test)]
fn distribution(dw: u32, dh: u32, proposal: &[f32]) -> Vec<[f32; 2]> {
    // Radiance-weighted solid-angle distribution, built at install time.
    let mut distribution = Vec::<[f64; 2]>::with_capacity((dw * dh) as usize);
    let mut total = 0f64;
    let columns: Vec<_> = (0..dw).map(|x| [-1i32,0,1].map(|dx|
        (x as i32+dx).rem_euclid(dw as i32) as usize)).collect();
    for y in 0..dh {
        let solid_angle = std::f64::consts::TAU / dw as f64
            * ((std::f64::consts::PI * y as f64 / dh as f64).cos()
                - (std::f64::consts::PI * (y + 1) as f64 / dh as f64).cos());
        let rows = [-1i32,0,1].map(|dy|
            (y as i32+dy).clamp(0,dh as i32-1) as usize*dw as usize);
        for x in 0..dw {
            // Keep each RGB sum's order, but share address/alpha preparation
            // between channels and hoist coordinate wrapping out of the stencil.
            let mut rgb = [0f64;3];
            for (dy,row) in rows.into_iter().enumerate() {
                for (dx,column) in columns[x as usize].into_iter().enumerate() {
                    let at=(row+column)*4;
                    let alpha=proposal[at+3] as f64;
                    let wx=if dx==1 {2.} else {1.};
                    let wy=if dy==1 {2.} else {1.};
                    for c in 0..3 { rgb[c]+=proposal[at+c] as f64*alpha*wx*wy; }
                }
            }
            let rgb=rgb.map(|v|v/16.);
            let luminance = crate::constants::LUMINANCE_REC709[0] as f64 * rgb[0]
                + crate::constants::LUMINANCE_REC709[1] as f64 * rgb[1]
                + crate::constants::LUMINANCE_REC709[2] as f64 * rgb[2];
            total += luminance * solid_angle;
            distribution.push([total, luminance]);
        }
    }
    if total > 0. {
        let mut result:Vec<[f32;2]>=distribution.into_iter().map(|v|[(v[0]/total) as f32,(v[1]/total) as f32]).collect();
        result.last_mut().unwrap()[0] = 1.;
        return result;
    } else {
        // Black environments use a uniform sphere proposal.
        let mut result=vec![[0f32;2];distribution.len()];
        for (i, value) in result.iter_mut().enumerate() {
            let y = i as u32 / dw;
            let row = (1. - (std::f32::consts::PI * y as f32 / dh as f32).cos()) / 2.;
            let next = (1. - (std::f32::consts::PI * (y + 1) as f32 / dh as f32).cos()) / 2.;
            *value = [
                row + (next - row) * ((i as u32 % dw + 1) as f32 / dw as f32),
                1. / (4. * std::f32::consts::PI),
            ];
        }
        result
    }
}

#[cfg(test)]
#[path = "environment_tests.rs"]
mod tests;

pub(crate) fn reduce_pipeline(r: &Renderer) -> &wgpu::ComputePipeline {
 let d = &r.device;
 r.pipelines.environment_reduce.get_or_init(|| {
        let module = native::shader_module(d, &native::ENVIRONMENT_REDUCE_REDUCE_ENVIRONMENT);
        let group = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HDR solid-angle reduction"),
            entries: &REDUCE_LAYOUT,
        });
        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HDR solid-angle reduction"),
            bind_group_layouts: &[Some(&group)],
            immediate_size: 0,
        });
        r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HDR solid-angle reduction"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(native::ENVIRONMENT_REDUCE_REDUCE_ENVIRONMENT.entry),
            compilation_options: Default::default(),
            cache: r.cache.as_ref(),
        })
    })
}
