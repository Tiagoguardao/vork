//! Render-level views over one installed scene and one evaluated frame.
use crate::*;
use vork_scene::v2;

/// Raster applies view membership while recording draws. Path camera rays still
/// use the whole-scene TLAS, so accepting a subtree can render excluded objects
/// (including a monitor reading its own initially empty capture).
pub(crate) fn require_supported_roots(input: &v2::Scene, order: &[String]) -> Result<()> {
    if !input.render.settings.integrator.is_path() { return Ok(()); }
    ensure!(input.render.root.is_none(),
        "render.root: path transport does not support subtree renders; use --profile std/final with depth of field and path-only effects disabled, or put the selected content in a separate scene and remove root");
    for name in order {
        ensure!(input.images[name].render.as_ref().unwrap().root.is_none(),
            "images[{name:?}].render.root: path transport does not support subtree renders; use --profile std/final with depth of field and path-only effects disabled, or render the capture from a separate scene without root");
    }
    Ok(())
}

#[cfg(test)]
mod subtree_tests {
    use super::*;

    #[test]
    fn path_subtrees_refuse_without_a_rendering_device() -> Result<()> {
        let mut value = serde_json::json!({"version":2,
            "render":{"settings":{"integrator":"path"}},
            "images":{"feed":{"render":{"root":"content"}}},
            "nodes":[]});
        let scene = |value| vork_scene::v2::from_test_value::<v2::Scene>(value);
        let live = vec!["feed".to_owned()];
        let input = scene(value.clone())?;
        // Unused captures must not restrict the main view.
        require_supported_roots(&input, &[])?;
        let error = require_supported_roots(&input, &live).unwrap_err().to_string();
        assert!(error.contains("images[\"feed\"].render.root"), "{error}");
        assert!(error.contains("path transport does not support subtree renders"), "{error}");
        value["render"]["root"] = "monitor".into();
        let error = require_supported_roots(&scene(value.clone())?, &[]).unwrap_err().to_string();
        assert_eq!(error,
            "render.root: path transport does not support subtree renders; use --profile std/final with depth of field and path-only effects disabled, or put the selected content in a separate scene and remove root");
        vork_scene::v2::replace_integrator(&mut value, serde_json::json!({"integrator":"raster"}));
        require_supported_roots(&scene(value.clone())?, &live)?;
        vork_scene::v2::replace_integrator(&mut value, serde_json::json!({"integrator":"path"}));
        value["render"]["root"] = serde_json::Value::Null;
        value["images"]["feed"]["render"]["root"] = serde_json::Value::Null;
        require_supported_roots(&scene(value)?, &live)?;
        Ok(())
    }
}

pub(crate) const PUBLICATION_LAYOUT: [wgpu::BindGroupLayoutEntry; 3] = [
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
    wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba16Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    },
    wgpu::BindGroupLayoutEntry {
        binding: 2,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
];

pub(crate) struct InstalledImageRender {
    pub name: String,
    pub camera: usize,
    pub settings: v2::Render,
    pub exposure_samples: u32,
    pub visibility_samples: u32,
    pub draws: Vec<bool>,
    pub objects: Vec<bool>,
    pub lights: Vec<bool>,
    pub pipeline: wgpu::ComputePipeline,
    pub output: wgpu::TextureView,
    texture: wgpu::Texture,
    pub level_zero: wgpu::TextureView,
    mips: materials::MipChain,
    time_dependent: bool,
}

impl InstalledImageRender {
    pub(crate) fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        for mip in 0..self.texture.mip_level_count() {
            let view = self.texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: mip, mip_level_count: Some(1), ..Default::default()
            });
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("reset render image"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None, depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
        }
    }

    fn publish(&self,encoder:&mut wgpu::CommandEncoder,bind:&wgpu::BindGroup,[width,height]:[u32;2],end_query:Option<(&wgpu::QuerySet,u32)>) {
        {
            let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:Some("publish render image"),
                timestamp_writes: end_query.filter(|_| self.mips.is_empty()).map(|(query_set,index)|wgpu::ComputePassTimestampWrites {query_set,beginning_of_pass_write_index:None,end_of_pass_write_index:Some(index)}),
            });
            pass.set_pipeline(&self.pipeline);pass.set_bind_group(0,bind,&[]);
            let group=constants::IMAGE_WORKGROUP_SIZE;
            pass.dispatch_workgroups(width.div_ceil(group),height.div_ceil(group),1);
        }
        self.mips.encode(encoder,end_query.map(|(query_set,index)| wgpu::RenderPassTimestampWrites {query_set,beginning_of_pass_write_index:None,end_of_pass_write_index:Some(index)}));
    }
}

pub(crate) fn plan(
    input: &v2::Scene,
    s: &InstalledScene,
    r: &Renderer,
    order: &[String],
) -> Result<Vec<InstalledImageRender>> {
    if order.is_empty() {
        return Ok(Vec::new());
    }
    let pipeline = publication_pipeline(r);
    let (mip_pipelines,mip_sampler)=materials::mip_resources(r);
    order
        .iter()
        .zip(&s.image_settings)
        .map(|(name, installed_settings)| {
            let source = input.images[name].render.as_ref().unwrap();
            let mut settings = input.images[name].resolved_render(&s.render).expect("a rendered image");
            settings.settings = installed_settings.clone();
            ensure!(
                settings
                    .resolution
                    .iter()
                    .all(|n| *n <= r.device.limits().max_texture_dimension_2d),
                crate::gpu_memory::device_limit(format_args!("render image {name}"), format_args!("{}x{} pixels",settings.resolution[0],settings.resolution[1]), format_args!("max_texture_dimension_2d={} pixels per axis",r.device.limits().max_texture_dimension_2d), "reduce image.render.resolution")
            );
            let (camera, exposure_samples, visibility_samples) = camera_sampling(&s.nodes, &settings, s.camera)?;
            let Subset {
                draws,
                objects,
                lights,
            } = subset(s, source.root.as_deref())?;
            // Rays may reach surfaces outside the capture subtree. Only absent consumers prove
            // that an excluded draw's clock cannot affect this capture.
            let integrator=&settings.settings.integrator;
            let ray_surfaces=s.plan.rays && (integrator.is_path()
                || matches!(integrator.reflections(),Some(vork_scene::Reflections::Auto|vork_scene::Reflections::Ray))
                || integrator.indirect().is_some_and(|gi|gi.method==vork_scene::FastGiMethod::Ray)
                || integrator.probe_volume().is_some()
                || integrator.raster().is_some_and(|r|matches!(r.refraction,vork_scene::Refraction::Auto|vork_scene::Refraction::Ray))
                || s.plan.effects.fibre);
            let shadows=lights.iter().any(|&included|included);
            let surface_time = s.draws.iter().enumerate().any(|(index,draw)| {
                let consumed=draws[index] || ray_surfaces || (shadows && s.plan.draws[index].casts_shadows);
                let id=s.materials[draw.material].program;
                consumed && id!=0 && s.surface_dependencies[id as usize-1].time
            });
            let time_dependent = surface_time || s.has_media
                || s.camera_programs.iter().any(|p|p.node==camera && p.uses_time)
                || s.environment.primary.as_ref().is_some_and(|p|p.uses_time)
                || s.scene.propagation.is_some() || settings.settings.integrator.is_path()
                || !settings.compositor.is_empty()
               ;
            let [width,height]=settings.resolution;
            let texture=gpu_memory::texture(&r.device,&wgpu::TextureDescriptor {
                label:Some("render image output"),size:extent(width,height),
                mip_level_count:32-width.max(height).leading_zeros(),sample_count:1,
                dimension:wgpu::TextureDimension::D2,format:wgpu::TextureFormat::Rgba16Float,
                usage:wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats:&[],
            });
            let mips=materials::MipChain::new(&r.device,&texture,None,&mip_pipelines[2],mip_sampler);
            let level_zero=texture.create_view(&wgpu::TextureViewDescriptor {mip_level_count:Some(1),..Default::default()});
            Ok(InstalledImageRender {
                name: name.clone(),
                time_dependent,
                camera,
                settings, exposure_samples, visibility_samples,
                draws,
                objects,
                lights,
                pipeline: pipeline.clone(),
                output: texture.create_view(&Default::default()),
                level_zero,mips,texture,
            })
        })
        .collect()
}

pub(crate) fn camera_sampling(nodes: &[EvalNode], settings: &v2::Render, default: usize) -> Result<(usize, u32, u32)> {
    let camera = settings.camera.as_deref().map(|name| named_node(nodes, name)).transpose()?.unwrap_or(default);
    let (exposure, visibility) = crate::plan::sampling_counts(settings, false);
    Ok((camera, exposure, visibility))
}

fn named_node(nodes: &[EvalNode], name: &str) -> Result<usize> {
    let mut matches = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.node.name == name);
    let first = matches
        .next()
        .with_context(|| format!("render node {name} is not installed"))?
        .0;
    ensure!(
        matches.next().is_none(),
        "render node {name} is ambiguous after asset expansion"
    );
    Ok(first)
}

/// Shared subtree membership for view planning and resource admission.
pub(crate) fn included_nodes(nodes: &[EvalNode], root: Option<&str>) -> Result<Vec<bool>> {
    let root = root.map(|name| named_node(nodes, name)).transpose()?;
    let mut included = vec![root.is_none(); nodes.len()];
    for (id, node) in nodes.iter().enumerate() {
        included[id] |= Some(id) == root || node.parent.is_some_and(|p| included[p]);
    }
    Ok(included)
}

pub(crate) struct Subset {
    pub draws: Vec<bool>,
    pub objects: Vec<bool>,
    pub lights: Vec<bool>,
}
pub(crate) fn subset(s: &InstalledScene, root: Option<&str>) -> Result<Subset> {
    let included = included_nodes(&s.nodes, root)?;
    let objects: Vec<_> = s.roots.iter().map(|root| included[root.node]).collect();
    let draws = s
        .draws
        .iter()
        .map(|draw| {
            draw.instances
                .clone()
                .any(|object| objects[object as usize])
        })
        .collect();
    let lights = s
        .light_plan
        .nodes
        .iter()
        .enumerate()
        .map(|(_, node)| included[*node])
        .collect();
    Ok(Subset {
        draws,
        objects,
        lights,
    })
}

pub(crate) struct CaptureView {
    pub(crate) view: crate::view::ViewState,
    uniforms: wgpu::Buffer,
    sampled_uniforms: Vec<Uniform>,
    camera_selections: Option<wgpu::Buffer>,
}
impl CaptureView {
    pub(crate) fn new(
        s: &InstalledScene,
        r: &Renderer,
        shared: &mut FrameSlot,
        desc: &crate::views::ViewDesc,
        image: &InstalledImageRender,
    ) -> Result<Self> {
        let subset = Subset {
            draws: image.draws.clone(),
            objects: image.objects.clone(),
            lights: image.lights.clone(),
        };
        let view = crate::view::ViewState::new(
            s,
            r,
            &image.settings,
            ReadbackPrecision::Float,
            desc,
            &shared.queries,
            crate::view::SharedBindings {
                points: &shared.point_publication.buffer,
                rays: shared.rays.as_ref(),
                compute: shared.compute_commands.as_ref(),
                objects: &shared.objects,
                deforms: &shared.deforms,
                materials: &shared.materials,
                shadow_bind: &shared.shadow_bind,
                output: Some(&shared.output),
            },
            Some(&subset),
        )?;
        let uniforms = crate::gpu_memory::buffer(&r.device, &wgpu::BufferDescriptor {
            label: Some("capture sample uniforms"),
            size: view.sample_count as u64 * std::mem::size_of::<Uniform>() as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_selections = view.camera_cull.sample_values(&r.device, view.sample_count as u32);
        Ok(Self {
            sampled_uniforms: Vec::with_capacity(view.sample_count),
            view,
            uniforms,
            camera_selections,
        })
    }

    pub(crate) fn prepare(
        &mut self,
        s: &InstalledScene,
        frame: &mut FrameSlot,
        image: &InstalledImageRender,
        base: Uniform,
        changes: &mut values::ChangeTracker,
        id: usize,
        dependency_changed: bool,
        lighting_revisions: [u64; 2],
    ) -> Result<bool> {
        self.sampled_uniforms.clear();
        for sample in 0..self.view.sample_count as u32 {
            self.sampled_uniforms.push(values::capture_uniform(s, frame, image, base, sample)?);
        }
        // GPU-authored scene data has no CPU input identity.
        // Their consumers replay even at the same time; compare observable inputs
        // only for captures whose complete state can be retained here.
        if s.compute.is_none() && !s.has_media {
            let mut inputs = Vec::new();
            // The shared stateless volume has its own exact input tracker. Retain
            // its revision, including updates made while this capture did not run.
            inputs.extend_from_slice(bytemuck::cast_slice(&lighting_revisions));
            let media=s.environment.media_request(&s.animation,&frame.sample_request)?.unwrap_or_else(||frame.sample_request.clone());
            inputs.extend_from_slice(&s.environment.sample_key(&frame.sample_request,&media)?);
            // Captures share the current parent sample's pose (independent capture
            // shutters are rejected). Include every immutable footage selection:
            // secondary hits, shadows and upstream captures can observe it too.
            // Use selected source frames, not publication revisions, so a seek or
            // another slot's overwrite cannot force replay of identical inputs.
            for source in s.footage.values().filter(|source|source.per_sample) {
                let request=frame.sample_request.exact()?;
                let clock=if source.canvas.is_some() {
                    source.canvas_instant(s,&frame.animation_values,&request)?
                } else {
                    let base=source.time.as_ref().map(|t|vork_time::exact::Germ::point(t.exact())).unwrap_or_else(||request.clone());
                    Some(s.animation.image_time_exact(&frame.animation_values,&request,&source.name,base)?)
                };
                inputs.extend(source.sample_key(clock.as_ref(),s.render.fps)?);
            }
            // Native raster shading has no time input. Programs, media, compositors
            // and animated path seeds do; keep the actual uniforms unchanged.
            for uniform in &self.sampled_uniforms {
                let mut input = *uniform;
                if !image.time_dependent {
                    input.scene[..3].fill(0.);
                    input.composite_shutter.fill(0.);
                    // Display programs receive nominal frame time separately
                    // from scene time. It is not an input to a static capture.
                    input.post[1] = 0.;
                }
                inputs.extend_from_slice(bytemuck::bytes_of(&input));
            }
            // Ray hits can see casters outside the image
            // subtree. These are evaluated values, not time/animation heuristics.
            for &object in &s.dynamic_objects {
                inputs.extend_from_slice(bytemuck::bytes_of(&frame.object_values[object].gpu()));
            }
            inputs.extend_from_slice(bytemuck::cast_slice(&frame.evaluated_lights));
            if let Some(materials) = &frame.materials {
                frame.changes.append_buffer_inputs(&materials.records, &mut inputs);
            }
            frame.geometry_chain.append_inputs(&frame.changes,&mut inputs);
            let changed = changes.changed(values::ChangeKey::Capture(id), &inputs);
            if !(changed || dependency_changed) {
                return Ok(false);
            }
        }
        if let Some(values) = &self.camera_selections {
            for (sample, uniform) in self.sampled_uniforms.iter().enumerate() {
                let projection = if uniform.sampling[3] > 1 {
                    let radius=crate::sampling::lens_selection_radius(uniform.post[3], uniform.lens[1], uniform.params[0], uniform.params[1]);
                    crate::sampling::selection_projection(uniform.vp, self.view.internal, radius)
                } else { uniform.vp };
                self.view.camera_cull.prepare_sample(projection, &frame.object_values, &frame.geometry_bounds,
                    &mut frame.changes, values, sample as u32);
            }
        }
        self.view.post.compositor_inputs.upload(s, base.scene[0], &mut frame.changes)?;
        frame.changes.upload(
            &self.uniforms,
            0,
            bytemuck::cast_slice(&self.sampled_uniforms),
        );
        let lights: Vec<_> = frame
            .evaluated_lights
            .iter()
            .zip(&image.lights)
            .map(|(light, included)| {
                let mut light = *light;
                if !included {
                    light.radiance[..3].fill(0.);
                }
                light
            })
            .collect();
        self.view.lighting.upload_lights(&mut frame.changes, &lights)?;
        let view = self.sampled_uniforms[0].view;
        if !self.view.transparent_order.is_empty() {
            let indirect = self.view.transparent_indirect.is_some();
            let sorted = passes::transparent::sort(
                s,
                &frame.object_values,
                &mut self.view.transparent_order,
                view,
                indirect,
            );
            if let Some((objects, args, runs)) = sorted {
                for (index, value) in objects {
                    frame.changes.upload(
                        self.view.transparent_instances.as_ref().unwrap(),
                        (index * std::mem::size_of::<u32>()) as u64,
                        bytemuck::bytes_of(&value),
                    );
                }
                frame.changes.upload(
                    self.view.transparent_indirect.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&args),
                );
                self.view.transparent_runs = runs;
            }
        }
        Ok(true)
    }

    pub(crate) fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &Renderer,
        environment:&crate::environment::Environment,
        queries: &wgpu::QuerySet,
        image: &InstalledImageRender,
        last: bool,
        denoise_timing: &mut denoise::Timing,
        segments: &mut crate::command_finish::Segments,
        changes: &mut crate::values::ChangeTracker,
    ) -> Result<()> {
        let samples = self.view.sample_count as u32;
        let frame = &mut self.view;
        for sample in 0..samples {
            frame.sample_index = sample as usize;
            encoder.copy_buffer_to_buffer(
                &self.uniforms,
                u64::from(sample) * std::mem::size_of::<Uniform>() as u64,
                &frame.uniform,
                0,
                std::mem::size_of::<Uniform>() as u64,
            );
            environment.publish_metadata(encoder,&frame.uniform);
            if let Some(values) = &self.camera_selections {
                frame.camera_cull.copy_sample(encoder, changes, values, sample);
            }
            let bits = crate::graph::RunBits {
                color: frame.color_enabled, final_sample: sample + 1 == samples,
                probes: false, fibre: false, images: false, pixels: false,
            };
            let view = &*frame;
            for step in &view.steps {
                if bits.runs(step.run) {
                    if step.pass == crate::graph::Pass::Denoise {
                        if let Some(stage) = &view.post.reconstruction { stage.encode(encoder,None); }
                        if let Some(stage) = &view.post.denoiser {
                            stage.run(renderer, encoder, view.sample_count, 1., denoise_timing, segments)?;
                        }
                    } else {
                        crate::graph::encode_view(encoder, view, step, renderer, segments)?;
                    }
                }
            }
        }
        image.publish(encoder,frame.publication.as_ref().expect("image publication installed"),
            [frame.width,frame.height],last.then_some((queries,timing::query("images")+1)));
        Ok(())
    }
}

#[cfg(all(test, vork_private_tests))]
include!("render_images_reuse_tests_private.rs");

pub(crate) fn publication_pipeline(r: &Renderer) -> wgpu::ComputePipeline {
    let shader = native::shader_module(&r.device, &native::RENDER_IMAGES_PUBLISH);
    let bind_layout = r.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("render image publication"),
        entries: &PUBLICATION_LAYOUT,
    });
    let layout = r.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render image publication"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });
    r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("render image publication"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some(native::RENDER_IMAGES_PUBLISH.entry),
        compilation_options: Default::default(),
        cache: r.cache.as_ref(),
    })
}

#[cfg(test)]
mod previs_views {
    use super::*;

    fn scene(main: &str, captures: &[&str], output: Option<&str>) -> Result<v2::Scene> {
        let mut value = serde_json::json!({"version":2,
            "materials":{
                "source":{"inputs":{"base_weight":0,"specular_weight":0,"emission_luminance":683}},
                "receiver":{"inputs":{"base_color":[0.8,0.8,0.8]}},
                "screen":{"inputs":{"base_weight":0,"specular_weight":0,"emission_luminance":683,"emission_color":[0.2,0.5,0.8]}}},
            "nodes":[{"name":"camera","type":"camera","translation":[0,0,5]},
                {"name":"screen_camera","type":"camera","translation":[20,0,5]},
                {"name":"world","children":[
                    {"name":"source","type":"mesh","visible_camera":false,"translation":[-1,1,2],"materials":["source"],"geometry":{"std":"rounded_box","size":[2,2,0.1],"radius":0}},
                    {"name":"receiver","type":"mesh","materials":["receiver"],"geometry":{"std":"rounded_box","size":[4,4,0.1],"radius":0}}]},
                {"name":"ui","children":[{"name":"screen","type":"mesh","translation":[20,0,0],"materials":["screen"],"geometry":{"std":"rounded_box","size":[4,4,0.1],"radius":0}}]},
                {"name":"empty"}],
            "render":{"root":main,"camera":if main=="world" {"camera"} else {"screen_camera"},"resolution":[32,32],"background":[0,0,0]}});
        if output == Some("world") {
            value["programs"]["receiver"] = serde_json::json!({"source": "
                struct Inputs {};
                Closure surface(ShadingContext ctx, Inputs inputs) {
                    Closure s = {};
                    s.base_weight = 1.;
                    s.base_color = float3(.8);
                    s.geometry_opacity = 1.;
                    s.geometry_normal = ctx.NORMAL;
                    return s;
                }"});
            value["materials"]["receiver"] = serde_json::json!({"program":"receiver"});
        }
        if !captures.is_empty() {
            // Tiny, nonzero contributions retain the producers and main beauty
            // without perceptibly changing the view under comparison.
            let mut inputs=serde_json::json!({"beauty":{"channel":"color"}});
            let mut fields="Image beauty;".to_string();
            let mut expression=format!("compositor_sample(i.beauty,c.uv)*{}",if output.is_some() {"1e-20"} else {"1"});
            for name in captures {
                value["images"][name]=serde_json::json!({"render":{"root":name,"camera":if *name=="world" {"camera"} else {"screen_camera"},"resolution":[32,32],"background":[0,0,0]}});
                inputs[name]=serde_json::json!({"image":name});
                fields.push_str(&format!("Image {name};"));
                expression.push_str(&format!("+compositor_sample(i.{name},c.uv)*{}",if output==Some(*name) {"1"} else {"1e-20"}));
            }
            value["composition"]=serde_json::json!({"space":"3d","compositor":[{"material":{
                "program":{"source":format!("struct Inputs {{{fields}}};float4 composite(CompositeContext c,Inputs i){{return {expression};}}")},"inputs":inputs}}]});
        }
        let mut scene=crate::test_scene(value,Path::new("."))?;
        scene.apply_settings(vork_scene::Profile::load("std/previs")?.resolve(&Default::default())?,"std/previs")?;
        Ok(scene)
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang"]
    fn emitter_providers_and_passes_are_view_local() -> Result<()> {
        let r=Renderer::new()?;
        for main in ["world","ui"] {
            let mut isolated=r.install(scene(main,&[],None)?,Path::new("."))?;
            let mut isolated_slot=isolated.create_frame_slot(&r,ReadbackPrecision::Float)?;
            let expected=isolated.render_frame(&r,&mut isolated_slot,0.)?.linear_rgba.context("isolated pixels")?;
            let mut mixed=r.install(scene(main,&["world","ui","empty"],None)?,Path::new("."))?;
            assert_eq!(mixed.render.settings.integrator.indirect().is_some(),main=="world","main {main}");
            let mut slot=mixed.create_frame_slot(&r,ReadbackPrecision::Float)?;
            let has_pass=|view:&crate::view::ViewState,pass|view.recording().steps.iter().any(|step|step.pass==pass);
            let assert_passes=|view:&crate::view::ViewState,needed| {
                assert_eq!(has_pass(view,crate::graph::Pass::Ao),needed,"GI pass");
                assert_eq!(has_pass(view,crate::graph::Pass::Reflections),needed,"reflection pass");
            };
            assert_passes(&slot.view,main=="world");
            for (image,capture) in mixed.graph.views.image_renders().zip(&slot.image_views) {
                let needed=image.name=="world";
                assert_eq!(image.settings.settings.integrator.indirect().is_some(),needed,"capture {}",image.name);
                assert_eq!(image.settings.settings.bounded_secondary_compilation,needed,"capture {}",image.name);
                assert_passes(&capture.view,needed);
                if !needed {
                    assert!(!has_pass(&capture.view,crate::graph::Pass::Denoise),"capture {}",image.name);
                }
            }
            // Capture-only emitter lighting still needs the shared source table.
            assert!(mixed.rays.as_ref().is_some_and(|rays| rays.emitter_weights.is_some()));
            let actual=mixed.render_frame(&r,&mut slot,0.)?.linear_rgba.context("mixed pixels")?;
            for c in 0..4 {
                let max=actual.iter().zip(&expected).map(|(a,b)|(a[c]-b[c]).abs()).fold(0_f32,f32::max);
                let rmse=(actual.iter().zip(&expected).map(|(a,b)|f64::from(a[c]-b[c]).powi(2)).sum::<f64>()/actual.len() as f64).sqrt();
                assert!(rmse<=1e-5 && max<=1e-4,"main {main}, channel {c}: RMSE={rmse}, max={max}");
            }
            assert!(actual.iter().any(|p|p[0]>0.01),"nonblack {main}");
        }
        // The qualifying capture must also preserve its lighting when the
        // main view changes providers. Exercise the authored receiver in both views.
        let mut reference: Option<Vec<[f32;4]>>=None;
        for main in ["world","ui"] {
            let mut installed=r.install(scene(main,&["world","ui","empty"],Some("world"))?,Path::new("."))?;
            let mut slot=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
            let pixels=installed.render_frame(&r,&mut slot,0.)?.linear_rgba.context("capture pixels")?;
            assert!(pixels.iter().any(|p| p[0] > 0.01), "nonblack authored receiver");
            for (image,capture) in installed.graph.views.image_renders().zip(&slot.image_views) {
                let needed=image.name=="world";
                assert!(!capture.sampled_uniforms.is_empty());
                for u in &capture.sampled_uniforms {
                    assert_eq!(u.flags[3],f32::from(needed),"{}: indirect uniform",image.name);
                    assert_eq!(u.ao[1],f32::from(needed),"{}: GI uniform",image.name);
                    assert_eq!(u.projection[2],if needed {constants::REFLECTIONS_RAY} else {constants::REFLECTIONS_ENVIRONMENT},"{}: reflection uniform",image.name);
                }
            }
            if let Some(expected)=&reference {
                for c in 0..4 {
                    let max=pixels.iter().zip(expected).map(|(a,b)|(a[c]-b[c]).abs()).fold(0_f32,f32::max);
                    let rmse=(pixels.iter().zip(expected).map(|(a,b)|f64::from(a[c]-b[c]).powi(2)).sum::<f64>()/pixels.len() as f64).sqrt();
                    assert!(rmse<=1e-5 && max<=1e-4,"capture channel {c}: RMSE={rmse}, max={max}");
                }
            }
            reference=Some(pixels);
        }
        Ok(())
    }
}
