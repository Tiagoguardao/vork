//! A canvas's image graph: the ordered graph a world runs, recorded by the same code (`record_images`), over
//! the canvas's float picture in the canvas's working encoding, before the canvas's view.
use super::*;
use vork_time::exact::Germ;

/// A canvas's installed graph. Its entries are recorded once, over buffers and image textures that stay for
/// the canvas's life; each draw publishes the frame's context, its animated inputs and the image inputs whose
/// source frame changed through one change tracker, as a world's frame slot does, then runs the recorded
/// entries.
///
/// The picture a canvas draws is associated RGBA in the encoding the canvas blends in (`composition.color_space`:
/// sRGB code values by default, linear light in a linear space), and so are the graph's images: an entry reads
/// and returns that encoding, before the canvas's view. Image inputs arrive in it too. The camera fields of an
/// entry's context are the identity and zero, and `ctx.samples` is one: a canvas is drawn once, through no
/// camera.
pub(crate) struct CanvasGraph {
    prepared: std::sync::Arc<CanvasGraphPrograms>,
    size: [u32; 2],
    fps: vork_time::Time,
    /// The drawn picture as the graph's first image, written by `ingest` each draw.
    pixels: wgpu::Buffer,
    /// The last image: the graph's result at the picture's size.
    output: wgpu::Buffer,
    dispatches: Vec<graph::Dispatch>,
    /// This draw's context.
    uniform: wgpu::Buffer,
    /// Animated input words by material.
    inputs: FrameInputs,
    images: Vec<ImageInput>,
    /// What the graph's buffers and image textures hold: the context, the animated inputs and each image's
    /// source frame (`ChangeKey::ImageClock`). A draw that fails forgets it, since its commands were not
    /// submitted, and the next draw publishes everything again.
    changes: std::cell::RefCell<crate::values::ChangeTracker>,
    /// Where a result that is not finite is recorded, unless the picture is drawn for a frame that collects its
    /// own diagnostics (`draw`).
    errors: crate::display_program::ErrorWord,
}
/// Immutable program installation shared by the graph's frame lanes. Each
/// instance binds its own writable images, uniforms and decoder state.
pub(crate) struct CanvasGraphPrograms {
    programs: std::collections::HashMap<String, Program>,
    ingest: wgpu::ComputePipeline,
    publish: wgpu::ComputePipeline,
    files: Vec<(String, FileInput)>,
}

enum FileInput {
    /// Published by the serial graph before admission. A single-frame file never changes.
    Still(wgpu::TextureView),
    Media(crate::footage::FileInput),
}

/// One image file a live entry reads: the resident texture a world's media image has, holding the encoding
/// the canvas blends in.
struct ImageInput {
    name: String,
    installed: crate::footage::Installed,
}

const INGEST: &str = "@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> pixels: array<vec4<f32>>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) p: vec3<u32>) {
    let size = textureDimensions(source); if (any(p.xy >= size)) { return; }
    pixels[p.y * size.x + p.x] = textureLoad(source, vec2<i32>(p.xy), 0);
}";

impl CanvasGraph {
    /// The graph of canvas `scene`, or `None` when its composition authors none.
    pub(crate) fn new(r: &Renderer, scene: &Scene, base: &Path, opened: &crate::footage::Opened) -> Result<Option<Self>> {
        let names = scene.compositor();
        if names.is_empty() { return Ok(None); }
        let [width, height] = scene.render.resolution;
        let bytes = u64::from(width) * u64::from(height) * constants::ACCUMULATION_PIXEL_BYTES;
        let limits = r.device.limits();
        ensure!(bytes <= u64::from(limits.max_storage_buffer_binding_size) && bytes <= limits.max_buffer_size,
            crate::gpu_memory::device_limit("canvas compositor picture", format_args!("{bytes} bytes"),
                format_args!("storage binding={} bytes, buffer={} bytes", limits.max_storage_buffer_binding_size, limits.max_buffer_size), "reduce the canvas resolution"));
        let programs = install_own(r, scene, base, opened)?;
        let pipeline = |label: &str, source: String| {
            let module = crate::timing::shader_module(&r.device, wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(source.into()) });
            // Through the persistent pipeline cache; the compiler chooses `cache`.
            r.compiler.compute_cached(&wgpu::ComputePipelineDescriptor {
                label: Some(label), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
            })
        };
        let ingest = pipeline("canvas compositor ingest", INGEST.to_owned());
        // The result becomes the canvas's float picture again. A value that is not finite, alpha included, is
        // zeroed and recorded with the code a view's non-finite result has, which a frame's diagnostics read
        // too. RGBA16F holds magnitudes up to 65504: a larger finite value saturates there.
        let publish = pipeline("canvas compositor publish", format!("@group(0) @binding(0) var<storage, read> pixels: array<vec4<f32>>;
@group(0) @binding(1) var picture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<storage, read_write> errors: array<atomic<u32>>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) p: vec3<u32>) {{
    let size = textureDimensions(picture); if (any(p.xy >= size)) {{ return; }}
    var value = pixels[p.y * size.x + p.x];
    if (!all(abs(value) <= vec4<f32>(3.402823466e38))) {{ atomicMax(&errors[0], {code}u); value = vec4<f32>(0.0); }}
    textureStore(picture, vec2<i32>(p.xy), clamp(value, vec4<f32>(-65504.0), vec4<f32>(65504.0)));
}}", code = vork_slang::registry::DISPLAY_NONFINITE));
        // Each image a live entry reads: a file (the scene refuses a rendered image), a still as a sequence of
        // one frame. The colour decoder converts whatever the file or its declaration says into the encoding
        // the canvas blends in, and the texture keeps it.
        let working = scene.composition.as_ref().map_or(vork_scene::v2::WorkingColorSpace::Srgb, vork_scene::v2::Composition::working_space);
        let mut images = Vec::new();
        let mut views = std::collections::BTreeMap::new();
        let mut files = Vec::new();
        for name in image_reads(&programs, names) {
            let image = &scene.images[&name];
            let file = image.file.as_ref().with_context(|| format!("composition.compositor: image {name} is rendered; a 2D canvas's compositor reads image files"))?;
            let path = base.join(file);
            let installed = (|| -> Result<_> {
                let source = if crate::footage::is_media(&path) { opened.canvas_source(&path, image.color_space)? }
                    else { crate::footage::Source::Sequence { files: vec![path.clone()], fps: None, inputs: opened.files.clone() } };
                crate::footage::Installed::from_source(r, &name, source, image.time.clone(), working, false, image.color_space, scene.render.fps)
            })().with_context(|| format!("composition.compositor: image {name} ({})", path.display()))?;
            let file = if crate::footage::is_media(&path) { FileInput::Media(installed.file_input()) }
                else { FileInput::Still(installed.view.clone()) };
            files.push((name.clone(), file));
            views.insert(name.clone(), (installed.view.clone(), images::associated_file(image)));
            images.push(ImageInput { name, installed });
        }
        let prepared = std::sync::Arc::new(CanvasGraphPrograms { programs, ingest, publish, files });
        Ok(Some(Self::bind(r, scene, prepared, images, views)?))
    }

    pub(crate) fn programs(&self) -> std::sync::Arc<CanvasGraphPrograms> {
        self.prepared.clone()
    }

    pub(crate) fn from_programs(r: &Renderer, scene: &Scene, prepared: std::sync::Arc<CanvasGraphPrograms>) -> Result<Self> {
        let mut images = Vec::new();
        let mut views = std::collections::BTreeMap::new();
        for (name, file) in &prepared.files {
            let image = &scene.images[name];
            let view = match file {
                FileInput::Still(view) => view.clone(),
                FileInput::Media(input) => {
                    let installed = crate::footage::Installed::from_file(r, name, image.time.clone(), input);
                    let view = installed.view.clone();
                    images.push(ImageInput { name: name.clone(), installed });
                    view
                }
            };
            views.insert(name.clone(), (view, images::associated_file(image)));
        }
        Self::bind(r, scene, prepared, images, views)
    }

    fn bind(r: &Renderer, scene: &Scene, prepared: std::sync::Arc<CanvasGraphPrograms>, images: Vec<ImageInput>,
        views: std::collections::BTreeMap<String, (wgpu::TextureView, bool)>) -> Result<Self> {
        let names = scene.compositor();
        let programs = &prepared.programs;
        let size = scene.render.resolution;
        let fps = scene.render.fps;
        let bytes = u64::from(size[0]) * u64::from(size[1]) * constants::ACCUMULATION_PIXEL_BYTES;
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| crate::gpu_memory::buffer(&r.device, &wgpu::BufferDescriptor {
            label: Some(label), size, usage, mapped_at_creation: false,
        });
        let pixels = buffer("canvas compositor picture", bytes, wgpu::BufferUsages::STORAGE);
        let uniform = buffer("canvas compositor context", std::mem::size_of::<Uniform>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let inputs = FrameInputs::new(programs, r, names);
        let (output, dispatches) = record_images(images::Sources::Canvas(&views), programs, r, &uniform, &pixels, names, &inputs.buffers, size,
            Normalization::None, graph::Pass::Compositor, buffer)?;
        Ok(Self { prepared, size, fps, pixels, output, dispatches, uniform, inputs, images, changes: Default::default(),
            errors: crate::display_program::ErrorWord::new(r, "canvas compositor errors") })
    }

    /// Run the graph over `source`, the canvas's drawn float picture, at the picture's local instant `request`,
    /// and return its result as a float picture of the same form: a texture of this draw's own, since its
    /// readers are recorded in commands submitted later. `instants` holds each image input's instant by name
    /// (`canvas::Document` evaluates them with the frame). The graph's commands follow the
    /// drawing that made `source`, in its frame submission when a prefix is open. `errors`: the shader diagnostics of the frame the picture is drawn for, which
    /// that frame refuses where it is collected; without them the graph's own word, cleared here and read
    /// through `errors` or `check`.
    pub(crate) fn draw(&self, r: &Renderer, source: &wgpu::Texture, request: &vork_time::Request, output_index: Option<u32>, instants: &[(String, Germ)],
        errors: Option<&wgpu::Buffer>, prefix: Option<&mut moda_engine::scratch::FramePrefix>) -> Result<wgpu::Texture> {
        let mut changes = self.changes.borrow_mut();
        let drawn = self.record(r, source, request, output_index, instants, errors, &mut changes);
        // A failed draw submitted nothing: what it recorded as published is not on the GPU.
        if drawn.is_err() { *changes = Default::default(); }
        let (picture, commands) = drawn?;
        if let Some(prefix) = prefix {
            prefix.append(commands);
        } else {
            r.queue.submit([commands]);
        }
        Ok(picture)
    }
    fn record(&self, r: &Renderer, source: &wgpu::Texture, request: &vork_time::Request, output_index: Option<u32>, instants: &[(String, Germ)],
        errors: Option<&wgpu::Buffer>, changes: &mut crate::values::ChangeTracker) -> Result<(wgpu::Texture, wgpu::CommandBuffer)> {
        ensure!(source.format() == wgpu::TextureFormat::Rgba16Float && [source.width(), source.height()] == self.size,
            "a canvas compositor reads the canvas's float picture");
        let time = request.numeric()?;
        // Two exactly representable words carry a numbered output's u32 to
        // Slang. An image sampled by a world keeps the world's numeric frame
        // uniform at its own local clock, including its precision at large times.
        let (frame_lo, frame_hi) = match output_index {
            Some(index) => ((index & 0xffff) as f32, (index >> 16) as f32),
            None => (request.nearest_frame(self.fps)? as f32, 0.),
        };
        let [w, h] = self.size;
        let mut frame = Uniform::zeroed();
        frame.view = glam::Mat4::IDENTITY;
        frame.composite_projection = glam::Mat4::IDENTITY;
        // One finished sample at the instant: nothing accumulates, and the shutter is closed.
        frame.temporal = [0., 1., 0., 0.];
        frame.scene = [time, frame_lo, frame_hi, self.fps.as_f32()];
        frame.display = [0., w as f32, h as f32, 0.];
        frame.composite_shutter = [time; 4];
        frame.composite_step = [0., 1., 1., 1.];
        changes.upload(&self.uniform, 0, bytemuck::bytes_of(&frame));
        // Sampled at the exact request, as the canvas's own tracks are.
        self.inputs.publish(&self.prepared.programs, changes, |animation, words| animation.words_request(request, time, words))?;
        let picture = crate::gpu_memory::texture(&r.device, &wgpu::TextureDescriptor {
            label: Some("canvas composited picture"), size: source.size(), mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba16Float,
            // COPY_SRC: tests read the picture back.
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
        });
        let mut encoder = r.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("canvas compositor") });
        if errors.is_none() { encoder.clear_buffer(self.errors.buffer(), 0, None); }
        // The changed words, as ordered copies in these commands.
        if !changes.bytes.is_empty() {
            let upload = init(&r.device, "canvas compositor uploads", &changes.bytes, wgpu::BufferUsages::COPY_SRC);
            for (destination, offset, source, size) in &changes.uploads { encoder.copy_buffer_to_buffer(&upload, *source, destination, *offset, *size); }
        }
        changes.bytes.clear();
        changes.uploads.clear();
        // An image input's texture is rewritten in these commands when its source frame changed.
        for image in &self.images {
            let (_, instant) = instants.iter().find(|(name, _)| *name == image.name).context("a compositor image has no instant")?;
            if changes.changed(crate::values::ChangeKey::ImageClock(image.name.clone()), &image.installed.sample_key(Some(instant), self.fps)?) {
                image.installed.encode(r, &mut encoder, Some(instant), self.fps, None).with_context(|| format!("composition.compositor: image {}", image.name))?;
            }
        }
        let groups = [w.div_ceil(8), h.div_ceil(8), 1];
        let source = source.create_view(&Default::default());
        let ingest = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("canvas compositor ingest"), layout: &self.prepared.ingest.get_bind_group_layout(0),
            entries: &[entry(0, wgpu::BindingResource::TextureView(&source)), entry(1, self.pixels.as_entire_binding())],
        });
        graph::Dispatch::new(&self.prepared.ingest, ingest, groups).encode_packet(&mut encoder, false);
        for dispatch in &self.dispatches { dispatch.encode_packet(&mut encoder, false); }
        let target = picture.create_view(&Default::default());
        let publish = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("canvas compositor publish"), layout: &self.prepared.publish.get_bind_group_layout(0),
            entries: &[entry(0, self.output.as_entire_binding()), entry(1, wgpu::BindingResource::TextureView(&target)),
                entry(2, errors.unwrap_or(self.errors.buffer()).as_entire_binding())],
        });
        graph::Dispatch::new(&self.prepared.publish, publish, groups).encode_packet(&mut encoder, false);
        Ok((picture, encoder.finish()))
    }
    /// The graph's own error word (four bytes) after the last `draw` without a frame's diagnostics. Copy it
    /// before the next `draw` clears it, and `display_program::refuse` it where the frame is collected.
    pub(crate) fn errors(&self) -> &wgpu::Buffer { self.errors.buffer() }
    /// Wait for the last `draw` and refuse a result that is not finite: for a picture with no collection point
    /// of its own.
    pub(crate) fn check(&self, r: &Renderer) -> Result<()> { self.errors.check(r) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn working_images_follow_last_bindings_and_belong_to_the_replica() -> Result<()> {
        let r = crate::test_renderer()?;
        let scene = Scene::from_json(&serde_json::to_vec(&json!({
            "version": 2, "composition": {"space": "2d", "compositor": [
                {"name": "keep", "material": "pass"}, "pass", "pass", "pass",
                {"material": "merge", "resolution": [8, 4]}
            ]}, "render": {"resolution": [16, 8]}, "nodes": [],
            "programs": {
                "pass": {"source": "struct Inputs {}; float4 composite(CompositeContext c, Inputs i) { return compositor_sample(c.uv); }"},
                "merge": {"source": "struct Inputs { Image saved; }; float4 composite(CompositeContext c, Inputs i) { return compositor_sample(c.uv) + compositor_sample(i.saved,c.uv) + float4(float(c.frame),0.,0.,1.); }"}
            }, "materials": {
                "pass": {"program": "pass"},
                "merge": {"program": "merge", "inputs": {"saved": {"step": "keep"}}}
            }
        }))?)?;
        let (graph, allocations) = crate::gpu_memory::test_buffer_allocations(||
            CanvasGraph::new(&r, &scene, Path::new("."), &Default::default()));
        let graph = graph?.unwrap();
        // keep + two alternating full-size steps + one smaller step. The full-size
        // resample may reuse an alternating step, but still binds keep read-only.
        let images: Vec<_> = allocations.iter().filter(|label| label.starts_with("Compositor image ")).collect();
        assert_eq!(images.len(), 4, "{images:?}");
        let replica = CanvasGraph::from_programs(&r, &scene, graph.programs())?;
        assert_ne!(replica.output, graph.output);

        // Queue both graphs and then reuse the first graph before collecting any
        // picture. Each consumer sees its own frame even after scratch is rewritten.
        let source = texture(&r.device, "blank canvas", 16, 8, 1,
            wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::TEXTURE_BINDING);
        let mut pictures = Vec::new();
        for (index, instance) in [(1, &graph), (2, &replica), (3, &graph)] {
            pictures.push((index, instance.draw(&r, &source, &vork_time::Request::Numeric(0.),
                Some(index), &[], None, None)?));
        }
        for (index, picture) in pictures.into_iter().rev() {
            let pixels = moda_engine::readback::read_texture_rgba16f(&r.device, &r.queue, &picture, 16, 8)?;
            for pixel in pixels.chunks_exact(4) {
                assert_eq!(pixel[0].to_f32(), index as f32);
                assert_eq!(pixel[3].to_f32(), 1.);
            }
        }
        Ok(())
    }
}
