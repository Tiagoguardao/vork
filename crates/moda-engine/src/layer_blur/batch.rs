//! Record independent capture filters in frame order. Compatible chains share a dispatch per stage across
//! texture-array layers. Raster batches keep separate attachments for blending and other fragment pipelines.
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::sync::{Arc, OnceLock};
use crate::device::OptionalPipeline;
use moda_text_draw::device::PipelineCreation;

const MAX_TARGETS: usize = 4;

pub(crate) struct FilterPipeline {
    module: &'static str,
    source: Cow<'static, str>,
    entry: &'static str,
    layout: wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
    auxiliary: Option<&'static str>,
    scalar: OnceLock<wgpu::RenderPipeline>,
    variants: Arc<FilterVariants>,
}

#[derive(Default)]
pub(crate) struct FilterVariants {
    array: OptionalPipeline<(wgpu::ComputePipeline, Option<wgpu::RenderPipeline>)>,
    raster: [OptionalPipeline<wgpu::RenderPipeline>; MAX_TARGETS - 1],
}

pub(crate) type FilterLibrary = std::sync::Mutex<HashMap<(&'static str, &'static str), Arc<FilterVariants>>>;

impl FilterPipeline {
    pub fn new(
        module: &'static str,
        entry: &'static str,
        source: Cow<'static, str>,
        layout: &wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
        auxiliary: Option<&'static str>,
        blend: Option<wgpu::BlendState>,
        library: &FilterLibrary,
    ) -> Arc<Self> {
        Arc::new(Self {
            module,
            entry,
            source,
            layout: layout.clone(),
            format,
            blend,
            auxiliary,
            scalar: Default::default(),
            variants: library.lock().unwrap().entry((module, entry)).or_default().clone(),
        })
    }

    // Width zero identifies the compute variant. These are scheduling hints only.
    fn hint(module: &str, entry: &str, width: usize) -> String { format!("canvas/filter/{module}/{entry}/{width}") }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn seen(&self, compiler: &moda_text_draw::device::OptionalCompiler, width: usize) -> bool {
        compiler.has_capture_hint(&Self::hint(self.module, self.entry, width))
    }

    fn source(&self, count: usize) -> Cow<'_, str> {
        if count == 1 {
            return self.source.as_ref().into();
        }
        let (uniform, uniform_type, image) = match self.module {
            "backdrop-blur" => ("blur", "BlurUniforms", "inputTexture"),
            "progressive-blur" => ("blur", "ProgressiveBlurUniforms", "inputTexture"),
            "layer-filter-shadow" => ("shadow", "ShadowUniforms", "inputTexture"),
            "layer-filter-texture" => ("tex", "TextureUniforms", "inputTexture"),
            "layer-filter-noise" => ("noise", "NoiseUniforms", "inputTexture"),
            "css-color-matrix" => ("matrix", "ColorMatrix", "picture"),
            _ => unreachable!("unknown layer filter"),
        };
        let mut source = self.source.to_string();
        for i in 1..count {
            writeln!(
                source,
                "@group({i}) @binding(0) var<uniform> {uniform}{i}: {uniform_type};"
            )
            .unwrap();
            writeln!(
                source,
                "@group({i}) @binding(1) var inputSampler{i}: sampler;"
            )
            .unwrap();
            writeln!(
                source,
                "@group({i}) @binding(2) var {image}{i}: texture_2d<f32>;"
            )
            .unwrap();
            if let Some(aux) = self.auxiliary {
                writeln!(
                    source,
                    "@group({i}) @binding(3) var {aux}{i}: texture_2d<f32>;"
                )
                .unwrap();
            }
        }
        source.push_str("struct FilterOutputs {\n");
        for i in 0..count {
            writeln!(source, "@location({i}) color{i}: vec4f,").unwrap();
        }
        let input = if self.module == "backdrop-blur" {
            "pixel: VSOut"
        } else {
            "@builtin(position) pixel: vec4f"
        };
        writeln!(
            source,
            "}}\n@fragment fn filter_batch({input}) -> FilterOutputs {{\nvar out: FilterOutputs;"
        )
        .unwrap();
        for i in 0..count {
            let suffix = if i == 0 { String::new() } else { i.to_string() };
            let extra = self
                .auxiliary
                .map(|aux| format!(", {aux}{suffix}"))
                .unwrap_or_default();
            writeln!(source,"out.color{i} = shade_{}(pixel, {uniform}{suffix}, inputSampler{suffix}, {image}{suffix}{extra});",self.entry).unwrap();
        }
        source.push_str("return out;\n}");
        source.into()
    }

    /// The caller can always draw with the scalar pipeline. Never join a compiler
    /// from the recording thread. A variant becomes eligible on publication.
    pub(super) fn max_targets(device: &wgpu::Device) -> usize {
        let limits = device.limits();
        MAX_TARGETS.min(limits.max_bind_groups as usize).min(limits.max_color_attachments as usize)
            .min((limits.max_color_attachment_bytes_per_sample / 8) as usize).max(1)
    }

    pub(super) fn raster_width(self: &Arc<Self>, device: &wgpu::Device, compatible: usize, limit: usize) -> usize {
        let width = compatible.min(limit);
        if width > 1 {
            let pipeline = self.clone();
            let (module, entry) = (self.module, self.entry);
            self.variants.raster[width - 2].request(device, "multi-target filter", move || Some(Self::hint(module, entry, width)), move |device, cache| {
                Ok(pipeline.build_raster(device, width, PipelineCreation::Optional(cache)))
            });
        }
        self.ready_width(width)
    }

    fn ready_width(&self, width: usize) -> usize {
        (2..=width).rev().find(|&n| self.variants.raster[n - 2].ready().is_some()).unwrap_or(1)
    }

    pub(super) fn request_array(self: &Arc<Self>, device: &wgpu::Device) {
        let pipeline = self.clone();
        let (module, entry) = (self.module, self.entry);
        self.variants.array.request(device, "array filter", move || Some(Self::hint(module, entry, 0)), move |device, cache| {
            let coordinates = (pipeline.module == "backdrop-blur").then(|| pipeline.build_coordinates(device, cache));
            Ok((pipeline.build_array(device, cache), coordinates))
        });
    }

    fn get(&self, device: &wgpu::Device, count: usize) -> &wgpu::RenderPipeline {
        if count == 1 {
            self.scalar.get_or_init(|| self.build_raster(device, 1, PipelineCreation::Required))
        } else {
            #[cfg(test)]
            crate::device::VARIANT_DRAWS.with_borrow_mut(|draws| draws.insert("multi-target filter"));
            self.variants.raster[count - 2].ready().expect("selected published raster width")
        }
    }

    fn build_raster(&self, device: &wgpu::Device, count: usize, creation: PipelineCreation) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(self.module),
            source: wgpu::ShaderSource::Wgsl(self.source(count)),
        });
        let groups = vec![Some(&self.layout); count];
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(self.module),
            bind_group_layouts: &groups,
            ..Default::default()
        });
        let targets = vec![
            Some(wgpu::ColorTargetState {
                format: self.format,
                blend: self.blend,
                write_mask: wgpu::ColorWrites::ALL,
            });
            count
        ];
        creation.render(
            device,
            &wgpu::RenderPipelineDescriptor {
                label: Some(self.module),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(if count == 1 {
                        self.entry
                    } else {
                        "filter_batch"
                    }),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            },
        )
    }
}

/// A filter stage describes its sampling extent before the batch chooses
/// individual attachments or array layers. Only the chosen storage is allocated.
#[derive(Clone)]
pub(crate) struct FilterSurface(std::rc::Rc<Surface>);
struct Surface {
    size: [u32; 2],
    format: wgpu::TextureFormat,
    view: std::cell::OnceCell<wgpu::TextureView>,
    individual: std::cell::Cell<bool>,
    binding_bytes: std::cell::Cell<Option<u64>>,
    failure: std::cell::OnceCell<std::rc::Rc<anyhow::Error>>,
    layer: std::cell::OnceCell<(wgpu::TextureView, u32)>,
}
impl PartialEq for FilterSurface {
    fn eq(&self, other: &Self) -> bool {
        std::rc::Rc::ptr_eq(&self.0, &other.0)
            || self
                .0
                .view
                .get()
                .zip(other.0.view.get())
                .is_some_and(|(a, b)| a == b)
    }
}
impl FilterSurface {
    pub fn new(size: [u32; 2], format: wgpu::TextureFormat) -> Self {
        Self(std::rc::Rc::new(Surface {
            size,
            format,
            view: Default::default(),
            individual: Default::default(),
            binding_bytes: Default::default(),
            failure: Default::default(),
            layer: Default::default(),
        }))
    }
    pub(crate) fn reject(&self, error: std::rc::Rc<anyhow::Error>) { let _ = self.0.failure.set(error); }
    pub(crate) fn failure(&self) -> Option<&std::rc::Rc<anyhow::Error>> { self.0.failure.get() }

    pub fn keep_individual(&self) {
        self.0.individual.set(true);
    }
    pub fn individual(&self) -> bool { self.0.individual.get() }

    pub fn resolve(&self, view: wgpu::TextureView) {
        self.0.view.set(view).expect("page records into one result");
    }

    pub fn existing(&self) -> Option<&wgpu::TextureView> {
        self.0.view.get()
    }
    pub fn external(view: wgpu::TextureView) -> Self {
        let tex = view.texture();
        let surface = Self::new([tex.width(), tex.height()], tex.format());
        surface.0.view.set(view).expect("new source");
        surface
    }
    pub fn layer(&self) -> Option<&(wgpu::TextureView, u32)> {
        self.0.layer.get()
    }
    pub fn set_layer(&self, array: wgpu::TextureView, index: u32) {
        self.0.layer.set((array, index)).expect("surface recorded once");
    }
    fn resource(&self) -> SurfaceResource {
        match self.existing().or_else(|| self.layer().map(|(v, _)| v)) {
            Some(view) => SurfaceResource::Texture(view.texture().clone()),
            None => SurfaceResource::Planned(std::rc::Rc::as_ptr(&self.0)),
        }
    }

    pub fn width(&self) -> u32 {
        self.0.size[0]
    }
    pub fn height(&self) -> u32 {
        self.0.size[1]
    }
    pub(crate) fn price(&self, device: &wgpu::Device, pool: &crate::scratch::ScratchPool) -> u64 {
        if let Some(bytes) = self.0.binding_bytes.get() { return bytes; }
        let bytes = pool.texture_price(device, self.attachment_key());
        self.0.binding_bytes.set(Some(bytes));
        bytes
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.0.binding_bytes.get().unwrap_or_else(|| u64::from(self.width()) * u64::from(self.height())
            * u64::from(self.format().block_copy_size(None).unwrap_or(8)))
    }

    fn format(&self) -> wgpu::TextureFormat {
        self.0.format
    }
    fn attachment_usage() -> wgpu::TextureUsages {
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC
    }

    pub(crate) fn attachment_key(&self) -> crate::scratch::TexKey {
        crate::scratch::TexKey::capture(self.width(), self.height(), 1, 1, self.format(), Self::attachment_usage())
    }

    pub fn view(
        &self,
        device: &wgpu::Device,
        scratch: &crate::scratch::ScratchPool,
    ) -> &wgpu::TextureView {
        self.0.view.get_or_init(|| {
            if let Some((array, layer)) = self.layer() {
                return scratch.layer_view(array.texture(), *layer);
            }
            let texture = scratch.texture(
                device,
                &wgpu::TextureDescriptor {
                    label: Some("Layer filter target"),
                    size: wgpu::Extent3d {
                        width: self.width(),
                        height: self.height(),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.format(),
                    usage: Self::attachment_usage(),
                    view_formats: &[],
                },
            );
            scratch.view(&texture)
        })
    }
}

#[derive(Clone)]
pub(crate) struct FilterDraw {
    pub pipeline: Arc<FilterPipeline>,
    pub parameters: Vec<f32>,
    pub sampler: wgpu::Sampler,
    pub input: FilterSurface,
    pub second: Option<FilterSurface>,
    pub target: FilterSurface,
    pub scissor: Option<[u32; 4]>,
    pub load: bool,
}

impl FilterDraw {
    fn key(&self) -> (*const FilterPipeline, u32, u32, Option<[u32; 4]>) {
        let tex = &self.target;
        (
            Arc::as_ptr(&self.pipeline),
            tex.width(),
            tex.height(),
            self.scissor,
        )
    }
}

/// Multiple views, including different layers of a source array, have one lifetime.
#[derive(Clone, PartialEq, Eq, Hash)]
enum SurfaceResource {
    Texture(wgpu::Texture),
    Planned(*const Surface),
}

/// Uses are counted in recording order. Remember the identity assigned before lazy
/// allocation so allocating a target does not change the key of its future readers.
struct SurfaceUses {
    resources: HashMap<*const Surface, SurfaceResource>,
    remaining: HashMap<SurfaceResource, usize>,
    outputs: HashSet<SurfaceResource>,
    painter: Vec<FilterSurface>,
    cached: HashSet<wgpu::Texture>,
}

impl SurfaceUses {
    fn new(chains: &[Vec<FilterDraw>], outputs: &[FilterSurface], cached: HashSet<wgpu::Texture>) -> Self {
        let mut uses = Self {
            resources: HashMap::new(),
            remaining: HashMap::new(),
            outputs: outputs.iter().map(FilterSurface::resource).collect(),
            painter: outputs.to_vec(), cached,
        };
        for draw in chains.iter().flatten() {
            for surface in draw.surfaces() {
                let key = uses.resources.entry(std::rc::Rc::as_ptr(&surface.0))
                    .or_insert_with(|| surface.resource());
                *uses.remaining.entry(key.clone()).or_default() += 1;
            }
        }
        uses
    }

    fn finish_surface(&mut self, surface: &FilterSurface, scratch: &crate::scratch::ScratchPool) {
        let key = &self.resources[&std::rc::Rc::as_ptr(&surface.0)];
        let count = self.remaining.get_mut(key).expect("registered draw");
        *count -= 1;
        if *count == 0 && !self.outputs.contains(key) {
            if let Some(view) = surface.existing().or_else(|| surface.layer().map(|(v, _)| v)) {
                scratch.recycle_attachment(view.texture());
            }
        }
    }

    fn finish(&mut self, draw: &FilterDraw, scratch: &crate::scratch::ScratchPool) {
        for surface in draw.surfaces() { self.finish_surface(surface, scratch); }
    }

    /// Already-resolved physical images belonging to unfinished draws or the
    /// painter. Newly assigned array slices may share one backing allocation.
    fn live_bytes(&self) -> u64 {
        let mut live = HashSet::new();
        for (key, count) in &self.remaining {
            if let SurfaceResource::Texture(t) = key {
                if *count > 0 && !self.cached.contains(t) { live.insert(t); }
            }
        }
        for out in &self.painter {
            if let Some(view) = out.existing().or_else(|| out.layer().map(|(v, _)| v)) {
                if !self.cached.contains(view.texture()) { live.insert(view.texture()); }
            }
        }
        live.into_iter().map(crate::scratch::texture_bytes).sum()
    }

}

impl FilterDraw {
    pub(crate) fn surfaces(&self) -> impl Iterator<Item = &FilterSurface> {
        [&self.input, &self.target].into_iter().chain(self.second.as_ref())
    }
}

/// Resident intervals in the ordered filter stages. Stage zero resolves the sources.
/// Results remain live for the painter. A previously bound source already has
/// an owner and consumes no new space in the proposed batch.
#[derive(Default)]
pub(crate) struct CaptureFootprint {
    intervals: HashMap<SurfaceResource, (usize, usize)>,
    changes: Vec<i64>,
    // Logical intervals remain independent when adjacent source draws merge.
    // Each physical interval appears in the stage deltas exactly once.
    aliases: HashMap<SurfaceResource, SurfaceResource>,
    physical: HashMap<SurfaceResource, (usize, usize, u64)>,
    pub(crate) fresh_uniforms: crate::scratch::UploadLayout,
    source_uploads: u64,
    pub(crate) uniforms: crate::scratch::UploadLayout,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct CaptureDemand {
    sources: u64,
    geometry: u64,
    filters: u64,
}

impl CaptureDemand {
    pub(crate) fn with_uniforms(self, upload: u64) -> (u64, u64) {
        let filters = self.filters + upload;
        (self.geometry + self.sources.max(filters), filters)
    }
}

impl CaptureFootprint {
    fn change(&mut self, at: usize, delta: i64) {
        if at == usize::MAX { return; }
        if self.changes.len() <= at { self.changes.resize(at + 1, 0); }
        self.changes[at] += delta;
    }

    fn remove_interval(&mut self, key: &SurfaceResource) {
        if let Some((start, end, bytes)) = self.physical.remove(key) {
            self.change(start, -(bytes as i64));
            self.change(end, bytes as i64);
        }
    }

    fn extend(&mut self, surface: &FilterSurface, start: usize, end: usize) {
        if surface.existing().is_some() || surface.layer().is_some() { return; }
        let key = surface.resource();
        let entry = self.intervals.entry(key.clone()).or_insert((usize::MAX, 0));
        entry.0 = entry.0.min(start);
        entry.1 = entry.1.max(end);
        let key = self.aliases.get(&key).cloned().unwrap_or(key);
        let old = self.physical.get(&key).copied();
        let range = old.map_or((start, end, surface.bytes()), |(a,b,bytes)| (a.min(start), b.max(end), bytes));
        if old != Some(range) {
            self.remove_interval(&key);
            self.change(range.0, range.2 as i64);
            self.change(range.1, -(range.2 as i64));
            self.physical.insert(key, range);
        }
    }

    /// Try the incoming chain before recording its page or allocating its targets.
    /// On rejection the caller closes this schedule and retries in an empty one.
    pub(crate) fn include(&mut self, source: &FilterSurface, coverage: Option<&FilterSurface>,
        draws: &[FilterDraw], output: &FilterSurface, page_uploads: u64) {
        self.extend(source, 0, 1);
        if let Some(coverage) = coverage { self.extend(coverage, 0, 1); }
        self.source_uploads += page_uploads;
        for (i, draw) in draws.iter().enumerate() {
            let stage = i + 1;
            for s in draw.surfaces() { self.extend(s, stage, stage + 1); }
            let size = draw.parameters.len() as u64 * 4;
            self.fresh_uniforms.add(size);
            self.uniforms.add(size);
        }
        let first_write = self.intervals.get(&output.resource()).map_or(draws.len(), |range| range.0);
        self.extend(output, first_write, usize::MAX);
    }

    /// Replace individually priced layer views with their shared backing, from
    /// resolve to the last stage that reads any slice. Individually copied slices
    /// retain their own interval beside that backing.
    pub(crate) fn share_layers<'a>(&mut self, bytes: u64, layers: impl Iterator<Item = &'a FilterSurface>) {
        let mut backing = None;
        let mut end = 0;
        for layer in layers {
            let key = layer.resource();
            end = end.max(self.intervals[&key].1);
            let physical = self.aliases.remove(&key).unwrap_or_else(|| key.clone());
            self.remove_interval(&physical);
            self.aliases.insert(key.clone(), backing.get_or_insert(key).clone());
        }
        if let Some(key) = backing {
            self.change(0, bytes as i64);
            self.change(end, -(bytes as i64));
            self.physical.insert(key, (0, end, bytes));
        }
    }

    pub(crate) fn demand(&self, sources: u64) -> CaptureDemand {
        let mut live = 0;
        let mut filter = 0;
        for (stage, delta) in self.changes.iter().enumerate() {
            live += delta;
            if stage > 0 { filter = filter.max(live as u64); }
        }
        CaptureDemand { sources, geometry: self.source_uploads, filters: filter }
    }

}

#[derive(Default)]
pub(crate) struct FilterBatch {
    chains: Vec<Vec<FilterDraw>>,
}

impl FilterBatch {
    pub fn add(&mut self, chain: Vec<FilterDraw>) {
        if !chain.is_empty() { self.chains.push(chain); }
    }
    /// Propagate source errors along filter dependencies. A failed source cannot
    /// publish an uninitialized result or suppress an independent sibling.
    pub(crate) fn discard_failed_sources(&mut self) {
        self.chains.retain(|chain| {
            let mut failed = false;
            for draw in chain {
                if let Some(error) = draw.input.failure().or_else(|| draw.second.as_ref().and_then(FilterSurface::failure))
                    .or_else(|| draw.load.then(|| draw.target.failure()).flatten()) {
                    draw.target.reject(error.clone());
                    failed = true;
                }
            }
            !failed
        });
    }

    pub(crate) fn uniform_bytes(&self, mut layout: crate::scratch::UploadLayout) -> u64 {
        for draw in self.chains.iter().flatten() { layout.add(draw.parameters.len() as u64 * 4); }
        layout.additional
    }

    pub fn requests(&self) -> HashMap<crate::scratch::TexKey, usize> {
        // A target's slot can be reused after its final filter read. Count
        // concurrently needed backings of each descriptor, including final outputs.
        let mut intervals = HashMap::new();
        for chain in &self.chains {
            for (stage, draw) in chain.iter().enumerate() {
                let entry = intervals.entry(draw.target.resource()).or_insert((draw.target.attachment_key(), stage, stage));
                entry.2 = entry.2.max(stage);
                for source in [&draw.input].into_iter().chain(draw.second.as_ref()) {
                    if let Some((_, _, end)) = intervals.get_mut(&source.resource()) { *end = (*end).max(stage); }
                }
            }
            if let Some(last) = chain.last() { intervals.get_mut(&last.target.resource()).unwrap().2 = usize::MAX; }
        }
        let mut events = HashMap::<_, Vec<i64>>::new();
        for (_, (key, start, end)) in intervals {
            let changes = events.entry(key).or_default();
            changes.resize(changes.len().max(start + 1), 0);
            changes[start] += 1;
            if end != usize::MAX {
                changes.resize(changes.len().max(end + 2), 0);
                changes[end + 1] -= 1;
            }
        }
        events.into_iter().map(|(key, stages)| {
            let mut live = 0;
            let peak = stages.into_iter().map(|delta| { live += delta; live }).max().unwrap_or(0);
            (key, peak as usize)
        }).collect()
    }

    pub(crate) fn cached_inputs(&self, cache: &super::cache::CaptureCache) -> HashSet<wgpu::Texture> {
        self.chains.iter().flatten().flat_map(FilterDraw::surfaces)
            .filter_map(|s| s.existing().or_else(|| s.layer().map(|(v, _)| v)))
            .map(wgpu::TextureView::texture).filter(|t| cache.owns(t)).cloned().collect()
    }

    pub fn record(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scratch: &crate::scratch::ScratchPool,
        sources: &[FilterSurface],
        outputs: &[FilterSurface],
        available_room: impl Fn() -> u64,
        cached: HashSet<wgpu::Texture>,
    ) {
        let mut uses = SurfaceUses::new(&self.chains, outputs, cached);
        for source in sources {
            let key = source.resource();
            if !uses.remaining.contains_key(&key) && !uses.outputs.contains(&key) {
                if let SurfaceResource::Texture(texture) = key { scratch.recycle_attachment(&texture); }
            }
        }
        if self.chains.is_empty() {
            return;
        }
        let _span = crate::prof::span("fx.batch_record");
        let width = FilterPipeline::max_targets(device);
        let mut encoder = crate::scratch::encoder(
            device,
            queue,
            &wgpu::CommandEncoderDescriptor {
                label: Some("Layer filters"),
            },
        );
        // A retained source may cease to be cache-owned after this recording.
        // Follow its consuming command buffer even if it is also a painter result.
        scratch.record_reads(&encoder, queue, self.chains.iter().flatten().flat_map(FilterDraw::surfaces)
            .filter_map(|s| s.existing().or_else(|| s.layer().map(|(v, _)| v)))
            .map(wgpu::TextureView::texture).filter(|t| !scratch.holds_texture(t)));
        let mut groups = std::collections::HashMap::<
            Vec<(usize, [u32; 2], [u32; 2], wgpu::TextureFormat)>,
            Vec<Vec<FilterDraw>>,
        >::new();
        let mut ordinary = Vec::new();
        for chain in self.chains {
            if array_chain(&chain, device) {
                let key = chain
                    .iter()
                    .map(|d| {
                        (
                            Arc::as_ptr(&d.pipeline) as usize,
                            [d.input.width(), d.input.height()],
                            [d.target.width(), d.target.height()],
                            d.input.format(),
                        )
                    })
                    .collect();
                groups.entry(key).or_default().push(chain);
            } else {
                ordinary.push(chain);
            }
        }
        let mut compiling_array = HashSet::new();
        let before_upload = scratch.upload_layout(queue, device, wgpu::BufferUsages::STORAGE).occupied();
        for group in groups.into_values() {
            let layers = device.limits().max_texture_array_layers as usize;
            let mut remaining = group.as_slice();
            while !remaining.is_empty() {
                let max = layers.min(remaining.len());
                // Price the source's actual backing and all other live chains before
                // requesting a specialization. The scalar recorder is always available.
                let uploads = scratch.upload_layout(queue, device, wgpu::BufferUsages::STORAGE);
                let available = available_room().saturating_sub(uploads.occupied().saturating_sub(before_upload));
                let live = uses.live_bytes();
                let mut low = 1;
                let mut high = max;
                while low < high {
                    let middle = (low + high + 1) / 2;
                    if array_storage(&remaining[..middle], &uses, live, uploads.clone(), device, scratch) <= available { low = middle; } else { high = middle - 1; }
                }
                let width = low;
                if width > 1 {
                    let mut ready = true;
                    for stage in &remaining[0] {
                        stage.pipeline.request_array(device);
                        ready &= stage.pipeline.variants.array.ready().is_some();
                    }
                    if !ready {
                        if !remaining[0].iter().any(|d| d.pipeline.variants.array.failed()) {
                            compiling_array.extend(remaining[0].iter().map(|d| Arc::as_ptr(&d.pipeline)));
                        }
                        ordinary.extend_from_slice(remaining);
                        break;
                    }
                }
                let (chunk, tail) = remaining.split_at(width);
                remaining = tail;
                if width == 1 {
                    ordinary.extend_from_slice(chunk);
                } else {
                    record_array(chunk, device, queue, scratch, &mut encoder, &mut uses);
                }
            }
        }
        let mut stages: Vec<Vec<FilterDraw>> = Vec::new();
        for chain in ordinary {
            for (i, draw) in chain.into_iter().enumerate() {
                if stages.len() == i {
                    stages.push(Vec::new());
                }
                stages[i].push(draw);
            }
        }
        for mut stage in stages {
            stage.sort_unstable_by_key(FilterDraw::key);
            for group in stage.chunk_by(|a, b| a.key() == b.key()) {
                let pipeline = &group[0].pipeline;
                let widest = if compiling_array.contains(&Arc::as_ptr(pipeline)) {
                    pipeline.ready_width(width.min(group.len()))
                } else { pipeline.raster_width(device, group.len(), width) };
                let mut remaining = group;
                while !remaining.is_empty() {
                    let arity = group[0].pipeline.ready_width(widest.min(remaining.len()));
                    let (draws, tail) = remaining.split_at(arity);
                    remaining = tail;
                    let attachments: Vec<_> = draws
                        .iter()
                        .map(|draw| {
                            Some(wgpu::RenderPassColorAttachment {
                                view: draw.target.view(device, scratch),
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: if draw.load {
                                        wgpu::LoadOp::Load
                                    } else {
                                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                                    },
                                    store: wgpu::StoreOp::Store,
                                },
                            })
                        })
                        .collect();
                    let first = &draws[0];
                    let pipeline = first.pipeline.get(device, draws.len());
                    let bindings: Vec<_> = draws
                        .iter()
                        .map(|draw| {
                            let mut bound = vec![
                                scratch.uniform(
                                    device,
                                    queue,
                                    "layer filter",
                                    &draw.parameters,
                                    false,
                                ),
                                crate::scratch::Bound::Sampler(draw.sampler.clone()),
                                crate::scratch::Bound::View(
                                    draw.input.view(device, scratch).clone(),
                                ),
                            ];
                            if let Some(view) = &draw.second {
                                bound.push(crate::scratch::Bound::View(
                                    view.view(device, scratch).clone(),
                                ));
                            }
                            scratch.bind(device, "layer filter", &draw.pipeline.layout, &bound)
                        })
                        .collect();
                    let ts = crate::gpu_profile::pass_timestamps();
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Layer filter stage"),
                        color_attachments: &attachments,
                        depth_stencil_attachment: None,
                        timestamp_writes: crate::gpu_profile::render_writes(&ts),
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    if let Some([x, y, w, h]) = first.scissor.and_then(|s| {
                        super::exec::fit(s, first.target.width(), first.target.height())
                    }) {
                        pass.set_scissor_rect(x, y, w, h);
                    }
                    pass.set_pipeline(pipeline);
                    for (i, binding) in bindings.iter().enumerate() {
                        pass.set_bind_group(i as u32, binding, &[]);
                    }
                    pass.draw(0..3, 0..1);
                    drop(pass);
                    for draw in draws { uses.finish(draw, scratch); }
                }
            }
        }
        crate::scratch::end_recording(queue, encoder);
    }
}

/// A linear chain whose sampling extent and original input are common to all
/// layers can run each stage as a single array dispatch. Loaded attachments
/// still use raster blending, including its native fixed-point rounding.
fn array_chain(chain: &[FilterDraw], device: &wgpu::Device) -> bool {
    let first = &chain[0];
    let input = &first.input;
    if input.layer().is_none() && input.existing().is_some_and(|v| {
        let tex = v.texture();
        !tex.usage().contains(wgpu::TextureUsages::COPY_SRC) || tex.depth_or_array_layers() != 1
    }) || !matches!(
        input.format(),
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba16Float
    ) || device.limits().max_storage_textures_per_shader_stage == 0
        || input.width() > 512
        || input.height() > 512
    {
        return false;
    }
    chain.iter().enumerate().all(|(i, d)| {
        let t = &d.target;
        d.scissor.is_none()
            && !d.load
            && t.width() <= 512 && t.height() <= 512
            && d.pipeline.entry != "fs_border"
            && matches!(
                d.pipeline.format,
                wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba16Float
            )
            && matches!(
                d.pipeline.module,
                "layer-filter-shadow" | "progressive-blur" | "backdrop-blur"
            )
            && (i == 0 || d.input == chain[i - 1].target)
            && d.second.as_ref().is_none_or(|s| *s == first.input)
    })
}

impl FilterPipeline {
    /// The scalar Gaussian reads the fullscreen triangle's interpolated UV. Reuse those
    /// hardware-interpolated bits for every array layer at this particular target size.
    fn coordinates(
        &self,
        device: &wgpu::Device,
        scratch: &crate::scratch::ScratchPool,
        encoder: &mut wgpu::CommandEncoder,
        w: u32,
        h: u32,
    ) -> wgpu::TextureView {
        let pipeline = self.variants.array.ready().and_then(|(_, uv)| uv.as_ref()).expect("published Gaussian coordinates");
        let texture = scratch.texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("filter UV"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                sample_count: 1,
                mip_level_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        let view = scratch.view(&texture);
        let ts = crate::gpu_profile::pass_timestamps();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("filter UV"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            timestamp_writes: crate::gpu_profile::render_writes(&ts),
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.draw(0..3, 0..1);
        drop(pass);
        view
    }
    fn build_coordinates(&self, device: &wgpu::Device, cache: Option<&wgpu::PipelineCache>) -> wgpu::RenderPipeline {
        let code = format!(
            "{}\n @fragment fn coordinates(in:VSOut)->@location(0) vec2f {{return in.uv;}}",
            self.source
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("filter UV"),
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("filter UV"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("coordinates"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rg32Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache,
        })
    }

    fn build_array(&self, device: &wgpu::Device, cache: Option<&wgpu::PipelineCache>) -> wgpu::ComputePipeline {
        let shader = array_shader(
            self.module,
            self.format,
            &self.source,
            self.auxiliary.is_some(),
            self.entry,
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Layer array"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Layer array"),
            layout: None,
            module: &shader,
            entry_point: Some("filter_layers"),
            compilation_options: Default::default(),
            cache,
        })
    }
}

fn array_shader(
    module: &str,
    target: wgpu::TextureFormat,
    source: &str,
    auxiliary: bool,
    entry: &str,
) -> String {
    let params = match module {
        "layer-filter-shadow" => "ShadowUniforms",
        "progressive-blur" => "ProgressiveBlurUniforms",
        "backdrop-blur" => "BlurUniforms",
        _ => unreachable!(),
    };
    let format = match target {
        wgpu::TextureFormat::Rgba8Unorm => "rgba8unorm",
        wgpu::TextureFormat::Rgba16Float => "rgba16float",
        _ => unreachable!(),
    };
    // Retain the actual scalar shading functions and working-color specialization.
    // The sampler adds a layer coordinate; the spatial coordinates are unchanged.
    let body = source
        .split("@fragment")
        .next()
        .unwrap()
        .lines()
        .filter(|line| !line.trim_start().starts_with("@group("))
        .collect::<Vec<_>>()
        .join("\n")
        .replace("texture_2d<f32>", "texture_2d_array<f32>")
        .replace("textureLoad(", "read_texel(")
        .replace("textureSampleLevel(", "read_layer(")
        .replace("textureSample(", "read_base(");
    let (coordinate_decl, position) = if module == "backdrop-blur" {
        (
            "@group(0) @binding(5) var coordinates: texture_2d<f32>;",
            "VSOut(pos,textureLoad(coordinates,vec2i(gid.xy),0).xy)",
        )
    } else {
        ("", "pos")
    };
    let aux_decl = if auxiliary {
        "@group(0) @binding(3) var original: texture_2d_array<f32>;"
    } else {
        ""
    };
    let aux_arg = if auxiliary { ", original" } else { "" };
    format!(
        r#"
{body}
var<private> layer: u32;
fn read_texel(t:texture_2d_array<f32>,p:vec2i,lod:i32) -> vec4f {{
  return textureLoad(t,p,i32(layer),lod);
}}
fn read_layer(t:texture_2d_array<f32>, s:sampler, uv:vec2f, lod:f32) -> vec4f {{
  return textureSampleLevel(t,s,uv,i32(layer),lod);
}}
fn read_base(t:texture_2d_array<f32>,s:sampler,uv:vec2f)->vec4f {{
  return textureSampleLevel(t,s,uv,i32(layer),0.0);
}}
{coordinate_decl}
@group(0) @binding(0) var<storage,read> records:array<{params}>;
@group(0) @binding(1) var sample_linear:sampler;
@group(0) @binding(2) var input_image:texture_2d_array<f32>;
{aux_decl}
@group(0) @binding(4) var output_image:texture_storage_2d_array<{format},write>;
@compute @workgroup_size(8,8)
fn filter_layers(@builtin(global_invocation_id) gid:vec3u) {{
  if any(gid.xy >= textureDimensions(output_image)) {{ return; }}
  layer = gid.z;
  let pos = vec4f(vec2f(gid.xy)+vec2f(0.5),0.0,1.0);
  let color = shade_{entry}({position},records[layer],sample_linear,input_image{aux_arg});
  textureStore(output_image,vec2i(gid.xy),i32(layer),color);
}}"#,
        entry = entry
    )
}

fn direct_layers(chains: &[Vec<FilterDraw>]) -> Option<&(wgpu::TextureView, u32)> {
    let input = &chains[0][0].input;
    input.layer().filter(|(array, first)| {
        *first == 0 && chains.iter().enumerate().all(|(z, chain)| {
            chain[0].input.layer().is_some_and(|(view, layer)| view == array && *layer == z as u32)
        })
    })
}

/// Largest simultaneous set of physical images and parameter records used by
/// this array schedule. Slices of the same source count once, including layers
/// outside the chunk. Both the gathered input and its source exist during copy.
fn array_storage(chains: &[Vec<FilterDraw>], uses: &SurfaceUses, live: u64, mut upload: crate::scratch::UploadLayout, device: &wgpu::Device, scratch: &crate::scratch::ScratchPool) -> u64 {
    let first = &chains[0];
    let n = chains.len() as u64;
    let mut reads = HashMap::new();
    for chain in chains {
        let source = &chain[0].input;
        let key = &uses.resources[&std::rc::Rc::as_ptr(&source.0)];
        *reads.entry(key).or_insert(0) += 1 + chain.iter().filter(|d| d.second.is_some()).count();
    }
    let released: u64 = reads.into_iter().filter_map(|(key, count)| match key {
        SurfaceResource::Texture(t) if uses.remaining[key] == count && !uses.outputs.contains(key) && !uses.cached.contains(t)
            => Some(crate::scratch::texture_bytes(t)),
        _ => None,
    }).sum();
    let price = |w, h, layers, format, usage| scratch.texture_price(device,
        crate::scratch::TexKey::capture(w, h, layers, 1, format, usage));
    use wgpu::TextureUsages as U;
    let input = &first[0].input;
    let gather = if direct_layers(chains).is_some() { 0 } else {
        price(input.width(), input.height(), n as u32, input.format(), U::TEXTURE_BINDING | U::COPY_DST)
    };
    let uv_bytes = |w, h| price(w, h, 1, wgpu::TextureFormat::Rg32Float, U::RENDER_ATTACHMENT | U::TEXTURE_BINDING);
    let mut coords = HashMap::new();
    for (i, draw) in first.iter().enumerate() {
        if draw.pipeline.module == "backdrop-blur" { coords.insert((draw.target.width(),draw.target.height()), i); }
        upload.add(n * draw.parameters.len() as u64 * 4);
    }
    let mut coordinate_bytes: u64 = coords.keys().map(|&(w,h)| uv_bytes(w, h)).sum();
    let original_last = first.iter().rposition(|d| d.second.is_some()).unwrap_or(0);
    let mut previous = 0;
    let mut peak = live + gather;
    let stationary = live.saturating_sub(released);
    for (stage, draw) in first.iter().enumerate() {
        let target = price(draw.target.width(), draw.target.height(), n as u32, draw.pipeline.format,
            U::STORAGE_BINDING | U::TEXTURE_BINDING | U::COPY_SRC);
        let original = if stage > original_last { 0 } else if gather > 0 { gather } else { released };
        peak = peak.max(stationary + original + coordinate_bytes + previous + target);
        previous = target;
        let size = (draw.target.width(), draw.target.height());
        if coords.get(&size) == Some(&stage) { coordinate_bytes -= uv_bytes(size.0, size.1); }
    }
    let individual: u64 = chains.iter().map(|c| &c.last().unwrap().target)
        .filter(|s| s.individual()).map(FilterSurface::bytes).sum();
    peak.max(stationary + previous + individual) + upload.additional
}

fn record_array(
    chains: &[Vec<FilterDraw>],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scratch: &crate::scratch::ScratchPool,
    encoder: &mut wgpu::CommandEncoder,
    uses: &mut SurfaceUses,
) {
    use crate::scratch::Bound;
    let input = &chains[0][0].input;
    let (w, h, n) = (input.width(), input.height(), chains.len() as u32);
    let extent = wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: n,
    };
    let slice = wgpu::Extent3d {
        depth_or_array_layers: 1,
        ..extent
    };
    let array = |size, format, usage| {
        scratch.texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("capture array"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            },
        )
    };
    // When the source render passes resolved into consecutive layers in filter order,
    // the first dispatch can sample the source array directly. Otherwise gather just
    // the required layers using exact texture copies.
    let direct = direct_layers(chains);
    let original = if let Some((array, _)) = direct {
        array.clone()
    } else {
        let inputs = array(extent, input.format(), wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        for (z, chain) in chains.iter().enumerate() {
            let source = &chain[0].input;
            let mut from = source.view(device, scratch).texture().as_image_copy();
            if let Some((view, layer)) = source.layer() {
                from = view.texture().as_image_copy();
                from.origin.z = *layer;
            }
            let mut to = inputs.as_image_copy();
            to.origin.z = z as u32;
            encoder.copy_texture_to_texture(from, to, slice);
            // Every later read of the original in this chain samples the gather.
            // The backing source can return immediately after its final copy.
            uses.finish_surface(source, scratch);
            for second in chain.iter().filter_map(|draw| draw.second.as_ref()) {
                uses.finish_surface(second, scratch);
            }
        }
        scratch.view(&inputs)
    };
    let layer = |z| wgpu::Origin3d { x: 0, y: 0, z };
    let mut previous = original.clone();
    // Materialize UVs before opening the compute pass. Track their last use
    // alongside the original input, which shadow composites may read again.
    let mut coordinates = HashMap::new();
    for (stage, draw) in chains[0].iter().enumerate().filter(|(_, d)| d.pipeline.module == "backdrop-blur") {
        let (w, h) = (draw.target.width(), draw.target.height());
        coordinates.entry((w,h))
            .or_insert_with(|| (draw.pipeline.coordinates(device, scratch, encoder, w, h), stage)).1 = stage;
    }
    let original_last = chains[0].iter().rposition(|d| d.second.is_some()).unwrap_or(0);
    let ts = crate::gpu_profile::pass_timestamps();
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Layer array"), timestamp_writes: crate::gpu_profile::compute_writes(&ts),
    });
    for (stage, first) in chains[0].iter().enumerate() {
        let (w, h) = (first.target.width(), first.target.height());
        let size = wgpu::Extent3d { width: w, height: h, ..extent };
        let output = array(size, first.pipeline.format, wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let view = scratch.view(&output);
        #[cfg(test)]
        crate::device::VARIANT_DRAWS.with_borrow_mut(|draws| draws.insert("array filter"));
        let pipe = &first.pipeline.variants.array.ready().expect("selected published compute pipeline").0;
        let mut data = Vec::with_capacity(first.parameters.len() * chains.len());
        for chain in chains {
            data.extend_from_slice(&chain[stage].parameters);
        }
        let records = scratch.upload_range(
            device,
            queue,
            "layer array records",
            bytemuck::cast_slice(&data),
            wgpu::BufferUsages::STORAGE,
            false,
        );
        let (buf, offset, len) = match &records {
            Bound::Range(buf, offset, size) => (buf, *offset, Some(*size)),
            Bound::Buffer(buf) => (buf, 0, None),
            _ => unreachable!(),
        };
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf,
                    offset,
                    size: len,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&first.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&previous),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&view),
            },
        ];
        if first.pipeline.module == "backdrop-blur" {
            entries.push(wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&coordinates[&(w, h)].0),
            });
        }
        if first.second.is_some() {
            entries.push(wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&original),
            });
        }
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer array"),
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
        });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), n);
        if stage > 0 { scratch.recycle_attachment(previous.texture()); }
        if stage == original_last && direct.is_none() { scratch.recycle_attachment(original.texture()); }
        if let Some((coords, _)) = coordinates.get(&(w,h)).filter(|(_, last)| *last == stage) {
            scratch.recycle_attachment(coords.texture());
        }
        for chain in chains {
            let draw = &chain[stage];
            if stage > 0 || direct.is_some() { uses.finish_surface(&draw.input, scratch); }
            uses.finish_surface(&draw.target, scratch);
            if direct.is_some() {
                if let Some(second) = &draw.second { uses.finish_surface(second, scratch); }
            }
        }
        previous = view;
    }
    drop(pass);
    let mut result_uses_array = false;
    for (z, chain) in chains.iter().enumerate() {
        let surface = &chain.last().unwrap().target;
        if !surface.0.individual.get() && surface.0.view.get().is_none() {
            result_uses_array = true;
            surface.0.layer.set((previous.clone(), z as u32)).expect("unallocated layer");
            let slice_view = scratch.layer_view(previous.texture(), z as u32);
            surface.0.view.set(slice_view).expect("unallocated result");
            continue;
        }
        let target = surface.view(device, scratch);
        let mut src = previous.texture().as_image_copy();
        src.origin = layer(z as u32);
        encoder.copy_texture_to_texture(src, target.texture().as_image_copy(), wgpu::Extent3d {
            width: surface.width(), height: surface.height(), depth_or_array_layers: 1,
        });
    }
    if !result_uses_array { scratch.recycle_attachment(previous.texture()); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shadow_array_shader_variants_validate() {
        for (module, entry, aux) in [
            ("progressive-blur", "fs", false),
            ("backdrop-blur", "fs", false),
            ("layer-filter-shadow", "fs_prepare", false),
            ("layer-filter-shadow", "fs_composite", true),
            ("layer-filter-shadow", "fs_source", false),
            ("layer-filter-shadow", "fs_feather", true),
        ] {
            for format in [
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureFormat::Rgba16Float,
            ] {
                let source = moda_shaders::module_source(module).unwrap().source;
                let source = array_shader(module, format, source, aux, entry);
                let ir = wgpu::naga::front::wgsl::parse_str(&source)
                    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                )
                .validate(&ir)
                .unwrap_or_else(|e| panic!("{module} {entry}: {e:?}"));
            }
        }
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    #[test]
    fn a_later_use_extends_the_shared_backing() {
        let left = FilterSurface::new([4, 4], wgpu::TextureFormat::Rgba8Unorm);
        let right = FilterSurface::new([4, 4], wgpu::TextureFormat::Rgba8Unorm);
        let mut cost = CaptureFootprint::default();
        cost.extend(&left, 0, 1);
        cost.extend(&right, 0, 2);
        cost.share_layers(192, [&left, &right].into_iter());
        cost.extend(&right, 4, 5);
        let live: Vec<_> = cost.changes.iter().scan(0, |n, d| { *n += d; Some(*n) }).collect();
        assert_eq!(live, [192, 192, 192, 192, 192, 0]);
    }

    #[test]
    fn copied_slice_and_passthrough_share_only_their_actual_backings() {
        let shape = || FilterSurface::new([4, 4], wgpu::TextureFormat::Rgba8Unorm);
        let (pass, short, copied) = (shape(), shape(), shape());
        let mut plan = CaptureFootprint::default();
        plan.extend(&pass, 0, usize::MAX);
        plan.extend(&short, 0, 2);
        plan.extend(&copied, 0, 3);
        plan.extend(&pass, 0, 2);
        plan.share_layers(3 * 64, [&pass, &short].into_iter());
        let live: Vec<_> = plan.changes.iter().scan(0, |live, delta| { *live += delta; Some(*live) }).collect();
        assert_eq!(live, [256, 256, 256, 192]);
        assert_eq!(plan.demand(320).with_uniforms(plan.uniforms.additional), (320, 256));
    }
}
