use crate::{FrameOutput, FrameReadback, ReadbackOutputs};
use anyhow::{Result, anyhow};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{Arc, mpsc},
    thread,
};


/// Blender ImageFormatSettings.color_depth for PNG colour: 8 or 16 bits per component.
/// Both carry the same exposure and display transform; control channels keep their own depths.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PngDepth {
    #[default]
    Eight,
    Sixteen,
}
impl std::str::FromStr for PngDepth {
    type Err = String;
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "8" => Ok(Self::Eight),
            "16" => Ok(Self::Sixteen),
            _ => Err(vork_scene::diagnostics::unknown("png_depth", value, ["8", "16"])),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Output {
    Aov(String),
    Color,
    Depth,
    ObjectIndex,
    ViewNormal,
    Motion,
    Camera,
}
impl std::str::FromStr for Output {
    type Err = String;
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        if let Some(name)=value.strip_prefix("aov:") {
            // AOV layers share the EXR namespace with built-in channels.
            return if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-')) {Err("AOV output requires a name using ASCII letters, digits, _ or -".into())}
            else if vork_scene::v2::OUTPUT_CHANNELS.contains(&name) {Err(format!("AOV {name:?} collides with a built-in channel and EXR layer; rename the AOV"))}
            else {Ok(Self::Aov(name.to_owned()))};
        }
        Self::builtin(value).ok_or_else(|| {
            ::vork_scene::v2::diagnostics::unknown("output channel", value, vork_scene::v2::OUTPUT_CHANNELS.iter().copied())
        })
    }
}
impl Output {
    /// A built-in channel by its token; vork_scene::v2::OUTPUT_CHANNELS lists the tokens.
    fn builtin(name: &str) -> Option<Self> {
        Some(match name {
            "color" => Self::Color,
            "depth" => Self::Depth,
            "object_index" => Self::ObjectIndex,
            "view_normal" => Self::ViewNormal,
            "motion" => Self::Motion,
            "camera" => Self::Camera,
            "shadow_catcher" => Self::Aov("shadow_catcher".into()),
            _ => return None,
        })
    }
    /// Every built-in channel, the default selection before declared AOVs.
    pub fn builtins() -> impl Iterator<Item = Self> {
        vork_scene::v2::OUTPUT_CHANNELS.iter().map(|name| Self::builtin(name).expect("every listed channel token parses"))
    }
}
#[derive(Clone)]
pub struct Outputs(pub Vec<Output>);
impl Outputs {
    pub fn new(outputs: Vec<Output>) -> Self {
        let mut unique = Vec::new();
        for output in outputs {
            if !unique.contains(&output) {
                unique.push(output);
            }
        }
        Self(unique)
    }
    pub fn has(&self, output: Output) -> bool {
        self.0.contains(&output)
    }
    pub fn readback(&self, packed: bool, float: bool, png_depth: PngDepth) -> ReadbackOutputs {
        let color = packed && self.has(Output::Color);
        ReadbackOutputs {
            aovs: self.0.iter().any(|o|matches!(o,Output::Aov(_))),
            color: color && png_depth == PngDepth::Eight,
            color16: color && png_depth == PngDepth::Sixteen,
            depth_ids: (packed && self.has(Output::Depth)) || self.has(Output::ObjectIndex),
            normals: packed && self.has(Output::ViewNormal),
            motion: packed && self.has(Output::Motion),
            float: crate::FloatOutputs {
                color: float && self.has(Output::Color),
                depth: float && self.has(Output::Depth),
                normals: float && self.has(Output::ViewNormal),
                motion: float && self.has(Output::Motion),
            },
        }
    }
}
/// The render thread only packs the mapped rows. The first PNG worker
/// converts the owned bytes in place. Every destination reads that result.
pub(crate) struct CanvasPixels(std::sync::LazyLock<Vec<u8>, Box<dyn FnOnce() -> Vec<u8> + Send + Sync>>);

impl CanvasPixels {
    pub(crate) fn new(mut bytes: Vec<u8>, format: wgpu::TextureFormat) -> Self {
        Self(std::sync::LazyLock::new(Box::new(move || {
            if format == wgpu::TextureFormat::Bgra8Unorm {
                for texel in bytes.as_chunks_mut::<4>().0 { texel.swap(0, 2); }
            }
            bytes
        })))
    }

    fn rgba(&self) -> &[u8] { &self.0 }
}

enum Input {
    Canvas { frame: usize, rgba: Arc<CanvasPixels>, lease: Arc<crate::ring::Lease> },
    CanvasFloat { frame: usize, rgba: Arc<Vec<[f32; 4]>>, lease: Arc<crate::ring::Lease> },
    Frame {
        frame: usize,
        pixels: FrameReadback,
    },
    Unpacked {
        frame: usize,
        pixels: Result<Arc<FrameOutput>>,
        lease: Option<Arc<crate::ring::Lease>>,
    },
    /// A refused frame: its output orders complete with no files.
    Skip {
        frame: usize,
    },
    Finish,
}
enum Job {
    Canvas { order: usize, frame: usize, rgba: Arc<CanvasPixels>, lease: Arc<crate::ring::Lease> },
    CanvasFloat { order: usize, frame: usize, rgba: Arc<Vec<[f32; 4]>>, lease: Arc<crate::ring::Lease> },
    Unpack {
        frame: usize,
        pixels: FrameReadback,
    },
    Plane {
        order: usize,
        frame: usize,
        outputs: Outputs,
        exr: bool,
        pixels: Arc<FrameOutput>,
        lease: Option<Arc<crate::ring::Lease>>,
    },
}
type Files = Vec<(String, Vec<u8>)>;
/// Whether a writer's destination is one still: a `.png` path. Anything else is a directory of frames.
fn one_still(out: &std::path::Path) -> bool {
    !crate::shot::directory_marked(out) && out.extension().is_some_and(|e| e.eq_ignore_ascii_case("png"))
}
/// Every PNG vork writes uses the fast setting: canvas or world, still or sequence frame, colour or control
/// channel. One policy, no split by kind: the smaller file the default setting gives costs far more time than
/// it saves bytes (about 1.8 s on a 4K still). Lossless either way.
const COMPRESSION: png::Compression = png::Compression::Fast;
fn encode_canvas(w: u32, h: u32, rgba: &[u8], display: vork_scene::v2::OutputDisplay) -> Result<Files> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, w, h);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(COMPRESSION);
        // The picture names its display, as a world's does.
        let cicp = cicp(display);
        if cicp.is_none() { encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual); }
        let mut writer = encoder.write_header()?;
        if let Some(cicp) = cicp { writer.write_chunk(png::chunk::ChunkType(*b"cICP"), &cicp)?; }
        writer.write_image_data(rgba)?;
    }
    Ok(vec![(".png".to_owned(), bytes)])
}
/// The cICP chunk (PNG Third Edition) naming a colour picture's display: H.273 primaries and transfer, RGB,
/// full range. An sRGB picture carries the sRGB chunk instead.
fn cicp(display: vork_scene::v2::OutputDisplay) -> Option<[u8; 4]> {
    use vork_scene::v2::OutputDisplay::*;
    match display {
        Srgb => None,
        DisplayP3 => Some([12, 13, 0, 1]),
        Rec2100Pq => Some([9, 16, 0, 1]),
        Rec2100Hlg => Some([9, 18, 0, 1]),
    }
}
fn encode(w: u32, h: u32, p: &FrameOutput, outputs: &Outputs, png_depth: PngDepth, display: vork_scene::v2::OutputDisplay) -> Result<Files> {
    use png::{
        BitDepth::{Eight, Sixteen},
        ColorType::{Grayscale, Rgb, Rgba},
    };
    let mut files = Vec::new();
    let mut png = |suffix: &str, color, depth, data: &[u8]| -> Result<()> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, w, h);
            encoder.set_color(color);
            encoder.set_depth(depth);
            encoder.set_compression(COMPRESSION);
            // The colour picture names its display; control channels are data.
            let cicp = if suffix == ".png" { cicp(display) } else { None };
            if suffix == ".png" && cicp.is_none() {encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);}
            let mut writer = encoder.write_header()?;
            if let Some(cicp) = cicp { writer.write_chunk(png::chunk::ChunkType(*b"cICP"), &cicp)?; }
            writer.write_image_data(data)?;
        }
        files.push((suffix.to_owned(), bytes));
        Ok(())
    };
    if outputs.has(Output::Color) {
        match png_depth {
            PngDepth::Eight => png(".png", Rgba, Eight, &p.rgba)?,
            PngDepth::Sixteen => {
                let words: Vec<_> = p.rgba16.iter().map(|v| v.map(u16::to_be)).collect();
                png(".png", Rgba, Sixteen, bytemuck::cast_slice(&words))?
            }
        }
    }
    // Keep one output item per pixel. Flattening byte arrays through nested
    // iterators left a scalar Iterator::next state machine on this hot path;
    // collecting fixed-size records lets LLVM vectorize the endian conversion.
    // PNG stores 16-bit components big-endian on every host.
    if outputs.has(Output::Depth) {
        let words: Vec<_> = p.depth_u16.iter().map(|v| v.to_be()).collect();
        png(".depth.png", Grayscale, Sixteen, bytemuck::cast_slice(&words))?;
    }
    if outputs.has(Output::ObjectIndex) {
        let words: Vec<_> = p.ids.iter().map(|v| v.to_be()).collect();
        png(".object_index.png", Grayscale, Sixteen, bytemuck::cast_slice(&words))?;
    }
    if outputs.has(Output::ViewNormal) {
        png(".view_normal.png", Rgb, Eight, &p.normals)?;
    }
    if outputs.has(Output::Motion) {
        let words: Vec<_> = p.motion.iter().map(|v| [v[0].to_be(), v[1].to_be(), 0u16]).collect();
        png(".motion.png", Rgb, Sixteen, bytemuck::cast_slice(&words))?;
    }
    for output in &outputs.0 {
        if let Output::Aov(name)=output {
            let aov=p.aovs.get(name).ok_or_else(||anyhow!("missing AOV {name}"))?;
            let components=if aov.kind==vork_scene::v2::AovType::Value {1} else {3};
            let bytes:Vec<_>=aov.pixels.iter().flat_map(|v|v[..components].iter().flat_map(|c|((c.clamp(0.,1.)*65535.).round() as u16).to_be_bytes())).collect();
            png(&if name=="shadow_catcher" {".shadow_catcher.png".into()} else {format!(".aov.{name}.png")},if components==1 {Grayscale} else {Rgb},Sixteen,&bytes)?;
        }
    }
    Ok(files)
}
fn float_channels<'a, const N: usize>(
    names: [&'a str; N],
    values: &Option<Vec<[f32; N]>>,
) -> Result<Vec<(&'a str, Vec<f32>)>> {
    let values = values
        .as_ref()
        .ok_or_else(|| anyhow!("EXR channels require float readback"))?;
    Ok(names
        .into_iter()
        .enumerate()
        .map(|(c, name)| (name, values.iter().map(|v| v[c]).collect()))
        .collect())
}

/// OpenEXR stores linear HDR and unquantized control signals; RGBA is premultiplied.
#[cfg(test)]
fn encode_exr(w: u32, h: u32, p: &FrameOutput, outputs: &Outputs, codec: crate::ExrCodec) -> Result<Files> {
    encode_exr_clock(w,h,p,outputs,codec,None,Default::default(),1)
}
/// A picture's associated RGBA as the default layer's R, G, B and A channels, its colour in the storage space.
fn beauty_channels(pixels: &[[f32; 4]], color: crate::output_device::RawColor) -> Vec<(&'static str, Vec<f32>)> {
    let mut channels: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(pixels.len()));
    for &pixel in pixels {
        for (channel, value) in channels.iter_mut().zip(color.store_associated(pixel)) { channel.push(value); }
    }
    ["R", "G", "B", "A"].into_iter().zip(channels).collect()
}
/// A canvas's EXR frame: its picture after its view (associated RGBA, sRGB code values), through the
/// writer a world's frame takes.
fn encode_exr_canvas(w: u32, h: u32, rgba: &[[f32; 4]], codec: crate::ExrCodec, fps: (i32, u32), color: crate::output_device::RawColor, zip_batch: usize) -> Result<Files> {
    anyhow::ensure!(rgba.len() == (w * h) as usize, "EXR channel size mismatch");
    let channels = beauty_channels(rgba, color).into_iter().map(|(channel, values)|
        exr::prelude::AnyChannel::new(exr::meta::attribute::Text::from_slice_unchecked(channel.as_bytes()), exr::prelude::FlatSamples::F32(values))).collect();
    write_exr(w, h, channels, codec, Some(fps), color, zip_batch)
}
/// `color`: the storage space of the file's colour (beauty and colour AOVs), which the file names in its
/// chromaticities and `vork:color_space`. Alpha, depth, indices, normals, motion and value AOVs are not colour.
fn encode_exr_clock(w: u32, h: u32, p: &FrameOutput, outputs: &Outputs, codec: crate::ExrCodec, fps: Option<(i32,u32)>, color: crate::output_device::RawColor, zip_batch: usize) -> Result<Files> {
    use exr::prelude::*;
    let mut all_channels = Vec::new();
    let count = (w * h) as usize;
    for output in &outputs.0 {
        // Plain OpenEXR: beauty R,G,B,A and depth Z in the default layer; every other
        // channel is <name>.<channel>.
        let (name, channels): (&str, Vec<(&str, Vec<f32>)>) = match output {
            Output::Color => {
                let pixels = p.linear_rgba.as_ref().ok_or_else(|| anyhow!("EXR channels require float readback"))?;
                ("", beauty_channels(pixels, color))
            },
            Output::Depth => (
                "",
                vec![(
                    "Z",
                    p.depth_f32
                        .as_ref()
                        .ok_or_else(|| anyhow!("EXR depth requires float readback"))?
                        .clone(),
                )],
            ),
            Output::ObjectIndex => ("object_index", vec![("X", p.ids.iter().map(|v| *v as f32).collect())]),
            Output::ViewNormal => ("view_normal", float_channels(["X", "Y", "Z"], &p.normals_f32)?),
            Output::Motion => ("motion", float_channels(["X", "Y"], &p.motion_f32)?),
            Output::Camera => continue,
            Output::Aov(name) => {
                let aov=p.aovs.get(name).ok_or_else(||anyhow!("missing AOV {name}"))?;
                let labels:&[&str]=if aov.kind==vork_scene::v2::AovType::Value {&["X"]} else {&["R","G","B"]};
                // shadow_catcher is a ratio of radiances, not a radiance: it stays in the working space.
                let stored=aov.kind==vork_scene::v2::AovType::Color && name!="shadow_catcher";
                (name.as_str(),labels.iter().enumerate().map(|(i,label)|(*label,aov.pixels.iter().map(|v|if stored {color.store([v[0],v[1],v[2]])[i]} else {v[i]}).collect())).collect())
            },
        };
        anyhow::ensure!(
            channels.iter().all(|(_, v)| v.len() == count),
            "EXR channel size mismatch"
        );
        all_channels.extend(channels.into_iter().map(|(channel, values)| {
            AnyChannel::new(
                exr::meta::attribute::Text::from_slice_unchecked(if name.is_empty() { channel.to_owned() } else { format!("{name}.{channel}") }.as_bytes()),
                FlatSamples::F32(values),
            )
        }));
    }
    write_exr(w, h, all_channels, codec, fps, color, zip_batch)
}
/// The one EXR file writer: the channels as one layer, with the names of the file's colour.
fn write_exr(w: u32, h: u32, all_channels: Vec<exr::prelude::AnyChannel<exr::prelude::FlatSamples>>, codec: crate::ExrCodec, fps: Option<(i32,u32)>,
    color: crate::output_device::RawColor, zip_batch: usize) -> Result<Files> {
    use exr::prelude::*;
    let mut files = Vec::new();
    if !all_channels.is_empty() {
        let mut image = Image::from_layer(Layer::new(
            (w as usize, h as usize),
            LayerAttributes {frames_per_second:fps,..Default::default()},
            codec.encoding(),
            AnyChannels::sort(all_channels.into()),
        ));
        // The file says what its colour is: the storage space's primaries and white, its name, and the
        // picture's reference domain: scene (a world's raw picture bypasses the view) or display (a canvas's
        // picture after its view).
        let [red, green, blue, white] = color.storage.chromaticities().map(|[x, y]| Vec2(x, y));
        image.attributes.chromaticities = Some(exr::meta::attribute::Chromaticities { red, green, blue, white });
        image.layer_data.attributes.other.insert(Text::from("vork:color_space"), AttributeValue::Text(Text::from(color.storage.name())));
        image.layer_data.attributes.other.insert(Text::from("vork:reference_domain"), AttributeValue::Text(Text::from(color.reference_domain())));
        let mut bytes = std::io::Cursor::new(Vec::new());
        if zip_batch > 1 {
            write_zip_blocks(&image, &mut bytes, zip_batch)?;
        } else {
            image.write().non_parallel().to_buffered(&mut bytes)?;
        }
        files.push((".exr".into(), bytes.into_inner()));
    }
    Ok(files)
}

type ExrImage = exr::image::Image<exr::image::Layer<exr::image::AnyChannels<exr::image::FlatSamples>>>;

/// Feed one EXR chunk writer in line order. A batch bounds scratch per frame.
fn write_zip_blocks(
    image: &ExrImage,
    bytes: &mut std::io::Cursor<Vec<u8>>,
    batch_size: usize,
) -> exr::error::UnitResult {
    use exr::block::{UncompressedBlock, writer::ChunksWriter};
    use exr::image::write::{
        WritableImage,
        layers::{LayersWriter, WritableLayers},
    };

    let headers = image.write().infer_meta_data();
    let layers = image.layer_data.create_writer(&headers);
    exr::block::write(bytes, headers, true, |meta, writer| {
        let headers = &meta.headers;
        let mut indices = meta.enumerate_ordered_header_block_indices();
        let mut batch = Vec::with_capacity(batch_size);
        loop {
            batch.extend(indices.by_ref().take(batch_size));
            if batch.is_empty() {
                return Ok(());
            }
            // Indexed collection keeps scanline order despite out-of-order completion.
            let chunks: Vec<_> = batch
                .par_iter()
                .map(|&(index, block)| {
                    let data = layers.extract_uncompressed_block(headers, block);
                    let chunk = UncompressedBlock { index: block, data }.compress_to_chunk(headers)?;
                    Ok::<_, exr::error::Error>((index, chunk))
                })
                .collect();
            for chunk in chunks {
                let (index, chunk) = chunk?;
                writer.write_chunk(index, chunk)?;
            }
            batch.clear();
        }
    })
}

/// One core-sized pool unpacks readbacks and compresses individual planes.
/// Ring leases bound mapped pixels, unpacked pixels, and ordered results.
pub struct FrameOutputWriter {
    jobs: mpsc::Sender<Input>,
    coordinator: thread::JoinHandle<Result<()>>,
    writer: thread::JoinHandle<Result<()>>,
    pub worker_count: usize,
}
impl FrameOutputWriter {
    pub(crate) fn canvas(out: PathBuf, w: u32, h: u32, pending_frames: usize, first_frame: usize, display: vork_scene::v2::OutputDisplay) -> Self {
        Self::new(out, w, h, Outputs(vec![Output::Color]), false, crate::ExrCodec::default(),
            PngDepth::Eight, pending_frames, first_frame, (1, 1), display, Default::default())
    }
    pub(crate) fn push_canvas(&self, frame: usize, rgba: Arc<CanvasPixels>, lease: Arc<crate::ring::Lease>) -> Result<()> {
        self.jobs.send(Input::Canvas { frame, rgba, lease })
            .map_err(|_| anyhow!("frame output workers stopped"))
    }
    /// A canvas's EXR destination (`x.%06d.exr`): one file per frame, its colour stored as `raw` says.
    pub(crate) fn canvas_exr(out: PathBuf, w: u32, h: u32, exr_codec: crate::ExrCodec, pending_frames: usize, first_frame: usize, fps: (i32, u32),
        raw: crate::output_device::RawColor) -> Self {
        Self::new(out, w, h, Outputs(vec![Output::Color]), true, exr_codec, PngDepth::Eight, pending_frames, first_frame, fps,
            vork_scene::v2::OutputDisplay::Srgb, raw)
    }
    /// One frame of a canvas's picture after its view: associated RGBA, sRGB code values.
    pub(crate) fn push_canvas_float(&self, frame: usize, rgba: Arc<Vec<[f32; 4]>>, lease: Arc<crate::ring::Lease>) -> Result<()> {
        self.jobs.send(Input::CanvasFloat { frame, rgba, lease })
            .map_err(|_| anyhow!("frame output workers stopped"))
    }
    pub fn new(
        out: PathBuf,
        w: u32,
        h: u32,
        outputs: Outputs,
        exr: bool,
        exr_codec: crate::ExrCodec,
        png_depth: PngDepth,
        pending_frames: usize,
        // Absolute index of the first pushed frame; output order counts from it.
        first_frame: usize,
        fps: (i32,u32),
        // The display colour PNGs are encoded for, which they name.
        display: vork_scene::v2::OutputDisplay,
        // The storage space of an EXR's colour, which it names.
        raw: crate::output_device::RawColor,
    ) -> Self {
        let plane_count = if exr { 1 } else { outputs.0.len() };
        // Only sufficiently large ZIP frames benefit from spending multiple
        // workers on one frame. Keep the batch bounded while frames overlap.
        let zip_blocks = if exr
            && exr_codec == crate::ExrCodec::Zip
            && u64::from(w) * u64::from(h) >= 256 * 1024
        {
            (h as usize).div_ceil(exr::prelude::Compression::ZIP16.scan_lines_per_block())
        } else {
            1
        };
        let zip_workers = zip_blocks.min(8);
        let worker_count = thread::available_parallelism()
            .map_or(1, usize::from)
            .min((pending_frames.max(1) * plane_count.max(1)).max(zip_workers));
        // A full ring can keep workers busy across frames. Spend extra
        // workers within a frame only when the ring has fewer frame credits.
        let workers_per_frame = (worker_count / pending_frames.max(1)).max(1);
        let zip_batch = zip_workers.min(workers_per_frame);

        let (requests, input) = mpsc::channel::<Input>();
        let (done, results) = mpsc::channel();
        let run_job = {
            let requests = requests.clone();
            let done = done.clone();
            move |job| {
                match job {
                    Job::Canvas { order, frame, rgba, lease } => {
                        let started = std::time::Instant::now();
                        let files = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encode_canvas(w, h, rgba.rgba(), display)))
                            .unwrap_or_else(|_| Err(anyhow!("frame compression worker panicked")));
                        crate::timing::output_interval(crate::timing::OutputStage::ImageEncode, started);
                        let _ = done.send((order, frame, files, Some(lease)));
                    }
                    Job::CanvasFloat { order, frame, rgba, lease } => {
                        let started = std::time::Instant::now();
                        let files = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encode_exr_canvas(w, h, &rgba, exr_codec, fps, raw, zip_batch)))
                            .unwrap_or_else(|_| Err(anyhow!("frame compression worker panicked")));
                        crate::timing::output_interval(crate::timing::OutputStage::ImageEncode, started);
                        let _ = done.send((order, frame, files, Some(lease)));
                    }
                    Job::Unpack { frame, pixels } => {
                        let lease = pixels.credit.clone();
                        let started = std::time::Instant::now();
                        let pixels =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                pixels.unpack()
                            }))
                            .map_err(|_| anyhow!("readback unpack worker panicked"))
                            .and_then(|result| result).map(Arc::new);
                        crate::timing::encoded(started);
                        let _ = requests.send(Input::Unpacked { frame, pixels, lease });
                    }
                    Job::Plane {
                        order,
                        frame,
                        outputs,
                        exr,
                        pixels,
                        lease,
                    } => {
                        let started = std::time::Instant::now();
                        let files =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                if exr {
                                    encode_exr_clock(w, h, &pixels, &outputs, exr_codec,Some(fps),raw,zip_batch)
                                } else {
                                    encode(w, h, &pixels, &outputs, png_depth, display)
                                }
                            }))
                            .unwrap_or_else(|_| {
                                Err(anyhow!("frame compression worker panicked"))
                            });
                        crate::timing::output_interval(crate::timing::OutputStage::ImageEncode, started);
                        let _ = done.send((order, frame, files, lease));
                    }
                }
            }
        };
        let coordinator = thread::spawn(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(worker_count)
                .thread_name(|index| format!("vork-output-{index}"))
                .build()?;
            // The coordinator waits outside the pool. Frame jobs and their
            // nested block jobs all run on the same workers.
            pool.in_place_scope_fifo(|scope| {
                let run_job = &run_job;
                let queue = |job| scope.spawn_fifo(move |_| run_job(job));
                let mut unpacking = 0;
                let mut finished = false;
                for request in input {
                    match request {
                        Input::Canvas { frame, rgba, lease } => {
                            queue(Job::Canvas { order: frame - first_frame, frame, rgba, lease });
                        }
                        Input::CanvasFloat { frame, rgba, lease } => {
                            queue(Job::CanvasFloat { order: frame - first_frame, frame, rgba, lease });
                        }
                        Input::Frame { frame, pixels } => {
                            unpacking += 1;
                            queue(Job::Unpack { frame, pixels });
                        }
                        Input::Unpacked {
                            frame,
                            pixels,
                            lease,
                        } => {
                            unpacking -= 1;
                            let groups = if exr {
                                vec![outputs.clone()]
                            } else {
                                outputs.0.iter().map(|o| Outputs(vec![o.clone()])).collect()
                            };
                            for (i, outputs) in groups.into_iter().enumerate() {
                                let order = (frame - first_frame) * plane_count + i;
                                match &pixels {
                                    Ok(pixels) => queue(Job::Plane {
                                        order,
                                        frame,
                                        outputs,
                                        exr,
                                        pixels: pixels.clone(),
                                        lease: lease.clone(),
                                    }),
                                    Err(error) => done.send((
                                        order,
                                        frame,
                                        Err(anyhow!(error.to_string())),
                                        lease.clone(),
                                    ))?,
                                }
                            }
                        }
                        Input::Skip { frame } => {
                            for i in 0..plane_count {
                                done.send(((frame - first_frame) * plane_count + i, frame, Ok(Files::new()), None))?;
                            }
                        }
                        Input::Finish => finished = true,
                    }
                    if finished && unpacking == 0 {
                        break;
                    }
                }
                Ok(())
            })
        });
        let writer = thread::spawn(move || {
            let mut pending = BTreeMap::new();
            let mut next = 0;
            let mut error = None;
            for (order, frame, files, lease) in results {
                pending.insert(order, (frame, files, lease));
                while let Some((frame, files, _lease)) = pending.remove(&next) {
                    let started = std::time::Instant::now();
                    let written = files.and_then(|files| {
                        for (suffix, bytes) in files {
                            let path = if exr {
                                let stem = out.file_stem().unwrap().to_string_lossy();
                                let stem = stem.replace("%06d", &format!("{frame:06}"));
                                out.with_file_name(format!("{stem}{suffix}"))
                            } else if one_still(&out) {
                                // A one-still PNG destination: colour at the path, other channels beside it.
                                if suffix == ".png" {
                                    out.clone()
                                } else {
                                    let stem = out.file_stem().unwrap().to_string_lossy();
                                    out.with_file_name(format!("{stem}{suffix}"))
                                }
                            } else {
                                out.join(format!("{frame:06}{suffix}"))
                            };
                            std::fs::write(path, bytes)?;
                        }
                        Ok(())
                    });
                    crate::timing::output_interval(crate::timing::OutputStage::ImageWrite, started);
                    if let Err(e) = written {
                        if error.is_none() {
                            error = Some(e);
                        }
                    }
                    next += 1;
                }
            }
            if !pending.is_empty() {
                return Err(anyhow!("frame output worker left a gap in output order"));
            }
            error.map_or(Ok(()), Err)
        });
        Self {
            jobs: requests,
            coordinator,
            writer,
            worker_count,
        }
    }
    pub fn push(&self, frame: usize, pixels: FrameReadback) -> Result<()> {
        self.jobs
            .send(Input::Frame { frame, pixels })
            .map_err(|_| anyhow!("frame output workers stopped"))
    }
    /// Write nothing for a refused frame, keeping later frames in order.
    pub fn skip(&self, frame: usize) -> Result<()> {
        self.jobs
            .send(Input::Skip { frame })
            .map_err(|_| anyhow!("frame output workers stopped"))
    }
    pub fn finish(self) -> Result<()> {
        let _ = self.jobs.send(Input::Finish);
        drop(self.jobs);
        let coordinated = self
            .coordinator
            .join()
            .map_err(|_| anyhow!("frame output coordinator panicked"));
        let result = self
            .writer
            .join()
            .map_err(|_| anyhow!("frame output writer panicked"))?;
        coordinated??;
        result
    }
}

#[cfg(test)]
#[path = "png_output_exr_tests.rs"]
mod exr_tests;

#[cfg(test)]
#[path = "png_output_canvas_writer_tests.rs"]
mod canvas_writer_tests;
