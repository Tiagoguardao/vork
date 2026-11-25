//! Display-frame media selection, adapted from an earlier portable
//! decoder: timestamp/keyframe lookup, sequential raw-frame pipe, indexed backwards
//! seeks. No clip is decoded into an unbounded in-memory frame collection.
use anyhow::{Context, Result, ensure};
use vork_time::exact::{Germ, Rational, decimal};
use std::{path::{Path, PathBuf}, process::Command, sync::Arc};

pub fn is_media(path: &Path) -> bool {
    is_sequence(path) || vork_scene::v2::is_movie_path(path)
}

pub(crate) fn is_sequence(path:&Path)->bool {
    path.file_name().and_then(|n|n.to_str()).is_some_and(|name| {
        name.match_indices('%').any(|(at,_)| {
            let tail=&name[at+1..];
            tail.starts_with('d') || tail.strip_prefix('0').is_some_and(|digits| {
                let count=digits.bytes().take_while(u8::is_ascii_digit).count();
                count>0 && digits[count..].starts_with('d')
            })
        })
    })
}

pub enum Source {
    Sequence { files:Vec<PathBuf>, fps:Option<vork_time::Time> },
    Video(Video),
}

type PathMap<T> = Arc<std::sync::Mutex<std::collections::BTreeMap<PathBuf, T>>>;

/// The media sources one document open has read, by path. The open owns it: every clock
/// plan it compiles and every image it installs takes a source from here, so a source is
/// read from disk once per open (a sequence's listing and first frame's rate, a movie's
/// probe) and all of them agree on what was read. It is dropped with the open; nothing
/// about a source is remembered for the next one.
#[derive(Clone, Default)]
pub struct Opened {
    sources: PathMap<Source>,
    headers: PathMap<Arc<MovieHeader>>,
    sequence_names: PathMap<PathBuf>,
    soundtracks: PathMap<Option<Arc<vork_audio::Source>>>,
    revisions: PathMap<Arc<vork_media::MediaRevision>>,
}
impl Opened {
    /// Keep the name bindings for a published revision without retaining
    /// picture decoder processes or frame pages from its validation.
    pub(crate) fn pins(&self) -> Self {
        Self { revisions: self.revisions.clone(), soundtracks: self.soundtracks.clone(), ..Default::default() }
    }
    pub(crate) fn revision(&self, path: &Path) -> Result<Arc<vork_media::MediaRevision>> {
        let key = std::path::absolute(path)?;
        let mut revisions = self.revisions.lock().unwrap();
        if let Some(known) = revisions.get(&key) { return Ok(known.clone()); }
        let target = path.canonicalize().with_context(|| format!("open media {}", path.display()))?;
        let revision = match revisions.get(&target) {
            Some(known) => known.clone(),
            None => vork_media::MediaRevision::open(&target)?,
        };
        revisions.insert(target, revision.clone());
        revisions.insert(key, revision.clone());
        Ok(revision)
    }
    pub(crate) fn optional_audio(&self, path: &Path) -> Result<Option<Arc<vork_audio::Source>>> {
        let revision = self.revision(path)?;
        let mut sounds = self.soundtracks.lock().unwrap();
        if let Some(source) = sounds.get(revision.original()) { return Ok(source.clone()); }
        let key = revision.original().to_owned();
        let source = vork_audio::Source::open_revision(revision)?.map(Arc::new);
        sounds.insert(key, source.clone());
        Ok(source)
    }
    pub(crate) fn audio(&self, path: &Path) -> Result<Arc<vork_audio::Source>> {
        self.optional_audio(path)?.context("audio: file has no audio stream")
    }
    pub(crate) fn held(&self, path: &Path) -> Result<Option<Arc<vork_media::MediaRevision>>> {
        Ok(self.revisions.lock().unwrap().get(&std::path::absolute(path)?).cloned())
    }
    pub(crate) fn file_version(&self, path: &Path) -> Result<crate::asset_revision::FileVersion> {
        let Some(revision) = self.held(path)? else {
            return Ok(crate::asset_revision::FileVersion::read(path)?);
        };
        let mut hash = blake3::Hasher::new();
        hash.update_reader(revision.reader()?)?;
        Ok(crate::asset_revision::FileVersion { target: revision.original().to_owned(), content: hash.finalize() })
    }
    fn movie_header(&self, revision: &vork_media::MediaRevision) -> Result<Arc<MovieHeader>> {
        let mut headers = self.headers.lock().unwrap();
        if let Some(header) = headers.get(revision.original()) { return Ok(header.clone()); }
        let header = Arc::new(MovieHeader::read(revision)?);
        if !header.unresolved_aperture() { headers.insert(revision.original().into(), header.clone()); }
        Ok(header)
    }
    pub(super) fn movie_info(&self, path: &Path) -> Result<serde_json::Value> {
        let revision = self.revision(path)?;
        self.movie_header(&revision)?.info()
    }
    pub(super) fn read<T>(&self, path: &Path, with: impl FnOnce(&Source) -> Result<T>) -> Result<T> {
        let revision = if is_sequence(path) { None } else { Some(self.revision(path)?) };
        let key = match &revision {
            Some(pin) => pin.original().to_owned(),
            None => {
                let spelling = std::path::absolute(path)?;
                let mut names = self.sequence_names.lock().unwrap();
                if let Some(key) = names.get(&spelling) { key.clone() } else {
                    let parent = spelling.parent().context("sequence needs a directory")?.canonicalize()?;
                    let key = parent.join(spelling.file_name().context("sequence needs a pattern")?);
                    names.insert(spelling, key.clone());
                    key
                }
            }
        };
        let mut sources = self.sources.lock().unwrap();
        if !sources.contains_key(&key) {
            let source = if let Some(revision) = revision {
                let remedy = || format!("movie {}; restore a playable movie and install ffmpeg/ffprobe on PATH", path.display());
                let header = self.movie_header(&revision).with_context(remedy)?;
                let source = Source::Video(Video::from_header(revision, &header).with_context(remedy)?);
                // Only this reader uses the fallback. Let the next reader measure it again.
                if header.unresolved_aperture() { return with(&source); }
                source
            } else { Source::open(&key)? };
            sources.insert(key.clone(), source);
        }
        with(&sources[&key])
    }
    /// The source at `path` as this open read it, with decoding state of its own.
    pub(crate) fn source(&self, path: &Path) -> Result<Source> {
        self.read(path, |source| Ok(source.fresh()))
    }
    /// Canvas paints and compositor inputs both enter the working-space conversion as SDR sRGB.
    pub(crate) fn canvas_source(&self, path: &Path, explicit: Option<vork_scene::v2::ColorSpace>) -> Result<Source> {
        self.read(path, |source| Ok(source.canvas_copy(source.canvas_maps_hdr(explicit))))
    }
    pub(crate) fn picture_span(&self, path: &Path, fps: vork_time::Time) -> Result<PictureSpan> {
        self.read(path, |source| match source {
            Source::Video(video) => Ok(video.picture_span(fps)),
            Source::Sequence {..} => anyhow::bail!("a sequence has no soundtrack"),
        })
    }
    /// That source's own length ([`Source::length`]).
    pub(crate) fn length(&self, path: &Path, fps: vork_time::Time) -> Result<Rational> {
        self.read(path, |source| source.length(fps))
    }
}

impl Source {
    /// What was read from disk, without this value's decoding state.
    pub(super) fn fresh(&self) -> Self {
        match self {
            Self::Sequence {files,fps} => Self::Sequence {files:files.clone(),fps:*fps},
            Self::Video(video) => Self::Video(video.fresh()),
        }
    }
    pub(super) fn canvas_maps_hdr(&self, explicit: Option<vork_scene::v2::ColorSpace>) -> bool {
        explicit.is_none() && matches!(self, Self::Video(video) if video.is_hdr())
    }
    pub(super) fn canvas_copy(&self, map_hdr: bool) -> Self {
        let mut source = self.fresh();
        if let Self::Video(video) = &mut source { video.map_hdr = map_hdr; }
        source
    }
    /// Files owned by this opened source, including every discovered sequence frame.
    pub fn files(&self) -> &[PathBuf] {
        match self {
            Self::Sequence {files,..} => files,
            Self::Video(video) => std::slice::from_ref(&video.path),
        }
    }

    pub fn open(path: &Path) -> Result<Self> {
        if is_sequence(path) {
            let files=sequence(path)?;
            let fps=if files[0].extension().is_some_and(|e|e.eq_ignore_ascii_case("exr")) {
                exr::meta::MetaData::read_from_file(&files[0],false)?.headers[0].own_attributes.frames_per_second
                    .map(|(n,d)|vork_time::Time::new(n.into(),d.into())).transpose().context("EXR sequence framesPerSecond must be positive")?
            } else {None};
            ensure!(fps.is_none_or(|fps|fps > vork_time::Time::ZERO),"EXR sequence framesPerSecond must be positive");
            return Ok(Self::Sequence {files,fps});
        }
        Ok(Self::Video(Video::open(path).with_context(||format!("movie {}; restore a playable movie and install ffmpeg/ffprobe on PATH",path.display()))?))
    }
    /// Natural texel size, without decoding a frame before the device's limits are known.
    pub(crate) fn dimensions(&self) -> Result<[u32; 2]> {
        match self {
            Self::Video(video) => Ok(video.size),
            Self::Sequence { files, .. } => {
                let path = &files[0];
                let (width, height, _) = crate::texture_source::header(image::ImageFormat::from_path(path)?, &std::fs::read(path)?, u32::MAX)?;
                Ok([width, height])
            }
        }
    }
    pub(crate) fn prepare_gpu(&mut self, r: &crate::Renderer) {
        if let Self::Video(video)=self {
            // A Linux FFmpeg subprocess cannot inspect or retry an NVDEC reference.
            // Only the in-process reader may enable NVIDIA decode on this platform.
            let hardware = !cfg!(target_os = "linux") && video.alpha_decoder.is_none() && video.container_crop.is_none()
                && r.adapter_info.vendor == crate::constants::NVIDIA_VENDOR_ID;
            if hardware != video.hardware { video.session = None; }
            video.hardware = hardware;
            #[cfg(vork_nvdecode)]
            {
                let uuid = unsafe { r.device.as_hal::<wgpu::hal::api::Vulkan>() }
                    .and_then(|hal| crate::footage_nvidia::device_uuid(&hal).ok());
                if uuid != video.nvidia {
                    video.deep = None;
                    video.nvidia = uuid;
                }
            }
        }
    }
    /// The content's own length in seconds: frames over the rate for a sequence; for a
    /// movie the stream's declared duration, else the displayed span plus one mean frame.
    pub(crate) fn length(&self,fps:vork_time::Time)->Result<Rational> {
        match self {
            Self::Sequence {files,fps:source_fps} => {
                let rate = source_fps.unwrap_or(fps).exact();
                Ok(Rational::from_integer(files.len().into())/rate)
            }
            Self::Video(video) => {
                if let Some(duration)=&video.duration {return Ok(duration.clone());}
                Ok(video.picture_span(fps).through(None)?.0)
            }
        }
    }
    pub fn timeline(&self,fps:vork_time::Time)->Result<Vec<vork_time::Clock>> {
        self.map_timeline(fps, |time| Ok(time.to_string().parse()?))
    }
    /// All identity consumers, including timeline generation, use these same
    /// boundaries. The decoder receives the selected integer, never a rounded time.
    pub(crate) fn frame_index_exact(&self, time:&Germ, fps:vork_time::Time)->Result<usize> {
        match self {
            Self::Sequence {files,fps:source_fps} => {
                let rate = source_fps.unwrap_or(fps).exact();
                Ok(time.scale(&rate).floor_clamped(0,i64::try_from(files.len()-1)?)? as usize)
            }
            Self::Video(video) => video.frame_index(time),
        }
    }
    #[cfg(test)]
    fn exact_timeline(&self, fps:vork_time::Time)->Result<Vec<Germ>> {
        self.map_timeline(fps, |time| Ok(Germ::point(time)))
    }
    fn map_timeline<T>(&self, fps:vork_time::Time, convert: impl Fn(Rational) -> Result<T>)->Result<Vec<T>> {
        Ok(match self {
            Self::Sequence {files,fps:source_fps} => {
                let rate = source_fps.unwrap_or(fps).exact();
                (0..files.len()).map(|n|convert(Rational::from_integer(n.into())/&rate)).collect::<Result<_>>()?
            }
            Self::Video(video) => {
                let origin = Rational::from_integer(video.index.pts(0)?.context("missing first picture")?.into());
                let mut times = Vec::new();
                let mut previous = None;
                video.index.visit_timestamps(|pts| {
                    if previous != Some(pts) {
                        times.push(convert((Rational::from_integer(pts.into()) - &origin) * &video.timebase)?);
                        previous = Some(pts);
                    }
                    Ok(())
                })?;
                times
            }
        })
    }
    pub(crate) fn frame_exact(&mut self,time:&Germ,fps:vork_time::Time,max_side:u32)->Result<image::DynamicImage> {
        let _decode = moda_engine::prof::span("canvas.input_decode");
        let index=self.frame_index_exact(time,fps)?;
        self.frame_at_index(index, max_side)
    }
    pub(crate) fn frame_at_index(&mut self, index:usize, max_side:u32)->Result<image::DynamicImage> {
        match self {
            Self::Sequence {files,..} => {
                let path=&files[index];
                crate::texture_source::decode(path,&std::fs::read(path)?,max_side)
                    .with_context(||format!("decode sequence frame {}; restore a valid frame or re-export the sequence",path.display()))
            }
            Self::Video(video) => video.frame_at(index,max_side),
        }
    }
    /// The frame at `time` with its colour in `upload`, the encoding the caller uploads
    /// (`texture_source::decode_color`): `explicit` is the image's declared space; without it a sequence
    /// frame's own description decides, and footage's stream tags. `light`: the image is light
    /// (an environment), and a colour that conversion carries outside the upload's gamut is limited to it
    /// (`texture_source::decode_light`).
    pub(crate) fn frame_color_exact(&mut self,time:&Germ,fps:vork_time::Time,max_side:u32,explicit:Option<vork_scene::v2::ColorSpace>,upload:vork_scene::v2::WorkingColorSpace,light:bool)->Result<image::DynamicImage> {
        let _decode = moda_engine::prof::span("canvas.input_decode");
        let index=self.frame_index_exact(time,fps)?;
        match self {
            Self::Sequence {files,..} => {
                let path=&files[index];
                let decode=if light {crate::texture_source::decode_light} else {crate::texture_source::decode_color};
                decode(path,&std::fs::read(path)?,max_side,explicit,upload)
                    .with_context(||format!("decode sequence frame {}",path.display()))
            }
            Self::Video(video) => {
                use vork_scene::v2::{ColorSpace as C,WorkingColorSpace as S};
                // The stream's own tags (ffprobe's names), when the image declares no space. HDR transfers and
                // wide primaries are not decoded yet and refuse. Every other SDR tag (BT.709, BT.601, sRGB, or
                // none) is read as sRGB, as Blender and browsers read video; linear light is read as linear.
                let source = match explicit {
                    Some(C::NonColor) => return video.frame_at(index,max_side),
                    Some(C::Srgb) => S::Srgb, Some(C::DisplayP3) => S::DisplayP3, Some(C::LinearRec709) => S::LinearRec709,
                    Some(C::LinearRec2020) => S::LinearRec2020, Some(C::ACEScg) => S::ACEScg,
                    None if video.map_hdr => S::Srgb,
                    None => {
                        let tagged=|what:&str|crate::texture_source::declared_colour(format!("footage {} is tagged {what}",video.path.display()));
                        if video.is_hdr() {return Err(tagged("PQ or HLG, which is not decoded yet"));}
                        match video.primaries.as_str() {
                            "smpte432" => return Err(tagged("Display P3 (smpte432): \"Display P3\" reads the tagged colours")),
                            "bt2020" => return Err(tagged("Rec.2020 (bt2020), which is not read yet")),
                            "smpte431" => return Err(tagged("DCI-P3 (smpte431), which is not read yet")),
                            _ => {}
                        }
                        if video.transfer=="linear" {S::LinearRec709} else {S::Srgb}
                    }
                };
                // A decoded video frame has no negative samples of its own.
                let frame=crate::texture_source::convert_image(video.frame_at(index,max_side)?,source,upload);
                Ok(if light {crate::texture_source::limit_to_gamut(frame)} else {frame})
            }
        }
    }
    pub(crate) fn identity(&self)->Result<blake3::Hash> {
        let mut hash=blake3::Hasher::new();
        match self {
            Self::Sequence {files,..} => for file in files {
                hash.update(crate::asset_revision::content(file)?.as_bytes());
            },
            Self::Video(video) => {
                let mut part = blake3::Hasher::new();
                part.update_reader(video.revision.reader()?)?;
                hash.update(part.finalize().as_bytes());
            }
        }
        Ok(hash.finalize())
    }
    pub fn frame(&mut self,time:impl Into<vork_time::Request>,fps:vork_time::Time,max_side:u32)->Result<image::DynamicImage> {
        self.frame_exact(&time.into().exact()?,fps,max_side)
    }

}

pub(crate) fn sequence(path: &Path) -> Result<Vec<PathBuf>> {
    let name = path.file_name().context("sequence needs a filename")?.to_str().context("sequence filename must be UTF-8")?;
    let start = name.find('%').context("sequence pattern missing %")?;
    let tail = &name[start+1..];
    let end = tail.find('d').context("sequence pattern must use %d or %04d")?;
    let digits = &tail[..end];
    ensure!(digits.is_empty() || (digits.starts_with('0') && digits[1..].chars().all(|c| c.is_ascii_digit())), "sequence pattern must use %d or %04d");
    let width = if digits.is_empty() {0} else {digits.parse::<usize>()?};
    ensure!(width <= 20, "sequence field width exceeds 20 digits");
    let prefix = &name[..start];
    let suffix = &tail[end+1..];
    ensure!(!suffix.contains('%'), "sequence must have one frame field");
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let mut frames = std::collections::BTreeMap::new();
    for entry in std::fs::read_dir(parent)? {
        let entry = entry?;
        let file = entry.file_name();
        let Some(file) = file.to_str() else {continue};
        let Some(number) = file.strip_prefix(prefix).and_then(|n| n.strip_suffix(suffix)) else {continue};
        if number.is_empty() || !number.chars().all(|c| c.is_ascii_digit()) {continue;}
        let index = number.parse::<u64>()?;
        if format!("{index:0width$}") != number {continue;}
        frames.insert(index, entry.path());
    }
    ensure!(!frames.is_empty(), "sequence {} has no frames", path.display());
    let mut missing = Vec::new();
    let mut previous = *frames.first_key_value().unwrap().0;
    for &index in frames.keys().skip(1) {
        if index > previous+1 {
            missing.push(if index == previous+2 {format!("{}",previous+1)} else {format!("{}..{}",previous+1,index-1)});
        }
        previous = index;
    }
    let mut errors = Vec::new();
    if !missing.is_empty() {
        errors.push(anyhow::anyhow!("sequence {} missing frames: {}; restore the missing frames or correct the pattern", path.display(), missing.join(", ")));
    }
    for frame in frames.values() {
        if let Err(error) = crate::source_file(frame) {
            errors.push(anyhow::Error::new(error).context(format!("sequence frame {}: restore a readable image file or correct the pattern", frame.display())));
        }
    }
    vork_scene::v2::diagnostics::joined(errors)?;
    Ok(frames.into_values().collect())
}

pub(super) use vork_media::{DecoderProcess, stop_decoders_for_exit};

pub(super) mod pixels;
use pixels::PixelInput;
struct Session { _child: DecoderProcess, input: PixelInput, next: usize }
pub struct Video {
    pub(super) path: PathBuf,
    pub(super) revision: Arc<vork_media::MediaRevision>,
    pub(super) opaque: bool,
    pub(super) alpha_decoder: Option<&'static str>,
    pub(super) map_hdr: bool,
    transfer: String,
    matrix_unspecified: bool,
    primaries: String,
    pub(super) index: Arc<vork_media::PictureIndex>,
    timebase: Rational,
    /// The stream's declared duration in seconds, when the container states one.
    duration: Option<Rational>,
    last_request: Option<usize>,
    #[cfg(test)]
    decoded: usize,
    pub(super) size: [u32;2],
    /// Present when the stream reports a container aperture. A successful
    /// geometry probe also records the decoder's native cropped pixel format.
    pub(super) container_crop: Option<ContainerCrop>,
    session: Option<Session>,
    pub(super) hardware: bool,
    #[cfg(vork_nvdecode)]
    nvidia: Option<[u8; 16]>,
    #[cfg(vork_nvdecode)]
    deep: Option<crate::footage_nvidia::DeepReader>,
    frames: std::collections::VecDeque<(usize, image::DynamicImage)>,
}
#[derive(Clone)]
pub(super) struct ContainerCrop { picture: Option<([u32; 2], String)> }

/// Undeclared movie length. Until EOF the last observed PTS gives a lower
/// bound. Only EOF determines the old last-frame hold (one mean frame).
pub(crate) struct PictureSpan {
    index: Arc<vork_media::PictureIndex>,
    timebase: Rational,
    fps: vork_time::Time,
}

fn displayed_by(time: &Germ, pts: i64, origin: i64, tick: &Rational) -> bool {
    let delta = Rational::from_integer(pts.into()) - Rational::from_integer(origin.into());
    time.compare_point(&(delta * tick)).is_ge()
}

impl PictureSpan {
    pub(crate) fn through(&self, until: Option<&Germ>) -> Result<(Rational, bool)> {
        let (first, last, count, complete) = self.index.span_through(|pts, origin|
            until.is_none_or(|time| displayed_by(time, pts, origin, &self.timebase)))?;
        let span = (Rational::from_integer(last.into()) - Rational::from_integer(first.into())) * &self.timebase;
        if !complete { return Ok((span, false)); }
        let frame = if count > 1 { &span / Rational::from_integer((count-1).into()) }
            else { decimal(&self.fps.to_string())?.recip() };
        Ok((span + frame, true))
    }
}

/// Facts available without opening a picture index.
struct MovieHeader {
    probe: serde_json::Value,
    size: [u32; 2],
    rotation: Option<i64>,
    crop: Option<ContainerCrop>,
    alpha_decoder: Option<&'static str>,
}

impl MovieHeader {
    fn read(revision: &vork_media::MediaRevision) -> Result<Self> {
        let path = revision.original();
        let probe = revision.input(Command::new("ffprobe").args(["-v","error","-select_streams","v:0","-show_streams","-show_entries",
            "stream=codec_name,pix_fmt,width,height,time_base,duration,avg_frame_rate,color_transfer,color_primaries,color_space:stream_tags=alpha_mode:stream_side_data=side_data_type,rotation:format=format_name,duration","-of","json"])
            )?.output().context("video inputs require ffprobe and ffmpeg on PATH")?;
        ensure!(probe.status.success(), "probe video {}: {}", path.display(), String::from_utf8_lossy(&probe.stderr));
        let value: serde_json::Value = serde_json::from_slice(&probe.stdout)?;
        revision.check_format(value["format"]["format_name"].as_str().unwrap_or(""))?;
        let stream = &value["streams"][0];
        let size = [stream["width"].as_u64().context("video width missing")? as u32,
                    stream["height"].as_u64().context("video height missing")? as u32];
        let codec = stream["codec_name"].as_str().unwrap_or("");
        // Matroska carries VPx alpha in BlockAdditional. The default software and hardware
        // decoders report only the colour plane's pixel format, even on an alpha track.
        let alpha_mode = stream["tags"].as_object().is_some_and(|tags| tags.iter()
            .any(|(k, v)| k.eq_ignore_ascii_case("alpha_mode") && v == "1"));
        let alpha_decoder = match (codec, alpha_mode) {
            ("vp9", true) => Some("libvpx-vp9"),
            ("vp8", true) => Some("libvpx"),
            _ => None,
        };
        let side_data = stream["side_data_list"].as_array().map(Vec::as_slice).unwrap_or_default();
        let rotation = side_data.iter().find_map(|side| side["rotation"].as_i64());
        let crop = side_data.iter().any(|side| side["side_data_type"] == "Frame Cropping").then(|| {
            let probe_frame = || Video::cropped_frame(revision, alpha_decoder);
            ContainerCrop { picture: probe_frame().or_else(probe_frame) }
        });
        let header = Self { probe: value, size, rotation, crop, alpha_decoder };
        if header.unresolved_aperture() {
            let [width, height] = header.display_size(None);
            crate::warnings::approximated("movie_clean_aperture", serde_json::json!({"movie":path}),
                format!("FFmpeg could not measure the clean aperture; using the {width}x{height} stream size may stretch the picture"));
        }
        Ok(header)
    }

    fn unresolved_aperture(&self) -> bool {
        self.crop.as_ref().is_some_and(|crop| crop.picture.is_none())
    }

    fn display_size(&self, frame_rotation: Option<i32>) -> [u32; 2] {
        if let Some((size, _)) = self.crop.as_ref().and_then(|crop| crop.picture.as_ref()) { return *size; }
        let [w, h] = self.size;
        if frame_rotation.map(i64::from).or(self.rotation).is_some_and(|angle| angle.rem_euclid(180) == 90) {
            [h, w]
        } else { [w, h] }
    }

    fn info(&self) -> Result<serde_json::Value> {
        let [width, height] = self.display_size(None);
        let stream = &self.probe["streams"][0];
        let duration: f64 = stream["duration"].as_str().filter(|v| *v != "N/A")
            .or_else(|| self.probe["format"]["duration"].as_str())
            .context("ask: movie has no duration")?.parse()?;
        let fps: vork_time::Time = stream["avg_frame_rate"].as_str().context("ask: movie has no frame rate")?.parse()?;
        ensure!(duration.is_finite() && duration >= 0. && fps > vork_time::Time::ZERO,
            "ask: movie timing is invalid; check it with ffprobe");
        Ok(serde_json::json!({"width":width,"height":height,"duration":duration,"fps":fps}))
    }
}

impl Video {
    fn is_hdr(&self) -> bool {
        matches!(self.transfer.as_str(), "smpte2084" | "arib-std-b67")
    }
    /// The probed stream again, with no decoder session and nothing decoded.
    fn fresh(&self) -> Self {
        Self {
            path: self.path.clone(), revision: self.revision.clone(), opaque: self.opaque,
            alpha_decoder: self.alpha_decoder, map_hdr: self.map_hdr,
            transfer: self.transfer.clone(), primaries: self.primaries.clone(), matrix_unspecified: self.matrix_unspecified,
            index: self.index.clone(), timebase: self.timebase.clone(), duration: self.duration.clone(),
            size: self.size, container_crop: self.container_crop.clone(),
            last_request: None, session: None, hardware: false, frames: Default::default(),
            #[cfg(vork_nvdecode)] nvidia: None,
            #[cfg(vork_nvdecode)] deep: None,
            #[cfg(test)] decoded: 0,
        }
    }
    fn frame_index(&self,time:&Germ)->Result<usize> {
        self.index.select(|pts, origin| displayed_by(time, pts, origin, &self.timebase))
    }
    fn picture_span(&self, fps: vork_time::Time) -> PictureSpan {
        PictureSpan { index: self.index.clone(), timebase: self.timebase.clone(), fps }
    }
    fn open(path: &Path) -> Result<Self> {
        let revision = vork_media::MediaRevision::open(path)?;
        let header = MovieHeader::read(&revision)?;
        Self::from_header(revision, &header)
    }
    fn from_header(revision: Arc<vork_media::MediaRevision>, header: &MovieHeader) -> Result<Self> {
        let stream = &header.probe["streams"][0];
        let (numerator,denominator)=stream["time_base"].as_str().context("video time base missing")?.split_once('/').context("invalid video time base")?;
        let timebase=decimal(&format!("{numerator}/{denominator}"))?;
        ensure!(timebase > Rational::from_integer(0.into()), "video time base must be positive");
        let duration=stream["duration"].as_str().and_then(|text|decimal(text).ok()).filter(|d|d>&Rational::from_integer(0.into()));
        ensure!(header.size.iter().all(|&n|n>0), "video has no displayed frames");
        let index = Arc::new(vork_media::PictureIndex::open(&revision, header.size)?);
        let opaque = vork_media::opaque_stream(stream["codec_name"].as_str().unwrap_or(""), stream["pix_fmt"].as_str().unwrap_or(""));
        let size = header.display_size(index.first_rotation());
        let matrix_unspecified = matches!(stream["color_space"].as_str(), None | Some("unknown"));
        Ok(Self {
            path: revision.original().into(), revision, opaque, alpha_decoder: header.alpha_decoder, map_hdr: false,
            transfer: stream["color_transfer"].as_str().unwrap_or("").into(),
            primaries: stream["color_primaries"].as_str().unwrap_or("").into(), matrix_unspecified,
            index, timebase, duration, size, container_crop: header.crop.clone(), last_request: None, session: None, hardware: false,
            #[cfg(vork_nvdecode)] nvidia: None,
            #[cfg(vork_nvdecode)] deep: None,
            frames: Default::default(), #[cfg(test)] decoded: 0,
        })
    }
    #[cfg(test)]
    fn frame(&mut self, time: f64, max_side:u32) -> Result<image::DynamicImage> {
        ensure!(self.size.iter().all(|&n|n<=max_side), crate::gpu_memory::device_limit(format_args!("video {}", self.path.display()), format_args!("{}x{}", self.size[0], self.size[1]), format_args!("max_texture_dimension_2d={max_side}"), "resize the source video or use a GPU with larger texture limits"));
        let index = self.frame_index(&Germ::point(vork_time::Clock::from(time).exact()))?;
        self.frame_at(index,max_side)
    }
    pub(super) fn pixel_anchor(&self, index: usize, next: usize) -> Result<Option<(usize, i64)>> {
        if index == next || !pixels::can_stamp_seek() { return Ok(None); }
        Ok(self.index.seek_anchor(index)?.filter(|(i, _)| *i != 0))
    }
    fn decoder_command(alpha_decoder: Option<&str>, hardware: bool) -> Command {
        let mut command=Command::new("ffmpeg");
        command.args(["-v","error","-nostdin","-nostats","-copyts"]);
        if !hardware {
            // Preserve the signalled left crop even when its first pixel is not
            // SIMD-aligned. Otherwise FFmpeg rounds down the offset before scaling.
            command.args(["-flags", "+unaligned"]);
        }
        if let Some(decoder) = alpha_decoder {
            command.args(["-c:v:0", decoder]);
        } else if hardware {
            command.args(["-hwaccel", "auto"]);
        }
        command
    }

    fn cropped_frame(revision: &vork_media::MediaRevision, alpha_decoder: Option<&str>) -> Option<([u32; 2], String)> {
        let mut command = Self::decoder_command(alpha_decoder, false);
        revision.input(command.args(["-hide_banner", "-v", "info", "-i"])).ok()?;
        let probe = command.args(["-map", "0:v:0", "-frames:v", "1", "-vf", "showinfo=checksum=0",
            "-c:v", "wrapped_avframe", "-f", "null", "-"]).output().ok()?;
        if !probe.status.success() { return None; }
        let log = String::from_utf8_lossy(&probe.stderr);
        let frame = log.lines().filter(|line| line.starts_with("[Parsed_showinfo_"))
            .filter_map(|line| line.split_once("] n:")).map(|(_, fields)| fields)
            .find(|fields| fields.split_whitespace().next() == Some("0"))?;
        let field = |key: &str| frame.split_whitespace().find_map(|field| field.strip_prefix(key));
        let (w, h) = field("s:")?.split_once('x')?;
        let size = [w.parse().ok()?, h.parse().ok()?];
        let format = field("fmt:")?;
        (size.iter().all(|&n| n > 0) && !format.is_empty() && format.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'))
            .then(|| (size, format.to_owned()))
    }

    fn command(&self, anchor: Option<(usize, i64)>, canvas_filter: Option<&str>, hardware: bool, pipes: &vork_media::PixelPipes) -> Result<Command> {
        let mut command = Self::decoder_command(self.alpha_decoder, hardware);
        if let Some((_,time))=anchor {
            // Ask the demuxer for the key at this absolute timestamp. Round
            // up to avoid asking for the previous GOP on a fractional microsecond.
            // With copyts, FFmpeg's accurate-seek trim can apply the movie's
            // start offset again. Inspect the actual landing PTS ourselves.
            let micros=(Rational::from_integer(time.into())*&self.timebase*Rational::from_integer(1_000_000.into())).ceil().to_integer().to_string();
            let (sign,digits)=micros.strip_prefix('-').map_or(("",micros.as_str()),|s|("-",s));
            let digits=format!("{digits:0>7}");
            let seek=format!("{sign}{}.{}",&digits[..digits.len()-6],&digits[digits.len()-6..]);
            command.args(["-noaccurate_seek","-seek_timestamp","1","-ss",&seek]);
            pixels::seek_input(&mut command);
        }
        self.revision.input(command.arg("-i"))?.args(["-map","0:v:0","-an","-sn","-fps_mode","passthrough"]);
        // Tone-map in float before the existing RGBA64 -> canvas rounding and association.
        // zscale reads the frame's transfer, primaries, matrix and range. For a bare HDR
        // transfer tag, use BT.2100 primaries and non-constant-luminance YCbCr.
        let mapped = self.map_hdr.then(|| {
            // Slice boundaries change zscale's reconstruction of subsampled chroma.
            // Use one whole-picture reconstruction; the following RGB filters can tile.
            let mut filter = String::from("zscale=t=linear:npl=100:threads=1");
            if self.primaries.is_empty() || self.primaries == "unknown" { filter.push_str(":pin=2020"); }
            if self.matrix_unspecified { filter.push_str(":min=2020_ncl"); }
            filter.push_str(",format=gbrapf32le,zscale=p=709,tonemap=clip,zscale=t=iec61966-2-1:m=gbr:r=full");
            if let Some(round) = canvas_filter {
                filter.push(',');
                filter.push_str(round);
            }
            filter
        });
        let colour = mapped.as_deref().or(canvas_filter);
        // Keep the cropped picture on its native planes until after the aperture
        // is applied. Negotiating RGB first changes odd subsampled crop edges.
        let crop_filter = self.container_crop.as_ref().and_then(|crop| crop.picture.as_ref()).map(|(_, format)| {
            let mut filter = format!("format={format}");
            if let Some(colour) = colour { filter.push(','); filter.push_str(colour); }
            filter
        });
        pixels::configure(&mut command, &self.timebase, crop_filter.as_deref().or(colour), anchor.is_some(), pipes.records_url());
        #[cfg(windows)]
        command.args(["-y", "-flush_packets", "1"]);
        // A seek can open on a resized keyframe. Rawvideo needs one fixed stride.
        command.args(["-s:v", &format!("{}x{}", self.size[0], self.size[1])]);
        command.args(["-f","rawvideo","-pix_fmt",if canvas_filter.is_some() {"rgba"} else {"rgba64le"},pipes.pixels_url()]);
        Ok(command)
    }
    pub(super) fn start_pixels(&self, anchor: Option<(usize, i64)>, filter: Option<&str>, hardware: bool)
        -> Result<(DecoderProcess, PixelInput, usize)> {
        let mut landing = anchor.filter(|_| pixels::can_stamp_seek());
        loop {
            let pipes = vork_media::PixelPipes::new().context("create movie output pipes")?;
            let command = self.command(landing, filter, hardware, &pipes)?;
            let (child, input, stamps) = pipes.start(command).context("start movie decoder")?;
            let next = landing.map_or(0, |(ordinal, _)| ordinal);
            let mut pixels = PixelInput::new(Box::new(input), stamps, &self.index, next, landing.is_some())?;
            if landing.is_some() && !pixels.check_landing()? {
                drop(child);
                self.index.reject_anchor(next)?;
                landing = self.index.seek_anchor(next)?.filter(|(i, _)| *i > 0 && *i < next);
                continue;
            }
            return Ok((child, pixels, next));
        }
    }
    pub(super) fn check_limit(&self, max_side:u32) -> Result<()> {
        ensure!(self.size.iter().all(|&n|n<=max_side), crate::gpu_memory::device_limit(format_args!("video {}", self.path.display()), format_args!("{}x{}", self.size[0], self.size[1]), format_args!("max_texture_dimension_2d={max_side}"), "resize the source video or use a GPU with larger texture limits"));
        Ok(())
    }
    fn frame_at(&mut self,index:usize,max_side:u32)->Result<image::DynamicImage> {
        self.check_limit(max_side)?;
        #[cfg(vork_nvdecode)]
        if let Some(uuid) = self.nvidia {
            if let Some((_, frame)) = self.frames.iter().find(|(i, _)| *i == index) {
                return Ok(frame.clone());
            }
            let result = self.deep.take().map_or_else(
                || crate::footage_nvidia::DeepReader::open(self, uuid), Ok
            ).and_then(|mut reader| {
                let image = reader.frame(self, index)?;
                self.deep = Some(reader);
                Ok(image)
            });
            match result {
                Ok(image) => {
                    self.frames.push_back((index, image.clone()));
                    while self.frames.len() > crate::constants::FOOTAGE_DECODE_FRAMES { self.frames.pop_front(); }
                    return Ok(image);
                }
                Err(error) => {
                    self.nvidia = None;
                    eprintln!("movie {} frame {index}: using software decoder ({error:#})", self.path.display());
                }
            }
        }
        let mut result = self.read_frame_at(index, false);
        if result.as_ref().is_err_and(|e| e.is::<pixels::WrongLanding>()) {
            self.session = None;
            result = self.read_frame_at(index, true);
        }
        match result {
            Err(error) if self.hardware => {
                self.hardware = false;
                self.session = None;
                eprintln!("movie {} frame {index}: retrying with software decode ({error:#})", self.path.display());
                self.read_frame_at(index, true)
            }
            other => other,
        }
    }
    fn read_frame_at(&mut self,index:usize,from_start:bool)->Result<image::DynamicImage> {
        let previous=self.last_request.replace(index);
        if let Some((_,frame)) = self.frames.iter().find(|(cached,_)|*cached==index) {return Ok(frame.clone());}
        // For a reverse walk retain the next predicted request too. The existing
        // bounded frame owner holds those two frames, not the whole decoded GOP.
        let preceding=previous.filter(|&p|p>index).map(|p|index.saturating_sub(p-index));
        let anchor = if from_start { None } else { self.pixel_anchor(index, self.session.as_ref().map_or(0, |s| s.next))? };
        let start=anchor.map_or(0,|(i,_)|i);
        // A forward jump also seeks when its keyframe is beyond the live pipe.
        if self.session.as_ref().is_none_or(|s|s.next>index || s.next<start) {
            self.session = None;
            let (child, input, next) = self.start_pixels(anchor, None, self.hardware)?;
            self.session = Some(Session {_child:child,input,next});
        }
        // A read can consume a partial frame before failing. Retain the pipe
        // only after a complete request; every error drops and reaps it so the
        // next uncached request restarts at its keyframe with aligned pixels.
        let mut session = self.session.take().unwrap();
        let mut bytes = vec![0; self.size[0] as usize * self.size[1] as usize * 8];
        let mut result=None;
        while session.next<=index {
            session.input.read_frame(&mut bytes).with_context(||format!("decode video {} frame {}; restore or re-encode the movie",self.path.display(),session.next))?;
            #[cfg(test)] {self.decoded+=1;}
            if session.next==index || preceding==Some(session.next) {
                let pixels=bytes.chunks_exact(2).map(|b|u16::from_le_bytes([b[0],b[1]])).collect();
                let image=image::DynamicImage::ImageRgba16(image::ImageBuffer::from_raw(self.size[0],self.size[1],pixels).context("video frame size")?);
                if session.next==index {result=Some(image.clone());}
                self.frames.push_back((session.next,image));
                while self.frames.len()>crate::constants::FOOTAGE_DECODE_FRAMES {self.frames.pop_front();}
            }
            session.next+=1;
        }
        let image=result.context("video frame missing")?;
        self.session=Some(session);
        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Stdio;
    #[test]
    fn oversized_video_reports_dimensions_limit_and_remedy_before_decode() -> Result<()> {
        let root=tempfile::tempdir()?;
        let path=root.path().join("source.mp4");
        ensure!(Command::new("ffmpeg").args(["-v","error","-f","lavfi","-i","color=s=32x16","-frames:v","1"]).arg(&path).status()?.success(), "create video");
        let mut video = Video::open(&path)?;
        for error in [video.frame(0., 8).unwrap_err(), video.frame_at(0, 8).unwrap_err()] {
            let message = error.to_string();
            for expected in ["source.mp4", "32x16", "max_texture_dimension_2d=8", "resize the source video", "vork help limits"] {
                assert!(message.contains(expected), "{message}");
            }
        }
        assert!(video.session.is_none());
        assert!(video.last_request.is_none());
        Ok(())
    }

    fn clean_aperture_fixture(dir: &Path) -> Result<()> {
        // Uncompressed YCbCr MOV has a full-raster clap atom. Adjust only its
        // eight rational components to place a 56x42 aperture at (6, 2).
        let encoded = Command::new("ffmpeg").args(["-v", "error", "-f", "lavfi", "-i",
            "testsrc2=s=64x48", "-frames:v", "1", "-pix_fmt", "uyvy422", "-c:v", "rawvideo"])
            .arg(dir.join("full.mov")).status()?;
        ensure!(encoded.success(), "create clean aperture fixture");
        let mut mov = std::fs::read(dir.join("full.mov"))?;
        let start = mov.windows(8).position(|b| b == b"\x00\x00\x00\x28clap").context("MOV clean aperture atom")? + 8;
        for (field, value) in mov[start..start+32].chunks_exact_mut(4).zip([56i32, 1, 42, 1, 2, 1, -1, 1]) {
            field.copy_from_slice(&value.to_be_bytes());
        }
        std::fs::write(dir.join("aperture.mov"), mov)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn movie_opens_with_ffprobe_alone() -> Result<()> {
        const CHILD: &str = "VORK_TEST_FFPROBE_ALONE";
        if let Some(dir) = std::env::var_os(CHILD) {
            let opened = Opened::default();
            let path = Path::new(&dir).join("aperture.mov");
            assert_eq!(opened.source(&Path::new(&dir).join("full.mov"))?.dimensions()?, [64, 48]);
            assert_eq!(opened.source(&path)?.dimensions()?, [64, 48]);
            // Restore FFmpeg in the same open. A failed probe must not freeze the stream size.
            std::os::unix::fs::symlink(std::env::var_os("VORK_TEST_RESTORE_FFMPEG").unwrap(), Path::new(&dir).join("ffmpeg"))?;
            let recovered = opened.source(&path)?;
            assert_eq!(recovered.dimensions()?, Source::open(&path)?.dimensions()?);
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        clean_aperture_fixture(dir.path())?;
        let tool = |name: &str| -> Result<PathBuf> {
            std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()).map(|dir| dir.join(name))
                .find(|path| path.is_file()).with_context(|| format!("find {name}"))?.canonicalize().map_err(Into::into)
        };
        std::os::unix::fs::symlink(tool("ffprobe")?, dir.path().join("ffprobe"))?;
        ensure!(Command::new(std::env::current_exe()?).args(["--exact", std::thread::current().name().unwrap(), "--nocapture"])
            .env(CHILD, dir.path()).env("PATH", dir.path()).env("VORK_TEST_RESTORE_FFMPEG", tool("ffmpeg")?).status()?.success(), "open without ffmpeg");
        Ok(())
    }

    #[test]
    fn software_movie_keeps_the_decoders_clean_aperture() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("aperture.mov");
        clean_aperture_fixture(dir.path())?;

        // A reporting probe uses the decoder's natural output. When an older
        // FFprobe omits the aperture, retain the 64x48 stream size it reported.
        let probe = Command::new("ffprobe").args(["-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream_side_data=side_data_type", "-of", "json"]).arg(&path).output()?;
        ensure!(probe.status.success(), "read fixture side data");
        let reported = String::from_utf8(probe.stdout)?.contains("Frame Cropping");
        let mut reference = Command::new("ffmpeg");
        reference.args(["-v", "error", "-i"]).arg(&path);
        if !reported { reference.args(["-s:v", "64x48"]); }
        let reference = reference.args(["-map", "0:v:0", "-frames:v", "1", "-pix_fmt", "rgba64le", "-f", "rawvideo", "-"]).output()?;
        ensure!(reference.status.success(), "decode clean aperture reference");
        let mut video = Video::open(&path)?;
        #[cfg(any(vork_nvdecode, vork_vtdecode))]
        if reported {
            assert!(crate::footage_device::admit(&video).unwrap_err().to_string().contains("container clean aperture"));
        }
        let frame = video.frame_at(0, 64)?.into_rgba16();
        let extent = match reference.stdout.len() {
            n if n == 56*42*8 => (56, 42),
            n if n == 64*48*8 => (64, 48),
            n => anyhow::bail!("unexpected clean aperture reference of {n} bytes"),
        };
        let scene = vork_scene::Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"}, "nodes":[], "images":{"plate":{"file":"aperture.mov"}}
        }))?)?;
        let info = crate::ask::scene(scene, dir.path(), &serde_json::json!([{"image":"plate","read":"info"}]))?;
        assert_eq!((info[0]["width"].as_u64(), info[0]["height"].as_u64()),
            (Some(u64::from(extent.0)), Some(u64::from(extent.1))), "ask displayed size");
        assert_eq!(frame.dimensions(), extent);
        assert!(frame.as_raw().iter().zip(reference.stdout.chunks_exact(2))
            .all(|(&v, b)| v == u16::from_le_bytes([b[0], b[1]])), "clean aperture pixels");
        Ok(())
    }

    #[test]
    fn software_movie_applies_the_exact_left_crop() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("cropped.mp4");
        let encoded = Command::new("ffmpeg").args(["-v", "error", "-f", "lavfi", "-i",
            "testsrc2=size=320x192:rate=24", "-frames:v", "6", "-threads", "2", "-c:v", "libx264",
            "-bsf:v", "h264_metadata=crop_left=4:crop_top=4"]).arg(&path).status()?;
        ensure!(encoded.success(), "encode offset crop");
        let raw = Command::new("ffmpeg").args(["-v", "error", "-apply_cropping", "0", "-i"])
            .arg(&path).args(["-vf", "crop=316:188:4:4:exact=1", "-pix_fmt", "rgba64le",
                "-f", "rawvideo", "-"]).output()?;
        let mut video = Video::open(&path)?;
        assert_eq!(video.size, [316, 188]);
        let stride = 316 * 188 * 8;
        ensure!(raw.status.success() && raw.stdout.len() == 6 * stride, "uncropped movie reference");
        for i in [0, 5, 2] {
            let frame = video.frame_at(i, 320)?.into_rgba16();
            let reference = raw.stdout[i * stride..(i + 1) * stride].chunks_exact(2);
            assert!(frame.as_raw().iter().zip(reference)
                .all(|(&v, b)| v == u16::from_le_bytes([b[0], b[1]])), "crop at frame {i}");
        }
        Ok(())
    }

    #[test]
    fn exact_nominal_playback_and_timeline_share_boundaries()->Result<()> {
        for (scene_rate, embedded) in [("24", None), ("24", Some("30000/1001")), ("60000/1001", None)] {
            let fps = scene_rate.parse()?;
            let source_fps = embedded.map(str::parse).transpose()?;
            let nominal_rate = source_fps.unwrap_or(fps);
            let source=Source::Sequence {files:vec![PathBuf::new();107893],fps:source_fps};
            assert_eq!(source.length(fps)?, vork_time::Request::Frame(107893, nominal_rate).exact()?.value().clone());
            let timeline=source.exact_timeline(fps)?;
            for (n,time) in timeline.iter().enumerate() {
                let nominal=vork_time::Request::Frame(n as u32,nominal_rate).exact()?;
                assert_eq!(time,&nominal);
                assert_eq!(source.frame_index_exact(&nominal,fps)?,n);
                if n>0 {
                    assert_eq!(source.frame_index_exact(&Germ::left(time.value().clone()),fps)?,n-1);
                }
            }
        }
        Ok(())
    }
    #[test]
    #[ignore="requires a real GPU and pinned Slang"]
    fn sequence_compositor_samples_its_own_extent_and_updates_with_the_clock()->Result<()> {
        let r = crate::test_renderer()?;
        let root=std::env::temp_dir().join(format!("vork-footage-compositor-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        let first=image::RgbaImage::from_fn(8,4,|x,_|image::Rgba(if x<4 {[255,0,0,255]} else {[0,255,0,255]}));
        first.save(root.join("1001.png"))?;
        image::RgbaImage::from_pixel(8,4,image::Rgba([0,0,255,255])).save(root.join("1002.png"))?;
        let input:crate::Scene=serde_json::from_value(serde_json::json!({"version":2,
            "composition":{"space":"3d","compositor":["comp"]},"render":{"resolution":[4,2],"fps":24,"motion_blur":{"shutter":1,"position":"end"},"settings":{"samples":4,"integrator":"path","denoiser":null}},
            "images":{"plate":{"file":"%04d.png"}},
            "programs":{"plate":{"source":"import vork; struct Inputs {Image plate;}; float4 composite(CompositeContext c,Inputs i) {return compositor_sample(i.plate,c.uv);}"}},
            "materials":{"comp":{"program":"plate","inputs":{"plate":{"image":"plate"}}}},
            "nodes":[{"type":"camera"}]}))?;
        let scene=r.install(input,&root)?;
        let mut slot=scene.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
        let outputs=crate::ReadbackOutputs {float:crate::FloatOutputs {color:true,..Default::default()},..Default::default()};
        slot.submit_outputs(&scene,&r,0.,outputs)?;
        let first=slot.collect(&r)?.linear_rgba.unwrap();
        assert_eq!(first[0],[1.,0.,0.,1.]);assert_eq!(first[3],[0.,1.,0.,1.]);
        slot.submit_outputs(&scene,&r,1./24.,outputs)?;
        assert!(slot.collect(&r)?.linear_rgba.unwrap().iter().all(|p|*p==[0.,0.,1.,1.]));
        // The same media clock also drives the World, without a compositor.
        let environment:crate::Scene=serde_json::from_value(serde_json::json!({"version":2,
            "render":{"resolution":[4,2],"fps":24,"settings":{"samples":1}},
            "images":{"plate":{"file":"%04d.png"}},
            "materials":{"environment_image":{"program":"std/environment_image","inputs":{"image":{"image":"plate"}}}},
            "nodes":[{"type":"camera"},{"type":"light","shape":"environment","material":"environment_image","energy":1}]}))?;
        let environment=r.install(environment,&root)?;
        let mut world=environment.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
        world.submit_outputs(&environment,&r,0.,outputs)?;
        let first=world.collect(&r)?.linear_rgba.unwrap();
        assert!(first.iter().any(|p|p[0]>0. || p[1]>0.));
        world.submit_outputs(&environment,&r,1./24.,outputs)?;
        for p in world.collect(&r)?.linear_rgba.unwrap() {assert!(p[0].abs()<0.001 && p[1].abs()<0.001 && (p[2]-1.).abs()<0.001,"retimed World: {p:?}");}
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires a real GPU and pinned Slang"]
    fn aborted_recording_rejects_all_slots_until_reinstall() -> Result<()> {
        // Both early per-sample uploads and late compositor reads can abort an
        // exposure after retained keys advance. Exercise streaming and batching.
        let r = crate::Renderer::new()?;
        let root = std::env::temp_dir().join(format!("vork-aborted-recording-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        for name in ["a", "b"] {
            for (frame, color) in [(0, [255, 0, 0, 255]), (1, [0, 0, 255, 255])] {
                image::RgbaImage::from_pixel(4, 4, image::Rgba(color))
                    .save(root.join(format!("{name}{frame}.png")))?;
            }
        }
        for early in [false, true] {
            for samples in [1, 2] {
                let mut input = serde_json::json!({"version":2,
                    "composition":{"space":"3d","compositor":["comp"]},"render":{"resolution":[4,4],"fps":24,"settings":{"samples":samples,"integrator":"path","denoiser":null}},
                    "images":{"a":{"file":"a%d.png"},"b":{"file":"b%d.png"}},
                    "programs":{"plate":{"source":"import vork; struct Inputs {Image a; Image b;}; float4 composite(CompositeContext c, Inputs i) {return 0.5*(compositor_sample(i.a,c.uv)+compositor_sample(i.b,c.uv));}"}},
                    "materials":{"comp":{"program":"plate","inputs":{"a":{"image":"a"},"b":{"image":"b"}}}},
                    "nodes":[{"type":"camera","name":"camera"}]});
                if early {
                    if samples == 2 {
                        input["render"]["motion_blur"] = serde_json::json!({"shutter":1});
                    }
                    input["images"]["capture"] = serde_json::json!({"composition":{"compositor":["capture_mix"]},"render":{
                        "camera":"camera","resolution":[4,4]
                    }});
                    input["materials"]["capture_mix"] = input["materials"]["comp"].clone();
                    input["programs"]["capture_copy"] = serde_json::json!({"source":"import vork; struct Inputs {Image capture;}; float4 composite(CompositeContext c, Inputs i) {return compositor_sample(i.capture,c.uv);}"});
                    input["materials"]["comp"] = serde_json::json!({"program":"capture_copy","inputs":{"capture":{"image":"capture"}}});
                }
                let input: crate::Scene = serde_json::from_value(input)?;
                let scene = r.install(input.clone(), &root)?;
                assert!(scene.footage.values().all(|source| source.per_sample == early));
                let mut slot = scene.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
                let mut other = scene.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
                assert_eq!(!slot.image_views.is_empty(), early, "early abort covers moved capture owners");
                let outputs = crate::ReadbackOutputs { float: crate::FloatOutputs {color:true,..Default::default()},..Default::default() };
                slot.submit_outputs(&scene, &r, 0., outputs)?;
                assert!(slot.collect(&r)?.linear_rgba.unwrap().iter().all(|p| *p == [1.,0.,0.,1.]));
                other.submit_outputs(&scene, &r, 0., outputs)?;
                std::fs::rename(root.join("b1.png"), root.join("b1.hidden"))?;
                let failure = slot.submit_outputs(&scene, &r, 1./24., outputs).unwrap_err();
                std::fs::rename(root.join("b1.hidden"), root.join("b1.png"))?;
                assert_eq!(failure.downcast_ref::<std::io::Error>().unwrap().kind(), std::io::ErrorKind::NotFound);
                if early && samples == 2 {
                    assert_eq!(slot.view.sample_index, 1, "one streaming sample submitted before the abort");
                }
                let queued = other.collect(&r).err().expect("queued publication must be refused");
                assert!(format!("{queued:#}").contains("publication invalidated"),
                    "aborted recording must discard queued images: {queued:#}");
                for target in [&mut slot, &mut other] {
                    let retry = target.submit_outputs(&scene, &r, 1./24., outputs);
                    assert!(retry.is_err(), "aborted recording admitted another submission (early={early}, samples={samples})");
                    let error = format!("{:#}", retry.unwrap_err());
                    assert!(error.contains("reinstall") && error.contains(&failure.root_cause().to_string()), "{error}");
                }
                let mut fresh = scene.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
                assert!(fresh.submit_outputs(&scene, &r, 1./24., outputs).is_err());
                // Reinstallation is the explicit recovery boundary; the renderer
                // remains usable and restored sources must publish the new frame.
                let clean = r.install(input, &root)?;
                let mut recovered = clean.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
                recovered.submit_outputs(&clean, &r, 2./24., outputs)?;
                assert!(recovered.collect(&r)?.linear_rgba.unwrap().iter().all(|p| *p == [0.,0.,1.,1.]));
            }
        }
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires a real GPU and pinned Slang"]
    fn aborted_first_tlas_recording_cannot_be_reused() -> Result<()> {
        let r = crate::Renderer::new()?;
        let root = std::env::temp_dir().join(format!("vork-aborted-tlas-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        std::fs::write(root.join("geometry.bin"), include_bytes!("../tests/fixtures/geometry.bin"))?;
        image::RgbaImage::from_pixel(4, 4, image::Rgba([0u8,0,255,255])).save(root.join("plate0.png"))?;
        let mut input: serde_json::Value = serde_json::from_str(include_str!("../tests/fixtures/analytic.json"))?;
        input["render"]["resolution"] = serde_json::json!([32,24]);
        input["render"]["settings"] = serde_json::json!({"samples":1,"integrator":"raster","denoiser":null});
        input["composition"]["space"] = "3d".into(); input["composition"]["compositor"] = serde_json::json!(["comp"]);
        input["images"] = serde_json::json!({"plate":{"file":"plate%d.png"}});
        input["programs"] = serde_json::json!({"plate":{"source":"import vork; struct Inputs {Image plate;}; float4 composite(CompositeContext c, Inputs i) {return 0.5*(compositor_sample(c.uv)+compositor_sample(i.plate,c.uv));}"}});
        input["materials"]["comp"] = serde_json::json!({"program":"plate","inputs":{"plate":{"image":"plate"}}});
        let input = vork_scene::v2::from_test_value(input)?;
        let scene = r.install(input, &root)?;
        assert!(!scene.footage["plate"].per_sample);
        let mut slot = scene.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
        assert_eq!(slot.rays.as_ref().expect("ray-shadow scene").builds, 0);
        std::fs::rename(root.join("plate0.png"), root.join("plate.hidden"))?;
        let error = slot.submit(&scene, &r, 0.).unwrap_err();
        std::fs::rename(root.join("plate.hidden"), root.join("plate0.png"))?;
        assert_eq!(error.downcast_ref::<std::io::Error>().unwrap().kind(), std::io::ErrorKind::NotFound);
        assert_eq!(slot.rays.as_ref().unwrap().builds, 1, "the first TLAS build was recorded before the abort");
        let retry = slot.submit(&scene, &r, 0.);
        assert!(retry.is_err(), "retry admitted a recorded-but-unsubmitted TLAS");
        let error = format!("{:#}", retry.unwrap_err());
        assert!(error.contains("reinstall"), "{error}");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn image_clock_uses_track_layers_for_reverse_hold_and_loop() -> Result<()> {
        let scene:crate::Scene=serde_json::from_value(serde_json::json!({"version":2,"images":{"plate":{"file":"plate.mp4"}},"nodes":[{"type":"camera"}],
            "animation":[{"loop":true,"tracks":[{"path":["images","plate","time"],"keys":[[0,2],[1,1],[2,1],[3,0]]}]}]}))?;
        let plan=crate::animation::Plan::compile(&scene,Path::new("."))?;
        for (time,expected) in [(0.,2.),(0.5,1.5),(1.5,1.),(2.5,0.5),(3.5,1.5)] {
            let mut samples=plan.samples();plan.sample_into(time,&mut samples);
            let request=vork_time::Request::Numeric(time).exact()?;
            assert!((plan.image_time_exact(&samples,&request,"plate",request.clone())?.numeric()?-expected).abs()<1e-5);
        }
        Ok(())
    }
    #[test]
    fn failed_video_reads_restart_at_the_same_request() -> Result<()> {
        let root=std::env::temp_dir().join(format!("vork-footage-recovery-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        let path=root.join("source.mp4");
        let status=Command::new("ffmpeg").args(["-v","error","-y","-f","lavfi","-i","testsrc2=size=16x16:rate=6:duration=2",
            "-c:v","libx264","-g","6","-keyint_min","6","-sc_threshold","0"]).arg(&path).status()?;
        ensure!(status.success(),"make recovery video fixture");
        let expected=Command::new("ffmpeg").args(["-v","error","-i"]).arg(&path)
            .args(["-fps_mode","passthrough","-f","rawvideo","-pix_fmt","rgba64le","pipe:1"]).output()?;
        ensure!(expected.status.success(),"decode recovery reference");
        let frame_bytes=16*16*8;
        // Model a decoder ending before a frame, inside a frame, or after
        // complete frames, on both forward and reverse requests. The data
        // demuxer copies the exact prefix into a real child stdout pipe.
        for (request,previous) in [(0,None),(4,None),(4,Some(5)),(10,None),(10,Some(11))] {
            for length in [0,frame_bytes/2,2*frame_bytes,2*frame_bytes+frame_bytes/2,4*frame_bytes,4*frame_bytes+frame_bytes/2] {
                if length/frame_bytes>request {continue;}
                let mut source=Video::open(&path)?;
                source.last_request=previous;
                let start=source.index.anchor(request)?.map_or(0,|(i,_)|i);
                assert_eq!(start,if request>=6 {6} else {0});
                let truncated=root.join("truncated.raw");
                std::fs::write(&truncated,&expected.stdout[start*frame_bytes..start*frame_bytes+length])?;
                let mut command=Command::new("ffmpeg");
                command.args(["-v","error","-f","data","-i"]).arg(&truncated)
                    .args(["-map","0:0","-c","copy","-f","data","pipe:1"])
                    .stdout(Stdio::piped());
                let child=DecoderProcess::start(command)?;
                let stdout=child.stdout();
                let (send, stamps) = std::sync::mpsc::channel();
                for i in start..=request {
                    let pts = source.index.pts(i)?.unwrap();
                    send.send(vork_media::PictureStamp::Displayed(pts))?;
                }
                let input = PixelInput::from_parts(Box::new(stdout), stamps, source.index.clone(), start, None);
                source.session=Some(Session {_child:child,input,next:start});
                let error=source.frame_at(request,32).unwrap_err();
                let failed_frame=start+length/frame_bytes;
                assert!(error.to_string().contains(&format!("frame {failed_frame}")),"{error:#}");
                assert_eq!(error.downcast_ref::<std::io::Error>().unwrap().kind(),std::io::ErrorKind::UnexpectedEof);
                assert!(source.session.is_none(),"failed decoder must be closed");
                assert!(source.frames.iter().all(|(index,_)|*index<failed_frame),"never cache a partial frame");
                // Retry the identical position without clearing state or seeking.
                let actual=source.frame_at(request,32)?.to_rgba16();
                let bytes:Vec<_>=actual.as_raw().iter().flat_map(|v|v.to_le_bytes()).collect();
                assert_eq!(bytes,expected.stdout[request*frame_bytes..(request+1)*frame_bytes]);
                assert_eq!(source.session.as_ref().unwrap().next,request+1);
            }
        }
        // An opened reader owns its source even when the authored name goes away.
        let mut source=Video::open(&path)?;
        let saved=root.join("saved.mp4");
        std::fs::rename(&path,&saved)?;
        assert!(Video::open(&path).is_err());
        let actual=source.frame_at(0,32)?.to_rgba16();
        let bytes:Vec<_>=actual.as_raw().iter().flat_map(|v|v.to_le_bytes()).collect();
        assert_eq!(bytes,expected.stdout[..frame_bytes]);
        drop(source);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn indexed_video_seeks_preserve_frames_and_bound_reverse_work() -> Result<()> {
        let root=std::env::temp_dir().join(format!("vork-footage-index-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        let path=root.join("long.mp4");
        let status=Command::new("ffmpeg").args(["-v","error","-y","-f","lavfi","-i","testsrc2=size=16x16:rate=30:duration=20",
            "-c:v","libx264","-g","30","-keyint_min","30","-sc_threshold","0","-bf","3","-output_ts_offset","7"]).arg(&path).status()?;
        ensure!(status.success(),"make long-GOP video fixture");
        let expected=Command::new("ffmpeg").args(["-v","error","-i"]).arg(&path)
            .args(["-fps_mode","passthrough","-f","rawvideo","-pix_fmt","rgba64le","pipe:1"]).output()?;
        ensure!(expected.status.success(),"decode video reference");
        let mut source=Video::open(&path)?;
        let mut requests:Vec<_>=(500..600).rev().collect();
        requests.extend([5,590,60,450,300,0,599,599,10]);
        for index in requests {
            let before=source.decoded;
            let request=Germ::point((Rational::from_integer(source.index.pts(index)?.unwrap().into())-Rational::from_integer(source.index.pts(0)?.unwrap().into()))*&source.timebase);
            assert_eq!(source.frame_index(&request)?,index);
            let actual=source.frame_at(source.frame_index(&request)?,32)?.to_rgba16();
            let bytes:Vec<_>=actual.as_raw().iter().flat_map(|v|v.to_le_bytes()).collect();
            assert_eq!(bytes,&expected.stdout[index*16*16*8..(index+1)*16*16*8],"seek frame {index}");
            assert!(source.decoded-before<=30,"seek {index} decoded {} frames",source.decoded-before);
            assert!(source.frames.len()<=crate::constants::FOOTAGE_DECODE_FRAMES);
        }
        drop(source);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn video_timestamps_hold_reverse_and_autorotate() -> Result<()> {
        let root=std::env::temp_dir().join(format!("vork-footage-video-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        for (name,color) in [("red",[255,0,0]),("green",[0,255,0]),("blue",[0,0,255])] {
            image::RgbImage::from_pixel(8,4,image::Rgb(color)).save(root.join(format!("{name}.png")))?;
        }
        std::fs::write(root.join("frames.txt"),"file 'red.png'\nduration 0.04\nfile 'green.png'\nduration 0.16\nfile 'blue.png'\nduration 0.08\nfile 'blue.png'\n")?;
        let video=root.join("vfr.mp4");
        let status=Command::new("ffmpeg").current_dir(&root).args(["-v","error","-y","-f","concat","-safe","0","-i","frames.txt","-fps_mode","vfr","-c:v","libx264","-pix_fmt","yuv444p"]).arg(&video).status()?;
        ensure!(status.success(),"make VFR video fixture");
        let mut source=Video::open(&video)?;
        source.frame(0.,16)?;source.frame(0.04,16)?;
        let next=source.session.as_ref().unwrap().next;
        source.frame(0.,16)?;source.frame(0.04,16)?;
        assert_eq!(source.session.as_ref().unwrap().next,next,"shuffled shutter samples retain the decoder session");
        let mut pts = Vec::new();
        source.index.visit_timestamps(|t| { pts.push(t); Ok(()) })?;
        assert!(pts.windows(3).any(|t| t[1]-t[0] != t[2]-t[1]));
        for (time,channel) in [(0.,0),(0.039,0),(0.04,1),(0.08,1),(0.19,1),(10.,2),(0.,0)] {
            let pixel=source.frame(time,16)?.to_rgb8()[(0,0)];
            assert!(pixel[channel]>240,"at {time}: {pixel:?}");
        }
        let offset=root.join("offset.mp4");
        let status=Command::new("ffmpeg").args(["-v","error","-y","-itsoffset","3","-i"]).arg(&video).args(["-c","copy"]).arg(&offset).status()?;
        ensure!(status.success(),"make edit-list video fixture");
        let mut shifted=Video::open(&offset)?;
        assert_eq!(shifted.frame_index(&Germ::point(decimal("0")?))?,0);
        assert!(shifted.frame(0.08,16)?.to_rgb8()[(0,0)][1]>240);
        drop(shifted);
        let rotated=root.join("rotated.mp4");
        let status=Command::new("ffmpeg").args(["-v","error","-y","-display_rotation","90","-i"]).arg(&video).args(["-c","copy"]).arg(&rotated).status()?;
        ensure!(status.success(),"make rotated video fixture");
        let mut rotated=Video::open(&rotated)?;
        let frame=rotated.frame(0.,16)?;assert_eq!((frame.width(),frame.height()),(4,8));
        drop(rotated);drop(source);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn exr_sequence_clock_uses_first_header_and_rejects_invalid_rates() -> Result<()> {
        use exr::prelude::*;
        let root=std::env::temp_dir().join(format!("vork-exr-clock-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        let write=|index,rate| -> Result<()> {
            let attributes=LayerAttributes {frames_per_second:rate,..Default::default()};
            let channels=SpecificChannels::rgb(|_:Vec2<usize>|(index as f32,0f32,0f32));
            Image::from_layer(Layer::new((1,1),attributes,Encoding::SMALL_LOSSLESS,channels)).write().to_file(root.join(format!("{index:04}.exr")))?;
            Ok(())
        };
        // A later header cannot change the sequence's clock.
        write(2,Some((120,1)))?;
        for rate in [Some((2,1)),None] {
            write(1,rate)?;
            let mut source=Source::open(&root.join("%04d.exr"))?;
            let fps=if rate.is_some() {2.} else {24.};
            assert_eq!(source.timeline(vork_time::Time::integer(24))?.iter().map(|t|t.as_f32()).collect::<Vec<_>>(),vec![0.,1./fps]);
            assert_eq!(source.frame_index_exact(&vork_time::Request::Numeric(0.04).exact()?,vork_time::Time::integer(24))?,0);
            assert_eq!(source.frame_index_exact(&vork_time::Request::Numeric(0.04).exact()?,vork_time::Time::integer(48))?,usize::from(rate.is_none()));
            for render_fps in [vork_time::Time::integer(24),vork_time::Time::integer(48)] {
                for time in [-1.,0.,0.04,0.5,100.,0.] {
                    let expected=1.+source.frame_index_exact(&vork_time::Request::Numeric(time).exact()?,render_fps)? as f32;
                    assert_eq!(source.frame(time,render_fps,16)?.to_rgb32f()[(0,0)][0],expected);
                }
                for (index,time) in source.timeline(render_fps)?.into_iter().enumerate() {assert_eq!(source.frame_index_exact(&vork_time::Request::from(&time).exact()?,render_fps)?,index);}
            }
        }
        for rate in [(0,1),(1,0),(-1,1)] {
            write(1,Some(rate))?;
            let error=Source::open(&root.join("%04d.exr")).err().context("invalid EXR rate accepted")?;
            assert!(format!("{error:#}").contains("framesPerSecond must be positive"),"{error:#}");
        }
        std::fs::remove_dir_all(root)?;Ok(())
    }

    #[test]
    fn sequences_start_at_first_disk_frame_hold_ends_and_reject_holes() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-footage-sequence-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        let dir = root.as_path();
        for n in [1001,1002] {
            image::RgbaImage::from_pixel(1,1,image::Rgba([(n-1000) as u8,0,0,255])).save(dir.join(format!("{n:04}.png")))?;
        }
        let mut source = Source::open(&dir.join("%04d.png"))?;
        for (time,expected) in [(-1.,1),(0.,1),(0.041,1),(1./24.,2),(100.,2)] {
            assert_eq!(source.frame(time,vork_time::Time::integer(24),16)?.to_rgba8()[ (0,0) ][0],expected);
        }
        std::fs::rename(dir.join("1002.png"),dir.join("1004.png"))?;
        let error = Source::open(&dir.join("%04d.png")).err().unwrap().to_string();
        assert!(error.contains("1002..1003"),"{error}");
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }
}

/// One resident texture, updated by ordered encoder copies rather than queue
/// writes so several shutter samples can coexist in one command submission.
pub(crate) struct Installed {
    pub name: String,
    pub time: Option<vork_time::Clock>,
    pub per_sample: bool,
    pub final_compositor: bool,
    pub view: wgpu::TextureView,
    source: Option<std::sync::Arc<std::sync::Mutex<Source>>>,
    /// A picture of a canvas node (`images.X.render.root` names a 2D composition): drawn by Moda's engine at the
    /// node's local time instead of read from a file.
    pub canvas: Option<crate::canvas_world::Picture>,
    texture: wgpu::Texture,
    mips: crate::materials::MipChain,
    /// The encoding the colour decoder is asked for.
    upload: vork_scene::v2::WorkingColorSpace,
    /// Whether an sRGB-encoded upload is decoded to linear light for the texture: a world reads scene-linear
    /// texels. A canvas's image graph reads the encoding the canvas blends in, which the decoder delivers as it
    /// is.
    linear: bool,
    color_space: Option<vork_scene::v2::ColorSpace>,
}
/// The opened decoder and the extent validated by the serial installation.
#[derive(Clone)]
pub(crate) struct FileInput {
    source: std::sync::Arc<std::sync::Mutex<Source>>,
    size: wgpu::Extent3d,
    upload: vork_scene::v2::WorkingColorSpace,
    linear: bool,
    color_space: Option<vork_scene::v2::ColorSpace>,
}
impl Installed {
    /// End external input work while the completed shot still owns GPU resources.
    pub(crate) fn finish(&mut self) {
        if let Some(source) = &self.source && let Source::Video(video) = &mut *source.lock().unwrap() {
            #[cfg(vork_nvdecode)] { video.deep = None; }
            video.session = None; // Session kills and reaps before closing its reader.
        }
    }
    /// The texture a canvas's picture is drawn into each frame: its size, scene-linear with associated alpha.
    pub(crate) fn canvas(r:&crate::Renderer,name:&str,picture:crate::canvas_world::Picture)->Result<Self> {
        let [width,height]=picture.size();
        let size=wgpu::Extent3d {width,height,depth_or_array_layers:1};
        let texture=crate::gpu_memory::texture(&r.device,&wgpu::TextureDescriptor {
            label:Some("canvas picture"),size,mip_level_count:32-width.max(height).leading_zeros(),sample_count:1,
            dimension:wgpu::TextureDimension::D2,format:wgpu::TextureFormat::Rgba16Float,
            usage:wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::COPY_DST|wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats:&[],
        });
        let (pipelines,sampler)=crate::materials::mip_resources(r);
        let mips=crate::materials::MipChain::new(&r.device,&texture,None,&pipelines[2],sampler);
        Ok(Self {name:name.into(),time:None,per_sample:true,final_compositor:false,view:texture.create_view(&Default::default()),texture,mips,source:None,canvas:Some(picture),upload:vork_scene::v2::WorkingColorSpace::LinearRec709,linear:true,color_space:None})
    }
    /// A world's media image: scene-linear texels.
    pub(crate) fn new(r:&crate::Renderer,name:&str,path:&Path,time:Option<vork_time::Clock>,srgb:bool,color_space:Option<vork_scene::v2::ColorSpace>,fps:vork_time::Time,opened:&Opened)->Result<Self> {
        Self::from_source(r,name,opened.source(path)?,time,crate::texture_source::upload(srgb),true,color_space,fps)
    }
    /// The resident texture of `source`. `upload`: the encoding the colour decoder is asked for. `linear`: an
    /// sRGB-encoded upload is decoded to linear light (`Installed::linear`).
    pub(crate) fn from_source(r:&crate::Renderer,name:&str,mut source:Source,time:Option<vork_time::Clock>,upload:vork_scene::v2::WorkingColorSpace,linear:bool,color_space:Option<vork_scene::v2::ColorSpace>,fps:vork_time::Time)->Result<Self> {
        source.prepare_gpu(r);
        let instant=time.as_ref().map(vork_time::Request::from).unwrap_or(vork_time::Request::Numeric(0.)).exact()?;
        let frame=source.frame_color_exact(&instant,fps,r.device.limits().max_texture_dimension_2d,color_space,upload,false)
            .map_err(|error|crate::texture_source::for_image(error,name))?;
        let file = FileInput {
            size: wgpu::Extent3d { width: frame.width(), height: frame.height(), depth_or_array_layers: 1 },
            source: std::sync::Arc::new(std::sync::Mutex::new(source)), upload, linear, color_space,
        };
        Ok(Self::from_file(r, name, time, &file))
    }

    pub(crate) fn file_input(&self) -> FileInput {
        FileInput { source: self.source.as_ref().expect("a file input").clone(), size: self.texture.size(),
            upload: self.upload, linear: self.linear, color_space: self.color_space }
    }

    /// A private upload texture for this lane. No probing, initial decode, or new decoder process.
    pub(crate) fn from_file(r: &crate::Renderer, name: &str, time: Option<vork_time::Clock>, file: &FileInput) -> Self {
        let size = file.size;
        let texture=crate::gpu_memory::texture(&r.device,&wgpu::TextureDescriptor {
            label:Some("footage image"),size,mip_level_count:32-size.width.max(size.height).leading_zeros(),sample_count:1,
            dimension:wgpu::TextureDimension::D2,format:wgpu::TextureFormat::Rgba16Float,
            usage:wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::COPY_DST|wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats:&[],
        });
        let (pipelines,sampler)=crate::materials::mip_resources(r);
        let mips=crate::materials::MipChain::new(&r.device,&texture,None,&pipelines[2],sampler);
        Self {name:name.into(),time,per_sample:true,final_compositor:false,view:texture.create_view(&Default::default()),
            texture,mips,source:Some(file.source.clone()),canvas:None,upload:file.upload,linear:file.linear,color_space:file.color_space}
    }
    /// Resolve a canvas's owning node before either sample or compositor publication.
    pub(crate) fn canvas_instant(&self,s:&crate::InstalledScene,samples:&crate::animation::Samples,request:&Germ)->Result<Option<Germ>> {
        let picture=self.canvas.as_ref().expect("canvas picture");
        let node=s.nodes.iter().position(|n|n.node.name==picture.node())
            .with_context(||format!("image {}: canvas node {:?} is not installed",self.name,picture.node()))?;
        s.animation.local_instant(samples,node,request)
            .map(|base|s.animation.image_time_exact(samples,request,&self.name,base)).transpose()
    }
    /// Decoded media has source frames; a canvas has a continuous, sided local instant.
    /// Absence is a publication too, distinct from both a picture and an invalidated key.
    pub(crate) fn sample_key(&self,time:Option<&Germ>,fps:vork_time::Time)->Result<Vec<u8>> {
        let Some(time)=time else {return Ok(vec![0])};
        let mut key=Vec::new();
        match &self.source {
            Some(source)=>{
                key.push(1);
                key.extend_from_slice(&(source.lock().unwrap().frame_index_exact(time,fps)? as u64).to_le_bytes());
            }
            None=>{key.push(2);time.append_key(&mut key);}
        }
        Ok(key)
    }
    /// `errors` is the owning frame's shader diagnostics: a canvas picture's view reports a non-finite result
    /// there, so the frame refuses it where it is collected and the picture needs no read-back of its own.
    pub(crate) fn encode(&self,r:&crate::Renderer,encoder:&mut wgpu::CommandEncoder,time:Option<&Germ>,fps:vork_time::Time,errors:Option<&wgpu::Buffer>)->Result<()> {
        let Some(time)=time else {
            for mip in 0..self.texture.mip_level_count() {
                let view=self.texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level:mip,mip_level_count:Some(1),..Default::default()
                });
                let _pass=encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label:Some("absent canvas picture"),
                    color_attachments:&[Some(wgpu::RenderPassColorAttachment {
                        view:&view,resolve_target:None,depth_slice:None,
                        ops:wgpu::Operations {load:wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),store:wgpu::StoreOp::Store},
                    })],..Default::default()
                });
            }
            return Ok(());
        };
        if let Some(picture)=&self.canvas {
            let level_zero=self.texture.create_view(&wgpu::TextureViewDescriptor {mip_level_count:Some(1),..Default::default()});
            picture.encode(r,encoder,&vork_time::Request::Exact(time.clone()),&level_zero,errors)?;
            self.mips.encode(encoder,None);
            return Ok(());
        }
        let _decode = moda_engine::prof::span("canvas.input_decode");
        let source=self.source.as_ref().expect("a file source");
        let frame=source.lock().unwrap().frame_color_exact(time,fps,r.device.limits().max_texture_dimension_2d,self.color_space,self.upload,false)
            .map_err(|error|crate::texture_source::for_image(error,&self.name))?;
        ensure!(frame.width()==self.texture.width() && frame.height()==self.texture.height(),"image {}: media frame dimensions changed; re-export every frame at the same resolution",self.name);
        let rgba=frame.into_rgba32f();
        let pitch=(self.texture.width()*8).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)*wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let mut data=vec![0u8;pitch as usize*self.texture.height() as usize];
        for (row,pixels) in rgba.as_raw().chunks_exact(self.texture.width() as usize*4).enumerate() {
            for (i,&value) in pixels.iter().enumerate() {
                let value=if self.linear && !self.upload.is_linear() && i%4!=3 {crate::texture_source::decode_srgb(value)} else {value};
                let at=row*pitch as usize+i*2;
                data[at..at+2].copy_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
            }
        }
        use wgpu::util::DeviceExt;
        let staging=r.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {label:Some("footage upload"),contents:&data,usage:wgpu::BufferUsages::COPY_SRC});
        encoder.copy_buffer_to_texture(wgpu::TexelCopyBufferInfo {buffer:&staging,layout:wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(pitch),rows_per_image:Some(self.texture.height())}},self.texture.as_image_copy(),self.texture.size());
        self.mips.encode(encoder,None);
        Ok(())
    }
}

#[cfg(test)]
mod oracle;
