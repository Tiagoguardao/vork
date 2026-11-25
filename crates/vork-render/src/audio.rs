//! Node sound uses the sound slice of the picture animation plan and the shared CPU samples.
use anyhow::{Context, Result, ensure};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};
use vork_audio::{Clip, Frame, SAMPLE_RATE, Source};
use vork_scene::{Node, Scene};

pub(crate) struct Mix {
    nodes: Vec<crate::EvalNode>,
    plan: crate::animation::Plan,
    sound: crate::animation::SoundPlan,
    /// Nodes with a file and their ancestors, ascending.
    audible: Vec<MixNode>,
    /// Nodes with a file, ascending, and each one's index into `sources`.
    voices: Vec<usize>,
    files: Vec<VoiceFile>,
    remaps: Vec<Option<(String, Option<vork_time::Clock>)>>,
    paints: PaintSelection,
    /// The distinct files the voices play.
    sources: Vec<Arc<Source>>,
}
struct MixNode {
    node: usize,
    parent: Option<usize>,
    voices: std::ops::Range<usize>,
    bus: bool,
}

struct VoiceFile {
    source: usize,
    /// End of the displayed movie; an explicit audio file plays to its own end.
    end: SoundEnd,
    movie: Option<String>,
}

enum SoundEnd {
    At(f64),
    FromPictures(crate::footage::PictureSpan),
}

struct CheckedEnd {
    /// Source seconds covered by the displayed-picture prefix for this voice.
    through: f64,
    /// Exact cutoff once known, infinity while the prefix extends past the requested time.
    end: f64,
}

impl SoundEnd {
    fn initial(&self) -> CheckedEnd {
        match self {
            Self::At(end) => CheckedEnd { through: f64::INFINITY, end: *end },
            Self::FromPictures(_) => CheckedEnd { through: f64::NEG_INFINITY, end: f64::INFINITY },
        }
    }

    fn cover(&self, position: f64, checked: &mut CheckedEnd) -> Result<()> {
        if position < checked.through {
            return Ok(());
        }
        if let Self::FromPictures(span) = self {
            let time = vork_time::exact::Germ::point(vork_time::Clock::from(position).exact());
            let (bound, complete) = span.through(Some(&time))?;
            let seconds = vork_time::exact::Germ::point(bound).seconds()?;
            if complete {
                checked.end = seconds;
                checked.through = f64::INFINITY;
            } else {
                checked.through = seconds;
            }
        }
        Ok(())
    }
}

/// Resolve animated paint binding with the same canvas evaluator as the picture.
/// Material-backed movies and explicit voices use their node clocks directly.
pub(crate) struct PaintSelection {
    voices: Vec<Option<(usize, usize)>>,
    scopes: Vec<PaintScope>,
}
struct PaintScope {
    owner: Option<usize>,
    owner_clocks: Vec<usize>,
    canvas: crate::canvas::SoundPaints,
}
impl PaintSelection {
    fn new(scene: &Scene, base: &Path, plan: &crate::animation::Plan, nodes: &[crate::EvalNode],
           voices: &[usize], files: &[VoiceFile], opened: &crate::footage::Opened) -> Result<Self> {
        let movies: std::collections::HashSet<_> = voices.iter().zip(files)
            .filter(|(node, file)| file.movie.is_some() && matches!(nodes[**node].node.kind,
                vork_scene::NodeKind::Path { .. } | vork_scene::NodeKind::Text { .. }))
            .map(|(&node, _)| nodes[node].node.name.as_str()).collect();
        let mut result = Self { voices: vec![None; voices.len()], scopes: Vec::new() };
        if movies.is_empty() { return Ok(result); }
        let indices: HashMap<_,_> = nodes.iter().enumerate().map(|(i, node)| (node.node.name.as_str(), i)).collect();
        let mut scopes = vec![(std::borrow::Cow::Borrowed(scene), None)];
        let mut locations = HashMap::new();
        fn contains(nodes: &[Node], keyed: &std::collections::HashSet<&str>) -> bool {
            nodes.iter().any(|node| keyed.contains(node.name.as_str()) || contains(&node.children, keyed))
        }
        while let Some((scope, owner)) = scopes.pop() {
            if scope.is_canvas() {
                let outer = scope.without_compositions()?;
                if contains(&outer.nodes, &movies) {
                    let canvas = crate::canvas::SoundPaints::new(&scope, base, &movies, opened)?;
                    for (name, item) in canvas.items() {
                        locations.insert(indices[name], (result.scopes.len(), item));
                    }
                    let owner_clocks = owner.map(|i| plan.sound_ancestors(i)).unwrap_or_default();
                    result.scopes.push(PaintScope { owner, owner_clocks, canvas });
                }
            }
            for part in scope.composition_parts() {
                if contains(&part.children, &movies) {
                    let inner = scope.composition_scene(&part.name, scope.render.resolution)?;
                    scopes.push((std::borrow::Cow::Owned(inner), Some(indices[part.name.as_str()])));
                }
            }
        }
        for (voice, (&node, file)) in voices.iter().zip(files).enumerate() {
            if file.movie.is_some() {
                result.voices[voice] = locations.get(&node).copied();
            }
        }
        Ok(result)
    }

    /// A voice can be selected for several disjoint intervals inside one ordinary
    /// sound block. Only the binding's own evaluator visits those intervals.
    pub(crate) fn windows(&self, voice: usize, movie: &str, plan: &crate::animation::Plan,
        first: i64, last: i64, scratch: &mut crate::animation::SoundScratch) -> Result<Option<Vec<std::ops::Range<i64>>>> {
        let Some((scope, item)) = self.voices[voice] else { return Ok(None) };
        let scope = &self.scopes[scope];
        let scope_rate = scope.owner.map(|owner| scratch.rate_bound(owner));
        let mut at = first;
        let mut windows: Vec<std::ops::Range<i64>> = Vec::new();
        while at < last {
            let scene = vork_time::exact::Germ::right(vork_time::exact::Rational::new(at.into(), SAMPLE_RATE.into()));
            let local = match scope.owner {
                Some(owner) if scope.canvas.animated(item) => plan.sound_local(owner, &scope.owner_clocks, &scene, scratch)?,
                _ => scene,
            };
            let end = scope.canvas.sample(item, &local, at, last, scope_rate.as_ref())?;
            if scope.canvas.shows(item, movie) {
                if let Some(window) = windows.last_mut().filter(|window| window.end == at) {
                    window.end = end;
                } else {
                    windows.push(at..end);
                }
            }
            at = end;
        }
        Ok((windows.as_slice() != [first..last]).then_some(windows))
    }

}

/// A voice's read after decoding this block. Grid-aligned playback becomes a slice;
/// clocks that need selection or reconstruction retain the general sample reader.
#[derive(Clone, Copy)]
enum VoiceRead {
    Silent,
    Aligned(i64),
    Clocked,
}

impl VoiceRead {
    fn new(
        clock: &crate::animation::SoundClock,
        clip: &Clip,
        end: f64,
        first: i64,
        last: i64,
    ) -> Result<Self> {
        if last - first > 1
            && let Some(start) = clock.aligned_start()
            && clip.aligned_frames(start, (last - first) as usize).is_some()
            && (end == f64::INFINITY || clock.seconds(last - 1)? < end)
        {
            return Ok(Self::Aligned(start));
        }
        Ok(Self::Clocked)
    }

    fn sample(clock: &crate::animation::SoundClock, clip: &Clip, end: f64, sample: i64) -> Result<Option<Frame>> {
        if !clock.sounds(sample) {
            return Ok(None);
        }
        let local = clock.seconds(sample)?;
        let speed = clock.speed(sample)?;
        let position = if speed < 0. { local - 1. / clip.rate as f64 } else { local };
        if position >= end {
            return Ok(None);
        }
        Ok(Some(clip.sample(position, speed)))
    }
}

/// Size of the contiguous work rows for one pass through the mix tree.
const TILE: usize = 1024;

/// Blocks scheduled, then decoded for and mixed, at a time: the schedule's memory bound.
const BATCH: usize = 4096;
impl Mix {
    pub(crate) fn new(scene: &Scene, base: &Path) -> Result<Self> {
        Self::new_in(scene, base, &Default::default())
    }

    pub(crate) fn new_in(scene: &Scene, base: &Path, opened: &crate::footage::Opened) -> Result<Self> {
        fn visit(list: &[Node], parent: Option<usize>, nodes: &mut Vec<crate::EvalNode>) {
            for node in list {
                let i = nodes.len();
                nodes.push(crate::EvalNode {
                    node: node.without_children(),
                    parent,
                    world: glam::Mat4::IDENTITY,
                    dynamic: false,
                    camera: None,
                    embedded: None,
                });
                visit(&node.children, Some(i), nodes);
            }
        }
        let mut nodes = Vec::new();
        visit(&scene.nodes, None, &mut nodes);
        let mut voices = Vec::new();
        let mut files = Vec::new();
        let mut sources: Vec<Arc<Source>> = Vec::new();
        let media = scene.media_readers();
        let mut remaps = Vec::new();
        let mut plan = crate::animation::Plan::compile_in(scene, base, None, opened)?;
        plan.mark_canvas_nodes(&mut nodes)?;
        for (i, n) in nodes.iter().enumerate() {
            let mut inputs = Vec::new();
            if let Some(file) = n.node.audio.as_ref().and_then(|a| a.file.as_ref()) {
                inputs.push((opened.audio(&base.join(file))?, None, SoundEnd::At(f64::INFINITY), None));
            }
            for name in media.node_media(&n.node) {
                let image = &scene.images[&name];
                let file = base.join(image.file.as_ref().unwrap());
                if crate::footage::is_sequence(&file) {
                    continue;
                }
                if let Some(source) = opened.optional_audio(&file)? {
                    let remap = (image.time.is_some() || plan.keys_image_time(&name))
                        .then(|| (name.clone(), image.time.clone()));
                    let end = match source.picture_duration() {
                        Some(duration) => SoundEnd::At(duration),
                        None => SoundEnd::FromPictures(opened.picture_span(&file, scene.render.fps)?),
                    };
                    inputs.push((source, remap, end, Some(name)));
                }
            }
            for (source, remap, end, movie) in inputs {
                remaps.push(remap);
                voices.push(i);
                let file = sources.iter().position(|known| Arc::ptr_eq(known, &source))
                    .unwrap_or_else(|| {
                        sources.push(source);
                        sources.len() - 1
                    });
                files.push(VoiceFile { source: file, end, movie });
            }
        }
        // The plan resolves every track's target, owner and clock; the sound slice below keeps
        // the ones sound reads. Layers stay whole: a looping layer's span covers all its tracks.
        let mut audible = std::collections::BTreeSet::new();
        for &voice in &voices {
            let mut parent = Some(voice);
            while let Some(i) = parent {
                audible.insert(i);
                parent = nodes[i].parent;
            }
        }
        let audible: Vec<usize> = audible.into_iter().collect();
        let sound = plan.sound_plan(&audible, |name| {
            nodes
                .iter()
                .position(|n| n.node.name == name)
                .with_context(|| format!("audio: unknown node {name:?}"))
        })?;
        let mut voice = 0;
        let audible: Vec<_> = audible.iter().map(|&node| {
            let first = voice;
            while voices.get(voice) == Some(&node) { voice += 1; }
            let parent = nodes[node].parent.map(|parent| audible.binary_search(&parent).unwrap());
            MixNode { node, parent, voices: first..voice, bus: false }
        }).collect();
        let mut audible = audible;
        for slot in 0..audible.len() {
            if let Some(parent) = audible[slot].parent { audible[parent].bus = true; }
        }
        let paints = PaintSelection::new(scene, base, &plan, &nodes, &voices, &files, opened)?;
        Ok(Self {
            nodes,
            plan,
            sound,
            audible,
            voices,
            files,
            remaps,
            paints,
            sources,
        })
    }
    pub(crate) fn node(&self, name: &str) -> Result<usize> {
        self.nodes
            .iter()
            .position(|n| n.node.name == name)
            .with_context(|| format!("audio: unknown node {name:?}"))
    }
    pub(crate) fn render_range(
        &self,
        first: usize,
        last: usize,
        fps: vork_time::Time,
        root: Option<usize>,
    ) -> Result<Vec<Frame>> {
        let end = (last as u64)
            .checked_add(1)
            .context("audio range overflow")?;
        let range = crate::audio_output::sample_range(first as u64..end, fps)?;
        self.render_samples(range.start, range.end, root)
    }
    pub(crate) fn render(&self, start: f64, end: f64, root: Option<usize>) -> Result<Vec<Frame>> {
        ensure!(
            start.is_finite() && end.is_finite() && end > start,
            "audio: window needs finite start < end"
        );
        let boundary = |seconds: f64| {
            let sample = seconds * SAMPLE_RATE as f64;
            if (sample - sample.round()).abs() < 1e-7 {
                sample.round()
            } else {
                sample.ceil()
            }
        };
        let first = boundary(start);
        let last = boundary(end);
        ensure!(
            first >= i64::MIN as f64 && last <= i64::MAX as f64,
            "audio: window exceeds one hour; render a shorter range"
        );
        self.render_samples(first as i64, last as i64, root)
    }

    fn render_samples(&self, first: i64, last: i64, root: Option<usize>) -> Result<Vec<Frame>> {
        let count = last
            .checked_sub(first)
            .context("audio window exceeds the sample clock")?;
        ensure!(
            (0..=48_000 * 3600).contains(&count),
            "audio: window exceeds one hour; render a shorter range"
        );
        let mut output = Vec::with_capacity(count as usize);
        let mut scratch = self.plan.sound_scratch();
        let base_gains: Vec<f32> = self
            .nodes
            .iter()
            .map(|n| n.node.audio.as_ref().map_or(1., |a| a.gain))
            .collect();
        let tile = (count as usize).clamp(1, TILE);
        let mut gains = crate::animation::GainTile::new(&self.sound, tile);
        let mut values = vec![[0.; 2]; self.audible.len() * tile];
        let mut reads = vec![VoiceRead::Silent; self.voices.len()];
        let mut direct = vec![None; self.audible.len()];
        let low = root.and_then(|node| self.audible.binary_search_by_key(&node, |n| n.node).ok()).unwrap_or(0);
        let mut readers: Vec<_> = self.sources.iter().map(|source| source.reader()).collect();
        // How many frames from its start each file has been read so far, the frames
        // themselves, and the blocks scheduled but not yet mixed.
        let mut needs = vec![0; self.sources.len()];
        let mut ends: Vec<_> = self.files.iter().map(|file| file.end.initial()).collect();
        let mut clips: Vec<Option<Arc<Clip>>> = vec![None; self.sources.len()];
        let mut decoded_through = vec![0; self.sources.len()];
        let mut blocks = Vec::new();
        let mut cursor = first;
        while cursor < last {
            // Schedule: each block is one exact evaluation, and says how far it reads each file.
            let mut scheduled = cursor;
            while scheduled < last && blocks.len() < BATCH {
                let block = self.plan.sound_block(
                    &self.sound,
                    &self.voices,
                    &self.remaps,
                    &self.paints,
                    scheduled,
                    last,
                    &mut scratch,
                    |voice| self.files[voice].movie.as_deref(),
                )?;
                for (voice, &i) in self.voices.iter().enumerate() {
                    let Some(clock) = block.clock(voice) else {
                        continue;
                    };
                    let rate = self.sources[self.files[voice].source].rate;
                    // A straight clock's extremes are its ends; a curved one is read at each sample.
                    let need = &mut needs[self.files[voice].source];
                    let mut read = |sample: i64| -> Result<()> {
                        if !clock.sounds(sample) {
                            return Ok(());
                        }
                        let speed = clock.speed(sample)?;
                        ensure!(
                            speed.is_finite()
                                && speed.abs() * rate as f64 / SAMPLE_RATE as f64 <= 256.,
                            "audio: node {:?}: resampling above 256 source samples per output sample is not built yet",
                            self.nodes[i].node.name
                        );
                        let position = clock.seconds(sample)?;
                        self.files[voice].end.cover(position, &mut ends[voice])?;
                        *need = (*need).max(Clip::reach(rate, position.min(ends[voice].end), speed));
                        Ok(())
                    };
                    if clock.straight() {
                        for window in clock.windows(scheduled..block.end()) {
                            read(window.start)?;
                            read(window.end - 1)?;
                        }
                    } else {
                        for sample in scheduled..block.end() {
                            read(sample)?;
                        }
                    }
                }
                scheduled = block.end();
                blocks.push(block);
            }
            // Drop a borrowed prefix only when it needs to grow. The decoder
            // writes new samples into the shared float buffer in place.
            for (source, reader) in readers.iter_mut().enumerate() {
                let need = needs[source];
                if need > decoded_through[source] {
                    clips[source] = None;
                    let clip = reader.prefix(need)?;
                    decoded_through[source] = if clip.frames.len() < need { usize::MAX } else { need };
                    clips[source] = Some(clip);
                }
            }
            let clip = |voice: usize| clips[self.files[voice].source].as_deref();
            for block in &blocks {
                for (voice, read) in reads.iter_mut().enumerate() {
                    *read = match (block.clock(voice), clip(voice)) {
                        (Some(clock), Some(clip)) =>
                            VoiceRead::new(clock, clip, ends[voice].end, cursor, block.end())?,
                        _ => VoiceRead::Silent,
                    };
                }
                for (node, pcm) in self.audible.iter().zip(&mut direct) {
                    *pcm = match (node.bus, &reads[node.voices.clone()]) {
                        (false, [VoiceRead::Aligned(start)]) => Some(*start as usize),
                        _ => None,
                    };
                }

                let mut first = cursor;
                while first < block.end() {
                    let len = (block.end() - first).min(tile as i64) as usize;
                    let offset = (first - cursor) as usize;
                    block.gains(&self.plan, &self.sound, first, len, &mut scratch, &base_gains, &mut gains)?;
                    for (slot, node) in self.audible.iter().enumerate() {
                        if direct[slot].is_some() { continue; }
                        let row = &mut values[slot * tile..slot * tile + len];
                        row.fill([0.; 2]);
                        for (index, voice) in node.voices.clone().enumerate() {
                            match reads[voice] {
                                VoiceRead::Silent => {}
                                VoiceRead::Aligned(start) => {
                                    let from = start as usize + offset;
                                    let frames = &clip(voice).unwrap().frames[from..from + len];
                                    if index == 0 {
                                        row.copy_from_slice(frames);
                                    } else {
                                        for (out, frame) in row.iter_mut().zip(frames) {
                                            out[0] += frame[0];
                                            out[1] += frame[1];
                                        }
                                    }
                                }
                                VoiceRead::Clocked => {
                                    let clock = block.clock(voice).unwrap();
                                    let clip = clip(voice).unwrap();
                                    for (j, out) in row.iter_mut().enumerate() {
                                        if let Some(frame) = VoiceRead::sample(clock, clip, ends[voice].end, first + j as i64)? {
                                            if index == 0 {
                                                *out = frame;
                                            } else {
                                                out[0] += frame[0];
                                                out[1] += frame[1];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let begin = output.len();
                    output.resize(begin + len, [0.; 2]);
                    let mixed = &mut output[begin..];
                    for (slot, node) in self.audible.iter().enumerate().skip(low).rev() {
                        let (ancestors, rows) = values.split_at_mut(slot * tile);
                        let row = match direct[slot] {
                            Some(start) => &clip(node.voices.start).unwrap().frames[start + offset..start + offset + len],
                            None => &rows[..len],
                        };
                        let only = root == Some(node.node);
                        let target = if only || (root.is_none() && node.parent.is_none()) {
                            Some(&mut *mixed)
                        } else {
                            node.parent.filter(|&parent| parent >= low)
                                .map(|parent| &mut ancestors[parent * tile..parent * tile + len])
                        };
                        let gain = gains.gain(&self.sound, node.node, len, base_gains[node.node]);
                        let valid = match gain {
                            crate::animation::Gain::Keyed(gain) => mix_row(row, gain.iter().copied(), target, only),
                            crate::animation::Gain::Constant(gain) => mix_row(row, std::iter::repeat(gain), target, only),
                        };
                        ensure!(valid, "audio: node {:?} has a nonfinite mix", self.nodes[node.node].node.name);
                    }
                    ensure!(mixed.iter().flatten().all(|v| v.is_finite()), "audio: nonfinite root mix; reduce source gains");
                    first += len as i64;
                }
                cursor = block.end();
            }
            blocks.clear();
        }
        Ok(output)
    }
}

fn has_audio(nodes: &[Node]) -> bool {
    nodes
        .iter()
        .any(|n| n.audio.is_some() || has_audio(&n.children))
}

pub(crate) fn may_have_audio(scene: &Scene) -> bool {
    has_audio(&scene.nodes) || scene.images.values().any(|image| image.file.as_ref().is_some_and(|file| {
        let path = Path::new(file);
        crate::footage::is_media(path) && !crate::footage::is_sequence(path)
    }))
}

pub(crate) struct MovieSound {
    wav: PathBuf,
    fps: vork_time::Time,
    first: u64,
}
impl Drop for MovieSound {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.wav);
    }
}
impl MovieSound {
    pub(crate) fn prepare(
        scene: &Scene,
        base: &Path,
        first: usize,
        last: usize,
        destinations: &[PathBuf],
        opened: &crate::footage::Opened,
    ) -> Result<Option<Self>> {
        if !may_have_audio(scene) {
            return Ok(None);
        }
        let mix = Mix::new_in(scene, base, opened)?;
        if mix.voices.is_empty() && !has_audio(&scene.nodes) {
            return Ok(None);
        }
        let root = scene
            .render
            .root
            .as_deref()
            .map(|n| mix.node(n))
            .transpose()?;
        let frames = mix.render_range(first, last, scene.render.fps, root)?;
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let wav = loop {
            let path = std::env::temp_dir().join(format!(
                "vork-audio-{}-{}.wav",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(_) => break path,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        };
        let sound = Self {
            wav,
            fps: scene.render.fps,
            first: first as u64,
        };
        vork_audio::write_wav(&sound.wav, &frames)?;
        let codec = if destinations
            .iter()
            .any(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("mp4")))
        {
            "aac"
        } else {
            "pcm_s16le"
        };
        let check = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(&sound.wav)
            .args(["-t", "0.01", "-c:a", codec, "-f", "null", "-"])
            .output()
            .context("audio: install ffmpeg to mux movie sound")?;
        ensure!(
            check.status.success(),
            "audio: {codec} encoding unavailable: {}",
            String::from_utf8_lossy(&check.stderr).trim()
        );
        Ok(Some(sound))
    }
    pub(crate) fn mux(&self, destinations: &[PathBuf], frames: u64) -> Result<()> {
        for path in destinations
            .iter()
            .filter(|p| crate::shot::video_destination(p))
        {
            crate::audio_output::mux_file(
                path,
                &self.wav,
                1.,
                0.,
                self.first..self.first + frames,
                self.fps,
            )?;
        }
        Ok(())
    }
}

/// Multiply each node after its own voices and children have been summed.
/// Visiting nodes in reverse order preserves the addition order at each bus.
fn mix_row(frames: &[Frame], gains: impl Iterator<Item = f32>, output: Option<&mut [Frame]>, replace: bool) -> bool {
    let mut finite = true;
    let frames = frames.iter().zip(gains).map(|(frame, gain)| frame.map(|value| value * gain));
    match output {
        Some(output) if replace => {
            for (out, frame) in output.iter_mut().zip(frames) {
                finite &= frame[0].is_finite() & frame[1].is_finite();
                *out = frame;
            }
        }
        Some(output) => {
            for (out, frame) in output.iter_mut().zip(frames) {
                finite &= frame[0].is_finite() & frame[1].is_finite();
                out[0] += frame[0];
                out[1] += frame[1];
            }
        }
        None => {
            for frame in frames {
                finite &= frame[0].is_finite() & frame[1].is_finite();
            }
        }
    }
    finite
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scene(nodes: serde_json::Value) -> Scene {
        Scene::from_json(
            &serde_json::to_vec(
                &serde_json::json!({"version":2,"composition":{"space":"2d"},"nodes":nodes}),
            )
            .unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn mix_reads_equal_length_replacement_with_preserved_mtime() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("sound.wav");
        let scene = scene(serde_json::json!([{"audio":{"file":"sound.wav"}}]));
        vork_audio::write_wav(&path, &[[0.25; 2]; 960])?;
        let render = || Mix::new(&scene, dir.path())?.render(0., 0.01, None);
        let original = render()?;

        let replacement = dir.path().join("replacement.wav");
        vork_audio::write_wav(&replacement, &[[-0.25; 2]; 960])?;
        replace_preserving_mtime(&path, &replacement)?;
        let updated = render()?;
        assert_eq!(original.len(), updated.len());
        assert!(original[100][0] > 0.2);
        assert!(updated[100][0] < -0.2);
        Ok(())
    }

    #[test]
    fn partial_ffmpeg_mix_releases_input_and_resumes_the_cached_revision() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let original = dir.path().join("original.wav");
        let path = dir.path().join("sound.flac");
        let renamed = dir.path().join("renamed.flac");
        let frames: Vec<_> = (0..96_000).map(|i| {
            let sample = ((i % 257) as f32 - 128.) / 256.;
            [sample, -sample]
        }).collect();
        vork_audio::write_wav(&original, &frames)?;
        ensure!(
            std::process::Command::new("ffmpeg").args(["-v", "error", "-i"])
                .arg(&original).arg(&path).status()?.success(),
            "encode test audio"
        );
        let scene = scene(serde_json::json!([{"audio":{"file":"sound.flac"}}]));
        let opened = crate::footage::Opened::default();
        let mix = Mix::new_in(&scene, dir.path(), &opened)?;
        let cached = &mix.sources[0];
        let first = mix.render(0., 0.1, None)?;
        assert_eq!(first, frames[..4_800]);
        let prefix = cached.prefix(1)?;
        assert!(prefix.frames.len() < frames.len());

        // Even a live Mix and its cached Source permit renaming the input.
        std::fs::rename(&path, &renamed)?;
        assert_eq!(mix.render(0., 0.1, None)?, first);
        assert!(Arc::ptr_eq(&prefix, &cached.prefix(1)?));
        let replacement = vec![[0.25, -0.25]; frames.len()];
        vork_audio::write_wav(&path, &replacement)?;
        let resumed = Mix::new_in(&scene, dir.path(), &opened.pins())?;
        assert!(Arc::ptr_eq(cached, &resumed.sources[0]));
        assert_eq!(resumed.render(0.1, 0.2, None)?, frames[4_800..9_600]);
        assert_eq!(Mix::new(&scene, dir.path())?.render(0.1, 0.2, None)?, replacement[4_800..9_600]);
        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[test]
    fn cached_silence_expires_when_equal_length_movie_gains_audio() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("clip.mov");
        let replacement = dir.path().join("replacement.mov");
        for (file, audio) in [(&path, false), (&replacement, true)] {
            let mut command = std::process::Command::new("ffmpeg");
            command.args(["-v", "error", "-f", "lavfi", "-i", "color=s=16x16:d=0.2"]);
            if audio {
                command.args([
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=duration=0.2",
                    "-c:a",
                    "pcm_s16le",
                ]);
            }
            ensure!(
                command
                    .args(["-c:v", "mpeg4"])
                    .arg(file)
                    .status()?
                    .success(),
                "create movie"
            );
        }
        let length = std::fs::metadata(&replacement)?.len();
        for file in [&path, &replacement] {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new().append(true).open(file)?;
            let size = u32::try_from(length - file.metadata()?.len() + 8)?;
            file.write_all(&size.to_be_bytes())?;
            file.write_all(b"free")?;
            file.write_all(&vec![0; size as usize - 8])?;
        }
        let scene = Scene::from_json(
            br#"{
            "version":2,"composition":{"space":"2d"},
            "images":{"clip":{"file":"clip.mov"}},
            "nodes":[{"type":"path","d":"M0 0H1V1Z","fill":{"image":"clip"}}]
        }"#,
        )?;
        let render = || Mix::new(&scene, dir.path())?.render(0., 0.1, None);
        assert!(render()?.iter().all(|f| *f == [0.; 2]));
        replace_preserving_mtime(&path, &replacement)?;
        assert!(render()?.iter().any(|f| f[0].abs() > 0.05));
        Ok(())
    }

    #[test]
    fn keyed_movie_paints_cut_sound_on_the_picture_sample() -> Result<()> {
        use serde_json::json;
        let dir = tempfile::tempdir()?;
        for (color, hz) in [("red", 440), ("blue", 880)] {
            let output = std::process::Command::new("ffmpeg")
                .args(["-v", "error", "-f", "lavfi", "-i"])
                .arg(format!("color=c={color}:s=128x64:r=4:d=1"))
                .args(["-f", "lavfi", "-i"])
                .arg(format!("sine=frequency={hz}:duration=1:sample_rate=48000"))
                .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0", "-c:a", "aac"])
                .arg(dir.path().join(format!("{color}.mp4")))
                .output()?;
            ensure!(output.status.success(), "create movie: {}", String::from_utf8_lossy(&output.stderr));
        }
        let paint = |name| json!({"image":name});
        let render_scene = |nodes, animation| {
            let scene = Scene::from_json(&serde_json::to_vec(&json!({
                "version":2, "composition":{"space":"2d"},
                "render":{"fps":4,"duration":1},
                "images":{"red":{"file":"red.mp4"},"blue":{"file":"blue.mp4"}},
                "nodes":nodes, "animation":animation
            }))?)?;
            Mix::new(&scene, dir.path())?.render_samples(0, 48_000, None)
        };
        let render = |nodes| render_scene(nodes, json!([]));
        let path = |property: &str, name| {
            let mut node = json!({"name":"cut","type":"path","d":"M0 0H128V64H0Z","fill":"none","stroke-width":2});
            node[property] = paint(name);
            node
        };
        let tone = |frames: &[Frame], hz: f64| {
            let (real, imag) = frames.iter().enumerate().fold((0., 0.), |(re, im), (i, frame)| {
                let phase = std::f64::consts::TAU * hz * i as f64 / f64::from(SAMPLE_RATE);
                let (sin, cos) = phase.sin_cos();
                (re + f64::from(frame[0]) * cos, im + f64::from(frame[0]) * sin)
            });
            2. * real.hypot(imag) / frames.len() as f64
        };
        let assert_tones = |frames: &[Frame], second_hz| {
            for (window, sounding, silent) in [
                (6_000..18_000, 440., 880.),
                (30_000..42_000, second_hz, 1320. - second_hz),
            ] {
                let frames = &frames[window];
                assert!(tone(frames, sounding) > 0.1, "missing {sounding} Hz");
                assert!(tone(frames, silent) < 0.002, "leaking {silent} Hz");
            }
        };
        let red = render(json!([path("fill", "red")]))?;
        let blue = render(json!([path("fill", "blue")]))?;
        assert_tones(&red, 440.);
        for (property, interpolation, start, end) in [
            ("fill", "STEP", 0., 0.5),
            ("fill", "LINEAR", 0.25, 0.75),
            ("stroke", "STEP", 0., 0.5),
        ] {
            let mut cut = path(property, "red");
            cut["animation"] = json!([{"tracks":[{"path":[property],"interpolation":interpolation,
                "keys":[[start,paint("red")],[end,paint("blue")]]}]}]);
            let mixed = render(json!([cut]))?;
            assert_tones(&mixed, 880.);
            assert_eq!(&mixed[..24_000], &red[..24_000], "{property} {interpolation}");
            assert_eq!(&mixed[24_000..], &blue[24_000..], "{property} {interpolation}");
        }

        // All three authorings select exactly the same movie at each sample.
        // Compare encoded WAVs as well as the two halves against the movie control.
        let width = json!({"path":["stroke-width"],"interpolation":"STEP","keys":[[0,0],[0.5,4]]});
        let redundant = json!({"path":["fill"],"keys":[[0,"white"],[1,"white"]]});
        let mut stroke = path("stroke", "red");
        stroke["fill"] = json!("white");
        stroke["stroke-width"] = json!(0);
        assert!(render(json!([stroke]))?.iter().all(|frame| *frame == [0.; 2]));
        for (value, control) in [(0, vec![[0.; 2]; red.len()]), (4, red.clone())] {
            stroke["stroke-width"] = json!(value);
            let expected = dir.path().join("static.wav");
            vork_audio::write_wav(&expected, &control)?;
            let expected = std::fs::read(expected)?;
            let width = json!({"path":["stroke-width"],"keys":[[0,value],[1,value]]});
            for tracks in [json!([]), json!([width]), json!([width, redundant])] {
                stroke["animation"] = if tracks.as_array().unwrap().is_empty() { json!([]) } else { json!([{"tracks":tracks}]) };
                let samples = render(json!([stroke]))?;
                assert_eq!(samples, control);
                let file = dir.path().join("variant.wav");
                vork_audio::write_wav(&file, &samples)?;
                assert_eq!(std::fs::read(file)?, expected);
            }
        }
        let mut expected = vec![[0.; 2]; 24_000];
        expected.extend_from_slice(&red[24_000..]);
        let wav = dir.path().join("selection.wav");
        vork_audio::write_wav(&wav, &expected)?;
        let bytes = std::fs::read(&wav)?;
        for tracks in [json!([width]), json!([width, redundant])] {
            stroke["animation"] = json!([{"tracks":tracks}]);
            let frames = render(json!([stroke]))?;
            assert_eq!(frames, expected);
            vork_audio::write_wav(&wav, &frames)?;
            assert_eq!(std::fs::read(&wav)?, bytes);
        }
        // A constant key gets one ordinary sound block for the entire export.
        let mut node = path("fill", "red");
        node["stroke"] = json!("white");
        node["animation"] = json!([{"tracks":[{"path":["stroke"],"keys":[[0,"white"],[3,"white"]]}]}]);
        let scene = Scene::from_json(&serde_json::to_vec(&json!({
            "version":2, "composition":{"space":"2d"}, "images":{"red":{"file":"red.mp4"}},
            "nodes":[node, {"audio":{"file":"red.mp4"}}]
        }))?)?;
        let mix = Mix::new(&scene, dir.path())?;
        let block = mix.plan.sound_block(&mix.sound, &mix.voices, &mix.remaps, &mix.paints,
            0, 192_000, &mut mix.plan.sound_scratch(), |i| mix.files[i].movie.as_deref())?;
        assert_eq!(block.end(), 192_000);
        assert_eq!(mix.render_samples(0, 48_000, None)?, red.iter().map(|f| f.map(|v| v + v)).collect::<Vec<_>>());

        let unrelated_text = json!({"type":"text","content":"Picture only","font":"missing.ttf"});
        let mixed = render_scene(json!([path("fill", "red"), unrelated_text]), json!([
            {"tracks":[{"path":["animation","paint","weight"],"keys":[[0,0],[1,1]]}]},
            {"name":"paint", "tracks":[{"path":["nodes","cut","fill"],"keys":[[0,paint("blue")]]}]}
        ]))?;
        assert_eq!(&mixed[..24_000], &red[..24_000]);
        assert_eq!(&mixed[24_000..], &blue[24_000..]);

        let font = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        let text = json!({"name":"text","type":"text","content":"A","font":font,
            "fill":{"image":"red"}});
        for (track, keys) in [
            (json!(["content"]), json!([[0,"A"],[0.5,[{"content":"A","fill":paint("blue")}]]])),
            (json!(["glyphs",0,"fill"]), json!([[0,paint("red")],[0.5,paint("blue")]])),
        ] {
            let mut text = text.clone();
            text["animation"] = json!([{"tracks":[{"path":track,"interpolation":"STEP","keys":keys}]}]);
            let mixed = render(json!([text]))?;
            assert_eq!(&mixed[..24_000], &red[..24_000]);
            assert_eq!(&mixed[24_000..], &blue[24_000..]);
        }

        // The parent and child each shift the clock. Compare against the single-movie
        // controls on those same clocks to check the source phase as well as the cut.
        let nested = |mut node: serde_json::Value| {
            node["active"] = json!([0.125,0.625]);
            json!([{"name":"parent","active":[0.125,0.875],"children":[node]}])
        };
        let nested_red = render(nested(path("fill", "red")))?;
        let nested_blue = render(nested(path("fill", "blue")))?;
        let mut cut = path("fill", "red");
        cut["animation"] = json!([{"tracks":[{"path":["fill"],"interpolation":"STEP",
            "keys":[[0,paint("red")],[0.2500625,paint("blue")]]}]}]);
        let mixed = render(nested(cut))?;
        assert_eq!(&mixed[..24_003], &nested_red[..24_003]);
        assert_eq!(&mixed[24_003..], &nested_blue[24_003..]);
        assert!(mixed[..12_000].iter().chain(&mixed[36_000..]).all(|frame| *frame == [0.; 2]));
        Ok(())
    }

    fn replace_preserving_mtime(path: &Path, replacement: &Path) -> Result<()> {
        let before = std::fs::metadata(path)?;
        std::fs::File::options()
            .write(true)
            .open(replacement)?
            .set_modified(before.modified()?)?;
        std::fs::rename(replacement, path)?;
        let after = std::fs::metadata(path)?;
        assert_eq!(before.len(), after.len());
        assert_eq!(before.modified()?, after.modified()?);
        Ok(())
    }

    #[test]
    fn contiguous_frame_ranges_tile_the_mix() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("vork-range-samples-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let source: Vec<_> = (0..10_000).map(|i| [i as f32 / 10_000.; 2]).collect();
        vork_audio::write_wav(&dir.join("sound.wav"), &source)?;
        let scene = scene(serde_json::json!([{"audio":{"file":"sound.wav"}}]));
        let mix = Mix::new(&scene, &dir)?;
        for (fps, lengths) in [
            (vork_time::Time::new(30_000, 1001)?, [1602, 1601, 1602]),
            (vork_time::Time::integer(24), [2000; 3]),
        ] {
            let mut joined = Vec::new();
            for (frame, length) in lengths.into_iter().enumerate() {
                let chunk = mix.render_range(frame, frame, fps, None)?;
                assert_eq!(chunk.len(), length);
                joined.extend(chunk);
            }
            assert_eq!(joined, mix.render_range(0, 2, fps, None)?);
        }
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn delayed_movie_and_aligned_wav_share_trim_loop_and_keyed_speed() -> Result<()> {
        use serde_json::json;
        let dir = std::env::temp_dir().join(format!("vork-movie-offset-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let source: Vec<_> = (0..48_000)
            .map(|i| {
                [if (12_000..13_000).contains(&i) {
                    0.25
                } else {
                    0.
                }; 2]
            })
            .collect();
        vork_audio::write_wav(&dir.join("pulse.wav"), &source)?;
        let mut aligned = vec![[0.; 2]; 12_000];
        aligned.extend(source);
        vork_audio::write_wav(&dir.join("aligned.wav"), &aligned)?;
        let status = std::process::Command::new("ffmpeg")
            .current_dir(&dir)
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=s=16x16:r=24:d=2",
                "-itsoffset",
                "0.25",
                "-i",
                "pulse.wav",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "mpeg4",
                "-c:a",
                "pcm_f32le",
                "delayed.mov",
            ])
            .status()?;
        ensure!(status.success(), "create timestamped movie");
        let mut doc = json!({"version":2, "composition":{"space":"2d"},
            "images":{"movie":{"file":"delayed.mov"}},
            "nodes":[{"type":"path","name":"clip","d":"M0 0H1V1Z","fill":{"image":"movie"},
                "time":{"offset":"1/4","duration":"3/4","loop":true}}],
            "animation":[{"tracks":[{"path":["nodes","clip","time","speed"],"interpolation":"STEP",
                "keys":[[0,1.5],[1,0.75]]}]}]});
        let render = |doc: &serde_json::Value| -> Result<Vec<Frame>> {
            let scene = Scene::from_json(&serde_json::to_vec(doc)?)?;
            Mix::new(&scene, &dir)?.render(0., 2., None)
        };
        let movie = render(&doc)?;
        assert!(movie.iter().filter(|f| f[0] > 0.2).count() > 1000);
        doc["nodes"][0]["fill"] = json!("none");
        doc["nodes"][0]["audio"] = json!({"file":"aligned.wav"});
        assert_eq!(movie, render(&doc)?);
        doc["nodes"][0]["audio"]["file"] = json!("delayed.mov");
        assert_eq!(movie, render(&doc)?);
        doc["nodes"][0]["fill"] = json!({"image":"movie"});
        assert_eq!(
            render(&doc)?,
            movie.iter().map(|v| v.map(|x| x + x)).collect::<Vec<_>>()
        );
        doc["nodes"][0]["audio"]["gain"] = json!(0);
        assert!(render(&doc)?.iter().all(|v| *v == [0.; 2]));
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn asset_export_never_silently_drops_sound() -> Result<()> {
        let base = std::env::temp_dir().join(format!("vork-export-sound-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        let path = base.join("scene.json");
        std::fs::write(
            &path,
            br#"{"version":2,"nodes":[{"name":"speaker","audio":{"gain":0.5},"type":"mesh","geometry":"g"}],"geometry":{"g":{"attributes":{"POSITION":"p"}}},"buffers":{"p":{"type":"vec3","data":[[0,0,0],[1,0,0],[0,1,0]]}}}"#,
        )?;
        let mut export = crate::export::Export {
            scene: path,
            out: base.join("scene.glb"),
            allow_losses: vec!["environment_lighting".into()],
            ..Default::default()
        };
        assert!(export.run().is_err());
        assert!(!export.out.exists());
        export.allow_losses.push("audio".into());
        let report = export.run()?;
        assert!(
            report["issues"]
                .as_array()
                .unwrap()
                .iter()
                .any(|i| i["code"] == "audio")
        );
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
    #[test]
    fn wav_shot_needs_neither_camera_nor_render_device() -> Result<()> {
        let base = std::env::temp_dir().join(format!("vork-wav-shot-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        let source: Vec<_> = (0..4800).map(|i| [i as f32 / 4800.; 2]).collect();
        vork_audio::write_wav(&base.join("source.wav"), &source)?;
        let path = base.join("scene.json");
        std::fs::write(&path, br#"{"version":2,"render":{"duration":0.1,"fps":30},"nodes":[{"audio":{"file":"source.wav"}}]}"#)?;
        let output = base.join("result.wav");
        let shot = crate::shot::Shot {
            scene: path,
            out: vec![output.clone().into()],
            outputs: Vec::new(),
            frame_start: Some(1),
            frame_end: Some(2),
            ..Default::default()
        }
        .render()?;
        assert_eq!(shot.report["samples"], 3200);
        assert!(shot.into_retained().is_none());
        assert_eq!(Clip::load(&output)?.frames, source[1600..]);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
    #[cfg(unix)]
    #[test]
    fn a_failed_wav_rerender_withdraws_the_previous_completion_record() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        // audit-core 9: WAV delivery takes the output run picture delivery takes.
        let base = std::env::temp_dir().join(format!("vork-wav-run-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base)?;
        vork_audio::write_wav(&base.join("source.wav"), &vec![[0.25; 2]; 4800])?;
        let path = base.join("scene.json");
        std::fs::write(&path, br#"{"version":2,"render":{"duration":0.1,"fps":30},"nodes":[{"audio":{"file":"source.wav"}}]}"#)?;
        let first = base.join("out").join("a.wav");
        let render = |out: Vec<std::path::PathBuf>, frame_end: usize| {
            crate::shot::Shot { scene: path.clone(), out: out.into_iter().map(Into::into).collect(), outputs: Vec::new(), frame_start: Some(0), frame_end: Some(frame_end), ..Default::default() }.render().map(|_| ())
        };
        let meta = crate::shot::sidecar(&first, "meta.json");
        let timings = crate::shot::sidecar(&first, "timings.json");
        render(vec![first.clone()], 2)?;
        assert!(meta.exists() && timings.exists());
        assert!(!crate::shot::sidecar(&first, "meta.json.pending").exists());
        assert_eq!(Clip::load(&first)?.frames.len(), 4800);
        // A rerender of different content whose second destination cannot be written: its
        // directory admits the run (the lock file is there) but no new file.
        let locked = base.join("locked");
        std::fs::create_dir_all(&locked)?;
        std::fs::write(locked.join(".vork-output.lock"), b"")?;
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o555))?;
        let second = locked.join("b.wav");
        let failed = render(vec![first.clone(), second.clone()], 0);
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755))?;
        let error = failed.map_err(|e| format!("{e:#}")).unwrap_err();
        assert!(!error.contains("remove previous output") && !error.contains("in use"), "the run began: {error}");
        assert_eq!(Clip::load(&first)?.frames.len(), 1600, "the first destination was replaced by the new WAV");
        assert!(!second.exists(), "the second WAV was not written");
        assert!(!meta.exists(), "the replaced WAV carries no completion record from the earlier render");
        assert!(!crate::shot::sidecar(&second, "meta.json").exists());
        // A destination that cannot be replaced fails before any data changes, and also
        // leaves no completion record.
        render(vec![first.clone()], 2)?;
        assert!(meta.exists());
        let blocked = base.join("out").join("c.wav");
        std::fs::create_dir_all(blocked.join("kept"))?;
        assert!(render(vec![first.clone(), blocked.clone()], 0).is_err());
        assert!(!meta.exists());
        // One destination named twice is one namespace, as for pictures.
        let error = render(vec![first.clone(), first.clone()], 2).map_err(|e| format!("{e:#}")).unwrap_err();
        assert!(error.contains("overlapping output namespaces"), "{error}");
        // A retry that can finish publishes again.
        render(vec![first.clone()], 2)?;
        assert!(first.exists() && meta.exists() && timings.exists());
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&std::fs::read(&meta)?)?["kind"], "audio");
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
    #[test]
    fn canvas_audio_readback_uses_the_audible_subtree() -> Result<()> {
        let base = std::env::temp_dir().join(format!("vork-ask-sound-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        vork_audio::write_wav(&base.join("constant.wav"), &vec![[0.5; 2]; 1024])?;
        let doc = scene(
            serde_json::json!([{"name":"bus","audio":{"gain":0.5},"children":[{"audio":{"file":"constant.wav"}}]}]),
        );
        let answer = crate::ask::scene(
            doc,
            &base,
            &serde_json::json!([
                {"audio":"bus","read":"loudness","window":[0,0.01]},
                {"audio":"bus","window":[0,0.0000625]},
                {"audio":"bus","read":"spectrum","window":[0,0.021333333333333333]}
            ]),
        )?;
        assert_eq!(answer[0]["rms"], 0.25);
        assert_eq!(answer[0]["peak"], 0.25);
        assert_eq!(answer[1]["samples"].as_array().unwrap().len(), 3);
        assert!((answer[2]["magnitudes"][0].as_f64().unwrap() - 0.25).abs() < 1e-9);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
    #[test]
    fn picture_only_clocks_stay_out_of_the_sound_plan() -> Result<()> {
        let base = std::env::temp_dir().join(format!("vork-mix-plan-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        let frames: Vec<_> = (0..4800).map(|i| [i as f32 / 4800.; 2]).collect();
        vork_audio::write_wav(&base.join("long.wav"), &frames)?;
        let mut nodes: Vec<_> = (0..8)
            .map(|i| serde_json::json!({"name":format!("shape{i}"),"time":{"start":0.001},"children":[{"active":[0,0.02]}]}))
            .collect();
        nodes.push(serde_json::json!({"name":"song","audio":{"file":"long.wav"}}));
        let mix = Mix::new(&scene(serde_json::Value::Array(nodes)), &base)?;
        assert_eq!(mix.sound.nodes(), [16]);
        assert!(mix.sound.spans());
        assert_eq!(mix.render(0., 0.01, None)?, frames[..480]);
        // Only what the window reads was decoded, and a later window continues from there.
        let decoded = |at: usize| Ok::<_, anyhow::Error>(mix.sources[0].prefix(at)?.frames.len());
        assert!((480..600).contains(&decoded(1)?), "{}", decoded(1)?);
        assert_eq!(mix.render(0.05, 0.06, None)?, frames[2400..2880]);
        assert!((2880..3000).contains(&decoded(1)?), "{}", decoded(1)?);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
    #[test]
    fn a_sounding_aligned_block_reads_pcm_as_a_slice() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let frames: Vec<_> = (0..1024)
            .map(|i| [(i as f32 * 0.13).sin(), (i as f32 * 0.73).cos()])
            .collect();
        vork_audio::write_wav(&dir.path().join("voice.wav"), &frames)?;
        let doc = scene(serde_json::json!([
            {"name":"aligned","time":{"start":"1/48000"},"audio":{"file":"voice.wav"}},
            {"name":"fractional","time":{"start":"1/96000"},"audio":{"file":"voice.wav"}}
        ]));
        let mix = Mix::new(&doc, dir.path())?;
        let mut scratch = mix.plan.sound_scratch();
        let first = 2;
        let block = mix.plan.sound_block(
            &mix.sound, &mix.voices, &mix.remaps, &mix.paints, first, 400, &mut scratch,
            |_| None,
        )?;
        let last = block.end();
        assert_eq!(last, 400);
        let clip = mix.sources[0].prefix(usize::MAX)?;
        let aligned = block.clock(0).unwrap();
        let fast = VoiceRead::new(aligned, &clip, f64::INFINITY, first, last)?;
        let VoiceRead::Aligned(start) = fast else {
            panic!("a forward sample-aligned block must borrow its PCM");
        };
        let slice = clip.aligned_frames(start, (last - first) as usize).unwrap();
        assert!(std::ptr::eq(slice, &clip.frames[1..399]));
        for (offset, sample) in (first..last).enumerate() {
            assert_eq!(
                slice[offset].map(f32::to_bits),
                clip.sample(aligned.seconds(sample)?, aligned.speed(sample)?).map(f32::to_bits),
            );
        }
        assert!(matches!(
            VoiceRead::new(block.clock(1).unwrap(), &clip, f64::INFINITY, first, last)?,
            VoiceRead::Clocked,
        ));
        let cutoff = aligned.seconds(last - 1)?;
        let cut = VoiceRead::new(aligned, &clip, cutoff, first, last)?;
        assert!(matches!(cut, VoiceRead::Clocked));
        assert!(VoiceRead::sample(aligned, &clip, cutoff, last - 1)?.is_none());
        Ok(())
    }
    #[test]
    fn blocks_read_what_an_exact_evaluation_reads_at_every_sample() -> Result<()> {
        let base = std::env::temp_dir().join(format!("vork-mix-blocks-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        let frames: Vec<_> = (0..480)
            .map(|i| [(i as f32 * 0.37).sin(), (i as f32 * 0.11).cos()])
            .collect();
        vork_audio::write_wav(&base.join("tone.wav"), &frames)?;
        // A window, a loop, a reversal, a resampled speed, a keyed existence, and gain keyed three
        // ways: a scene ramp, a scene step and a looping layer on the bus's own clock.
        let doc = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2,"composition":{"space":"2d"},
            "nodes":[
                {"name":"bus","audio":{"gain":0.5},"time":{"start":0.0003,"duration":0.0211,"loop":true},
                 "animation":[{"loop":true,"tracks":[{"path":["children","looped","audio","gain"],"keys":[[0,0.25],[0.0031,1]]}]}],
                 "children":[
                    {"name":"looped","time":{"duration":0.0041,"loop":true},"audio":{"file":"tone.wav"}},
                    {"name":"late","active":[0.0101,0.0173],"time":{"speed":-1},"audio":{"file":"tone.wav","gain":2}}]},
                {"name":"ramped","time":{"start":0.0007,"speed":0.75},"audio":{"file":"tone.wav"}},
                {"name":"stepped","active":[0.002,0.027],"audio":{"file":"tone.wav"}},
                {"name":"flag","audio":{"file":"tone.wav"}}],
            "animation":[{"tracks":[
                {"path":["nodes","ramped","audio","gain"],"keys":[[0,0],[0.02,1]]},
                {"path":["nodes","stepped","audio","gain"],"keys":[[0,1],[0.01251,0.25]],"interpolation":"STEP"},
                {"path":["nodes","flag","active"],"keys":[[0,false],[0.0052,true],[0.0199,false]],"interpolation":"STEP"}]}]
        }))?)?;
        let mix = Mix::new(&doc, &base)?;
        assert!(mix.sound.spans());
        let whole = mix.render(0., 0.03, None)?;
        assert_eq!(whole.len(), 1440);
        assert!(whole.iter().any(|frame| frame[0] != 0.));
        for (sample, frame) in whole.iter().enumerate() {
            let one = mix.render(
                sample as f64 / SAMPLE_RATE as f64,
                (sample + 1) as f64 / SAMPLE_RATE as f64,
                None,
            )?;
            assert_eq!(one.len(), 1);
            assert_eq!(one[0].map(f32::to_bits), frame.map(f32::to_bits), "sample {sample}");
        }
        // Keyed speeds: a ramp from a standing start with a clocked, windowed child and a gain
        // layer on its own clock; a step speed that stands for a while; a ramp read in scene
        // seconds under a retimed parent; a ramp keyed by the parent. Blocks carry each as the
        // polynomial through exact evaluations; one-sample renders are those evaluations.
        let keyed = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2,"composition":{"space":"2d"},
            "nodes":[
                {"name":"bent","time":{"duration":0.02},"audio":{"file":"tone.wav"},
                 "animation":[{"tracks":[{"path":["audio","gain"],"keys":[[0,0.5],[0.005,1]]}]}],
                 "children":[{"name":"under","active":[0.002,0.008],"time":{"start":0.001,"speed":0.5},"audio":{"file":"tone.wav"}}]},
                {"name":"stepper","time":{"duration":0.02},"audio":{"file":"tone.wav"}},
                {"name":"shot","time":{"speed":2,"duration":0.05},"children":[
                    {"name":"inner","time":{"duration":0.02},"audio":{"file":"tone.wav"}}]},
                {"name":"owner","time":{"start":0.0005,"duration":0.03},
                 "animation":[{"tracks":[
                    {"path":["children","kid","time","speed"],"keys":[[0,1],[0.01,0.25]]},
                    {"path":["children","kid","audio","gain"],"keys":[[0,0],[0.01,1]]}]}],
                 "children":[{"name":"kid","time":{"duration":0.02},"audio":{"file":"tone.wav"}}]}],
            "animation":[{"tracks":[
                {"path":["nodes","bent","time","speed"],"keys":[[0,0],[0.004,1.5],[0.011,0.5]]},
                {"path":["nodes","stepper","time","speed"],"keys":[[0,0.5],[0.003,0],[0.006,2]],"interpolation":"STEP"},
                {"path":["nodes","inner","time","speed"],"keys":[[0,1],[0.02,3]]}]}]
        }))?)?;
        let mix = Mix::new(&keyed, &base)?;
        assert!(mix.sound.spans());
        let whole = mix.render(0., 0.02, None)?;
        assert_eq!(whole.len(), 960);
        assert!(whole.iter().any(|frame| frame[0] != 0.));
        for (sample, frame) in whole.iter().enumerate() {
            let one = mix.render(
                sample as f64 / SAMPLE_RATE as f64,
                (sample + 1) as f64 / SAMPLE_RATE as f64,
                None,
            )?;
            assert_eq!(one[0].map(f32::to_bits), frame.map(f32::to_bits), "keyed sample {sample}");
        }
        assert_eq!(mix.render(0.002, 0.004, None)?, whole[96..192]);
        // A clock that stands for an instant silences what is below it at that instant, even a
        // child whose own rate is not zero there: the ramp starts from rest at sample 0.
        let rest = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2,"composition":{"space":"2d"},
            "nodes":[{"name":"ramp","time":{"duration":0.05},
                "animation":[{"tracks":[{"path":["children","rider","time","speed"],"keys":[[0,1],[0.01,1]]}]}],
                "children":[{"name":"rider","time":{"duration":0.05},"audio":{"file":"tone.wav"}}]}],
            "animation":[{"tracks":[{"path":["nodes","ramp","time","speed"],"keys":[[0,0],[0.01,1]]}]}]
        }))?)?;
        let mix = Mix::new(&rest, &base)?;
        let whole = mix.render(0., 0.01, None)?;
        assert_eq!(whole[0], [0.; 2]);
        assert!(whole[1][1] != 0.);
        assert_eq!(mix.render(0., 1. / SAMPLE_RATE as f64, None)?, whole[..1]);
        for sample in [1, 2, 3, 100, 479] {
            let one = mix.render(sample as f64 / SAMPLE_RATE as f64, (sample + 1) as f64 / SAMPLE_RATE as f64, None)?;
            assert_eq!(one[0].map(f32::to_bits), whole[sample].map(f32::to_bits), "rider sample {sample}");
        }
        // Three ramps deep: a degree-eight clock, still carried in blocks where it fits.
        let deep = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2,"composition":{"space":"2d"},
            "nodes":[{"name":"p","time":{"duration":1},
                "animation":[{"tracks":[{"path":["children","q","time","speed"],"keys":[[0,0.5],[1,2]]}]}],
                "children":[{"name":"q","time":{"duration":1},
                    "animation":[{"tracks":[{"path":["children","r","time","speed"],"keys":[[0,0.5],[1,2]]}]}],
                    "children":[{"name":"r","time":{"duration":1},"audio":{"file":"tone.wav"}}]}]}],
            "animation":[{"tracks":[{"path":["nodes","p","time","speed"],"keys":[[0,0.5],[1,2]]}]}]
        }))?)?;
        let mix = Mix::new(&deep, &base)?;
        let whole = mix.render(0., 0.05, None)?;
        assert!(whole.iter().any(|frame| frame[0] != 0.));
        for sample in (0..2400).step_by(7) {
            let one = mix.render(sample as f64 / SAMPLE_RATE as f64, (sample + 1) as f64 / SAMPLE_RATE as f64, None)?;
            assert_eq!(one[0].map(f32::to_bits), whole[sample].map(f32::to_bits), "deep sample {sample}");
        }
        // A cubic speed is read exactly at every sample; a window of it matches the whole.
        let cubic = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2,"composition":{"space":"2d"},
            "nodes":[{"name":"bent","time":{"duration":0.01},"audio":{"file":"tone.wav"}}],
            "animation":[{"tracks":[{"path":["nodes","bent","time","speed"],"keys":[[0,1,1,1],[0.01,1,1,1]],"interpolation":"CUBICSPLINE"}]}]
        }))?)?;
        let mix = Mix::new(&cubic, &base)?;
        // More one-sample blocks than one batch schedules.
        assert!(mix.render(0., 0.1, None)?.len() > BATCH);
        let whole = mix.render(0., 0.01, None)?;
        assert!(whole.iter().any(|frame| frame[0] != 0.));
        assert_eq!(mix.render(0.002, 0.004, None)?, whole[96..192]);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
    #[test]
    fn nested_clocks_gain_active_reverse_and_range() -> Result<()> {
        let base = std::env::temp_dir().join(format!("vork-mix-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        let frames: Vec<_> = (0..480).map(|i| [i as f32 / 480.; 2]).collect();
        vork_audio::write_wav(&base.join("ramp.wav"), &frames)?;
        let doc = scene(serde_json::json!([
            {"name":"bus","audio":{"gain":0.5},"children":[
                {"name":"source","visible":false,"active":[0.01,0.03],"time":{"speed":-1},"audio":{"file":"ramp.wav","gain":2}}
            ]}
        ]));
        let mix = Mix::new(&doc, &base)?;
        let all = mix.render(0., 0.03, None)?;
        assert!(all[..480].iter().all(|x| *x == [0.; 2]));
        assert!((all[480][0] - frames[479][0]).abs() < 1e-6);
        assert!((all[481][0] - frames[478][0]).abs() < 1e-6);
        assert!(all[960..].iter().all(|x| *x == [0.; 2]));
        assert_eq!(mix.render(0.012, 0.018, None)?, all[576..864]);
        let forward = scene(
            serde_json::json!([{"time":{"duration":0.005},"children":[{"audio":{"file":"ramp.wav"}}]}]),
        );
        let forward = Mix::new(&forward, &base)?.render(0., 0.01, None)?;
        assert!(forward[240..].iter().all(|x| *x == [0.; 2]));
        let gain = scene(
            serde_json::json!([{"name":"source","audio":{"file":"ramp.wav"},"animation":[{"tracks":[{"path":["audio","gain"],"keys":[[0,0],[0.01,1]]}]}]}]),
        );
        let gain = Mix::new(&gain, &base)?.render(0., 0.01, None)?;
        assert!((gain[240][0] - 0.25).abs() < 1e-6);
        let held = scene(
            serde_json::json!([{"time":{"speed":0},"children":[{"audio":{"file":"ramp.wav"}}]}]),
        );
        assert!(
            Mix::new(&held, &base)?
                .render(0., 0.01, None)?
                .iter()
                .all(|x| *x == [0.; 2])
        );
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
}
