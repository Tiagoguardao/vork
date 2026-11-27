//! CPU sound samples and analysis. No rendering device or output-frame clock.
use anyhow::{Context, Result, ensure};
use std::{
    io::{BufReader, Read},
    path::Path,
    process::{ChildStdout, Stdio},
    sync::{Arc, Mutex},
};
pub const SAMPLE_RATE: u32 = 48_000;
pub type Frame = [f32; 2];

#[derive(Clone)]
pub struct Clip {
    pub rate: u32,
    pub frames: Vec<Frame>,
}

/// The shared, decoded prefix of a sound file. Readers own the active decoders. The source pins their input revision.
pub struct Source {
    revision: Arc<vork_media::MediaRevision>,
    pub rate: u32,
    channels: u32,
    picture: Option<PictureTimeline>,
    /// Sample frames in the file, when its header says: a WAV does, a decoder's stream does not.
    length: Option<usize>,
    decoded: Mutex<Decoded>,
}
struct PictureTimeline {
    origin: f64,
    duration: Option<f64>,
    /// Packet timestamps can round by one container tick, plus one output sample on rescaling.
    tolerance: f64,
}
struct Decoded {
    /// The frames decoded so far, from the first.
    clip: Arc<Clip>,
    whole: bool,
}

/// Decoding state for one mix, opened only when the mix needs new samples.
pub struct Reader<'a> {
    source: &'a Source,
    stream: Option<Stream>,
    position: usize,
}
enum Stream {
    Wav(Wav),
    External(External),
}
struct Wav {
    input: BufReader<vork_media::MediaReader>,
    spec: hound::WavSpec,
    sample_bytes: usize,
    length: usize,
}

impl Wav {
    fn open(input: vork_media::MediaReader) -> hound::Result<Self> {
        let reader = hound::WavReader::new(BufReader::new(input))?;
        let spec = reader.spec();
        let length = reader.duration() as usize;
        let samples = reader.len();
        let mut input = reader.into_inner();
        // Hound leaves the input at the PCM payload, immediately after its
        // four-byte chunk length. Recover the stored width, which can include
        // padding beyond the valid bits reported in the WAV spec.
        let sample_bytes = if samples > 0 {
            let mut size = [0; 4];
            input.seek_relative(-4)?;
            input.read_exact(&mut size)?;
            (u32::from_le_bytes(size) / samples) as usize
        } else {
            0
        };
        Ok(Self {
            input,
            spec,
            sample_bytes,
            length,
        })
    }

    fn frame_bytes(&self) -> usize {
        self.sample_bytes * usize::from(self.spec.channels)
    }

    fn append(&mut self, output: &mut Vec<Frame>, wanted: usize) -> Result<()> {
        use hound::SampleFormat::{Float, Int};
        if wanted == 0 {
            return Ok(());
        }
        let spec = self.spec;
        let channels = usize::from(spec.channels);
        match (spec.sample_format, self.sample_bytes, spec.bits_per_sample) {
            (Int, 1, 8) => self.read(output, wanted, channels, |[x]| {
                f32::from(i16::from(x) - 128) * (1. / 128.)
            }),
            (Int, 2, 16) => self.read(output, wanted, channels, |bytes| {
                f32::from(i16::from_le_bytes(bytes)) * (1. / 32_768.)
            }),
            (Int, 3, 24) => self.read(output, wanted, channels, |[a, b, c]| pcm24(a, b, c)),
            (Int, 4, 24) => self.read(output, wanted, channels, |[a, b, c, _]| pcm24(a, b, c)),
            (Int, 4, 32) => self.read(output, wanted, channels, |bytes| {
                // Scaling by this power of two commutes with f32 rounding.
                i32::from_le_bytes(bytes) as f32 * (1. / 2_147_483_648.)
            }),
            (Float, 4, 32) => {
                let first = output.len();
                self.read(output, wanted, channels, f32::from_le_bytes)?;
                ensure!(
                    output[first..].iter().flatten().all(|x| x.is_finite()),
                    "audio: nonfinite source sample"
                );
                Ok(())
            }
            _ => Err(hound::Error::Unsupported.into()),
        }
    }

    fn read<const N: usize>(
        &mut self,
        output: &mut Vec<Frame>,
        mut wanted: usize,
        channels: usize,
        sample: impl Fn([u8; N]) -> f32,
    ) -> Result<()> {
        let mut bytes = [0; 32 * 1024];
        let stride = N * channels;
        output.reserve(wanted);
        while wanted > 0 {
            let count = wanted.min(bytes.len() / stride);
            let chunk = &mut bytes[..count * stride];
            self.input.read_exact(chunk)?;
            if channels == 1 {
                output.extend(chunk.as_chunks::<N>().0.iter().map(|&raw| [sample(raw); 2]));
            } else {
                output.extend(
                    chunk
                        .as_chunks::<N>()
                        .0
                        .chunks_exact(2)
                        .map(|pair| [sample(pair[0]), sample(pair[1])]),
                );
            }
            wanted -= count;
        }
        Ok(())
    }
}

fn pcm24(a: u8, b: u8, c: u8) -> f32 {
    // The signed sample occupies the low 24 bits, including in the padded layout.
    i32::from_le_bytes([0, a, b, c]) as f32 * (1. / 2_147_483_648.)
}

/// The user's decoder, blocked on its output pipe between reads.
struct External {
    child: vork_media::DecoderProcess,
    stdout: ChildStdout,
    errors: Option<std::thread::JoinHandle<Vec<u8>>>,
}
impl Source {
    fn new(revision: Arc<vork_media::MediaRevision>, rate: u32, channels: u32, length: Option<usize>) -> Self {
        Self {
            revision,
            rate,
            channels,
            picture: None,
            length,
            decoded: Mutex::new(Decoded {
                clip: Arc::new(Clip {
                    rate,
                    frames: Vec::new(),
                }),
                whole: false,
            }),
        }
    }
    /// Read the layout and first sample frame to validate the file at open.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_optional(path)?.context("audio: file has no audio stream")
    }
    /// An implicit soundtrack is absent when the file has no audio stream.
    pub fn open_optional(path: &Path) -> Result<Option<Self>> {
        Self::open_revision(vork_media::MediaRevision::open(path)?)
    }

    pub fn open_revision(revision: Arc<vork_media::MediaRevision>) -> Result<Option<Self>> {
        let Some((source, stream)) = Self::layout(revision)? else {
            return Ok(None);
        };
        Reader {
            source: &source,
            stream,
            position: 0,
        }
        .prefix(1)?;
        Ok(Some(source))
    }

    fn layout(revision: Arc<vork_media::MediaRevision>) -> Result<Option<(Self, Option<Stream>)>> {
        let path = revision.original();
        if let Ok(reader) = Wav::open(revision.reader()?) {
            let spec = reader.spec;
            ensure!(
                matches!(spec.channels, 1 | 2),
                "audio: {} channels are not built yet; use mono or stereo",
                spec.channels
            );
            ensure!(spec.sample_rate > 0, "audio: sample rate must be positive");
            let source = Self::new(
                revision,
                spec.sample_rate,
                spec.channels.into(),
                Some(reader.length),
            );
            return Ok(Some((source, Some(Stream::Wav(reader)))));
        }
        // The user's decoder is a separate executable, never linked or bundled.
        let probe = revision
            .probe(&[
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,channels,time_base,duration:stream_disposition=attached_pic:format=format_name",
                "-of",
                "json",
            ])
            .context("audio: install ffmpeg and ffprobe to decode this file")?;
        ensure!(
            probe.status.success(),
            "audio: cannot probe {}: {}",
            path.display(),
            String::from_utf8_lossy(&probe.stderr).trim()
        );
        let probe: serde_json::Value = serde_json::from_slice(&probe.stdout)?;
        revision.check_format(probe["format"]["format_name"].as_str().unwrap_or(""))?;
        let streams = probe["streams"].as_array().context("audio: no streams")?;
        let Some(audio) = streams.iter().find(|stream| stream["codec_type"] == "audio") else {
            return Ok(None);
        };
        let channels = audio["channels"].as_u64().context("audio: missing channel count")?;
        ensure!(
            matches!(channels, 1 | 2),
            "audio: {channels} channels are not built yet; use mono or stereo"
        );
        let mut source = Self::new(revision.clone(), SAMPLE_RATE, channels as u32, None);
        if let Some(video) = streams
            .iter()
            .find(|stream| stream["codec_type"] == "video")
            .filter(|stream| stream["disposition"]["attached_pic"].as_u64() != Some(1))
        {
            let tick = |stream: &serde_json::Value| -> Result<f64> {
                let (num, den) = stream["time_base"].as_str().and_then(|base| base.split_once('/'))
                    .context("audio: missing stream time base")?;
                let (num, den) = (num.parse::<u32>()?, den.parse::<u32>()?);
                ensure!(num > 0 && den > 0, "audio: invalid stream time base");
                Ok(f64::from(num) / f64::from(den))
            };
            source.picture = Some(PictureTimeline {
                origin: vork_media::first_picture_pts(&revision)? as f64 * tick(video)?,
                duration: video["duration"].as_str().and_then(|duration| duration.parse::<f64>().ok())
                    .filter(|duration| duration.is_finite() && *duration > 0.),
                tolerance: tick(audio)? + 1. / f64::from(SAMPLE_RATE),
            });
        }
        Ok(Some((source, None)))
    }
    /// The picture stream header's duration, when it declares one.
    pub fn picture_duration(&self) -> Option<f64> {
        self.picture.as_ref().and_then(|picture| picture.duration)
    }
    /// The file's length in sample frames at `rate`. Only a decoder's stream end tells the
    /// length of a file without a WAV header, so that file is decoded to its end.
    pub fn frames(&self) -> Result<usize> {
        match self.length {
            Some(length) => Ok(length),
            None => Ok(self.prefix(usize::MAX)?.frames.len()),
        }
    }
    pub fn reader(&self) -> Reader<'_> {
        Reader {
            source: self,
            stream: None,
            position: 0,
        }
    }

    /// Read a prefix and close the decoder before returning.
    pub fn prefix(&self, frames: usize) -> Result<Arc<Clip>> {
        self.reader().prefix(frames)
    }

    /// Open the decoder at the file's first frame.
    fn start(&self) -> Result<Stream> {
        if self.length.is_some() {
            let reader = Wav::open(self.revision.reader()?)
                .with_context(|| format!("audio: {}", self.revision.original().display()))?;
            let spec = reader.spec;
            ensure!(
                u32::from(spec.channels) == self.channels
                    && spec.sample_rate == self.rate
                    && Some(reader.length) == self.length,
                "audio: {} changed while it was open",
                self.revision.original().display()
            );
            return Ok(Stream::Wav(reader));
        }
        let mut decoder = vork_media::tool_command("ffmpeg");
        decoder.args(["-v", "error", "-nostdin"]);
        if self.picture.is_some() {
            decoder.arg("-copyts");
        }
        self.revision.input(decoder.arg("-i"))?;
        let mono = "pan=stereo|c0=c0|c1=c0";
        if let Some(picture) = &self.picture {
            // Preserve packet timing through resampling. first_pts pads/trims at the
            // picture origin; async fills or cuts later hard timestamp discontinuities.
            let layout = if self.channels == 1 {
                format!("{mono},")
            } else {
                String::new()
            };
            let filter = format!(
                "{layout}asettb=1/48000,asetpts=PTS-round(({})*48000),aresample=48000:async=1:first_pts=0:min_hard_comp={}",
                picture.origin, picture.tolerance
            );
            decoder.args(["-af", &filter]);
        } else if self.channels == 1 {
            decoder.args(["-af", mono]);
        }
        decoder
            .args([
                "-map", "0:a:0", "-f", "f32le", "-ac", "2", "-ar", "48000", "pipe:1",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let child = vork_media::DecoderProcess::start(decoder)
            .context("audio: install ffmpeg to decode this file")?;
        let stdout = child.stdout();
        let mut stderr = child.stderr().expect("piped decoder errors");
        // Drained beside the samples so a talkative decoder never blocks on its error pipe.
        let errors = std::thread::spawn(move || {
            let mut text = Vec::new();
            let _ = stderr.read_to_end(&mut text);
            text
        });
        Ok(Stream::External(External {
            child,
            stdout,
            errors: Some(errors),
        }))
    }
}

impl Reader<'_> {
    /// Read at least this many frames, or all of a shorter file. A live decoder
    /// continues across batches; a new one advances to the shared prefix first.
    pub fn prefix(&mut self, frames: usize) -> Result<Arc<Clip>> {
        let source = self.source;
        let mut decoded = source.decoded.lock().unwrap();
        let have = decoded.clip.frames.len();
        if decoded.whole || have >= frames {
            if decoded.whole {
                self.stream = None;
            }
            return Ok(decoded.clip.clone());
        }

        // Take ownership before reading so failures close the decoder without
        // changing the already published samples.
        let (mut stream, position) = match self.stream.take() {
            Some(stream) => (stream, self.position),
            None => (source.start()?, 0),
        };
        if have > position {
            match &mut stream {
                Stream::Wav(wav) => {
                    wav.input
                        .seek_relative(((have - position) * wav.frame_bytes()) as i64)?;
                }
                Stream::External(decoder) => {
                    // Replay from the origin so codec, timestamps and resampling
                    // match a continuous decode. Discard the cached PCM in the pipe.
                    let bytes = (have - position) as u64 * 8;
                    let skipped = std::io::copy(
                        &mut (&mut decoder.stdout).take(bytes),
                        &mut std::io::sink(),
                    )?;
                    if skipped != bytes {
                        decoder.finish(source.revision.original())?;
                        anyhow::bail!(
                            "audio: {} ended before the cached prefix",
                            source.revision.original().display()
                        );
                    }
                }
            }
        }
        let wanted = frames - have;
        let output = &mut Arc::make_mut(&mut decoded.clip).frames;
        let append = match &mut stream {
            Stream::Wav(wav) => wav.append(output, wanted.min(wav.length - have)),
            Stream::External(decoder) => decoder.append(output, wanted, source.revision.original()),
        };
        // Failed reads leave the shared prefix valid and close this decoder.
        if let Err(error) = append {
            output.truncate(have);
            return Err(error);
        }
        let end = output.len();
        decoded.whole = end < frames || source.length == Some(end);
        self.position = end;
        if !decoded.whole {
            self.stream = Some(stream);
        }
        Ok(decoded.clip.clone())
    }
}

impl External {
    fn append(&mut self, output: &mut Vec<Frame>, mut wanted: usize, path: &Path) -> Result<()> {
        let mut bytes = [0; 32 * 1024];
        let mut partial = 0;
        let first = output.len();
        while wanted > 0 {
            let limit = wanted.min(bytes.len() / 8) * 8;
            let read = match self.stdout.read(&mut bytes[partial..limit]) {
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                result => result?,
            };
            if read == 0 {
                self.finish(path)?;
                ensure!(partial == 0, "audio: incomplete decoded sample frame");
                break;
            }
            let count = partial + read;
            let (frames, remainder) = bytes[..count].as_chunks::<8>();
            output.extend(frames.iter().map(|&[a, b, c, d, e, f, g, h]| {
                [
                    f32::from_le_bytes([a, b, c, d]),
                    f32::from_le_bytes([e, f, g, h]),
                ]
            }));
            wanted -= frames.len();
            partial = remainder.len();
            bytes.copy_within(count - partial..count, 0);
        }
        ensure!(
            output[first..].iter().flatten().all(|x| x.is_finite()),
            "audio: nonfinite decoded sample"
        );
        Ok(())
    }

    fn finish(&mut self, path: &Path) -> Result<()> {
        let status = self.child.wait()?;
        let errors = self
            .errors
            .take()
            .and_then(|errors| errors.join().ok())
            .unwrap_or_default();
        ensure!(
            status.success(),
            "audio: cannot decode {}: {}",
            path.display(),
            String::from_utf8_lossy(&errors).trim()
        );
        Ok(())
    }
}

impl Clip {
    /// Decode a whole file.
    pub fn load(path: &Path) -> Result<Self> {
        let source = Source::open(path)?;
        let clip = source.prefix(usize::MAX)?;
        drop(source);
        Ok(Arc::try_unwrap(clip).unwrap_or_else(|clip| (*clip).clone()))
    }
    pub fn duration(&self) -> f64 {
        self.frames.len() as f64 / self.rate as f64
    }

    /// How many frames from the start of a source at `rate` `sample(seconds, speed)` can read:
    /// the frame under `seconds`, the reconstruction window after it and two frames of slack.
    pub fn reach(rate: u32, seconds: f64, speed: f64) -> usize {
        if seconds.is_nan() || seconds < 0. {
            return 0;
        }
        let radius = (16. / Self::cutoff(rate, speed)).ceil();
        (seconds * rate as f64 + radius + 2.) as usize
    }
    fn cutoff(rate: u32, speed: f64) -> f64 {
        (SAMPLE_RATE as f64 / (rate as f64 * speed.abs())).min(1.)
    }

    /// A run of aligned unit-speed output frames, when every read can use the PCM directly.
    pub fn aligned_frames(&self, start: i64, count: usize) -> Option<&[Frame]> {
        let start = usize::try_from(start).ok()?;
        let end = start.checked_add(count)?;
        // In this range the rounded seconds and their product with the rate differ
        // from the exact frame number by less than 2^-25, inside sample's 1e-7 test.
        // A frame strictly inside the slice also precedes the clip's rounded duration.
        if self.rate != SAMPLE_RATE || end > 1 << 27 {
            return None;
        }
        self.frames.get(start..end)
    }

    /// Windowed sinc reconstruction, with a rate-dependent antialias cutoff.
    pub fn sample(&self, seconds: f64, speed: f64) -> Frame {
        if seconds < 0. || seconds >= self.duration() || speed == 0. {
            return [0.; 2];
        }
        let position = seconds * self.rate as f64;
        if self.rate == SAMPLE_RATE
            && speed.abs() == 1.
            && (position - position.round()).abs() < 1e-7
        {
            return self
                .frames
                .get(position.round() as usize)
                .copied()
                .unwrap_or([0.; 2]);
        }
        let cutoff = Self::cutoff(self.rate, speed);
        let radius = (16. / cutoff).ceil() as i64;
        let center = position.floor() as i64;
        let mut sum = [0.; 2];
        let mut weight = 0.;
        for i in center - radius + 1..=center + radius {
            let x = (i as f64 - position) * cutoff;
            if x.abs() >= 16. {
                continue;
            }
            let sinc = if x.abs() < 1e-12 {
                1.
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            };
            let w = cutoff * sinc * (0.5 + 0.5 * (std::f64::consts::PI * x / 16.).cos());
            weight += w;
            if let Some(frame) = usize::try_from(i).ok().and_then(|i| self.frames.get(i)) {
                for c in 0..2 {
                    sum[c] += frame[c] as f64 * w;
                }
            }
        }
        sum.map(|x| (x / weight) as f32)
    }
}

pub fn write_wav(path: &Path, frames: &[Frame]) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 2,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    for &sample in frames.iter().flatten() {
        ensure!(sample.is_finite(), "audio: nonfinite mix sample");
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

pub struct Loudness {
    pub rms: f64,
    pub peak: f64,
}
pub fn loudness(frames: &[Frame]) -> Loudness {
    let mut energy = 0.;
    let mut peak: f64 = 0.;
    for &x in frames.iter().flatten() {
        energy += f64::from(x).powi(2);
        peak = peak.max(f64::from(x).abs());
    }
    Loudness {
        rms: if frames.is_empty() {
            0.
        } else {
            (energy / (frames.len() * 2) as f64).sqrt()
        },
        peak,
    }
}

/// Host analysis convention: Hann (N-1), mono average, 2/sum(window).
/// No temporal smoothing: the caller explicitly chooses the analysis window.
pub fn spectrum(frames: &[Frame]) -> Result<Vec<f64>> {
    use rustfft::{FftPlanner, num_complex::Complex};
    let n = frames.len();
    ensure!(
        n >= 4 && n.is_power_of_two() && n <= 65536,
        "audio spectrum: window must contain a power of two sample frames, 4..65536"
    );
    let mut window_sum = 0.;
    let mut input: Vec<_> = frames
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let w = 0.5 - 0.5 * (2. * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
            window_sum += w;
            Complex::new((f64::from(f[0]) + f64::from(f[1])) * 0.5 * w, 0.)
        })
        .collect();
    ensure!(
        window_sum > 0.,
        "audio spectrum: Hann window needs at least four samples"
    );
    FftPlanner::new().plan_fft_forward(n).process(&mut input);
    Ok(input[..=n / 2]
        .iter()
        .enumerate()
        .map(|(i, x)| x.norm() / window_sum * if i == 0 || i == n / 2 { 1. } else { 2. })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tone(hz: f64, rate: u32, n: usize) -> Clip {
        Clip {
            rate,
            frames: (0..n)
                .map(|i| {
                    [(2. * std::f64::consts::PI * hz * i as f64 / rate as f64).sin() as f32; 2]
                })
                .collect(),
        }
    }
    #[test]
    fn spectrum_and_loudness_measure_a_known_signal() {
        let clip = tone(1500., SAMPLE_RATE, 4096);
        let s = spectrum(&clip.frames).unwrap();
        assert!((s[128] - 1.).abs() < 1e-5);
        let level = loudness(&clip.frames);
        assert!((level.rms - 0.5_f64.sqrt()).abs() < 1e-6);
        assert!((level.peak - 1.).abs() < 1e-6);
        let cancelled: Vec<_> = clip.frames.iter().map(|x| [x[0], -x[0]]).collect();
        assert!(spectrum(&cancelled).unwrap().iter().all(|x| *x == 0.));
    }
    #[test]
    fn resampling_changes_pitch_and_suppresses_aliases() {
        let low = tone(3000., SAMPLE_RATE, 48000);
        let high = tone(18000., SAMPLE_RATE, 48000);
        let low_samples: Vec<_> = (1000..5096)
            .map(|i| low.sample(2. * i as f64 / SAMPLE_RATE as f64, 2.))
            .collect();
        let high_samples: Vec<_> = (1000..5096)
            .map(|i| high.sample(2. * i as f64 / SAMPLE_RATE as f64, 2.))
            .collect();
        let spectrum = spectrum(&low_samples).unwrap();
        assert!((spectrum[512] - 1.).abs() < 0.001);
        assert!(loudness(&high_samples).rms < 0.001);
    }
    #[test]
    #[ignore = "requires user-installed ffmpeg and ffprobe"]
    fn short_packet_gap_is_silent_on_the_picture_timeline() -> Result<()> {
        let path = std::env::temp_dir().join(format!("vork-audio-gap-{}.mkv", std::process::id()));
        let status = std::process::Command::new("ffmpeg").args([
            "-v", "error", "-y", "-f", "lavfi", "-i", "color=s=16x16:r=24:d=1",
            "-f", "lavfi", "-i", "aevalsrc=0.25:s=48000:d=1",
            "-af", "asetnsamples=n=1024,aselect='not(eq(n,10))'",
            "-c:v", "mpeg4", "-c:a", "pcm_s16le",
        ]).arg(&path).status()?;
        ensure!(status.success(), "create packet-gap movie");
        let clip = Clip::load(&path)?;
        // Matroska packet timestamps round to milliseconds. Check inside the
        // missing 1024-sample block, and either side of it.
        assert_eq!(clip.frames[10_000], [0.25; 2]);
        assert!(clip.frames[10_300..11_200].iter().all(|frame| *frame == [0.; 2]));
        assert_eq!(clip.frames[11_500], [0.25; 2]);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires user-installed ffmpeg and ffprobe"]
    fn external_decoder_preserves_mono_amplitude() {
        let path = std::env::temp_dir().join(format!("vork-audio-mono-{}.bin", std::process::id()));
        let mut wav = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 1,
                sample_rate: SAMPLE_RATE,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        for _ in 0..480 {
            wav.write_sample(0.25_f32).unwrap();
        }
        wav.finalize().unwrap();
        let decoded = Clip::load(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(decoded.frames.len(), 480);
        assert!(decoded.frames.iter().all(|f| *f == [0.25; 2]));
    }
    #[test]
    fn readers_share_the_prefix_and_resume_at_the_first_missing_frame() {
        let path =
            std::env::temp_dir().join(format!("vork-audio-source-{}.wav", std::process::id()));
        let frames: Vec<Frame> = (0..1000).map(|i| [i as f32, -(i as f32)]).collect();
        write_wav(&path, &frames).unwrap();
        let source = Source::open(&path).unwrap();
        assert_eq!((source.rate, source.frames().unwrap()), (SAMPLE_RATE, 1000));
        // Opening cached the first frame. A later reader reuses the prefix.
        assert_eq!(source.decoded.lock().unwrap().clip.frames, frames[..1]);
        let prefix = source.prefix(100).unwrap();
        assert_eq!(prefix.frames, frames[..100]);
        assert!(Arc::ptr_eq(&prefix, &source.prefix(40).unwrap()));
        let mut reader = source.reader();
        assert_eq!(reader.prefix(101).unwrap().frames, frames[..101]);
        assert_eq!(prefix.frames.len(), 100);
        assert!(reader.stream.is_some());
        // A different reader can extend the prefix while this one is still live.
        assert_eq!(source.prefix(120).unwrap().frames, frames[..120]);
        let whole = reader.prefix(usize::MAX).unwrap();
        assert_eq!(whole.frames, frames);
        assert!(reader.stream.is_none());
        assert!(Arc::ptr_eq(&whole, &source.prefix(5000).unwrap()));
        drop(source);
        assert_eq!(Clip::load(&path).unwrap().frames, frames);
        std::fs::remove_file(path).unwrap();
        // Every frame a sample reads lies inside its reach.
        for (rate, speed) in [(SAMPLE_RATE, 1.), (44_100, 1.), (SAMPLE_RATE, -3.5), (8_000, 0.25)] {
            for seconds in [0., 0.01, 0.5] {
                let reach = Clip::reach(rate, seconds, speed);
                let clip = Clip {
                    rate,
                    frames: vec![[1.; 2]; reach + 64],
                };
                let mut short = clip.clone();
                short.frames.truncate(reach);
                short.frames.extend([[9.; 2]; 64]);
                assert_eq!(clip.sample(seconds, speed), short.sample(seconds, speed));
            }
        }
    }
    #[test]
    fn bulk_wav_matches_hound_for_each_integer_layout() -> Result<()> {
        let path = std::env::temp_dir().join(format!("vork-audio-layout-{}.wav", std::process::id()));
        for (bits, bytes) in [(8, 1), (16, 2), (24, 3), (24, 4), (32, 4)] {
            let half = 1_i64 << (bits - 1);
            let scale = half as f64;
            for channels in [1, 2] {
                let spec = hound::WavSpec { channels, sample_rate: SAMPLE_RATE, bits_per_sample: bits, sample_format: hound::SampleFormat::Int };
                let mut writer = hound::WavWriter::new_with_spec_ex(
                    std::io::BufWriter::new(std::fs::File::create(&path)?),
                    hound::WavSpecEx { spec, bytes_per_sample: bytes },
                )?;
                for i in 0..40_000_i64 {
                    let sample = ((i * 67_108_861) % (2 * half) - half) as i32;
                    writer.write_sample(sample)?;
                }
                writer.finalize()?;
                let stored = hound::WavReader::open(&path)?.into_samples::<i32>()
                    .map(|x| x.map(|x| (f64::from(x) / scale) as f32))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let expected: Vec<_> = stored.chunks_exact(channels as usize)
                    .map(|pair| [pair[0], pair[channels as usize - 1]])
                    .collect();
                let source = Source::open(&path)?;
                assert_eq!(source.prefix(139)?.frames, expected[..139], "{bits} bits, {channels} channels");
                assert_eq!(source.prefix(usize::MAX)?.frames, expected, "{bits} bits, {channels} channels");
            }
        }
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn a_file_that_does_not_decode_is_refused_when_it_is_opened() {
        let path =
            std::env::temp_dir().join(format!("vork-audio-broken-{}.wav", std::process::id()));
        let frames = vec![[0.5, -0.5]; 3];
        let cut = |frames_kept: u64| {
            write_wav(&path, &frames).unwrap();
            let length = std::fs::metadata(&path).unwrap().len();
            let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            file.set_len(length - (3 - frames_kept) * 8).unwrap();
        };
        // The header promises three frames and none follow.
        cut(0);
        assert!(Source::open(&path).is_err());
        // Open reads the layout and the first frame, not the file: damage further in is
        // reported by the mix that reaches it.
        cut(1);
        let source = Source::open(&path).unwrap();
        assert!(source.prefix(3).is_err());
        // A header alone is not a sound file.
        std::fs::write(&path, b"RIFF").unwrap();
        assert!(Source::open(&path).is_err());
        std::fs::remove_file(&path).unwrap();
        assert!(Source::open(&path).is_err());
    }
    #[test]
    #[ignore = "requires user-installed ffmpeg and ffprobe"]
    fn an_undecodable_compressed_file_is_refused_when_it_is_opened() {
        let path =
            std::env::temp_dir().join(format!("vork-audio-broken-{}.mp3", std::process::id()));
        std::fs::write(&path, vec![0x55; 4096]).unwrap();
        assert!(Source::open(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }
    #[cfg(unix)]
    #[test]
    fn wav_content_selects_the_same_reader_through_either_symlink() {
        let dir = std::env::temp_dir().join(format!("vork-wav-suffix-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("a.wav");
        let pcm = dir.join("b.pcm");
        let alias = dir.join("voice.wav");
        let mut writer = hound::WavWriter::create(&wav, hound::WavSpec {
            channels: 1, sample_rate: 22_050, bits_per_sample: 16, sample_format: hound::SampleFormat::Int,
        }).unwrap();
        for i in 0..500 { writer.write_sample(i as i16 * 17).unwrap(); }
        writer.finalize().unwrap();
        std::fs::copy(&wav, &pcm).unwrap();
        let read = |target| {
            std::os::unix::fs::symlink(target, &alias).unwrap();
            let clip = Clip::load(&alias.canonicalize().unwrap()).unwrap();
            std::fs::remove_file(&alias).unwrap();
            assert_eq!(clip.rate, 22_050);
            clip.frames
        };
        assert_eq!(read(wav), read(pcm));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn wav_round_trip_retains_unclamped_stereo() {
        let path = std::env::temp_dir().join(format!("vork-audio-test-{}.wav", std::process::id()));
        let frames = vec![[2., -3.], [0.125, -0.25], [0., 0.]];
        write_wav(&path, &frames).unwrap();
        let clip = Clip::load(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(clip.frames, frames);
        assert_eq!(clip.rate, SAMPLE_RATE);
    }
}
