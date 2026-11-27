//! The displayed-picture clock used by movie textures, canvas readers and soundtracks.
mod process;
mod records;
mod revision;
pub use process::{DecoderProcess, PixelPipes, stop_decoders_for_exit};
pub use vork_inputs::tool_command;
pub use revision::{MediaBinding, MediaReader, MediaRevision};

use anyhow::{Context, Result, ensure};
use std::{
    io::{BufRead, BufReader},
    process::{ChildStdout, Command, Stdio},
    sync::Mutex,
};

fn opaque_format(format: &str) -> bool {
    matches!(
        format,
        "yuv420p" | "yuv422p" | "yuv444p" | "nv12" | "rgb24" | "bgr24"
    )
}

/// AVC has no alpha plane. Other codecs can change sample description mid-stream.
pub fn opaque_stream(codec: &str, format: &str) -> bool {
    codec == "h264" && opaque_format(format)
}

/// Y′CbCr coefficients for an opaque picture accepted by the native NV12 converter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativeMatrix {
    Bt709 = 1,
    Bt601 = 2,
}

impl NativeMatrix {
    pub fn av_colorspace(self) -> i32 {
        match self {
            Self::Bt709 => 1, // AVCOL_SPC_BT709
            Self::Bt601 => 6, // AVCOL_SPC_SMPTE170M
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct MissingCropOffsets;

// The clock stays usable when an older probe cannot certify the source window.
type ProbedMatrix = Result<Option<NativeMatrix>, MissingCropOffsets>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Picture {
    pts: i64,
    raw_pts: Option<i64>,
    position: i64,
    key: bool,
    matrix: ProbedMatrix,
    rotation: Option<i32>,
}

// The coded frame from `ffprobe -apply_cropping 0` includes alignment
// padding. A zero-origin display window may end before its right/bottom edges.
fn matches_source_raster(fields: &str, [width, height]: [u32; 2]) -> Result<bool, MissingCropOffsets> {
    let (mut coded_width, mut coded_height) = (0u32, 0u32);
    let mut edges = [None; 4];
    for field in fields.split('|') {
        let Some((name, value)) = field.split_once('=') else { continue };
        let edge = match name {
            "width" => { coded_width = value.parse().unwrap_or(0); continue; }
            "height" => { coded_height = value.parse().unwrap_or(0); continue; }
            "crop_left" => 0,
            "crop_top" => 1,
            "crop_right" => 2,
            "crop_bottom" => 3,
            _ => continue,
        };
        let Ok(n) = value.parse::<u32>() else { return Ok(false) };
        edges[edge] = Some(n);
    }
    match edges {
        [Some(0), Some(0), Some(right), Some(bottom)] =>
            Ok(coded_width.checked_sub(right) == Some(width) && coded_height.checked_sub(bottom) == Some(height)),
        [None, None, None, None] if [coded_width, coded_height] == [width, height] => Ok(true),
        [None, None, None, None] if coded_width >= width && coded_height >= height => Err(MissingCropOffsets),
        _ => Ok(false),
    }
}

struct ProbeCommand {
    command: Command,
    // VideoToolbox clears AVFrame.crop_* before publishing a pixel buffer.
    // Verify its source rectangle against the unrotated stream raster here.
    source_raster: Option<[u32; 2]>,
}

struct Probe {
    child: DecoderProcess,
    stdout: BufReader<ChildStdout>,
    line: String,
    source_raster: Option<[u32; 2]>,
}

impl Probe {
    fn pictures(revision: &MediaRevision, source_raster: Option<[u32; 2]>) -> Result<ProbeCommand> {
        let fields = if source_raster.is_some() {
            "frame=pts,pkt_pos,best_effort_timestamp,key_frame,pix_fmt,color_space,width,height,crop_left,crop_top,crop_right,crop_bottom:frame_side_data=rotation"
        } else if cfg!(target_os = "linux") {
            "frame=pts,pkt_pos,best_effort_timestamp,key_frame,pix_fmt,color_space,color_range,chroma_location:frame_side_data=side_data_type,rotation"
        } else {
            "frame=pts,pkt_pos,best_effort_timestamp,key_frame,pix_fmt,color_space:frame_side_data=rotation"
        };
        Self::command(revision, fields, source_raster)
    }

    fn command(revision: &MediaRevision, fields: &'static str, source_raster: Option<[u32; 2]>) -> Result<ProbeCommand> {
        let mut command = tool_command("ffprobe");
        if source_raster.is_some() {
            command.args(["-apply_cropping", "0"]);
        }
        // The picture clock reads metadata and discards the reconstructed pixels.
        command
            .args([
                "-v",
                "error",
                "-skip_loop_filter",
                "all",
                "-select_streams",
                "v:0",
                "-show_entries",
                fields,
                "-of",
                "compact=p=1",
            ])
            .stdout(Stdio::piped());
        revision.attach(&mut command)?;
        Ok(ProbeCommand { command, source_raster })
    }

    fn spawn(ProbeCommand { command, source_raster }: ProbeCommand) -> Result<Self> {
        let child = DecoderProcess::start(command).context("probe displayed video frames")?;
        Ok(Self {
            stdout: BufReader::new(child.stdout()),
            child,
            line: String::new(),
            source_raster,
        })
    }
    fn start(command: ProbeCommand) -> Result<(Self, Picture)> {
        let mut probe = Self::spawn(command)?;
        let first = probe.next()?.context("video has no displayed frames")?;
        Ok((probe, first))
    }

    fn next(&mut self) -> Result<Option<Picture>> {
        loop {
            self.line.clear();
            if self.stdout.read_line(&mut self.line)? == 0 {
                ensure!(
                    self.child.wait()?.success(),
                    "probe displayed video frames failed"
                );
                return Ok(None);
            }
            let Some(fields) = self.line.trim_end().strip_prefix("frame|") else {
                continue;
            };
            let mut pts = None;
            let mut raw_pts: Option<i64> = None;
            let mut position = -1;
            let mut key = false;
            let mut opaque = false;
            let mut rotation = None;
            #[cfg(target_os = "linux")]
            let mut supported_metadata = true;
            #[cfg(not(target_os = "linux"))]
            let supported_metadata = true;
            // swscale uses BT.601 for an unspecified matrix.
            let mut matrix = Some(NativeMatrix::Bt601);
            for field in fields.split('|') {
                if let Some(t) = field.strip_prefix("best_effort_timestamp=") {
                    pts = t.parse().ok();
                }
                if let Some(t) = field.strip_prefix("pkt_pos=") {
                    position = t.parse().unwrap_or(-1);
                }
                if let Some(t) = field.strip_prefix("pts=") {
                    raw_pts = t.parse().ok();
                }
                if let Some(fmt) = field.strip_prefix("pix_fmt=") {
                    opaque = opaque_format(fmt) || (cfg!(target_os = "linux") && fmt == "yuv420p10le");
                }
                #[cfg(target_os = "linux")]
                {
                    if let Some(range) = field.strip_prefix("color_range=") {
                        supported_metadata &= matches!(range, "tv" | "unknown");
                    }
                    if let Some(location) = field.strip_prefix("chroma_location=") {
                        supported_metadata &= matches!(location, "left" | "unspecified" | "unknown");
                    }
                    if field.rsplit(':').next() == Some("side_data_type=3x3 displaymatrix") {
                        supported_metadata = false;
                    }
                }
                // FFmpeg autorotates with the first display matrix on the frame.
                if rotation.is_none() {
                    if let Some(angle) = field.rsplit(':').next().and_then(|f| f.strip_prefix("rotation=")) {
                        rotation = angle.parse().ok();
                    }
                }
                if let Some(value) = field.strip_prefix("color_space=") {
                    matrix = match value {
                        "bt709" => Some(NativeMatrix::Bt709),
                        "bt470bg" | "smpte170m" | "unknown" => Some(NativeMatrix::Bt601),
                        _ => None,
                    };
                }
                if field == "key_frame=1" {
                    key = true;
                }
            }
            return Ok(Some(Picture {
                raw_pts,
                rotation,
                position,
                pts: pts.context("video frame has no presentation timestamp")?,
                key,
                // NVDEC exposes pixels and sequence metadata. The displayed-frame
                // probe supplies the range, chroma siting and orientation checks.
                // VideoToolbox checks those fields on its decoded AVFrame.
                matrix: match (matrix.filter(|_| opaque && supported_metadata), self.source_raster) {
                    (Some(matrix), Some(size)) => matches_source_raster(fields, size).map(|fits| fits.then_some(matrix)),
                    (matrix, _) => Ok(matrix),
                },
            }));
        }
    }
}

/// The audio mixer needs only the first displayed timestamp.
pub fn first_picture_pts(revision: &MediaRevision) -> Result<i64> {
    let command = Probe::command(revision, "frame=best_effort_timestamp", None)?;
    Ok(Probe::start(command)?.1.pts)
}

enum Tail {
    Complete,
    // Opening an unused source must leave no child reading ahead.
    Pending {
        command: Option<ProbeCommand>,
        first: Picture,
    },
    Reading(Probe),
    Failed(String),
}

struct State {
    records: records::Records,
    // A key at the end of the prefix needs one more PTS (or EOF) to resolve ties.
    pending: Option<Picture>,
    first: i64,
    first_rotation: Option<i32>,
    last: i64,
    previous: Option<i64>,
    anchor: usize,
    tail: Tail,
}

impl State {
    fn new(first: Picture, tail: Tail) -> Self {
        Self {
            records: Default::default(),
            pending: Some(first),
            first: first.pts,
            first_rotation: first.rotation,
            last: first.pts,
            previous: None,
            anchor: 0,
            tail,
        }
    }
    fn len(&self) -> usize {
        self.records.len() + usize::from(self.pending.is_some())
    }

    fn append(&mut self, next: Option<Picture>) -> Result<()> {
        let f = self.pending.context("movie has no pending picture")?;
        ensure!(
            next.is_none_or(|n| n.pts >= f.pts),
            "video presentation timestamps must be ordered"
        );
        if f.key && self.previous.is_some_and(|p| p < f.pts) && next.is_none_or(|n| f.pts < n.pts) {
            self.anchor = self.records.len();
        }
        self.records
            .push(f.pts, f.raw_pts, f.position, f.matrix, self.anchor)?;
        self.previous = Some(f.pts);
        self.last = next.unwrap_or(f).pts;
        self.pending = next;
        Ok(())
    }

    fn at(&mut self, i: usize) -> Result<(i64, ProbedMatrix)> {
        if i == self.records.len() {
            let f = self.pending.context("movie frame missing")?;
            return Ok((f.pts, f.matrix));
        }
        let record = self.records.get(i)?;
        Ok((record.pts, record.matrix))
    }

    fn anchor_at(&mut self, ordinal: usize) -> Result<records::Record> {
        // The next PTS disambiguates a key picture sharing its timestamp.
        self.through(ordinal.saturating_add(1))?;
        ensure!(ordinal < self.len(), "video frame {ordinal} missing");
        let record = self.records.get(ordinal)?;
        if record.anchor == ordinal {
            Ok(record)
        } else {
            self.records.get(record.anchor)
        }
    }

    fn extend(&mut self) -> Result<bool> {
        let result = (|| {
            if let Tail::Pending {
                command,
                first: expected,
            } = &mut self.tail
            {
                let (mut probe, first) =
                    Probe::start(command.take().context("missing frame probe")?)?;
                ensure!(
                    first == *expected && probe.next()? == self.pending,
                    "movie changed after opening; reopen the scene"
                );
                self.tail = Tail::Reading(probe);
            }
            let frame = match &mut self.tail {
                Tail::Complete => return Ok(false),
                Tail::Failed(error) => anyhow::bail!("{error}"),
                Tail::Reading(probe) => probe.next()?,
                Tail::Pending { .. } => unreachable!(),
            };
            self.append(frame)?;
            Ok(frame.is_some())
        })();
        match &result {
            Ok(false) => self.tail = Tail::Complete,
            Err(err) => self.tail = Tail::Failed(format!("{err:#}")),
            Ok(true) => {}
        }
        result
    }

    fn cover(&mut self, at_or_before: &impl Fn(i64, i64) -> bool) -> Result<()> {
        while at_or_before(self.last, self.first) && self.extend()? {}
        Ok(())
    }

    fn through(&mut self, ordinal: usize) -> Result<()> {
        while self.len() <= ordinal && self.extend()? {}
        Ok(())
    }
}

/// All readers of an opened movie share the same uninterrupted displayed-picture prefix.
pub struct PictureIndex(Mutex<State>);

#[derive(Clone, Copy, Debug)]
pub enum PictureStamp {
    Unavailable,
    Displayed(i64),
    PacketPosition(i64),
}

impl PictureIndex {
    pub fn open(revision: &MediaRevision, unrotated_size: [u32; 2]) -> Result<Self> {
        let source_raster = cfg!(target_os = "macos").then_some(unrotated_size);
        let (mut probe, first) = Probe::start(Probe::pictures(revision, source_raster)?)?;
        let second = probe.next()?;
        drop(probe);
        let mut state = State::new(first, Tail::Complete);
        state.append(second)?;
        if second.is_some() {
            // Bind the deferred probe to its own cursor while the name is
            // available. A native session can grow this clock after replacement.
            state.tail = Tail::Pending {
                command: Some(Probe::pictures(revision, source_raster)?),
                first,
            };
        }
        Ok(Self(Mutex::new(state)))
    }

    /// Display-matrix rotation on the first decoded picture, when present.
    pub fn first_rotation(&self) -> Option<i32> {
        self.0.lock().unwrap().first_rotation
    }

    pub fn pts(&self, ordinal: usize) -> Result<Option<i64>> {
        let mut index = self.0.lock().unwrap();
        index.through(ordinal)?;
        if ordinal >= index.len() {
            return Ok(None);
        }
        Ok(Some(index.at(ordinal)?.0))
    }

    /// Use the verified display picture's matrix. Hardware decoders can attach
    /// different colour metadata to the same Y′CbCr samples.
    pub fn native_matrix_at(&self, ordinal: usize) -> Result<Option<NativeMatrix>> {
        let mut index = self.0.lock().unwrap();
        index.through(ordinal)?;
        index.at(ordinal)?.1.map_err(|_| anyhow::anyhow!(
            "this padded frame requires software decoding because FFprobe omitted its crop offsets. Install FFprobe 6.1 or newer on PATH to use VideoToolbox"
        ))
    }

    /// Select the last picture whose time is at or before the requested clock.
    pub fn select(&self, at_or_before: impl Fn(i64, i64) -> bool) -> Result<usize> {
        let mut index = self.0.lock().unwrap();
        index.cover(&at_or_before)?;
        let origin = index.first;
        let (mut left, mut right) = (0, index.len());
        while left < right {
            let mid = left + (right - left) / 2;
            if at_or_before(index.at(mid)?.0, origin) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        Ok(left.saturating_sub(1))
    }

    /// First and last known PTS, displayed count, and whether EOF is known.
    pub fn span_through(
        &self,
        at_or_before: impl Fn(i64, i64) -> bool,
    ) -> Result<(i64, i64, usize, bool)> {
        let mut state = self.0.lock().unwrap();
        state.cover(&at_or_before)?;
        Ok((
            state.first,
            state.last,
            state.len(),
            matches!(state.tail, Tail::Complete),
        ))
    }

    /// Effective key picture in the uninterrupted displayed-frame index.
    pub fn anchor(&self, ordinal: usize) -> Result<Option<(usize, i64)>> {
        let key = self.0.lock().unwrap().anchor_at(ordinal)?;
        Ok((key.anchor != 0).then_some((key.anchor, key.pts)))
    }

    /// Address the demuxer in track ticks. The returned ordinal is only a
    /// candidate until its decoder reports the indexed packet position.
    pub fn seek_anchor(&self, ordinal: usize) -> Result<Option<(usize, i64)>> {
        let mut state = self.0.lock().unwrap();
        let mut key = state.anchor_at(ordinal)?;
        while key.rejected
            || key.position < 0
            || (key.anchor == 0 && state.len() > 1 && state.at(1)?.0 == key.pts)
        {
            if key.anchor == 0 {
                return Ok(None);
            }
            key = state.anchor_at(key.anchor - 1)?;
        }
        Ok(Some((key.anchor, key.raw_pts.unwrap_or(key.pts))))
    }

    /// Verify the pixel decoder's display time, or its packet identity after a seek.
    pub fn matches(&self, ordinal: usize, stamp: PictureStamp) -> Result<bool> {
        let mut state = self.0.lock().unwrap();
        state.through(ordinal)?;
        if ordinal >= state.len() {
            return Ok(false);
        }
        let (pts, position) = if ordinal == state.records.len() {
            let picture = state.pending.context("missing pending picture")?;
            (picture.pts, picture.position)
        } else {
            let record = state.records.get(ordinal)?;
            (record.pts, record.position)
        };
        Ok(match stamp {
            PictureStamp::Unavailable => false,
            PictureStamp::Displayed(time) => time == pts,
            PictureStamp::PacketPosition(pos) => pos >= 0 && pos == position,
        })
    }

    /// A portable decoder observed a different first timestamp. Keep that
    /// landing ineligible for every subsequent reader of this revision.
    pub fn reject_anchor(&self, ordinal: usize) -> Result<()> {
        let mut state = self.0.lock().unwrap();
        if ordinal < state.records.len() {
            state.records.disable_anchor(ordinal)?;
        }
        Ok(())
    }

    /// Stream the requested timeline into the caller's representation without
    /// retaining another movie-sized vector of timestamps.
    pub fn visit_timestamps(&self, mut visit: impl FnMut(i64) -> Result<()>) -> Result<()> {
        let mut state = self.0.lock().unwrap();
        while state.extend()? {}
        for i in 0..state.len() {
            visit(state.at(i)?.0)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
