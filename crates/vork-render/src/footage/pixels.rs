//! Pair each raw picture with the timestamp reported by its own FFmpeg process.
//! Drain the machine records independently of the pixel pipe. Either pipe can fill first.
use super::Rational;
use anyhow::{Context, Result};
use std::{
    io::{BufRead, BufReader, Read},
    process::Command,
    sync::{Arc, LazyLock, mpsc},
};
use vork_media::{PictureIndex, PictureStamp};

// FFmpeg 5/6 expose POS on decoded frames. Starting with FFmpeg 7
// (libavcodec 61), setts can carry the demuxed packet position into the decoder.
#[derive(Clone, Copy)]
enum SeekStamp {
    FramePosition,
    InputFilter,
}
fn probe_seek_stamp() -> Option<SeekStamp> {
    let list = vork_inputs::tool_output("ffmpeg", &["-bsfs"]).ok()?;
    if !list.status.success() {
        return None;
    }
    let banner = String::from_utf8_lossy(&list.stderr);
    let version = banner
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("libavcodec "))?;
    let major = version
        .trim_start()
        .split('.')
        .next()?
        .trim()
        .parse::<u32>()
        .ok()?;
    if major < 61 {
        return Some(SeekStamp::FramePosition);
    }
    String::from_utf8_lossy(&list.stdout)
        .lines()
        .any(|line| line.trim() == "setts")
        .then_some(SeekStamp::InputFilter)
}

fn seek_stamp() -> Option<SeekStamp> {
    static STAMP: LazyLock<Option<SeekStamp>> = LazyLock::new(probe_seek_stamp);
    *STAMP
}

pub(super) fn can_stamp_seek() -> bool {
    seek_stamp().is_some()
}

pub(super) fn seek_input(command: &mut Command) {
    if matches!(seek_stamp(), Some(SeekStamp::InputFilter)) {
        // FFmpeg's encoder stats preserve the decoder's input PTS. Carry
        // the byte position there so duplicate raw PTS cannot alias a landing.
        command.args(["-bsf:v:0", "setts=pts=POS:dts=DTS"]);
    }
}

pub(super) fn configure(
    command: &mut Command,
    tick: &Rational,
    filter: Option<&str>,
    sought: bool,
    records: &str,
) {
    let mut chain = if sought && matches!(seek_stamp(), Some(SeekStamp::InputFilter)) {
        command.args([
            "-stats_enc_pre",
            records,
            "-stats_enc_pre_fmt",
            "vork-pos {ptsi}",
        ]);
        String::new()
    } else {
        let stamp = if sought {
            "setpts=if(isnan(POS)\\,-1\\,POS)".to_owned()
        } else {
            format!("settb={tick}")
        };
        let url = records.replace(':', r"\:");
        format!(
            r"{stamp},metadata=mode=add:key=vork_frame:value=1,metadata=mode=print:key=vork_frame:file='{url}':direct=1,"
        )
    };
    // Rawvideo carries only pixels. Give the output muxer increasing times even
    // when display timestamps repeat or packet offsets go backwards in B frames.
    chain.push_str("settb=1/1,setpts=N");
    if let Some(filter) = filter {
        chain.push(',');
        chain.push_str(filter);
    }
    command.args(["-vf", &chain, "-enc_time_base", "1:1"]);
}

fn stamp(line: &str, sought: bool) -> Option<PictureStamp> {
    if let Some(value) = line.strip_prefix("vork-pos ") {
        return Some(
            value
                .trim()
                .parse()
                .map(PictureStamp::PacketPosition)
                .unwrap_or(PictureStamp::Unavailable),
        );
    }
    if !line.starts_with("frame:") {
        return None;
    }
    let value = line
        .split_once("pts:")
        .and_then(|(_, fields)| fields.split_whitespace().next())
        .and_then(|value| value.parse().ok());
    Some(match value {
        Some(pos) if sought => PictureStamp::PacketPosition(pos),
        Some(pts) => PictureStamp::Displayed(pts),
        None => PictureStamp::Unavailable,
    })
}

#[derive(Debug)]
pub(crate) struct WrongLanding(usize);

impl std::fmt::Display for WrongLanding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "decoded packet differs from the displayed index at picture {}",
            self.0
        )
    }
}
impl std::error::Error for WrongLanding {}

pub(crate) struct PixelInput {
    input: Box<dyn Read + Send>,
    stamps: mpsc::Receiver<PictureStamp>,
    index: Arc<PictureIndex>,
    next: usize,
    landing_checked: bool,
    anchor: Option<usize>,
}

impl PixelInput {
    pub(super) fn new(
        input: Box<dyn Read + Send>,
        stream: impl Read + Send + 'static,
        index: &Arc<PictureIndex>,
        next: usize,
        sought: bool,
    ) -> Result<Self> {
        let (send, stamps) = mpsc::channel();
        std::thread::Builder::new()
            .name("movie timestamps".into())
            .spawn(move || {
                let mut stream = BufReader::new(stream);
                let mut line = String::new();
                loop {
                    line.clear();
                    match stream.read_line(&mut line) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {}
                    }
                    if let Some(stamp) = stamp(&line, sought) {
                        if send.send(stamp).is_err() {
                            break;
                        }
                    }
                }
            })?;
        Ok(Self::from_parts(
            input,
            stamps,
            index.clone(),
            next,
            sought.then_some(next),
        ))
    }

    pub(super) fn from_parts(
        input: Box<dyn Read + Send>,
        stamps: mpsc::Receiver<PictureStamp>,
        index: Arc<PictureIndex>,
        next: usize,
        anchor: Option<usize>,
    ) -> Self {
        Self {
            input,
            stamps,
            index,
            next,
            landing_checked: false,
            anchor,
        }
    }

    pub(super) fn check_landing(&mut self) -> Result<bool> {
        if let Ok(stamp) = self.stamps.recv() {
            self.landing_checked = self.index.matches(self.next, stamp)?;
        }
        Ok(self.landing_checked)
    }

    pub(crate) fn read_frame(&mut self, bytes: &mut [u8]) -> Result<()> {
        self.input.read_exact(bytes)?;
        if !std::mem::take(&mut self.landing_checked) {
            let stamp = self
                .stamps
                .recv()
                .context("movie decoder omitted a frame timestamp")?;
            if !self.index.matches(self.next, stamp)? {
                if let Some(anchor) = self.anchor {
                    self.index.reject_anchor(anchor)?;
                    return Err(WrongLanding(self.next).into());
                }
                anyhow::bail!(
                    "decoded picture {} ({stamp:?}) differs from displayed index",
                    self.next
                );
            }
        }
        self.next += 1;
        Ok(())
    }
}
