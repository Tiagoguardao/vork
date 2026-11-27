//! FFmpeg timing receipts share the renderer's content-addressed store.
use crate::disk_cache;
use anyhow::{Context, Result, ensure};
use std::path::PathBuf;

pub(crate) fn clock_help(dependency: &[u8]) -> Result<String> {
    let identity = vork_inputs::tool_identity("ffmpeg").context("read ffmpeg timing capability identity")?;
    let mut key = blake3::Hasher::new();
    key.update(identity.digest().as_bytes());
    // This report includes runtime libav* versions: a shared-library upgrade
    // can change capabilities without replacing the ffmpeg executable.
    key.update(dependency);
    let path: PathBuf = disk_cache::Store::host().path("ffmpeg-clock", 1, key.finalize());
    if let Some(help) = disk_cache::read(&path).and_then(|bytes| String::from_utf8(bytes).ok()) {
        return Ok(help);
    }
    let help = vork_encode::probe_video_clock_capabilities()?;
    // A new on-disk receipt can outlive this worker. Publish only if the executable
    // used to measure it still matches the selection captured by the worker.
    let selected = vork_inputs::tool_command("ffmpeg");
    ensure!(
        vork_inputs::Executable::open(selected.get_program().as_ref())
            .context("recheck ffmpeg timing capability identity")? == identity,
        "ffmpeg changed while checking timing capabilities; retry the render"
    );
    let _ = disk_cache::write(&path, help.as_bytes());
    Ok(help)
}

/// What a first video render asks FFmpeg before it draws, asked by `vork setup` so the answer is in the store:
/// one receipt per kind of video file, since each names FFmpeg by a different report. Each is asked here
/// directly; a render's own check keeps one answer per process, so through it the second kind would never be
/// asked or stored. A machine without a usable FFmpeg has nothing to store: the render that needs one says so.
/// macOS asks FFmpeg nothing about timing (`vork_encode::validate_video_clock_capabilities`).
pub(crate) fn warm() {
    if cfg!(target_os = "macos") { return; }
    for file in ["setup.mp4", "setup.mov"] {
        if let Ok(dependency) = vork_encode::validate_video_output(std::path::Path::new(file)) {
            let _ = clock_help(&dependency);
        }
    }
}
