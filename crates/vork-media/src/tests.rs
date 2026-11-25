use super::*;
use serde_json::Value;
use std::path::Path;

fn ffmpeg(args: &[&str], out: &Path) -> Result<()> {
    ensure!(
        Command::new("ffmpeg")
            .args(["-v", "error", "-y"])
            .args(args)
            .arg(out)
            .status()?
            .success(),
        "generate {}",
        out.display()
    );
    Ok(())
}

fn probe(path: &Path, frames: bool) -> Result<Value> {
    let mut cmd = Command::new("ffprobe");
    cmd.args(["-v", "error", "-select_streams", "v:0", "-show_streams"]);
    if frames {
        cmd.arg("-show_frames");
    }
    let out = cmd.args(["-of", "json"]).arg(path).output()?;
    ensure!(out.status.success(), "probe fixture");
    Ok(serde_json::from_slice(&out.stdout)?)
}

fn compare(path: &Path) -> Result<()> {
    let info = probe(path, false)?;
    let revision = MediaRevision::open(path)?;
    let stream = &info["streams"][0];
    let size = [
        stream["width"].as_u64().unwrap().try_into()?,
        stream["height"].as_u64().unwrap().try_into()?,
    ];
    let index = PictureIndex::open(&revision, size)?;
    {
        let state = index.0.lock().unwrap();
        assert!(matches!(state.tail, Tail::Pending { .. }));
        assert_eq!(state.len(), 2);
    }
    let (origin, covered, count, complete) = index.span_through(|pts, origin| pts <= origin)?;
    assert!(!complete);
    assert!(covered > origin && count >= 2);
    let reference = probe(path, true)?;
    assert_eq!(
        info["streams"][0]["duration"],
        reference["streams"][0]["duration"]
    );
    let frames = reference["frames"].as_array().unwrap();
    let pts: Vec<_> = frames
        .iter()
        .map(|f| f["best_effort_timestamp"].as_i64().unwrap())
        .collect();
    let mut anchor = None;
    for (i, &t) in pts.iter().enumerate() {
        if i > 0
            && frames[i]["key_frame"] == 1
            && pts[i - 1] < t
            && pts.get(i + 1).is_none_or(|&n| n > t)
        {
            anchor = Some((i, t));
        }
        assert_eq!(index.anchor(i)?, anchor, "anchor {}", path.display());
        assert_eq!(index.pts(i)?, Some(t), "PTS {}", path.display());
    }
    let mut times = Vec::new();
    index.visit_timestamps(|t| {
        times.push(t);
        Ok(())
    })?;
    assert_eq!(times, pts);
    Ok(())
}

#[test]
fn decoded_prefix_matches_the_displayed_frames() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("b.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x48:r=30:d=3",
            "-c:v",
            "libx264",
            "-g",
            "12",
            "-bf",
            "3",
            "-output_ts_offset",
            "7",
        ],
        &path,
    )?;
    compare(&path)?;
    let source = path.to_str().unwrap();
    let mkv = dir.path().join("cues.mkv");
    ffmpeg(&["-i", source, "-c", "copy"], &mkv)?;
    compare(&mkv)?;
    let fragmented = dir.path().join("fragments.mp4");
    ffmpeg(
        &[
            "-i",
            source,
            "-c",
            "copy",
            "-movflags",
            "frag_keyframe+empty_moov",
        ],
        &fragmented,
    )?;
    compare(&fragmented)?;
    let trimmed = dir.path().join("trim.mov");
    ffmpeg(
        &["-ss", "0.17", "-i", source, "-t", "1.2", "-c", "copy"],
        &trimmed,
    )?;
    compare(&trimmed)?;
    let vfr = dir.path().join("vfr.mov");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x48:r=30:d=3",
            "-vf",
            "select='not(mod(n,3)*mod(n,7))'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-g",
            "8",
            "-bf",
            "0",
        ],
        &vfr,
    )?;
    compare(&vfr)?;
    let open_gop = dir.path().join("open-gop.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=96x64:r=30:d=3",
            "-c:v",
            "libx264",
            "-x264-params",
            "open-gop=1:keyint=12:min-keyint=12:scenecut=0",
        ],
        &open_gop,
    )?;
    compare(&open_gop)?;
    let duplicates = dir.path().join("duplicates.mkv");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x48:r=30:d=2",
            "-vf",
            "setpts=2*floor(N/2)+gte(N\\,30)*15",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "ffv1",
            "-g",
            "2",
        ],
        &duplicates,
    )?;
    let probe = probe(&duplicates, true)?;
    let times: Vec<_> = probe["frames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["best_effort_timestamp"].as_i64().unwrap())
        .collect();
    assert!(times.windows(2).any(|p| p[0] == p[1]));
    assert!(times.windows(2).any(|p| p[1] - p[0] > 400));
    compare(&duplicates)?;
    Ok(())
}

#[test]
fn complete_moov_with_a_short_media_payload_uses_displayed_frames() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("short.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x48:r=30:d=3",
            "-c:v",
            "libx264",
            "-bf",
            "0",
            "-g",
            "12",
            "-movflags",
            "faststart",
        ],
        &path,
    )?;
    compare(&path)?;
    let mut file = std::fs::read(&path)?;
    let mut pos = 0;
    loop {
        let size = u32::from_be_bytes(file[pos..pos + 4].try_into()?) as usize;
        if &file[pos + 4..pos + 8] == b"mdat" {
            let short = size / 2;
            file[pos..pos + 4].copy_from_slice(&(short as u32).to_be_bytes());
            file.truncate(pos + short);
            break;
        }
        pos += size;
    }
    std::fs::write(&path, file)?;
    compare(&path)
}

#[test]
fn field_coded_track_uses_displayed_ordinals() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("fields.mov");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x48:r=30:d=2",
            "-c:v",
            "libx264",
            "-g",
            "12",
            "-flags",
            "+ildct+ilme",
            "-x264-params",
            "tff=1",
        ],
        &path,
    )?;
    compare(&path)
}

#[test]
fn first_picture_reports_its_display_rotation() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("orientation.mkv");
    ffmpeg(
        &[
            "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10",
            "-frames:v", "4", "-c:v", "libx264", "-bsf:v",
            "h264_metadata=display_orientation=insert:rotate=90",
        ],
        &path,
    )?;
    let revision = MediaRevision::open(&path)?;
    let index = PictureIndex::open(&revision, [64, 48])?;
    assert_eq!(index.first_rotation(), Some(90));
    #[cfg(target_os = "linux")]
    assert_eq!(index.native_matrix_at(0)?, None);
    Ok(())
}

#[test]
fn source_raster_allows_trailing_padding_and_requires_the_indexed_extent() {
    let padded = "width=320|height=192|crop_left=0|crop_top=0|crop_right=4|crop_bottom=16";
    assert_eq!(matches_source_raster(padded, [316, 176]), Ok(true));
    // Same visible dimensions, moved four pixels within the coded frame.
    let left = padded.replace("crop_left=0", "crop_left=4").replace("crop_right=4", "crop_right=0");
    let top = padded.replace("crop_top=0", "crop_top=4").replace("crop_bottom=16", "crop_bottom=12");
    assert_eq!(matches_source_raster(&left, [316, 176]), Ok(false));
    assert_eq!(matches_source_raster(&top, [316, 176]), Ok(false));
    assert_eq!(matches_source_raster(padded, [320, 180]), Ok(false));
    assert_eq!(matches_source_raster("width=320|height=192", [320, 192]), Ok(true));
    assert_eq!(matches_source_raster("width=320|height=192", [316, 176]), Err(MissingCropOffsets));
}

#[test]
fn source_probe_sees_a_changed_crop_at_the_same_display_size() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let segment = dir.path().join("segment.h264");
    let mut joined = Vec::new();
    for (source, crop) in [
        ("testsrc2=s=64x48:r=30", "h264_metadata=crop_right=0:crop_bottom=0"),
        ("testsrc2=s=68x52:r=30", "h264_metadata=crop_left=4:crop_top=4"),
    ] {
        ffmpeg(&["-f", "lavfi", "-i", source, "-frames:v", "2",
            "-c:v", "libx264", "-threads", "1", "-bf", "0", "-bsf:v", crop], &segment)?;
        joined.extend(std::fs::read(&segment)?);
    }
    std::fs::write(&segment, joined)?;
    let stream = dir.path().join("changing.ts");
    ffmpeg(&["-r", "30", "-i", segment.to_str().unwrap(), "-c", "copy"], &stream)?;

    let revision = MediaRevision::open(&stream)?;
    let (mut probe, first) = Probe::start(Probe::pictures(&revision, Some([64, 48]))?)?;
    let mut eligible = vec![first.matrix.ok().flatten().is_some()];
    while let Some(picture) = probe.next()? {
        eligible.push(picture.matrix.ok().flatten().is_some());
    }
    assert_eq!(eligible, [true, true, false, false]);
    Ok(())
}

#[test]
fn padded_pictures_without_ffprobe_crop_fields_explain_the_software_fallback() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("padded.mp4");
    ffmpeg(&["-f", "lavfi", "-i", "color=s=1920x1080", "-frames:v", "2",
        "-c:v", "libx264", "-threads", "1"], &path)?;
    let revision = MediaRevision::open(&path)?;
    // Project the fields a pre-6.1 probe supplies for the 1920x1088 coded raster.
    let command = Probe::command(&revision, "frame=best_effort_timestamp,width,height,pix_fmt,color_space",
        Some([1920, 1080]))?;
    let (mut probe, first) = Probe::start(command)?;
    let mut state = State::new(first, Tail::Complete);
    state.append(probe.next()?)?;
    let index = PictureIndex(Mutex::new(state));
    for ordinal in 0..2 {
        assert!(index.pts(ordinal)?.is_some(), "portable picture clock");
        let reason = index.native_matrix_at(ordinal).unwrap_err().to_string();
        assert!(reason.contains("software decoding") && reason.contains("crop offsets") && reason.contains("6.1"), "{reason}");
    }
    Ok(())
}
