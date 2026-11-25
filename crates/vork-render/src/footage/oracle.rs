//! Compare actual discontinuous reads against a single uninterrupted decode.
//! Comparing two readers that both use the same seek index can hide a wrong landing.
use super::*;
use crate::footage_canvas::CanvasDecoder;
use serde_json::Value;

fn oracle(path: &Path, video: &mut Video, requests: &[usize]) -> Result<usize> {
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp,key_frame:frame_side_data=",
            "-of",
            "json",
        ])
        .arg(path)
        .output()?;
    ensure!(probe.status.success(), "full frame probe");
    let json: Value = serde_json::from_slice(&probe.stdout)?;
    let frames = json["frames"]
        .as_array()
        .context("missing displayed frames")?;
    let pts: Vec<_> = frames
        .iter()
        .map(|f| f["best_effort_timestamp"].as_i64().unwrap())
        .collect();
    let output = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(path)
        .args([
            "-map",
            "0:v:0",
            "-an",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba64le",
            "pipe:1",
        ])
        .output()?;
    let bytes = (video.size[0] * video.size[1] * 8) as usize;
    ensure!(
        output.status.success() && output.stdout.len() == bytes * pts.len(),
        "uninterrupted pixel count"
    );
    if path.file_stem().is_some_and(|s| s == "damaged") {
        ensure!(
            pts.len() == 89 && pts[10] - pts[9] == 2,
            "fixture must drop sample ten"
        );
        for requests in [
            (0..89).collect::<Vec<_>>(),
            vec![41, 10, 9, 11, 63, 22, 88, 0],
        ] {
            // Each reader grows its own decoded prefix. Start the second
            // walk beyond the first post-damage key, without priming its clock.
            let mut world = Video::open(path)?;
            let canvas_video = Video::open(path)?;
            let mut canvas = CanvasDecoder::default();
            for i in requests {
                let expected = output.stdout[i * bytes..(i + 1) * bytes]
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]));
                let deep = world.frame_at(i, 256)?.into_rgba16();
                ensure!(
                    deep.as_raw().iter().copied().eq(expected.clone()),
                    "damaged world ordinal {i}"
                );
                let frame = canvas.frame(&canvas_video, i, false, 256)?;
                ensure!(
                    frame
                        .rgba_premul
                        .iter()
                        .copied()
                        .eq(expected.map(|v| ((u32::from(v) + 128) / 257) as u8)),
                    "damaged canvas ordinal {i}"
                );
            }
            ensure!(
                world.index.pts(89)?.is_none() && canvas_video.index.pts(89)?.is_none(),
                "damaged movie retained a phantom final picture"
            );
        }
        let clock = Video::open(path)?;
        ensure!(
            clock.frame_index(&Germ::point(decimal("1/3")?))? == 9,
            "hold across damage"
        );
    }
    let mut canvas = CanvasDecoder::default();
    // Exercise the production time selector before asking either reader for
    // pixels. FFprobe and the uninterrupted byte stream independently specify
    // which pixels a request at 1/5 second must produce.
    let fifth = decimal("1/5")?;
    let selected = video.frame_index(&Germ::point(fifth.clone()))?;
    let expected_index = pts
        .partition_point(|&stamp| {
            Rational::from_integer((stamp - pts[0]).into()) * &video.timebase <= fifth
        })
        .saturating_sub(1);
    ensure!(
        selected == expected_index,
        "time 0.2 s selected {selected}, expected {expected_index}"
    );
    if path.file_stem().is_some_and(|s| s == "conflict") {
        ensure!(
            selected == 4
                && output.stdout[4 * bytes..5 * bytes] != output.stdout[5 * bytes..6 * bytes],
            "conflicting clock must discriminate pictures 4 and 5"
        );
    }
    for (i, reference) in
        std::iter::once((selected, expected_index)).chain(requests.iter().map(|&r| {
            let i = r.min(pts.len() - 1);
            (i, i)
        }))
    {
        let expected = &output.stdout[reference * bytes..(reference + 1) * bytes];
        let world = video.frame_at(i, 256)?.into_rgba16();
        let pixels = expected
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]));
        ensure!(
            world.as_raw().iter().copied().eq(pixels.clone()),
            "deep pixels at {i}, PTS {}",
            pts[i]
        );
        let canvas = canvas.frame(video, i, false, 256)?;
        ensure!(
            canvas
                .rgba_premul
                .iter()
                .copied()
                .eq(pixels.map(|c| ((u32::from(c) + 128) / 257) as u8)),
            "canvas pixels at {i}, PTS {}",
            pts[i]
        );
    }
    if path
        .file_stem()
        .is_some_and(|s| s == "conflict" || s == "repeated-raw")
    {
        let mut reverse = Video::open(path)?;
        for i in (0..pts.len()).rev() {
            let before = reverse.decoded;
            let frame = reverse.frame_at(i, 256)?.into_rgba16();
            ensure!(
                frame
                    .as_raw()
                    .iter()
                    .copied()
                    .eq(output.stdout[i * bytes..(i + 1) * bytes]
                        .chunks_exact(2)
                        .map(|b| u16::from_le_bytes([b[0], b[1]]))),
                "repaired clock reverse pixels {i}"
            );
            // Stay within one GOP even after an earlier PTS repair.
            ensure!(
                reverse.decoded - before <= 12,
                "reverse request {i} decoded {} pictures",
                reverse.decoded - before
            );
        }
    }
    // Check the full index only after the pixel requests so time selection and
    // random access exercise the same unprimed prefix as a real first look.
    let mut key = None;
    for (i, &stamp) in pts.iter().enumerate() {
        if i > 0
            && frames[i]["key_frame"] == 1
            && pts[i - 1] < stamp
            && pts.get(i + 1).is_none_or(|&next| stamp < next)
        {
            key = Some((i, stamp));
        }
        ensure!(video.index.pts(i)? == Some(stamp), "PTS at {i}");
        ensure!(video.index.anchor(i)? == key, "anchor at {i}");
        if pts.get(i + 1) != Some(&stamp) {
            ensure!(
                video.index.select(|p, _| p <= stamp)? == i,
                "duplicate selection at {i}"
            );
        }
    }
    ensure!(video.index.pts(pts.len())?.is_none(), "extra indexed frame");
    Ok(pts.len())
}

fn compare_generated(mode: &str) -> Result<()> {
    let dir = tempfile::tempdir()?;
    let generator = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/footage/movies.py");
    ensure!(
        Command::new("python3")
            .arg(generator)
            .arg(dir.path())
            .arg(mode)
            .status()?
            .success(),
        "generate movies"
    );
    let mut files = std::fs::read_dir(dir.path())?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.sort();
    let mut pictures = 0;

    for path in &files {
        let mut video = Video::open(path)?;

        let requests = if mode == "pages" {
            [8192, 4095, 4096, 4110, 31, 8200, 0]
        } else {
            [12, 41, 13, usize::MAX, 1, 25, 0]
        };
        let count = oracle(path, &mut video, &requests)
            .with_context(|| format!("fixture {}", path.display()))?;
        match path.file_stem().and_then(|s| s.to_str()) {
            Some("paged-prefix") => {
                assert_eq!(video.index.pts(4095)?, video.index.pts(4096)?);
            }
            Some("sparse-sync") => {
                assert_eq!(video.index.anchor(12)?, Some((12, 42)));
            }
            Some("conflict") => {
                assert!(video.index.anchor(12)?.is_some());
                assert_eq!(video.index.seek_anchor(12)?.map(|a| a.0), Some(12));
            }
            Some("fractional") => {
                assert_eq!(count, 89);
                assert_eq!(video.index.pts(88)?, Some(118));
            }
            Some("duplicates") => assert_eq!(video.index.pts(12)?, video.index.pts(13)?),
            Some("refresh") => {
                assert_eq!(video.index.anchor(12)?, None);
                // Exercise the portable guard even if handed the unsafe container hint.
                let (_child, _, start) =
                    video.start_pixels(Some((12, video.index.pts(12)?.unwrap())), None, false)?;
                ensure!(
                    start == 0,
                    "accepted delayed recovery as the twelfth picture"
                );
            }
            _ => {}
        }
        pictures += count;
    }
    eprintln!(
        "{} movies, {pictures} displayed PTS/anchor pairs, {} deep/canvas seek comparisons",
        files.len(),
        files.len() * 16
    );
    Ok(())
}

#[test]
fn movie_table_regressions() -> Result<()> {
    compare_generated("regressions")
}

#[test]
#[ignore = "Seeded MOV/MP4 combinations compared to complete FFprobe and FFmpeg decodes"]
fn generated_movie_differential() -> Result<()> {
    compare_generated("variants")
}

#[test]
fn opened_movie_survives_replacement_between_readers() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let old = dir.path().join("original.mkv");
    let live = dir.path().join("clip.mkv");
    let replacement = dir.path().join("replacement.mkv");
    for (path, source, hz) in [
        (&old, "testsrc2=s=64x48:r=24:d=1", 330),
        (&replacement, "testsrc=s=32x32:r=30:d=1", 990),
    ] {
        ensure!(
            Command::new("ffmpeg")
                .args([
                    "-v", "error", "-f", "lavfi", "-i", source, "-f", "lavfi", "-i"
                ])
                .arg(format!("sine=frequency={hz}:duration=1:sample_rate=48000"))
                .args(["-c:a", "pcm_s16le", "-c:v", "libx264", "-g", "8"])
                .arg(path)
                .status()?
                .success(),
            "create revision"
        );
    }
    std::fs::copy(&old, &live)?;
    let opened = Opened::default();
    let pin = opened.revision(&live)?;
    let prefix_open = Opened::default();
    let Source::Video(mut pending) = prefix_open.source(&live)? else {
        unreachable!()
    };
    assert!(!pending.index.span_through(|_, _| false)?.3);
    // Replace after obtaining a handle, before header, first-picture, prefix,
    // native/portable pixels or the mix inspect the path.
    std::fs::rename(replacement, &live)?;
    #[cfg(target_os = "macos")]
    if let Err(e) = pin.independent_file() {
        let expected = format!(
            "{} was replaced after the scene opened it; reopen the scene",
            live.display()
        );
        ensure!(e.to_string() == expected, "{e}");
        let error = pending
            .frame_at(12, 256)
            .err()
            .context("CLI decoder opened replaced file")?;
        ensure!(format!("{error:#}").contains(&expected), "{error:#}");
        return Ok(());
    }
    ensure!(
        oracle(&old, &mut pending, &[12, 3, 18])? == 24,
        "replacement changed a pending decoded prefix"
    );
    let Source::Video(mut video) = opened.source(&live)? else {
        unreachable!()
    };
    assert!(Arc::ptr_eq(&pin, &video.revision));
    ensure!(
        oracle(&old, &mut video, &[16, 4, 23, 8])? == 24,
        "replacement leaked into the old picture index"
    );
    fn scene(file: &str) -> Result<crate::Scene> {
        Ok(serde_json::from_value(serde_json::json!({
            "version":2,"composition":{"space":"2d"},"images":{"clip":{"file":file}},
            "nodes":[{"type":"path","d":"M0 0H1V1Z","fill":{"image":"clip"},"audio":{"file":file}}]
        }))?)
    }
    let mix = |file, opened: &Opened| {
        crate::audio::Mix::new_in(&scene(file)?, dir.path(), opened)?.render(0., 0.25, None)
    };
    let old_sound = mix("original.mkv", &Opened::default())?;
    assert_eq!(mix("clip.mkv", &opened)?, old_sound);
    let fresh = Opened::default();
    assert_ne!(mix("clip.mkv", &fresh)?, old_sound);
    let Source::Video(mut new_video) = fresh.source(&live)? else {
        unreachable!()
    };
    ensure!(
        oracle(&live, &mut new_video, &[17, 3, 29])? == 30,
        "new open did not see replacement"
    );
    Ok(())
}

#[test]
fn spilled_ordinals_and_duplicate_pts_across_pages() -> Result<()> {
    compare_generated("pages")
}
