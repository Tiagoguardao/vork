use crate::footage::Video;
use anyhow::{Context, Result, ensure};
use std::{ffi::c_void, ptr::NonNull};

#[cfg(vork_vtdecode)]
use crate::footage_metal::Session;
#[cfg(vork_nvdecode)]
use crate::footage_nvidia::Session;

unsafe extern "C" {
    fn vork_decode_supported_size(width: i32, height: i32) -> i32;
    fn vork_decode_seek(decoder: *mut c_void, pts: i64) -> i32;
    fn vork_decode_next(decoder: *mut c_void, position: *mut i64, pts: *mut i64) -> i32;
    fn vork_decode_close(decoder: *mut c_void);
}
pub(super) struct Native(pub(super) NonNull<c_void>);
// The session is used exclusively. The worker owns the decoder after initialization.
unsafe impl Send for Native {}
impl Drop for Native {
    fn drop(&mut self) {
        unsafe { vork_decode_close(self.0.as_ptr()) };
    }
}
pub(super) fn select_frame(
    decoder: *mut c_void,
    pictures: &vork_media::PictureIndex,
    next: &mut usize,
    sought: &mut bool,
    index: usize,
) -> Result<()> {
    let mut anchor = if *next == index { None } else { pictures.seek_anchor(index)? };
    let mut reposition = *next > index || anchor.is_some_and(|(start, _)| *next < start);
    'landing: loop {
        let start = anchor.map_or(0, |(i, _)| i);
        if reposition {
            let time = match anchor {
                Some((_, raw)) => raw,
                None => pictures.pts(0)?.context("missing first picture")?,
            };
            ensure!(unsafe { vork_decode_seek(decoder, time) } == 0, "seek failed");
            *next = start;
            *sought = true;
        }
        while *next <= index {
            let (mut position, mut pts) = (0, 0);
            match unsafe { vork_decode_next(decoder, &mut position, &mut pts) } {
                0 => {}
                -2 => anyhow::bail!("NVDEC reference failed after three submissions before picture {}", *next),
                _ => anyhow::bail!("native decoder ended before picture {}", *next),
            }
            let stamp = if *sought || (cfg!(vork_nvdecode) && position >= 0) {
                vork_media::PictureStamp::PacketPosition(position)
            } else {
                vork_media::PictureStamp::Displayed(pts)
            };
            if !pictures.matches(*next, stamp)? {
                if reposition && *next == start && start != 0 {
                    pictures.reject_anchor(start)?;
                    anchor = pictures.seek_anchor(start - 1)?;
                    reposition = true;
                    continue 'landing;
                }
                anyhow::bail!("decoded picture {stamp:?} differs from picture {}", *next);
            }
            *next += 1;
        }
        break;
    }
    Ok(())
}
pub(super) fn admit(video: &Video) -> Result<()> {
    ensure!(video.container_crop.is_none(), "container clean aperture requires software decoding");
    ensure!(video.alpha_decoder.is_none(), "VPx alpha requires the libvpx decoder");
    ensure!(!video.map_hdr, "HDR transfer requires the canvas tone map");
    let [w, h] = video.size;
    ensure!(
        video.index.native_matrix_at(0)?.is_some() &&
            unsafe { vork_decode_supported_size(w as i32, h as i32) } != 0,
        "unsupported video geometry, pixel format or colour matrix"
    );
    Ok(())
}
/// Overlap the next contiguous decode with drawing the displayed texture. Only
/// a second sequential request starts prediction. The worker owns all decoder
/// and transfer state, and can hold at most one unrequested frame.
pub(super) struct Reader {
    requests: Option<std::sync::mpsc::Sender<usize>>,
    frames: std::sync::mpsc::Receiver<Result<wgpu::Texture>>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl Reader {
    pub(super) fn open(gpu: &wgpu::Device, queue: &wgpu::Queue, video: &Video) -> Result<Self> {
        admit(video)?;
        let mut session = Session::open(gpu, queue, video)?;
        let pictures = video.index.clone();
        let mut next = 0;
        let mut sought = false;
        let (requests, input) = std::sync::mpsc::channel::<usize>();
        let (output, frames) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("hardware footage".into())
            .spawn(move || {
                let mut read = |index| -> Result<wgpu::Texture> {
                    let matrix = pictures.native_matrix_at(index)?.context("unsupported pixel format or colour matrix")?;
                    select_frame(session.decoder(), &pictures, &mut next, &mut sought, index)?;
                    session.convert(matrix.av_colorspace())
                };
                let mut previous = None;
                let mut predicted = None;
                for index in input {
                    let frame = match predicted.take() {
                        Some((i, result)) if i == index => result,
                        _ => read(index),
                    };
                    let ok = frame.is_ok();
                    if output.send(frame).is_err() || !ok {
                        break;
                    }
                    if index.checked_sub(1).is_some_and(|i| Some(i) == previous)
                        && pictures.pts(index + 1).is_ok_and(|pts| pts.is_some())
                    {
                        predicted = Some((index + 1, read(index + 1)));
                    }
                    previous = Some(index);
                }
            })?;
        Ok(Self {
            requests: Some(requests),
            frames,
            thread: Some(thread),
        })
    }
    pub(super) fn frame(&mut self, index: usize) -> Result<wgpu::Texture> {
        self.requests
            .as_ref()
            .unwrap()
            .send(index)
            .context("hardware footage reader stopped")?;
        self.frames
            .recv()
            .context("hardware footage reader stopped")?
    }
}
impl Drop for Reader {
    fn drop(&mut self) {
        self.requests.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{footage::Source, footage_canvas::CanvasDecoder};

    #[test]
    #[ignore = "requires native hardware footage decode"]
    fn retained_device_frames_match_portable_decoder_across_seeks() -> Result<()> {
        let gpu = crate::test_renderer()?;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("b-frames.mp4");
        let encoded = std::process::Command::new("ffmpeg")
            .args([
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=96x64:rate=12:duration=2",
                "-c:v",
                "libx264",
                "-bf",
                "2",
                "-g",
                "6",
                "-colorspace",
                "bt709",
                "-color_range",
                "tv",
            ])
            .arg(&path)
            .status()?;
        ensure!(encoded.success(), "encode B-frame fixture");
        let mut source = Source::open(&path)?;
        source.prepare_gpu(&gpu);
        let Source::Video(video) = source else {
            unreachable!()
        };
        let mut native = Reader::open(&gpu.device, &gpu.queue, &video)?;
        let mut portable = CanvasDecoder::default();
        let mut held = Vec::new();
        for index in [0, 1, 17, 16, 9, 0, 1, 1, 23] {
            held.push((
                native.frame(index)?,
                portable.frame(&video, index, false, 128)?,
            ));
        }
        for (texture, reference) in held {
            compare_texture(&gpu, &texture, &reference.rgba_premul)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires native hardware footage decode and FFmpeg 6.1+ with libx264"]
    fn canvas_offset_crop_matches_the_exact_software_crop() -> Result<()> {
        use crate::footage_canvas::CanvasDevice;
        use moda_assets::AssetEntry;
        use std::process::Command;

        let gpu = crate::test_renderer()?;
        let device = CanvasDevice::new(&gpu, false);
        let dir = tempfile::tempdir()?;
        let full = dir.path().join("full.mp4");
        let cropped = dir.path().join("cropped.mp4");
        ensure!(Command::new("ffmpeg").args(["-v", "error", "-f", "lavfi", "-i",
            "testsrc2=s=320x180:r=30", "-frames:v", "16", "-c:v", "libx264",
            "-threads", "2", "-bf", "0", "-g", "16"]).arg(&full).status()?.success(),
            "encode H.264 control");
        ensure!(Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(&full)
            .args(["-c", "copy", "-bsf:v", "h264_metadata=crop_left=4:crop_top=4"])
            .arg(&cropped).status()?.success(), "set SPS crop");

        // Both streams have bottom macroblock padding. The original has a zero
        // origin and must exercise the native reader on this device.
        let mut control = Source::open(&full)?;
        let frame = device.decoder().canvas_frame(&mut control, 0, false, 320, [320, 180])?;
        ensure!(matches!(frame, AssetEntry::DeviceFrame(..)), "expected native control");

        // Use the same compressed pictures without the offset metadata and
        // crop in source YUV before swscale's conversion to deep RGB.
        let raw = Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(&full)
            .args(["-vf", "crop=316:176:4:4:exact=1", "-pix_fmt", "rgba64le",
                "-f", "rawvideo", "-"]).output()?;
        let stride = 316 * 176 * 4;
        ensure!(raw.status.success() && raw.stdout.len() == 16 * stride * 2, "crop reference");
        let expected: Vec<u8> = raw.stdout.chunks_exact(2)
            .map(|b| ((u32::from(u16::from_le_bytes([b[0], b[1]])) + 128) / 257) as u8)
            .collect();

        let mut source = Source::open(&cropped)?;
        assert_eq!(source.dimensions()?, [316, 176]);
        let mut reader = device.decoder();
        for i in (0..16).chain([0, 9, 15]) {
            let AssetEntry::Loaded(image) = reader.canvas_frame(&mut source, i, false, 320, [316, 176])?
            else { anyhow::bail!("expected portable frame {i} for offset crop") };
            ensure!(image.rgba_premul[..] == expected[i * stride..(i + 1) * stride],
                "offset crop differs from the exact reference at frame {i}");
        }
        Ok(())
    }

    #[cfg(vork_nvdecode)]
    #[test]
    #[ignore = "requires NVDEC and FFmpeg with libx264 and libx265"]
    fn trimmed_preroll_stays_native_across_seek_to_start() -> Result<()> {
        use std::process::Command;
        let gpu = crate::test_renderer()?;
        let dir = tempfile::tempdir()?;
        for (codec, open_gop) in [("libx264", false), ("libx265", false), ("libx265", true)] {
            let original = dir.path().join(format!("{codec}-{open_gop}.mp4"));
            let trimmed = dir.path().join(format!("{codec}-{open_gop}-trimmed.mp4"));
            let mut encode = Command::new("ffmpeg");
            encode.args(["-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x192:rate=24:duration=4",
                "-threads", "2", "-c:v", codec, "-bf", "2", "-g", "48", "-colorspace", "bt709"]);
            if codec == "libx265" {
                let params = if open_gop {
                    "pools=1:open-gop=1:b-adapt=0:scenecut=0:log-level=error"
                } else {
                    "pools=1:open-gop=0:log-level=error"
                };
                encode.args(["-x265-params", params]);
            }
            ensure!(encode.arg(&original).status()?.success(), "encode {codec}");
            ensure!(Command::new("ffmpeg").args(["-v", "error", "-ss", "1.5", "-i"])
                .arg(&original).args(["-c", "copy"]).arg(&trimmed).status()?.success(), "trim {codec}");
            let packets = Command::new("ffprobe").args(["-v", "error", "-select_streams", "v:0",
                "-show_entries", "packet=flags", "-of", "csv=p=0"]).arg(&trimmed).output()?;
            ensure!(packets.status.success() && packets.stdout.split(|&b| b == b'\n').next()
                .is_some_and(|flags| flags.contains(&b'D')), "{codec} fixture lacks discarded preroll");
            let raw = Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(&trimmed)
                .args(["-fps_mode", "passthrough", "-pix_fmt", "rgba64le", "-f", "rawvideo", "-"]).output()?;
            let stride = 320 * 192 * 8;
            ensure!(raw.status.success() && raw.stdout.len() == 60 * stride, "{codec} software reference");
            let Source::Video(video) = Source::open(&trimmed)? else { unreachable!() };
            // Reader has no portable fallback. Every requested texture must come from NVDEC.
            let mut native = Reader::open(&gpu.device, &gpu.queue, &video)?;
            for i in (0..60).chain([0, 1, 36, 0]) {
                let expected = raw.stdout[i * stride..(i + 1) * stride].chunks_exact(2)
                    .map(|b| ((u32::from(u16::from_le_bytes([b[0], b[1]])) + 128) / 257) as u8)
                    .collect::<Vec<_>>();
                let texture = native.frame(i).with_context(|| format!("{codec} trimmed picture {i}"))?;
                compare_texture(&gpu, &texture, &expected)?;
            }
        }
        Ok(())
    }

    #[cfg(vork_nvdecode)]
    #[test]
    #[ignore = "requires NVDEC and the FFmpeg 10-bit libx265 encoder"]
    fn main10_native_canvas_and_deep_match_software() -> Result<()> {
        use crate::footage_nvidia::{DeepReader, device_uuid};
        use std::process::Command;
        let gpu = crate::test_renderer()?;
        let hal = unsafe { gpu.device.as_hal::<wgpu::hal::api::Vulkan>() }.context("Vulkan")?;
        let uuid = device_uuid(&hal)?;
        drop(hal);
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("main10.mp4");
        ensure!(Command::new("ffmpeg").args(["-v", "error", "-f", "lavfi", "-i",
            "testsrc2=size=320x192:rate=24", "-frames:v", "24", "-pix_fmt", "yuv420p10le",
            "-c:v", "libx265", "-x265-params", "pools=1:frame-threads=1:keyint=12:log-level=error",
            "-colorspace", "bt709"]).arg(&path).status()?.success(), "encode Main10");
        let raw = Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(&path)
            .args(["-fps_mode", "passthrough", "-pix_fmt", "rgba64le", "-f", "rawvideo", "-"]).output()?;
        let stride = 320 * 192 * 8;
        ensure!(raw.status.success() && raw.stdout.len() == 24 * stride, "Main10 software reference");
        let Source::Video(video) = Source::open(&path)? else { unreachable!() };
        let mut canvas = Reader::open(&gpu.device, &gpu.queue, &video)?;
        let mut deep = DeepReader::open(&video, uuid)?;
        for i in (0..24).chain([1, 17, 0]) {
            let expected = &raw.stdout[i * stride..(i + 1) * stride];
            let components = deep.frame(&video, i)?.into_rgba16();
            ensure!(components.as_raw().iter().zip(expected.chunks_exact(2))
                .all(|(&c, e)| c == u16::from_le_bytes([e[0], e[1]])), "Main10 RGBA16 picture {i}");
            let rounded = expected.chunks_exact(2)
                .map(|e| ((u32::from(u16::from_le_bytes([e[0], e[1]])) + 128) / 257) as u8)
                .collect::<Vec<_>>();
            compare_texture(&gpu, &canvas.frame(i)?, &rounded)?;
        }
        Ok(())
    }

    #[cfg(vork_nvdecode)]
    #[test]
    #[ignore = "requires NVDEC and FFmpeg with libxvid"]
    fn avi_packet_identity_and_deep_components_across_seeks() -> Result<()> {
        use crate::footage_nvidia::{DeepReader, device_uuid};
        use std::process::Command;
        let gpu = crate::test_renderer()?;
        let uuid = device_uuid(&unsafe { gpu.device.as_hal::<wgpu::hal::api::Vulkan>() }.unwrap())?;
        let dir = tempfile::tempdir()?;
        for codec in ["libx264", "libxvid"] {
            let path = dir.path().join(format!("{codec}.avi"));
            ensure!(Command::new("ffmpeg").args(["-v", "error", "-f", "lavfi", "-i",
                "testsrc2=size=320x180:rate=24", "-frames:v", "60", "-threads", "1",
                "-bf", "2", "-g", "12", "-c:v", codec]).arg(&path).status()?.success(), "encode AVI");
            let mut reference = Command::new("ffmpeg");
            reference.args(["-v", "error"]);
            // MPEG-4 Part 2 permits decoder-dependent IDCT rounding. Use the
            // independent FFmpeg hardware decoder to isolate unpacking and seeking.
            if codec == "libxvid" {
                reference.args(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]);
            }
            reference.arg("-i").arg(&path);
            if codec == "libxvid" {
                reference.args(["-vf", "hwdownload,format=nv12,format=yuv420p"]);
            }
            let raw = reference.args(["-fps_mode", "passthrough", "-pix_fmt", "rgba64le", "-f", "rawvideo", "-"]).output()?;
            ensure!(raw.status.success(), "decode AVI reference");
            let Source::Video(video) = Source::open(&path)? else { unreachable!() };
            let mut reader = DeepReader::open(&video, uuid)?;
            // AVI omits packet PTS. libxvid also packs two VOPs in a chunk.
            for i in (0..24).chain([3, 15, 2, 28, 0]) {
                let image = reader.frame(&video, i).with_context(|| format!("{codec} frame {i}"))?.into_rgba16();
                let expected = &raw.stdout[i * 320 * 180 * 8..(i + 1) * 320 * 180 * 8];
                ensure!(image.as_raw().iter().zip(expected.chunks_exact(2))
                    .all(|(&a, e)| a == u16::from_le_bytes([e[0], e[1]])), "{codec} RGBA16 frame {i}");
            }
        }
        Ok(())
    }

    #[test]
    #[cfg(vork_nvdecode)]
    #[ignore = "requires NVIDIA AV1 hardware decode and FFmpeg with libaom-av1"]
    fn av1_native_frame_matches_software_decode() -> Result<()> {
        use std::process::Command;
        let gpu = crate::test_renderer()?;
        let dir = tempfile::tempdir()?;
        for grain in [false, true] {
            let path = dir.path().join(format!("av1-grain-{grain}.mp4"));
            let mut encode = Command::new("ffmpeg");
            encode.args(["-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x192:rate=24",
                "-frames:v", "36", "-threads", "2", "-c:v", "libaom-av1", "-cpu-used", "6", "-colorspace", "bt709"]);
            if grain {
                encode.args(["-aom-params", "film-grain-test=1"]);
            }
            ensure!(encode.arg(&path).status()?.success(), "encode AV1 film grain {grain}");
            let raw = Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(&path)
                .args(["-pix_fmt", "rgba64le", "-f", "rawvideo", "pipe:1"]).output()?;
            let frame_bytes = 320 * 192 * 8;
            ensure!(raw.status.success() && raw.stdout.len() == 36 * frame_bytes, "decode AV1 reference");
            let Source::Video(video) = Source::open(&path)? else { unreachable!() };
            let mut native = Reader::open(&gpu.device, &gpu.queue, &video)?;
            for (index, deep) in raw.stdout.chunks_exact(frame_bytes).enumerate() {
                let reference = deep.chunks_exact(2)
                    .map(|b| ((u32::from(u16::from_le_bytes([b[0], b[1]])) + 128) / 257) as u8)
                    .collect::<Vec<_>>();
                compare_texture(&gpu, &native.frame(index)?, &reference)?;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires native hardware footage decode"]
    fn intra_refresh_native_seeks_match_uninterrupted_pixels() -> Result<()> {
        use std::process::Command;
        let gpu = crate::test_renderer()?;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("recovery.mp4");
        ensure!(Command::new("ffmpeg").args(["-v", "error", "-f", "lavfi", "-i",
            "testsrc2=size=128x96:rate=30:duration=3", "-c:v", "libx264", "-bf", "0",
            "-x264-params", "intra-refresh=1:keyint=12:scenecut=0", "-colorspace", "bt709",
            "-color_range", "tv"]).arg(&path).status()?.success(), "generate intra refresh fixture");
        let raw = Command::new("ffmpeg").args(["-v","error","-i"]).arg(&path)
            .args(["-map","0:v:0","-fps_mode","passthrough","-f","rawvideo","-pix_fmt","rgba64le","pipe:1"]).output()?;
        let stride = 128*96*8;
        ensure!(raw.status.success() && raw.stdout.len() == stride*90, "reference decode");
        let Source::Video(video) = Source::open(&path)? else { unreachable!() };
        let mut native = Reader::open(&gpu.device, &gpu.queue, &video)?;
        for i in [12, 53, 18, 0, 89, 12] {
            let reference = raw.stdout[i*stride..(i+1)*stride].chunks_exact(2)
                .map(|b| ((u32::from(u16::from_le_bytes([b[0], b[1]])) + 128)/257) as u8).collect::<Vec<_>>();
            compare_texture(&gpu, &native.frame(i)?, &reference)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires native hardware footage decode"]
    fn held_native_reader_seeks_after_replacement_and_clock_repair() -> Result<()> {
        let gpu = crate::test_renderer()?;
        let dir = tempfile::tempdir()?;
        let live = dir.path().join("repeated-raw.mp4");
        use std::process::Command;
        let generator = dir.path().join("movies.py");
        std::fs::write(&generator, include_str!("../tests/fixtures/footage/movies.py"))?;
        ensure!(Command::new("python3").arg(&generator).arg(dir.path()).arg("native-held")
            .status()?.success(), "generate late-moov fixture");
        let Source::Video(video) = Source::open(&live)? else { unreachable!() };
        let requests = [41, 13, 12, 11, 10, 0, 89];
        let mut portable = CanvasDecoder::default();
        let reference = requests.iter().map(|&i| portable.frame(&video, i, false, 256))
            .collect::<Result<Vec<_>>>()?;
        // Open a fresh index so the native lookup also grows its held clock.
        let Source::Video(video) = Source::open(&live)? else { unreachable!() };
        std::fs::rename(dir.path().join("control.mp4"), &live)?;
        let mut native = Reader::open(&gpu.device, &gpu.queue, &video)?;
        for (i, frame) in requests.into_iter().zip(reference) {
            compare_texture(&gpu, &native.frame(i)?, &frame.rgba_premul)?;
        }
        Ok(())
    }

    #[cfg(vork_nvdecode)]
    #[test]
    #[ignore = "requires NVDEC and the FFmpeg libvpx-vp9 encoder"]
    fn damaged_picture_preserves_display_ordinals_across_native_fallback() -> Result<()> {
        use crate::footage_canvas::CanvasDevice;
        use moda_assets::AssetEntry;
        use vork_time::exact::{Germ, Rational};

        let gpu = crate::test_renderer()?;
        let dir = tempfile::tempdir()?;
        let generator = dir.path().join("movies.py");
        std::fs::write(&generator, include_str!("../tests/fixtures/footage/movies.py"))?;
        let status = std::process::Command::new("python3")
            .arg(generator).arg(dir.path()).arg("native-damage").status()?;
        ensure!(status.success(), "generate damaged VP9 sample");

        let Source::Video(clean) = Source::open(&dir.path().join("clean.mp4"))? else {
            unreachable!()
        };
        let mut reference = CanvasDecoder::default();
        let mut source = Source::open(&dir.path().join("damaged.mp4"))?;
        source.prepare_gpu(&gpu);
        let mut reader = CanvasDevice::new(&gpu, false).decoder();
        let fps = vork_time::Time::integer(30);
        for tick in (0..=12).chain([9, 11]) {
            let time = Germ::point(Rational::new(tick.into(), 30.into()));
            let ordinal = source.frame_index_exact(&time, fps)?;
            // The software clock drops packet ten. Hold nine at its time,
            // then display the undamaged key at eleven as ordinal ten.
            assert_eq!(ordinal, if tick >= 10 { tick - 1 } else { tick });
            if tick == 10 {
                // The canvas retains the image it already displayed at nine.
                // Keep the next pixel read contiguous across the damaged packet.
                continue;
            }
            let frame = reader.canvas_frame(&mut source, ordinal, false, 320, [320, 180])?;
            if tick == 0 {
                ensure!(matches!(frame, AssetEntry::DeviceFrame(..)), "exercise native playback");
            }
            let expected = reference.frame(&clean, tick, false, 320)?;
            match frame {
                AssetEntry::DeviceFrame(_, texture) => {
                    compare_texture(&gpu, &texture, &expected.rgba_premul)?;
                }
                AssetEntry::Loaded(image) => assert_eq!(image.rgba_premul, expected.rgba_premul),
                _ => anyhow::bail!("missing decoded canvas picture"),
            }
        }
        Ok(())
    }

    #[cfg(vork_nvdecode)]
    #[test]
    #[ignore = "requires NVDEC and the FFmpeg libvpx-vp9 encoder"]
    fn vp9_resize_rejects_incompatible_surfaces() -> Result<()> {
        use std::process::Command;
        let gpu = crate::test_renderer()?;
        let dir = tempfile::tempdir()?;
        for (file, size) in [("first.webm", "320x192"), ("second.webm", "384x192")] {
            let source = format!("testsrc2=size={size}:rate=24");
            ensure!(Command::new("ffmpeg").current_dir(dir.path()).args(["-v", "error", "-f",
                "lavfi", "-i", &source, "-frames:v", "12", "-threads", "2", "-c:v", "libvpx-vp9",
                "-cpu-used", "6", "-colorspace", "bt709", file]).status()?.success(), "encode VP9 segment");
        }
        std::fs::write(dir.path().join("parts.txt"), "file first.webm\nfile second.webm\n")?;
        ensure!(Command::new("ffmpeg").current_dir(dir.path()).args(["-v", "error", "-f", "concat",
            "-i", "parts.txt", "-c", "copy", "resize.webm"]).status()?.success(), "join VP9 segments");
        let path = dir.path().join("resize.webm");
        let Source::Video(video) = Source::open(&path)? else { unreachable!() };
        let mut portable = CanvasDecoder::default();
        let reference = (0..24)
            .map(|i| portable.frame(&video, i, false, 384))
            .collect::<Result<Vec<_>>>()?;

        let mut source = Source::open(&path)?;
        source.prepare_gpu(&gpu);
        let mut reader = crate::footage_canvas::CanvasDevice::new(&gpu, false).decoder();
        for (request, i) in (0..24).chain([0, 17, 12, 23]).enumerate() {
            let frame = reader.canvas_frame(&mut source, i, false, 384, [320, 192])?;
            match frame {
                moda_assets::AssetEntry::DeviceFrame(_, texture) if request < 12 => {
                    compare_texture(&gpu, &texture, &reference[i].rgba_premul)?;
                }
                moda_assets::AssetEntry::Loaded(image) if request >= 12 => {
                    assert_eq!(image.rgba_premul, reference[i].rgba_premul, "portable frame {i}");
                }
                _ => anyhow::bail!("unexpected VP9 decoder at frame {i}"),
            }
        }
        Ok(())
    }

    pub(super) fn compare_texture(
        gpu: &crate::Renderer,
        texture: &wgpu::Texture,
        reference: &[u8],
    ) -> Result<()> {
        assert_eq!(texture.mip_level_count(), 1);
        let row = texture.width() * 4;
        let pitch = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        assert_eq!(reference.len(), (row * texture.height()) as usize);
        let read = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("decoded frame comparison"),
            size: u64::from(pitch) * u64::from(texture.height()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut copy = gpu.device.create_command_encoder(&Default::default());
        copy.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &read,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(pitch),
                    rows_per_image: None,
                },
            },
            texture.size(),
        );
        let submission = gpu.queue.submit([copy.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })?;
        rx.recv()??;
        let data = read.slice(..).get_mapped_range();
        for (actual, expected) in data
            .chunks_exact(pitch as usize)
            .zip(reference.chunks_exact(row as usize))
        {
            assert_eq!(&actual[..row as usize], expected);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "nv12_vectors.rs"]
mod vectors;
