//! Read ahead on the decoder pipe and publish immutable canvas pixels. The world
//! texture consumer samples the deep intermediate and uses its own session.
use super::footage::{DecoderProcess, Video};
use anyhow::{Context, Result};
#[cfg(test)]
use std::process::Command;
use moda_assets::PreparedImage;
use std::{
    sync::{Arc, mpsc},
    thread,
};

// Direct YUV -> RGBA8 uses different swscale coefficients and rounding. Keep
// the reference conversion, then round v/257 exactly. zscale vectorizes that
// final step on planar RGB. Its alpha rounding differs, so only use it when
// every frame is opaque. The packed LUT preserves all four channels.
const OPAQUE_FILTER: &str = "format=rgba64le,format=gbrp16le,zscale=matrixin=gbr:matrix=gbr:rangein=full:range=full:dither=none,format=gbrp,format=rgba";
const RGBA_FILTER: &str = "format=rgba64le,lutrgb=r=round(val/257)*256:g=round(val/257)*256:b=round(val/257)*256:a=round(val/257)*256,format=rgba";

fn canvas_filter(opaque: bool) -> &'static str {
    if opaque
        && vork_inputs::tool_output("ffmpeg", &["-hide_banner", "-filters"])
            .is_ok_and(|output| {
                output.status.success()
                    && String::from_utf8_lossy(&output.stdout)
                        .lines()
                        .any(|line| line.split_whitespace().nth(1) == Some("zscale"))
            })
    {
        OPAQUE_FILTER
    } else {
        RGBA_FILTER
    }
}

struct Session {
    child: Option<DecoderProcess>,
    ready: Option<mpsc::Receiver<Result<Arc<PreparedImage>>>>,
    worker: Option<thread::JoinHandle<()>>,
    next: usize,
}

impl Drop for Session {
    fn drop(&mut self) {
        self.child.take(); // Reap before the producer can close the reader.
        self.ready.take(); // Unblock a producer waiting to publish.
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Session {
    fn start(
        child: DecoderProcess,
        mut input: super::footage::pixels::PixelInput,
        size: [u32; 2],
        straight: bool,
        opaque: bool,
        next: usize,
    ) -> Result<Self> {
        let (sender, ready) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("movie input".into())
            .spawn(move || {
                let components = size[0] as usize * size[1] as usize * 4;
                // One displayed, one queued, one being filled. Extra holders (e.g.
                // an asset-store snapshot) keep their immutable pixels until released.
                let allocate = || {
                    Arc::new(PreparedImage::from_premultiplied_with_coverage(
                        vec![0; components], size[0], size[1], 0,
                    ))
                };
                let mut pool: [Arc<PreparedImage>; 4] = std::array::from_fn(|_| allocate());
                for index in (0..pool.len()).cycle() {
                    let shared = &mut pool[index];
                    if Arc::strong_count(shared) != 1 || Arc::strong_count(&shared.rgba_premul) != 1
                    {
                        *shared = allocate();
                    }
                    // Dissociate weak GPU cache keys before publishing new contents.
                    let frame = Arc::make_mut(shared);
                    let pixels = Arc::make_mut(&mut frame.rgba_premul);
                    if let Err(error) = input.read_frame(pixels) {
                        let _ = sender.send(Err(error));
                        break;
                    }
                    if opaque {
                        frame.coverage_permille = 1000;
                    } else {
                        let covered = if straight {
                            pixels.chunks_exact(4).filter(|p| p[3] != 0).count() as u64
                        } else {
                            moda_assets::decode::premultiply_in_place(pixels)
                        };
                        frame.coverage_permille = moda_assets::decode::coverage_permille(
                            covered,
                            (components / 4) as u64,
                        );
                    }
                    frame.straight = straight;
                    if sender.send(Ok(shared.clone())).is_err() {
                        break;
                    }
                }
            });
        match worker {
            Ok(worker) => Ok(Self {
                child: Some(child),
                ready: Some(ready),
                worker: Some(worker),
                next,
            }),
            Err(error) => Err(error.into()),
        }
    }
}

#[derive(Default)]
pub(crate) struct CanvasDecoder {
    #[cfg(any(vork_nvdecode, vork_vtdecode))]
    device: Option<(wgpu::Device, wgpu::Queue)>,
    #[cfg(any(vork_nvdecode, vork_vtdecode))]
    native: Option<crate::footage_device::Reader>,
    filter: Option<&'static str>,
    session: Option<Session>,
    software_only: bool,
    previous: Option<usize>,
    predicted: Option<(usize, Arc<PreparedImage>)>,
}

#[derive(Default)]
pub(super) struct CanvasDevice {
    #[cfg(any(vork_nvdecode, vork_vtdecode))]
    gpu: Option<(wgpu::Device, wgpu::Queue)>,
}

impl CanvasDevice {
    pub(super) fn new(renderer: &crate::Renderer, straight: bool) -> Self {
        #[cfg(vork_nvdecode)]
        let supported = renderer.adapter_info.vendor == crate::constants::NVIDIA_VENDOR_ID;
        #[cfg(vork_vtdecode)]
        let supported = renderer.adapter_info.backend == wgpu::Backend::Metal;
        #[cfg(not(any(vork_nvdecode, vork_vtdecode)))]
        let _ = (renderer, straight);
        Self {
            #[cfg(any(vork_nvdecode, vork_vtdecode))]
            gpu: (supported && !straight).then(|| (renderer.device.clone(), renderer.queue.clone())),
        }
    }

    pub(super) fn decoder(&self) -> CanvasDecoder {
        CanvasDecoder {
            #[cfg(any(vork_nvdecode, vork_vtdecode))]
            device: self.gpu.clone(),
            ..Default::default()
        }
    }
}

impl CanvasDecoder {
    #[cfg(any(vork_nvdecode, vork_vtdecode))]
    pub(crate) fn device_frame(
        &mut self,
        video: &Video,
        index: usize,
    ) -> Option<moda_assets::AssetEntry> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SERIAL: AtomicU64 = AtomicU64::new(1);
        let (device, queue) = self.device.as_ref()?;
        let frame = (|| {
            if self.native.is_none() {
                self.native = Some(crate::footage_device::Reader::open(device, queue, video)?);
            }
            self.native.as_mut().unwrap().frame(index)
        })();
        match frame {
            Ok(texture) => Some(moda_assets::AssetEntry::DeviceFrame(
                moda_assets::ExternalImageMeta {
                    width: video.size[0],
                    height: video.size[1],
                    scale_factor: (1., 1.),
                    upload_id: SERIAL.fetch_add(1, Ordering::Relaxed),
                },
                Arc::new(texture),
            )),
            Err(err) => {
                eprintln!(
                    "movie {} frame {index}: using portable decoder ({err})",
                    video.path.display()
                );
                self.native = None;
                self.device = None;
                None
            }
        }
    }

    pub(super) fn canvas_frame(
        &mut self, source: &mut crate::footage::Source, index: usize, straight: bool, limit: u32, size: [u32; 2],
    ) -> Result<moda_assets::AssetEntry> {
        let _decode = moda_engine::prof::span("canvas.input_decode");
        if let crate::footage::Source::Video(video) = source {
            #[cfg(any(vork_nvdecode, vork_vtdecode))]
            if let Some(entry) = self.device_frame(video, index) { return Ok(entry); }
            return Ok(moda_assets::AssetEntry::Loaded(self.frame(video, index, straight, limit)?));
        }
        let frame = source.frame_at_index(index, limit)?.into_rgba8();
        anyhow::ensure!([frame.width(), frame.height()] == size, "frame dimensions changed; use a sequence with constant dimensions");
        let mut pixels = frame.into_raw();
        let prepared = if straight {
            PreparedImage::from_straight(pixels, size[0], size[1])
        } else {
            let covered = moda_assets::decode::premultiply_in_place(&mut pixels);
            PreparedImage::from_premultiplied_with_coverage(pixels, size[0], size[1], covered)
        };
        Ok(moda_assets::AssetEntry::Loaded(Arc::new(prepared)))
    }

    pub(crate) fn frame(
        &mut self,
        video: &Video,
        index: usize,
        straight: bool,
        limit: u32,
    ) -> Result<Arc<PreparedImage>> {
        video.check_limit(limit)?;
        let hardware = video.hardware && !self.software_only;
        let mut result = self.read_frame(video, index, straight, hardware, false);
        if result.as_ref().is_err_and(|e| e.is::<super::footage::pixels::WrongLanding>()) {
            self.session = None;
            self.predicted = None;
            result = self.read_frame(video, index, straight, hardware, true);
        }
        match result {
            Err(error) if hardware => {
                eprintln!(
                    "movie {} frame {index}: retrying with software decode ({error:#})",
                    video.path.display()
                );
                self.software_only = true;
                self.session = None;
                self.predicted = None;
                self.read_frame(video, index, straight, false, true)
            }
            result => result,
        }
    }

    fn read_frame(
        &mut self,
        video: &Video,
        index: usize,
        straight: bool,
        hardware: bool,
        from_start: bool,
    ) -> Result<Arc<PreparedImage>> {
        let previous = self.previous.replace(index);
        if let Some((cached, frame)) = self.predicted.take() {
            if cached == index {
                return Ok(frame);
            }
        }
        let preceding = previous
            .filter(|&p| p > index)
            .map(|p| index.saturating_sub(p - index));
        let session = &mut self.session;
        let anchor = if from_start { None } else { video.pixel_anchor(index, session.as_ref().map_or(0, |s| s.next))? };
        let start = anchor.map_or(0, |(i, _)| i);
        if session
            .as_ref()
            .is_none_or(|s| s.next > index || s.next < start)
        {
            *session = None;
            let filter = *self.filter.get_or_insert_with(|| canvas_filter(video.opaque));
            let (child, input, next) = video.start_pixels(anchor, Some(filter), hardware)?;
            *session = Some(Session::start(child, input, video.size, straight, video.opaque, next)?);
        }
        let mut live = session.take().unwrap();
        loop {
            let frame = live
                .ready
                .as_ref()
                .unwrap()
                .recv()
                .context("canvas movie worker stopped")?
                .with_context(|| {
                    format!("decode video {} frame {}", video.path.display(), live.next)
                })?;
            if preceding == Some(live.next) && live.next != index {
                self.predicted = Some((live.next, frame.clone()));
            }
            live.next += 1;
            if live.next > index {
                *session = Some(live);
                return Ok(frame);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::footage::Source;

    #[test]
    fn opaque_canvas_matches_deep_rounding_and_reuses_pixels() -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("vork-canvas-opaque-{}.mp4", std::process::id()));
        let status = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x48:rate=12:duration=1",
                "-colorspace",
                "bt709",
                "-color_range",
                "tv",
                "-c:v",
                "libx264",
            ])
            .arg(&path)
            .status()?;
        anyhow::ensure!(status.success(), "create 8-bit opaque fixture");
        let Source::Video(video) = Source::open(&path)? else {
            unreachable!()
        };
        let mut reference = Source::open(&path)?;
        let mut decoder = CanvasDecoder::default();
        let mut buffers = std::collections::HashSet::new();
        for index in 0..12 {
            let frame = decoder.frame(&video, index, false, 128)?;
            let expected = reference
                .frame(
                    index as f32 / 12. + 0.001,
                    vork_time::Time::integer(12),
                    128,
                )?
                .into_rgba8();
            assert_eq!(frame.rgba_premul.as_slice(), expected.as_raw());
            assert_eq!(frame.coverage_permille, 1000);
            buffers.insert(frame.rgba_premul.as_ptr());
        }
        assert_eq!(buffers.len(), 4);
        drop(decoder);
        drop(reference);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn canvas_bytes_match_deep_frames_across_seeks_and_loops() -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("vork-canvas-deep-{}.mkv", std::process::id()));
        let status = Command::new("ffmpeg").args(["-v","error","-y","-f","lavfi","-i",
            "testsrc2=size=64x48:rate=12:duration=1,format=yuva444p10le,geq=lum=lum(X\\,Y):cb=cb(X\\,Y):cr=cr(X\\,Y):a=mod(X*37+Y*7\\,1024)",
            "-c:v","ffv1","-g","4"]).arg(&path).status()?;
        anyhow::ensure!(status.success(), "create 10-bit alpha fixture");
        for straight in [true, false] {
            let Source::Video(video) = Source::open(&path)? else {
                unreachable!()
            };
            let mut source = Source::open(&path)?;
            let mut decoder = CanvasDecoder::default();
            let mut held = Vec::new();
            for index in [0, 1, 2, 4, 8, 10, 9, 8, 7, 0, 1, 1, 11] {
                let frame = decoder.frame(&video, index, straight, 128)?;
                let mut expected = source
                    .frame(
                        index as f32 / 12.0 + 0.001,
                        vork_time::Time::integer(12),
                        128,
                    )?
                    .into_rgba8()
                    .into_raw();
                if !straight {
                    for pixel in expected.chunks_exact_mut(4) {
                        for c in 0..3 {
                            pixel[c] =
                                ((u16::from(pixel[c]) * u16::from(pixel[3]) + 127) / 255) as u8;
                        }
                    }
                }
                held.push((frame, expected));
            }
            for (frame, expected) in held {
                assert_eq!(*frame.rgba_premul, expected);
            }
        }
        std::fs::remove_file(path)?;
        Ok(())
    }
}
