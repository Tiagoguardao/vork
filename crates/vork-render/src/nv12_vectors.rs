// Keep the exact synthetic values through FFmpeg and NVDEC. Metal consumes
// the identical bytes in an IOSurface as VideoToolbox can reject I_PCM.
// Baseline H.264 I_PCM carries literal 4:2:0 samples, including excursions
// outside TV range. A lossy encoder would move the rounding-boundary vectors.
use super::*;
use crate::{footage::Source, footage_canvas::CanvasDecoder};
use std::process::Command;

// NVDEC on L4 requires coded width >= 48 for H.264.
const W: usize = 64;
const H: usize = 32;

unsafe extern "C" {
    fn vork_nv12_reference(nv12: *const u8, w: i32, h: i32, matrix: i32, rgba: *mut u8);
}

fn frames() -> Vec<Vec<u8>> {
    let make = |luma: &dyn Fn(usize, usize) -> u8, uv: &dyn Fn(usize, usize) -> [u8; 2]| {
        let mut bytes = Vec::new();
        for y in 0..H {
            for x in 0..W {
                bytes.push(luma(x, y));
            }
        }
        for y in 0..H / 2 {
            for x in 0..W / 2 {
                bytes.extend(uv(x, y));
            }
        }
        bytes
    };
    let mut frames = Vec::new();
    for row in [0, 1, 2, H / 2 - 3, H / 2 - 2, H / 2 - 1] {
        frames.push(make(&|_, _| 128, &|x, y| {
            if y != row {
                [128, 128]
            } else if x % 2 == 0 {
                [255, 128]
            } else {
                [128, 0]
            }
        }));
    }
    frames.push(make(&|x, y| (x + W * y) as u8, &|x, y| {
        [(17 * x) as u8, (17 * y) as u8]
    }));
    frames.push(make(&|x, _| if x % 2 == 0 { 255 } else { 0 }, &|_, y| {
        if y % 2 == 0 { [255, 0] } else { [0, 255] }
    }));
    // Constant chroma sums exactly to 4096. These green components land on
    // opposite sides of the /257 half-step for 601 and 709, respectively.
    for (y, u, v) in [(16, 160, 28), (17, 128, 93), (23, 95, 116), (29, 95, 131)] {
        frames.push(make(&|_, _| y, &|_, _| [u, v]));
    }
    frames
}

fn pcm_stream(frames: &[Vec<u8>]) -> Vec<u8> {
    // Constrained Baseline SPS/PPS for 64x32, one slice, CAVLC, POC type 2.
    const HEADERS: &[u8] = &[
        0x0, 0x0, 0x0, 0x1, 0x67, 0x42, 0xc0, 0xa, 0xda, 0x11, 0x6c, 0x4, 0x40, 0x0, 0x0, 0x3, 0x0,
        0x40, 0x0, 0x0, 0xc, 0x83, 0xc4, 0x89, 0xa8, 0x0, 0x0, 0x0, 0x1, 0x68, 0xce, 0xf, 0x2c,
        0x80, 0x0, 0x0, 0x0, 0x1, 0x65,
    ];
    let mut stream = Vec::new();
    for frame in frames {
        stream.extend(HEADERS);
        // IDR I slice, deblocking disabled. Each I_PCM macroblock begins
        // with ue(25) and zero alignment bits, followed by Y, Cb, Cr planes.
        let mut slice = vec![0xb8, 0x4a];
        for my in 0..H / 16 {
            for mx in 0..W / 16 {
                slice.extend([0x0d, 0]);
                for y in my * 16..my * 16 + 16 {
                    let x = y * W + mx * 16;
                    slice.extend(&frame[x..x + 16]);
                }
                for channel in 0..2 {
                    for y in my * 8..my * 8 + 8 {
                        for x in mx * 8..mx * 8 + 8 {
                            slice.push(frame[W * H + y * W + 2 * x + channel]);
                        }
                    }
                }
            }
        }
        slice.push(0x80);
        for byte in slice {
            if byte <= 3 && stream.ends_with(&[0, 0]) {
                stream.push(3);
            }
            stream.push(byte);
        }
    }
    stream
}

#[cfg(vork_vtdecode)]
struct MetalVectors(NonNull<c_void>);

#[cfg(vork_vtdecode)]
unsafe extern "C" {
    fn vork_vector_conversion_open(shader: *const std::ffi::c_char) -> *mut c_void;
    fn vork_conversion_close(conversion: *mut c_void);
    fn vork_metal_vector(
        conversion: *mut c_void,
        input: *const u8,
        w: i32,
        h: i32,
        matrix: i32,
        output: *mut u8,
    ) -> i32;
}

#[cfg(vork_vtdecode)]
impl MetalVectors {
    fn open() -> Result<Self> {
        let shader = std::ffi::CString::new(include_str!(concat!(env!("OUT_DIR"), "/nv12.metal")))?;
        Ok(Self(
            NonNull::new(unsafe { vork_vector_conversion_open(shader.as_ptr()) })
                .context("open production Metal converter for raw vectors")?,
        ))
    }

    fn frame(&self, input: &[u8], matrix: i32) -> Result<Vec<u8>> {
        assert_eq!(input.len(), W * H * 3 / 2);
        let mut rgba = vec![0; W * H * 4];
        ensure!(
            unsafe {
                vork_metal_vector(
                    self.0.as_ptr(),
                    input.as_ptr(),
                    W as i32,
                    H as i32,
                    matrix,
                    rgba.as_mut_ptr(),
                )
            } != 0,
            "convert and read Metal vector"
        );
        Ok(rgba)
    }
}

#[cfg(vork_vtdecode)]
impl Drop for MetalVectors {
    fn drop(&mut self) {
        unsafe { vork_conversion_close(self.0.as_ptr()) };
    }
}

fn check_vectors(gpu: Option<&crate::Renderer>) -> Result<()> {
    let dir = tempfile::tempdir()?;
    let frames = frames();
    #[cfg(vork_vtdecode)]
    let metal = gpu.map(|_| MetalVectors::open()).transpose()?;
    let stream = dir.path().join("vectors.h264");
    std::fs::write(&stream, pcm_stream(&frames))?;
    for matrix in [1, 2, 5, 6] {
        let path = dir.path().join(format!("matrix-{matrix}.mp4"));
        let metadata = format!(
            "h264_metadata=matrix_coefficients={matrix}:colour_primaries=1:transfer_characteristics=1:video_full_range_flag=0:chroma_sample_loc_type=0"
        );
        let status = Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(&stream)
            .args(["-c", "copy", "-bsf:v", &metadata])
            .arg(&path)
            .status()?;
        ensure!(status.success(), "mux NV12 test vectors");
        let Source::Video(video) = Source::open(&path)? else {
            unreachable!()
        };
        let mut portable = CanvasDecoder::default();
        #[cfg(vork_nvdecode)]
        let mut native = gpu
            .map(|gpu| Reader::open(&gpu.device, &gpu.queue, &video))
            .transpose()?;
        #[cfg(vork_nvdecode)]
        let mut deep = gpu.map(|gpu| -> Result<_> {
            let hal = unsafe { gpu.device.as_hal::<wgpu::hal::api::Vulkan>() }.context("Vulkan")?;
            let uuid = crate::footage_nvidia::device_uuid(&hal)?;
            Ok((crate::footage_nvidia::DeepReader::open(&video, uuid)?, Source::open(&path)?))
        }).transpose()?;
        for (i, frame) in frames.iter().enumerate() {
            let mut rgba = vec![0; W * H * 4];
            unsafe {
                vork_nv12_reference(
                    frame.as_ptr(),
                    W as i32,
                    H as i32,
                    matrix,
                    rgba.as_mut_ptr(),
                )
            };
            let ffmpeg = portable.frame(&video, i, false, W as u32)?;
            assert_eq!(
                rgba.as_slice(),
                ffmpeg.rgba_premul.as_slice(),
                "matrix {matrix} frame {i}"
            );
            #[cfg(vork_nvdecode)]
            if let Some((reader, reference)) = deep.as_mut() {
                assert_eq!(
                    reader.frame(&video, i)?.into_rgba16().as_raw(),
                    reference.frame_at_index(i, W as u32)?.into_rgba16().as_raw(),
                    "RGBA16 matrix {matrix} frame {i}"
                );
            }
            #[cfg(vork_vtdecode)]
            if let Some(metal) = &metal {
                assert_eq!(
                    metal.frame(frame, matrix)?,
                    rgba,
                    "Metal matrix {matrix} frame {i}"
                );
            }
            #[cfg(vork_nvdecode)]
            if let (Some(gpu), Some(native)) = (gpu, native.as_mut()) {
                super::tests::compare_texture(gpu, &native.frame(i)?, &rgba)?;
            }
        }
    }
    Ok(())
}

#[test]
fn integer_vectors_match_ffmpeg() -> Result<()> {
    for (w, h, ok) in [
        (30, 32, false),
        (32, 30, false),
        (32, 33, false),
        (33, 32, false),
        (32, 32, true),
    ] {
        assert_eq!(unsafe { vork_decode_supported_size(w, h) } != 0, ok);
    }
    check_vectors(None)
}

#[test]
#[ignore = "requires native hardware footage decode"]
fn native_nv12_vectors_match_reference_and_ffmpeg() -> Result<()> {
    check_vectors(Some(&crate::test_renderer()?))
}
