//! The decoder worker owns the CVPixelBuffer until conversion returns. The
//! returned private RGBA texture can outlive both the worker and the decoder.
use crate::{footage::Video, footage_device::Native};
use anyhow::{Context, Result};
use objc2::rc::Retained;
use std::{
    ffi::{CString, c_char, c_void},
    ptr::NonNull,
};
use wgpu::hal::api::Metal;

unsafe extern "C" {
    fn vork_decode_open(input_fd: i32, w: i32, h: i32) -> *mut c_void;
    fn vork_decode_pixels(decoder: *mut c_void, matrix: i32) -> *mut c_void;
    fn vork_conversion_open(device: *mut c_void, shader: *const c_char) -> *mut c_void;
    fn vork_conversion_close(conversion: *mut c_void);
    fn vork_conversion_frame(
        conversion: *mut c_void,
        pixels: *mut c_void,
        matrix: i32,
    ) -> *mut c_void;
}
pub(super) struct Session {
    native: Native,
    conversion: NonNull<c_void>,
    device: wgpu::Device,
    size: [u32; 2],
}
// All native state is exclusively accessed by the reader worker after open.
unsafe impl Send for Session {}
impl Drop for Session {
    fn drop(&mut self) {
        unsafe { vork_conversion_close(self.conversion.as_ptr()) }
    }
}
impl Session {
    pub(super) fn open(gpu: &wgpu::Device, _queue: &wgpu::Queue, video: &Video) -> Result<Self> {
        let [w, h] = video.size;
        use std::os::fd::{AsFd, AsRawFd};
        let input_fd = video.revision.as_fd().as_raw_fd();
        let shader = CString::new(include_str!(concat!(env!("OUT_DIR"), "/nv12.metal")))?;
        let hal = unsafe { gpu.as_hal::<Metal>() }.context("VideoToolbox canvas requires Metal")?;
        let native = Native(
            NonNull::new(unsafe { vork_decode_open(input_fd, w as i32, h as i32) })
                .context("VideoToolbox unavailable")?,
        );
        let raw = Retained::as_ptr(hal.raw_device()) as *mut c_void;
        let conversion = NonNull::new(unsafe { vork_conversion_open(raw, shader.as_ptr()) })
            .context("Metal conversion unavailable")?;
        Ok(Self {
            native,
            conversion,
            device: gpu.clone(),
            size: video.size,
        })
    }
    pub(super) fn decoder(&self) -> *mut c_void {
        self.native.0.as_ptr()
    }
    pub(super) fn convert(&mut self, matrix: i32) -> Result<wgpu::Texture> {
        let [width, height] = self.size;
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        unsafe {
            let pixels = NonNull::new(vork_decode_pixels(self.decoder(), matrix))
                .context("unsupported VideoToolbox frame")?;
            let raw = vork_conversion_frame(self.conversion.as_ptr(), pixels.as_ptr(), matrix);
            let raw = Retained::from_raw(raw.cast()).context("Metal frame conversion failed")?;
            let format = wgpu::TextureFormat::Rgba8Unorm;
            let hal = wgpu::hal::metal::Device::texture_from_raw(
                raw,
                format,
                objc2_metal::MTLTextureType::Type2D,
                1,
                1,
                extent.into(),
            );
            Ok(self.device.create_texture_from_hal::<Metal>(
                hal,
                &wgpu::TextureDescriptor {
                    label: Some("decoded canvas frame"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                },
            ))
        }
    }
}
