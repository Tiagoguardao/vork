//! CUDA writes one device-local transfer buffer. A queued Vulkan copy gives
//! each retained canvas frame independent storage. Wait only for the previous
//! copy before returning ownership to CUDA, never for a retained canvas draw.
use super::footage::Video;
use anyhow::{Context, Result, ensure};
use ash::{khr::external_memory_fd, vk};
use std::{
    ffi::{CString, c_char, c_void},
    os::fd::{FromRawFd, IntoRawFd, OwnedFd},
    ptr::NonNull,
};
use wgpu::hal::{api::Vulkan, vulkan};
unsafe extern "C" {
    fn vork_decode_open(
        input_fd: i32,
        fd: i32,
        bytes: u64,
        uuid: *const u8,
        width: i32,
        height: i32,
        pitch: i32,
        ptx: *const c_char,
    ) -> *mut c_void;
    fn vork_decode_convert(decoder: *mut c_void, matrix: i32, host: *mut c_void) -> i32;
}
use crate::footage_device::Native;
pub(super) struct Session {
    size: [u32; 2],
    native: Native,
    buffer: wgpu::Buffer,
    raw_buffer: vk::Buffer,
    raw_device: ash::Device,
    family: u32,
    gpu: wgpu::Device,
    queue: wgpu::Queue,
    pending: Option<wgpu::SubmissionIndex>,
    pitch: u32,
}
struct Allocation {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}
impl Drop for Allocation {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

pub(super) fn device_uuid(hal: &vulkan::Device) -> Result<[u8; 16]> {
    let instance = hal.shared_instance().raw_instance();
    let physical = hal.raw_physical_device();
    let properties = unsafe { instance.get_physical_device_properties(physical) };
    ensure!(
        properties.vendor_id == 0x10de,
        "NVDEC requires an NVIDIA Vulkan device"
    );
    let mut ids = vk::PhysicalDeviceIDProperties::default();
    let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut ids);
    unsafe { instance.get_physical_device_properties2(physical, &mut properties2) };

    Ok(ids.device_uuid)
}

fn open_native(
    video: &Video,
    uuid: [u8; 16],
    fd: Option<OwnedFd>,
    bytes: u64,
    pitch: u32,
) -> Result<Native> {
    use std::os::fd::{AsFd, AsRawFd};
    let input_fd = video.revision.as_fd().as_raw_fd();
    let ptx = CString::new(include_str!(concat!(env!("OUT_DIR"), "/nv12.ptx")))?;
    let [width, height] = video.size;
    let ptr = unsafe {
        vork_decode_open(
            input_fd,
            fd.map_or(-1, IntoRawFd::into_raw_fd),
            bytes,
            uuid.as_ptr(),
            width as i32,
            height as i32,
            pitch as i32,
            ptx.as_ptr(),
        )
    };
    Ok(Native(
        NonNull::new(ptr).context("CUDA video decoder unavailable")?,
    ))
}

pub(super) struct DeepReader {
    native: Native,
    next: usize,
    sought: bool,
}

impl DeepReader {
    pub(super) fn open(video: &Video, uuid: [u8; 16]) -> Result<Self> {
        crate::footage_device::admit(video)?;
        let [width, height] = video.size;
        let pitch = width.checked_mul(8).context("RGBA16 row too large")?;
        let native = open_native(
            video,
            uuid,
            None,
            u64::from(pitch) * u64::from(height),
            pitch,
        )?;
        Ok(Self {
            native,
            next: 0,
            sought: false,
        })
    }
    pub(super) fn frame(&mut self, video: &Video, index: usize) -> Result<image::DynamicImage> {
        let matrix = video
            .index
            .native_matrix_at(index)?
            .context("unsupported deep frame format or colour matrix")?;
        let decoder = self.native.0.as_ptr();
        crate::footage_device::select_frame(
            decoder,
            &video.index,
            &mut self.next,
            &mut self.sought,
            index,
        )?;
        let [w, h] = video.size;
        let mut image = image::ImageBuffer::<image::Rgba<u16>, Vec<u16>>::new(w, h);
        ensure!(
            unsafe {
                vork_decode_convert(decoder, matrix.av_colorspace(), image.as_mut_ptr().cast())
            } == 0,
            "CUDA RGBA16 conversion failed"
        );
        Ok(image::DynamicImage::ImageRgba16(image))
    }
}

impl Session {
    pub(super) fn open(gpu: &wgpu::Device, queue: &wgpu::Queue, video: &Video) -> Result<Self> {
        let [width, height] = video.size;
        let hal = unsafe { gpu.as_hal::<Vulkan>() }.context("NVDEC requires Vulkan")?;
        ensure!(
            hal.enabled_device_extensions()
                .contains(&external_memory_fd::NAME),
            "VK_KHR_external_memory_fd unavailable"
        );
        let device = hal.raw_device();
        let instance = hal.shared_instance().raw_instance();
        let physical = hal.raw_physical_device();
        let pitch = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes = u64::from(pitch) * u64::from(height);
        ensure!(
            bytes <= gpu.limits().max_buffer_size,
            "footage exceeds transfer buffer limit"
        );
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let info = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .push_next(&mut external);
        let raw_buffer = unsafe { device.create_buffer(&info, None) }?;
        let mut allocation = Allocation {
            device: device.clone(),
            buffer: raw_buffer,
            memory: vk::DeviceMemory::null(),
        };
        let requirements = unsafe { device.get_buffer_memory_requirements(raw_buffer) };
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let memory_type = memory_properties
            .memory_types_as_slice()
            .iter()
            .enumerate()
            .find(|(i, m)| {
                requirements.memory_type_bits & (1 << i) != 0
                    && m.property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .context("no device-local external memory type")?
            .0 as u32;
        let mut export = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(raw_buffer);
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut export)
            .push_next(&mut dedicated);
        allocation.memory = unsafe { device.allocate_memory(&allocate, None) }?;
        unsafe { device.bind_buffer_memory(raw_buffer, allocation.memory, 0) }?;
        let export_api = external_memory_fd::Device::new(instance, device);
        let fd = unsafe {
            export_api.get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(allocation.memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD),
            )
        }?;
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };

        let native = open_native(
            video,
            device_uuid(&hal)?,
            Some(fd),
            requirements.size,
            pitch,
        )?;
        let buffer = unsafe {
            gpu.create_buffer_from_hal::<Vulkan>(
                vulkan::Buffer::from_raw_managed(
                    raw_buffer,
                    allocation.memory,
                    0,
                    requirements.size,
                ),
                &wgpu::BufferDescriptor {
                    label: Some("NVDEC converted pixels"),
                    size: bytes,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )
        };
        // wgpu now owns both handles.
        allocation.buffer = vk::Buffer::null();
        allocation.memory = vk::DeviceMemory::null();

        let mut session = Self {
            size: video.size,
            native,
            buffer,
            raw_buffer,
            raw_device: device.clone(),
            family: hal.queue_family_index(),
            gpu: gpu.clone(),
            queue: queue.clone(),
            pitch,
            pending: None,
        };
        let mut init = gpu.create_command_encoder(&Default::default());
        init.clear_buffer(&session.buffer, 0, None);
        session.pending = Some(queue.submit([init.finish(), session.ownership(true)]));
        Ok(session)
    }
    fn ownership(&self, release: bool) -> wgpu::CommandBuffer {
        let mut encoder = self.gpu.create_command_encoder(&Default::default());
        let (src, dst, src_access, dst_access, src_stage, dst_stage) = if release {
            (
                self.family,
                vk::QUEUE_FAMILY_EXTERNAL,
                vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            )
        } else {
            (
                vk::QUEUE_FAMILY_EXTERNAL,
                self.family,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            )
        };
        let barrier = vk::BufferMemoryBarrier::default()
            .buffer(self.raw_buffer)
            .offset(0)
            .size(self.buffer.size())
            .src_queue_family_index(src)
            .dst_queue_family_index(dst)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access);
        // SAFETY: wgpu serializes native recording. The buffer remains alive; this
        // only changes queue ownership, while wgpu tracks its COPY_DST use.
        unsafe {
            encoder.as_hal_mut::<Vulkan, _, _>(|encoder| {
                self.raw_device.cmd_pipeline_barrier(
                    encoder.unwrap().raw_handle(),
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[barrier],
                    &[],
                );
            })
        }
        encoder.finish()
    }

    pub(super) fn decoder(&self) -> *mut c_void {
        self.native.0.as_ptr()
    }
    pub(super) fn convert(&mut self, matrix: i32) -> Result<wgpu::Texture> {
        let decoder = self.decoder();
        if let Some(submission_index) = self.pending.take() {
            self.gpu.poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            })?;
        }
        ensure!(
            unsafe { vork_decode_convert(decoder, matrix, std::ptr::null_mut()) } == 0,
            "unsupported decoded frame or CUDA conversion failure"
        );
        let [width, height] = self.size;
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = self.gpu.create_texture(&wgpu::TextureDescriptor {
            label: Some("decoded canvas frame"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut copy = self.gpu.create_command_encoder(&Default::default());
        copy.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.pitch),
                    rows_per_image: None,
                },
            },
            texture.as_image_copy(),
            extent,
        );
        self.pending =
            Some(
                self.queue
                    .submit([self.ownership(false), copy.finish(), self.ownership(true)]),
            );
        Ok(texture)
    }
}
