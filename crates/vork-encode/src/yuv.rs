//! The software H.264 input: planar limited-range BT.709, packed on the GPU.
use std::sync::OnceLock;
use wgpu::*;

pub(super) struct Conversion {
    device: Device,
    pipeline: ComputePipeline,
    packed: Buffer,
    // Callers that supply copy-only textures can still use the encoder.
    rgba_copy: OnceLock<Texture>,
    bgra_copy: OnceLock<Texture>,
}

impl Conversion {
    pub(super) fn new(device: &Device, width: u32, height: u32) -> Option<Self> {
        let limits = device.limits();
        let size = Self::size(width, height);
        // swscale uses a shorter vertical kernel for very small images.
        if height < 12
            || size > u64::from(limits.max_storage_buffer_binding_size)
            || limits.max_storage_buffers_per_shader_stage == 0
        {
            return None;
        }
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("BT.709 yuv420p"),
            source: ShaderSource::Wgsl(include_str!("yuv.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("BT.709 yuv420p"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions {
                constants: &[(
                    "x86_rounding",
                    f64::from(u8::from(cfg!(target_arch = "x86_64"))),
                )],
                ..Default::default()
            },
            cache: None,
        });
        let packed = device.create_buffer(&BufferDescriptor {
            label: Some("packed yuv420p"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Some(Self {
            device: device.clone(),
            pipeline,
            packed,
            rgba_copy: OnceLock::new(),
            bgra_copy: OnceLock::new(),
        })
    }

    pub(super) fn size(width: u32, height: u32) -> u64 {
        // Each invocation writes eight luma bytes and four bytes per chroma plane.
        u64::from(width.div_ceil(8)) * 8 * u64::from(height) * 3 / 2
    }

    pub(super) fn record(&self, commands: &mut CommandEncoder, source: &Texture, staging: &Buffer) {
        let sampled = if source.usage().contains(TextureUsages::TEXTURE_BINDING) {
            source
        } else {
            let cached = if source.format() == TextureFormat::Bgra8Unorm {
                &self.bgra_copy
            } else {
                &self.rgba_copy
            };
            let copy = cached.get_or_init(|| {
                self.device.create_texture(&TextureDescriptor {
                    label: Some("encoder sampled RGB"),
                    size: source.size(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: source.format(),
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                    view_formats: &[],
                })
            });
            commands.copy_texture_to_texture(
                source.as_image_copy(),
                copy.as_image_copy(),
                source.size(),
            );
            copy
        };
        let view = sampled.create_view(&TextureViewDescriptor {
            mip_level_count: Some(1),
            ..Default::default()
        });
        let bind = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("encoder yuv420p input and output"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.packed.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = commands.begin_compute_pass(&ComputePassDescriptor {
                label: Some("encoder RGB to yuv420p"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(source.width().div_ceil(64), source.height().div_ceil(16), 1);
        }
        // Commands share one device-local buffer. Each copy precedes the next
        // dispatch on the same queue, and each ring slot has its own staging.
        commands.copy_buffer_to_buffer(&self.packed, 0, staging, 0, self.packed.size());
    }

    pub(super) fn pack(view: &[u8], width: u32, height: u32, pixels: &mut Vec<u8>) {
        pixels.clear();
        if width % 8 == 0 {
            pixels.extend_from_slice(view);
        } else {
            let luma_stride = width.div_ceil(8) as usize * 8;
            let (luma, chroma) = view.split_at(luma_stride * height as usize);
            for row in luma.chunks_exact(luma_stride) {
                pixels.extend_from_slice(&row[..width as usize]);
            }
            for row in chroma.chunks_exact(luma_stride / 2) {
                pixels.extend_from_slice(&row[..width as usize / 2]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::util::{DeviceExt, TextureDataOrder};

    #[test]
    fn packs_bt709_planes_at_unaligned_width() -> anyhow::Result<()> {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default()))?;
        let side = 18;
        let mut bgra = Vec::new();
        for y in 0..side {
            for x in 0..side {
                bgra.extend([
                    (x * 131 + y * 13 + x * y * 7) as u8,
                    (x * 23 + y * 173 + x * y * 3) as u8,
                    (x * 67 + y * 29 + x * y * 11) as u8,
                    (x * 5 + y * 23) as u8,
                ]);
            }
        }
        let source = device.create_texture_with_data(
            &queue,
            &TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: side,
                    height: side,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Bgra8Unorm,
                usage: TextureUsages::COPY_SRC,
                view_formats: &[],
            },
            TextureDataOrder::LayerMajor,
            &bgra,
        );
        let conversion = Conversion::new(&device, side, side).unwrap();
        let staging = device.create_buffer(&BufferDescriptor {
            label: None,
            size: Conversion::size(side, side),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        conversion.record(&mut encoder, &source, &staging);
        let submission = queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        staging.map_async(MapMode::Read, .., move |status| {
            let _ = tx.send(status);
        });
        device.poll(PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })?;
        rx.recv()??;
        let mut packed = Vec::new();
        Conversion::pack(&staging.get_mapped_range(..), side, side, &mut packed);
        staging.unmap();

        let luma = (side * side) as usize;
        assert_eq!(packed.len(), luma * 3 / 2);
        let (y, uv) = packed.split_at(luma);
        let (u, v) = uv.split_at(luma / 4);
        let actual = [y, u, v].map(|plane| {
            plane.iter().fold(0xcbf29ce484222325_u64, |hash, &byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
            })
        });
        // FNV-1a of the three planes from FFmpeg 6.1.1's same-size
        // BT.709 limited conversion, with -cpuflags 0 for scalar rounding.
        let chroma = if cfg!(target_arch = "x86_64") {
            [0x3db14c65ac4fc18b, 0x002f8b72edb3033f]
        } else {
            [0x0c28e97f5209045a, 0x48c98f962136a0ac]
        };
        assert_eq!(actual, [0xc7114c671210e3cf, chroma[0], chroma[1]]);
        Ok(())
    }
}
