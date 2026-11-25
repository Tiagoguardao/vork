use crate::{util::DeviceExt as _, *};
use core::{future::Future, pin::pin, task};

#[test]
#[ignore = "requires a Vulkan adapter with indirect draw validation"]
fn discard_query_and_indirect_validation_before_draw() {
    // Native adapter and device requests complete on their first poll (as in Device::noop).
    fn ready<T>(request: impl Future<Output = T>) -> T {
        match pin!(request).poll(&mut task::Context::from_waker(task::Waker::noop())) {
            task::Poll::Ready(value) => value,
            task::Poll::Pending => panic!("expected synchronous native device setup"),
        }
    }

    let mut desc = InstanceDescriptor::new_without_display_handle();
    desc.backends = Backends::VULKAN;
    desc.flags.insert(InstanceFlags::VALIDATION_INDIRECT_CALL);
    let instance = Instance::new(desc);
    let adapter = ready(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = ready(adapter.request_device(&Default::default())).unwrap();

    let tex = TextureDescriptor {
        label: None,
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    };
    let source = device.create_texture_with_data(
        &queue,
        &TextureDescriptor {
            usage: tex.usage | TextureUsages::TEXTURE_BINDING,
            ..tex
        },
        util::TextureDataOrder::LayerMajor,
        &[255, 0, 0, 255],
    );
    let source_view = source.create_view(&Default::default());
    let target = device.create_texture(&TextureDescriptor {
        usage: tex.usage | TextureUsages::COPY_SRC,
        ..tex
    });
    let target_view = target.create_view(&Default::default());
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(
            r"
            @group(0) @binding(0) var source: texture_2d<f32>;
            @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
                let p = array(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
                return vec4f(p[i], 0, 1);
            }
            @fragment fn fs() -> @location(0) vec4f {
                return textureLoad(source, vec2i(0), 0) + vec4f(0,1,0,1);
            }
        "
            .into(),
        ),
    });
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(tex.format.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(&source_view),
        }],
    });
    // With INDIRECT_FIRST_INSTANCE disabled, the second draw must become a no-op.
    let indirect = device.create_buffer_init(&util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[3u32, 1, 0, 0, 3, 1, 0, 1]),
        usage: BufferUsages::INDIRECT,
    });
    let queries = device.create_query_set(&QuerySetDescriptor {
        label: None,
        ty: QueryType::Occlusion,
        count: 2,
    });
    let resolve = device.create_buffer(&BufferDescriptor {
        label: None,
        size: 16,
        usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&BufferDescriptor {
        label: None,
        size: 36,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.begin_render_pass(&RenderPassDescriptor {
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &source_view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Load,
                store: StoreOp::Discard,
            },
        })],
        ..Default::default()
    });

    // Reuse the same query slots for the opposite outcomes and retain both resolves.
    for (round, offsets) in [[0, 16], [16, 0]].into_iter().enumerate() {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &target_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLUE),
                    store: StoreOp::Store,
                },
            })],
            occlusion_query_set: Some(&queries),
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        for (query, offset) in offsets.into_iter().enumerate() {
            pass.begin_occlusion_query(query as u32);
            pass.draw_indirect(&indirect, offset);
            pass.end_occlusion_query();
        }
        drop(pass);
        encoder.resolve_query_set(&queries, 0..2, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 16 * round as u64, 16);
        if round == 0 {
            // Capture the first sample of the discarded texture, before another pass can clear it.
            encoder.copy_texture_to_buffer(
                target.as_image_copy(),
                TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: TexelCopyBufferLayout {
                        offset: 32,
                        ..Default::default()
                    },
                },
                tex.size,
            );
        }
    }
    queue.submit([encoder.finish()]);
    // A driver can return plausible counts even when a query was used without a reset.
    let validation = hal::VALIDATION_CANARY.get_and_reset();
    assert!(validation.is_empty(), "Vulkan validation: {validation:#?}");
    readback
        .slice(..)
        .map_async(MapMode::Read, |result| result.unwrap());
    device.poll(PollType::wait_indefinitely()).unwrap();
    let data = readback.slice(..).get_mapped_range();
    let hits: &[u64] = bytemuck::cast_slice(&data[..32]);
    assert_eq!(
        [hits[0] != 0, hits[1] != 0, hits[2] != 0, hits[3] != 0],
        [true, false, false, true],
        "query results across reuse: {hits:?}"
    );
    assert_eq!(
        &data[32..],
        &[0, 255, 0, 255],
        "the draw must sample transparent black after discard"
    );
}
