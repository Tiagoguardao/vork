use super::{CommandEncoderStatus, RenderPassColorAttachment, RenderPassDescriptor};
use crate::global::Global;
use alloc::{vec, vec::Vec};
use core::{cell::RefCell, fmt::Debug};

std::thread_local! {
    static BODIES: RefCell<Option<Vec<*const ()>>> = const { RefCell::new(None) };
}

pub(super) fn record_body(buffer: &dyn hal::DynCommandBuffer) {
    BODIES.with_borrow_mut(|bodies| {
        if let Some(bodies) = bodies {
            bodies.push(core::ptr::from_ref(buffer).cast());
        }
    });
}

fn checked<T, E: Debug>((value, error): (T, Option<E>)) -> T {
    assert!(error.is_none(), "{error:?}");
    value
}

#[test]
#[ignore = "requires a Vulkan or Metal adapter"]
fn native_buffers_per_pass() {
    let backend = if cfg!(metal) {
        wgt::Backends::METAL
    } else {
        wgt::Backends::VULKAN
    };
    let global = Global::new(
        "pre-pass test",
        wgt::InstanceDescriptor {
            backends: backend,
            ..wgt::InstanceDescriptor::new_without_display_handle()
        },
        None,
    );
    let adapter = global
        .request_adapter(&Default::default(), backend, None)
        .unwrap();
    let (device, _) = global
        .adapter_request_device(adapter, &Default::default(), None, None)
        .unwrap();
    let attachment = || {
        let texture = checked(global.device_create_texture(
            device,
            &wgt::TextureDescriptor {
                label: None,
                size: wgt::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgt::TextureDimension::D2,
                format: wgt::TextureFormat::Rgba8Unorm,
                usage: wgt::TextureUsages::RENDER_ATTACHMENT | wgt::TextureUsages::COPY_DST,
                view_formats: vec![],
            },
            None,
        ));
        let view = checked(global.texture_create_view(texture, &Default::default(), None));
        (texture, view)
    };
    let (texture, view) = attachment();
    // A fresh attachment makes the query reset the only reason to open this pre-pass.
    let (_, query_view) = attachment();
    let query = checked(global.device_create_query_set(
        device,
        &wgt::QuerySetDescriptor {
            label: None,
            ty: wgt::QueryType::Occlusion,
            count: 1,
        },
        None,
    ));
    let encoder = checked(global.device_create_command_encoder(
        device,
        &wgt::CommandEncoderDescriptor { label: None },
        None,
    ));
    let empty_compute = || {
        let mut pass = checked(global.command_encoder_begin_compute_pass(
            encoder,
            &super::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            },
        ));
        global.compute_pass_end(&mut pass).unwrap();
    };

    let transition = || {
        global
            .command_encoder_transition_resources(
                encoder,
                core::iter::empty(),
                core::iter::once(wgt::TextureTransition {
                    texture,
                    selector: None,
                    state: wgt::TextureUses::COPY_DST,
                }),
            )
            .unwrap();
    };
    let render = |view, reset_query: bool| {
        let mut pass = checked(
            global.command_encoder_begin_render_pass(
                encoder,
                &RenderPassDescriptor {
                    color_attachments: vec![Some(RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        load_op: wgt::LoadOp::Clear(wgt::Color::BLACK),
                        store_op: wgt::StoreOp::Store,
                    })]
                    .into(),
                    occlusion_query_set: reset_query.then_some(query),
                    ..Default::default()
                },
            ),
        );
        if reset_query {
            global
                .render_pass_begin_occlusion_query(&mut pass, 0)
                .unwrap();
            global.render_pass_end_occlusion_query(&mut pass).unwrap();
        }
        global.render_pass_end(&mut pass).unwrap();
    };

    empty_compute();
    render(view, false);
    render(query_view, true); // Query reset only.
    transition();
    render(view, true); // Reset and transition must share one pre-pass.
    transition();
    render(view, false); // Only Vulkan records this transition.
    empty_compute();

    BODIES.set(Some(Vec::new()));
    let buffer = checked(global.command_encoder_finish(
        encoder,
        &wgt::CommandBufferDescriptor { label: None },
        None,
    ));
    let buffer = global.hub.command_buffers.get(buffer);
    let data = buffer.data.lock();
    let CommandEncoderStatus::Finished(data) = &*data else {
        panic!("expected finished encoder")
    };
    // Identify the bodies by their live HAL allocations at close(), before any fixups.
    // A count alone would also pass if the reset landed after its body or before the first pass.
    let encoder = &data.encoder;
    let bodies = BODIES.take().unwrap();
    assert_eq!(bodies.len(), 6);
    let submission_order: Vec<_> = encoder
        .list
        .iter()
        .map(|buffer| {
            let address = core::ptr::from_ref(&**buffer).cast();
            bodies.iter().position(|body| *body == address)
        })
        .collect();
    let mut expected = vec![
        Some(0),
        Some(1), // plain compute and render
        None,
        Some(2), // query reset, render
        None,
        None,
        Some(3), // transition command, shared pre-pass, render
        None,    // transition command
    ];
    if backend == wgt::Backends::VULKAN {
        expected.push(None); // explicit transition back to COLOR_TARGET
    }
    expected.extend([Some(4), Some(5)]);
    assert_eq!(submission_order, expected);
}
