//! Diagnostic receipt semantics for every canvas output path.
use moda_engine::prof::HostTimings;
use serde_json::{json, Value};

pub(crate) fn frame(host: &HostTimings, gpu: Option<moda_engine::gpu_profile::GpuTime>) -> Value {
    let get = |name| host.get(name).copied().unwrap_or(0.);
    let mut value = json!({
        "evaluation_ms": get("$evaluation"), "planning_ms": get("$planning"),
        "coverage_planning_ms": get("$coverage_planning"), "host_draw_ms": get("$host_draw"),
        "completion_wait_ms": get("$completion_wait"), "output_conversion_ms": get("$output_conversion"),
        "encoder_handoff_ms": get("$encoder_handoff"), "input_decode_ms": get("$input_decode"),
        "other_host_ms": get("$other_host"), "encoder_start_ms": 0.,
        "nested_world_host_ms": get("$nested_world"),
        "gpu_busy_ms": gpu.filter(|g| g.untimed_passes == 0).map(|g| g.busy_ms + get("canvas.world_gpu")),
        "gpu_span_ms": gpu.map(|g| g.span_ms),
        "gpu_timed_passes": gpu.map(|g| g.timed_passes),
        "gpu_untimed_passes": gpu.map(|g| g.untimed_passes),
    });
    value["gpu_available"] = gpu.is_some().into();
    value
}
pub(crate) fn total(samples: &[Value], setup: &HostTimings, start: f64, finish: f64) -> Value {
    let mut value = json!({});
    if let Some(first) = samples.first() {
        for (key, _) in first["breakdown"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(key, _)| key.ends_with("_ms"))
        {
            value[key] = if samples.iter().any(|s| s["breakdown"][key].is_null()) {
                Value::Null
            } else {
                samples
                    .iter()
                    .map(|s| s["breakdown"][key].as_f64().unwrap_or(0.))
                    .sum::<f64>()
                    .into()
            };
        }
    }
    value["encoder_start_ms"] = start.into();
    value["encoder_finish_ms"] = finish.into();
    value["setup_input_decode_ms"] = setup.get("$input_decode").copied().unwrap_or(0.).into();
    value["setup_host_spans_inclusive_ms"] = serde_json::to_value(setup).unwrap();
    value["semantics"] = json!({
        "host": "exclusive CPU wall spans; planning excludes coverage planning; nested world host excludes its separately reported completion waits",
        "gpu_busy_ms": "sum of pass timestamp intervals plus nested-world pass intervals; overlaps host and completion waits; excludes queue idle gaps",
        "gpu_span_ms": "first to last canvas pass, including queue idle gaps; overlaps GPU busy and host; do not add",
        "completion_wait_ms": "host acceptance/readback and output credit waits; overlaps GPU busy and other frames",
        "encoder_handoff_ms": "host encoder release/push and writer handoff including bounded ring backpressure; completion-driven push precedes acceptance/readback; asynchronous GPU/encode/mux work runs outside this span",
        "encoder_start_ms": "prepare plus open; prepare overlaps document/device setup; render-level only, per-frame zero",
        "frame_wall_ms": "frame-start-to-delivery latency; frames overlap; never sum as render wall time",
        "host_spans_inclusive_ms": "nested detail, not additive; $ entries are exclusive categories",
        "gpu_unavailable": "null rather than zero; gpu_untimed_passes reports invalid timestamps"
    });
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn asynchronous_gpu_queries_grow_and_reuse_without_dropping_passes() -> anyhow::Result<()> {
        let r = crate::Renderer::new()?;
        let _capture = moda_engine::gpu_profile::capture_state();
        let marker = crate::timing::Marker::new(&r);
        let mut reuse = None;
        for count in [1100, 3] {
            assert!(moda_engine::gpu_profile::begin_frame(
                &r.device,
                &r.queue,
                reuse.take()
            ));
            {
                let _world = moda_engine::gpu_profile::pause();
                assert!(!moda_engine::gpu_profile::enabled());
                {
                    let _canvas = moda_engine::gpu_profile::resume();
                    assert!(moda_engine::gpu_profile::enabled());
                }
                assert!(!moda_engine::gpu_profile::enabled());
            }
            assert!(moda_engine::gpu_profile::enabled());
            let mut encoder = r.device.create_command_encoder(&Default::default());
            for _ in 0..count {
                let pair = moda_engine::gpu_profile::pass_timestamps();
                marker.encode_timestamps(
                    &mut encoder,
                    moda_engine::gpu_profile::compute_writes(&pair).unwrap(),
                );
            }
            let pending = moda_engine::gpu_profile::end_frame(&mut encoder).unwrap();
            let submission = r.queue.submit([encoder.finish()]);
            let (send, recv) = std::sync::mpsc::channel();
            let buffers: Vec<_> = pending.readbacks().collect();
            for buffer in &buffers {
                let send = send.clone();
                buffer.map_async(wgpu::MapMode::Read, .., move |result| {
                    send.send(result).unwrap();
                });
            }
            r.wait(Some(submission))?;
            for _ in &buffers {
                recv.recv()??;
            }
            let time = pending.time();
            assert_eq!(time.timed_passes, count);
            assert_eq!(time.untimed_passes, 0);
            assert!(time.busy_ms > 0.);
            assert!(time.span_ms >= time.busy_ms);
            drop(buffers);
            reuse = Some(pending);
        }
        r.errors.check()?;
        Ok(())
    }
    #[test]
    fn totals_preserve_unavailable_gpu_and_separate_startup_from_frames() {
        let samples = vec![
            json!({"breakdown": frame(&HostTimings::from([("$evaluation", 2.)]), None)}),
            json!({"breakdown": frame(&HostTimings::from([("$evaluation", 3.)]), None)}),
        ];
        let total = total(&samples, &HostTimings::new(), 4., 5.);
        assert_eq!(total["evaluation_ms"], 5.);
        assert!(total["gpu_busy_ms"].is_null());
        assert_eq!(total["encoder_start_ms"], 4.);
        assert_eq!(total["encoder_finish_ms"], 5.);
    }
}

pub(crate) fn category(name: &str) -> Option<&'static str> {
    Some(match name {
        "canvas.evaluation" => "$evaluation",
        "canvas.rows" | "canvas.plan" => "$planning",
        "canvas.plan.coverage" => "$coverage_planning",
        "canvas.draw" | "canvas.compositor" => "$host_draw",
        "canvas.output.wait_readback" => "$completion_wait",
        "canvas.output.convert_record" | "canvas.output.encoder_convert" => "$output_conversion",
        "canvas.output.handoff" | "canvas.output.encoder_release" => "$encoder_handoff",
        "canvas.input_decode" => "$input_decode",
        "canvas.worlds" => "$nested_world",
        _ => return None,
    })
}
