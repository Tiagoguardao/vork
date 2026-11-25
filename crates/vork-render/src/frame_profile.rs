//! Internal, compile-time opt-in stage ledger. Ordinary builds contain no probes.
//! VORK_FRAME_PROFILE names an append-only JSONL file. Intervals are inclusive:
//! e.g. byte_compare is inside prepare_values, itself inside encode_sample.
//! Wait calls count host synchronization boundaries, not PCIe transactions.
use std::{
    cell::RefCell,
    collections::BTreeMap,
    fs::OpenOptions,
    io::Write,
    sync::{Mutex, OnceLock},
    time::Instant,
};
#[derive(Default, serde::Serialize)]
struct Ledger {
    phase: &'static str,
    frame: u32,
    time: f32,
    wall_ms: f64,
    spans_ms: BTreeMap<&'static str, f64>,
    details: BTreeMap<&'static str, serde_json::Value>,
    counts: BTreeMap<&'static str, u64>,
}
thread_local! { static ACTIVE: RefCell<Vec<Ledger>> = const { RefCell::new(Vec::new()) }; }
fn sink() -> Option<&'static Mutex<std::fs::File>> {
    static FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    FILE.get_or_init(|| {
        std::env::var_os("VORK_FRAME_PROFILE").map(|path| {
            Mutex::new(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .expect("open VORK_FRAME_PROFILE"),
            )
        })
    })
    .as_ref()
}
pub(crate) struct Event(Option<Instant>);
pub(crate) fn event(phase: &'static str, frame: u32, time: f32) -> Event {
    Event(sink().map(|_| {
        ACTIVE.with(|a| {
            a.borrow_mut().push(Ledger {
                phase,
                frame,
                time,
                ..Default::default()
            })
        });
        Instant::now()
    }))
}
impl Drop for Event {
    fn drop(&mut self) {
        if let Some(start) = self.0 {
            let mut ledger = ACTIVE.with(|a| a.borrow_mut().pop().unwrap());
            ledger.wall_ms = start.elapsed().as_secs_f64() * 1000.;
            let mut file = sink().unwrap().lock().unwrap();
            serde_json::to_writer(&mut *file, &ledger).expect("write frame profile");
            writeln!(file).expect("write frame profile newline");
        }
    }
}
pub(crate) struct Span(&'static str, Option<Instant>);
pub(crate) fn span(name: &'static str) -> Span {
    Span(
        name,
        ACTIVE.with(|a| (!a.borrow().is_empty()).then(Instant::now)),
    )
}
impl Drop for Span {
    fn drop(&mut self) {
        if let Some(start) = self.1 {
            ACTIVE.with(|a| {
                if let Some(ledger) = a.borrow_mut().last_mut() {
                    *ledger.spans_ms.entry(self.0).or_default() +=
                        start.elapsed().as_secs_f64() * 1000.;
                }
            });
        }
    }
}
pub(crate) fn count(name: &'static str, amount: u64) {
    ACTIVE.with(|a| {
        if let Some(ledger) = a.borrow_mut().last_mut() {
            *ledger.counts.entry(name).or_default() += amount;
        }
    });
}

pub(crate) fn detail(name: &'static str, value: serde_json::Value) {
    ACTIVE.with(|a| {
        if let Some(ledger) = a.borrow_mut().last_mut() {
            ledger.details.insert(name, value);
        }
    });
}
pub(crate) fn report(report: &serde_json::Value) {
    if let Some(sink) = sink() {
        let mut file = sink.lock().unwrap();
        serde_json::to_writer(&mut *file, &serde_json::json!({"render_report": report}))
            .expect("write render profile");
        writeln!(file).expect("write render profile newline");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_ledgers_keep_world_and_canvas_counters_separate() {
        ACTIVE.with(|a| a.borrow_mut().push(Ledger::default()));
        count("queue_submissions", 2);
        ACTIVE.with(|a| a.borrow_mut().push(Ledger::default()));
        {
            profile_span!(_span, "scene_evaluation");
            assert!(ACTIVE.with(|a| a.borrow().last().unwrap().spans_ms.is_empty()));
            count("queue_submissions", 3);
        }
        let child = ACTIVE.with(|a| a.borrow_mut().pop().unwrap());
        assert_eq!(child.counts["queue_submissions"], 3);
        assert!(child.spans_ms["scene_evaluation"].is_finite());
        let parent = ACTIVE.with(|a| a.borrow_mut().pop().unwrap());
        assert_eq!(parent.counts["queue_submissions"], 2);
        assert!(parent.spans_ms.is_empty());
        count("queue_submissions", 99);
        assert!(ACTIVE.with(|a| a.borrow().is_empty()));
    }
}
