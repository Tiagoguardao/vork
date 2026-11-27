//! Visible losses are render results, not debug logging. A shot owns its collector;
//! installation workers explicitly inherit it, so concurrent shots stay isolated.
use std::{cell::RefCell, path::{Path, PathBuf}, sync::{Arc, Mutex, Weak, atomic::{AtomicU64, Ordering}}};
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RenderWarning {
    pub code: &'static str,
    /// Authored object, program, binding or value; runtime losses name the output frame and counters.
    pub requested: serde_json::Value,
    /// Omitted/substituted behaviour.
    pub dropped: String,
}
#[derive(Clone, Debug, PartialEq)]
enum Notice {
    Warning(RenderWarning),
    Diagnostic(String),
}

/// Ordered notices from one piece of work, kept with the value it produced.
#[derive(Clone, Default)]
pub(crate) struct Notices(Vec<Notice>);

#[derive(Default)]
struct Recorded {
    notices: Mutex<Vec<Notice>>,
    emissions: AtomicU64,
}

#[derive(Clone, Default)]
pub(crate) struct Collector {
    recorded: Arc<Recorded>,
    quiet: bool,
    parent: Option<Arc<Collector>>,
    sources: Option<crate::asset_revision::SourceFiles>,
}

thread_local! { static CURRENT: RefCell<Option<Collector>> = const { RefCell::new(None) }; }
pub(crate) struct Guard(Option<Collector>);
impl Drop for Guard {
    fn drop(&mut self) { CURRENT.with(|c| *c.borrow_mut() = self.0.take()); }
}
impl Collector {
    /// CPU read-back and candidate checks collect notices without printing them.
    pub(crate) fn quiet() -> Self { Self { quiet: true, ..Default::default() } }
    pub(crate) fn with_sources(&self, sources: &crate::asset_revision::SourceFiles) -> Self {
        Self { sources: Some(sources.clone()), ..self.clone() }
    }
    pub(crate) fn enter(&self) -> Guard {
        Guard(CURRENT.with(|c| c.replace(Some(self.clone()))))
    }
    pub(crate) fn current() -> Option<Self> { CURRENT.with(|c| c.borrow().clone()) }
    pub(crate) fn snapshot(&self) -> Vec<RenderWarning> {
        self.recorded.notices.lock().unwrap().iter().filter_map(|notice| match notice {
            Notice::Warning(warning) => Some(warning.clone()),
            Notice::Diagnostic(_) => None,
        }).collect()
    }

    fn record(&self, notice: Notice) {
        self.recorded.emissions.fetch_add(1, Ordering::Relaxed);
        // Keep duplicate emissions flowing to the request for warning tracking.
        // Structured warnings are reported once, at their first occurrence.
        let mut notices = self.recorded.notices.lock().unwrap();
        let fresh = !matches!(notice, Notice::Warning(_)) || !notices.contains(&notice);
        if fresh {
            notices.push(notice.clone());
        }
        if let Some(parent) = &self.parent {
            parent.record(notice);
        } else if fresh && !self.quiet {
            print(&notice);
        }
    }
}

impl Notices {
    /// Observe exactly the producer being memoized. Also deliver its notices to
    /// the enclosing request, including those emitted before a refusal.
    pub(crate) fn capture<T>(work: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<(T, Self)> {
        let parent = Collector::current();
        let collector = Collector {
            sources: parent.as_ref().and_then(|collector| collector.sources.clone()),
            parent: parent.map(Arc::new),
            ..Default::default()
        };
        let (result, notices) = Self::record_with(collector, work);
        result.map(|value| (value, notices))
    }

    /// A parallel producer hands notices back with its result. The caller
    /// replays them in input order when it joins the work.
    pub(crate) fn defer<T>(work: impl FnOnce() -> T) -> (T, Self) {
        let collector = Collector {
            quiet: true,
            sources: Collector::current().and_then(|collector| collector.sources),
            ..Default::default()
        };
        Self::record_with(collector, work)
    }

    fn record_with<T>(collector: Collector, work: impl FnOnce() -> T) -> (T, Self) {
        let _scope = collector.enter();
        let value = work();
        let notices = std::mem::take(&mut *collector.recorded.notices.lock().unwrap());
        (value, Self(notices))
    }

    pub(crate) fn replay(&self) {
        for notice in &self.0 { emit(notice.clone()); }
    }

    pub(crate) fn is_empty(&self) -> bool { self.0.is_empty() }
}

/// Detect a notice from a cached frame evaluation, even if the request has
/// already reported the same warning. The token does not retain the request.
pub(crate) struct Checkpoint(Weak<Recorded>, u64);

impl Checkpoint {
    pub(crate) fn current() -> Option<Self> {
        let collector = Collector::current()?;
        Some(Self(Arc::downgrade(&collector.recorded), collector.recorded.emissions.load(Ordering::Relaxed)))
    }

    pub(crate) fn emitted(&self) -> bool {
        self.0.upgrade().is_some_and(|recorded| recorded.emissions.load(Ordering::Relaxed) != self.1)
    }

    pub(crate) fn same_request(&self, other: &Self) -> bool { self.0.ptr_eq(&other.0) }
}

/// Keep a private asset snapshot's filename out of authored diagnostics.
pub(crate) fn source_name(path: &Path) -> PathBuf {
    CURRENT.with(|current| {
        current.borrow().as_ref().and_then(|collector| collector.sources.as_ref())
            .map_or_else(|| path.to_owned(), |sources| sources.authored_name(path))
    })
}

/// Plain diagnostics go to stderr verbatim and retain their position among
/// other notices. They are not structured render warnings.
pub(crate) fn diagnostic(message: &str) {
    emit(Notice::Diagnostic(message.to_owned()));
}

pub(crate) fn dropped(code: &'static str, requested: serde_json::Value, dropped: String) {
    emit(Notice::Warning(RenderWarning { code, requested, dropped }));
}
pub(crate) fn approximated(code: &'static str, requested: serde_json::Value, approximation: String) {
    dropped(code, requested, approximation);
}

fn emit(notice: Notice) {
    if let Some(collector) = Collector::current() { collector.record(notice); }
    else { print(&notice); }
}

fn print(notice: &Notice) {
    match notice {
        Notice::Diagnostic(message) => eprintln!("{message}"),
        Notice::Warning(warning) => {
            if matches!(warning.code, "nonfinite_samples" | "previs_material_emission" | "openpbr_approximation") {
                eprintln!("WARNING: {}", warning.dropped);
            } else {
                // No ANSI or TTY detection: the same conspicuous record survives logs.
                eprintln!("\nWARNING: REQUESTED EFFECT DROPPED [{}]\n  requested: {}\n  dropped: {}\n", warning.code, warning.requested, warning.dropped);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn provisional_environment_selection_does_not_decide_admission() {
        let scene = vork_scene::v2::Scene::from_json(br#"{"version":2,"nodes":[{"type":"camera"},{"type":"light","shape":"environment"},{"type":"light","shape":"environment"},{"type":"light","shape":"environment"}]}"#).unwrap();
        let collector = Collector::default();
        let _scope = collector.enter();
        let settings = crate::environment_node::Settings::from_scene(&scene).unwrap();
        assert_eq!(settings.source, Some(1));
        let warnings = collector.snapshot();
        assert!(warnings.is_empty());
    }

    #[test]
    fn capture_replays_order_and_detects_duplicate_frame_emissions() -> anyhow::Result<()> {
        let warn = |code| dropped(code, serde_json::Value::Null, code.into());
        let first = Collector::quiet();
        let _request = first.enter();
        warn("admission");
        let ((), installed) = Notices::capture(|| {
            warn("admission");
            let worker = Collector::current().unwrap();
            std::thread::spawn(move || {
                let _scope = worker.enter();
                dropped("install", serde_json::Value::Null, "install".into());
            }).join().unwrap();
            diagnostic("compiler");
            Ok(())
        })?;
        let frame = Checkpoint::current().unwrap();
        warn("install");
        assert!(frame.emitted());
        warn("frame");

        let second = Collector::quiet();
        let _request = second.enter();
        assert!(!frame.same_request(&Checkpoint::current().unwrap()));
        warn("admission");
        installed.replay();
        assert_eq!(*second.recorded.notices.lock().unwrap(), &first.recorded.notices.lock().unwrap()[..3]);
        warn("frame");
        assert_eq!(first.snapshot(), second.snapshot());
        Ok(())
    }

    #[test]
    fn collectors_isolate_shots_and_inherit_workers_without_duplicate_records() {
        let first = Collector::default();
        let _scope = first.enter();
        dropped("test", serde_json::json!({"node":"a","stage":"lit"}), "lit stage; retaining surface".into());
        let worker = Collector::current().unwrap();
        std::thread::spawn(move || {
            let _scope = worker.enter();
            dropped("test", serde_json::json!({"node":"a","stage":"lit"}), "lit stage; retaining surface".into());
        }).join().unwrap();
        let second = Collector::default();
        { let _scope = second.enter(); dropped("other", serde_json::json!({"node":"b"}), "displacement".into()); }
        assert_eq!(first.snapshot().len(), 1);
        assert_eq!(second.snapshot().len(), 1);
        assert_eq!(first.snapshot()[0].requested["stage"], "lit");
        assert_eq!(serde_json::to_value(first.snapshot()).unwrap()[0]["code"], "test");
    }
}
