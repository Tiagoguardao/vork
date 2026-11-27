//! Background workers from overlapping installs can belong to different compilers.
use std::sync::Mutex;

#[derive(Default)]
struct Workers {
    active: usize,
    compiled: bool,
    on_idle: Option<(fn(), bool)>,
}

impl Workers {
    fn after_reply(&mut self, callback: fn()) -> bool {
        let compiled = std::mem::take(&mut self.compiled);
        if self.active > 0 {
            let (_, recheck) = self.on_idle.get_or_insert((callback, false));
            *recheck |= compiled;
        }
        compiled
    }

    fn retire(&mut self) -> Option<fn()> {
        self.active -= 1;
        if self.active != 0 { return None; }
        let (callback, recheck) = self.on_idle.take()?;
        (std::mem::take(&mut self.compiled) || recheck).then_some(callback)
    }
}

static WORKERS: Mutex<Workers> = Mutex::new(Workers { active: 0, compiled: false, on_idle: None });

pub(crate) fn started() {
    WORKERS.lock().unwrap().active += 1;
}

pub(crate) fn compiled() {
    WORKERS.lock().unwrap().compiled = true;
}

/// Called after flushing a session reply. A cold compile schedules maintenance,
/// including a final pass once overlapping workers have dropped their scratch.
pub fn after_compile_reply(callback: fn()) {
    let compiled = WORKERS.lock().unwrap().after_reply(callback);
    if compiled { callback(); }
}

pub(crate) fn retired() {
    let callback = WORKERS.lock().unwrap().retire();
    if let Some(callback) = callback {
        callback();
    }
}

#[test]
fn overlapping_installs_arm_only_the_last_worker() {
    let mut workers = Workers::default();
    assert!(!workers.after_reply(|| {}));
    workers.active += 1;
    assert!(!workers.after_reply(|| {}));
    assert!(workers.retire().is_none());

    workers.active += 2;
    assert!(!workers.after_reply(|| {}));
    workers.compiled = true;
    assert!(workers.retire().is_none());
    assert!(workers.retire().is_some());
    assert!(!workers.after_reply(|| {}));

    workers.active += 1;
    workers.compiled = true;
    assert!(workers.after_reply(|| {}));
    assert!(!workers.after_reply(|| {}));
    assert!(workers.retire().is_some());
    assert!(!workers.after_reply(|| {}));

    workers.active += 1;
    workers.compiled = true;
    assert!(workers.retire().is_none());
    assert!(workers.after_reply(|| {}));
    assert!(!workers.after_reply(|| {}));
}
