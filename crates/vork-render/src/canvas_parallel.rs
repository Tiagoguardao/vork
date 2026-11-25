//! Memory admission for optional canvas replicas.
use std::sync::Arc;

/// A qualification-only total-memory ceiling. Native budgets remain authoritative;
/// this can only make admission more conservative, including on a larger test GPU.
fn ceiling() -> Option<u64> {
    std::env::var("VORK_CANVAS_MEMORY_BUDGET").ok()?.parse().ok()
}
pub(crate) fn headroom(device: &wgpu::Device) -> Option<(u64, u64)> {
    let (used, mut free) = crate::residency::camera_usage_headroom(device)?;
    if let Some(cap) = ceiling() { free = free.min(cap.saturating_sub(used)); }
    #[cfg(target_os = "macos")]
    { free = free.min(unified_free()?); }
    Some((used, free))
}
#[cfg(target_os = "macos")]
pub(crate) fn unified_free() -> Option<u64> {
    // XNU's free_count already includes speculative pages (osfmk/kern/host.c).
    // Purgeable pages can be discarded without application paging. Inactive
    // application pages and swap are not capacity for replication.
    unsafe {
        // One process-lifetime send right; repeated probes must not leak ports.
        static HOST: std::sync::OnceLock<libc::mach_port_t> = std::sync::OnceLock::new();
        let host = *HOST.get_or_init(|| libc::mach_host_self());
        let mut stats: libc::vm_statistics64 = std::mem::zeroed();
        let mut count = libc::HOST_VM_INFO64_COUNT;
        let status = libc::host_statistics64(host, libc::HOST_VM_INFO64,
            &mut stats as *mut _ as *mut libc::integer_t, &mut count);
        let page_size = libc::sysconf(libc::_SC_PAGESIZE);
        if status != libc::KERN_SUCCESS || page_size <= 0 { return None; }
        (u64::from(stats.free_count) + u64::from(stats.purgeable_count)).checked_mul(page_size as u64)
    }
}
#[cfg(target_os = "macos")]
fn host_working_set() -> Option<u64> {
    crate::canvas_host_memory::macos_resident_sizes().map(|(rss, _)| rss)
}
/// Leave native command buffers, staging and other applications room to grow.
/// The mandatory serial target/ring are never charged to optional capacity.
pub(crate) fn budget(device: &wgpu::Device, discrete: bool, host: Arc<crate::canvas_host_memory::Guard>) -> Option<Arc<wgpu::optional_allocations::Budget>> {
    let (initial_used, free) = headroom(device)?;
    #[cfg(target_os = "macos")]
    let initial_rss = host_working_set()?;
    // HAL Performance allocation sizes permit a 256 MiB device block and a
    // 128 MiB host block (vendor/wgpu-hal/src/lib.rs::AllocationSizes). Retain
    // room for both plus command/descriptor allocation, even at tight headroom.
    let reserve = (free / 8).max(512 * 1024 * 1024);
    let limit = free.saturating_sub(reserve).min(host.allocation_room()?);
    let device = device.clone();
    let available = move || {
        let (used, current_free) = headroom(&device)?;
        let fixed = free.saturating_sub(used.saturating_sub(initial_used));
        // host_statistics is rate-limited by XNU. Do not regain credit from a
        // cached free-page report while our own CPU/GPU working set grows.
        #[cfg(target_os = "macos")]
        let fixed = fixed.saturating_sub(host_working_set()?.saturating_sub(initial_rss));
        Some(current_free.min(fixed).saturating_sub(reserve).min(host.allocation_room()?))
    };
    Some(if discrete {
        wgpu::optional_allocations::Budget::new_batched(limit, available)
    } else {
        // Unified memory can change during CPU evaluation/decode within a frame.
        wgpu::optional_allocations::Budget::new(limit, available)
    })
}
/// Conservatively reserve the output ring before native encoder open. The
/// max-buffer limit remains a separate single-allocation check, never capacity.
pub(crate) fn ring_limit(free: Option<u64>, bytes_per_slot: u64, serial: usize, wanted: usize) -> usize {
    let Some(free) = free else { return serial; };
    let optional = free.saturating_sub(free / 8) / bytes_per_slot.max(1);
    wanted.min(usize::try_from(optional).unwrap_or(usize::MAX)).max(serial)
}

/// A short export can have fewer slots than twice its worker count even on
/// main. Only reduce that count when memory actually shortened the ring.
pub(crate) fn replicas_for_ring(wanted: usize, admitted: usize, full: usize) -> usize {
    if admitted == full { wanted } else { wanted.min(admitted / 2).max(1) }
}

/// Grow a drained pixel ring for the workers actually admitted. Its serial entries
/// are also valid worker entries, and the encoder owns its native surfaces separately.
pub(crate) fn extra_slots(serial: usize, limit: usize, admitted: usize) -> usize {
    limit.min(admitted.saturating_mul(2)).saturating_sub(serial)
}

/// Service times for one accepted frame. The ordered output handoff runs on
/// the producer thread. Waiting for a worker reply belongs only to wall time.
#[derive(Clone, Copy, Default)]
pub(crate) struct FrameWork {
    pub plan_ms: f64,
    pub draw_ms: f64,
    pub output_ms: f64,
}

impl FrameWork {
    fn period(self, lanes: usize) -> f64 {
        (self.plan_ms + self.output_ms)
            .max(self.draw_ms / lanes as f64)
            .max(self.output_ms * crate::constants::CANVAS_PARALLEL_DELIVERY_MARGIN)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Resize { Keep, Grow, Retire }

#[derive(Default)]
pub(crate) struct Schedule {
    work: FrameWork,
    samples: usize,
    first: Option<std::time::Instant>,
    last: Option<std::time::Instant>,
    warmup: usize,
    cold_ms: f64,
    growing: usize,
    redundant: usize,
    // Compare the trial to the configuration immediately before publication
    // and resize. Remember a failed prediction until the workload changes.
    trial: Option<(f64, f64)>,
    rejected: Option<(usize, f64)>,
}

impl Schedule {
    pub(crate) fn observe(&mut self, frame: usize, work: FrameWork, now: std::time::Instant) {
        if frame == 0 { self.cold_ms = work.plan_ms + work.draw_ms; }
        self.last = Some(now);
        if self.first.is_none() || self.warmup > 0 {
            self.warmup = self.warmup.saturating_sub(1);
            self.first = Some(now);
            return;
        }
        self.samples += 1;
        self.work.plan_ms += work.plan_ms;
        self.work.draw_ms += work.draw_ms;
        self.work.output_ms += work.output_ms;
    }

    pub(crate) fn resized(&mut self, slots: usize) {
        self.samples = 0;
        self.work = FrameWork::default();
        self.first = None;
        self.growing = 0;
        self.redundant = 0;
        // Omit the first trip through each slot, including lazy lane setup.
        self.warmup = slots;
    }

    pub(crate) fn decide(&mut self, lanes: usize, limit: usize, remaining: usize, install_ms: f64, diagnostics: bool) -> Resize {
        use crate::constants::{CANVAS_PARALLEL_MIN_FRAMES, CANVAS_PARALLEL_PROBE_FRAMES, CANVAS_PARALLEL_PROBE_WINDOWS, CANVAS_PARALLEL_SETUP_MARGIN};
        let windows = if self.trial.is_some() { CANVAS_PARALLEL_PROBE_WINDOWS } else { 1 };
        if self.samples < CANVAS_PARALLEL_PROBE_FRAMES * windows { return Resize::Keep; }
        let n = self.samples as f64;
        let work = FrameWork { plan_ms: self.work.plan_ms / n, draw_ms: self.work.draw_ms / n, output_ms: self.work.output_ms / n };
        let wall = (self.last.unwrap() - self.first.unwrap()).as_secs_f64() * 1000. / n;
        self.samples = 0;
        self.first = self.last;
        self.work = FrameWork::default();
        if let Some((baseline, predicted_gain)) = self.trial.take() {
            // A marginal change within five percent of the prior accepted-frame
            // period does not justify retaining another renderer and its memory.
            if diagnostics { eprintln!("canvas concurrency: {lanes}-lane trial {wall:.3} ms/frame; previous {baseline:.3}"); }
            if wall >= baseline * 0.95 {
                self.rejected = Some((lanes, predicted_gain));
                return Resize::Retire;
            }
        }
        if lanes > 1 && work.period(lanes - 1) <= work.period(lanes) {
            self.redundant += 1;
            if self.redundant >= CANVAS_PARALLEL_PROBE_WINDOWS { return Resize::Retire; }
        } else { self.redundant = 0; }

        let gain = work.period(lanes) - work.period(lanes + 1);
        let startup = (install_ms + self.cold_ms) * CANVAS_PARALLEL_SETUP_MARGIN;
        if lanes < limit && remaining >= CANVAS_PARALLEL_MIN_FRAMES && gain * remaining as f64 > startup
            && !self.rejected.is_some_and(|(count, prior)| count == lanes + 1 && gain <= prior * 2.) {
            self.growing += 1;
            if self.growing >= CANVAS_PARALLEL_PROBE_WINDOWS {
                if diagnostics { eprintln!("canvas concurrency: trial from {lanes} lanes; plan={:.3} draw={:.3} output={:.3} ms/frame, predicted gain={gain:.3}", work.plan_ms, work.draw_ms, work.output_ms); }
                self.trial = Some((wall, gain));
                return Resize::Grow;
            }
        } else { self.growing = 0; }
        Resize::Keep
    }

    pub(crate) fn cancel_trial(&mut self, requested: usize) {
        if let Some((_, gain)) = self.trial.take() { self.rejected = Some((requested, gain)); }
        self.growing = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn producer_bound_frames_do_not_start_a_trial() {
        let work = FrameWork { plan_ms: 3.8, draw_ms: 0.95, output_ms: 0.2 };
        assert_eq!(work.period(1), work.period(4));
    }

    #[test]
    fn an_unprofitable_third_lane_is_retired_once() {
        use std::time::{Duration, Instant};
        let work = FrameWork { plan_ms: 1., draw_ms: 12., output_ms: 0.1 };
        let mut schedule = Schedule::default();
        let mut now = Instant::now();
        let mut frame = 0;
        let mut advance = |schedule: &mut Schedule, lanes, period, count| {
            for _ in 0..count {
                now += Duration::from_millis(period);
                schedule.observe(frame, work, now);
                frame += 1;
                let action = schedule.decide(lanes, 4, 1500, 1., false);
                if action != Resize::Keep { return action; }
            }
            Resize::Keep
        };
        assert_eq!(advance(&mut schedule, 1, 15, 32), Resize::Grow);
        schedule.resized(4);
        assert_eq!(advance(&mut schedule, 2, 8, 48), Resize::Grow);
        schedule.resized(6);
        assert_eq!(advance(&mut schedule, 3, 9, 48), Resize::Retire);
        schedule.resized(4);
        assert_eq!(advance(&mut schedule, 2, 8, 96), Resize::Keep);
    }

    #[test]
    fn reduced_admission_reuses_the_serial_slots() {
        assert_eq!(extra_slots(3, 8, 2), 1);
        assert_eq!(extra_slots(3, 8, 4), 5);
        assert_eq!(extra_slots(3, 3, 3), 0);
    }
    #[test]
    fn memory_does_not_change_the_count_for_short_exports() {
        assert_eq!(replicas_for_ring(8, 9, 9), 8);
        assert_eq!(replicas_for_ring(3, 3, 3), 3);
        assert_eq!(replicas_for_ring(8, 8, 16), 4);
        assert_eq!(replicas_for_ring(8, 3, 16), 1);
    }
    #[test]
    fn unknown_and_tight_budgets_keep_the_serial_ring() {
        assert_eq!(ring_limit(None, 1024, 3, 16), 3);
        assert_eq!(ring_limit(Some(1024), 1024, 3, 16), 3);
        assert_eq!(ring_limit(Some(8192), 1024, 3, 16), 7);
        assert_eq!(ring_limit(Some(u64::MAX), u64::MAX, 3, 16), 3);
    }
}

// Compile the excluded dependency's allocation contracts in the renderer tests.
#[cfg(test)]
#[path = "../../../vendor/wgpu/src/optional_allocations.rs"]
mod allocation_contract;
