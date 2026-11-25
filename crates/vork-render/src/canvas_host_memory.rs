//! Host memory admission for canvas frame workers.

const MIB: u64 = 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Sample {
    pub rss: u64,
    peak: u64,
    pub available: u64,
    pub source: &'static str,
}

impl Sample {
    pub(crate) fn reserve(self) -> u64 {
        // Keep the existing 32 MiB floor for in-flight frame growth in
        // addition to the native allocator reserve and each lane's claim.
        (self.available / 8).max(512 * MIB).saturating_add(32 * MIB)
    }

    pub(crate) fn resident_peak(self) -> u64 {
        self.peak.max(self.rss)
    }
}

/// Capacity promised to constructors which can run concurrently. Make the whole
/// batch's claim before starting any of its constructors; RSS need not have moved.
#[derive(Default)]
pub(crate) struct Reservations {
    held: u64,
}

impl Reservations {
    pub(crate) fn claim(&mut self, available: u64, floor: u64, bytes: u64) -> bool {
        let Some(total) = self.held.checked_add(bytes) else { return false };
        if total > available.saturating_sub(floor) { return false; }
        self.held = total;
        true
    }
}

// Some transitions replace a complete retained history. Its replacement can
// contain a different set of drawing nodes, batches and tessellations. The
// current history's reservation cannot authorize that second working set.
std::thread_local! {
    static WORKERS_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn with_workers<T>(work: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) { WORKERS_ACTIVE.set(self.0); }
    }
    let _restore = Restore(WORKERS_ACTIVE.replace(true));
    work()
}

pub(crate) fn require_serial_growth() -> Result<(), wgpu::optional_allocations::Refused> {
    if WORKERS_ACTIVE.get() { Err(wgpu::optional_allocations::Refused) } else { Ok(()) }
}

use std::sync::Mutex;
use std::time::{Duration, Instant};

pub(crate) struct Guard {
    monitor: Monitor,
    diagnostics: bool,
    pressure: Mutex<Pressure>,
}

#[derive(Default)]
struct Pressure {
    watch: Option<(Sample, u64)>,
    checked: Option<Instant>,
    stopped: bool,
    reservations: Reservations,
}

impl Guard {
    pub(crate) fn new(diagnostics: bool) -> Option<Self> {
        Some(Self {
            monitor: Monitor::new()?,
            diagnostics,
            pressure: Mutex::default(),
        })
    }

    pub(crate) fn sample(&self, baseline: Option<Sample>) -> Option<Sample> {
        let mut sample = self.monitor.sample()?;
        if let Some(baseline) = baseline.filter(|_| cfg!(target_os = "macos")) {
            // XNU can return cached VM statistics. Debit new RSS immediately.
            sample.available = sample.available.min(
                baseline
                    .available
                    .saturating_sub(sample.rss.saturating_sub(baseline.rss)),
            );
        }
        Some(sample)
    }

    pub(crate) fn claim_workers(&self, sample: Sample, floor: u64, worker_bytes: u64, slots: u64) -> bool {
        // Slot resources will take their own byte leases at their actual backend
        // constructors. Require room for both so the selected pool fits its ring.
        self.pressure.lock().unwrap().reservations.claim(sample.available.saturating_sub(slots), floor, worker_bytes)
    }

    pub(crate) fn release_workers(&self, bytes: u64) {
        self.pressure.lock().unwrap().reservations.held -= bytes;
    }

    /// Host credit for the same byte leases that protect GPU buffers, textures
    /// and queue upload staging. Native placement can use host memory even when
    /// a discrete device also reports space on a local heap.
    pub(crate) fn allocation_room(&self) -> Option<u64> {
        let state = self.pressure.lock().unwrap();
        if state.stopped { return None; }
        let sample = self.sample(state.watch.map(|(baseline, _)| baseline))?;
        let floor = state.watch.map_or_else(|| sample.reserve(), |(_, floor)| floor);
        Some(sample.available.saturating_sub(floor).saturating_sub(state.reservations.held))
    }

    pub(crate) fn watch(&self, baseline: Sample, floor: u64) {
        let mut state = self.pressure.lock().unwrap();
        state.watch = Some((baseline, floor));
        state.checked = None;
    }

    pub(crate) fn check(&self, frame: usize) -> Result<(), wgpu::optional_allocations::Refused> {
        let mut state = self.pressure.lock().unwrap();
        if state.stopped {
            return Err(wgpu::optional_allocations::Refused);
        }
        if let Some((baseline, floor)) = state.watch {
            let now = Instant::now();
            if state
                .checked
                .is_none_or(|last| now.duration_since(last) >= Duration::from_millis(50))
            {
                state.checked = Some(now);
                let sample = self.sample(Some(baseline));
                if sample.is_none_or(|sample| sample.available < floor) {
                    state.stopped = true;
                    if self.diagnostics {
                        eprintln!(
                            "canvas concurrency: host pressure at frame {frame}; sample={sample:?}, reserve={floor}"
                        );
                    }
                    return Err(wgpu::optional_allocations::Refused);
                }
            }
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::Sample;
    use std::fs;
    use std::io::ErrorKind;
    use std::path::{Path, PathBuf};

    struct Cgroup {
        path: PathBuf,
        limit: &'static str,
        usage: &'static str,
    }

    pub(crate) struct Monitor(Vec<Cgroup>);

    fn kib(contents: &str, field: &str) -> Option<u64> {
        contents.lines().find_map(|line| {
            let mut fields = line.strip_prefix(field)?.split_whitespace();
            let value = fields.next()?.parse::<u64>().ok()?;
            (fields.next()? == "kB").then(|| value.saturating_mul(1024))
        })
    }

    // mountinfo escapes spaces, tabs, newlines and backslashes with octal.
    fn mount_path(path: &str) -> PathBuf {
        PathBuf::from(
            path.replace("\\040", " ")
                .replace("\\011", "\t")
                .replace("\\012", "\n")
                .replace("\\134", "\\"),
        )
    }

    fn membership_path(member: &Path, root: &Path, mount: &Path) -> PathBuf {
        // A cgroup namespace can report membership relative to the mounted root.
        let relative = member
            .strip_prefix(root)
            .unwrap_or_else(|_| member.strip_prefix("/").unwrap_or(member));
        // A membership above the namespace root can appear as /../... .
        // Never follow that path out of the cgroup mount.
        if relative
            .components()
            .any(|part| part == std::path::Component::ParentDir)
        {
            mount.to_owned()
        } else {
            mount.join(relative)
        }
    }

    impl Monitor {
        pub(crate) fn new() -> Option<Self> {
            let memberships = fs::read_to_string("/proc/self/cgroup").ok()?;
            let mounts = fs::read_to_string("/proc/self/mountinfo").ok()?;
            let mut groups = Vec::new();
            for (controllers, member) in memberships
                .lines()
                .filter_map(|line| line.split_once(':'))
                .filter_map(|(_, rest)| rest.split_once(':'))
            {
                let (kind, limit, usage) = if controllers.is_empty() {
                    ("cgroup2", "memory.max", "memory.current")
                } else if controllers.split(',').any(|name| name == "memory") {
                    ("cgroup", "memory.limit_in_bytes", "memory.usage_in_bytes")
                } else {
                    continue;
                };
                for (mount, fs) in mounts.lines().filter_map(|line| line.split_once(" - ")) {
                    let mut fs = fs.split_whitespace();
                    if fs.next()? != kind
                        || (kind == "cgroup" && !fs.nth(1)?.split(',').any(|name| name == "memory"))
                    {
                        continue;
                    }
                    let mut fields = mount.split_whitespace().skip(3);
                    let root = mount_path(fields.next()?);
                    let mount = mount_path(fields.next()?);
                    let leaf = membership_path(Path::new(member), &root, &mount);
                    for path in leaf.ancestors().take_while(|path| path.starts_with(&mount)) {
                        match fs::metadata(path.join(limit)) {
                            Ok(_) => groups.push(Cgroup {
                                path: path.to_owned(),
                                limit,
                                usage,
                            }),
                            Err(error) if error.kind() == ErrorKind::NotFound => {}
                            Err(_) => return None,
                        }
                    }
                }
            }
            Some(Self(groups))
        }

        pub(crate) fn sample(&self) -> Option<Sample> {
            let mut cgroup = u64::MAX;
            for group in &self.0 {
                let limit = fs::read_to_string(group.path.join(group.limit)).ok()?;
                if limit.trim() == "max" {
                    continue;
                }
                let limit = limit.trim().parse::<u64>().ok()?;
                // v1 uses kernel LONG_MAX rounded down to a page as unlimited.
                if limit >= 1 << 62 {
                    continue;
                }
                let used = fs::read_to_string(group.path.join(group.usage)).ok()?;
                cgroup = cgroup.min(limit.saturating_sub(used.trim().parse::<u64>().ok()?));
            }
            let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
            let physical = kib(&meminfo, "MemAvailable:")?;
            let status = fs::read_to_string("/proc/self/status").ok()?;
            Some(Sample {
                rss: kib(&status, "VmRSS:")?,
                peak: kib(&status, "VmHWM:")?,
                available: physical.min(cgroup),
                source: if cgroup <= physical {
                    "cgroup"
                } else {
                    "MemAvailable"
                },
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn resolves_mount_roots_and_namespace_membership() {
            let mount = Path::new("/sys/fs/cgroup");
            let root = Path::new("/host/container");
            assert_eq!(
                membership_path(Path::new("/host/container/job"), root, mount),
                mount.join("job")
            );
            assert_eq!(
                membership_path(Path::new("/job"), root, mount),
                mount.join("job")
            );
            assert_eq!(membership_path(Path::new("/../host"), root, mount), mount);
            assert_eq!(mount_path(r"/a\040b\134c"), Path::new(r"/a b\c"));
        }
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::Sample;
    use crate::canvas_parallel::unified_free;

    pub(crate) struct Monitor;

    impl Monitor {
        pub(crate) fn new() -> Option<Self> {
            Some(Self)
        }

        pub(crate) fn sample(&self) -> Option<Sample> {
            let (rss, peak) = super::macos_resident_sizes()?;
            Some(Sample {
                rss,
                peak,
                available: unified_free()?,
                source: "Mach free and purgeable",
            })
        }
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn macos_resident_sizes() -> Option<(u64, u64)> {
    // SAFETY: task_info writes into a correctly sized Mach info struct.
    unsafe {
        let mut info: libc::mach_task_basic_info = std::mem::zeroed();
        let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
        if libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as *mut libc::integer_t,
            &mut count,
        ) != libc::KERN_SUCCESS
        {
            return None;
        }
        Some((info.resident_size, info.resident_size_max))
    }
}

#[cfg(target_os = "windows")]
mod platform {
    use super::Sample;
    use std::mem::size_of;
    use windows_sys::Win32::System::{
        ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
        SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX},
        Threading::GetCurrentProcess,
    };

    pub(crate) struct Monitor;

    impl Monitor {
        pub(crate) fn new() -> Option<Self> {
            Some(Self)
        }

        pub(crate) fn sample(&self) -> Option<Sample> {
            let mut rss = PROCESS_MEMORY_COUNTERS {
                cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
                ..Default::default()
            };
            let mut system = MEMORYSTATUSEX {
                dwLength: size_of::<MEMORYSTATUSEX>() as u32,
                ..Default::default()
            };
            // SAFETY: the pseudo-handle and initialized output buffers are valid.
            unsafe {
                if K32GetProcessMemoryInfo(
                    GetCurrentProcess(),
                    &mut rss,
                    size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
                ) == 0
                    || GlobalMemoryStatusEx(&mut system) == 0
                {
                    return None;
                }
            }
            Some(Sample {
                rss: rss.WorkingSetSize as u64,
                peak: rss.PeakWorkingSetSize as u64,
                available: system.ullAvailPhys.min(system.ullAvailPageFile),
                source: "Windows physical and commit availability",
            })
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
mod platform {
    use super::Sample;
    pub(crate) struct Monitor;
    impl Monitor {
        pub(crate) fn new() -> Option<Self> {
            None
        }
        pub(crate) fn sample(&self) -> Option<Sample> {
            None
        }
    }
}

pub(crate) use platform::Monitor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_constructors_cannot_spend_the_same_headroom() {
        let mut reserved = Reservations::default();
        assert!(reserved.claim(1000, 200, 300));
        assert!(!reserved.claim(1000, 200, 600));
        assert!(reserved.claim(1000, 200, 500));
        assert!(!reserved.claim(1000, 200, 1));
        assert!(!reserved.claim(u64::MAX, 0, u64::MAX));
    }

}
