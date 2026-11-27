//! `vork session`: a kept-alive worker driven by JSON lines on stdin and stdout.
//!
//! One request per line, one reply per line, in order; renderer progress and
//! warnings keep going to stderr. The device, pipeline caches and the installed
//! scene live for the whole session; editing is `open` again with the whole
//! document. Same-build private: the SDK `Session` objects speak it; the framing
//! is not a public contract.
use anyhow::{Context, Result};
use serde::Deserialize;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use vork_render::shot::{Output, Shot};
use vork_render::shot::worker::{REUSE_KEY, Source, Worker};

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    /// A scene file, or the whole document as JSON with the directory its relative files resolve against.
    Open {
        #[serde(default)]
        scene: Option<PathBuf>,
        #[serde(default)]
        document: Option<serde_json::Value>,
        #[serde(default)]
        base: Option<PathBuf>,
    },
    Render(RenderRequest),
    Ask { questions: serde_json::Value },
    Close,
}
/// Today's `render` options, as JSON.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RenderRequest {
    #[serde(deserialize_with = "one_or_many")]
    out: Vec<serde_json::Value>,
    #[serde(default)]
    at: Option<vork_time::Clock>,
    #[serde(default)]
    frame_start: Option<usize>,
    #[serde(default)]
    frame_end: Option<usize>,
    #[serde(default)]
    profile: Option<String>,
    #[serde(default)]
    set: serde_json::Map<String, serde_json::Value>,
    /// Default colour; ["all"] selects controls and every declared AOV.
    #[serde(default)]
    channels: Vec<String>,
    #[serde(default)]
    exr_codec: Option<String>,
    #[serde(default)]
    png_depth: Option<String>,
    #[serde(default)]
    no_write: bool,
    #[serde(default)]
    gpu_only: bool,
    #[serde(default)]
    diagnostics: bool,
    /// The first picture available instead of the final one: one sample, no
    /// denoise, labelled `provisional: true`; render again for the final.
    #[serde(default)]
    provisional: bool,
}
fn one_or_many<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Vec<serde_json::Value>, D::Error> {
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(match value { serde_json::Value::Array(values) => values, value => vec![value] })
}
fn usage(error: anyhow::Error) -> anyhow::Error { error.context(vork_render::failure::FailureKind::Usage) }
impl RenderRequest {
    fn shot(self) -> Result<Shot> {
        // Exactly what `--set KEY=VALUE` accepts: a string value is read as the
        // CLI reads it (resolution also as WxH or W,H), so the messages hold.
        let mut settings = serde_json::Map::new();
        let provisional = self.provisional;
        for (key, value) in self.set {
            anyhow::ensure!(vork_scene::CallOverrides::KEYS.contains(&key.as_str()),
                "{}", usage(anyhow::anyhow!("unknown render-call field {key}; fields: {}", vork_scene::CallOverrides::KEYS.join(", "))));
            let value = match value {
                serde_json::Value::String(text) => vork_scene::CallOverrides::parse(&format!("{key}={text}")).map_err(usage)?.1,
                value => value,
            };
            settings.insert(key, value);
        }
        if provisional {
            settings.insert("samples".into(), 1.into());
            settings.insert("denoise".into(), false.into());
        }
        let outputs = if self.channels.is_empty() { vec![Output::Color] }
            else if self.channels == ["all"] { Vec::new() }
            else { self.channels.iter().map(|s| s.parse()).collect::<Result<Vec<Output>, _>>().map_err(|e| usage(anyhow::anyhow!(e)))? };
        Ok(Shot {
            outputs,
            out: self.out.into_iter().map(|v| match v.as_str() { Some(s) => s.parse().map_err(anyhow::Error::msg), None => serde_json::from_value(v).map_err(|e| anyhow::anyhow!("output descriptor: {e}")) }).collect::<Result<_>>().map_err(usage)?,
            profile: Some(vork_scene::Profile::load(self.profile.as_deref().unwrap_or(vork_scene::DEFAULT_PROFILE)).map_err(usage)?),
            settings,
            at: self.at,
            frame_start: self.frame_start,
            frame_end: self.frame_end,
            exr_codec: match self.exr_codec { Some(codec) => codec.parse().map_err(|e: String| usage(anyhow::anyhow!(e)))?, None => Default::default() },
            png_depth: match self.png_depth { Some(depth) => depth.parse().map_err(|e: String| usage(anyhow::anyhow!(e)))?, None => Default::default() },
            no_write: self.no_write,
            gpu_only: self.gpu_only,
            diagnostics: self.diagnostics,
            provisional,
            ..Default::default()
        })
    }
}

fn failure(id: &serde_json::Value, error: &anyhow::Error) -> serde_json::Value {
    let kind = match vork_render::failure::exit_code(error) { 2 => "usage", 3 => "scene", 5 => "io", _ => "device" };
    serde_json::json!({"id": id, "ok": false, "kind": kind, "error": format!("{error:#}")})
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod scratch {
    use std::os::fd::RawFd;
    use std::sync::{Mutex, MutexGuard, OnceLock, mpsc::{sync_channel, SyncSender}};
    use std::time::{Duration, Instant};

    const IDLE_GRACE: Duration = Duration::from_millis(25);

    struct Activity {
        last_reply: Option<Instant>,
        buffered_input: bool,
    }

    static SESSION: Mutex<Activity> = Mutex::new(Activity { last_reply: None, buffered_input: false });

    pub(super) struct RequestGuard(MutexGuard<'static, Activity>);

    impl Drop for RequestGuard {
        fn drop(&mut self) {
            self.0.last_reply = Some(Instant::now());
        }
    }

    impl RequestGuard {
        pub(super) fn buffered_input(&mut self, buffered: bool) {
            self.0.buffered_input = buffered;
        }
    }

    pub(super) fn request() -> RequestGuard {
        RequestGuard(SESSION.lock().unwrap())
    }

    pub(super) fn can_wait_for_input(fd: RawFd) -> bool {
        // SAFETY: F_GETFL takes no argument beyond the descriptor.
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if flags < 0 || flags & libc::O_ACCMODE == libc::O_WRONLY || flags & libc::O_NONBLOCK != 0 {
            return false;
        }

        // A noncanonical terminal with VMIN=0 can return zero on an idle
        // read. Its VTIME timer starts at read, which poll would postpone.
        let mut termios = std::mem::MaybeUninit::<libc::termios>::uninit();
        // SAFETY: tcgetattr initializes termios on success. Nonterminals fail.
        if unsafe { libc::tcgetattr(fd, termios.as_mut_ptr()) } != 0 {
            return true;
        }
        let termios = unsafe { termios.assume_init() };
        termios.c_lflag & libc::ICANON != 0 || termios.c_cc[libc::VMIN] != 0
    }

    pub(super) fn wait_for_input(fd: RawFd) -> std::io::Result<()> {
        let mut input = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
        loop {
            // SAFETY: poll receives one valid entry and waits for input or EOF.
            if unsafe { libc::poll(&mut input, 1, -1) } >= 0 { return Ok(()); }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted { return Err(error); }
        }
    }

    fn input_idle(fd: RawFd) -> bool {
        let mut input = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
        // SAFETY: poll receives one valid entry and returns immediately.
        // EOF and poll errors also take precedence over maintenance.
        (unsafe { libc::poll(&mut input, 1, 0) }) == 0
    }

    fn wait_until_idle(session: &Mutex<Activity>, fd: RawFd) -> MutexGuard<'_, Activity> {
        loop {
            let activity = session.lock().unwrap();
            let since_reply = activity.last_reply.map_or(Duration::ZERO, |at| at.elapsed());
            let remaining = IDLE_GRACE.saturating_sub(since_reply);
            if remaining.is_zero() && !activity.buffered_input && input_idle(fd) {
                return activity;
            }
            drop(activity);
            std::thread::sleep(if remaining.is_zero() { IDLE_GRACE } else { remaining });
        }
    }

    fn worth_trimming(free: usize) -> bool {
        // Cold Classroom retained 1.6 GiB of scratch, Water over 20 GiB.
        free >= 1 << 30
    }

    pub(super) fn schedule() {
        static QUEUE: OnceLock<Option<SyncSender<()>>> = OnceLock::new();
        let queue = QUEUE.get_or_init(|| {
            let (tx, rx) = sync_channel(1);
            let started = std::thread::Builder::new().spawn(move || {
                while rx.recv().is_ok() {
                    loop {
                        let _idle = wait_until_idle(&SESSION, libc::STDIN_FILENO);
                        let _ = rx.try_recv();
                        // SAFETY: glibc synchronizes both the arena summary and trim.
                        if !worth_trimming(unsafe { libc::mallinfo2().fordblks }) {
                            break;
                        }
                        // Input can arrive while glibc is summing the arenas.
                        // In that case keep the same check pending.
                        if input_idle(libc::STDIN_FILENO) {
                            unsafe { libc::malloc_trim(0); }
                            break;
                        }
                    }
                }
            });
            // The session can keep serving requests if the OS refused the thread.
            started.ok().map(|_| tx)
        });
        // One pending check covers all retirements before it starts.
        if let Some(queue) = queue {
            let _ = queue.try_send(());
        }
    }

    #[test]
    fn threshold_is_one_gibibyte() {
        assert!(!worth_trimming((1 << 30) - 1));
        assert!(worth_trimming(1 << 30));
    }

    #[test]
    fn readable_input_and_eof_defer_maintenance() {
        use std::io::Write;
        use std::os::{fd::AsRawFd, unix::net::UnixStream};
        let (reader, mut writer) = UnixStream::pair().unwrap();
        assert!(input_idle(reader.as_raw_fd()));
        writer.write_all(b"\n").unwrap();
        assert!(!input_idle(reader.as_raw_fd()));

        let (reader, writer) = UnixStream::pair().unwrap();
        drop(writer);
        assert!(!input_idle(reader.as_raw_fd()));
    }

    #[test]
    fn pending_check_waits_for_the_reply_and_prefetched_input() {
        use std::os::{fd::AsRawFd, unix::net::UnixStream};
        let (reader, _writer) = UnixStream::pair().unwrap();
        let session = Mutex::new(Activity { last_reply: None, buffered_input: false });
        let mut request = session.lock().unwrap();
        std::thread::scope(|scope| {
            let (started, waiting) = sync_channel(0);
            let (checked, completed) = sync_channel(1);
            let session = &session;
            scope.spawn(move || {
                started.send(()).unwrap();
                let _idle = wait_until_idle(session, reader.as_raw_fd());
                checked.send(Instant::now()).unwrap();
            });
            waiting.recv().unwrap();
            assert!(completed.try_recv().is_err());
            request.last_reply = Some(Instant::now());
            request.buffered_input = true;
            drop(request);
            assert!(completed.recv_timeout(2 * IDLE_GRACE).is_err());

            let replied = Instant::now();
            let mut follow_up = session.lock().unwrap();
            follow_up.buffered_input = false;
            follow_up.last_reply = Some(replied);
            drop(follow_up);
            assert!(completed.recv_timeout(Duration::from_secs(2)).unwrap() >= replied + IDLE_GRACE);
        });
    }
}

fn open(worker: &mut Worker, scene: Option<PathBuf>, document: Option<serde_json::Value>, base: Option<PathBuf>) -> Result<serde_json::Value> {
    let opened = match (scene, document) {
        (Some(path), None) if base.is_none() => worker.open(Source::File(&path))?,
        (None, Some(document)) => worker.open(Source::Document { document, base: base.as_deref().unwrap_or(Path::new(".")) })?,
        _ => return Err(usage(anyhow::anyhow!(
            "open takes scene (a file path) or document (the whole scene as JSON, with base naming the directory for its relative files), not both"))),
    };
    Ok(serde_json::json!({"revision": opened.revision, REUSE_KEY: opened.reuse.name(), "changed": opened.changed}))
}

/// Serve requests until `close` or end of input, then exit without driver teardown.
pub fn run(gpu: Option<vork_render::GpuSelection>) -> Result<()> {
    if let Some(index) = gpu { vork_render::select_gpu(index); }
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    use std::os::fd::AsRawFd;
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    let mut stdin = {
        use std::os::fd::AsFd;
        // Own the only read buffer so maintenance can see prefetched requests
        // that are no longer visible to poll.
        let fd = std::io::stdin().as_fd().try_clone_to_owned().context("duplicate session input")?;
        std::io::BufReader::new(std::fs::File::from(fd))
    };
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
    let mut reply = |value: serde_json::Value| -> Result<()> {
        serde_json::to_writer(&mut stdout, &value).map_err(|e| crate::output::error(e.into()))?;
        stdout.write_all(b"\n").map_err(crate::output::error)?;
        stdout.flush().map_err(crate::output::error)
    };
    let mut worker = Worker::new();
    loop {
        // Leave ready input visible to maintenance until we hold the guard.
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        if stdin.buffer().is_empty() {
            let fd = stdin.get_ref().as_raw_fd();
            if scratch::can_wait_for_input(fd) {
                scratch::wait_for_input(fd).context("wait for session input")?;
            }
        }
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        let mut request_guard = scratch::request();
        let mut line = String::new();
        let read = stdin.read_line(&mut line);
        // Rust's Stdin treats EBADF (including write-only stdin) as EOF.
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        let read = match read {
            Err(error) if error.raw_os_error() == Some(libc::EBADF) => Ok(0),
            other => other,
        };
        if read.context("read session request")? == 0 { break; }
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        request_guard.buffered_input(!stdin.buffer().is_empty());
        if line.trim().is_empty() { continue; }
        // Match BufRead::lines, including for the request depth diagnostic.
        if line.ends_with('\n') {
            line.pop();
            if line.ends_with('\r') { line.pop(); }
        }
        // An inline document is one level inside the request.
        // A request nested past the limit carries a document nested past it: that is the scene's refusal, not a
        // malformed request, and the session goes on to the next request.
        let read = match vork_scene::diagnostics::too_deep(line.as_bytes(), vork_scene::MAX_JSON_DEPTH + 1, "request") {
            Some(refusal) => Err(refusal.context(vork_render::failure::FailureKind::Scene)),
            None => vork_scene::diagnostics::read_value(line.as_bytes()).map_err(|error| usage(anyhow::anyhow!("request is not a JSON object: {error}"))),
        };
        let mut value: serde_json::Value = match read {
            Ok(value) => value,
            Err(error) => {
                reply(failure(&serde_json::Value::Null, &error))?;
                continue;
            }
        };
        let id = value.as_object_mut().and_then(|object| object.remove("id")).unwrap_or(serde_json::Value::Null);
        let request: Request = match serde_json::from_value(value) {
            Ok(request) => request,
            Err(error) => { reply(failure(&id, &usage(anyhow::anyhow!("{error}; requests are open {{scene | document, base}}, render {{out, at, profile, set, channels, provisional}}, ask {{questions}} and close"))))?; continue; }
        };
        let result = match request {
            Request::Ask { questions } => worker.ask(&questions).map_err(crate::ask::single_line).map(|answers|serde_json::json!({
                "id":id,"ok":true,"revision":worker.revision(),"answers":answers,
            })),
            Request::Open { scene, document, base } => open(&mut worker, scene, document, base).map(|mut opened| {
                opened["id"] = id.clone();
                opened["ok"] = true.into();
                opened
            }),
            Request::Render(request) => { let provisional = request.provisional; request.shot().and_then(|shot| worker.render(shot)).map(|(report, reuse)| {
                serde_json::json!({
                    "id": id, "ok": true, "revision": worker.revision(), REUSE_KEY: reuse.name(), "installs": worker.installs(),
                    "provisional": provisional,
                    "device_reused": report["device_reused"], "install_reused": report["install_reused"],
                    "result": {"warnings": report["warnings"], "exposure": report["exposure"], "sidecars": report["sidecars"], "frames": report["frames"]},
                    "timing": {"wall_ms": report["wall_ms"], "device_ms": report["device_ms"], "install_ms": report["install_ms"],
                        "ready_ms": report["ready_ms"], "first_frame_ms": report["first_frame_ms"], "loop_wall_ms": report["loop_wall_ms"]},
                    "report": if report["render_settings"].is_null() { serde_json::Value::Null } else { serde_json::json!({"render_settings": report["render_settings"]}) },
                })
            }) }
            Request::Close => {
                worker.close();
                reply(serde_json::json!({"id": id, "ok": true}))?;
                break;
            }
        };
        match result {
            Ok(value) => reply(value)?,
            Err(error) => reply(failure(&id, &error))?,
        }
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        {
            drop(request_guard);
            vork_render::after_compile_reply(scratch::schedule);
        }
    }
    // Every output is delivered and every reply flushed; skip driver teardown like one-shot renders.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    let _shutdown_guard = scratch::request();
    drop(stdout);
    std::io::stderr().flush()?;
    worker.finish_pipeline_save();
    worker.close();
    vork_render::shot::exit_process(0)
}
