//! Independent pixel and frame-record outputs, with FFmpeg diagnostics on stderr.
use crate::DecoderProcess;
use std::{
    io,
    process::{Command, Stdio},
};

#[cfg(unix)]
pub struct PixelPipes {
    input: io::PipeReader,
    output: io::PipeWriter,
}

#[cfg(unix)]
impl PixelPipes {
    pub fn new() -> io::Result<Self> {
        let (input, output) = io::pipe()?;
        Ok(Self { input, output })
    }

    pub fn records_url(&self) -> &str {
        "pipe:3"
    }

    pub fn pixels_url(&self) -> &str {
        "pipe:1"
    }

    pub fn start(
        self,
        mut command: Command,
    ) -> io::Result<(
        DecoderProcess,
        std::os::unix::net::UnixStream,
        io::PipeReader,
    )> {
        use std::os::{
            fd::{AsRawFd, OwnedFd},
            unix::{net::UnixStream, process::CommandExt},
        };
        let (pixels, writer) = UnixStream::pair()?;
        command.stdout(Stdio::from(OwnedFd::from(writer)));
        let output = self.output;
        // Only the child gets an inheritable descriptor. Keep the original
        // CLOEXEC writer owned by Command until spawn returns.
        unsafe {
            command.pre_exec(move || {
                let fd = output.as_raw_fd();
                let result = if fd == 3 {
                    libc::fcntl(fd, libc::F_SETFD, 0)
                } else {
                    libc::dup2(fd, 3)
                };
                if result < 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        Ok((DecoderProcess::start(command)?, pixels, self.input))
    }
}

// Windows only maps the three standard handles into the child's CRT. FFmpeg
// writes records to stdout. The rawvideo muxer opens a local named pipe once;
// a filtergraph can open and close its metadata output during initialization.
#[cfg(windows)]
pub struct PixelPipes {
    input: std::fs::File,
    url: String,
}

#[cfg(windows)]
impl PixelPipes {
    pub fn new() -> io::Result<Self> {
        use std::{
            os::windows::io::FromRawHandle,
            sync::atomic::{AtomicU64, Ordering},
        };
        use windows_sys::Win32::{
            Foundation::INVALID_HANDLE_VALUE, Storage::FileSystem::*, System::Pipes::*,
        };
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let name = format!(
            "vork-frames-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let url = format!("//./pipe/{name}");
        let wide: Vec<_> = url.replace('/', "\\").encode_utf16().chain([0]).collect();
        let handle = unsafe {
            CreateNamedPipeW(
                wide.as_ptr(),
                PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
                PIPE_NOWAIT | PIPE_REJECT_REMOTE_CLIENTS,
                1,
                0,
                65536,
                0,
                std::ptr::null(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            input: unsafe { std::fs::File::from_raw_handle(handle) },
            url,
        })
    }

    pub fn records_url(&self) -> &str {
        "pipe:1"
    }

    pub fn pixels_url(&self) -> &str {
        &self.url
    }

    pub fn start(
        self,
        mut command: Command,
    ) -> io::Result<(
        DecoderProcess,
        Box<dyn io::Read + Send>,
        std::process::ChildStdout,
    )> {
        use std::{os::windows::io::AsRawHandle, ptr};
        use windows_sys::Win32::{
            Foundation::{ERROR_NO_DATA, ERROR_PIPE_CONNECTED, ERROR_PIPE_LISTENING},
            System::Pipes::*,
        };
        command.stdout(Stdio::piped());
        let child = DecoderProcess::start(command)?;
        let records = child.stdout();
        let handle = self.input.as_raw_handle();
        // The filter may fail to initialize. Wait for a client only while the
        // decoder is alive, then do ordinary blocking pixel reads.
        let mut exited = false;
        loop {
            if unsafe { ConnectNamedPipe(handle, ptr::null_mut()) } == 0 {
                let err = io::Error::last_os_error();
                match err.raw_os_error().map(|v| v as u32) {
                    Some(ERROR_PIPE_CONNECTED) => break,
                    // A short decode can close before the first poll. Its queued
                    // bytes are still readable from the server end.
                    Some(ERROR_NO_DATA) => return Ok((child, Box::new(self.input), records)),
                    Some(ERROR_PIPE_LISTENING) => {}
                    _ => return Err(err),
                }
                // Recheck the pipe once after exit in case the writer connected
                // between ConnectNamedPipe and try_wait.
                if exited {
                    return Ok((child, Box::new(io::empty()), records));
                }
                exited = child.0.lock().unwrap().try_wait()?.is_some();
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        if unsafe { SetNamedPipeHandleState(handle, &PIPE_WAIT, ptr::null(), ptr::null()) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok((child, Box::new(self.input), records))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn exit_before_opening_pixel_pipe_yields_eof() -> anyhow::Result<()> {
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error", "-vork-missing-option"])
            .stderr(Stdio::null());
        let (child, mut pixels, mut records) = PixelPipes::new()?.start(cmd)?;
        assert_eq!(pixels.read(&mut [0])?, 0);
        assert_eq!(records.read(&mut [0])?, 0);
        assert!(!child.wait()?.success());
        Ok(())
    }

    #[test]
    fn ffmpeg_emits_records_separately_from_pixels() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let path = tmp.path().join("frames.gray");
        std::fs::write(&path, [21, 42])?;
        let pipe = PixelPipes::new()?;
        let url = pipe.records_url().replace(':', r"\:");
        let filter = format!(
            r"metadata=mode=add:key=test:value=1,metadata=mode=print:key=test:file='{url}':direct=1"
        );
        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "gray",
            "-video_size",
            "1x1",
            "-i",
        ])
        .arg(path)
        .args([
            "-vf",
            &filter,
            "-threads:v",
            "1",
            "-f",
            "rawvideo",
            "-y",
            pipe.pixels_url(),
        ]);
        let (child, mut raw, mut records) = pipe.start(cmd)?;
        let mut pixels = Vec::new();
        raw.read_to_end(&mut pixels)?;
        let mut text = String::new();
        records.read_to_string(&mut text)?;
        assert!(child.wait()?.success());
        assert_eq!(pixels, [21, 42]);
        assert_eq!(
            text.lines().filter(|l| l.starts_with("frame:")).count(),
            2,
            "{text}"
        );
        Ok(())
    }
}
