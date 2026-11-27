//! Hold the opened file across atomic replacement. Each consumer has its own
//! logical position, including when an open file has lost its last name.
use anyhow::{Context, Result, ensure};
use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Arc,
};

/// An open file's native identity, or a per-open identity on filesystems that
/// cannot supply one. In that case a new live capture always adopts its new handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MediaBinding {
    File { volume: u64, index: u128 },
    Open(u64),
}

impl MediaBinding {
    fn from_file(file: &File, metadata: &std::fs::Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let _ = file;
            if metadata.ino() != 0 {
                return Self::File { volume: metadata.dev(), index: metadata.ino().into() };
            }
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::Storage::FileSystem::{FILE_ID_INFO, FileIdInfo, GetFileInformationByHandleEx};
            let _ = metadata;
            let mut info = FILE_ID_INFO::default();
            // The live handle and the entire writable FILE_ID_INFO outlive the call.
            if unsafe { GetFileInformationByHandleEx(file.as_raw_handle(), FileIdInfo,
                (&raw mut info).cast(), std::mem::size_of::<FILE_ID_INFO>() as u32) } != 0 {
                let index = u128::from_le_bytes(info.FileId.Identifier);
                if index != 0 { return Self::File { volume: info.VolumeSerialNumber, index }; }
            }
        }
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_OPEN: AtomicU64 = AtomicU64::new(0);
        Self::Open(NEXT_OPEN.fetch_add(1, Ordering::Relaxed))
    }
}

pub struct MediaRevision {
    binding: MediaBinding,
    original: PathBuf,
    file: Arc<File>,
    signal: Option<vork_inputs::FileSignal>,
}

#[cfg(unix)]
impl std::os::fd::AsFd for MediaRevision {
    fn as_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        self.file.as_fd()
    }
}

/// A cursor over the held regular file. Unix dup shares an offset, so Rust
/// readers use pread. Windows ReOpenFile gives each reader a new file object.
pub struct MediaReader {
    file: File,
    offset: u64,
}
impl Read for MediaReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(&self.file, buf, self.offset)?;
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(&self.file, buf, self.offset)?;
        #[cfg(not(any(unix, windows)))]
        let n = {
            self.file.seek(SeekFrom::Start(self.offset))?;
            self.file.read(buf)?
        };
        self.offset += n as u64;
        Ok(n)
    }
}
impl Seek for MediaReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.offset = match pos {
            SeekFrom::Start(pos) => Some(pos),
            SeekFrom::Current(delta) => self.offset.checked_add_signed(delta),
            SeekFrom::End(delta) => self.file.metadata()?.len().checked_add_signed(delta),
        }
        .ok_or(io::ErrorKind::InvalidInput)?;
        Ok(self.offset)
    }
}

impl MediaRevision {
    pub fn open(path: &Path) -> Result<Arc<Self>> {
        Self::open_named(path, path)
    }

    /// Private revision files can supply the descriptor, but only the live
    /// author-visible binding may reach a decoder, a verifier or a diagnostic.
    pub fn open_named(backing: &Path, path: &Path) -> Result<Arc<Self>> {
        let mut options = File::options();
        options.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            use windows_sys::Win32::Storage::FileSystem::*;
            options.share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE);
        }
        // Opening a FIFO for reading must not block admission before fstat.
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NONBLOCK);
        }
        let mut file = options
            .open(backing)
            .with_context(|| format!("open media {}", path.display()))?;
        let metadata = file.metadata()?;
        ensure!(
            metadata.is_file(),
            "{} is not a regular file; restore a readable regular file or correct its path",
            path.display()
        );
        // A list with relative entries can fail before ffprobe emits format_name:
        // its first member would be opened relative to the descriptor path.
        let mut signature = Vec::new();
        (&mut file).take(20).read_to_end(&mut signature)?;
        file.rewind()?;
        let binding = MediaBinding::from_file(&file, &metadata);
        let file = Arc::new(file);
        let revision = Self {
            binding,
            original: path.into(),
            signal: vork_inputs::FileSignal::held(path, file.clone()),
            file,
        };
        if signature == b"ffconcat version 1.0" {
            revision.check_format("concat")?;
        } else if signature.starts_with(b"#EXTM3U") {
            revision.check_format("hls")?;
        }
        Ok(Arc::new(revision))
    }

    /// A scene pins one regular file for all consumers of a media revision.
    pub fn check_format(&self, names: &str) -> Result<()> {
        if let Some(list) = names.split(',').find(|n| matches!(*n, "concat" | "hls")) {
            anyhow::bail!(
                "{} is a {list} media list; one descriptor cannot pin its member files. Export a self-contained media file and use that file instead",
                self.original.display()
            );
        }
        Ok(())
    }

    pub fn original(&self) -> &Path {
        &self.original
    }

    /// Identity of the file object behind the live descriptor. While the descriptor
    /// is held, replacing its name cannot recycle the old object's identity.
    pub fn signal(&self) -> Option<&vork_inputs::FileSignal> { self.signal.as_ref() }

    pub fn binding(&self) -> MediaBinding { self.binding }

    pub fn reader(&self) -> io::Result<MediaReader> {
        #[cfg(unix)]
        let file = self.file.try_clone()?;
        #[cfg(not(unix))]
        let file = self.independent_file()?;
        Ok(MediaReader { file, offset: 0 })
    }

    /// An independently seekable file for a native or external decoder.
    pub fn independent_file(&self) -> io::Result<File> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            #[cfg(target_os = "linux")]
            {
                use std::os::fd::AsRawFd;
                if let Ok(file) = File::open(format!("/proc/self/fd/{}", self.file.as_raw_fd())) {
                    return Ok(file);
                }
            }
            // Opening /dev/fd on Darwin is a dup. Open the vnode by name and
            // check the resulting handle against the still-live pin instead.
            let pin = self.file.metadata()?;
            #[cfg(target_os = "macos")]
            let by_id = PathBuf::from(format!("/.vol/{}/{}", pin.dev(), pin.ino()));
            let names = [
                self.original.as_path(),
                #[cfg(target_os = "macos")]
                &by_id,
            ];
            for name in names {
                use std::os::unix::fs::OpenOptionsExt;
                if let Ok(file) = File::options()
                    .read(true)
                    .custom_flags(libc::O_NONBLOCK)
                    .open(name)
                {
                    let other = file.metadata()?;
                    if (pin.dev(), pin.ino()) == (other.dev(), other.ino()) {
                        return Ok(file);
                    }
                }
            }
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "{} was replaced after the scene opened it; reopen the scene",
                    self.original.display()
                ),
            ))
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::{AsRawHandle, FromRawHandle};
            use windows_sys::Win32::{
                Foundation::{GENERIC_READ, INVALID_HANDLE_VALUE},
                Storage::FileSystem::*,
            };
            let handle = unsafe {
                ReOpenFile(
                    self.file.as_raw_handle(),
                    GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    0,
                )
            };
            if handle == INVALID_HANDLE_VALUE {
                return Err(io::Error::last_os_error());
            }
            Ok(unsafe { File::from_raw_handle(handle) })
        }
        #[cfg(not(any(unix, windows)))]
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "this platform has no held media handle route",
        ))
    }

    /// The file publication protects the input and the worker owns the decoder installation.
    pub fn probe(&self, args: &'static [&'static str]) -> Result<std::process::Output> {
        Ok(self.attach(vork_inputs::tool_command("ffprobe").args(args))?.output()?)
    }

    /// Set stdin and append the decoder input operand (after -i for FFmpeg).
    pub fn input<'a>(&self, command: &'a mut Command) -> Result<&'a mut Command> {
        self.attach(command)
    }

    pub(crate) fn attach<'a>(&self, command: &'a mut Command) -> Result<&'a mut Command> {
        let file = self.independent_file()?;
        command.stdin(Stdio::from(file));
        #[cfg(unix)]
        if Path::new("/dev/fd").is_dir() {
            return Ok(command.arg("/dev/fd/0"));
        }
        require_fd_protocol(command.get_program())?;
        command.arg("fd:");
        Ok(command)
    }
}

// FFmpeg and FFprobe may come from different installations. Cache successful
// input-protocol probes for each executable on the current search path.
fn has_fd_input(protocols: &str) -> bool {
    protocols
        .split_once("Input:")
        .and_then(|(_, s)| s.split("Output:").next())
        .is_some_and(|inputs| inputs.lines().any(|l| l.trim() == "fd"))
}

fn require_fd_protocol(program: &std::ffi::OsStr) -> Result<()> {
    let output = vork_inputs::tool_output(program, &["-hide_banner", "-protocols"])
        .with_context(|| format!("check input protocols for {}", program.to_string_lossy()))?;
    let protocols = String::from_utf8_lossy(&output.stdout);
    ensure!(
        output.status.success() && has_fd_input(&protocols),
        "{} lacks the fd: input protocol; install an FFmpeg build with fd input support",
        program.to_string_lossy()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fd_must_be_an_input_protocol() {
        assert!(has_fd_input(
            "Supported file protocols:\r\nInput:\r\n  fd\r\nOutput:\r\n  pipe\r\n"
        ));
        assert!(!has_fd_input("Input:\n  pipe\nOutput:\n  fd\n"));
        assert!(!has_fd_input("Input:\n  fdisk\nOutput:\n"));
    }

    #[test]
    fn relative_lists_are_rejected_before_opening_members() -> Result<()> {
        let dir = tempfile::tempdir()?;
        for (format, contents) in [
            ("concat", "ffconcat version 1.0\nfile relative.mp4\n"),
            (
                "hls",
                "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:2,\nrelative.ts\n#EXT-X-ENDLIST\n",
            ),
        ] {
            let path = dir.path().join("renamed.mov");
            std::fs::write(&path, contents)?;
            let err = MediaRevision::open(&path)
                .err()
                .context("accepted list")?
                .to_string();
            for expected in [path.to_str().unwrap(), format, "member files"] {
                assert!(err.contains(expected), "{err}");
            }
        }
        Ok(())
    }

    #[test]
    fn a_private_backing_cannot_become_the_decoder_name() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let backing = dir.path().join("private-copy");
        let name = dir.path().join("plate.mp4");
        std::fs::write(&backing, b"ffconcat version 1.0\nfile relative.mp4\n")?;
        let error = MediaRevision::open_named(&backing, &name).err().context("accepted playlist")?;
        assert!(error.to_string().contains(name.to_str().unwrap()));
        assert!(!error.to_string().contains("private-copy"));
        Ok(())
    }

    #[test]
    fn held_readers_have_independent_offsets_after_replacement() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let live = dir.path().join("file");
        let next = dir.path().join("next");
        std::fs::write(&live, b"original")?;
        let pin = MediaRevision::open(&live)?;
        let mut first = pin.reader()?;
        first.seek(SeekFrom::Start(3))?;
        std::fs::write(&next, b"replaced")?;
        std::fs::rename(&next, &live)?;
        let mut second = pin.reader()?;
        let mut text = String::new();
        second.read_to_string(&mut text)?;
        assert_eq!(text, "original");
        text.clear();
        first.read_to_string(&mut text)?;
        assert_eq!(text, "ginal");
        Ok(())
    }
}
