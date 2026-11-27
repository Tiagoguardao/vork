//! Host observations for a kept request. The reader that produces a value also
//! owns the operation that can reproduce it. An unscoped one-shot reads live.
use std::{
    any::{Any, type_name},
    cell::RefCell,
    collections::BTreeMap,
    fmt,
    hash::{Hash, Hasher},
    io,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// Preserve native errors (including errno) and Rust errors with custom messages.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ReadError {
    raw: Option<i32>,
    kind: io::ErrorKind,
    message: String,
}

impl From<io::Error> for ReadError {
    fn from(e: io::Error) -> Self {
        Self {
            raw: e.raw_os_error(),
            kind: e.kind(),
            message: e.to_string(),
        }
    }
}
impl ReadError {
    pub fn error(&self) -> io::Error {
        match self.raw {
            Some(code) => io::Error::from_raw_os_error(code),
            None => io::Error::new(self.kind, self.message.clone()),
        }
    }
}

/// A successful read can use matching Unix metadata as a witness only after
/// the timestamp tick containing the last change had ended before that read.
#[derive(Clone, Debug)]
pub struct FileSignal {
    file: Arc<std::fs::File>,
    target: PathBuf,
    version: FileStat,
    observed_ns: i128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileStat {
    device: u64,
    inode: u64,
    size: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}
const SECOND_NS: i128 = 1_000_000_000;

impl FileStat {
    fn proof_after_ns(&self) -> i128 {
        // Infer precision from this inode, including ext filesystems with old
        // inode formats. Whole-second stamps are ambiguous (FAT has a two-second
        // tick). Even fractional stamps can come from a coarse kernel clock.
        let granularity = if self.modified.1 == 0 || self.changed.1 == 0 {
            2 * SECOND_NS
        } else {
            SECOND_NS
        };
        let (sec, nsec) = self.modified.max(self.changed);
        i128::from(sec) * SECOND_NS + i128::from(nsec) + granularity
    }

    fn read(meta: std::fs::Metadata) -> Option<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            (meta.ino() != 0 && (meta.is_file() || meta.is_dir())).then(|| Self {
                device: meta.dev(), inode: meta.ino(), size: meta.size(),
                modified: (meta.mtime(), meta.mtime_nsec()), changed: (meta.ctime(), meta.ctime_nsec()),
            })
        }
        #[cfg(not(unix))]
        { let _ = meta; None }
    }
}
impl FileSignal {
    pub fn open(path: &Path) -> Option<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            let target = path.canonicalize().ok()?;
            FileStat::read(target.metadata().ok()?)?;
            let file = std::fs::File::options().read(true).custom_flags(libc::O_NONBLOCK).open(&target).ok()?;
            Self::held(&target, Arc::new(file))
        }
        #[cfg(not(unix))]
        { let _ = path; None }
    }
    pub fn held(path: &Path, file: Arc<std::fs::File>) -> Option<Self> {
        // Sample before collecting metadata and reading the value. Waiting in
        // between looks cannot make an earlier ambiguous observation conclusive.
        let elapsed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok()?;
        let observed_ns = i128::try_from(elapsed.as_nanos()).ok()?;
        let version = FileStat::read(file.metadata().ok()?)?;
        Some(Self { target: path.canonicalize().ok()?, file, version, observed_ns })
    }
    pub fn target(&self) -> &Path { &self.target }
    pub fn file(&self) -> &std::fs::File { &self.file }
    pub fn refreshed(&self, path: &Path) -> Option<Self> {
        let next = Self::held(&self.target, self.file.clone())?;
        next.matches_binding(path).then_some(next)
    }
    pub fn current(&self, path: &Path) -> bool {
        self.observed_ns > self.version.proof_after_ns() && self.matches_binding(path)
    }
    /// Identify a racily clean file binding for a host-local cache. Re-observe
    /// the path before reusing the cached value.
    pub fn binding_stamp(&self, path: &Path) -> Option<blake3::Hash> {
        if !self.current(path) {
            return None;
        }
        let mut hash = blake3::Hasher::new();
        hash.update(self.target.as_os_str().as_encoded_bytes());
        let FileStat { device, inode, size, modified, changed } = self.version;
        for field in [device, inode, size] {
            hash.update(&field.to_le_bytes());
        }
        for (seconds, nanos) in [modified, changed] {
            hash.update(&seconds.to_le_bytes());
            hash.update(&nanos.to_le_bytes());
        }
        Some(hash.finalize())
    }
    fn matches_binding(&self, path: &Path) -> bool {
        let stamp = |meta: io::Result<std::fs::Metadata>| meta.ok().and_then(FileStat::read);
        stamp(self.file.metadata()) == Some(self.version)
            && stamp(path.metadata()) == Some(self.version)
            && path.canonicalize().is_ok_and(|target| target == self.target)
    }
}

pub trait Query: Send + Sync + 'static {
    type Value: Clone + Eq + Hash + Send + Sync;
    fn key(&self) -> &Path;
    fn read(&self) -> io::Result<Self::Value>;
    fn signal(&self) -> Option<FileSignal> { None }
    fn read_bound(&self, _signal: Option<&FileSignal>) -> io::Result<Self::Value> {
        self.read()
    }
    fn unchanged(&self, signal: &FileSignal, _value: &Self::Value) -> bool {
        signal.current(self.key())
    }

}

type Key = (&'static str, PathBuf);
trait Witness: Any + Send + Sync {
    fn agrees(&self, other: &dyn Witness) -> bool;
    fn current(&self) -> bool;
    fn live_hash(&self, hash: &mut Stamp);
    fn as_any(&self) -> &dyn Any;
}
struct Observation<Q: Query> {
    query: Q,
    signal: Mutex<Option<FileSignal>>,
    result: Result<Q::Value, ReadError>,
}
impl<Q: Query> Witness for Observation<Q> {
    fn agrees(&self, other: &dyn Witness) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|observed| self.result == observed.result)
    }
    fn current(&self) -> bool {
        self.with_live_result(|live| live == &self.result)
    }
    fn live_hash(&self, hash: &mut Stamp) {
        self.with_live_result(|live| live.hash(hash))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl<Q: Query> Observation<Q> {
    fn with_live_result<T>(&self, visit: impl FnOnce(&Result<Q::Value, ReadError>) -> T) -> T {
        let mut signal = self.signal.lock().unwrap();
        if signal.as_ref().zip(self.result.as_ref().ok())
            .is_some_and(|(signal, value)| self.query.unchanged(signal, value)) {
            return visit(&self.result);
        }
        let candidate = self.query.signal();
        let result = self.query.read_bound(candidate.as_ref()).map_err(ReadError::from);
        if result == self.result {
            *signal = candidate.filter(|signal| result.as_ref().is_ok_and(|value| self.query.unchanged(signal, value)));
        }
        visit(&result)
    }
}
struct Stamp(blake3::Hasher);
impl Hasher for Stamp {
    fn write(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }
    fn finish(&self) -> u64 {
        u64::from_le_bytes(self.0.finalize().as_bytes()[..8].try_into().unwrap())
    }
}

#[derive(Clone, Default)]
pub struct Receipt(Arc<Mutex<BTreeMap<Key, Arc<dyn Witness>>>>);
impl fmt::Debug for Receipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.0.lock().unwrap().keys())
            .finish()
    }
}

thread_local! { static ACTIVE: RefCell<Option<Receipt>> = const { RefCell::new(None) }; }
pub struct Scope {
    previous: Option<Receipt>,
    // This scope restores a thread-local value and must drop on its entering thread.
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Scope {
    fn replace(next: Option<Receipt>) -> Self {
        Self {
            previous: ACTIVE.with(|active| active.replace(next)),
            _thread: Default::default(),
        }
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|active| active.replace(self.previous.take()));
    }
}
impl Receipt {
    pub fn enter(&self) -> Scope {
        Scope::replace(Some(self.clone()))
    }
    pub fn active() -> Self {
        ACTIVE.with(|active| active.borrow().clone().unwrap_or_default())
    }
    pub fn fork(&self) -> Self {
        Self(Arc::new(Mutex::new(self.0.lock().unwrap().clone())))
    }
    pub fn is_empty(&self) -> bool {
        self.0.lock().unwrap().is_empty()
    }
    pub fn same(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.0, &other.0) {
            return true;
        }
        let prior = other.0.lock().unwrap().clone();
        let entries = self.0.lock().unwrap();
        entries.len() == prior.len()
            && entries.iter().all(|(key, entry)| {
                prior
                    .get(key)
                    .is_some_and(|before| entry.agrees(before.as_ref()))
            })
    }
    /// A new admission can consume a prior file observation when its own binding
    /// agrees with the old value and the held file still occupies the live path.
    pub fn adopt_file<Q: Query>(&self, previous: &Self, query: &Q) {
        let key = (type_name::<Q>(), query.key().to_owned());
        let Some(entry) = previous.0.lock().unwrap().get(&key).cloned() else { return; };
        let observed = entry.as_any().downcast_ref::<Observation<Q>>().unwrap();
        let unchanged = observed.signal.lock().unwrap().as_ref().zip(observed.result.as_ref().ok())
            .is_some_and(|(signal, value)| query.unchanged(signal, value));
        if unchanged { self.0.lock().unwrap().entry(key).or_insert(entry); }
    }
    pub fn valid(&self) -> bool {
        let _live = Scope::replace(None);
        self.0.lock().unwrap().values().all(|entry| entry.current())
    }
    pub fn live_stamp(&self) -> blake3::Hash {
        let _live = Scope::replace(None);
        let mut hash = Stamp(blake3::Hasher::new());
        for (key, entry) in self.0.lock().unwrap().iter() {
            key.hash(&mut hash);
            entry.live_hash(&mut hash);
        }
        hash.0.finalize()
    }
}

/// Repeated reads under the same scope consume the same answer. Validation
/// calls precisely this query's implementation again, outside the scope.
pub fn read<Q: Query>(query: Q) -> io::Result<Q::Value> {
    read_with(query, Q::read_bound)
}

/// The initial read may have an already-open descriptor. The receipt retains
/// the lookup and its change signal for verification.
pub fn read_with<Q: Query>(
    query: Q,
    held: impl FnOnce(&Q, Option<&FileSignal>) -> io::Result<Q::Value>,
) -> io::Result<Q::Value> {
    let Some(receipt) = ACTIVE.with(|active| active.borrow().clone()) else {
        return held(&query, None);
    };
    let key = (type_name::<Q>(), query.key().to_owned());
    let result = |entry: &Arc<dyn Witness>| {
        entry
            .as_any()
            .downcast_ref::<Observation<Q>>()
            .unwrap()
            .result
            .as_ref()
            .cloned()
            .map_err(ReadError::error)
    };
    if let Some(entry) = receipt.0.lock().unwrap().get(&key) {
        return result(entry);
    }
    // A semantic reader may itself use other recording readers. Do I/O
    // outside the lock and give racing consumers the first published answer.
    let signal = query.signal();
    let observed = held(&query, signal.as_ref()).map_err(ReadError::from);
    let signal = signal.filter(|signal| observed.as_ref().is_ok_and(|value| query.unchanged(signal, value)));
    let mut entries = receipt.0.lock().unwrap();
    result(entries.entry(key).or_insert_with(|| {
        Arc::new(Observation {
            query,
            signal: Mutex::new(signal),
            result: observed,
        })
    }))
}

struct CurrentDir;
impl Query for CurrentDir {
    type Value = PathBuf;
    fn key(&self) -> &Path {
        Path::new("")
    }
    fn read(&self) -> io::Result<PathBuf> {
        std::env::current_dir()
    }
}
pub fn current_dir() -> io::Result<PathBuf> {
    read(CurrentDir)
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Executable {
    path: PathBuf,
    bytes: blake3::Hash,
    mode: u32,
}
impl Executable {
    pub fn digest(&self) -> blake3::Hash { self.bytes }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn open(path: &Path) -> io::Result<Self> {
        let path = path.canonicalize()?;
        if !path.metadata()?.is_file() {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        let file = std::fs::File::open(&path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        #[cfg(unix)]
        let mode = {
            use std::os::unix::fs::PermissionsExt;
            metadata.permissions().mode()
        };
        #[cfg(not(unix))]
        let mode = u32::from(metadata.permissions().readonly());
        let bytes = blake3::Hasher::new().update_reader(file)?.finalize();
        Ok(Self { path, bytes, mode })
    }
}

struct Canonical(PathBuf);
impl Query for Canonical {
    type Value = PathBuf;
    fn key(&self) -> &Path {
        &self.0
    }
    fn read(&self) -> io::Result<PathBuf> {
        self.0.canonicalize()
    }
}
struct Contents(PathBuf);
impl Query for Contents {
    type Value = blake3::Hash;
    fn key(&self) -> &Path {
        &self.0
    }
    fn read(&self) -> io::Result<Self::Value> {
        if !self.0.metadata()?.is_file() {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        let file = std::fs::File::open(&self.0)?;
        Ok(blake3::Hasher::new().update_reader(file)?.finalize())
    }
}

/// Aliases and content are independent observations. Several library names
/// may refer to one large binary, whose bytes are read once in a capture.
pub fn installed_content(path: &Path) -> io::Result<(PathBuf, blake3::Hash)> {
    let canonical = read(Canonical(path.into()))?;
    let digest = read(Contents(canonical.clone()))?;
    Ok((canonical, digest))
}

/// Installation observations are process-wide. A worker restart selects a new build.
pub fn installed<Q: Query>(query: Q) -> io::Result<Q::Value> {
    use std::sync::OnceLock;
    type Entries = BTreeMap<Key, Box<dyn Any + Send + Sync>>;
    type Cell<T> = Arc<OnceLock<Result<T, ReadError>>>;
    static ENTRIES: OnceLock<Mutex<Entries>> = OnceLock::new();
    let key = (type_name::<Q>(), query.key().to_owned());
    let cell = {
        let mut entries = ENTRIES.get_or_init(Mutex::default).lock().unwrap();
        entries.entry(key).or_insert_with(|| Box::new(Cell::<Q::Value>::default()))
            .downcast_ref::<Cell<Q::Value>>().unwrap().clone()
    };
    cell.get_or_init(|| Receipt::live(|| query.read()).map_err(ReadError::from))
        .as_ref().cloned().map_err(ReadError::error)
}

/// Resolve an installed command once. Keep its native spelling for launch (a
/// multi-call executable can use argv[0]) and the identity of the selected image.
struct Program(PathBuf);
impl Query for Program {
    type Value = (PathBuf, Result<Executable, ReadError>);
    fn key(&self) -> &Path { &self.0 }
    fn read(&self) -> io::Result<Self::Value> {
        let mut name = self.0.clone();
        if cfg!(windows) && name.extension().is_none() { name.set_extension("exe"); }
        let path = if name.components().count() == 1 {
            program_directories().into_iter().map(|dir| dir.join(&name)).find(|p| executable(p))
        } else { Some(name) };
        // Keep the native Command error for an absent executable, or when the
        // OS uses a default search path and the process has no PATH setting.
        let path = path.and_then(|p| std::path::absolute(p).ok()).unwrap_or_else(|| self.0.clone());
        let identity = Executable::open(&path).map_err(ReadError::from);
        Ok((path, identity))
    }
}

fn executable(path: &Path) -> bool {
    if !path.is_file() { return false; }
    #[cfg(unix)]
    {
        use std::{ffi::CString, os::unix::ffi::OsStrExt};
        let Ok(path) = CString::new(path.as_os_str().as_bytes()) else { return false; };
        // The NUL-terminated path lives throughout this check of effective access.
        unsafe { libc::faccessat(libc::AT_FDCWD, path.as_ptr(), libc::X_OK, libc::AT_EACCESS) == 0 }
    }
    #[cfg(not(unix))]
    { true }
}

fn program_directories() -> Vec<PathBuf> {
    let mut directories = Vec::new();
    #[cfg(windows)]
    {
        use std::{ffi::OsString, os::windows::ffi::OsStringExt};
        use windows_sys::Win32::System::SystemInformation::{GetSystemDirectoryW, GetWindowsDirectoryW};
        // std::process searches the application and the two system directories
        // before the parent's PATH when the command inherits its environment.
        if let Ok(exe) = std::env::current_exe() && let Some(dir) = exe.parent() {
            directories.push(dir.to_owned());
        }
        for directory in [GetSystemDirectoryW, GetWindowsDirectoryW] {
            let mut buf = vec![0u16; 260];
            loop {
                // Both functions take a writable UTF-16 buffer and its capacity.
                let n = unsafe { directory(buf.as_mut_ptr(), buf.len() as u32) } as usize;
                if n == 0 { break; }
                if n < buf.len() {
                    directories.push(OsString::from_wide(&buf[..n]).into());
                    break;
                }
                buf.resize(n + 1, 0);
            }
        }
    }
    if let Some(paths) = std::env::var_os("PATH") {
        directories.extend(std::env::split_paths(&paths).filter(|p| !cfg!(windows) || !p.as_os_str().is_empty()));
    }
    directories
}

fn selected_program(program: &std::ffi::OsStr) -> (PathBuf, Result<Executable, ReadError>) {
    installed(Program(program.into())).expect("command selection retains discovery errors")
}

/// Use the same installation identity for persistent capability receipts and launches.
pub fn tool_identity(program: impl AsRef<std::ffi::OsStr>) -> io::Result<Executable> {
    selected_program(program.as_ref()).1.map_err(|error| error.error())
}

pub fn tool_command(program: impl AsRef<std::ffi::OsStr>) -> std::process::Command {
    let program = program.as_ref();
    let (path, _) = selected_program(program);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let mut command = std::process::Command::new(path);
        command.arg0(program);
        command
    }
    #[cfg(not(unix))]
    { std::process::Command::new(path) }
}

fn query_key(value: impl Hash) -> PathBuf {
    let mut key = Stamp(blake3::Hasher::new());
    value.hash(&mut key);
    key.0.finalize().to_hex().as_str().into()
}

struct ToolQuery {
    key: PathBuf,
    program: std::ffi::OsString,
    args: &'static [&'static str],
}
#[derive(Clone, Eq, PartialEq)]
pub struct ToolOutput(pub std::process::Output);
impl Hash for ToolOutput {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.status.to_string().hash(state);
        self.0.stdout.hash(state);
        self.0.stderr.hash(state);
    }
}
impl Query for ToolQuery {
    type Value = ToolOutput;
    fn key(&self) -> &Path {
        &self.key
    }
    fn read(&self) -> io::Result<ToolOutput> {
        tool_command(&self.program)
            .args(self.args)
            .output()
            .map(ToolOutput)
    }
}
pub fn tool_output(
    program: impl AsRef<std::ffi::OsStr>,
    args: &'static [&'static str],
) -> io::Result<std::process::Output> {
    let program = program.as_ref();
    installed(ToolQuery {
        key: query_key((program, args)),
        program: program.into(),
        args,
    })
    .map(|out| out.0)
}

impl Receipt {
    /// A semantic read covers the primitive operations used to compute it.
    /// Repeating the semantic operation must be sufficient to verify its result.
    pub fn live<T>(read: impl FnOnce() -> T) -> T {
        let _live = Scope::replace(None);
        read()
    }
    pub fn is_recording() -> bool {
        ACTIVE.with(|active| active.borrow().is_some())
    }
}

impl Receipt {
    /// Record a query evaluated against a held input (for example an open
    /// movie descriptor). Verification evaluates its live binding with read().
    pub fn record<Q: Query>(
        &self,
        query: Q,
        result: Result<Q::Value, ReadError>,
    ) -> io::Result<()> {
        let key = (type_name::<Q>(), query.key().to_owned());
        let observed = Arc::new(Observation { query, result, signal: Mutex::default() });
        match self.0.lock().unwrap().entry(key) {
            std::collections::btree_map::Entry::Occupied(entry) => {
                if !entry.get().agrees(observed.as_ref()) {
                    return Err(io::Error::other(format!(
                        "host input {} {:?} changed during capture; retry the command",
                        entry.key().0,
                        entry.key().1
                    )));
                }
            }
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(observed);
            }
        }
        Ok(())
    }
}

struct Bytes(PathBuf);
impl Query for Bytes {
    fn read_bound(&self, signal: Option<&FileSignal>) -> io::Result<Self::Value> {
        use std::io::Read;
        let Some(signal) = signal else { return self.read(); };
        let mut bytes = Vec::new();
        signal.file().read_to_end(&mut bytes)?;
        Ok(bytes.into())
    }
    fn signal(&self) -> Option<FileSignal> { FileSignal::open(&self.0) }
    type Value = Arc<[u8]>;
    fn key(&self) -> &Path {
        &self.0
    }
    fn read(&self) -> io::Result<Self::Value> {
        std::fs::read(&self.0).map(Into::into)
    }
}
/// Read authored bytes. The receipt owns exactly what the caller consumed.
pub fn bytes(path: impl AsRef<Path>) -> io::Result<Arc<[u8]>> {
    read(Bytes(path.as_ref().into()))
}

struct Directory(PathBuf);
impl Query for Directory {
    fn signal(&self) -> Option<FileSignal> { FileSignal::open(&self.0) }
    type Value = Arc<[PathBuf]>;
    fn key(&self) -> &Path {
        &self.0
    }
    fn read(&self) -> io::Result<Self::Value> {
        let mut children = std::fs::read_dir(&self.0)?
            .map(|e| e.map(|e| e.path()))
            .collect::<io::Result<Vec<_>>>()?;
        children.sort();
        Ok(children.into())
    }
}
pub fn directory(path: &Path) -> io::Result<Arc<[PathBuf]>> {
    read(Directory(path.into()))
}

struct Environment(PathBuf);
impl Query for Environment {
    type Value = Option<std::ffi::OsString>;
    fn key(&self) -> &Path {
        &self.0
    }
    fn read(&self) -> io::Result<Self::Value> {
        Ok(std::env::var_os(self.0.as_os_str()))
    }
}
pub fn var_os(name: &str) -> Option<std::ffi::OsString> {
    read(Environment(name.into())).expect("reading an environment setting is infallible")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Count(Arc<AtomicUsize>);
    impl Query for Count {
        type Value = usize;
        fn key(&self) -> &Path {
            Path::new("fixture")
        }
        fn read(&self) -> io::Result<usize> {
            Ok(self.0.load(Ordering::Relaxed))
        }
    }

    #[test]
    fn the_read_that_supplied_the_value_is_the_verifier() {
        let count = Arc::new(AtomicUsize::new(1));
        let receipt = Receipt::default();
        let scope = receipt.enter();
        assert_eq!(read(Count(count.clone())).unwrap(), 1);
        count.store(2, Ordering::Relaxed);
        assert_eq!(read(Count(count.clone())).unwrap(), 1);
        assert!(!receipt.valid());
        let other = Receipt::default();
        let other_scope = other.enter();
        assert_eq!(read(Count(count.clone())).unwrap(), 2);
        assert!(other.valid());
        assert!(!other.same(&receipt));
        drop(other_scope);
        drop(scope);
        assert_eq!(read(Count(count)).unwrap(), 2);
    }

    #[test]
    fn two_resources_can_share_the_same_held_input() {
        let count = Arc::new(AtomicUsize::new(1));
        let receipt = Receipt::default();
        receipt.record(Count(count.clone()), Ok(1)).unwrap();
        receipt.record(Count(count.clone()), Ok(1)).unwrap();
        assert!(receipt.record(Count(count), Ok(2)).is_err());
        assert!(receipt.valid());
    }

    #[test]
    fn installations_are_observed_once_across_receipts() {
        let value = Arc::new(AtomicUsize::new(1));
        let receipt = Receipt::default();
        let _scope = receipt.enter();
        assert_eq!(installed(Count(value.clone())).unwrap(), 1);
        value.store(2, Ordering::Relaxed);
        assert_eq!(installed(Count(value)).unwrap(), 1);
        assert!(receipt.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn matching_stat_skips_the_read_and_changed_stat_verifies_content() -> io::Result<()> {
        // Advance the observation clock across the strict tick boundary without
        // sleeping or relying on the test host's timestamp resolution.
        struct FileQuery(PathBuf, Arc<AtomicUsize>, Arc<std::sync::atomic::AtomicBool>);
        impl Query for FileQuery {
            type Value = Vec<u8>;
            fn key(&self) -> &Path { &self.0 }
            fn read(&self) -> io::Result<Self::Value> {
                self.1.fetch_add(1, Ordering::Relaxed);
                std::fs::read(&self.0)
            }
            fn signal(&self) -> Option<FileSignal> {
                let mut signal = FileSignal::open(&self.0)?;
                signal.observed_ns = signal.version.proof_after_ns() + i128::from(self.2.load(Ordering::Relaxed));
                Some(signal)
            }
        }
        let file = std::env::temp_dir().join(format!("vork-stat-test-{}", std::process::id()));
        let next = file.with_extension("new");
        std::fs::write(&file, b"one")?;
        let read_count = Arc::new(AtomicUsize::new(0));
        let receipt = Receipt::default();
        let _scope = receipt.enter();
        let safely_old = Arc::new(std::sync::atomic::AtomicBool::new(false));
        assert_eq!(read(FileQuery(file.clone(), read_count.clone(), safely_old.clone()))?, b"one");
        assert!(receipt.valid());
        assert_eq!(read_count.load(Ordering::Relaxed), 2);
        safely_old.store(true, Ordering::Relaxed);
        assert!(receipt.valid()); // The first safely old read renews the signal.
        assert!(receipt.valid());
        assert_eq!(read_count.load(Ordering::Relaxed), 3);
        std::fs::write(&next, b"one")?;
        std::fs::rename(&next, &file)?;
        assert!(receipt.valid());
        assert!(receipt.valid());
        assert_eq!(read_count.load(Ordering::Relaxed), 4);
        let mtime = file.metadata()?.modified()?;
        std::fs::write(&file, b"two")?;
        std::fs::File::open(&file)?.set_modified(mtime)?;
        assert!(!receipt.valid());
        assert_eq!(read_count.load(Ordering::Relaxed), 5);
        std::fs::remove_file(file)
    }

    #[test]
    fn native_and_custom_read_errors_round_trip() {
        for error in [
            io::Error::from_raw_os_error(20),
            io::Error::new(io::ErrorKind::InvalidData, "invalid header"),
        ] {
            let observed = ReadError::from(error);
            assert_eq!(ReadError::from(observed.error()), observed);
        }
    }
}

impl Receipt {
    pub fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    /// Scoped jobs inherit their installation's reader. A synchronous caller
    /// already has the reader of the request it is admitting.
    pub fn enter_worker(&self) -> Option<Scope> {
        (!Self::is_recording()).then(|| self.enter())
    }

    pub fn include(&self, other: &Self) -> io::Result<()> {
        if self.same_owner(other) {
            return Ok(());
        }
        let prior = other.0.lock().unwrap().clone();
        let mut entries = self.0.lock().unwrap();
        for (key, entry) in &prior {
            if entries
                .get(key)
                .is_some_and(|present| !present.agrees(entry.as_ref()))
            {
                return Err(io::Error::other(format!(
                    "host input {} {:?} changed during installation; retry the command",
                    key.0, key.1
                )));
            }
        }
        for (key, entry) in prior {
            entries.entry(key).or_insert(entry);
        }
        Ok(())
    }
}
