//! The file identity shared by session installation, export proofs and host asset caches.
use std::{collections::{BTreeMap, BTreeSet}, io, sync::{Arc, Mutex}};
use tempfile::{Builder, TempPath};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FileVersion {
    pub target: PathBuf,
    pub content: blake3::Hash,
    /// Immutable copies need only their bytes. Movie and audio readers keep a live file.
    pub binding: Option<vork_media::MediaBinding>,
}

impl FileVersion {
    pub(crate) fn read_bound(path: &Path, signal: Option<&vork_inputs::FileSignal>) -> io::Result<Self> {
        let Some(signal) = signal else { return Self::read(path); };
        let content = blake3::Hasher::new().update_reader(signal.file())?.finalize();
        Ok(Self { target: signal.target().to_owned(), content, binding: None })
    }
    pub fn read(path: &Path) -> io::Result<Self> {
        let target = path.canonicalize()?;
        let content = observed_content(&target)?;
        Ok(Self { target, content, binding: None })
    }
}

pub(crate) fn content(path: &Path) -> io::Result<blake3::Hash> {
    let mut hash = blake3::Hasher::new();
    hash.update_reader(crate::source_file(path)?)?;
    Ok(hash.finalize())
}

/// Snapshot the source closure enumerated by admission.
pub(crate) fn file_versions(paths: BTreeSet<PathBuf>, opened: &crate::footage::Opened) -> Option<Vec<(PathBuf, FileVersion)>> {
    paths.into_iter().map(|path| opened.file_version(&path).ok().map(|version| (path, version))).collect()
}

/// A source keeps its authored name alongside its private backing or its exact
/// open failure. Readers use the name for their output and the backing for I/O.
#[derive(Clone, Default)]
pub struct SourceFiles {
    bindings: Arc<BTreeMap<PathBuf, SourceFile>>,
    discovered: Option<Arc<Discovered>>,
}

#[derive(Default)]
struct Discovered {
    origin: Option<PathBuf>,
    files: Mutex<BTreeMap<PathBuf, SourceFile>>,
    aliases: Mutex<BTreeMap<PathBuf, PathBuf>>,
    reads: vork_inputs::Receipt,
}

#[derive(Clone)]
enum Backing {
    Authored,
    Copy(Arc<TempPath>),
    Failed(vork_inputs::ReadError),
}

#[derive(Clone)]
pub(crate) struct SourceFile {
    pub name: PathBuf,
    backing: Backing,
}

struct Asset(PathBuf);
impl vork_inputs::Query for Asset {
    fn read_bound(&self, signal: Option<&vork_inputs::FileSignal>) -> io::Result<FileVersion> {
        FileVersion::read_bound(&self.0, signal)
    }
    fn signal(&self) -> Option<vork_inputs::FileSignal> { vork_inputs::FileSignal::open(&self.0) }
    type Value = FileVersion;
    fn key(&self) -> &Path { &self.0 }
    fn read(&self) -> io::Result<FileVersion> { FileVersion::read(&self.0) }
}

/// Publication and lazy discovery use the same immutable copy. The source
/// owns its backing, including when retained by an independent query reader.
pub(crate) fn save(name: &Path, version: &FileVersion) -> io::Result<Arc<TempPath>> {
    let suffix = name.extension().map_or(String::new(), |s| format!(".{}", s.to_string_lossy()));
    let copy = Builder::new().prefix("vork-revision-").suffix(&suffix).tempfile()?.into_temp_path();
    std::fs::copy(&version.target, &copy)?;
    if content(&copy)? != version.content || name.canonicalize()? != version.target {
        return Err(io::Error::other(format!("asset {} changed during installation; retry the command", name.display())));
    }
    Ok(Arc::new(copy))
}

impl SourceFiles {
    pub fn publish(&mut self) {
        self.discovered = Some(Arc::new(Discovered {
            origin: vork_inputs::current_dir().ok(),
            ..Default::default()
        }));
    }

    /// Published readers use the spelling as it was bound when their revision
    /// opened, even when its working directory has since been renamed.
    pub fn key(&self, path: &Path) -> io::Result<PathBuf> {
        if path.is_relative() && let Some(origin) = self.discovered.as_ref().and_then(|d| d.origin.as_ref()) {
            return std::path::absolute(origin.join(path));
        }
        std::path::absolute(path)
    }

    pub fn current(&self) -> bool {
        self.discovered.as_ref().is_none_or(|d| d.reads.valid())
    }

    pub fn stamp(&self, hash: &mut blake3::Hasher) {
        if let Some(discovered) = &self.discovered { hash.update(discovered.reads.live_stamp().as_bytes()); }
    }

    pub fn bind(&mut self, name: &Path, backing: Result<Arc<TempPath>, vork_inputs::ReadError>) -> io::Result<()> {
        let backing = match backing { Ok(copy) => Backing::Copy(copy), Err(error) => Backing::Failed(error) };
        let source = SourceFile { name: name.into(), backing };
        let key = self.key(name)?;
        let entries = Arc::make_mut(&mut self.bindings);
        if let Backing::Copy(copy) = &source.backing { entries.insert(copy.to_path_buf(), source.clone()); }
        entries.insert(key, source);
        Ok(())
    }

    pub(crate) fn alias(&self, backing: &Path, name: &Path) {
        if let Some(discovered) = &self.discovered {
            discovered.aliases.lock().unwrap().entry(backing.to_owned()).or_insert_with(|| name.to_owned());
        }
    }

    pub(crate) fn authored_name(&self, path: &Path) -> PathBuf {
        if let Some(source) = self.bindings.get(path) { return source.name.clone(); }
        self.discovered.as_ref().and_then(|d| d.aliases.lock().unwrap().get(path).cloned())
            .unwrap_or_else(|| path.to_owned())
    }

    pub fn source(&self, path: &Path) -> SourceFile {
        let Some(discovered) = &self.discovered else {
            return SourceFile { name: path.into(), backing: Backing::Authored };
        };
        let name = path.to_owned();
        let key = match self.key(path) {
            Ok(key) => key,
            Err(error) => return SourceFile { name, backing: Backing::Failed(error.into()) },
        };
        if let Some(source) = self.bindings.get(&key) { return source.clone(); }
        // Readers can discover files after publication. Never fall through
        // to a mutable author path in an installation or query reader.
        let mut files = discovered.files.lock().unwrap();
        files.entry(key).or_insert_with_key(|key| {
            let _scope = discovered.reads.enter();
            let copy = vork_inputs::read(Asset(key.clone())).and_then(|version| save(key, &version));
            let backing = match copy { Ok(copy) => Backing::Copy(copy), Err(error) => Backing::Failed(error.into()) };
            SourceFile { name, backing }
        }).clone()
    }
}
impl SourceFile {
    fn backing(&self) -> io::Result<&Path> {
        match &self.backing {
            Backing::Authored => Ok(&self.name),
            Backing::Copy(copy) => Ok(copy),
            Backing::Failed(error) => Err(error.error()),
        }
    }
    pub fn digest(&self) -> io::Result<blake3::Hash> { content(self.backing()?) }

    pub fn media_name(&self, key: PathBuf) -> io::Result<PathBuf> {
        match &self.backing {
            Backing::Authored => self.name.canonicalize(),
            Backing::Copy(_) => Ok(key),
            Backing::Failed(error) => Err(error.error()),
        }
    }

    pub fn open_media(&self, name: &Path) -> anyhow::Result<Arc<vork_media::MediaRevision>> {
        vork_media::MediaRevision::open_named(self.backing()?, name)
    }

    /// The raster store needs a different key for different publications of
    /// the same name. Only the authored path is given to its file reader.
    pub fn paint_key(&self) -> io::Result<String> {
        match &self.backing {
            Backing::Authored => Ok(self.name.canonicalize().unwrap_or_else(|_| self.name.clone()).display().to_string()),
            Backing::Copy(copy) => {
                let generation = blake3::hash(copy.as_os_str().as_encoded_bytes());
                Ok(format!("vork-paint:{generation}:{}", self.name.display()))
            }
            Backing::Failed(error) => Err(error.error()),
        }
    }

    pub fn paint_name(key: &str) -> &str {
        key.strip_prefix("vork-paint:").and_then(|binding| binding.split_once(':')).map_or(key, |(_, name)| name)
    }

    pub fn read(&self) -> io::Result<Vec<u8>> { std::fs::read(self.backing()?) }
    pub fn open_regular(&self) -> io::Result<std::fs::File> { crate::source_file(self.backing()?) }
    pub fn dimensions(&self, upright: bool) -> image::ImageResult<(u32, u32)> {
        let path = self.backing()?;
        if upright { moda_assets::decode::upright_dimensions(path) } else { image::image_dimensions(path) }
    }
}

/// Hashing a readable regular file and detecting the native read failure for a
/// directory both use the ordinary file read. Never enter a FIFO or device.
pub(crate) fn observed_content(path: &Path) -> io::Result<blake3::Hash> {
    let kind = std::fs::metadata(path)?.file_type();
    let file = if kind.is_file() || kind.is_dir() { std::fs::File::open(path)? } else { crate::source_file(path)? };
    Ok(blake3::Hasher::new().update_reader(file)?.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reader_discovering_an_unlisted_file_pins_and_verifies_it() -> io::Result<()> {
        let dir = tempfile::tempdir()?;
        let file = dir.path().join("late.png");
        let mut absent = SourceFiles::default();
        absent.publish();
        let missing = absent.source(&file);
        assert_eq!(missing.read().unwrap_err().kind(), io::ErrorKind::NotFound);
        std::fs::write(&file, b"first")?;
        assert!(!absent.current());
        assert_eq!(missing.read().unwrap_err().kind(), io::ErrorKind::NotFound);

        let mut published = SourceFiles::default();
        published.publish();
        let source = published.source(&file);
        assert_eq!(source.name, file);
        assert!(published.current());
        std::fs::write(&file, b"other")?;
        assert!(!published.current());
        assert_eq!(published.source(&file).read()?, b"first");
        drop(published);
        assert_eq!(source.read()?, b"first");
        Ok(())
    }
}
