//! The file identity shared by session installation, export proofs and host asset caches.
use std::{collections::BTreeSet, io};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FileVersion {
    pub target: PathBuf,
    pub content: blake3::Hash,
}

impl FileVersion {
    pub fn read(path: &Path) -> io::Result<Self> {
        let target = path.canonicalize()?;
        let content = content(&target)?;
        Ok(Self { target, content })
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
