//! Immutable copies of admitted assets for a failed edit. Each generation
//! shares unchanged files with its predecessor. Installations read from these copies.
use super::*;
use std::{collections::BTreeMap, sync::Arc};
use tempfile::{Builder, TempPath};

#[derive(Clone, Default)]
pub(super) struct PublishedFiles {
    pub(super) media: crate::footage::Opened,
    assets: Arc<BTreeMap<PathBuf, (FileVersion, Arc<TempPath>)>>,
    containers: Arc<std::sync::Mutex<BTreeMap<PathBuf, Option<TempPath>>>>,
}

impl PublishedFiles {
    pub(super) fn capture(files: &[(PathBuf, FileVersion)], previous: &Self, media: &crate::footage::Opened) -> Result<Self> {
        let mut published = BTreeMap::new();
        for (path, version) in files {
            // Held movies and sound files retain their authored spelling. The
            // rollback render receives this same map, including after a rename.
            if media.held(path)?.is_some() { continue; }
            let absolute = std::path::absolute(path)?;
            let saved = if let Some((_, copy)) = previous.assets.get(&absolute).filter(|(old, _)| old == version) {
                copy.clone()
            } else {
                let suffix = absolute.extension().map_or(String::new(), |s| format!(".{}", s.to_string_lossy()));
                let copy = Builder::new().prefix("vork-revision-").suffix(&suffix).tempfile()?.into_temp_path();
                std::fs::copy(&version.target, &copy)?;
                anyhow::ensure!(crate::asset_revision::content(&copy)? == version.content && path.canonicalize()? == version.target,
                    "asset {} changed during installation; retry the command", path.display());
                Arc::new(copy)
            };
            published.insert(absolute, (version.clone(), saved));
        }
        Ok(Self { media: media.pins(), assets: Arc::new(published), containers: Default::default() })
    }

    pub(super) fn name_decode_source(&self, mut error: anyhow::Error) -> anyhow::Error {
        if let Some(location) = crate::failure::cause_mut::<crate::texture_source::DecodeLocation>(&mut error) {
            self.name_source(&mut location.path);
        } else if let Some(font) = crate::failure::cause_mut::<vork_std::moda::InvalidFont>(&mut error) {
            self.name_source(&mut font.path);
        }
        error
    }

    fn name_source(&self, path: &mut PathBuf) {
        if let Some((original, _)) = self.assets.iter().find(|(_, (_, copy))| copy.as_os_str() == path.as_os_str()) {
            path.clone_from(original);
        }
    }

    /// Source modules use the compiler's original virtual paths. Other resources
    /// use private files. Rewrite glTF sidecar URIs before handing them to gltf.
    pub(super) fn relocate(&self, scene: &mut Scene, base: &Path) -> Result<()> {
        if self.assets.is_empty() { return Ok(()); }
        let mut containers = self.containers.lock().unwrap();
        {
            let executable = std::env::current_exe()?.canonicalize()?;
            let directory = executable.parent().context("locate executable directory")?;
            for library in vork_scene::libraries::LIBRARIES {
                for data in library.images {
                    if let vork_scene::libraries::ProgramImage::File(relative) = data.image {
                        let path = directory.join(relative);
                        if let Some((_, copy)) = self.assets.get(&path) {
                            scene.library_files.insert(path, copy.to_path_buf());
                        }
                    }
                }
            }
        }
        // Rewrite containers while the field still identifies the resource type.
        fn node_files<'a>(nodes: &'a mut [Node], files: &mut Vec<&'a mut String>) {
            for node in nodes {
                files.extend(node.animation.iter_mut().filter_map(|l| l.clip.as_mut()?.file.as_mut()));
                if let NodeKind::Asset { file, .. } = &mut node.kind {
                    if !crate::splats::is_file(&*file) { files.push(file); }
                }
                node_files(&mut node.children, files);
            }
        }
        let mut files: Vec<_> = scene.animation.iter_mut().filter_map(|l| l.clip.as_mut()?.file.as_mut()).collect();
        node_files(&mut scene.nodes, &mut files);
        for file in files {
            let original = std::path::absolute(base.join(&*file))?;
            let Some((_, copy)) = self.assets.get(&original) else { continue };
            if !containers.contains_key(&original) {
                containers.insert(original.clone(), self.container(&original, copy.as_ref())?);
            }
            if let Some(container) = &containers[&original] {
                *file = container.to_str().context("revision resource path is not UTF-8")?.to_owned();
            }
        }
        let programs = scene.programs.clone();
        scene.map_resource_paths(|file| {
            let original = std::path::absolute(base.join(&*file))?;
            let Some((_, copy)) = self.assets.get(&original) else { return Ok(()); };
            *file = copy.to_str().context("revision resource path is not UTF-8")?.to_owned();
            Ok(())
        })?;
        scene.programs = programs;
        Ok(())
    }

    fn container(&self, original: &Path, saved: &Path) -> Result<Option<TempPath>> {
        let bytes = std::fs::read(saved)?;
        let mut glb = bytes.starts_with(b"glTF").then(|| gltf::binary::Glb::from_slice(&bytes)).transpose()?;
        let json = glb.as_ref().map_or(bytes.as_slice(), |container| container.json.as_ref());
        let mut doc: serde_json::Value = serde_json::from_slice(vork_scene::diagnostics::without_bom(json))?;
        let mut changed = false;
        for table in ["buffers", "images"] {
            for entry in doc.get_mut(table).and_then(serde_json::Value::as_array_mut).into_iter().flatten() {
                let Some(uri) = entry["uri"].as_str().filter(|u| !u.starts_with("data:")) else { continue; };
                let source = match uri.strip_prefix("file://").or_else(|| uri.strip_prefix("file:")) {
                    Some(path) => PathBuf::from(path),
                    None => original.parent().context("glTF directory")?.join(urlencoding::decode(uri)?.as_ref()),
                };
                let (_, blob) = self.assets.get(&std::path::absolute(&source)?)
                    .with_context(|| format!("missing published glTF sidecar {}", source.display()))?;
                entry["uri"] = format!("file:{}", blob.display()).into();
                changed = true;
            }
        }
        if !changed { return Ok(None); }
        let mut copy = Builder::new().prefix("vork-revision-").suffix(if glb.is_some() { ".glb" } else { ".gltf" }).tempfile()?;
        if let Some(glb) = &mut glb {
            glb.json = serde_json::to_vec(&doc)?.into();
            glb.to_writer(copy.as_file_mut())?;
        } else {
            serde_json::to_writer(copy.as_file_mut(), &doc)?;
        }
        Ok(Some(copy.into_temp_path()))
    }
}
