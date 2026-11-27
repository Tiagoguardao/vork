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
        let mut media = media.pins();
        media.files.publish();
        for (path, version) in files {
            // Held movies and sound files retain their authored spelling. The
            // rollback render receives this same map, including after a rename.
            if media.held(path)?.is_some() { continue; }
            let absolute = std::path::absolute(path)?;
            let saved = if let Some((_, copy)) = previous.assets.get(&absolute).filter(|(old, _)| old == version) {
                copy.clone()
            } else {
                crate::asset_revision::save(path, version)?
            };
            media.files.bind(path, Ok(saved.clone()))?;
            published.insert(absolute, (version.clone(), saved));
        }
        Ok(Self { media, assets: Arc::new(published), ..Default::default() })
    }

    pub(super) fn saved(&self, path: &Path) -> Result<Option<&Path>> {
        let path = std::path::absolute(path)?;
        Ok(self.assets.get(&path).map(|(_, copy)| copy.as_ref().as_ref()))
    }

    pub(super) fn scene(&self, scene: &Scene, base: &Path) -> Result<Scene> {
        self.relocated(scene, base, None)
    }

    pub(super) fn install_scene(&self, scene: &Scene, base: &Path, files: Option<&[(PathBuf, FileVersion)]>) -> Result<Scene> {
        // Canvas identity hashes the resolved scene. Relocate only its draw
        // inputs when admission proves that list complete.
        self.relocated(scene, base, files.filter(|_| scene.is_canvas()))
    }

    pub(super) fn retain_install_containers(&mut self, previous: &Self) {
        self.containers.clone_from(&previous.containers);
    }

    /// Source modules use the compiler's original virtual paths. Other resources
    /// use private files. Rewrite glTF sidecar URIs before handing them to gltf.
    fn relocated(&self, scene: &Scene, base: &Path, install_files: Option<&[(PathBuf, FileVersion)]>) -> Result<Scene> {
        let mut scene = scene.clone();
        if self.assets.is_empty() { return Ok(scene); }
        let included = |path: &Path| {
            install_files.is_none_or(|files| files.binary_search_by(|(file, _)| file.as_path().cmp(path)).is_ok())
        };
        let mut containers = self.containers.lock().unwrap();
        {
            let executable = std::env::current_exe()?.canonicalize()?;
            let directory = executable.parent().context("locate executable directory")?;
            for library in vork_scene::libraries::LIBRARIES {
                for data in library.images {
                    if let vork_scene::libraries::ProgramImage::File(relative) = data.image {
                        let path = directory.join(relative);
                        if included(&path) && let Some((_, copy)) = self.assets.get(&path) {
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
            let path = base.join(&*file);
            if !included(&path) { continue; }
            let original = std::path::absolute(&path)?;
            let Some((_, copy)) = self.assets.get(&original) else { continue };
            if !containers.contains_key(&original) {
                containers.insert(original.clone(), self.container(&original, copy.as_ref())?);
            }
            if let Some(container) = &containers[&original] {
                self.media.files.alias(container, &path);
                *file = container.to_str().context("revision resource path is not UTF-8")?.to_owned();
            }
        }
        // Image readers receive the binding with their Opened input. The scene
        // carries the author's spelling through checks, warnings and refusals.
        let images = std::mem::take(&mut scene.images);
        let programs = std::mem::take(&mut scene.programs);
        scene.map_resource_paths(|file| {
            let path = base.join(&*file);
            if included(&path) && let Some(copy) = self.saved(&path)? {
                *file = copy.to_str().context("revision resource path is not UTF-8")?.to_owned();
            }
            Ok(())
        })?;
        scene.images = images;
        scene.programs = programs;
        Ok(scene)
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
