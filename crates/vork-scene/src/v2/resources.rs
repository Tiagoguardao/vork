use super::*;
use anyhow::Context;
impl Scene {
    /// Resolve file-backed scene resources before writing a relocatable view.
    /// Inline source keeps its virtual filename so imports retain their base.
    pub fn make_resource_paths_absolute(&mut self, base:&Path) -> Result<()> {
        let base=std::path::absolute(base)?;
        fn absolute(file:&mut String,base:&Path) -> Result<()> {
            let path=Path::new(file);
            if path.is_relative() {*file=base.join(path).to_str().context("expanded resource path is not UTF-8")?.to_owned();}
            Ok(())
        }
        self.map_resource_paths(|file| absolute(file, &base))
    }

    /// Remap the normalized scene's external resource slots, preserving inline data.
    pub fn map_resource_paths(&mut self, mut map: impl FnMut(&mut String) -> Result<()>) -> Result<()> {
        macro_rules! relocate {
            (string, $access:expr) => {
                for file in $access { map(file)?; }
            };
            (json, $access:expr) => {
                for slot in $access {
                    if let Some(file) = slot.as_str() {
                        let mut file = file.to_owned();
                        map(&mut file)?;
                        *slot = file.into();
                    }
                }
            };
        }
        super::inline::resource_files!(relocate, self);
        fn tracks(tracks: &mut [Track], map: &mut impl FnMut(&mut String) -> Result<()>) -> Result<()> {
            for track in tracks {
                let Some(field @ ("font" | "content")) = track.path.last().map(String::as_str) else { continue; };
                if matches!(track.path.as_slice(), [root, _, inputs, _] if root == "materials" && inputs == "inputs") { continue; }
                let Keys::Inline(keys) = &mut track.keys else { continue; };
                for value in keys.iter_mut().filter_map(|key| key.get_mut(1)) {
                    let mut font = |value: &mut serde_json::Value| -> Result<()> {
                        let mut files: FontFiles = serde_json::from_value(value.clone())?;
                        for file in files.files_mut() { map(file)?; }
                        *value = serde_json::to_value(files)?;
                        Ok(())
                    };
                    if field == "font" { font(value)?; }
                    else if let Some(runs) = value.as_array_mut() {
                        for file in runs.iter_mut().filter_map(|run| run.get_mut("font")).filter(|file| !file.is_null()) {
                            font(file)?;
                        }
                    }
                }
                // Rebuild the sampled values from the relocated authored keys.
                *track = serde_json::from_value(serde_json::to_value(&*track)?)?;
            }
            Ok(())
        }
        fn node_tracks(nodes: &mut [Node], map: &mut impl FnMut(&mut String) -> Result<()>) -> Result<()> {
            for node in nodes {
                for layer in &mut node.animation {
                    if let Some(layer_tracks) = &mut layer.tracks { tracks(layer_tracks, map)?; }
                }
                node_tracks(&mut node.children, map)?;
            }
            Ok(())
        }
        for layer in &mut self.animation {
            if let Some(layer_tracks) = &mut layer.tracks { tracks(layer_tracks, &mut map)?; }
        }
        node_tracks(&mut self.nodes, &mut map)
    }
}
