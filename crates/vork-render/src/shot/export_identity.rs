//! Identity shared by independently rendered MP4 ranges and the full WAV.
use std::{collections::BTreeSet, path::{Path, PathBuf}};
use vork_scene::Scene;
use crate::{asset_revision::file_versions, identity::digest};

/// Publish a proof only when admission can enumerate and read all inputs.
/// Relative source names keep copies staged in different worker directories
/// comparable. The cache's canonical target paths serve local reuse only.
pub(super) fn identify(
    scene: &Scene,
    base: &Path,
    files: BTreeSet<PathBuf>,
    profile: &vork_scene::Profile,
    default_denoise_keep_samples: Option<bool>,
    build_version: Option<&str>,
    opened: &crate::footage::Opened,
) -> Option<String> {
    let base = std::path::absolute(if base.as_os_str().is_empty() { Path::new(".") } else { base }).ok()?;
    let executable = std::env::current_exe().ok()?.canonicalize().ok()?;
    let library = executable.parent()?.join("libraries");
    let mut assets = Vec::with_capacity(files.len());
    for (path, version) in file_versions(files, opened)? {
        let path = std::path::absolute(path).ok()?;
        let (scope, key) = if let Ok(relative) = path.strip_prefix(&library) {
            ("library", relative)
        } else if let Ok(relative) = path.strip_prefix(&base) {
            ("scene", relative)
        } else {
            ("absolute", path.as_path())
        };
        assets.push((scope, key.to_str()?.to_owned(), version.content.to_hex().to_string()));
    }
    assets.sort_unstable();

    let identity = digest(&(
        "vork-export-v2",
        scene,
        assets,
        (vork_scene::libraries::profile(&profile.name).map(|_| &profile.name), profile.level, profile.version),
        (scene.render.settings.promotion_denoise, scene.render.settings.bounded_secondary_compilation,
            scene.render.settings.approximate_field_boundaries),
        default_denoise_keep_samples,
        env!("VORK_EXPORT_BUILD_ID"),
        env!("CARGO_PKG_VERSION"),
        env!("VORK_BUILD_PROFILE"),
        option_env!("VORK_NATIVE_FOOTAGE_READER"),
        build_version,
    )).ok()?;
    Some(format!("v2:{identity}"))
}

/// Called after finalization (including mux), while this run owns the destination.
pub(crate) fn record(meta: &mut serde_json::Value, identity: &str, path: &Path) -> anyhow::Result<()> {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    std::io::copy(&mut std::fs::File::open(path)?, &mut hash)?;
    meta["payload_sha256"] = format!("{:x}", hash.finalize()).into();
    meta["export_identity"] = identity.into();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proof_tracks_the_embedded_build_and_effective_profile() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let mut scene = Scene::from_json(br#"{"version":2,"composition":{"space":"2d"},"nodes":[]}"#)?;
        let profile = |name| vork_scene::Profile::parse(name, br#"{"version":1,"level":1}"#);
        let local = profile("/worker/a.json")?;
        scene.render.settings = local.resolve(&Default::default())?;
        let proof = |scene: &Scene, profile: &vork_scene::Profile, keep_samples, build_version: Option<&str>| {
            identify(scene, dir.path(), Default::default(), profile, keep_samples, build_version, &Default::default()).unwrap()
        };
        let original = proof(&scene, &local, None, Some("build-a"));
        assert_eq!(original, proof(&scene, &profile("/other/b.json")?, None, Some("build-a")));
        assert_ne!(original, proof(&scene, &local, None, Some("build-b")));
        let bundled = vork_scene::Profile::load("std/final")?;
        let mut final_scene = scene.clone();
        final_scene.render.settings = bundled.resolve(&Default::default())?;
        assert_ne!(proof(&final_scene, &bundled, Some(false), None), proof(&final_scene, &bundled, Some(true), None));
        scene.render.settings.seed += 1;
        assert_ne!(original, proof(&scene, &local, None, Some("build-a")));
        Ok(())
    }
}

#[cfg(all(test, unix))]
mod held_revision_tests {
    use super::*;
    use anyhow::{Context, Result, ensure};
    use std::{fs, os::unix::fs::symlink, process::Command, sync::Arc};

    #[test]
    fn directory_switch_keeps_the_admitted_mix_pictures_and_export_proof() -> Result<()> {
        let root = tempfile::tempdir()?;
        let base = root.path();
        for (name, color, hz) in [("a", "red", 330), ("b", "blue", 990)] {
            fs::create_dir(base.join(name))?;
            ensure!(Command::new("ffmpeg").args(["-v","error","-f","lavfi","-i"])
                .arg(format!("color={color}:s=16x16:r=5:d=1"))
                .args(["-f","lavfi","-i"]).arg(format!("sine=frequency={hz}:duration=1"))
                .args(["-c:v","libx264","-c:a","pcm_s16le"]).arg(base.join(name).join("clip.mkv"))
                .status()?.success(), "create {name} fixture");
        }
        let current = base.join("current");
        symlink("a", &current)?;
        let path = current.join("clip.mkv");
        let scene: Scene = serde_json::from_value(serde_json::json!({
            "version":2,"composition":{"space":"2d"},"images":{"clip":{"file":"current/clip.mkv"}},
            "nodes":[{"type":"path","d":"M0 0H1V1Z","fill":{"image":"clip"},"audio":{"file":"current/clip.mkv"}}]
        }))?;
        let opened = crate::footage::Opened::default();
        let files = crate::inspect_files(&scene, base, crate::InputIdentity::Export, &opened)?
            .context("complete export closure")?;
        let proof = |media: &crate::footage::Opened| -> Result<_> {
            identify(&scene, base, files.clone(), &vork_scene::Profile::load("std/final")?,
                None, None, media).context("export proof")
        };
        let mix = |media: &crate::footage::Opened|
            crate::audio::Mix::new_in(&scene, base, media)?.render(0., 0.25, None);
        let pixels = |media: &crate::footage::Opened| -> Result<_> {
            Ok(media.source(&path)?.frame(0., vork_time::Time::integer(24), 32)?.to_rgba8())
        };
        // With only a painted movie, admission must also bind the input that
        // proof will hash, before the first pixel reader opens.
        let mut silent = scene.clone();
        silent.nodes[0].audio = None;
        let paint = crate::footage::Opened::default();
        let painted_files = crate::inspect_files(&silent, base, crate::InputIdentity::Export, &paint)?
            .context("painted export closure")?;
        let held = paint.held(&path)?.context("painted movie admitted")?;
        assert!(painted_files.contains(&path));
        let control = crate::footage::Opened::default();
        let (a_proof, a_sound, a_picture) = (proof(&control)?, mix(&control)?, pixels(&control)?);
        let pin = opened.revision(&path)?;
        assert!(Arc::ptr_eq(&pin, &opened.revision(&base.join("a/clip.mkv"))?));
        symlink("b", base.join("next"))?;
        fs::rename(base.join("next"), &current)?;
        assert!(Arc::ptr_eq(&held, &paint.revision(&path)?));
        assert_eq!(pixels(&paint)?, a_picture);
        assert_eq!(proof(&opened)?, a_proof);
        assert_eq!(mix(&opened)?, a_sound);
        assert_eq!(pixels(&opened)?, a_picture);
        let fresh = crate::footage::Opened::default();
        assert_ne!(proof(&fresh)?, a_proof);
        assert_ne!(mix(&fresh)?, a_sound);
        assert_ne!(pixels(&fresh)?, a_picture);
        Ok(())
    }
}
