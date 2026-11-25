//! Run on NVIDIA's Linux Vulkan driver to exercise both ray admission and readback.
#![cfg(target_os = "linux")]

use std::{
    path::Path,
    process::{Command, Output},
};

fn render(root: &Path, scene: &str, profile: &str, denoise: bool) -> Output {
    std::fs::write(root.join("scene.json"), scene).unwrap();
    let binary = std::env::var_os("VORK_BIN").unwrap_or_else(|| env!("CARGO_BIN_EXE_vork").into());
    Command::new(binary)
        .current_dir(root)
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("RUST_BACKTRACE", "1")
        .args([
            "render",
            "scene.json",
            "--profile",
            profile,
            "--out",
            "movie.mp4",
            "--set",
            if denoise {
                "denoise=true"
            } else {
                "denoise=false"
            },
        ])
        .output()
        .unwrap()
}

#[test]
#[ignore = "requires Linux NVIDIA Vulkan with libcuda, FFmpeg and pinned Slang"]
fn hidden_cuda_ray_device_is_refused_before_video_starts() {
    let root = std::env::temp_dir().join(format!("vork-hidden-cuda-ray-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let previous = b"previous output";
    std::fs::write(root.join("movie.mp4"), previous).unwrap();

    let output = render(
        &root,
        include_str!("../../../examples/first-scene.json"),
        "std/draft",
        false,
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(4), "{stderr}");
    assert!(
        stderr.contains("Vulkan acceleration structures need the rendering GPU visible to CUDA"),
        "{stderr}"
    );
    assert_eq!(std::fs::read(root.join("movie.mp4")).unwrap(), previous);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "requires Linux NVIDIA Vulkan, FFmpeg with libx264, ffprobe, OIDN and pinned Slang"]
fn hidden_cuda_exports_a_non_ray_world_with_libx264() {
    let root =
        std::env::temp_dir().join(format!("vork-hidden-cuda-readback-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    // A visible emissive surface with no receiver lobes needs no acceleration structures.
    let scene = r#"{
        "version":2,
        "materials":{"source":{"inputs":{"base_weight":0,"specular_weight":0,"emission_luminance":1}}},
        "nodes":[
            {"type":"camera","translation":[3,2,5],"target":[0,0,0]},
            {"type":"mesh","geometry":{"std":"sphere"},"materials":["source"]}
        ],
        "render":{"resolution":[320,240],"duration":0.125}
    }"#;
    let output = render(&root, scene, "std/previs", true);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(
        stderr.contains("Video encoder: FFmpeg libx264 (readback)"),
        "{stderr}"
    );
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("movie.mp4.meta.json")).unwrap()).unwrap();
    assert_eq!(metadata["denoiser"], "cpu");
    let probe = Command::new("ffprobe")
        .current_dir(&root)
        .args([
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_read_frames",
            "-of",
            "json",
            "movie.mp4",
        ])
        .output()
        .unwrap();
    assert!(
        probe.status.success(),
        "{}",
        String::from_utf8_lossy(&probe.stderr)
    );
    let info: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
    let stream = &info["streams"][0];
    assert_eq!(stream["codec_name"], "h264");
    assert_eq!(stream["width"], 320);
    assert_eq!(stream["height"], 240);
    assert_eq!(stream["nb_read_frames"], "3");
    std::fs::remove_dir_all(root).unwrap();
}
