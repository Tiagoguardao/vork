#[test]
#[ignore = "requires a real GPU, pinned Slang and FFmpeg with libvpx, zscale and tonemap"]
fn canvas_footage_matches_ffmpeg_at_the_requested_instant() {
    assert_eq!(std::env::var("VORK_GPU_TESTS").as_deref(), Ok("1"));
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/footage_formats.py");
    let status = std::process::Command::new("python3")
        .arg(script)
        .arg(env!("CARGO_BIN_EXE_vork"))
        .status()
        .expect("run FFmpeg reference comparison");
    assert!(status.success());
}
