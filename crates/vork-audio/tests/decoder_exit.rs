// The exit sweep is process-wide, so this test owns its process.
#[test]
fn exit_reaps_a_decoder_with_unread_audio() -> anyhow::Result<()> {
    let path = std::env::temp_dir().join(format!("vork-short-prefix-{}.flac", std::process::id()));
    let encoded = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-f", "lavfi", "-i", "sine=duration=30", "-y"])
        .arg(&path).status()?;
    assert!(encoded.success());
    let source = vork_audio::Source::open(&path)?;
    let mut reader = source.reader();
    assert_eq!(reader.prefix(2)?.frames.len(), 2);
    vork_media::stop_decoders_for_exit();
    let result = reader.prefix(usize::MAX);
    drop(reader);
    std::fs::remove_file(path)?;
    assert!(result.is_err(), "the decoder completed beyond the prefix after teardown");
    Ok(())
}
