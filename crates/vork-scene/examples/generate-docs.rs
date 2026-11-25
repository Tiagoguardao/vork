fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (path, text) in vork_scene::v2::documents() {
        if std::env::args().any(|arg| arg == "--check") {
            assert_eq!(
                std::fs::read_to_string(root.join(path)).unwrap(),
                text,
                "{path} is stale; run scripts/generate-docs"
            );
        } else {
            std::fs::write(root.join(path), text).unwrap();
        }
    }
}
