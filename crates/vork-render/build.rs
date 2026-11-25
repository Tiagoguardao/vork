#[path = "../vork-slang/src/registry.rs"]
#[allow(dead_code)]
mod registry;
#[path = "src/object_abi.rs"]
#[allow(dead_code)]
mod object_abi;
#[path = "src/binding_abi.rs"]
#[allow(dead_code)]
mod binding_abi;
use std::{fs, path::Path};
#[path = "build_slang.rs"]
mod slang;
fn hash_tree(path: &Path, hash: &mut blake3::Hasher) {
    let mut entries: Vec<_> = fs::read_dir(path)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            hash_tree(&path, hash);
        } else {
            let name = path.to_string_lossy();
            let bytes = fs::read(&path).unwrap();
            hash.update(&(name.len() as u64).to_le_bytes());
            hash.update(name.as_bytes());
            hash.update(&(bytes.len() as u64).to_le_bytes());
            hash.update(&bytes);
        }
    }
}
// Local builds may use installed headers. Distribution builds supply their pinned tree.
fn mac_ffmpeg(decoder: &mut cc::Build) -> Option<(Vec<std::path::PathBuf>, String)> {
    if let Ok(av) = pkg_config::Config::new().range_version("60".."64")
        .cargo_metadata(false).probe("libavformat")
    {
        let major = av.version.split('.').next().unwrap();
        // Local-only. Distribution builds bypass pkg-config and never embed this prefix.
        if let Ok(dir) = pkg_config::get_variable("libavformat", "libdir") {
            decoder.define("VORK_LOCAL_AV_LIBDIR", Some(format!("{dir:?}").as_str()));
            return Some((av.include_paths, format!("{major}:local-first")));
        }
        return Some((av.include_paths, major.to_owned()));
    }
    for prefix in ["/opt/homebrew/opt/ffmpeg", "/usr/local/opt/ffmpeg"] {
        let prefix = Path::new(prefix);
        let include = prefix.join("include");
        let header = include.join("libavformat/version_major.h");
        println!("cargo:rerun-if-changed={}", header.display());
        let Ok(version) = fs::read_to_string(header) else { continue };
        let major = version.lines().find_map(|line| {
            let mut words = line.split_whitespace();
            (words.next()? == "#define" && words.next()? == "LIBAVFORMAT_VERSION_MAJOR")
                .then(|| words.next()?.parse::<u32>().ok()).flatten()
        });
        if let Some(major) = major.filter(|v| (60..64).contains(v)) {
            return Some((vec![include], major.to_string()));
        }
    }
    None
}
fn build_hwdecode() {
    println!("cargo:rustc-check-cfg=cfg(vork_nvdecode)");
    println!("cargo:rustc-check-cfg=cfg(vork_vtdecode)");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rerun-if-changed=native/hwdecode");
        let mut decoder = cc::Build::new();
        println!("cargo:rerun-if-env-changed=VORK_MAC_DIST_AV_HEADERS");
        let avformat = if let Some(include) = std::env::var_os("VORK_MAC_DIST_AV_HEADERS") {
            let major = "63";
            decoder.include(include).define("VORK_DIST_AVFORMAT_MAJOR", major);
            major.to_owned()
        } else if let Some((includes, avformat)) = mac_ffmpeg(&mut decoder) {
            decoder.includes(includes);
            avformat
        } else {
            println!("cargo:warning=FFmpeg headers unavailable, using portable footage reader");
            return;
        };
        // The Metal compiler receives a string with no filesystem include path.
        let shader = include_str!("native/hwdecode/nv12.metal")
            .replace("#include \"nv12.h\"", include_str!("native/hwdecode/nv12.h"));
        fs::write(Path::new(&std::env::var_os("OUT_DIR").unwrap()).join("nv12.metal"), shader).unwrap();
        decoder.file("native/hwdecode/reference.c").file("native/hwdecode/decoder.c").compile("vork_vtdecode");
        cc::Build::new().flag("-fobjc-arc").files(["native/hwdecode/metal.m", "native/hwdecode/metal_vectors.m"]).compile("vork_vtplanes");
        for framework in ["Metal", "Foundation", "CoreVideo"] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
        println!("cargo:rustc-cfg=vork_vtdecode");
        println!("cargo:rustc-env=VORK_NATIVE_FOOTAGE_READER=videotoolbox:{avformat}");
        return;
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") { return; }
    println!("cargo:rerun-if-changed=../vork-encode/native/nvcodec");
    let dir = "native/hwdecode";
    println!("cargo:rerun-if-changed={dir}");
    // libav structures are versioned. A build without the tested headers keeps
    // the portable reader. The shared libraries are resolved at runtime.
    let Ok(av) = pkg_config::Config::new().range_version("60".."61")
        .cargo_metadata(false).probe("libavformat") else { return };
    let ptx = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("nv12.ptx");
    let result = std::process::Command::new("clang++").args([
        "-O2", "-x", "cuda", "--cuda-device-only", "-nocudainc", "-nocudalib",
        "--cuda-gpu-arch=sm_75", "-S", "native/hwdecode/nv12.cu", "-o",
    ]).arg(ptx).status();
    if !result.is_ok_and(|s| s.success()) { return; }
    cc::Build::new().includes(av.include_paths).file("native/hwdecode/reference.c").file("native/hwdecode/nvdec.c").compile("vork_nvdecode");
    println!("cargo:rustc-cfg=vork_nvdecode");
    println!("cargo:rustc-env=VORK_NATIVE_FOOTAGE_READER=nvdec:60");
}
fn main() {
    build_hwdecode();
    println!("cargo:rustc-env=VORK_BUILD_PROFILE={}", std::env::var("PROFILE").unwrap());
    // Explicit opt-in: directory layout must not change test admission.
    println!("cargo:rustc-check-cfg=cfg(vork_private_tests)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VORK_PRIVATE_TESTS");
    if std::env::var("VORK_PRIVATE_TESTS").as_deref() == Ok("1") {
        println!("cargo:rustc-cfg=vork_private_tests");
    }
    // Texture bytes depend on codecs, not unrelated rendering changes or GPU IDs.
    let mut textures = blake3::Hasher::new();
    for file in [
        "src/compressed_textures.rs",
        "src/gpu_bc7.rs",
        "src/constants.rs",
        "shaders/mip.slang",
        "shaders/bc7.slang",
        "shaders/bc7_refine.slangh",
        "../../Cargo.lock",
        "../../dist/slang.json",
    ] {
        println!("cargo:rerun-if-changed={file}");
        textures.update(&fs::read(file).unwrap());
    }
    hash_tree(Path::new("../../vendor/basisu_c_sys"), &mut textures);
    println!("cargo:rerun-if-changed=../../vendor/basisu_c_sys");
    hash_tree(Path::new("shaders/third_party/directxtex"), &mut textures);
    println!("cargo:rerun-if-changed=shaders/third_party/directxtex");
    println!(
        "cargo:rustc-env=VORK_TEXTURE_BUILD_ID={}",
        textures.finalize()
    );
    let mut hash = blake3::Hasher::new();
    // One conservative build identity covers every host generator, pipeline
    // descriptor and shader. Consumer code and Git history are not cache inputs.
    for tree in [
        "src",
        "shaders",
        "../vork-scene/src",
        "../vork-slang/src",
        "../vork-mesh/src",
        "../vork-std/src",
    ] {
        println!("cargo:rerun-if-changed={tree}");
        hash_tree(Path::new(tree), &mut hash);
    }
    let closure = std::env::var("DEP_VORK_OPENPBR_CLOSURE").unwrap();
    println!("cargo:rustc-env=VORK_CLOSURE_WGSL={closure}");
    let out = std::env::var("OUT_DIR").unwrap();
    fs::write(Path::new(&out).join("gpu_object.rs"), object_abi::rust()).unwrap();
    let closure_dir = Path::new(&closure).parent().unwrap();
    let (slang, shader_sources) = slang::compile(Path::new(&out), closure_dir);
    // Native Slang output is portable across devices and unrelated host builds.
    // Include shader generators as well as their inputs; pipeline descriptors and
    // scene loading do not participate in this artifact's identity.
    let mut shaders = blake3::Hasher::new();
    shaders.update(slang.as_bytes());
    shaders.update(shader_sources.as_bytes());
    for tree in ["shaders", "../vork-slang/src"] {
        hash_tree(Path::new(tree), &mut shaders);
    }
    for file in [
        closure.as_str(),
        "build_slang.rs",
        "src/slang_forward.rs",
        "src/programs.rs",
        "src/forward_variant.rs",
        "src/surface_native.rs",
        "src/camera_native.rs",
        "../../libraries/std/openpbr_core.slang",
    ] {
        let bytes = fs::read(file).unwrap();
        shaders.update(&(bytes.len() as u64).to_le_bytes());
        shaders.update(&bytes);
    }
    println!("cargo:rustc-env=VORK_SHADER_ID={}", shaders.finalize());
    hash.update(slang.as_bytes());
    for file in [
        closure.as_str(),
        "build.rs",
        "build_slang.rs",
        "../../libraries/std/openpbr_core.slang",
        "src/forward_variant.rs",
        "Cargo.toml",
        "../../Cargo.toml",
        "../../Cargo.lock",
        "../vork-mesh/Cargo.toml",
        "../vork-scene/Cargo.toml",
        "../vork-scene/build.rs",
        "../vork-slang/Cargo.toml",
    ] {
        println!("cargo:rerun-if-changed={file}");
        let bytes = fs::read(file).unwrap();
        hash.update(&(bytes.len() as u64).to_le_bytes());
        hash.update(&bytes);
    }
    // Reflected programs and generated native code depend on these sources,
    // not the host executable's target, optimization level or GPU adapter.
    println!("cargo:rustc-env=VORK_PORTABLE_ID={}", hash.finalize());
    for variable in ["TARGET", "OPT_LEVEL", "CARGO_ENCODED_RUSTFLAGS"] {
        println!("cargo:rerun-if-env-changed={variable}");
        let value = std::env::var(variable).unwrap_or_default();
        hash.update(&(value.len() as u64).to_le_bytes());
        hash.update(value.as_bytes());
    }
    println!("cargo:rustc-env=VORK_BUILD_ID={}", hash.finalize());

    // Export proof also covers local edits to the canvas, sound and output
    // implementations. These sources are separate from renderer cache inputs.
    for tree in [
        "native/hwdecode",
        "../moda-assets/src",
        "../moda-color/src",
        "../moda-engine/src",
        "../moda-plain-text/src",
        "../moda-scene/src",
        "../moda-shaders/src",
        "../moda-shaders/wgsl",
        "../moda-text/src",
        "../moda-text-draw/src",
        "../vork-audio/src",
        "../vork-encode/native",
        "../vork-encode/src",
        "../vork-media/src",
        "../vork-splat/src",
        "../vork-time/src",
    ] {
        println!("cargo:rerun-if-changed={tree}");
        hash_tree(Path::new(tree), &mut hash);
    }
    println!("cargo:rustc-env=VORK_EXPORT_BUILD_ID={}", hash.finalize());
}
