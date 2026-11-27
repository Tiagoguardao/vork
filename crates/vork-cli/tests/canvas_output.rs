use std::{fs, process::Command};

// A 1x1 opaque green RGB PNG.
const GREEN_PNG: &[u8] = &[
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
    0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x50, 0xf8, 0xaf, 0x00,
    0x00, 0x02, 0x82, 0x01, 0x40, 0x49, 0x67, 0x57, 0x5f, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e,
    0x44, 0xae, 0x42, 0x60, 0x82, 0x41,
];

#[test]
#[ignore = "requires a supported GPU and VORK_GPU_TESTS=1"]
fn refused_canvas_picture_preserves_completed_still() {
    assert_eq!(std::env::var("VORK_GPU_TESTS").as_deref(), Ok("1"));
    let root = std::env::temp_dir().join(format!("vork-canvas-output-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let write_scene = |fill: &str| {
        fs::write(
            root.join("scene.json"),
            format!(
                r#"{{"version":2,"composition":{{"space":"2d"}},
                    "render":{{"resolution":[32,32],"duration":1,"fps":1}},
                    "images":{{"source":{{"file":"source.png"}}}},
                    "nodes":[{{"type":"path","d":"M0 0H32V32H0Z","fill":{fill}}}]}}"#
            ),
        )
        .unwrap();
    };
    let render = |success: bool| {
        let result = Command::new(env!("CARGO_BIN_EXE_vork"))
            .current_dir(&root)
            .args(["render", "scene.json", "--out", "still.png"])
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert_eq!(result.status.success(), success, "{stderr}");
        if !success {
            assert!(
                stderr.contains("image decode") && stderr.contains("source.png"),
                "{stderr}"
            );
        }
    };
    let read = |name| fs::read(root.join(name)).unwrap_or_else(|error| panic!("{name}: {error}"));

    fs::write(root.join("source.png"), GREEN_PNG).unwrap();
    write_scene(r#""red""#);
    render(true);
    let pixels = read("still.png");
    let receipt = read("still.png.meta.json");

    // Retain IHDR and the start of IDAT, so the image has dimensions but cannot decode.
    fs::write(root.join("source.png"), &GREEN_PNG[..42]).unwrap();
    write_scene(r#"{"image":"source"}"#);
    render(false);
    assert_eq!(read("still.png"), pixels);
    assert_eq!(read("still.png.meta.json"), receipt);

    fs::write(root.join("source.png"), GREEN_PNG).unwrap();
    render(true);
    assert_ne!(read("still.png"), pixels);
    assert_ne!(read("still.png.meta.json"), receipt);
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "requires a supported GPU, pinned Slang and VORK_GPU_TESTS=1"]
fn scaled_path_matte_and_visible_paint_sample_the_same_pixel_neighbors() {
    use serde_json::json;
    assert_eq!(std::env::var("VORK_GPU_TESTS").as_deref(), Ok("1"));
    let root = std::env::temp_dir().join(format!("vork-matte-derivative-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let rect = json!({"type":"path", "name":"matte", "d":"M8 8H72V56H8Z",
                      "scale":[1.5,1.25,1], "fill":"red"});
    let mut scene = json!({
        "version":2, "composition":{"space":"2d", "viewBox":[0,0,128,96]},
        "render":{"resolution":[128,96], "background":"black"}, "nodes":[rect]
    });
    let render = |scene: &serde_json::Value| {
        fs::write(root.join("scene.json"), serde_json::to_vec(scene).unwrap()).unwrap();
        let run = Command::new(env!("CARGO_BIN_EXE_vork"))
            .current_dir(&root)
            .args(["render", "scene.json", "--out", "still.png"])
            .output().unwrap();
        assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
        fs::read(root.join("still.png")).unwrap()
    };
    let css = render(&scene);
    scene["programs"] = json!({"ink":{"source":
        "struct Inputs {}; float4 paint(PaintContext c, Inputs i) { float a = saturate(fwidth(floor(c.position.x * 1.5))); return float4(a,0,0,a); }"
    }});
    scene["materials"] = json!({"red":{"program":"ink"}});
    scene["nodes"][0]["fill"] = json!({"material":"red"});
    assert!(render(&scene) == css, "visible derivative differs from CSS");
    scene["nodes"][0]["mask-type"] = json!("alpha");
    scene["nodes"].as_array_mut().unwrap().push(json!({
        "type":"path", "d":"M0 0H128V96H0Z", "fill":"red", "mask":"matte"
    }));
    assert!(render(&scene) == css, "matte derivative differs from visible paint");
    fs::remove_dir_all(root).unwrap();
}
