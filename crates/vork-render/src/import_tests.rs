    #[test]
    fn optional_node_instancing_warns_for_each_affected_node() {
        let collector = crate::warnings::Collector::default();
        let _scope = collector.enter();
        let root = std::env::temp_dir().join(format!("vork-optional-instancing-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let data: Vec<u8> = [-1f32,0.,0., 1.,0.,0., 0.,1.,0., -3.,0.,0., -1.,0.,0., 1.,0.,0., 3.,0.,0.]
            .into_iter().flat_map(f32::to_le_bytes).collect();
        std::fs::write(root.join("mesh.bin"), data).unwrap();
        let asset = serde_json::json!({"asset":{"version":"2.0"},"extensionsUsed":["EXT_mesh_gpu_instancing"],
            "buffers":[{"uri":"mesh.bin","byteLength":84}],
            "bufferViews":[{"buffer":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":48}],
            "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[-1,0,0],"max":[1,1,0]},
                {"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"}],
            "meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],
            "nodes":[{"name":"first","mesh":0,"extensions":{"EXT_mesh_gpu_instancing":{"attributes":{"TRANSLATION":1}}}},
                {"name":"second","mesh":0,"translation":[0,2,0],"scale":[2,1,1],"extensions":{"EXT_mesh_gpu_instancing":{"attributes":{"TRANSLATION":1}}}}],
            "scenes":[{"nodes":[0,1]}],"scene":0});
        let path = root.join("scene.gltf");
        std::fs::write(&path, serde_json::to_vec(&asset).unwrap()).unwrap();
        super::open(&path).unwrap();
        std::fs::remove_dir_all(root).unwrap();
        let warnings = collector.snapshot();
        assert_eq!(warnings.len(), 2);
        assert!(warnings.iter().all(|warning| warning.code == "gltf_instancing" && warning.dropped.contains("Export separate mesh nodes")));
    }
    #[test]
    fn cyclic_node_hierarchies_are_errors() {
        for nodes in [
            serde_json::json!([{"children":[0]}]),
            serde_json::json!([{"children":[1]},{"children":[2]},{"children":[0]}]),
            serde_json::json!([{}, {"children":[1]}]),
        ] {
            let bytes = serde_json::to_vec(&serde_json::json!({"asset":{"version":"2.0"},"nodes":nodes})).unwrap();
            let error = super::from_slice(&bytes).unwrap_err().to_string();
            assert!(error.contains("node hierarchy contains a cycle"), "{error}");
        }
        super::from_slice(br#"{"asset":{"version":"2.0"},"nodes":[{"children":[1,2]},{"children":[3]},{"children":[3]},{}]}"#).unwrap();
    }

    #[test]
    fn geometry_admission_respects_storage_before_allocation_limits() {
        let limits = wgpu::Limits {max_buffer_size: 256 << 20,
            max_storage_buffer_binding_size: 128 << 20, ..Default::default()};
        let limit = u64::from(limits.max_storage_buffer_binding_size);
        assert_eq!(super::admit_geometry_streams(&limits, [limit; 3]).unwrap(), limit);
        // The random-scene regression fits a device allocation but cannot be
        // bound for hit reconstruction. Refuse before creating GPU resources.
        for stream in 0..3 {
            let mut bytes = [16; 3]; bytes[stream] = 181_678_560;
            let error = super::admit_geometry_streams(&limits, bytes).unwrap_err().to_string();
            assert!(error.contains(["vertices", "attributes", "indices"][stream])
                && error.contains("181678560") && error.contains("134217728"), "{error}");
        }
        let generous = wgpu::Limits {max_storage_buffer_binding_size: 256 << 20, ..limits.clone()};
        super::admit_geometry_streams(&generous, [181_678_560; 3]).unwrap();
        let small = wgpu::Limits {max_buffer_size: 64 << 20, ..limits};
        assert_eq!(super::admit_geometry_streams(&small, [16; 3]).unwrap(), 64 << 20);
    }
    use super::*;

    #[test]
    fn unsupported_asset_formats_report_the_format_before_parsing() {
        for path in ["character.fbx", "character.FBX", "mesh.obj", "external.usdz"] {
            let error = format!("{:#}", open(Path::new(path)).unwrap_err());
            assert!(error.contains("unsupported asset format"), "{error}");
            assert!(error.contains("glTF/GLB, SPZ and Gaussian PLY"), "{error}");
        }
        for bytes in [b"Kaydara FBX Binary  \0\x1a\0".as_slice(), b"  ; FBX 7.4.0 project file"] {
            let error = format!("{:#}", from_slice(bytes).unwrap_err());
            assert!(error.contains("unsupported asset format FBX"), "{error}");
        }
        assert!(format!("{:#}", from_slice(b"{bad gltf json").unwrap_err()).contains("key must be a string"));
    }

    fn glb(json: &[u8], length: u32) -> Vec<u8> {
        let mut bytes = b"glTF".to_vec();
        bytes.extend(2u32.to_le_bytes());
        bytes.extend(length.to_le_bytes());
        bytes.extend((json.len() as u32).to_le_bytes());
        bytes.extend(b"JSON");
        bytes.extend(json);
        bytes
    }

    /// QA fuzz reproducers. Each panicked in gltf-json's validation, gltf's
    /// unwraps and `unreachable!` type matches, or vork's bufferView slice; each
    /// is now an error naming the problem.
    #[test]
    #[cfg(vork_private_tests)]
    fn malformed_documents_are_errors() {
        let seeds = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-fuzz/seeds/gltf");
        for (seed, cause) in [
            ("position-accessor-out-of-range.gltf", "the glTF parser failed on malformed input (upstream panic at gltf-json-1.4.1/src/mesh.rs:151"),
            ("image-without-source.gltf", "glTF image 0 has neither a uri nor a bufferView"),
            ("image-view-without-mime-type.glb", "glTF image 0 is stored in a bufferView without the required mimeType"),
            ("image-view-past-buffer.glb", "glTF bufferView 5 (172 bytes at offset 574) runs past buffer 0 (272 bytes)"),
            ("buffer-uri-not-utf8.gltf", "the glTF buffer reader failed on malformed input (upstream panic at gltf-1.4.1/src/import.rs:50"),
            ("index-accessor-type.gltf", "indices: accessor 2 is Vec2 F32; glTF allows [Scalar] of [U8, U16, U32]"),
            ("texcoord-accessor-type.gltf", "TexCoords(0): accessor 2 is Vec2 U32; glTF allows [Vec2] of [U8, U16, F32]"),
            ("accessor-element-size.glb", "animation 0 channel 0 input: accessor 4 is Vec3 F32; glTF allows [Scalar] of [F32]"),
        ] {
            let error = format!("{:#}", crate::fuzz::gltf(&seeds.join(seed)).expect_err(seed));
            assert!(error.contains(cause), "{seed}: {error}");
        }
    }

    #[test]
    fn short_accessor_stride_is_an_error() {
        let bytes=br#"{"asset":{"version":"2.0"},"buffers":[{"byteLength":12,"uri":"data:application/octet-stream;base64,AAAAAAAAAAAAAAAA"}],"bufferViews":[{"buffer":0,"byteLength":12,"byteStride":4}],"accessors":[{"bufferView":0,"componentType":5126,"count":1,"type":"VEC3","min":[0,0,0],"max":[0,0,0]}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}]}
"#;
        let error=from_slice(bytes).unwrap_err().to_string();
        assert!(error.contains("byteStride 4 is smaller than its 12-byte element"),"{error}");
        let bytes=br#"{"asset":{"version":"2.0"},"buffers":[{"byteLength":16,"uri":"data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAAAAAA=="}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":1},{"buffer":0,"byteOffset":4,"byteLength":12,"byteStride":4}],"accessors":[{"componentType":5126,"count":1,"type":"VEC3","min":[0,0,0],"max":[0,0,0],"sparse":{"count":1,"indices":{"bufferView":0,"componentType":5121},"values":{"bufferView":1}}}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}]}
"#;
        let error=from_slice(bytes).unwrap_err().to_string();
        assert!(error.contains("sparse views must not declare byteStride"),"{error}");
    }

    #[test]
    fn malformed_glb_lengths_are_errors() {
        // Fuzz reproducers: a declared length below the header underflows gltf's `length - 12`
        // in Glb::from_reader (version 2) and Glb::from_slice (version 0).
        for bytes in [*b"glTF\x02\0\0\0\0\0\0\0", *b"glTF\0\0\0\0\0\0\0\0"] {
            let error = from_slice(&bytes).unwrap_err().to_string();
            assert!(error.contains("GLB header length 0"), "{error}");
        }
        let json = br#"{"asset":{"version":"2.0"}}     "#;
        let length = 20 + json.len() as u32;
        let error = from_slice(&glb(json, u32::MAX)).unwrap_err().to_string();
        assert!(error.contains("GLB header length"), "{error}");
        let mut chunk_past_end = glb(json, length);
        chunk_past_end[12] += 4;
        let error = from_slice(&chunk_past_end).unwrap_err().to_string();
        assert!(error.contains("GLB chunk at byte 12"), "{error}");
        let error = from_slice(&glb(json, 16)[..16]).unwrap_err().to_string();
        assert!(error.contains("GLB chunk header at byte 12"), "{error}");
        // A chunk appended past the declared length is not read.
        let mut appended = glb(json, length);
        appended.extend(4u32.to_le_bytes());
        appended.extend(b"BIN\0");
        appended.extend([0; 4]);
        let error = from_slice(&appended).unwrap_err().to_string();
        assert!(error.contains("GLB header length"), "{error}");
        assert!(from_slice(&glb(json, length)).unwrap().blob.is_none());
    }

    #[test]
    fn gltf_json_accepts_a_leading_byte_order_mark() {
        let json = br#"{"asset":{"version":"2.0"},"nodes":[{"name":"a"}]}"#;
        let document = from_slice(&[b"\xEF\xBB\xBF".as_slice(), json].concat()).unwrap().document;
        assert_eq!(document.nodes().next().unwrap().name(), Some("a"));
        // Binary glTF has no byte-order mark, valid or with a malformed length.
        let json = br#"{"asset":{"version":"2.0"}}     "#;
        for length in [20 + json.len() as u32, 12] {
            let error = from_slice(&[b"\xEF\xBB\xBF".as_slice(), &glb(json, length)].concat()).unwrap_err().to_string();
            assert!(error.contains("byte-order mark precedes binary glTF"), "{error}");
        }
    }
