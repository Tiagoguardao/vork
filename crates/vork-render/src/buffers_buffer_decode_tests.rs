    use super::*;
    #[test]
    fn shared_sidecar_decodes_only_each_authored_range() -> Result<()> {
        let file =
            std::env::temp_dir().join(format!("vork-attribute-shared-{}.bin", std::process::id()));
        let mut bytes = vec![0xcc; 16];
        for value in [1.25_f32, -2., 3.5] {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend(0xffff_ffff_u32.to_le_bytes());
        bytes.extend(17_u32.to_le_bytes());
        bytes.push(0xcc); // Explicit counts must ignore unrelated trailing bytes.
        std::fs::write(&file, bytes)?;
        let decode = |element_type: &str, offset, count| -> Result<AttributeData> {
            AttributeData::decode(
                &vork_scene::v2::from_test_value(json!({"type":element_type,
                "file":&file,"offset":offset,"count":count}))?,
                Path::new("."),
            )
        };
        assert_eq!(
            decode("vec3", 16, 1)?.floats::<3>()?,
            vec![[1.25, -2., 3.5]]
        );
        assert_eq!(decode("u32", 28, 2)?.words, vec![u32::MAX, 17]);
        std::fs::remove_file(file)?;
        Ok(())
    }

    #[test]
    fn integer_bits_and_nested_inline_values_match_scene_validation() -> Result<()> {
        let buffer = vork_scene::v2::from_test_value(json!({"type":"i32","data":[[-1]]}))?;
        assert_eq!(
            AttributeData::decode(&buffer, Path::new("."))?.words,
            vec![u32::MAX]
        );
        let buffer = vork_scene::v2::from_test_value(
            json!({"type":"mat4","data":[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]}),
        )?;
        assert_eq!(AttributeData::decode(&buffer, Path::new("."))?.count(), 1);
        Ok(())
    }
    #[test]
    fn inline_scratch_is_cleared_between_differently_nested_rows() -> Result<()> {
        let buffer = vork_scene::v2::from_test_value(json!({"type":"vec3","data":[
            [1,2,3], [[4],[5,6]], [[],7,[8,9]], [10,[11,[12]]]
        ]}))?;
        let actual = AttributeData::decode(&buffer, Path::new("."))?;
        assert_eq!(actual.words, (1..=12).map(|n| (n as f32).to_bits()).collect::<Vec<_>>());
        for row in [json!([1,2]), json!([1,2,3,4]), json!([1,"bad",3])] {
            let buffer = vork_scene::v2::from_test_value(json!({"type":"vec3","data":[[1,2,3],row]}))?;
            assert!(AttributeData::decode(&buffer, Path::new(".")).is_err());
        }
        Ok(())
    }

    #[test]
    fn bb107_sdk_written_implicit_field_loads() -> Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bb107/written.json");
        let scene = vork_scene::v2::Scene::load(&path)?;
        let buffers = SceneBuffers::decode(&scene, path.parent().unwrap())?;
        assert_eq!(buffers.data["field"].words, (0..1024).map(|i| ((i % 16) as f32 - 7.5).to_bits()).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn dense_binary_cells_use_dimensions_and_file_range() -> Result<()> {
        let path = std::env::temp_dir().join(format!("vork-dense-bb107-{}.bin", std::process::id()));
        let cells: Vec<f32> = (0..1024).map(|i| (i % 16) as f32 - 7.5).collect();
        let mut bytes = vec![0u8; 16];
        bytes.extend(cells.iter().flat_map(|v| v.to_le_bytes()));
        bytes.extend([0u8; 4]);
        std::fs::write(&path, &bytes)?;
        for count in [None, Some(1024)] {
            let buffer = vork_scene::v2::from_test_value(json!({"type":"f32", "dims":[16,8,8], "file":path, "offset":16, "count":count}))?;
            assert_eq!(AttributeData::decode(&buffer, Path::new("."))?.words,
                cells.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        std::fs::write(&path, &bytes[..bytes.len()-8])?;
        let buffer = vork_scene::v2::from_test_value(json!({"type":"f32", "dims":[16,8,8], "file":path, "offset":16}))?;
        assert!(AttributeData::decode(&buffer, Path::new(".")).err().unwrap().to_string().contains("too short"));
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn binary_ranges_partial_elements_and_nonfinite_values_are_rejected() -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("vork-attribute-decode-{}.bin", std::process::id()));
        std::fs::write(&path, [0u8; 4])?;
        let decode = |ty: &str, offset: u64, count: Option<u32>| -> Result<AttributeData> {
            AttributeData::decode(
                &vork_scene::v2::from_test_value(
                    json!({"type":ty,"file":path,"offset":offset,"count":count}),
                )?,
                Path::new("."),
            )
        };
        assert!(decode("u32", 8, None).is_err());
        assert!(decode("u32", 0, Some(2)).is_err());
        assert!(decode("vec2", 0, None).is_err());
        std::fs::write(&path, f32::NAN.to_le_bytes())?;
        assert!(decode("f32", 0, None).is_err());
        assert_eq!(decode("u32", 0, None)?.words, vec![f32::NAN.to_bits()]);
        std::fs::remove_file(path)?;
        Ok(())
    }
