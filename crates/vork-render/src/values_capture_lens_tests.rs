    use super::*;

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn imported_visible_keys_reach_objects_lights_and_shadow_reservations() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let mut gltf: serde_json::Value = vork_scene::v2::from_test_str(include_str!("../tests/goldens/skinned-box.gltf"))?;
        gltf["nodes"].as_array_mut().unwrap().push(serde_json::json!({"name":"parent","children":[0,3]}));
        gltf["nodes"].as_array_mut().unwrap().push(serde_json::json!({"name":"lamp","translation":[0,2,3],"extensions":{"KHR_lights_punctual":{"light":0}}}));
        gltf["scenes"][0]["nodes"] = serde_json::json!([1,2]);
        gltf["extensionsUsed"] = serde_json::json!(["KHR_lights_punctual"]);
        gltf["extensions"] = serde_json::json!({"KHR_lights_punctual":{"lights":[{"type":"point","intensity":100}]}});
        let file = std::env::temp_dir().join(format!("vork-visible-import-{}.gltf",std::process::id()));
        std::fs::write(&file,serde_json::to_vec(&gltf)?)?;
        let input = vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "render":{"resolution":[16,16],"settings":{"samples":1}},
            "materials":{"plain":{"inputs":{"base_color":[1,0,0],"emission_color":[1,0,0],"emission_luminance":1}}},
            "nodes":[{"type":"camera","name":"cam","translation":[0,0,4]},
                {"type":"asset","name":"actor","file":file,"materials":{"*":"plain"},"nodes":{"parent":{"visible":false}}}],
            "animation":[{"tracks":[{"path":["nodes","actor","nodes","parent","visible"],"keys":[[0,false],[1,true]]}]}]
        }))?;
        let installed = renderer.install(input,Path::new("."));
        std::fs::remove_file(&file)?;
        let mut installed = installed?;
        assert!(installed.plan.draws.iter().all(|draw| draw.casts_shadows));
        let mut frame = installed.create_frame_slot(&renderer,crate::ReadbackPrecision::Float)?;
        for (time,shown) in [(0.,false),(1.,true),(0.,false)] {
            let output = installed.render_frame(&renderer, &mut frame, time)?;
            assert_eq!(output.ids.iter().any(|id| *id != 0),shown,"imported visibility at {time}");
            assert!(!frame.evaluated_lights.is_empty());
            for light in &frame.evaluated_lights {
                assert_eq!(light.shadow_offsets[constants::LIGHT_CONTRIBUTION_WORD] & constants::RAY_VISIBILITY_ALL,
                    if shown {constants::RAY_VISIBILITY_ALL} else {0});
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn animated_visible_hides_environment_and_lights_through_parents() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let input = vork_scene::v2::from_test_value(serde_json::json!({
            "version":2,"render":{"resolution":[16,16],"duration":2,"settings":{"samples":1}},
            "nodes":[{"type":"camera","name":"cam"},
                {"type":"node","name":"rig","visible":false,"children":[
                    {"type":"light","name":"world","shape":"environment","color":[1,1,1]},
                    {"type":"light","name":"lamp","shape":"point","shadowed_by":[]}]}],
            "animation":[{"tracks":[{"path":["nodes","rig","visible"],"keys":[[0,false],[1,true]]}]}]
        }))?;
        let installed = renderer.install(input, Path::new("."))?;
        let mut frame = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
        let mut changes = LightingTracker::default();
        for (time, visible) in [(0.,false),(0.999,false),(1.,true),(0.5,false)] {
            let (_,_,_,uniform) = frame.prepare_values(&installed,&renderer,time,time,&mut changes)?;
            assert_eq!(uniform.environment_links[0] & constants::RAY_VISIBILITY_ALL,
                if visible {constants::RAY_VISIBILITY_ALL} else {0});
            assert_eq!(uniform.medium[3],u32::from(visible));
            assert!(!frame.evaluated_lights.is_empty());
            for light in &frame.evaluated_lights {
                assert_eq!(light.shadow_offsets[constants::LIGHT_CONTRIBUTION_WORD] & constants::RAY_VISIBILITY_ALL,
                    if visible {constants::RAY_VISIBILITY_ALL} else {0});
            }
        }
        Ok(())
    }
