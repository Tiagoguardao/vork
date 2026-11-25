    use super::*;
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn installation_uses_animated_poses_and_reports_evaluated_coincidence() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let document = serde_json::json!({"version":2,
            "render":{"resolution":[8,8]},
            "buffers":{"p":{"type":"vec3","data":[[0,0,0],[1,0,0],[0,1,0]]}},
            "geometry":{"triangle":{"attributes":{"POSITION":"p"}}},
            "nodes":[{"type":"node","name":"rig","children":[
                {"type":"camera","name":"cam","target":"subject"},
                {"type":"mesh","name":"aimed","geometry":"triangle","target":"subject"}]},
                {"type":"node","name":"subject"}],
            "animation":[{"tracks":[
                {"path":["nodes","rig","translation"],"keys":[[0,[0,0,3]],[1,[0,0,4]]]},
                {"path":["nodes","subject","translation"],"keys":[[0,[1,0,0]],[1,[2,0,0]]]}]}]});
        let installed = renderer.install(vork_scene::v2::from_test_value(document.clone())?, Path::new("."))?;
        let mut values = offset_storage(&installed);
        sample_frame(&installed, 0., &mut values.samples, &mut values.nodes, &mut values.worlds)?;
        for (node, world) in installed.nodes.iter().zip(&values.worlds) {
            assert!(node.world.abs_diff_eq(*world, 1e-6));
        }
        for (root, object) in installed.roots.iter().zip(&installed.objects) {
            assert!(object.model.abs_diff_eq(root.world(&values.worlds), 1e-6));
        }
        let mut coincident = document;
        coincident["animation"][0]["tracks"][0]["keys"][0][1] = serde_json::json!([1,0,0]);
        let error = renderer.install(vork_scene::v2::from_test_value(coincident)?, Path::new(".")).err().expect("evaluated coincidence");
        assert!(error.to_string().contains("node \"cam\""), "{error:#}");
        assert!(error.to_string().contains("sample time 0"), "{error:#}");
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn keyed_camera_uses_requested_initial_time_and_checks_sampled_poses() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let document = serde_json::json!({"version":2,
            "render":{"resolution":[8,8],"fps":1,"duration":2},
            "nodes":[{"type":"camera","name":"cam","target":"subject"},
                {"type":"node","name":"subject"}],
            "animation":[{"tracks":[{"path":["nodes","cam","translation"],
                "keys":[[0,[0,0,0]],[1,[0,0,3]]]}]}]});
        let installed = renderer.install_with_options(vork_scene::v2::from_test_value(document.clone())?, Path::new("."),
            InstallOptions { initial_time: 1_f32.into(), ..Default::default() })?;
        assert!(installed.nodes[installed.camera].world.w_axis.truncate().abs_diff_eq(Vec3::new(0.,0.,3.), 1e-6));
        // Repeated seeking must sample authoring defaults rather than the installed pose.
        let mut values = offset_storage(&installed);
        for time in [0.5,1.,0.25] {
            sample_frame(&installed, time, &mut values.samples, &mut values.nodes, &mut values.worlds)?;
            assert!((values.worlds[installed.camera].w_axis.z - 3.*time).abs() < 1e-6);
        }
        let mut valid = document.clone();
        valid["animation"][0]["tracks"][0]["keys"][0][1] = serde_json::json!([0,0,3]);
        let mut installed = renderer.install(vork_scene::v2::from_test_value(valid.clone())?, Path::new("."))?;
        let mut slot = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
        installed.render_frame(&renderer, &mut slot, 0.)?;
        valid["render"]["motion_blur"] = serde_json::json!({"shutter":1,"position":"start"});
        valid["render"]["settings"] = serde_json::json!({"samples":1});
        valid["animation"][0]["tracks"][0]["keys"][1][1] = serde_json::json!([0,0,0]);
        let mut installed = renderer.install(vork_scene::v2::from_test_value(valid)?, Path::new("."))?;
        let mut slot = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
        // Sampled motion blur does not evaluate unused shutter endpoints.
        installed.render_frame(&renderer, &mut slot, 0.)?;
        let error = installed.render_frame(&renderer, &mut slot, 1.).err().expect("coincident sampled pose");
        // The frame-abort context wraps the pose error; the cause names the node
        // and the one evaluated shutter sample: the middle of [1, 2] at 1 fps.
        let chain = format!("{error:#}");
        assert!(chain.ends_with("node \"cam\": target coincides with the node origin at sample time 1.5"), "{chain}");
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn offset_instances_use_aimed_worlds_and_shifted_parents() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let mut input: serde_json::Value = vork_scene::v2::from_test_str(include_str!("../tests/fixtures/analytic.json"))?;
        input["render"] = serde_json::json!({"resolution":[8,8],"duration":3,"settings":{"samples":1}});
        input["buffers"]["translations"] = serde_json::json!({"type":"vec3","data":[[0,0,0],[0,0,0]]});
        input["buffers"]["offsets"] = serde_json::json!({"type":"f32","data":[0,1]});
        input["nodes"] = serde_json::json!([
            {"type":"camera","translation":[0,0,8]},
            {"type":"node","name":"parent","children":[
                {"type":"mesh","name":"aimed","geometry":"shape_box","target":[4,0,-2],
                 "instances":{"attributes":{"TRANSLATION":"translations","time_offset":"offsets"}}}]}]);
        input["animation"] = serde_json::json!([{"tracks":[
            {"path":["nodes","parent","translation"],"keys":[[0,[0,0,0]],[2,[2,0,0]]]}
        ]}]);
        let base=Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let mut installed=renderer.install(vork_scene::v2::from_test_value(input)?,&base)?;
        assert_eq!(installed.roots.len(),2);
        let mut frame=installed.create_frame_slot(&renderer,crate::ReadbackPrecision::Float)?;
        for time in [0.,1.,0.25] {
            installed.render_frame(&renderer, &mut frame, time)?;
            for (object,root) in installed.roots.iter().enumerate() {
                let eye=Vec3::new(time+root.time_offset,0.,0.);
                let expected=Mat4::look_to_rh(eye,(Vec3::new(4.,0.,-2.)-eye).normalize(),Vec3::Y).inverse();
                assert!(frame.object_values[object].model.abs_diff_eq(expected,1e-5),"time {time}, offset {}",root.time_offset);
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn additive_rotation_layers_compose_after_target_and_move_children() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let input=vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "render":{"resolution":[8,8],"duration":2},
            "nodes":[{"type":"camera","name":"cam","translation":[0,0,6]},
                {"type":"node","name":"aimed","target":"subject","rotation":{"euler":[0,0,0.2]},
                    "children":[{"type":"node","name":"child","translation":[0,0,-2]}]},
                {"type":"node","name":"subject","translation":[0,0,-3]}],
            "animation":[{"tracks":[{"path":["nodes","subject","translation"],"keys":[[0,[0,0,-3]],[1,[2,0,-3]]]}]},
                {"blend":"combine","tracks":[{"path":["nodes","aimed","rotation"],"keys":[[0,{"euler":[0,0,0]}],[1,{"euler":[0,0.7,0]}]]}]}]
        }))?;
        let installed=renderer.install(input,Path::new("."))?;
        let index=|name:&str| installed.nodes.iter().position(|node|node.node.name==name).unwrap();
        let mut values=offset_storage(&installed);
        for time in [0.,0.25,1.,0.5] {
            sample_frame(&installed,time,&mut values.samples,&mut values.nodes,&mut values.worlds)?;
            let target=Vec3::new(2.*time,0.,-3.);
            let expected=Mat4::look_to_rh(Vec3::ZERO,target.normalize(),Vec3::Y).inverse()
                * Mat4::from_rotation_z(0.2) * Mat4::from_rotation_y(0.7*time);
            assert!(values.worlds[index("aimed")].abs_diff_eq(expected,1e-5),"aim at {time}");
            assert!(values.worlds[index("child")].w_axis.truncate().abs_diff_eq(expected.transform_point3(Vec3::new(0.,0.,-2.)),1e-5),"child at {time}");
        }
        Ok(())
    }
    // Aim and focus follow animated targets at every evaluated time, including
    // shutter sub-frame times: keyed targets and targets under an animated parent.
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn aim_and_focus_follow_animated_targets_at_any_time() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let base = std::path::Path::new(".");
        let scene = serde_json::json!({"version":2,
            "render":{"resolution":[16,16],"fps":24,"duration":2,"motion_blur":{"shutter":0.5},"settings":{"samples":2}},
            "nodes":[
                {"type":"camera","name":"cam","target":"subject","translation":[0,1,6],"focus_object":"held","material":{"program":"std/camera","inputs":{"use_dof":true}}},
                {"type":"light","name":"spot","shape":"spot","target":"subject","translation":[3,4,2]},
                {"type":"light","name":"key","shape":"sun","target":"held"},
                {"type":"node","name":"rig","children":[{"type":"node","name":"held","translation":[0,0,-1]}]},
                {"type":"node","name":"subject"}],
            "animation":[
                {"tracks":[{"path":["nodes","subject","translation"],"keys":[[0,[-2,0,0]],[2,[2,1,-3]]]},
                    {"path":["nodes","rig","rotation"],"keys":[[0,[0,0,0,1]],[2,[0,0.7071068,0,0.7071068]]]},
                    {"path":["nodes","rig","translation"],"keys":[[0,[0,5,0]],[2,[4,2,1]]]}]}]});
        let s = renderer.install(vork_scene::v2::from_test_value(scene)?, &base)?;
        let index = |name: &str| s.nodes.iter().position(|n| n.node.name == name).unwrap();
        let (cam, subject, spot, key, held) = (index("cam"), index("subject"), index("spot"), index("key"), index("held"));
        let mut values = offset_storage(&s);
        let mut targets = Vec::new();
        // Frame and sub-frame (shutter) times.
        for time in [0., 0.3, 0.5 + 1. / 96., 1.13, 1.75] {
            sample_frame(&s, time, &mut values.samples, &mut values.nodes, &mut values.worlds)?;
            let worlds = &values.worlds;
            let aims = |from: usize, to: usize| {
                let eye = worlds[from].w_axis.truncate();
                let forward = worlds[from].transform_vector3(Vec3::NEG_Z).normalize();
                forward.dot((worlds[to].w_axis.truncate() - eye).normalize())
            };
            assert!(aims(cam, subject) > 0.99999, "camera at {time}: {}", aims(cam, subject));
            assert!(aims(spot, subject) > 0.99999, "spot at {time}: {}", aims(spot, subject));
            assert!(aims(key, held) > 0.99999, "sun at {time}: {}", aims(key, held));
            let dof = values.nodes[cam].as_ref().unwrap().camera.as_ref().unwrap();
            assert!((dof.focus - crate::targets::focus_distance(worlds[cam], worlds[held])).abs() < 1e-5, "focus at {time}");
            targets.push((worlds[subject].w_axis.truncate(), worlds[held].w_axis.truncate()));
        }
        assert!(targets.windows(2).all(|pair| pair[0].0.distance(pair[1].0) > 1e-4 && pair[0].1.distance(pair[1].1) > 1e-4), "{targets:?}");
        Ok(())
    }
