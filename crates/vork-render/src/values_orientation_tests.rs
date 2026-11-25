    use super::*;
    #[test]
    fn change_keys_detect_length_changes_and_equal_prefixes() {
        let mut changes = super::LightingTracker::default();
        let key = super::LightingKey::Probes;
        assert!(changes.changed(key.clone(), &[1, 2, 3, 4]));
        assert!(!changes.changed(key.clone(), &[1, 2, 3, 4]));
        assert!(changes.changed(key.clone(), &[1, 2]));
        assert!(!changes.changed(key.clone(), &[1, 2]));
        assert!(changes.changed(key, &[1, 2, 3, 4]));
    }

    #[test]
    fn camera_view_accepts_zero_and_small_nonzero_scale() {
        for scale in [0.,1e-8] {
            let world=Mat4::from_scale(Vec3::new(1.,1.,scale));
            let nodes=vec![EvalNode { camera: None,node:vork_scene::v2::from_test_value(serde_json::json!({"type":"camera","scale":[1.,1.,scale]})).unwrap(),
                world,parent:None,dynamic:false,embedded:None}];
            let view=node_view(0,&nodes,&[world]).unwrap();
            assert!(view.is_finite());
            assert!(view.abs_diff_eq(Mat4::IDENTITY,1e-6));
        }
    }

    #[test]
    fn zero_scale_light_retains_target_orientation() {
        let node: Node = vork_scene::v2::from_test_value(serde_json::json!({
            "type":"light","shape":"sun","scale":[0,0,0],"target":[1,0,0]})).unwrap();
        let world = node.transform.matrix();
        let nodes = vec![EvalNode { camera: None,node,world,parent:None,dynamic:false,embedded:None}];
        let frame = node_view(0,&nodes,&[world]).unwrap().inverse();
        assert!(frame.is_finite());
        assert!(frame.transform_vector3(Vec3::NEG_Z).abs_diff_eq(Vec3::X,1e-6));
    }

    #[test]
    fn rounded_view_axis_keeps_unit_length_when_only_magnitude_is_quantized() {
        let direction = Vec3::ONE.normalize();
        let eye = Vec3::splat(1_048_576.);
        let axis = view_axis(eye, direction);
        assert!(axis.abs_diff_eq(direction, 1e-6));
        let view = Mat4::look_to_rh(eye, axis, Vec3::Y);
        for basis in [Vec3::X, Vec3::Y, Vec3::Z] {
            assert!((view.transform_vector3(basis).length() - 1.).abs() < 1e-6);
        }
    }

    #[test]
    fn ordinary_camera_view_retains_the_shadow_fit_frame() {
        let eye = Vec3::new(7., 5., 11.);
        let rotation = glam::Quat::from_xyzw(-0.14237286, 0.27649665, 0.041459076, 0.9495055);
        let world = Mat4::from_rotation_translation(rotation, eye);
        let nodes = vec![EvalNode { camera: None, node: vork_scene::v2::from_test_value(serde_json::json!({"type":"camera"})).unwrap(),
            world, parent: None, dynamic: false, embedded: None }];
        let target = eye + world.transform_vector3(Vec3::NEG_Z).normalize();
        let expected = Mat4::look_at_rh(eye, target, world.transform_vector3(Vec3::Y).normalize());
        assert_eq!(node_view(0, &nodes, &[world]).unwrap(), expected);
    }

    #[test]
    fn camera_orientation_is_independent_of_world_translation() {
        let rotation=glam::Quat::from_rotation_y(0.3)*glam::Quat::from_rotation_z(0.2);
        for offset in [0.,10_000_000.,100_000_000.] {
            let world=Mat4::from_rotation_translation(rotation,Vec3::splat(offset));
            let nodes=vec![EvalNode { camera: None,node:vork_scene::v2::from_test_value(serde_json::json!({"type":"camera"})).unwrap(),
                world,parent:None,dynamic:false,embedded:None}];
            let view=node_view(0,&nodes,&[world]).unwrap();
            for axis in [Vec3::X,Vec3::Y,Vec3::Z] {
                assert!(view.transform_vector3(rotation*axis).abs_diff_eq(axis,1e-6));
            }
        }
    }

    #[test]
    fn rotation_roll_and_parent_space_transform() {
        let node: Node = vork_scene::v2::from_test_value(serde_json::json!({
            "type": "camera", "rotation": [0.,0.,0.70710677,0.70710677]
        }))
        .unwrap();
        let world = node.transform.matrix();
        let mut nodes = vec![EvalNode { camera: None,
            node,
            world,
            parent: None,
            dynamic: false,
            embedded: None,
        }];
        let rolled = node_view(0, &nodes, &[world]).unwrap().inverse();
        assert!(rolled.transform_vector3(Vec3::X).abs_diff_eq(Vec3::Y, 1e-5));
        assert!(
            rolled
                .transform_vector3(Vec3::NEG_Z)
                .abs_diff_eq(Vec3::NEG_Z, 1e-5)
        );
        // Camera orientation composes with the parent transform.
        nodes[0].node.transform.rotation =
            Some(glam::Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2).to_array());
        nodes[0].parent = Some(1);
        let parent = Mat4::from_translation(Vec3::new(5., 2., 3.));
        nodes.push(EvalNode { camera: None,
            node: vork_scene::v2::from_test_value(serde_json::json!({"type":"node","children":[]})).unwrap(),
            world: parent,
            parent: None,
            dynamic: false,
            embedded: None,
        });
        let aimed_world = node_world(&nodes[0].node, parent).unwrap();
        let aimed = node_view(0, &nodes, &[aimed_world, parent])
            .unwrap()
            .inverse();
        assert!(
            aimed
                .transform_vector3(Vec3::NEG_Z)
                .abs_diff_eq(Vec3::X, 1e-5)
        );
    }
