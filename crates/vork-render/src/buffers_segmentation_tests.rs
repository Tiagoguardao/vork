    use super::*;
    fn eval(name: &str, parent: Option<usize>) -> EvalNode {
        let node: Node = vork_scene::v2::from_test_value(json!({"type":"node","name":name})).unwrap();
        EvalNode { camera: None,
            node,
            parent,
            world: Mat4::IDENTITY,
            dynamic: false,
            embedded: None,
        }
    }
    #[test]
    fn imported_masks_keep_siblings_independent_after_grouping() -> Result<()> {
        let overrides = vork_scene::v2::from_test_value(json!({
            "Wheel":{"visible_camera":false},
            "#3":{"visible_glossy":false}
        }))?;
        let wheel = imported_visibility_mask(&overrides, &[(0, "Car".into()), (2, "Wheel".into())]);
        let body = imported_visibility_mask(&overrides, &[(0, "Car".into()), (3, "Body".into())]);
        assert_eq!(wheel & constants::RAY_CAMERA, 0);
        assert_ne!(wheel & constants::RAY_SPECULAR, 0);
        assert_ne!(body & constants::RAY_CAMERA, 0);
        assert_eq!(body & constants::RAY_SPECULAR, 0);
        assert_eq!(
            imported_visibility_mask(&overrides, &[(4, "Other".into())]),
            constants::RAY_VISIBILITY_ALL
        );
        Ok(())
    }

    /// The installed preorder of an authored tree.
    fn installed(input: &::vork_scene::v2::Scene) -> Vec<EvalNode> {
        fn visit(nodes: &[Node], parent: Option<usize>, out: &mut Vec<EvalNode>) {
            for node in nodes {
                let index = out.len();
                out.push(EvalNode { camera: None,
                    node: node.without_children(),
                    parent,
                    world: Mat4::IDENTITY,
                    dynamic: false,
                    embedded: None,
                });
                visit(&node.children, Some(index), out);
            }
        }
        let mut out = Vec::new();
        visit(&input.nodes, None, &mut out);
        out
    }
    #[test]
    fn immutable_point_ids_are_unique_for_inline_and_file_buffers() -> Result<()> {
        let directory=std::env::temp_dir().join(format!("vork-point-ids-{}",std::process::id()));
        std::fs::create_dir_all(&directory)?;
        std::fs::write(directory.join("ids.bin"),bytemuck::cast_slice(&[7u32,7]))?;
        for ids in [json!({"type":"u32","data":[7,7]}),json!({"type":"u32","file":"ids.bin","count":2})] {
            let mut value=json!({"version":2,"nodes":[{"type":"camera"}],"buffers":{
                "p":{"type":"vec3","data":[[0,0,0],[1,0,0]]},"w":{"type":"f32","data":[1,1]},"ids":ids},
                "geometry":{"points":{"topology":"points","attributes":{"POSITION":"p"},"widths":"w","ids":"ids"}}});
            let scene=crate::test_scene(value.clone(),&directory)?;
            let error=SceneBuffers::decode(&scene,&directory).err().context("duplicate immutable point IDs admitted")?;
            assert!(format!("{error:#}").contains("duplicate point identity 7"));
            value["buffers"]["ids"]=json!({"type":"u32","data":[7,8]});
            SceneBuffers::decode(&crate::test_scene(value,&directory)?,&directory)?;
        }
        std::fs::remove_dir_all(&directory)?;
        Ok(())
    }

    fn attributes(ids: Vec<u32>, transforms: usize) -> SceneBuffers {
        let mut data = std::collections::BTreeMap::new();
        data.insert(
            "ids".into(),
            AttributeData {
                element_type: ::vork_scene::v2::BufferType::U32,
                width: 1,
                words: ids,
            },
        );
        data.insert(
            "transforms".into(),
            AttributeData {
                element_type: ::vork_scene::v2::BufferType::Vec3,
                width: 3,
                words: vec![0; transforms * 3],
            },
        );
        SceneBuffers { data, grids: Default::default() }
    }
    #[test]
    fn visibility_is_anded_through_the_installed_parent_chain() {
        let mut parent = eval("parent", None);
        parent.node.visibility.camera = false;
        parent.node.visibility.glossy = false;
        let mut child = eval("child", Some(0));
        child.node.visibility.shadow = false;
        let mask = node_visibility_mask(&vec![parent, child], 1);
        assert_eq!(mask & constants::RAY_CAMERA, 0);
        assert_eq!(mask & constants::RAY_SPECULAR, 0);
        assert_eq!(mask & constants::RAY_SHADOW, 0);
        assert_ne!(mask & constants::RAY_DIFFUSE, 0);
    }
    #[test]
    fn hidden_nodes_and_imported_ancestors_disable_every_ray() -> Result<()> {
        let mut parent = eval("parent", None);
        parent.node.visibility.visible = false;
        let child = eval("child", Some(0));
        assert_eq!(node_visibility_mask(&vec![parent, child], 1), 0);
        let overrides = vork_scene::v2::from_test_value(json!({"Car":{"visible":false}}))?;
        assert_eq!(imported_visibility_mask(&overrides, &[(0,"Car".into()),(1,"Wheel".into())]),0);
        assert_eq!(imported_visibility_mask(&overrides, &[(2,"Other".into())]),constants::RAY_VISIBILITY_ALL);
        Ok(())
    }
    #[test]
    fn only_explicit_ids_are_inherited_by_unnamed_meshes() -> Result<()> {
        for explicit in [None, Some(7)] {
            let input = vork_scene::v2::from_test_value(json!({"version":2,"nodes":[{
                "type":"node","name":"props","pass_index":explicit,"children":[
                    {"type":"mesh","geometry":"g"},{"type":"mesh","geometry":"g"}]}]}))?;
            let nodes = installed(&input);
            let attrs = attributes(vec![], 0);
            let mut ids = SegmentationIds::new(&input, &attrs)?;
            let a = ids.prepare(1, &[], &nodes, &attrs)?.instance(None)?;
            let b = ids.prepare(2, &[], &nodes, &attrs)?.instance(None)?;
            if explicit.is_some() {
                assert_eq!((a, b), (7, 7));
            } else {
                assert_eq!((a, b), (1, 2));
            }
        }
        Ok(())
    }

    #[test]
    fn imported_mesh_nodes_get_distinct_automatic_ids_but_share_primitive_ids() -> Result<()> {
        let input = vork_scene::v2::from_test_value(
            json!({"version":2,"nodes":[{"type":"asset","name":"car","file":"car.glb"}]}),
        )?;
        let nodes = installed(&input);
        let attrs = attributes(vec![], 0);
        let mut ids = SegmentationIds::new(&input, &attrs)?;
        for (index, expected) in [(3, 1), (4, 2), (3, 1)] {
            assert_eq!(
                ids.prepare(0, &[(index, format!("mesh{index}"))], &nodes, &attrs)?
                    .instance(None)?,
                expected
            );
        }
        let metadata = ids.metadata();
        assert_eq!(metadata[0]["gltf_node_index"], 3);
        assert_eq!(metadata[1]["gltf_node_index"], 4);
        Ok(())
    }

    #[test]
    fn reserves_explicit_ids_and_reports_the_matched_imported_ancestor() -> Result<()> {
        let input = vork_scene::v2::from_test_value(
            json!({"version":2,"nodes":[{"type":"node","name":"parent","children":[{"type":"asset","name":"car","file":"car.glb","nodes":{"#3":{"pass_index":7}}}],"pass_index":1}]}),
        )?;
        let nodes = installed(&input);
        let none = attributes(vec![], 0);
        let mut ids = SegmentationIds::new(&input, &none)?;
        assert_eq!(ids.automatic(9, None)?, 2);
        assert_eq!(ids.automatic(9, None)?, 2);
        let draw = ids.prepare(1, &[(3, "Car".into()), (4, "Wheel".into())], &nodes, &none)?;
        assert_eq!(draw.instance(None)?, 7);
        let metadata = ids.metadata();
        assert_eq!(metadata[0]["gltf_node_name"], "Car");
        assert_eq!(metadata[0]["gltf_node_index"], 3);
        Ok(())
    }
    #[test]
    fn class_label_metadata_is_per_id_and_instance_ids_override_subtree_ids() -> Result<()> {
        let input = vork_scene::v2::from_test_value(
            json!({"version":2,"nodes":[{"type":"mesh","name":"scatter","geometry":"g","instances":{"attributes":{"TRANSLATION":"transforms","pass_index":"ids"}},"pass_index":8}]}),
        )?;
        let attrs = attributes(vec![77; 100], 100);
        let nodes = installed(&input);
        let mut ids = SegmentationIds::new(&input, &attrs)?;
        for _ in 0..2 {
            let draw = ids.prepare(0, &[], &nodes, &attrs)?;
            for index in 0..100 {
                assert_eq!(draw.instance(Some(index))?, 77);
            }
        }
        let metadata = ids.metadata();
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0]["instance_count"], 100);
        assert!(metadata[0].get("owners").is_none());
        Ok(())
    }
    #[test]
    fn rejects_background_id_and_mismatched_instance_ids() -> Result<()> {
        for node in [
            json!({"type":"node","pass_index":0}),
            json!({"type":"asset","file":"x.glb","nodes":{"#0":{"pass_index":0}}}),
        ] {
            let input = vork_scene::v2::from_test_value(json!({"version":2,"nodes":[node]}))?;
            assert!(SegmentationIds::new(&input, &attributes(vec![], 0)).is_err());
        }
        let input = vork_scene::v2::from_test_value(
            json!({"version":2,"nodes":[{"type":"mesh","geometry":"g","instances":{"attributes":{"TRANSLATION":"transforms","pass_index":"ids"}}}]}),
        )?;
        assert!(SegmentationIds::new(&input, &attributes(vec![2, 3], 1)).is_err());
        assert!(SegmentationIds::new(&input, &attributes(vec![0], 1)).is_err());
        Ok(())
    }
