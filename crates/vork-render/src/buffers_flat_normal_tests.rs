    use super::*;
    #[test]
    fn omitted_normals_split_faces_without_losing_attributes_or_identity() -> Result<()> {
        let mut input=json!({"version":2,"nodes":[{"type":"camera"}],
            "buffers":{
                "p":{"type":"vec3","data":[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]},
                "i":{"type":"u32","data":[0,1,2,0,3,1]},
                "uv":{"type":"vec2","data":[[0,0],[1,0],[0,1],[1,1]]},
                "t":{"type":"vec4","data":[[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1]]},
                "marker":{"type":"f32","data":[10,20,30,40]},
                "n":{"type":"vec3","data":[[0,1,1],[0,1,1],[0,1,1],[0,1,1]]}},
            "geometry":{"mesh":{"attributes":{"POSITION":"p","TEXCOORD_0":"uv","TANGENT":"t","_marker":"marker"},
                "indices":"i","face_vertex_counts":[3,3]}}});
        let load=|value:serde_json::Value| -> Result<geometry::Geometry> {
            let scene: ::vork_scene::v2::Scene=::vork_scene::v2::from_test_value(value)?;
            geometry_v2(&SceneBuffers::decode(&scene,Path::new("."))?,&scene.geometry["mesh"],&["_marker".into()],4,None)
        };
        let flat=load(input.clone())?;
        assert_eq!(flat.indices,vec![0,1,2,3,4,5]);
        assert_eq!(flat.vertex_ids,vec![0,1,2,0,3,1]);
        assert_eq!(flat.primitive_ids,vec![0,1]);
        for (index,vertex) in flat.vertices.iter().enumerate() {
            assert_eq!(vertex.NORMAL,if index<3 {[0.,0.,1.]} else {[0.,1.,0.]});
            assert_eq!(vertex.TANGENT,[1.,0.,0.,1.]);
            assert_eq!(flat.attributes["_marker"][index][0],[10.,20.,30.,10.,40.,20.][index]);
        }
        assert_eq!(flat.vertices[3].TEXCOORD_0,[0.,0.]);
        assert_eq!(flat.vertices[4].TEXCOORD_0,[1.,1.]);
        input["geometry"]["mesh"]["attributes"]["NORMAL"]=json!("n");
        let smooth=load(input)?;
        assert_eq!(smooth.vertices.len(),4);
        assert_eq!(smooth.indices,vec![0,1,2,0,3,1]);
        assert!(smooth.vertices.iter().all(|v|v.NORMAL==Vec3::new(0.,1.,1.).normalize().to_array()));
        Ok(())
    }
