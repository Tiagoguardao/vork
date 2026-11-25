    use super::*;
    #[test]
    fn instance_columns_default_compose_and_validate() -> Result<()> {
        let mut value=json!({"version":2,"nodes":[],"buffers":{
            "t":{"type":"vec3","data":[[2,3,4]]},"r":{"type":"vec4","data":[[0,0,0.6,0.8]]},
            "s":{"type":"vec3","data":[[-2,3,4]]}}});
        let definition: vork_scene::Instances=vork_scene::v2::from_test_value(json!({"attributes":{"TRANSLATION":"t","ROTATION":"r","SCALE":"s"}}))?;
        let decode=|v:&serde_json::Value| SceneBuffers::decode(&vork_scene::v2::from_test_value(v.clone())?,Path::new("."));
        let expected=Mat4::from_scale_rotation_translation(Vec3::new(-2.,3.,4.),glam::Quat::from_xyzw(0.,0.,0.6,0.8).normalize(),Vec3::new(2.,3.,4.));
        assert!(instance_transforms(&definition,&decode(&value)?)?[0].abs_diff_eq(expected,1e-6));
        let only_translation: vork_scene::Instances=vork_scene::v2::from_test_value(json!({"attributes":{"TRANSLATION":"t"}}))?;
        assert_eq!(instance_transforms(&only_translation,&decode(&value)?)?[0],Mat4::from_translation(Vec3::new(2.,3.,4.)));
        value["buffers"]["r"]["data"]=json!([[0,0,0,0]]);
        assert!(instance_transforms(&definition,&decode(&value)?).unwrap_err().to_string().contains("unit"));
        value["buffers"]["r"]["data"]=json!([[0,0,0,1],[0,0,0,1]]);
        assert!(instance_transforms(&definition,&decode(&value)?).unwrap_err().to_string().contains("equal counts"));
        value["buffers"]["r"]["data"]=json!([[0,0,0,1]]);value["buffers"]["s"]["data"]=json!([[0,1,1]]);
        assert!(crate::collapsed(instance_transforms(&definition,&decode(&value)?)?[0]));
        Ok(())
    }
