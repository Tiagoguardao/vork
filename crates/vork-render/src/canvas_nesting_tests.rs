    use super::*;
    /// A canvas nested to the stated depth is read, built, evaluated and hashed on a scene thread, as the command
    /// thread and the hashing worker do it.
    #[test]
    fn a_canvas_nested_to_the_stated_depth_builds_and_hashes() {
        let groups = vork_scene::MAX_NODE_DEPTH;
        let mut text = String::from(r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[64,64]},"nodes":["#);
        for depth in 0..groups { text.push_str(&format!(r#"{{"name":"g{depth}","children":["#)); }
        text.push_str(r##"{"type":"path","name":"leaf","d":"M0 0H16V16H0Z","fill":"#38bdf8"}"##);
        for _ in 0..groups { text.push_str("]}"); }
        text.push_str(r#"],"animation":[{"tracks":[{"path":["nodes","leaf","translation"],"keys":[[0,[0,0,0]],[1,[8,0,0]]]}]}]}"#);
        let built = vork_scene::scene_thread().spawn(move || -> Result<(usize, Option<usize>)> {
            let scene = Scene::from_json(text.as_bytes())?;
            let document = Document::new(&scene, Path::new("."))?;
            document.frame(&vork_time::Request::Numeric(0.5))?;
            let hashed = std::thread::scope(|scope| {
                vork_scene::scene_thread().spawn_scoped(scope, || Kept::identity(&scene, Path::new("."))).unwrap().join().unwrap()
            })?;
            ensure!(hashed == Kept::identity(&scene, Path::new("."))?, "the scene hashes the same on any scene thread");
            Ok((document.items.len(), document.rest[groups].parent))
        }).unwrap().join().unwrap().unwrap();
        assert_eq!(built, (groups + 1, Some(groups - 1)));
    }
