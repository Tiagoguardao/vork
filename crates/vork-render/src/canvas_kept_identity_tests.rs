    use super::*;

    /// A kept canvas serves the scenes whose sorted JSON is the same text, under the same directory, and no other.
    #[test]
    fn a_kept_canvas_serves_exactly_the_scenes_with_the_same_content() -> Result<()> {
        let documents = [
            r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[100,100]},"nodes":[
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red"},
                {"name":"g","children":[{"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"blue"}]}]}"#,
            // The same document with its keys in another order.
            r#"{"nodes":[
                {"fill":"red","d":"M0 0H20V20H0Z","type":"path","name":"a"},
                {"children":[{"fill":"blue","name":"b","d":"M30 0H50V20H30Z","type":"path"}],"name":"g"}],
                "render":{"resolution":[100,100]},"composition":{"space":"2d"},"version":2}"#,
            // A paint, a geometry, a name, a placement in the tree, an order, a render setting, a track.
            r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[100,100]},"nodes":[
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"lime"},
                {"name":"g","children":[{"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"blue"}]}]}"#,
            r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[100,100]},"nodes":[
                {"name":"a","type":"path","d":"M0 0H20V21H0Z","fill":"red"},
                {"name":"g","children":[{"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"blue"}]}]}"#,
            r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[100,100]},"nodes":[
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red"},
                {"name":"g","children":[{"name":"c","type":"path","d":"M30 0H50V20H30Z","fill":"blue"}]}]}"#,
            r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[100,100]},"nodes":[
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red"},
                {"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"blue"},
                {"name":"g","children":[]}]}"#,
            r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[100,100]},"nodes":[
                {"name":"g","children":[{"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"blue"}]},
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red"}]}"#,
            r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[100,101]},"nodes":[
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red"},
                {"name":"g","children":[{"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"blue"}]}]}"#,
            r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[100,100]},"nodes":[
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red"},
                {"name":"g","children":[{"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"blue"}]}],
                "animation":[{"tracks":[{"path":["nodes","a","opacity"],"keys":[[0,1],[1,0]]}]}]}"#,
        ];
        let scenes = documents.iter().map(|document| Scene::from_json(document.as_bytes())).collect::<Result<Vec<_>>>()?;
        let base = Path::new("/a");
        let mut shared = 0;
        for (i, a) in scenes.iter().enumerate() {
            for (j, b) in scenes.iter().enumerate() {
                let same = serde_json::to_vec(a)? == serde_json::to_vec(b)?;
                assert_eq!(Kept::identity(a, base)? == Kept::identity(b, base)?, same, "documents {i} and {j}");
                shared += usize::from(same && i != j);
            }
        }
        // Only the reordered document shares, and every edit is its own scene.
        assert_eq!(shared, 2);
        // The directory the scene's files resolve against is part of what was built.
        assert_ne!(Kept::identity(&scenes[0], base)?, Kept::identity(&scenes[0], Path::new("/b"))?);
        Ok(())
    }
