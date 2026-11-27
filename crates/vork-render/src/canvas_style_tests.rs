    use super::*;
    use serde_json::json;
    fn doc(node:serde_json::Value,tracks:serde_json::Value)->Document {
        let scene=Scene::from_json(&serde_json::to_vec(&json!({"version":2,"composition":{"space":"2d"},"nodes":[node],"animation":[{"tracks":tracks}]})).unwrap()).unwrap();
        Document::new(&scene,Path::new(".")).unwrap()
    }
    fn drawing(doc:&Document,time:f32)->serde_json::Value {
        let frame=doc.snapshot(&vork_time::Request::Numeric(time)).unwrap();
        let nodes:Vec<_>=doc.items.iter().enumerate().map(|(i,n)|frame.nodes[i].as_deref().unwrap_or(&n.node)).collect();
        crate::canvas_draw::document(&doc.page,&nodes,&frame.values).unwrap()
    }
    /// A group that collapses to zero scale in the frame its text child loses a keyed glyph: the group is
    /// evaluated first whatever the track order, so the child's batches are not built against a parent that
    /// no longer draws. The kept document equals a fresh one.
    #[test]
    fn a_collapsing_parent_is_evaluated_before_its_changed_child() {
        let font=Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        let node=json!({"name":"g","children":[{"name":"n","type":"text","content":"AB","font":font}]});
        let child=[json!({"path":["nodes","n","content"],"keys":[[0,"AB"],[1,"A"]],"interpolation":"STEP"}),
            json!({"path":["nodes","n","glyphs",1,"opacity"],"keys":[[0,0.25]]})];
        let parent=json!({"path":["nodes","g","scale"],"keys":[[0,[1,1,1]],[1,[0,0,1]]],"interpolation":"STEP"});
        for child_first in [true, false] {
            let tracks=if child_first {json!([child[0],child[1],parent])} else {json!([parent,child[0],child[1]])};
            let kept=doc(node.clone(),tracks.clone());
            assert!(kept.snapshot(&vork_time::Request::Numeric(0.)).unwrap().batches().next().is_some(), "child_first={child_first}");
            let frame=kept.snapshot(&vork_time::Request::Numeric(1.)).expect("a collapsed group's text is not built");
            let fresh=doc(node.clone(),tracks).snapshot(&vork_time::Request::Numeric(1.)).unwrap();
            // The kept document's text cache has a revision and a composition from the frame that drew the text;
            // that identity is not picture content.
            for (kept, fresh) in frame.values.iter().zip(&fresh.values) {
                assert!(kept.transform==fresh.transform && kept.opacity==fresh.opacity && kept.visible==fresh.visible
                    && kept.texts==fresh.texts && kept.path==fresh.path, "child_first={child_first}");
            }
            assert_eq!(frame.batches().count(), 0, "child_first={child_first}");
        }
    }

    #[test]
    fn keyed_layout_piece_bounds_use_sampled_text() {
        let font=Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        for (unit, content) in [("glyphs", "AB"), ("words", "A B"), ("lines", "A\nB")] {
            for owned in [false, true] {
                let node=json!({"name":"n","type":"text","content":"A","font":font,"white-space":"pre-line"});
                let prefix=if owned {vec![]} else {vec!["nodes", "n"]};
                let mut content_path=prefix.clone();content_path.push("content");
                let mut piece_path=prefix;piece_path.extend([unit,"1","opacity"]);
                let tracks=json!([
                    {"path":content_path,"keys":[[0,content],[1,"A"]],"interpolation":"STEP"},
                    {"path":piece_path,"keys":[[0,0.25]]}
                ]);
                let animated=if owned {
                    let mut node=node.clone();node["animation"]=json!([{"tracks":tracks}]);doc(node,json!([]))
                } else {doc(node.clone(),tracks)};
                let mut expected=node;expected["content"]=json!(content);
                let expected=doc(expected,json!([{"path":["nodes","n",unit,1,"opacity"],"keys":[[0,0.25]]}]));
                for time in [-1.,0.,0.75] {assert_eq!(drawing(&animated,time),drawing(&expected,time), "{unit}, owned={owned}");}
                let error=animated.snapshot(&vork_time::Request::Numeric(1.)).err().expect("sampled short text must refuse").to_string();
                assert!(error.contains("index 1 is past the end"), "{error}");
                let mut restored = drawing(&animated,0.);
                let mut reference = drawing(&expected,0.);
                // A seek rebuilds text and advances its cache revision; that identity is not picture content.
                for picture in [&mut restored, &mut reference] {
                    for node in picture["nodes"].as_array_mut().unwrap() {
                        if let Some(binding) = node.get_mut("vorkText") { binding[2] = json!(0); }
                        if let Some(paint) = node.get_mut("vorkPaint") { *paint = json!(0); }
                    }
                }
                assert_eq!(restored,reference);
            }
        }
    }

    #[test]
    fn frames_are_classified_by_the_largest_changed_track_kind() {
        let doc = doc(serde_json::json!({"name":"p","type":"path","d":"M0 0H10V10H0Z","fill":"#ff0000"}), serde_json::json!([
            {"path":["nodes","p","translation"],"keys":[[0,[0,0,0]],[1,[10,0,0]],[2,[10,0,0]],[3,[10,0,0]],[4,[10,0,0]]]},
            {"path":["nodes","p","opacity"],"keys":[[0,[1]],[1,[1]],[2,[0.5]],[3,[0.5]],[4,[0.5]]]},
            {"path":["nodes","p","fill"],"keys":[[0,[1,0,0,1]],[2,[1,0,0,1]],[3,[0,0,1,1]],[4,[0,0,1,1]]]},
            {"path":["nodes","p","stroke-width"],"keys":[[0,[0]],[3,[0]],[4,[2]]]}
        ]));
        assert_eq!(doc.kinds, [ChangeKind::Placement, ChangeKind::Placement, ChangeKind::PaintValue, ChangeKind::Content].map(Some));
        // First frame, a move, a hold, a fade, a colour change, a stroke change.
        for time in [0., 0.5, 0.5, 1.5, 2.5, 3.5] { doc.snapshot(&vork_time::Request::Numeric(time)).unwrap(); }
        assert_eq!(doc.change_counters(), serde_json::json!({"unchanged":1,"placement":2,"paint_value":1,"content":1,"structure":1}));
    }

    /// A frame that only moves nodes is evaluated for those nodes alone; it equals a full evaluation.
    #[test]
    fn placement_frames_match_full_evaluation() {
        let node = json!({"name":"g","children":[
            {"name":"a","type":"path","d":"M0 0H10V10H0Z","fill":"red"},
            {"name":"b","type":"path","d":"M0 0H10V10H0Z","fill":"blue"},
            {"name":"still","type":"path","d":"M0 0H10V10H0Z","fill":"lime"}]});
        let tracks = json!([
            {"path":["nodes","g","translation"],"keys":[[0,[0,0,0]],[2,[40,10,0]]]},
            {"path":["nodes","a","scale"],"keys":[[0,[1,1,1]],[1,[2,0.5,1]],[2,[1,1,1]]]},
            {"path":["nodes","b","opacity"],"keys":[[0,[1]],[2,[0.2]]]},
            {"path":["nodes","b","translation"],"keys":[[0,[0,0,0]],[1,[5,5,0]],[2,[5,5,0]]]}]);
        let kept = doc(node.clone(), tracks.clone());
        let mut placed = 0;
        for step in [0, 3, 7, 8, 8, 12, 20, 15, 1] {
            let time = step as f32 / 10.;
            let before = kept.change_counters()["placement"].as_u64().unwrap();
            let frame = kept.snapshot(&vork_time::Request::Numeric(time)).unwrap();
            // A placement frame evaluates the items its changed tracks name.
            if kept.change_counters()["placement"].as_u64().unwrap() > before {
                let (items, _) = kept.changed_items(&kept.samples.borrow());
                placed += 1;
                assert!(!items.iter().any(|(item, _)| *item == kept.by_name["still"]), "a node no track moves is not visited");
            }
            let fresh = doc(node.clone(), tracks.clone()).snapshot(&vork_time::Request::Numeric(time)).unwrap();
            assert!(frame.values == fresh.values, "time {time}");
        }
        assert!(placed >= 6, "moves are evaluated sparsely: {placed}");
    }

    /// A text node whose fill or stroke alone changes keeps its text and takes new paints: the kept document's
    /// node, values and batches equal a fresh document's at every time, flat colours and a gradient alike, and
    /// its layout and unit tables are the ones it had (it is not built again).
    #[test]
    fn a_text_fill_change_repaints_the_kept_text() {
        let font=Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        let node=json!({"name":"n","type":"text","content":[{"content":"A"},{"content":"B","fill":"lime"}],"font":font,"fill":"red","stroke":"black","stroke-width":1});
        let tracks=json!([
            {"path":["nodes","n","fill"],"keys":[[0,"red"],[1,"blue"],[2,"linear-gradient(red, blue)"],[3,"none"]],"interpolation":"STEP"},
            {"path":["nodes","n","stroke"],"keys":[[0,"black"],[1.5,"white"]],"interpolation":"STEP"}]);
        let kept=doc(node.clone(),tracks.clone());
        let mut layout=None;
        for time in [0.,0.5,1.,1.5,2.,3.,1.,0.] {
            let frame=kept.snapshot(&vork_time::Request::Numeric(time)).unwrap();
            let fresh_doc=doc(node.clone(),tracks.clone());
            let fresh=fresh_doc.snapshot(&vork_time::Request::Numeric(time)).unwrap();
            let specs=|frame:&Snapshot| frame.batches().map(|draw| (draw.sub, draw.spec.run_colors.clone(), draw.spec.fill_override.clone(), draw.spec.stroke_color.clone(), draw.spec.html.clone())).collect::<Vec<_>>();
            assert!(specs(&frame)==specs(&fresh), "time {time}");
            assert!(frame.values[0].text_composition==fresh.values[0].text_composition && frame.values[0].texts==fresh.values[0].texts, "time {time}");
            let (kept_text,fresh_text)=(kept.styles.borrow(),fresh_doc.styles.borrow());
            if let (Some(a),Some(b))=(kept_text[0].as_ref().and_then(|s|s.text.as_ref()),fresh_text[0].as_ref().and_then(|s|s.text.as_ref())) {
                assert!(a.fill==b.fill && a.stroke==b.stroke && a.spec.run_colors==b.spec.run_colors, "time {time}");
                // One layout through every repaint, and the same unit tables: a text built again would have its own.
                let first=layout.get_or_insert_with(||(a.layout.clone(),a.units.clone()));
                assert!(Arc::ptr_eq(&first.0,&a.layout) && Arc::ptr_eq(&first.1,&a.units), "time {time}");
            }
        }
    }

    // Concurrent asset publication can legitimately rebuild the plan. These fixtures
    // exercise incremental edits with stable resource inputs.
    fn run_with_stable_assets(test: fn() -> Result<()>) -> Result<()> {
        const CHILD: &str = "VORK_TEST_STABLE_PAINT_ASSETS";
        let thread = std::thread::current();
        let name = thread.name().context("unnamed paint test")?;
        if std::env::var(CHILD).as_deref() == Ok(name) {
            return test();
        }
        let output = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", name]).env(CHILD, name).output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        ensure!(output.status.success() && stdout.lines().any(|line| line == "running 1 test"),
            "{name} did not pass exactly one test\n{stdout}\n{}", String::from_utf8_lossy(&output.stderr));
        Ok(())
    }

    #[test]
    fn text_paint_revisions_keep_geometry_and_match_fresh_plans() -> Result<()> {
        run_with_stable_assets(text_paint_revisions_keep_geometry_and_match_fresh_plans_fixture)
    }

    fn text_paint_revisions_keep_geometry_and_match_fresh_plans_fixture() -> Result<()> {
        let font = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        let node = json!({"name":"n","type":"text","content":"Paint","font":font,"font-size":32,"fill":"red",
            "stroke":"blue","stroke-width":2});
        let tracks = json!([
            {"path":["nodes","n","fill"],"keys":[[0,"red"],[1,"green"],[2,"blue"]]},
            {"path":["nodes","n","stroke"],"keys":[[0,"blue"],[1,"red"],[2,"lime"]]}]);
        let kept = doc(node.clone(), tracks.clone());
        let mut content = None;
        let mut last_paint = 0;
        for time in [0., 0.25, 0.75, 1.5, 0.25] {
            let frame = kept.snapshot(&vork_time::Request::Numeric(time))?;
            let revision = frame.values[0].text_revision;
            assert_eq!(*content.get_or_insert(revision), revision);
            assert!(frame.values[0].text_paint_revisions[0] > last_paint);
            last_paint = frame.values[0].text_paint_revisions[0];
            let actual = kept.plan_snapshot(&frame, &[])?;
            let fresh = doc(node.clone(), tracks.clone());
            let fresh_frame = fresh.snapshot(&vork_time::Request::Numeric(time))?;
            let expected = fresh.plan_snapshot(&fresh_frame, &[])?;
            assert_eq!(format!("{:?}", actual.items), format!("{:?}", expected.items), "time {time}");
            if time != 0. {
                assert!(actual.items.iter().any(|item| matches!(item, moda_engine::seams::DrawItem::Text(b) if b.content_key.is_none())), "paint patches bypass content preparation");
            }
        }
        Ok(())
    }

    #[test]
    fn text_paint_and_path_placement_share_the_incremental_writer() -> Result<()> {
        run_with_stable_assets(text_paint_and_path_placement_share_the_incremental_writer_fixture)
    }

    fn text_paint_and_path_placement_share_the_incremental_writer_fixture() -> Result<()> {
        let font = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        let node = json!({"name":"group","children":[
            {"name":"n","type":"text","content":"Paint","font":font,"font-size":32,"fill":"red"},
            {"name":"p","type":"path","d":"M0 50H20V70H0Z","fill":"blue"}]});
        let tracks = json!([
            {"path":["nodes","n","fill"],"keys":[[0,"red"],[1,"green"]]},
            {"path":["nodes","p","translation"],"keys":[[0,[0,0,0]],[1,[20,0,0]]]}]);
        let kept = doc(node.clone(), tracks.clone());
        let mut geometry: Option<moda_text::vector::VectorNodeOps> = None;
        for time in [0., 0.25, 0.75, 0.25] {
            let frame = kept.snapshot(&vork_time::Request::Numeric(time))?;
            let actual = kept.plan_snapshot(&frame, &[])?;
            let fresh = doc(node.clone(), tracks.clone());
            let frame = fresh.snapshot(&vork_time::Request::Numeric(time))?;
            let expected = fresh.plan_snapshot(&frame, &[])?;
            assert_eq!(format!("{:?}", actual.items), format!("{:?}", expected.items));
            let batch = actual.items.iter().find_map(|item| match item {
                moda_engine::seams::DrawItem::Text(b) => Some(b), _ => None,
            }).unwrap();
            let vector = batch.vector.as_ref().unwrap();
            if let Some(old) = &geometry {
                assert!(batch.content_key.is_none(), "mixed writes must repaint text");
                assert!(Arc::ptr_eq(&old.curve_points, &vector.curve_points));
                for (a,b) in old.ops.iter().zip(&vector.ops) {
                    assert!(Arc::ptr_eq(&a.records, &b.records));
                    assert!(Arc::ptr_eq(&a.tris, &b.tris));
                }
            }
            geometry = Some(vector.clone());
        }
        Ok(())
    }

    #[test]
    fn glyph_fill_repaints_kept_batches_when_coverage_is_unchanged() -> Result<()> {
        run_with_stable_assets(glyph_fill_repaints_kept_batches_when_coverage_is_unchanged_fixture)
    }

    fn glyph_fill_repaints_kept_batches_when_coverage_is_unchanged_fixture() -> Result<()> {
        let font = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        let node = json!({"name":"n","type":"text","content":"AB","font":font,"font-size":32,"fill":"red"});
        let tracks = json!([{ "path":["nodes","n","glyphs",0,"fill"],"keys":[[0,"blue"],[1,"green"]]}]);
        let kept = doc(node.clone(), tracks.clone());
        let mut content = None;
        let mut previous: Vec<Arc<moda_text_draw::TextQuadBatch>> = Vec::new();
        for time in [0., 0.25, 0.75, 0.25] {
            let frame = kept.snapshot(&vork_time::Request::Numeric(time))?;
            assert_eq!(*content.get_or_insert(frame.values[0].text_revision), frame.values[0].text_revision);
            let actual = kept.plan_snapshot(&frame, &[])?;
            let fresh = doc(node.clone(), tracks.clone());
            let frame = fresh.snapshot(&vork_time::Request::Numeric(time))?;
            let expected = fresh.plan_snapshot(&frame, &[])?;
            assert_eq!(format!("{:?}", actual.items), format!("{:?}", expected.items));
            let batches: Vec<_> = actual.items.iter().filter_map(|item| match item {
                moda_engine::seams::DrawItem::Text(b) => Some(b.clone()), _ => None,
            }).collect();
            assert_eq!(batches.len(), 2);
            if !previous.is_empty() {
                assert!(batches[0].content_key.is_none(), "the keyed glyph repainted");
                assert!(Arc::ptr_eq(&previous[1], &batches[1]), "the unkeyed glyph kept its batch");
                for (old, batch) in previous.iter().zip(&batches) {
                    for (a, b) in old.vector.as_ref().unwrap().ops.iter().zip(&batch.vector.as_ref().unwrap().ops) {
                        assert!(Arc::ptr_eq(&a.records, &b.records));
                    }
                }
            }
            previous = batches;
        }
        Ok(())
    }

    #[test]
    fn dense_text_recolour_reuses_glyph_geometry() -> Result<()> {
        let font = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        let names: Vec<_> = (0..4096).map(|i| format!("label{i}")).collect();
        let nodes: Vec<_> = names.iter().map(|name| json!({
            "name":name, "type":"text", "content":"A", "font":font, "fill":"red"
        })).collect();
        let tracks: Vec<_> = names.iter().map(|name| json!({
            "path":["nodes",name,"fill"], "keys":[[0,"red"],[1,"blue"]]
        })).collect();
        let scene = Scene::from_json(&serde_json::to_vec(&json!({
            "version":2, "composition":{"space":"2d"}, "nodes":nodes, "animation":[{"tracks":tracks}]
        }))?)?;
        let doc = Document::new(&scene, Path::new("."))?;
        let first = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        drop(doc.plan_snapshot(&first, &[])?);
        let next = doc.snapshot(&vork_time::Request::Numeric(1.))?;
        let plan = doc.plan_snapshot(&next, &[])?;
        let texts: Vec<_> = plan.items.iter().filter_map(|item| match item {
            moda_engine::seams::DrawItem::Text(batch) => Some(batch), _ => None,
        }).collect();
        assert_eq!(texts.len(), names.len());
        assert!(texts.iter().all(|b| b.content_key.is_none()), "all resident glyph batches use their repaint recipe");
        Ok(())
    }

    /// A snapshot planned by another document is that snapshot's state, not the other document's own latest, even
    /// when both have taken as many snapshots.
    #[test]
    fn a_snapshot_is_planned_as_its_own_state_by_any_document() {
        let node = json!({"name":"n","type":"path","d":"M0 0H10V10H0Z","fill":"red"});
        let tracks = json!([{"path":["nodes","n","translation"],"keys":[[0,[0,0,0]],[1,[20,0,0]]]}]);
        let (a, b) = (doc(node.clone(), tracks.clone()), doc(node, tracks));
        let early = a.snapshot(&vork_time::Request::Numeric(0.)).unwrap();
        let late = b.snapshot(&vork_time::Request::Numeric(1.)).unwrap();
        assert_ne!(early.values[0].transform.translation[0], late.values[0].transform.translation[0]);
        // `b` plans `a`'s snapshot: its scene holds the early state, as a fresh document's would.
        b.plan_snapshot(&early, &[]).unwrap();
        let nodes: Vec<_> = b.items.iter().map(|item| &item.node).collect();
        b.retained.borrow().as_ref().unwrap().assert_fresh(&b.page, &nodes, &early.values);
    }

    /// A world's picture is given by a plan and kept in the evaluated table: a frame that then moves another
    /// item evaluates in place, keeps the picture, and passes the check against a whole evaluation.
    #[test]
    fn a_worlds_picture_stays_in_the_table_across_frames() -> Result<()> {
        let scene = Scene::from_json(&serde_json::to_vec(&json!({"version": 2, "composition": {"space": "2d"}, "render": {"resolution": [32, 32]},
            "buffers": {"p": {"type": "vec3", "data": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}},
            "geometry": {"g": {"attributes": {"POSITION": "p"}}},
            "nodes": [
                {"name": "shot", "composition": {"space": "3d"}, "children": [
                    {"type": "camera", "name": "cam", "translation": [0, 0, 5]}, {"type": "mesh", "name": "tri", "geometry": "g"}]},
                {"name": "box", "type": "path", "d": "M0 0H8V8H0Z", "fill": "red"}],
            "animation": [{"tracks": [{"path": ["nodes", "box", "translation"], "keys": [[0, [0, 0, 0]], [1, [10, 0, 0]]]}]}]}))?)?;
        let doc = Document::new(&scene, Path::new("."))?;
        let world = doc.worlds[0].item;
        doc.snapshot(&vork_time::Request::Numeric(0.))?;
        // What a plan does with the world's picture.
        let picture = Some(("picture".to_string(), [32, 32]));
        doc.evaluated.borrow_mut().as_mut().unwrap().state.values[world].world = picture.clone();
        let moved = doc.snapshot(&vork_time::Request::Numeric(0.5))?;
        assert_eq!(doc.change_counters()["placement"], json!(1), "the box alone moves");
        assert!(moved.values[world].world == picture, "the table keeps the picture");
        assert_eq!(moved.values[doc.by_name["box"]].transform.translation[0], 5.);
        Ok(())
    }

    #[test]
    fn solid_colour_tracks_are_paint_values_however_they_are_spelled() {
        let kinds = |fill: &str, keys: serde_json::Value| doc(serde_json::json!({"name":"p","type":"path","d":"M0 0H10V10H0Z","fill":fill}),
            serde_json::json!([{"path":["nodes","p","fill"],"keys":keys}])).kinds.clone();
        assert_eq!(kinds("red", serde_json::json!([[0,[1,0,0,1]],[1,[0,0,1,1]]])), [Some(ChangeKind::PaintValue)]);
        assert_eq!(kinds("red", serde_json::json!([[0,"red"],[1,"blue"]])), [Some(ChangeKind::PaintValue)]);
        // A key that is not a colour, or a rest paint that is not one, changes the paint's kind.
        assert_eq!(kinds("red", serde_json::json!([[0,"red"],[1,"linear-gradient(red, blue)"]])), [Some(ChangeKind::Content)]);
        assert_eq!(kinds("linear-gradient(red, blue)", serde_json::json!([[0,"red"],[1,"blue"]])), [Some(ChangeKind::Content)]);
        assert_eq!(kinds("none", serde_json::json!([[0,"red"],[1,"blue"]])), [Some(ChangeKind::Content)]);
    }

    #[test]
    fn gradients_of_one_form_are_paint_values() {
        let doc = |fill: &str, tracks: serde_json::Value| doc(serde_json::json!({"name":"p","type":"path","d":"M0 0H10V10H0Z","fill":fill}), tracks);
        let fill = |keys: serde_json::Value, step: bool| if step { serde_json::json!([{"path":["nodes","p","fill"],"keys":keys,"interpolation":"STEP"}]) }
            else { serde_json::json!([{"path":["nodes","p","fill"],"keys":keys}]) };
        let gradient = "linear-gradient(red, blue)";
        // A stop's colour and the gradient's angle change; type, interpolation and stop count do not.
        let stops = doc(gradient, fill(serde_json::json!([[0,gradient],[1,"linear-gradient(green, blue)"]]), false));
        assert_eq!(stops.kinds, [Some(ChangeKind::PaintValue)]);
        for time in [0., 0.25, 0.5] { stops.snapshot(&vork_time::Request::Numeric(time)).unwrap(); }
        assert_eq!(stops.change_counters(), serde_json::json!({"unchanged":0,"placement":0,"paint_value":2,"content":0,"structure":1}));
        assert_eq!(doc(gradient, fill(serde_json::json!([[0,gradient],[1,"linear-gradient(90deg, red, blue)"]]), false)).kinds, [Some(ChangeKind::PaintValue)]);
        // Another type, another stop count, or a colour at rest: the paint's form changes.
        assert_eq!(doc(gradient, fill(serde_json::json!([[0,gradient],[1,"radial-gradient(red, blue)"]]), true)).kinds, [Some(ChangeKind::Content)]);
        assert_eq!(doc(gradient, fill(serde_json::json!([[0,gradient],[1,"linear-gradient(red, green, blue)"]]), true)).kinds, [Some(ChangeKind::Content)]);
        assert_eq!(doc("red", fill(serde_json::json!([[0,gradient],[1,"linear-gradient(green, blue)"]]), false)).kinds, [Some(ChangeKind::Content)]);
        // Opacity scales the paint that is there, whatever its form.
        let opacity = serde_json::json!([{"path":["nodes","p","fill-opacity"],"keys":[[0,[1]],[1,[0]]]}]);
        assert_eq!(doc(gradient, opacity.clone()).kinds, [Some(ChangeKind::PaintValue)]);
        assert_eq!(doc("red", opacity).kinds, [Some(ChangeKind::PaintValue)]);
    }

    #[test]
    fn a_paint_another_layer_can_change_the_form_of_is_content_on_every_track() {
        let scene = Scene::from_json(&serde_json::to_vec(&json!({"version":2,"composition":{"space":"2d"},
            "nodes":[{"name":"p","type":"path","d":"M0 0H10V10H0Z","fill":"red"}],
            "animation":[
                {"tracks":[{"path":["nodes","p","fill"],"keys":[[0,"red"],[1,"blue"]]}]},
                {"tracks":[{"path":["nodes","p","fill"],"keys":[[0,"red"],[1,"linear-gradient(red, blue)"]],"interpolation":"STEP"}]}]})).unwrap()).unwrap();
        assert_eq!(Document::new(&scene, Path::new(".")).unwrap().kinds, [Some(ChangeKind::Content); 2]);
    }

    #[test]
    fn layer_weight_tracks_keep_the_kind_of_what_they_drive() {
        let scene = |driven: serde_json::Value| {
            let scene = Scene::from_json(&serde_json::to_vec(&json!({"version":2,"composition":{"space":"2d"},
                "nodes":[{"name":"p","type":"path","d":"M0 0H10V10H0Z","fill":"red"}],
                "animation":[
                    {"tracks":[{"path":["animation","driven","weight"],"keys":[[0,[0]],[1,[1]]]}]},
                    {"name":"driven","tracks":[driven]}]})).unwrap()).unwrap();
            Document::new(&scene, Path::new(".")).unwrap()
        };
        let moved = scene(json!({"path":["nodes","p","translation"],"keys":[[0,[10,0,0]],[1,[10,0,0]]]}));
        assert_eq!(moved.kinds, [None, Some(ChangeKind::Placement)]);
        for time in [0.25, 0.5, 0.75] { moved.snapshot(&vork_time::Request::Numeric(time)).unwrap(); }
        assert_eq!(moved.change_counters(), json!({"unchanged":0,"placement":2,"paint_value":0,"content":0,"structure":1}));
        let widened = scene(json!({"path":["nodes","p","stroke-width"],"keys":[[0,[4]],[1,[4]]]}));
        for time in [0.25, 0.5, 0.75] { widened.snapshot(&vork_time::Request::Numeric(time)).unwrap(); }
        assert_eq!(widened.change_counters(), json!({"unchanged":0,"placement":0,"paint_value":0,"content":2,"structure":1}));
    }

    #[test]
    fn published_plans_preserve_frames_across_worker_schedules() -> Result<()> {
        fn picture(document: &Document, time: f32) -> (serde_json::Value, Vec<(Arc<moda_text::ops::TextNodeSpec>, [f32; 16])>) {
            let frame = document.snapshot(&vork_time::Request::Numeric(time)).unwrap();
            let nodes: Vec<_> = document.items.iter().enumerate().map(|(i, n)| frame.nodes[i].as_deref().unwrap_or(&n.node)).collect();
            let mut picture = crate::canvas_draw::document(&document.page, &nodes, &frame.values).unwrap();
            // Each retained history has its own cache revision. Compare picture
            // content and the bound text specs, not those history-local stamps.
            for node in picture["nodes"].as_array_mut().unwrap() {
                if let Some(binding) = node.get_mut("vorkText") { binding[2] = json!(0); }
                if let Some(paint) = node.get_mut("vorkPaint") { *paint = json!(0); }
            }
            (picture, frame.batches().map(|draw| (draw.spec.clone(), draw.model)).collect())
        }
        let font = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        let scene = Scene::from_json(&serde_json::to_vec(&json!({
            "version":2,"composition":{"space":"2d"},
            "nodes":[{"name":"g","active":[0,3],"children":[
                {"name":"p","type":"path","d":"M0 0H20V20H0Z","fill":"red"},
                {"name":"t","type":"text","content":"AB","font":font}
            ]}],
            "animation":[{"tracks":[
                {"path":["nodes","p","translation"],"keys":[[0,[0,0,0]],[2,[25,15,0]]]},
                {"path":["nodes","t","content"],"keys":[[0,"AB"],[1,"CD"]],"interpolation":"STEP"},
                {"path":["nodes","g","opacity"],"keys":[[0,1],[2,0.4]]}
            ]}]
        }))?)?;
        let original = Document::new(&scene, Path::new("."))?;
        let expected: Vec<_> = (0..32).map(|frame| {
            picture(&Document::new(&scene, Path::new(".")).unwrap(), frame as f32 / 8.)
        }).collect();
        // Keep all issued frames alive while advancing and rewinding the one
        // producer. A late GPU consumer must still see exactly its issued plan.
        for count in 1..=4 {
            let mut lanes: Vec<Vec<_>> = (0..count).map(|_| Vec::new()).collect();
            for frame in (0..32).chain((0..32).rev()) {
                assert_eq!(picture(&original, frame as f32 / 8.), expected[frame]);
                let plan = original.plan(&[])?;
                let shared = plan.shared().expect("complete plan");
                assert!(std::ptr::eq(&*plan, shared.coverage.as_ref()));
                lanes[frame % count].push((shared, format!("{:?}", plan.items)));
            }
            std::thread::scope(|scope| {
                for lane in lanes {
                    scope.spawn(move || for (plan, expected) in lane {
                        assert_eq!(format!("{:?}", plan.coverage.items), expected);
                    });
                }
            });
        }
        Ok(())
    }

    #[test]
    fn sampled_state_reuses_holds_and_matches_fresh_reverse_seeks() {
        let node=json!({"name":"n","type":"path","d":"M0 0H20V20H0Z","fill":"red"});
        let tracks=json!([
            {"path":["nodes","n","translation"],"keys":[[0,[0,0,0]],[1,[20,0,0]]],"interpolation":"STEP"},
            {"path":["nodes","n","opacity"],"keys":[[0,1],[1,0.4]],"interpolation":"STEP"}
        ]);
        let retained=doc(node.clone(),tracks.clone());
        let first=retained.snapshot(&vork_time::Request::Numeric(0.)).unwrap();
        let held=retained.snapshot(&vork_time::Request::Numeric(0.5)).unwrap();
        assert!(first.values == held.values);
        assert_eq!(retained.change_counters()["unchanged"], json!(1), "a held frame evaluates nothing");
        for time in [0.75,1.,2.,0.25,1.,0.] {
            let fresh=doc(node.clone(),tracks.clone());
            assert_eq!(drawing(&retained,time),drawing(&fresh,time),"time {time}");
        }
        // A consumer's edited test frame cannot change another snapshot or
        // trick the retained planner's state identity check.
        let mut edited=retained.snapshot(&vork_time::Request::Numeric(0.)).unwrap();
        edited.values[0].transform.translation[0]=17.;
        assert_eq!(first.values[0].transform.translation[0],0.);
    }

    #[test]
    fn sampled_state_tracks_clock_presence_and_recovers_after_errors() {
        let node=json!({"name":"parent","active":[1,3],"children":[
            {"name":"n","type":"path","d":"M0 0H20V20H0Z","fill":"red"}]});
        let retained=doc(node.clone(),json!([]));
        for time in [0.,0.5,1.,2.,3.,2.,0.] {
            assert_eq!(drawing(&retained,time),drawing(&doc(node.clone(),json!([])),time));
        }
        let a=retained.snapshot(&vork_time::Request::Numeric(1.)).unwrap();
        let b=retained.snapshot(&vork_time::Request::Numeric(2.)).unwrap();
        assert!(a.values == b.values,"local time alone does not change node values");

        let retained=doc(json!({"name":"n","type":"path","d":"M0 0H20V20H0Z","fill":"red"}),json!([
            {"path":["nodes","n","translation"],"keys":[[0,[0,0,0]],[1,[0,0,10]]],"interpolation":"STEP"}
        ]));
        let good=retained.snapshot(&vork_time::Request::Numeric(0.)).unwrap();
        assert!(retained.snapshot(&vork_time::Request::Numeric(1.)).is_err());
        let recovered=retained.snapshot(&vork_time::Request::Numeric(0.)).unwrap();
        assert_eq!(good.values[0].transform,recovered.values[0].transform);
        assert_eq!(retained.change_counters()["structure"], json!(2), "a failed evaluation discards the table: the next frame evaluates every item");
    }

    #[test]
    fn keyed_whitespace_and_width_defer_static_piece_bounds() {
        let font=Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        for (field, rest, sampled, unit, index) in [
            ("white-space",json!("pre-line"),json!("normal"),"glyphs",2),
            ("width",json!(1000),json!(20),"lines",1),
        ] {
            let mut node=json!({"name":"n","type":"text","content":"A\nB","font":font});node[field]=rest;
            let animated=doc(node.clone(),json!([
                {"path":["nodes","n",field],"keys":[[0,sampled]]},
                {"path":["nodes","n",unit,index,"opacity"],"keys":[[0,0.5]]}
            ]));
            node[field]=sampled;
            let expected=doc(node,json!([{"path":["nodes","n",unit,index,"opacity"],"keys":[[0,0.5]]}]));
            assert_eq!(drawing(&animated,0.),drawing(&expected,0.),"{field}");
        }
    }

    #[test]
    fn css_background_defaults_to_continuous_interpolation() {
        let scene=Scene::from_json(br##"{"version":2,"composition":{"space":"2d"},"render":{"background":"red"},"nodes":[],"animation":[{"tracks":[{"path":["render","background"],"keys":[[0,"red"],[2,"blue"]]}]}]}"##).unwrap();
        let doc=Document::new(&scene,Path::new(".")).unwrap();
        assert_eq!(doc.snapshot(&vork_time::Request::Numeric(1.)).unwrap().background,Some([0.5,0.,0.5,1.]));
    }

    #[test]
    fn animated_path_filter_and_paint_match_an_authored_midpoint() {
        let node=json!({"name":"n","type":"path","d":"M0 0H10V10H0Z","fill":"red","stroke":"white","stroke-width":2});
        let animated=doc(node.clone(),json!([
            {"path":["nodes","n","d"],"keys":[[0,"M0 0H10V10H0Z"],[2,"M0 0H30V50H0Z"]]},
            {"path":["nodes","n","filter"],"keys":[[0,"none"],[2,"blur(8px)"]]},
            {"path":["nodes","n","fill"],"keys":[[0,"red"],[2,"blue"]]},
            {"path":["nodes","n","stroke-width"],"keys":[[0,2],[2,6]]}
        ]));
        let mut midpoint=node;midpoint["d"]=json!("M0 0H20V30H0Z");midpoint["filter"]=json!("blur(4px)");midpoint["fill"]=json!("rgb(127.5 0 127.5)");midpoint["stroke-width"]=json!(4);
        assert_eq!(drawing(&animated,1.),drawing(&doc(midpoint,json!([])),1.));
        for time in [0.,0.5,1.,1.,2.,0.] {
            let frame=animated.snapshot(&vork_time::Request::Numeric(time)).unwrap();
            let nodes:Vec<_>=animated.items.iter().enumerate().map(|(i,n)|frame.nodes[i].as_deref().unwrap_or(&n.node)).collect();
            let mut retained=animated.retained.borrow_mut();
            match retained.as_mut(){Some(r)=>r.update(&animated.page,&nodes,&frame.values).unwrap(),None=>*retained=Some(crate::canvas_draw::Retained::new(&animated.page,&nodes,&frame.values).unwrap())};
            retained.as_ref().unwrap().assert_fresh(&animated.page,&nodes,&frame.values);
        }
    }
    #[test]
    fn keyed_clip_geometry_updates_every_reference() {
        let animated=doc(json!({"name":"root","children":[
            {"name":"clip","type":"path","d":"M0 0H10V10H0Z"},
            {"name":"a","type":"path","d":"M0 0H40V40H0Z","fill":"red","clip-path":"clip"},
            {"name":"b","type":"path","d":"M0 0H40V40H0Z","fill":"blue","clip-path":"clip","translation":[50,0,0]}
        ]}),json!([{"path":["nodes","clip","d"],"keys":[[0,"M0 0H10V10H0Z"],[2,"M0 0H30V30H0Z"]]}]));
        for time in [0.,0.5,1.,2.,0.] {
            let frame=animated.snapshot(&vork_time::Request::Numeric(time)).unwrap();
            let nodes:Vec<_>=animated.items.iter().enumerate().map(|(i,n)|frame.nodes[i].as_deref().unwrap_or(&n.node)).collect();
            let mut retained=animated.retained.borrow_mut();
            match retained.as_mut(){Some(r)=>r.update(&animated.page,&nodes,&frame.values).unwrap(),None=>*retained=Some(crate::canvas_draw::Retained::new(&animated.page,&nodes,&frame.values).unwrap())};
            retained.as_ref().unwrap().assert_fresh(&animated.page,&nodes,&frame.values);
        }
    }

    #[test]
    fn disabled_filter_layer_restores_authored_filter() {
        let scene=Scene::from_json(&serde_json::to_vec(&json!({"version":2,"composition":{"space":"2d"},"nodes":[{"name":"n","type":"path","d":"M0 0H20V20H0Z","fill":"red"}],"animation":[
            {"tracks":[{"path":["animation","effect","weight"],"keys":[[0,1],[1,0]]},{"path":["nodes","n","fill-rule"],"keys":[[0,"evenodd"]]}]},
            {"name":"effect","tracks":[{"path":["nodes","n","filter"],"keys":[[0,"blur(4px)"]]}]}
        ]})).unwrap()).unwrap();
        let animated=Document::new(&scene,Path::new(".")).unwrap();
        let rest=doc(json!({"name":"n","type":"path","d":"M0 0H20V20H0Z","fill":"red","fill-rule":"evenodd"}),json!([]));
        let _=drawing(&animated,0.);
        assert_eq!(drawing(&animated,1.),drawing(&rest,1.));
    }
    #[test]
    fn text_settings_start_from_typed_defaults() {
        let font=Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        let node=json!({"name":"n","type":"text","content":"Keys","font":font});
        let animated=doc(node.clone(),json!([
            {"path":["nodes","n","font-variation-settings"],"keys":[[0,"normal"],[2,"normal"]]},
            {"path":["nodes","n","font-feature-settings"],"keys":[[0,"normal"],[2,"normal"]]}
        ]));
        assert_eq!(drawing(&animated,1.),drawing(&doc(node,json!([])),1.));
    }
    #[test]
    fn text_node_and_glyph_gradients_use_interpolated_stops() {
        let font=Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        let node=json!({"name":"n","type":"text","content":"A","font":font,"fill":"linear-gradient(red, blue)"});
        let keys=json!([[0,"linear-gradient(red, blue)"],[2,"linear-gradient(blue, red)"]]);
        let animated=doc(node.clone(),json!([{"path":["nodes","n","fill"],"keys":keys}]));
        let _=animated.snapshot(&vork_time::Request::Numeric(1.)).unwrap();
        let styles=animated.styles.borrow();
        let text=styles[0].as_ref().unwrap().text.as_ref().unwrap();
        let expected=crate::canvas_draw::gradient_paint("linear-gradient(rgb(127.5 0 127.5), rgb(127.5 0 127.5))",[0.,0.,text.size[0],text.size[1]]).unwrap();
        assert_eq!(text.fill[0],PaintSource::Moda(expected.clone()));
        let glyph=doc(node,json!([
            {"path":["nodes","n","glyphs",0,"fill"],"keys":[[0,"linear-gradient(red 0px, blue 10px)"],[2,"linear-gradient(blue 0px, red 30px)"]]},
            {"path":["nodes","n","font-size"],"keys":[[0,16],[2,32]]}
        ]));
        let frame=glyph.snapshot(&vork_time::Request::Numeric(1.)).unwrap();
        fn paints(parts:&[crate::canvas_draw::TextPart])->Vec<moda_scene::value::Value> {
            parts.iter().flat_map(|part|match part {
                crate::canvas_draw::TextPart::Paint{paint,..}=>vec![paint.clone()],
                crate::canvas_draw::TextPart::Group{children,..}=>paints(children),
                _=>Vec::new(),
            }).collect()
        }
        let styles=glyph.styles.borrow();let text=styles[0].as_ref().unwrap().text.as_ref().unwrap();
        let expected=crate::canvas_draw::gradient_paint("linear-gradient(rgb(127.5 0 127.5) 0px, rgb(127.5 0 127.5) 20px)",[0.,0.,text.size[0],text.size[1]]).unwrap();
        let mut actual=paints(&frame.values[0].text_composition);
        assert!((actual[0]["stops"][1]["offset"].as_f64().unwrap()-expected["stops"][1]["offset"].as_f64().unwrap()).abs()<1e-12);
        actual[0]["stops"][1]["offset"]=expected["stops"][1]["offset"].clone();
        assert_eq!(actual,vec![expected]);
    }
    #[test]
    fn gradients_interpolate_stops_and_node_clocks_drive_styles() {
        let node=json!({"name":"n","type":"path","time":{"speed":2},"d":"M0 0H100V100H0Z","fill":"linear-gradient(red, blue)",
            "animation":[{"tracks":[{"path":["fill"],"keys":[[0,"linear-gradient(red, blue)"],[2,"linear-gradient(blue, red)"]]}]}]});
        let animated=doc(node,json!([]));
        let static_doc=doc(json!({"name":"n","type":"path","d":"M0 0H100V100H0Z","fill":"linear-gradient(rgb(127.5 0 127.5), rgb(127.5 0 127.5))"}),json!([]));
        assert_eq!(drawing(&animated,0.5),drawing(&static_doc,0.5));
    }

    #[test]
    fn fading_discrete_styles_use_the_typed_rest_values() {
        for explicit in [false, true] {
            let styles = [
                ("mask-type", "luminance", "alpha"),
                ("mix-blend-mode", "normal", "multiply"),
                ("isolation", "auto", "isolate"),
            ];
            let mut node = json!({"name":"n","type":"path","d":"M0 0H20V20H0Z","fill":"red"});
            let mut keyed = node.clone();
            let mut tracks = Vec::new();
            for (property, rest, sampled) in styles {
                if explicit { node[property] = json!(rest); }
                keyed[property] = json!(sampled);
                tracks.push(json!({
                    "path":["nodes","n",property],"interpolation":"STEP","keys":[[0,sampled]]
                }));
            }
            let scene = Scene::from_json(&serde_json::to_vec(&json!({
                "version":2,"composition":{"space":"2d"},"nodes":[node],
                "animation":[
                    {"tracks":[{"path":["animation","effect","weight"],"keys":[[0,1],[1,0]]}]},
                    {"name":"effect","tracks":tracks}
                ]
            })).unwrap()).unwrap();
            let animated = Document::new(&scene, Path::new(".")).unwrap();
            let at_rest = doc(node, json!([]));
            let at_key = doc(keyed, json!([]));
            for (time, expected) in [(0., &at_key), (0.25, &at_key), (0.75, &at_rest),
                (1., &at_rest), (0.75, &at_rest), (0.25, &at_key)] {
                assert_eq!(drawing(&animated, time), drawing(expected, time), "explicit={explicit}, time={time}");
                let snapshot = animated.snapshot(&vork_time::Request::Numeric(time)).unwrap();
                let actual = snapshot.nodes[0].as_deref().unwrap_or(&animated.items[0].node);
                let expected = &expected.items[0].node;
                assert_eq!((actual.mask_type, actual.blend, actual.isolation),
                    (expected.mask_type, expected.blend, expected.isolation));
            }
        }
    }
