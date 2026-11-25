    use super::*;

    #[test]
    fn scrolling_rich_text_through_path_clips_plans_without_refusal() -> Result<()> {
        // Reduced from the author-trial feed: nested scrolling rows, rich
        // spans, and an SVG clip whose bounds do not begin at the origin.
        for path in ["M0 80 h960 v460 h-960 Z", "M0 80 H960 L0 540 Z"] {
            let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                "version":2, "composition":{"space":"2d"},
                "render":{"resolution":[960,540], "duration":4},
                "nodes":[
                    {"type":"path", "name":"feed-clip", "d":path, "fill":"white"},
                    {"name":"feed", "clip-path":"feed-clip", "children":[
                        {"name":"scroll", "children":[
                            {"name":"row", "children":[
                                {"type":"text", "content":[{"content":"starred "}, {"content":"codehike"}],
                                    "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf",
                                    "font-size":16, "translation":[108,49,0], "fill":"black"}
                            ]}
                        ]}
                    ]}
                ],
                "animation":[{"tracks":[{"path":["nodes","scroll","translation"],
                    "keys":[[0,[0,-100,0]],[1,[0,50,0]],[2.9,[0,490,0]],[3,[0,50,0]]]}]}]
            }))?)?;
            let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
            for time in [2.9, 1., 0., 3., 2.9] {
                let frame = doc.snapshot(&vork_time::Request::Numeric(time))?;
                doc.plan_snapshot(&frame, &[]).with_context(|| format!("clip {path}, time {time}"))?;
            }
        }
        Ok(())
    }

    #[test]
    fn text_paints_and_geometry_clips_use_the_host_layout() -> Result<()> {
        for fill in ["none", "linear-gradient(to right, red, blue)", "conic-gradient(red, blue)"] {
            let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                "version": 2, "composition": {"space": "2d"},
                "render": {"resolution": [512, 256]},
                "nodes": [
                    {"type": "path", "d": "M0 0H512V256H0Z", "fill": "lime", "clip-path": "title"},
                    {"type": "text", "name": "title", "content": "AB", "font-size": 80,
                        "font": "../vork-std/tests/fonts/LiberationSans-Regular.ttf", "fill": fill,
                        "stroke": "red", "stroke-width": 8, "opacity": 0.2},
                    {"type":"text", "name":"other", "content":"AB", "font-size":80,
                        "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf", "fill":"blue"}
                ],
                "animation": [{"tracks": [
                    {"path": ["nodes", "title", "glyphs", 1, "translation"], "keys": [[0, [300, 0, 0]]]},
                    {"path": ["nodes", "title", "glyphs", 1, "opacity"], "keys": [[0, 0.4]]},
                    {"path": ["nodes", "other", "glyphs", 1, "translation"], "keys": [[0, [-50, 0, 0]]]},
                    {"path": ["nodes", "other", "glyphs", 1, "fill"], "keys": [[0, "green"]]}
                ]}]
            }))?)?;
            let document = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
            let frame = document.snapshot(&vork_time::Request::Numeric(0.))?;
            let clip = frame.batches().find(|draw| draw.item == 1 && draw.sub == frame.values[1].texts).unwrap();
            assert_eq!(clip.spec.unit_transforms[1].off, [300., 0.]);
            assert_eq!(clip.spec.fill_override.as_deref(), Some("white"));
            assert!(clip.spec.stroke_color.is_none());
            assert!(clip.spec.unit_transforms.iter().all(|x| x.opacity == 1.));
            let plan = document.plan_snapshot(&frame, &[])?;
            assert!(plan.masks.iter().any(|mask| mask.size[0] > 300.), "moved glyphs must extend the capture beyond the layout box");
        }
        Ok(())
    }

    #[test]
    fn fractional_text_unit_opacity_composes_fill_and_stroke() -> Result<()> {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"));
        for level in ["glyphs", "words", "lines"] {
            for content in [serde_json::json!("I"), serde_json::json!([
                {"content":"A", "stroke":"red"}, {"content":"B", "stroke":"blue"}
            ])] {
                for opacity in [0., 0.5, 1.] {
                    let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                        "version":2, "composition":{"space":"2d"},
                        "nodes":[{"type":"text", "name":"title", "content":content,
                            "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf",
                            "fill":"black", "stroke":"black", "stroke-width":10}],
                        "animation":[{"tracks":[{"path":["nodes","title",level,0,"opacity"],
                            "keys":[[0,opacity]]}]}]
                    }))?)?;
                    let doc = Document::new(&scene, base)?;
                    let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
                    assert_eq!(!frame.values[0].text_composition.is_empty(), opacity == 0.5);
                    if opacity == 0.5 {
                        use crate::canvas_draw::TextPart;
                        let TextPart::Group { opacity, children } = &frame.values[0].text_composition[0] else {
                            panic!("the faded {level} must compose as a group");
                        };
                        assert_eq!(*opacity, 0.5);
                        assert!(!children.is_empty());
                        assert!(frame.batches().flat_map(|d| d.spec.unit_transforms.iter())
                            .all(|x| x.opacity == 0. || x.opacity == 1.));
                        let nodes: Vec<_> = doc.items.iter().map(|item| &item.node).collect();
                        let document = crate::canvas_draw::document(&doc.page, &nodes, &frame.values)?;
                        assert!(document["nodes"].as_array().unwrap().iter().any(|node|
                            node["type"] == "group" && node["opacity"] == 0.5));
                        doc.plan_snapshot(&frame, &[])?;
                    }
                }
            }
        }
        for (fill, stroke, width) in [("black", "none", 10), ("none", "black", 10), ("black", "black", 0),
            ("transparent", "black", 10), ("black", "transparent", 10)] {
            let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                "version":2, "composition":{"space":"2d"},
                "nodes":[{"type":"text", "name":"title", "content":"I",
                    "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf",
                    "fill":fill, "stroke":stroke, "stroke-width":width}],
                "animation":[{"tracks":[{"path":["nodes","title","glyphs",0,"opacity"],"keys":[[0,0.5]]}]}]
            }))?)?;
            assert!(Document::new(&scene, base)?.snapshot(&vork_time::Request::Numeric(0.))?.values[0].text_composition.is_empty());
        }
        Ok(())
    }

    #[test]
    fn equal_text_unit_opacities_keep_independent_pictures() -> Result<()> {
        use crate::canvas_draw::TextPart;
        for (level, content) in [("glyphs", "HH"), ("words", "H H"), ("lines", "H H")] {
            let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                "version":2, "composition":{"space":"2d"},
                "nodes":[{"type":"text", "name":"title", "content":content,
                    "white-space":"pre-line", "font-size":100,
                    "width":if level == "lines" { Some(100) } else { None::<u32> },
                    "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf",
                    "fill":"white", "stroke":"white", "stroke-width":4}],
                "animation":[{"tracks":[
                    {"path":["nodes","title",level,0,"opacity"], "keys":[[0,0.5]]},
                    {"path":["nodes","title",level,1,"opacity"], "keys":[[0,0.5]]},
                    {"path":["nodes","title",level,1,"translation"],
                        "keys":[[0,if level == "lines" { [0,-115,0] } else { [-70,0,0] }]]}
                ]}]
            }))?)?;
            let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
            let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
            let groups: Vec<_> = frame.values[0].text_composition.iter().filter_map(|part| {
                if let TextPart::Group { opacity, children } = part { Some((*opacity, children)) } else { None }
            }).collect();
            // Two overlapping solid pictures at 0.5 must source-over to 0.75, not 0.5.
            assert_eq!(groups.len(), 2, "separately faded {level} need independent pictures");
            for (opacity, children) in groups {
                assert_eq!(opacity, 0.5);
                assert!(!children.is_empty());
            }
            doc.plan_snapshot(&frame, &[])?;
        }
        Ok(())
    }

    #[test]
    fn unit_opacity_batches_share_layout_payloads() -> Result<()> {
        // Independent opacity pictures must not duplicate the full layout payloads.
        let count = 512;
        for equal in [true, false] {
            let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                "version":2, "composition":{"space":"2d"},
                "nodes":[{"type":"text", "name":"title", "content":"A".repeat(count),
                    "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf", "font-size":12,
                    "width":1000, "fill":"white", "stroke":"black", "stroke-width":1}],
                "animation":[{"tracks": (0..count).map(|i| serde_json::json!({
                    "path":["nodes","title","glyphs",i,"opacity"],
                    "keys":[[0,if equal || i % 2 == 0 { 0.5 } else { 0.25 }]]
                })).collect::<Vec<_>>()}]
            }))?)?;
            let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
            let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
            assert_eq!(frame.batches().count(), count);
            let first = &frame.batches().next().unwrap().spec;
            assert_eq!(first.unit_transforms.len(), count);
            for draw in frame.batches() {
                assert!(Arc::ptr_eq(&first.html, &draw.spec.html));
                assert!(Arc::ptr_eq(&first.unit_transforms, &draw.spec.unit_transforms));
                assert!(Arc::ptr_eq(&first.unit_groups, &draw.spec.unit_groups));
                assert!(Arc::ptr_eq(&first.unit_fill, &draw.spec.unit_fill));
            }
            assert_eq!(frame.batches().map(|d| d.spec.unit_range.as_ref().unwrap().len()).sum::<usize>(), count);
        }
        Ok(())
    }

    #[test]
    fn nested_text_opacity_and_seeks_keep_composition() -> Result<()> {
        use crate::canvas_draw::TextPart;
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"},
            "nodes":[{"type":"text", "name":"title", "content":"AB CD", "font-size":80,
                "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf",
                "fill":"white", "stroke":"black", "stroke-width":8},
                {"type":"text", "name":"steady", "content":"AB CD", "font-size":80,
                    "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf",
                    "fill":"blue", "stroke":"red", "stroke-width":4}],
            "animation":[{"tracks":[
                {"path":["nodes","steady","glyphs",0,"translation"], "keys":[[0,[25,0,0]]]},
                {"path":["nodes","steady","words",0,"opacity"], "keys":[[0,0.8]]},
                {"path":["nodes","title","glyphs",0,"translation"], "keys":[[0,[0,0,0]],[1,[10,0,0]]]},
                {"path":["nodes","title","lines",0,"opacity"], "keys":[[0,0.5],[1,1]]},
                {"path":["nodes","title","words",0,"opacity"], "keys":[[0,0.4],[1,1]]},
                {"path":["nodes","title","glyphs",0,"opacity"], "keys":[[0,0.3],[1,1]]},
                {"path":["nodes","title","glyphs",1,"fill"], "keys":[[0,"red"]]},
                {"path":["nodes","title","glyphs",1,"stroke"], "keys":[[0,"blue"]]}
            ]}]
        }))?)?;
        let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
        let nodes: Vec<_> = doc.items.iter().map(|item| &item.node).collect();
        let mut steady_spec = None;
        for time in [0., 0., 0.5, 1., 0.5, 0.] {
            let frame = doc.snapshot(&vork_time::Request::Numeric(time))?;
            let steady = frame.batches().find(|draw| draw.item == 1).unwrap();
            if let Some(previous) = &steady_spec { assert!(Arc::ptr_eq(previous, &steady.spec)); }
            steady_spec = Some(steady.spec.clone());
            assert_eq!(frame.values[1].text_revision, 1);
            assert_eq!(steady.spec.unit_transforms[0].off, [25., 0.]);
            let title = frame.batches().find(|draw| draw.item == 0).unwrap();
            assert!((title.spec.unit_transforms[0].off[0] - time as f32 * 10.).abs() < 1e-5);
            if time == 0. {
                let TextPart::Group { opacity: line, children } = &frame.values[0].text_composition[0] else { panic!("line") };
                let TextPart::Group { opacity: word, children } = &children[0] else { panic!("word") };
                let TextPart::Group { opacity: glyph, .. } = &children[0] else { panic!("glyph") };
                for (actual, expected) in [(*line, 0.5), (*word, 0.4), (*glyph, 0.3)] {
                    assert!((actual - expected).abs() < 1e-6);
                }
            }
            assert_eq!(frame.values[0].text_composition.is_empty(), time == 1.);
            doc.plan_snapshot(&frame, &[])?;
            doc.retained.borrow().as_ref().unwrap().assert_fresh(&doc.page, &nodes, &frame.values);
        }
        Ok(())
    }

    #[test]
    fn css_text_paints_reach_moda_as_rgba() -> Result<()> {
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests");
        for paint in ["rgb(100%, 0%, 0%)", "rgb(255 0 0)", "RGB(255 0 0 / 100%)",
            "hsl(1turn 100% 50%)", "hwb(0 0% 0%)", "red", "#f00"] {
            for content in [serde_json::json!("Hello"), serde_json::json!([{"content":"Hello", "fill":paint,"stroke":paint}])] {
                let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                    "version": 2, "composition": {"space": "2d"}, "nodes": [{
                        "type": "text", "name":"title", "content": content, "font": "fonts/LiberationSans-Regular.ttf",
                        "fill": paint, "stroke": paint
                    }], "animation":[{"tracks":[{"path":["nodes","title","glyphs",0,"fill"],"keys":[[0,paint]]}]}]
                }))?)?;
                let document = Document::new(&scene, &base)?;
                let text = document.items[0].text.as_ref().unwrap();
                assert!(text.fill.iter().all(|p| *p == PaintSource::Color([1., 0., 0., 1.])), "{paint}");
                assert_eq!(text.stroke, text.fill, "{paint}");
                // The colours reach Moda with the draw, not in the layout's HTML.
                assert!(text.spec.run_colors.iter().all(|colour| *colour == [1., 0., 0., 1.]), "{paint}: {:?}", text.spec.run_colors);
                assert!(!text.spec.html.contains("color:"), "{paint}: {}", text.spec.html);
                assert!(document.snapshot(&vork_time::Request::Numeric(0.))?.batches().next().is_some());
            }
        }
        Ok(())
    }

    #[test]
    fn missing_canvas_still_loads_magenta_but_corrupt_still_refuses() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let scene = Scene::from_json(br#"{
            "version":2, "composition":{"space":"2d"}, "render":{"resolution":[16,16]},
            "images":{"photo":{"file":"missing.png"}},
            "nodes":[{"type":"path","d":"M0 0H16V16H0Z","fill":{"image":"photo","fit":"fill"}}]
        }"#)?;
        let warning = crate::warnings::Collector::quiet();
        let _scope = warning.enter();
        let document = Document::new(&scene, dir.path())?;
        document.install_images()?;
        let frame = document.snapshot(&vork_time::Request::Numeric(0.))?;
        let nodes: Vec<_> = document.items.iter().map(|item| &item.node).collect();
        let drawing = crate::canvas_draw::document(&document.page, &nodes, &frame.values)?;
        let source = drawing["nodes"][0]["fills"][0]["paint"]["src"].as_str()
            .context("expected an image paint")?;
        let asset = moda_assets::engine_seam::resolve(source).context("expected an installed texture")?;
        let moda_assets::engine_seam::ResolvedPixels::Heap(pixels) = asset.pixels else {
            panic!("expected decoded image pixels");
        };
        assert_eq!(&pixels[..4], &[255, 0, 255, 255]);
        assert_eq!(warning.snapshot()[0].code, "missing_image");
        std::fs::write(dir.path().join("missing.png"), b"corrupt image")?;
        let error = Document::new(&scene, dir.path()).err().expect("corrupt image must refuse");
        assert!(format!("{error:#}").contains("Invalid PNG signature"), "{error:#}");
        Ok(())
    }
