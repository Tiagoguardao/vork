    use super::*;

    fn document(content: &str, extra: serde_json::Value) -> Result<Document> {
        let mut node = serde_json::json!({"name":"title","type":"text","content":content,
            "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf","font-size":40});
        node.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2,"composition":{"space":"2d"},"nodes":[node]
        }))?)?;
        Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))
    }

    const WIDE: &str = "../moda-text/tests/fonts/colour/Wide.ttf";
    const NARROW: &str = "../moda-text/tests/fonts/colour/Narrow.ttf";

    fn emoji_doc(nodes: serde_json::Value, animation: serde_json::Value) -> Result<Document> {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"}, "nodes":nodes, "animation":animation
        }))?)?;
        Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))
    }

    fn emoji_ops(doc: &Document, scale: f64) -> Result<Vec<moda_text::vector::VectorEmoji>> {
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let mut emojis = Vec::new();
        for draw in frame.batches() {
            if let Some(batch) = doc.fonts.build(draw, 0., [0., 0., 2048., 2048.], scale)? {
                emojis.extend(batch.vector.as_ref().unwrap().emojis.iter().cloned());
            }
        }
        Ok(emojis)
    }

    #[test]
    fn emoji_font_order_follows_each_node_and_run_into_the_raster() -> Result<()> {
        use serde_json::json;
        for (first, second, advances, colours) in [
            (WIDE, NARROW, [64.,32.], [[255,0,0],[0,0,255]]),
            (NARROW, WIDE, [32.,64.], [[0,0,255],[255,0,0]]),
        ] {
            let doc = emoji_doc(json!([
                {"type":"text","name":"one","font":[first,second],"font-size":64,"content":"✅","stroke":"black","stroke-width":3},
                {"type":"text","name":"two","font":[second,first],"font-size":64,"content":"✅","stroke":"black","stroke-width":3},
                {"type":"text","name":"runs","font":first,"font-size":64,"content":[
                    {"font":[first,second],"content":"✅"},{"font":[second,first],"content":"✅"}]}
            ]), json!([]))?;
            for (item, width) in doc.items.iter().zip([advances[0], advances[1], advances.iter().sum()]) {
                assert_eq!(item.text.as_ref().unwrap().layout.metrics[0][0], width);
            }
            let ops = emoji_ops(&doc, 1.)?;
            let engine = doc.fonts.host.canvas_engine();
            let strokes: Vec<_> = ops.iter().filter(|op| op.effect.is_some())
                .map(|op| op.image(engine).unwrap().width).collect();
            assert_eq!(strokes.len(), 2);
            assert_eq!(strokes[0].cmp(&strokes[1]), advances[0].partial_cmp(&advances[1]).unwrap());
            let fills: Vec<_> = ops.iter().filter(|op| op.effect.is_none()).collect();
            assert_eq!(fills.len(), 4);
            for (op, rgb) in fills.iter().zip(colours.into_iter().cycle()) {
                let run = op.shaped.as_ref().unwrap();
                let cell = doc.fonts.host.canvas_engine().emoji_raster(run, op.ppem).unwrap();
                let opaque = cell.image.rgba.chunks_exact(4).find(|p| p[3] >= 250).unwrap();
                assert!(opaque[..3].iter().zip(rgb).all(|(c, expected)| c.abs_diff(expected) <= 3));
            }
        }
        Ok(())
    }

    #[test]
    fn retained_emoji_records_keep_the_selected_face_across_scale_changes() -> Result<()> {
        use serde_json::json;
        let doc = emoji_doc(json!([{
            "type":"text","name":"title","font":WIDE,"font-size":64,
            "content":[{"content":"✅","font":WIDE},{"content":"✅","font":NARROW}]
        }]), json!([{"tracks":[{"path":["nodes","title","scale"],
            "keys":[[0,[1,1,1]],[1,[3,1,1]]]}]}]))?;
        let mut face_ids = None;
        for (time, ppem) in [(0., 64), (1., 192), (0., 64)] {
            let frame = doc.snapshot(&vork_time::Request::Numeric(time))?;
            let plan = doc.plan_snapshot(&frame, &[])?;
            let mut ids = Vec::new();
            for item in &plan.items {
                let moda_engine::seams::DrawItem::Text(batch) = item else { continue };
                let vector = batch.vector.as_ref().unwrap();
                assert!(vector.uniform_padding.is_none());
                for op in &batch.ops {
                    let moda_text::ops::TextOpPipeline::Emoji { emoji } = op.pipeline else { continue };
                    let draw = &vector.emojis[emoji as usize];
                    assert_eq!(draw.ppem, ppem);
                    let run = draw.shaped.as_ref().unwrap();
                    ids.push((run.id, run.advance_em));
                }
            }
            assert_eq!(ids.len(), 2);
            assert_ne!(ids[0].0, ids[1].0);
            assert_eq!([ids[0].1, ids[1].1], [1., 0.5]);
            assert_eq!(&ids, face_ids.get_or_insert_with(|| ids.clone()));
        }
        Ok(())
    }

    #[test]
    fn emoji_base_fill_is_emitted_once_across_fill_and_stroke_passes() -> Result<()> {
        use serde_json::json;
        let doc = emoji_doc(json!([{
            "type":"text","name":"title","font":WIDE,"font-size":64,"content":"✅🟥",
            "fill":"green","stroke":"black","stroke-width":3
        }]), json!([{"tracks":[{"path":["nodes","title","glyphs",1,"fill"],"keys":[[0,"red"]]}]}]))?;
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        assert!(frame.batches().count() >= 3);
        let mut base = Vec::new();
        let mut strokes = Vec::new();
        for draw in frame.batches() {
            let mut draw = draw.clone();
            Arc::make_mut(&mut draw.spec).opacity = 0.5;
            if let Some(batch) = doc.fonts.build(&draw, 0., [0.,0.,512.,512.], 1.)? {
                for glyph in &batch.vector.as_ref().unwrap().emojis {
                    if glyph.effect.is_some() { strokes.push(glyph.cluster.clone()); }
                    else { base.push((glyph.cluster.clone(), glyph.opacity)); }
                }
            }
        }
        base.sort_by(|a,b| a.0.cmp(&b.0));
        strokes.sort();
        assert_eq!(base, [("✅".into(), 0.5), ("🟥".into(), 0.5)]);
        assert_eq!(strokes, ["✅", "🟥"]);
        Ok(())
    }

    #[test]
    fn emoji_ink_owns_the_query_and_unit_pivot_at_independent_raster_scales() -> Result<()> {
        use serde_json::json;
        let node = json!([{"type":"text","name":"title","font":WIDE,"font-size":100,"content":"✅"}]);
        let rest = emoji_doc(node.clone(), json!([]))?;
        let text = rest.items[0].text.as_ref().unwrap();
        let ink = text.layout.vector.as_ref().unwrap().code_point_boxes(1)[0].unwrap();
        let baseline = text.layout.metrics[0][1];
        for (got, expected) in ink.into_iter().zip([10.,baseline-60.,80.,baseline]) {
            assert!((got - expected).abs() <= 1., "{ink:?}, baseline {baseline}");
        }
        let bounds = rest.measure("title", "node", "bounds", &vork_time::Request::Numeric(0.),
            &mut rest.bounds_batch(["title"].into_iter()))?;
        for (value, expected) in bounds.as_array().unwrap().iter().flat_map(|xy| xy.as_array().unwrap()).zip(ink) {
            assert!((value.as_f64().unwrap() - expected).abs() < 0.01, "queried {bounds}, ink {ink:?}");
        }
        let normal = emoji_ops(&rest, 1.)?.remove(0);
        let zoomed = emoji_ops(&rest, 4.)?.remove(0);
        let engine = rest.fonts.host.canvas_engine();
        let normal_image = engine.emoji_raster(normal.shaped.as_ref().unwrap(), normal.ppem).unwrap();
        let zoomed_image = engine.emoji_raster(zoomed.shaped.as_ref().unwrap(), zoomed.ppem).unwrap();
        assert!(zoomed_image.image.width >= 3 * normal_image.image.width);
        assert!(f64::from(normal.corners[0]) < ink[0]);
        let moving = emoji_doc(node, json!([{"tracks":[
            {"path":["nodes","title","glyphs",0,"scale"], "keys":[[0,[2,2,1]]]}
        ]}]))?;
        let moved = emoji_ops(&moving, 1.)?.remove(0);
        let cx = (ink[0] + ink[2]) / 2.;
        let cy = (ink[1] + ink[3]) / 2.;
        for (before, after) in normal.corners.chunks_exact(2).zip(moved.corners.chunks_exact(2)) {
            assert!((f64::from(after[0]) - (2.*f64::from(before[0])-cx)).abs() <= 2.);
            assert!((f64::from(after[1]) - (2.*f64::from(before[1])-cy)).abs() <= 2.);
        }
        Ok(())
    }

    #[test]
    fn emoji_nonpack_sequences_and_positioned_colour_glyphs() -> Result<()> {
        use serde_json::json;
        let tag = "🏴\u{e0078}\u{e007f}";
        // The first font has the bare flag. Only the second covers the tag sequence.
        let ordered = emoji_doc(json!([{"type":"text","name":"title","font":[NARROW,WIDE],
            "font-size":100,"content":tag}]), json!([]))?;
        let ordered_ops = emoji_ops(&ordered, 1.)?;
        assert_eq!(ordered_ops[0].shaped.as_ref().unwrap().advance_em, 1.);
        for cluster in ["🇽🇽", tag, "✅‍🟥"] {
            let doc = emoji_doc(json!([{"type":"text","name":"title","font":WIDE,"font-size":100,
                "content":cluster}]), json!([]))?;
            let ops = emoji_ops(&doc, 1.)?;
            assert_eq!(ops.len(), 1);
            let glyph = &ops[0];
            assert_eq!(glyph.cluster, cluster);
            let run = glyph.shaped.as_ref().unwrap();
            let [x0,y0,x1,y1] = run.ink_em;
            if cluster == "✅‍🟥" {
                // Two marks, positioned by GPOS. Their ink extends beyond one em.
                assert!((run.advance_em - 1.875).abs() < 0.001, "{run:?}");
                assert!((x1 - x0 - 1.575).abs() < 0.02, "{run:?}");
            } else {
                // The custom flag ligature is half as tall as the bare flag.
                assert!((y1 - y0 - 0.3).abs() < 0.02, "{run:?}");
            }
        }
        Ok(())
    }

    #[test]
    fn text_shadows_have_a_capable_owner() -> Result<()> {
        for (content, font) in [("مرحبا", "NotoSansArabic-Regular.ttf"), ("AB", "LiberationSans-Regular.ttf")] {
            let doc = document(content, serde_json::json!({
                "font": format!("../vork-std/tests/fonts/{font}"), "filter":"drop-shadow(8px 4px 0px red)"
            }))?;
            let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
            let nodes = doc.items.iter().map(|item| &item.node).collect::<Vec<_>>();
            let lowered = crate::canvas_draw::document(&doc.page, &nodes, &frame.values)?;
            // Vector and raster text alike: the group's captured picture casts the shadow, from the text's own alpha.
            let owner = lowered["nodes"].as_array().unwrap().iter().find(|n| n["effects"].as_array().is_some_and(|fx| !fx.is_empty())).unwrap();
            assert_eq!(owner["type"], "group", "{content}: {lowered}");
            assert_eq!(owner["effects"][0]["type"], "filter-shadow", "{content}: {lowered}");
        }
        Ok(())
    }

    #[test]
    fn text_units_preserve_unicode_spaces_and_blank_lines() -> Result<()> {
        for content in ["A\u{a0}\u{a0}B", "\u{2007}A\u{202f}", "A\n\nB"] {
            let doc = document(content, serde_json::json!({"white-space":"pre-line"}))?;
            let text = doc.items[0].text.as_ref().unwrap();
            assert_eq!(text.code_points, content.chars().count(), "{content:?}");
            assert_eq!(text.units[&TextUnit::Glyphs].len(), v2::unit_ranges(content, TextUnit::Glyphs).len(), "{content:?}");
        }
        Ok(())
    }

    #[test]
    fn preserved_unicode_glyph_targets_reach_the_draw() -> Result<()> {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2,"composition":{"space":"2d"},
            "nodes":[{"name":"title","type":"text","content":"A\u{a0}\u{a0}B",
                "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf","font-size":40}],
            "animation":[{"tracks":[{"path":["nodes","title","glyphs",3,"translation"],"keys":[[0,[30,0,0]]]}]}]
        }))?)?;
        let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let first = frame.batches().next().unwrap();
        assert_eq!(first.spec.unit_transforms[3].off[0], 30.);
        assert_eq!(first.spec.unit_transforms[2].off[0], 0.);
        Ok(())
    }

    #[test]
    fn sanitized_controls_do_not_lose_the_following_ink_box() -> Result<()> {
        let content = format!("A{}B", "\u{2060}".repeat(16));
        let doc = document(&content, serde_json::json!({}))?;
        let boxes = doc.items[0].text.as_ref().unwrap().layout.vector.as_ref().unwrap().code_point_boxes(content.chars().count());
        assert!(boxes[17].is_some(), "B keeps its authored character index after sanitization");
        Ok(())
    }

    #[test]
    fn variable_weight_ink_boxes_use_the_drawn_instance() -> Result<()> {
        let make = |extra: serde_json::Value| {
            let mut node = serde_json::json!({"font":"../moda-text/tests/fonts/NotoSans-WeightSubset.ttf"});
            node.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            document("Hamburgefonts", node)
        };
        let regular = make(serde_json::json!({}))?;
        let heavy = make(serde_json::json!({"font-weight":800}))?;
        let explicit = make(serde_json::json!({"font-variation-settings":"'wght' 800"}))?;
        let boxes = |doc: &Document| doc.items[0].text.as_ref().unwrap().layout.vector.as_ref().unwrap().code_point_boxes("Hamburgefonts".chars().count());
        assert_ne!(boxes(&regular), boxes(&heavy));
        assert_eq!(boxes(&heavy), boxes(&explicit));
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and a colour emoji font"]
    fn canvas_renders_colour_emoji_in_the_measured_text() -> Result<()> {
        let renderer = Renderer::new()?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        for (cluster, content) in [
            ("🙂", serde_json::json!("A🙂B")),
            ("👨‍👩‍👧‍👦", serde_json::json!("A👨‍👩‍👧‍👦B")),
            ("👍🏽", serde_json::json!("A👍🏽B")),
            ("🇺🇸", serde_json::json!("A🇺🇸B")),
            ("1️⃣", serde_json::json!([{"content":"A1","fill":"black"},{"content":"️⃣B","fill":"white"}])),
            ("©️", serde_json::json!([{"content":"A©","fill":"black"},{"content":"\u{2060}️B","fill":"white"}])),
        ] {
            let doc = document("", serde_json::json!({"content":content}))?;
            let text = doc.items[0].text.as_ref().unwrap();
            let boxes = text.layout.vector.as_ref().unwrap().code_point_boxes(text.code_points);
            let [left, top, right, bottom] = boxes[1].expect("the emoji has a measured box");
            assert!(right > left && bottom > top);
            let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
            let texture = engine.render_to_texture(&doc.plan_snapshot(&frame, &[])?)?;
            let (w, h) = (texture.width(), texture.height());
            let rgba = moda_engine::readback::read_texture_rgba(&renderer.device, &renderer.queue, &texture, w, h)?;
            let mut ink = false;
            let mut chromatic = false;
            for y in top.max(0.) as u32..(bottom.ceil() as u32).min(h) {
                for x in left.max(0.) as u32..(right.ceil() as u32).min(w) {
                    let pixel = &rgba[((y * w + x) * 4) as usize..][..4];
                    if pixel[3] > 100 {
                        ink |= pixel[..3].iter().any(|&channel| (20..230).contains(&channel));
                        chromatic |= pixel[..3].iter().max().unwrap() - pixel[..3].iter().min().unwrap() > 20;
                    }
                }
            }
            assert!(ink, "{cluster} has no ink in its box");
            if matches!(cluster, "🙂" | "👍🏽" | "🇺🇸" | "1️⃣") {
                assert!(chromatic, "{cluster} lost its colours");
            }

        }
        Ok(())
    }

    #[test]
    fn text_presentation_across_styled_runs_stays_text() {
        let catalog = moda_text::catalog::Catalog::from_catalog_json(&serde_json::json!({}));
        let paragraphs = moda_text::parser::parse_rich_text_to_paragraphs(
            "<p><span style='color: blue'>A✈</span><span style='color: red'>\u{fe0e}B</span></p>"
        );
        let runs = moda_text::runs::map_paragraphs_to_msdf_runs(&paragraphs, &catalog);
        assert!(runs.iter().all(|run| run.kind == moda_text::runs::RunKind::Text), "{runs:?}");
        assert_eq!(runs.iter().map(|run| run.text.as_str()).collect::<String>(), "A✈B");
        for run in &runs {
            assert_eq!(run.color, if run.text.contains('B') { "red" } else { "blue" });
        }
    }

    #[test]
    fn emoji_classification_ignores_style_boundaries() {
        use moda_text::{emoji::Segment, runs::RunKind};
        let catalog = moda_text::catalog::Catalog::from_catalog_json(&serde_json::json!({}));
        for text in ["A✈\u{fe0e}B", "©\u{fe0f}", "1\u{fe0f}\u{20e3}", "👍🏽", "👩\u{200d}💻", "🇺🇸", "🏴\u{e0067}\u{e0062}\u{e007f}"] {
            let expected = moda_text::emoji::split_text_into_segments(text);
            for split in text.char_indices().skip(1).map(|(i, _)| i) {
                let paragraphs = moda_text::parser::parse_rich_text_to_paragraphs(&format!(
                    "<p><span style='color: blue'>{}</span><span style='color: red'>{}</span></p>",
                    &text[..split], &text[split..]
                ));
                let runs = moda_text::runs::map_paragraphs_to_msdf_runs(&paragraphs, &catalog);
                let mut actual = Vec::new();
                for run in runs {
                    match (run.kind, actual.last_mut()) {
                        (RunKind::Text, Some(Segment::Text(previous))) => previous.push_str(&run.text),
                        (RunKind::Text, _) => actual.push(Segment::Text(run.text)),
                        (RunKind::Emoji, _) => {
                            assert_eq!(run.color, "blue", "emoji uses its base's style");
                            actual.push(Segment::Emoji(run.text));
                        }
                    }
                }
                let mut canonical = Vec::new();
                for segment in &expected {
                    match (segment, canonical.last_mut()) {
                        (Segment::Text(text), Some(Segment::Text(previous))) => previous.push_str(text),
                        _ => canonical.push(segment.clone()),
                    }
                }
                assert_eq!(actual, canonical, "{text:?} split at byte {split}");
            }
        }
    }

    #[test]
    fn missing_glyph_refusal_identifies_character() {
        let error = match document("A \u{10ffff} B", serde_json::json!({})) {
            Ok(_) => panic!("missing glyph must refuse instead of drawing notdef"),
            Err(error) => format!("{error:#}"),
        };
        assert_eq!(error.lines().count(), 1, "{error}");
        for part in ["node \"title\"", "\\u{10ffff}"] {
            assert!(error.contains(part), "{error}");
        }
        assert!(!error.contains("lane"), "{error}");
    }

    #[test]
    fn complex_text_stroke_refuses_without_internal_lane_details() {
        let error = match document("مرحبا", serde_json::json!({
            "font":"../vork-std/tests/fonts/NotoSansArabic-Regular.ttf", "stroke":"red","stroke-width":1
        })) {
            Ok(_) => panic!("complex text strokes are not built"),
            Err(error) => format!("{error:#}"),
        };
        assert!(error.contains("node \"title\""), "{error}");
        assert!(error.contains("not built yet"), "{error}");
        assert_eq!(error.lines().count(), 1, "{error}");
        assert!(!error.contains("Moda") && !error.contains("tier"), "{error}");
    }
