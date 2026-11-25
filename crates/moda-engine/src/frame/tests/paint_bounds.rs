use super::*;
use moda_scene::value::{Value, json};

fn plan(nodes: Vec<Value>, root: &str) -> Result<CoveragePlan> {
    let scene = Scene::load(&json!({"nodes":nodes,"pages":[{
        "id":"p","width":240,"height":200,"nodeIds":[root]
    }]}))?;
    let options = RenderOptions {
        transparent: true,
        ..present_test_options(4)
    };
    let (plan, warnings) = build_coverage_plan(&scene, 0, &options)?;
    assert!(warnings.skipped.is_empty(), "{warnings:?}");
    Ok(plan)
}

#[test]
#[ignore = "requires a working GPU adapter"]
fn emitted_paint_matches_full_target_splits() -> Result<()> {
    crate::test_fonts();
    let engine =
        Engine::new_with_frame_format(gpu("painted splits"), wgpu::TextureFormat::Rgba16Float)?;
    let triangle = json!({"id":"shape","type":"path","x":50,"y":70,"width":60,"height":75,"closed":true,
        "points":[{"type":"move","x":0,"y":75},{"type":"line","x":24,"y":0},{"type":"line","x":48,"y":75}],
        "opacity":0.5,"fills":[{"id":"base","paint":{"type":"solid","color":"#4ad"}}]});
    let mut shadow = triangle.clone();
    shadow["fills"].as_array_mut().unwrap().push(json!({
        "id":"top","paint":{"type":"solid","color":"#e81"},"effects":[{
            "id":"s","type":"drop-shadow","color":"#06f","blur":3,"offset":{"x":68,"y":0}
        }]
    }));
    let mut miter = triangle;
    miter["strokes"] = json!([{"id":"s","width":20,"join":"miter",
        "paint":{"type":"solid","color":"#eb5"},"blendMode":"screen"}]);
    let warped_text = json!({"id":"text","type":"richtext","x":20,"y":60,"width":170,"height":70,
        "distort":{"type":"matrix","matrix":[1,0,0.52,0,1,0.2,0.06,0,1]},
        "htmlContent":"<p style='font-size:28px;color:#dc4411'>Ink</p>"});
    let group = json!({"id":"g","type":"group","opacity":0.5,"children":["text"]});
    for (name, p) in [
        ("entry shadow", plan(vec![shadow], "shape")?),
        ("miter", plan(vec![miter], "shape")?),
        ("warped text", plan(vec![group, warped_text], "g")?),
    ] {
        let mut reference = p.clone();
        let mut splits = 0;
        for item in &mut reference.items {
            if let DrawItem::BackdropSplit(split) = item {
                **split = crate::backdrop::blend::isolation_split(
                    &split.node_id,
                    split.op.clone(),
                    [0., 0., 240., 200.],
                );
                splits += 1;
            }
        }
        assert!(splits >= 2, "{name} must exercise a split");
        let read = |plan| -> Result<_> {
            let texture = engine.render_plan_to_texture(plan)?;
            crate::readback::read_texture_rgba16f(
                &engine.gpu.device,
                &engine.gpu.queue,
                &texture,
                240,
                200,
            )
        };
        assert!(
            read(&p)? == read(&reference)?,
            "{name} diverged from its full-target composite"
        );
    }
    Ok(())
}

#[test]
fn filtered_group_shadow_drops_the_outer_page_clip() -> Result<()> {
    let p = plan(
        vec![
            json!({"id":"g","type":"group","children":["caster"],"effects":[{"id":"b","type":"layer-blur","radius":14}]}),
            json!({"id":"caster","type":"rectangle","x":-9,"y":70,"width":6,"height":30,
            "fills":[{"id":"f","paint":{"type":"solid","color":"#155"}}],
            "effects":[{"id":"s","type":"drop-shadow","color":"#ff1437","blur":1,"spread":8,"offset":{"x":-2,"y":3}}]}),
        ],
        "g",
    )?;
    let shadow = p
        .items
        .iter()
        .find_map(|draw| {
            let DrawItem::LayerBlurComposite { group, .. } = draw else {
                return None;
            };
            group.items.iter().find_map(|item| match item {
                DrawItem::Shadow { instance, .. } => Some(instance),
                _ => None,
            })
        })
        .expect("filtered source includes its shadow");
    assert_eq!(shadow.page_clip_bounds, INFINITE_CLIP);
    Ok(())
}

#[test]
fn empty_paint_layers_and_groups_emit_no_split() -> Result<()> {
    // The screen layer needs its own split. Half opacity also needs an outer
    // fade when the two fills overlap. At zero width neither has any paint.
    for (kind, width, opacity, expected_splits) in [
        ("rectangle", 0, 0.5, 0),
        ("rectangle", 0, 1.0, 0),
        ("path", 48, 0.5, 0),
        ("path", 48, 1.0, 0),
        ("rectangle", 48, 0.5, 4),
    ] {
        let shape = json!({"id":"bar","type":kind,"x":20,"y":20,"width":width,"height":24,
            "points":[],"opacity":opacity,"fills":[
                {"id":"a","paint":{"type":"solid","color":"#49f"}},
                {"id":"b","blendMode":"screen","paint":{"type":"solid","color":"#e82"}}
            ]});
        let scene = Scene::load(&json!({"nodes":[shape],
            "pages":[{"id":"p","width":240,"height":200,"nodeIds":["bar"]}]}))?;
        let (plan, warnings) = build_coverage_plan(&scene, 0, &RenderOptions {
            transparent: true, ..present_test_options(4)
        })?;
        let splits = plan.items.iter().filter(|i| matches!(i, DrawItem::BackdropSplit(_))).count();
        let bodies = plan.items.iter().filter(|i| matches!(i, DrawItem::Body { .. }
            | DrawItem::SolidShapeSdf { .. } | DrawItem::GradientShapeSdf { .. })).count();
        assert_eq!(bodies, if expected_splits == 0 { 0 } else { 2 }, "{kind} width {width}: {warnings:?}");
        assert_eq!(splits, expected_splits, "{kind} width {width} opacity {opacity}: {warnings:?}");
    }
    Ok(())
}

#[test]
fn off_target_stack_keeps_the_paint_that_the_parent_filter_inspects() -> Result<()> {
    let fills = json!([
        {"id":"a","paint":{"type":"solid","color":"#49f"}},
        {"id":"b","paint":{"type":"solid","color":"#e82"}}
    ]);
    // Cover both a rejected outer pair and a rejected per-entry pair.
    for stacked in [true, false] {
        let mut shape = json!({"id":"box","type":"rectangle","x":-40,"y":65,"width":34,"height":45,
            "appearanceOrder":["b","a"],"opacity":0.3,"fills":fills});
        if stacked {
            shape["blendMode"] = json!("multiply");
        } else {
            shape["opacity"] = json!(1.0);
            shape["fills"][1]["blendMode"] = json!("multiply");
        }
        let scene = Scene::load(&json!({"nodes":[shape,
            {"id":"g","type":"group","children":["box"],"effects":[{"id":"b","type":"layer-blur","radius":18}]}],
            "pages":[{"id":"p","width":240,"height":200,"nodeIds":["g"]}]}))?;
        let (plan, warnings) = build_coverage_plan(&scene, 0, &RenderOptions {
            transparent: true, ..present_test_options(4)
        })?;
        assert!(warnings.skipped.iter().any(|w| w.starts_with("g: subtree filter cannot isolate")), "{warnings:?}");
        assert!(!plan.items.iter().any(|it| matches!(it, DrawItem::LayerBlurComposite { .. })));
        let alpha: Vec<_> = plan.items.iter().filter_map(|it| match it {
            DrawItem::Body { blend_mode: BlendMode::Multiply, instance, .. } => Some(instance.fill_color[3]),
            _ => None,
        }).collect();
        assert_eq!(alpha, if stacked { vec![0.3; 2] } else { vec![1.] });
    }
    Ok(())
}

#[test]
fn horizon_crossing_text_has_only_its_finite_clip_as_host_bounds() -> Result<()> {
    crate::test_fonts();
    let text = json!({"id":"t","type":"richtext","x":5,"y":2,"width":96,"height":45,
        "htmlContent":"<p style='font-size:28px'>Ink</p>"});
    let mut batch = plan(vec![text], "t")?.items.into_iter().find_map(|item| match item {
        DrawItem::Text(batch) => Some(batch),
        _ => None,
    }).expect("shaped glyphs");
    let glyphs = std::sync::Arc::make_mut(&mut batch);
    let bounds = glyphs.local_bounds().unwrap();
    let horizon = (bounds[0] + bounds[2]) as f32 * 0.5;
    glyphs.distort = Some([1., 0., 0., 0., 1., 0., -1., 0., horizon]);
    for (clip, expected) in [
        (INFINITE_CLIP, None),
        ([4., 6., 40., 30.], Some([4., 6., 44., 36.])),
    ] {
        std::sync::Arc::make_mut(&mut batch).page_clip_bounds = clip;
        let item = DrawItem::Text(batch.clone());
        assert_eq!(crate::layer_blur::shadow::draw_bounds(&item), expected);
        assert!(crate::layer_blur::shadow::all_items_canvas_bounds(&[item]).is_some());
    }
    Ok(())
}
