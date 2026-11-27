    use super::*;

    fn css_document(nodes: serde_json::Value, animation: serde_json::Value) -> Document {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"},
            "render":{"resolution":[100,100]}, "nodes":nodes, "animation":animation
        })).unwrap()).unwrap();
        Document::new(&scene, Path::new(".")).unwrap()
    }

    #[test]
    fn path_mask_and_clip_keep_the_consumers_coordinates_across_parent_collapse() -> Result<()> {
        use serde_json::json;
        for reference in ["mask", "clip-path"] {
            let doc = css_document(json!([
                {"name":"group", "children":[{"children":[
                    {"name":"matte", "type":"path", "d":"M0 0H80V80Z", "fill":"white"}
                ]}]},
                {"type":"path", "d":"M0 0H80V80Z", "fill":"red", (reference):"matte"}
            ]), json!([{"tracks":[{"path":["nodes","group","scale"],
                "keys":[[0,[0,1,1]],[1,[1,1,1]]]}]}]));
            for time in [0., 0.25, 1., 0.] {
                doc.frame(&vork_time::Request::Numeric(time))?;
                assert!(!doc.plan(&[])?.items.is_empty(), "{reference} at {time}");
            }
        }
        Ok(())
    }

    #[test]
    fn text_clip_uses_only_the_glyph_batches_built_for_this_frame() -> Result<()> {
        use serde_json::json;
        let doc = css_document(json!([
            {"name":"group","children":[{"name":"clip","type":"text","content":"A","font-size":48,
                "font":concat!(env!("CARGO_MANIFEST_DIR"), "/../vork-std/tests/fonts/LiberationSans-Regular.ttf")}]},
            {"type":"path","d":"M0 0H100V100H0Z","fill":"red","clip-path":"clip"}
        ]), json!([{"tracks":[{"path":["nodes","group","scale"],"keys":[[0,[0,1,1]],[1,[1,1,1]]]}]}]));
        for (at, empty) in [(0., true), (1., false), (0., true)] {
            doc.frame(&vork_time::Request::Numeric(at))?;
            assert_eq!(doc.plan(&[])?.items.is_empty(), empty, "text clip at {at}");
        }
        Ok(())
    }

    #[test]
    fn collapsed_movie_matte_keeps_its_vector_stroke() -> Result<()> {
        use serde_json::json;
        let scene = Scene::from_json(&serde_json::to_vec(&json!({
            "version":2, "composition":{"space":"2d"}, "render":{"resolution":[64,64]},
            "images":{"film":{"file":"crates/vork-render/tests/fixtures/small-movie.mp4"}},
            "nodes":[{"scale":[0,1,1],"children":[{"name":"cut","type":"path","d":"M8 8H56V56H8Z",
                "fill":{"image":"film"}, "stroke":"white","stroke-width":4}]},
                {"type":"path","d":"M0 0H64V64H0Z","fill":"red","mask":"cut"}]
        }))?)?;
        let doc = Document::new(&scene, &Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))?;
        doc.frame(&vork_time::Request::Numeric(0.))?;
        assert!(!doc.plan(&[])?.items.is_empty(), "the matte's solid stroke still draws");
        Ok(())
    }

    #[test]
    fn moving_caption_over_movie_keeps_one_plan_through_seek() -> Result<()> {
        // The movie reader publishes into the process asset store.
        const CHILD: &str = "VORK_TEST_MOVING_CAPTION";
        if std::env::var_os(CHILD).is_none() {
            let name = std::thread::current().name().unwrap().to_owned();
            ensure!(std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", &name]).env(CHILD, "1").status()?.success(), "caption test failed");
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let movie = dir.path().join("clip.mp4");
        ensure!(std::process::Command::new("ffmpeg").args(["-v","error","-f","lavfi","-i",
            "testsrc2=s=64x64:r=8:d=1","-pix_fmt","yuv420p"]).arg(&movie).status()?.success(), "movie fixture");
        use serde_json::json;
        let scene = Scene::from_json(&serde_json::to_vec(&json!({
            "version":2,"composition":{"space":"2d"},"render":{"resolution":[160,90]},
            "images":{"clip":{"file":movie}},
            "nodes":[{"name":"plate","type":"path","d":"M0 0H160V90H0Z","fill":{"image":"clip","fit":"fill"}},
                {"name":"title","type":"text","content":"A B","font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf","font-size":24}],
            "animation":[{"tracks":[
                {"path":["nodes","title","translation"],"keys":[[0,[0,0,0]],[1,[10,5,0]]]},
                {"path":["nodes","title","words",0,"fill"],"keys":[[0,"red"],[0.5,"blue"],[1,"red"]]},
                {"path":["nodes","title","words",1,"fill"],"keys":[[0,"red"],[0.5,"green"],[1,"red"]]}
            ]}]
        }))?)?;
        let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
        doc.install_images()?;
        let mut revisions = Vec::new();
        for time in [0., 0.125, 0.375, 0.625, 0.875, 0., 0.5, 0.25] {
            let snapshot = doc.snapshot(&vork_time::Request::Numeric(time))?;
            let _plan = doc.plan_snapshot(&snapshot, &[])?;
            revisions.push(moda_assets::engine_seam::store_generation());
        }
        assert!(revisions.windows(2).all(|p| p[0] != p[1]), "movie advanced across every edit and seek");
        assert_eq!(doc.retained.borrow().as_ref().unwrap().full_plans, 1);
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn refused_matte_swap_restores_the_store_after_a_real_publication() -> Result<()> {
        use crate::shot::worker::{Source, Worker};
        use moda_assets::engine_seam::{installed, store_generation};
        use serde_json::json;
        // The generation belongs to the process. Only this worker may change it.
        const CHILD: &str = "VORK_TEST_REFUSED_MATTE_SWAP";
        if std::env::var_os(CHILD).is_none() {
            let name = std::thread::current().name().unwrap().to_owned();
            ensure!(std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", &name, "--ignored"]).env(CHILD, "1").status()?.success(), "matte publication test failed");
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let base = dir.path();
        ensure!(std::process::Command::new("ffmpeg").args(["-v","error","-f","lavfi","-i",
            "testsrc2=s=64x64:r=4:d=1","-c:v","libx264","-threads","1"]).arg(base.join("a.mp4")).status()?.success(), "movie fixture");
        std::fs::copy(base.join("a.mp4"), base.join("b.mp4"))?;
        let mut doc = json!({"version":2,"composition":{"space":"2d"},"render":{"resolution":[64,64]},
            "images":{"film":{"file":"a.mp4"}},"nodes":[
                {"name":"matte","type":"path","d":"M0 0H64V64H0Z","fill":{"image":"film"}},
                {"name":"body","type":"path","d":"M0 0H64V64H0Z","mask":"matte","fill":"red","stroke":"white","stroke-width":1}]});
        let mut worker = Worker::new();
        image::RgbaImage::from_pixel(1, 1, image::Rgba([255u8; 4])).save(base.join("still.png"))?;
        let mut rejected = doc.clone();
        rejected["images"]["film"]["file"] = json!("still.png");
        rejected["nodes"][1]["stroke-dasharray"] = json!([0.00001, 0.00001]);
        assert!(installed().is_none());
        let before = store_generation();
        let error = worker.open(Source::Document { document: rejected, base }).unwrap_err();
        assert!(format!("{error:#}").contains("stroke-dasharray expansion"), "{error:#}");
        assert!(store_generation() > before, "the rejected open must have published its image");
        assert!(installed().is_none(), "the first refusal must restore the empty store");

        worker.open(Source::Document { document: doc.clone(), base })?;
        let shot = crate::shot::Shot { out: vec![base.join("look.png").into()], ..Default::default() };
        doc["nodes"][1]["stroke-dasharray"] = json!([0.00001, 0.00001]);
        for (file, swap) in [("a.mp4", false), ("b.mp4", true)] {
            worker.render(shot.clone())?;
            doc["images"]["film"]["file"] = file.into();
            let store = installed().unwrap();
            let before = store_generation();
            let error = worker.open(Source::Document { document: doc.clone(), base }).unwrap_err();
            let message = format!("{error:#}");
            assert!(message.contains("update scene") && message.contains("stroke-dasharray expansion"), "{message}");
            // Observe at the refused open itself, before another render can publish.
            // Restoring an unchanged store does not advance its resource generation.
            let after = store_generation();
            assert_eq!(after > before, swap, "only the missing matte should publish");
            moda_assets::engine_seam::install_store(store);
            assert_eq!(store_generation(), after, "refusal must already have restored the original resources");
        }
        Ok(())
    }

    #[test]
    fn a_visible_path_publishes_only_its_selected_footage_paint() -> Result<()> {
        const CHILD: &str = "VORK_TEST_SELECTED_FOOTAGE_PAINT";
        if std::env::var_os(CHILD).is_none() {
            let name = std::thread::current().name().unwrap().to_owned();
            ensure!(std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", &name]).env(CHILD, "1").status()?.success(), "selected paint test failed");
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        for (name, red) in [("first", 64u8), ("second", 192)] {
            image::RgbaImage::from_pixel(8, 8, image::Rgba([red, 0, 0, 255]))
                .save(dir.path().join(format!("{name}00.png")))?;
        }
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"},
            "images":{"a":{"file":"first%02d.png"},"b":{"file":"second%02d.png"}},
            "nodes":[{"name":"p","type":"path","d":"M0 0H8V8Z","fill":{"image":"a","fit":"fill"}}],
            "animation":[{"tracks":[{"path":["nodes","p","fill"],"interpolation":"STEP",
                "keys":[[0,{"image":"a","fit":"fill"}],[1,{"image":"b","fit":"fill"}],[2,"red"]]}]}]
        }))?)?;
        let doc = Document::new(&scene, dir.path())?;
        for (time, count) in [(0., 1), (1., 1), (2., 0), (0., 1)] {
            doc.frame(&vork_time::Request::Numeric(time))?;
            let mut store = (*moda_assets::engine_seam::installed().unwrap()).clone();
            assert_eq!(store.clear_prefixed("vork:canvas-media/"), count);
        }
        Ok(())
    }

    #[test]
    fn replica_hands_a_late_subtree_reveal_back_before_reinstall() -> Result<()> {
        let doc = css_document(serde_json::json!([{"name":"g", "scale":[0,0,1],
            "children":[{"name":"p", "type":"path", "d":"M0 0H80V80Z", "fill":"red"}]}]),
            serde_json::json!([{"tracks":[{"path":["nodes","g","scale"],"interpolation":"STEP",
                "keys":[[0,[0,0,1]],[1,[1,1,1]]]}]}]));
        let at = vork_time::Request::Numeric(1.);
        doc.frame(&vork_time::Request::Numeric(0.))?;
        doc.plan(&[])?;
        let before = doc.retained.borrow().as_ref().unwrap().picture_scene();
        let refused = crate::canvas_host_memory::with_workers(|| doc.frame(&at)).err().expect("unreserved subtree");
        assert!(refused.is::<crate::canvas_host_memory::SerialGrowth>());
        assert_eq!(doc.retained.borrow().as_ref().unwrap().picture_scene(), before);
        doc.frame(&at)?;
        assert!(!doc.plan(&[])?.items.is_empty());
        Ok(())
    }

    #[test]
    fn css_blur_plans_at_large_target_sigmas() -> Result<()> {
        for group in [false, true] {
            for (sigma, scale) in [
                (100., 1.), (100.01, 1.), (128., 1.), (500., 1.),
                (1e30, 1.), (50., 2.), (50.01, 2.),
                (200., 0.5),
            ] {
                let mut node = serde_json::json!({"name":"shape", "type":"path",
                    "d":"M0 0H120V80H0Z", "fill":"red"});
                if group { node = serde_json::json!({"name":"group", "children":[node]}); }
                node["filter"] = serde_json::json!(format!("blur({sigma}px)"));
                node["scale"] = serde_json::json!([scale, scale, 1.]);
                let doc = css_document(serde_json::json!([node]), serde_json::json!([]));
                let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
                doc.plan_snapshot(&frame, &[])?;
            }
        }
        Ok(())
    }

    #[test]
    fn wide_blur_captures_one_pixel_strip() -> Result<()> {
        for width in [0.5, 1.0] {
            for reach in [3.0, 75.0, 76.0, 299.0, 300.0, 300.01, 301.0, 384.0] {
                let doc = css_document(serde_json::json!([{
                    "name":"strip", "type":"path", "d":format!("M50 5h{width}v90h-{width}Z"),
                    "fill":"white", "filter":format!("blur({}px)", reach / 3.0)
                }]), serde_json::json!([]));
                let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
                let plan = doc.plan_snapshot(&frame, &[])?;
                assert!(plan.items.iter().any(|item| matches!(
                    item, moda_engine::seams::DrawItem::LayerBlurComposite { .. }
                )), "width {width}, reach {reach}");
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn budget_bound_gaussian_preserves_translated_coverage() -> Result<()> {
        let renderer = Renderer::new()?;
        let engine = crate::canvas_draw::Engine::new_picture(
            &renderer, true, v2::WorkingColorSpace::LinearRec709)?;
        let (width, height) = (8192, 768);
        for phase in [0.0, 0.25, 0.5, 0.75] {
            // Offscreen contours fix a roughly 6000 × 6700 capture. Its sides
            // fit individually, but their product forces the area reduction.
            let x = 4096.0 + phase;
            let path = format!("M-64 64h1v640h-1Z M{x} 64h0.5v640h-0.5Z M6000 6700h1v1h-1Z");
            let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                "version":2,"composition":{"space":"2d","color_space":"Linear Rec.709"},
                "render":{"resolution":[width,height],"background":"transparent"},
                "nodes":[{"name":"strip","type":"path","d":path,"fill":"white","filter":"blur(4px)"}]
            }))?)?;
            let doc = Document::new(&scene, Path::new("."))?;
            let frame = doc.snapshot(&vork_time::Request::Numeric(0.0))?;
            let plan = doc.plan_snapshot(&frame, &[])?;
            let group = plan.items.iter().find_map(|item| match item {
                moda_engine::seams::DrawItem::LayerBlurComposite { group, .. } => Some(group),
                _ => None,
            }).expect("filtered strip");
            let [_, _, w, h] = group.region_canvas;
            assert!(w.max(h) < 8192. && w*h > 4096.*4096.);
            assert!(f64::from(group.width_px) / w < 0.7);
            let texture = engine.render_to_texture(&plan)?;
            let rgba = moda_engine::readback::read_texture_rgba16f(
                &renderer.device, &renderer.queue, &texture, width, height)?;
            // The Gaussian is normalized and the full visible contour and its kernel
            // lie inside the target. Integrate linear alpha before display encoding.
            let alpha: f64 = rgba.chunks_exact(4).map(|p| f64::from(p[3].to_f32())).sum();
            assert!((alpha - 320.0).abs() < 3.2, "phase {phase}: integrated alpha {alpha}, expected 320");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn uniform_blur_matches_full_resolution_strip_convolution() -> Result<()> {
        const SIZE: usize = 1024;
        const X: usize = 512;
        let renderer = Renderer::new()?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, true, v2::WorkingColorSpace::LinearRec709)?;
        let render = |width: f64, sigma: f64| -> Result<Vec<f32>> {
            let mut strip = serde_json::json!({"name":"strip","type":"path","fill":"white",
                "d":format!("M512 192h{width}v640h-{width}Z")});
            if sigma > 0.0 { strip["filter"] = serde_json::json!(format!("blur({sigma}px)")); }
            let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                "version":2,"composition":{"space":"2d","color_space":"Linear Rec.709"},
                "render":{"resolution":[SIZE,SIZE],"background":"black"},"nodes":[strip]
            }))?)?;
            let doc = Document::new(&scene, Path::new("."))?;
            let frame = doc.snapshot(&vork_time::Request::Numeric(0.0))?;
            let texture = engine.render_to_texture(&doc.plan_snapshot(&frame, &[])?)?;
            let rgba = moda_engine::readback::read_texture_rgba16f(
                &renderer.device, &renderer.queue, &texture, SIZE as u32, SIZE as u32)?;
            Ok(rgba.chunks_exact(4).map(|p| p[0].to_f32()).collect())
        };
        for width in [0.5, 1.0] {
            let sharp = render(width, 0.0)?;
            // Both strips occupy one column. Convolving the measured coverage in
            // that column vertically and its impulse horizontally is the full
            // resolution 2D convolution, with transparent continuation.
            assert!(sharp[512 * SIZE + X] > 0.0);
            for reach in [3., 75., 76., 299., 299.99, 300., 300.01, 301., 384., 6000., 12000.] {
                let sigma = reach / 3.0;
                let actual = render(width, sigma)?;
                let radius = (6.0 * sigma).ceil() as usize;
                let mut weights: Vec<_> = (0..=radius).map(|d| (-0.5 * (d as f64 / sigma).powi(2)).exp()).collect();
                let sum = 1.0 + 2.0 * weights[1..].iter().sum::<f64>();
                weights.iter_mut().for_each(|w| *w /= sum);
                let weight = |offset: usize| weights.get(offset).copied().unwrap_or(0.0);
                let mut max_error = 0.0_f64;
                let mut reference_mass = 0.0;
                for y in 0..SIZE {
                    let column: f64 = (192..832).map(|sy| f64::from(sharp[sy * SIZE + X]) * weight(y.abs_diff(sy))).sum();
                    for x in 0..SIZE {
                        let expected = column * weight(x.abs_diff(X));
                        reference_mass += expected;
                        max_error = max_error.max((f64::from(actual[y * SIZE + x]) - expected).abs());
                    }
                }
                let mass = actual.iter().map(|&v| f64::from(v)).sum::<f64>();
                assert!(max_error <= 1.0 / 255.0, "width {width}, reach {reach}: max error {max_error}");
                // At very wide reaches the entire picture is below an eight-bit
                // level. A missing strip would pass the absolute-error test.
                assert!((mass / reference_mass - 1.0).abs() < 0.03,
                    "width {width}, reach {reach}: mass {mass}, reference {reference_mass}");
            }
        }
        Ok(())
    }

    #[test]
    fn css_nested_blurs_keep_every_capture_and_validate_the_parent() -> Result<()> {
        use moda_engine::seams::DrawItem;
        fn captures(items: &[DrawItem]) -> usize {
            items.iter().map(|item| match item {
                DrawItem::LayerBlurComposite { group, .. } => 1 + captures(&group.items),
                _ => 0,
            }).sum()
        }
        for depth in [2, 3, 4] {
            for oversized in [false, true] {
                let mut node = serde_json::json!({"name":"shape", "type":"path",
                    "d":"M20 20H80V80H20Z", "fill":"red", "filter":"blur(1px)"});
                for level in 1..depth {
                    node = serde_json::json!({"name":format!("parent{level}"),
                        "filter":if oversized && level == depth - 1 {"blur(500px)"} else {"blur(2px)"},
                        "children":[node]});
                }
                let doc = css_document(serde_json::json!([node]), serde_json::json!([]));
                let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
                assert_eq!(captures(&doc.plan_snapshot(&frame, &[])?.items), depth);
            }
        }
        Ok(())
    }

    #[test]
    fn css_blur_uses_small_nonzero_transform_scales() -> Result<()> {
        for scale in [0.000001, 0.00005, 0.0001, 0.0002] {
            for group in [false, true] {
                let extent = 50. / scale;
                let mut node = serde_json::json!({"name":"scaled", "type":"path",
                    "d":format!("M0 0H{extent}V{extent}H0Z"), "fill":"red"});
                if group { node = serde_json::json!({"name":"group", "children":[node]}); }
                node["scale"] = serde_json::json!([scale, scale, 1.]);
                node["filter"] = serde_json::json!("blur(200px)");
                let doc = css_document(serde_json::json!([node]), serde_json::json!([]));
                let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
                doc.plan_snapshot(&frame, &[])?;
            }
        }
        Ok(())
    }

    #[test]
    fn css_blur_plans_visible_anisotropy() -> Result<()> {
        // Both axis orders, ordinary scales, inherited shear, and the review's
        // tiny-scale input exercise the shared leaf and subtree planner.
        for scale in [[0.00005, 1., 1.], [1., 0.00005, 1.], [0.5, 1., 1.]] {
            for group in [false, true] {
                for sigma in [2., 80.] {
                    let mut node = serde_json::json!({"name":"shape", "type":"path",
                        "d":format!("M0 0H{}V{}H0Z", 50. / scale[0], 50. / scale[1]),
                        "fill":"red"});
                    if group { node = serde_json::json!({"name":"group", "children":[node]}); }
                    node["scale"] = serde_json::json!(scale);
                    node["filter"] = serde_json::json!(format!("blur({sigma}px)"));
                    let doc = css_document(serde_json::json!([node]), serde_json::json!([]));
                    let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
                    doc.plan_snapshot(&frame, &[])?;
                }
            }
        }
        // Rotated nonuniform ancestors produce shear; the leaf's local scale
        // alone cannot identify whether its projected blur is isotropic.
        let doc = css_document(serde_json::json!([{"name":"parent", "scale":[2.,1.,1.],
            "children":[{"name":"child", "rotation":[0.,0.,0.38268343,0.9238795],
                "type":"path", "d":"M0 0H40V40H0Z", "fill":"red", "filter":"blur(2px)"}]}]),
            serde_json::json!([]));
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        doc.plan_snapshot(&frame, &[])?;
        Ok(())
    }

    #[test]
    fn css_blur_keeps_conformal_and_invisible_transforms() -> Result<()> {
        for scale in [[0.00005, 0.00005, 1.], [-1., 1., 1.], [1., 1., 1.],
            [0.00005, 0.0001, 1.]] {
            let doc = css_document(serde_json::json!([{"name":"shape", "type":"path",
                "d":"M0 0H50V50H0Z", "fill":"red", "scale":scale,
                "rotation":[0.,0.,0.38268343,0.9238795], "filter":"blur(2px)"}]),
                serde_json::json!([]));
            let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
            doc.plan_snapshot(&frame, &[])?;
        }
        Ok(())
    }

    #[test]
    fn css_blur_visibility_uses_the_larger_reach_at_the_cutoff() -> Result<()> {
        // The reaches straddle 0.5px within projection-rounding tolerance.
        // Accepting this near-conformal transform must still retain its blur.
        let doc = css_document(serde_json::json!([{"name":"shape", "type":"path",
            "d":"M0 0H300V300H0Z", "fill":"red", "scale":[0.1666,0.1667,1.],
            "filter":"blur(1px)"}]), serde_json::json!([]));
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let plan = doc.plan_snapshot(&frame, &[])?;
        assert!(plan.items.iter().any(|item| matches!(item,
            moda_engine::seams::DrawItem::LayerBlurComposite { .. })),
            "the larger reach is visible, so the blur must remain");
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn held_root_picture_matches_fresh_draws_after_edits() -> Result<()> {
        let renderer = Renderer::new()?;
        let engine = crate::canvas_draw::Engine::new(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        let reference = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        let doc = css_document(serde_json::json!([
            {"name":"shape", "type":"path", "d":"M10 10H60V60H10Z", "fill":"red"},
            {"name":"other", "type":"path", "d":"M40 40H90V90H40Z", "fill":"blue", "opacity":0.5}
        ]), serde_json::json!([]));
        for step in [0, 1, 2, 3, 0] {
            let mut frame = doc.snapshot(&vork_time::Request::Numeric(step as f32))?;
            match step {
                1 => frame.values[0].transform.translation[0] = 13.,
                2 => frame.values[1].opacity = 0.2,
                3 => frame.background = Some([0., 1., 0., 1.]),
                _ => {},
            }
            let plan = doc.plan_snapshot(&frame, &[])?;
            let expected = reference.render_to_texture(&plan)?;
            let expected = moda_engine::readback::read_texture_rgba(
                &renderer.device, &renderer.queue, &expected, 100, 100)?;
            // First render may populate a resource cache. A second draw must
            // establish a stable held picture, and the third must reuse it.
            for _ in 0..2 { engine.render_to_texture(&plan)?; }
            let before = engine.profile_counters();
            let actual = engine.render_to_texture(&plan)?;
            let after = engine.profile_counters();
            assert_eq!(after["picture_hits"].as_u64(), before["picture_hits"].as_u64().map(|v| v + 1));
            assert_eq!(before["bundle_replays"], after["bundle_replays"]);
            let actual = moda_engine::readback::read_texture_rgba(
                &renderer.device, &renderer.queue, &actual, 100, 100)?;
            assert_eq!(actual, expected, "step {step}");
        }
        assert_eq!(reference.profile_counters()["picture_hits"], 0);
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn css_nested_blurs_match_composed_gaussians_and_tiny_scale_renders() -> Result<()> {
        let renderer = Renderer::new()?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        let render = |node: serde_json::Value| -> Result<Vec<u8>> {
            let nodes = if node.is_array() { node } else { serde_json::json!([node]) };
            let doc = css_document(nodes, serde_json::json!([]));
            let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
            let texture = engine.render_to_texture(&doc.plan_snapshot(&frame, &[])?)?;
            moda_engine::readback::read_texture_rgba(
                &renderer.device, &renderer.queue, &texture, 100, 100)
        };
        let square = || serde_json::json!({"name":"shape", "type":"path",
            "d":"M20 20H80V80H20Z", "fill":"red"});
        // Independent Gaussian convolutions add their variances. Compare associated
        // RGBA with explicit tolerances for finite kernels and capture quantization.
        for (sigmas, combined) in [(vec![3., 4.], 5.), (vec![1., 2., 2.], 3.)] {
            let mut nested = square();
            nested["filter"] = serde_json::json!(format!("blur({}px)", sigmas[0]));
            for (level, sigma) in sigmas.iter().enumerate().skip(1) {
                nested = serde_json::json!({"name":format!("parent{level}"),
                    "filter":format!("blur({sigma}px)"), "children":[nested]});
            }
            let actual = render(nested)?;
            let mut reference = square();
            reference["filter"] = serde_json::json!(format!("blur({combined}px)"));
            let expected = render(reference)?;
            let max = actual.iter().zip(&expected).map(|(a,b)| a.abs_diff(*b)).max().unwrap();
            let rmse = (actual.iter().zip(&expected).map(|(a,b)| (*a as f64 - *b as f64).powi(2))
                .sum::<f64>() / actual.len() as f64).sqrt();
            assert!(max <= 12 && rmse <= 3., "{sigmas:?}: max={max}, RMSE={rmse}");
        }
        // The child's clip is sampled before the parent blur, while the mask
        // above the parent clips the final filtered result.
        let masked = render(serde_json::json!([
            {"name":"parent", "filter":"blur(4px)", "clip-path":"top", "children":[
                {"name":"child", "filter":"blur(3px)", "clip-path":"left", "children":[square()]}]},
            {"name":"left", "type":"path", "d":"M0 0H50V100H0Z"},
            {"name":"top", "type":"path", "d":"M0 0H100V50H0Z"}
        ]))?;
        let alpha = |x: usize, y: usize| masked[(y * 100 + x) * 4 + 3];
        assert!(alpha(51, 40) > 0, "parent blur spreads past the child's clip");
        assert_eq!(alpha(40, 51), 0, "outer clip masks the parent result");
        assert!(alpha(40, 40) > 200, "interior remains visible");
        let scaled = render(serde_json::json!({"name":"scaled", "type":"path",
            "d":"M0 0H1000000V1000000H0Z", "fill":"red",
            "scale":[0.00005,0.00005,1.], "filter":"blur(200px)"}))?;
        let reference = render(serde_json::json!({"name":"reference", "type":"path",
            "d":"M0 0H50V50H0Z", "fill":"red"}))?;
        assert_eq!(scaled, reference, "0.01 target-pixel sigma preserves the square");
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn css_gradients_ignore_fully_transparent_stop_rgb() -> Result<()> {
        let renderer = Renderer::new()?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        for kind in ["linear-gradient(to right,", "radial-gradient(", "conic-gradient("] {
            let mut reference: Option<Vec<u8>> = None;
            for stop in ["rgba(0,0,255,0)", "rgba(255,0,0,0)", "transparent"] {
                let doc = css_document(serde_json::json!([{"name":"shape", "type":"path",
                    "d":"M0 0H100V100H0Z", "fill":format!("{kind}{stop}, blue)")}]), serde_json::json!([]));
                let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
                let texture = engine.render_to_texture(&doc.plan_snapshot(&frame, &[])?)?;
                let rgba = moda_engine::readback::read_texture_rgba(
                    &renderer.device, &renderer.queue, &texture, 100, 100)?;
                if let Some(expected) = &reference {
                    let max = rgba.iter().zip(expected.iter()).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
                    assert!(max <= 1, "{kind}{stop}: hidden RGB changed pixels by {max}");
                } else {
                    assert!(rgba.chunks_exact(4).any(|p| p[3] > 32 && p[3] < 224), "exercise translucent pixels");
                    reference = Some(rgba);
                }
            }
        }
        // Nonzero alpha at both ends must also be associated during interpolation.
        let doc = css_document(serde_json::json!([{"name":"shape", "type":"path",
            "d":"M0 0H100V100H0Z",
            "fill":"linear-gradient(to right, rgba(255,0,0,0.25), rgba(0,0,255,0.75))"}]), serde_json::json!([]));
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let texture = engine.render_to_texture(&doc.plan_snapshot(&frame, &[])?)?;
        let rgba = moda_engine::readback::read_texture_rgba(
            &renderer.device, &renderer.queue, &texture, 100, 100)?;
        let pixel = &rgba[(50 * 100 + 50) * 4..][..4];
        for (actual, expected) in pixel.iter().zip([32u8, 0, 97, 128]) {
            assert!(actual.abs_diff(expected) <= 2, "premultiplied midpoint: {pixel:?}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn retained_picture_outputs_survive_later_draws() -> Result<()> {
        let renderer = Renderer::new()?;
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version": 2, "composition": {"space": "2d"},
            "render": {"resolution": [32, 32]},
            "nodes": [{"name":"shape", "type":"path", "d":"M0 0H16V32H0Z", "fill":"red"}]
        }))?)?;
        let doc = Document::new(&scene, Path::new("."))?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        let mut frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let first = engine.render_to_texture(&doc.plan_snapshot(&frame, &[])?)?;
        let read = |texture: &wgpu::Texture| moda_engine::readback::read_texture_rgba(
            &renderer.device, &renderer.queue, texture, 32, 32);
        let expected = read(&first)?;
        frame.values[doc.by_name["shape"]].transform.translation[0] = 16.;
        let second = engine.render_to_texture(&doc.plan_snapshot(&frame, &[])?)?;
        assert_ne!(read(&second)?, expected, "the second drawing must change");
        assert_eq!(read(&first)?, expected, "a deferred consumer must still see the first drawing");
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn text_filters_follow_glyph_alpha_for_every_paint() -> Result<()> {
        let renderer = Renderer::new()?;
        let dir = std::env::temp_dir().join(format!("vork-text-filter-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255])).save(dir.join("paint.png"))?;
        let font = dir.join("font.ttf");
        std::fs::write(&font, include_bytes!("../../vork-std/tests/fonts/LiberationSans-Regular.ttf"))?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        // Large glyphs: the counters of the O and the gaps of the H must be wider than twice the widest reach
        // probed below plus the shadow's offset, or no pixel is far enough from both the ink and its shadow to
        // be a hole and the oracle probes nothing.
        const W: i32 = 512;
        const H: i32 = 320;
        for fill in [serde_json::json!("red"),
            serde_json::json!("linear-gradient(to right, red, blue)"),
            serde_json::json!({"image":"paint"})] {
            let render = |filter: &str| -> Result<Vec<u8>> {
                let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                    "version":2, "composition":{"space":"2d"},
                    "render":{"resolution":[W,H]},
                    "images":{"paint":{"file":"paint.png"}},
                    "nodes":[{"type":"text", "content":"OHO", "font":font,
                        "font-size":160, "translation":[24,16,0], "fill":fill, "filter":filter}]
                }))?)?;
                let doc = Document::new(&scene, &dir)?;
                doc.install_images()?;
                let pictures = Pictures::new(&renderer, &doc, &dir, true)?;
                let texture = doc.draw(&engine, &pictures, &renderer, &vork_time::Request::Numeric(0.), None)?;
                let rgba = moda_engine::readback::read_texture_rgba(
                    &renderer.device, &renderer.queue, &texture, W as u32, H as u32)?;
                Ok(rgba.chunks_exact(4).map(|p| p[3]).collect())
            };
            let source = render("none")?;
            assert!(source.iter().any(|a| *a > 250), "text must render");
            let bounds = source.iter().enumerate().filter(|(_, a)| **a > 0)
                .fold([W, H, 0, 0], |[x0, y0, x1, y1], (i, _)| {
                    let (x, y) = (i as i32 % W, i as i32 / W);
                    [x0.min(x), y0.min(y), x1.max(x), y1.max(y)]
                });
            let sample = |x: i32, y: i32| -> u8 {
                if (0..W).contains(&x) && (0..H).contains(&y) {
                    source[(y * W + x) as usize]
                } else { 0 }
            };
            for (filter, shadow, radius) in [
                ("drop-shadow(12px 8px 0px black)", true, 2),
                ("drop-shadow(12px 8px 4px black)", true, 14),
                ("blur(2px)", false, 8),
                ("drop-shadow(12px 8px 0px black) blur(2px)", true, 8),
            ] {
                let alpha = render(filter)?;
                assert!(alpha.iter().any(|a| *a > 128), "{fill} {filter}: text must remain visible");
                let mut holes = 0;
                let mut shadow_pixels = 0;
                let mut halo_pixels = 0;
                for y in bounds[1]..=bounds[3] + if shadow { 8 } else { 0 } {
                    for x in bounds[0]..=bounds[2] + if shadow { 12 } else { 0 } {
                        // Allow the finite Gaussian reach (3 sigma), plus two
                        // pixels for the different text/mask antialiasing paths.
                        let supported = (-radius..=radius).any(|dy| (-radius..=radius).any(|dx|
                            sample(x + dx, y + dy) > 0
                                || (shadow && sample(x + dx - 12, y + dy - 8) > 0)));
                        let actual = alpha[(y * W + x) as usize];
                        if !supported {
                            holes += 1;
                            assert!(actual <= 3, "{fill} {filter}: glyph hole/background ({x},{y}) has alpha {actual}");
                        }
                        if !shadow && sample(x, y) == 0 && actual > 3 { halo_pixels += 1; }
                        if shadow && sample(x, y) == 0 && sample(x - 12, y - 8) > 250 {
                            shadow_pixels += usize::from(actual > 80);
                        }
                    }
                }
                assert!(holes > 100, "{fill} {filter}: the oracle must probe glyph holes and inter-glyph gaps, and probed {holes}");
                if shadow { assert!(shadow_pixels > 100, "{fill} {filter}: the offset glyphs must cast a visible shadow ({shadow_pixels} pixels)"); }
                else { assert!(halo_pixels > 100, "{fill} {filter}: blur must spread beyond the glyph edges ({halo_pixels} pixels)"); }
            }
        }
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn parent_shadows_follow_glyph_alpha_through_nested_filters() -> Result<()> {
        let renderer = Renderer::new()?;
        let dir = std::env::temp_dir().join(format!("vork-nested-text-filter-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255])).save(dir.join("paint.png"))?;
        std::fs::write(dir.join("font.ttf"), include_bytes!("../../vork-std/tests/fonts/LiberationSans-Regular.ttf"))?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        for fill in [serde_json::json!("red"),
            serde_json::json!("linear-gradient(to right, red, blue)"),
            serde_json::json!("radial-gradient(red, blue)"),
            serde_json::json!("conic-gradient(red, blue, red)"),
            serde_json::json!({"image":"paint"})] {
            for depth in [1, 3] {
                let render = |shadow: bool| -> Result<Vec<u8>> {
                    let mut child = serde_json::json!({"name":"text", "type":"text", "content":"OHO",
                        "font":"font.ttf", "font-size":80, "translation":[24,16,0], "fill":fill,
                        "filter":"blur(2px)"});
                    for level in 1..depth {
                        child = serde_json::json!({"name":format!("filter-{level}"),
                            "filter":"blur(0.5px)", "children":[child]});
                    }
                    let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                        "version":2, "composition":{"space":"2d"}, "render":{"resolution":[640,160]},
                        "images":{"paint":{"file":"paint.png"}},
                        "nodes":[{"name":"parent", "filter":if shadow {"drop-shadow(300px 0px 0px black)"} else {"none"},
                            "children":[child]}]
                    }))?)?;
                    let doc = Document::new(&scene, &dir)?;
                    doc.install_images()?;
                    let pictures = Pictures::new(&renderer, &doc, &dir, true)?;
                    let texture = doc.draw(&engine, &pictures, &renderer, &vork_time::Request::Numeric(0.), None)?;
                    let rgba = moda_engine::readback::read_texture_rgba(
                        &renderer.device, &renderer.queue, &texture, 640, 160)?;
                    Ok(rgba.chunks_exact(4).map(|p| p[3]).collect())
                };
                let source = render(false)?;
                let shadow = render(true)?;
                let mut holes = 0;
                let mut ink = 0;
                // The far offset isolates the shadow. Probe holes and glyph cores
                // away from antialiasing/blur edges, independently of paint RGB.
                for y in 24..100usize {
                    for x in 24..220usize {
                        let supported = (-2..=2isize).any(|dy| (-2..=2isize).any(|dx| {
                            let (sx, sy) = (x as isize + dx, y as isize + dy);
                            sx >= 0 && sy >= 0 && source[sy as usize * 640 + sx as usize] > 0
                        }));
                        let actual = shadow[y * 640 + x + 300];
                        if !supported {
                            holes += 1;
                            assert!(actual <= 3, "{fill} depth {depth}: shadow fills glyph hole/gap at {x},{y}: {actual}");
                        }
                        // The source is already blurred; its strokes need not
                        // contain fully opaque pixels. Probe 75% coverage.
                        if source[y * 640 + x] > 192 {
                            ink += 1;
                            assert!(actual > 80, "{fill} depth {depth}: missing glyph shadow at {x},{y}");
                        }
                    }
                }
                assert!(holes > 100, "must exercise holes and gaps");
                assert!(ink > 100, "{fill} depth {depth}: only {ink} high-coverage probes; max source alpha {:?}", source.iter().max());
            }
        }
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn rich_text_clip_keeps_exact_shape_with_offscreen_siblings() -> Result<()> {
        let renderer = Renderer::new()?;
        let dir = std::env::temp_dir().join(format!("vork-rich-clip-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("font.ttf"), include_bytes!("../../vork-std/tests/fonts/LiberationSans-Regular.ttf"))?;
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"}, "render":{"resolution":[240,100]},
            "nodes":[
                {"name":"triangle", "type":"path", "d":"M20 20 H60 L20 60 Z", "fill":"white"},
                {"clip-path":"triangle", "children":[
                    {"type":"text", "content":"H", "font":"font.ttf", "font-size":80, "fill":"red"},
                    {"type":"text", "content":[{"content":"hidden "},{"content":"rich text"}],
                        "font":"font.ttf", "font-size":20, "translation":[0,-100,0]}]},
                {"type":"text", "content":"H", "font":"font.ttf", "font-size":80,
                    "translation":[120,0,0], "fill":"red"}]
        }))?)?;
        let doc = Document::new(&scene, &dir)?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        let pictures = Pictures::new(&renderer, &doc, &dir, true)?;
        for _ in 0..2 {
            let texture = doc.draw(&engine, &pictures, &renderer, &vork_time::Request::Numeric(0.), None)?;
            let rgba = moda_engine::readback::read_texture_rgba(
                &renderer.device, &renderer.queue, &texture, 240, 100)?;
            let alpha = |x: usize, y: usize| rgba[(y * 240 + x) * 4 + 3];
            assert!(alpha(46, 25) > 192, "mask removed interior glyph coverage");
            assert!(alpha(46, 55) <= 3, "glyph leaked outside triangle but inside its bounding box");
            assert!(alpha(166, 25) > 192 && alpha(166, 55) > 192, "unclipped reference must contain both probes");
        }
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn ancestor_shadows_follow_masks_at_every_composite_and_paint_depth() -> Result<()> {
        let renderer = Renderer::new()?;
        let dir = std::env::temp_dir().join(format!("vork-composite-mask-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255])).save(dir.join("paint.png"))?;
        std::fs::write(dir.join("font.ttf"), include_bytes!("../../vork-std/tests/fonts/LiberationSans-Regular.ttf"))?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        for text in [false, true] {
        for fill in [serde_json::json!("red"),
            serde_json::json!("linear-gradient(to right, red, blue)"),
            serde_json::json!("radial-gradient(red, blue)"),
            serde_json::json!("conic-gradient(red, blue, red)"),
            serde_json::json!({"image":"paint"})] {
            for paint_depth in 0..4 {
                for leaf_filter in [false, true] {
                    // One tile per subset of masks. Paint may precede deeper
                    // masks; those must not leak back to an earlier sibling.
                    let mut nodes = Vec::new();
                    for placements in 0..16 {
                        let mut child = None;
                        for level in (0..4).rev() {
                            let mut children = Vec::new();
                            if level == paint_depth {
                                let mut paint = if text {
                                    serde_json::json!({"type":"text", "content":"H", "font":"font.ttf", "font-size":80})
                                } else { serde_json::json!({"type":"path", "d":"M0 0H80V80H0Z"}) };
                                paint["fill"] = fill.clone();
                                paint["filter"] = serde_json::json!(if leaf_filter { "blur(2px)" } else { "none" });
                                children.push(paint);
                            }
                            if let Some(child) = child { children.push(child); }
                            let mut group = serde_json::json!({"filter":"blur(0.5px)", "children":children});
                            if placements & (1 << level) != 0 { group["clip-path"] = serde_json::json!("triangle"); }
                            child = Some(group);
                        }
                        nodes.push(serde_json::json!({"translation":[(placements % 4) * 240, (placements / 4) * 100, 0],
                            "filter":"drop-shadow(120px 0px 0px black)", "children":[child.unwrap()]}));
                    }
                    nodes.push(serde_json::json!({"name":"triangle", "type":"path",
                        "d":"M20 20H60L20 60Z", "fill":"white"}));
                    let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                        "version":2, "composition":{"space":"2d"}, "render":{"resolution":[960,400]},
                        "images":{"paint":{"file":"paint.png"}}, "nodes":nodes
                    }))?)?;
                    let doc = Document::new(&scene, &dir)?;
                    doc.install_images()?;
                    let pictures = Pictures::new(&renderer, &doc, &dir, true)?;
                    // A second draw exercises retained plans/resources as well.
                    for _ in 0..2 {
                        let texture = doc.draw(&engine, &pictures, &renderer, &vork_time::Request::Numeric(0.), None)
                            .with_context(|| format!("{fill} text {text} depth {paint_depth} leaf {leaf_filter}"))?;
                        let rgba = moda_engine::readback::read_texture_rgba(
                            &renderer.device, &renderer.queue, &texture, 960, 400)?;
                        for placements in 0..16usize {
                            let alpha = |x, y| rgba[(((placements / 4) * 100 + y) * 960
                                + (placements % 4) * 240 + 120 + x) * 4 + 3];
                            assert!(alpha(46, 25) > 192, "missing shadow: {fill} text {text} depth {paint_depth}, masks {placements}");
                            let clipped = placements & ((1 << (paint_depth + 1)) - 1) != 0;
                            if clipped {
                                assert!(alpha(46, 55) <= 3,
                                    "mask lost: {fill} text {text} depth {paint_depth}, masks {placements}, leaf {leaf_filter}");
                            } else {
                                assert!(alpha(46, 55) > 192,
                                    "sibling mask leaked: {fill} text {text} depth {paint_depth}, masks {placements}");
                            }
                        }
                    }
                }
            }
        }
        }
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn image_and_conic_strokes_draw_only_the_outline() -> Result<()> {
        let renderer = Renderer::new()?;
        let dir = std::env::temp_dir().join(format!("vork-paint-{}-{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()));
        std::fs::create_dir(&dir)?;
        image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255])).save(dir.join("paint.png"))?;
        for paint in [serde_json::json!({"image": "paint"}), serde_json::json!("conic-gradient(red, blue, red)")] {
            let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                "version": 2, "composition": {"space": "2d"}, "render": {"resolution": [128, 128]},
                "images": {"paint": {"file": "paint.png"}},
                "nodes": [{"type": "path", "d": "M32 32H96V96H32Z", "fill": "none", "stroke": paint, "stroke-width": 8}]
            }))?)?;
            let doc = Document::new(&scene, &dir)?;
            doc.install_images()?;
            let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
            let pictures = Pictures::new(&renderer, &doc, &dir, true)?;
            let texture = doc.draw(&engine, &pictures, &renderer, &vork_time::Request::Numeric(0.), None)?;
            let rgba = moda_engine::readback::read_texture_rgba(&renderer.device, &renderer.queue, &texture, 128, 128)?;
            let alpha = |x: usize, y: usize| rgba[(y * 128 + x) * 4 + 3];
            assert!(alpha(32, 64) > 250, "the paint must cover the stroke centre");
            assert_eq!(alpha(64, 64), 0, "a stroke must leave the interior empty");
            assert_eq!(alpha(20, 64), 0, "a stroke must not paint outside its width");
        }
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn explicit_zero_length_paths_draw_round_and_square_caps() -> Result<()> {
        let scene = Scene::from_json(br#"{
            "version":2, "composition":{"space":"2d"}, "render":{"resolution":[48,16]},
            "nodes":[
                {"type":"path","d":"M6 8L6 8","fill":"none","stroke":"red","stroke-width":8,"stroke-linecap":"round"},
                {"type":"path","d":"M16 8L16 8","fill":"none","stroke":"red","stroke-width":8,"stroke-linecap":"square"},
                {"type":"path","d":"M26 8Z","fill":"none","stroke":"red","stroke-width":8,"stroke-linecap":"round"},
                {"type":"path","d":"M36 8","fill":"none","stroke":"red","stroke-width":8,"stroke-linecap":"round"},
                {"type":"path","d":"M44 8L44 8","fill":"none","stroke":"red","stroke-width":8}
            ]
        }"#)?;
        let base = Path::new(".");
        let renderer = Renderer::new()?;
        let doc = Document::new(&scene, base)?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        let pictures = Pictures::new(&renderer, &doc, base, true)?;
        let texture = doc.draw(&engine, &pictures, &renderer, &vork_time::Request::Numeric(0.), None)?;
        let rgba = moda_engine::readback::read_texture_rgba(&renderer.device, &renderer.queue, &texture, 48, 16)?;
        let alpha = |x: usize, y: usize| rgba[(y * 48 + x) * 4 + 3];
        assert_eq!(alpha(6, 8), 255);
        assert_eq!(alpha(2, 4), 0, "round cap leaves the corner clear");
        assert_eq!(alpha(16, 8), 255);
        assert_eq!(alpha(12, 4), 255, "square cap covers the corner");
        assert_eq!(alpha(26, 8), 255, "an explicit close has a round cap");
        assert_eq!(alpha(36, 8), 0, "a lone move has no stroke");
        assert_eq!(alpha(44, 8), 0, "a butt cap on a zero-length segment has no stroke");
        Ok(())
    }

    #[test]
    fn retained_canvas_updates_match_fresh_scene() -> Result<()> {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version": 2, "composition": {"space": "2d"},
            "nodes": [
                {"name":"solid", "type":"path", "d":"M0 0H30V30H0Z", "fill":"lime", "children":[
                    {"name":"nested-clip", "type":"path", "d":"M0 0H20V20H0Z"}
                ]},
                {"name":"clipped", "type":"path", "d":"M0 0H40V40H0Z", "fill":"blue", "clip-path":"nested-clip"},
                {"name":"clip", "type":"path", "d":"M0 0H50V50H0Z"},
                {"name":"group", "clip-path":"clip", "children":[
                    {"name":"shape", "type":"path", "d":"M0 0H100V100H0Z", "fill":"red",
                     "stroke":"blue", "stroke-width":3, "stroke-dasharray":[4,2], "paint-order":"stroke"},
                    {"name":"text", "type":"text", "content":"Cache", "font-size":20,
                     "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf"}
                ]}
            ]
        }))?)?;
        let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
        let nodes: Vec<_> = doc.items.iter().map(|i| &i.node).collect();
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let mut values = frame.values.clone();
        let mut retained = crate::canvas_draw::Retained::new(&doc.page, &nodes, &values)?;
        for step in 0..20 {
            for (i, value) in values.iter_mut().enumerate() {
                value.transform.translation[0] = (step * (i + 1)) as f32;
                value.transform.scale[0] = if step == 7 { 0. } else { 1. + step as f32 * 0.1 };
                value.opacity = (step % 5) as f32 * 0.2;
                value.visible = step % 4 != 0;
                if let Some(path) = &mut value.path {
                    path.stroke_width = (step % 3) as f32;
                    path.stroke_dashoffset = step as f32 * 0.5;
                    path.fill_opacity = 0.3 + step as f32 * 0.01;
                }
            }
            retained.update(&doc.page, &nodes, &values)?;
            retained.assert_fresh(&doc.page, &nodes, &values);
        }
        retained.update(&doc.page, &nodes, &frame.values)?;
        retained.assert_fresh(&doc.page, &nodes, &frame.values);
        Ok(())
    }

    /// An image or conic stroke is drawn through its coverage, with its picture in a rectangle around it. Its
    /// width and the paint opacities are fields of the nodes it already has: they are written in place, and the
    /// canvas is not lowered again.
    #[test]
    fn coverage_strokes_follow_their_values_without_an_install() -> Result<()> {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version": 2, "composition": {"space": "2d"},
            "nodes": [
                {"name":"ring", "type":"path", "d":"M32 32H96V96H32Z", "fill":"red",
                 "stroke":"conic-gradient(red, blue, red)", "stroke-width":8},
                {"name":"still", "type":"path", "d":"M0 0H10V10H0Z", "fill":"lime"}
            ]
        }))?)?;
        let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
        let nodes: Vec<_> = doc.items.iter().map(|i| &i.node).collect();
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let mut values = frame.values.clone();
        let mut retained = crate::canvas_draw::Retained::new(&doc.page, &nodes, &values)?;
        for step in 1..8 {
            let path = values[doc.by_name["ring"]].path.as_mut().unwrap();
            path.stroke_width = 2. + step as f32;
            path.fill_opacity = 1. - step as f32 * 0.1;
            path.stroke_opacity = 0.2 + step as f32 * 0.1;
            retained.update(&doc.page, &nodes, &values)?;
            retained.assert_fresh(&doc.page, &nodes, &values);
        }
        let counters: serde_json::Value = serde_json::from_str(&retained.profile_counters())?;
        assert!(counters["scene_installs"] == 0 && counters["row_write_frames"] == 7, "values are written into rows: {counters}");
        Ok(())
    }

    #[test]
    fn indexed_canvas_batches_match_fresh_nested_and_clipped_plans() -> Result<()> {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"}, "nodes":[
                {"name":"g","children":[
                    {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red"},
                    {"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"linear-gradient(blue, lime)"}]},
                {"name":"mask","type":"path","d":"M0 0H80V80H0Z"},
                {"name":"clipped","type":"path","d":"M0 0H100V100H0Z","fill":"blue","clip-path":"mask"}
            ]
        }))?)?;
        let doc = Document::new(&scene, Path::new("."))?;
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let nodes: Vec<_> = doc.items.iter().map(|item| &item.node).collect();
        let mut values = frame.values.clone();
        let mut retained = crate::canvas_draw::Retained::new(&doc.page, &nodes, &values)?;
        let builder: crate::canvas_draw::CanvasTextBuilder = std::rc::Rc::new(|_,_,_,_,_,_,_,_| Ok(None));
        let mut previous_plan = crate::canvas_draw::plan(&mut retained, &doc.page, builder.clone())?;
        let counter = |r: &crate::canvas_draw::Retained, name: &str| -> u64 {
            serde_json::from_str::<serde_json::Value>(&r.profile_counters()).unwrap()[name].as_u64().unwrap()
        };
        for (edits, content, patches) in [
            (vec![("g",11.),("a",3.)], false, true),
            (vec![("a",0.),("g",0.)], false, true),
            // The fade crosses opaque, so the planner prepares `a` again; `b` follows its recipe.
            (vec![("a",5.),("g",2.)], true, true),
            // The generated clip copy is under the translated wrapper, so its
            // mask changes too. Recorded units update the mask and rebuild its
            // consumers in place, retaining the rest of the plan.
            (vec![("clipped",7.)], false, true),
            (vec![("mask",9.)], false, true),
        ] {
            let changes_mask = edits.iter().any(|(name, _)| matches!(*name, "clipped" | "mask"));
            let before = counter(&retained, "incr_patch_hits");
            if content {
                values[doc.by_name["a"]].opacity = 0.5;
                retained.update(&doc.page, &nodes, &values)?;
            }
            // More than one update before planning must preserve the whole batch.
            for (name, x) in edits {
                values[doc.by_name[name]].transform.translation[0] = x;
                retained.update(&doc.page, &nodes, &values)?;
            }
            let actual = crate::canvas_draw::plan(&mut retained, &doc.page, builder.clone())?;
            if changes_mask {
                assert_ne!(actual.masks, previous_plan.masks, "translated clip geometry must update the mask plan");
            }
            assert_eq!(counter(&retained, "incr_patch_hits") - before, u64::from(patches));
            let mut fresh = crate::canvas_draw::Retained::new(&doc.page, &nodes, &values)?;
            let expected = crate::canvas_draw::plan(&mut fresh, &doc.page, builder.clone())?;
            assert_eq!(format!("{:?}", *actual), format!("{:?}", *expected));
            previous_plan = actual;
        }
        Ok(())
    }

    /// The oracle for typed placement writes: after every random batch of moves, turns, scales and fades, the
    /// retained scene equals a freshly written one and the retained plan equals a freshly built one.
    /// `uniform` names the nodes whose scale stays uniform (a blurred node refuses any other).
    fn placement_oracle(nodes: serde_json::Value, names: &[&str], leaves: &[&str], uniform: &[&str], group_fades: bool) -> Result<serde_json::Value> {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"}, "nodes": nodes}))?)?;
        let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let nodes: Vec<_> = doc.items.iter().map(|item| &item.node).collect();
        let fonts = doc.fonts.clone();
        let draws: HashMap<_, _> = frame.batches().map(|d| ((d.item, d.sub), d.clone())).collect();
        let builder: crate::canvas_draw::CanvasTextBuilder = std::rc::Rc::new(move |item, sub, model, z, clip, scale, opacity, _| {
            let mut placed: Draw = draws.get(&(item, sub)).context("a text batch the frame did not plan")?.clone();
            placed.model = model;
            Arc::make_mut(&mut placed.spec).opacity = opacity;
            fonts.build(&placed, z, clip, scale)
        });
        let mut values = frame.values.clone();
        let mut retained = crate::canvas_draw::Retained::new(&doc.page, &nodes, &values)?;
        crate::canvas_draw::plan(&mut retained, &doc.page, builder.clone())?;
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed >> 40) as f32 / (1u32 << 24) as f32 };
        for step in 0..400 {
            let batches = 1 + (next() * 3.) as usize;
            for _ in 0..batches {
                let name = names[(next() * names.len() as f32) as usize % names.len()];
                let leaf = leaves.contains(&name);
                let value = &mut values[doc.by_name[name]];
                match (next() * 5.) as u32 {
                    0 => value.transform.translation[0] = (next() * 200. - 100.).round() / 4.,
                    1 => value.transform.translation[1] = next() * 60. - 30.,
                    2 => { let half = next() * std::f32::consts::PI; *value.transform.rotation_mut() = [0., 0., half.sin(), half.cos()]; }
                    // Within and across tessellation bands, mirrored now and then.
                    3 => {
                        let scale = 0.4 + next() * 3.;
                        let other = [if next() < 0.2 { -scale } else { scale }, 0.4 + next() * 3., 1.];
                        value.transform.scale = if uniform.contains(&name) { [scale, scale, 1.] } else { other };
                    }
                    // A group that fades composites its children (not a placement): now and then only.
                    _ if leaf => value.opacity = if next() < 0.2 { 1. } else { 0.05 + next() * 0.9 },
                    _ if group_fades => value.opacity = if next() < 0.15 { 0.5 } else { 1. },
                    _ => value.transform.translation[0] = next() * 8.,
                }
                // More than one update before planning must preserve the whole batch.
                retained.update(&doc.page, &nodes, &values)?;
            }
            retained.assert_fresh(&doc.page, &nodes, &values);
            let actual = crate::canvas_draw::plan(&mut retained, &doc.page, builder.clone())?;
            let mut fresh = crate::canvas_draw::Retained::new(&doc.page, &nodes, &values)?;
            let expected = crate::canvas_draw::plan(&mut fresh, &doc.page, builder.clone())?;
            assert_eq!(format!("{:?}", *actual), format!("{:?}", *expected), "step {step}");
        }
        Ok(serde_json::from_str(&retained.profile_counters())?)
    }

    #[test]
    fn typed_placements_match_fresh_scenes_and_plans_over_random_edits() -> Result<()> {
        // Plain paths and groups: nearly every frame follows recipes.
        let plain = placement_oracle(serde_json::json!([
            {"name":"g","children":[
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red"},
                {"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"linear-gradient(blue, lime)"},
                {"name":"h","children":[
                    {"name":"c","type":"path","d":"M0 0C10 0 20 10 20 20S0 30 0 20Z","fill":"#336699","opacity":0.5},
                    {"name":"d","type":"path","d":"M5 5H9V9H5Z","fill":"#102030"}]}]},
            {"name":"lone","type":"path","d":"M60 60H90V90H60Z","fill":"#ffcc00cc"}
        ]), &["g", "a", "b", "h", "c", "d", "lone"], &["a", "b", "c", "d", "lone"], &[], false)?;
        assert!(plain["placement_frames"].as_u64().unwrap() > 400 && plain["row_write_frames"] == 0 && plain["scene_installs"] == 0, "placements are written typed: {plain}");
        assert!(plain["incr_placement_nodes"].as_u64().unwrap() > 200, "recipes follow placements: {plain}");
        // Strokes, clips, text and fading groups: recipes where they hold, the planner elsewhere.
        let mixed = placement_oracle(serde_json::json!([
            {"name":"g","children":[
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red","stroke":"#00ff0080","stroke-width":2},
                {"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"linear-gradient(blue, lime)"},
                {"name":"h","children":[
                    {"name":"c","type":"path","d":"M0 0C10 0 20 10 20 20S0 30 0 20Z","fill":"#336699","opacity":0.5}]}]},
            {"name":"lone","type":"path","d":"M60 60H90V90H60Z","fill":"#ffcc00cc"},
            {"name":"mask","type":"path","d":"M0 0H80V80H0Z"},
            {"name":"clipped","type":"path","d":"M0 0H100V100H0Z","fill":"blue","clip-path":"mask"},
            {"name":"text","type":"text","content":"Move","font-size":20,
             "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf"}
        ]), &["g", "a", "b", "h", "c", "lone", "clipped", "mask", "text"], &["a", "b", "c", "lone"], &[], true)?;
        assert!(mixed["placement_frames"].as_u64().unwrap() > 100, "placements are written typed: {mixed}");
        assert!(mixed["incr_placement_nodes"].as_u64().unwrap() > 30, "recipes follow placements: {mixed}");
        // A blurred node and a fading group are units: they are planned whole when they or their content move,
        // and the plain nodes beside them keep following recipes.
        let units = placement_oracle(serde_json::json!([
            {"name":"soft","type":"path","d":"M0 0H30V30H0Z","fill":"#cc3366","filter":"blur(3px)"},
            {"name":"fade","opacity":0.5,"children":[
                {"name":"in","type":"path","d":"M40 0H60V20H40Z","fill":"blue"},
                {"name":"in2","type":"path","d":"M50 10H70V30H50Z","fill":"lime"}]},
            {"name":"free","type":"path","d":"M60 60H90V90H60Z","fill":"#ffcc00"},
            {"name":"g","children":[
                {"name":"m","type":"path","d":"M0 40H20V60H0Z","fill":"red"},
                {"name":"n","type":"path","d":"M20 40H40V60H20Z","fill":"#102030"}]}
        ]), &["soft", "fade", "in", "free", "g", "m", "n", "free", "m", "n"], &["free", "m", "n"], &["soft"], false)?;
        assert!(units["incr_placement_nodes"].as_u64().unwrap() > 20, "plain nodes beside units follow recipes: {units}");
        Ok(())
    }

    /// Sparse placement frames through evaluation, the scene write and the plan: unchanged bodies before and
    /// between moved ones, a parent move, frames the recipes decline (a band change, a colour change), and a
    /// frame evaluated but never planned (so the next frame's moved-items list has a stale base).
    #[test]
    fn sparse_placement_frames_plan_like_fresh_documents() -> Result<()> {
        let source = serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"},
            "nodes":[
                {"name":"still0","type":"path","d":"M0 0H10V10H0Z","fill":"#111111"},
                {"name":"g","children":[
                    {"name":"a","type":"path","d":"M20 0H30V10H20Z","fill":"red"},
                    {"name":"still1","type":"path","d":"M40 0H50V10H40Z","fill":"#222222"},
                    {"name":"b","type":"path","d":"M60 0C70 0 70 10 60 10Z","fill":"blue"}]},
                {"name":"still2","type":"path","d":"M80 0H90V10H80Z","fill":"#333333"}],
            "animation":[{"tracks":[
                {"path":["nodes","g","translation"],"keys":[[0,[0,0,0]],[1,[0,0,0]],[2,[30,12,0]]]},
                {"path":["nodes","a","translation"],"keys":[[0,[0,0,0]],[1,[9,3,0]],[3,[9,3,0]]]},
                {"path":["nodes","b","scale"],"keys":[[0,[1,1,1]],[2,[1,1,1]],[3,[5,5,1]]]},
                {"path":["nodes","still2","fill"],"keys":[[0,[0.2,0.2,0.2,1]],[1.4,[0.2,0.2,0.2,1]],[1.6,[1,0,0,1]],[3,[1,0,0,1]]]}
            ]}]
        }))?;
        let document = || Document::new(&Scene::from_json(&source).unwrap(), Path::new("."));
        let kept = document()?;
        // (time, plan it): an unplanned frame leaves the scene one revision behind the next frame's base.
        for (time, planned) in [(0., true), (0.2, true), (0.5, true), (0.5, true), (0.9, true), (1.2, true), (1.3, false),
            (1.35, true), (1.5, true), (1.7, true), (2.2, true), (2.6, true), (2.9, true), (0.4, true), (0.6, true)] {
            let frame = kept.snapshot(&vork_time::Request::Numeric(time))?;
            if !planned { continue; }
            let plan = kept.plan_snapshot(&frame, &[])?;
            let fresh = document()?;
            let fresh_frame = fresh.snapshot(&vork_time::Request::Numeric(time))?;
            let expected = fresh.plan_snapshot(&fresh_frame, &[])?;
            assert_eq!(format!("{:?}", *plan), format!("{:?}", *expected), "time {time}");
        }
        let counters: serde_json::Value = serde_json::from_str(&kept.retained.borrow().as_ref().unwrap().profile_counters())?;
        assert!(counters["placement_frames"].as_u64().unwrap() >= 6, "moves are written sparsely: {counters}");
        assert!(counters["incr_placement_nodes"].as_u64().unwrap() >= 4, "recipes follow them: {counters}");
        Ok(())
    }

    #[test]
    fn retained_shadow_offsets_follow_each_transform_occurrence() -> Result<()> {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version": 2, "composition": {"space": "2d", "viewBox": [0, 0, 200, 100], "preserveAspectRatio": "none"},
            "render": {"resolution": [400, 300]},
            "nodes": [
                {"name":"mask", "type":"path", "d":"M0 0H50V50H0Z", "fill":"white",
                 "filter":"drop-shadow(3px 5px 2px red) blur(1px)"},
                {"name":"clip", "type":"path", "d":"M0 0H50V50H0Z"},
                {"name":"parent", "children":[
                    {"name":"shape", "type":"path", "d":"M0 0H40V30H0Z", "fill":"red",
                     "filter":"drop-shadow(8px 4px 0px blue)"},
                    {"name":"split", "type":"path", "d":"M0 0H40V30H0Z", "fill":"red",
                     "stroke":"blue", "stroke-width":2, "paint-order":"stroke",
                     "clip-path":"clip", "mask":"mask", "filter":"drop-shadow(8px 4px 0px lime)"},
                    {"name":"text", "type":"text", "content":"A", "font-size":24,
                     "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf",
                     "filter":"drop-shadow(8px 4px 0px lime)"},
                    {"name":"batched", "type":"text", "font-size":24, "stroke-width":2,
                     "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf",
                     "content":[{"content":"A", "stroke":"red"}, {"content":"B", "stroke":"blue"}],
                     "filter":"drop-shadow(8px 4px 0px lime)"},
                    {"name":"group", "filter":"drop-shadow(8px 4px 0px blue)", "children":[
                        {"type":"path", "d":"M0 0H40V30H0Z", "fill":"red"}
                    ]}
                ]},
                {"name":"other", "scale":[3,2,1], "mask":"mask", "children":[
                    {"type":"path", "d":"M0 0H40V30H0Z", "fill":"blue"}
                ]}
            ]
        }))?)?;
        let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
        let nodes: Vec<_> = doc.items.iter().map(|i| &i.node).collect();
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        assert!(frame.values[doc.by_name["batched"]].texts > 1);
        // Isolate own, ancestor and mask-instance changes; seek back after each edit.
        for name in ["shape", "split", "text", "batched", "group", "parent", "other", "mask"] {
            let mut values = frame.values.clone();
            let mut retained = crate::canvas_draw::Retained::new(&doc.page, &nodes, &values)?;
            let i = doc.by_name[name];
            for (scale, angle) in [([2., 3., 1.], 0.), ([2., 3., 1.], 1.2), ([-1., 2., 1.], -0.7)] {
                values[i].transform.scale = scale;
                values[i].transform.rotation = Some(glam::Quat::from_rotation_z(angle).to_array());
                retained.update(&doc.page, &nodes, &values)?;
                retained.assert_fresh(&doc.page, &nodes, &values);
            }
            // Rebuilding the hierarchy must refresh the effect bindings as well.
            values[i].visible = false;
            retained.update(&doc.page, &nodes, &values)?;
            values[i].visible = true;
            retained.update(&doc.page, &nodes, &values)?;
            retained.update(&doc.page, &nodes, &frame.values)?;
            retained.assert_fresh(&doc.page, &nodes, &frame.values);
        }
        Ok(())
    }

    #[test]
    fn plain_text_placement_preserves_glyph_records_and_content_replans() -> Result<()> {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"}, "nodes":[
                {"name":"shown", "type":"text", "content":"Cache", "font-size":40,
                 "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf"},
                {"visible":false, "children":[
                    {"type":"text", "content":"Hidden", "font-size":40,
                     "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf"}
                ]}
            ]
        }))?)?;
        let base = Path::new(env!("CARGO_MANIFEST_DIR"));
        let doc = Document::new(&scene, base)?;
        let mut frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let again = doc.snapshot(&vork_time::Request::Numeric(1.))?;
        assert_eq!(frame.batches().count(), 1);
        assert!(Arc::ptr_eq(&frame.batches().next().unwrap().spec, &again.batches().next().unwrap().spec));
        let index = doc.by_name["shown"];
        let texts = |plan: &moda_engine::seams::CoveragePlan| plan.items.iter().filter_map(|item| {
            if let moda_engine::seams::DrawItem::Text(batch) = item { Some(batch.clone()) } else { None }
        }).collect::<Vec<_>>();
        let mut previous_records = None;
        let mut previous_counts = (0, 0);
        for step in 0..5 {
            match step {
                1 => frame.values[index].transform.translation[0] = 17.,
                2 => {
                    Arc::make_mut(&mut frame.draws[index][0].spec).fill_override = Some("red".into());
                    assert_ne!(again.draws[index][0].spec.fill_override.as_deref(), Some("red"));
                    frame.values[index].text_revision += 1;
                }
                3 => frame.values[index].transform.scale[0] = 0.7,
                4 => frame.values[index].opacity = 0.3,
                _ => {}
            }
            let cached = doc.plan_snapshot(&frame, &[])?;
            let fresh = Document::new(&scene, base)?.plan_snapshot(&frame, &[])?;
            assert_eq!(texts(&cached).len(), 1);
            assert_eq!(texts(&cached), texts(&fresh), "step {step}");
            let (hits, misses) = doc.retained.borrow().as_ref().unwrap().text_cache_counts();
            let batches = texts(&cached);
            let records = &batches[0].vector.as_ref().unwrap().ops[0].records;
            match step {
                0 => assert_eq!((hits, misses), (0, 1)),
                1 | 3 => {
                    assert_eq!((hits, misses), previous_counts, "a resident transform skips text preparation");
                    assert!(Arc::ptr_eq(records, previous_records.as_ref().unwrap()));
                }
                2 => assert_eq!(misses, previous_counts.1 + 1, "editing content enters the text builder"),
                4 => {
                    assert_eq!(misses, previous_counts.1, "group opacity keeps glyph geometry");
                    assert!(Arc::ptr_eq(records, previous_records.as_ref().unwrap()));
                }
                _ => unreachable!(),
            }
            previous_records = Some(records.clone());
            previous_counts = (hits, misses);
        }
        Ok(())
    }

    /// Full animation oracle and CPU-only planning benchmark, with external qualification scenes.
    #[test]
    #[ignore = "requires VORK_CANVAS_CORPUS pointing at the external canvas qualification scenes"]
    fn retained_canvas_corpus() -> Result<()> {
        let root = PathBuf::from(std::env::var("VORK_CANVAS_CORPUS")?);
        let mut ports: Vec<_> = std::fs::read_dir(&root)?.collect::<std::io::Result<Vec<_>>>()?;
        ports.sort_by_key(|p| p.file_name());
        for port in ports {
            let path = port.path().join("port.json");
            if !path.exists() || port.file_name().to_string_lossy().starts_with("05-") { continue; }
            let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
            let duration = json["render"]["duration"].as_f64().unwrap_or(1.);
            let fps = json["render"]["fps"].as_f64().unwrap_or(30.);
            let scene = Scene::from_json(&serde_json::to_vec(&json)?)?;
            let doc = Document::new(&scene, &port.path())?;
            let nodes: Vec<_> = doc.items.iter().map(|i| &i.node).collect();
            crate::canvas_draw::install_images(&Default::default(), &nodes, &[])?;
            let frames = (duration * fps).round() as usize;
            let mut elapsed = std::time::Duration::ZERO;
            for f in (0..frames).chain((0..frames).rev().step_by(17)) {
                let start = Instant::now();
                let frame = doc.snapshot(&vork_time::Request::Frame(f as u32, scene.render.fps))?;
                let _plan = doc.plan_snapshot(&frame, &[])?;
                elapsed += start.elapsed();
                let nodes: Vec<_> = doc.items.iter().enumerate().map(|(i,item)|frame.nodes[i].as_deref().unwrap_or(&item.node)).collect();
                let mut page=doc.page.clone();page.background=frame.background;
                doc.retained.borrow().as_ref().unwrap().assert_fresh(&page, &nodes, &frame.values);
            }
            eprintln!("{}: {:.3} ms/plan ({frames} forward frames plus reverse seeks)", port.file_name().to_string_lossy(), elapsed.as_secs_f64() * 1000. / (frames + frames.div_ceil(17)) as f64);
        }
        Ok(())
    }
