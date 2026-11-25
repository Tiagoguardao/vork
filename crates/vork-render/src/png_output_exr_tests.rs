    use super::*;
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn rendered_png16_preserves_precision_and_scales_dither() -> Result<()> {
        let r = crate::test_renderer()?;
        assert!(!crate::InstallOptions::default().readback.color16);
        let readback = ReadbackOutputs { color: true, color16: true, ..Default::default() };
        let scene = serde_json::from_value(serde_json::json!({
            "version": 2, "nodes": [{"type": "camera"}],
            "composition":{"space":"3d","view":{"program": "display"}},"render": {"resolution": [16,16], "background": [0.25,0.25,0.25],
                "settings": {"samples": 1}},
            "programs": {"display": {"source": "import std.view; struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs i) {return finish_view(ctx.color,ctx,1.);}"}}
        }))?;
        let installed = r.install_with_options(scene, std::path::Path::new("."), crate::InstallOptions {
            readback, gpu_color: false, ..Default::default()
        })?;
        let mut frame = installed.create_frame_slot(&r, crate::ReadbackPrecision::Packed)?;
        frame.submit_outputs(&installed, &r, 0., readback)?;
        let pixels = frame.collect(&r)?;
        assert_eq!(pixels.rgba16.len(), 256);
        for (low, high) in pixels.rgba.chunks_exact(4).zip(&pixels.rgba16) {
            for c in 0..4 {
                assert!((f32::from(low[c])/255. - f32::from(high[c])/65535.).abs() <= 1./255.);
            }
            // A constant 0.25 input must not retain 8-bit dither in RGBA16.
            assert!((f32::from(high[0])/65535. - 0.25).abs() <= 1.5/65535.);
        }
        let files = encode(16,16,&pixels,&Outputs::new(vec![Output::Color]),PngDepth::Sixteen,Default::default())?;
        let path = std::env::temp_dir().join(format!("vork-png16-{}.png",std::process::id()));
        std::fs::write(&path,&files[0].1)?;
        let bytes = std::fs::read(&path)?;
        std::fs::remove_file(&path)?;
        let mut reader = png::Decoder::new(std::io::Cursor::new(bytes)).read_info()?;
        let mut decoded = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut decoded)?;
        assert_eq!(info.bit_depth,png::BitDepth::Sixteen);
        let expected: Vec<u8> = pixels.rgba16.iter().flatten().flat_map(|v|v.to_be_bytes()).collect();
        assert_eq!(&decoded[..info.buffer_size()],expected.as_slice());
        Ok(())
    }
    #[test]
    fn every_png_uses_the_fast_setting_and_decodes_to_its_pixels() -> Result<()> {
        assert!(matches!(COMPRESSION, png::Compression::Fast));
        let rgba: Vec<u8> = (0..64 * 64 * 4).map(|i| (i * 7 % 251) as u8).collect();
        let mut reader = png::Decoder::new(std::io::Cursor::new(encode_canvas(64, 64, &rgba, Default::default())?.remove(0).1)).read_info()?;
        let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut pixels)?;
        assert_eq!(&pixels[..info.buffer_size()], rgba.as_slice());
        Ok(())
    }
    #[test]
    fn aov_channels_cannot_shadow_built_in_layers() {
        assert!("aov:heat".parse::<Output>().is_ok());
        assert!("aov:color".parse::<Output>().unwrap_err().contains("collides"));
        assert!("aov:object_index".parse::<Output>().unwrap_err().contains("collides"));
        assert_eq!(Output::builtins().count(), vork_scene::v2::OUTPUT_CHANNELS.len());
        assert!("ZIP".parse::<crate::ExrCodec>().is_err());
        assert_eq!(("8".parse(), "16".parse()), (Ok(PngDepth::Eight), Ok(PngDepth::Sixteen)));
        assert!("32".parse::<PngDepth>().is_err());
    }
    #[test]
    fn channel_grammar_is_namespaced_and_builtin_defaults_are_stable() {
        // Unimplemented selections are unknown channels, not reserved promises.
        for token in ["aov:", "aov:a:b", "aov:a.b", "aov:../file", "unknown:x", "cryptomatte_object"] {
            assert!(token.parse::<Output>().is_err(), "{token}");
        }
        for token in ["cryptomatte:object", "cryptomatte:material", "cryptomatte:asset", "lightgroup:key"] {
            assert!(token.parse::<Output>().unwrap_err().starts_with("unknown output channel"), "{token}");
        }
        let defaults: Vec<Output> = Output::builtins().collect();
        assert_eq!(defaults, vec![Output::Color, Output::Depth, Output::ObjectIndex, Output::ViewNormal, Output::Motion, Output::Camera, Output::Aov("shadow_catcher".into())]);
    }
    #[test]
    fn multilayer_and_png_control_units_round_trip() -> Result<()> {
        use exr::prelude::*;
        let mut pixels = FrameOutput {
            aovs: [("heat".into(), crate::frame::AovPixels {
                kind: vork_scene::v2::AovType::Value,
                pixels: vec![[0.25, 0., 0., 0.], [0.75, 0., 0., 0.]],
            })].into(),
            rgba: vec![128, 128, 255, 128, 0, 0, 0, 0],
            rgba16: vec![[0x0102, 0x8000, 0xffff, 0x7fff], [0, 0, 0, 0]],
            linear_rgba: Some(vec![[0.09, 0.2158605 * 0.5, 2., 0.5], [0., 0., 0., 0.]]),
            depth_f32: Some(vec![1.2345679, 1e10]),
            normals_f32: Some(vec![[0.1234567, 0.2345678, 0.964], [0., 0., 0.]]),
            motion_f32: Some(vec![[0., 0.], [0., 0.]]),
            depth_u16: vec![100, 0],
            ids: vec![65535, 0],
            normals: vec![255, 128, 128, 0, 0, 0],
            motion: vec![[32768, 32768], [32768, 32768]],
            camera: serde_json::json!({}),
            exposure: None,
            timing: crate::FrameTiming {
                cpu_prepare_ms: 0.,
                cpu_submit_ms: 0.,
                cpu_finish_ms: 0.,
                denoise_wait_ms: 0.,
                denoise_pack_ms: 0.,
                denoise_filter_ms: 0.,
                denoise_upload_ms: 0.,
                gpu_ms: 0.,
                nonfinite: None,
                dropped_light_samples: 0,
                dropped_path_samples: 0,
                invalid_pixels: Default::default(),
                terminated_medium_paths: 0,
                discarded_medium_radiance: 0.,
                gpu_elapsed_ms: 0.,
                timing_source: "unmeasured",
                gpu_spans_ms: vec![],
                sample_count: 1,
                variants_pending: false,
                path_kernel: None,
            },
        };
        // Raw storage converts colour only: beauty and colour AOVs. Alpha, indices, value AOVs and the catcher
        // ratio keep their values, and the file names its primaries, white point, space and reference domain.
        // The second beauty pixel is light without coverage (a bloom halo on transparent film): it keeps its energy.
        {
            use vork_scene::v2::{AovType, WorkingColorSpace as S};
            let world: vork_scene::v2::Scene = serde_json::from_value(serde_json::json!({"version":2,"nodes":[{"type":"camera"}]}))?;
            let mut raw = pixels.clone();
            raw.linear_rgba = Some(vec![[2., -0.25, 0.5, 0.5], [0.5, 0.25, 0.125, 0.]]);
            let colour = [[2., -0.25, 0.5, 1.], [0.5, 0.25, 0.125, 0.]];
            for name in ["tint", "shadow_catcher"] {
                raw.aovs.insert(name.into(), crate::frame::AovPixels { kind: AovType::Color, pixels: colour.to_vec() });
            }
            let outputs = Outputs(vec![Output::Color, Output::ObjectIndex, Output::Aov("heat".into()), Output::Aov("tint".into()), Output::Aov("shadow_catcher".into())]);
            let mut default_bytes = None;
            for (storage, expected) in [
                (None, [2., -0.25, 0.5]),
                (Some(S::LinearRec709), [2., -0.25, 0.5]),
                (Some(S::LinearRec2020), [1.19414353, -0.08600947, 0.45857722]),
                (Some(S::ACEScg), [1.16500378, -0.08197488, 0.44874606]),
            ] {
                let destination = vork_scene::v2::OutputDestination { color_space: storage, .."raw.%06d.exr".into() };
                let color = crate::output_device::RawColor::of(&world, &destination);
                let storage = storage.unwrap_or(S::LinearRec709);
                assert_eq!(color.storage, storage);
                let encoded = encode_exr_clock(1, 2, &raw, &outputs, crate::ExrCodec::default(), Some((24, 1)), color, 1)?;
                let decoded = read().no_deep_data().largest_resolution_level().all_channels().all_layers().all_attributes()
                    .from_buffered(std::io::Cursor::new(&encoded[0].1))?;
                let layer = &decoded.layer_data[0];
                let channel = |name: &str| layer.channel_data.list.iter().find(|c| c.name.to_string() == name).unwrap().sample_data.values_as_f32().collect::<Vec<_>>();
                let halo = color.store([0.5, 0.25, 0.125]);
                for (i, name) in ["R", "G", "B"].into_iter().enumerate() {
                    for prefix in ["", "tint."] {
                        let values = channel(&format!("{prefix}{name}"));
                        assert!((values[0] - expected[i]).abs() < 2e-6 && values[1] == halo[i], "{storage:?} {prefix}{name}: {values:?}");
                    }
                    assert_eq!(channel(&format!("shadow_catcher.{name}")), vec![colour[0][i], colour[1][i]], "the catcher ratio is not converted");
                }
                assert_eq!(channel("A"), vec![0.5, 0.]);
                assert_eq!(channel("object_index.X"), vec![65535., 0.]);
                assert_eq!(channel("heat.X"), vec![0.25, 0.75]);
                let [red, green, blue, white] = storage.chromaticities().map(|[x, y]| Vec2(x, y));
                assert_eq!(decoded.attributes.chromaticities, Some(exr::meta::attribute::Chromaticities { red, green, blue, white }));
                assert_eq!(layer.attributes.other[&Text::from("vork:color_space")], AttributeValue::Text(Text::from(storage.name())));
                assert_eq!(layer.attributes.other[&Text::from("vork:reference_domain")], AttributeValue::Text(Text::from("scene")));
                assert_eq!(layer.attributes.frames_per_second, Some((24, 1)));
                // The working space stores the samples as they are, named or omitted: the same file.
                if storage == S::LinearRec709 {
                    assert_eq!([channel("R")[0], channel("G")[0], channel("B")[0]].map(f32::to_bits), [2f32, -0.25, 0.5].map(f32::to_bits));
                    assert_eq!(default_bytes.get_or_insert_with(|| encoded[0].1.clone()), &encoded[0].1);
                }
            }
        }
        // A canvas's EXR: its picture after its view, associated sRGB code values, through the
        // same writer. It is display-referred; omitted storage keeps the code values as they are, and a linear
        // space takes display-linear light, converted on unassociated colour.
        {
            use vork_scene::v2::WorkingColorSpace as S;
            let canvas: vork_scene::v2::Scene = serde_json::from_value(serde_json::json!({"version":2,"composition":{"space":"2d"},"nodes":[]}))?;
            let picture = [[0.25, 0.25, 0.25, 0.5], [1.5, -0.25, 0.5, 1.]];
            for (storage, expected) in [(None, picture), (Some(S::Srgb), picture),
                (Some(S::LinearRec709), [[0.21404114 * 0.5, 0.21404114 * 0.5, 0.21404114 * 0.5, 0.5], [2.5371552, -0.019349845, 0.21404114, 1.]])] {
                let destination = vork_scene::v2::OutputDestination { color_space: storage, .."frames.%06d.exr".into() };
                let color = crate::output_device::RawColor::of(&canvas, &destination);
                let encoded = encode_exr_canvas(1, 2, &picture, crate::ExrCodec::default(), (24, 1), color, 1)?;
                let decoded = read().no_deep_data().largest_resolution_level().all_channels().all_layers().all_attributes()
                    .from_buffered(std::io::Cursor::new(&encoded[0].1))?;
                let layer = &decoded.layer_data[0];
                let channel = |name: &str| layer.channel_data.list.iter().find(|c| c.name.to_string() == name).unwrap().sample_data.values_as_f32().collect::<Vec<_>>();
                for (i, name) in ["R", "G", "B", "A"].into_iter().enumerate() {
                    let values = channel(name);
                    for pixel in 0..2 {
                        if storage == Some(S::LinearRec709) { assert!((values[pixel] - expected[pixel][i]).abs() < 2e-5, "{name}: {values:?}"); }
                        else { assert_eq!(values[pixel].to_bits(), expected[pixel][i].to_bits(), "{name}: {values:?}"); }
                    }
                }
                let storage = storage.unwrap_or(S::Srgb);
                let [red, green, blue, white] = storage.chromaticities().map(|[x, y]| Vec2(x, y));
                assert_eq!(decoded.attributes.chromaticities, Some(exr::meta::attribute::Chromaticities { red, green, blue, white }));
                assert_eq!(layer.attributes.other[&Text::from("vork:color_space")], AttributeValue::Text(Text::from(storage.name())));
                assert_eq!(layer.attributes.other[&Text::from("vork:reference_domain")], AttributeValue::Text(Text::from("display")));
                assert_eq!(layer.attributes.frames_per_second, Some((24, 1)));
            }
        }
        // The public PNG protocol uses network byte order, including the zero
        // blue component of the RGB16 motion image. Decode the actual files so
        // this also checks their PNG type, depth and dimensions.
        let pngs = encode(1, 2, &pixels, &Outputs(vec![Output::Depth, Output::ObjectIndex, Output::Motion]), PngDepth::Eight, Default::default())?;
        for (suffix, color, expected) in [
            (".depth.png", png::ColorType::Grayscale, vec![0, 100, 0, 0]),
            (".object_index.png", png::ColorType::Grayscale, vec![255, 255, 0, 0]),
            (".motion.png", png::ColorType::Rgb, vec![128, 0, 128, 0, 0, 0, 128, 0, 128, 0, 0, 0]),
        ] {
            let bytes = &pngs.iter().find(|(name, _)| name == suffix).unwrap().1;
            let mut reader = png::Decoder::new(std::io::Cursor::new(bytes)).read_info()?;
            assert!(reader.info().srgb.is_none(), "control channels are data");
            let mut decoded = vec![0; reader.output_buffer_size().unwrap()];
            let info = reader.next_frame(&mut decoded)?;
            assert_eq!((info.width, info.height, info.color_type, info.bit_depth), (1, 2, color, png::BitDepth::Sixteen));
            assert_eq!(&decoded[..info.buffer_size()], &expected, "{suffix}");
        }
        // Colour follows the selected depth; 16-bit components are big-endian RGBA16.
        for (depth, bit_depth, expected) in [
            (PngDepth::Eight, png::BitDepth::Eight, vec![128, 128, 255, 128, 0, 0, 0, 0]),
            (PngDepth::Sixteen, png::BitDepth::Sixteen, vec![1, 2, 128, 0, 255, 255, 127, 255, 0, 0, 0, 0, 0, 0, 0, 0]),
        ] {
            let files = encode(1, 2, &pixels, &Outputs(vec![Output::Color]), depth, Default::default())?;
            assert_eq!(files.len(), 1);
            assert_eq!(files[0].0, ".png");
            let mut reader = png::Decoder::new(std::io::Cursor::new(&files[0].1)).read_info()?;
            assert_eq!(reader.info().srgb,Some(png::SrgbRenderingIntent::Perceptual));
            let mut decoded = vec![0; reader.output_buffer_size().unwrap()];
            let info = reader.next_frame(&mut decoded)?;
            assert_eq!((info.width, info.height, info.color_type, info.bit_depth), (1, 2, png::ColorType::Rgba, bit_depth));
            assert_eq!(&decoded[..info.buffer_size()], &expected, "{depth:?}");
        }
        // A picture for another display names it in cICP and carries no sRGB chunk; its pixels are the same
        // plane, and control channels beside it stay untagged data.
        let chunks = |bytes: &[u8]| {
            let (mut chunks, mut offset) = (Vec::new(), 8);
            while offset < bytes.len() {
                let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                chunks.push((bytes[offset + 4..offset + 8].to_vec(), bytes[offset + 8..offset + 8 + length].to_vec()));
                offset += 12 + length;
            }
            chunks
        };
        let srgb = encode(1, 2, &pixels, &Outputs(vec![Output::Color]), PngDepth::Eight, Default::default())?;
        assert!(chunks(&srgb[0].1).iter().all(|(name, _)| name != b"cICP"));
        let wide = encode(1, 2, &pixels, &Outputs(vec![Output::Color, Output::Depth]), PngDepth::Eight, vork_scene::v2::OutputDisplay::DisplayP3)?;
        let tagged = chunks(&wide[0].1);
        assert_eq!(tagged.iter().find(|(name, _)| name == b"cICP").map(|(_, data)| data.as_slice()), Some(&[12, 13, 0, 1][..]));
        assert!(tagged.iter().all(|(name, _)| !matches!(name.as_slice(), b"sRGB" | b"gAMA" | b"cHRM" | b"iCCP")));
        let position = |name: &[u8]| tagged.iter().position(|(chunk, _)| chunk == name).unwrap();
        assert!(position(b"cICP") < position(b"IDAT"));
        assert!(chunks(&wide[1].1).iter().all(|(name, _)| !matches!(name.as_slice(), b"cICP" | b"sRGB")));
        assert_eq!(png::Decoder::new(std::io::Cursor::new(&wide[0].1)).read_info()?.info().srgb, None);
        for (display, expected) in [(vork_scene::v2::OutputDisplay::Rec2100Pq, [9, 16, 0, 1]), (vork_scene::v2::OutputDisplay::Rec2100Hlg, [9, 18, 0, 1])] {
            assert_eq!(cicp(display), Some(expected));
        }
        // A canvas's picture names its display the same way.
        let canvas = encode_canvas(1, 1, &[1, 2, 3, 255], vork_scene::v2::OutputDisplay::DisplayP3)?;
        let tagged = chunks(&canvas[0].1);
        assert_eq!(tagged.iter().find(|(name, _)| name == b"cICP").map(|(_, data)| data.as_slice()), Some(&[12, 13, 0, 1][..]));
        assert!(tagged.iter().all(|(name, _)| name != b"sRGB"));
        assert!(chunks(&encode_canvas(1, 1, &[1, 2, 3, 255], Default::default())?[0].1).iter().any(|(name, _)| name == b"sRGB"));
        let files = encode_exr(
            1,
            2,
            &pixels,
            &Outputs(vec![
                Output::Color,
                Output::Depth,
                Output::ObjectIndex,
                Output::ViewNormal,
                Output::Motion,
                Output::Camera,
                Output::Aov("heat".into()),
            ]),
            crate::ExrCodec::default(),
        )?;
        assert_eq!(files.len(), 1);
        let image = read()
            .no_deep_data()
            .largest_resolution_level()
            .all_channels()
            .all_layers()
            .all_attributes()
            .from_buffered(std::io::Cursor::new(&files[0].1))?;
        assert_eq!(image.layer_data[0].encoding.compression, Compression::ZIP16);
        assert_eq!(image.layer_data.len(), 1);
        assert_eq!(image.layer_data[0].channel_data.list.len(), 12);
        let sample = |channel: &str| {
            image.layer_data[0]
                .channel_data
                .list
                .iter()
                .find(|c| c.name.to_string() == channel)
                .unwrap()
                .sample_data
                .values_as_f32()
                .collect::<Vec<_>>()
        };
        assert_eq!(sample("Z"), [1.2345679, 1e10]);
        assert_eq!(sample("object_index.X"), [65535., 0.]);
        assert_eq!(sample("heat.X"), [0.25, 0.75]);
        assert_eq!(sample("R"), [0.09, 0.]);
        assert_eq!(sample("G"), [0.2158605 * 0.5, 0.]);
        assert_eq!(sample("B"), [2., 0.]);
        assert_eq!(sample("A"), [0.5, 0.]);
        assert_eq!(sample("view_normal.X"), [0.1234567, 0.]);
        assert_eq!(sample("motion.X"), [0., 0.]);
        assert_eq!(sample("motion.Y"), [0., 0.]);
        // Beauty and depth sit in the default layer; every other layer is a
        // built-in channel name, reserved against AOV names at scene validation.
        for channel in &image.layer_data[0].channel_data.list {
            let name = channel.name.to_string();
            match name.split_once('.') {
                None => assert!(["R", "G", "B", "A", "Z"].contains(&name.as_str()), "{name}"),
                Some((layer, _)) => assert!(layer == "heat" || vork_scene::v2::OUTPUT_CHANNELS.contains(&layer), "{name}"),
            }
        }
        for name in ["none", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab"] {
            let codec: crate::ExrCodec = name.parse().unwrap();
            let files = encode_exr(1, 2, &pixels, &Outputs(vec![Output::Color, Output::Depth, Output::ObjectIndex]), codec)?;
            let image = read().no_deep_data().largest_resolution_level().all_channels()
                .all_layers().all_attributes().from_buffered(std::io::Cursor::new(&files[0].1))?;
            assert_eq!(std::mem::discriminant(&image.layer_data[0].encoding.compression),
                std::mem::discriminant(&codec.encoding().compression), "{name}");
            let ids = image.layer_data[0].channel_data.list.iter()
                .find(|c| c.name.to_string() == "object_index.X").unwrap()
                .sample_data.values_as_f32().collect::<Vec<_>>();
            assert_eq!(ids, [65535., 0.], "{name} must retain exact object IDs");
        }
        // Declared lists beyond the former shader budget keep their names and
        // values through EXR and explicit --channels selections.
        for count in [0, 1, 3, 8, 32] {
            pixels.aovs = (0..count).map(|i| (format!("a{i}"), crate::frame::AovPixels {
                kind: if i % 2 == 0 {vork_scene::v2::AovType::Value} else {vork_scene::v2::AovType::Color},
                pixels: vec![[(i+1) as f32 / 64., 0.5, 0.75, 1.]; 2],
            })).collect();
            let outputs = Outputs::new(std::iter::once(Output::Color).chain((0..count).map(|i| Output::Aov(format!("a{i}")))).collect());
            let files = encode_exr(1, 2, &pixels, &outputs, crate::ExrCodec::default())?;
            let image = read().no_deep_data().largest_resolution_level().all_channels()
                .all_layers().all_attributes().from_buffered(std::io::Cursor::new(&files[0].1))?;
            let channels = &image.layer_data[0].channel_data.list;
            assert_eq!(channels.len(), 4 + (0..count).map(|i| if i%2==0 {1} else {3}).sum::<usize>());
            for i in 0..count {
                for (component, expected) in if i%2==0 {vec![("X", (i+1) as f32/64.)]} else {vec![("R", (i+1) as f32/64.), ("G", 0.5), ("B", 0.75)]} {
                    let name = format!("a{i}.{component}");
                    let channel = channels.iter().find(|c| c.name.to_string()==name).unwrap();
                    assert_eq!(channel.sample_data.values_as_f32().collect::<Vec<_>>(), vec![expected;2]);
                }
            }
            if count>0 {
                let selection: Output = format!("aov:a{}",count-1).parse().unwrap();
                let selected = Outputs::new(vec![selection]);
                let png = encode(1,2,&pixels,&selected,PngDepth::Eight,Default::default())?;
                assert_eq!(png.len(),1);
                assert_eq!(png[0].0,format!(".aov.a{}.png",count-1));
                let files = encode_exr(1,2,&pixels,&selected,crate::ExrCodec::default())?;
                let image = read().no_deep_data().largest_resolution_level().all_channels()
                    .all_layers().all_attributes().from_buffered(std::io::Cursor::new(&files[0].1))?;
                assert!(image.layer_data[0].channel_data.list.iter().all(|c| c.name.to_string().starts_with(&format!("a{}.",count-1))));
            }
        }
        Ok(())
    }

    #[test]
    fn parallel_zip_keeps_float_bits_and_order_across_batches() -> Result<()> {
        use exr::prelude::*;

        // Three scanline blocks. The last one has a partial block and batch.
        let (width, height) = (64, 33);
        let count = (width * height) as usize;
        let samples = |channel: usize| {
            (0..count)
                .map(|i| f32::from_bits(match (i + channel) % 7 {
                    0 => 0,
                    1 => 0x8000_0000,
                    2 => 0x7fc0_0123,
                    3 => 0x7f80_0000,
                    4 => 1,
                    _ => 0x3f00_0000 + (i + channel) as u32,
                }))
                .collect::<Vec<_>>()
        };
        let names = ["R", "A", "motion.X", "Z"];
        let encode = |batch| {
            let channels = names.iter().enumerate().map(|(i, &name)| {
                AnyChannel::new(name, FlatSamples::F32(samples(i)))
            }).collect();
            write_exr(width, height, channels, crate::ExrCodec::Zip, Some((24, 1)), Default::default(), batch)
        };

        let serial = encode(1)?;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
        let parallel = pool.install(|| encode(2))?;
        // Custom attributes are stored in a hash map. Compare their values
        // and the ordered offset table and chunks independently.
        let mut serial_bytes = std::io::Cursor::new(serial[0].1.as_slice());
        let mut parallel_bytes = std::io::Cursor::new(parallel[0].1.as_slice());
        let serial_meta = exr::meta::MetaData::read_from_buffered(&mut serial_bytes, true)?;
        let parallel_meta = exr::meta::MetaData::read_from_buffered(&mut parallel_bytes, true)?;
        assert_eq!(serial_meta.headers, parallel_meta.headers);
        assert_eq!(serial_bytes.position(), parallel_bytes.position());
        let chunks = serial_bytes.position() as usize;
        assert_eq!(&serial[0].1[chunks..], &parallel[0].1[chunks..]);

        let decoded = read().no_deep_data().largest_resolution_level().all_channels()
            .all_layers().all_attributes().from_buffered(std::io::Cursor::new(&parallel[0].1))?;
        let layer = &decoded.layer_data[0];
        assert_eq!(layer.encoding.compression, Compression::ZIP16);
        assert_eq!(layer.encoding.line_order, LineOrder::Increasing);
        assert_eq!(layer.attributes.frames_per_second, Some((24, 1)));
        for (i, name) in names.into_iter().enumerate() {
            let channel = layer.channel_data.list.iter().find(|c| c.name.to_string() == name).unwrap();
            let FlatSamples::F32(decoded) = &channel.sample_data else {
                panic!("{name} must remain float32");
            };
            assert!(decoded.iter().map(|v| v.to_bits()).eq(samples(i).into_iter().map(f32::to_bits)), "{name}");
        }
        Ok(())
    }
