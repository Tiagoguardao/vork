    use super::*;

    #[test]
    fn zero_opacity_requires_static_table_proof() -> Result<()> {
        let mut material = PendingMaterial::program(0, false);
        let mut builder = RecordBuilder::openpbr();
        builder.set("geometry_opacity", &[0.]);
        material.record = builder.record;
        assert!(material.clone().finish().zero_opacity);
        material.opacity_animated = true;
        assert!(!material.clone().finish().zero_opacity);
        material.opacity_animated = false;
        material.record.tag[2] |= constants::FLAG_VERTEX_ALPHA;
        assert!(!material.clone().finish().zero_opacity);
        material.record.tag[2] &= !constants::FLAG_VERTEX_ALPHA;
        material.record.tag[0] = 1;
        assert!(!material.clone().finish().zero_opacity);
        material.record.tag[0] = 0;
        builder.bind("geometry_opacity", Tap::plain(0), 3)?;
        material.record = builder.record;
        assert!(!material.clone().finish().zero_opacity);
        builder = RecordBuilder::openpbr();
        builder.set("geometry_opacity", &[f32::MIN_POSITIVE]);
        material.record = builder.record;
        assert!(!material.finish().zero_opacity);
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn mip_chain_preserves_odd_extent_premultiplied_coverage() -> Result<()> {
        let r = crate::test_renderer()?;
        let (pipelines,sampler)=mip_resources(&r);
        let pipeline=&pipelines[2];
        let readback=crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {
            label:Some("mip area readback"),size:256,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false,
        });
        for [width,height] in [[3u32,1u32],[3,3],[480,270]] {
            let levels=32-width.max(height).leading_zeros();
            let texture=crate::gpu_memory::texture(&r.device,&wgpu::TextureDescriptor {
                label:Some("mip area test"),size:extent(width,height),mip_level_count:levels,sample_count:1,
                dimension:wgpu::TextureDimension::D2,format:wgpu::TextureFormat::Rgba16Float,
                usage:wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::RENDER_ATTACHMENT|wgpu::TextureUsages::COPY_DST|wgpu::TextureUsages::COPY_SRC,view_formats:&[],
            });
            let pixels:Vec<u16>=(0..width*height).flat_map(|p| {
                let value=half::f16::from_f32(if p%width%3==1 {0.} else {1.}).to_bits();
                [value;4]
            }).collect();
            r.queue.write_texture(wgpu::TexelCopyTextureInfo {texture:&texture,mip_level:0,origin:wgpu::Origin3d::ZERO,aspect:wgpu::TextureAspect::All},
                bytemuck::cast_slice(&pixels),wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(width*8),rows_per_image:Some(height)},extent(width,height));
            let mut encoder=r.device.create_command_encoder(&Default::default());
            MipChain::new(&r.device,&texture,None,pipeline,sampler).encode(&mut encoder,None);
            encoder.copy_texture_to_buffer(wgpu::TexelCopyTextureInfo {texture:&texture,mip_level:levels-1,origin:wgpu::Origin3d::ZERO,aspect:wgpu::TextureAspect::All},
                wgpu::TexelCopyBufferInfo {buffer:&readback,layout:wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(256),rows_per_image:Some(1)}},extent(1,1));
            let submission=crate::command_finish::Segments::submit(&r.queue, encoder);
            let (tx,rx)=std::sync::mpsc::channel();
            readback.map_async(wgpu::MapMode::Read,..,move |value| {let _=tx.send(value);});
            r.wait(Some(submission))?;rx.recv()??;
            let bytes=readback.get_mapped_range(..);
            for (c,&bits) in bytemuck::cast_slice::<u8,u16>(&bytes[..8]).iter().enumerate() {
                let value=half::f16::from_bits(bits).to_f32();
                assert!((value-2./3.).abs()<0.003,"{width}x{height} channel {c}: {value}");
            }
            drop(bytes);readback.unmap();
        }
        Ok(())
    }

    #[test]
    fn gltf_ior_sentinel_is_refused_before_shading() -> Result<()> {
        // The no-op device runs the actual importer without opening a GPU.
        let renderer = pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
        let render_images = Default::default();
        let mut textures = TextureInstaller::new(&renderer, &render_images);
        let witness: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/gltf-conventions/s15/witness.gltf"
        ))?;
        for ior in [0., 0.5, 1., 1.33, 1.5, 2.42] {
            let mut input = witness.clone();
            input["materials"][0]["extensions"]["KHR_materials_ior"]["ior"] = serde_json::json!(ior);
            let doc = gltf::Gltf::from_slice(&serde_json::to_vec(&input)?)?.document;
            let material = doc.materials().next().unwrap();
            let result = install_gltf(&mut textures, &doc, Some(&material), &[], &[]);
            if ior < 1. {
                let error = result.err().expect("glTF IOR sentinel must not enter the shading model");
                assert_eq!(error.to_string(), "KHR_materials_ior: infinite or sub-unit IOR is unsupported; use a finite ior of at least 1; see vork help upgrade");
            } else {
                assert_eq!(result?.record.get("specular_ior"), &[ior]);
            }
        }
        let doc = gltf::Gltf::from_slice(br#"{"asset":{"version":"2.0"}}"#)?.document;
        assert_eq!(install_gltf(&mut textures, &doc, None, &[], &[])?.record.get("specular_ior"), &[1.5]);
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn gltf_alpha_modes_keep_coverage_and_distinct_residency() -> Result<()> {
        let r = crate::test_renderer()?;
        let render_images = Default::default();
        let mut textures = TextureInstaller::new(&r, &render_images);
        let doc = gltf::Gltf::from_slice(
            br#"{
            "asset":{"version":"2.0"},
            "images":[{"uri":"alpha.png"}],"textures":[{"source":0}],
            "materials":[
                {"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}},
                {"alphaMode":"MASK","pbrMetallicRoughness":{"baseColorTexture":{"index":0}}},
                {"alphaMode":"BLEND","pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}
            ]
        }"#,
        )?
        .document;
        let pixels: Vec<_> = (0..16).flat_map(|i| [17, 42, 93, (i * 17) as u8]).collect();
        let key = blake3::hash(&pixels);
        let images = [gltf::image::Data {
            pixels,
            format: gltf::image::Format::R8G8B8A8,
            width: 4,
            height: 4,
        }.into()];
        for material in doc.materials() {
            let installed=install_gltf(&mut textures, &doc, Some(&material), &images, &[key])?;
            assert_eq!(installed.record.tag[0],0,"ordinary glTF stays in the shared material table");
            assert_ne!(installed.record.tag[2]&vork_slang::registry::FLAG_SAMPLE_MATERIAL,0,"textured imports retain covered-sample evaluation");
            assert!(installed.record.bound("base_color"),"imported texture taps stay in the table");
            assert!(!installed.transparent,"non-transmitting imports use bounded camera visibility");
            assert_eq!(installed.alpha_mask,material.alpha_mode()!=gltf::material::AlphaMode::Opaque);
            assert_eq!(installed.record.extra[1],match material.alpha_mode() {gltf::material::AlphaMode::Blend=>2.,gltf::material::AlphaMode::Mask=>3.,gltf::material::AlphaMode::Opaque=>0.});
        }
        let resident = textures.resident.borrow();
        assert!(resident.contains_key(&(key, true, false)));
        assert!(resident.contains_key(&(key, true, true)));
        assert_eq!(resident.len(), 2);
        r.errors.check()?;
        Ok(())
    }

    /// Regenerate with VORK_BLESS=1 cargo test -p vork-render generated_slang.
    #[test]
    fn generated_slang_modules_are_current() {
        for (name, text) in input_slang() {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("shaders/generated/{name}.slang"));
            if std::env::var_os("VORK_BLESS").is_some() {
                std::fs::write(&path, &text).unwrap();
            }
            assert_eq!(
                std::fs::read_to_string(&path).unwrap_or_default(),
                text,
                "{} is stale; regenerate with VORK_BLESS=1 cargo test -p vork-render generated_slang",
                path.display()
            );
        }
    }
    use ::vork_scene::v2::Scene;
    #[test]
    fn every_input_has_a_constant_and_a_tap_byte() {
        let words = std::mem::size_of_val(&MaterialRecord::zeroed().values) / 4;
        for (index, p) in OPENPBR.iter().enumerate() {
            assert!(
                value_word(index) + value_width(p.kind) <= words,
                "{}",
                p.name
            );
        }
        assert!(OCCLUSION_INPUT < constants::MATERIAL_INPUT_WORDS * 4);
        let base = OPENPBR.iter().position(|p| p.name == "base_color").unwrap();
        assert_eq!(
            value_location(2, base),
            Some((
                (2 * std::mem::size_of::<MaterialRecord>()
                    + std::mem::offset_of!(MaterialRecord, values)
                    + value_word(base) * 4) as u64,
                3
            ))
        );
    }
    #[test]
    fn sidedness_changes_material_data_without_changing_shader_features() -> Result<()> {
        // Native table and authored-program sidedness is material data, not code.
        let native_scene = scene(serde_json::json!({
            "paint":{}, "front":{"double_sided":false}, "both":{"double_sided":true}}));
        let front = openpbr_pending(&native_scene, Some("front"), &Default::default(), values(&native_scene, "front")?).finish();
        let both = openpbr_pending(&native_scene, Some("both"), &Default::default(), values(&native_scene, "both")?).finish();
        assert_eq!(front.features, both.features);
        assert_eq!(PendingMaterial::program(1, false).finish().features,
                   PendingMaterial::program(1, true).finish().features);

        // Adding a material-less glTF must not introduce a single-sided variant.
        let renderer = pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
        let render_images = Default::default();
        let mut textures = TextureInstaller::new(&renderer, &render_images);
        let mut input: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/gltf-conventions/s06/witness.gltf"))?;
        let document = gltf::Gltf::from_slice(&serde_json::to_vec(&input)?)?.document;
        let missing = install_gltf(&mut textures, &document, None, &[], &[])?;
        assert_eq!(missing.record.tag[2] & FLAG_DOUBLE_SIDED, 0);
        let features = missing.finish().features;
        input["materials"] = serde_json::json!([{"doubleSided":true}]);
        let document = gltf::Gltf::from_slice(&serde_json::to_vec(&input)?)?.document;
        let material = document.materials().next().unwrap();
        let both = install_gltf(&mut textures, &document, Some(&material), &[], &[])?;
        assert_ne!(both.record.tag[2] & FLAG_DOUBLE_SIDED, 0);
        assert_eq!(features, both.finish().features);
        Ok(())
    }

    #[test]
    fn gltf_missing_material_uses_single_sided_defaults() -> Result<()> {
        let renderer = pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
        let render_images = Default::default();
        let mut textures = TextureInstaller::new(&renderer, &render_images);
        let original: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/gltf-conventions/s06/witness.gltf"))?;
        let document = gltf::Gltf::from_slice(&serde_json::to_vec(&original)?)?.document;
        let missing = install_gltf(&mut textures, &document, None, &[], &[])?;
        let default_key = missing.key();
        assert!(!missing.double_sided);
        assert!(!missing.preserve_solid_backfaces);
        assert_eq!(missing.record.tag[2] & FLAG_DOUBLE_SIDED, 0);
        for material in [serde_json::json!({}), serde_json::json!({"doubleSided":false}),
                         serde_json::json!({"doubleSided":true})] {
            let double_sided = material.get("doubleSided").and_then(|v|v.as_bool()).unwrap_or(false);
            let mut input = original.clone();
            input["materials"] = serde_json::json!([material]);
            input["meshes"][0]["primitives"][0]["material"] = serde_json::json!(0);
            let document = gltf::Gltf::from_slice(&serde_json::to_vec(&input)?)?.document;
            let material = document.materials().next().unwrap();
            let pending = install_gltf(&mut textures, &document, Some(&material), &[], &[])?;
            assert_eq!(pending.double_sided, double_sided);
            assert!(!pending.preserve_solid_backfaces);
            assert_eq!(pending.record.tag[2] & FLAG_DOUBLE_SIDED != 0, double_sided);
            if !double_sided { assert_eq!(pending.key(), default_key); }
        }
        let native: ::vork_scene::v2::Material = serde_json::from_value(serde_json::json!({}))?;
        assert!(native.double_sided, "native authored materials retain their own default");
        assert!(PendingMaterial::program(1, false).finish().preserve_solid_backfaces);
        let native_scene = scene(serde_json::json!({"paint":{"double_sided":false}}));
        assert!(openpbr_pending(&native_scene, Some("paint"), &Default::default(), values(&native_scene, "paint")?).finish().preserve_solid_backfaces);
        // The volume extension overrides surface sidedness, independently of
        // transmission weight; unlit ignores the other material extensions.
        for thickness in [0., 1.] {
            for transmission in [0., 1.] {
                for unlit in [false, true] {
                    let mut input = original.clone();
                    input["extensionsUsed"] = serde_json::json!(["KHR_materials_volume", "KHR_materials_transmission"]);
                    let mut extensions = serde_json::json!({
                        "KHR_materials_volume":{"thicknessFactor":thickness},
                        "KHR_materials_transmission":{"transmissionFactor":transmission}});
                    if unlit {
                        extensions["KHR_materials_unlit"] = serde_json::json!({});
                        input["extensionsUsed"].as_array_mut().unwrap().push("KHR_materials_unlit".into());
                    }
                    input["materials"] = serde_json::json!([{"doubleSided":false,"extensions":extensions}]);
                    input["meshes"][0]["primitives"][0]["material"] = serde_json::json!(0);
                    let document = gltf::Gltf::from_slice(&serde_json::to_vec(&input)?)?.document;
                    let material = document.materials().next().unwrap();
                    let pending = install_gltf(&mut textures, &document, Some(&material), &[], &[])?;
                    let both = thickness > 0. && !unlit;
                    assert_eq!(pending.double_sided, both);
                    assert!(!pending.preserve_solid_backfaces);
                    assert_eq!(pending.record.tag[2] & FLAG_DOUBLE_SIDED != 0, both);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn gltf_surface_tint_and_volume_absorption_have_independent_inputs() -> Result<()> {
        let renderer = pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
        let render_images = Default::default();
        let mut textures = TextureInstaller::new(&renderer, &render_images);
        for witness in [
            include_str!("../tests/fixtures/gltf-conventions/s09/witness.gltf"),
            include_str!("../tests/fixtures/gltf-conventions/s10/witness.gltf"),
        ] {
            let mut input: serde_json::Value = serde_json::from_str(witness)?;
            for base in [[1., 0., 0., 1.], [0.25, 0.5, 0.75, 1.]] {
                input["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] = serde_json::json!(base);
                for absorption in [None, Some([1., 1., 1.]), Some([0.5, 0.75, 0.25])] {
                    let extensions = input["materials"][0]["extensions"].as_object_mut().unwrap();
                    extensions.remove("KHR_materials_volume");
                    if let Some(color) = absorption {
                        extensions.insert("KHR_materials_volume".into(), serde_json::json!({
                            "thicknessFactor": 1, "attenuationDistance": 2, "attenuationColor": color
                        }));
                    }
                    let doc = gltf::Gltf::from_slice(&serde_json::to_vec(&input)?)?.document;
                    let material = doc.materials().next().unwrap();
                    let installed = install_gltf(&mut textures, &doc, Some(&material), &[], &[])?;
                    assert_eq!(installed.record.get("base_color"), &base[..3]);
                    assert_eq!(installed.record.get("transmission_color"), &absorption.unwrap_or([1.; 3]));
                    assert_eq!(installed.record.get("transmission_depth"), &[if absorption.is_some() {2.} else {0.}]);
                    assert_ne!(installed.record.tag[2] & constants::FLAG_VERTEX_COLOR, 0);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn gltf_source_uv_indices_resolve_independently_of_packed_channels() -> Result<()> {
        let mut sources = GltfUvSources::default();
        let third = sources.bind(2)?;
        assert_eq!(third & constants::FLAG_UV_SECOND, 0);
        assert_ne!(third & constants::FLAG_UV_BASIS, 0);
        assert_eq!(sources.bind(2)?, third);
        assert_eq!(sources.bind(7)? & constants::FLAG_UV_SECOND, 1);
        assert_eq!(sources.sources(), [2, 7]);
        assert!(sources.bind(0).unwrap_err().to_string().contains("two distinct UV sets"));
        let mut sources = GltfUvSources::default();
        assert_eq!(sources.bind(1)? & constants::FLAG_UV_SECOND, 1);
        assert_eq!(sources.bind(0)?, 0);
        assert_eq!(sources.sources(), [0, 1]);
        let transform = serde_json::json!({"texCoord":7,"scale":[2.,3.],"offset":[0.2,0.4],"rotation":0.5});
        let (set, scale, rotation, offset) = gltf_uv(0, Some(&transform))?;
        assert_eq!((set, scale, rotation, offset), (7, [2.,3.], 0.5, [0.2,0.4]));
        Ok(())
    }

    fn scene(materials: serde_json::Value) -> Scene {
        vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],
            "images":{"a":{"file":"a.png"},"d":{"file":"d.png","color_space":"Non-Color"}},
            "materials":materials,
            "programs":{"x":{"source":""}},
            "animation":[{"tracks":[{"path":["materials","paint","inputs","base_color"],"keys":[[0,[0,1,0]]]}]}]}))
        .unwrap()
    }
    fn values(scene: &Scene, name: &str) -> Result<OpenPbrRecord> {
        let mut words = std::collections::BTreeMap::new();
        openpbr_values(
            scene,
            Some(name),
            &animated_inputs(scene)
                .unwrap()
                .remove(name)
                .unwrap_or_default(),
            |image| {
                let next = words.len() as u64 + 7;
                Ok(*words.entry(image.to_owned()).or_insert(next))
            },
        )
    }

    #[test]
    fn receiver_queries_require_possible_solid_transmission() {
        let material = |thin: bool, program: bool, animated: bool, mapped: bool, transmission: bool| {
            let mut pending = PendingMaterial::program(u32::from(program), false);
            let mut builder = RecordBuilder::openpbr();
            builder.set("geometry_thin_walled", &[f32::from(thin)]);
            pending.record.values = builder.record.values;
            pending.transmissive = transmission;
            if animated {pending.unique = Some("animated".into());}
            if mapped {
                let input = param("geometry_thin_walled").0;
                pending.record.inputs[input] |= 1;
            }
            pending.finish()
        };
        // Variant gates must agree with the planner, including tracks, programs
        // and mapped wall type. A false gate removes the entire transport path.
        for thin in [false, true] {
            for program in [false, true] {
                for animated in [false, true] {
                    for mapped in [false, true] {
                        for transmission in [false, true] {
                            let m = material(thin, program, animated, mapped, transmission);
                            assert_eq!(m.features & FEATURE_TRANSMISSION != 0, transmission);
                            assert_eq!(m.features & FEATURE_SOLID_TRANSMISSION != 0, m.solid_transmission);
                        }
                    }
                }
            }
        }
        assert!(material(false, false, false, false, true).solid_transmission);
        assert!(!material(true, false, false, false, true).solid_transmission);
        assert!(material(true, true, false, false, true).solid_transmission);
        assert!(material(true, false, true, false, true).solid_transmission);
        assert!(material(true, false, false, true, true).solid_transmission);
        assert!(!material(false, true, true, true, false).solid_transmission);
    }
    #[test]
    fn solid_shadow_absorption_keeps_unknowns_but_omits_clear_interfaces() {
        let material=|depth:f32,color:[f32;3],thin:bool,program:bool,animated:bool,mapped:Option<&str>| {
            let mut pending=PendingMaterial::program(u32::from(program),false);
            let mut builder=RecordBuilder::openpbr();builder.set("transmission_depth",&[depth]);builder.set("transmission_color",&color);builder.set("geometry_thin_walled",&[f32::from(thin)]);
            pending.record.values=builder.record.values;pending.transmissive=true;
            if animated {pending.unique=Some("animated".into());}
            if let Some(field)=mapped {pending.record.inputs[param(field).0]|=1;}
            pending.finish().solid_absorption
        };
        assert!(!material(0.,[0.5;3],false,false,false,None));
        assert!(!material(1.,[1.;3],false,false,false,None));
        assert!(material(1.,[0.5,1.,1.],false,false,false,None));
        assert!(!material(1.,[0.5;3],true,false,false,None));
        assert!(material(0.,[1.;3],false,true,false,None));
        assert!(material(0.,[1.;3],false,false,true,None));
        assert!(material(0.,[0.5;3],false,false,false,Some("transmission_depth")));
        assert!(material(1.,[1.;3],false,false,false,Some("transmission_color")));
    }

    #[test]
    fn null_connections_omit_only_proven_refracting_solids() {
        let material=|transmission:bool,thin:bool,ior:f32,weight:f32,dispersion:f32,program:bool,animated:bool,mapped:Option<&str>| {
            let mut pending=PendingMaterial::program(u32::from(program),false);
            let mut builder=RecordBuilder::openpbr();
            builder.set("geometry_thin_walled",&[f32::from(thin)]);builder.set("specular_ior",&[ior]);
            builder.set("specular_weight",&[weight]);builder.set("transmission_dispersion_scale",&[dispersion]);
            pending.record.values=builder.record.values;pending.transmissive=transmission;
            if animated {pending.unique=Some("animated".into());}
            if let Some(field)=mapped {pending.record.inputs[param(field).0]|=1;}
            let fact=pending.null_connection();
            assert_eq!(pending.finish().null_connection,fact);
            fact
        };
        // Without transmission nothing passes, whatever else is unknown.
        assert!(!material(false,false,1.,0.,1.,true,true,Some("specular_ior")));
        assert!(!material(false,true,1.5,1.,0.,false,false,None));
        // A constant refracting solid blocks; every unknown keeps the evaluation.
        assert!(!material(true,false,1.5,1.,0.,false,false,None));
        assert!(!material(true,false,0.75,1.,0.,false,false,None));
        assert!(material(true,false,1.,1.,0.,false,false,None));
        assert!(material(true,false,1.5,0.5,0.,false,false,None));
        assert!(material(true,false,1.5,1.,0.25,false,false,None));
        assert!(material(true,true,1.5,1.,0.,false,false,None));
        assert!(material(true,false,1.5,1.,0.,true,false,None));
        assert!(material(true,false,1.5,1.,0.,false,true,None));
        for field in ["geometry_thin_walled","specular_ior","specular_weight","transmission_dispersion_scale"] {
            assert!(material(true,false,1.5,1.,0.,false,false,Some(field)),"{field}");
        }
    }

    #[test]
    fn previs_scalar_alpha_changes_only_its_material_slot() -> Result<()> {
        let mut s=scene(serde_json::json!({
            "opaque":{"inputs":{}},
            "alpha":{"inputs":{"geometry_opacity":0.35}},
            "cutoff":{"inputs":{"geometry_opacity":0.35,"alpha_cutoff":0.5}},
            "continuous":{"opacity":"transmittance","inputs":{"geometry_opacity":0.35}},
            "glass":{"inputs":{"geometry_opacity":0.35,"transmission_weight":1.}}
        }));
        s.animation.clear();
        let tracks=Default::default();
        for (name, slot) in [("opaque",0.),("alpha",2.),("cutoff",3.),("continuous",1.)] {
            let v=openpbr_values(&s,Some(name),&tracks,|_|Ok(0))?;
            assert!(!v.previs_requires_path(&tracks),"{name}");
            let material=openpbr_pending(&s,Some(name),&tracks,v);
            assert_eq!(material.record.extra[1],slot,"{name}");
            assert_eq!(material.alpha_mask, name=="alpha" || name=="cutoff");
        }
        assert!(openpbr_values(&s,Some("glass"),&tracks,|_|Ok(0))?.previs_requires_path(&tracks));
        for input in ["geometry_opacity","alpha_cutoff"] {
            s.materials.get_mut("alpha").unwrap().inputs.clear();
            s.materials.get_mut("alpha").unwrap().inputs.insert(input.into(),
                serde_json::from_value(serde_json::json!({"image":"d"}))?);
            assert!(openpbr_values(&s,Some("alpha"),&tracks,|_|Ok(1))?.previs_requires_path(&tracks));
        }
        for input in ["geometry_opacity","alpha_cutoff"] {
            let tracks=std::collections::BTreeSet::from([input.to_owned()]);
            assert!(openpbr_values(&s,Some("opaque"),&tracks,|_|Ok(0))?.previs_requires_path(&tracks));
        }
        Ok(())
    }

    #[test]
    fn subsurface_accepts_dithered_and_blended_coverage() {
        let mut s=scene(serde_json::json!({
            "coverage":{"inputs":{"geometry_opacity":0.5,"subsurface_weight":0.5}},
            "transmittance":{"opacity":"transmittance","inputs":{"geometry_opacity":0.5,"subsurface_weight":0.5}}
        }));
        s.animation.clear();
        assert!(values(&s,"coverage").is_ok());
        assert!(values(&s,"transmittance").is_ok());
    }
    #[test]
    fn transmission_preserves_subsurface_substrate() {
        let mut s=scene(serde_json::json!({"mixed":{"inputs":{
            "transmission_weight":0.5,"subsurface_weight":0.5
        }}}));
        s.animation.clear();
        assert!(values(&s,"mixed").unwrap().capabilities.subsurface);
    }
    #[test]
    fn shading_features_preserve_metal_tracks_textures_and_programs() {
        let s = scene(serde_json::json!({
            "paint": {},
            "coat": {"inputs": {"coat_weight": 0.5}},
            "metal": {"inputs": {"base_metalness": 0.5}},
            "both": {"inputs": {"base_metalness": 1, "fuzz_weight": 0.5}},
            "texture": {"inputs": {"base_metalness": {"image": "d"}}}
        }));
        let finish = |v: OpenPbrRecord| {
            let mut pending = PendingMaterial::program(0, false);
            pending.record = v.builder.record;
            pending.shading = v.shading;
            pending.finish().features as u8 & (SHADING_LAYERS | SHADING_METAL)
        };
        for (name, expected) in [
            ("paint", 0),
            ("coat", SHADING_LAYERS),
            ("metal", SHADING_METAL),
            ("both", SHADING_LAYERS | SHADING_METAL),
            ("texture", SHADING_METAL),
        ] {
            assert_eq!(finish(values(&s, name).unwrap()), expected, "{name}");
        }
        // A zero initial weight can become metallic on a later frame/instance phase.
        let animated = ["base_metalness".to_owned()].into_iter().collect();
        let tracked = openpbr_values(&s, Some("paint"), &animated, |_| Ok(0)).unwrap();
        assert_eq!(finish(tracked), SHADING_METAL);
        assert_eq!(
            PendingMaterial::program(1, false).finish().features as u8
                & (SHADING_LAYERS | SHADING_METAL),
            SHADING_LAYERS | SHADING_METAL
        );
    }
    #[test]
    fn gltf_closure_state_is_independent_of_vertex_colour_multiplication() {
        for imported in [false,true] {
            for transmissive in [false,true] {
                for unlit in [false,true] {
                    let mut material=PendingMaterial::program(0,false);
                    material.transmissive=transmissive;
                    material.record.tag[2] |= if imported {constants::FLAG_VERTEX_COLOR} else {0}
                        | if unlit {vork_slang::registry::FLAG_GLTF_UNLIT} else {0};
                    let needed=imported && (transmissive || unlit);
                    assert_eq!(material.gltf_tint(),needed);
                    assert_eq!(material.clone().finish().features & FEATURE_GLTF_TINT != 0,needed);
                    // Even default-white glTF colour must retain its multiply:
                    // interpolated barycentric weights need not sum to exactly one.
                    assert_eq!(material.record.tag[2] & constants::FLAG_VERTEX_COLOR != 0,imported);
                    if imported && transmissive {
                        assert!(material.record.screen_exit_needs_color(true));
                    }
                }
            }
        }
    }

    #[test]
    fn specialization_keeps_dynamic_and_program_capabilities() {
        let s = scene(serde_json::json!({
            "paint": {},
            "coat": {"inputs": {"coat_weight": 0.5}},
            "normal": {"inputs": {"geometry_normal": {"image":"d"}}},
            "coat_normal": {"inputs": {"coat_weight":1,"geometry_coat_normal": {"image":"d"}}},
            "tinted": {"inputs": {"specular_color": [0.4, 0.6, 0.8]}}
        }));
        let finish = |name| {
            let v = values(&s, name).unwrap();
            let mut p = PendingMaterial::program(0, true);
            p.record = v.builder.record;
            p.shading = v.shading;
            p.finish().features
        };
        assert_eq!(finish("paint"), FEATURE_SPECIALIZED);
        assert_ne!(finish("coat") & FEATURE_COAT, 0);
        assert_eq!(finish("coat") & FEATURE_FUZZ, 0);
        assert_ne!(
            finish("normal") & (1 << (param("geometry_normal").0 + 2)),
            0
        );
        assert_ne!(finish("tinted") & FEATURE_TINT, 0);
        assert_ne!(finish("coat_normal") & texture_feature("geometry_coat_normal"), 0);
        assert_ne!(finish("coat_normal") & FEATURE_COAT, 0);
        let mut program = PendingMaterial::program(1, false);
        program.integrate_opacity(Some(vork_scene::Opacity::Coverage),false,false);
        let program = program.finish().features;
        let required =
            FEATURE_COAT | FEATURE_FUZZ | FEATURE_TINT | FEATURE_MASK | 3;
        assert_eq!(program & required, required);
        assert_eq!(program & FEATURE_LOBES, 0);
        assert_eq!(program & (FEATURE_ANISOTROPY | FEATURE_FILM), FEATURE_ANISOTROPY | FEATURE_FILM);
        let mut hair = PendingMaterial::program(1, false);
        hair.hair = true;
        assert_eq!(hair.finish().features & (FEATURE_ANISOTROPY | FEATURE_FILM), 0);
        let mut mixed = PendingMaterial::program(1, false);
        mixed.lobes = true;
        assert_ne!(mixed.finish().features & FEATURE_LOBES, 0);
        let mut tracked = PendingMaterial::program(0, true);
        tracked.shading = SHADING_LAYERS | SHADING_METAL;
        tracked.unique = Some("paint".into());
        assert_eq!(
            tracked.finish().features & required,
            required & !FEATURE_MASK
        );
    }

    #[test]
    fn scene_variant_budget_merges_only_drawn_materials_conservatively() {
        let mut materials: Vec<_> = (0..=constants::SCENE_SHADING_VARIANTS)
            .map(|i| {
                let mut m = PendingMaterial::program(0, true).finish();
                m.features |= 1 << (i + 2);
                m
            })
            .collect();
        let original: Vec<_> = materials.iter().map(|m| m.features).collect();
        let layout = crate::geometry::VertexLayout::for_vertices(&[Vertex::zeroed()], false);
        bound_features(
            &mut materials,
            (0..constants::SCENE_SHADING_VARIANTS).map(|i| (layout, i)),
        );
        assert_eq!(
            materials.iter().map(|m| m.features).collect::<Vec<_>>(),
            original,
            "unused material does not consume budget"
        );
        let len = materials.len();
        bound_features(&mut materials, (0..len).map(|i| (layout, i)));
        let union = original.iter().fold(0, |a, b| a | b);
        assert!(materials.iter().all(|m| m.features == union));
        for (m, before) in materials.iter().zip(original) {
            assert_eq!(m.features & before, before, "no capability lost");
        }
    }

    #[test]
    fn scene_variant_budget_counts_layout_multiplication() {
        let layout = crate::geometry::VertexLayout::for_vertices(&[Vertex::zeroed()], false);
        let mut other = layout;
        other.deform = !other.deform;
        let mut materials: Vec<_> = (0..constants::SCENE_SHADING_VARIANTS)
            .map(|i| {
                let mut m = PendingMaterial::program(0, true).finish();
                m.features |= 1 << (i + 2);
                m
            })
            .collect();
        let len = materials.len();
        bound_features(
            &mut materials,
            [layout, other]
                .into_iter()
                .flat_map(|l| (0..len).map(move |i| (l, i))),
        );
        assert!(
            materials
                .iter()
                .all(|m| m.features == materials[0].features)
        );
    }

    #[test]
    fn too_many_mandatory_layouts_use_generic_shading() {
        let base = crate::geometry::VertexLayout::for_vertices(&[Vertex::zeroed()], false);
        let uv_sets = [
            base.uv,
            [0, constants::VERTEX_DEFAULT_UV],
            [constants::VERTEX_DEFAULT_UV, 0],
            [0, 0],
            [0, 1],
        ];
        let layouts: Vec<_> = uv_sets
            .into_iter()
            .flat_map(|uv| {
                (0..6).map(move |flags| crate::geometry::VertexLayout {
                points:false,
                    deform: flags >= 4,
                    tangent: flags >= 4 || flags & 2 != 0,
                    color: flags & 1 != 0,
                    uv,
                    ..base
                })
            })
            .take(constants::SCENE_SHADING_VARIANTS + 1)
            .collect();
        let mut masked = PendingMaterial::program(0, false);
        masked.integrate_opacity(Some(vork_scene::Opacity::Coverage),false,false);
        let mut materials = [masked.finish()];
        bound_features(&mut materials, layouts.into_iter().map(|l| (l, 0)));
        assert_eq!(
            materials[0].features,
            FeatureMask::from(SHADING_LAYERS | SHADING_METAL),
            "generic shading retains both dynamic lobes and every default texture/light path"
        );
    }

    #[test]
    fn spatial_subsurface_reserves_animated_thin_walled_state() {
        let scene: vork_scene::v2::Scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,"nodes":[],
            "materials":{"leaf":{"inputs":{"subsurface_weight":1,"geometry_thin_walled":true}}}})).unwrap();
        for tracked in [false,true] {
            let dynamic=if tracked {["geometry_thin_walled".to_owned()].into_iter().collect()} else {Default::default()};
            let record=openpbr_values(&scene,Some("leaf"),&dynamic,|_|Ok(0)).unwrap();
            assert!(record.capabilities.subsurface);
            assert_eq!(record.spatial_subsurface(),tracked);
        }
    }

    #[test]
    fn subsurface_capability_reserves_animated_zero_weight() {
        let scene=scene(serde_json::json!({
            "plain":{}, "skin":{"inputs":{"subsurface_weight":0.5}}
        }));
        for (name, tracked, expected) in [("plain",false,false),("plain",true,true),("skin",false,true)] {
            let animated=if tracked {["subsurface_weight".to_owned()].into_iter().collect()} else {Default::default()};
            let values=openpbr_values(&scene,Some(name),&animated,|_|Ok(0)).unwrap();
            assert_eq!(values.capabilities.subsurface,expected);
            let material=openpbr_pending(&scene,Some(name),&animated,values).finish();
            assert_eq!(material.subsurface,expected,"{name}, tracked {tracked}");
            assert_eq!(material.features & (FEATURE_SUBSURFACE|FEATURE_HAIR),0,"scene unions are never raster keys");
            assert_eq!(installed_features(&[material]) & FEATURE_SUBSURFACE != 0,expected,"{name}, tracked {tracked}");
        }
    }

    #[test]
    fn final_scattering_retains_unknowns_and_unions_only_installed_materials() {
        let weights=["base_weight","specular_weight","coat_weight","fuzz_weight","transmission_weight","subsurface_weight"];
        let mut empty=PendingMaterial::program(0,false);
        for name in weights {empty.record.values[value_word(param(name).0)]=0.;}
        assert!(!empty.clone().finish().scattering);
        for name in weights {
            for value in [1.,-1.,1e-30] {
                let mut active=empty.clone();active.record.values[value_word(param(name).0)]=value;
                assert!(active.finish().scattering,"{name}={value}");
            }
            let mut bound=empty.clone();bound.record.inputs[param(name).0]=1;
            assert!(bound.finish().scattering,"bound {name}");
            let input=serde_json::json!({"base_weight":0,"specular_weight":0,"coat_weight":0,"fuzz_weight":0,"transmission_weight":0,"subsurface_weight":0});
            let scene=scene(serde_json::json!({"material":{"inputs":input}}));
            let animated=[name.to_owned()].into_iter().collect();
            let values=openpbr_values(&scene,Some("material"),&animated,|_|Ok(0)).unwrap();
            assert!(openpbr_pending(&scene,Some("material"),&animated,values).finish().scattering,"tracked {name}");
        }
        let mut program=PendingMaterial::program(1,false);
        assert!(program.clone().finish().scattering,"unproven program");
        program.response_program &= !FEATURE_SCATTERING;
        assert!(!program.clone().finish().scattering,"program record defaults are not its output");
        let mut materials=vec![empty.finish(),program.finish()];
        assert_eq!(installed_features(&materials)&FEATURE_SCATTERING,0);
        materials.push(PendingMaterial::program(2,false).finish());
        let union=installed_features(&materials);materials.reverse();
        assert_eq!(union,installed_features(&materials));
        assert_ne!(union&FEATURE_SCATTERING,0);
        assert!(materials.iter().all(|m|m.features&FEATURE_SCATTERING==0),"no raster variants");
    }

    #[test]
    fn installed_features_are_the_union_of_every_material() {
        // An unproven program keeps every lobe; its proofs remove them.
        let unproven=PendingMaterial::program(1,false);
        let mut proven=unproven.clone();proven.response_program=0;
        let mut fibre=proven.clone();fibre.hair=true;
        let mut coated=proven.clone();coated.response_program=FEATURE_COAT;
        assert!(unproven.clone().finish().subsurface);
        assert!(!proven.clone().finish().subsurface);
        let plain=installed_features(&[proven.clone().finish()]);
        assert_eq!(plain & (FEATURE_SUBSURFACE|FEATURE_HAIR|FEATURE_COAT),0);
        assert_eq!(plain & (FEATURE_SPECIALIZED|FEATURE_INSTALLED),FEATURE_SPECIALIZED|FEATURE_INSTALLED);
        // No material's own key claims to be the scene's.
        assert_eq!(proven.clone().finish().features & FEATURE_INSTALLED,0);
        let union=installed_features(&[proven.clone().finish(),fibre.finish(),coated.finish(),unproven.finish()]);
        assert_eq!(union & (FEATURE_SUBSURFACE|FEATURE_HAIR|FEATURE_COAT),FEATURE_SUBSURFACE|FEATURE_HAIR|FEATURE_COAT);
        // Order never changes the key: the same scene always gets the same kernel.
        let mut reversed=[proven.clone().finish(),PendingMaterial::program(1,false).finish()];
        let forward=installed_features(&reversed);reversed.reverse();
        assert_eq!(forward,installed_features(&reversed));
    }

    #[test]
    fn installed_coverage_keeps_bound_animated_and_programmed_opacity() -> Result<()> {
        use crate::forward_variant::FEATURE_SURFACE_COVERAGE;
        let mut s=scene(serde_json::json!({
            "plain":{},
            "glass":{"inputs":{"transmission_weight":1.}},
            "alpha":{"inputs":{"geometry_opacity":0.4}},
            "cutoff":{"inputs":{"alpha_cutoff":0.}},
            "image":{"inputs":{"geometry_opacity":{"image":"d"}}}
        }));
        s.animation.clear();
        let covered=|materials:&[Material]| installed_features(materials)&FEATURE_SURFACE_COVERAGE!=0;
        for (name,tracked,expected) in [
            ("plain",None,false),("glass",None,false),("alpha",None,true),
            ("cutoff",None,true),("image",None,true),
            ("plain",Some("geometry_opacity"),true),("plain",Some("alpha_cutoff"),true),
        ] {
            let animated=tracked.map(str::to_owned).into_iter().collect();
            let values=openpbr_values(&s,Some(name),&animated,|_|Ok(1))?;
            let material=openpbr_pending(&s,Some(name),&animated,values).finish();
            assert_eq!(material.features&FEATURE_SURFACE_COVERAGE,0,"not a per-material proof");
            assert_eq!(covered(&[material]),expected,"{name}, {tracked:?}");
        }
        // The existing installed program proof selects coverage integration.
        // A single possible partial material, even unused by a draw, keeps it.
        let mut program=PendingMaterial::program(1,false);
        assert!(!covered(&[program.clone().finish()]));
        program.integrate_opacity(None,false,true);
        let mut materials=[PendingMaterial::program(0,false).finish(),program.finish()];
        assert!(covered(&materials));
        materials.reverse();
        assert!(covered(&materials));
        Ok(())
    }

    #[test]
    fn surface_response_capabilities_follow_the_authored_inputs() {
        let mut program=PendingMaterial::program(1,false);
        assert_eq!(program.clone().finish().features & (FEATURE_ANISOTROPY|FEATURE_FILM),FEATURE_ANISOTROPY|FEATURE_FILM);
        program.response_program=0;
        assert_eq!(program.clone().finish().features & (FEATURE_COAT|FEATURE_FUZZ|FEATURE_TINT),0);
        for feature in [FEATURE_COAT,FEATURE_FUZZ,FEATURE_TINT] {
            program.response_program=feature;
            assert_eq!(program.clone().finish().features & (FEATURE_COAT|FEATURE_FUZZ|FEATURE_TINT),feature);
        }
        program.response_program=0;
        assert_eq!(program.clone().finish().features & (FEATURE_ANISOTROPY|FEATURE_FILM),0);
        program.response_program=FEATURE_FILM;
        assert_eq!(program.finish().features & (FEATURE_ANISOTROPY|FEATURE_FILM),FEATURE_FILM);
        let plain=PendingMaterial::program(0,false);
        assert_eq!(plain.clone().finish().features & (FEATURE_ANISOTROPY|FEATURE_FILM),0);
        let mut color_track=plain.clone();color_track.unique=Some("paint".into());
        assert_eq!(color_track.finish().features & (FEATURE_ANISOTROPY|FEATURE_FILM),0);
        let mut anisotropy_track=plain.clone();anisotropy_track.response_animated=FEATURE_ANISOTROPY;
        assert_eq!(anisotropy_track.finish().features & (FEATURE_ANISOTROPY|FEATURE_FILM),FEATURE_ANISOTROPY);
        let mut film_track=plain;film_track.response_animated=FEATURE_FILM;
        assert_eq!(film_track.finish().features & (FEATURE_ANISOTROPY|FEATURE_FILM),FEATURE_FILM);
        for (input,capability) in [("specular_roughness_anisotropy",FEATURE_ANISOTROPY),("coat_roughness_anisotropy",FEATURE_COAT_ANISOTROPY),("thin_film_weight",FEATURE_FILM)] {
            let mut pending=PendingMaterial::program(0,false);
            pending.record.values[value_word(param(input).0)]=0.5;
            assert_ne!(pending.finish().features&capability,0);
        }
    }

    #[test]
    fn coat_anisotropy_capability_is_independent_of_base() {
        let plain=PendingMaterial::program(0,false);
        let mut base=plain.clone();base.record.values[value_word(param("specular_roughness_anisotropy").0)]=0.7;
        assert_eq!(base.finish().features & FEATURE_COAT_ANISOTROPY,0);
        for source in 0..4 {
            let mut coat=plain.clone();
            match source {
                0 => coat.record.values[value_word(param("coat_roughness_anisotropy").0)]=0.7,
                1 => coat.response_animated=FEATURE_COAT_ANISOTROPY,
                2 => coat.response_program=FEATURE_COAT_ANISOTROPY,
                _ => coat.record.inputs[param("coat_roughness_anisotropy").0]=1,
            }
            let mask=FEATURE_ANISOTROPY|FEATURE_COAT_ANISOTROPY;
            assert_eq!(coat.finish().features & mask,mask,"source {source}");
        }
    }

    #[test]
    fn unit_ior_proof_reserves_nonunit_bound_animated_and_unknown_weights() {
        let plain=PendingMaterial::program(0,false);
        assert!(!plain.weighted_ior());
        for weight in [0.,0.25,0.999,2.] {
            let mut material=plain.clone();material.record.values[value_word(param("specular_weight").0)]=weight;
            assert!(material.weighted_ior());
        }
        let mut bound=plain.clone();bound.record.inputs[param("specular_weight").0]=1;
        assert!(bound.weighted_ior());
        let mut animated=plain.clone();animated.response_animated=FEATURE_WEIGHTED_IOR;
        assert!(animated.weighted_ior());
        let mut color_track=plain;color_track.unique=Some("paint".into());
        assert!(!color_track.weighted_ior());
        let mut program=PendingMaterial::program(1,false);
        assert!(program.weighted_ior());
        program.response_program=0;
        assert!(!program.weighted_ior());
    }

    #[test]
    fn coat_frame_parameters_use_spec_metadata_and_defaults() {
        let (_, anisotropy)=param("coat_roughness_anisotropy");
        assert_eq!(anisotropy.kind,OpenPbrKind::Float);
        assert_eq!(anisotropy.default,&[0.]);
        assert_eq!((anisotropy.min,anisotropy.max),(0.,1.));
        let (_,tangent)=param("geometry_coat_tangent");
        assert_eq!(tangent.kind,OpenPbrKind::Vector3);
        assert_eq!(tangent.default_text,"unperturbed tangent");
        for inputs in [serde_json::json!({"coat_roughness_anisotropy":0}),serde_json::json!({"coat_roughness_anisotropy":1,"geometry_coat_tangent":[0,1,0]})] {
            let s=scene(serde_json::json!({"paint":{"inputs":inputs}}));
            assert!(values(&s,"paint").is_ok());
        }
    }

    #[test]
    fn approximated_inputs_warn_once_per_material_and_keep_shaded_values() {
        for (inputs, animated, warn) in [
            (serde_json::json!({}), vec![], false),
            (serde_json::json!({"base_diffuse_roughness":0,"transmission_scatter":[0,0,0],"transmission_scatter_anisotropy":0}), vec![], false),
            (serde_json::json!({"base_diffuse_roughness":0.5,"transmission_scatter":[0.2,0.3,0.4],"transmission_scatter_anisotropy":-0.5}), vec![], true),
            (serde_json::json!({"base_diffuse_roughness":{"image":"d"}}), vec![], true),
            (serde_json::json!({}), vec!["transmission_scatter_anisotropy"], true),
        ] {
            let s = scene(serde_json::json!({"paint":{"inputs":inputs}}));
            let collector = crate::warnings::Collector::default();
            let _scope = collector.enter();
            let animated = animated.into_iter().map(str::to_string).collect();
            let install = || {
                let values = openpbr_values(&s, Some("paint"), &animated, |_| Ok(0))?;
                Ok::<_, anyhow::Error>(openpbr_pending(&s, Some("paint"), &animated, values))
            };
            let installed = install().unwrap();
            install().unwrap();
            let warnings = collector.snapshot();
            assert_eq!(warnings.len(), usize::from(warn));
            if warn {
                assert_eq!(warnings[0].code, "openpbr_approximation");
                assert_eq!(warnings[0].requested["material"], "paint");
                assert!(warnings[0].dropped.contains("Lambert"));
                assert!(warnings[0].dropped.contains("Beer-only"));
            }
            let default = RecordBuilder::openpbr();
            for p in OPENPBR.iter().filter(|p| p.closure) {
                assert_eq!(installed.record.get(p.name), default.get(p.name), "{}", p.name);
            }
        }
    }

    #[test]
    fn openpbr_records_reject_what_is_not_applied() {
        for (inputs, why) in [
            (
                serde_json::json!({"specular_roughness":"a"}),
                "sRGB data image",
            ),
            (
                serde_json::json!({"geometry_thin_walled":{"image":"d"}}),
                "boolean texture",
            ),
        ] {
            let s = scene(serde_json::json!({"paint":{"inputs":inputs}}));
            assert!(values(&s, "paint").is_err(), "accepted {why}");
        }
    }
    #[test]
    fn scalar_texture_diagnostic_prescribes_a_valid_colour_space() {
        let mut s = scene(serde_json::json!({"paint":{"inputs":{"specular_roughness":{"image":"a"}}}}));
        let error = values(&s, "paint").err().expect("sRGB scalar texture must fail");
        assert!(error.to_string().contains("color_space \"Non-Color\""), "{error}");
        let mut authored = serde_json::to_value(&s).unwrap();
        authored["images"]["a"]["color_space"] = "Non-Color".into();
        s = vork_scene::v2::from_test_value(authored).unwrap();
        values(&s, "paint").unwrap();
    }
    /// Only dithered coverage selects the depth and control entries that read
    /// SV_PrimitiveID; opaque, cutout and blended table materials never do.
    #[test]
    fn only_dithered_materials_select_hashed_coverage_entries() {
        use crate::{bundles::Kind, forward_variant::FEATURE_DITHERED, pipelines::Role};
        use vork_scene::Opacity::{Transmittance, Coverage};
        for (method, transmission, dithered) in [
            (None, false, false),
            (Some(Transmittance), false, false),
            (Some(Coverage), false, true),
            (Some(Coverage), true, true),
            (None, true, false),
        ] {
            let mut pending = PendingMaterial::program(0, false);
            if method.is_some() || transmission {
                pending.integrate_opacity(method, transmission, false);
            }
            let material = pending.finish();
            assert_eq!(material.dithered, dithered, "{method:?} {transmission}");
            assert_eq!(material.features & FEATURE_DITHERED, 0, "a pipeline-role bit, not a material capability");
            let flags = crate::plan::DrawFlags { hair: false, transparent: material.transparent, moving: false,
                casts_shadows: true, alpha_mask: material.alpha_mask };
            let mut controls = 0;
            for &kind in Role::kinds(&flags, true) {
                match Role::of(kind, &flags, &material) {
                    Role::Controls(m) | Role::TransparentControls(m) => {
                        controls += 1;
                        assert_eq!(m & FEATURE_DITHERED != 0, dithered, "{method:?} {transmission}");
                    }
                    Role::MaskedDepth { hashed } => assert_eq!(hashed, dithered),
                    _ => (),
                }
            }
            assert_eq!(controls, 1);
            if material.alpha_mask && !material.transparent {
                assert_eq!(Role::of(Kind::Depth, &flags, &material), Role::MaskedDepth { hashed: dithered });
            }
        }
    }

    #[test]
    fn dithered_programs_cast_coverage_shadows() {
        let mut pending=PendingMaterial::program(1,false);
        pending.integrate_opacity(Some(vork_scene::Opacity::Coverage),false,true);
        let material=pending.finish();
        assert!(material.alpha_mask,"camera dither still needs evaluated coverage");
        assert!(material.shadow_coverage);
        assert!(material.shadow_animated);

    }

    #[test]
    fn material_emission_receivers_follow_effective_weights() {
        for (inputs,tracks,expected) in [
            (serde_json::json!({}),vec![],true),
            (serde_json::json!({"base_weight":0,"specular_weight":0}),vec![],false),
            (serde_json::json!({"transmission_weight":1}),vec![],true),
            (serde_json::json!({"transmission_weight":1,"geometry_thin_walled":true,"specular_ior":1}),vec![],false),
            (serde_json::json!({"transmission_weight":1,"geometry_thin_walled":true,"specular_weight":0}),vec![],false),
            (serde_json::json!({"transmission_weight":1,"geometry_thin_walled":true,"specular_ior":1}),vec!["specular_weight"],false),
            (serde_json::json!({"transmission_weight":1,"geometry_thin_walled":true,"specular_ior":1}),vec!["specular_ior"],true),
            (serde_json::json!({"transmission_weight":1,"geometry_thin_walled":true,"specular_ior":1}),vec!["geometry_thin_walled"],true),
            (serde_json::json!({"base_metalness":1}),vec![],true),
            (serde_json::json!({"base_weight":0,"specular_weight":0,"fuzz_weight":1}),vec![],true),
            (serde_json::json!({"base_weight":0,"specular_weight":0}),vec!["fuzz_weight"],true),
            (serde_json::json!({"base_weight":0,"specular_weight":0}),vec!["base_weight"],true),
            (serde_json::json!({"subsurface_weight":1}),vec![],true),
        ] {
            let s=scene(serde_json::json!({"paint":{"inputs":inputs}}));
            let tracks=tracks.into_iter().map(str::to_owned).collect();
            let pending=openpbr_pending(&s,Some("paint"),&tracks,values(&s,"paint").unwrap());
            assert_eq!(pending.emission_receiver_mask()!=0,expected,"{}: {tracks:?}",s.materials["paint"].inputs.len());
        }
        let mut program=PendingMaterial::program(1,false);
        assert_ne!(program.emission_receiver_mask(),0,"unknown program scattering remains possible");
        program.response_program &= !FEATURE_SCATTERING;
        assert_eq!(program.emission_receiver_mask(),0,"proved unlit output needs no receiver");
    }

    /// Zero emission factors (an authored black colour or zero luminance, a texture
    /// times either) are not emitters; nonzero constants and emission tracks are.
    #[test]
    fn provably_dark_emission_is_not_an_emitter() {
        let pending = |record, emission_animated| {
            let mut p = PendingMaterial::program(0, false);
            p.record = record;
            p.emission_animated = emission_animated;
            p.finish()
        };
        let s: Scene = vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],
            "images":{"a":{"file":"a.png"}},
            "materials":{
                "plain":{},
                "black":{"inputs":{"emission_color":[0,0,0],"emission_luminance":1}},
                "off_map":{"inputs":{"emission_color":{"image":"a"},"emission_luminance":0}},
                "lit_map":{"inputs":{"emission_color":{"image":"a"},"emission_luminance":1}},
                "glow":{"inputs":{"emission_luminance":1}}}}))
        .unwrap();
        for (name, emits) in [("plain", false), ("black", false), ("off_map", false),
            ("lit_map", true), ("glow", true)] {
            let record = values(&s, name).unwrap().builder.record;
            let m = pending(record, false);
            assert_eq!(m.source_emission, emits, "{name}");
            let tracked = pending(record, true);
            assert!(tracked.source_emission, "{name} with an emission track");
        }
    }

    #[test]
    fn constant_source_proof_keeps_spatial_and_layered_emission() {
        let s=scene(serde_json::json!({"paint":{"inputs":{"emission_luminance":3}}}));
        let mut plain=PendingMaterial::program(0,false);
        plain.record=values(&s,"paint").unwrap().builder.record;
        assert!(plain.constant_source_emission());
        // Unrelated receiver response cannot invalidate a source-only proof.
        let mut metal=plain.clone();metal.record.values[value_word(param("base_metalness").0)]=1.;
        assert!(metal.constant_source_emission());
        for input in ["emission_color","emission_luminance","geometry_opacity","alpha_cutoff","coat_weight","fuzz_weight"] {
            let mut bound=plain.clone();bound.record.inputs[param(input).0]=1;
            assert!(!bound.constant_source_emission(),"bound {input}");
        }
        for (input,feature) in [("coat_weight",FEATURE_COAT),("fuzz_weight",FEATURE_FUZZ)] {
            let mut weighted=plain.clone();weighted.record.values[value_word(param(input).0)]=0.5;
            assert!(!weighted.constant_source_emission(),"nonzero {input}");
            let mut animated=plain.clone();animated.response_animated|=feature;
            assert!(!animated.constant_source_emission(),"initially zero tracked {input}");
        }
        for flag in [FLAG_VERTEX_ALPHA,vork_slang::registry::FLAG_GLTF_UNLIT] {
            let mut varying=plain.clone();varying.record.tag[2]|=flag;
            assert!(!varying.constant_source_emission());
        }
        let mut program=plain.clone();program.record.tag[0]=1;program.source_emission=true;
        assert!(!program.constant_source_emission());
        program.source_emission=false;
        assert!(program.constant_source_emission(),"proved dark programs cannot be sampled sources");
        let mut animated=plain;animated.emission_animated=true;animated.opacity_animated=true;
        animated.record.values[value_word(param("emission_luminance").0)]=0.;
        assert!(animated.constant_source_emission(),"constant record tracks are read at the current time");
        animated.response_animated|=FEATURE_COAT;
        assert!(!animated.constant_source_emission(),"initially dark tracked sources still reserve layers");
    }

    #[test]
    fn animated_records_never_merge_with_identical_static_ones() {
        let s = scene(serde_json::json!({"paint":{}}));
        let record = values(&s, "paint").unwrap().builder.record;
        let pending = |unique: Option<&str>| PendingMaterial {
            uv_sources: [0, 1],
            emission_animated: false,
            source_emission: false,
            record,
            taps: Vec::new(),
            alpha_mask: false,
            double_sided: false,
            preserve_solid_backfaces: true,
            transparent: false,
            transmissive: false,
            lobes: false,
            lit: false,
            hair: false,
            physical_hair: false,
            opacity_animated: false,
            shading: 0,
            response_animated: 0,
            receiver_animated: 0,
            response_program: 0,
            unique: unique.map(str::to_owned),
        };
        assert_ne!(pending(None).key(), pending(Some("paint")).key());
        let native = pending(None);
        let mut imported = native.clone();
        imported.preserve_solid_backfaces = false;
        assert_ne!(native.key(), imported.key());
    }
    #[test]
    fn material_tracks_read_the_installed_record() {
        let s =
            scene(serde_json::json!({"paint":{"inputs":{"base_color":[1.,1.,1.]}},"p":{"program":"x"}}));
        let mut records = vec![MaterialRecord::zeroed(); 8];
        records[3] = values(&s, "paint").unwrap().builder.record;
        records[7] = records[3];
        let bytes = bytemuck::cast_slice::<_, u8>(&records);
        let base = OPENPBR.iter().position(|p| p.name == "base_color").unwrap();
        let targets = track_targets(("paint", "base_color"), &[3, 7], bytes).unwrap();
        assert_eq!(
            targets,
            vec![(3, base, vec![1., 1., 1.]), (7, base, vec![1., 1., 1.])]
        );
        for name in ["base_diffuse_roughness", "transmission_scatter", "transmission_scatter_anisotropy"] {
            assert!(track_targets(("paint", name), &[3, 7], bytes).unwrap().is_empty());
        }
        let roughness = track_targets(("paint", "specular_roughness"), &[3], bytes).unwrap();
        assert_eq!(roughness[0].2, vec![0.3]);
        assert!(track_targets(("paint", "thin_film_weight"), &[3], bytes).is_ok());
        assert!(track_targets(("p", "anything"), &[3], bytes).is_err());
        assert!(track_targets(("paint", "base_color"), &[9], bytes).is_err());
        assert_eq!(
            animated_inputs(&s).unwrap()["paint"]
                .iter()
                .collect::<Vec<_>>(),
            ["base_color"]
        );
    }
