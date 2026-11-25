    use super::*;
    #[test]
    #[ignore="requires real GPU"]
    fn gpu_environment_moments_and_cdf_match_numerical_references()->Result<()> {
        let r=Renderer::new()?;
        let varying=image::Rgb32FImage::from_fn(17,9,|x,y|image::Rgb([(x+1) as f32*10000.,(y+1) as f32*5000.,(x*y+1) as f32]));
        let black=image::Rgb32FImage::new(17,9);
        let hdr=image::Rgb32FImage::from_fn(8,4,|x,y|image::Rgb([(0.5+((y*8+x)%3) as f32*0.5)*1e38,1e38,2e38]));
        let mut seam=image::Rgb32FImage::new(8,4);seam.put_pixel(0,2,image::Rgb([2e6,0.,0.]));
        for image in [varying,black,hdr,seam] {
            let (width,height)=image.dimensions();
            let data=source_rgba(&image,&image);
            let source=environment_program::source_texture(&r,[width,height]);
            r.queue.write_texture(source.as_image_copy(),bytemuck::cast_slice(&data),wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(width*16),rows_per_image:Some(height)},source.size());
            let evaluated=evaluated::Evaluation::new(&r,&source);let (actual,scale)=evaluated.update(&r)?;
            let first=image.pixels().next().unwrap().0.map(f32::to_bits);
            let uniform=image.pixels().all(|pixel|pixel.0.map(f32::to_bits)==first);
            assert_eq!(evaluated.uniform(),uniform.then_some(first));
            assert!((scale-radiance_scale(&image)).abs()<1e-5+1e-6*radiance_scale(&image));
            let reference=analyze(&image);
            // Signed moments can cancel to zero. Float32 reduction/trigonometric
            // roundoff scales with integrated channel energy, not that residual.
            for (a,b) in actual.into_iter().zip(reference) {for c in 0..3 {
                if reference[0][c]==0. {assert_eq!(a[c],0.,"empty channel acquired energy");}
                else {assert!((a[c]-b[c]).abs()<0.02+2e-6*reference[0][c].abs().max(b[c].abs()),"{a:?} != {b:?}");}
            }}
            let expected=distribution(width,height,&data[data.len()/2..]);
            let size=expected.len() as u64*8;
            let read=crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {label:Some("CDF oracle"),size,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
            let mut encoder=r.device.create_command_encoder(&Default::default());encoder.copy_buffer_to_buffer(evaluated.distribution.as_ref().unwrap(),0,&read,0,size);r.queue.submit([encoder.finish()]);
            let(tx,rx)=std::sync::mpsc::channel();read.map_async(wgpu::MapMode::Read,..,move |v|{let _=tx.send(v);});r.wait(None)?;rx.recv()??;
            let mapped=read.get_mapped_range(..);let values:&[[f32;2]]=bytemuck::cast_slice(&mapped);
            for (a,b) in values.iter().zip(expected) {for c in 0..2 {assert!((a[c]-b[c]).abs()<2e-6,"CDF {a:?} != {b:?}");}}
        }
        Ok(())
    }

    #[test]
    #[ignore="requires real GPU"]
    fn warm_raster_cache_skips_proposal_and_matches_path_energy()->Result<()> {
        let r=Renderer::new()?;
        let fixtures=Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let mut input:serde_json::Value=vork_scene::v2::with_test_settings(include_str!("../tests/fixtures/analytic.json"),include_str!("../tests/presets/analytic-7fefd52a.json"))?;
        input["animation"]=serde_json::json!([]);
        input["render"]=serde_json::json!({"resolution":[8,8],"settings":{"integrator":"raster","samples":64,"denoiser":null}});
        input["programs"]=serde_json::json!({"world":{"source":"struct Inputs {}; float3 environment(EnvironmentContext c,Inputs i){return float3(1.);}"}});
        input["materials"]=serde_json::json!({"world":{"program":"world"},"white":{"inputs":{"base_color":[1,1,1],"specular_weight":0}}});
        input["nodes"]=serde_json::json!([
            {"name":"camera","type":"camera","translation":[0,0,5],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":0.1}}},
            {"type":"mesh","geometry":"shape_box","materials":["white"],"scale":[20,20,0.25]},
            {"type":"light","shape":"environment","material":"world","energy":1,"color":[1,1,1]}]);
        let render=|scene:&InstalledScene|->Result<Vec<[f32;4]>> {
            let mut frame=scene.create_frame_slot(&r,ReadbackPrecision::Float)?;
            frame.submit_outputs(scene,&r,0.,ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()})?;
            Ok(frame.collect(&r)?.linear_rgba.context("linear pixels")?)
        };
        // Seed the normal cache and wait only in the test for optional persistence.
        let cold=r.install(vork_scene::v2::from_test_value(input.clone())?,&fixtures)?;
        let original=render(&cold)?;
        r.wait(None)?;
        disk_cache::wait_persistence()?;
        let (warm,allocations)=crate::gpu_memory::test_buffer_allocations(||r.install(vork_scene::v2::from_test_value(input.clone())?,&fixtures));
        let warm=warm?;
        assert!(!allocations.iter().any(|label|label=="uniform environment proposal"),"complete warm install allocated a dummy: {allocations:?}");
        assert_eq!(warm.environment.distribution_size,[1,1],"warm raster allocated a proposal");
        assert!(warm.environment.distribution.is_none());
        let cached=render(&warm)?;
        for (a,b) in cached.iter().zip(&original) {for c in 0..3 {assert!((a[c]-b[c]).abs()<1e-5,"cached raster changed energy: {a:?} {b:?}");}}
        // Exercise the validation-only GPU path and its actual allocation/dispatch
        // plan independently of the cached metadata used for lighting.
        let source=environment_program::install_source(&r,&warm.scene,"world",&fixtures)?.0;
        let validation=evaluated::Evaluation::validation(&r,&source.texture);
        assert_eq!(validation.work(),(false,0,2),"warm validation must allocate no proposal/CDF and dispatch only peaks and summary");
        assert_eq!(validation.update(&r)?.1,warm.environment.initial_filter_scale());
        for invalid in [-1.,f32::NAN] {
            r.queue.write_texture(source.texture.as_image_copy(),bytemuck::cast_slice(&[invalid,0.,0.,1.]),wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(16),rows_per_image:Some(1)},extent(1,1));
            assert!(validation.update(&r).is_err(),"warm source validation accepted {invalid}");
        }
        input["render"]["settings"]["integrator"]=serde_json::json!("path");
        input["render"]["settings"]["samples"]=serde_json::json!(256);
        let path=r.install(vork_scene::v2::from_test_value(input)?,&fixtures)?;
        assert_ne!(path.environment.distribution_size,[1,1],"warm path lost its proposal");
        let reference=render(&path)?;
        for c in 0..3 {
            let mean=|pixels:&Vec<[f32;4]>|pixels.iter().map(|p|p[c] as f64).sum::<f64>()/pixels.len() as f64;
            let raster=mean(&cached);let path=mean(&reference);
            eprintln!("cached raster / std/path energy channel {c}: {raster} / {path}");
            assert!((path-1.).abs()<0.03 && (raster-path).abs()<0.03,"raster {raster}, path {path}");
        }
        r.errors.check()?;Ok(())
    }

    #[test]
    #[ignore="requires real GPU"]
    fn held_environment_inputs_skip_production_evaluation()->Result<()> {
        let r=Renderer::new()?;
        let scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "programs":{"world":{"source":"struct Inputs {float gain=1.;}; float3 environment(EnvironmentContext c,Inputs i){return float3(i.gain);}"}},
            "materials":{"world":{"program":"world"}},
            "nodes":[{"type":"light","shape":"environment","material":"world"}],
            "animation":[{"tracks":[{"path":["materials","world","inputs","gain"],"interpolation":"STEP","keys":[[0,1],[1,2]]}]}]}))?;
        let environment=Environment::new(&r,environment_node::Settings::from_scene(&scene)?,&scene,Path::new("."))?;
        let mut changes=values::LightingTracker::default();
        environment.update(&r,0.,0.,&mut changes)?;
        let program=&environment.animated.as_ref().context("animated environment")?.program;
        // Perturb the GPU source without touching the production change tracker.
        // A held update must leave it alone, proving evaluation itself was skipped.
        let source=program.evaluate_source(&r,1.,1.)?;
        environment.update(&r,0.9,0.9,&mut changes)?;
        assert!(source.read(&r)?.image.pixels().all(|p|p.0==[2.;3]));
        program.evaluate_source(&r,0.,0.)?;
        environment.update(&r,1.,1.,&mut changes)?;
        assert!(source.read(&r)?.image.pixels().all(|p|p.0==[2.;3]));
        r.errors.check()?;Ok(())
    }

    #[test]
    #[ignore="requires real GPU"]
    fn animated_metadata_matches_path_energy_across_queued_slots_and_rewinds()->Result<()> {
        let mut r=Renderer::new()?;
        // Match the existing numerical transport oracle: production path firefly
        // clamps deliberately bias HDR energy, so disable them for this check.
        r.reference_transport_bounces=Some(64);
        let fixtures=Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let base:serde_json::Value=vork_scene::v2::with_test_settings(include_str!("../tests/fixtures/analytic.json"),include_str!("../tests/presets/analytic-7fefd52a.json"))?;
        let mut reference=None;
        for path in [true,false] {for scenario in ["plain","capture"] {
            let capture=scenario=="capture";
            let high=40000.;
            let mut input=base.clone();input["animation"]=serde_json::json!([]);
            input["render"]=serde_json::json!({"camera":"camera","resolution":[8,8],"background":"environment","settings":{"integrator":if path {"path"} else {"raster"},"denoiser":null,"samples":128}});
            if !path {input["render"]["settings"]["fast_gi"]=serde_json::Value::Null;input["render"]["settings"]["reflections"]=serde_json::json!("environment");}
            input["programs"]=serde_json::json!({"world":{"source":format!("struct Inputs {{}}; float3 environment(EnvironmentContext c,Inputs i){{return float3(c.time<.5?1.:c.time<1.5?{high:.1}:2.);}}")}});
            input["materials"]=serde_json::json!({"world":{"program":"world"},"white":{"inputs":{"base_color":[1,1,1],"specular_weight":0}}});
            input["nodes"]=serde_json::json!([
                {"type":"camera","name":"camera","translation":[0,0,5],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":0.1}}},
                {"type":"mesh","geometry":"shape_box","materials":["white"],"scale":[20,20,0.25]},
                {"type":"light","shape":"environment","material":"world","energy":2,"color":[0.5,0.375,0.25]}]);
            if capture {
                input["images"]=serde_json::json!({"capture":{"render":{"camera":"camera","resolution":[8,8],"background":"environment"}}});
                input["programs"]["copy"]=serde_json::json!({"source":"struct Inputs {Image source;}; float4 composite(CompositeContext c,Inputs i){return compositor_load(i.source,int2(c.uv*c.size));}"});
                input["materials"]["copy"]=serde_json::json!({"program":"copy","inputs":{"source":{"image":"capture"}}});
                input["composition"]["space"] = "3d".into(); input["composition"]["compositor"] = serde_json::json!(["copy"]);
            }
            // Cross the filter scale threshold within the existing sample
            // output format. Medium mean consumption has its own oracle below.
            let installed=r.install(vork_scene::v2::from_test_value(input)?,&fixtures)?;
            let mut slots=[installed.create_frame_slot(&r,ReadbackPrecision::Float)?,installed.create_frame_slot(&r,ReadbackPrecision::Float)?];
            let outputs=ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()};
            slots[0].submit_outputs(&installed,&r,0.,outputs)?;
            slots[1].submit_outputs(&installed,&r,1.,outputs)?;
            for (index,time,gain) in [(0,0.,1.),(1,1.,high),(0,2.,2.),(0,0.,1.)] {
                let pixels=slots[index].collect(&r)?.linear_rgba.unwrap();
                let mean:[f64;3]=std::array::from_fn(|c|pixels.iter().map(|p|p[c] as f64/gain).sum::<f64>()/pixels.len() as f64);
                if reference.is_none() {for c in 0..3 {assert!((mean[c]/[1.,0.75,0.5][c]-1.).abs()<0.03,"std/path white energy {mean:?}");} reference=Some(mean);}
                let expected=reference.unwrap();
                for c in 0..3 {assert!((mean[c]/expected[c]-1.).abs()<0.03,"path={path} scenario={scenario} time={time}: {mean:?} vs std/path {expected:?}");}
                eprintln!("animated energy path={path} scenario={scenario} time={time}: {mean:?}");
                if index==1 {slots[0].submit_outputs(&installed,&r,2.,outputs)?;}
                else if time==2. {slots[0].submit_outputs(&installed,&r,0.,outputs)?;}
            }
        }}
        r.errors.check()?;Ok(())
    }



    #[test]
    #[ignore="requires real GPU"]
    fn animated_environment_rejects_earlier_invalid_shutter_samples_and_recovers()->Result<()> {
        let r=Renderer::new()?;
        let last=1.+crate::sampling::shutter_time(7,8)-0.5;
        for invalid in ["-1.","asfloat(0x7fc00000u)"] {
            let source=format!("struct Inputs {{}}; float3 environment(EnvironmentContext c,Inputs i){{return float3(c.time>=.5 && c.time<=1.5 && abs(c.time-{last:.9})>.001 ? {invalid} : 1.);}}");
            let input=serde_json::json!({"version":2,
                "render":{"resolution":[8,8],"fps":1,"background":"environment","motion_blur":{"shutter":1},"settings":{"samples":8,"denoiser":null,"fast_gi":null,"reflections":"environment"}},
                "programs":{"world":{"source":source}},"materials":{"world":{"program":"world"}},
                "nodes":[{"type":"camera"},{"type":"light","shape":"environment","material":"world"}]});
            let installed=r.install(vork_scene::v2::from_test_value(input.clone())?,Path::new("."))?;
            assert_eq!(installed.plan.exposure_samples,8);
            let mut slot=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
            let outputs=ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()};
            slot.submit_outputs(&installed,&r,1.,outputs)?;
            let error=slot.collect(&r).err().context("invalid earlier shutter sample was accepted")?;
            assert!(error.to_string().contains("radiance must be finite and nonnegative"),"{error:#}");
            assert!(installed.environment.sampled_mean(&r)?.iter().all(|v|v.is_finite() && *v>0.),"last sample must be valid so diagnostics prove accumulation");
            // A failed publication invalidates the installation, including its
            // other slots. Reinstall before checking a later valid exposure.
            let installed=r.install(vork_scene::v2::from_test_value(input)?,Path::new("."))?;
            let mut slot=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
            slot.submit_outputs(&installed,&r,3.,outputs)?;
            assert!(slot.collect(&r)?.linear_rgba.unwrap().iter().flatten().all(|v|v.is_finite()));
        }
        r.errors.check()?;Ok(())
    }

    #[test]
    fn source_storage_cannot_use_a_filtering_sampler() {
        let entries=layout_entries();
        assert!(!entries.iter().any(|e| matches!(e.ty,wgpu::BindingType::Sampler(_))));
        // Native SPIR-V bypasses wgpu's dense Vulkan binding remap.
        for (index,entry) in entries.iter().enumerate() {assert_eq!(entry.binding,index as u32);}
        for entry in [entries[0],REDUCE_LAYOUT[0]] {
            assert!(matches!(entry.ty,wgpu::BindingType::Texture {
                sample_type:wgpu::TextureSampleType::Uint,..
            }));
        }
        assert_eq!(SOURCE_FORMAT,wgpu::TextureFormat::Rgba32Uint);
    }

    #[test]
    fn source_layers_keep_independent_full_precision_radiance() {
        let full=image::Rgb32FImage::from_raw(2,1,vec![2e6,1e6,5e5,12.,16.,20.]).unwrap();
        let residual=image::Rgb32FImage::from_raw(2,1,vec![20.,10.,5.,12.,16.,20.]).unwrap();
        let packed=source_rgba(&full,&residual);
        assert_eq!(packed,vec![2e6,1e6,5e5,1.,12.,16.,20.,1.,20.,10.,5.,1.,12.,16.,20.,1.]);
    }

    #[test]
    fn proposal_normalization_preserves_float32_hdr_range() {
        let original:Vec<f32>=(0..8*4).flat_map(|i|[0.5+(i%3) as f32*0.5,1.,2.,1.]).collect();
        let scaled:Vec<f32>=original.chunks_exact(4).flat_map(|p|[p[0]*1e38,p[1]*1e38,p[2]*1e38,p[3]]).collect();
        let a=distribution(8,4,&original);let b=distribution(8,4,&scaled);
        for (a,b) in a.iter().zip(b) {for k in 0..2 {assert!(b[k].is_finite() && (a[k]-b[k]).abs()<1e-7);}}
    }

    #[test]
    fn proposal_support_covers_bilinear_neighbors_and_longitude_seam() {
        let mut image=vec![0f32;8*4*4];image[(2*8)*4]=2e6;image[(2*8)*4+3]=1.;
        let cdf=distribution(8,4,&image);
        for y in 1..=3 {for x in [7,0,1] {
            let at=y*8+x;let mass=cdf[at][0]-if at>0 {cdf[at-1][0]} else {0.};
            assert!(mass>0.,"cell {x},{y}");
        }}
    }
    #[test]
    fn filter_parameters_preserve_hdr_scale_and_abi() {
        for scale in [1., 2., 1000., f32::MAX] {
            let p = filter_parameters(3, 0, scale, [glam::DVec3::splat(2.); 4], [64, 32]);
            assert_eq!(std::mem::size_of_val(&p), 96);
            assert!(p[0][2].is_finite() && p[0][2] > 0.);
            assert!((p[0][2] as f64 * scale as f64 - 1.).abs() < 1e-6);
            assert_eq!(p[5], [64., 32., 0., 0.]);
            assert_eq!(p[1][0], (2. / scale as f64) as f32);
        }
        for scale in [0., -1., f32::NAN, f32::INFINITY] {
            assert!(std::panic::catch_unwind(|| filter_parameters(0, 0, scale, [glam::DVec3::ZERO; 4], [1, 1])).is_err());
        }
    }

    // Instrument the undefined intrinsic domain so Vulkan's benign atan2(0,0)
    // cannot conceal the Metal failure. Exercise the production mapping itself.
    #[test]
    #[ignore = "requires GPU and pinned Slang"]
    fn equirect_poles_never_evaluate_undefined_longitude() -> Result<()> {
        let r=Renderer::new()?;let d=&r.device;
        let prelude=include_str!("../shaders/vork_prelude.slang");
        let mapping=prelude.split("float2 equirect_uv(").nth(1).unwrap().split("float2 environment_uv(").next().unwrap();
        let mapping=format!("float2 checked_equirect_uv({mapping}").replace("atan2(","checked_atan2(");
        let shader=format!(r#"
import vork_prelude;
import vork_constants;
[[vk::binding(0,0)]] RWStructuredBuffer<float4> result;
float checked_atan2(float y,float x) {{
    if (x==0. && y==0.) {{result[0]=float4(0.);return asfloat(0x7fc00000u);}}
    return atan2(y,x);
}}
{mapping}
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {{
    result[0]=float4(1.);
    for (uint i=0u;i<8u;i++) {{
        let x=asfloat((i&1u)*0x80000000u);let z=asfloat((i&2u)*0x40000000u);
        let direction=float3(x,i<4u ? 1. : -1.,z);
        result[1u+i]=float4(checked_equirect_uv(direction),equirect_uv(direction));
    }}
    result[9]=float4(checked_equirect_uv(float3(1.,0.,0.)),checked_equirect_uv(float3(0.,0.,1.)));
}}
"#);
        let entries=[crate::pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false})];
        let pipeline=crate::medium::test_pipeline(&r,&shader,&crate::medium::Native::default(),&entries,(1,1,1))?;
        let output=d.create_buffer(&wgpu::BufferDescriptor {label:None,size:160,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
        let read=d.create_buffer(&wgpu::BufferDescriptor {label:None,size:160,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,output.as_entire_binding())]});
        let mut encoder=d.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(1,1,1);}
        encoder.copy_buffer_to_buffer(&output,0,&read,0,160);r.queue.submit([encoder.finish()]);
        let(tx,rx)=std::sync::mpsc::channel();read.map_async(wgpu::MapMode::Read,..,move |v|{let _=tx.send(v);});r.wait(None)?;rx.recv()??;
        let mapped=read.get_mapped_range(..);let results:&[[f32;4]]=bytemuck::cast_slice(&mapped);
        assert_eq!(results[0],[1.;4],"equirectangular mapping evaluated atan2(0,0)");
        for i in 0..8 {let v=if i<4 {0.} else {1.};assert_eq!(results[i+1],[0.5,v,0.5,v]);}
        assert_eq!(results[9],[0.5,0.5,0.75,0.5]);
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU and pinned Slang"]
    fn filtered_environment_writes_sh_without_source_sampling() -> Result<()> {
        let r=Renderer::new()?;let d=&r.device;
        let source=d.create_texture(&wgpu::TextureDescriptor {label:None,
            size:wgpu::Extent3d {width:1,height:1,depth_or_array_layers:2},
            mip_level_count:1,sample_count:1,dimension:wgpu::TextureDimension::D2,
            format:SOURCE_FORMAT,usage:wgpu::TextureUsages::TEXTURE_BINDING,view_formats:&[]});
        let output=d.create_texture(&wgpu::TextureDescriptor {label:None,
            size:wgpu::Extent3d {width:1,height:1,depth_or_array_layers:1},
            mip_level_count:1,sample_count:1,dimension:wgpu::TextureDimension::D2,
            format:wgpu::TextureFormat::Rgba16Float,
            usage:wgpu::TextureUsages::STORAGE_BINDING|wgpu::TextureUsages::COPY_SRC,view_formats:&[]});
        let parameters=[[0f32,1.,1.,0.],[0.5,0.25,0.125,0.],[0.;4],[0.;4],[0.;4],[1.,1.,0.,0.]];
        let params=init(d,"SH fixture",bytemuck::cast_slice(&parameters),wgpu::BufferUsages::UNIFORM);
        let cdf=init(d,"SH unused CDF",bytemuck::cast_slice(&[1f32,1.]),wgpu::BufferUsages::STORAGE);
        let source_view=source.create_view(&Default::default());let view=output.create_view(&Default::default());
        let pipeline=Environment::pipeline(&r);
        let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[
            entry(0,wgpu::BindingResource::TextureView(&source_view)),entry(1,wgpu::BindingResource::TextureView(&view)),entry(2,params.as_entire_binding()),entry(3,cdf.as_entire_binding())]});
        let read=d.create_buffer(&wgpu::BufferDescriptor {label:None,size:256,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        let mut encoder=d.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(1,1,1);}
        encoder.copy_texture_to_buffer(output.as_image_copy(),wgpu::TexelCopyBufferInfo {buffer:&read,layout:wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(256),rows_per_image:Some(1)}},output.size());
        r.queue.submit([encoder.finish()]);
        let(tx,rx)=std::sync::mpsc::channel();read.map_async(wgpu::MapMode::Read,..,move |v|{let _=tx.send(v);});r.wait(None)?;rx.recv()??;
        let mapped=read.get_mapped_range(..);
        let actual:Vec<f32>=mapped[..8].chunks_exact(2).map(|v|half::f16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f32()).collect();
        assert_eq!(actual,[0.5,0.25,0.125,1.]);
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU and pinned Slang"]
    fn filtered_environment_preserves_constant_hdr_in_every_mode() -> Result<()> {
        let r=Renderer::new()?;let d=&r.device;
        let source=d.create_texture(&wgpu::TextureDescriptor {label:None,
            size:wgpu::Extent3d {width:1,height:1,depth_or_array_layers:2},
            mip_level_count:1,sample_count:1,dimension:wgpu::TextureDimension::D2,
            format:SOURCE_FORMAT,usage:wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::COPY_DST,view_formats:&[]});
        let output=d.create_texture(&wgpu::TextureDescriptor {label:None,
            size:wgpu::Extent3d {width:1,height:2,depth_or_array_layers:1},
            mip_level_count:1,sample_count:1,dimension:wgpu::TextureDimension::D2,
            format:wgpu::TextureFormat::Rgba16Float,
            usage:wgpu::TextureUsages::STORAGE_BINDING|wgpu::TextureUsages::COPY_SRC,view_formats:&[]});
        r.queue.write_texture(source.as_image_copy(), bytemuck::cast_slice(&[4e6f32,2e6,1e6,1.,2e6,1e6,5e5,1.]), wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(16),rows_per_image:Some(1)}, source.size());
        for (mode,mip) in [(1,0),(0,0),(0,1),(0,3),(0,5)] {
        let parameters=filter_parameters(mip,mode,4e6,[glam::DVec3::new(2e6,1e6,5e5),glam::DVec3::ZERO,glam::DVec3::ZERO,glam::DVec3::ZERO],[1,1]);
        let params=init(d,"SH fixture",bytemuck::cast_slice(&parameters),wgpu::BufferUsages::UNIFORM);
        let cdf=init(d,"SH unused CDF",bytemuck::cast_slice(&[1f32,1.]),wgpu::BufferUsages::STORAGE);
        let source_view=source.create_view(&Default::default());let view=output.create_view(&Default::default());
        let pipeline=Environment::pipeline(&r);
        let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[
            entry(0,wgpu::BindingResource::TextureView(&source_view)),entry(1,wgpu::BindingResource::TextureView(&view)),entry(2,params.as_entire_binding()),entry(3,cdf.as_entire_binding())]});
        let read=d.create_buffer(&wgpu::BufferDescriptor {label:None,size:512,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        let mut encoder=d.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(1,1,1);}
        encoder.copy_texture_to_buffer(output.as_image_copy(),wgpu::TexelCopyBufferInfo {buffer:&read,layout:wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(256),rows_per_image:Some(2)}},output.size());
        r.queue.submit([encoder.finish()]);
        let(tx,rx)=std::sync::mpsc::channel();read.map_async(wgpu::MapMode::Read,..,move |v|{let _=tx.send(v);});r.wait(None)?;rx.recv()??;
        let mapped=read.get_mapped_range(..);
        for row in 0..2 {
            let actual:Vec<f32>=mapped[row*256..row*256+8].chunks_exact(2).map(|v|half::f16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f32()).collect();
            assert_eq!(actual,[0.5,0.25,0.125,1.],"mode {mode}, mip {mip}, row {row}");
        }
        drop(mapped);
        read.unmap();
        }
        Ok(())
    }

    /// Integer source bindings make optional float32 filtering impossible. Check
    /// decoding, interpolation, addressing and residual multiplication together.
    #[test]
    #[ignore = "requires GPU and pinned Slang"]
    fn source_bits_interpolate_hdr_without_float_filtering() -> Result<()> {
        let r=Renderer::new()?;
        let d=&r.device;
        let pixels=[[2e6f32,1e6,5e5,0.25],[4e6,2e6,1e6,0.75],
            [8.,16.,32.,1.],[64.,128.,256.,0.]];
        let image=d.create_texture(&wgpu::TextureDescriptor {
            label:Some("HDR source bits regression"),
            size:wgpu::Extent3d {width:2,height:2,depth_or_array_layers:2},
            mip_level_count:1,sample_count:1,dimension:wgpu::TextureDimension::D2,
            format:SOURCE_FORMAT,
            usage:wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::COPY_DST,
            view_formats:&[],
        });
        let pixels= pixels.into_iter().chain(pixels.map(|p|[p[0]*p[3],p[1]*p[3],p[2]*p[3],1.])).collect::<Vec<_>>();
        r.queue.write_texture(image.as_image_copy(),bytemuck::cast_slice(&pixels),
            wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(32),rows_per_image:Some(2)},image.size());
        let shader=r#"
import vork_prelude;
[[vk::binding(0,0)]] Texture2DArray<uint4> source;
[[vk::binding(1,0)]] RWStructuredBuffer<float4> result;
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {
    result[0]=float4(environment_radiance(source,float2(.5,.5)),1.);
    result[1]=float4(environment_radiance(source,float2(.5,.5),0u,true),1.);
    result[2]=float4(environment_radiance(source,float2(0.,.25)),1.);
    result[3]=float4(environment_radiance(source,float2(0.,.25),1u),1.);
    result[4]=float4(environment_radiance(source,float2(0.,.25),2u),1.);
    result[5]=float4(environment_radiance(source,float2(.5,.5),4u),1.);
    result[6]=float4(1.);
    for (uint mode=0u;mode<8u;mode++) {
        for (int x=-8;x<=8;x++) {for (int y=-8;y<=8;y++) {
            let uv=float2(float(x),float(y))*.25;
            checked_environment_radiance(source,uv,mode);
            checked_environment_radiance(source,uv*1e20,mode);
        }}
    }
    result[7]=float4(environment_radiance(source,float2(1e20,.25),0u),1.);
    // Seam: u=1/8 on a 2-texel row puts the sample at x=-0.25 (cell -1),
    // blending the last column into the first.
    result[8]=float4(environment_radiance(source,float2(.125,.25)),1.);

}
"#;
        // Replace only the load with an explicit coordinate assertion. This
        // exercises production addressing without Vulkan robust access masking
        // invalid reads. Extreme finite UVs catch integer conversion overflow.
        let prelude=include_str!("../shaders/vork_prelude.slang");
        let addressing=prelude.split("float3 environment_radiance(").nth(1).unwrap()
            .split("// Invocation-local").next().unwrap();
        let addressing=format!("float3 checked_environment_radiance({addressing}")
            .replace("environment_texel(","checked_environment_texel(");
        let checked=r#"
float3 checked_environment_texel(Texture2DArray<uint4> image,int2 cell,bool residual) {
    uint width,height,layers;image.GetDimensions(width,height,layers);
    if (any(cell<int2(0)) || any(cell>=int2(width,height))) {result[6]=float4(0.);}
    return float3(0.);
}
"#;
        let shader=format!("{checked}\n{addressing}\n{shader}");
        let entries=[
            wgpu::BindGroupLayoutEntry {visibility:wgpu::ShaderStages::COMPUTE,..crate::pipelines::environment_texture_entry(0)},
            crate::pipelines::buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false}),
        ];
        let pipeline=crate::medium::test_pipeline(&r,&shader,&crate::medium::Native::default(),&entries,(1,1,1))?;
        let output=d.create_buffer(&wgpu::BufferDescriptor {label:None,size:144,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
        let read=d.create_buffer(&wgpu::BufferDescriptor {label:None,size:144,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        let view=image.create_view(&Default::default());
        let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,wgpu::BindingResource::TextureView(&view)),entry(1,output.as_entire_binding())]});
        let mut encoder=d.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(1,1,1);}
        encoder.copy_buffer_to_buffer(&output,0,&read,0,144);r.queue.submit([encoder.finish()]);
        let(tx,rx)=std::sync::mpsc::channel();read.map_async(wgpu::MapMode::Read,..,move |v|{let _=tx.send(v);});r.wait(None)?;rx.recv()??;
        let mapped=read.get_mapped_range(..);let results:&[[f32;4]]=bytemuck::cast_slice(&mapped);
        assert_eq!(results[6],[1.;4],"source load coordinates escaped the texture");
        assert_eq!(results[7],[3e6,1.5e6,750000.,1.],"wrap must precede integer conversion");
        for channel in 0..3 {
            // Negative seam cell -1 wraps to the last column: b+(a-b)*0.75.
            let (a,b)=(pixels[0][channel],pixels[1][channel]);
            let expected=b+(a-b)*0.75;
            assert!((results[8][channel]-expected).abs()<=expected.abs()*1e-6,"seam channel {channel}: {:?}, expected {expected}",results[8]);
        }
        for channel in 0..3 {
            let a=pixels[0][channel];let b=pixels[1][channel];
            let c=pixels[2][channel];let e=pixels[3][channel];
            let expected=[(a+b+c+e)*0.25,(a*0.25+b*0.75+c)*0.25,(a+b)*0.5,a,a,e];
            for (i,value) in expected.into_iter().enumerate() {
                assert!((results[i][channel]-value).abs()<=value.abs()*1e-6,"sample {i}, channel {channel}: {:?}, expected {value}",results[i]);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang and Vulkan"]
    fn hdr_proposal_keeps_selected_cell_pdf_at_real_resolution() -> Result<()> {
        let r=Renderer::new()?;let d=&r.device;
        let shader=r#"
import environment_proposal;
import vork_constants;
[[vk::binding(0,0)]] StructuredBuffer<float2> cdf;
[[vk::binding(1,0)]] RWStructuredBuffer<float4> result;
[shader("compute")][numthreads(64,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {
    let i=id.x;let u=float2((float(i)+.5)/65536.,(float(reversebits(i)>>16u)+.5)/65536.);
    float pdf;let direction=environment_proposal_sample(cdf,uint2(2048,1024),u,pdf);
    result[i]=float4(direction,pdf);
}
"#;
        let entries=[
            crate::pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage{read_only:true}),
            crate::pipelines::buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage{read_only:false}),
        ];
        let pipeline=crate::medium::test_pipeline(&r,shader,&crate::medium::Native::default(),&entries,(64,1,1))?;
        for row in [1u32,350,1022] {
            let at=row*2048+1024;
            let values:Vec<[f32;2]>=(0..2048*1024).map(|i|[if i>=at {1.} else {0.},0.]).collect();
            let table=init(d,"HDR CDF",bytemuck::cast_slice(&values),wgpu::BufferUsages::STORAGE);
            let output=d.create_buffer(&wgpu::BufferDescriptor {label:None,size:65536*16,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
            let read=d.create_buffer(&wgpu::BufferDescriptor {label:None,size:65536*16,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
            let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,table.as_entire_binding()),entry(1,output.as_entire_binding())]});
            let mut encoder=d.create_command_encoder(&Default::default());
            {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(1024,1,1);}
            encoder.copy_buffer_to_buffer(&output,0,&read,0,65536*16);r.queue.submit([encoder.finish()]);
            let(tx,rx)=std::sync::mpsc::channel();read.map_async(wgpu::MapMode::Read,..,move |v|{let _=tx.send(v);});r.wait(None)?;rx.recv()??;
            let mapped=read.get_mapped_range(..);let results:&[[f32;4]]=bytemuck::cast_slice(&mapped);
            let theta = std::f64::consts::PI / 1024.;
            let omega = std::f64::consts::TAU / 2048. * ((row as f64 * theta).cos() - ((row + 1) as f64 * theta).cos());
            let expected=1./omega as f32;
            for value in results {assert!((value[3]/expected-1.).abs()<0.0002,"row {row}: {value:?}, PDF {expected}");}
            drop(mapped);read.unmap();
        }
        Ok(())
    }

    /// Independent angular moments and cell mass check the production GPU CDF
    /// inversion and PDF, including an all-black map and sparse HDR support.
    #[test]
    #[ignore = "requires pinned Slang and Vulkan"]
    fn environment_proposal_pdf_and_sampling() -> Result<()> {
        let r=Renderer::new()?;let d=&r.device;
        let shader=r#"
import environment_proposal;
import vork_constants;
import vork_prelude;
[[vk::binding(0,0)]] StructuredBuffer<float2> cdf;
[[vk::binding(1,0)]] RWStructuredBuffer<float4> result;
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {
    let cell=id.x;let row=cell/8u;let col=cell%8u;
    let phi=REG_TAU*((float(col)+0.5)/8.-0.5);
    let theta=REG_PI*(float(row)+0.5)/4.;
    let center=float3(sin(theta)*cos(phi),cos(theta),sin(theta)*sin(phi));
    let omega=REG_TAU/8.*(cos(REG_PI*float(row)/4.)-cos(REG_PI*float(row+1u)/4.));
    var count=0.;var moment=0.;
    for (uint i=0u;i<65536u;i++) {
        let u=float2((float(i)+0.5)/65536.,(float(reversebits(i)>>16u)+0.5)/65536.);
        float sampled_pdf;let direction=environment_proposal_sample(cdf,uint2(8,4),u,sampled_pdf);
        let uv=equirect_uv(direction);
        let actual=min(uint2(float2(frac(uv.x),clamp(uv.y,0.,1.))*float2(8,4)),uint2(7,3));
        if (actual.y*8u+actual.x==cell) {count+=1.;moment+=direction.y;}
    }
    result[cell]=float4(count/65536.,moment/65536.,environment_proposal_pdf(cdf,uint2(8,4),center)*omega,0.);
}
"#;
        let entries=[
            crate::pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage{read_only:true}),
            crate::pipelines::buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage{read_only:false}),
        ];
        let pipeline=crate::medium::test_pipeline(&r,shader,&crate::medium::Native::default(),&entries, (1, 1, 1))?;
        for sparse in [false,true] {
            let mut pixels=vec![0f32;32*4];
            for p in pixels.chunks_exact_mut(4) {p[3]=1.;}
            if sparse {for (index,value) in [(3,500.),(17,1.)] {for channel in 0..3 {pixels[index*4+channel]=value;}}}
            let values=distribution(8,4,&pixels);
            let table=init(d,"CDF test",bytemuck::cast_slice(&values),wgpu::BufferUsages::STORAGE);
            let output=d.create_buffer(&wgpu::BufferDescriptor{label:None,size:512,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
            let readback=d.create_buffer(&wgpu::BufferDescriptor{label:None,size:512,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
            let bind=d.create_bind_group(&wgpu::BindGroupDescriptor{label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,table.as_entire_binding()),entry(1,output.as_entire_binding())]});
            let mut encoder=d.create_command_encoder(&Default::default());
            {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(32,1,1);}
            encoder.copy_buffer_to_buffer(&output,0,&readback,0,512);
            let submission=r.queue.submit([encoder.finish()]);
            let(tx,rx)=std::sync::mpsc::channel();readback.map_async(wgpu::MapMode::Read,..,move|v|{let _=tx.send(v);});
            d.poll(wgpu::PollType::Wait{submission_index:Some(submission),timeout:None})?;rx.recv()??;
            let mapped=readback.get_mapped_range(..);let actual:&[[f32;4]]=bytemuck::cast_slice(&mapped);
            for (i,v) in actual.iter().enumerate() {
                let mass=values[i][0]-if i==0 {0.} else {values[i-1][0]};
                let row=i/8;let angle=std::f64::consts::PI/4.;
                let mean=((angle*row as f64).cos()+(angle*(row+1) as f64).cos()) as f32*0.5;
                assert!(v.iter().all(|x|x.is_finite()),"sparse={sparse} cell{i}: {v:?}");
                // Histogram bins round-trip directions through GPU atan2/acos;
                // allow 0.05% absolute mass at cell edges. PDF mass stays tight.
                assert!((v[0]-mass).abs()<0.0005,"sample mass sparse={sparse} cell{i}: {v:?}, expected{mass}");
                assert!((v[1]-mass*mean).abs()<0.0005,"angular moment sparse={sparse} cell{i}: {v:?}");
                assert!((v[2]-mass).abs()<0.00001,"PDF mass sparse={sparse} cell{i}: {v:?}, expected{mass}");
            }
        }
        r.errors.check()?;Ok(())
    }
    /// A row of two equal one-ULP cells after a large prefix: the conditional
    /// column choice must stay 1/2 each with uniform azimuth inside each cell,
    /// which a search against the rounded global target loses.
    #[test]
    #[ignore = "requires pinned Slang and Vulkan"]
    fn environment_proposal_keeps_conditional_precision_after_large_prefix() -> Result<()> {
        let r=Renderer::new()?;let d=&r.device;
        let shader=r#"
import environment_proposal;
import vork_constants;
import vork_prelude;
[[vk::binding(0,0)]] StructuredBuffer<float2> cdf;
[[vk::binding(1,0)]] RWStructuredBuffer<float4> result;
[shader("compute")][numthreads(64,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {
    let u=float2((float(id.x)+0.5)/65536.,0.75+asfloat(0x33800000u));
    float sampled_pdf;let uv=equirect_uv(environment_proposal_sample(cdf,uint2(2,3),u,sampled_pdf));
    result[id.x]=float4(uv,0.,0.);
}
"#;
        let entries=[
            crate::pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage{read_only:true}),
            crate::pipelines::buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage{read_only:false}),
        ];
        let pipeline=crate::medium::test_pipeline(&r,shader,&crate::medium::Native::default(),&entries,(64,1,1))?;
        let ulp=f32::from_bits(0x33800000);
        let values:[[f32;2];6]=[[0.375,0.],[0.75,0.],[0.75+ulp,0.],[0.75+2.*ulp,0.],[0.875,0.],[1.,0.]];
        let table=init(d,"CDF test",bytemuck::cast_slice(&values),wgpu::BufferUsages::STORAGE);
        let bytes=65536*16;
        let output=d.create_buffer(&wgpu::BufferDescriptor{label:None,size:bytes,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
        let readback=d.create_buffer(&wgpu::BufferDescriptor{label:None,size:bytes,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        let bind=d.create_bind_group(&wgpu::BindGroupDescriptor{label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,table.as_entire_binding()),entry(1,output.as_entire_binding())]});
        let mut encoder=d.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(1024,1,1);}
        encoder.copy_buffer_to_buffer(&output,0,&readback,0,bytes);
        let submission=r.queue.submit([encoder.finish()]);
        let(tx,rx)=std::sync::mpsc::channel();readback.map_async(wgpu::MapMode::Read,..,move|v|{let _=tx.send(v);});
        d.poll(wgpu::PollType::Wait{submission_index:Some(submission),timeout:None})?;rx.recv()??;
        let mapped=readback.get_mapped_range(..);let uv:&[[f32;4]]=bytemuck::cast_slice(&mapped);
        let mut count=[0usize;2];let mut moment=[0f64;2];
        for p in uv {
            assert!(p[1]>1./3.-1e-4 && p[1]<2./3.+1e-4,"row {p:?}");
            let x=(p[0].rem_euclid(1.)*2.) as f64;let column=(x as usize).min(1);
            count[column]+=1;moment[column]+=x-column as f64;
        }
        for c in 0..2 {
            assert!((count[c] as f64/65536.-0.5).abs()<0.002,"column {c}: {count:?}");
            assert!((moment[c]/count[c] as f64-0.5).abs()<0.01,"column {c} mean in-cell azimuth {}",moment[c]/count[c] as f64);
        }
        drop(mapped);r.errors.check()?;Ok(())
    }
    #[test]
    #[ignore = "requires pinned Slang and Vulkan"]
    fn environment_mis_energy_links_and_null_connections() -> Result<()> {
        let r=Renderer::new()?;
        let fixtures=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let base:serde_json::Value=vork_scene::v2::with_test_settings(include_str!("../tests/fixtures/analytic.json"), include_str!("../tests/presets/analytic-7fefd52a.json"))?;
        let directory=std::env::temp_dir().join(format!("vork-environment-test-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()));
        std::fs::create_dir(&directory)?;
        let image=directory.join("white.png");
        image::RgbImage::from_pixel(16,8,image::Rgb([255u8;3])).save(&image)?;
        let hemisphere=directory.join("hemisphere.png");
        let mut half_image=image::RgbImage::from_pixel(16,8,image::Rgb([0u8;3]));
        for y in 0..8 {for x in 4..12 {half_image.put_pixel(x,y,image::Rgb([255u8;3]));}}
        half_image.save(&hemisphere)?;
        // Repeated installs must preserve the source proposal on warm caches.
        for (scenario,expected) in [
            ("open",1.),("open",1.),
            ("blocked",0.),("unlinked",0.),
            ("null-linked",1.),
            ("rotated-dark",0.),("rotated-white",1.),
        ] {
            let mut input=base.clone();input["animation"]=serde_json::json!([]);
            input["render"]=serde_json::json!({"resolution":[8,8],"background":"environment","settings":{"integrator":"path","denoiser":null,"samples":256}});
            input["images"]=serde_json::json!({"world":{"file":image,"color_space":"Linear Rec.709"}});
            input["materials"]=serde_json::json!({
                "environment_image":{"program":"std/environment_image","inputs":{"image":{"image":"world"}}},
                "white":{"inputs":{"base_color":[1,1,1],"specular_weight":0}},
                "block":{"inputs":{"base_color":[0,0,0],"specular_weight":0,"geometry_opacity":if scenario=="null-linked" {0.} else {1.}}}});
            input["nodes"]=serde_json::json!([
                {"name":"camera","type":"camera","translation":[0,0,5],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":0.1}}},
                {"name":"slab","type":"mesh","geometry":"shape_box","materials":["white"],"light_link_tags":["receiver"],"scale":[20,20,0.25]},
                {"name":"world","type":"light","shape":"environment","material":"environment_image","energy":1,"color":[1,1,1],"illuminates":if scenario=="unlinked" {serde_json::json!([])} else {serde_json::json!(["receiver"])}}
            ]);
            if scenario=="blocked" || scenario=="null-linked" {
                input["nodes"].as_array_mut().unwrap().push(serde_json::json!({"name":"blocker","type":"mesh","geometry":"shape_box","materials":["block"],"visible_camera":false,"light_link_tags":["other"],"translation":[0,0,3],"scale":[4000,4000,0.1]}));
            }
            if scenario.starts_with("rotated-") {
                input["images"]["world"]["file"]=serde_json::json!(hemisphere);
                let sine=if scenario=="rotated-dark" {std::f32::consts::FRAC_1_SQRT_2} else {-std::f32::consts::FRAC_1_SQRT_2};
                vork_scene::v2::set_transform(&mut input["nodes"][2], serde_json::json!({"rotation":[0,sine,0,std::f32::consts::FRAC_1_SQRT_2]}));
            }
            let installed=r.install(vork_scene::v2::from_test_value(input)?,&fixtures)?;
            {assert_ne!(installed.environment.distribution_size,[1,1],"cached filters dropped source CDF");}
            let mut frame=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
            frame.submit_outputs(&installed,&r,0.,ReadbackOutputs{float:FloatOutputs{color:true,..Default::default()},..Default::default()})?;
            let pixels=frame.collect(&r)?.linear_rgba.unwrap();
            let mean=pixels.iter().map(|p|p[0]).sum::<f32>()/pixels.len() as f32;
            assert!((mean-expected).abs()<0.03,"{scenario}: mean{mean}, expected{expected}");
        }
        std::fs::remove_dir_all(&directory)?;
        r.errors.check()?;Ok(())
    }
