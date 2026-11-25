    use super::*;
    use serde_json::json;

    fn scene() -> Result<Scene> {
        Ok(vork_scene::v2::from_test_value(json!({"version":2,
            "composition":{"space":"3d","view":{"program":"std/standard","inputs":{}}},"render":{"dither":false,"resolution":[16,16],"fps":2,"duration":2,"settings":{"integrator":"raster","samples":1}},
            "nodes":[{"type":"camera","name":"cam"},{"type":"mesh","geometry":"g","name":"actors","materials":["m"],"instances":{"attributes":{"TRANSLATION":"positions"}}}],
            "buffers":{"p":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[0,1,0]]},"positions":{"type":"vec3","data":[[0,0,-5],[8,0,-5],[30,0,-5],[40,0,-5]]}},
            "geometry":{"g":{"attributes":{"POSITION":"p"}}},
            "materials":{"m":{"inputs":{"base_weight":0,"specular_weight":0,"emission_color":[1,0,0],"emission_luminance":1}}},
            "animation":[{"tracks":[
                {"path":["nodes","cam","translation"],"interpolation":"STEP","keys":[[0,[0,0,0]],[1,[8,0,0]]]},
                {"path":["materials","m","inputs","emission_color"],"keys":[[0,[1,0,0]],[0.5,[0,1,0]],[1,[0,0,1]]]}
            ]}]}))?)
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn static_selection_retains_work_across_shading_slots_repeat_and_seek() -> Result<()> {
        let r = crate::test_renderer()?;
        for blur in [false,true] {
        let mut input = scene()?;
        if blur {
            input.render.motion_blur = Some(vork_scene::v2::from_test_value(json!({"shutter":0.5,"position":"center"}))?);
            input.render.settings.samples = 4;
        }
        let mut s = r.install(input, Path::new("."))?;
        let mut slots = [s.create_frame_slot(&r, crate::ReadbackPrecision::Float)?,s.create_frame_slot(&r, crate::ReadbackPrecision::Float)?];
        let mut first: [Option<Vec<[f32;4]>>;2] = [None,None];
        for (slot,time,changes_camera) in [(0,0.,true),(1,0.5,true),(0,0.5,false),(1,0.,false),(0,1.,true),(1,0.5,false),(0,0.,true)] {
            let before = slots[slot].view.camera_cull.test_visits();
            let actual = s.render_frame(&r,&mut slots[slot],time)?;
            let after = slots[slot].view.camera_cull.test_visits();
            if !blur && changes_camera { assert!(after > before,"camera must prepare at {time}"); }
            else if !blur { assert_eq!(after,before,"shading-only time {time} rescanned objects"); }
            let mut fresh = s.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
            let expected = s.render_frame(&r,&mut fresh,time)?;
            crate::test_linear::assert_linear_eq((actual).linear_rgba.as_ref().expect("linear readback"), (expected).linear_rgba.as_ref().expect("linear readback"));
            assert_eq!(actual.ids,expected.ids);
            crate::test_linear::assert_depth_eq(actual.depth_f32.as_ref().expect("float depth readback"), expected.depth_f32.as_ref().expect("float depth readback"));
            assert!(actual.ids.iter().any(|&id|id!=0),"selected actor must remain visible");
            if let Some(prior) = first[slot].as_ref() {
                if time == 0. && slot == 0 { crate::test_linear::assert_linear_eq(actual.linear_rgba.as_ref().unwrap(),prior); }
            } else { first[slot] = actual.linear_rgba; }
        }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn preparing_capture_staging_preserves_live_selection() -> Result<()> {
        let r = crate::test_renderer()?;
        let s = r.install(scene()?, Path::new("."))?;
        let frame = s.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
        let cull = &frame.view.camera_cull;
        let vp = Mat4::perspective_rh(1., 1., 0.1, 100.);
        let mut changes = ChangeTracker::default();
        changes.sample = 1;
        cull.prepare_exposure(vp, &frame.object_values, &[], &mut changes, false);
        let samples = cull.sample_values(&r.device, 1).unwrap();
        cull.prepare_sample(vp * Mat4::from_translation(Vec3::new(-8., 0., 0.)),
            &frame.object_values, &[], &mut changes, &samples, 0);
        let before = cull.test_visits();
        changes.sample += 1;
        changes.bytes.clear();
        changes.uploads.clear();
        cull.prepare_exposure(vp, &frame.object_values, &[], &mut changes, false);
        assert_eq!(cull.test_visits(), before, "staging did not replace live destinations");
        assert!(changes.uploads.is_empty(), "staging must not invalidate live CPU byte history");
        r.errors.check()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn selection_observes_publications_bounds_and_capture_buffer_replacement() -> Result<()> {
        let r = crate::test_renderer()?;
        let mut s = r.install(scene()?, Path::new("."))?;
        let frame = s.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
        // Exercise the CPU receiver contract with a producer-owned local bound.
        // No modified scene is rendered: ownership is supplied to selection here.
        s.draws[0].computed = true;
        let cull = crate::passes::camera_cull::Resources::new(&s,&r,&frame.view.uniform,&frame.objects,s.material_table.records(),None,None);
        let mut objects = frame.object_values.clone();
        let vp = Mat4::perspective_rh(1.,1.,0.1,100.);
        let mut changes = ChangeTracker::default();
        changes.sample = 1;
        cull.prepare_exposure(vp,&objects,&[None],&mut changes,false);
        let count = cull.test_visits();
        assert_eq!(count,objects.len());
        let next = |changes: &mut ChangeTracker| { changes.sample+=1;changes.bytes.clear();changes.uploads.clear(); };
        next(&mut changes);
        cull.prepare_exposure(vp,&objects,&[None],&mut changes,false);
        assert_eq!(cull.test_visits(),count);
        assert!(changes.uploads.is_empty());
        let outside = Some((Vec3::splat(100.),Vec3::splat(101.)));
        cull.prepare_exposure(vp,&objects,&[outside],&mut changes,false);
        assert_eq!(cull.test_visits(),2*count,"new evaluated bounds invalidate");
        assert!(!changes.uploads.is_empty(),"changed bounds must publish selection");
        next(&mut changes);
        cull.prepare_exposure(vp,&objects,&[outside],&mut changes,false);
        assert_eq!(cull.test_visits(),2*count);
        objects[0].model = Mat4::from_translation(Vec3::new(1.,0.,-5.));
        changes.upload(&frame.objects,0,bytemuck::bytes_of(&objects[0].gpu()));
        cull.prepare_exposure(vp,&objects,&[outside],&mut changes,false);
        assert_eq!(cull.test_visits(),3*count);
        // A second publication has the same sample stamp; it must not retain
        // the first publication's selection.
        objects[0].metadata[1] = 0;
        changes.upload(&frame.objects,0,bytemuck::bytes_of(&objects[0].gpu()));
        cull.prepare_exposure(vp,&objects,&[outside],&mut changes,false);
        assert_eq!(cull.test_visits(),4*count);
        next(&mut changes);
        changes.upload(s.material_table.records(),0,&[1,2,3,4]);
        cull.prepare_exposure(vp,&objects,&[outside],&mut changes,false);
        assert_eq!(cull.test_visits(),4*count,"shading publication must retain selection");
        let samples = cull.sample_values(&r.device,2).unwrap();
        cull.prepare_sample(vp,&objects,&[None],&mut changes,&samples,0);
        cull.prepare_sample(vp*Mat4::from_translation(Vec3::new(-8.,0.,0.)),&objects,&[None],&mut changes,&samples,1);
        let mut encoder = r.device.create_command_encoder(&Default::default());
        cull.copy_sample(&mut encoder,&mut changes,&samples,0);
        for (buffer,offset,source,size) in &changes.uploads {
            r.queue.write_buffer(buffer,*offset,&changes.bytes[*source as usize..(*source+*size) as usize]);
        }
        r.queue.submit([encoder.finish()]);
        let before = cull.test_visits();
        next(&mut changes);
        cull.prepare_exposure(vp,&objects,&[outside],&mut changes,false);
        assert_eq!(cull.test_visits(),before+count,"capture restoration invalidates the live selection");
        for (buffer,offset,source,size) in &changes.uploads {
            r.queue.write_buffer(buffer,*offset,&changes.bytes[*source as usize..(*source+*size) as usize]);
        }
        let commands = cull.commands().unwrap();
        let read = r.device.create_buffer(&wgpu::BufferDescriptor { label:None,size:commands.size()+cull.instances.size(),
            usage:wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,mapped_at_creation:false });
        let mut encoder = r.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(commands,0,&read,0,commands.size());
        encoder.copy_buffer_to_buffer(&cull.instances,0,&read,commands.size(),cull.instances.size());
        let submission = r.queue.submit([encoder.finish()]);
        let (tx,rx) = std::sync::mpsc::channel();
        read.map_async(wgpu::MapMode::Read,..,move |result| {let _=tx.send(result);});
        r.device.poll(wgpu::PollType::Wait {submission_index:Some(submission),timeout:None})?;
        rx.recv()??;
        let mapped = read.get_mapped_range(..);
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        assert_eq!(words[1],0,"CPU selection must replace the capture's nonzero GPU count, even when its bytes equal the earlier CPU selection");
        assert!(words[commands.size() as usize / 4..].iter().all(|&word| word==0),
            "CPU selection must also replace the capture's instance map");
        drop(mapped);
        read.unmap();
        r.errors.check()?;
        Ok(())
    }
