    use super::*;
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn material_tables_sample_beyond_1024_textures_and_64_samplers() -> Result<()> {
        let r = crate::test_renderer()?;
        let images=Default::default();
        let mut installer=TextureInstaller::new(&r,&images);
        let samplers:Vec<_>=(0..65).map(|i|r.device.create_sampler(&wgpu::SamplerDescriptor {lod_min_clamp:i as f32/128.,lod_max_clamp:32.,..Default::default()})).collect();
        for i in 0..1025 {
            let texture=r.device.create_texture(&wgpu::TextureDescriptor {label:Some("descriptor capacity fixture"),size:extent(1,1),mip_level_count:1,sample_count:1,dimension:wgpu::TextureDimension::D2,format:wgpu::TextureFormat::Rgba8Unorm,usage:wgpu::TextureUsages::COPY_DST|wgpu::TextureUsages::TEXTURE_BINDING,view_formats:&[]});
            r.queue.write_texture(wgpu::TexelCopyTextureInfo {texture:&texture,mip_level:0,origin:wgpu::Origin3d::ZERO,aspect:wgpu::TextureAspect::All},&[23,191,91,255],wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(4),rows_per_image:Some(1)},extent(1,1));
            installer.slot(texture.create_view(&Default::default()),samplers[i%65].clone());
        }
        assert_eq!(installer.counts()?,(1025,65));
        let material_layout=r.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:None,entries:&crate::pipelines::material_entries_for(1025,65)});
        let table=installer.table(&material_layout,vec![0;16],false)?;
        let output_layout=r.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:None,entries:&[crate::pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false})]});
        let output=crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {label:None,size:16,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
        let readback=crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {label:None,size:16,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
        let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&output_layout,entries:&[entry(0,output.as_entire_binding())]});
        let layout=r.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {label:None,bind_group_layouts:&[Some(&output_layout),Some(&material_layout)],immediate_size:0});
        let source="import vork_material_table; [[vk::binding(0,0)]] RWStructuredBuffer<float4> output : register(u0); [shader(\"compute\")][numthreads(1,1,1)] void capacity_main() {output[0]=material_image(uint2(1024,64),float2(.5),float2(0),float2(0));}";
        let mut modules=crate::native::SOURCES.to_vec();modules.push(("descriptor_capacity",source));
        let metal=r.adapter_info.backend==wgpu::Backend::Metal;
        let (_,target,flags)=if metal {crate::forward_variant::METAL_TARGET} else {crate::forward_variant::SPIRV_TARGET};
        let registers=crate::binding_abi::material_registers(1);
        let args:Vec<_>=["-target",target].into_iter().chain(flags.iter().copied()).chain(registers.iter().map(String::as_str)).collect();
        let code=vork_slang::Compiler::discover().compile_modules(&modules,"descriptor_capacity",&[("capacity_main","compute")],&args)?;
        let shader=if metal {crate::native::module(&r.device,"descriptor capacity",None,Some(&String::from_utf8(code)?),(1,1,1))}
            else {crate::native::module(&r.device,"descriptor capacity",Some(&code),None,(1,1,1))};
        let pipeline=r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {label:None,layout:Some(&layout),module:&shader,entry_point:Some("capacity_main"),compilation_options:Default::default(),cache:None});
        let mut encoder=r.device.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.set_bind_group(1,table.bind(),&[]);pass.dispatch_workgroups(1,1,1);}
        encoder.copy_buffer_to_buffer(&output,0,&readback,0,16);
        let submission=crate::command_finish::Segments::submit(&r.queue,encoder);
        let (tx,rx)=std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read,..,move |value|{let _=tx.send(value);});
        r.wait(Some(submission))?;rx.recv()??;
        let mapped=readback.get_mapped_range(..);
        let color:&[f32]=bytemuck::cast_slice(&mapped);
        for (actual,expected) in color.iter().zip([23.,191.,91.,255.]) {assert!((actual-expected/255.).abs()<1e-5);}
        Ok(())
    }
