//! A paint unit supplies a fragment function. Its path uses the regular phase
//! vertex streams, compact instance and placement columns, masks and blend state.
use super::*;
use crate::paint::PathProgramSource;
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::Arc,
};

type Pipelines = HashMap<(u32, u32, BlendMode), wgpu::RenderPipeline>;

#[derive(Default)]
pub(super) struct Installed {
    sources: Arc<[PathProgramSource]>,
    modules: HashMap<u32, wgpu::ShaderModule>,
    pipelines: Arc<Pipelines>,
    mask_pipelines: Arc<HashMap<(u32, bool), wgpu::RenderPipeline>>,
    inputs: Option<(wgpu::Buffer, wgpu::BindGroup)>,
    words: Vec<u32>,
    source_hash: u64,
    pub(super) capture_identity: Option<u64>,
    errors: Option<wgpu::Buffer>,
}

impl Installed {
    pub(super) fn snapshot(&self) -> (Arc<Pipelines>, Option<wgpu::BindGroup>) {
        (
            self.pipelines.clone(),
            self.inputs.as_ref().map(|p| p.1.clone()),
        )
    }
}

impl Engine {
    /// Publish the table before submitting a frame. A worker owns its buffer.
    pub fn set_path_programs(
        &self,
        sources: &Arc<[PathProgramSource]>,
        words: &[u32],
        errors: Option<&wgpu::Buffer>,
    ) -> Result<bool> {
        let mut p = self.path_programs.borrow_mut();
        let code = !Arc::ptr_eq(&p.sources, sources) && p.sources != *sources;
        let binding_changed = p.errors.as_ref() != errors;
        if !code && !binding_changed && p.words == words {
            return Ok(false);
        }
        if code {
            let previous = p.sources.clone();
            p.modules
                .retain(|unit, _| p_source_unchanged(*unit, &previous, sources));
            Arc::make_mut(&mut p.pipelines)
                .retain(|(unit, _, _), _| p_source_unchanged(*unit, &previous, sources));
            Arc::make_mut(&mut p.mask_pipelines)
                .retain(|(unit, _), _| p_source_unchanged(*unit, &previous, sources));
            p.sources = sources.clone();
            let mut hash = rustc_hash::FxHasher::default();
            sources.hash(&mut hash);
            p.source_hash = hash.finish();
        }
        let resize = p.words.len() != words.len() || binding_changed;
        p.errors = errors.cloned();
        if resize && !words.is_empty() {
            let size = std::mem::size_of_val(words) as u64;
            anyhow::ensure!(
                size <= u64::from(self.gpu.device.limits().max_storage_buffer_binding_size),
                "path material inputs exceed the device storage binding limit"
            );
            let buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Path material inputs and clocks"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let group = self
                .gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Path material inputs"),
                    layout: &self.layouts.program_inputs_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: errors
                                .context("missing path error word")?
                                .as_entire_binding(),
                        },
                    ],
                });
            p.inputs = Some((buf, group));
        }
        if words.is_empty() {
            p.inputs = None;
        }
        if let Some((buf, _)) = &p.inputs {
            let first = if resize {
                0
            } else {
                words
                    .iter()
                    .zip(&p.words)
                    .take_while(|(a, b)| a == b)
                    .count()
            };
            let end = if resize {
                words.len()
            } else {
                words
                    .iter()
                    .zip(&p.words)
                    .rposition(|(a, b)| a != b)
                    .map_or(first, |i| i + 1)
            };
            if first < end {
                self.gpu.queue.write_buffer(
                    buf,
                    first as u64 * 4,
                    bytemuck::cast_slice(&words[first..end]),
                );
            }
        }
        // A frame worker can see masks published by the serial engine. Their
        // source plan contains offsets, so distinguish the values behind them.
        p.capture_identity = (!sources.is_empty()).then(|| {
            let mut hash = rustc_hash::FxHasher::default();
            p.source_hash.hash(&mut hash);
            words.hash(&mut hash);
            hash.finish()
        });
        p.words.clear();
        p.words.extend_from_slice(words);
        self.retained_masks.borrow_mut().take();
        self.capture_cache.lock().unwrap().clear();
        if resize
            || code
            || self
                .prepared_frame
                .borrow()
                .as_ref()
                .is_some_and(|p| p.gpu.lock().unwrap().captures.is_some())
        {
            self.prepared_frame.borrow_mut().take();
        }
        Ok(true)
    }

    pub fn clear_path_errors(&self) {
        if let Some(errors) = &self.path_programs.borrow().errors {
            self.gpu.queue.write_buffer(errors, 0, &[0; 4]);
        }
    }

    pub(super) fn mask_programs(&self, masks: &[crate::seams::MaskPlan], picture_projection: [f32; 4]) -> Result<crate::effects::mask::MaskPrograms> {
        let mut p = self.path_programs.borrow_mut();
        for draw in masks.iter().flat_map(|mask| mask.draws.iter()) {
            let crate::seams::MaskPaint::Program { body, stroke, .. } = &draw.paint else { continue };
            let unit = body.binding.unit;
            let uv = *stroke && body.binding.stroke_uv;
            if !p.mask_pipelines.contains_key(&(unit, uv)) {
                let source = p.sources.get(unit as usize).context("missing mask paint unit")?;
                let pipe = crate::effects::mask::coverage_pipeline(
                    &self.gpu.device, self.working.as_ref(), Some((source, uv, &self.layouts.program_inputs_bgl)),
                )?;
                Arc::make_mut(&mut p.mask_pipelines).insert((unit, uv), pipe);
            }
        }
        Ok(crate::effects::mask::MaskPrograms { picture_projection, pipelines: p.mask_pipelines.clone(), inputs: p.inputs.as_ref().map(|i| i.1.clone()) })
    }

    pub(super) fn prepare_path_program(
        &self,
        unit: u32,
        samples: u32,
        blend: BlendMode,
    ) -> Result<()> {
        let mut p = self.path_programs.borrow_mut();
        if p.pipelines.contains_key(&(unit, samples, blend)) {
            return Ok(());
        }
        if !p.modules.contains_key(&unit) {
            let source = p
                .sources
                .get(unit as usize)
                .ok_or_else(|| anyhow!("path paint unit {unit} missing"))?;
            let module = self
                .gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Path paint"),
                    source: wgpu::ShaderSource::Wgsl(source.body_module().into()),
                });
            p.modules.insert(unit, module);
        }
        if std::env::var_os("MODA_PIPELINE_STATS").is_some() {
            eprintln!("canvas path pipeline unit {unit}, samples {samples}, blend {blend:?}");
        }
        let pipeline = Self::create_path_pipeline(
            &self.gpu.device,
            &p.modules[&unit],
            &self.layouts.program_layout,
            "path-gradient",
            1,
            samples,
            blend.fixed_function(),
            self.frame_format,
            moda_text_draw::device::PipelineCreation::Required,
        )?;
        Arc::make_mut(&mut p.pipelines).insert((unit, samples, blend), pipeline);
        Ok(())
    }
}

fn p_source_unchanged(unit: u32, previous: &[PathProgramSource], current: &[PathProgramSource]) -> bool {
    current
        .get(unit as usize)
        .is_some_and(|s| Some(s) == previous.get(unit as usize))
}
