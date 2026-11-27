//! CPU-side installation of the path paint table. One unit per Slang program,
//! one word range per material, and one time value per local clock domain.
use crate::{animation, disk_cache, programs};
use anyhow::{Context, Result};
use moda_scene::paint::PathProgram;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::Path,
    sync::Arc,
};
use vork_scene::v2;

#[derive(Default)]
pub(crate) struct Paints {
    pub bindings: HashMap<String, PathProgram>,
    pub sources: Arc<[moda_engine::paint::PathProgramSource]>,
    defaults: Arc<[u32]>,
    animated: bool,
    clock: bool,
}

impl Paints {
    pub fn install(
        scene: &v2::Scene,
        base: &Path,
        names: BTreeSet<String>,
        plan: &mut animation::Plan,
    ) -> Result<Self> {
        if names.is_empty() {
            return Ok(Self::default());
        }
        let store = disk_cache::Store::host();
        let abi = programs::abi();
        let compiler = vork_slang::Compiler::discover().for_paint();
        let mut result = Self::default();
        let mut units = HashMap::new();
        let mut sources = Vec::new();
        let mut defaults = vec![0]; // slot zero supplies time to an untimed program
        let keyed = plan.material_inputs();
        for name in names {
            let material = scene
                .materials
                .get(&name)
                .with_context(|| format!("unknown path material {name:?}"))?;
            if !units.contains_key(&material.program) {
                let (source, origin) = programs::material_source(scene, base, &material.program)?;
                let key = blake3::hash(b"canvas-path-paint-v1");
                let program = programs::cached_proof(
                    &store,
                    key,
                    &compiler,
                    &abi,
                    &source,
                    &origin,
                    |compiler| {
                        if std::env::var_os("MODA_PIPELINE_STATS").is_some() {
                            eprintln!("canvas path Slang compile {:?}", material.program);
                        }
                        compiler.compile_paint(&abi, &source, &origin, "path")
                    },
                )
                .with_context(|| {
                    format!("path material {name:?}, program {:?}", material.program)
                })?;
                let index = u32::try_from(sources.len())?;
                let module =
                    fragment_module(&program).with_context(|| format!("materials.{name}"))?;
                sources.push(module);
                units.insert(material.program.clone(), (index, program));
            }
            let (unit, program) = &units[&material.program];
            let inputs: BTreeMap<_, _> = material
                .inputs
                .iter()
                .map(|(k, v)| Ok((k.clone(), serde_json::to_value(v)?)))
                .collect::<Result<_>>()?;
            let mut words = program
                .inputs
                .pack(&inputs, |field, _| {
                    anyhow::bail!("path paint input {field}: expected a numeric value")
                })
                .with_context(|| format!("materials.{name}.inputs"))?;
            let offset = defaults.len();
            if keyed.iter().any(|(m, _)| m == &name) {
                programs::bind_input_words(
                    plan,
                    &name,
                    &program.inputs,
                    &mut words,
                    offset,
                    0.,
                    false,
                )?;
                result.animated = true;
            }
            defaults.extend(words);
            result.clock |= program.time;
            result.bindings.insert(
                name,
                PathProgram {
                    unit: *unit,
                    site: 0,
                    inputs: u32::try_from(offset)?,
                    clock: 0,
                    stroke_uv: program.stroke_uv,
                },
            );
        }
        result.defaults = defaults.into();
        result.sources = sources.into();
        Ok(result)
    }

    pub fn clock_slot(&self, plan: &animation::Plan, i: usize) -> Result<u32> {
        Ok(if self.clock {
            u32::try_from(self.defaults.len() + plan.canvas_clock_domain(i))?
        } else {
            0
        })
    }

    pub fn sample(
        &self,
        plan: &animation::Plan,
        samples: &animation::Samples,
        scene: &vork_time::exact::Germ,
    ) -> Result<Arc<[u32]>> {
        if !self.animated && !self.clock {
            return Ok(self.defaults.clone());
        }
        let mut words = self.defaults.to_vec();
        if self.animated {
            for (binding, value) in plan.material_values(samples, 0.) {
                let value = programs::sample_words(value, binding);
                words[binding.slot..binding.slot + value.len()].copy_from_slice(&value);
            }
        }
        if self.clock {
            if plan.has_clocks() {
                for domain in &samples.clocks {
                    words.push(
                        domain
                            .as_ref()
                            .map(|t| t.numeric())
                            .transpose()?
                            .unwrap_or(0.)
                            .to_bits(),
                    );
                }
            } else {
                words.push(scene.numeric()?.to_bits());
            }
        }
        Ok(words.into())
    }
}

fn fragment_module(program: &vork_slang::PaintProgram) -> Result<moda_engine::paint::PathProgramSource> {
    let source = moda_engine::paint::PathProgramSource {
        entry: program.entry.clone(), wgsl: program.wgsl.clone(),
    };
    let module = source.body_module();
    let ir = wgpu::naga::front::wgsl::parse_str(&module)
        .map_err(|e| anyhow::anyhow!(e.emit_to_string(&module)))?;
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
    )
    .validate(&ir)
    .with_context(|| "invalid path fragment")?;
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires pinned Slang"]
    fn example_fragments_validate_with_the_path_vertex_and_mask() -> Result<()> {
        let base =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/motion-graphics/path-paint");
        for file in ["cloth.slang", "ink.slang"] {
            let path = base.join(file);
            let source = std::fs::read_to_string(&path)?;
            let program = vork_slang::Compiler::discover().compile_paint(
                &programs::abi(),
                &source,
                &path,
                "path",
            )?;
            fragment_module(&program)?;
        }
        Ok(())
    }
}
