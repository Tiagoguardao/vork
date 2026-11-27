//! A numeric fragment stage over an existing canvas path. Only code and the
//! reflected word layout enter this artifact. Every material supplies its own
//! base in the input table.
use super::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct PaintProgram {
    pub wgsl: String,
    pub entry: String,
    pub inputs: InputLayout,
    pub stroke_uv: bool,
    pub time: bool,
    pub dependencies: Vec<PathBuf>,
}

impl Compiler {
    /// Scope the authored program and its resolved imports to the canvas paint ABI.
    pub fn for_paint(mut self) -> Self {
        self.stage_module = Some(("canvas_paint", include_str!("canvas_paint.slang")));
        self
    }

    pub fn compile_paint(
        &self,
        abi: &Abi,
        source: &str,
        origin: &Path,
        tag: &str,
    ) -> Result<PaintProgram> {
        if self.stage_module.is_none() {
            return self.clone().for_paint().compile_paint(abi, source, origin, tag);
        }
        let compiled = self.compile_with_capacity(abi, source, origin, tag, None, false, true, false, |inputs, declarations| {
            ensure!(declarations.contains("paint"), "path paint requires float4 paint(PaintContext, Inputs)");
            for field in &inputs.fields {
                ensure!(matches!(field.ty, Type::Float | Type::Int | Type::Uint | Type::Vector(_))
                    && field.enumeration.is_none(), "paint input {}: path paint accepts numeric inputs only", field.name);
            }
            let decode = inputs.slang_decode();
            Ok(format!(r#"[shader("compute")][numthreads(1,1,1)] void vork_entry() {{}}
export float4 vork_paint_{tag}(float2 position, float2 uv, float2 strokeUV, float time, uint base) {{
    {decode}
    return paint(PaintContext(position, uv, strokeUV, time), inputs);
}}"#))
        })?;
        let entry = format!("vork_paint_{tag}_0_{tag}");
        let (stroke_uv, time) = effects::paint_reads(&compiled.output.wgsl, &entry)?;
        Ok(PaintProgram {
            entry,
            wgsl: compiled.output.wgsl,
            inputs: compiled.inputs,
            stroke_uv,
            time,
            dependencies: compiled.output.dependencies,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires pinned Slang"]
    fn paint_context_is_scoped_to_paint_including_imported_helpers() -> Result<()> {
        let abi = Abi {
            aov_count: 0,
            context: Abi::spec_context(),
            closure: openpbr_fields(),
        };
        let compiler = Compiler::discover();
        let dir = Workspace::new()?;
        let origin = dir.0.join("ink.slang");
        fs::write(
            dir.0.join("ink_helper.slang"),
            "import vork; public float across(PaintContext c) { return c.stroke_uv.y; }",
        )?;
        let source = "import ink_helper; struct Inputs {}; float4 paint(PaintContext c, Inputs i) { return float4(across(c)); }";
        let ordinary = compiler.reflect_inputs(&abi, source, &origin).unwrap_err();
        assert!(format!("{ordinary:#}").contains("PaintContext"));
        let paint = compiler.for_paint();
        let prepared = paint.prepare_program(&abi, source, &origin)?;
        let (program, receipt) = prepared
            .compiler()
            .capture_inputs(|c| c.compile_paint(&abi, source, &origin, "ink"))?;
        receipt.verify_proof(&prepared.receipt())?;
        assert!(program.stroke_uv);
        assert!(!program.time);
        assert!(
            program
                .dependencies
                .contains(&dir.0.join("ink_helper.slang"))
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang"]
    fn only_live_paint_coordinates_request_the_extra_stream() -> Result<()> {
        let abi = Abi {
            aov_count: 0,
            context: Abi::spec_context(),
            closure: openpbr_fields(),
        };
        for (expression, stroke, time) in [
            ("fwidth(c.uv.x) + i.gain", false, false),
            ("c.stroke_uv.y * i.gain + c.time", true, true),
        ] {
            let source = format!(
                "struct Inputs {{ float gain=1; }}; float4 paint(PaintContext c, Inputs i) {{ return float4({expression}); }}"
            );
            let program = Compiler::discover().compile_paint(
                &abi,
                &source,
                Path::new("paint.slang"),
                "test",
            )?;
            assert_eq!(
                (program.stroke_uv, program.time),
                (stroke, time),
                "{}",
                program.wgsl
            );
        }
        Ok(())
    }
}
