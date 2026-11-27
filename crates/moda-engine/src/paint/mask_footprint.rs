//! Select the picture-grid evaluator only for Slang units that emit a derivative.
use super::PathProgramSource;
use anyhow::Result;
use wgpu::naga;

impl PathProgramSource {
    pub(crate) fn mask_library(&self) -> Result<String> {
        let mut uses_derivatives = false;
        if ["dpdx", "dpdy", "fwidth"]
            .iter()
            .any(|op| self.wgsl.contains(op))
        {
            let source = format!(
                "fn vork_material_word(i: u32) -> u32 {{ return i; }}\n{}",
                self.wgsl
            );
            let module = naga::front::wgsl::parse_str(&source)
                .map_err(|error| anyhow::anyhow!(error.emit_to_string(&source)))?;
            uses_derivatives = module.functions.iter().any(|(_, f)| {
                f.expressions
                    .iter()
                    .any(|(_, e)| matches!(e, naga::Expression::Derivative { .. }))
            });
        }
        Ok(format!(
            "{}\nconst pathPaintUsesDerivatives = {uses_derivatives};",
            self.library()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_derivatives_in_helpers_select_the_picture_grid() -> Result<()> {
        for (expression, expected) in [
            ("x", false),
            ("dpdxCoarse(x)", true),
            ("fwidth(floor(x))", true),
        ] {
            let unit = PathProgramSource {
                wgsl: format!("fn dpdy_helper(x: f32) -> f32 {{ return {expression}; }}"),
                entry: "dpdy_helper".into(),
            };
            let library = unit.mask_library()?;
            assert!(library.contains(&unit.wgsl));
            assert!(library.contains(&format!("pathPaintUsesDerivatives = {expected};")));
        }
        Ok(())
    }
}
