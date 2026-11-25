//! Error-only migration refusals. No entry accepts an old spelling.
//! Keep wording here; callers own detection, error context and exit classification.

/// Removed/renamed spelling and its exact message template.
pub const REFUSALS: &[(&str, &str)] = &[
    ("moved fields", "{moved}{space}; see vork help upgrade"),
    ("render.display", "render.display was removed; use composition.view"),
    ("render.view", "render.view: use composition.view"),
    ("render.compositor", "{old} was removed; use {new}"),
    ("composition.space.suffix", "; composition.space is required too: write \"3d\" (a scene without a composition is a world)"),
    ("view.inputs.dither_intensity", "{name}: view inputs.dither_intensity was removed; output encoding owns dither: delete the input (1, the default, draws as before; \
                 0 or more than 1 has no setting: dither scales with the output's bit depth, and grain is a compositor program); see vork help upgrade"),
    ("composition.display", "{path}.display: use composition.view; see vork help upgrade"),
    ("composition.space", "{path}.space is required: write \"3d\" for a world (what a scene without a composition is) or \"2d\" for a canvas; see vork help upgrade"),
    ("compositor.node", "compositor {node:NAME} was renamed to {step:NAME}; see vork help upgrade"),
    ("std/text.align_x", "std/text.align_x {align_x} was removed; std/text does not wrap: write left, center or right, or use wrapped canvas text for justification; see vork help upgrade"),
    ("preservesPitch | preserves_pitch", "unknown field {name}; use time.pitch: keep; see `vork help upgrade`"),
    ("MathOperation.FRACTION | MathOperation.fraction", "unknown {kind} {name}; it was renamed {new}; see `vork help upgrade`"),
    ("program.std/fog", "unknown program \"std/fog\"; this program was removed; see `vork help upgrade`"),
    ("program.blender/principled_volume", "unknown program \"blender/principled_volume\"; this program was removed; see `vork help upgrade`"),
    ("program.blender.volume", "unknown program \"blender.volume\"; this program was removed; see `vork help upgrade`"),
    ("program.post/vignette", "unknown program \"post/vignette\"; this program was removed; see `vork help upgrade`"),
    ("program.post/lens_distortion", "unknown program \"post/lens_distortion\"; this program was removed; see `vork help upgrade`"),
    ("program.post/glare", "unknown program \"post/glare\"; this program was removed; see `vork help upgrade`"),
    ("program.std/highlight", "unknown program \"std/highlight\"; this program was removed; see `vork help upgrade`"),
    ("animation.layer.blend.add", "animation.layer.blend: use \"combine\" instead of \"add\"; see `vork help upgrade`"),
    ("program.blender/principled", "unknown program \"blender/principled\"; use std/openpbr; see `vork help upgrade`"),
    ("post/white_balance | post.white_balance", "{name} was removed; std.white_balance is a library, not a program: write a composite program that imports it and returns white_balance(color, inputs); see `vork help upgrade`"),
    ("std/particles | std.particles", "{name} was removed; use fluid.particles; see `vork help upgrade`"),
    ("std/display | std.display", "{name} was removed; use std.view: finish_view(srgb, ctx, gamma); see `vork help upgrade`"),
];

/// Substitute caller-supplied values in one pass: authored names cannot introduce placeholders.
pub fn message(field: &str, values: &[(&str, &str)]) -> String {
    known_message(field, values).expect("migration refusal is registered")
}

/// Error-only lookup; a miss leaves ordinary unknown-name diagnostics in charge.
pub fn known_message(field: &str, values: &[(&str, &str)]) -> Option<String> {
    let template = REFUSALS.iter().find(|(keys, _)| keys.split(" | ").any(|key| key == field))?.1;
    let mut result = String::new();
    let mut rest = template;
    while let Some(start) = rest.find('{') {
        result.push_str(&rest[..start]);
        let after = &rest[start..];
        let Some(end) = after.find('}') else { result.push_str(after); return Some(result); };
        let key = &after[1..end];
        if let Some((_, value)) = values.iter().find(|(name, _)| *name == key) {
            result.push_str(value);
        } else {
            // Literal examples such as {node:NAME} remain exactly as written.
            result.push_str(&after[..=end]);
        }
        rest = &after[end + 1..];
    }
    result.push_str(rest);
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_refusal_is_unique_one_line_and_keeps_literal_examples() {
        let mut keys = std::collections::BTreeSet::new();
        for (field, template) in REFUSALS {
            for key in field.split(" | ") {
                assert!(keys.insert(key), "duplicate refusal: {key}");
                assert_eq!(message(key, &[]), *template, "{key}");
            }
            assert!(!template.contains(['\n', '\r']), "{field}");

        }
        assert_eq!(message("compositor.node", &[]),
            "compositor {node:NAME} was renamed to {step:NAME}; see vork help upgrade");
        assert_eq!(message("render.compositor", &[("old", "{new}"), ("new", "composition.compositor")]),
            "{new} was removed; use composition.compositor");
    }
}
