//! Unstable, internal: all Rust APIs in this crate are free to change.
//!
//! Language-independent JSON fragments from the same implementation as @voyager-so/vork.
//! The embedded engine has no host bindings, filesystem or network access.
mod expand;
pub use expand::{expand_scene, expand_scene_with, load_scene};
pub mod environment;
mod extrude;
pub mod moda;
mod sky;
mod sweep;
mod text;
pub use text::input_help as text_input_help;

use anyhow::{Context, Result};
#[cfg(test)]
use serde_json::Value;

pub const HELP: &str = include_str!("help.txt");

// Native counterpart of validate.ts clampRadii: one proportional scale satisfies
// all edge constraints. Callers validate finite, nonnegative radii first.
fn clamp_radii(radii: &[f32], edges: &[(f32, &[usize])]) -> Vec<f32> {
    let maximum = radii.iter().copied().fold(0f32, f32::max);
    if maximum == 0. {
        return radii.to_vec();
    }
    let mut fitted = maximum;
    for (length, corners) in edges {
        let sum: f32 = corners.iter().map(|&i| radii[i] / maximum).sum();
        if sum > 0. {
            fitted = fitted.min(length / sum);
        }
    }
    if fitted == maximum {
        radii.to_vec()
    } else {
        radii.iter().map(|r| (r / maximum) * fitted).collect()
    }
}

pub fn evaluate(function: &str, argument: &str) -> Result<String> {
    evaluate_uncached(function, argument, &|_, _| Ok(()))
}

fn evaluate_uncached(function: &str, argument: &str,
    observed: &dyn Fn(&std::path::Path, &[u8]) -> Result<()>,
) -> Result<String> {
    evaluate_observed(function, argument, observed, || None, |_| {})
}

/// Only the embedded geometry builders have closed inputs. The caller owns the
/// key (including these sources and the complete call), storage and eviction.
/// Native builders keep their ordinary path, including text's font reads.
pub fn evaluate_with_cache(
    function: &str,
    argument: &str,
    read: impl FnOnce() -> Option<String>,
    write: impl FnOnce(&str),
) -> Result<String> {
    evaluate_observed(function, argument, &|_, _| Ok(()), read, write)
}

/// Report the bytes used by native builders, including recursive expansion.
/// Pure builders can still use their ordinary host cache.
pub fn evaluate_observed(
    function: &str,
    argument: &str,
    observed: &dyn Fn(&std::path::Path, &[u8]) -> Result<()>,
    read: impl FnOnce() -> Option<String>,
    write: impl FnOnce(&str),
) -> Result<String> {
    match function {
        "expand" => {
            let mut scene = serde_json::from_str(argument)?;
            expand_scene_with(&mut scene, &vork_inputs::current_dir()?, &|name, args|
                evaluate_uncached(name, args, observed))?;
            return Ok(serde_json::to_string(&scene)?);
        }
        "sky" => return sky::evaluate(argument),
        "text" => return text::evaluate(argument, observed),
        "extrude" => return extrude::evaluate(argument),
        "sweep" => return sweep::evaluate(argument),
        _ => {}
    }
    if let Some(result) = read() { return Ok(result); }
    let runtime = rquickjs::Runtime::new().context("initialize standard-library engine")?;
    let context =
        rquickjs::Context::full(&runtime).context("initialize standard-library context")?;
    let script = format!(
        "{}\nJSON.stringify(stdEvaluate({},JSON.parse({})), (key,value) => {{ if (typeof value === 'number' && !Number.isFinite(value)) throw new Error('non-finite standard-library result at '+key); return value; }})",
        include_str!("sdk.js"),
        serde_json::to_string(function)?,
        serde_json::to_string(argument)?
    );
    let result = context.with(|ctx| {
        ctx.eval::<String, _>(script).map_err(|error| {
            let exception = ctx.catch();
            let message = exception
                .as_exception()
                .and_then(|e| e.message())
                .unwrap_or_else(|| error.to_string());
            anyhow::anyhow!("standard-library function failed: {message}")
        })
    })?;
    write(&result);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_geometry_reuses_exact_results_without_hiding_changed_or_invalid_calls() {
        let cache = std::cell::RefCell::new(std::collections::BTreeMap::new());
        let writes = std::cell::Cell::new(0);
        let cached = |function: &str, argument: &str| {
            let key = (function.to_owned(), argument.to_owned());
            evaluate_with_cache(function, argument,
                || cache.borrow().get(&key).cloned(),
                |result| {
                    writes.set(writes.get() + 1);
                    cache.borrow_mut().insert(key.clone(), result.to_owned());
                })
        };
        let calls = [
            ("sphere", r#"{"radius":0.85,"segments":16,"rings":8}"#),
            ("sphere", r#"{"radius":0.9,"segments":16,"rings":8}"#),
            ("box", r#"{"size":[0.9,0.9,0.9]}"#),
            ("grid", r#"{"size":[200,200]}"#),
        ];
        for (function, argument) in calls {
            let expected = evaluate(function, argument).unwrap();
            assert_eq!(cached(function, argument).unwrap(), expected);
            assert_eq!(cached(function, argument).unwrap(), expected);
        }
        assert_eq!(writes.get(), calls.len());
        assert!(cached("sphere", r#"{"radius":-1}"#).is_err());
        let error = cached("shpere", "{}").unwrap_err().to_string();
        assert!(error.contains("unknown standard-library function shpere"), "{error}");
        assert_eq!(writes.get(), calls.len(), "failed calls are never published");
    }

    #[test]
    fn file_dependent_builders_do_not_consult_closed_geometry_cache() {
        let argument = serde_json::json!({"body":"A", "font": std::env::temp_dir()
            .join(format!("vork-std-cache-missing-font-{}.ttf", std::process::id()))}).to_string();
        let error = evaluate_with_cache("text", &argument,
            || panic!("a font read cannot be replaced by a closed-input cache hit"),
            |_| panic!("a file-dependent result cannot be published here")).unwrap_err();
        assert!(error.chain().any(|cause| cause.is::<std::io::Error>()));
    }

    #[test]
    fn geometry_names_default_to_builder() {
        for (builder, options) in [
            ("box", serde_json::json!({})),
            ("sphere", serde_json::json!({})),
            ("grid", serde_json::json!({})),
            ("cylinder", serde_json::json!({})),
            ("cone", serde_json::json!({})),
            ("torus", serde_json::json!({})),
            ("capsule", serde_json::json!({})),
            ("revolve", serde_json::json!({"profile":[[0,0],[1,0],[1,1],[0,1]]})),
            ("rounded_box", serde_json::json!({})),
            (
                "extrude",
                serde_json::json!({"outline":[[0,0],[1,0],[0,1]]}),
            ),
            (
                "text",
                serde_json::json!({"body":"x","font":concat!(env!("CARGO_MANIFEST_DIR"),"/tests/fonts/LiberationSans-Regular.ttf")}),
            ),
        ] {
            let implicit: Value =
                serde_json::from_str(&evaluate(builder, &options.to_string()).unwrap()).unwrap();
            let mut explicit = options.clone();
            explicit["name"] = builder.into();
            let expected: Value =
                serde_json::from_str(&evaluate(builder, &explicit.to_string()).unwrap()).unwrap();
            assert_eq!(implicit, expected, "{builder}");
            assert_eq!(
                implicit["geometry"][builder]["attributes"]["POSITION"],
                format!("{builder}_position")
            );
            assert_eq!(implicit["buffers"].as_object().unwrap().len(), 4);
            explicit["name"] = "custom".into();
            let named: Value =
                serde_json::from_str(&evaluate(builder, &explicit.to_string()).unwrap()).unwrap();
            assert_eq!(
                named["geometry"]["custom"]["attributes"]["POSITION"],
                "custom_position"
            );
            for name in [
                serde_json::json!(""),
                serde_json::json!("bad/name"),
                Value::Null,
            ] {
                explicit["name"] = name;
                assert!(
                    evaluate(builder, &explicit.to_string()).is_err(),
                    "{builder}"
                );
            }
        }
    }

    #[test]
    fn oversized_radii_fit_the_full_extent() {
        for (requested, canonical) in [
            (
                r#"{"size":[0.18,0.18,0.012],"radius":0.02}"#,
                r#"{"size":[0.18,0.18,0.012],"radius":0.006}"#,
            ),
            (
                r#"{"size":[1,0,2],"radius":1}"#,
                r#"{"size":[1,0,2],"radius":0}"#,
            ),
        ] {
            assert_eq!(
                evaluate("rounded_box", requested).unwrap(),
                evaluate("rounded_box", canonical).unwrap()
            );
        }
    }

    #[test]
    fn geometry_and_errors() {
        let fragment: Value = serde_json::from_str(
            &evaluate("rounded_box", r#"{"name":"cube","radius":0}"#).unwrap(),
        )
        .unwrap();
        assert_eq!(
            fragment["buffers"]["cube_position"]["data"]
                .as_array()
                .unwrap()
                .len(),
            24
        );
        for argument in [
            r#"{"radius":-1}"#,
            r#"{"segment":8}"#,
            r#"{"size":[1,2]}"#,
            "not JSON",
        ] {
            assert!(evaluate("rounded_box", argument).is_err(), "{argument}");
        }
        for argument in [r#"{"font":"font.ttf"}"#, r#"{"body":"x"}"#] {
            assert!(
                evaluate("text", argument)
                    .unwrap_err()
                    .to_string()
                    .contains("missing field")
            );
        }
    }
}
