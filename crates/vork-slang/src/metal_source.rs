//! What Slang's MSL needs before Apple's compiler sees it: every local
//! initialised, its floating-point directives, and rejection of Slang's known
//! unbound acceleration-structure lowering failure (this also covers authored
//! graphs outside our corpus).
use anyhow::{Result, bail};
use std::ffi::OsStr;

/// Finish the code Slang emitted for `args`. Every Metal module vork compiles,
/// shipped or at runtime, passes through here; other targets are returned as is.
///
/// Locals are initialised first (see [`initialize_locals`]). Then, because Slang
/// emits source and its fp-mode flag cannot configure the later Metal compile in
/// a passthrough API, `-fp-mode precise` becomes standard source directives,
/// independent of shader names.
///
/// This file is hashed into the compiled-shader identity (`VORK_SHADER_ID`,
/// vork-render's build.rs), so a change here never meets an older cache entry.
pub fn finalize<S: AsRef<OsStr>>(args: &[S], code: Vec<u8>) -> Result<Vec<u8>> {
    let pair = |names: &[&str], value: &str| args.windows(2)
        .any(|pair| names.iter().any(|name| pair[0].as_ref() == *name) && pair[1].as_ref() == value);
    if !pair(&["-target"], "metal") { return Ok(code); }
    let mut source = String::new();
    if pair(&["-fp-mode", "-floating-point-mode"], "precise") {
        source.push_str("#pragma METAL fp math_mode(safe)\n#pragma METAL fp contract(off)\n");
    }
    source.push_str(&initialize_locals(std::str::from_utf8(&code)?));
    // No function reaches Apple's compiler marked noinline. Slang's outlining
    // bounds its own optimization graph; the hint is not semantics. With it,
    // scene-linked path stages never returned from a dispatch on M5 Max
    // (#2580, bisected there) and the GPU watchdog can freeze the machine.
    // The 3D revert to 0.6.1 (#2708) dropped this; restored here.
    let source = source.replace("__attribute__((noinline))\n", "").replace("__attribute__((noinline))", "");
    Ok(source.into_bytes())
}

/// Tokens with their byte offsets; comments, strings and preprocessor lines are skipped.
fn tokens(source: &str) -> Vec<(usize, &str)> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                while i < bytes.len() && bytes[i] != b'\n' { i += 1; }
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                i += 2;
                while i + 1 < bytes.len() && &bytes[i..i + 2] != b"*/" { i += 1; }
                i = (i + 2).min(bytes.len());
            }
            b'"' | b'\'' => {
                let quote = bytes[i]; i += 1;
                while i < bytes.len() {
                    let c = bytes[i]; i += 1;
                    if c == quote { break; }
                    if c == b'\\' { i = (i + 1).min(bytes.len()); }
                }
            }
            b'#' => {
                while i < bytes.len() && bytes[i] != b'\n' { i += 1; }
            }
            c if c.is_ascii_whitespace() => { i += 1; }
            c if c.is_ascii_alphanumeric() || c == b'_' => {
                let start = i; i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') { i += 1; }
                tokens.push((start, &source[start..i]));
            }
            _ => {
                // Emitted MSL punctuation is ASCII; skip any UTF-8 codepoint in
                // diagnostics/comments rather than slicing through its bytes.
                let end = i + source[i..].chars().next().unwrap().len_utf8();
                tokens.push((i, &source[i..end])); i = end;
            }
        }
    }
    tokens
}

const INITIALIZER: &str = " = {}";

/// Give every local declared without an initialiser a value: `T x;` becomes
/// `T x = {};`.
///
/// Slang's `-zero-initialize` assigns such locals member by member, which leaves
/// bytes no member owns unwritten: a `float3` occupies sixteen bytes in MSL and
/// its fourth lane is never stored. With stale bytes in that lane of a local
/// `array<float3, 3>` (`hit_shadow_offset`'s normals), Apple's compiler produced
/// a path kernel whose picture was dark; a zero there gave the right one.
/// Value-initialising the whole object leaves no such bytes, for arrays, structs
/// and scalars alike.
///
/// Only block-scope declarations are touched. Parameters, globals, struct
/// members, pointers, references, declarations that already have an
/// initialiser, declarations directly under a `switch` (a later label may not
/// jump past an initialiser) and Metal's own object types (textures, samplers,
/// ray-tracing objects, anything namespace-qualified) are left as Slang wrote them.
pub fn initialize_locals(source: &str) -> String {
    #[derive(Clone, Copy, PartialEq)]
    enum Scope { File, Body, Switch, Other }
    let tokens = tokens(source);
    let at = |i: usize| tokens.get(i).map_or("", |token| token.1);
    let mut inserts = Vec::new();
    let mut scopes = vec![Scope::File];
    // The token before each open parenthesis, and before the last one closed.
    let (mut parens, mut closed) = (Vec::new(), "");
    let mut statement = false;
    for i in 0..tokens.len() {
        if statement && *scopes.last().unwrap() == Scope::Body { declaration(&tokens, i, &mut inserts); }
        let before = if i == 0 { "" } else { at(i - 1) };
        statement = false;
        match at(i) {
            "{" => {
                let scope = match *scopes.last().unwrap() {
                    Scope::File if before == ")" => Scope::Body,
                    Scope::File if i >= 2 && at(i - 2) == "namespace" => Scope::File,
                    Scope::Body | Scope::Switch if before == ")" && closed == "switch" => Scope::Switch,
                    Scope::Body | Scope::Switch if matches!(before, ")" | "else" | "do" | "{" | "}" | ";" | ":") => Scope::Body,
                    _ => Scope::Other,
                };
                scopes.push(scope);
                statement = true;
            }
            "}" => {
                if scopes.len() > 1 { scopes.pop(); }
                statement = true;
            }
            ";" => statement = parens.is_empty(),
            "(" => {
                parens.push(before);
                // A `for` header opens with a declaration of its own.
                statement = before == "for";
            }
            ")" => closed = parens.pop().unwrap_or(""),
            _ => {}
        }
    }
    let mut out = String::with_capacity(source.len() + inserts.len() * INITIALIZER.len());
    let mut from = 0;
    for offset in inserts {
        out.push_str(&source[from..offset]);
        out.push_str(INITIALIZER);
        from = offset;
    }
    out.push_str(&source[from..]);
    out
}

/// If a declaration statement starts at token `i`, record where its
/// uninitialised declarators need an initialiser.
fn declaration(tokens: &[(usize, &str)], mut i: usize, inserts: &mut Vec<usize>) {
    const NOT_A_TYPE: &[&str] = &["return", "break", "continue", "else", "do", "goto", "case", "default", "if", "for",
        "while", "switch", "typedef", "using", "struct", "class", "union", "enum", "static", "extern", "new", "delete",
        "sizeof", "template", "namespace"];
    // Metal's object types construct themselves, some only explicitly.
    const OBJECTS: &[&str] = &["texture", "depth", "intersect", "acceleration_structure", "visible_function_table",
        "patch_control_point"];
    const QUALIFIERS: &[&str] = &["thread", "const", "volatile", "constexpr", "device", "constant"];
    let at = |i: usize| tokens.get(i).map_or("", |token| token.1);
    let name = |token: &str| token.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && !NOT_A_TYPE.contains(&token) && !QUALIFIERS.contains(&token);
    while QUALIFIERS.contains(&at(i)) { i += 1; }
    let head = at(i);
    if !name(head) || matches!(head, "sampler" | "ray") { return; }
    i += 1;
    if at(i) == "<" && OBJECTS.iter().any(|object| head.starts_with(object)) { return; }
    if at(i) == "<" {
        let mut depth = 0usize;
        loop {
            match at(i) {
                "<" => depth += 1,
                ">" => depth -= 1,
                "" | ";" | "{" | "}" => return,
                _ => {}
            }
            i += 1;
            if depth == 0 { break; }
        }
    }
    while QUALIFIERS.contains(&at(i)) { i += 1; }
    loop {
        if !name(at(i)) { return; }
        i += 1;
        while at(i) == "[" {
            while !matches!(at(i), "]" | "" | ";") { i += 1; }
            if at(i) != "]" { return; }
            i += 1;
        }
        // Anything else is an initialiser, a call, a range or not a declaration.
        if !matches!(at(i), ";" | ",") { return; }
        let last = tokens[i - 1];
        inserts.push(last.0 + last.1.len());
        if at(i) == ";" { return; }
        i += 1;
    }
}

pub fn validate(source: &str) -> Result<()> {
    let tokens: Vec<_> = tokens(source).into_iter().map(|token| token.1).collect();
    let (mut braces, mut parens) = (0usize, 0usize);
    for (at, token) in tokens.iter().enumerate() {
        match *token {
            "{" => braces += 1,
            "}" => braces = braces.saturating_sub(1),
            "(" => parens += 1,
            ")" => parens = parens.saturating_sub(1),
            "acceleration_structure" if braces == 0 && parens == 0 && tokens.get(at + 1) == Some(&"<") => {
                let mut angle = 1usize;
                let mut end = at + 2;
                while end < tokens.len() && angle > 0 {
                    match tokens[end] { "<" => angle += 1, ">" => angle -= 1, _ => {} }
                    end += 1;
                }
                while matches!(tokens.get(end).copied(), Some("*" | "&" | "const" | "volatile" | "thread" | "device" | "constant")) { end += 1; }
                if let (Some(name), Some(next)) = (tokens.get(end), tokens.get(end + 1)) {
                    if matches!(*next, ";" | "=" | "[" | ",") {
                        bail!("Metal shader compilation refused: Slang emitted an unbound file-scope acceleration structure `{name}`. This ray-query helper graph is not supported by the pinned Slang Metal compiler; simplify nested trace/coverage calls or render on Vulkan.");
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialises_every_uninitialised_local_and_nothing_else() {
        let source = r#"#include <metal_stdlib>
#pragma clang diagnostic ignored "-Wunused-variable"
using namespace metal;
struct Hit_0
{
    float3 position_0;
    array<float3, int(3)> normals_0;
};
float3 global_0;
namespace vork { float scale; }
static const array<float, int(2)> table_0 = { 1.0, 2.0 };
float3 declared_0(float3 p_0, const Hit_0 thread* hit_0);
float4 shadow_0(array<float3, int(3)> p_0, const Hit_0 thread* hit_0, texture2d<float, access::sample> image_0)
{
    thread array<float3, int(3)> n_0;
    array<float4, int(12)> rows_0;
    matrix<float,int(4),int(4)>  model_0;
    Hit_0 copy_0;
    const Hit_0 thread* pointer_0;
    Hit_0 thread& reference_0 = copy_0;
    float a_0, b_0;
    float c_0[2];
    float set_0 = 1.0f;
    float3 braced_0 { 0.0 };
    Hit_0 list_0 = { float3(0.0), { } };
    uint i_0; // float not_code;
    /* bool not_code; */
    const char* text_0 = "float inside; { ";
    texture2d<float, access::sample> view_0;
    sampler sampler_0;
    raytracing::intersection_query<raytracing::instancing> query_0;
    intersection_query<instancing, triangle_data> query_1;
    depth_0 user_0;
    for(int j_0; j_0 < 3; j_0++)
    {
        bool inner_0;
        n_0[j_0] = p_0[j_0] * float3(a_0 * b_0);
    }
    for(;;)
    {
        break;
    }
    if(a_0 < b_0)
        return float4(0.0);
    else
    {
        float else_0;
    }
    switch(i_0)
    {
    case 0U:
        float jumped_0;
        {
            float nested_0;
        }
        break;
    default:
        break;
    }
    struct Local_0 { float member_0; } local_0;
    do
    {
        uint k_0;
    }
    while(false);
    a_0 * b_0;
    return float4(n_0[0], 1.0);
}
[[kernel]] void main_0(uint3 id_0 [[thread_position_in_grid]])
{
    float4 last_0;
}
"#;
        let initialised = initialize_locals(source);
        let lines: Vec<_> = source.lines().zip(initialised.lines())
            .filter(|(before, after)| before != after).map(|(_, after)| after.trim()).collect();
        assert_eq!(source.lines().count(), initialised.lines().count());
        assert_eq!(lines, [
            "thread array<float3, int(3)> n_0 = {};",
            "array<float4, int(12)> rows_0 = {};",
            "matrix<float,int(4),int(4)>  model_0 = {};",
            "Hit_0 copy_0 = {};",
            "float a_0 = {}, b_0 = {};",
            "float c_0[2] = {};",
            "uint i_0 = {}; // float not_code;",
            "depth_0 user_0 = {};",
            "for(int j_0 = {}; j_0 < 3; j_0++)",
            "bool inner_0 = {};",
            "float else_0 = {};",
            "float nested_0 = {};",
            "uint k_0 = {};",
            "float4 last_0 = {};",
        ]);
        // Already initialised source is left alone.
        assert_eq!(initialize_locals(&initialised), initialised);
    }

    #[test]
    fn finalises_metal_only_and_keeps_the_precise_contract() {
        const SOURCE: &str = "void f()\n{\n    float3 x;\n}\n";
        const INITIALISED: &str = "void f()\n{\n    float3 x = {};\n}\n";
        const SAFE: &str = "#pragma METAL fp math_mode(safe)\n#pragma METAL fp contract(off)\n";
        let finalise = |args: &[&str]| String::from_utf8(finalize(args, SOURCE.into()).unwrap()).unwrap();
        assert_eq!(finalise(&["-target", "metal", "-O2"]), INITIALISED);
        assert_eq!(finalise(&["-target", "metal", "-fp-mode", "precise"]), format!("{SAFE}{INITIALISED}"));
        assert_eq!(finalise(&["-target", "metal", "-floating-point-mode", "precise"]), format!("{SAFE}{INITIALISED}"));
        assert_eq!(finalise(&["-target", "spirv", "-fp-mode", "precise"]), SOURCE);
        assert_eq!(finalize(&["-target", "spirv"], vec![0xff, 0x00]).unwrap(), [0xff, 0x00]);
    }
    const AS: &str = "metal::raytracing::acceleration_structure<metal::raytracing::instancing>";

    #[test]
    fn distinguishes_resource_globals_from_context_fields_and_parameters() {
        let good = format!("#include <metal_stdlib>\nstruct KernelContext {{ {AS} ray_scene_0; }}; void helper({AS} ray_scene_0); void entry({AS} scene) {{}} // {{\n");
        validate(&good).unwrap();
        let bad = format!("{good}\n{AS} ray_scene_0;");
        let message = validate(&bad).unwrap_err().to_string();
        assert!(message.contains("unbound file-scope acceleration structure `ray_scene_0`"));
        assert!(message.contains("Vulkan"));
    }

    #[test]
    fn handles_multiline_declarations_comments_and_return_types() {
        validate(&format!("{AS} result(); /* {{ */ const char* s=\"{{}}\";")).unwrap();
        assert!(validate(&format!("{AS}\n/* comment */ illegal;")).is_err());
    }
}
