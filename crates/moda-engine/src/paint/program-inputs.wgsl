@group(3) @binding(0) var<storage,read> materialWords: array<u32>;
@group(3) @binding(1) var<storage,read_write> paintError: atomic<u32>;
fn vork_material_word(i: u32) -> u32 { return materialWords[i]; }

fn evaluatePathPaint(local: vec2f, bounds: vec4f, strokeUV: vec2f, stroke: bool, clock: u32, inputs: u32) -> vec4f {
    let extent = bounds.zw;
    let uv = select(vec2f(0.5), (local - bounds.xy) / select(vec2f(1), extent, extent != vec2f(0)), extent != vec2f(0));
    let along = select(strokeUV.x, strokeUV.x - 1., strokeUV.x > 1.);
    return PAINT_ENTRY(local, uv, select(vec2f(0), vec2f(along, strokeUV.y), stroke), bitcast<f32>(materialWords[clock]), inputs);
}

fn checkedPathPaint(paint: vec4f, site: u32, stroke: bool) -> vec4f {
    if any((bitcast<vec4u>(paint) & vec4u(0x7f800000u)) == vec4u(0x7f800000u)) {
        atomicMax(&paintError, 1u + 2u * site + u32(stroke));
        return vec4f(0);
    }
    return paint;
}
