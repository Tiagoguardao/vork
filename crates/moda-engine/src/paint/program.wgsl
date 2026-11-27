// Same byte layout as SolidPathInstance; both kinds use the placement column.
struct PaintRecord {
    bounds: vec4f,
    opacity: vec4f,
    slot: u32, inputs: u32, clock: u32, depth: f32,
    clip: vec4f,
}
@group(1) @binding(0) var<storage,read> paths: array<PaintRecord>;
@group(1) @binding(1) var<storage,read> models: array<mat4x4f>;
struct MaskUniforms { enabled: u32, origin: vec2f, invSize: vec2f }
@group(2) @binding(0) var maskSampler: sampler;
@group(2) @binding(1) var maskTexture: texture_2d<f32>;
@group(2) @binding(2) var<uniform> mask: MaskUniforms;
struct PaintVertex {
    @builtin(position) pixel: vec4f,
    @location(0) local: vec2f,
    @location(1) strokeUV: vec2f,
    @location(2) world: vec2f,
    @location(3) @interpolate(flat) clip: vec4f,
    @location(4) @interpolate(flat) instance: u32,
    @location(5) @interpolate(flat) stroke: f32,
}
@vertex fn vertexMain(@location(0) local: vec2f, @location(1) stroke: f32,
    @location(2) strokeUV: vec2f, @builtin(instance_index) i: u32) -> PaintVertex {
    let path = paths[i];
    let world = pathPlacement(models[i], path.slot) * vec4f(local, 0, 1);
    var pixel = frameUniforms.viewProjection * world;
    pixel.z = path.depth + select(0., -0.000001490116119384765625, stroke > 0.5);
    return PaintVertex(pixel, local, strokeUV, world.xy, projectRectClip(path.clip), i, stroke);
}

@fragment fn fragmentMain(v: PaintVertex) -> @location(0) vec4f {
    let path = paths[v.instance];
    let stroke = v.stroke > 0.5;
    let value = evaluatePathPaint(v.local, path.bounds, v.strokeUV, stroke, path.clock, path.inputs);
    var coverage = select(path.opacity.x, path.opacity.y, v.stroke > 0.5);
    if mask.enabled != 0u {
        let p = (v.world - mask.origin) * mask.invSize;
        coverage *= textureSampleLevel(maskTexture, maskSampler, p, 0.).r;
        if any(p < vec2f(0)) || any(p > vec2f(1)) { discard; }
    }
    if coverage <= 0. || outsideRectClip(v.pixel.xy, v.clip) { discard; }
    let paint = checkedPathPaint(value, bitcast<u32>(path.opacity.z), stroke);
    if paint.a <= 0. { discard; }
    return paint * coverage;
}
