// Integrate the piecewise-constant resolved RGBA of this tile over a filter texel.
// Texels outside the tile are supplied by the other tiles (or transparent).
struct Grid {
    first_edge: vec2f,
    step: vec2f,
    first_cell: vec2f,
    _pad: vec2f,
}
@group(0) @binding(0) var<uniform> grid: Grid;
@group(0) @binding(2) var source: texture_2d<f32>;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let edge = grid.first_edge + (floor(position.xy) - grid.first_cell) * grid.step;
    let lo = max(edge, vec2f(0.0));
    let hi = min(edge + grid.step, vec2f(textureDimensions(source)));
    var sum = vec4f(0.0);
    for (var y = i32(floor(lo.y)); y < i32(ceil(hi.y)); y++) {
        let dy = min(hi.y, f32(y + 1)) - max(lo.y, f32(y));
        for (var x = i32(floor(lo.x)); x < i32(ceil(hi.x)); x++) {
            let dx = min(hi.x, f32(x + 1)) - max(lo.x, f32(x));
            sum += textureLoad(source, vec2i(x, y), 0) * (dx * dy);
        }
    }
    return sum / (grid.step.x * grid.step.y);
}
