// working-color begin
fn workingRgb(c: vec3<f32>) -> vec3<f32> { return c; }
fn workingToSrgb(c: vec3<f32>) -> vec3<f32> { return c; }
// working-color end

struct ColorMatrix {
    rows: array<vec4<f32>, 4>,
    offset: vec4<f32>,
}
@group(0) @binding(0) var<uniform> matrix: ColorMatrix;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var picture: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let vertices = array(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
    return vec4f(vertices[i], 0, 1);
}

fn shade_fs(pos: vec4<f32>, matrix: ColorMatrix, inputSampler: sampler, picture: texture_2d<f32>) -> vec4<f32> {
    let pixel = textureLoad(picture, vec2i(pos.xy), 0);
    if (pixel.a == 0) { return vec4f(0); }
    let c = vec4f(workingToSrgb(pixel.rgb / pixel.a), pixel.a);
    let result = clamp(vec4f(dot(matrix.rows[0], c), dot(matrix.rows[1], c),
                             dot(matrix.rows[2], c), dot(matrix.rows[3], c)) + matrix.offset,
                       vec4f(0), vec4f(1));
    return vec4f(workingRgb(result.rgb) * result.a, result.a);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  return shade_fs(pos, matrix, inputSampler, picture);
}
