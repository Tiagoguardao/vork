
struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) @interpolate(flat) clip: vec4f };
// #include "rect-clip.wgsl"
@group(0) @binding(1) var<uniform> model: mat4x4<f32>;
@group(0) @binding(2) var<uniform> color: vec4f; // keep interface parity; used for alpha only if needed
@group(0) @binding(3) var<uniform> params: vec4f; // x=zDepth, y=nodeOpacity
@group(0) @binding(4) var<uniform> pageClip: vec4f; // x, y, w, h in world space
@group(1) @binding(0) var samp: sampler;
@group(1) @binding(1) var texImg: texture_2d<f32>;

@vertex fn vs(@location(0) pos: vec2f, @location(1) uv: vec2f) -> VSOut {
  var o: VSOut;
  let world = model * vec4f(pos, 0.0, 1.0);
  o.pos = frameUniforms.viewProjection * world;
  o.pos.z = params.x;
  o.uv = uv;
  o.clip = projectRectClip(pageClip);
  return o;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  if (outsideRectClip(in.pos.xy, in.clip)) {
    discard;
  }
  let rgba = textureSample(texImg, samp, in.uv);
  // Apply node opacity to alpha only (match MSDF semantics)
  let finalAlpha = rgba.a * params.y;
  return vec4f(rgba.rgb, finalAlpha);
}
    