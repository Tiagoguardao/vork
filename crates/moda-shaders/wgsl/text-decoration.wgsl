// #include "working-color.wgsl"

struct VSOut { @builtin(position) pos: vec4f, @location(0) @interpolate(flat) clip: vec4f };
// #include "frame-uniform.wgsl"
// #include "rect-clip.wgsl"
@group(0) @binding(1) var<uniform> model: mat4x4<f32>;
@group(0) @binding(2) var<uniform> color: vec4f;
@group(0) @binding(3) var<uniform> params: vec4f; // x=zDepth, y=nodeOpacity
@group(0) @binding(4) var<uniform> pageClip: vec4f; // x, y, w, h in world space

@vertex fn vs(@location(0) pos: vec2f, @location(1) _uv: vec2f) -> VSOut {
  var o: VSOut;
  let world = model * vec4f(pos, 0.0, 1.0);
  o.pos = frameUniforms.viewProjection * world;
  o.pos.z = params.x; // enforce scene depth
  o.clip = projectRectClip(pageClip);
  return o;
}

@fragment fn fsClip(in: VSOut) -> @location(0) vec4f {
  if (outsideRectClip(in.pos.xy, in.clip)) {
    discard;
  }
  // Straight alpha output, apply node opacity to alpha only (match MSDF pipeline semantics)
  // NOTE: Blur not supported for solid shapes (bullets, decorations) - would require geometry expansion
  let finalAlpha = color.a * params.y;
  return vec4f(workingRgb(color.rgb), finalAlpha);
}
    