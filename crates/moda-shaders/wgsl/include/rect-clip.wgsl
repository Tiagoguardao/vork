// Project the axis-aligned clip independently of the primitive. Interpolating
// world positions across a large triangle can lose whole pixels near the clip.
fn projectRectClip(bounds: vec4<f32>) -> vec4<f32> {
  let scale = frameUniforms.worldToPixel.xy;
  let offset = frameUniforms.worldToPixel.zw;
  return vec4<f32>(fma(bounds.xy, scale, offset), fma(bounds.xy + bounds.zw, scale, offset));
}

fn outsideRectClip(pixel: vec2<f32>, bounds: vec4<f32>) -> bool {
  return any(pixel < bounds.xy) || any(pixel > bounds.zw);
}
