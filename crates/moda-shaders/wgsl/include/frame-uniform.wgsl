struct ClipFrameUniforms {
  viewProjection: mat4x4<f32>,
  worldToPixel: vec4<f32>, // scale.xy, offset.xy
}

@group(0) @binding(0) var<uniform> frameUniforms: ClipFrameUniforms;
