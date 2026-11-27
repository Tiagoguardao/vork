struct SolidFrame {
  viewProjection: mat4x4<f32>,
  worldToPixel: vec4<f32>,
  groupRange: vec2<u32>,
  groupAnchor: vec2<f32>,
  groupHigh: vec2<f32>,
  groupTail: vec2<f32>,
  groupGrid: vec2<f32>,
}
@group(0) @binding(0) var<uniform> frameUniforms: SolidFrame;
