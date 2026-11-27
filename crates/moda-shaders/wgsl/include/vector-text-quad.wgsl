// Preserve the glyph quad diagonal used to interpolate outline coordinates.
const quadCorners = array<vec2<f32>, 6>(
  vec2<f32>(0, 0), vec2<f32>(1, 0), vec2<f32>(0, 1),
  vec2<f32>(1, 0), vec2<f32>(1, 1), vec2<f32>(0, 1));

fn kineticGlyphLocal(
  pen: vec2<f32>, em: vec2<f32>, emBox: vec4<f32>, fontSizePx: f32,
  glyphLin: vec4<f32>, glyphOff: vec2<f32>,
) -> vec2<f32> {
  let baseLocal = pen + em * fontSizePx;
  if (glyphLin.x == 1.0 && glyphLin.y == 0.0 && glyphLin.z == 0.0 && glyphLin.w == 1.0 &&
      glyphOff.x == 0.0 && glyphOff.y == 0.0) {
    return baseLocal;
  }
  let center = pen + (emBox.xy + emBox.zw) * 0.5 * fontSizePx;
  let rel = baseLocal - center;
  return vec2<f32>(
    glyphLin.x * rel.x + glyphLin.y * rel.y + glyphOff.x,
    glyphLin.z * rel.x + glyphLin.w * rel.y + glyphOff.y,
  ) + center;
}

struct GlyphVertex {
  local: vec2<f32>,
  em: vec2<f32>,
};

fn glyphVertex(corner: vec2<f32>, pen: vec2<f32>, emBox: vec4<f32>, size: f32,
               lin: vec4<f32>, offset: vec2<f32>, aa: f32) -> GlyphVertex {
  if (aa == 0.0) {
    let em = mix(emBox.xy, emBox.zw, corner);
    return GlyphVertex(kineticGlyphLocal(pen, em, emBox, size, lin, offset), em);
  }
  // Transform the effect-expanded em box first. Only the varying is pulled
  // back to glyph space: the rasterized vertices have a fixed local margin.
  let center = (emBox.xy + emBox.zw) * 0.5;
  let halfSize = (emBox.zw - emBox.xy) * 0.5 * size;
  let extent = vec2<f32>(dot(abs(lin.xy), halfSize), dot(abs(lin.zw), halfSize)) + aa;
  let delta = (2.0 * corner - 1.0) * extent;
  // CPU rejects numerically collapsed maps. Normalize before the determinant
  // so a large uniform scale cannot overflow its products.
  let scale = max(max(abs(lin.x), abs(lin.y)), max(abs(lin.z), abs(lin.w)));
  let m = lin / scale;
  let bc = m.y * m.z;
  let det = fma(m.x, m.w, -bc) + fma(-m.y, m.z, bc);
  let p = (delta / scale) / size;
  let em = center + vec2<f32>(m.w * p.x - m.y * p.y, m.x * p.y - m.z * p.x) / det;
  return GlyphVertex(pen + center * size + offset + delta, em);
}
