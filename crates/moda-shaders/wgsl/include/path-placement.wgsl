fn pathPlacement(placement: mat4x4f, slot: u32) -> mat4x4f {
  var model = placement;
  if (slot >= frameUniforms.groupRange.x && slot < frameUniforms.groupRange.y) {
    var origin = model[3].xy - frameUniforms.groupAnchor;
    let grid = frameUniforms.groupGrid;
    if (grid.x > 0.0) { origin.x = round(origin.x / grid.x) * grid.x; }
    if (grid.y > 0.0) { origin.y = round(origin.y / grid.y) * grid.y; }
    model[3].x = select(paddedBound(origin.x, frameUniforms.groupHigh.x, frameUniforms.groupTail.x), frameUniforms.groupHigh.x, origin.x == 0.0);
    model[3].y = select(paddedBound(origin.y, frameUniforms.groupHigh.y, frameUniforms.groupTail.y), frameUniforms.groupHigh.y, origin.y == 0.0);
  }
  return model;
}
