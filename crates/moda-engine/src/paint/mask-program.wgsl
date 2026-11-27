struct MaskPaintVertex {
    @builtin(position) pixel: vec4f,
    @location(0) world: vec2f,
    @location(1) local: vec2f,
    @location(2) strokeUV: vec2f,
}
fn maskPaintVertex(input: VSIn, strokeUV: vec2f) -> MaskPaintVertex {
    let v = maskVertex(input);
    return MaskPaintVertex(v.position, v.worldPos, v.localPos, strokeUV);
}
@vertex fn vs_paint(input: VSIn) -> MaskPaintVertex {
    return maskPaintVertex(input, vec2f(0));
}
@vertex fn vs_stroke_paint(input: VSIn, @location(1) strokeUV: vec2f) -> MaskPaintVertex {
    return maskPaintVertex(input, strokeUV);
}
// Broadcast one lane's RGBA across its quad with core fragment operations.
// The absolute difference between a nonnegative value and zero is that value.
// Two exact 16-bit halves preserve every f32 bit, including nonfinite results.
fn quadSelectedHalf(halfWord: vec4u, selected: bool) -> vec4u {
    let source = select(vec4f(0), vec4f(halfWord), selected);
    return vec4u(abs(dpdyFine(abs(dpdxFine(source)))));
}
fn quadSelectedPaint(value: vec4f, selected: bool) -> vec4f {
    let bits = bitcast<vec4u>(value);
    let low = quadSelectedHalf(bits & vec4u(65535u), selected);
    let high = quadSelectedHalf(bits >> vec4u(16u), selected);
    return bitcast<vec4f>(low | (high << vec4u(16u)));
}

// The coverage atlas can be finer or coarser than the output. A derivative of a
// discontinuous paint needs the values on the actual neighboring picture pixels.
// Each atlas lane takes one turn as anchor. Its hardware quad evaluates the
// picture quad containing that anchor, including helpers beyond triangle edges.
fn sampleMaskPaint(v: MaskPaintVertex) -> vec4f {
    let p = globals.program;
    let stroke = p.w != 0u;
    if !pathPaintUsesDerivatives {
        return evaluatePathPaint(v.local, globals.gradBounds, v.strokeUV, stroke, p.y, p.x);
    }
    let gain = globals.pictureFromAtlas.xy;
    let offset = globals.pictureFromAtlas.zw;
    let atlas = floor(v.pixel.xy) + 0.5;
    let quad = floor(atlas * 0.5) * 2.0 + 0.5;
    let lane = vec2u(atlas - quad);
    let picture = fma(atlas, gain, offset);
    let coords = vec4f(v.local, v.strokeUV);
    let dx = dpdxFine(coords);
    let dy = dpdyFine(coords);
    var result = vec4f(0);
    for (var i = 0u; i < 4u; i++) {
        let anchor = vec2u(i & 1u, i >> 1u);
        let pixel = vec2i(floor(fma(quad + vec2f(anchor), gain, offset)));
        let parity = vec2u(pixel & vec2i(1));
        let pictureSample = vec2f(pixel - vec2i(parity)) + vec2f(lane) + 0.5;
        let shift = (pictureSample - picture) / gain;
        let at = fma(dy, vec4f(shift.y), fma(dx, vec4f(shift.x), coords));
        let value = evaluatePathPaint(at.xy, globals.gradBounds, at.zw, stroke, p.y, p.x);
        let anchorValue = quadSelectedPaint(value, all(lane == parity));
        if all(lane == anchor) { result = anchorValue; }
    }
    return result;
}

@fragment fn fs_paint(v: MaskPaintVertex) -> @location(0) vec4f {
    let p = globals.program;
    let stroke = p.w != 0u;
    let value = sampleMaskPaint(v);
    let ancestor = maskAncestor(v.world);
    if ancestor <= 0. || globals.color.a <= 0. { discard; }
    let paint = checkedPathPaint(value, p.z, stroke);
    if paint.a <= 0. { discard; }
    return maskCoverage(vec4f(workingToSrgb(paint.rgb / paint.a), globals.color.a * paint.a), ancestor);
}
