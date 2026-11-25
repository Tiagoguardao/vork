// The rounded sum has the sign of its larger operand. The part of the small
// operand retained in that sum has the sign of the small operand (or is zero).
// Subtract their magnitudes so a relaxed driver preserves both rounding steps.
fn sameSignDifference(a: f32, b: f32) -> f32 {
  let magnitude = abs(a) - abs(b);
  return select(magnitude, -magnitude, bitcast<u32>(b) >= 0x80000000u);
}

// Compensated addition of a glyph bound and the two-word CPU antialiasing margin.
fn paddedBound(value: f32, high: f32, tail: f32) -> f32 {
  let swap = abs(value) < abs(high);
  let large = select(value, high, swap);
  let small = select(high, value, swap);
  let sum = large + small;
  let rounded = sameSignDifference(sum, large);
  let error = select(sameSignDifference(small, rounded), small, rounded == 0.0);
  let correction = error + tail;
  let result = sum + correction;
  // If adding the tail to the residual rounded onto a midpoint, its discarded
  // bits decide between the two adjacent bounds. Compare on the f32 lattice.
  if (correction == error && tail != 0.0 && error != 0.0 && sum == result) {
    let step = select(0xffffffffu, 1u, (result >= 0.0) == (error > 0.0));
    let neighbor = bitcast<f32>(bitcast<u32>(result) + step);
    if (abs(error) == abs(sameSignDifference(neighbor, result)) * 0.5 && (error > 0.0) == (tail > 0.0)) {
      return neighbor;
    }
  }
  return result;
}
