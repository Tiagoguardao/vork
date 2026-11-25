//! The numeric gain pass for a scheduled sound block.
use super::*;

impl SoundBlock {
    /// Prepare the gain rows for this tile in layer order, carrying the last
    /// sampled key value across tiles, including through a zero weight.
    pub(crate) fn gains(
        &self,
        plan: &Plan,
        sound: &SoundPlan,
        first: i64,
        len: usize,
        samples: &mut SoundScratch,
        base: &[f32],
        tile: &mut GainTile,
    ) -> Result<()> {
        let GainTile {
            gains,
            weights,
            layer_weight,
            clock,
        } = tile;
        for (value, &node) in gains.constants.iter_mut().zip(&sound.keyed) {
            *value = Some(base[node]);
        }
        for block in &self.layers {
            if let Some(slot) = sound.weight_slots[block.layer] {
                weights.constants[slot] = Some(plan.layers[block.layer].source.weight);
            }
        }

        let samples = &mut samples.samples;
        let mut previous: Option<&BlockLayer> = None;
        for block in &self.layers {
            let layer = &plan.layers[block.layer];
            if !previous.is_some_and(|p| block.same_clock(p, plan)) {
                clock.fill(block, layer, self.first, first, len)?;
                previous = Some(block);
            }
            let present = layer.owner_node.is_none() || block.owner.is_some();
            let weight = match (present, sound.weight_slots[block.layer]) {
                (false, _) => Gain::Constant(0.),
                (true, None) => Gain::Constant(layer.source.weight),
                (true, Some(slot)) => match weights.row(slot, len) {
                    Gain::Constant(value) => Gain::Constant(value),
                    Gain::Keyed(values) => {
                        layer_weight[..len].copy_from_slice(values);
                        Gain::Keyed(&layer_weight[..len])
                    }
                },
            };
            let additive = matches!(layer.source.blend, v2::Blend::Add);
            for block_track in &block.tracks {
                let track = &layer.tracks[block_track.flat - plan.layer_starts[block.layer]];
                let Channel::Numeric(curve) = &track.channel else {
                    continue;
                };
                let Sample::Numeric(held) = &mut samples.values[block_track.flat] else {
                    unreachable!()
                };
                let (row, gain) = if let Some(node) = block_track.gain {
                    let slot = sound.gain_slots[node].unwrap();
                    (gains.row_mut(slot, len), true)
                } else {
                    let slot = sound.weight_slots[track.weight_target.unwrap()].unwrap();
                    (weights.row_mut(slot, len), false)
                };
                let weight = if block_track.absent {
                    Gain::Constant(0.)
                } else {
                    weight
                };
                let mut output = GainOutput {
                    row,
                    weight,
                    additive,
                    gain,
                    last: &mut held[0],
                };
                clock.apply(curve, block_track.upper, &mut output);
                // Image-time scheduling can also sample weight tracks. The retained value
                // is shared, and its general sampler should read its own interval again.
                samples.held[block_track.flat] = Held::No;
                let w = weight.at(len - 1);
                samples.weights[block_track.flat] = if w == 0. { 0. } else { w };
            }
        }
        Ok(())
    }
}

/// Reused storage for a small tile of the mix. Rows are contiguous in time.
pub(crate) struct GainTile {
    gains: GainRows,
    weights: GainRows,
    layer_weight: Vec<f32>,
    clock: GainClock,
}

impl GainTile {
    pub(crate) fn new(sound: &SoundPlan, stride: usize) -> Self {
        Self {
            gains: GainRows::new(sound.keyed.len(), stride),
            weights: GainRows::new(sound.driven, stride),
            layer_weight: vec![0.; stride],
            clock: GainClock {
                times: Vec::with_capacity(stride),
                runs: Vec::with_capacity(stride),
            },
        }
    }

    pub(crate) fn gain(&self, sound: &SoundPlan, node: usize, len: usize, base: f32) -> Gain<'_> {
        sound.gain_slots[node].map_or(Gain::Constant(base), |slot| self.gains.row(slot, len))
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Gain<'a> {
    Constant(f32),
    Keyed(&'a [f32]),
}

impl Gain<'_> {
    fn at(self, index: usize) -> f32 {
        match self {
            Self::Constant(value) => value,
            Self::Keyed(values) => values[index],
        }
    }
}

struct GainRows {
    stride: usize,
    constants: Vec<Option<f32>>,
    values: Vec<f32>,
}

impl GainRows {
    fn new(count: usize, stride: usize) -> Self {
        Self {
            stride,
            constants: vec![None; count],
            values: vec![0.; count * stride],
        }
    }

    fn row(&self, slot: usize, len: usize) -> Gain<'_> {
        match self.constants[slot] {
            Some(value) => Gain::Constant(value),
            None => Gain::Keyed(&self.values[slot * self.stride..slot * self.stride + len]),
        }
    }

    fn row_mut(&mut self, slot: usize, len: usize) -> GainRow<'_> {
        GainRow {
            constant: &mut self.constants[slot],
            values: &mut self.values[slot * self.stride..slot * self.stride + len],
        }
    }
}

struct GainRow<'a> {
    constant: &'a mut Option<f32>,
    values: &'a mut [f32],
}

impl GainRow<'_> {
    fn samples(&mut self) -> &mut [f32] {
        if let Some(value) = self.constant.take() {
            self.values.fill(value);
        }
        self.values
    }
}

impl BlockLayer {
    fn same_clock(&self, other: &Self, plan: &Plan) -> bool {
        let a = &plan.layers[self.layer];
        let b = &plan.layers[other.layer];
        let loop_span = |layer: &CompiledLayer| {
            layer
                .span
                .filter(|&(a, b)| layer.source.looping && b > a)
                .map(|(a, b)| (a.to_bits(), b.to_bits()))
        };
        loop_span(a) == loop_span(b)
            && match (&self.owner, &other.owner) {
                (Some(Carried::Along(a)), Some(Carried::Along(b))) => a == b,
                (Some(Carried::One(a)), Some(Carried::One(b))) => a.to_bits() == b.to_bits(),
                (None, None) if a.owner_node.is_none() && b.owner_node.is_none() => {
                    a.source.offset.to_bits() == b.source.offset.to_bits()
                        && a.source.speed.to_bits() == b.source.speed.to_bits()
                }
                (None, None) => a.owner_node.is_some() && b.owner_node.is_some(),
                _ => false,
            }
    }
}

struct GainClock {
    times: Vec<f32>,
    /// Monotone numeric clock runs, split at wraps. Empty for a curved clock
    /// or one whose numeric wrap cannot be resolved with machine arithmetic.
    runs: Vec<std::ops::Range<usize>>,
}

impl GainClock {
    fn fill(
        &mut self,
        block: &BlockLayer,
        layer: &CompiledLayer,
        origin: i64,
        first: i64,
        len: usize,
    ) -> Result<()> {
        self.times.clear();
        self.runs.clear();
        let straight = match &block.owner {
            Some(Carried::Along(line)) => {
                let from = first - origin;
                let until = from + len as i64;
                let straight = line.is_line();
                let initial = line.numeric(from)?;
                if straight && initial.to_bits() == line.numeric(until - 1)?.to_bits() {
                    self.times.resize(len, initial);
                } else {
                    line.numeric_run(from..until, &mut self.times)?;
                }
                straight
            }
            Some(Carried::One(time)) => {
                self.times.resize(len, *time);
                true
            }
            None if layer.owner_node.is_none() => {
                self.times.extend((first..first + len as i64).map(|i| {
                    let time = (i as f64 / SAMPLE_RATE as f64) as f32;
                    layer.source.offset + time * layer.source.speed
                }));
                true
            }
            None => return Ok(()),
        };

        let looping = layer.span.filter(|&(a, b)| layer.source.looping && b > a);
        let finite = self.times[0].is_finite() && self.times[len - 1].is_finite();
        match looping {
            None if straight && finite => self.runs.push(0..len),
            None => {}
            Some((a, b)) if straight && self.wrap(a, b - a) => {}
            Some(_) => {
                self.runs.clear();
                self.times
                    .iter_mut()
                    .for_each(|time| *time = layer.looped(*time));
            }
        }
        Ok(())
    }

    fn wrap(&mut self, start: f32, period: f32) -> bool {
        let delta = |time: f32| f64::from(time - start);
        let period = f64::from(period);
        let last = self.times.len() - 1;
        let first_delta = delta(self.times[0]);
        let last_delta = delta(self.times[last]);
        // Within this range the integer multiple of the f32 period is exact
        // in f64. The rounded f32 subtraction precedes the remainder, as in
        // CompiledLayer::looped. Subtracting the multiple gives the same
        // remainder as fmod, including the correction for a negative input.
        if !period.is_finite()
            || first_delta.abs().max(last_delta.abs()) > period * f64::from(1 << 24)
            || !first_delta.is_finite()
            || !last_delta.is_finite()
        {
            return false;
        }
        let rising = first_delta <= last_delta;
        let mut first = 0;
        while first <= last {
            let x = delta(self.times[first]);
            let cycle = (x / period).floor();
            // Resolve a quotient rounded onto the next integer against the
            // exact boundary products.
            let cycle = if x < period * cycle {
                cycle - 1.
            } else {
                cycle
            };
            let lower = period * cycle;
            let upper = period * (cycle + 1.);
            let in_cycle = |time: &f32| {
                if rising {
                    delta(*time) < upper
                } else {
                    delta(*time) >= lower
                }
            };
            let count = if in_cycle(&self.times[last]) {
                last + 1 - first
            } else {
                self.times[first..].partition_point(in_cycle)
            };
            let end = first + count;
            for time in &mut self.times[first..end] {
                let x = delta(*time);
                let remainder = if lower == 0. {
                    x as f32
                } else if x == lower {
                    0_f32.copysign(x as f32)
                } else {
                    (x - lower) as f32
                };
                *time = start + remainder;
            }
            self.runs.push(first..end);
            first = end;
        }
        true
    }

    fn apply(&self, curve: &Curve, upper: Option<usize>, output: &mut GainOutput<'_>) {
        if let Gain::Constant(weight) = output.weight
            && weight == 0.
        {
            if output.gain {
                output.blend_constant(0..output.row.values.len(), *output.last, 0.);
            }
            return;
        }
        let fixed = upper.or_else(|| (curve.times.len() == 1).then_some(0));
        if let Some(upper) = fixed {
            self.apply_run(curve, upper, output, 0..self.times.len());
        } else if self.runs.is_empty() {
            for i in 0..self.times.len() {
                let upper = curve.times.partition_point(|&key| key <= self.times[i]);
                self.apply_run(curve, upper, output, i..i + 1);
            }
        } else {
            for run in &self.runs {
                let rising = self.times[run.start] <= self.times[run.end - 1];
                let mut first = run.start;
                while first < run.end {
                    let upper = curve.times.partition_point(|&key| key <= self.times[first]);
                    let in_interval = |time: &f32| {
                        if rising {
                            curve.times.get(upper).is_none_or(|key| time < key)
                        } else {
                            upper == 0 || *time >= curve.times[upper - 1]
                        }
                    };
                    let end = if in_interval(&self.times[run.end - 1]) {
                        run.end
                    } else {
                        first + self.times[first..run.end].partition_point(in_interval)
                    };
                    self.apply_run(curve, upper, output, first..end);
                    first = end;
                }
            }
        }
    }

    fn apply_run(
        &self,
        curve: &Curve,
        upper: usize,
        output: &mut GainOutput<'_>,
        range: std::ops::Range<usize>,
    ) {
        let times = &self.times[range.clone()];
        let (a, b) = (upper.saturating_sub(1), upper.min(curve.times.len() - 1));
        let start = curve.times[a];
        let dt = curve.times[b] - start;
        if curve.interpolation == Interpolation::Step {
            output.hold(times, range, curve.values[a * curve.width]);
        } else if curve.interpolation == Interpolation::Linear {
            let (va, vb) = (curve.values[a * curve.width], curve.values[b * curve.width]);
            if a == b {
                output.hold(times, range, linear_key(va, vb, 0.));
            } else {
                // A numeric interval was selected from these rounded sample
                // times. Its interpolation parameter is already in [0, 1].
                output.sample(times, range, |time| linear_key(va, vb, (time - start) / dt));
            }
        } else {
            let key = |i, part| curve.values[(i * 3 + part) * curve.width];
            let [va, ta, vb, tb] = [key(a, 1), key(a, 2), key(b, 1), key(b, 0)];
            let cubic = |time| {
                let u = key_fraction(time, start, dt);
                let [h00, h10, h01, h11] = hermite_basis(u, 1., 2., 3.);
                h00 * va + h10 * dt * ta + h01 * vb + h11 * dt * tb
            };
            if a == b {
                output.hold(times, range, cubic(start));
            } else {
                output.sample(times, range, cubic);
            }
        }
    }
}

struct GainOutput<'a> {
    row: GainRow<'a>,
    weight: Gain<'a>,
    additive: bool,
    gain: bool,
    last: &'a mut f32,
}

impl GainOutput<'_> {
    fn blend_constant(&mut self, range: std::ops::Range<usize>, value: f32, weight: f32) {
        if range.len() == self.row.values.len()
            && let Some(base) = self.row.constant.as_mut()
        {
            *base = blend_scalar(*base, value, weight, self.additive);
        } else {
            for base in &mut self.row.samples()[range] {
                *base = blend_scalar(*base, value, weight, self.additive);
            }
        }
    }

    fn hold(&mut self, times: &[f32], range: std::ops::Range<usize>, value: f32) {
        if let Gain::Constant(weight) = self.weight {
            *self.last = value;
            self.blend_constant(range, value, weight);
        } else {
            self.sample(times, range, |_| value);
        }
    }

    #[inline]
    fn sample(
        &mut self,
        times: &[f32],
        range: std::ops::Range<usize>,
        sample: impl Fn(f32) -> f32,
    ) {
        let first = range.start;
        let output = &mut self.row.samples()[range];
        match self.weight {
            Gain::Constant(weight) => {
                // Zero constant weights are handled before interval selection.
                for (base, &time) in output.iter_mut().zip(times) {
                    *base = blend_scalar(*base, sample(time), weight, self.additive);
                }
                *self.last = sample(*times.last().unwrap());
            }
            Gain::Keyed(weights) => {
                for ((base, &time), &weight) in output.iter_mut().zip(times).zip(&weights[first..])
                {
                    if weight != 0. {
                        *self.last = sample(time);
                        *base = blend_scalar(*base, *self.last, weight, self.additive);
                    } else if self.gain {
                        *base = blend_scalar(*base, *self.last, 0., self.additive);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_wrap_runs_preserve_the_float_remainder() {
        for (start, period) in [
            (0., 2.),
            (-0., 0.0031),
            (0.0005, 0.0031),
            (-3., 5.123),
            (1., 1e-6),
            (0., f32::MIN_POSITIVE),
        ] {
            let mut raw: Vec<_> = (-2000..=2000)
                .map(|n| start + n as f32 * period / 32.)
                .collect();
            for n in [-16_777_215., -125., -1., 0., 1., 125., 16_777_215.] {
                let boundary = start + n * period;
                raw.extend([boundary.next_down(), boundary, boundary.next_up()]);
            }
            raw.retain(|time| f64::from((*time - start).abs()) <= f64::from(period) * 16_777_216.);
            raw.sort_by(f32::total_cmp);
            for _ in 0..2 {
                let mut clock = GainClock {
                    times: raw.clone(),
                    runs: Vec::new(),
                };
                assert!(clock.wrap(start, period));
                for (&time, &actual) in raw.iter().zip(&clock.times) {
                    let expected = start + (time - start).rem_euclid(period);
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{start} + ({time} - {start}) mod {period}"
                    );
                }
                assert_eq!(clock.runs.iter().flat_map(Clone::clone).count(), raw.len());
                raw.reverse();
            }
        }
    }
}
