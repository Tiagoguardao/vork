//! Seek and walk a repeatable style list without expanding its common period.
use super::dash::{
    BASE_DASH_STROKE_WIDTH, normalize_dash_pattern, scale_dash_pattern, seed_dash_phase,
};
use super::moda_engine_path_types::LineCap;
use super::tessellator::MAX_DASH_STEPS;
use moda_scene::dash::{DashList, SampledDash};
use num_bigint::{BigInt, BigUint, Sign};
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(super) enum DashPattern<'a> {
    Static(&'a [f64]),
    Sampled(&'a SampledDash),
}

#[derive(Clone, Copy)]
pub(super) enum ContourDash<'a> {
    Static(&'a [f64]),
    Compact(&'a CompactDash),
}

pub(super) struct CompactDash {
    sources: Vec<Source>,
    period: Option<u128>,
    rounded: Option<Vec<f64>>,
    unit: i32,
    capped: bool,
}

impl CompactDash {
    pub(super) fn new(
        dash: &SampledDash,
        width: f64,
        cap: LineCap,
        corners: bool,
    ) -> Result<Self, String> {
        // This is the input boundary for both sampled styles and raw Moda
        // strokes. Reject invalid numbers before interpreting them as dyadics.
        if !dash.scale.is_finite() || dash.scale < 0. {
            return Err("sampled stroke-dasharray scale must be finite and nonnegative".into());
        }
        let authored = validate_list(&dash.list)?;
        let capped = matches!(cap, LineCap::Round | LineCap::Square);
        let period = dash
            .list
            .period()
            .filter(|&count| count > 0)
            .and_then(|count| count.checked_mul(1 + count % 2));
        let limit = if corners {
            MAX_DASH_STEPS / 3
        } else {
            authored.saturating_mul(2).min(MAX_DASH_STEPS / 3)
        };

        // Resolve the mix once, keeping exact coefficients for both forms.
        let mut sources = Vec::new();
        Source::collect(&dash.list, &1u8.into(), 0, &mut sources);

        // Keep normalized f32 lengths and seam snapping where the period
        // is bounded by the inputs. Round authored runs before calibration.
        if let Some(count) = period.filter(|&count| count <= limit as u128) {
            let unit = sources
                .iter()
                .filter_map(|source| source.exponent)
                .min()
                .unwrap_or(0);
            let values: Option<Vec<_>> = (0..count as usize)
                .map(|index| {
                    let exact: BigUint = sources
                        .iter()
                        .map(|source| {
                            source.exact(source.values[index % source.values.len()], unit)
                        })
                        .sum();
                    let length = to_distance(&exact, unit);
                    let rounded = length as f32;
                    let stored = f64::from(rounded) * dash.scale * BASE_DASH_STROKE_WIDTH / width;
                    // Raw Moda lengths can be outside the f32 range.
                    (stored.is_finite() && (exact == BigUint::ZERO || rounded > 0.))
                        .then_some(stored)
                })
                .collect();
            if let Some(values) = values {
                let values = moda_scene::paint::zero_dashes_as_dots(
                    Some(&values),
                    capped.then_some("round"),
                    width,
                )
                .unwrap();
                let values = normalize_dash_pattern(scale_dash_pattern(&values, width));
                if values.iter().sum::<f64>().is_finite() {
                    return Ok(Self {
                        sources: Vec::new(),
                        period,
                        rounded: Some(values),
                        unit: 0,
                        capped: false,
                    });
                }
            }
        }

        let (scale, exponent) = dyadic(dash.scale);
        if scale == 0 {
            sources.clear();
        } else {
            for source in &mut sources {
                source.weight *= scale;
                source.weight_exponent += exponent;
                source.exponent = source.exponent.map(|unit| unit + exponent);
            }
        }
        let unit = sources
            .iter()
            .filter_map(|source| source.exponent)
            .chain(capped.then(|| dyadic(0.01).1))
            .min()
            .unwrap_or(0);
        Ok(Self {
            sources,
            period,
            rounded: None,
            unit,
            capped,
        })
    }

    pub(super) fn corners(&self) -> Option<&[f64]> {
        self.rounded.as_deref()
    }

    pub(super) fn is_solid(&self) -> bool {
        self.rounded.as_ref().map_or_else(
            || self.sources.iter().all(|source| source.exponent.is_none()),
            |values| values.iter().all(|&value| value == 0.),
        )
    }

    pub(super) fn on_fraction(&self) -> f64 {
        let (on, total) = if let Some(values) = &self.rounded {
            (values.iter().step_by(2).sum(), values.iter().sum())
        } else {
            let mean = |even| {
                self.sources
                    .iter()
                    .map(|source| source.mean(self.unit, even))
                    .sum::<f64>()
            };
            (mean(true) * 0.5, mean(false))
        };
        if total > 0. && total.is_finite() {
            (on / total).clamp(0., 1.)
        } else {
            1.
        }
    }

    pub(super) fn phase(&self, offset: f64) -> Result<Phase<'_>, String> {
        if !offset.is_finite() {
            return Err(
                "stroke-dashoffset must be finite after pathLength calibration".to_string(),
            );
        }
        if let Some(values) = &self.rounded {
            let (index, is_on, remaining) =
                seed_dash_phase(values, offset).expect("a solid dash does not need a phase");
            return Ok(Phase {
                is_on,
                remaining,
                cursor: Cursor::Rounded { values, index },
                #[cfg(test)]
                index: index.into(),
            });
        }

        // A binary float is an integer times a power of two. Accumulate those
        // integers in a common unit, so neither a large skipped run nor a
        // large run index can round away a small phase. For a negative offset,
        // count backwards directly from the seam.
        let mut reverse = offset < 0.;
        let (distance, offset_exponent) = dyadic(offset.abs());
        let unit = self.unit.min(offset_exponent);
        if distance == 0 {
            let cursor = FactoredCursor::new(self, &BigInt::ZERO);
            let [on, gap] = cursor.pair_lengths(unit, 0);
            if on != BigUint::ZERO || gap != BigUint::ZERO {
                return Ok(finish_phase(
                    cursor,
                    BigInt::ZERO,
                    on,
                    gap,
                    BigUint::ZERO,
                    unit,
                    false,
                ));
            }
        }
        let mut distance = BigUint::from(distance) << (offset_exponent - unit) as usize;
        let build_prefixes = |reverse| {
            self.sources
                .iter()
                .map(|source| source.prefixes(unit, reverse))
                .collect::<Vec<_>>()
        };
        let mut prefixes = build_prefixes(reverse);
        if let Some(period) = self.period {
            distance %= prefix_at(&prefixes, &BigUint::from(period));
            if reverse && distance == BigUint::ZERO {
                reverse = false;
                prefixes = build_prefixes(false);
            }
        }

        // Search pair boundaries. A capped zero dash borrows from its own
        // gap, so it cannot move any of these boundaries.
        let before = |pair: &BigUint| {
            let end = prefix_at(&prefixes, &(pair << 1usize));
            if reverse {
                end < distance
            } else {
                end <= distance
            }
        };
        let (mut lo, mut hi) = (BigUint::ZERO, BigUint::from(1u8));
        while before(&hi) {
            lo.clone_from(&hi);
            hi <<= 1usize;
        }
        while &hi - &lo > BigUint::from(1u8) {
            let mid = (&lo + &hi) >> 1usize;
            if before(&mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let pair = &lo << 1usize;
        let into_pair = distance - prefix_at(&prefixes, &pair);
        let start = if reverse {
            -BigInt::from(pair) - 2u8
        } else {
            BigInt::from(pair)
        };
        let cursor = FactoredCursor::new(self, &start);
        let [on, gap] = cursor.pair_lengths(unit, 0);
        Ok(finish_phase(
            cursor, start, on, gap, into_pair, unit, reverse,
        ))
    }
}

fn validate_list(list: &DashList) -> Result<usize, String> {
    match list {
        DashList::Lengths(values) => {
            if !values.iter().all(|&value| value.is_finite() && value >= 0.) {
                return Err(
                    "sampled stroke-dasharray lengths must be finite and nonnegative".into(),
                );
            }
            Ok(values.len())
        }
        DashList::Mix(left, right, weight) => {
            if !(0. ..=1.).contains(weight) {
                return Err("sampled stroke-dasharray mix weight must be between 0 and 1".into());
            }
            let left = validate_list(left)?;
            let right = validate_list(right)?;
            Ok(match *weight {
                0. => left,
                1. => right,
                _ => left.saturating_add(right),
            })
        }
    }
}

fn finish_phase(
    mut cursor: FactoredCursor<'_>,
    _start: BigInt,
    on: BigUint,
    gap: BigUint,
    into_pair: BigUint,
    unit: i32,
    reverse: bool,
) -> Phase<'_> {
    let (is_on, remaining) = if reverse {
        if into_pair <= gap {
            (false, into_pair)
        } else {
            (true, into_pair - gap)
        }
    } else if into_pair < on {
        (true, on - into_pair)
    } else {
        (false, on + gap - into_pair)
    };
    if !is_on {
        cursor.advance();
    }
    Phase {
        is_on,
        remaining: to_distance(&remaining, unit),
        cursor: Cursor::Factored(cursor),
        #[cfg(test)]
        index: _start + u8::from(!is_on),
    }
}

/// Odd significand and its power of two, including subnormals.
fn dyadic(value: f64) -> (u64, i32) {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1 << 52) - 1);
    let mantissa = if exponent == 0 {
        fraction
    } else {
        fraction | (1 << 52)
    };
    if mantissa == 0 {
        return (0, 0);
    }
    let zeros = mantissa.trailing_zeros();
    (mantissa >> zeros, exponent.max(1) - 1075 + zeros as i32)
}

fn to_distance(value: &BigUint, unit: i32) -> f64 {
    // Round to the f64 significand, or to the subnormal grid near zero.
    let shift = value
        .bits()
        .saturating_sub(53)
        .max((-1074 - unit).max(0) as u64);
    let mut high = u64::try_from(value >> shift).unwrap();
    if shift > 0
        && value.bit(shift - 1)
        && (high % 2 == 1 || value.trailing_zeros().unwrap() < shift - 1)
    {
        high += 1;
    }
    if high == 0 {
        return 0.;
    }
    let exponent = unit + shift as i32;
    let exponent_floor = exponent.max(-1022);
    high as f64 * 2f64.powi(exponent_floor) * 2f64.powi(exponent - exponent_floor)
}

struct Source {
    values: Arc<[f64]>,
    weight: BigUint,
    weight_exponent: i32,
    exponent: Option<i32>,
}

impl Source {
    fn new(values: Arc<[f64]>, weight: BigUint, weight_exponent: i32) -> Self {
        let exponent = values
            .iter()
            .copied()
            .filter(|&value| value > 0.)
            .map(|value| dyadic(value).1 + weight_exponent)
            .min();
        Self {
            values,
            weight,
            weight_exponent,
            exponent,
        }
    }

    fn collect(list: &DashList, weight: &BigUint, exponent: i32, out: &mut Vec<Self>) {
        if weight == &BigUint::ZERO {
            return;
        }
        match list.active() {
            DashList::Lengths(values) => {
                if !values.is_empty() {
                    out.push(Self::new(values.clone(), weight.clone(), exponent));
                }
            }
            DashList::Mix(left, right, mix) => {
                let (mantissa, shift) = dyadic(f64::from(*mix));
                let numerator = BigUint::from(mantissa);
                let denominator = BigUint::from(1u8) << (-shift) as usize;
                Self::collect(
                    left,
                    &(weight * (&denominator - &numerator)),
                    exponent + shift,
                    out,
                );
                Self::collect(right, &(weight * numerator), exponent + shift, out);
            }
        }
    }

    fn mean(&self, unit: i32, even: bool) -> f64 {
        let count = self.values.len();
        let stride = if even && count % 2 == 0 { 2 } else { 1 };
        self.values
            .iter()
            .step_by(stride)
            .map(|&value| to_distance(&self.exact(value, unit), unit) / (count / stride) as f64)
            .sum()
    }

    fn exact(&self, value: f64, unit: i32) -> BigUint {
        let (mantissa, exponent) = dyadic(value);
        if mantissa == 0 {
            return BigUint::ZERO;
        }
        (&self.weight * mantissa) << (exponent + self.weight_exponent - unit) as usize
    }

    fn prefixes(&self, unit: i32, reverse: bool) -> Vec<BigUint> {
        let mut prefix = Vec::with_capacity(self.values.len() + 1);
        prefix.push(BigUint::ZERO);
        for index in 0..self.values.len() {
            let index = if reverse {
                self.values.len() - 1 - index
            } else {
                index
            };
            prefix.push(prefix.last().unwrap() + self.exact(self.values[index], unit));
        }
        prefix
    }
}

fn prefix_at(prefixes: &[Vec<BigUint>], index: &BigUint) -> BigUint {
    prefixes
        .iter()
        .map(|prefix| {
            let length = (prefix.len() - 1) as u64;
            let tail = u64::try_from(index % length).unwrap() as usize;
            (index / length) * prefix.last().unwrap() + &prefix[tail]
        })
        .sum()
}

pub(super) struct Phase<'a> {
    pub(super) is_on: bool,
    pub(super) remaining: f64,
    pub(super) cursor: Cursor<'a>,
    #[cfg(test)]
    index: BigInt,
}

#[derive(Clone)]
pub(super) enum Cursor<'a> {
    Rounded { values: &'a [f64], index: usize },
    Factored(FactoredCursor<'a>),
}

impl Cursor<'_> {
    pub(super) fn next_length(&mut self) -> f64 {
        match self {
            Self::Rounded { values, index } => {
                *index = (*index + 1) % values.len();
                values[*index]
            }
            Self::Factored(cursor) => {
                cursor.advance();
                let unit = cursor.pattern.unit;
                let length = if cursor.pattern.capped {
                    let [on, gap] = cursor.pair_lengths(unit, -isize::from(!cursor.on));
                    if cursor.on { on } else { gap }
                } else {
                    cursor.length_at(unit, 0)
                };
                to_distance(&length, unit)
            }
        }
    }
}

#[derive(Clone)]
pub(super) struct FactoredCursor<'a> {
    pattern: &'a CompactDash,
    indices: Vec<usize>,
    on: bool,
}

impl<'a> FactoredCursor<'a> {
    fn new(pattern: &'a CompactDash, start: &BigInt) -> Self {
        let indices = pattern
            .sources
            .iter()
            .map(|source| {
                let count = source.values.len();
                let index = u64::try_from(start.magnitude() % count as u64).unwrap() as usize;
                if start.sign() == Sign::Minus && index != 0 {
                    count - index
                } else {
                    index
                }
            })
            .collect();
        Self {
            pattern,
            indices,
            on: true,
        }
    }

    fn length_at(&self, unit: i32, relative: isize) -> BigUint {
        self.pattern
            .sources
            .iter()
            .zip(&self.indices)
            .map(|(source, &index)| {
                let index =
                    (index as isize + relative).rem_euclid(source.values.len() as isize) as usize;
                source.exact(source.values[index], unit)
            })
            .sum()
    }

    fn pair_lengths(&self, unit: i32, relative: isize) -> [BigUint; 2] {
        let mut on = self.length_at(unit, relative);
        let mut gap = self.length_at(unit, relative + 1);
        if self.pattern.capped {
            let old_on = to_distance(&on, unit);
            let [dot, _] =
                moda_scene::paint::zero_dash_pair_as_dot(old_on, to_distance(&gap, unit), 1.);
            if dot != old_on {
                let (significand, exponent) = dyadic(dot);
                let dot = BigUint::from(significand) << (exponent - unit) as usize;
                gap -= &dot - &on;
                on = dot;
            }
        }
        [on, gap]
    }

    fn advance(&mut self) {
        for (source, index) in self.pattern.sources.iter().zip(&mut self.indices) {
            *index += 1;
            if *index == source.values.len() {
                *index = 0;
            }
        }
        self.on = !self.on;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tess::dash::{PathSegment, Seg, walk_path_by_arc_length_phased, walk_path_runs};

    #[test]
    fn exact_mix_endpoints_walk_like_the_active_dash_list() {
        let segment = [PathSegment {
            seg: Seg::Line {
                p0: [0., 0.],
                p1: [12., 0.],
            },
            arc_length: 12.,
        }];
        let expected = super::super::dash::walk_path_by_arc_length(&segment, &[1., 2.]);
        for (repeats, rounded) in [(1, true), (41, false)] {
            let active = DashList::Lengths([1., 2.].repeat(repeats).into());
            let inactive = DashList::Lengths(vec![2f64.powi(60); 37].into());
            for list in [inactive.mix(&active, 1.), active.mix(&inactive, 0.)] {
                let (list, scale) = if rounded {
                    (list, 1.)
                } else {
                    let zeros = DashList::Lengths(vec![0.; 37].into());
                    (zeros.mix(&list, 0.5), 2.)
                };
                let dash = SampledDash { list, scale };
                let runs = CompactDash::new(&dash, 4., LineCap::Butt, false).unwrap();
                assert_eq!(runs.corners().is_some(), rounded);
                assert!(!runs.is_solid());
                let mut phase = runs.phase(0.).unwrap();
                let actual = walk_path_runs(&segment, phase.is_on, phase.remaining, || {
                    Ok(phase.cursor.next_length())
                })
                .unwrap();
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn corner_alignment_uses_the_active_endpoints_period() {
        let active = DashList::Lengths([1., 2.].into());
        let inactive = DashList::Lengths(vec![2f64.powi(60); 37].into());
        for list in [inactive.mix(&active, 1.), active.mix(&inactive, 0.)] {
            let dash = SampledDash { list, scale: 1. };
            let runs = CompactDash::new(&dash, 4., LineCap::Butt, true).unwrap();
            assert_eq!(runs.corners(), Some([1., 2.].as_slice()));
        }
    }

    #[test]
    fn raw_sampled_strokes_reject_invalid_weights_before_seeking() {
        for weight in [-0.5, 1.5, 2.] {
            let stroke: moda_scene::paint::Stroke = serde_json::from_value(serde_json::json!({
                "id": "sample",
                "paint": {"type": "solid", "color": "black"},
                "width": 4.,
                "sampledDash": {
                    "list": {"Mix": [{"Lengths": [2., 3.]}, {"Lengths": [5., 7., 11.]}, weight]},
                    "scale": 1.
                }
            }))
            .unwrap();
            let dash = stroke.sampled_dash.as_ref().unwrap();
            use super::super::moda_engine_path_types::{FillRule, LineJoin, PathCmd, StrokeAlign};
            let geometry = super::super::tessellator::tessellate_with_pattern_cached(
                &[
                    PathCmd::Move { x: 0., y: 0. },
                    PathCmd::Line { x: 16., y: 0. },
                ],
                stroke.width,
                false,
                None,
                Some(DashPattern::Sampled(dash)),
                0.,
                LineCap::Butt,
                LineJoin::Miter,
                FillRule::NonZero,
                StrokeAlign::Center,
                false,
            );
            let error = geometry.stroke_dropped.as_deref().unwrap();
            assert!(error.contains("mix weight"), "{error}");
        }
        let source = DashList::Lengths(vec![1., 2.].into());
        let dash = SampledDash {
            list: source.mix(&source, f32::NAN),
            scale: 1.,
        };
        assert!(CompactDash::new(&dash, 4., LineCap::Butt, false).is_err());
    }

    #[test]
    fn short_round_dashes_keep_the_legacy_seam() {
        let tiny = 2f64.powi(-18);
        let left = DashList::Lengths(vec![240., tiny, 240., 12.].into());
        let right = DashList::Lengths(vec![272., tiny, 272., 12.].into());
        let dash = SampledDash {
            list: left.mix(&right, 0.5),
            scale: 1.,
        };
        let runs = CompactDash::new(&dash, 4., LineCap::Round, false).unwrap();

        let mut phase = runs.phase(512.).unwrap();
        assert!(!phase.is_on);
        assert_eq!(phase.remaining, 12.);
        assert_eq!(phase.cursor.next_length(), 256.);
    }

    #[test]
    fn seek_and_walk_keep_products_with_underflowing_weights() {
        let zero = DashList::Lengths(vec![0.; 37].into());
        let mut list = DashList::Lengths(vec![2f64.powi(1023); 41].into());
        for _ in 0..15 {
            list = zero.mix(&list, 2f32.powi(-74));
        }
        let dash = SampledDash { list, scale: 1. };
        let runs = CompactDash::new(&dash, 4., LineCap::Butt, false).unwrap();

        let mut phase = runs.phase(0.).unwrap();
        let expected = 2f64.powi(1023 - 15 * 74);
        assert_eq!(phase.remaining, expected);
        assert_eq!(phase.cursor.next_length(), expected);
        assert_eq!(phase.cursor.next_length(), expected);
    }

    #[test]
    fn coprime_periods_walk_only_the_contour_and_keep_phase() {
        let left: Vec<_> = (0..4093).map(|i| 2. + f64::from(i % 7)).collect();
        let right: Vec<_> = (0..4091).map(|i| 3. + f64::from(i % 11)).collect();
        let list = DashList::Lengths(left.clone().into())
            .mix(&DashList::Lengths(right.clone().into()), 0.5);
        let dash = SampledDash { list, scale: 1. };
        let shell = [
            PathSegment {
                seg: Seg::Line {
                    p0: [0., 0.],
                    p1: [16., 0.],
                },
                arc_length: 16.,
            },
            PathSegment {
                seg: Seg::Line {
                    p0: [16., 0.],
                    p1: [16., 8.],
                },
                arc_length: 8.,
            },
            PathSegment {
                seg: Seg::Line {
                    p0: [16., 8.],
                    p1: [0., 0.],
                },
                arc_length: 320f64.sqrt(),
            },
        ];
        let length =
            |i: i128| (left[i.rem_euclid(4093) as usize] + right[i.rem_euclid(4091) as usize]) / 2.;
        let period = 4093 * 4091 * 2;
        let distance = left.iter().sum::<f64>() * (period / 4093) as f64 / 2.
            + right.iter().sum::<f64>() * (period / 4091) as f64 / 2.;
        let dash = CompactDash::new(&dash, 4., LineCap::Butt, false).unwrap();
        for (start, offset) in [(0, 1.), (0, 1. + distance), (-2, -1.)] {
            let expected: Vec<_> = (start..start + 24).map(length).collect();
            let phase = if start < 0 {
                length(-2) + length(-1) - 1.
            } else {
                1.
            };
            let expected = walk_path_by_arc_length_phased(&shell, &expected, phase);
            let Phase {
                is_on,
                remaining,
                mut cursor,
                ..
            } = dash.phase(offset).unwrap();
            let mut steps = 0;
            let actual = walk_path_runs(&shell, is_on, remaining, || {
                steps += 1;
                Ok(cursor.next_length())
            })
            .unwrap();
            assert_eq!(actual, expected);
            assert!(steps < 12, "{steps}");
        }
    }
    #[test]
    fn skipped_fractional_runs_and_consumed_runs_use_the_same_lengths() {
        let a = 0.1;
        let b = 0.3;
        let left = DashList::Lengths(vec![a; 4093].into());
        let right = DashList::Lengths(vec![b; 4091].into());
        let dash = SampledDash {
            list: left.mix(&right, 0.5),
            scale: 1.5,
        };
        let runs = CompactDash::new(&dash, 2., LineCap::Butt, false).unwrap();
        let index = 300_000;
        let authored = (a + b) / 2.;
        let offset = (index as f64 + 0.25) * authored * dash.scale;
        let mut phase = runs.phase(offset).unwrap();
        assert_eq!(phase.index, BigInt::from(index));
        assert!((phase.remaining - 0.75 * authored * dash.scale).abs() < 1e-10);
        assert!((phase.cursor.next_length() - authored * dash.scale).abs() < 1e-14);
    }
    #[test]
    fn small_negative_offsets_keep_their_phase_in_a_very_large_period() {
        let list = [4093, 4091, 4079, 4073, 4057]
            .into_iter()
            .map(|len| DashList::Lengths(vec![2.; len].into()))
            .reduce(|a, b| a.mix(&b, 0.5))
            .unwrap();
        let dash = SampledDash { list, scale: 1. };
        let runs = CompactDash::new(&dash, 2., LineCap::Butt, false).unwrap();
        let phase = runs.phase(-0.25).unwrap();
        assert_eq!(phase.index, BigInt::from(-1));
        assert_eq!(phase.remaining, 0.25);
    }

    #[test]
    fn negative_seam_keeps_small_runs_next_to_huge_runs() {
        let source = |len| {
            let mut lengths = vec![2.; len];
            lengths[0] = 1e30;
            DashList::Lengths(lengths.into())
        };
        let dash = SampledDash {
            list: source(1001).mix(&source(1003), 0.5),
            scale: 1.,
        };
        let runs = CompactDash::new(&dash, 4., LineCap::Butt, false).unwrap();
        for (offset, index, remaining) in [
            (-0.25, -1, 0.25),
            (-2., -1, 2.),
            (-2.25, -2, 0.25),
            (-4., -2, 2.),
            (0., 0, 1e30),
            (0.25, 0, 1e30),
        ] {
            let phase = runs.phase(offset).unwrap();
            assert_eq!(phase.index, BigInt::from(index), "offset {offset}");
            assert_eq!(phase.remaining, remaining, "offset {offset}");
        }
    }

    #[test]
    fn huge_offsets_work_when_the_common_period_and_index_overflow_u128() {
        let list = [
            101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
            191, 193,
        ]
        .into_iter()
        .map(|len| DashList::Lengths(vec![0.25; len].into()))
        .reduce(|a, b| a.mix(&b, 0.5))
        .unwrap();
        assert_eq!(list.period(), None);
        let dash = SampledDash { list, scale: 1. };
        let runs = CompactDash::new(&dash, 4., LineCap::Butt, false).unwrap();
        let offset = 2f64.powi(127);
        for sign in [1, -1] {
            let phase = runs.phase(offset * f64::from(sign)).unwrap();
            assert_eq!(phase.index, BigInt::from(sign) << 129usize);
            assert!(phase.is_on);
            assert_eq!(phase.remaining, 0.25);
        }
    }

    #[test]
    fn exact_loop_edges_and_dot_cuts_keep_the_calibrated_phase() {
        let dash = SampledDash {
            list: DashList::Lengths(vec![0.; 1001].into()).mix(
                &DashList::Lengths(vec![2., 2., 0., 2., 2., 2., 0., 2.].into()),
                0.5,
            ),
            scale: 2.,
        };
        let runs = CompactDash::new(&dash, 4., LineCap::Round, false).unwrap();
        let period = 12. * 1001.;
        for offset in [-0.25, 0., 0.25, 4., 4.005, 4.015, 4.25] {
            let phase = runs.phase(offset).unwrap();
            for turns in [-32., 31.] {
                let wrapped = runs.phase(offset + turns * period).unwrap();
                assert_eq!(wrapped.is_on, phase.is_on, "offset {offset} turns {turns}");
                assert!(
                    (wrapped.remaining - phase.remaining).abs() < 1e-8,
                    "offset {offset} turns {turns}"
                );
            }
        }
        let dot = runs.phase(4.).unwrap();
        assert_eq!(dot.index, BigInt::from(2));
        assert_eq!(dot.remaining, 0.01);
        let gap = runs.phase(4.25).unwrap();
        assert_eq!(gap.index, BigInt::from(3));
        assert_eq!(gap.remaining, 1.75);
    }
}
