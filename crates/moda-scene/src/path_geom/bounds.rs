//! Positional error budget in local units. Each Bézier edge gets 8 ulps of its
//! largest absolute control coordinate on that axis. An SVG arc gets 32 ulps of
//! max(|endpoints|, corrected radii) times its aspect ratio and the conditioning
//! of endpoint-to-centre conversion: max(1, 1/max(√ε, √|1-h²|)), where h is
//! the normalized half-chord before radii correction and ε is binary64 epsilon.
//! The square root is inherent at a diameter. None of these budgets is a cutoff.

use super::arc::{arc_to_beziers, ArcParams};
use crate::blend::BlendPathPoint as Cmd;
use std::borrow::Borrow;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl Bounds {
    pub fn center(&self) -> (f64, f64) {
        (
            self.min_x.midpoint(self.max_x),
            self.min_y.midpoint(self.max_y),
        )
    }
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    fn include_cubic(&mut self, from: [f64; 2], c1: [f64; 2], c2: [f64; 2], to: [f64; 2]) -> Option<()> {
        let axis = |i: usize| AxisBezier::Cubic([from[i], c1[i], c2[i], to[i]]).bounds();
        let (left, right) = axis(0)?;
        let (bottom, top) = axis(1)?;
        self.include(left, bottom);
        self.include(right, top);
        Some(())
    }

    pub(crate) fn include(&mut self, x: f64, y: f64) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
    }
}

/// The hull of the endpoints and interior extrema. Move points count; a point or
/// axis-aligned line has a valid zero extent. Empty and non-finite paths have no box.
pub fn path_bounds<C: Borrow<Cmd>>(cmds: impl IntoIterator<Item = C>) -> Option<Bounds> {
    measure(cmds, arc_extents)
}

/// Bounds of the path the tessellator draws, including its cubic approximation of each SVG arc.
pub fn drawn_path_bounds<C: Borrow<Cmd>>(cmds: impl IntoIterator<Item = C>) -> Option<Bounds> {
    measure(cmds, |bounds, from, arc| {
        let mut start = [from.0, from.1];
        for b in arc_to_beziers(from.0, from.1, arc) {
            bounds.include_cubic(start, b.cp1, b.cp2, b.end)?;
            start = b.end;
        }
        Some(())
    })
}

fn measure<C: Borrow<Cmd>>(
    cmds: impl IntoIterator<Item = C>,
    arc: impl Fn(&mut Bounds, (f64, f64), &ArcParams) -> Option<()>,
) -> Option<Bounds> {
    let mut b = Bounds {
        min_x: f64::INFINITY,
        min_y: f64::INFINITY,
        max_x: f64::NEG_INFINITY,
        max_y: f64::NEG_INFINITY,
    };
    let mut from = (0.0, 0.0);
    for cmd in cmds {
        let cmd = cmd.borrow();
        let (Cmd::Move { x, y }
        | Cmd::Line { x, y }
        | Cmd::Quadratic { x, y, .. }
        | Cmd::Curve { x, y, .. }
        | Cmd::Arc { x, y, .. }) = *cmd;
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let to = (x, y);
        b.include(x, y);
        if !matches!(cmd, Cmd::Move { .. }) {
            b.include(from.0, from.1);
        }
        match *cmd {
            Cmd::Quadratic { cp, .. } => {
                let (lo, hi) = AxisBezier::Quadratic([from.0, cp.0, x]).bounds()?;
                let (bottom, top) = AxisBezier::Quadratic([from.1, cp.1, y]).bounds()?;
                b.include(lo, bottom);
                b.include(hi, top);
            }
            Cmd::Curve { cp1, cp2, .. } => {
                b.include_cubic(from.into(), cp1.into(), cp2.into(), to.into())?;
            }
            Cmd::Arc {
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
                ..
            } => {
                if !rx.is_finite() || !ry.is_finite() || !x_axis_rotation.is_finite() {
                    return None;
                }
                arc(&mut b, from, &ArcParams { x, y, rx, ry, x_axis_rotation, large_arc, sweep })?;
            }
            Cmd::Move { .. } | Cmd::Line { .. } => {}
        }
        from = to;
    }
    b.min_x.is_finite().then_some(b)
}

/// One coordinate of a segment. Bounds, trim crossings and monotone subdivision
/// all use the same stationary parameters.
#[derive(Clone, Copy)]
pub enum AxisBezier {
    Line([f64; 2]),
    Quadratic([f64; 3]),
    Cubic([f64; 4]),
}

impl AxisBezier {
    fn controls(&self) -> &[f64] {
        match self {
            Self::Line(p) => p,
            Self::Quadratic(p) => p,
            Self::Cubic(p) => p,
        }
    }

    /// Distinct interior derivative roots in increasing order.
    pub fn extrema(self) -> impl Iterator<Item = f64> {
        let controls = self.controls();
        let magnitude = controls.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        // Keep differences and products representable at extreme exponents.
        let scale = if magnitude != 0.0 && !(1e-140..=1e140).contains(&magnitude) {
            magnitude
        } else {
            1.0
        };
        let edge = |i: usize| controls[i + 1] / scale - controls[i] / scale;
        let mut roots = match self {
            Self::Line(_) => [f64::NAN; 2],
            Self::Quadratic(_) => {
                let d0 = edge(0);
                [d0 / (d0 - edge(1)), f64::NAN]
            }
            Self::Cubic(_) => {
                let (d0, d1, d2) = (edge(0), edge(1), edge(2));
                // B'/3 = a t² + 2 b t + d0. The discriminant divided by four
                // simplifies to d1²-d0*d2. q/a and d0/q avoid cancellation.
                let b = d1 - d0;
                let a = (d2 - d1) - b;
                if a == 0.0 {
                    [-0.5 * d0 / b, f64::NAN]
                } else {
                    let disc = d1 * d1 - d0 * d2;
                    if disc == 0.0 {
                        [-b / a, f64::NAN]
                    } else {
                        let q = -b - disc.sqrt().copysign(b);
                        [q / a, d0 / q]
                    }
                }
            }
        };
        if roots[0] == roots[1] {
            roots[1] = f64::NAN;
        }
        if roots[0] > roots[1] {
            roots.swap(0, 1);
        }
        roots.into_iter().filter(|&t| t > 0.0 && t < 1.0)
    }

    pub fn at(self, t: f64) -> f64 {
        let lerp = |a, b| (1.0 - t) * a + t * b;
        match self {
            Self::Line([a, b]) => lerp(a, b),
            Self::Quadratic([a, b, c]) => lerp(lerp(a, b), lerp(b, c)),
            Self::Cubic([a, b, c, d]) => {
                let mid = lerp(b, c);
                lerp(lerp(lerp(a, b), mid), lerp(mid, lerp(c, d)))
            }
        }
    }

    pub fn bounds(self) -> Option<(f64, f64)> {
        let p = self.controls();
        if p.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let (mut lo, mut hi) = (p[0].min(p[p.len() - 1]), p[0].max(p[p.len() - 1]));
        if !p[1..p.len() - 1].iter().all(|&v| lo <= v && v <= hi) {
            for t in self.extrema() {
                let v = self.at(t);
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        Some((lo, hi))
    }
}

/// In unit-circle coordinates the chord's perpendicular n points from its
/// midpoint toward one possible centre. Its signed distance is ±sqrt(1-h²)
/// for half-chord length h. The sweep picks one side of the chord, so testing
/// the ellipse's four support points requires no angle wrapping or subdivision.
fn arc_extents(b: &mut Bounds, from: (f64, f64), arc: &ArcParams) -> Option<()> {
    let ArcParams { rx, ry, x_axis_rotation: degrees, large_arc: large, sweep, x, y } = *arc;
    let to = (x, y);
    if rx == 0.0 || ry == 0.0 || from == to {
        return Some(());
    }
    let (sin, cos) = (degrees % 360.0).to_radians().sin_cos();
    let (dx, dy) = (from.0 * 0.5 - to.0 * 0.5, from.1 * 0.5 - to.1 * 0.5);
    let (rx, ry) = (rx.abs(), ry.abs());
    let (u, v) = ((cos * dx + sin * dy) / rx, (cos * dy - sin * dx) / ry);
    let h = u.hypot(v);
    if h == 0.0 || !h.is_finite() {
        return None;
    }
    let (nx, ny) = (v / h, -u / h);
    let correction = h.max(1.0);
    let (rx, ry) = (rx * correction, ry * correction);
    let offset = ((1.0 - h).max(0.0) * (1.0 + h.min(1.0))).sqrt();
    let offset = if large == sweep { -offset } else { offset };
    for (mid, a, c, lo, hi) in [
        (
            from.0.midpoint(to.0),
            rx * cos,
            -ry * sin,
            &mut b.min_x,
            &mut b.max_x,
        ),
        (
            from.1.midpoint(to.1),
            rx * sin,
            ry * cos,
            &mut b.min_y,
            &mut b.max_y,
        ),
    ] {
        let radius = a.hypot(c);
        let projection = nx * a + ny * c;
        if !radius.is_finite() || !projection.is_finite() {
            return None;
        }
        for sign in [-1.0, 1.0] {
            let side = sign * projection / radius + offset;
            if (sweep && side <= 0.0) || (!sweep && side >= 0.0) {
                // For a shallow arc the centre displacement and the radius nearly
                // cancel. Rationalize 1 - k*z using the along-chord projection
                // and the half-chord, retaining even a sub-ulp sagitta.
                let z = projection / radius;
                let k = sign * offset;
                let delta = if k * z < -0.5 {
                    let along = (nx * c - ny * a) / radius;
                    let chord = h.min(1.0) * z;
                    sign * radius * (along * along + chord * chord) / (1.0 - k * z)
                } else {
                    sign * radius + offset * projection
                };
                let edge = mid + delta;
                if !edge.is_finite() {
                    return None;
                }
                *lo = lo.min(edge);
                *hi = hi.max(edge);
            }
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binary64_quadratic_and_nearly_quadratic_cubic() {
        let start = Cmd::Move { x: 296.1, y: 100.0 };
        let q = Cmd::Quadratic {
            cp: (1910.3, 500.0),
            x: 314.9,
            y: 900.0,
        };
        let c = Cmd::Curve {
            cp1: (296.1 + (1910.3 - 296.1) * 2.0 / 3.0, 300.0),
            cp2: (314.9 + (1910.3 - 314.9) * 2.0 / 3.0, 600.0),
            x: 314.9,
            y: 900.0,
        };
        for cmd in [q, c] {
            let b = path_bounds([start.clone(), cmd]).unwrap();
            assert!((b.max_x - 1107.9275299102692).abs() < 2e-12, "{b:?}");
        }
    }

    #[test]
    fn stationary_inflection_has_one_root_and_a_shallow_arc_keeps_its_sagitta() {
        let c = AxisBezier::Cubic([0.0, 1.0, 0.0, 1.0]);
        assert_eq!(c.extrema().collect::<Vec<_>>(), [0.5]);
        let arc = Cmd::Arc {
            x: 2.0,
            y: 0.0,
            rx: 1e10,
            ry: 1e10,
            x_axis_rotation: 0.0,
            large_arc: false,
            sweep: true,
        };
        let b = path_bounds([arc]).unwrap();
        assert!((b.min_y + 5e-11).abs() < 2e-26, "{b:?}");
        assert_eq!((b.min_x, b.max_x, b.max_y), (0.0, 2.0, 0.0));
    }

    #[test]
    fn flat_tiny_and_nonfinite() {
        let quad = |h| Cmd::Quadratic {
            cp: (1.0, h),
            x: 2.0,
            y: 0.0,
        };
        assert_eq!(path_bounds([quad(0.0)]).unwrap().max_y, 0.0);
        assert_eq!(path_bounds([quad(1e-250)]).unwrap().max_y, 5e-251);
        assert!(path_bounds([quad(f64::NAN)]).is_none());
        assert!(path_bounds([
            quad(1.0),
            Cmd::Line {
                x: f64::INFINITY,
                y: 0.0
            }
        ])
        .is_none());
    }
}
