//! Bézier arc length shared by path queries, stroke markers, trims and dashes.
//! Quadratics are elevated to cubics and SVG arcs use `arc_to_beziers`.

use kurbo::{CubicBez, ParamCurveArclen};

type P = [f64; 2];

/// Use f32 unit roundoff so distinct representable stroke distances stay distinct.
pub fn distance_tolerance(length: f64) -> f64 {
    length * (0.5 * f64::from(f32::EPSILON))
}

// Subtract the origin before subdividing so translating a small curve does
// not round its intermediate controls back onto the original coordinates.
fn local_cubic(a: P, b: P, c: P, d: P) -> CubicBez {
    let local = |p: P| (p[0] - a[0], p[1] - a[1]);
    CubicBez::new((0., 0.), local(b), local(c), local(d))
}

pub fn cubic_arc_length(a: P, b: P, c: P, d: P) -> f64 {
    let curve = local_cubic(a, b, c, d);
    let polygon =
        curve.p0.distance(curve.p1) + curve.p1.distance(curve.p2) + curve.p2.distance(curve.p3);
    if polygon == 0. || !polygon.is_finite() {
        return polygon;
    }
    curve.arclen(distance_tolerance(polygon))
}

pub fn find_t_at_arc_length_cubic(a: P, b: P, c: P, d: P, target: f64, total: f64) -> f64 {
    if target <= 0. {
        return 0.;
    }
    if target >= total {
        return 1.;
    }
    local_cubic(a, b, c, d).inv_arclen(target, distance_tolerance(total))
}

/// Cubic controls for the quadratic from `from` to `to` through control `cp`.
pub fn elevated_controls(from: P, cp: P, to: P) -> (P, P) {
    let elevate = |p: P| {
        [
            p[0] + (2.0 / 3.0) * (cp[0] - p[0]),
            p[1] + (2.0 / 3.0) * (cp[1] - p[1]),
        ]
    };
    (elevate(from), elevate(to))
}

pub fn quadratic_arc_length(from: P, cp: P, to: P) -> f64 {
    let cp = [cp[0] - from[0], cp[1] - from[1]];
    let to = [to[0] - from[0], to[1] - from[1]];
    let (c1, c2) = elevated_controls([0., 0.], cp, to);
    cubic_arc_length([0., 0.], c1, c2, to)
}
