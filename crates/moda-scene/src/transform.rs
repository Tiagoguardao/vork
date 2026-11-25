//! Affine-transform helpers — line-for-line port of
//! `core/scene/affineTransform.ts:43-179` plus path bounds
//! (`utils/geometryHelpers.ts`: boundsFromPath :692-849, arc bounds :864-971).
//!
//! All math in f64 (the TS engine is f64 — DECISION.md §1.4 rationale).

use crate::types::{Affine, Transform2D};
use crate::value::Value;

pub fn identity() -> Affine {
    Affine {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        e: 0.0,
        f: 0.0,
    }
}

/// `translateAffine` (affineTransform.ts:47-49).
pub fn translate(tx: f64, ty: f64) -> Affine {
    Affine {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        e: tx,
        f: ty,
    }
}

/// `affineFromTRS` (affineTransform.ts:55-73). Rotation in DEGREES.
pub fn from_trs(t: Transform2D) -> Affine {
    let rad = t.rotation * std::f64::consts::PI / 180.0;
    let cos = rad.cos();
    let sin = rad.sin();
    Affine {
        a: cos * t.scale_x,
        b: sin * t.scale_x,
        c: -sin * t.scale_y,
        d: cos * t.scale_y,
        e: t.x,
        f: t.y,
    }
}

/// `affineFromTRS` with a shear (affineTransform.ts): `T · R · Shear(tan skewX) · S`. `skew_x_deg`
/// leans the local y axis along the local x axis before the rotation (nodes.ts `skewX`), which
/// makes every 2×3 matrix a node transform.
pub fn from_trsk(t: Transform2D, skew_x_deg: f64) -> Affine {
    if skew_x_deg == 0.0 || !skew_x_deg.is_finite() {
        return from_trs(t);
    }
    let rad = t.rotation * std::f64::consts::PI / 180.0;
    let cos = rad.cos();
    let sin = rad.sin();
    let k = (skew_x_deg.clamp(-89.0, 89.0) * std::f64::consts::PI / 180.0).tan();
    Affine {
        a: cos * t.scale_x,
        b: sin * t.scale_x,
        c: (cos * k - sin) * t.scale_y,
        d: (sin * k + cos) * t.scale_y,
        e: t.x,
        f: t.y,
    }
}

/// `composeAffine` (affineTransform.ts:81-90): apply(compose(A,B), p) ==
/// apply(A, apply(B, p)).
pub fn compose(a: Affine, b: Affine) -> Affine {
    Affine {
        a: a.a * b.a + a.c * b.b,
        b: a.b * b.a + a.d * b.b,
        c: a.a * b.c + a.c * b.d,
        d: a.b * b.c + a.d * b.d,
        e: a.a * b.e + a.c * b.f + a.e,
        f: a.b * b.e + a.d * b.f + a.f,
    }
}

/// `applyAffineToPoint` (affineTransform.ts:92-97).
pub fn apply_point(p: (f64, f64), t: Affine) -> (f64, f64) {
    (t.a * p.0 + t.c * p.1 + t.e, t.b * p.0 + t.d * p.1 + t.f)
}

/// `transformedAABB` (affineTransform.ts:156-179).
pub fn transformed_aabb(local: (f64, f64, f64, f64), t: Affine) -> (f64, f64, f64, f64) {
    let (x, y, w, h) = local;
    let corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)];
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for c in corners {
        let p = apply_point(c, t);
        min_x = min_x.min(p.0);
        min_y = min_y.min(p.1);
        max_x = max_x.max(p.0);
        max_y = max_y.max(p.1);
    }
    (min_x, min_y, max_x, max_y)
}

// ---------------------------------------------------------------------------
// boundsFromPath (utils/geometryHelpers.ts:692-849) — path bounds for the
// display list, masks and the v3 animation evaluator.
// ---------------------------------------------------------------------------

pub use crate::path_geom::bounds::Bounds;

/// `boundsFromPath` over raw JSON path points (geometryHelpers.ts:692-849).
pub fn bounds_from_path(points: &[Value]) -> Option<Bounds> {
    bounds_from_points(crate::path_points::PathPoints::Json(points))
}

/// `boundsFromPath` over a node's points ([`crate::types::RawNode::points`]).
pub fn bounds_from_points(points: crate::path_points::PathPoints<'_>) -> Option<Bounds> {
    crate::path_geom::bounds::path_bounds(points.draw_commands())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    /// An arc's bulge past its endpoints is in the bounds, on the side its sweep flag draws: `A30 30 0 0 0 60 60`
    /// from (0,60), a drop's round bottom, was bounded by the other half of its circle (60 tall, drawn 90).
    #[test]
    fn arc_bounds_follow_the_sweep_and_rotation() {
        let drop = [
            json!({ "type": "move", "x": 30.0, "y": 0.0 }),
            json!({ "type": "curve", "cp1": { "x": 30.0, "y": 0.0 }, "cp2": { "x": 0.0, "y": 42.0 }, "x": 0.0, "y": 60.0 }),
            json!({ "type": "arc", "rx": 30.0, "ry": 30.0, "xAxisRotation": 0.0, "largeArc": false, "sweep": false, "x": 60.0, "y": 60.0 }),
            json!({ "type": "curve", "cp1": { "x": 60.0, "y": 42.0 }, "cp2": { "x": 30.0, "y": 0.0 }, "x": 30.0, "y": 0.0 }),
        ];
        let b = bounds_from_path(&drop).unwrap();
        let r6 = |v: f64| (v * 1e6).round() / 1e6;
        assert_eq!(
            (r6(b.min_x), r6(b.min_y), r6(b.max_x), r6(b.max_y)),
            (0.0, 0.0, 60.0, 90.0)
        );

        let mut seed: u64 = 7;
        let mut r = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..400 {
            let (x0, y0) = (r() * 200.0, r() * 200.0);
            let arc = crate::path_geom::arc::ArcParams {
                rx: 5.0 + r() * 120.0,
                ry: 5.0 + r() * 120.0,
                x_axis_rotation: r() * 360.0 - 180.0,
                large_arc: r() < 0.5,
                sweep: r() < 0.5,
                x: r() * 200.0,
                y: r() * 200.0,
            };
            let points = [
                json!({ "type": "move", "x": x0, "y": y0 }),
                json!({ "type": "arc", "rx": arc.rx, "ry": arc.ry, "xAxisRotation": arc.x_axis_rotation,
                        "largeArc": arc.large_arc, "sweep": arc.sweep, "x": arc.x, "y": arc.y }),
            ];
            let b = bounds_from_path(&points).unwrap();
            let c = crate::path_geom::arc::arc_endpoint_to_center(x0, y0, &arc).unwrap();
            let (mut lo_x, mut lo_y, mut hi_x, mut hi_y) =
                (x0.min(arc.x), y0.min(arc.y), x0.max(arc.x), y0.max(arc.y));
            for i in 0..=2048 {
                let t = c.theta1 + c.dtheta * i as f64 / 2048.0;
                let (ux, uy) = (c.rx * t.cos(), c.ry * t.sin());
                let x = c.cx + ux * c.phi.cos() - uy * c.phi.sin();
                let y = c.cy + ux * c.phi.sin() + uy * c.phi.cos();
                (lo_x, lo_y, hi_x, hi_y) = (lo_x.min(x), lo_y.min(y), hi_x.max(x), hi_y.max(y));
            }
            for (got, want) in [
                (b.min_x, lo_x),
                (b.min_y, lo_y),
                (b.max_x, hi_x),
                (b.max_y, hi_y),
            ] {
                assert!(
                    (got - want).abs() < 0.01,
                    "{got} vs drawn {want} for {points:?}"
                );
            }
        }
    }

    #[test]
    fn compose_matches_apply_order() {
        // affineTransform.ts:75-90 contract.
        let a = from_trs(Transform2D {
            x: 10.0,
            y: 5.0,
            rotation: 30.0,
            scale_x: 2.0,
            scale_y: 1.0,
        });
        let b = from_trs(Transform2D {
            x: -3.0,
            y: 7.0,
            rotation: -15.0,
            scale_x: 1.0,
            scale_y: 3.0,
        });
        let p = (4.2, -1.1);
        let lhs = apply_point(p, compose(a, b));
        let rhs = apply_point(apply_point(p, b), a);
        assert!((lhs.0 - rhs.0).abs() < 1e-12 && (lhs.1 - rhs.1).abs() < 1e-12);
    }

    #[test]
    fn from_trsk_is_rotate_shear_scale() {
        // T · R · Shear(tan skew) · S, the TS `affineFromTRS` with `skewX`.
        let t = Transform2D {
            x: 7.0,
            y: -3.0,
            rotation: 30.0,
            scale_x: 2.0,
            scale_y: 0.5,
        };
        let m = from_trsk(t, 20.0);
        let k = 20f64.to_radians().tan();
        let r = from_trs(Transform2D {
            scale_x: 1.0,
            scale_y: 1.0,
            ..t
        });
        let shear_scale = Affine {
            a: 2.0,
            b: 0.0,
            c: k * 0.5,
            d: 0.5,
            e: 0.0,
            f: 0.0,
        };
        let want = compose(r, shear_scale);
        for (got, want) in [
            (m.a, want.a),
            (m.b, want.b),
            (m.c, want.c),
            (m.d, want.d),
            (m.e, want.e),
            (m.f, want.f),
        ] {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        // No shear is exactly from_trs.
        assert_eq!(from_trsk(t, 0.0), from_trs(t));
        // The top edge stays on the x axis; the bottom-left corner leans by tan(skew)·h.
        let unrotated = from_trsk(
            Transform2D {
                x: 0.0,
                y: 0.0,
                rotation: 0.0,
                scale_x: 1.0,
                scale_y: 1.0,
            },
            45.0,
        );
        let bl = apply_point((0.0, 10.0), unrotated);
        assert!((bl.0 - 10.0).abs() < 1e-9 && (bl.1 - 10.0).abs() < 1e-9);
        assert_eq!(apply_point((10.0, 0.0), unrotated), (10.0, 0.0));
    }

    #[test]
    fn bounds_from_path_cubic_extrema() {
        // geometryHelpers.ts:754-779 — bounds must include the curve bulge.
        let pts = vec![
            json!({"type":"move","x":0,"y":0}),
            json!({"type":"curve","x":100,"y":0,"cp1":{"x":0,"y":100},"cp2":{"x":100,"y":100}}),
        ];
        let b = bounds_from_path(&pts).unwrap();
        assert!(b.max_y > 70.0 && b.max_y <= 75.0); // bulge = 75 at t=0.5
        assert_eq!((b.min_x, b.min_y, b.max_x), (0.0, 0.0, 100.0));
    }
}
