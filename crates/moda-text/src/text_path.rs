//! Type on a path (vector-app F7.2, design-editor E7.16): text laid along an open or closed path, like
//! Illustrator's Type on a Path tool.
//!
//! The node's text is laid out as one run (every paragraph on the path in turn, a space apart); each glyph is then
//! moved so its advance centre sits on the path at its distance along it, and turned to the path's tangent there
//! (Illustrator's Rainbow effect). On a bend the distance along the path isn't the advance: Illustrator's curvature
//! spacing ([`PathSpacing`], [`CURVE_SPACING_CONVEX`]) packs type on the outside of a bend closer along the path and
//! spreads type on the inside. Options, as Illustrator's Type on a Path Options:
//!
//! - `start_offset`: where the text starts, a fraction of the path length (the start bracket);
//! - the paragraph alignment: left from the start, centred in what's left of the path, or right-aligned to its end;
//! - `flip`: the text runs the other way along the path, on its other side;
//! - `align`: which line of the text sits on the path — the baseline, the ascender, the descender or the centre.
//!
//! On an open path, glyphs whose centre falls past its end are hidden (Illustrator's overflow); on a closed one the
//! seam is an end bracket: glyphs crossing it are hidden, never wrapped onto earlier text. The draw walk (`vector::ops`)
//! applies the placement as each glyph's transform, so exports that read the engine's glyph geometry (SVG outlines,
//! PDF outlines, Create Outlines) place it exactly the same.

/// Which line of the text sits on the path (Illustrator's Align to Path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextPathAlign {
    #[default]
    Baseline,
    Ascender,
    Descender,
    Center,
}

impl TextPathAlign {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "ascender" | "top" => Self::Ascender,
            "descender" | "bottom" => Self::Descender,
            "center" | "centre" | "middle" => Self::Center,
            _ => Self::Baseline,
        }
    }
}

/// A node's path, flattened, with its options.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TextPathSpec {
    /// Node-local px, in path order (a closed path does not repeat its first point).
    pub points: Vec<(f64, f64)>,
    pub closed: bool,
    /// 0..1 of the path length.
    pub start_offset: f64,
    pub flip: bool,
    pub align: TextPathAlign,
}

/// A path command in node-local px, the input to [`flatten`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathCmd {
    Move(f64, f64),
    Line(f64, f64),
    Quad(f64, f64, f64, f64),
    Cubic(f64, f64, f64, f64, f64, f64),
}

/// The first subpath of `cmds`, flattened to a polyline (curves to within about 0.1px).
pub fn flatten(cmds: &[PathCmd]) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = Vec::new();
    let mut cur = (0.0, 0.0);
    for (i, cmd) in cmds.iter().enumerate() {
        match *cmd {
            PathCmd::Move(x, y) => {
                if i > 0 && !out.is_empty() {
                    break; // text follows the first subpath only
                }
                cur = (x, y);
                out.push(cur);
            }
            PathCmd::Line(x, y) => {
                if out.is_empty() {
                    out.push(cur);
                }
                cur = (x, y);
                out.push(cur);
            }
            PathCmd::Quad(cx, cy, x, y) => {
                if out.is_empty() {
                    out.push(cur);
                }
                let (x0, y0) = cur;
                let n = segments_for(((cx - x0).hypot(cy - y0) + (x - cx).hypot(y - cy)).max(1.0));
                for k in 1..=n {
                    let t = k as f64 / n as f64;
                    let u = 1.0 - t;
                    out.push((
                        u * u * x0 + 2.0 * u * t * cx + t * t * x,
                        u * u * y0 + 2.0 * u * t * cy + t * t * y,
                    ));
                }
                cur = (x, y);
            }
            PathCmd::Cubic(c1x, c1y, c2x, c2y, x, y) => {
                if out.is_empty() {
                    out.push(cur);
                }
                let (x0, y0) = cur;
                let hull = (c1x - x0).hypot(c1y - y0)
                    + (c2x - c1x).hypot(c2y - c1y)
                    + (x - c2x).hypot(y - c2y);
                let n = segments_for(hull.max(1.0));
                for k in 1..=n {
                    let t = k as f64 / n as f64;
                    let u = 1.0 - t;
                    let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
                    out.push((
                        a * x0 + b * c1x + c * c2x + d * x,
                        a * y0 + b * c1y + c * c2y + d * y,
                    ));
                }
                cur = (x, y);
            }
        }
    }
    out.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9);
    out
}

/// Enough segments for a curve whose control hull is `len` px long to stay within ~0.1px.
fn segments_for(len: f64) -> usize {
    ((len / 4.0).ceil() as usize).clamp(4, 256)
}

/// Curvature spacing (Illustrator's, measured against Illustrator 2026 with Spacing 0): on a bend, a glyph's advance
/// and the distance it takes along the path differ. Their ratio, advance ÷ path distance, is `1 + κ·c·size`: κ the
/// path's signed curvature on the text's side (positive where the path bends away from the text, the convex side),
/// `size` the font size in px, `c` [`CURVE_SPACING_CONVEX`] or [`CURVE_SPACING_CONCAVE`]. So type on the outside of a
/// bend sits closer along the path than its advances (its tops fan out), and type on the inside further apart.
pub const CURVE_SPACING_CONVEX: f64 = 0.17;
/// The inside-of-the-bend coefficient of the curvature spacing (see [`CURVE_SPACING_CONVEX`]).
pub const CURVE_SPACING_CONCAVE: f64 = 0.43;
/// The curvature is averaged over a window this many font sizes long (about a glyph), so it varies smoothly along
/// a flattened curve.
const CURVE_WINDOW_EM: f64 = 0.5;
/// A turn sharper than this at one vertex of the flattened path is a corner, not a curve: it adds no curvature, so
/// the straight segments either side of it keep plain advances. (Flattened curves turn far less per vertex.)
const CORNER_TURN: f64 = std::f64::consts::FRAC_PI_3;
/// Integration step of the advance → path distance map, in px of advance.
const SPACING_STEP: f64 = 1.0;

/// Advance ÷ path distance at signed curvature `k` for type `size` px (see [`CURVE_SPACING_CONVEX`]).
pub fn curve_spacing_ratio(k: f64, size: f64) -> f64 {
    let c = if k > 0.0 {
        CURVE_SPACING_CONVEX
    } else {
        CURVE_SPACING_CONCAVE
    };
    (1.0 + k * size * c).clamp(0.2, 5.0)
}

/// Arc-length parametrised path.
#[derive(Debug, Clone)]
pub struct PathMeasure {
    pts: Vec<(f64, f64)>,
    /// cumulative length at each point
    cum: Vec<f64>,
    closed: bool,
    /// Tangent angle of each segment of non-zero length (unwrapped, corners taken out), at its midpoint's distance:
    /// the curvature is the change of this angle along the path.
    mids: Vec<f64>,
    thetas: Vec<f64>,
    /// Closed paths: the angle gained over one lap (±2π for a simple smooth loop, less its corners).
    lap_turn: f64,
    /// False when every segment has the same (corner-free) direction: plain advances, exactly.
    curved: bool,
}

/// `a` − `b` wrapped to (−π, π].
fn angle_diff(a: f64, b: f64) -> f64 {
    let mut d = a - b;
    while d > std::f64::consts::PI {
        d -= std::f64::consts::TAU;
    }
    while d <= -std::f64::consts::PI {
        d += std::f64::consts::TAU;
    }
    d
}

/// A turn at one vertex that counts as curvature (a corner counts as none).
fn curve_turn(d: f64) -> f64 {
    if d.abs() > CORNER_TURN {
        0.0
    } else {
        d
    }
}

impl PathMeasure {
    /// `None` for a path with no length.
    pub fn new(points: &[(f64, f64)], closed: bool, reverse: bool) -> Option<Self> {
        let mut pts: Vec<(f64, f64)> = points.to_vec();
        if closed && pts.len() > 1 {
            let first = pts[0];
            let last = *pts.last().unwrap();
            if (first.0 - last.0).abs() > 1e-9 || (first.1 - last.1).abs() > 1e-9 {
                pts.push(first);
            }
        }
        if reverse {
            pts.reverse();
        }
        if pts.len() < 2 {
            return None;
        }
        let mut cum = Vec::with_capacity(pts.len());
        let mut acc = 0.0;
        cum.push(0.0);
        for w in pts.windows(2) {
            acc += (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1);
            cum.push(acc);
        }
        if acc <= 1e-9 {
            return None;
        }
        let mut mids = Vec::with_capacity(pts.len());
        let mut thetas = Vec::with_capacity(pts.len());
        let mut raws: Vec<f64> = Vec::with_capacity(pts.len());
        for (i, w) in pts.windows(2).enumerate() {
            let (dx, dy) = (w[1].0 - w[0].0, w[1].1 - w[0].1);
            if dx.hypot(dy) <= 1e-9 {
                continue;
            }
            let raw = dy.atan2(dx);
            let theta = match (raws.last(), thetas.last()) {
                (Some(&prev_raw), Some(&prev)) => prev + curve_turn(angle_diff(raw, prev_raw)),
                _ => raw,
            };
            raws.push(raw);
            mids.push((cum[i] + cum[i + 1]) * 0.5);
            thetas.push(theta);
        }
        let lap_turn = match (closed, raws.first(), raws.last()) {
            (true, Some(&first), Some(&last)) => {
                thetas.last().unwrap() + curve_turn(angle_diff(first, last)) - thetas[0]
            }
            _ => 0.0,
        };
        let curved = thetas.iter().any(|t| (t - thetas[0]).abs() > 1e-12) || lap_turn.abs() > 1e-12;
        Some(Self {
            pts,
            cum,
            closed,
            mids,
            thetas,
            lap_turn,
            curved,
        })
    }

    /// The smoothed tangent angle at distance `s` (any real `s` on a closed path, which gains `lap_turn` a lap).
    fn theta(&self, s: f64) -> f64 {
        let (m, th) = (&self.mids, &self.thetas);
        let n = m.len();
        if n == 0 {
            return 0.0;
        }
        let interp = |s: f64, (s0, t0): (f64, f64), (s1, t1): (f64, f64)| {
            if s1 - s0 > 1e-12 {
                t0 + (t1 - t0) * ((s - s0) / (s1 - s0))
            } else {
                t0
            }
        };
        if self.closed {
            let len = self.length();
            let laps = (s / len).floor();
            let r = s - laps * len;
            let base = laps * self.lap_turn;
            if r < m[0] {
                return base
                    + interp(r, (m[n - 1] - len, th[n - 1] - self.lap_turn), (m[0], th[0]));
            }
            if r >= m[n - 1] {
                return base
                    + interp(r, (m[n - 1], th[n - 1]), (m[0] + len, th[0] + self.lap_turn));
            }
            let i = m.partition_point(|&v| v <= r);
            return base + interp(r, (m[i - 1], th[i - 1]), (m[i], th[i]));
        }
        if s <= m[0] {
            return th[0];
        }
        if s >= m[n - 1] {
            return th[n - 1];
        }
        let i = m.partition_point(|&v| v <= s);
        interp(s, (m[i - 1], th[i - 1]), (m[i], th[i]))
    }

    /// The path's signed curvature at distance `s` (1/px), averaged over `window` px: positive where the path turns
    /// clockwise on screen (y down), which bends it away from type set on it (the type's up is the tangent turned
    /// a quarter anticlockwise). Corners count as straight.
    pub fn curvature(&self, s: f64, window: f64) -> f64 {
        if !self.curved {
            return 0.0;
        }
        let h = window.max(1e-6) * 0.5;
        let (a, b) = if self.closed {
            (s - h, s + h)
        } else {
            let len = self.length();
            let s = s.clamp(0.0, len);
            ((s - h).max(0.0), (s + h).min(len))
        };
        if b - a <= 1e-9 {
            return 0.0;
        }
        (self.theta(b) - self.theta(a)) / (b - a)
    }

    /// One integration step of the advance → distance map: `d` px of advance from distance `s` (midpoint rule).
    fn spacing_step(&self, s: f64, d: f64, size: f64) -> f64 {
        if size <= 0.0 {
            return s + d;
        }
        let window = CURVE_WINDOW_EM * size;
        let k1 = 1.0 / curve_spacing_ratio(self.curvature(s, window), size);
        let k2 = 1.0 / curve_spacing_ratio(self.curvature(s + d * k1 * 0.5, window), size);
        s + d * k2
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Closed-path type never wraps: the whole ink extent must fit before the seam.
    /// `at` still wraps for geometric/curvature queries, so text callers check this first.
    pub fn fits_closed_glyph(&self, centre: f64, half_width: f64) -> bool {
        !self.closed || (centre - half_width >= -1e-9 && centre + half_width <= self.length() + 1e-9)
    }

    pub fn length(&self) -> f64 {
        *self.cum.last().unwrap()
    }

    /// The point at distance `s` along the path and the unit tangent there; `None` past the ends of an open path.
    /// A closed path wraps around.
    pub fn at(&self, s: f64) -> Option<((f64, f64), (f64, f64))> {
        let len = self.length();
        let s = if self.closed {
            s.rem_euclid(len)
        } else if !(-1e-9..=len + 1e-9).contains(&s) {
            return None;
        } else {
            s.clamp(0.0, len)
        };
        // First segment whose end is at or past s.
        let i = match self.cum.binary_search_by(|c| c.partial_cmp(&s).unwrap()) {
            Ok(i) => i.max(1),
            Err(i) => i.clamp(1, self.cum.len() - 1),
        };
        let (a, b) = (self.pts[i - 1], self.pts[i]);
        let seg = self.cum[i] - self.cum[i - 1];
        let t = if seg > 0.0 {
            (s - self.cum[i - 1]) / seg
        } else {
            0.0
        };
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let dl = dx.hypot(dy).max(1e-12);
        Some(((a.0 + dx * t, a.1 + dy * t), (dx / dl, dy / dl)))
    }
}

/// The font size along the text: `(advance, size)` steps in ascending advance, each size holding from its advance to
/// the next step's (the first also before it). Built from the laid-out segments (each word and space at its own size),
/// so text whose size changes partway takes each glyph's curvature spacing at its own size.
pub type PathSizes = [(f64, f64)];

/// The size at `advance` px into the text (see [`PathSizes`]); 0 for none.
pub fn size_at(sizes: &PathSizes, advance: f64) -> f64 {
    let i = sizes.partition_point(|&(a, _)| a <= advance);
    sizes.get(i.saturating_sub(1)).map_or(0.0, |&(_, size)| size)
}

/// The advance → path distance map of text set from distance `start` (curvature spacing, [`CURVE_SPACING_CONVEX`]):
/// `distance(advance)` is where the point `advance` px into the text lands along the path. Each step of advance is
/// spaced at the font size the text has there ([`PathSizes`]), so a size change moves what follows it by what its
/// own glyphs take. (Before, the whole run up to a glyph was re-integrated at that glyph's size, so text jumped — a gap
/// or an overlap — where its size changed: 15 px after 400 px of 64 px type then 16 px type on a 200 px circle.) It
/// integrates from the start on a fixed grid of advance, so a result doesn't depend on the order of the queries;
/// ascending queries continue from the last grid point instead of starting over. The TS twin is `TextPathSpacing`.
#[derive(Debug, Clone)]
pub struct PathSpacing<'a> {
    measure: &'a PathMeasure,
    start: f64,
    sizes: &'a PathSizes,
    /// Last grid point reached: advance and distance.
    a: f64,
    s: f64,
}

impl<'a> PathSpacing<'a> {
    pub fn new(measure: &'a PathMeasure, start: f64, sizes: &'a PathSizes) -> Self {
        Self {
            measure,
            start,
            sizes,
            a: 0.0,
            s: start,
        }
    }

    /// The distance along the path of the point `advance` px into the text.
    pub fn distance(&mut self, advance: f64) -> f64 {
        if !self.measure.curved || advance <= 0.0 || self.sizes.iter().all(|&(_, size)| size <= 0.0) {
            return self.start + advance;
        }
        if advance < self.a {
            self.a = 0.0;
            self.s = self.start;
        }
        while self.a + SPACING_STEP <= advance {
            let size = size_at(self.sizes, self.a + SPACING_STEP * 0.5).max(0.0);
            self.s = self.measure.spacing_step(self.s, SPACING_STEP, size);
            self.a += SPACING_STEP;
        }
        let rest = advance - self.a;
        if rest > 0.0 {
            let size = size_at(self.sizes, self.a + rest * 0.5).max(0.0);
            self.measure.spacing_step(self.s, rest, size)
        } else {
            self.s
        }
    }
}

/// The along-path distance where text `text_width` px of advance long starts, for the paragraph alignment; its sizes
/// ([`PathSizes`]) set the curvature spacing of its length along the path.
pub fn start_distance(
    measure: &PathMeasure,
    start_offset: f64,
    align: &str,
    text_width: f64,
    sizes: &PathSizes,
) -> f64 {
    let len = measure.length();
    let start = start_offset.clamp(0.0, 1.0) * len;
    // The text's length along the path from a candidate start; the start moves until the text fits as aligned.
    let along = |from: f64| PathSpacing::new(measure, from, sizes).distance(text_width) - from;
    let settle = |f: &dyn Fn(f64) -> f64, guess: f64| (0..6).fold(guess, |st, _| f(st));
    let placed = match align {
        "center" => settle(
            &|st| start + (len - start - along(st)) / 2.0,
            start + (len - start - text_width) / 2.0,
        ),
        "right" => settle(&|st| len - along(st), len - text_width),
        _ => start,
    };
    // Overset centred/right-aligned type starts at the start bracket and clips at the seam.
    if measure.is_closed() { placed.max(start) } else { placed }
}

/// The glyph transform that puts a glyph on the path: the shader draws `pen + lin·(q − C) + C + off` (q the outline
/// point in px from the pen, C the em box centre in px); this returns the `(lin, off)` that first applies the glyph's
/// own `(lin, off)`, then turns the result about its anchor `anchor` (the glyph's advance centre on the layout
/// baseline) to the tangent `t`, and moves the anchor to `on_path` shifted by `normal_shift` px along the glyph's
/// own down direction (Align to Path).
#[allow(clippy::too_many_arguments)]
pub fn place_on_path(
    pen: (f64, f64),
    c: (f64, f64),
    lin: [f32; 4],
    off: [f32; 2],
    anchor: (f64, f64),
    on_path: (f64, f64),
    t: (f64, f64),
    normal_shift: f64,
) -> ([f32; 4], [f32; 2]) {
    let (cos, sin) = (t.0, t.1);
    let [a, cc, b, d] = lin.map(|v| v as f64);
    // R · M, with M = [[a, cc], [b, d]] and R = [[cos, −sin], [sin, cos]].
    let lin2 = [
        (cos * a - sin * b) as f32,
        (cos * cc - sin * d) as f32,
        (sin * a + cos * b) as f32,
        (sin * cc + cos * d) as f32,
    ];
    // G(q) = A' + R(F(q) − P), F(q) = pen + M(q − C) + C + off  ⇒  off' = A' + R(pen + C + off − P) − pen − C.
    let target = (
        on_path.0 - sin * normal_shift,
        on_path.1 + cos * normal_shift,
    );
    let v = (
        pen.0 + c.0 + off[0] as f64 - anchor.0,
        pen.1 + c.1 + off[1] as f64 - anchor.1,
    );
    let rv = (cos * v.0 - sin * v.1, sin * v.0 + cos * v.1);
    let off2 = [
        (target.0 + rv.0 - pen.0 - c.0) as f32,
        (target.1 + rv.1 - pen.1 - c.1) as f32,
    ];
    (lin2, off2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 32 px type throughout.
    const SZ32: &PathSizes = &[(0.0, 32.0)];

    #[test]
    fn measure_and_wrap() {
        let m =
            PathMeasure::new(&[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)], false, false).unwrap();
        assert!((m.length() - 200.0).abs() < 1e-9);
        let (p, t) = m.at(150.0).unwrap();
        assert!((p.0 - 100.0).abs() < 1e-9 && (p.1 - 50.0).abs() < 1e-9);
        assert!((t.0).abs() < 1e-9 && (t.1 - 1.0).abs() < 1e-9);
        assert!(m.at(201.0).is_none());
        let closed = PathMeasure::new(
            &[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
            true,
            false,
        )
        .unwrap();
        assert!((closed.length() - 400.0).abs() < 1e-9);
        let (p, _) = closed.at(450.0).unwrap();
        assert!((p.0 - 50.0).abs() < 1e-9 && p.1.abs() < 1e-9);
        // Reversed: starts at the old end, runs back.
        let r = PathMeasure::new(&[(0.0, 0.0), (100.0, 0.0)], false, true).unwrap();
        let (p, t) = r.at(10.0).unwrap();
        assert!((p.0 - 90.0).abs() < 1e-9 && (t.0 + 1.0).abs() < 1e-9);
    }

    #[test]
    fn closed_text_ink_clips_at_the_seam_and_overset_alignment_keeps_the_start_bracket() {
        let m = PathMeasure::new(&[(0.0,0.0),(100.0,0.0),(100.0,100.0),(0.0,100.0)], true, false).unwrap();
        assert!(!m.fits_closed_glyph(395.0,10.0));
        assert!(!m.fits_closed_glyph(5.0,10.0));
        assert!(m.fits_closed_glyph(390.0,10.0));
        assert!(!m.fits_closed_glyph(410.0,5.0));
        for align in ["left","center","right"] {
            assert_eq!(start_distance(&m,0.75,align,600.0,&[(0.0,20.0)]),300.0);
        }
    }

    #[test]
    fn flatten_curves_and_first_subpath() {
        let pts = flatten(&[
            PathCmd::Move(0.0, 0.0),
            PathCmd::Cubic(0.0, 50.0, 100.0, 50.0, 100.0, 0.0),
            PathCmd::Move(500.0, 500.0),
            PathCmd::Line(600.0, 500.0),
        ]);
        assert!(pts.len() > 5);
        assert_eq!(pts[0], (0.0, 0.0));
        assert_eq!(*pts.last().unwrap(), (100.0, 0.0));
        // The midpoint of that symmetric cubic is (50, 37.5).
        let mid = pts
            .iter()
            .min_by(|a, b| (a.0 - 50.0).abs().partial_cmp(&(b.0 - 50.0).abs()).unwrap())
            .unwrap();
        assert!((mid.1 - 37.5).abs() < 1.0, "{mid:?}");
    }

    /// The upper half of a circle of radius `r` about (0, 0), left to right over the top (clockwise on screen, so
    /// type set on it sits outside the bend), as two quarter cubics.
    fn upper_semicircle(r: f64) -> Vec<(f64, f64)> {
        let k = 0.552_284_749_8 * r;
        flatten(&[
            PathCmd::Move(-r, 0.0),
            PathCmd::Cubic(-r, -k, -k, -r, 0.0, -r),
            PathCmd::Cubic(k, -r, r, -k, r, 0.0),
        ])
    }

    #[test]
    fn curvature_spacing_on_arcs() {
        for r in [150.0, 300.0] {
            let pts = upper_semicircle(r);
            for size in [32.0, 64.0] {
                // flip: the same arc walked the other way, type on its inside.
                for flip in [false, true] {
                    let m = PathMeasure::new(&pts, false, flip).unwrap();
                    let (k, c) = if flip {
                        (-1.0 / r, CURVE_SPACING_CONCAVE)
                    } else {
                        (1.0 / r, CURVE_SPACING_CONVEX)
                    };
                    let mid = m.length() * 0.5;
                    assert!(
                        (m.curvature(mid, size * 0.5) - k).abs() < 0.03 / r,
                        "κ at R {r}: {}", m.curvature(mid, size * 0.5) * r
                    );
                    let start = m.length() * 0.2;
                    let advance = m.length() * 0.4;
                    let along = PathSpacing::new(&m, start, &[(0.0, size)]).distance(advance) - start;
                    let ratio = advance / along;
                    let want = 1.0 + k * c * size;
                    assert!(
                        (ratio - want).abs() < 0.002,
                        "R {r} size {size} flip {flip}: advance/distance {ratio}, want {want}"
                    );
                    // Outside the bend the glyphs sit closer along the path than their advances; inside, further.
                    assert_eq!(along < advance, !flip);
                }
            }
        }
    }

    #[test]
    fn straight_segments_and_corners_keep_plain_advances() {
        // A polyline with right-angle corners, open and closed: exactly the advances, even past the corners.
        let sq = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)];
        for closed in [false, true] {
            let m = PathMeasure::new(&sq, closed, false).unwrap();
            assert_eq!(m.curvature(200.0, 32.0), 0.0);
            let mut sp = PathSpacing::new(&m, 15.0, &[(0.0, 32.0)]);
            for a in [0.0, 3.5, 150.0, 190.0, 210.0, 399.0] {
                assert_eq!(sp.distance(a), 15.0 + a);
            }
        }
        // Straight runs of a path that also curves: plain advances well away from the bend.
        let mut pts = vec![(-400.0, 0.0)];
        pts.extend(upper_semicircle(100.0).into_iter().map(|(x, y)| (x - 100.0, y)));
        let m = PathMeasure::new(&pts, false, false).unwrap();
        assert!(m.curvature(100.0, 32.0).abs() < 1e-12);
        assert!((PathSpacing::new(&m, 10.0, &[(0.0, 32.0)]).distance(150.0) - 160.0).abs() < 1e-9);
    }

    #[test]
    fn curvature_wraps_around_a_closed_path() {
        let r = 120.0;
        let k = 0.552_284_749_8 * r;
        let ring = flatten(&[
            PathCmd::Move(-r, 0.0),
            PathCmd::Cubic(-r, -k, -k, -r, 0.0, -r),
            PathCmd::Cubic(k, -r, r, -k, r, 0.0),
            PathCmd::Cubic(r, k, k, r, 0.0, r),
            PathCmd::Cubic(-k, r, -r, k, -r, 0.0),
        ]);
        let m = PathMeasure::new(&ring, true, false).unwrap();
        for s in [0.0, 1.0, m.length() - 1.0, m.length() * 0.5, m.length() * 3.25] {
            assert!((m.curvature(s, 16.0) - 1.0 / r).abs() < 0.02 / r, "at {s}");
        }
        // Order-independent: a query after a later one gives the same distance.
        let mut sp = PathSpacing::new(&m, m.length() - 40.0, SZ32);
        let late = sp.distance(300.0);
        let early = sp.distance(100.5);
        assert_eq!(early, PathSpacing::new(&m, m.length() - 40.0, SZ32).distance(100.5));
        assert_eq!(late, PathSpacing::new(&m, m.length() - 40.0, SZ32).distance(300.0));
    }

    #[test]
    fn centred_and_right_aligned_text_fits_along_the_bend() {
        let m = PathMeasure::new(&upper_semicircle(150.0), false, false).unwrap();
        let (w, size) = (200.0, 48.0);
        let sizes = [(0.0, size)];
        let right = start_distance(&m, 0.0, "right", w, &sizes);
        let end = PathSpacing::new(&m, right, &sizes).distance(w);
        assert!((end - m.length()).abs() < 1e-3, "{end} vs {}", m.length());
        let centre = start_distance(&m, 0.0, "center", w, &sizes);
        let end = PathSpacing::new(&m, centre, &sizes).distance(w);
        assert!((centre - (m.length() - end)).abs() < 1e-3);
    }

    #[test]
    fn placement_rotates_about_the_anchor() {
        // Identity glyph, pen (10, 0), C (5, -5); anchor at the pen + 5 on the baseline; a vertical tangent (down)
        // at (100, 100): the anchor lands on (100, 100) and the glyph's x axis points down.
        let (lin, off) = place_on_path(
            (10.0, 0.0),
            (5.0, -5.0),
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
            (15.0, 0.0),
            (100.0, 100.0),
            (0.0, 1.0),
            0.0,
        );
        let draw = |q: (f64, f64)| {
            let [a, c, b, d] = lin.map(|v| v as f64);
            let (px, py) = (q.0 - 5.0, q.1 + 5.0);
            (
                10.0 + a * px + c * py + 5.0 + off[0] as f64,
                b * px + d * py - 5.0 + off[1] as f64,
            )
        };
        let anchor = draw((5.0, 0.0));
        assert!(
            (anchor.0 - 100.0).abs() < 1e-4 && (anchor.1 - 100.0).abs() < 1e-4,
            "{anchor:?}"
        );
        let right = draw((15.0, 0.0));
        assert!(
            (right.0 - 100.0).abs() < 1e-4 && (right.1 - 110.0).abs() < 1e-4,
            "{right:?}"
        );
        // Align ascender: the anchor moves 8px along the glyph's down (here −x).
        let (_, off2) = place_on_path(
            (10.0, 0.0),
            (5.0, -5.0),
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
            (15.0, 0.0),
            (100.0, 100.0),
            (0.0, 1.0),
            8.0,
        );
        assert!((off2[0] - off[0] + 8.0).abs() < 1e-4);
    }

    /// The TS twin (`TextPathSpacing`, packages/editor/test/textPathSpacing.test.ts) asserts these same numbers.
    #[test]
    fn twin_reference_values() {
        let m = PathMeasure::new(&upper_semicircle(150.0), false, false).unwrap();
        let f = PathMeasure::new(&upper_semicircle(150.0), false, true).unwrap();
        let mut sp = PathSpacing::new(&m, 12.5, SZ32);
        let mut fp = PathSpacing::new(&f, 12.5, SZ32);
        let got = [
            sp.distance(7.25),
            sp.distance(100.0),
            sp.distance(333.3),
            fp.distance(7.25),
            fp.distance(100.0),
            fp.distance(333.3),
            m.curvature(200.0, 16.0),
            start_distance(&m, 0.1, "right", 180.0, &[(0.0, 40.0)]),
            start_distance(&f, 0.1, "center", 180.0, &[(0.0, 40.0)]),
        ];
        let want = [
            19.49716014482633,
            108.99348087981387,
            334.1277157638648,
            20.479615056638384,
            122.61116077967428,
            379.4613515651582,
            0.006711429537130196,
            299.0369636036636,
            157.55082450113775,
        ];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-9, "{got:?}");
        }
    }

    /// The twin fixture's paths, as `PathPoint` JSON (the editor's, which the TS twin flattens).
    fn twin_cases() -> Vec<serde_json::Value> {
        use serde_json::json;
        let mv = |x: f64, y: f64| json!({ "type": "move", "x": x, "y": y });
        let ln = |x: f64, y: f64| json!({ "type": "line", "x": x, "y": y });
        let cu = |c1: (f64, f64), c2: (f64, f64), x: f64, y: f64| {
            json!({ "type": "curve", "x": x, "y": y, "cp1": { "x": c1.0, "y": c1.1 }, "cp2": { "x": c2.0, "y": c2.1 } })
        };
        // An ellipse about (c, d) from its leftmost point, clockwise on screen over the top.
        let ellipse = |c: f64, d: f64, rx: f64, ry: f64| {
            let (kx, ky) = (0.552_284_749_8 * rx, 0.552_284_749_8 * ry);
            vec![
                mv(c - rx, d),
                cu((c - rx, d - ky), (c - kx, d - ry), c, d - ry),
                cu((c + kx, d - ry), (c + rx, d - ky), c + rx, d),
                cu((c + rx, d + ky), (c + kx, d + ry), c, d + ry),
                cu((c - kx, d + ry), (c - rx, d + ky), c - rx, d),
            ]
        };
        // Glyph anchors: the advance centres of `n` glyphs of varied widths at `size`.
        let anchors = |n: usize, size: f64| -> (Vec<f64>, f64) {
            let mut pen = 0.0;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let w = size * (0.3 + 0.05 * ((i * 7) % 11) as f64);
                out.push(((pen + w * 0.5) * 1e6_f64).round() / 1e6);
                pen += w;
            }
            (out, (pen * 1e6_f64).round() / 1e6)
        };
        #[allow(clippy::too_many_arguments)]
        fn case(
            name: &str,
            path: Vec<serde_json::Value>,
            closed: bool,
            flip: bool,
            offset: f64,
            align: &str,
            sizes: Vec<(f64, f64)>,
            (a, width): (Vec<f64>, f64),
        ) -> serde_json::Value {
            serde_json::json!({
                "name": name, "path": path, "closed": closed, "flip": flip, "startOffset": offset, "align": align,
                "sizes": sizes.iter().map(|&(a, s)| [a, s]).collect::<Vec<_>>(), "textWidth": width, "anchors": a,
            })
        }
        let k = 0.552_284_749_8 * 150.0;
        let rect = vec![mv(0.0, 0.0), ln(400.0, 0.0), ln(400.0, 200.0), ln(0.0, 200.0), ln(0.0, 0.0)];
        let arc = vec![mv(-150.0, 0.0), cu((-150.0, -k), (-k, -150.0), 0.0, -150.0), cu((k, -150.0), (150.0, -k), 150.0, 0.0)];
        let wave = vec![mv(30.0, 170.0), cu((110.0, 40.0), (410.0, 300.0), 490.0, 170.0)];
        vec![
            case("circle r200, 200 glyphs", ellipse(300.0, 300.0, 200.0, 200.0), true, false, 0.0, "left", vec![(0.0, 11.0)], anchors(200, 11.0)),
            case("circle r200 flipped", ellipse(300.0, 300.0, 200.0, 200.0), true, true, 0.0, "left", vec![(0.0, 11.0)], anchors(200, 11.0)),
            case("ellipse, offset 30 %, centred", ellipse(300.0, 200.0, 260.0, 120.0), true, false, 0.3, "center", vec![(0.0, 24.0)], anchors(40, 24.0)),
            case("rectangle, offset 10 %, right", rect, true, false, 0.1, "right", vec![(0.0, 20.0)], anchors(40, 20.0)),
            case("arc, mixed sizes", arc, false, false, 0.05, "left", vec![(0.0, 48.0), (150.0, 14.0), (300.0, 30.0)], anchors(30, 20.0)),
            case("wave, centred", wave, false, false, 0.0, "center", vec![(0.0, 32.0)], anchors(20, 32.0)),
        ]
    }

    fn twin_cmds(path: &serde_json::Value) -> Vec<PathCmd> {
        let n = |v: &serde_json::Value| v.as_f64().unwrap();
        path.as_array()
            .unwrap()
            .iter()
            .map(|p| match p["type"].as_str().unwrap() {
                "move" => PathCmd::Move(n(&p["x"]), n(&p["y"])),
                "line" => PathCmd::Line(n(&p["x"]), n(&p["y"])),
                _ => PathCmd::Cubic(
                    n(&p["cp1"]["x"]),
                    n(&p["cp1"]["y"]),
                    n(&p["cp2"]["x"]),
                    n(&p["cp2"]["y"]),
                    n(&p["x"]),
                    n(&p["y"]),
                ),
            })
            .collect()
    }

}
