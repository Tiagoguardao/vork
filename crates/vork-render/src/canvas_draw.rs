//! The 2D canvas drawn by Moda's 2D engine (crates `moda-engine`, `moda-scene`, `moda-assets`): the canvas is
//! lowered once to an installed Moda scene (a page the size of the output, its background the page fill), and
//! each frame writes its changed fields into the scene rows bound to them.
//! Moda plans and draws its nodes using the editor's per-node caches. vork's names and meanings are SVG 2's
//! and CSS's; this module spells them in Moda's document model:
//!
//! - a `path` is a Moda path node: its `d` as Moda's path points, `fill` and `stroke` as one fill layer
//!   and one stroke layer (a CSS colour as a solid paint, a CSS gradient as Moda's linear, radial or conic
//!   gradient over the path's bounding box, an image as an image paint), the `stroke-*` properties as the
//!   stroke's width, cap, join, miter limit and dashes (SVG px; Moda scales a dash pattern by half the
//!   stroke width, so lengths are divided by it), `fill-opacity` and `stroke-opacity` as the layers'
//!   opacity; `paint-order: stroke` draws the stroke node under the fill node.
//! - a node's `opacity` is the Moda node's opacity (a group composites as one unit, as CSS opacity does);
//!   its `filter` (`drop-shadow()`, `blur()`) is the Moda node's filter-shadow and layer-blur effects, both
//!   layer filters over the node's rendered picture, children included. CSS gives both functions a standard
//!   deviation and Moda's blur and radius are twice it. A CSS drop shadow is the picture's own alpha, offset,
//!   blurred and tinted, drawn under the picture: Moda's `filter-shadow`, not its `drop-shadow` (Figma's,
//!   cast from the opaque shape).
//! - `clip-path` and `mask` wrap the node in a Moda clip-mask group whose mask child is a copy of the named
//!   node in the node's coordinates: a vector mask (its fill geometry) for `clip-path`, a luminance mask for
//!   `mask`. SVG's order holds: filter, then clip, then mask, then opacity.
//! - a `text` node is a Moda richtext node of its laid-out box, which Moda's frame walk draws through the
//!   builder vork installs (`moda_engine::text_draw::install_text_builder`): vork's text spec, drawn by
//!   moda-text-draw at the model matrix, z and clip Moda computes. Its `filter` is on a group around it.
//! - `viewBox` and `preserveAspectRatio` are a root group's scale and offset.
//! - a world (`composition: {space: "3d"}`) is a Moda path node, a rectangle the size of the canvas's box, filled
//!   with the world's picture: a GPU texture registered with Moda's engine (`canvas_world`).
use anyhow::{Context, Result, bail, ensure};
use moda_scene::path_points::{CompactKind, CompactPoints};
use moda_scene::value::{K, Map, Number, RecordLayout, Value, json};
use std::sync::Arc;
use std::{cell::RefCell, collections::{BTreeMap, BTreeSet}, path::{Path, PathBuf}, rc::Rc};
use vork_scene::v2;

/// A canvas node as Moda draws it at one sampled instant.
#[derive(Clone, PartialEq)]
pub(crate) struct Node {
    pub parent: Option<usize>,
    pub clip_path: Option<usize>,
    pub mask: Option<usize>,
    pub mask_type: v2::MaskType,
    pub blend: v2::MixBlendMode,
    pub isolation: v2::Isolation,
    pub filter: Filter,
    pub kind: Kind,
}
#[derive(Clone, PartialEq)]
pub(crate) enum Kind {
    Group,
    /// The laid-out box (px).
    Text { size: [f64; 2] },
    Path(std::sync::Arc<PathItem>),
    /// A world (`composition: {space: "3d"}`): its picture fills this box `[x0, y0, x1, y1]`, the canvas's box.
    World { bounds: [f64; 4] },
}
/// One node's animated values at one instant.
#[derive(Clone, PartialEq)]
pub(crate) struct Values {
    pub style_revision: u64,
    pub transform: v2::Transform,
    pub visible: bool,
    pub opacity: f32,
    pub path: Option<PathValues>,
    /// How many batches a text node draws at this instant (styled paints split a node into several).
    pub texts: usize,
    pub text_composition: Vec<TextPart>,
    /// A world's picture at this instant: the image source Moda reads it from, and its size in px.
    pub world: Option<(String, [u32; 2])>,
    /// Content identity for the host's external text specs; Moda includes it in its text cache key.
    pub text_revision: u64,
    /// Picture revision for externally held paint values, independent of glyph geometry.
    pub text_paint_revisions: TextPaintRevisions,
}
pub(crate) type TextPaintRevisions = smallvec::SmallVec<[u64; 1]>;
/// Text-unit pictures, using Moda's ordinary group-opacity composition.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TextPart {
    Batch(usize),
    Paint { sub: usize, paint: Value, size: [f64; 2], bounds: [f64; 4] },
    Group { opacity: f32, children: Vec<TextPart> },
}
/// A path's keyed fields at one instant.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct PathValues {
    /// A keyed fill or stroke colour (straight sRGB-encoded RGBA), replacing the static paint.
    pub fill: Option<[f32; 4]>,
    pub stroke: Option<[f32; 4]>,
    pub stroke_width: f32,
    pub stroke_dashoffset: f32,
    pub fill_opacity: f32,
    pub stroke_opacity: f32,
}

/// A path's geometry as Moda holds it: compact points (kinds and coordinates in flat arrays), parsed from `d`
/// once and shared by every Moda node that draws the path. A path with an arc keeps Moda's point objects (Moda
/// has no compact arc).
#[derive(Clone)]
pub(crate) enum Points {
    Compact(Arc<CompactPoints>),
    Json(Vec<Value>),
}
impl Points {
    fn is_empty(&self) -> bool {
        match self { Self::Compact(points) => points.is_empty(), Self::Json(points) => points.is_empty() }
    }
    /// Moda's point objects.
    #[cfg(test)]
    fn json(&self) -> Vec<Value> {
        match self { Self::Compact(points) => points.to_json(), Self::Json(points) => points.clone() }
    }
    /// What a node's `points` field holds: the point objects, or `null` where compact points take their place.
    fn field(&self) -> Value {
        match self { Self::Compact(_) => Value::Null, Self::Json(points) => Value::Array(points.clone()) }
    }
}
impl PartialEq for Points {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Compact(a), Self::Compact(b)) => Arc::ptr_eq(a, b) || a == b,
            (Self::Json(a), Self::Json(b)) => a == b,
            _ => false,
        }
    }
}

/// A path's static data: Moda's points and the resolved paints.
#[derive(Clone, PartialEq)]
pub(crate) struct PathItem {
    points: Points,
    closed: Value,
    /// The geometric bounds `[x0, y0, x1, y1]` of the path (Moda's `tess::path_bounds`), the box a gradient or
    /// an image is placed in.
    bounds: Option<[f64; 4]>,
    /// Measured contour length in local units for SVG `pathLength` calibration.
    length: f64,
    fill: PaintSource,
    stroke: PaintSource,
    fill_rule: v2::FillRule,
    linecap: v2::LineCap,
    linejoin: v2::LineJoin,
    miterlimit: f32,
    dasharray: Vec<f32>,
    sampled_dash: Option<moda_scene::dash::DashList>,
    path_length: Option<f32>,
    stroke_first: bool,
    pub(super) paint_clock: u32,
    /// The static values the tracks key from.
    pub values: PathValues,
}
impl PathItem {
    pub(crate) fn measured_length(&self) -> f64 {
        if self.path_length.is_some() {
            self.length
        } else {
            let cmds = match &self.points {
                Points::Compact(points) => moda_engine::tess::moda_engine_path_types::cmds_from_path_points(
                    moda_scene::path_points::PathPoints::Compact(points),
                ),
                Points::Json(points) => moda_engine::tess::moda_engine_path_types::cmds_from_node_points(points),
            };
            measure_path(&cmds, &self.closed)
        }
    }
    pub(crate) fn set_dash(&mut self, dash: &moda_scene::dash::DashList) {
        match dash {
            moda_scene::dash::DashList::Lengths(lengths) => {
                self.dasharray = lengths.iter().map(|value| *value as f32).collect();
            }
            moda_scene::dash::DashList::Mix(..) => self.sampled_dash = Some(dash.clone()),
        }
    }

    fn stroke_coverage(&self) -> bool {
        matches!(&self.stroke, PaintSource::Moda(p) if matches!(p["type"].as_str(), Some("image" | "conic")) && !missing_media(p))
    }
}
/// A paint resolved against its path and the installed material table.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PaintSource {
    None,
    Color([f32; 4]),
    Program(moda_scene::paint::PathProgram),
    Moda(Value),
}

/// What a paint keeps while its values change: a solid colour, or a gradient of one type, repeat mode,
/// interpolation space and stop count (the gradients that interpolate into each other).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PaintForm {
    Solid,
    Gradient { kind: Value, repeat: Value, interpolation: Value, stops: usize },
}
impl PaintForm {
    /// The form of Moda gradient paint JSON; `None` for an image.
    pub(crate) fn of_gradient(paint: &Value) -> Option<Self> {
        Some(Self::Gradient { kind: paint["type"].clone(), repeat: paint["repeat"].clone(),
            interpolation: paint["interpolation"].clone(), stops: paint["stops"].as_array()?.len() })
    }
}

impl PathItem {
    /// Final fill and stroke consumed by the picture writer and soundtrack.
    pub(crate) fn paints(&self, values: &PathValues) -> [std::borrow::Cow<'_, PaintSource>; 2] {
        use std::borrow::Cow;
        let resolve = |color: Option<[f32; 4]>, source| match color {
            Some(color) => Cow::Owned(PaintSource::Color(color)),
            None => Cow::Borrowed(source),
        };
        [resolve(values.fill, &self.fill), if values.stroke_width > 0. {
            resolve(values.stroke, &self.stroke)
        } else { Cow::Borrowed(&PaintSource::None) }]
    }

    /// The form of the path's fill (or stroke) at rest; `None` without a paint, or with an image.
    pub(crate) fn paint_form(&self, fill: bool) -> Option<PaintForm> {
        match if fill { &self.fill } else { &self.stroke } {
            PaintSource::Color(_) => Some(PaintForm::Solid),
            PaintSource::Moda(paint) => PaintForm::of_gradient(paint),
            PaintSource::None | PaintSource::Program(_) => None,
        }
    }
}

impl PaintSource {
    pub fn colour(&self) -> Option<[f32; 4]> { match self { Self::Color(c) => Some(*c), _ => None } }
    pub fn json(&self) -> Option<Value> { match self {
        Self::None => None,
        Self::Color(c) => Some(json!({"type": "solid", "color": vork_std::moda::moda_color(*c)})),
        Self::Program(binding) => Some(program_record(*binding)),
        Self::Moda(p) => Some(p.clone()),
    } }
}

fn program_record(binding: moda_scene::paint::PathProgram) -> Value {
    static LAYOUT: std::sync::LazyLock<RecordLayout<6>> = std::sync::LazyLock::new(|| {
        RecordLayout::new([K::Type, K::Unit, K::Inputs, K::StrokeUv, K::Clock, K::Site])
    });
    LAYOUT.record(["program".into(), binding.unit.into(), binding.inputs.into(),
        binding.stroke_uv.into(), binding.clock.into(), binding.site.into()])
}

pub(crate) fn resolve_paint(images: &v2::Table<v2::Image>, base: &Path, paint: &v2::Paint, bounds: Option<[f64; 4]>, here: &str, media: &Media) -> Result<PaintSource> {
    Ok(match paint {
        v2::Paint::None(_) => PaintSource::None,
        v2::Paint::Material(p) => media.program(&p.material)?,
        v2::Paint::Css(text) if text.contains("gradient(") => {
            let Some(bounds) = bounds else { return Ok(PaintSource::None) };
            PaintSource::Moda(gradient_paint(text, bounds).with_context(|| here.to_string())?)
        }
        v2::Paint::Css(text) => v2::text::paint(text).with_context(|| here.to_string())?.map_or(PaintSource::None, PaintSource::Color),
        v2::Paint::Image(image) => {
            let Some(bounds) = bounds else { return Ok(PaintSource::None) };
            PaintSource::Moda(image_paint(images, base, image, bounds, media).with_context(|| format!("{here} image {:?}", image.image))?)
        }
    })
}

/// Keep a paint in its authored box while its coverage (stroke or animated glyphs) extends beyond it.
fn expanded_paint(paint: &Value, [px0, py0, px1, py1]: [f64; 4], [x0, y0, x1, y1]: [f64; 4]) -> Value {
    let (w, h) = (px1 - px0, py1 - py0);
    let mut paint = paint.clone();
    let (bw, bh) = ((x1 - x0).max(1e-9), (y1 - y0).max(1e-9));
    for field in ["start", "end", "center"] {
        if let Some(p) = paint.get_mut(field) {
            if let (Some(x), Some(y)) = (p["x"].as_f64(), p["y"].as_f64()) {
                p["x"] = json!((x * w / 100. + px0 - x0) / bw * 100.);
                p["y"] = json!((y * h / 100. + py0 - y0) / bh * 100.);
            }
        }
    }
    for (field, ratio) in [("radiusX", w / bw), ("radiusY", h / bh)] {
        if let Some(radius) = paint[field].as_f64() { paint[field] = json!(radius * ratio); }
    }
    if paint["type"] == "image" {
        let placement = &mut paint["placement"];
        let nw = placement["naturalSize"]["width"].as_f64().unwrap_or(1.);
        let nh = placement["naturalSize"]["height"].as_f64().unwrap_or(1.);
        let fit = placement["fit"].as_str().unwrap_or("tile");
        let scale = match fit {
            "contain" => Some([(w / nw).min(h / nh); 2]),
            "cover" => Some([(w / nw).max(h / nh); 2]),
            "fill" => Some([w / nw, h / nh]),
            _ => None,
        };
        if let Some([sx, sy]) = scale {
            placement["fit"] = json!("crop");
            placement["scale"] = json!({"x": sx, "y": sy});
            placement["offset"] = json!({"x": px0 + (w - nw * sx) / 2., "y": py0 + (h - nh * sy) / 2.});
        }
    }
    paint
}

#[path = "canvas_color_filter.rs"]
mod color_filter;
pub(crate) use color_filter::ColorFunction;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FilterOp {
    Blur(f64),
    Shadow([f64; 2], f64, [f32; 4]),
    Color(ColorFunction, f64),
}
impl FilterOp {
    pub fn identity(self) -> Self {
        match self {
            Self::Blur(_) => Self::Blur(0.),
            Self::Shadow(..) => Self::Shadow([0.;2], 0., [0.;4]),
            Self::Color(f,_) => Self::Color(f,f.identity()),
        }
    }
    pub fn same_kind(&self, other: &Self) -> bool {
        match (self,other) {
            (Self::Color(f,_),Self::Color(g,_)) => f == g,
            _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
    }
}
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Filter(pub Vec<FilterOp>);

impl Filter {
    fn single_capture(&self) -> bool {
        matches!(self.0.as_slice(), [] | [_] | [FilterOp::Shadow(..), FilterOp::Blur(_)])
    }
    fn has_shadow(&self) -> bool { self.0.iter().any(|op| matches!(op, FilterOp::Shadow(..))) }
}

pub(crate) fn parse_filter(text: &str) -> Result<Filter> {
    let mut filter = Filter::default();
    let text = text.trim();
    ensure!(!text.is_empty(), "filter: write CSS filter functions or none");
    if text == "none" {
        return Ok(filter);
    }
    let mut rest = text;
    while !rest.is_empty() {
        let open = rest.find('(').with_context(|| format!("filter {text:?}: write CSS filter functions such as blur(4px) drop-shadow(2px 2px 4px black), or none"))?;
        let name = rest[..open].trim();
        let close = matching_paren(rest, open).with_context(|| format!("filter {text:?}: {name}( is not closed"))?;
        let args = &rest[open + 1..close];
        match name {
            "blur" => {
                let args = args.trim();
                let sigma = if args.is_empty() { 0. } else { css_length(args).with_context(|| format!("filter {text:?}: blur() takes a length in px"))? };
                ensure!(sigma >= 0., "filter {text:?}: blur() takes a nonnegative length");
                ensure!((2. * sigma).is_finite(), "filter {text:?}: blur() radius exceeds the numeric range");
                filter.0.push(FilterOp::Blur(sigma));
            }
            "drop-shadow" => {
                let mut lengths = Vec::new();
                let mut color = None;
                for token in split_top_level(args, ' ') {
                    match css_length(&token) {
                        Ok(length) if color.is_none() || lengths.is_empty() || lengths.len() < 3 => lengths.push(length),
                        Err(error) if token.chars().next().is_some_and(|c| c.is_ascii_digit() || matches!(c, '+' | '-' | '.')) || token.ends_with("px") => {
                            return Err(error.context(format!("filter {text:?}: drop-shadow()")));
                        }
                        _ => {
                            ensure!(color.is_none(), "filter {text:?}: drop-shadow() takes one colour");
                            color = Some(vork_std::moda::css_color(&token).with_context(|| format!("filter {text:?}: drop-shadow()"))?);
                        }
                    }
                }
                ensure!(matches!(lengths.len(), 2 | 3), "filter {text:?}: drop-shadow() takes an x and y offset, an optional standard deviation and an optional colour");
                let sigma = lengths.get(2).copied().unwrap_or(0.);
                ensure!(sigma >= 0., "filter {text:?}: drop-shadow()'s standard deviation must not be negative");
                ensure!((2. * sigma).is_finite(), "filter {text:?}: drop-shadow() standard deviation exceeds the numeric range");
                // CSS: an omitted colour is currentcolor; a canvas node has no current colour but black.
                filter.0.push(FilterOp::Shadow([lengths[0], lengths[1]], sigma, color.unwrap_or([0., 0., 0., 1.])));
            }
            other => {
                let function = ColorFunction::parse(other).with_context(|| format!("unknown filter function {other}()"))?;
                filter.0.push(FilterOp::Color(function, function.argument(args)?));
            }
        }
        rest = rest[close + 1..].trim_start();
    }
    Ok(filter)
}

fn matching_paren(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in text[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + i);
                }
            }
            _ => {}
        }
    }
    None
}

/// `text` split at `separator` outside parentheses, empty pieces dropped.
fn split_top_level(text: &str, separator: char) -> Vec<String> {
    let (mut out, mut current, mut depth) = (Vec::new(), String::new(), 0);
    for c in text.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ => {}
        }
        if (c == separator || (separator == ' ' && c.is_whitespace())) && depth == 0 {
            if !current.trim().is_empty() {
                out.push(current.trim().to_string());
            }
            current.clear();
        } else {
            current.push(c);
        }
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

/// A CSS length in px (`4px`, or a bare `0`).
fn css_length(token: &str) -> Result<f64> {
    let token = token.trim();
    let number = token.strip_suffix("px").or_else(|| (token.parse::<f64>().ok() == Some(0.)).then_some(token))
        .with_context(|| format!("{token:?} is not a CSS length; write nonzero lengths in px"))?;
    let value: f64 = number.parse().with_context(|| format!("{token:?} is not a length in px; write a number followed by px"))?;
    ensure!(value.is_finite(), "{token:?} is not a finite length");
    Ok(value)
}

/// A CSS angle in degrees (`deg`, `rad`, `grad`, `turn`, or a bare `0`).
fn css_angle(token: &str) -> Option<f64> {
    let token = token.trim();
    let unit = |suffix: &str, scale: f64| token.strip_suffix(suffix).and_then(|n| n.parse::<f64>().ok()).map(|n| n * scale);
    unit("deg", 1.).or_else(|| unit("grad", 0.9)).or_else(|| unit("rad", 180. / std::f64::consts::PI)).or_else(|| unit("turn", 360.))
        .or_else(|| (token.parse::<f64>().ok() == Some(0.)).then_some(0.)).filter(|v| v.is_finite())
}

/// A stop or position value: a fraction of a reference (`%`) or an absolute length.
#[derive(Clone, Copy, Debug)]
enum Position {
    Percent(f64),
    Px(f64),
}
impl Position {
    fn parse(token: &str) -> Option<Self> {
        let token = token.trim();
        if let Some(n) = token.strip_suffix('%') {
            return n.parse::<f64>().ok().filter(|v| v.is_finite()).map(Self::Percent);
        }
        css_length(token).ok().map(Self::Px)
    }
    fn resolve(self, reference: f64) -> f64 {
        match self {
            Self::Percent(p) => p / 100. * reference,
            Self::Px(v) => v,
        }
    }
}

/// A CSS `<color-stop-list>` with positions resolved to fractions of the gradient line (`unit` px long, or for a
/// conic gradient, of a turn) after CSS Images 3 §3.5.1's fix-up.
fn color_stops(stops: &[String], unit: f64, conic: bool, text: &str) -> Result<Vec<(f64, [f32; 4])>> {
    let mut list: Vec<(Option<f64>, [f32; 4])> = Vec::new();
    for stop in stops {
        let tokens = split_top_level(stop, ' ');
        let (color, positions) = tokens.split_first().with_context(|| format!("{text:?}: an empty colour stop"))?;
        let color = vork_std::moda::css_color(color).with_context(|| format!("{text:?}: a colour hint or a stop that does not start with a colour is not built"))?;
        ensure!(positions.len() <= 2, "{text:?}: a colour stop takes at most two positions");
        if positions.is_empty() {
            list.push((None, color));
        }
        for position in positions {
            let fraction = if conic {
                if let Some(p) = position.strip_suffix('%').and_then(|n| n.parse::<f64>().ok()) { p / 100. } else {
                    css_angle(position).with_context(|| format!("{text:?}: a conic stop's position is an angle or a percentage"))? / 360.
                }
            } else {
                Position::parse(position).with_context(|| format!("{text:?}: a stop's position is a length in px or a percentage"))?.resolve(unit) / unit.max(1e-9)
            };
            list.push((Some(fraction), color));
        }
    }
    ensure!(list.len() >= 2, "{text:?}: a gradient takes at least two colour stops");
    ensure!(list.len() <= moda_engine::paint::gradient::MAX_GRADIENT_STOPS,
        "{text:?}: Moda's engine draws at most {} colour stops", moda_engine::paint::gradient::MAX_GRADIENT_STOPS);
    if list[0].0.is_none() {
        list[0].0 = Some(0.);
    }
    let last = list.len() - 1;
    if list[last].0.is_none() {
        list[last].0 = Some(1.);
    }
    let mut max = f64::NEG_INFINITY;
    for stop in &mut list {
        if let Some(p) = &mut stop.0 {
            *p = p.max(max);
            max = *p;
        }
    }
    let mut i = 0;
    while i < list.len() {
        if list[i].0.is_some() {
            i += 1;
            continue;
        }
        let start = i - 1;
        let mut end = i;
        while list[end].0.is_none() {
            end += 1;
        }
        let (a, b) = (list[start].0.unwrap(), list[end].0.unwrap());
        for (k, stop) in list[start + 1..end].iter_mut().enumerate() {
            stop.0 = Some(a + (b - a) * (k + 1) as f64 / (end - start) as f64);
        }
        i = end;
    }
    Ok(list.into_iter().map(|(p, c)| (p.unwrap(), c)).collect())
}

/// CSS Color 4's `in <space>` for a gradient: the interpolation spaces Moda's gradient shaders draw.
fn interpolation(tokens: &[String], text: &str) -> Result<Option<&'static str>> {
    match tokens {
        [] => Ok(None),
        [space] => match space.as_str() {
            "srgb" => Ok(Some("srgb")),
            "srgb-linear" => Ok(Some("srgb-linear")),
            "oklab" => Ok(Some("oklab")),
            "oklch" => Ok(Some("oklch")),
            other => bail!("{text:?}: interpolation in {other} is not built; srgb, srgb-linear, oklab and oklch are"),
        },
        [space, hue, word] if space == "oklch" && hue == "shorter" && word == "hue" => Ok(Some("oklch")),
        _ => bail!("{text:?}: write `in srgb`, `in srgb-linear`, `in oklab` or `in oklch`"),
    }
}

/// A CSS `linear-gradient()`, `radial-gradient()` or `conic-gradient()` (or a `repeating-` one) over a box
/// `[x0, y0, x1, y1]`, as a Moda gradient paint: its points in percent of the box.
pub(crate) fn gradient_paint(text: &str, bounds: [f64; 4]) -> Result<Value> {
    let trimmed = text.trim();
    let open = trimmed.find('(').context("not a gradient")?;
    let (name, repeating) = match trimmed[..open].trim() {
        name @ ("linear-gradient" | "radial-gradient" | "conic-gradient") => (name, false),
        "repeating-linear-gradient" => ("linear-gradient", true),
        "repeating-radial-gradient" => ("radial-gradient", true),
        "repeating-conic-gradient" => bail!("{text:?}: repeating-conic-gradient() is not built"),
        other => bail!("{text:?}: {other}() is not built; linear-gradient(), radial-gradient() and conic-gradient() are"),
    };
    let close = matching_paren(trimmed, open).with_context(|| format!("{text:?}: unclosed gradient"))?;
    ensure!(trimmed[close + 1..].trim().is_empty(), "{text:?}: one gradient per paint");
    let args = split_top_level(&trimmed[open + 1..close], ',');
    let (w, h) = ((bounds[2] - bounds[0]).max(1e-6), (bounds[3] - bounds[1]).max(1e-6));
    // The first argument is the gradient's geometry when it does not start with a colour.
    let first_is_geometry = args.first().is_some_and(|a| {
        let first = split_top_level(a, ' ');
        first.first().is_some_and(|t| vork_std::moda::css_color(t).is_err())
    });
    let (geometry, stops) = if first_is_geometry { (split_top_level(&args[0], ' '), &args[1..]) } else { (Vec::new(), &args[..]) };
    // `in <space>` may close the geometry.
    let (geometry, space) = match geometry.iter().position(|t| t == "in") {
        Some(i) => (geometry[..i].to_vec(), interpolation(&geometry[i + 1..], text)?),
        None => (geometry, None),
    };
    let pct = |x: f64, y: f64| json!({"x": x / w * 100., "y": y / h * 100.});
    let mut paint = match name {
        "linear-gradient" => {
            // CSS Images 3 §3.1: an angle (0deg up, clockwise), or `to` a side or corner; default `to bottom`.
            let (dx, dy) = match geometry.as_slice() {
                [] => (0., 1.),
                [angle] => {
                    let a = css_angle(angle).with_context(|| format!("{text:?}: {angle:?} is not an angle or `to <side>`"))?.to_radians();
                    (a.sin(), -a.cos())
                }
                [to, sides @ ..] if to == "to" && matches!(sides.len(), 1 | 2) => {
                    let (mut sx, mut sy) = (0., 0.);
                    for side in sides {
                        match side.as_str() {
                            "left" => sx = -1.,
                            "right" => sx = 1.,
                            "top" => sy = -1.,
                            "bottom" => sy = 1.,
                            other => bail!("{text:?}: {other:?} is not a side"),
                        }
                    }
                    // A corner: the line perpendicular to the diagonal through the other two corners.
                    if sx != 0. && sy != 0. { (sx * h, sy * w) } else { (sx, sy) }
                }
                _ => bail!("{text:?}: a linear gradient starts with an angle or `to <side or corner>`"),
            };
            let norm = (dx * dx + dy * dy).sqrt();
            let (dx, dy) = (dx / norm, dy / norm);
            let length = (w * dx).abs() + (h * dy).abs();
            let (cx, cy) = (w / 2., h / 2.);
            let stops = color_stops(stops, length, false, text)?;
            json!({"type": "linear", "start": pct(cx - dx * length / 2., cy - dy * length / 2.),
                "end": pct(cx + dx * length / 2., cy + dy * length / 2.), "stops": moda_stops(&stops)})
        }
        "radial-gradient" => {
            // CSS Images 3 §3.2: [<shape> || <size>] [at <position>]; default an ellipse to the farthest corner, centred.
            let at = geometry.iter().position(|t| t == "at");
            let (shape_size, position) = match at { Some(i) => (&geometry[..i], &geometry[i + 1..]), None => (&geometry[..], &[][..]) };
            let (cx, cy) = css_position(position, w, h, text)?;
            let mut circle = None;
            let mut extent = None;
            let mut lengths = Vec::new();
            for token in shape_size {
                match token.as_str() {
                    "circle" => circle = Some(true),
                    "ellipse" => circle = Some(false),
                    k @ ("closest-side" | "farthest-side" | "closest-corner" | "farthest-corner") => extent = Some(k),
                    other => lengths.push(Position::parse(other).with_context(|| format!("{text:?}: {other:?} is not a radial gradient's shape or size"))?),
                }
            }
            let circle = circle.unwrap_or(lengths.len() == 1);
            let sides = [cx, w - cx, cy, h - cy].map(f64::abs);
            let (rx, ry) = match (lengths.as_slice(), extent.unwrap_or("farthest-corner")) {
                ([r], _) if circle => {
                    ensure!(matches!(r, Position::Px(_)), "{text:?}: a circle's radius is a length in px");
                    (r.resolve(0.), r.resolve(0.))
                }
                ([a, b], _) if !circle => (a.resolve(w), b.resolve(h)),
                ([], keyword) => {
                    let (near_x, far_x) = (sides[0].min(sides[1]), sides[0].max(sides[1]));
                    let (near_y, far_y) = (sides[2].min(sides[3]), sides[2].max(sides[3]));
                    match (keyword, circle) {
                        ("closest-side", true) => (near_x.min(near_y), near_x.min(near_y)),
                        ("farthest-side", true) => (far_x.max(far_y), far_x.max(far_y)),
                        ("closest-corner", true) => (near_x.hypot(near_y), near_x.hypot(near_y)),
                        ("farthest-corner", true) => (far_x.hypot(far_y), far_x.hypot(far_y)),
                        ("closest-side", false) => (near_x, near_y),
                        ("farthest-side", false) => (far_x, far_y),
                        // CSS: the ellipse with the side ratio that passes through the corner.
                        ("closest-corner", false) => (near_x * std::f64::consts::SQRT_2, near_y * std::f64::consts::SQRT_2),
                        _ => (far_x * std::f64::consts::SQRT_2, far_y * std::f64::consts::SQRT_2),
                    }
                }
                _ => bail!("{text:?}: a circle takes one radius, an ellipse two"),
            };
            let stops = color_stops(stops, rx, false, text)?;
            json!({"type": "radial", "center": pct(cx, cy), "radiusX": rx / w * 100., "radiusY": ry / h * 100., "rotation": 0,
                "stops": moda_stops(&stops)})
        }
        _ => {
            // CSS Images 4 §3.3: [from <angle>] [at <position>].
            let at = geometry.iter().position(|t| t == "at");
            let (from, position) = match at { Some(i) => (&geometry[..i], &geometry[i + 1..]), None => (&geometry[..], &[][..]) };
            let angle = match from {
                [] => 0.,
                [word, angle] if word == "from" => css_angle(angle).with_context(|| format!("{text:?}: {angle:?} is not an angle"))?,
                _ => bail!("{text:?}: a conic gradient starts with `from <angle>` and `at <position>`"),
            };
            let (cx, cy) = css_position(position, w, h, text)?;
            let stops = color_stops(stops, 1., true, text)?;
            json!({"type": "conic", "center": pct(cx, cy), "angle": angle, "stops": moda_stops(&stops)})
        }
    };
    if repeating {
        paint["repeat"] = json!(true);
    }
    if let Some(space) = space {
        paint["interpolation"] = json!(space);
    }
    Ok(paint)
}

fn moda_stops(stops: &[(f64, [f32; 4])]) -> Value {
    stops.iter().map(|(offset, color)| json!({"offset": offset, "color": vork_std::moda::moda_color(*color)})).collect()
}

/// A CSS `<position>` (one or two keywords, lengths or percentages) in a `w` × `h` box; omitted = the centre.
fn css_position(tokens: &[String], w: f64, h: f64, text: &str) -> Result<(f64, f64)> {
    let horizontal = |t: &str| match t { "left" => Some(0.), "center" => Some(0.5), "right" => Some(1.), _ => None };
    let vertical = |t: &str| match t { "top" => Some(0.), "center" => Some(0.5), "bottom" => Some(1.), _ => None };
    let value = |t: &str, size: f64| Position::parse(t).map(|p| p.resolve(size));
    match tokens {
        [] => Ok((w / 2., h / 2.)),
        [one] => {
            if let Some(y) = vertical(one).filter(|_| one != "center") { return Ok((w / 2., y * h)); }
            let x = horizontal(one).map(|f| f * w).or_else(|| value(one, w)).with_context(|| format!("{text:?}: {one:?} is not a position"))?;
            Ok((x, h / 2.))
        }
        [a, b] => {
            // Keywords may come in either order; lengths are x then y.
            if vertical(a).is_some() && a != "center" && horizontal(b).is_some() {
                return Ok((horizontal(b).unwrap() * w, vertical(a).unwrap() * h));
            }
            let x = horizontal(a).map(|f| f * w).or_else(|| value(a, w)).with_context(|| format!("{text:?}: {a:?} is not a horizontal position"))?;
            let y = vertical(b).map(|f| f * h).or_else(|| value(b, h)).with_context(|| format!("{text:?}: {b:?} is not a vertical position"))?;
            Ok((x, y))
        }
        _ => bail!("{text:?}: a position of more than two values is not built"),
    }
}

/// Moda's path points for path commands (`move`, `line`, `curve`, `quadratic`, `arc`, absolute): compact when
/// every command has a compact kind and finite coordinates, which is what Moda's own load makes of the point
/// objects (each coordinate written as a float); point objects otherwise.
fn moda_points(cmds: &[vork_std::moda::PathCmd]) -> Points {
    use vork_std::moda::PathCmd;
    let mut kinds = Vec::with_capacity(cmds.len());
    let mut coords = Vec::with_capacity(cmds.len() * 2);
    let compact = cmds.iter().all(|cmd| {
        match *cmd {
            PathCmd::Move { x, y } => { kinds.push(CompactKind::Move); coords.extend([x, y]); }
            PathCmd::Line { x, y } => { kinds.push(CompactKind::Line); coords.extend([x, y]); }
            PathCmd::Curve { x, y, cp1, cp2 } => { kinds.push(CompactKind::Curve); coords.extend([x, y, cp1.0, cp1.1, cp2.0, cp2.1]); }
            PathCmd::Quadratic { x, y, cp } => { kinds.push(CompactKind::Quadratic); coords.extend([x, y, cp.0, cp.1]); }
            PathCmd::Arc { .. } => return false,
        }
        true
    });
    if compact {
        if let Some(points) = CompactPoints::from_floats(&kinds, &coords) {
            return Points::Compact(Arc::new(points));
        }
    }
    Points::Json(moda_point_objects(cmds))
}

/// Moda's point objects for path commands.
fn moda_point_objects(cmds: &[vork_std::moda::PathCmd]) -> Vec<Value> {
    use vork_std::moda::PathCmd;
    cmds.iter().map(|cmd| match *cmd {
        PathCmd::Move { x, y } => json!({"type": "move", "x": x, "y": y}),
        PathCmd::Line { x, y } => json!({"type": "line", "x": x, "y": y}),
        PathCmd::Curve { x, y, cp1, cp2 } => json!({"type": "curve", "x": x, "y": y, "cp1": {"x": cp1.0, "y": cp1.1}, "cp2": {"x": cp2.0, "y": cp2.1}}),
        PathCmd::Quadratic { x, y, cp } => json!({"type": "quadratic", "x": x, "y": y, "cp": {"x": cp.0, "y": cp.1}}),
        PathCmd::Arc { x, y, rx, ry, x_axis_rotation, large_arc, sweep } =>
            json!({"type": "arc", "x": x, "y": y, "rx": rx, "ry": ry, "xAxisRotation": x_axis_rotation, "largeArc": large_arc, "sweep": sweep}),
    }).collect()
}

fn measure_path(cmds: &[vork_std::moda::PathCmd], closure: &Value) -> f64 {
    use moda_engine::tess::{PathClosure, PathContours};

    PathContours::new(cmds, PathClosure::from_field(closure)).measured_length()
}

/// A path node's data for Moda: its `d` as Moda's points, its paints resolved against its bounds.
pub(crate) fn path_item(images: &v2::Table<v2::Image>, base: &Path, node: &v2::Node, media: &Media) -> Result<PathItem> {
    sampled_path_item(images,base,node,None,media)
}

pub(crate) fn sampled_path_item(images: &v2::Table<v2::Image>, base: &Path, node: &v2::Node, sampled: Option<&crate::animation::style::StyleValue>, media: &Media) -> Result<PathItem> {
    let v2::NodeKind::Path { d, fill, fill_rule, fill_opacity, stroke, stroke_width, stroke_linecap, stroke_linejoin, stroke_miterlimit,
        stroke_dasharray, stroke_dashoffset, stroke_opacity, path_length, paint_order } = &node.kind else { unreachable!() };
    let name = &node.name;
    ensure!(stroke_miterlimit.is_finite() && *stroke_miterlimit>=1.,"path {name:?}: stroke-miterlimit must be at least 1");
    ensure!(path_length.is_none_or(|v|v.is_finite() && v>0.),"path {name:?}: pathLength must be positive");
    let (cmds, closed) = if let Some(sampled)=sampled {sampled.path_commands()?} else if d.trim() == "none" || d.trim().is_empty() { (Vec::new(), Vec::new()) } else {
        vork_std::moda::path_commands(d).with_context(|| format!("path {name:?}"))?
    };
    let points = moda_points(&cmds);
    let needs_bounds=|paint:&v2::Paint| matches!(paint,v2::Paint::Image(_)) || matches!(paint,v2::Paint::Css(s) if s.contains("gradient("));
    let bounds = if needs_bounds(fill) || needs_bounds(stroke) {moda_engine::tess::path_bounds(&cmds)} else {None};
    let closed = moda_engine::tess::PathClosure::to_field(closed);
    let length = if path_length.is_some() { measure_path(&cmds, &closed) } else { 0. };
    let paint = |field: &str, value: &v2::Paint| resolve_paint(images, base, value, bounds, &format!("path {name:?}: {field}"), media);
    let fill = paint("fill", fill)?;
    let stroke = paint("stroke", stroke)?;
    let color = |paint: &PaintSource| match paint { PaintSource::Color(color) => Some(*color), _ => None };
    let fill_color = color(&fill);
    let stroke_color = color(&stroke);
    Ok(PathItem {
        points,
        closed,
        bounds,
        length,
        fill,
        stroke,
        fill_rule: *fill_rule,
        linecap: *stroke_linecap,
        linejoin: *stroke_linejoin,
        miterlimit: *stroke_miterlimit,
        dasharray: stroke_dasharray.lengths().to_vec(),
        sampled_dash: None,
        path_length: *path_length,
        stroke_first: paint_order.stroke_first(), paint_clock: 0,
        values: PathValues {
            fill: fill_color,
            stroke: stroke_color,
            stroke_width: *stroke_width,
            stroke_dashoffset: *stroke_dashoffset,
            fill_opacity: *fill_opacity,
            stroke_opacity: *stroke_opacity,
        },
    })
}

const MEDIA_PREFIX: &str = "vork:canvas-media/";

/// Movie and sequence paints bind by their publication key, including copies inside masks.
fn missing_media(paint: &Value) -> bool {
    paint["src"].as_str().is_some_and(|src| src.starts_with(MEDIA_PREFIX)
        && moda_assets::engine_seam::installed().is_none_or(|store| store.dimensions(src).is_none()))
}

/// A render samples its instant. An edit only needs a publication for each image its planner reads.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum MediaPublication { Frame, Missing }

/// Image names own clocks and publication keys. Visible source pictures own decoders.
pub(crate) struct Media {
    pub(super) files: crate::asset_revision::SourceFiles,
    /// Sound can resolve the same paints using metadata alone.
    pub(super) publish_assets: bool,
    pub(super) paints: Arc<crate::canvas_program::Paints>,
    images: Vec<MediaImage>,
    reader_items: Vec<usize>,
    sources: Vec<crate::footage::Source>,
    // (held source, displayed ordinal). Equal requests share immutable pixels.
    live: BTreeMap<(usize, usize), LivePicture>,
    device: crate::footage_canvas::CanvasDevice,
    fps: vork_time::Time,
    max_side: u32,
    /// Whether the canvas takes an image as its decoded texels, straight alpha, at its natural size: so for a
    /// canvas that blends in another encoding than sRGB. For an sRGB canvas Moda's asset store prepares an
    /// image as a browser prepares a bitmap: associated in 8 bits, and a still sized (a small or an oversized
    /// one resampled on its sRGB bytes). A canvas in another encoding converts, associates and resamples in
    /// that encoding, so nothing may round or filter the image's sRGB bytes first: a still is stored under
    /// its natural key (`moda_assets::natural_key`), a footage frame unassociated.
    decoded_texels: bool,
}
struct MediaImage {
    name: String,
    path: PathBuf,
    src: Rc<String>,
    source: usize,
    size: [u32; 2],
    time: Option<vork_time::Clock>,
    frame: Option<usize>,
}
#[derive(Clone)]
struct LivePicture {
    decoder: Rc<RefCell<crate::footage_canvas::CanvasDecoder>>,
    entry: moda_assets::AssetEntry,
}
impl Drop for Media {
    fn drop(&mut self) {
        if !self.publish_assets || self.images.iter().all(|image| Rc::strong_count(&image.src) > 1) { return; }
        if let Some(installed) = moda_assets::engine_seam::installed() {
            let mut store = (*installed).clone();
            for image in &self.images {
                if Rc::strong_count(&image.src) == 1 { store.remove(&image.src); }
            }
            moda_assets::engine_seam::install_store(std::sync::Arc::new(store));
        }
    }
}
/// Whether a canvas paint's image is sRGB as Moda's decoders read it: an sRGB upload that declares no other
/// space. Canvas paints do not go through the colour decoder yet, so an embedded description is not read.
/// The decoded image is sRGB code values whatever encoding the canvas blends in: Moda's engine uploads it in
/// that encoding (`working_color`).
fn paint_is_srgb(image: &v2::Image) -> bool {
    image.is_srgb() && matches!(image.color_space, None | Some(v2::ColorSpace::Srgb))
}
impl Media {
    fn program(&self, name: &str) -> Result<PaintSource> {
        let binding = self.paints.bindings.get(name).with_context(|| format!("unknown path material {name:?}"))?;
        Ok(PaintSource::Program(*binding))
    }


    pub(super) fn reader_items(&self) -> &[usize] { &self.reader_items }
    pub(crate) fn is_empty(&self) -> bool { self.images.is_empty() }
    #[cfg(test)]
    pub(crate) fn new(scene: &vork_scene::Scene, base: &Path) -> Result<Self> {
        Self::with_readers(scene, base, None, &Default::default(), None)
    }

    pub(super) fn with_readers(scene: &vork_scene::Scene, base: &Path,
        wanted: Option<&std::collections::HashSet<&str>>, opened: &crate::footage::Opened,
        previous: Option<(&Self, &BTreeSet<&Path>)>) -> Result<Self> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut names = std::collections::BTreeSet::new();
        let mut reader_items = Vec::new();
        fn visit(media: &v2::MediaReaders<'_>, nodes: &[v2::Node], at: &mut usize,
                 names: &mut std::collections::BTreeSet<String>, items: &mut Vec<usize>) {
            for node in nodes {
                let media_names = media.node_media(node);
                if !media_names.is_empty() {
                    items.push(*at);
                    names.extend(media_names);
                }
                *at += 1;
                visit(media, &node.children, at, names, items);
            }
        }
        if wanted.is_none() {
            visit(&scene.media_readers(), &scene.nodes, &mut 0, &mut names, &mut reader_items);
        }
        let mut images = Vec::new();
        let mut sources = Vec::new();
        let decoded_texels = scene.composition.as_ref().is_some_and(|composition| composition.working_space() != v2::WorkingColorSpace::Srgb);
        let previous = previous.filter(|(old, _)| old.decoded_texels == decoded_texels);
        let mut live = BTreeMap::new();
        let mut by_path = BTreeMap::new();
        for name in names {
            let image = &scene.images[&name];
            ensure!(paint_is_srgb(image), "image {name:?}: a 2D paint reads sRGB images; an image in another colour space is not built yet");
            let path = base.join(image.file.as_ref().expect("media file"));
            // A staged document owns its frame stamps. Its unchanged bindings can keep the
            // publication key and the immutable picture, including the decoder that can continue it.
            let held = previous.filter(|(_, files)| files.contains(path.as_path())).and_then(|(old, _)| {
                let index = old.images.binary_search_by(|img| img.name.cmp(&name)).ok()?;
                let img = &old.images[index];
                (img.path == path && img.time == image.time).then_some((old, img))
            });
            let key = if crate::footage::is_sequence(&path) { path.clone() } else { opened.revision(&path)?.original().to_owned() };
            // Equal clocks can share pixels only when the canvas decodes them the same way.
            // An explicit sRGB declaration can give a tagged HDR movie two different pictures.
            let bind = |probed: &crate::footage::Source| -> Result<_> {
                let map_hdr = probed.canvas_maps_hdr(image.color_space);
                let binding = match by_path.entry((key, map_hdr)) {
                    std::collections::btree_map::Entry::Occupied(entry) => *entry.get(),
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        let size = match held { Some((_, img)) => img.size, None => probed.dimensions()? };
                        let index = sources.len();
                        sources.push(probed.canvas_copy(map_hdr));
                        *entry.insert((index, size))
                    }
                };
                let same_pixels = !matches!(probed, crate::footage::Source::Video(video) if video.map_hdr != map_hdr);
                Ok((binding, same_pixels))
            };
            let ((source, size), same_pixels) = match held {
                Some((old, img)) => bind(&old.sources[img.source]),
                None => opened.read(&path, bind),
            }?;
            let (src, frame) = if let Some((old, img)) = held.filter(|_| same_pixels) {
                if let Some(i) = img.frame && let Some(picture) = old.live.get(&(img.source, i)) {
                    live.entry((source, i)).or_insert_with(|| picture.clone());
                }
                (img.src.clone(), img.frame)
            } else {
                (Rc::new(format!("{MEDIA_PREFIX}{id}/{}", images.len())), None)
            };
            images.push(MediaImage { src, name, path, source, size, time: image.time.clone(), frame });
        }
        Ok(Self { paints: Default::default(), files: opened.files.clone(), publish_assets: wanted.is_none(), images, reader_items, sources, live,
            device: Default::default(),
            fps: scene.render.fps, max_side: u32::MAX, decoded_texels })
    }
    pub(crate) fn prepare_gpu(&mut self, renderer: &crate::Renderer) -> Result<()> {
        let limit = renderer.device.limits().max_texture_dimension_2d;
        self.max_side = limit;
        for image in &mut self.images {
            ensure!(image.size.iter().all(|&side| side <= limit), crate::gpu_memory::DeviceLimit(format!("image {:?}: {}x{} exceeds this device's {limit} px texture limit; resize the source", image.name, image.size[0], image.size[1])));
        }
        for source in &mut self.sources { source.prepare_gpu(renderer); }
        self.device = crate::footage_canvas::CanvasDevice::new(renderer, self.decoded_texels);
        Ok(())
    }
    /// Returns whether any publication appeared or disappeared (advancing its picture keeps the same binding).
    pub(crate) fn update(&mut self, plan: &crate::animation::Plan, samples: &crate::animation::Samples, request: &vork_time::exact::Germ, publication: MediaPublication, needed: impl Fn(&str) -> bool) -> Result<bool> {
        if self.images.is_empty() { return Ok(false); }
        let wanted = self.images.iter().map(|image| -> Result<Option<usize>> {
            // Reserve all held pictures during an edit, including movies outside its zero-time paints.
            if publication == MediaPublication::Missing && image.frame.is_some() { return Ok(image.frame); }
            if !needed(&image.src) { return Ok(None); }
            let time = plan.footage_time(samples, request, &image.name, image.time.as_ref())?;
            time.map(|time| self.sources[image.source].frame_index_exact(&time, self.fps)).transpose()
        }).collect::<Result<Vec<_>>>()?;
        if self.images.iter().zip(&wanted).all(|(image, &index)| image.frame == index) { return Ok(false); }

        // Reserve held pictures first. A moving clock cannot take the decoder of a frozen alias.
        let mut next = BTreeMap::new();
        for (image, &index) in self.images.iter().zip(&wanted) {
            if let Some(key) = index.map(|i| (image.source, i))
                && let Some(picture) = self.live.remove(&key) {
                next.insert(key, picture);
            }
        }
        for (image, &index) in self.images.iter().zip(&wanted) {
            let Some(index) = index else { continue };
            let std::collections::btree_map::Entry::Vacant(slot) = next.entry((image.source, index)) else { continue };
            // Continue this clock's reader when possible, including across an alias handoff.
            let old = image.frame.and_then(|i| self.live.remove(&(image.source, i))).or_else(|| {
                let (&key, _) = self.live.range((image.source, 0)..=(image.source, usize::MAX)).next()?;
                self.live.remove(&key)
            });
            let decoder = old.map(|picture| picture.decoder).unwrap_or_else(|| Rc::new(RefCell::new(self.device.decoder())));
            let entry = decoder.borrow_mut().canvas_frame(&mut self.sources[image.source], index, self.decoded_texels,
                self.max_side, image.size).with_context(|| format!("canvas image {:?}", image.name))?;
            slot.insert(LivePicture { decoder, entry });
        }
        let bindings_changed = self.images.iter().zip(&wanted).any(|(img, next)| img.frame.is_some() != next.is_some());
        let mut store = moda_assets::engine_seam::installed().map(|s| (*s).clone()).unwrap_or_default();
        for (image, index) in self.images.iter_mut().zip(wanted) {
            if image.frame == index { continue; }
            if let Some(index) = index {
                store.insert(image.src.as_ref().clone(), next[&(image.source, index)].entry.clone());
            } else {
                store.remove(&image.src);
            }
            image.frame = index;
        }
        moda_assets::engine_seam::install_store(Arc::new(store));
        self.live = next;
        Ok(bindings_changed)
    }
}

pub(crate) fn resolve_animated_paint(images: &v2::Table<v2::Image>, base: &Path, bounds: [f64;4], paint: &crate::animation::style::Paint, media: &Media) -> Result<PaintSource> {
            use crate::animation::style::Paint as P;
            Ok(match paint {
                P::None => PaintSource::None,
                P::Color(c) => PaintSource::Color(*c),
                P::Material(name) => media.program(name)?,
                P::Image(v) => PaintSource::Moda(image_paint(images,base,&serde_json::from_value(v.clone())?,bounds,media)?),
                P::Gradient(s) => PaintSource::Moda(gradient_paint(s,bounds)?),
                P::Mix(a,b,t) => {
                    let value=if let (P::Gradient(a),P::Gradient(b))=(a.as_ref(),b.as_ref()) {
                        interpolate_gradient(a,b,*t,bounds)?
                    } else {
                        let (PaintSource::Moda(a),PaintSource::Moda(b))=(resolve_animated_paint(images,base,bounds,a,media)?,resolve_animated_paint(images,base,bounds,b,media)?) else {unreachable!("only gradients form a paint mix")};
                        mix_gradient_values(&a,&b,*t,bounds)?
                    };
                    PaintSource::Moda(value)
                }
            })
        }

impl PathItem {
    pub fn set_paint(&mut self, images: &v2::Table<v2::Image>, base: &Path, field: &str, paint: &crate::animation::style::Paint, media: &Media) -> Result<()> {
        use crate::animation::style::Paint as P;
        let paint=match paint {
            P::None=>PaintSource::None,
            P::Color(color)=>PaintSource::Color(*color),
            P::Material(name)=>media.program(name)?,
            _=>{let Some(bounds)=self.bounds else {return Ok(())};resolve_animated_paint(images,base,bounds,paint,media)?},
        };
        let color=if let PaintSource::Color(c)=&paint {Some(*c)} else {None};
        match field { "fill"=>{self.fill=paint;self.values.fill=color;}, "stroke"=>{self.stroke=paint;self.values.stroke=color;}, _=>unreachable!() }
        Ok(())
    }
}

pub(crate) fn blend_paint(a:&PaintSource,b:&PaintSource,t:f32,bounds:[f64;4])->Result<PaintSource> {
    if t<=0. || a==b {return Ok(a.clone())} if t>=1. {return Ok(b.clone())}
    Ok(match (a,b) {
        (PaintSource::Color(a),PaintSource::Color(b))=>PaintSource::Color(crate::animation::style::color_mix(*a,*b,t)),
        (PaintSource::Moda(a),PaintSource::Moda(b)) if a["stops"].is_array() && b["stops"].is_array()=>PaintSource::Moda(mix_gradient_values(a,b,t,bounds)?),
        _=>if t<0.5 {a.clone()}else{b.clone()},
    })
}

/// CSS Images 4 defers interpolation of a gradient mixing length and percentage stops.
pub(crate) fn gradient_discrete(a:&str,b:&str)->bool {
    fn mixed(s:&str)->bool {
        if s.contains("conic-gradient("){return false}
        let Some(open)=s.find('(') else{return false};let Some(close)=s.rfind(')') else{return false};
        let args=split_top_level(&s[open+1..close],',');let(mut length,mut percent)=(false,false);
        for part in args {
            let tokens=split_top_level(&part,' ');
            if tokens.first().is_none_or(|v|vork_std::moda::css_color(v).is_err()){continue}
            for token in tokens.iter().skip(1) {percent|=token.ends_with('%');length|=token.ends_with("px") || token.parse::<f64>().is_ok();}
        }
        length && percent
    }
    if !mixed(a) && !mixed(b){return false}
    let (Ok(a),Ok(b))=(gradient_paint(a,[0.,0.,100.,100.]),gradient_paint(b,[0.,0.,100.,100.])) else{return false};
    a["type"]==b["type"] && a["repeat"]==b["repeat"] && a["interpolation"]==b["interpolation"] && a["stops"].as_array().map(Vec::len)==b["stops"].as_array().map(Vec::len)
}

/// Gradients interpolate their geometry and corresponding colour stops, after resolving lengths in
/// the shape's box. Incompatible images have no continuous interpolation in the built paint model.
pub(crate) fn interpolate_gradient(a: &str,b: &str,t:f32,bounds:[f64;4])->Result<Value> {
    let mut left=gradient_paint(a,bounds)?;
    let mut right=gradient_paint(b,bounds)?;
    if left["type"]!=right["type"] || left["repeat"]!=right["repeat"] || left["interpolation"]!=right["interpolation"]
        || left["stops"].as_array().map(Vec::len)!=right["stops"].as_array().map(Vec::len) {bail!("gradient cross-fade and repeating stop matching are not built; use STEP or matching gradient types, interpolation spaces and stop counts")}
    if gradient_discrete(a,b) {return Ok(if t<0.5 {left}else{right})}
    // A linear gradient interpolates its angle, not its end-point vectors (opposing directions
    // would collapse the gradient). Resolve both stop lists on that interpolated line.
    if left["type"]=="linear" {
        let angle=|p:&Value| {
            let dx=(p["end"]["x"].as_f64().unwrap()-p["start"]["x"].as_f64().unwrap())*(bounds[2]-bounds[0]);
            let dy=(p["end"]["y"].as_f64().unwrap()-p["start"]["y"].as_f64().unwrap())*(bounds[3]-bounds[1]);
            dx.atan2(-dy).to_degrees().rem_euclid(360.)
        };
        let authored_angle=|s:&str,p:&Value| {
            let open=s.find('(').unwrap(); let args=split_top_level(&s[open+1..s.rfind(')').unwrap()],',');
            args.first().and_then(|v|split_top_level(v,' ').first().and_then(|v|css_angle(v))).unwrap_or_else(||angle(p))
        };
        let mut x=authored_angle(a,&left); let mut y=authored_angle(b,&right);
        let keyword=|s:&str|s[s.find('(').unwrap()+1..].trim_start().starts_with("to ");
        if keyword(a) && keyword(b) && (x-y).abs()>180. {if x<y {x+=360.}else{y+=360.}}
        let angle=x+(y-x)*f64::from(t);
        if x!=y {
        let replace=|s:&str| {
            let open=s.find('(').unwrap(); let mut args=split_top_level(&s[open+1..s.rfind(')').unwrap()],',');
            if args.first().is_some_and(|a|split_top_level(a,' ').first().is_some_and(|v|vork_std::moda::css_color(v).is_err())) {args.remove(0);}
            format!("{}({angle}deg, {})",&s[..open],args.join(", "))
        };
        let space=left.get("interpolation").cloned();
        left=gradient_paint(&replace(a),bounds)?;right=gradient_paint(&replace(b),bounds)?;
        if let Some(space)=space {left["interpolation"]=space.clone();right["interpolation"]=space;}
        }
    }
    if left["type"]=="radial" {
        let rx=left["radiusX"].as_f64().unwrap()+(right["radiusX"].as_f64().unwrap()-left["radiusX"].as_f64().unwrap())*f64::from(t);
        let radius=rx/100.*(bounds[2]-bounds[0]);
        let stops=|s:&str|->Result<Value> {
            let open=s.find('(').unwrap();let args=split_top_level(&s[open+1..s.rfind(')').unwrap()],',');
            let skip=usize::from(args.first().is_some_and(|a|split_top_level(a,' ').first().is_some_and(|v|vork_std::moda::css_color(v).is_err())));
            Ok(moda_stops(&color_stops(&args[skip..],radius,false,s)?))
        };
        left["stops"]=stops(a)?;right["stops"]=stops(b)?;
    }
    mix_gradient_values(&left,&right,t,bounds)
}

fn mix_gradient_values(left:&Value,right:&Value,t:f32,bounds:[f64;4])->Result<Value> {
    ensure!(left["type"]==right["type"] && left["repeat"]==right["repeat"] && left["interpolation"]==right["interpolation"]
        && left["stops"].as_array().map(Vec::len)==right["stops"].as_array().map(Vec::len),
        "gradient cross-fade and repeating stop matching are not built; use STEP or matching gradient types, interpolation spaces and stop counts");
    fn mix(a:&Value,b:&Value,t:f32)->Value {
        match (a,b) {
            (Value::Number(a),Value::Number(b))=>json!(a.as_f64().unwrap()+(b.as_f64().unwrap()-a.as_f64().unwrap())*f64::from(t)),
            (Value::Array(a),Value::Array(b)) if a.len()==b.len()=>a.iter().zip(b).map(|(a,b)|mix(a,b,t)).collect(),
            (Value::Object(a),Value::Object(b))=>a.iter().map(|(key,value)| {
                let value=if key=="color" {
                    let a=vork_std::moda::css_color(value.as_str().unwrap()).unwrap();
                    let b=vork_std::moda::css_color(b[key].as_str().unwrap()).unwrap();
                    json!(vork_std::moda::moda_color(crate::animation::style::color_mix(a,b,t)))
                }else{mix(value,&b[key],t)};
                (key.clone(),value)
            }).collect(),
            _=>if t<0.5 {a.clone()}else{b.clone()},
        }
    }
    let mut result=mix(left,right,t);
    if left["type"]=="linear" && (left["start"]!=right["start"] || left["end"]!=right["end"]) {
        let w=(bounds[2]-bounds[0]).max(1e-6);let h=(bounds[3]-bounds[1]).max(1e-6);
        let angle=|p:&Value| {
            let dx=(p["end"]["x"].as_f64().unwrap()-p["start"]["x"].as_f64().unwrap())*w;
            let dy=(p["end"]["y"].as_f64().unwrap()-p["start"]["y"].as_f64().unwrap())*h;
            dx.atan2(-dy).rem_euclid(std::f64::consts::TAU)
        };
        let a=angle(left);let b=angle(right);let angle=a+(b-a)*f64::from(t);
        let dx=angle.sin();let dy=-angle.cos();let length=w*dx.abs()+h*dy.abs();
        result["start"]=json!({"x":50.-dx*length/w*50.,"y":50.-dy*length/h*50.});
        result["end"]=json!({"x":50.+dx*length/w*50.,"y":50.+dy*length/h*50.});
    }
    Ok(result)
}

const EXPLICIT_SRGB_PAINT: &str = "vork-srgb-paint:";

fn still_paint_source(key: &str) -> (&str, bool, bool) {
    let natural = moda_assets::natural_source(key);
    let source = natural.unwrap_or(key);
    let declared = source.strip_prefix(EXPLICIT_SRGB_PAINT);
    (crate::asset_revision::SourceFile::paint_name(declared.unwrap_or(source)), natural.is_some(), declared.is_some())
}

/// An image paint (`{image, fit}`) as Moda's: CSS `object-fit` in the path's box, or, without `fit`, the image at
/// its natural size from the box's top-left corner, repeated (the image's `wrap_s` and `wrap_t`, glTF's
/// `repeat` by default, as a CSS background repeats).
fn image_paint(images: &v2::Table<v2::Image>, base: &Path, paint: &v2::ImagePaint, bounds: [f64; 4], media: &Media) -> Result<Value> {
    let image = images.get(&paint.image).with_context(||format!("unknown image {:?}",paint.image))?;
    ensure!(image.render.is_none(), "a rendered image as a 2D paint is not built yet; give a file");
    let file = image.file.as_deref().context("give the image a file")?;
    let extension = Path::new(file).extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
    ensure!(matches!(extension.as_str(), "png" | "jpg" | "jpeg") || crate::footage::is_media(Path::new(file)), "a 2D paint reads PNG, JPEG, movies and frame sequences; {file:?} is not built yet");
    ensure!(paint_is_srgb(image), "a 2D paint reads sRGB images; an image in another colour space is not built yet");
    // The soundtrack consumes the resolved binding only. Placement and decoded
    // pixels are picture inputs and do not determine which image this paint selects.
    if !media.publish_assets {
        return Ok(json!({"type":"image", "src":paint.image}));
    }
    let path = base.join(file);
    let binding = media.images.iter().find(|item| item.name == paint.image);
    let (nw, nh, src) = match binding {
        Some(item) => (item.size[0], item.size[1], item.src.as_ref().clone()),
        None => {
            let input = media.files.source(&path);
            let (width, height, source) = match input.dimensions(true) {
                Ok((width, height)) => {
                    (width, height, input.paint_key()?)
                }
                Err(image::ImageError::IoError(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                    // Keep a texture so fit, crop and tiling keep their normal 1×1 semantics.
                    const MAGENTA_PNG: &str = concat!(
                        "data:image/png;base64,",
                        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z/AfAAQAAf8iCjrwAAAAAElFTkSuQmCC",
                    );
                    crate::warnings::dropped("missing_image", serde_json::json!({"file":input.name}),
                        "using a 1x1 magenta texture; restore the image or correct its path.".into());
                    (1, 1, MAGENTA_PNG.into())
                }
                Err(error) => return Err(error).with_context(|| format!("read {}", input.name.display())),
            };
            let source = if image.color_space == Some(v2::ColorSpace::Srgb) {
                format!("{EXPLICIT_SRGB_PAINT}{source}")
            } else { source };
            let source = if media.decoded_texels { moda_assets::natural_key(&source) } else { source };
            (width, height, source)
        }
    };
    let (nw, nh) = (f64::from(nw), f64::from(nh));
    let [x0, y0, x1, y1] = bounds;
    let placement = match paint.fit {
        None => {
            ensure!(image.wrap_s == v2::Wrap::Repeat && image.wrap_t == v2::Wrap::Repeat,
                "an image paint without fit repeats; wrap_s and wrap_t other than repeat are not built yet");
            json!({"fit": "tile", "offset": {"x": x0, "y": y0}, "scale": {"x": 1, "y": 1}, "repeat": "repeat", "naturalSize": {"width": nw, "height": nh}})
        }
        Some(v2::Fit::None) => json!({"fit": "crop", "offset": {"x": x0 + (x1 - x0 - nw) / 2., "y": y0 + (y1 - y0 - nh) / 2.},
            "scale": {"x": 1, "y": 1}, "naturalSize": {"width": nw, "height": nh}}),
        Some(fit) => json!({"fit": match fit { v2::Fit::Contain => "contain", v2::Fit::Cover => "cover", _ => "fill" },
            "naturalSize": {"width": nw, "height": nh}}),
    };
    Ok(json!({"type": "image", "src": src, "placement": placement}))
}

/// The output's page: a Moda document page of `size` px with the canvas's background as its fill.
#[derive(Clone, PartialEq)]
pub(crate) struct Page {
    pub size: [u32; 2],
    /// Canvas units to output px (per axis) and the offset, from `viewBox` and `preserveAspectRatio`.
    pub scale: [f64; 2],
    pub offset: [f64; 2],
    /// A CSS colour behind every node, or `None` (transparent).
    pub background: Option<[f32; 4]>,
    /// The encoding the canvas blends in, as Moda's engine and its planner take it (`working_color`); `None`
    /// is sRGB.
    pub working: Option<moda_engine::working_color::WorkingColor>,
}

/// One frame's Moda nodes, typed: no JSON text or JSON tree is built for them.
struct Lowered {
    /// The document without its nodes: the page, the config and the layout fields.
    envelope: Map,
    /// Each Moda node's fields, with its path geometry when vork holds that compact (the fields then keep
    /// `"points": null` in its place).
    nodes: Vec<(Map, Option<Arc<CompactPoints>>)>,
    /// Canvas fields bound to the row that the writer emitted.
    bindings: Vec<Bound>,
    /// Items whose painted body the writer emitted inside a mask source.
    mask_paint_items: Vec<usize>,
    /// How many canvas items were lowered.
    items: usize,
}
impl Lowered {
    /// The canvas emits supported nodes in installation order, so its row handles survive hydration.
    fn load(self) -> Result<(moda_scene::Scene, Bindings)> {
        let count = self.nodes.len();
        ensure!(u32::try_from(count).is_ok(), "canvas exceeds the scene row limit");
        let scene = moda_scene::Scene::load_nodes(self.envelope, self.nodes)?;
        ensure!(scene.doc().nodes.len() == count, "canvas install dropped a lowered row");
        let bindings = Bindings::new(self.bindings, self.items, self.mask_paint_items);
        Ok((scene, bindings))
    }
    /// The nodes as Moda holds them before hydration.
    #[cfg(test)]
    fn raw_nodes(nodes: Vec<(Map, Option<Arc<CompactPoints>>)>) -> impl Iterator<Item = moda_scene::RawNode> {
        nodes.into_iter().map(|(fields, points)| match points {
            Some(points) => moda_scene::RawNode::with_shared_points(fields, points),
            None => moda_scene::RawNode::new(fields),
        })
    }
}

/// A canvas item's field in the installed scene: the row of the Moda node that holds it.
struct Bound {
    item: usize,
    row: u32,
    field: BoundField,
}
/// Every binding of the installed scene, by item.
struct Bindings {
    mask_paint_items: Vec<usize>,
    /// Sorted by item; each item's bindings keep the order the writer made them in.
    list: Vec<Bound>,
    /// Item `i`'s bindings are `list[first[i]..first[i + 1]]`.
    first: Vec<u32>,
    /// The effects bindings each item's transform feeds (a canvas-space shadow offset follows every transform
    /// above it): item `i`'s are `fed[fed_first[i]..fed_first[i + 1]]`, as indices into `list`.
    fed_first: Vec<u32>,
    fed: Vec<u32>,
}
impl Bindings {
    fn new(mut list: Vec<Bound>, items: usize, mask_paint_items: Vec<usize>) -> Self {
        // Stable: an item's bindings stay in the writer's order.
        list.sort_by_key(|bound| bound.item);
        let mut first = vec![0u32; items + 1];
        for bound in &list { first[bound.item + 1] += 1; }
        for item in 0..items { first[item + 1] += first[item]; }
        let mut fed_first = vec![0u32; items + 1];
        for bound in &list {
            if let BoundField::Effects(effect) = &bound.field {
                for &item in &effect.transforms { fed_first[item + 1] += 1; }
            }
        }
        for item in 0..items { fed_first[item + 1] += fed_first[item]; }
        let mut fed = vec![0u32; fed_first[items] as usize];
        let mut next = fed_first.clone();
        for (at, bound) in list.iter().enumerate() {
            if let BoundField::Effects(effect) = &bound.field {
                for &item in &effect.transforms {
                    fed[next[item] as usize] = at as u32;
                    next[item] += 1;
                }
            }
        }
        Self { list, first, fed_first, fed, mask_paint_items }
    }
    /// Where `item`'s bindings sit in `list`.
    fn of(&self, item: usize) -> std::ops::Range<usize> {
        self.first[item] as usize..self.first[item + 1] as usize
    }
    /// The effects bindings `item`'s transform feeds, as indices into `list`.
    fn fed_by(&self, item: usize) -> &[u32] {
        &self.fed[self.fed_first[item] as usize..self.fed_first[item + 1] as usize]
    }
}

/// What of an item changed since the values the installed scene holds, as the evaluation that made the new
/// values names it.
pub(crate) mod change {
    pub const TRANSLATION: u8 = 1;
    pub const ROTATION: u8 = 8;
    pub const SCALE: u8 = 16;
    pub const TRANSFORM: u8 = TRANSLATION | ROTATION | SCALE;
    /// Its opacity.
    pub const OPACITY: u8 = 2;
    /// Anything else: paint, geometry, style, text, a world's picture, visibility.
    pub const REST: u8 = 4;
    /// Only the authored CSS filter changed.
    pub const FILTER: u8 = 32;
    pub const ALL: u8 = TRANSFORM | OPACITY | REST | FILTER;
}

/// The form an item's Moda nodes were lowered in: everything that decides which Moda nodes there are for it and
/// how they nest. While an item keeps its form, its changes are field writes into rows the bindings name; when
/// one's form differs the canvas is lowered again.
#[derive(Clone, PartialEq)]
struct Form {
    shape: Shape,
    /// The text pieces the item's nodes were written for: which pieces there are and how they nest. Their
    /// paints, boxes and opacities are fields, and are not kept here.
    composition: Box<[TextShape]>,
    css: Option<Box<CssShape>>,
}
/// Which piece of a text composition a node was written for, without its values.
#[derive(Clone, PartialEq)]
enum TextShape {
    Batch(usize),
    Paint(usize),
    Group(Vec<TextShape>),
}
impl TextShape {
    fn of(parts: &[TextPart]) -> Vec<TextShape> {
        parts.iter().map(|part| match part {
            TextPart::Batch(sub) => TextShape::Batch(*sub),
            TextPart::Paint { sub, .. } => TextShape::Paint(*sub),
            TextPart::Group { children, .. } => TextShape::Group(TextShape::of(children)),
        }).collect()
    }
    /// Whether `parts` are these pieces, nested the same way.
    fn holds(shape: &[TextShape], parts: &[TextPart]) -> bool {
        shape.len() == parts.len() && shape.iter().zip(parts).all(|(shape, part)| match (shape, part) {
            (TextShape::Batch(a), TextPart::Batch(b)) => a == b,
            (TextShape::Paint(a), TextPart::Paint { sub: b, .. }) => a == b,
            (TextShape::Group(a), TextPart::Group { children: b, .. }) => TextShape::holds(a, b),
            _ => false,
        })
    }
}
#[derive(Clone, Copy, PartialEq)]
struct Shape {
    visible: bool,
    zero_scale: bool,
    texts: usize,
    /// For a path with keyed values: whether it has a keyed fill, a keyed stroke and a stroke width.
    keyed: Option<(bool, bool, bool)>,
    world: bool,
    parent: Option<usize>,
    clip_path: Option<usize>,
    mask: Option<usize>,
    kind: KindForm,
}
#[derive(Clone, Copy, PartialEq)]
enum KindForm {
    Group,
    /// A path: its paints and geometry are fields of its path nodes. A stroke drawn through its coverage (an
    /// image or conic stroke) is other nodes than a plain stroke, and needs bounds to be any.
    Path { empty: bool, stroke_first: bool, fill: bool, stroke: bool, coverage: bool, bounds: bool },
    /// Text: its box, and what it draws (named by the revision its nodes carry), are fields of its nodes.
    Text,
    /// A world: its node holds its box (fixed when the canvas is read), and its picture is a field.
    World { bounds: [u64; 4] },
}
impl Shape {
    fn of(node: &Node, v: &Values) -> Self {
        Self {
            visible: v.visible,
            zero_scale: v.transform.scale[0] == 0. || v.transform.scale[1] == 0.,
            texts: v.texts,
            keyed: v.path.map(|p| (p.fill.is_some(), p.stroke.is_some(), p.stroke_width > 0.)),
            world: v.world.is_some(),
            parent: node.parent,
            clip_path: node.clip_path,
            mask: node.mask,
            kind: match &node.kind {
                Kind::Group => KindForm::Group,
                Kind::Path(path) => KindForm::Path {
                    empty: path.points.is_empty(), stroke_first: path.stroke_first,
                    fill: !matches!(path.fill, PaintSource::None), stroke: !matches!(path.stroke, PaintSource::None),
                    coverage: path.stroke_coverage(), bounds: path.bounds.is_some(),
                },
                Kind::Text { .. } => KindForm::Text,
                Kind::World { bounds } => KindForm::World { bounds: bounds.map(f64::to_bits) },
            },
        }
    }
}
/// Only styled nodes need a retained description of their CSS wrappers and effect bindings.
#[derive(Clone, PartialEq)]
struct CssShape {
    mask_type: v2::MaskType,
    blend: v2::MixBlendMode,
    isolation: v2::Isolation,
    stages: usize,
    single_capture: bool,
    shadow: bool,
    active: bool,
}

impl CssShape {
    fn of(node: &Node) -> Option<Self> {
        if node.filter.0.is_empty() && node.blend == v2::MixBlendMode::Normal
            && node.isolation == v2::Isolation::Auto && node.mask_type == v2::MaskType::Luminance {
            return None;
        }
        Some(Self {
            mask_type: node.mask_type,
            blend: node.blend,
            isolation: node.isolation,
            stages: node.filter.0.len(),
            single_capture: node.filter.single_capture(),
            shadow: node.filter.has_shadow(),
            active: node.filter.0.iter().any(|op| !matches!(op, FilterOp::Blur(sigma) if *sigma == 0.)),
        })
    }
}

impl Form {
    fn of(node: &Node, v: &Values) -> Self {
        Self {
            shape: Shape::of(node, v),
            composition: TextShape::of(&v.text_composition).into_boxed_slice(),
            css: CssShape::of(node).map(Box::new),
        }
    }
    /// Whether `node` at `v` would be lowered in this form.
    fn holds(&self, node: &Node, v: &Values) -> bool {
        TextShape::holds(&self.composition, &v.text_composition) && self.shape == Shape::of(node, v)
            && self.css.as_deref() == CssShape::of(node).as_ref()
    }
}

/// One frame's Moda document.
fn write_document(page: &Page, nodes: &[&Node], values: &[Values]) -> Result<Lowered> {
    let items = nodes.len();
    let mut out = Writer { nodes, values, list: Vec::new(), children: vec![Vec::new(); nodes.len()], bindings: Vec::new(), transform_path: Vec::new(), masks: 0, mask_paint_items: BTreeSet::new() };
    for (i, node) in nodes.iter().enumerate() {
        if let Some(parent) = node.parent {
            out.children[parent].push(i);
        }
    }
    // A node named by clip-path or mask is not painted on its own (SVG: clipPath and mask contents).
    let mut referenced = vec![false; nodes.len()];
    for node in nodes {
        for target in [node.clip_path, node.mask].into_iter().flatten() {
            referenced[target] = true;
        }
    }
    let mut roots = Vec::new();
    for i in (0..nodes.len()).filter(|&i| nodes[i].parent.is_none() && !referenced[i]) {
        let id = format!("n{i}");
        if out.node(i, &id, &referenced, &mut Vec::new(), glam::DMat2::from_diagonal(glam::DVec2::from_array(page.scale)))?.is_some() {
            roots.push(id);
        }
    }
    // viewBox placement: one root group.
    let root = "canvas".to_string();
    out.list.push(json!({"id": root, "type": "group", "x": page.offset[0], "y": page.offset[1], "rotation": 0,
        "scaleX": page.scale[0], "scaleY": page.scale[1], "children": roots}));
    let fills = match page.background {
        Some(color) => json!([{"id": "background", "paint": {"type": "solid", "color": vork_std::moda::moda_color(color)}}]),
        None => json!([]),
    };
    let Value::Object(envelope) = json!({
        "pages": [{"id": "page", "name": "canvas", "width": page.size[0], "height": page.size[1], "nodeIds": [root], "fills": fills}],
        "config": {"renderPageClipping": true},
        "canvasRootNodeIds": [],
        "layoutMode": "vertical",
    }) else { unreachable!("an object literal") };
    // A path's compact geometry rides beside the Moda nodes that draw it: the path and geometry bindings name
    // exactly those nodes, and their `points` field is the `null` placeholder.
    let mut geometry: Vec<Option<Arc<CompactPoints>>> = vec![None; out.list.len()];
    for binding in &out.bindings {
        if !matches!(binding.field, BoundField::Path { .. } | BoundField::Geometry) { continue; }
        let Kind::Path(path) = &nodes[binding.item].kind else { continue };
        let row = binding.row as usize;
        if let Points::Compact(points) = &path.points {
            if out.list[row]["points"].is_null() { geometry[row] = Some(points.clone()); }
        }
    }
    let nodes = out.list.into_iter().zip(geometry).map(|(node, points)| match node {
        Value::Object(fields) => (fields, points),
        _ => unreachable!("every written node is an object"),
    }).collect();
    Ok(Lowered { envelope, nodes, bindings: out.bindings, items, mask_paint_items: out.mask_paint_items.into_iter().collect() })
}

/// The frame's document as JSON, for tests that read what the writer lowers.
#[cfg(test)]
pub(super) fn document(page: &Page, nodes: &[&Node], values: &[Values]) -> Result<serde_json::Value> {
    let lowered = write_document(page, nodes, values)?;
    let mut document = lowered.envelope;
    let nodes: Vec<Value> = Lowered::raw_nodes(lowered.nodes).map(|node| Value::Object(node.into_json_map())).collect();
    document.insert("nodes".into(), Value::Array(nodes));
    Ok(Value::Object(document).to_json())
}

/// The next `Retained::id`.
static RETAINED_IDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The installed Moda scene and the editor's bounded tessellation caches. The canvas's nodes are lowered to
/// Moda nodes once, at install; after that each canvas field is written into the scene row bound to it, as
/// typed values. The writer runs again, and the scene is installed anew, only when visibility, zero scale, paint
/// presence or text batch count changes which Moda nodes there are.
///
/// The scene's rows are the only installed copy of the canvas's values: nothing here keeps the values of the
/// last frame to compare the next against. The evaluation names the items it changed; a field is compared, where
/// it is compared at all, with the row that holds it.
pub(crate) struct Retained {
    background: Option<[f32;4]>,
    /// The form each item's Moda nodes were lowered in.
    forms: Vec<Form>,
    scene: moda_scene::Scene,
    caches: moda_engine::residency::PlanCaches,
    hashes: moda_engine::residency::RowHashes,
    bindings: Bindings,
    /// Which retained scene this is, unique in the process: with the scene's clock it names one state of one
    /// scene.
    id: u64,
    plan_index: moda_engine::residency::FramePlanIndex,
    /// The scene clock the last plan was made at; `None` before the first.
    planned_at: Option<u64>,
    /// How many states the scene's plan has had; `None` before the first plan. Every plan made, and every write
    /// into the plan, is another state, whatever caused it (a scene write, a resource that settled, another
    /// shader clock): this names the plan for the engine's packet, and a change to the plan starts from it.
    plan_revision: Option<u64>,
    pub(super) full_plans: u64,
    /// Rows whose placement was written since the last plan (the scene's dirty list for the planner).
    moved: moda_scene::stamps::Subscription,
    painted: moda_scene::stamps::Subscription,
    content: moda_scene::stamps::Subscription,
    /// Per binding (as `bindings.list`), whether it is a transform whose row takes typed placement writes
    /// (`Scene::placeable`), with the scene clock that was read at: it stands until something dirties everything.
    placeable: Option<(u64, Vec<Option<bool>>)>,
    pending_placements: Vec<moda_scene::scene::PlacementEdit<(usize, u8, Option<BoundOpacity>)>>,
    /// Frames whose every write was a typed placement; frames that wrote other fields into rows; and scenes
    /// installed after the first.
    placement_frames: u64,
    write_frames: u64,
    installs: u64,
}
impl Retained {
    pub(super) fn mask_paint_items(&self) -> &[usize] { &self.bindings.mask_paint_items }

    pub(crate) fn measured_boxes(&self, plan: &Plan, count: usize, wanted: &std::collections::HashSet<usize>) -> Vec<Option<[f64; 4]>> {
        let mut boxes = vec![None::<[f64; 4]>; count];
        let mut pending: Vec<_> = plan.items.iter().collect();
        while let Some(draw) = pending.pop() {
            if let moda_engine::seams::DrawItem::LayerBlurComposite { group, .. } = draw {
                pending.extend(&group.items);
                continue;
            }
            let bounds = std::cell::OnceCell::new();
            let mut owner = moda_engine::residency::drawitem_node_id(draw)
                .map(moda_engine::paint::layers::layer_owner_id);
            while let Some(id) = owner {
                // Lowered authored items are n{index}. Paint wrappers and the page
                // have other ids and contribute through their authored ancestors.
                if let Some(row) = id.strip_prefix('n').and_then(|s| s.parse::<usize>().ok())
                    .filter(|index| wanted.contains(index)).and_then(|index| boxes.get_mut(index))
                    && let Some(b) = *bounds.get_or_init(|| moda_engine::layer_blur::shadow::draw_bounds(draw)) {
                    *row = Some(row.map_or(b, |a| [a[0].min(b[0]), a[1].min(b[1]),
                        a[2].max(b[2]), a[3].max(b[3])]));
                }
                owner = self.scene.parent_map().get(id).map(String::as_str);
            }
        }
        boxes
    }

    pub fn new(page: &Page, nodes: &[&Node], values: &[Values]) -> Result<Self> {
        let (mut scene, bindings) = write_document(page, nodes, values)?.load()?;
        let hashes = Default::default();
        let moved = scene.subscribe(moda_scene::stamps::Group::Placement);
        let painted = scene.subscribe(moda_scene::stamps::Group::Paint);
        let content = scene.subscribe(moda_scene::stamps::Group::Content);
        Ok(Self { background:page.background, forms: nodes.iter().zip(values).map(|(node, v)| Form::of(node, v)).collect(), scene, hashes, bindings, caches: Default::default(), id: RETAINED_IDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed), plan_index: Default::default(),
            planned_at: None, plan_revision: None, full_plans: 0, moved, painted, content,
            placeable: None, pending_placements: Vec::new(), placement_frames: 0, write_frames: 0, installs: 0 })
    }

    /// The scene's clock: it moves on every write to the scene and on nothing else, so everything made from the
    /// scene (the plan, the plan index, a held picture) is keyed by it.
    fn generation(&self) -> u64 {
        self.scene.stamps().clock()
    }

    pub(super) fn picture_scene(&self) -> (u64, u64) { (self.id, self.generation()) }

    /// The placement rows written since the last plan, when nothing else was: `None` when content, paint or
    /// structure changed too (or nothing was planned yet). Paint can still use the common span writer.
    fn placement_batch(&self) -> Option<&[u32]> {
        use moda_scene::stamps::Group;
        let stamps = &self.scene.stamps();
        let at = self.planned_at?;
        ([Group::Paint, Group::Content, Group::Structure].into_iter().all(|group| stamps.latest(group) <= at))
            .then(|| stamps.dirty(self.moved))
    }

    /// A plan was made of the scene as it stands.
    fn planned(&mut self) {
        self.planned_at = Some(self.scene.stamps().clock());
        self.scene.clear_dirty(self.moved);
        self.scene.clear_dirty(self.painted);
        self.scene.clear_dirty(self.content);
    }

    /// Lower the canvas again and install the scene that makes: the path of the first install, with every row
    /// dirty. The new scene continues the old one's clock and dirty lists, and everything made before is stale.
    fn install(&mut self, page: &Page, nodes: &[&Node], values: &[Values]) -> Result<()> {
        crate::canvas_host_memory::require_serial_growth()?;
        let (mut scene, bindings) = write_document(page, nodes, values)?.load()?;
        scene.continue_stamps_from(&self.scene);
        self.scene = scene;
        self.hashes = Default::default();
        self.bindings = bindings;
        self.placeable = None;
        self.background = page.background;
        self.forms = nodes.iter().zip(values).map(|(node, v)| Form::of(node, v)).collect();
        self.installs += 1;
        Ok(())
    }

    pub(super) fn profile_counters(&self) -> String {
        let stats = self.caches.stats_json();
        let (bytes, held, total) = self.plan_index.retained_size();
        format!("{},\"full_plans\":{},\"plan_retained_bytes\":{bytes},\"plan_resident_items\":{held},\"plan_total_items\":{total},\"placement_frames\":{},\"row_write_frames\":{},\"scene_installs\":{}}}", stats.trim_end_matches('}'),
            self.full_plans, self.placement_frames, self.write_frames, self.installs)
    }

    /// Bring `placeable` up to the scene: it is derived from the scene's records, and stands until something
    /// dirties everything.
    fn refresh_placeable(&mut self) {
        let all = self.scene.stamps().all();
        if self.placeable.as_ref().is_some_and(|(read_at, _)| *read_at >= all) { return; }
        // A sparse edit should classify only the bindings it visits.
        self.placeable = Some((self.scene.stamps().clock(), vec![None; self.bindings.list.len()]));
    }

    /// Bring the scene to `values`, given the items an evaluation may have changed since the values the scene
    /// holds and what of each (`change`). Each named field is written into the row bound to it: a transform or a
    /// body's opacity as typed placement numbers, anything else as typed fields. No node is copied or looked up
    /// by name, nothing is compared with a kept copy of the last values, and the cost follows `changed`.
    ///
    /// `changed` is an account of what may differ, not a proof that it does (a later layer can override a track
    /// that changed): the scene's writers leave a field that already holds its value alone, so a named field
    /// that did not change stamps nothing. `node(item)` is the item's node as it now stands.
    ///
    /// Returns false, having changed nothing, when a changed item would be lowered in another form (which Moda
    /// nodes there are changes): the caller then calls [`Self::update`], which installs.
    pub fn write<'a>(&mut self, page: &Page, node: impl Fn(usize) -> &'a Node, values: &[Values], changed: &[(usize, u8)]) -> Result<bool> {
        if self.background != page.background || self.forms.len() != values.len() { return Ok(false); }
        // A transform changes an item's form only through a zero scale: that alone is asked of a moved item.
        // Anything else of the item can change its form, and is asked whole.
        let zero = |v: &Values| v.transform.scale[0] == 0. || v.transform.scale[1] == 0.;
        if changed.iter().any(|&(item, what)| {
            if what & (change::REST | change::FILTER) != 0 { !self.forms[item].holds(node(item), &values[item]) }
            else { what & change::SCALE != 0 && self.forms[item].shape.zero_scale != zero(&values[item]) }
        }) {
            return Ok(false);
        }
        let _profile = moda_engine::prof::span("canvas.plan.update.writes");
        self.refresh_placeable();
        let (mut placements, mut rows) = (false, false);
        let before = self.scene.stamps().clock();
        // The effects bindings this batch touches, each written once after the items: an effect under several
        // moved ancestors reads all of their transforms as they now stand, whichever of them named it.
        let mut pending = std::mem::take(&mut self.pending_placements);
        let numbers = |&(item, what, opacity): &(usize, u8, Option<BoundOpacity>)| {
            let new = &values[item];
            let t = &new.transform;
            let f32n = |v| Number::from_f32_shortest(v).expect("finite placement");
            let transform = (what & change::TRANSFORM != 0).then(|| moda_scene::scene::NodePlacement {
                translation: (what & change::TRANSLATION != 0).then(|| [f32n(t.translation[0]), f32n(t.translation[1])]),
                rotation: (what & change::ROTATION != 0).then(|| {
                    let q = glam::Quat::from_array(t.resolved_rotation()).normalize();
                    Number::from_f64((2. * f64::from(q.z).atan2(f64::from(q.w))).to_degrees()).expect("finite rotation")
                }),
                scale: (what & change::SCALE != 0).then(|| [f32n(t.scale[0]), f32n(t.scale[1])]),
            });
            let opacity = opacity.and_then(|opacity| match opacity {
                BoundOpacity::Plain => Number::from_f32_shortest(new.opacity.clamp(0., 1.)),
                BoundOpacity::Product(factor) => Number::from_f64(f64::from(new.opacity.clamp(0., 1.)) * factor),
            });
            (transform, opacity)
        };
        let mut effects_touched: Vec<u32> = Vec::new();
        for &(item, what) in changed {
            let new = &values[item];
            if what & (change::TRANSFORM | change::OPACITY) != 0 {
                for at in self.bindings.of(item) {
                    let bound = &self.bindings.list[at];
                    let BoundField::Transform { opacity } = &bound.field else { continue };
                    let t = &new.transform;
                    // The typed writer classifies group opacity by its isolation state before and after the write.
                    let typed = self.placeable.as_mut().is_some_and(|(read_at, table)| self.scene.stamps().all() <= *read_at
                        && *table[at].get_or_insert_with(|| self.scene.placeable(bound.row)))
                        && t.translation[0].is_finite() && t.translation[1].is_finite() && t.scale[0].is_finite() && t.scale[1].is_finite()
                        && glam::Quat::from_array(t.resolved_rotation()).normalize().is_finite() && new.opacity.is_finite();
                    if typed {
                        // The numbers `transform_fields` writes: f32 translation and scale, f64 degrees.
                        let opacity = if what & change::OPACITY != 0 { *opacity } else { None };
                        pending.push(moda_scene::scene::PlacementEdit::new(bound.row, (item, what, opacity)));
                        continue;
                    }
                    placements |= self.scene.write_placements(&mut pending, numbers);
                    rows |= self.scene.write_row(bound.row, |fields| {
                        if what & change::TRANSFORM != 0 {
                            for (key, value) in transform_values(&new.transform).expect("a zero scale is another form") {
                                fields.set_known(key, value);
                            }
                        }
                        if what & change::OPACITY != 0 {
                            if let Some(opacity) = opacity {
                                // The writer gives a group's opacity as f32, and a body's collapsed opacity as an
                                // f64 product. Keep that rounding boundary byte for byte.
                                fields.set_known(K::Opacity, match opacity {
                                    BoundOpacity::Plain => json!(new.opacity.clamp(0., 1.)),
                                    BoundOpacity::Product(factor) => json!(f64::from(new.opacity.clamp(0., 1.)) * factor),
                                });
                            }
                        }
                    }).map_err(anyhow::Error::msg)?;
                }
            }
            // A canvas-space shadow offset follows every transform above it.
            if what & change::TRANSFORM != 0 {
                effects_touched.extend_from_slice(self.bindings.fed_by(item));
            }
            if what & change::REST == 0 {
                if what & change::FILTER != 0 {
                    for at in self.bindings.of(item) {
                        if matches!(self.bindings.list[at].field, BoundField::Effects(_)) {
                            effects_touched.push(at as u32);
                        }
                    }
                }
                continue;
            }
            placements |= self.scene.write_placements(&mut pending, numbers);
            let item_node = node(item);
            // A path's paints, made once for the path nodes that carry them.
            let mut paints = None;
            for at in self.bindings.of(item) {
                let bound = &self.bindings.list[at];
                match bound.field {
                    BoundField::Transform { .. } => continue,
                    BoundField::Effects(_) => {
                        effects_touched.push(at as u32);
                        continue;
                    }
                    _ => {}
                }
                rows |= self.scene.write_row(bound.row, |fields| {
                    let mut geometry = None;
                    match &bound.field {
                        BoundField::Transform { .. } | BoundField::Effects(_) => unreachable!(),
                        BoundField::Geometry => {
                            if let Kind::Path(path) = &item_node.kind {
                                // The row holds the geometry it was given: it compares by identity first, then by
                                // value, and takes only geometry that differs.
                                match &path.points {
                                    Points::Compact(points) => geometry = Some(points.clone()),
                                    Points::Json(points) => { fields.set_known(K::Points, Value::Array(points.clone())); }
                                }
                                fields.set_known(K::Closed, path.closed.clone());
                            }
                        }
                        BoundField::World => {
                            let (src, size) = new.world.as_ref().expect("a world without a picture is another form");
                            fields.set_known(K::Fills, world_fill(src, *size));
                        }
                        BoundField::Text { sub } => {
                            let content = json!([item as u64, *sub as u64, new.text_revision]);
                            if fields.node().known_field_eq(K::VorkText, &content) {
                                fields.set_known(K::VorkPaint, json!(new.text_paint_revisions.get(*sub).copied().unwrap_or(0)));
                                return;
                            }
                            fields.set_known(K::VorkText, content);
                            fields.set_known(K::VorkPaint, json!(new.text_paint_revisions.get(*sub).copied().unwrap_or(0)));
                            // The laid-out box of every text node of the item, clip copies included.
                            if let Kind::Text { size, .. } = &item_node.kind {
                                fields.set_known(K::Width, json!(size[0]));
                                fields.set_known(K::Height, json!(size[1]));
                            }
                        }
                        BoundField::TextPaint { part } => {
                            if let Some(TextPart::Paint { paint, size, bounds, .. }) = text_part(&new.text_composition, part) {
                                fields.set_known(K::Points, Value::Array(rect_points(*bounds)));
                                fields.set_known(K::Fills, json!([{"id": "fill", "paint": expanded_paint(paint, [0., 0., size[0], size[1]], *bounds)}]));
                            }
                        }
                        BoundField::TextFade { part } => {
                            if let Some(TextPart::Group { opacity, .. }) = text_part(&new.text_composition, part) {
                                fields.set_known(K::Opacity, json!(opacity));
                            }
                        }
                        BoundField::Path { part } => {
                            if let Kind::Path(path) = &item_node.kind {
                                // The picture's geometry is its rectangle, not the path's.
                                if *part != PathPart::Picture {
                                    match &path.points {
                                        Points::Compact(points) => geometry = Some(points.clone()),
                                        Points::Json(points) => { fields.set_known(K::Points, Value::Array(points.clone())); }
                                    }
                                    fields.set_known(K::Closed, path.closed.clone());
                                    fields.set_known(K::FillRule, json!(match path.fill_rule { v2::FillRule::NonZero => "nonzero", v2::FillRule::EvenOdd => "evenodd" }));
                                }
                                let v = new.path.unwrap_or(path.values);
                                let (fill, stroke) = paints.get_or_insert_with(|| {
                                    let writer = Writer { nodes: &[], values, list: Vec::new(), children: Vec::new(), bindings: Vec::new(), transform_path: Vec::new(), masks: 0, mask_paint_items: BTreeSet::new() };
                                    writer.path_paints(item, path, &v)
                                });
                                // Each part's share of the paints, as `Writer::body_inner` gives it.
                                match part {
                                    PathPart::Paints | PathPart::Fill | PathPart::Stroke => {
                                        let fills: Vec<_> = fill.iter().filter(|_| *part != PathPart::Stroke).cloned().collect();
                                        let strokes: Vec<_> = stroke.iter().filter(|_| *part != PathPart::Fill).cloned().collect();
                                        fields.set_known(K::Fills, Value::Array(fills));
                                        fields.set_known(K::Strokes, Value::Array(strokes));
                                    }
                                    PathPart::Coverage => {
                                        let white: Vec<Value> = stroke.iter().map(|stroke| {
                                            let mut white = stroke.clone();
                                            white["paint"] = json!({"type": "solid", "color": "#ffffffff"});
                                            white
                                        }).collect();
                                        fields.set_known(K::Fills, json!([]));
                                        fields.set_known(K::Strokes, Value::Array(white));
                                    }
                                    PathPart::Picture => {
                                        if let (Some(stroke), Some(bounds)) = (stroke.as_ref(), path.bounds) {
                                            let rect = coverage_rect(path, &v, bounds);
                                            fields.set_known(K::Points, Value::Array(rect_points(rect)));
                                            fields.set_known(K::Fills, json!([{"id": "fill", "paint": expanded_paint(&stroke["paint"], bounds, rect)}]));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if let Some(points) = geometry { fields.set_shared_points(points); }
                }).map_err(anyhow::Error::msg)?;
            }
        }
        placements |= self.scene.write_placements(&mut pending, numbers);
        self.pending_placements = pending;
        let only_effects = self.planned_at == Some(before) && self.scene.stamps().clock() == before;
        effects_touched.sort_unstable();
        effects_touched.dedup();
        for at in effects_touched {
            let bound = &self.bindings.list[at as usize];
            rows |= Self::write_effects(&mut self.scene, bound, page, node(bound.item), values)?;
        }
        let parameter = if only_effects { Some(K::Effects) }
            else if self.planned_at == Some(before) && changed.iter().all(|&(_, what)| what == change::OPACITY) { Some(K::Opacity) }
            else { None };
        self.plan_index.record_parameter_writes(parameter.map(|field| (before, self.scene.stamps().clock(), field)));
        use moda_scene::stamps::Group;
        rows |= self.scene.stamps().latest(Group::Paint) > before || self.scene.stamps().latest(Group::Content) > before;
        if rows { self.write_frames += 1; } else if placements { self.placement_frames += 1; }
        Ok(true)
    }

    /// Write an effects binding's node its effects: the item's filter, with offsets carried through the
    /// transforms above it as they now stand. Returns whether the row changed.
    fn write_effects(scene: &mut moda_scene::Scene, bound: &Bound, page: &Page, node: &Node, values: &[Values]) -> Result<bool> {
        let BoundField::Effects(effect) = &bound.field else { return Ok(false) };
        let linear = effect.transforms.iter().fold(
            glam::DMat2::from_diagonal(glam::DVec2::from_array(page.scale)),
            |parent, &i| parent * transform_linear(&values[i].transform));
        let list = match effect.stage {
            Some(i) => effect_list(&node.filter.0[i..=i], linear),
            None => effects(&node.filter, linear),
        };
        scene.write_row(bound.row, |fields| fields.set_known(K::Effects, Value::Array(list)))
            .map_err(anyhow::Error::msg)
    }

    /// Bring the scene to `values` without knowing what changed: every item's fields are made and compared with
    /// the rows that hold them, and the canvas is lowered again and installed when that is not enough. The cost
    /// follows the canvas, so this is for the first frame after an error, a frame whose base the scene does not
    /// hold, and tests; an evaluation that knows what it changed calls [`Self::write`].
    pub fn update(&mut self, page: &Page, nodes: &[&Node], values: &[Values]) -> Result<()> {
        if nodes.len() == values.len() {
            let every: Vec<(usize, u8)> = (0..values.len()).map(|item| (item, change::ALL)).collect();
            if self.write(page, |item| nodes[item], values, &every)? { return Ok(()); }
        }
        let _install_profile = moda_engine::prof::span("canvas.plan.update.install");
        self.install(page, nodes, values)
    }
}

fn world_fill(src: &str, [w, h]: [u32; 2]) -> Value {
    json!([{"id": "fill", "paint": {"type": "image", "src": src,
        "placement": {"fit": "fill", "naturalSize": {"width": w, "height": h}}}, "opacity": 1}])
}

struct Writer<'a> {
    nodes: &'a [&'a Node],
    values: &'a [Values],
    list: Vec<Value>,
    children: Vec<Vec<usize>>,
    bindings: Vec<Bound>,
    transform_path: Vec<usize>,
    /// How many mask copies enclose the node being written. A node is written once outside masks, and once
    /// for every node that names it (or an ancestor) as a clip or mask.
    masks: usize,
    mask_paint_items: BTreeSet<usize>,
}

/// A group carries its opacity directly. A collapsed body multiplies it by its paint opacity.
#[derive(Clone, Copy)]
enum BoundOpacity {
    Plain,
    Product(f64),
}
/// Which of a path item's Moda path nodes a binding writes: each holds its own share of the item's paints.
#[derive(Clone, Copy, PartialEq)]
enum PathPart {
    /// The one node that draws both fill and stroke.
    Paints,
    /// The fill alone (`paint-order: stroke`, or beside a stroke drawn through its coverage).
    Fill,
    /// The stroke alone (`paint-order: stroke`).
    Stroke,
    /// An image or conic stroke's coverage: the stroke in white, the mask its picture is drawn through.
    Coverage,
    /// That stroke's picture: a rectangle around the stroke, filled with the paint in its authored box.
    Picture,
}
impl PathPart {
    /// The part a path node written by `Writer::body_inner` plays, from the suffix that writer gives its id.
    fn of(id: &str) -> Self {
        if id.ends_with(".coverage") { Self::Coverage }
        else if id.ends_with(".paint") { Self::Picture }
        else if id.ends_with(".stroke") { Self::Stroke }
        else if id.ends_with(".fill") { Self::Fill }
        else { Self::Paints }
    }
}

/// The piece at `at` in a text composition (an index at each level of nesting).
fn text_part<'a>(parts: &'a [TextPart], at: &[usize]) -> Option<&'a TextPart> {
    let (first, rest) = at.split_first()?;
    let part = parts.get(*first)?;
    if rest.is_empty() { return Some(part); }
    match part {
        TextPart::Group { children, .. } => text_part(children, rest),
        _ => None,
    }
}

/// The rectangle an image or conic stroke's picture covers: the path's bounds grown by the widest the stroke
/// reaches (a miter).
fn coverage_rect(path: &PathItem, v: &PathValues, bounds: [f64; 4]) -> [f64; 4] {
    let pad = f64::from(v.stroke_width) * f64::from(path.miterlimit.max(2.));
    [bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad]
}

/// Moda's point objects for a rectangle.
fn rect_points([x0, y0, x1, y1]: [f64; 4]) -> Vec<Value> {
    vec![json!({"type": "move", "x": x0, "y": y0}), json!({"type": "line", "x": x1, "y": y0}),
        json!({"type": "line", "x": x1, "y": y1}), json!({"type": "line", "x": x0, "y": y1})]
}

struct EffectBinding {
    transforms: Vec<usize>,
    stage: Option<usize>,
}

enum BoundField {
    Transform { opacity: Option<BoundOpacity> },
    Path { part: PathPart },
    Text { sub: usize },
    /// The picture of a text paint: its rectangle and the paint, for the piece at `part` in the item's
    /// composition (an index at each level of nesting).
    TextPaint { part: Vec<usize> },
    /// The group that fades the pieces at `part` in the item's composition together.
    TextFade { part: Vec<usize> },
    World,
    Geometry,
    /// Actual writer ancestry, including each occurrence of a referenced mask.
    Effects(Box<EffectBinding>),
}

/// Moda's local transform fields (`x`, `y`, `rotation` in degrees, `scaleX`, `scaleY`) for a canvas transform,
/// or `None` when a scale is zero (the node draws nothing).
fn transform_values(t: &v2::Transform) -> Option<[(K, Value); 5]> {
    (t.scale[0] != 0. && t.scale[1] != 0.).then(|| placement_fields(t))
}

fn placement_fields(t: &v2::Transform) -> [(K, Value); 5] {
    let [x, y, _] = t.translation;
    let [sx, sy, _] = t.scale;
    let q = glam::Quat::from_array(t.resolved_rotation()).normalize();
    let degrees = (2. * f64::from(q.z).atan2(f64::from(q.w))).to_degrees();
    // An f32 is the f64 its shortest decimal text reads back as (what a JSON document carried); a non-finite
    // value is null, as JSON writes it.
    let f32n = |v: f32| Number::from_f32_shortest(v).map_or(Value::Null, Value::Number);
    [(K::X, f32n(x)), (K::Y, f32n(y)), (K::Rotation, Value::from(degrees)),
        (K::ScaleX, f32n(sx)), (K::ScaleY, f32n(sy))]
}

pub(crate) fn placed_matrix(page: &Page, chain: &[&v2::Transform]) -> Result<[f64; 16]> {
    use moda_scene::{
        transform::{compose, from_trs},
        types::Transform2D,
    };
    let mut m = from_trs(Transform2D {
        x: page.offset[0],
        y: page.offset[1],
        rotation: 0.,
        scale_x: page.scale[0],
        scale_y: page.scale[1],
    });
    for t in chain.iter().rev() {
        let fields = placement_fields(t);
        let [x, y, rotation, scale_x, scale_y] =
            fields.map(|(_, v)| v.as_f64().unwrap_or(f64::NAN));
        m = compose(
            m,
            from_trs(Transform2D {
                x,
                y,
                rotation,
                scale_x,
                scale_y,
            }),
        );
    }
    let matrix = [
        m.a, m.b, 0., 0., m.c, m.d, 0., 0., 0., 0., 1., 0., m.e, m.f, 0., 1.,
    ];
    ensure!(
        matrix.iter().all(|v| v.is_finite()),
        "ask: nonfinite canvas transform"
    );
    Ok(matrix)
}

fn transform_fields(t: &v2::Transform) -> Option<Map> {
    Some(transform_values(t)?.into_iter().map(|(key, value)| (moda_scene::value::Key::from(key), value)).collect())
}

/// The linear part of the same local transform written by `transform_fields`.
fn transform_linear(t: &v2::Transform) -> glam::DMat2 {
    let q = glam::Quat::from_array(t.resolved_rotation()).normalize();
    let angle = 2. * f64::from(q.z).atan2(f64::from(q.w));
    glam::DMat2::from_angle(angle) * glam::DMat2::from_diagonal(glam::DVec2::new(f64::from(t.scale[0]), f64::from(t.scale[1])))
}

impl Writer<'_> {
    /// Write node `i` and its subtree with Moda id `id`. Return its root row when it draws.
    /// `stack` holds the nodes being written as masks, to refuse a node that masks itself.
    fn node(&mut self, i: usize, id: &str, referenced: &[bool], stack: &mut Vec<usize>, parent_linear: glam::DMat2) -> Result<Option<usize>> {
        self.transform_path.push(i);
        let result = self.node_inner(i, id, referenced, stack, parent_linear);
        self.transform_path.pop();
        result
    }

    fn node_inner(&mut self, i: usize, id: &str, referenced: &[bool], stack: &mut Vec<usize>, parent_linear: glam::DMat2) -> Result<Option<usize>> {
        let values = &self.values[i];
        if !values.visible {
            return Ok(None);
        }
        let Some(mut outer) = transform_fields(&values.transform) else { return Ok(None) };
        let linear = parent_linear * transform_linear(&values.transform);
        let node = &self.nodes[i];
        // The node's own picture at identity: its body and its children, under its filter.
        let mut content = Vec::new();
        if let Some(body) = self.body(i, &format!("{id}.body"))? {
            if self.masks != 0 { self.mask_paint_items.insert(i); }
            content.push(body);
        }
        let body_row = self.list.len().checked_sub(1);
        for &child in &self.children[i].clone() {
            if referenced[child] {
                continue;
            }
            // Outside masks an item is written once, so its index names it: ids do not grow with depth.
            // Inside a mask copy the id carries the copy's path, which tells the copies apart.
            let child_id = if self.masks == 0 { format!("n{child}") } else { format!("{id}/{child}") };
            if self.node(child, &child_id, referenced, stack, linear)?.is_some() {
                content.push(child_id);
            }
        }
        if content.is_empty() {
            return Ok(None);
        }
        let wraps = node.clip_path.is_some() || node.mask.is_some();
        let lone_body = content.len() == 1 && self.children[i].iter().all(|c| referenced[*c]) && !wraps;
        // Richtext opacity multiplies each glyph. A text body keeps node opacity and filter on a group even
        // for one batch: the group's captured picture is what fades and what is filtered.
        let text_body = lone_body && self.list[body_row.expect("written body")]["type"] == "richtext";
        let single_capture = node.filter.single_capture();
        let effects = if single_capture { effects(&node.filter, linear) } else { Vec::new() };
        // Other lone bodies without clip or mask carry the transform, opacity and filter themselves. A drop
        // shadow is the exception: CSS fades a node after its filter, and Moda fades a leaf before it. For
        // blur() the two are the same picture; a shadow drawn under a faded source is not (the shadow would
        // show through an opaque shape as it fades). Such a node keeps its opacity on the group around the
        // filtered picture, as a text node does.
        let blended = node.blend != v2::MixBlendMode::Normal;
        let isolated = node.isolation == v2::Isolation::Isolate;
        if blended || isolated {
            outer.insert("blendMode".into(), json!(node.blend));
        }
        if isolated {
            outer.insert("isolation".into(), json!("isolate"));
        }
        if lone_body && !text_body && !node.filter.has_shadow() && single_capture && !blended && !isolated {
            let row = body_row.expect("written body");
            let object = self.list[row].as_object_mut().unwrap();
            object.extend(outer);
            self.bindings.push(Bound { item: i, row: row as u32, field: BoundField::Transform {
                opacity: Some(BoundOpacity::Product(object.get("opacity").and_then(Value::as_f64).unwrap_or(1.))),
            }});
            object.insert("id".into(), json!(id));
            let opacity = f64::from(values.opacity.clamp(0., 1.)) * object.get("opacity").and_then(Value::as_f64).unwrap_or(1.);
            object.insert("opacity".into(), json!(opacity));
            if !effects.is_empty() {
                object.insert("effects".into(), json!(effects));
            }
            if !effects.is_empty() {
                self.bindings.push(Bound { item: i, row: row as u32, field: BoundField::Effects(Box::new(EffectBinding { transforms: self.transform_path.clone(), stage: None })) });
            }
            return Ok(Some(row));
        }
        let mut inner = if content.len() == 1 && effects.is_empty() { content[0].clone() } else {
            let core = format!("{id}.core");
            if !effects.is_empty() {
                self.bindings.push(Bound { item: i, row: self.list.len() as u32, field: BoundField::Effects(Box::new(EffectBinding { transforms: self.transform_path.clone(), stage: None })) });
            }
            self.list.push(json!({"id": core, "type": "group", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1,
                "children": content, "effects": effects}));
            core
        };
        if !single_capture {
            for (stage, op) in node.filter.0.iter().enumerate() {
                let group = format!("{id}.filter-{stage}");
                self.bindings.push(Bound { item: i, row: self.list.len() as u32, field: BoundField::Effects(Box::new(EffectBinding {
                    transforms: self.transform_path.clone(), stage: Some(stage),
                })) });
                self.list.push(json!({"id":group,"type":"group","children":[inner],
                    "effects":effect_list(std::slice::from_ref(op), linear)}));
                inner = group;
            }
        }
        for (target, kind) in [(node.clip_path, "vector"), (node.mask, node.mask.map_or("luminance", |target| match self.nodes[target].mask_type {
            vork_scene::v2::MaskType::Alpha => "alpha",
            vork_scene::v2::MaskType::Luminance => "luminance",
        }))] {
            let Some(target) = target else { continue };
            ensure!(!stack.contains(&target), "clip-path and mask: node {target} masks itself through the nodes that name it");
            // Alpha masks cover the composited picture once. A luminance mask uses the existing mask plan.
            if kind == "alpha" {
                let picture = self.list.last().filter(|row| row["id"] == inner.as_str());
                let single_paint = picture.is_some_and(|row| row["type"] == "path"
                    && ["fills", "strokes"].iter().map(|key| row[*key].as_array().map_or(0, Vec::len)).sum::<usize>() <= 1);
                if !single_paint {
                    let picture_id = format!("{id}.mask-content");
                    self.list.push(json!({"id":picture_id,"type":"group","children":[inner],"isolation":"isolate","blendMode":"normal"}));
                    inner = picture_id;
                }
            }
            if let Some(picture) = self.list.iter_mut().rev().find(|row| row["id"] == inner.as_str()) {
                Self::capture_mask_child(picture);
            }
            stack.push(i);
            let mask = self.mask(target, &format!("{id}.{kind}"), kind == "vector", referenced, stack, linear)?;
            stack.pop();
            let group = format!("{id}.{kind}-group");
            match mask {
                Some(row) => {
                    // A group of shapes clips by its rendered coverage (Moda's alpha mask: every fill is opaque), as
                    // Moda's vector mask reads one shape's outline only.
                    let mask_id = self.set_mask_type(row, kind);
                    self.list.push(json!({"id": group, "type": "group", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1,
                        "children": [mask_id, inner], "clipMaskNodeId": mask_id}));
                    inner = group;
                }
                // A mask that draws nothing hides everything (SVG: an empty clip or a transparent mask).
                None => return Ok(None),
            }
        }
        outer.insert("id".into(), json!(id));
        outer.insert("type".into(), json!("group"));
        outer.insert("children".into(), json!([inner]));
        outer.insert("opacity".into(), json!(values.opacity.clamp(0., 1.)));
        self.bindings.push(Bound { item: i, row: self.list.len() as u32, field: BoundField::Transform { opacity: Some(BoundOpacity::Plain) } });
        self.list.push(Value::Object(outer));
        Ok(Some(self.list.len() - 1))
    }

    /// On either side of a mask, an isolated group's children compose before its mask is applied.
    /// Its CSS filter effects, if any, already belong to inner rows of this group.
    fn capture_mask_child(row: &mut Value) {
        if row["isolation"] == "isolate" {
            row["effects"] = json!([{"id":"mask-content","type":"mask-capture"}]);
        }
    }

    fn set_mask_type(&mut self, row: usize, kind: &str) -> String {
        let node = &mut self.list[row];
        Self::capture_mask_child(node);
        node["maskType"] = json!(if kind == "vector" && node["type"] == "group" { "alpha" } else { kind });
        node["id"].as_str().unwrap().to_owned()
    }

    /// A copy of node `i`'s subtree as a mask, in the masked node's coordinates (its own transform kept). For a
    /// clip (`vector`), every path keeps only its fill geometry: an opaque fill, no stroke, opacity or filter.
    fn mask(&mut self, i: usize, id: &str, vector: bool, referenced: &[bool], stack: &mut Vec<usize>, parent_linear: glam::DMat2) -> Result<Option<usize>> {
        self.masks += 1;
        let result = self.mask_inner(i, id, vector, referenced, stack, parent_linear);
        self.masks -= 1;
        result
    }

    fn mask_inner(&mut self, i: usize, id: &str, vector: bool, referenced: &[bool], stack: &mut Vec<usize>, parent_linear: glam::DMat2) -> Result<Option<usize>> {
        if !vector {
            if !matches!(self.nodes[i].kind, Kind::Text { .. } | Kind::World { .. }) {
                return self.node(i, id, referenced, stack, parent_linear);
            }
            let Some(body) = self.node(i, &format!("{id}.picture"), referenced, stack, parent_linear)? else { return Ok(None) };
            if self.list[body]["type"] == "group" {
                return Ok(Some(body));
            }
            // Render the source picture through the same host builder as visible text. A group also
            // retains colours, strokes, effects and nested masks for luminance coverage.
            self.list.push(json!({"id": id, "type": "group", "x": 0, "y": 0,
                "children": [self.list[body]["id"]]}));
            return Ok(Some(self.list.len() - 1));
        }
        let values = &self.values[i];
        if !values.visible {
            return Ok(None);
        }
        let Some(mut fields) = transform_fields(&values.transform) else { return Ok(None) };
        let linear = parent_linear * transform_linear(&values.transform);
        let solid = |id: &str, path: &PathItem| json!({"id": id, "type": "path", "x": 0, "y": 0, "points": path.points.field(), "closed": path.closed,
            "fillRule": "nonzero", "fills": [{"id": "fill", "paint": {"type": "solid", "color": "#000000ff"}}]});
        // A lone path is the mask node itself, at its transform: Moda's vector mask (its outline at full coverage)
        // reads a shape node's geometry.
        if let Kind::Path(path) = &self.nodes[i].kind {
            if self.children[i].is_empty() && self.nodes[i].clip_path.is_none() {
                if path.points.is_empty() {
                    return Ok(None);
                }
                let mut node = solid(id, path);
                node.as_object_mut().unwrap().extend(fields);
                self.bindings.push(Bound { item: i, row: self.list.len() as u32, field: BoundField::Transform { opacity: None } });
                self.bindings.push(Bound { item: i, row: self.list.len() as u32, field: BoundField::Geometry });
                self.list.push(node);
                return Ok(Some(self.list.len() - 1));
            }
        }
        let mut children = Vec::new();
        if let Kind::Path(path) = &self.nodes[i].kind {
            if !path.points.is_empty() {
                let body = format!("{id}.body");
                self.list.push(solid(&body, path));
                self.bindings.push(Bound { item: i, row: (self.list.len() - 1) as u32, field: BoundField::Geometry });
                children.push(body);
            }
        }
        if let Kind::World { bounds: [x0, y0, x1, y1] } = &self.nodes[i].kind {
            let body = format!("{id}.picture-shape");
            self.list.push(json!({"id": body, "type": "path", "x": 0, "y": 0, "closed": true,
                "points": [{"type": "move", "x": x0, "y": y0}, {"type": "line", "x": x1, "y": y0},
                    {"type": "line", "x": x1, "y": y1}, {"type": "line", "x": x0, "y": y1}],
                "fills": [{"id": "fill", "paint": {"type": "solid", "color": "#ffffffff"}}]}));
            children.push(body);
        }
        // The clip glyphs follow the ordinary text batches when the evaluator built them.
        let sub = values.texts;
        if let Kind::Text { size, .. } = &self.nodes[i].kind && let Some(&paint) = values.text_paint_revisions.get(sub) {
            let body = format!("{id}.glyphs");
            self.list.push(json!({"id": body, "type": "richtext", "x": 0, "y": 0,
                "width": size[0], "height": size[1], "vorkText": [i, sub, values.text_revision], "vorkPaint": paint}));
            self.bindings.push(Bound { item: i, row: (self.list.len() - 1) as u32, field: BoundField::Text { sub } });
            children.push(body);
        }
        for &child in &self.children[i].clone() {
            if referenced[child] {
                continue;
            }
            let child_id = format!("{id}/{child}");
            if self.mask(child, &child_id, true, referenced, stack, linear)?.is_some() {
                children.push(child_id);
            }
        }
        if children.is_empty() {
            return Ok(None);
        }
        // The clip source's own clip intersects its fill geometry before its transform.
        // A group mask is rendered as alpha coverage, preserving nested intersections.
        if let Some(target) = self.nodes[i].clip_path {
            ensure!(target != i && !stack.contains(&target), "clip-path and mask: node {target} masks itself through the nodes that name it");
            stack.push(i);
            let mask = self.mask(target, &format!("{id}.clip"), true, referenced, stack, linear)?;
            stack.pop();
            let Some(mask) = mask else { return Ok(None) };
            let mask = self.set_mask_type(mask, "vector");
            let clipped = format!("{id}.clipped");
            children.insert(0, mask.clone());
            self.list.push(json!({"id": clipped, "type": "group", "x": 0, "y": 0, "rotation": 0,
                "scaleX": 1, "scaleY": 1, "children": children, "clipMaskNodeId": mask}));
            children = vec![clipped];
        }
        fields.insert("id".into(), json!(id));
        fields.insert("type".into(), json!("group"));
        fields.insert("children".into(), json!(children));
        self.bindings.push(Bound { item: i, row: self.list.len() as u32, field: BoundField::Transform { opacity: None } });
        self.list.push(Value::Object(fields));
        Ok(Some(self.list.len() - 1))
    }

    /// Node `i`'s own drawing at identity (a path's fill and stroke, a text's box), or `None`.
    fn body(&mut self, i: usize, id: &str) -> Result<Option<String>> {
        let start = self.list.len();
        let result = self.body_inner(i, id)?;
        for (index, node) in self.list[start..].iter().enumerate() {
            let row = (start + index) as u32;
            if matches!(self.nodes[i].kind, Kind::Path(_)) {
                if node["type"] == "path" {
                    let part = PathPart::of(node["id"].as_str().unwrap());
                    self.bindings.push(Bound { item: i, row, field: BoundField::Path { part } });
                }
            } else if matches!(self.nodes[i].kind, Kind::World { .. }) && node["type"] == "path" {
                self.bindings.push(Bound { item: i, row, field: BoundField::World });
            } else if node["type"] == "richtext" {
                self.bindings.push(Bound { item: i, row,
                    field: BoundField::Text { sub: node["vorkText"][1].as_u64().unwrap() as usize } });
            }
        }
        Ok(result)
    }

    fn body_inner(&mut self, i: usize, id: &str) -> Result<Option<String>> {
        match &self.nodes[i].kind {
            Kind::Group => Ok(None),
            Kind::World { bounds: [x0, y0, x1, y1] } => {
                let Some((src, [w, h])) = &self.values[i].world else { return Ok(None) };
                let corner = |x: f64, y: f64| json!({"type": if x == *x0 && y == *y0 { "move" } else { "line" }, "x": x, "y": y});
                self.list.push(json!({"id": id, "type": "path", "x": 0, "y": 0,
                    "points": [corner(*x0, *y0), corner(*x1, *y0), corner(*x1, *y1), corner(*x0, *y1)], "closed": true, "fillRule": "nonzero",
                    "fills": world_fill(src, [*w, *h]), "strokes": []}));
                Ok(Some(id.to_string()))
            }
            Kind::Text { size, .. } => {
                // `vorkText` names the text item and its batch for the text builder (`plan`); Moda keeps fields it
                // does not know.
                let texts = self.values[i].texts;
                let richtext = |id: &str, sub: usize| json!({"id": id, "type": "richtext", "x": 0, "y": 0, "width": size[0], "height": size[1],
                    "vorkText": [i as u64, sub as u64, self.values[i].text_revision], "vorkPaint": self.values[i].text_paint_revisions.get(sub).copied().unwrap_or(0)});
                if !self.values[i].text_composition.is_empty() {
                    // `at` is where `parts` sits in the item's composition: each paint's picture and each fading
                    // group is bound by its place there, so their values are fields of nodes that stay.
                    fn write(list: &mut Vec<Value>, bindings: &mut Vec<Bound>, item: usize, at: &mut Vec<usize>, id: &str, parts: &[TextPart], richtext: &impl Fn(&str, usize) -> Value) {
                        let mut children = Vec::new();
                        for (index, part) in parts.iter().enumerate() {
                            let child = format!("{id}.{index}");
                            at.push(index);
                            match part {
                                TextPart::Batch(sub) => list.push(richtext(&child, *sub)),
                                TextPart::Paint { sub, paint, size, bounds: [x0, y0, x1, y1] } => {
                                    let paint = expanded_paint(paint, [0., 0., size[0], size[1]], [*x0, *y0, *x1, *y1]);
                                    let glyphs = format!("{child}.glyphs");
                                    let mask = format!("{child}.mask");
                                    let picture = format!("{child}.paint");
                                    list.push(richtext(&glyphs, *sub));
                                    list.push(json!({"id": mask, "type": "group", "x": 0, "y": 0,
                                        "maskType": "alpha", "children": [glyphs]}));
                                    bindings.push(Bound { item, row: list.len() as u32, field: BoundField::TextPaint { part: at.clone() } });
                                    list.push(json!({"id": picture, "type": "path", "x": 0, "y": 0,
                                        "points": [{"type": "move", "x": x0, "y": y0}, {"type": "line", "x": x1, "y": y0},
                                            {"type": "line", "x": x1, "y": y1}, {"type": "line", "x": x0, "y": y1}],
                                        "closed": true, "fills": [{"id": "fill", "paint": paint}]}));
                                    list.push(json!({"id": child, "type": "group", "x": 0, "y": 0,
                                        "children": [mask, picture], "clipMaskNodeId": mask}));
                                }
                                TextPart::Group { opacity, children } => {
                                    write(list, bindings, item, at, &child, children, richtext);
                                    list.last_mut().unwrap()["opacity"] = json!(opacity);
                                    bindings.push(Bound { item, row: (list.len() - 1) as u32, field: BoundField::TextFade { part: at.clone() } });
                                }
                            }
                            at.pop();
                            children.push(child);
                        }
                        list.push(json!({"id": id, "type": "group", "x": 0, "y": 0, "rotation": 0,
                            "scaleX": 1, "scaleY": 1, "children": children}));
                    }
                    write(&mut self.list, &mut self.bindings, i, &mut Vec::new(), id, &self.values[i].text_composition, &richtext);
                    return Ok(Some(id.to_string()));
                }
                match texts {
                    0 => Ok(None),
                    1 => {
                        self.list.push(richtext(id, 0));
                        Ok(Some(id.to_string()))
                    }
                    _ => {
                        let children: Vec<String> = (0..texts).map(|sub| format!("{id}.{sub}")).collect();
                        for (sub, child) in children.iter().enumerate() {
                            self.list.push(richtext(child, sub));
                        }
                        self.list.push(json!({"id": id, "type": "group", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1, "children": children}));
                        Ok(Some(id.to_string()))
                    }
                }
            }
            Kind::Path(path) => {
                if path.points.is_empty() {
                    return Ok(None);
                }
                let v = self.values[i].path.unwrap_or(path.values);
                let (fill, stroke) = self.path_paints(i, path, &v);
                static PATH: std::sync::LazyLock<RecordLayout<9>> = std::sync::LazyLock::new(|| RecordLayout::new([
                    K::Id, K::Type, K::X, K::Y, K::Points, K::Closed, K::FillRule, K::Fills, K::Strokes,
                ]));
                let base = |id: &str, fills: Vec<Value>, strokes: Vec<Value>| PATH.record([
                    id.into(), "path".into(), 0.into(), 0.into(), path.points.field(), path.closed.clone(),
                    match path.fill_rule { v2::FillRule::NonZero => "nonzero", v2::FillRule::EvenOdd => "evenodd" }.into(),
                    Value::Array(fills), Value::Array(strokes),
                ]);
                if let Some(stroke) = stroke.as_ref().filter(|stroke| matches!(stroke["paint"]["type"].as_str(), Some("image" | "conic"))) {
                    let Some(bounds) = path.bounds else { return Ok(None) };
                    let rect = coverage_rect(path, &v, bounds);
                    let coverage = format!("{id}.coverage");
                    let mask = format!("{id}.mask");
                    let picture = format!("{id}.paint");
                    let outline = format!("{id}.stroke");
                    let mut white = stroke.clone();
                    white["paint"] = json!({"type": "solid", "color": "#ffffffff"});
                    self.list.push(base(&coverage, vec![], vec![white]));
                    self.list.push(json!({"id": mask, "type": "group", "x": 0, "y": 0,
                        "maskType": "alpha", "children": [coverage]}));
                    self.list.push(json!({"id": picture, "type": "path", "x": 0, "y": 0, "closed": true,
                        "points": rect_points(rect),
                        "fills": [{"id": "fill", "paint": expanded_paint(&stroke["paint"], bounds, rect)}]}));
                    self.list.push(json!({"id": outline, "type": "group", "x": 0, "y": 0,
                        "children": [mask, picture], "clipMaskNodeId": mask}));
                    let mut children = vec![outline];
                    if let Some(fill) = fill {
                        let body = format!("{id}.fill");
                        self.list.push(base(&body, vec![fill], vec![]));
                        if path.stroke_first { children.push(body); } else { children.insert(0, body); }
                    }
                    self.list.push(json!({"id": id, "type": "group", "x": 0, "y": 0, "children": children}));
                    return Ok(Some(id.to_string()));
                }
                match (fill, stroke) {
                    (None, None) => Ok(None),
                    (Some(fill), Some(stroke)) if path.stroke_first => {
                        // paint-order: stroke under the fill — two nodes, composited as one group.
                        let (under, over) = (format!("{id}.stroke"), format!("{id}.fill"));
                        self.list.push(base(&under, vec![], vec![stroke]));
                        self.list.push(base(&over, vec![fill], vec![]));
                        self.list.push(json!({"id": id, "type": "group", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1, "children": [under, over]}));
                        Ok(Some(id.to_string()))
                    }
                    (fill, stroke) => {
                        self.list.push(base(id, fill.into_iter().collect(), stroke.into_iter().collect()));
                        Ok(Some(id.to_string()))
                    }
                }
            }
        }
    }

    fn path_paints(&self, item: usize, path: &PathItem, v: &PathValues) -> (Option<Value>, Option<Value>) {
        static SOLID: std::sync::LazyLock<RecordLayout<2>> = std::sync::LazyLock::new(|| RecordLayout::new([K::Type, K::Color]));
        static FILL: std::sync::LazyLock<RecordLayout<3>> = std::sync::LazyLock::new(|| RecordLayout::new([K::Id, K::Paint, K::Opacity]));
        let [fill, stroke] = path.paints(v).map(|source| match source.as_ref() {
            PaintSource::Color(color) => Some(SOLID.record(["solid".into(), vork_std::moda::moda_color(*color).into()])),
            PaintSource::Program(binding) => Some(program_record(moda_scene::paint::PathProgram {
                clock: path.paint_clock, site: item as u32, ..*binding
            })),
            PaintSource::Moda(paint) if missing_media(paint) => Some(SOLID.record(["solid".into(), "#00000000".into()])),
            PaintSource::Moda(paint) => Some(paint.clone()),
            PaintSource::None => None,
        });
        let fill = fill.map(|paint| FILL.record(["fill".into(), paint, json!(v.fill_opacity.clamp(0., 1.))]));
        let stroke = stroke.map(|paint| Self::stroke_layer(path, v, paint));
        (fill, stroke)
    }

    /// One Moda stroke layer: SVG px dash lengths scaled by `pathLength` and divided by half the stroke width
    /// (Moda's `scale_dash_pattern` multiplies its pattern by `width / 2`).
    fn stroke_layer(path: &PathItem, v: &PathValues, paint: Value) -> Value {
        let width = f64::from(v.stroke_width);
        let mut layer = json!({"id": "stroke", "paint": paint, "width": width, "opacity": v.stroke_opacity.clamp(0., 1.),
            "cap": match path.linecap { v2::LineCap::Butt => "butt", v2::LineCap::Round => "round", v2::LineCap::Square => "square" },
            "join": match path.linejoin { v2::LineJoin::Miter => "miter", v2::LineJoin::Round => "round", v2::LineJoin::Bevel => "bevel" },
            "miterLimit": path.miterlimit});
        let sum: f32 = path.dasharray.iter().sum();
        if sum > 0. || path.sampled_dash.is_some() {
            let length_scale = path.path_length.map_or(1., |authored| path.length / f64::from(authored));
            let to_moda = |px: f64| px * length_scale * 2. / width;
            if let Some(list) = &path.sampled_dash {
                layer["sampledDash"] = json!(moda_scene::dash::SampledDash { list: list.clone(), scale: length_scale });
            } else {
                layer["dash"] = path.dasharray.iter().map(|v| to_moda(f64::from(*v))).collect();
            }
            layer["dashOffset"] = json!(to_moda(f64::from(v.stroke_dashoffset)));
        }
        layer
    }
}

/// A node's filter as Moda's layer filters, in CSS's order; `linear` (the node, ancestor and viewBox linear
/// transform) converts the authored offset vector to canvas space, where Moda takes a shadow's offset.
fn effects(filter: &Filter, linear: glam::DMat2) -> Vec<Value> {
    effect_list(&filter.0, linear)
}
fn effect_list(ops: &[FilterOp], linear: glam::DMat2) -> Vec<Value> {
    let mut out = Vec::new();
    for op in ops { match *op {
    FilterOp::Shadow([x, y], sigma, color) => {
        let offset = linear * glam::DVec2::new(x, y);
        let (x, y) = (offset.x, offset.y);
        // CSS Filter Effects: the input's alpha, offset, blurred by the standard deviation, tinted, under the
        // input. That is Moda's filter shadow, whose blur is a radius (σ = blur / 2).
        out.push(json!({"id": "drop-shadow", "type": "filter-shadow", "color": vork_std::moda::moda_color(color), "blur": 2. * sigma,
            "offset": {"x": x, "y": y}}));
    }
    FilterOp::Blur(sigma) if sigma > 0. => {
        // Moda's layer blur radius follows Figma: σ = radius / 2.
        out.push(json!({"id": "blur", "type": "layer-blur", "radius": 2. * sigma}));
    }
    FilterOp::Blur(_) => {}
    FilterOp::Color(function, amount) => out.push(json!({"id":"css-color","type":"color-matrix","values":function.matrix(amount)})),
    }}
    out
}

/// The worlds' pictures of one frame, put in Moda's asset store as GPU captures under their `src` keys (the
/// previous frame's leave it).
pub(crate) fn install_captures(captures: &[(String, [u32; 2], u64)]) {
    let prefix = format!("{}gpu/", moda_engine::effects::PIXEL_SINK_SRC_PREFIX);
    let mut store = moda_assets::engine_seam::installed().map(|installed| (*installed).clone()).unwrap_or_default();
    store.clear_prefixed(&prefix);
    for (src, [width, height], upload_id) in captures {
        store.insert_external(src.clone(), moda_assets::ExternalImageMeta { width: *width, height: *height, scale_factor: (1., 1.), upload_id: *upload_id });
    }
    moda_assets::engine_seam::install_store(std::sync::Arc::new(store));
}

/// Every image the paths paint, read and decoded once into Moda's asset store (`moda_assets::engine_seam`),
/// which Moda's frame walk reads them from. A paint's `src` is the store's key: a path, or the path's natural
/// key for a canvas that takes decoded texels (`Media::decoded_texels`), which Moda's loader prepares so.
pub(crate) fn install_images(files: &crate::asset_revision::SourceFiles, nodes: &[&Node], text_paints: &[&PaintSource]) -> Result<()> {
    let _decode = moda_engine::prof::span("canvas.input_decode");
    let installed=moda_assets::engine_seam::installed();
    let mut requests = BTreeSet::new();
    let paints = nodes.iter().flat_map(|node| match &node.kind {
        Kind::Path(path) => vec![&path.fill, &path.stroke],
        _ => Vec::new(),
    }).chain(text_paints.iter().copied());
    for paint in paints {
        if let PaintSource::Moda(paint) = paint {
            if let Some(src) = paint.get("src").and_then(Value::as_str) {
                if src.starts_with(MEDIA_PREFIX) { continue; }
                if installed.as_ref().is_some_and(|s|s.dimensions(src).is_some()){continue;}
                requests.insert(src.to_string());
            }
        }
    }
    if requests.is_empty() {
        return Ok(());
    }
    let store = moda_assets::fetch::prefetch_canvas_rasters(&requests, still_paint_source, |src| {
        let input = files.source(Path::new(src));
        input.read().with_context(|| format!("read {}", input.name.display()))
    }, |src, bytes, natural, srgb| {
        // The missing-file swatch is an untagged PNG embedded in the renderer.
        if src.starts_with("data:") {
            return if natural { moda_assets::decode::prepare_raster_natural(bytes) }
                else { moda_assets::decode::prepare_raster(bytes) };
        }
        let input = files.source(Path::new(src));
        let path = &input.name;
        let decoded = crate::texture_source::decode_canvas_still(path, bytes, srgb.then_some(v2::ColorSpace::Srgb))
            .map_err(|error| crate::texture_source::decode_context(error, "image decode", path))?;
        let (width, height) = (decoded.width(), decoded.height());
        moda_assets::decode::prepare_rgba(decoded.into_rgba8().into_raw(), width, height, !natural, false)
            .with_context(|| format!("prepare image {}", path.display()))
    })?;
    // Animated paints can discover different images on concurrent frame workers.
    // Publish additions together: separate read/replace operations could lose another
    // worker's newly decoded image and leave its draw with a loading placeholder.
    // Decoding stays outside this lock; installed pixels are immutable and shared.
    static PUBLISH: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _publish = PUBLISH.lock().expect("canvas image publication");
    let mut merged = moda_assets::engine_seam::installed().map(|installed| (*installed).clone()).unwrap_or_default();
    merged.extend(store);
    moda_assets::engine_seam::install_store(std::sync::Arc::new(merged));
    Ok(())
}

/// Moda's image store as it stands: the process's, keyed by path.
pub(crate) type Images = Option<std::sync::Arc<moda_assets::AssetStore>>;

/// Empty Moda's image store, so the next `install_images` reads every file again, and hand back what it held.
/// A kept-alive process calls this before it builds a document whose files may have changed since they were read.
pub(crate) fn forget_images() -> Images {
    let held = moda_assets::engine_seam::installed();
    if held.is_some() {
        moda_assets::engine_seam::install_store(std::sync::Arc::new(Default::default()));
    }
    held
}

/// Restore the process's image store to the snapshot taken before a temporary installation.
pub(crate) fn restore_images(held: Images) {
    match held {
        Some(store) => moda_assets::engine_seam::install_store(store),
        None => moda_assets::engine_seam::clear_store(),
    }
}

/// A frame retains the host text builder for deferred mask captures as well as its initial walk.
pub(crate) struct Plan {
    translation: Option<moda_engine::frame::GroupTranslation>,
    picture_text: std::rc::Weak<CanvasTextFn>,
    resources: [u64; 2],
    coverage: std::sync::Arc<moda_engine::seams::CoveragePlan>,
    /// How this plan came from the last one of its scene: the state that one was (`revision`), and what was
    /// done to it. `None` for a scene's first plan, and for the plan as it already stood.
    change: Option<(u64, Change)>,
    /// Which state of its scene's plan this is (`Retained::plan_revision`).
    revision: u64,
    packet: bool,
    record_budget: moda_engine::frame::RecordBudget,
    addresses: Option<std::sync::Arc<moda_engine::residency::ItemAddresses>>,
    options: moda_engine::RenderOptions,
    text: moda_engine::text_draw::TextBuilder,
    /// Which installed scene this plan is of, and that scene's clock when it was planned: a picture drawn of the
    /// plan is the picture of that scene until its clock moves.
    scene: (u64, u64),
}
/// The executable part of a serial plan. Geometry, laid-out text and mask rows
/// stay in the plan that produced the serial frame. Workers never need a text
/// builder or a retained scene to execute it.
#[derive(Clone)]
pub(super) struct SharedPlan {
    pub(super) coverage: std::sync::Arc<moda_engine::seams::CoveragePlan>,
    key: moda_engine::frame::PlanKey,
    record_budget: Option<usize>,
    scene: (u64, u64),
    resources: [u64; 2],
}

impl Plan {
    pub(super) fn shared(&self) -> Option<SharedPlan> {
        if self.addresses.is_some() || self.translation.is_some() { return None; }
        Some(SharedPlan { coverage: self.coverage.clone(), record_budget: self.packet.then_some(self.record_budget.available), scene: self.scene, resources: self.resources, key: moda_engine::frame::PlanKey {
            owner: self.scene.0, revision: self.revision,
        } })
    }
}

/// What planning did to a scene's plan (`moda_engine::frame::ChangedItems`).
enum Change {
    /// It wrote these items of the plan where they stand.
    Written(moda_engine::residency::PlanWritten),
    /// It made the plan again whole; this is the plan before.
    Whole(std::sync::Arc<moda_engine::seams::CoveragePlan>),
}
impl std::ops::Deref for Plan {
    type Target = moda_engine::seams::CoveragePlan;
    fn deref(&self) -> &Self::Target { &self.coverage }
}

/// Draw text nodes whose batches are built here, through the entry every text node takes after shaping:
/// the plan's text builder hands Moda a `TextQuadBatch`, and Moda draws it. `vork setup` runs it so the text
/// pipelines are in the driver cache without a font: a pipeline's key does not depend on a glyph's shape or a
/// raster's pixels, only on which pass draws it. The node draws each lane vork's canvas text takes: the
/// vector lane (a glyph run that is a square outline, in the analytic glyph pass, and a triangle, as
/// decorations are drawn), and the raster lane of text the vector lane declines (right-to-left and complex
/// scripts, drawn as one textured quad; an emoji draws through the same pipeline). Two nodes draw them: a plain
/// one, on the page's four samples, and a blurred one, which Moda draws into a capture of one sample; a text
/// pipeline is created per sample count (a shadowed node's capture has the page's four). `float`: on a float
/// canvas's Rgba16Float picture instead of an 8-bit one.
pub(crate) fn warm_text(renderer: &crate::Renderer, float: bool) -> Result<()> {
    use moda_text::vector::ops::{VectorAnalyticRecord, VectorNodeOps, VectorOp, VectorOpPipeline};
    let node = |filter| Node { parent: None, clip_path: None, mask: None, mask_type: Default::default(), blend: Default::default(), isolation: Default::default(), filter, kind: Kind::Text { size: [48., 48.] } };
    let (plain, blurred) = (node(Filter::default()), node(Filter(vec![FilterOp::Blur(1.)])));
    let values = Values { style_revision: 0, transform: v2::Transform::default(), visible: true, opacity: 1., path: None, texts: 2,
        text_composition: Vec::new(), text_revision: 0, text_paint_revisions: Default::default(), world: None };
    let page = Page { size: [64, 64], scale: [1., 1.], offset: [0., 0.], background: None, working: None };
    let mut retained = Retained::new(&page, &[&plain, &blurred], &[values.clone(), values])?;
    let op = |pipeline, tris: Vec<f32>, records: Vec<VectorAnalyticRecord>| VectorOp {
        pipeline, color: [1.; 4], opacity: 1., blur_px: 0., stroke_px: 0., additive: false, tris: tris.into(), records: records.into(), gradient: None,
        paint: None,
    };
    // One em square as four quadratic segments (em units, y down, pen origin), and a glyph record that places
    // it at 32 px; then a triangle, as an underline's rectangle is drawn.
    let square = vec![0., 0., 0.5, 0., 1., 0.,  1., 0., 1., -0.5, 1., -1.,  1., -1., 0.5, -1., 0., -1.,  0., -1., 0., -0.5, 0., 0.];
    let glyph = VectorAnalyticRecord { pen_x: 8., baseline_y: 40., font_size_px: 32., em_box: [-0.1, -1.1, 1.1, 0.1], first_curve: 0, curve_count: 4,
        glyph_lin: [1., 0., 0., 1.], glyph_off: [0., 0.], glyph_opacity: 1., local_aa_px: 0. };
    let batch = |vector, degraded| std::sync::Arc::new(moda_text_draw::TextQuadBatch {
        emoji_engine: None,
        node_id: "setup-text".into(), z: 0., model_matrix: glam::Mat4::IDENTITY.to_cols_array(), page_clip_bounds: [0., 0., 64., 64.],
        ops: Vec::new(), degraded, vector, distort: None, content_key: None, paint_geometry_key: None,
    });
    // The raster lane's picture: a few opaque texels.
    let raster = moda_text::degraded::DegradedRaster { rgba: vec![255; 8 * 8 * 4], width: 8, height: 8, logical_w: 8., logical_h: 8.,
        quality_scale: 1., pad_top: 0., pad_bottom: 0., fill_alpha: 1. };
    let batches = [
        batch(Some(VectorNodeOps {
            uniform_padding: None, baked_padding_scale: None,
            ops: vec![op(VectorOpPipeline::Analytic, Vec::new(), vec![glyph]), op(VectorOpPipeline::Tess, vec![8., 44., 40., 44., 40., 46.], Vec::new())],
            curve_points: std::sync::Arc::new(square), gradient: None, emojis: Vec::new(),
        }), None),
        batch(None, Some(moda_text_draw::seams::DegradedQuad { raster: std::sync::Arc::new(raster), offset: [0., 0.], opacity: 1. })),
    ];
    let text: CanvasTextBuilder = std::rc::Rc::new(move |_, index, model, z, clip, _, _, _| {
        let mut placed = batches[index.min(batches.len() - 1)].clone();
        let batch = std::sync::Arc::make_mut(&mut placed);
        batch.model_matrix = model;
        batch.z = z;
        batch.page_clip_bounds = clip;
        Ok(Some(placed))
    });
    let plan = plan(&mut retained, &page, text)?;
    Engine::new(renderer, float, v2::WorkingColorSpace::Srgb)?.render_to_texture(&plan)?;
    Ok(())
}

/// Moda's frame walk over one document: the plan and its warnings. Text nodes are drawn by `text`, given the
/// vork item index and batch of a richtext node, its model matrix, z, clip, render scale and effective opacity.
pub(crate) type CanvasTextBuilder = std::rc::Rc<CanvasTextFn>;
type CanvasTextFn = dyn Fn(usize, usize, [f32; 16], f64, [f32; 4], f64, f64, Option<&moda_text_draw::TextQuadBatch>) -> Result<Option<std::sync::Arc<moda_text_draw::TextQuadBatch>>>;
pub(crate) fn plan(
    retained: &mut Retained,
    page: &Page,
    text: CanvasTextBuilder,
) -> Result<Plan> {

    let mut result = plan_for_draw(retained, page, text, PlanUse::Snapshot)?;
    result.materialize(retained)?;
    Ok(result)
}

/// A serial packet can bind its parent's translation at submission. Readers of a
/// snapshot (queries and independent GPU workers) need ordinary current matrices.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PlanUse { Packet, Snapshot }

pub(super) fn plan_for_draw(retained: &mut Retained, page: &Page, text: CanvasTextBuilder, usage: PlanUse) -> Result<Plan> {
    // Restoring trimmed spans can tessellate and shape text. Retire optional
    // lanes before growing the serial working set to restore those outputs.
    if retained.plan_index.needs_materialization() { crate::canvas_host_memory::require_serial_growth()?; }
    let resources = picture_resources();
    let picture_text = std::rc::Rc::downgrade(&text);
    let paint_text = text.clone();
    let identity = retained.id;
    let geometry_key = move |node: &moda_scene::types::RawNode| {
        let at = node.fields().get_known(K::VorkText)?.as_array()?;
        Some([identity, at.first()?.as_u64()?, at.get(1)?.as_u64()?, at.get(2)?.as_u64()?])
    };
    let builder: moda_engine::text_draw::TextBuilder = std::rc::Rc::new(move |node, model, z, clip, scale| {
        let at = |k: usize| node.fields().get_known(K::VorkText).and_then(|v| v.get(k)).and_then(Value::as_u64).map(|v| v as usize);
        let (Some(item), Some(sub)) = (at(0), at(1)) else { bail!("a richtext node vork did not write") };
        let opacity = node.f64_known(K::Opacity).unwrap_or(1.);
        Ok(match text(item, sub, model, z, clip, scale, opacity, None)? {
            Some(mut batch) => {
                // Planner coverage belongs to this lowered occurrence, not the
                // authored text name shared by its paint and mask batches.
                std::sync::Arc::make_mut(&mut batch).node_id = node.id().to_string();
                std::sync::Arc::make_mut(&mut batch).paint_geometry_key = geometry_key(node);
                moda_engine::text_draw::TextBuildOutcome::Item(Box::new(moda_engine::seams::DrawItem::Text(batch)))
            },
            None => moda_engine::text_draw::TextBuildOutcome::Empty,
        })
    });
    moda_engine::text_draw::install_text_builder(Some(builder.clone()));
    let opts = options(page);
    retained.caches.begin_frame();
    // The scene's world columns follow its writes (the subtrees of the rows written since the last frame): every
    // matrix read below, by row or by name, comes from them.
    {
        let _worlds_profile = moda_engine::prof::span("canvas.plan.worlds");
        retained.scene.refresh_worlds();
    }
    let generation = retained.generation();
    let frame_key = moda_engine::residency::page_frame_key(generation, 0, &opts);
    let patch_profile = moda_engine::prof::span("canvas.plan.patch");
    // Whether the scene's dirty list holds the complete change since the last plan: placement rows, and nothing
    // else written.
    let batch = retained.placement_batch().is_some_and(|rows| !rows.is_empty());
    // The state the plan stands at: what a change to it starts from.
    let from = retained.plan_revision;
    // Placement writes first follow their recipes, by row, reading no hash and no name. What the recipes decline
    // goes to the planner node by node, and after that to a whole plan; both read current hashes. Either writes
    // the plan the index holds, in place.
    let patchable = retained.planned_at.is_some_and(|at| {
        let stamps = retained.scene.stamps();
        stamps.latest(moda_scene::stamps::Group::Structure) <= at
            && stamps.all() <= at
            && [retained.moved, retained.painted, retained.content].into_iter()
                .any(|list| !stamps.dirty(list).is_empty())
    });
    let painter: moda_engine::text_draw::TextPainter = std::rc::Rc::new(move |node, old, opacity| {
        if old.paint_geometry_key != geometry_key(node) { return None; }
        let at = node.fields().get_known(K::VorkText)?.as_array()?;
        paint_text(at.first()?.as_u64()? as usize, at.get(1)?.as_u64()? as usize,
            old.model_matrix, old.z, old.page_clip_bounds, 1., opacity, Some(old)).ok().flatten()
    });
    moda_engine::text_draw::install_text_painter(Some(painter));
    let moved = retained.scene.stamps().dirty(retained.moved);
    let mut combined;
    let rows = if patchable && !batch {
        combined = moved.to_vec();
        combined.extend_from_slice(retained.scene.stamps().dirty(retained.painted));
        combined.extend_from_slice(retained.scene.stamps().dirty(retained.content));
        combined.sort_unstable();
        combined.dedup();
        &combined[..]
    } else if patchable { moved } else { &[] };
    if usage == PlanUse::Packet && batch && let [row] = rows && let Some(revision) = from
        && let Some((coverage, translation)) = retained.plan_index.translated_group(&retained.scene, &opts, *row) {
        moda_engine::text_draw::install_text_builder(None);
        moda_engine::text_draw::install_text_painter(None);
        return Ok(Plan { addresses: retained.plan_index.addresses(&coverage), options: opts,
            record_budget: retained.plan_index.record_budget(), picture_text, resources, coverage,
            translation: Some(translation), change: None, revision, packet: true, text: builder,
            scene: (retained.id, generation) });
    }
    let patched = retained.plan_index.update(&retained.scene, &retained.hashes, &opts, generation,
        frame_key, rows, batch, &mut retained.caches);
    drop(patch_profile);
    // The scene's clock moves on every write and on nothing else, so the frame key made of it names the scene
    // as it stands: it is the plan's whole identity where it is held. No display list is hashed for it.
    let planned = if let Some((plan, warnings)) = patched {
        let written = retained.plan_index.take_written();
        Ok((plan, warnings, from.map(|from| (from, Change::Written(written))), None))
    } else {
        let planning_profile = moda_engine::prof::span("canvas.plan.coverage");
        // Nothing written since this plan was made (when it was not held, it is made again).
        let planned = match retained.plan_index.held(frame_key, opts.shader_time_s) {
            // The plan as it stands: the same state.
            Some((plan, warnings)) => Ok((plan, warnings, None, from)),
            None if from.is_some() && crate::canvas_host_memory::require_serial_growth().is_err() => {
                Err(crate::canvas_host_memory::SerialGrowth.into())
            }
            None => {
                collect_page(retained, &opts, frame_key).map(|(plan, warnings, before)|
                    (plan, warnings, from.zip(before).map(|(from, before)| (from, Change::Whole(before))), None))
            }
        };
        drop(planning_profile);
        planned
    };
    moda_engine::text_draw::install_text_builder(None);
    moda_engine::text_draw::install_text_painter(None);
    let (plan, warnings, change, same) = planned?;
    retained.planned();
    // A plan made or written is the state after the last.
    let revision = same.unwrap_or_else(|| from.map_or(0, |from| from + 1));
    retained.plan_revision = Some(revision);
    // Nothing Moda declines is drawn as nothing without saying so. A path whose dashes leave nothing to draw at
    // this instant (a stroke drawn on from nothing) is empty geometry, not a decline.
    warnings.ensure_drawn()?;
    let packet = retained.plan_index.holds(&plan);
    let addresses = retained.plan_index.addresses(&plan);
    Ok(Plan { translation: None, addresses, options: opts, record_budget: retained.plan_index.record_budget(), picture_text, resources, coverage: plan, change, revision, packet, text: builder, scene: (retained.id, generation) })
}

fn collect_page(retained: &mut Retained, opts: &moda_engine::RenderOptions, key: u64)
    -> Result<(std::sync::Arc<moda_engine::seams::CoveragePlan>, std::sync::Arc<moda_engine::frame::PlanWarnings>, Option<std::sync::Arc<moda_engine::seams::CoveragePlan>>)> {
    retained.full_plans += 1;
    let (plan, warnings, aux) = moda_engine::frame::build_coverage_plan_indexed_with_dl(
        &retained.scene, 0, opts, Some((&mut retained.caches, &retained.hashes)), None)?;
    let (plan, warnings) = (std::sync::Arc::new(plan), std::sync::Arc::new(warnings));
    let before = retained.plan_index.seed(&retained.scene, opts, &plan, &warnings, aux,
        retained.generation(), key, &mut retained.caches);
    Ok((plan, warnings, before))
}

impl Plan {
    pub(super) fn materialize(&mut self, retained: &mut Retained) -> Result<()> {
        if self.addresses.is_none() && self.translation.is_none() { return Ok(()); }
        self.restore_or_collect(retained)
    }

    fn restore_or_collect(&mut self, retained: &mut Retained) -> Result<()> {
        moda_engine::text_draw::install_text_builder(Some(self.text.clone()));
        let full = if self.translation.take().is_some() { None } else {
            retained.plan_index.materialize(&self.coverage, &retained.scene, &retained.hashes,
                &self.options, &mut retained.caches)
        };
        let translated = full.is_some() && match &mut self.change {
            Some((_, Change::Written(written))) => {
                self.addresses.as_ref().is_some_and(|a| written.materialize(a).is_some())
            }
            _ => true,
        };
        if let Some(full) = full.filter(|_| translated) {
            moda_engine::text_draw::install_text_builder(None);
            self.coverage = full;
        } else {
            let key = moda_engine::residency::page_frame_key(retained.generation(), 0, &self.options);
            let fresh = collect_page(retained, &self.options, key);
            moda_engine::text_draw::install_text_builder(None);
            let (plan, warnings, _) = fresh?;
            warnings.ensure_drawn()?;
            self.revision = retained.plan_revision.unwrap_or(self.revision) + 1;
            retained.planned();
            self.scene = retained.picture_scene();
            self.resources = picture_resources();
            retained.plan_revision = Some(self.revision);
            self.packet = retained.plan_index.holds(&plan);
            self.record_budget = retained.plan_index.record_budget();
            self.coverage = plan;
            self.change = None;
        }
        self.addresses = None;
        Ok(())
    }

    fn packet_change(&self) -> Option<moda_engine::frame::PlanChange<'_>> {
        use moda_engine::frame::{ChangedItems, PlanChange};
        let (from, change) = self.change.as_ref()?;
        let items = match change {
            Change::Written(written) if written.recollect => return None,
            Change::Written(written) => ChangedItems::Written {
                evicted: &written.evicted, placed: &written.placed, replaced: &written.replaced, masks_changed: written.masks_changed,
            },
            Change::Whole(before) => ChangedItems::Whole(before),
        };
        Some(PlanChange { from: *from, items })
    }
}

fn options(page: &Page) -> moda_engine::RenderOptions {
    let opaque = page.background.is_some_and(|c| c[3] >= 1.);
    moda_engine::RenderOptions {
        scale: 1.,
        transparent: !opaque,
        preserve_page_fill: !opaque && page.background.is_some(),
        sample_count: 4,
        // The plan is for this canvas's engine: Moda plans a luminance mask by the encoding it blends in.
        working: page.working,
        ..Default::default()
    }
}

/// The encoding a canvas blends in, as Moda's engine and its planner take it: nothing for sRGB, the encoding
/// Moda's shaders are written in; otherwise the two conversions between sRGB, which every authored colour and
/// every decoded image is in, and the working encoding (`moda_engine::working_color`), with the pinned matrices
/// and the sRGB transfer (`WorkingColorSpace::conversion`).
pub(crate) fn working_color(working: v2::WorkingColorSpace) -> Option<moda_engine::working_color::WorkingColor> {
    use moda_engine::working_color::{Conversion, WorkingColor};
    let srgb = v2::WorkingColorSpace::Srgb;
    let conversion = |from: v2::WorkingColorSpace, to: v2::WorkingColorSpace| {
        let (decode, matrix, encode) = from.conversion(to).steps();
        Conversion { decode, matrix: matrix.map(|matrix| matrix.to_cols_array_2d()), encode }
    };
    (working != srgb).then(|| WorkingColor { from_srgb: conversion(srgb, working), to_srgb: conversion(working, srgb) })
}

/// Moda's engine on vork's device.
pub(crate) struct Engine {
    inner: moda_engine::Engine,
    cache_registration: crate::residency::Registration,
    reuse_picture: bool,
    publishing: std::cell::Cell<bool>,
    published: std::cell::RefCell<Option<SharedPlan>>,
    held: std::cell::RefCell<Option<HeldPicture>>,
    picture_hits: std::cell::Cell<u64>,
}
struct HeldPicture {
    /// The scene and clock the picture was drawn of (`Plan::scene`): it is that scene's picture until the clock
    /// moves or a resource generation does.
    scene: (u64, u64),
    source: PictureSource,
    resources: [u64; 2],
    texture: wgpu::Texture,
}
enum PictureSource {
    Serial(std::rc::Weak<CanvasTextFn>),
    Shared(moda_engine::frame::PlanKey),
}
fn picture_resources() -> [u64; 2] {
    [moda_assets::engine_seam::store_generation(),
        moda_engine::text_draw::text_host_generation()]
}
impl Engine {
    /// `float`: the canvas keeps a float picture (`canvas::Document::float`), so Moda draws into Rgba16Float;
    /// otherwise into Rgba8Unorm, as a browser's canvas does. `working`: the encoding the canvas blends in.
    pub fn new(renderer: &crate::Renderer, float: bool, working: v2::WorkingColorSpace) -> Result<Self> {
        let gpu = moda_engine::GpuContext {
            #[cfg(not(target_arch = "wasm32"))]
            pipeline_device: None,
            device: renderer.device.clone(),
            queue: renderer.queue.clone(),
            adapter_info: renderer.adapter_info.clone(),
            // The MSAA count Moda's 2D pipelines draw at (`manifest.pinned_constants.msaa_sample_count`), which
            // vork's devices take for Rgba8Unorm and Depth24Plus.
            max_sample_count: 4,
            max_sample_count_bgra: 4,
            max_sample_count_f16: 4,
            surface_copy_src: false,
        };
        let engine = moda_engine::Engine::new_with_color(gpu,
            if float { wgpu::TextureFormat::Rgba16Float } else { wgpu::TextureFormat::Rgba8Unorm }, working_color(working))?;
        Ok(Self::with_engine(renderer, engine))
    }

    pub(super) fn draw_resources(&self) -> moda_engine::frame::DrawResources { self.inner.draw_resources() }

    pub(super) fn with_engine(renderer: &crate::Renderer, mut engine: moda_engine::Engine) -> Self {
        // Like Moda's editor, retain GPU caches as well as the installed CPU scene.
        engine.set_retained(true);
        let device = renderer.device.clone();
        let ledger = renderer.residency.clone();
        let reclaim_device = renderer.device.clone();
        engine.set_scratch_memory_policy(
            move || crate::residency::camera_usage_headroom(&device)
                // Native allocator device/host block growth and command storage
                // need the same reserve as optional canvas replicas.
                .map(|(_, free)| free.saturating_sub(512 * 1024 * 1024)),
            move |bytes| {
                let target = bytes.saturating_add(512 * 1024 * 1024);
                while crate::residency::camera_usage_headroom(&reclaim_device).is_some_and(|(_, free)| free < target) {
                    if !ledger.reclaim() { break; }
                    // Releasing a slot can merely unpin its backing cache.
                    // Actual native headroom, not an owner's estimated weight,
                    // decides whether enough memory has been reclaimed.
                    let _ = reclaim_device.poll(wgpu::PollType::Poll);
                }
            });
        // Every frame of a clip is drawn here: keep geometry and record slots on the GPU between them.
        engine.set_offscreen_vertex_residency(true);
        let cache_registration = renderer.residency.register(engine.frame_cache_reclaimer());
        Self { inner: engine, cache_registration, reuse_picture: true, publishing: Default::default(), published: Default::default(), held: Default::default(), picture_hits: Default::default() }
    }
    pub(super) fn path_programs(&self, sources: &std::sync::Arc<[moda_engine::paint::PathProgramSource]>, words: &[u32], errors: Option<&wgpu::Buffer>) -> Result<()> {
        if self.inner.set_path_programs(sources, words, errors)? { self.held.borrow_mut().take(); }
        Ok(())
    }

    pub(super) fn serial_mask_captures(&self) -> u64 { self.inner.serial_mask_captures() }

    pub(super) fn publishing(&self) -> bool { self.publishing.get() }

    pub(super) fn publish_masks(&self, enabled: bool) -> moda_engine::frame::SharedCaptures {
        let previous = self.publishing.replace(enabled);
        if enabled && !previous {
            // A held serial picture predates publication. Draw it once to
            // provide its exact plan and any deferred mask captures.
            self.held.borrow_mut().take();
        } else if previous && !enabled {
            self.forget_published();
            self.inner.finish_shared_draws();
        }
        self.inner.publish_mask_captures(enabled)
    }

    pub(super) fn published(&self, scene: (u64, u64)) -> Option<SharedPlan> {
        self.published.borrow().as_ref().filter(|plan| plan.scene == scene && plan.resources == picture_resources()).cloned()
    }

    pub(super) fn forget_published(&self) {
        self.published.borrow_mut().take();
    }

    pub(super) fn publish(&self, plan: &Plan) -> Option<SharedPlan> {
        let shared = plan.shared();
        *self.published.borrow_mut() = shared.clone();
        shared
    }

    pub(super) fn draw_shared(&self, plan: &SharedPlan, captures: &moda_engine::frame::SharedCaptures) -> Result<wgpu::Texture> {
        let _draw = moda_engine::prof::span("canvas.draw");
        self.cache_registration.touch();
        let resources = picture_resources();
        if let Some(picture) = self.held.borrow().as_ref()
            && picture.scene == plan.scene && picture.resources == resources && resources == plan.resources
            && matches!(&picture.source, PictureSource::Shared(key) if *key == plan.key) {
            self.picture_hits.set(self.picture_hits.get().wrapping_add(1));
            return Ok(picture.texture.clone());
        }
        self.held.borrow_mut().take();
        self.inner.clear_path_errors();
        let result = self.inner.render_shared_plan_to_texture(&plan.coverage, plan.key, plan.record_budget, captures);
        self.inner.drop_deferred();
        if let Ok(texture) = &result && plan.resources == resources && resources == picture_resources() {
            self.held.replace(Some(HeldPicture {
                scene: plan.scene, source: PictureSource::Shared(plan.key), resources, texture: texture.clone(),
            }));
        }
        result
    }

    pub(super) fn publish_images(&self, images: &std::sync::Arc<moda_engine::paint::draw::SharedImages>) {
        self.inner.publish_images(images);
    }
    pub fn share_images(&mut self, images: std::sync::Arc<moda_engine::paint::draw::SharedImages>) {
        self.inner.share_images(images);
    }
    /// Embedded pictures can have deferred consumers, so only their resolve textures must stay distinct.
    pub fn new_picture(renderer: &crate::Renderer, float: bool, working: v2::WorkingColorSpace) -> Result<Self> {
        let mut engine = Self::new(renderer, float, working)?;
        engine.inner.set_reuse_frame_targets(false);
        engine.reuse_picture = false;
        Ok(engine)
    }
    pub fn profile_counters(&self) -> Value {
        let stats = self.inner.prepared_frame_stats();
        json!({"packet_hits": stats.hits, "packet_misses": stats.misses,
            "packet_rebuilds": stats.rebuilds, "bundle_replays": stats.bundle_replays,
            "geometry_upload_bytes": stats.upload_bytes, "text_upload_bytes": stats.text_bytes,
            "patched_items": stats.patched_items, "patch_bytes": stats.patch_bytes,
            "merged_bodies": stats.merged_bodies, "picture_hits": self.picture_hits.get()})
    }
    pub(super) fn picture(&self, scene: (u64, u64), text: &std::rc::Weak<CanvasTextFn>) -> Option<wgpu::Texture> {
        if !self.reuse_picture { return None; }
        let held = self.held.borrow();
        let picture = held.as_ref()?;
        if picture.scene != scene || !matches!(&picture.source, PictureSource::Serial(builder) if builder.ptr_eq(text)) || picture.resources != picture_resources() { return None; }
        self.cache_registration.touch();
        self.picture_hits.set(self.picture_hits.get().wrapping_add(1));
        Some(picture.texture.clone())
    }
    pub(crate) fn record_for_output(&self) -> moda_engine::scratch::OutputRecording<'_> {
        self.inner.record_for_output()
    }

    /// Draw a plan into Moda's reusable target: associated colour, Rgba8Unorm or (a float canvas) Rgba16Float,
    /// sampled and copied.
    /// Submit consumers before the next draw reuses it; Moda submits the drawing itself.
    pub(super) fn render_resident(&self, plan: &mut Plan, retained: &mut Retained) -> Result<wgpu::Texture> {
        let mut touched = match &plan.change {
            Some((_, Change::Written(w))) if plan.addresses.is_some() => w.placed.iter().map(|w| w.at)
                .chain(w.replaced.iter().map(|(i, _)| *i)).collect(),
            _ => Vec::new(),
        };
        let result = self.draw_revision(plan, retained);
        if self.publishing.get() && result.is_ok() { self.publish(plan); }
        let key = moda_engine::frame::PlanKey { owner: plan.scene.0, revision: plan.revision };
        let keep_recording = retained.plan_index.finish_draw(&retained.scene, &mut touched, self.inner.owned_packet_storage(key));
        self.inner.fit_owned_records(key, retained.plan_index.record_budget().available, keep_recording);
        result
    }
    fn draw_revision(&self, plan: &mut Plan, retained: &mut Retained) -> Result<wgpu::Texture> {
        if let Some(texture) = self.picture(plan.scene, &plan.picture_text) { return Ok(texture); }
        self.inner.clear_path_errors();
        // All owner revisions attempt the same packet write, including a fully resident plan.
        let key = moda_engine::frame::PlanKey { owner: plan.scene.0, revision: plan.revision };
        if plan.packet && plan.resources == picture_resources() {
            self.cache_registration.touch();
            if let Some(texture) = self.inner.try_translated_bundle(&plan.coverage, key, plan.packet_change().as_ref(), plan.addresses.as_deref(), plan.translation) {
                if self.reuse_picture {
                    *self.held.borrow_mut() = Some(HeldPicture { scene: plan.scene, source: PictureSource::Serial(plan.picture_text.clone()),
                        resources: plan.resources, texture: texture.clone() });
                }
                return Ok(texture);
            }
        }
        plan.materialize(retained)?;
        self.render_to_texture(plan)
    }
    pub fn render_to_texture(&self, plan: &Plan) -> Result<wgpu::Texture> {
        self.cache_registration.touch();
        let resources = picture_resources();
        if let Some(texture) = self.picture(plan.scene, &plan.picture_text) { return Ok(texture); }
        // A failed draw must not leave an older picture eligible for reuse.
        self.held.borrow_mut().take();
        moda_engine::text_draw::install_text_builder(Some(plan.text.clone()));
        let key = moda_engine::frame::PlanKey { owner: plan.scene.0, revision: plan.revision };
        let change = plan.packet_change();
        let result = self.inner.render_owned_plan_to_texture(
            &plan.coverage, key, change.as_ref(), plan.packet.then_some(plan.record_budget),
        );
        moda_engine::text_draw::install_text_builder(None);
    moda_engine::text_draw::install_text_painter(None);
        // Submission owns GPU resource lifetimes; exports need not keep old CPU
        // plans through the editor's idle-time destruction queue.
        self.inner.drop_deferred();
        if self.reuse_picture {
            if let Ok(texture) = &result {
                // Rendering may populate resource caches; only hold a picture whose
                // dependencies stayed unchanged throughout the draw.
                if plan.resources == resources && resources == picture_resources() {
                    *self.held.borrow_mut() = Some(HeldPicture {
                        scene: plan.scene, source: PictureSource::Serial(plan.picture_text.clone()), resources, texture: texture.clone(),
                    });
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn near(a: &Value, b: f64) -> bool { (a.as_f64().unwrap() - b).abs() < 1e-6 }

    #[test]
    fn write_document_records_only_painted_mask_items_across_copies() -> Result<()> {
        let scene: vork_scene::Scene = serde_json::from_value(serde_json::json!({
            "version": 2, "composition": {"space": "2d"},
            "nodes": [{"type": "path", "d": "M0 0H8V8Z", "fill": "none"}]
        }))?;
        let base = Path::new(".");
        let media = Media::new(&scene, base)?;
        let path = Arc::new(path_item(&scene.images, base, &scene.nodes[0], &media)?);
        let node = |kind, parent, mask| Node {
            kind, parent, mask, clip_path: None, mask_type: Default::default(),
            blend: Default::default(), isolation: Default::default(), filter: Default::default(),
        };
        let text = Kind::Text { size: [8., 8.] };
        let group = node(Kind::Group, None, None);
        let ink = node(text.clone(), Some(0), None);
        let switched = node(Kind::Path(path.clone()), Some(0), Some(3));
        let nested_matte = node(text.clone(), None, None);
        let consumer = node(text, None, Some(0));
        let clip = node(Kind::Path(path.clone()), None, None);
        let mut clipped = consumer.clone();
        clipped.clip_path = Some(6);
        let nodes = [&group, &ink, &switched, &nested_matte, &consumer, &clipped, &clip];
        let page = Page { size: [16, 16], scale: [1.; 2], offset: [0.; 2], background: None, working: None };
        let mut values = vec![Values {
            transform: Default::default(), visible: true, opacity: 1., path: None, texts: 1,
            text_composition: Vec::new(), style_revision: 0, text_revision: 0,
            text_paint_revisions: Default::default(), world: None,
        }; nodes.len()];

        // A sampled fill switches from no paint to white. Only then is its nested text matte emitted.
        for (fill, expected) in [(None, &[1][..]), (Some([1.; 4]), &[1, 2, 3][..])] {
            values[2].path = Some(PathValues { fill, ..path.values });
            let doc = write_document(&page, &nodes, &values)?;
            assert_eq!(doc.mask_paint_items, expected);
            let ink_copies = doc.bindings.iter().filter(|b| b.item == 1 && matches!(b.field, BoundField::Text { .. })).count();
            assert_eq!(ink_copies, 2, "both consumers draw the shared group's text");
        }
        Ok(())
    }

    #[test]
    fn visible_aliases_share_split_retire_and_reopen_the_held_movie() -> Result<()> {
        // Isolate the process-wide asset store from other renderer tests.
        const CHILD: &str = "VORK_TEST_VISIBLE_FOOTAGE";
        if std::env::var_os(CHILD).is_none() {
            let name = std::thread::current().name().unwrap().to_owned();
            ensure!(std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", &name]).env(CHILD, "1").status()?.success(), "visible footage test failed");
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("cuts.mp4");
        ensure!(std::process::Command::new("ffmpeg").args(["-v", "error", "-f", "lavfi", "-i",
            "testsrc2=s=64x48:r=10:d=1", "-c:v", "libx264"]).arg(&path).status()?.success(), "encode fixture");
        let mut images = serde_json::Map::new();
        let mut nodes = Vec::new();
        for i in 0..32 {
            let name = format!("clip{i:02}");
            images.insert(name.clone(), serde_json::json!({"file":"cuts.mp4"}));
            nodes.push(serde_json::json!({"name":name, "type":"path", "d":"M0 0H1V1Z",
                "fill":{"image":name}, "time":{"offset":if i == 1 { "1/10" } else { "0/10" }}}));
        }
        let scene: vork_scene::Scene = serde_json::from_value(serde_json::json!({
            "version":2, "composition":{"space":"2d"}, "images":images, "nodes":nodes
        }))?;
        let mut media = Media::new(&scene, dir.path())?;
        let mut plan = crate::animation::Plan::compile(&scene, dir.path())?;
        let mut nodes: Vec<_> = scene.nodes.iter().map(|node| crate::install::EvalNode {
            node: node.without_children(), parent: None, world: glam::Mat4::IDENTITY,
            dynamic: false, camera: None, embedded: None,
        }).collect();
        plan.mark_canvas_nodes(&mut nodes)?;
        let mut samples = plan.samples();
        let mut show = |media: &mut Media, tick: u32, visible: &[usize]| -> Result<()> {
            let request = vork_time::Request::Frame(tick, vork_time::Time::integer(10));
            plan.sample_request_into(tick as f32 / 10., &request, &mut samples)?;
            let shown: Vec<_> = (0..32).map(|i| visible.contains(&i)).collect();
            let keys: Vec<_> = visible.iter().map(|&i| media.images[i].src.clone()).collect();
            media.update(&plan, &samples, &request.exact()?, MediaPublication::Frame, |src| keys.iter().any(|k| k.as_str() == src))?;
            let store = moda_assets::engine_seam::installed().unwrap();
            for (i, image) in media.images.iter().enumerate() {
                assert_eq!(store.image(&image.src).is_some(), shown[i]);
            }
            Ok(())
        };
        let frame = |media: &Media, i: usize| {
            moda_assets::engine_seam::installed().unwrap().image(&media.images[i].src)
                .unwrap().rgba_premul.clone()
        };
        show(&mut media, 0, &[0, 1, 2])?;
        assert_eq!(media.sources.len(), 1);
        assert_eq!(media.live.len(), 2);
        let original = frame(&media, 0);
        assert!(Arc::ptr_eq(&original, &frame(&media, 2)));
        let second = frame(&media, 1);
        assert_ne!(*original, *second);
        // Clip 0 now requests the picture formerly held by clip 1.
        show(&mut media, 1, &[0])?;
        assert!(Arc::ptr_eq(&second, &frame(&media, 0)));
        assert_eq!(media.live.len(), 1);
        show(&mut media, 2, &[])?;
        assert!(media.live.is_empty());
        assert_eq!(Arc::strong_count(&second), 1);
        std::fs::rename(&path, dir.path().join("held.mp4"))?;
        std::fs::write(&path, b"replacement is not a movie")?;
        for i in 2..32 {
            show(&mut media, 0, &[i])?;
            assert_eq!(media.live.len(), 1);
            assert_eq!(*frame(&media, i), *original);
        }

        show(&mut media, 5, &[0, 2])?;
        let held = frame(&media, 0);
        let reader = media.live[&(media.images[0].source, 5)].decoder.clone();
        let zero = vork_time::Request::Numeric(0.);
        plan.sample_request_into(0., &zero, &mut samples)?;
        let visible = [media.images[0].src.clone(), media.images[3].src.clone()];
        media.update(&plan, &samples, &zero.exact()?, MediaPublication::Missing, |src| visible.iter().any(|key| key.as_str() == src))?;
        assert!(Arc::ptr_eq(&held, &frame(&media, 0))); // visible at zero
        assert!(Arc::ptr_eq(&held, &frame(&media, 2))); // hidden at zero
        assert!(Rc::ptr_eq(&reader, &media.live[&(media.images[0].source, 5)].decoder));
        assert_eq!(*frame(&media, 3), *original); // newly published at zero
        Ok(())
    }

    #[test]
    fn hdr_override_on_a_pinned_alias_selects_a_new_publication() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("hdr.mkv");
        ensure!(std::process::Command::new("ffmpeg").args(["-v", "error", "-f", "lavfi", "-i",
            "color=s=64x48,setparams=color_trc=smpte2084", "-frames:v", "1", "-c:v", "ffv1"])
            .arg(&path).status()?.success(), "encode PQ fixture");
        let mut scene: vork_scene::Scene = serde_json::from_value(serde_json::json!({
            "version": 2, "composition": {"space": "2d"},
            "images": {"a": {"file": "hdr.mkv"}, "b": {"file": "hdr.mkv", "color_space": "sRGB"}},
            "nodes": [{"type": "path", "d": "M0 0H1V1Z", "fill": {"image": "a"}, "stroke": {"image": "b"}}]
        }))?;
        let opened = crate::footage::Opened::default();
        let old = Media::with_readers(&scene, dir.path(), None, &opened, None)?;
        assert_ne!(old.images[0].source, old.images[1].source);
        scene.images.get_mut("b").unwrap().color_space = None;
        std::fs::remove_file(&path)?;
        let files = BTreeSet::from([path.as_path()]);
        let edited = Media::with_readers(&scene, dir.path(), None, &opened.pins(), Some((&old, &files)))?;
        assert_eq!(edited.images[0].source, edited.images[1].source);
        assert_eq!(edited.images[0].src, old.images[0].src);
        assert_ne!(edited.images[1].src, old.images[1].src);
        Ok(())
    }

    #[cfg(any(vork_nvdecode, vork_vtdecode))]
    #[test]
    #[ignore = "requires native hardware footage decode"]
    fn unsupported_native_size_publishes_requested_portable_frame() -> Result<()> {
        let gpu = crate::test_renderer()?;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("small.mp4");
        ensure!(std::process::Command::new("ffmpeg").args(["-v", "error", "-f", "lavfi", "-i",
            "nullsrc=s=16x16:r=10:d=1,geq=lum='16+20*N':cb=128:cr=128",
            "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p"]).arg(&path).status()?.success(), "encode small movie");
        let scene: vork_scene::Scene = serde_json::from_value(serde_json::json!({
            "version": 2, "composition": {"space": "2d"}, "images": {"clip": {"file": "small.mp4"}},
            "nodes": [{"type": "path", "d": "M0 0H1V1Z", "fill": {"image": "clip"}}]
        }))?;
        let mut media = Media::new(&scene, dir.path())?;
        media.prepare_gpu(&gpu)?;
        let plan = crate::animation::Plan::compile(&scene, dir.path())?;
        let mut samples = plan.samples();
        let crate::footage::Source::Video(video) = crate::footage::Source::open(&path)? else { unreachable!() };
        let mut reference = crate::footage_canvas::CanvasDecoder::default();
        let uninterrupted = (0..10).map(|i| reference.frame(&video, i, false, 16)).collect::<Result<Vec<_>>>()?;
        for index in [7, 2] {
            let request = vork_time::Request::Frame(index, vork_time::Time::integer(10));
            plan.sample_request_into(index as f32 / 10., &request, &mut samples)?;
            media.update(&plan, &samples, &request.exact()?, MediaPublication::Frame, |_| true)?;
            let store = moda_assets::engine_seam::installed().context("published movie frame")?;
            let actual = store.image(&media.images[0].src).context("portable frame in the asset store")?;
            assert_eq!(actual.rgba_premul, uninterrupted[index as usize].rgba_premul);
        }
        Ok(())
    }

    #[test]
    fn page_resize_and_layout_change_recollect_rendered_mask_membership() -> Result<()> {
        let page = Page { size: [64, 48], scale: [1.; 2], offset: [0.; 2], background: None, working: None };
        let text: CanvasTextBuilder = Rc::new(|_, _, _, _, _, _, _, _| Ok(None));
        let mut retained = Retained::new(&page, &[], &[])?;
        let mut scene = moda_scene::Scene::load(&json!({
            "layoutMode":"freeform",
            "pages":[{"id":"page","width":64,"height":48,"freeformX":20,"freeformY":10,"nodeIds":["clip"]}],
            "nodes":[
                {"id":"clip","type":"group","clipMaskNodeId":"source","children":["source","body"]},
                {"id":"source","type":"group","maskType":"luminance","children":["ink"]},
                {"id":"ink","type":"rectangle","width":20,"height":15,"fills":[{"paint":{"type":"solid","color":"#ffffff"}}]},
                {"id":"body","type":"rectangle","width":40,"height":30,"fills":[{"paint":{"type":"solid","color":"#ffffff"}}]}
            ]
        }))?;
        scene.continue_stamps_from(&retained.scene);
        retained.scene = scene;
        let draw = |r: &mut Retained| -> Result<_> {
            let p = plan_for_draw(r, &page, text.clone(), PlanUse::Packet)?;
            let mask = &p.masks[0];
            let key = mask.image.as_ref().unwrap().rendered.as_ref().unwrap().cache_key.unwrap();
            Ok((key, mask.origin, p.page_size))
        };
        let initial = draw(&mut retained)?;
        assert_eq!(draw(&mut retained)?, initial);
        assert_eq!(retained.full_plans, 1);

        retained.scene.page_mut("page").unwrap().0.insert("width".into(), json!(80));
        retained.scene.reexpand_page_background("page");
        let resized = draw(&mut retained)?;
        assert_eq!(retained.full_plans, 2);
        assert_ne!(initial.0, resized.0);
        assert_eq!(resized.2[0], 80.);

        // Document layout is immutable in place. Installation continues the scene clock.
        let mut doc = retained.scene.doc().clone();
        doc.layout_mode = Some("vertical".into());
        let mut replacement = moda_scene::Scene::from_document(doc)?;
        replacement.continue_stamps_from(&retained.scene);
        retained.scene = replacement;
        let laid_out = draw(&mut retained)?;
        assert_eq!(retained.full_plans, 3);
        assert_ne!(resized.0, laid_out.0);
        assert_ne!(resized.1, laid_out.1);
        Ok(())
    }

    #[test]
    fn losing_the_materialization_source_collects_and_seeds_the_current_scene() -> Result<()> {
        let mut page = Page { size: [64, 48], scale: [1.; 2], offset: [0.; 2], background: Some([1., 0., 0., 1.]), working: None };
        let text: CanvasTextBuilder = std::rc::Rc::new(|_, _, _, _, _, _, _, _| Ok(None));
        let mut retained = Retained::new(&page, &[], &[])?;
        let mut planned = plan_for_draw(&mut retained, &page, text.clone(), PlanUse::Packet)?;
        let old = format!("{:?}", planned.coverage);
        page.background = Some([0., 1., 0., 1.]);
        retained.update(&page, &[], &[])?;
        retained.plan_index = Default::default();
        planned.restore_or_collect(&mut retained)?;
        let mut fresh = Retained::new(&page, &[], &[])?;
        let expected = plan(&mut fresh, &page, text)?;
        assert_ne!(old, format!("{:?}", planned.coverage));
        assert_eq!(format!("{:?}", planned.coverage), format!("{:?}", expected.coverage));
        assert!(planned.change.is_none() && retained.plan_index.holds(&planned.coverage));
        Ok(())
    }

    #[test]
    fn mixed_path_length_and_dash_match_svg() -> Result<()> {
        use moda_engine::tess::{PathClosure, tessellate_combined_with_dash_trim};
        use moda_engine::tess::moda_engine_path_types::{FillRule, LineCap, LineJoin, StrokeAlign};

        let scene: v2::Scene = serde_json::from_value(serde_json::json!({
            "version": 2, "composition": {"space": "2d"},
            "nodes": [
                {"name": "out-back", "type": "path", "d": "M0 0H10 M20 0H30Z"},
                {"name": "mixed", "type": "path", "d": "M10 10H20 M30 20H36V28Z",
                    "fill": "none", "stroke": "black", "stroke-width": 2,
                    "stroke-dasharray": [2, 3], "stroke-dashoffset": 1},
                {"name": "calibrated", "type": "path", "d": "M10 10H20 M30 20H36V28Z",
                    "fill": "none", "stroke": "black", "stroke-width": 2,
                    "stroke-dasharray": [2, 3], "stroke-dashoffset": 1, "pathLength": 17},
                {"name": "turn", "type": "path", "d": "M85 55 q100 0 0 1 h-80",
                    "fill": "none", "stroke": "white", "stroke-width": 2,
                    "stroke-dasharray": [4, 3], "pathLength": 90}
            ]
        }))?;
        let base = Path::new(".");
        let media = Media::new(&scene, base)?;
        assert_eq!(path_item(&scene.images, base, &scene.nodes[0], &media)?.measured_length(), 30.);
        let mixed = "M10 10H20 M30 20H36V28Z";
        let samples = [
            [10.5, 10.], [15.5, 10.], [18.5, 10.], [30.5, 20.], [34.5, 20.],
            [36., 23.5], [36., 25.5], [33.3, 24.4], [31.5, 22.],
        ];
        // Chromium getTotalLength() and isPointInStroke() on independently authored SVG.
        let expected_strokes = [
            (mixed, 34., samples, [true, true, false, true, true, true, false, false, false]),
            (mixed, 34., samples, [true, false, true, true, false, true, true, true, true]),
            ("M85 55 q100 0 0 1 h-80", 180.017578125,
                [0, 4, 8, 14, 22, 27, 35, 43, 51].map(|i| [84.5 - i as f32, 56.]),
                [true, true, false, true, false, true, false, true, false]),
        ];
        for (node, (d, length, samples, expected)) in scene.nodes[1..].iter().zip(expected_strokes) {
            let path = path_item(&scene.images, base, node, &media)?;
            assert!((path.measured_length() - length).abs() < 0.01);
            let (cmds, _) = vork_std::moda::path_commands(d)?;
            let layer = Writer::stroke_layer(&path, &path.values, Value::Null);
            let dash: Vec<_> = layer["dash"].as_array().unwrap().iter()
                .map(|v| v.as_f64().unwrap()).collect();
            let mesh = tessellate_combined_with_dash_trim(
                &cmds, 2., PathClosure::from_field(&path.closed), None, Some(&dash),
                layer["dashOffset"].as_f64().unwrap(), LineCap::Butt, LineJoin::Miter,
                FillRule::NonZero, StrokeAlign::Center,
            );
            let covered = samples.map(|p: [f32; 2]| {
                mesh.positions.chunks_exact(6).zip(mesh.vertex_types.chunks_exact(3))
                    .filter(|(_, kinds)| kinds[0] == 1.)
                    .any(|(t, _)| {
                        let vertices = [[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]];
                        let [a, b, c] = vertices;
                        let cross = |a: [f32; 2], b: [f32; 2], p: [f32; 2]| {
                            (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
                        };
                        let sides = [cross(a, b, p), cross(b, c, p), cross(c, a, p)];
                        cross(a, b, c) != 0. &&
                            (sides.iter().all(|&d| d >= 0.) || sides.iter().all(|&d| d <= 0.))
                    })
            });
            assert_eq!(covered, expected, "{}", node.name);
        }
        Ok(())
    }

    #[test]
    fn css_gradients_place_moda_gradients_in_the_box() {
        // `to right` over a 200x100 box: the gradient line runs along the middle, edge to edge.
        let p = gradient_paint("linear-gradient(to right, red, blue)", [10., 20., 210., 120.]).unwrap();
        assert_eq!(p["type"], "linear");
        assert!(near(&p["start"]["x"], 0.) && near(&p["start"]["y"], 50.) && near(&p["end"]["x"], 100.) && near(&p["end"]["y"], 50.));
        assert_eq!(p["stops"][0]["color"], "#ff0000ff");
        assert!(near(&p["stops"][1]["offset"], 1.));
        // 45deg in a square: corner to corner.
        let p = gradient_paint("linear-gradient(45deg, red, blue)", [0., 0., 100., 100.]).unwrap();
        assert!(near(&p["start"]["x"], 0.) && near(&p["start"]["y"], 100.) && near(&p["end"]["x"], 100.) && near(&p["end"]["y"], 0.));
        // Stops without positions spread evenly; a repeated position is a hard stop.
        let p = gradient_paint("repeating-linear-gradient(red, lime, blue 50%, white 50%)", [0., 0., 10., 10.]).unwrap();
        assert!(near(&p["stops"][1]["offset"], 0.25) && near(&p["stops"][3]["offset"], 0.5));
        assert_eq!(p["repeat"], true);
        // An ellipse to the farthest corner, centred: radii √2 × half the box.
        let p = gradient_paint("radial-gradient(red, blue)", [0., 0., 200., 100.]).unwrap();
        assert!(near(&p["radiusX"], 50. * std::f64::consts::SQRT_2) && near(&p["radiusY"], 50. * std::f64::consts::SQRT_2));
        let p = gradient_paint("radial-gradient(circle 20px at left top, red, blue)", [0., 0., 200., 100.]).unwrap();
        assert!(near(&p["center"]["x"], 0.) && near(&p["radiusX"], 10.) && near(&p["radiusY"], 20.));
        let p = gradient_paint("conic-gradient(from 90deg at 25% 75%, red, blue)", [0., 0., 100., 100.]).unwrap();
        assert!(near(&p["angle"], 90.) && near(&p["center"]["x"], 25.) && near(&p["center"]["y"], 75.));
        let p = gradient_paint("linear-gradient(to bottom in oklab, red, blue)", [0., 0., 1., 1.]).unwrap();
        assert_eq!(p["interpolation"], "oklab");
        for (text, line) in [
            ("linear-gradient(red)", "at least two colour stops"),
            ("linear-gradient(red, 30%, blue)", "colour hint"),
            ("linear-gradient(in hsl, red, blue)", "interpolation in hsl is not built"),
            ("elliptical-gradient(red, blue)", "elliptical-gradient() is not built"),
        ] {
            let error = format!("{:#}", gradient_paint(text, [0., 0., 1., 1.]).unwrap_err());
            assert!(error.contains(line), "{error}");
        }
    }

    #[test]
    fn text_batches_keep_lowered_occurrence_identity_across_retained_plans() -> Result<()> {
        let node = Node { parent: None, clip_path: None, mask: None, mask_type: Default::default(), blend: Default::default(), isolation: Default::default(), filter: Filter::default(),
            kind: Kind::Text { size: [80., 80.] } };
        let mut values = Values { style_revision: 0, transform: v2::Transform::default(), visible: true,
            opacity: 1., path: None, texts: 2, text_composition: Vec::new(), text_revision: 0, text_paint_revisions: Default::default(), world: None };
        let page = Page { size: [256, 128], scale: [1., 1.], offset: [0., 0.], background: None, working: None };
        let mut retained = Retained::new(&page, &[&node], &[values.clone()])?;
        // A host can serve shared shaped glyphs for multiple lowered batches.
        // Planner identities must be distinct without mutating that host cache.
        let shared = std::sync::Arc::new(moda_text_draw::TextQuadBatch {
            emoji_engine: None,
            node_id: "authored-text".into(), z: 0., model_matrix: glam::Mat4::IDENTITY.to_cols_array(),
            page_clip_bounds: [0., 0., 256., 128.], ops: Vec::new(),
            degraded: None, vector: Some(moda_text::vector::VectorNodeOps {
                uniform_padding: Some((16., 1.)), ..Default::default()
            }), distort: None, content_key: None, paint_geometry_key: None,
        });
        let source = shared.clone();
        let builder: CanvasTextBuilder = std::rc::Rc::new(move |_, _, model, z, clip, _, _, _| {
            let mut batch = source.clone();
            let placed = std::sync::Arc::make_mut(&mut batch);
            placed.model_matrix = model;
            placed.z = z;
            placed.page_clip_bounds = clip;
            Ok(Some(batch))
        });
        let mut previous = None;
        for _ in 0..3 {
            let plan = plan(&mut retained, &page, builder.clone())?;
            if let Some(old) = &previous {
                assert!(std::sync::Arc::ptr_eq(old, &plan.coverage), "unchanged frames reuse their plan");
            }
            previous = Some(plan.coverage.clone());
            let ids: Vec<_> = plan.coverage.items.iter().filter_map(|item| match item {
                moda_engine::seams::DrawItem::Text(batch) => Some(batch.node_id.as_str()),
                _ => None,
            }).collect();
            assert_eq!(ids.len(), 2);
            assert_ne!(ids[0], ids[1]);
            for id in ids { assert!(retained.scene.effective_node(id).is_some(), "unresolved text owner {id}"); }
            assert_eq!(shared.node_id, "authored-text");
        }
        // The plan index is the plan's one holder: a plan it does not hold (one past the byte budget) is made again.
        let before_eviction = previous.take().unwrap();
        retained.plan_index = Default::default();
        let rebuilt = plan(&mut retained, &page, builder.clone())?;
        assert!(!std::sync::Arc::ptr_eq(&before_eviction, &rebuilt.coverage));
        assert_eq!(format!("{:?}", before_eviction.items), format!("{:?}", rebuilt.coverage.items));
        previous = Some(rebuilt.coverage);
        for step in 1..8 {
            values.transform.translation[0] = (4 - step) as f32;
            values.transform.scale[0] = 1. + step as f32 / 8.;
            let placements = retained.caches.incr_placement_nodes;
            if step >= 4 { values.text_revision += 1; }
            let hits = retained.caches.incr_patch_hits;
            retained.update(&page, &[&node], &[values.clone()])?;
            let changed = plan(&mut retained, &page, builder.clone())?;
            if step < 4 {
                assert_eq!(retained.caches.incr_placement_nodes - placements, 2);
            }
            assert_eq!(retained.caches.incr_patch_hits - hits, 1,
                "node-local placement and content edits reuse the indexed plan");
            assert!(!std::sync::Arc::ptr_eq(previous.as_ref().unwrap(), &changed.coverage));
            let mut fresh = Retained::new(&page, &[&node], &[values.clone()])?;
            let expected = plan(&mut fresh, &page, builder.clone())?;
            assert_eq!(format!("{:?}", changed.coverage.items), format!("{:?}", expected.coverage.items));
            previous = Some(changed.coverage.clone());
        }
        Ok(())
    }

    fn grouped_disks() -> Result<(Vec<Node>, Vec<Values>, Page)> {
        use std::sync::Arc;
        let path = Arc::new(PathItem {
            points: moda_points(&vork_std::moda::path_commands("M-2 0A2 2 0 1 0 2 0A2 2 0 1 0-2 0Z")?.0),
            closed: true.into(), bounds: None, length: 0.,
            fill: PaintSource::Color([1.,0.,0.,1.]), stroke: PaintSource::None,
            fill_rule: Default::default(), linecap: Default::default(), linejoin: Default::default(),
            miterlimit: 4., dasharray: Vec::new(), sampled_dash: None, path_length: None, stroke_first: false, paint_clock: 0,
            values: PathValues {fill:Some([1.,0.,0.,1.]),stroke:None,stroke_width:0.,stroke_dashoffset:0.,fill_opacity:1.,stroke_opacity:1.},
        });
        let mut nodes = vec![Node {parent:None,clip_path:None,mask:None,mask_type:Default::default(),blend:Default::default(),isolation:Default::default(),filter:Default::default(),kind:Kind::Group}];
        nodes.extend((0..40).map(|_| Node {parent:Some(0),clip_path:None,mask:None,mask_type:Default::default(),blend:Default::default(),isolation:Default::default(),filter:Default::default(),kind:Kind::Path(path.clone())}));
        let mut values = vec![Values {style_revision:0,transform:Default::default(),visible:true,opacity:1.,
            path:Some(path.values),texts:0,text_composition:Vec::new(),world:None,text_revision:0,text_paint_revisions:Default::default()};nodes.len()];
        values[0].path=None;
        for (i, value) in values.iter_mut().enumerate().skip(1) { value.transform.translation=[8. + i as f32 * 2.,40.,0.]; }
        let page=Page {size:[320,180],scale:[1.,1.],offset:[0.,0.],background:None,working:None};
        Ok((nodes, values, page))
    }

    #[test]
    fn translated_lowered_group_materializes_the_same_order_and_values_as_a_fresh_plan() -> Result<()> {
        let (nodes, mut values, page) = grouped_disks()?;
        let refs: Vec<_> = nodes.iter().collect();
        let builder: CanvasTextBuilder = std::rc::Rc::new(|_,_,_,_,_,_,_,_| Ok(None));
        let mut retained = Retained::new(&page, &refs, &values)?;
        plan(&mut retained, &page, builder.clone())?;
        for (row, x) in [(0, 0.21333334), (0, 7.45), (0, -0.65), (1, 13.25)] {
            values[row].transform.translation[0] = x;
            assert!(retained.write(&page, |i| refs[i], &values, &[(row, change::TRANSLATION)])?);
            let mut prepared = plan_for_draw(&mut retained, &page, builder.clone(), PlanUse::Packet)?;
            assert_eq!(prepared.translation.is_some(), row == 0);
            if row != 0 || x < 0. {
                prepared.materialize(&mut retained)?;
                let mut fresh = Retained::new(&page, &refs, &values)?;
                let expected = plan(&mut fresh, &page, builder.clone())?;
                assert_eq!(format!("{:?}", prepared.coverage), format!("{:?}", expected.coverage));
            }
        }
        Ok(())
    }

    #[test]
    fn parent_pan_resumes_after_a_child_step_and_publishes_current_worker_matrices() -> Result<()> {
        let (nodes, mut values, page) = grouped_disks()?;
        let refs: Vec<_> = nodes.iter().collect();
        let text: CanvasTextBuilder = Rc::new(|_, _, _, _, _, _, _, _| Ok(None));
        let mut retained = Retained::new(&page, &refs, &values)?;
        plan(&mut retained, &page, text.clone())?;

        // A nonintegral parent pose straddles an exponent boundary for this child.
        values[0].transform.translation[0] = 5.3333335;
        values[1].transform.translation[0] = 15.25;
        retained.write(&page, |i| refs[i], &values, &[(0, change::TRANSLATION), (1, change::TRANSLATION)])?;
        let changed = plan_for_draw(&mut retained, &page, text.clone(), PlanUse::Packet)?;
        assert!(changed.translation.is_none() && matches!(changed.change, Some((_, Change::Written(_)))));
        drop(changed);
        for x in [6.25, -1.75, 3.5] {
            values[0].transform.translation[0] = x;
            retained.write(&page, |i| refs[i], &values, &[(0, change::TRANSLATION)])?;
            let packet = plan_for_draw(&mut retained, &page, text.clone(), PlanUse::Packet)?;
            assert!(packet.translation.is_some(), "parent run at {x}");
        }

        let worker = crate::canvas_host_memory::with_workers(|| plan(&mut retained, &page, text.clone()))?;
        assert!(worker.shared().is_some());
        let mut fresh = Retained::new(&page, &refs, &values)?;
        let expected = plan(&mut fresh, &page, text)?;
        assert_eq!(format!("{:?}", worker.coverage), format!("{:?}", expected.coverage));
        Ok(())
    }

    #[test]
    fn lowered_group_opacity_writes_only_composition_outputs() -> Result<()> {
        let (nodes, mut values, page) = grouped_disks()?;
        values[0].opacity = 0.6;
        let refs=nodes.iter().collect::<Vec<_>>();
        let mut retained=Retained::new(&page,&refs,&values)?;
        let builder: CanvasTextBuilder=std::rc::Rc::new(|_,_,_,_,_,_,_,_| Ok(None));
        plan(&mut retained,&page,builder.clone())?;
        for opacity in [0.4,0.7,0.3,0.8] {
            values[0].opacity=opacity;
            retained.update(&page,&refs,&values)?;
            let changed=plan(&mut retained,&page,builder.clone())?;
            let Some((_,Change::Written(written)))=&changed.change else { panic!("opacity must patch its plan"); };
            assert_eq!(written.replaced.len(),2,"the lowered group must retain child draw records");
            assert!(written.placed.is_empty());
            let mut fresh=Retained::new(&page,&refs,&values)?;
            let expected=plan(&mut fresh,&page,builder.clone())?;
            assert_eq!(format!("{:?}",changed.coverage),format!("{:?}",expected.coverage));
        }
        Ok(())
    }

    #[test]
    fn qa_2d_shadow_offsets_follow_viewbox_and_node_transforms() {
        for (scale, parent_scale, angle, expected) in [
            ([2., 2.], [1., 1., 1.], 0., [16., 8.]),
            ([2., 3.], [1., 1., 1.], 0., [16., 12.]),
            ([2., 3.], [2., 4., 1.], 0., [32., 48.]),
            ([2., 3.], [1., 1., 1.], std::f32::consts::FRAC_PI_2, [-8., 24.]),
        ] {
            for (group, batches) in [(false, 1), (false, 3), (true, 1), (true, 3)] {
                let mut nodes = vec![
                    Node { parent: None, clip_path: None, mask: None, mask_type: Default::default(), blend: Default::default(), isolation: Default::default(), filter: Filter::default(), kind: Kind::Group },
                    Node { parent: Some(0), clip_path: None, mask: None, mask_type: Default::default(), blend: Default::default(), isolation: Default::default(),
                        filter: parse_filter("drop-shadow(8px 4px 0px red)").unwrap(), kind: Kind::Text { size: [8., 8.] } },
                ];
                if group {
                    nodes[1].kind = Kind::Group;
                    nodes.push(Node { parent: Some(1), clip_path: None, mask: None, mask_type: Default::default(), blend: Default::default(), isolation: Default::default(), filter: Filter::default(), kind: Kind::Text { size: [8., 8.] } });
                }
                let mut values = vec![Values { style_revision: 0, transform: v2::Transform::default(), visible: true, opacity: 1., path: None, texts: batches, text_composition: Vec::new(), text_revision: 0, text_paint_revisions: Default::default(), world: None }; nodes.len()];
                values[0].transform.scale = parent_scale;
                values[0].transform.rotation = Some(glam::Quat::from_rotation_z(angle).to_array());
                let page = Page { size: [256, 128], scale, offset: [23., 45.], background: None, working: None };
                let doc = document(&page, &nodes.iter().collect::<Vec<_>>(), &values).unwrap();
                let effect = doc["nodes"].as_array().unwrap().iter()
                    .flat_map(|node| node["effects"].as_array().into_iter().flatten())
                    .find(|effect| effect["type"] == "filter-shadow").unwrap();
                // A lone text batch, several batches and a group alike cast the shadow from a
                // captured picture, which receives its offset in canvas space.
                for (axis, expected) in ["x", "y"].into_iter().zip(expected) {
                    assert!((effect["offset"][axis].as_f64().unwrap() - expected).abs() < 1e-5, "{effect}");
                }
                assert_eq!(effect["blur"], 0.);
            }
        }
    }

    #[test]
    fn css_filters_are_one_drop_shadow_then_one_blur() {
        let f = parse_filter("drop-shadow(2px 4px 6px rgba(0, 0, 0, 0.5)) blur(3px)").unwrap();
        let FilterOp::Shadow([x, y], blur, color) = f.0[0] else { panic!() };
        assert_eq!((x, y, blur, f.0[1]), (2., 4., 6., FilterOp::Blur(3.)));
        assert_eq!(color, [0., 0., 0., 128. / 255.]);
        let glow = parse_filter("drop-shadow(#36f9c5 0 0 24px)").unwrap();
        assert!(matches!(glow.0[0], FilterOp::Shadow(_, 24., _)));
        let effects = effects(&parse_filter("drop-shadow(1px 1px 2px black) blur(5px)").unwrap(), glam::DMat2::IDENTITY);
        // Both lengths are standard deviations; Moda's shadow blur and layer-blur radius are twice theirs.
        assert_eq!(effects[0]["type"], "filter-shadow");
        assert_eq!(effects[0]["blur"], 4.);
        assert_eq!(effects[1]["radius"], 10.);
        assert!(parse_filter("none").unwrap().0.is_empty());
        for (text, line) in [
            ("blur(1e308px)", "blur() radius exceeds the numeric range"),
            ("drop-shadow(1px)", "an x and y offset"),
            ("drop-shadow(1px 1px -2px)", "standard deviation must not be negative"),
            ("drop-shadow(1px 1px 1e308px)", "drop-shadow() standard deviation exceeds the numeric range"),
            ("blur(1.2)", "write nonzero lengths in px"),
            ("drop-shadow(0 2 3 #00000070)", "write nonzero lengths in px"),
            ("drop-shadow(red -2 0)", "write nonzero lengths in px"),
            ("drop-shadow(0 0 3)", "write nonzero lengths in px"),
            ("drop-shadow(0 2em red)", "write nonzero lengths in px"),
        ] {
            let error = format!("{:#}", parse_filter(text).unwrap_err());
            assert!(error.contains(line), "{error}");
        }
    }
}

#[cfg(test)]
impl Retained {
    pub(super) fn text_cache_counts(&self) -> (u64, u64) {
        (self.caches.text_hits, self.caches.text_misses)
    }

    pub(super) fn assert_fresh(&self, page: &Page, nodes: &[&Node], values: &[Values]) {
        let (fresh, _) = write_document(page, nodes, values).unwrap().load().unwrap();
        assert_eq!(self.scene.node_order().len(), fresh.node_order().len());
        for id in fresh.node_order() {
            assert_eq!(self.scene.node(id).unwrap().to_json_map(), fresh.node(id).unwrap().to_json_map(), "node {id}");
        }
        for id in fresh.node_order() {
            assert_eq!(self.scene.accumulated_affine(id), fresh.accumulated_affine(id), "transform {id}");
        }
    }
}

#[cfg(test)]
mod retained_world_tests {
    use super::*;

    #[test]
    fn text_node_opacity_stays_outside_glyph_drawing() {
        let page = Page { size: [320, 180], scale: [1., 1.], offset: [0., 0.], background: None, working: None };
        for filter in [Filter::default(), parse_filter("drop-shadow(8px 4px 0px red)").unwrap()] {
            let node = Node { parent: None, clip_path: None, mask: None, mask_type: Default::default(), blend: Default::default(), isolation: Default::default(), filter,
                kind: Kind::Text { size: [160., 120.] } };
            let nodes = [&node];
            let mut values = vec![Values { style_revision: 0, transform: Default::default(), visible: true, opacity: 0.5,
                path: None, texts: 1, text_composition: Vec::new(), text_revision: 0, text_paint_revisions: Default::default(), world: None }];
            let mut retained = Retained::new(&page, &nodes, &values).unwrap();
            for opacity in [0.5, 1., 0.25, 0., 0.5] {
                values[0].opacity = opacity;
                values[0].transform.translation[0] += 10.;
                retained.update(&page, &nodes, &values).unwrap();
                retained.assert_fresh(&page, &nodes, &values);
                let doc = document(&page, &nodes, &values).unwrap();
                let records = doc["nodes"].as_array().unwrap();
                let outer = records.iter().find(|n| n["id"] == "n0").unwrap();
                assert_eq!(outer["type"], "group", "node opacity must compose the complete text picture");
                assert_eq!(outer["opacity"], serde_json::json!(opacity));
                let text = records.iter().find(|n| n["type"] == "richtext").unwrap();
                assert_eq!(text["opacity"].as_f64().unwrap_or(1.), 1., "glyphs must not inherit node opacity");
                if node.filter.has_shadow() {
                    assert!(text["effects"].is_null(), "a text shadow is cast from the captured picture, not by a glyph pass");
                    let core = records.iter().find(|n| n["id"] == "n0.core").unwrap();
                    assert_eq!(core["type"], "group");
                    assert_eq!(core["effects"][0]["type"], "filter-shadow");
                    assert_eq!(core["effects"][0]["offset"], serde_json::json!({"x": 8., "y": 4.}));
                }
            }
        }
    }

    /// A text's filter is a field of the group around it: a filter that changes between two shadows is written
    /// into that row, and the canvas is not lowered again.
    #[test]
    fn a_text_filter_that_keeps_its_form_is_written_into_its_group_row() {
        let page = Page { size: [320, 180], scale: [1., 1.], offset: [0., 0.], background: None, working: None };
        let text = |filter: &str| Node { parent: None, clip_path: None, mask: None, mask_type: Default::default(), blend: Default::default(), isolation: Default::default(), filter: parse_filter(filter).unwrap(),
            kind: Kind::Text { size: [160., 120.] } };
        let mut values = vec![Values { style_revision: 0, transform: Default::default(), visible: true, opacity: 1.,
            path: None, texts: 1, text_composition: Vec::new(), text_revision: 0, text_paint_revisions: Default::default(), world: None }];
        let first = text("drop-shadow(8px 4px 0px red)");
        let mut retained = Retained::new(&page, &[&first], &values).unwrap();
        for (step, filter) in ["drop-shadow(2px 9px 3px blue)", "drop-shadow(8px 4px 0px red)", "drop-shadow(-5px 1px 6px #00ff0080)"].into_iter().enumerate() {
            let node = text(filter);
            values[0].style_revision = step as u64 + 1;
            retained.update(&page, &[&node], &values).unwrap();
            retained.assert_fresh(&page, &[&node], &values);
        }
        assert_eq!(retained.installs, 0, "a filter of the same form is a field write");
    }

    #[test]
    fn vector_clip_style_edits_keep_geometry_and_translation_eligibility() {
        use std::sync::Arc;
        let path = Arc::new(PathItem {
            points: Points::Json(vec![json!({"type":"move","x":0,"y":0}), json!({"type":"line","x":20,"y":0}),
                json!({"type":"line","x":20,"y":20})]), closed: true.into(), bounds: Some([0.,0.,20.,20.]), length: 60.,
            fill: PaintSource::Color([1.,0.,0.,1.]), stroke: PaintSource::None,
            fill_rule: Default::default(), linecap: Default::default(), linejoin: Default::default(),
            miterlimit: 4., dasharray: Vec::new(), sampled_dash: None, path_length: None, stroke_first: false, paint_clock: 0,
            values: PathValues {fill:None,stroke:None,stroke_width:0.,stroke_dashoffset:0.,fill_opacity:1.,stroke_opacity:1.},
        });
        let source = Node {parent:None,clip_path:None,mask:None,mask_type:Default::default(),blend:Default::default(),isolation:Default::default(),filter:Default::default(),kind:Kind::Path(path.clone())};
        for compact in [false, true] {
            moda_scene::path_points::set_compact_on_load(compact);
            let mut nodes = vec![source.clone(),source.clone(),source.clone(),source.clone()];
            nodes[1].clip_path=Some(0); nodes[2].clip_path=Some(0);
            let mut values = vec![Values {style_revision:0,transform:Default::default(),visible:true,opacity:1.,
                path:Some(path.values),texts:0,text_composition:Vec::new(),world:None,text_revision:0,text_paint_revisions:Default::default()};4];
            let page=Page {size:[64,64],scale:[1.,1.],offset:[0.,0.],background:None,working:None};
            let mut retained=Retained::new(&page,&nodes.iter().collect::<Vec<_>>(),&values).unwrap();
            let copies:Vec<u32>=retained.bindings.list.iter().filter(|b| matches!(b.field,BoundField::Geometry))
                .map(|b| b.row).collect();
            assert_eq!(copies.len(),2);
            for step in 1..5 {
                values[0].style_revision=step;
                let mut changed=(*path).clone(); changed.fill=PaintSource::Color([0.,0.,step as f32/5.,1.]);
                nodes[0].kind=Kind::Path(Arc::new(changed));
                let generation=retained.generation();
                retained.update(&page,&nodes.iter().collect::<Vec<_>>(),&values).unwrap();
                assert_eq!(retained.generation(),generation,"ignored clip paint must not mutate the scene");
                retained.planned();
                values[3].transform.translation[0]=step as f32;
                retained.update(&page,&nodes.iter().collect::<Vec<_>>(),&values).unwrap();
                assert!(retained.placement_batch().is_some_and(|rows| !rows.is_empty()));
                retained.assert_fresh(&page,&nodes.iter().collect::<Vec<_>>(),&values);
            }
            let mut changed=(*path).clone();
            let Points::Json(points)=&mut changed.points else { unreachable!() };
            points[1]["x"]=json!(30); changed.closed=false.into();
            nodes[0].kind=Kind::Path(Arc::new(changed.clone())); values[0].style_revision+=1;
            retained.update(&page,&nodes.iter().collect::<Vec<_>>(),&values).unwrap();
            for row in copies {
                let node=retained.scene.row(row).unwrap();
                assert!(node.field_eq_json("points",&Value::Array(changed.points.json())));
                assert_eq!(node.bool("closed"),Some(false));
            }
            retained.assert_fresh(&page,&nodes.iter().collect::<Vec<_>>(),&values);
        }
        moda_scene::path_points::set_compact_on_load(true);
    }

    #[test]
    fn background_changes_keep_the_installed_page_current() {
        let mut page = Page { size: [64, 64], scale: [1., 1.], offset: [0., 0.], background: None, working: None };
        let mut retained = Retained::new(&page, &[], &[]).unwrap();
        for background in [Some([1., 0., 0., 0.5]), Some([0., 0., 1., 1.]), None] {
            page.background = background;
            retained.update(&page, &[], &[]).unwrap();
            retained.assert_fresh(&page, &[], &[]);
        }
    }

    #[test]
    fn world_picture_changes_keep_the_installed_scene_current() {
        let node = Node { parent: None, clip_path: None, mask: None, mask_type: Default::default(), blend: Default::default(), isolation: Default::default(), filter: Filter::default(),
            kind: Kind::World { bounds: [0., 0., 64., 64.] } };
        let nodes = [&node];
        let page = Page { size: [64, 64], scale: [1., 1.], offset: [0., 0.], background: None, working: None };
        let mut values = vec![Values { style_revision: 0, transform: Default::default(), visible: true, opacity: 1.,
            path: None, texts: 0, text_composition: Vec::new(), text_revision: 0, text_paint_revisions: Default::default(), world: None }];
        let mut retained = Retained::new(&page, &nodes, &values).unwrap();
        for (step, picture) in [Some(("a", [64, 64])), Some(("b", [64, 64])), Some(("b", [32, 32])), None,
            Some(("a", [64, 64]))].into_iter().enumerate() {
            values[0].world = picture.map(|(src, size)| (src.into(), size));
            values[0].transform.translation[0] = step as f32;
            values[0].opacity = 0.4;
            retained.update(&page, &nodes, &values).unwrap();
            retained.assert_fresh(&page, &nodes, &values);
        }
    }
}
