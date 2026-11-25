//! The 2D canvas (END-STATE-API §7): a scene with `composition: {space: "2d"}` is a canvas of
//! `render.resolution` pixels (or its `viewBox` placed in them), origin top-left, y down, whose group, text
//! and path nodes paint in tree order. Each frame is drawn by Moda's 2D engine (`canvas_draw`): the canvas's
//! nodes as one Moda document, planned and drawn with Moda's pipelines into a 4x MSAA target. Text is laid
//! out by Moda's text engine (crates `moda-text`, `moda-text-draw`): each text node becomes Moda
//! `TextNodeSpec`s (its styled runs spans of its HTML), its glyph, word and line tracks Moda's per-code-point
//! unit transforms, and its batches are drawn where Moda's frame walk places the text node. Text Moda's
//! vector lane declines (right-to-left and complex scripts) is drawn by its degraded tier, a cosmic-text
//! raster. The canvas blends in encoded sRGB, as CSS and SVG do, or in the encoding its composition names
//! (`composition.color_space`): its authored colours and the pictures it paints are sRGB code values, which
//! Moda's engine converts where they enter it, before it filters or blends them (`canvas_draw::working_color`).
//! Frames are delivered to the destinations a 3D shot writes.
//!
//! Node clocks (§8) are the 3D renderer's: `active`, `time` and a node's own `animation` are evaluated by the
//! same animation plan, so an absent node draws nothing, children included, and a node's tracks run on its
//! local time. A world (`composition: {space: "3d"}`) is drawn by the 3D renderer at its node's local time and
//! placed as one picture (`canvas_world`).
use crate::failure::{FailureKind, device, panicked, scene as scene_error, setup};
use crate::shot::{self, Output, RenderedShot};
use crate::Renderer;
use anyhow::{Context, Result, bail, ensure};
use glam::{DMat2, DVec2, Quat};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use vork_encode::{Backend, EncodeSettings, Encoder, Surface};
use vork_scene::v2::{self, NodeKind, TextUnit, TrackTarget, UnitProperty};
use vork_scene::Scene;
use crate::canvas_draw::{MediaPublication, PaintForm, PaintSource, TextPaintRevisions};

macro_rules! ensure_usage {
    ($condition:expr, $($message:tt)*) => {
        if !$condition { return Err(anyhow::anyhow!($($message)*).context(FailureKind::Usage)); }
    };
}

/// The levels of pieces, innermost first: a glyph's transform applies first, then its word's, then its line's.
const LEVELS: [TextUnit; 3] = [TextUnit::Glyphs, TextUnit::Words, TextUnit::Lines];

/// Columns in glyph, word and line order, shared with LEVELS.
struct TextLevels<T>([T; 3]);

impl<T> std::ops::Index<&TextUnit> for TextLevels<T> {
    type Output = T;
    fn index(&self, unit: &TextUnit) -> &T {
        &self.0[match unit { TextUnit::Glyphs => 0, TextUnit::Words => 1, TextUnit::Lines => 2 }]
    }
}

#[derive(Clone, Default)]
struct TextTracks {
    pieces: bool,
    paint_units: [Option<Arc<std::collections::BTreeSet<(TextUnit, usize)>>>; 2],
    glyphs: bool,
}

/// One node of the canvas tree, flattened in paint order.
struct Item {
    name: String,
    visible: bool,
    text: Option<TextItem>,
    /// How Moda draws the node.
    node: crate::canvas_draw::Node,
}
/// What an item's evaluation starts from, held apart from the item's larger parts (its name, text and node) so
/// evaluating an item that only moves reads this alone: its parent, its placement and opacity at rest, a path's
/// keyed numbers at rest, and whether it is text.
struct Rest {
    parent: Option<usize>,
    transform: v2::Transform,
    opacity: f32,
    path: Option<crate::canvas_draw::PathValues>,
    text: bool,
}
/// Apply one numeric placement track in layer order. Both the initial evaluation and
/// subsequent writes to a placement column use the same blending arithmetic.
fn blend_placement(part: u8, transform: &mut v2::Transform, opacity: &mut f32,
    value: &[f32], weight: f32, additive: bool) {
    use crate::animation::{blend, BlendKind};
    use crate::canvas_draw::change;
    match part {
        change::TRANSLATION => blend(&mut transform.translation, value, weight, additive, BlendKind::Add),
        change::SCALE => blend(&mut transform.scale, value, weight, additive, BlendKind::Multiply),
        change::ROTATION => blend(transform.rotation_mut(), value, weight, additive, BlendKind::Rotation),
        change::OPACITY => blend(std::slice::from_mut(opacity), value, weight, additive, BlendKind::Add),
        _ => unreachable!("a numeric placement column"),
    }
}

/// Every item's children, in paint order, as one array.
struct Children {
    /// Item `i`'s children are `items[first[i]..first[i + 1]]`.
    first: Vec<usize>,
    items: Vec<usize>,
}
impl Children {
    /// From each item's parent. Items are in paint order, a parent before its children, so each item's
    /// children come out ascending.
    fn new(rest: &[Rest]) -> Self {
        let mut first = vec![0; rest.len() + 1];
        for parent in rest.iter().filter_map(|item| item.parent) { first[parent + 1] += 1; }
        for item in 0..rest.len() { first[item + 1] += first[item]; }
        let mut items = vec![0; first[rest.len()]];
        let mut next = first.clone();
        for (item, at_rest) in rest.iter().enumerate() {
            if let Some(parent) = at_rest.parent {
                items[next[parent]] = item;
                next[parent] += 1;
            }
        }
        Self { first, items }
    }
    fn of(&self, item: usize) -> &[usize] {
        &self.items[self.first[item]..self.first[item + 1]]
    }
}
/// A node of the canvas's scene, where the scene holds it, with its parent's item.
struct SceneNode<'a> {
    node: &'a v2::Node,
    parent: Option<usize>,
}
impl crate::animation::ClockNode for SceneNode<'_> {
    fn node(&self) -> &v2::Node { self.node }
    fn parent(&self) -> Option<usize> { self.parent }
    /// The mark is the 3D installer's; the canvas evaluates its items from their tracks.
    fn set_dynamic(&mut self) {}
}
/// A paint: straight-alpha sRGB, or `none`.
type Paint = Option<[f32; 4]>;
/// A text node laid out once: Moda's spec at rest, and, in the code points of Moda's canonical plain text
/// (the space its unit transforms index), each glyph, word and line, with the ink box it covers at rest.
#[derive(Clone)]
struct TextItem {
    /// The node's HTML (each run a span with its font, naming its paint; the HTML as drawn, colours and all, when
    /// the degraded tier rasters it), its runs' colours, box and wrapping; no stroke, no motion.
    spec: moda_text::ops::TextNodeSpec,
    /// Each code point's run. This and the three tables below are of the text's content and layout, not its
    /// paints: text that takes new paints shares them.
    run_of: Arc<Vec<usize>>,
    code_points: usize,
    clip_source: bool,
    units: Arc<TextLevels<Vec<Vec<usize>>>>,
    /// Per level, the piece each code point belongs to.
    piece_of: Arc<TextLevels<Vec<Option<usize>>>>,
    /// Per level, each piece's ink box at rest (node px: left, top, right, bottom).
    boxes: Arc<TextLevels<Vec<Option<[f64; 4]>>>>,
    /// The centre of all the node's ink at rest: Moda turns every code point about it when all form one unit.
    centre: DVec2,
    /// Each code point's fill and stroke at rest (its run's, else the node's).
    fill: Vec<PaintSource>,
    stroke: Vec<PaintSource>,
    stroke_width: f64,
    tracks: TextTracks,
    /// The text laid out: the one layout its box, lines, ink boxes and draws read.
    layout: Arc<Laid>,
    degraded_paints: Vec<(PaintSource, String)>,
    /// The laid-out box (px): its width and the text's height.
    size: [f64; 2],
}
impl TextItem {
    /// Boundaries follow authored runs and independently keyed paint units.
    fn paint_ranges(&self, stroke: bool) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        let keyed = self.tracks.paint_units[usize::from(stroke)].as_deref();
        let mut start = 0;
        std::iter::from_fn(move || {
            if start >= self.code_points { return None; }
            let end = (start + 1..self.code_points).find(|&cp| {
                self.run_of[cp] != self.run_of[start] || keyed.is_some_and(|units| LEVELS.iter().any(|level| {
                    let (a, b) = (self.piece_of[level][start], self.piece_of[level][cp]);
                    a != b && [a, b].into_iter().flatten().any(|i| units.contains(&(*level, i)))
                }))
            }).unwrap_or(self.code_points);
            let range = start..end;
            start = end;
            Some(range)
        })
    }
    /// Moda's vector lane declined this text (right-to-left or complex script); its degraded tier draws it.
    fn degraded(&self) -> bool { self.layout.vector.is_none() }
}
/// A text node's accepted layout, made once for its content and shared by every node with the same content: the
/// same HTML without its colours (text, runs, fonts and their axes, size, spacing, alignment, white space), box
/// width and wrapping, on this document's fonts. Where a node draws and in what colours is no part of it.
struct Laid {
    metrics: Vec<[f64; 2]>,
    /// Moda's vector layout; `None` when its vector lane declines the text and its degraded tier draws it.
    vector: Option<moda_text::vector::VectorLayout>,
    /// The box's width (px): the widest line when the text has no width of its own.
    width: f64,
    /// The text's height (px).
    height: f64,
    /// Per line, the source code points its segments hold, in order.
    lines: Vec<Vec<usize>>,

}
/// What lays text out: its HTML with paints named, not coloured (`TextHtml::layout`), its width and whether it wraps. The line-height policy is CSS for every node.
type LaidKey = (Arc<str>, Option<u64>, bool);
/// One glyph's, word's or line's animated fields, in the text node's space.
#[derive(Clone, PartialEq)]
struct UnitValue {
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
    opacity: f32,
    /// Paint layers resolve after the sampled text layout is known.
    fill: Vec<UnitPaint>,
    stroke: Vec<UnitPaint>,
    origin: Option<[(f32, f32); 2]>,
}
#[derive(Clone, Default, PartialEq)]
struct UnitValues(Vec<((TextUnit, usize), UnitValue)>);

impl UnitValues {
    fn position(&self, key: &(TextUnit, usize)) -> Result<usize, usize> {
        self.0.binary_search_by_key(key, |entry| entry.0)
    }

    fn get(&self, key: &(TextUnit, usize)) -> Option<&UnitValue> {
        self.position(key).ok().map(|slot| &self.0[slot].1)
    }

    fn write(&mut self, key: (TextUnit, usize)) -> &mut UnitValue {
        let slot = match self.position(&key) {
            Ok(slot) => slot,
            Err(slot) => {
                self.0.insert(slot, (key, UnitValue::default()));
                slot
            }
        };
        &mut self.0[slot].1
    }
}

#[derive(Clone, PartialEq)]
enum UnitPaint {
    Css(Arc<crate::animation::style::StyleValue>, f32),
    Numeric([f32;4], f32, bool),
}
impl Default for UnitValue {
    fn default() -> Self {
        Self { translation: [0.; 3], rotation: [0., 0., 0., 1.], scale: [1.; 3], opacity: 1., fill: Vec::new(), stroke: Vec::new(), origin: None }
    }
}
/// One batch Moda draws: a text node's item, which of the node's batches it is, its spec at an instant and its
/// model matrix (canvas px, where Moda's frame walk places the node), and the layout it draws.
#[derive(Clone)]
struct Draw {
    item: usize,
    sub: usize,
    spec: Arc<moda_text::ops::TextNodeSpec>,
    model: [f32; 16],
    layout: Arc<Laid>,
}

/// A plan can outlive the next evaluated frame. It holds one shared directory of text rows.
/// Updating an item detaches its directory page and batch vector while earlier plans keep theirs.
const DRAW_ROWS_PER_PAGE: usize = 64;
type DrawPage = [Option<Arc<Vec<Draw>>>; DRAW_ROWS_PER_PAGE];

#[derive(Clone)]
struct TextDrawRows(Arc<Vec<Option<Arc<DrawPage>>>>);

impl TextDrawRows {
    fn new(items: usize) -> Self {
        Self(Arc::new(vec![None; items.div_ceil(DRAW_ROWS_PER_PAGE)]))
    }

    fn get(&self, item: usize) -> &[Draw] {
        self.0.get(item / DRAW_ROWS_PER_PAGE).and_then(Option::as_ref)
            .and_then(|page| page[item % DRAW_ROWS_PER_PAGE].as_ref())
            .map_or(&[], |row| row.as_slice())
    }

    fn slot(&mut self, item: usize) -> &mut Option<Arc<Vec<Draw>>> {
        let directory = Arc::make_mut(&mut self.0);
        let page = directory[item / DRAW_ROWS_PER_PAGE].get_or_insert_with(|| Arc::new(std::array::from_fn(|_| None)));
        &mut Arc::make_mut(page)[item % DRAW_ROWS_PER_PAGE]
    }

    fn replace(&mut self, item: usize, draws: &mut Vec<Draw>) {
        let next = (!draws.is_empty()).then(|| Arc::new(std::mem::take(draws)));
        if let Some(old) = std::mem::replace(self.slot(item), next).and_then(Arc::into_inner) {
            *draws = old;
        }
    }
}

impl std::ops::Index<usize> for TextDrawRows {
    type Output = [Draw];
    fn index(&self, item: usize) -> &[Draw] { self.get(item) }
}

#[cfg(test)]
impl std::ops::IndexMut<usize> for TextDrawRows {
    fn index_mut(&mut self, item: usize) -> &mut [Draw] {
        Arc::make_mut(self.slot(item).as_mut().expect("snapshot text row")).as_mut_slice()
    }
}

/// A canvas document: its text laid out, its fonts loaded, its tracks compiled.
pub(crate) struct Document {
    prepared: Arc<Prepared>,
    styles: RefCell<Vec<Option<StyledItem>>>,
    fonts: Fonts,
    /// The tracks' samples, reused from frame to frame.
    samples: RefCell<crate::animation::Samples>,
    media: RefCell<crate::canvas_draw::Media>,
    retained: RefCell<Option<crate::canvas_draw::Retained>>,
    text_specs: RefCell<Vec<Option<TextSpecs>>>,
    evaluated: RefCell<Option<Evaluated>>,
    /// Whether the installed scene is owed the whole evaluated table. A frame writes the scene as it ends
    /// (`write_rows`), so this is set only before the first install, after a failure part way through a write,
    /// and, for a canvas with worlds, after a frame that needs a whole write: those wait for the plan, which has
    /// the worlds' pictures.
    stale: std::cell::Cell<bool>,
    /// The text builder made of the evaluated table's batches; `None` when a frame changed a batch since.
    text: RefCell<Option<crate::canvas_draw::CanvasTextBuilder>>,
    /// Which document this is, unique in the process, and how many snapshots tests have taken of it.
    #[cfg(test)]
    id: u64,
    #[cfg(test)]
    snapshots: std::cell::Cell<u64>,
    /// Scratch for naming the changed items: per item, 0 or one more than its slot among them.
    item_marks: RefCell<Vec<u32>>,
    /// Frames by what changed since the frame before: unchanged, then each `ChangeKind`.
    changes: std::cell::Cell<[u64; 5]>,
}
/// Visit the sampled paints, including images selected by path, run and glyph tracks.
fn visit_image_paints(node: &crate::canvas_draw::Node, values: &crate::canvas_draw::Values, mut image: impl FnMut(&str)) {
    let mut paint = |value: &moda_scene::value::Value| {
        if value["type"] == "image" && let Some(src) = value["src"].as_str() { image(src); }
    };
    if let (crate::canvas_draw::Kind::Path(path), Some(values)) = (&node.kind, &values.path) {
        for source in path.paints(values) {
            if let PaintSource::Moda(value) = source.as_ref() { paint(value); }
        }
    }
    fn text(parts: &[crate::canvas_draw::TextPart], paint: &mut impl FnMut(&moda_scene::value::Value)) {
        use crate::canvas_draw::TextPart;
        for part in parts {
            match part {
                TextPart::Paint { paint: value, .. } => paint(value),
                TextPart::Group { children, .. } => text(children, paint),
                TextPart::Batch(_) => {}
            }
        }
    }
    text(&values.text_composition, &mut paint);
}

/// The canvas evaluator's resolved movie paints, without frame decoding or GPU work.
pub(crate) struct SoundPaints {
    document: Document,
    items: Vec<usize>,
    names: RefCell<Vec<Option<BTreeSet<String>>>>,
    bindings: Vec<crate::animation::BindingPlan>,
    scratch: RefCell<crate::animation::SoundScratch>,
}

impl SoundPaints {
    pub(crate) fn new(scene: &Scene, base: &Path, wanted: &std::collections::HashSet<&str>, opened: &crate::footage::Opened) -> Result<Self> {
        let document = Document::new_with_paints(scene, base, Some(wanted), opened)?;
        let items: Vec<_> = document.items.iter().enumerate()
            .filter(|(_, item)| wanted.contains(item.name.as_str())).map(|(i, _)| i).collect();
        let names = RefCell::new(vec![None; items.len()]);
        let bindings = items.iter().map(|&item| {
            let tracks = document.tracks_of(item).iter().copied().filter(|&flat|
                document.kinds[flat as usize] == Some(ChangeKind::Content));
            document.plan.binding_plan(item, tracks)
        }).collect();
        let scratch = RefCell::new(document.plan.sound_scratch());
        Ok(Self { document, items, names, bindings, scratch })
    }

    pub(crate) fn items(&self) -> impl Iterator<Item = (&str, usize)> + '_ {
        self.items.iter().enumerate().map(|(slot, &item)| (self.document.items[item].name.as_str(), slot))
    }

    pub(crate) fn animated(&self, slot: usize) -> bool { !self.bindings[slot].is_empty() }

    pub(crate) fn sample(&self, slot: usize, instant: &vork_time::exact::Germ, first: i64, last: i64,
        scope_rate: Option<&vork_time::exact::Rational>) -> Result<i64> {
        let doc = &self.document;
        let binding = &self.bindings[slot];
        let item = self.items[slot];
        let mut names = self.names.borrow_mut();
        if binding.is_empty() && names[slot].is_some() { return Ok(last); }
        let mut samples = doc.samples.borrow_mut();
        let end = doc.plan.sample_binding(binding, instant, first, last, scope_rate,
            &mut self.scratch.borrow_mut(), &mut samples)?;
        if names[slot].is_none() || !samples.changed().is_empty() {
            let mut values = doc.unevaluated(item);
            let mut node = None;
            doc.evaluate_item(&samples, item, true, true, &mut doc.styles.borrow_mut(),
                &mut doc.text_specs.borrow_mut(), &mut node, &mut values, &mut Vec::new())?;
            let mut movies = BTreeSet::new();
            let node = node.as_deref().unwrap_or(&doc.items[item].node);
            visit_image_paints(node, &values, |name| { movies.insert(name.to_owned()); });
            names[slot] = Some(movies);
        }
        Ok(end)
    }

    pub(crate) fn shows(&self, item: usize, movie: &str) -> bool {
        self.names.borrow()[item].as_ref().expect("sampled paints").contains(movie)
    }
}

/// Bounds of every authored subtree at the last instant a question measured.
struct MeasuredBounds {
    time: vork_time::Request,
    boxes: Vec<Option<[f64; 4]>>,
    effect: Vec<Option<usize>>,
}
pub(crate) struct BoundsBatch {
    wanted: std::collections::HashSet<usize>,
    measured: Option<MeasuredBounds>,
}
impl MeasuredBounds {
    fn answer(&self, doc: &Document, item: usize, read: &str) -> Result<serde_json::Value> {
        if let Some(effect) = self.effect[item] {
            bail!("ask: bounds depend on mask, clip or filter on {:?}; remove it to measure the geometry", doc.items[effect].name);
        }
        let Some([x, y, right, bottom]) = self.boxes[item] else {
            return Ok(serde_json::Value::Null);
        };
        if read == "bounds" {
            return Ok(serde_json::json!([[x, y], [right, bottom]]));
        }
        crate::ask::screen_box([x, y, right, bottom], f64::from(doc.page.size[0]), f64::from(doc.page.size[1]))
    }
}

/// Immutable installed inputs shared by concurrently sampled frames. Each document owns
/// its samples, stamped scene, planner and draw caches; no frame copies these inputs.
pub(crate) struct Prepared {
    opened: crate::footage::Opened,
    /// The canvas's scene, kept when its image graph or a view stage is prepared from it: it keeps a float
    /// picture (`float`), or it blends in another encoding than sRGB, which only its view turns into code
    /// values a reader or a display takes.
    view: Option<Scene>,
    /// `float`.
    float: bool,
    /// The encoding the canvas blends in (`composition.color_space`).
    working: v2::WorkingColorSpace,
    /// The image files the canvas's image graph may read, by name, each with its fixed time when it has one
    /// (`images.X.time`): every frame carries their instants (`Frame::images`).
    graph_images: Vec<(String, Option<vork_time::Clock>)>,
    images: v2::Table<v2::Image>,
    base: PathBuf,
    font_lists: Vec<Vec<PathBuf>>,
    /// What a restyling track starts from, for each item one names (`restyled_field`); `None` for every other
    /// item. By item.
    authored: Vec<Option<Box<Authored>>>,
    items: Vec<Item>,
    /// Each item's rest values, by item.
    rest: Vec<Rest>,
    children: Children,
    /// Each named node's item.
    by_name: HashMap<String, usize>,
    plan: crate::animation::Plan,
    /// The output page Moda draws on: its size, the viewBox placement and the background.
    page: crate::canvas_draw::Page,
    /// The worlds in the canvas: each one's item and its scene, drawn at `size` px.
    worlds: Vec<WorldPart>,
    /// Each track's kind, by flat track index (every track of a canvas is authored, so that is `Plan::sampled`
    /// order); `None` for a control track (a layer's weight).
    kinds: Vec<Option<ChangeKind>>,
    /// Each track's item, when it targets a node of this canvas or a piece of its text.
    track_items: Vec<Option<usize>>,
    /// What a change of each track changes of its item (`canvas_draw::change`), by flat track index.
    track_parts: Vec<u8>,
    /// Each item's tracks, ascending: item `i`'s are `item_tracks[item_tracks_first[i]..item_tracks_first[i + 1]]`.
    /// An item is evaluated by blending these from its rest values, and visits no other track.
    item_tracks_first: Vec<u32>,
    item_tracks: Vec<u32>,
    /// The tracks of the page's background: the tracks that target no item and still draw.
    page_tracks: Vec<u32>,
}
impl std::ops::Deref for Document {
    type Target = Prepared;
    fn deref(&self) -> &Prepared { &self.prepared }
}
/// A world inside the canvas.
pub(crate) struct WorldPart {
    item: usize,
    pub scene: Scene,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum StyleField {
    Fill,
    Stroke,
    StrokeWidth,
    Property(v2::StyleProperty),
}

impl StyleField {
    fn name(self) -> &'static str {
        match self {
            Self::Fill => "fill",
            Self::Stroke => "stroke",
            Self::StrokeWidth => "stroke-width",
            Self::Property(property) => property.name(),
        }
    }
}

struct ReferenceFields {
    path_data: StyleField,
    filter: StyleField,
    clip_path: StyleField,
    mask: StyleField,
}

static REFERENCE_FIELDS: std::sync::LazyLock<ReferenceFields> = std::sync::LazyLock::new(|| {
    let property = |name| StyleField::Property(v2::StyleProperty::parse(name).expect("registered canvas field"));
    ReferenceFields { path_data: property("d"), filter: property("filter"), clip_path: property("clip-path"), mask: property("mask") }
});

#[derive(Default, PartialEq)]
struct StyleFields(Vec<(StyleField, Arc<crate::animation::style::StyleValue>)>);

impl StyleFields {
    fn get(&self, key: StyleField) -> Option<&Arc<crate::animation::style::StyleValue>> {
        self.0.iter().find(|entry| entry.0 == key).map(|entry| &entry.1)
    }

    fn set(&mut self, key: StyleField, value: Arc<crate::animation::style::StyleValue>) {
        if let Some(entry) = self.0.iter_mut().find(|entry| entry.0 == key) {
            entry.1 = value;
        } else {
            // Preserve the authored field-application order when multiple tracks write one node.
            let position = self.0.partition_point(|entry| entry.0.name() < key.name());
            self.0.insert(position, (key, value));
        }
    }

    fn content(&self, include_paint: bool) -> impl Iterator<Item = &(StyleField, Arc<crate::animation::style::StyleValue>)> {
        self.0.iter().filter(move |(field, _)| match field {
            StyleField::Fill | StyleField::Stroke => include_paint,
            _ => *field != REFERENCE_FIELDS.filter && *field != REFERENCE_FIELDS.clip_path && *field != REFERENCE_FIELDS.mask,
        })
    }

}
struct StyledItem {
    fields: StyleFields,
    draw: Arc<crate::canvas_draw::Node>,
    text: Option<Arc<TextItem>>,
    revision: u64,
    text_revision: u64,
}
#[derive(Clone)]
struct TextSpecs {
    paint_revisions: TextPaintRevisions,
    revision: u64,
    content_revision: u64,
    style_revision: Option<u64>,
    units: UnitValues,
    specs: Vec<Arc<moda_text::ops::TextNodeSpec>>,
    composition: Vec<crate::canvas_draw::TextPart>,
    clip: Option<Arc<moda_text::ops::TextNodeSpec>>,
}
/// The scene's fonts as Moda's text lanes read them, shared with the text builder Moda's frame walk calls.
#[derive(Clone)]
struct Fonts {
    host: Arc<moda_text::TextHost>,
    store: std::rc::Rc<moda_text::vector::VectorFontStore>,
    /// The layouts in use, by content: an entry lives as long as a text item or draw holds it.
    layouts: std::rc::Rc<RefCell<Layouts>>,
}
#[derive(Default)]
struct Layouts {
    by_content: HashMap<LaidKey, std::sync::Weak<Laid>>,
    /// The entry count at which dropped layouts are next swept out.
    sweep_at: usize,
}
/// One instant of a canvas, as asked for. Its values are the document's evaluated table (`Evaluated`), which
/// stands at this instant until the next frame is evaluated.
struct Frame {
    /// The instant asked for, exactly: an authored view's `ctx.time`, and what its animated inputs are sampled
    /// at, so a view's keys fall on the same side of a boundary as the canvas's own.
    request: vork_time::Request,
    /// World clocks advance even when their surrounding canvas is unchanged.
    worlds: Vec<Option<vork_time::Request>>,
    /// The instant each image of the canvas's image graph shows (`Document::graph_images`): scene time, or the
    /// image's own fixed or keyed time.
    images: Vec<(String, vork_time::exact::Germ)>,
}
/// Every item's values at one instant, and the text batches to draw.
#[derive(Clone)]
struct FrameState {
    background: Paint,
    nodes: Vec<Option<Arc<crate::canvas_draw::Node>>>,
    values: Vec<crate::canvas_draw::Values>,
    /// Each item's text batches, in the order it draws them (`Draw::sub`); empty for an item that draws none.
    draws: TextDrawRows,
    /// Which nodes draw at all: visible, present on their clocks, and so is every ancestor.
    shown: Vec<bool>,
    /// Of those, the ones no zero scale at or above them collapses: a text node's batches are built only then.
    drawing: Vec<bool>,
}
impl FrameState {
    /// Every text batch, in item order.
    #[cfg(test)]
    fn batches(&self) -> impl Iterator<Item = &Draw> {
        (0..self.values.len()).flat_map(|item| self.draws.get(item))
    }
}
/// The document's evaluated table: every item's values at the last frame evaluated, written in place by each
/// frame. There is one, it is never copied, and it has no revision: the document's samples stand at the same
/// frame, so the tracks the next sampling reports changed are what differs from it.
struct Evaluated {
    present: Vec<bool>,
    state: FrameState,
}
/// What a frame changed of the evaluated table.
enum Changed {
    /// Nothing: the table stands.
    Nothing,
    /// These items, each with what of it changed (`canvas_draw::change`); and whether every changed track draws
    /// through an item of the canvas.
    Items(Vec<(usize, u8)>, bool),
    /// Every item was evaluated.
    Every,
}
/// A test's copy of the evaluated table at one frame: tests keep frames side by side, read their values and edit
/// them by hand, which the document's own table (one, in place) does not allow.
#[cfg(test)]
struct Snapshot {
    state: FrameState,
    /// The document it was taken of (`Document::id`), and which snapshot of that document it is, counted from one.
    document: u64,
    serial: u64,
    edited: bool,
}
#[cfg(test)]
impl std::ops::Deref for Snapshot {
    type Target = FrameState;
    fn deref(&self) -> &FrameState { &self.state }
}
#[cfg(test)]
impl std::ops::DerefMut for Snapshot {
    /// A test edits a frame's values by hand: the result is a state no evaluation made.
    fn deref_mut(&mut self) -> &mut FrameState {
        self.edited = true;
        &mut self.state
    }
}
/// What a track's change asks of a frame, decided once at install. Ordered by how much a frame redoes:
/// a frame's kind is the largest kind among the tracks whose sample or weight changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ChangeKind {
    /// A node's translation, rotation, scale or opacity: its placement, and its descendants'.
    Placement,
    /// A path's colour, the stops and geometry of a gradient of one form, or paint opacity: values of the paint
    /// it already has.
    PaintValue,
    /// Geometry, stroke shape, text, a paint's kind or a filter's parameters: that node is prepared again.
    Content,
    /// Which nodes draw, the page, and clip and mask relations.
    Structure,
}
impl ChangeKind {
    /// A track's kind, or `None` for a control track that draws nothing itself (a layer's weight): its effect
    /// is the changed weight of the tracks it drives, which carry their own kinds.
    /// `form` is the one paint form the track's values keep, if they do; `rest(node, fill)` the form of the
    /// path's paint at rest (a colour over a gradient, an image or `none` changes the paint's kind or presence;
    /// so does a gradient over one of another form). `path(node)` says the node is a path that owns its paints.
    fn of(target: &TrackTarget, numeric: bool, form: Option<PaintForm>, rest: impl Fn(&str, bool) -> Option<PaintForm>,
        path: impl Fn(&str) -> bool) -> Option<Self> {
        use v2::NodeProperty as P;
        Some(match target {
            TrackTarget::LayerWeight { .. } => return None,
            TrackTarget::Node { property: P::Translation | P::Rotation | P::Scale | P::Opacity, .. } if numeric => Self::Placement,
            // The writer drops an invisible node from the generated hierarchy.
            TrackTarget::Node { property: P::Visibility(_), .. } => Self::Structure,
            TrackTarget::Node { node, property: property @ (P::Fill | P::Stroke) } if form.is_some() && rest(node, *property == P::Fill) == form => Self::PaintValue,
            // Opacity scales whatever paint is there: a colour, a gradient or an image.
            TrackTarget::Node { node, property: P::FillOpacity | P::StrokeOpacity } if numeric && path(node) => Self::PaintValue,
            TrackTarget::Node { property: P::Style(p), .. } if matches!(p.name(), "clip-path" | "mask" | "mask-type" | "mix-blend-mode" | "isolation") => Self::Structure,
            TrackTarget::Render(v2::RenderProperty::Background) => Self::Structure,
            _ => Self::Content,
        })
    }
}

/// The node and the authored field a track restyles: a style, paint or stroke-width track's value is mixed from
/// that field as authored, and a changed content field makes the node again with the new value in it. `None`
/// for every other track. (A stroke-width track restyles a text node; a path's stroke width is a number of its
/// own.)
fn restyled_field(target: &TrackTarget) -> Option<(&str, StyleField)> {
    use v2::NodeProperty as P;
    match target {
        TrackTarget::Node { node, property: P::Style(p) } => Some((node.as_str(), StyleField::Property(*p))),
        TrackTarget::Node { node, property: P::Fill } => Some((node.as_str(), StyleField::Fill)),
        TrackTarget::Node { node, property: P::Stroke } => Some((node.as_str(), StyleField::Stroke)),
        TrackTarget::Node { node, property: P::StrokeWidth } => Some((node.as_str(), StyleField::StrokeWidth)),
        _ => None,
    }
}
/// An item some track restyles, as authored: the typed node (no children, no layers of its own), which a
/// restyle copies and writes its changed fields into, and the authored value of each field such a track names,
/// which that track's value is mixed from.
struct Authored {
    node: v2::Node,
    fields: Vec<(StyleField, serde_json::Value)>,
}
impl Authored {
    /// The authored value of a field a track names; null for one the node does not write.
    fn field(&self, name: StyleField) -> &serde_json::Value {
        static NULL: serde_json::Value = serde_json::Value::Null;
        self.fields.iter().find(|(field, _)| *field == name).map_or(&NULL, |(_, value)| value)
    }
}

/// A CSS colour (or `none`) as straight-alpha sRGB.
fn paint(here: &str, value: &str) -> Result<Paint> {
    v2::text::paint(value).with_context(|| here.to_string())
}
/// A paint as the CSS colour Moda reads, quantised to 8 bits a channel.
fn css(paint: [f32; 4]) -> String {
    let [r, g, b, a] = paint.map(|c| (c.clamp(0., 1.) * 255.).round() as u8);
    format!("rgba({r}, {g}, {b}, {})", f32::from(a) / 255.)
}
fn quantised(paint: Paint) -> Option<[u8; 4]> {
    paint.filter(|p| p[3] > 0.).map(|p| p.map(|c| (c.clamp(0., 1.) * 255.).round() as u8))
}

/// The HTML Moda's parser reads for a text node: one paragraph per line of `pre-line` text, else one
/// paragraph; each stretch of one run a span with the node's CSS properties and the run's own. When tracks
/// address the node's glyphs, every letter is its own glyph: ligatures are off (`liga`, `clig`, `dlig` and
/// `hlig` 0, after the authored features), so a ligature spanning addressed glyphs draws as its letters.
fn html(kind: &NodeKind, family: &dyn Fn(Option<&v2::FontFiles>) -> String, separate_letters: bool) -> Result<String> {
    Ok(text_markup(kind, family, separate_letters, false)?.0)
}
/// What lays a text node out, and its runs' colours.
struct TextHtml {
    /// The node's HTML with each run naming its paint (`--nf-paint`: the run's index; the empty line's is the run
    /// count) where the drawn HTML (`html`) has its `color`. Runs and shaping are those of the drawn HTML, and text
    /// that differs only in its colours has one layout.
    layout: String,
    /// The paints: each run's colour as Moda draws that `color`, then the empty line's (the node's own).
    colors: Vec<[f32; 4]>,
}
fn text_html(kind: &NodeKind, family: &dyn Fn(Option<&v2::FontFiles>) -> String, separate_letters: bool) -> Result<TextHtml> {
    let (layout, colors) = text_markup(kind, family, separate_letters, true)?;
    Ok(TextHtml { layout, colors })
}
/// A run's flat fill as the CSS colour Moda reads: its own or the node's; white under a gradient or an image
/// (the paint is drawn through the glyphs' coverage), transparent for `none`.
fn run_fill(run: &v2::Run, fill: &v2::Paint) -> Result<String> {
    let fill = match run.fill.as_ref().unwrap_or(fill) {
        v2::Paint::None(_) => None,
        v2::Paint::Css(css) if !css.contains("gradient(") => v2::text::paint(css)?,
        _ => Some([1.; 4]),
    };
    Ok(fill.map(css).unwrap_or_else(|| "rgba(0, 0, 0, 0)".into()))
}
/// A text node's paints as a draw supplies them (`TextNodeSpec::run_colors`): each run's colour, then the empty
/// line's. What `text_markup` returns beside the HTML, without making the HTML.
fn text_colors(kind: &NodeKind) -> Result<Vec<[f32; 4]>> {
    let NodeKind::Text { content, fill, .. } = kind else { unreachable!() };
    content.runs().iter().chain([&v2::Run::default()]).map(|run| Ok(moda_text::vector::run_color(&run_fill(run, fill)?))).collect()
}
/// A text node's HTML and its runs' colours. `named`: each run names its paint (the layout's HTML); else each
/// run's `color` is its flat fill (the HTML as drawn, which the degraded tier rasters).
fn text_markup(kind: &NodeKind, family: &dyn Fn(Option<&v2::FontFiles>) -> String, separate_letters: bool, named: bool) -> Result<(String, Vec<[f32; 4]>)> {
    let NodeKind::Text { content, font_size, font_weight, font_style, font_variation_settings, font_feature_settings, letter_spacing,
        line_height, text_align, direction, white_space, fill, .. } = kind else { unreachable!() };
    let runs = content.runs();
    let style = |run: &v2::Run, paint: Option<usize>| -> Result<(String, String)> {
        let mut style = format!("font-family: '{}'; font-size: {}px; font-weight: {}; font-style: {}; letter-spacing: {}px", family(run.font.as_ref()),
            run.font_size.unwrap_or(*font_size), run.font_weight.unwrap_or(*font_weight).value(),
            match run.font_style.unwrap_or(*font_style) { v2::FontStyle::Normal => "normal", v2::FontStyle::Italic => "italic", v2::FontStyle::Oblique => "oblique" },
            run.letter_spacing.unwrap_or(*letter_spacing));
        match line_height {
            v2::LineHeight::Normal(_) => style.push_str("; line-height: normal"),
            v2::LineHeight::Multiple(multiple) => style.push_str(&format!("; line-height: {multiple}")),
        }
        let variations = run.font_variation_settings.as_ref().unwrap_or(font_variation_settings);
        let features = run.font_feature_settings.as_ref().unwrap_or(font_feature_settings);
        if variations != "normal" { style.push_str(&format!("; font-variation-settings: {variations}")); }
        const NO_LIGATURES: &str = "\"liga\" 0, \"clig\" 0, \"dlig\" 0, \"hlig\" 0";
        match (features != "normal", separate_letters) {
            (true, true) => style.push_str(&format!("; font-feature-settings: {features}, {NO_LIGATURES}")),
            (true, false) => style.push_str(&format!("; font-feature-settings: {features}")),
            (false, true) => style.push_str(&format!("; font-feature-settings: {NO_LIGATURES}")),
            (false, false) => {}
        }
        let fill = run_fill(run, fill)?;
        match paint {
            Some(paint) => style.push_str(&format!("; {}: {paint}", moda_text::types::PAINT_PROPERTY)),
            None => style.push_str(&format!("; color: {fill}")),
        }
        Ok((vork_std::moda::escape_html(&style).replace('"', "&quot;"), fill))
    };
    // Each run's style and colour as drawn, then the empty line's (the node's own).
    let styled = runs.iter().chain([&v2::Run::default()]).enumerate()
        .map(|(paint, run)| style(run, named.then_some(paint))).collect::<Result<Vec<_>>>()?;
    let colors: Vec<[f32; 4]> = styled.iter().map(|(_, fill)| moda_text::vector::run_color(fill)).collect();
    let styles: Vec<String> = styled.into_iter().map(|(style, _)| style).collect();
    let span = |run: usize, text: &str| format!("<span style=\"{}\">{}</span>", styles[run], vork_std::moda::escape_html(text));
    let block = format!("text-align: {}; direction: {}", text_align.css(), match direction { v2::Direction::Ltr => "ltr", v2::Direction::Rtl => "rtl" });
    let laid = v2::laid_out_runs(content, *white_space);
    let html = laid.split(|(c, _)| *c == '\n')
        .map(|line| {
            let mut spans = String::new();
            let mut start = 0;
            while start < line.len() {
                let run = line[start].1;
                let end = line[start..].iter().position(|(_, r)| *r != run).map_or(line.len(), |n| start + n);
                spans.push_str(&span(run, &line[start..end].iter().map(|(c, _)| *c).collect::<String>()));
                start = end;
            }
            if line.is_empty() {
                // An empty line keeps the height of the node's text.
                spans = format!("<span style=\"{}\"></span>", styles[runs.len()]);
            }
            format!("<p style=\"{block}\">{spans}</p>")
        })
        .collect();
    Ok((html, colors))
}

/// For each line of a layout (the source code points its segments hold), the code points of `plain` it holds:
/// its characters in order, the white space at its breaks with it.
fn line_code_points(layout: &[Vec<usize>], plain: &[char]) -> Vec<Vec<usize>> {
    let mut lines = Vec::new();
    let mut next = 0;
    for line in layout {
        let mut held = Vec::new();
        for &cp in line {
            if cp >= plain.len() { continue; }
            held.extend((next..=cp).filter(|&i| plain[i] != '\n'));
            next = next.max(cp+1);
        }
        lines.push(held);
    }
    if let Some(last) = lines.last_mut() { last.extend((next..plain.len()).filter(|&i| plain[i] != '\n')); }
    lines
}
fn union(a: Option<[f64; 4]>, b: [f64; 4]) -> [f64; 4] {
    match a { None => b, Some(a) => [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])] }
}

fn build_text(
    node: &v2::Node, images: &v2::Table<v2::Image>, base: &Path,
    lists: &[Vec<PathBuf>], fonts: &Fonts,
    tracks: TextTracks,
    styles: &StyleFields, media: &crate::canvas_draw::Media,
) -> Result<Option<TextItem>> {
    v2::Scene::validate_text(node, images)?;
    let files = |font: &v2::FontFiles| -> Vec<PathBuf> {font.files().iter().map(|file|base.join(file)).collect()};
    Ok(match &node.kind {
                NodeKind::Text { font, content, width, white_space, stroke_width, fill, .. } => {
                    let family = |run_font: Option<&v2::FontFiles>| {
                        let list = files(run_font.unwrap_or(font));
                        vork_std::moda::family(lists.iter().position(|l| *l == list).expect("collected font list"))
                    };
                    let TextHtml { layout: layout_html, colors } = text_html(&node.kind, &family, tracks.glyphs)?;
                    let layout_html: Arc<str> = layout_html.into();
                    let here = format!("node {:?}", node.name);
                    let laid = v2::laid_out_runs(content, *white_space);
                    let plain_text: String = laid.iter().map(|(ch, _)| *ch).collect();
                    let nowrap = width.is_none() || *white_space == v2::WhiteSpace::Nowrap;
                    let mut spec = moda_text::ops::TextNodeSpec {
                        node_id: node.name.clone(), html: layout_html.clone(), width: width.map(f64::from), opacity: 1., auto_width: nowrap, line_height_policy: moda_text::types::LineHeightPolicy::Css, ..Default::default()
                    };
                    // The layout is of the text without its colours: each run names its paint, and the draw has them.
                    // So the spec's HTML is the layout's, and a change of colour leaves it as it is.
                    spec.run_colors = Arc::new(colors);
                    // The degraded tier lays out and rasters the HTML as drawn, colours and all: it is made only for
                    // text that tier draws, and there the spec holds it.
                    let drawn = || self::html(&node.kind, &family, tracks.glyphs);
                    let accepted = fonts.lay_out(&spec, layout_html, &here, &drawn)?;
                    let is_degraded = accepted.vector.is_none();
                    if is_degraded { spec.html = drawn()?.into(); }
                    // Without a width the box is the widest line, so text-align places lines within it.
                    if width.is_none() { spec.width = Some(accepted.width); }
                    let plain: Vec<char> = plain_text.chars().collect();
                    let units = TextLevels(LEVELS.map(|unit| {
                        if unit == TextUnit::Lines { return line_code_points(&accepted.lines, &plain); }
                        v2::unit_ranges(&plain_text, unit).into_iter().map(|range| {
                            let start = plain_text[..range.start].chars().count();
                            (start..start + plain_text[range].chars().count()).collect::<Vec<_>>()
                        }).collect()
                    }));
                    let piece_of = TextLevels(LEVELS.map(|unit| {
                        let mut of = vec![None; plain.len()];
                        for (index, code_points) in units[&unit].iter().enumerate() {
                            for &cp in code_points { of[cp] = Some(index); }
                        }
                        of
                    }));
                    let run_of: Vec<usize> = laid.iter().map(|(_, run)| *run).collect();
                    let height = accepted.height;
                    // The box Moda's frame walk bounds the node by: its width, and the text's height.
                    let size = [spec.width.unwrap_or(0.), height];
                    let (fills, strokes) = text_paints(node, images, base, styles, media, size, &run_of)?;
                    // A paint as `text_paints` resolves it, for the degraded tier's coverage below.
                    let bounds = Some([0., 0., size[0], size[1]]);
                    let resolve = |field: StyleField, value:Option<&v2::Paint>, node_value:&v2::Paint| -> Result<PaintSource> {
                        if value.is_none() {
                            if let Some(crate::animation::style::StyleValue::Paint(paint))=styles.get(field).map(|v|v.as_ref()) {
                                return crate::canvas_draw::resolve_animated_paint(images,base,bounds.unwrap(),paint,media);
                            }
                        }
                        crate::canvas_draw::resolve_paint(images,base,value.unwrap_or(node_value),bounds,&format!("{here}: {}", field.name()),media)
                    };
                    let mut degraded_paints = Vec::new();
                    if is_degraded && fills.iter().any(|p| matches!(p, PaintSource::Moda(_))) {
                        for source in &fills {
                            if matches!(source, PaintSource::None) || degraded_paints.iter().any(|(p, _)| p == source) { continue; }
                            let mut coverage = node.kind.clone();
                            if let NodeKind::Text { content, fill: coverage_fill, .. } = &mut coverage {
                                let selected = |value: Option<&v2::Paint>| -> Result<v2::Paint> {
                                    Ok(if resolve(StyleField::Fill,value,fill)? == *source {
                                        v2::Paint::Css("white".into())
                                    } else { v2::Paint::NONE })
                                };
                                if let v2::Content::Runs(runs) = content {
                                    for run in runs { run.fill = Some(selected(run.fill.as_ref())?); }
                                }
                                *coverage_fill = selected(None)?;
                            }
                            degraded_paints.push((source.clone(), self::html(&coverage, &family, false)?));
                        }
                    }
                    let pieces_keyed = tracks.pieces;
                    if is_degraded {
                        ensure!(!pieces_keyed, "{here}: glyph, word and line tracks on right-to-left or complex-script text are not built yet");
                        ensure!(strokes.iter().all(|s| matches!(s, PaintSource::None)), "{here}: stroke on right-to-left or complex-script text is not built yet");
                    }
                    let (boxes, centre) = if pieces_keyed && !is_degraded {
                        let cp_boxes = accepted.vector.as_ref().map_or_else(Vec::new, |vector| vector.code_point_boxes(plain.len()));
                        let boxes = TextLevels(LEVELS.map(|unit| units[&unit].iter().map(|cps|
                            cps.iter().filter_map(|&cp| cp_boxes[cp]).fold(None, |a, b| Some(union(a, b)))
                        ).collect()));
                        let all = cp_boxes.iter().flatten().fold(None, |a, b| Some(union(a, *b)));
                        (boxes, all.map_or(DVec2::ZERO, |b| DVec2::new((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)))
                    } else {
                        (TextLevels(Default::default()), DVec2::ZERO)
                    };
                    Some(TextItem {
                        spec, run_of: Arc::new(run_of), code_points: plain.len(), clip_source: false, units: Arc::new(units), piece_of: Arc::new(piece_of), boxes: Arc::new(boxes), centre, fill: fills, stroke: strokes, stroke_width: f64::from(*stroke_width),
                        tracks, layout: accepted, degraded_paints, size,
                    })
                }
                NodeKind::Node {} | NodeKind::Path { .. } => None,
                _ => bail!("node {:?}: only group, text and path nodes are built in a 2D canvas", node.name),
            })
}

/// Each code point's fill and stroke: its run's, else the node's (a paint track's value, in `styles`, when the run
/// has none of its own), resolved in the text's box. `run_of` is each code point's run.
fn text_paints(node: &v2::Node, images: &v2::Table<v2::Image>, base: &Path,
    styles: &StyleFields, media: &crate::canvas_draw::Media,
    size: [f64; 2], run_of: &[usize]) -> Result<(Vec<PaintSource>, Vec<PaintSource>)> {
    let NodeKind::Text { content, stroke, stroke_width, fill, .. } = &node.kind else { unreachable!() };
    let here = format!("node {:?}", node.name);
    let runs = content.runs();
    let bounds = Some([0., 0., size[0], size[1]]);
    let resolve = |field: StyleField, value:Option<&v2::Paint>, node_value:&v2::Paint| -> Result<PaintSource> {
        if value.is_none() {
            if let Some(crate::animation::style::StyleValue::Paint(paint))=styles.get(field).map(|v|v.as_ref()) {
                return crate::canvas_draw::resolve_animated_paint(images,base,bounds.unwrap(),paint,media);
            }
        }
        crate::canvas_draw::resolve_paint(images,base,value.unwrap_or(node_value),bounds,&format!("{here}: {}", field.name()),media)
    };
    let paints = |field: StyleField, node_value: &v2::Paint, run_value: &dyn Fn(&v2::Run) -> Option<v2::Paint>| -> Result<Vec<PaintSource>> {
        let per_run = runs.iter().map(|run| resolve(field,run_value(run).as_ref(),node_value)).collect::<Result<Vec<_>>>()?;
        resolve(field,None,node_value)?;
        Ok(run_of.iter().map(|&run|per_run[run].clone()).collect())
    };
    let fills = paints(StyleField::Fill, fill, &|run| run.fill.clone())?;
    let strokes = if *stroke_width > 0. { paints(StyleField::Stroke, stroke, &|run| run.stroke.clone())? } else { vec![PaintSource::None; run_of.len()] };
    Ok((fills, strokes))
}

/// The text `kept` with the paints `node` has: what `build_text` makes of `node` when only its fill or stroke
/// differs from the node `kept` was made from, for text Moda's vector lane draws. Its layout, box, pieces and HTML
/// do not depend on those, so none is made again or copied (the tables are shared): the runs' colours and each
/// code point's paints are made.
fn repainted_text(node: &v2::Node, kept: &TextItem, images: &v2::Table<v2::Image>, base: &Path,
    styles: &StyleFields, media: &crate::canvas_draw::Media) -> Result<TextItem> {
    v2::Scene::validate_text(node, images)?;
    let (fill, stroke) = text_paints(node, images, base, styles, media, kept.size, &kept.run_of)?;
    let mut spec = kept.spec.clone();
    spec.run_colors = Arc::new(text_colors(&node.kind)?);
    // Every part is named, so a part added to the text is placed here as shared or as made again.
    Ok(TextItem {
        spec, fill, stroke,
        run_of: kept.run_of.clone(), units: kept.units.clone(), piece_of: kept.piece_of.clone(), boxes: kept.boxes.clone(), layout: kept.layout.clone(),
        code_points: kept.code_points, clip_source: kept.clip_source, centre: kept.centre, stroke_width: kept.stroke_width,
        tracks: kept.tracks.clone(), degraded_paints: kept.degraded_paints.clone(), size: kept.size,
    })
}

fn canvas_tracks(scene: &Scene) -> Result<Vec<(&v2::Track, TrackTarget)>> {
    let mut tracks = Vec::new();
    for (layer, targets) in scene.animation.iter().zip(scene.track_targets()?) {
        tracks.extend(layer.tracks.iter().flatten().zip(targets));
    }
    for owned in scene.node_owned_layers()? {
        tracks.extend(owned.layer.tracks.iter().flatten().zip(owned.targets));
    }
    Ok(tracks)
}

/// A canvas scene's font lists, one family each: every selected text node's and run's, authored or keyed.
fn font_lists(scene: &Scene, base: &Path, tracks: &[(&v2::Track, TrackTarget)],
              includes: &impl Fn(&str) -> bool) -> Result<Vec<Vec<PathBuf>>> {
    fn insert(lists: &mut Vec<Vec<PathBuf>>, base: &Path, font: &v2::FontFiles) {
        let list: Vec<_> = font.files().iter().map(|file| base.join(file)).collect();
        if !lists.contains(&list) { lists.push(list); }
    }
    fn collect(nodes: &[v2::Node], base: &Path, includes: &impl Fn(&str) -> bool, lists: &mut Vec<Vec<PathBuf>>) {
        for node in nodes {
            if let NodeKind::Text { font, content, .. } = &node.kind
                && includes(&node.name)
            {
                insert(lists, base, font);
                for run in content.runs().iter() {
                    if let Some(font) = &run.font { insert(lists, base, font); }
                }
            }
            collect(&node.children, base, includes, lists);
        }
    }
    let mut lists = Vec::new();
    collect(&scene.nodes, base, includes, &mut lists);
    for (track, target) in tracks {
        if let (TrackTarget::Node { node, property: v2::NodeProperty::Style(property) }, v2::Values::Inline(values)) =
            (target, &track.values)
            && includes(node)
        {
            match property.name() {
                "font" => for value in values {
                    insert(&mut lists, base, &serde_json::from_value(value.clone())?);
                },
                "content" => for value in values {
                    let content: v2::Content = serde_json::from_value(value.clone())?;
                    for run in content.runs().iter() {
                        if let Some(font) = &run.font { insert(&mut lists, base, font); }
                    }
                },
                _ => {}
            }
        }
    }
    Ok(lists)
}

/// The files a canvas reads that the scene's other sources do not list: its font files and the files of the
/// images its paints name, authored or keyed. A self-contained movie has one pinned file. A session declines a frame sequence or
/// rendered image, or tracks that do not resolve, so it reads those inputs again
/// for every render. An image no paint names is not read and is not a source.
pub(crate) fn sources(scene: &Scene, base: &Path, files: &mut BTreeSet<PathBuf>, for_export: bool, opened: &crate::footage::Opened) -> bool {
    let mut complete = true;
    // Nodes inside nested worlds' canvases are read too: more files than this document's own, never fewer.
    match canvas_tracks(scene).and_then(|tracks| font_lists(scene, base, &tracks, &|_| true)) {
        Ok(lists) => files.extend(lists.into_iter().flatten()),
        Err(_) => complete = false,
    }
    let mut painted: BTreeSet<&str> = BTreeSet::new();
    fn authored<'a>(nodes: &'a [v2::Node], painted: &mut BTreeSet<&'a str>) {
        for node in nodes {
            let paints: Vec<&v2::Paint> = match &node.kind {
                NodeKind::Path { fill, stroke, .. } => vec![fill, stroke],
                NodeKind::Text { fill, stroke, content, .. } => {
                    let mut paints = vec![fill, stroke];
                    if let v2::Content::Runs(runs) = content {
                        paints.extend(runs.iter().flat_map(|run| run.fill.iter().chain(&run.stroke)));
                    }
                    paints
                }
                _ => Vec::new(),
            };
            for paint in paints {
                if let v2::Paint::Image(binding) = paint { painted.insert(&binding.image); }
            }
            authored(&node.children, painted);
        }
    }
    authored(&scene.nodes, &mut painted);
    // A keyed paint or keyed text content names its image inside a track value: every `image` name in any
    // track value counts, which lists more than the paints' own and never fewer.
    fn keyed<'a>(value: &'a serde_json::Value, painted: &mut BTreeSet<&'a str>) {
        match value {
            serde_json::Value::Object(fields) => {
                if let Some(name) = fields.get("image").and_then(serde_json::Value::as_str) { painted.insert(name); }
                for value in fields.values() { keyed(value, painted); }
            }
            serde_json::Value::Array(values) => for value in values { keyed(value, painted); },
            _ => {}
        }
    }
    let owned = scene.node_owned_layers();
    if owned.is_err() { complete = false; }
    let owned = owned.unwrap_or_default();
    for tracks in scene.animation.iter().map(|layer| &layer.tracks).chain(owned.iter().map(|owned| &owned.layer.tracks)) {
        for track in tracks.iter().flatten() {
            if let v2::Values::Inline(values) = &track.values { for value in values { keyed(value, &mut painted); } }
        }
    }
    for name in painted {
        // An unknown name is the document's error to report.
        let Some(image) = scene.images.get(name) else { continue };
        if image.render.is_some() { complete = false; }
        let Some(file) = image.file.as_deref() else { continue };
        let path = base.join(file);
        if !for_export && crate::footage::is_sequence(&path) {
            complete = false;
        } else if crate::footage::is_sequence(&path) {
            match crate::footage::sequence(&path) {
                Ok(sequence) => files.extend(sequence),
                Err(_) => complete = false,
            }
        } else {
            if crate::footage::is_media(&path) && opened.revision(&path).is_err() {
                complete = false;
            }
            files.insert(path);
        }
    }
    complete
}

/// Whether any of `nodes` or their descendants is a 3D world (`composition: {space: "3d"}`).
pub(crate) fn has_worlds(nodes: &[v2::Node]) -> bool {
    nodes.iter().any(|node| node.composition.as_ref().is_some_and(|c| c.space == v2::Space::World) || has_worlds(&node.children))
}

impl Document {
    /// Lay out every text node of a canvas scene and compile its tracks.
    pub(crate) fn new(scene: &Scene, base: &Path) -> Result<Self> {
        Self::new_with_paints(scene, base, None, &Default::default())
    }

    /// With a selection, keep the full clock tree and prepare only the named items' paint and layout.
    pub(crate) fn new_with_paints(scene: &Scene, base: &Path, wanted: Option<&std::collections::HashSet<&str>>, opened: &crate::footage::Opened) -> Result<Self> {
        Self::prepare_document(scene, base, wanted, opened, None)
    }

    fn prepare_document(scene: &Scene, base: &Path, wanted: Option<&std::collections::HashSet<&str>>,
        opened: &crate::footage::Opened, previous: Option<(&Self, &BTreeSet<&Path>)>) -> Result<Self> {
        ensure!(scene.is_canvas(), "not a 2D canvas");
        let includes = |name: &str| wanted.is_none_or(|wanted| wanted.contains(name));
        let [w, h] = scene.render.resolution;
        let composition = scene.composition.as_ref().expect("a canvas has a composition");
        let aspect = v2::AspectRatio::parse(composition.preserve_aspect_ratio.as_deref()).map_err(anyhow::Error::msg)?;
        let (scale, offset) = aspect.placement(scene.canvas_view_box(), [w, h]);
        // The worlds' insides are documents of their own; this one holds their nodes, empty.
        let whole = scene;
        let outer = scene.without_compositions()?;
        let scene: &Scene = &outer;
        let tracks = canvas_tracks(scene)?;
        let lists = font_lists(scene, base, &tracks, &includes)?;
        let fonts = if let Some((old, _)) = previous.filter(|(old, _)| old.font_lists == lists) {
            old.fonts.clone()
        } else {
            let host = Arc::new(vork_std::moda::host(&vork_std::moda::read(&lists)?)?);
            Fonts { store: std::rc::Rc::new(host.vector_font_store_lenient()), host, layouts: Default::default() }
        };
        // Which pieces' fields each text node's tracks key, and CSS colour keys read once.
        let mut keyed = HashMap::<&str, TextTracks>::new();
        for (_, target) in &tracks {
            let TrackTarget::Text { node, unit, index, property } = target else { continue };
            let fields = keyed.entry(node.as_str()).or_default();
            fields.pieces = true;
            if matches!(property, UnitProperty::Fill | UnitProperty::Stroke) {
                let keyed = fields.paint_units[usize::from(*property == UnitProperty::Stroke)].get_or_insert_with(Default::default);
                Arc::make_mut(keyed).insert((*unit, *index));
            }
            fields.glyphs |= *unit == TextUnit::Glyphs;

        }
        let previous_media = previous.map(|(doc, files)| (doc.media.borrow(), files));
        let media = crate::canvas_draw::Media::with_readers(scene, base, wanted, opened,
            previous_media.as_ref().map(|(readers, files)| (&**readers, *files)))?;
        // The scene's nodes in paint order, each where the scene holds it, with its parent's item: a walk with
        // its own stack, so the tree's depth costs no call stack.
        let mut flat: Vec<SceneNode> = Vec::new();
        let mut open = vec![(None, scene.nodes.iter())];
        while let Some((parent, siblings)) = open.last_mut() {
            match siblings.next() {
                Some(node) => {
                    let parent = *parent;
                    open.push((Some(flat.len()), node.children.iter()));
                    flat.push(SceneNode { node, parent });
                }
                None => { open.pop(); }
            }
        }
        let mut items = Vec::with_capacity(flat.len());
        // The fields each node's tracks restyle.
        let mut restyled: HashMap<&str, Vec<StyleField>> = HashMap::new();
        for (node, field) in tracks.iter().filter_map(|(_, target)| restyled_field(target)) {
            let fields = restyled.entry(node).or_default();
            if !fields.contains(&field) { fields.push(field); }
        }
        let mut authored: Vec<Option<Box<Authored>>> = Vec::with_capacity(flat.len());
        let mut references = Vec::with_capacity(flat.len());
        let mut worlds = Vec::new();
        let mut rest: Vec<Rest> = Vec::with_capacity(flat.len());
        for &SceneNode { node, parent } in &flat {
            let selected = includes(&node.name);
            let text = if selected {
                build_text(node, &scene.images, base, &lists, &fonts, keyed.get(node.name.as_str()).cloned().unwrap_or_default(), &StyleFields::default(), &media)?
            } else { None };
            // A stroke-width track restyles a text node only.
            let fields: Vec<StyleField> = restyled.get(node.name.as_str()).into_iter().flatten().copied()
                .filter(|field| *field != StyleField::StrokeWidth || text.is_some()).collect();
            authored.push(if !selected || fields.is_empty() { None } else {
                let mut source_node=node.without_children();
                source_node.animation.clear();
                // The fields' authored values, as the node writes them: read from the written node, which is not kept.
                let mut source=serde_json::to_value(&source_node)?;
                // These computed defaults are omitted by the authoring serializer.
                if let NodeKind::Text {font_variation_settings,font_feature_settings,..}=&source_node.kind {
                    source["font-variation-settings"]=serde_json::json!(font_variation_settings);
                    source["font-feature-settings"]=serde_json::json!(font_feature_settings);
                }
                let fields = fields.into_iter().filter_map(|field| Some((field, source.get(field.name())?.clone()))).collect();
                Some(Box::new(Authored { node: source_node, fields }))
            });
            let world = node.composition.as_ref().is_some_and(|c| c.space == v2::Space::World);
            let kind = match (&node.kind, &text) {
                (_, Some(text)) => crate::canvas_draw::Kind::Text { size: text.size },
                _ if world && wanted.is_none() => {
                    // The world's picture fills the canvas's box, at the output's pixel density.
                    let [x, y, vw, vh] = whole.canvas_view_box().map(f64::from);
                    let px = [(vw * scale[0]).round().max(1.) as u32, (vh * scale[1]).round().max(1.) as u32];
                    worlds.push(WorldPart { item: items.len(), scene: whole.world_part(&node.name, px)? });
                    crate::canvas_draw::Kind::World { bounds: [x, y, x + vw, y + vh] }
                }
                (NodeKind::Path { .. }, _) if selected => crate::canvas_draw::Kind::Path(Arc::new(crate::canvas_draw::path_item(&scene.images, base, node, &media)?)),
                _ => crate::canvas_draw::Kind::Group,
            };
            let filter = match &node.filter {
                Some(filter) if selected => crate::canvas_draw::parse_filter(filter).with_context(|| format!("node {:?}", node.name))?,
                _ => Default::default(),
            };
            let draw = crate::canvas_draw::Node { parent, clip_path: None, mask: None, mask_type: node.mask_type, blend: node.mix_blend_mode, isolation: node.isolation, filter, kind };
            rest.push(Rest { parent, transform: node.transform.clone(), opacity: node.opacity, text: text.is_some(),
                path: match &draw.kind { crate::canvas_draw::Kind::Path(path) => Some(path.values), _ => None } });
            items.push(Item { name: node.name.clone(), visible: node.visibility.visible, text, node: draw });
            references.push((node.clip_path.clone(), node.mask.clone()));
        }
        let by_name: HashMap<String, usize> = items.iter().enumerate().filter(|(_, item)| !item.name.is_empty()).map(|(i, item)| (item.name.clone(), i)).collect();
        // clip-path and mask name nodes; the scene checked that they exist.
        for (i, (clip_path, mask)) in references.into_iter().enumerate() {
            items[i].node.clip_path = clip_path.and_then(|name| by_name.get(&name).copied());
            items[i].node.mask = mask.and_then(|name| by_name.get(&name).copied());
        }
        let children = Children::new(&rest);
        let mut clip_sources = vec![false; items.len()];
        let mut pending: Vec<_> = items.iter().filter_map(|item| item.node.clip_path).collect();
        for (track,target) in &tracks {
            if matches!(target,TrackTarget::Node{property:v2::NodeProperty::Style(p),..} if p.name()=="clip-path") {
                if let v2::Values::Inline(values)=&track.values {pending.extend(values.iter().filter_map(|v|v.as_str().and_then(|name|by_name.get(name).copied())));}
            }
        }
        while let Some(source) = pending.pop() {
            if std::mem::replace(&mut clip_sources[source], true) { continue; }
            pending.extend(children.of(source).iter().copied());
        }
        for (source, _) in clip_sources.into_iter().enumerate().filter(|(_,used)| *used) {
            if let Some(text) = &mut items[source].text { text.clip_source = true; }
        }
        // Static line counts are authoritative only if no track can change layout.
        // Animated layouts are checked against the sampled pieces in frame().
        let dynamic_lines: BTreeSet<_> = tracks.iter()
            .filter_map(|(_, target)| target.changes_text_pieces(TextUnit::Lines)).collect();
        for (track, target) in &tracks {
            let TrackTarget::Text { node, unit: TextUnit::Lines, index, .. } = target else { continue };
            if dynamic_lines.contains(node.as_str()) { continue; }
            let Some(text) = by_name.get(node).and_then(|&i| items[i].text.as_ref()) else { continue };
            let count = text.units[&TextUnit::Lines].len();
            ensure!(*index < count, "track {}: node {node:?} has {count} lines; index {index} is past the end", v2::display_path(&track.path));
        }
        let mut plan = crate::animation::Plan::compile_in(scene, base, None, opened)?;
        plan.mark_canvas_nodes(&mut flat)?;
        let samples = RefCell::new(plan.samples_naming_changes());
        let mut kinds: Vec<Option<ChangeKind>> = {
            // Paths own their paints; text paints are part of glyph batches.
            let path = |name: &str| by_name.get(name).and_then(|&i| match &items[i].node.kind {
                crate::canvas_draw::Kind::Path(path) if items[i].text.is_none() => Some(path),
                _ => None,
            });
            let samples = samples.borrow();
            plan.sampled(&samples).zip(plan.sampled_paint_forms()).map(|((target, sample, _, _), form)|
                ChangeKind::of(target, matches!(sample, crate::animation::Sample::Numeric(_)), form,
                    |name, fill| path(name).and_then(|path| path.paint_form(fill)), |name| path(name).is_some())).collect()
        };
        // A paint that any track can turn into another form is content for every track on it: layers mix.
        {
            let samples = samples.borrow();
            let paint_of = |target: &TrackTarget| match target {
                TrackTarget::Node { node, property: property @ (v2::NodeProperty::Fill | v2::NodeProperty::Stroke) } =>
                    Some((node.clone(), *property == v2::NodeProperty::Fill)),
                _ => None,
            };
            let unstable: BTreeSet<(String, bool)> = plan.sampled(&samples).zip(&kinds)
                .filter(|(_, kind)| **kind != Some(ChangeKind::PaintValue))
                .filter_map(|((target, ..), _)| paint_of(target)).collect();
            for ((target, ..), kind) in plan.sampled(&samples).zip(&mut kinds) {
                if paint_of(target).is_some_and(|paint| unstable.contains(&paint)) { *kind = Some(ChangeKind::Content); }
            }
        }
        let track_items: Vec<Option<usize>> = plan.sampled(&samples.borrow()).map(|(target, ..)| match target {
            TrackTarget::Node { node, .. } | TrackTarget::Text { node, .. } => by_name.get(node).copied(),
            _ => None,
        }).collect();
        let track_parts: Vec<u8> = plan.sampled(&samples.borrow()).map(|(target, ..)| match target {
            TrackTarget::Node { property: v2::NodeProperty::Translation, .. } => crate::canvas_draw::change::TRANSLATION,
            TrackTarget::Node { property: v2::NodeProperty::Rotation, .. } => crate::canvas_draw::change::ROTATION,
            TrackTarget::Node { property: v2::NodeProperty::Scale, .. } => crate::canvas_draw::change::SCALE,
            TrackTarget::Node { property: v2::NodeProperty::Opacity, .. } => crate::canvas_draw::change::OPACITY,
            TrackTarget::Node { property: v2::NodeProperty::Style(p), .. } if p.name() == "filter" => crate::canvas_draw::change::FILTER,
            _ => crate::canvas_draw::change::REST,
        }).collect();
        // The sampler names tracks by flat index, and these tables are in authored-track order: the same order
        // when every track is authored, as a canvas's are (only an imported clip's channels have no target).
        ensure!(kinds.len() == plan.track_count() && track_items.len() == plan.track_count(),
            "internal: a canvas track without an authored target");
        let mut item_tracks_first = vec![0u32; items.len() + 1];
        for item in track_items.iter().flatten() { item_tracks_first[item + 1] += 1; }
        for item in 0..items.len() { item_tracks_first[item + 1] += item_tracks_first[item]; }
        let mut item_tracks = vec![0u32; item_tracks_first[items.len()] as usize];
        {
            let mut next = item_tracks_first.clone();
            for (track, item) in track_items.iter().enumerate() {
                if let Some(item) = item {
                    item_tracks[next[*item] as usize] = track as u32;
                    next[*item] += 1;
                }
            }
        }
        let page_tracks: Vec<u32> = plan.sampled(&samples.borrow()).enumerate()
            .filter(|(_, (target, ..))| matches!(target, TrackTarget::Render(v2::RenderProperty::Background)))
            .map(|(track, _)| track as u32).collect();
        let item_marks = RefCell::new(vec![0; items.len()]);
        let background = match &scene.render.background {
            v2::Background::Css(color) => Some(vork_std::moda::css_color(color).context("render.background")?),
            _ => None,
        };
        let working = composition.working_space();
        let page = crate::canvas_draw::Page { size: [w, h], scale, offset, background, working: crate::canvas_draw::working_color(working) };
        let text_specs = RefCell::new((0..items.len()).map(|_| None).collect());
        let styles = RefCell::new((0..items.len()).map(|_| None).collect());
        // The precision rule: the canvas holds a world, it authors a view or an image graph, or it blends in
        // linear light.
        let float = !worlds.is_empty() || composition.view.is_some() || !composition.compositor.is_empty() || working.is_linear();
        let graph_images = whole.compositor_image_reads(whole.compositor()).into_iter()
            .filter_map(|name| whole.images.get(&name).map(|image| image.time.clone()).map(|time| (name, time))).collect();
        let prepared = Arc::new(Prepared {
            opened: opened.clone(),
            view: (float || working != v2::WorkingColorSpace::Srgb).then(|| whole.clone()),
            float, working, graph_images, images: scene.images.clone(), base: base.into(),
            font_lists: lists, authored, items, rest,
            children, by_name, plan, page, worlds, kinds, track_items, track_parts,
            item_tracks_first, item_tracks, page_tracks,
        });
        Ok(Self { prepared, fonts, samples, styles, media: RefCell::new(media),
            retained: Default::default(), text_specs, evaluated: Default::default(),
            stale: std::cell::Cell::new(true), text: Default::default(), changes: Default::default(), item_marks,
            #[cfg(test)]
            id: Self::next_id(),
            #[cfg(test)]
            snapshots: Default::default(),
        })
    }
    #[cfg(test)]
    fn next_id() -> u64 {
        static IDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        IDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
    /// Item `i` as authored. Only an item a track restyles is read this way, and `new` kept exactly those.
    fn authored(&self, i: usize) -> Result<&Authored> {
        self.authored[i].as_deref().with_context(|| format!("internal: node {:?} is restyled by a track and was not kept as authored", self.items[i].name))
    }
    /// Read static images once into Moda's asset store.
    pub(crate) fn install_images(&self) -> Result<()> {
        let nodes: Vec<&crate::canvas_draw::Node> = self.items.iter().map(|item| &item.node).collect();
        let paints: Vec<_> = self.items.iter().filter_map(|item| item.text.as_ref())
            .flat_map(|text| text.fill.iter().chain(&text.stroke)).collect();
        crate::canvas_draw::install_images(&nodes, &paints)
    }
    /// Whether the canvas keeps a float picture (RGBA16F) instead of the 8-bit one a browser's canvas keeps:
    /// only when something in it needs the precision. It holds a world, whose viewed picture arrives floating
    /// point; it authors a view or an image graph (`composition.compositor`), which read and return float
    /// colour; or it blends in linear light (`Linear Rec.709`, `Linear Rec.2020`, `ACEScg`), whose dark
    /// values eight bits cannot hold. An encoded space (sRGB, Display P3) holds its code values in eight bits.
    /// A float picture is rounded once, where it reaches an 8-bit destination; an 8-bit picture is copied as
    /// it is.
    pub(crate) fn float(&self) -> bool {
        self.float
    }
    /// The encoding the canvas blends in.
    pub(crate) fn working(&self) -> v2::WorkingColorSpace {
        self.working
    }
    /// The canvas's picture at `request` as the world that reads it takes it: associated sRGB code values,
    /// Rgba8Unorm as drawn or, where the picture is not that already (a float canvas, an authored view,
    /// another working encoding), Rgba16Float with its image graph and its view applied
    /// (`Pictures::reader_stage`). `errors`: the shader diagnostics of the frame the picture is drawn for, where its graph and its
    /// view record a result they cannot use; that frame refuses it where it is collected. Without them each
    /// one's own error word is read back here, which waits for the GPU.
    pub(crate) fn draw(&self, engine: &crate::canvas_draw::Engine, worlds: &Pictures, renderer: &Renderer, request: &vork_time::Request,
        errors: Option<&wgpu::Buffer>) -> Result<wgpu::Texture> {
        let frame = self.frame(request)?;
        let (drawn, _) = worlds.draw(renderer, self, &frame, None, engine, errors, None, || {})?;
        if let (Some(graph), None) = (&worlds.graph, errors) { graph.check(renderer)?; }
        let Some(view) = &worlds.view else { return Ok(drawn) };
        // Each draw's viewed picture is a texture of its own: its readers are recorded in commands submitted
        // later (a world's shutter samples, a frame's finish), after further draws.
        let viewed = crate::gpu_memory::texture(&renderer.device, &wgpu::TextureDescriptor {
            label: Some("canvas viewed picture"), size: drawn.size(), mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2, format: view.format(renderer),
            // COPY_SRC: tests read the picture back.
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
        });
        let mut encoder = renderer.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("canvas view") });
        view.encode(renderer, &mut encoder, &drawn, &viewed, &frame.request, errors)?;
        #[cfg(feature = "frame-profile")]
        crate::frame_profile::count("queue_submissions", 1);
        renderer.queue.submit([encoder.finish()]);
        if errors.is_none() { view.check(renderer)?; }
        Ok(viewed)
    }

    /// Frames evaluated so far, by what changed since the frame before.
    pub(crate) fn change_counters(&self) -> serde_json::Value {
        let [unchanged, placement, paint_value, content, structure] = self.changes.get();
        serde_json::json!({"unchanged": unchanged, "placement": placement, "paint_value": paint_value,
            "content": content, "structure": structure})
    }

    pub(crate) fn measure_image_time(
        &self,
        name: &str,
        image: &v2::Image,
        request: &vork_time::Request,
    ) -> Result<vork_time::exact::Germ> {
        let mut samples = self.plan.samples();
        self.plan.sample_request_into(request.numeric()?, request, &mut samples)?;
        let scene_time = request.exact()?;
        self.plan.footage_time(&samples, &scene_time, name, image.time.as_ref())?
            .with_context(|| format!("ask: image {name:?} is absent at this time"))
    }

    pub(crate) fn bounds_batch<'a>(&self, names: impl Iterator<Item = &'a str>) -> BoundsBatch {
        BoundsBatch { wanted: names.filter_map(|name| self.by_name.get(name).copied()).collect(), measured: None }
    }

    pub(crate) fn measure(
        &self,
        name: &str,
        kind: &str,
        read: &str,
        request: &vork_time::Request,
        bounds: &mut BoundsBatch,
    ) -> Result<serde_json::Value> {
        use serde_json::{Value, json};
        let &i = self.by_name.get(name)
            .with_context(|| format!("ask: unknown canvas node {name:?}"))?;
        if kind == "node" && read != "transform"
            && let Some(cached) = bounds.measured.as_ref().filter(|b| b.time == *request) {
            return cached.answer(self, i, read);
        }
        let effect;
        let frame = self.frame(request)?;
        {
            let evaluated = self.evaluated.borrow();
            let state = &evaluated.as_ref().unwrap().state;
            let node = state.nodes[i].as_deref().unwrap_or(&self.items[i].node);
            match kind {
                "path" => {
                    let crate::canvas_draw::Kind::Path(path) = &node.kind else {
                        bail!("ask: {name:?} must be a path")
                    };
                    return Ok(if state.shown[i] {
                        json!(path.measured_length())
                    } else {
                        Value::Null
                    });
                }
                "text" => {
                    let styles = self.styles.borrow();
                    let text = state.nodes[i]
                        .as_ref()
                        .and(styles[i].as_ref())
                        .and_then(|style| style.text.as_deref())
                        .or(self.items[i].text.as_ref())
                        .with_context(|| format!("ask: {name:?} must be text"))?;
                    if !state.shown[i] {
                        return Ok(Value::Null);
                    }
                    let laid = &text.layout;
                    let lines: Vec<_> = laid
                        .metrics
                        .iter()
                        .map(
                            |[advance, baseline]| json!({ "advance":advance, "baseline":baseline }),
                        )
                        .collect();
                    return Ok(
                        json!({"box":[[0.,0.],[laid.width,laid.height]], "line_count":lines.len(), "lines":lines}),
                    );
                }
                "node" if read == "transform" => {
                    if !state.shown[i] {
                        return Ok(Value::Null);
                    }
                    let chain: Vec<_> = std::iter::successors(Some(i), |&n| self.rest[n].parent)
                        .map(|n| &state.values[n].transform)
                        .collect();
                    return Ok(json!(crate::canvas_draw::placed_matrix(
                        &self.page, &chain
                    )?));
                }
                "node" => {
                    // These operations determine their silhouette from pixels. A plan's
                    // capture allocation or mask box cannot answer geometric bounds.
                    let mut ancestors = vec![None; self.items.len()];
                    let mut subtree = vec![None; self.items.len()];
                    for (j, item) in self.items.iter().enumerate() {
                        let node = state.nodes[j].as_deref().unwrap_or(&item.node);
                        if state.shown[j] && (node.clip_path.is_some() || node.mask.is_some()
                            || !node.filter.0.is_empty()) {
                            subtree[j] = Some(j);
                        }
                        ancestors[j] = self.rest[j].parent.and_then(|p| ancestors[p]).or(subtree[j]);
                    }
                    // Preorder indices preserve the first offending node in the original walk.
                    for j in (0..subtree.len()).rev() {
                        if let (Some(p), Some(child)) = (self.rest[j].parent, subtree[j]) {
                            subtree[p] = Some(subtree[p].map_or(child, |v| v.min(child)));
                        }
                    }
                    for (first, ancestor) in subtree.iter_mut().zip(ancestors) {
                        *first = ancestor.or(*first);
                    }
                    if let Some(j) = subtree[i] {
                        bail!("ask: bounds depend on mask, clip or filter on {:?}; remove it to measure the geometry", self.items[j].name);
                    }
                    effect = subtree;
                }
                _ => bail!("ask: {kind} is unavailable on a canvas"),
            }
        }
        // The world image is a placed rectangle. Planning its geometry only requires
        // the dimensions the renderer would register for the completed picture.
        let mut captures = Vec::new();
        let pictures = self
            .worlds
            .iter()
            .zip(&frame.worlds)
            .enumerate()
            .map(|(i, (world, time))| {
                time.as_ref()?;
                let src = format!("{}gpu/ask/{i}", moda_engine::effects::PIXEL_SINK_SRC_PREFIX);
                let size = world.scene.render.resolution;
                captures.push((src.clone(), size, 0));
                Some((src, size))
            })
            .collect::<Vec<_>>();
        if !captures.is_empty() {
            crate::canvas_draw::install_captures(&captures);
        }
        let plan = self.plan(&pictures)?;
        let retained = self.retained.borrow();
        let boxes = retained.as_ref().unwrap().measured_boxes(&plan, self.items.len(), &bounds.wanted);
        bounds.measured.insert(MeasuredBounds { time: request.clone(), boxes, effect }).answer(self, i, read)
    }

    /// One instant: every node's animated values (transform, visibility, opacity, a path's keyed fields), and the
    /// batches of each text node in paint order: its Moda specs with its pieces' motion and paint.
    ///
    /// The document's samples and its evaluated table move together: the sampler reports the tracks that changed
    /// since its last call, and that is what differs from the table. A frame that fails after sampling leaves no
    /// table, so the next one is evaluated whole. The values are written into the table; the frame returned is the
    /// instant and its worlds' clocks.
    fn frame(&self, request: &vork_time::Request) -> Result<Frame> {
        let _profile = moda_engine::prof::span("canvas.evaluation");
        let frame = self.evaluate(request).and_then(|(frame, changed)| {
            let bindings_changed = self.publish_frame_media(request, MediaPublication::Frame)?;
            // A shared paint can appear or disappear on a matte whose own tracks stood still.
            self.write_rows(if bindings_changed { &Changed::Every } else { &changed })?;
            Ok(frame)
        });
        if frame.is_err() { self.evaluated.borrow_mut().take(); }
        frame
    }

    /// Publish the media read by the evaluated rows before planning their picture.
    fn publish_frame_media(&self, request: &vork_time::Request, publication: MediaPublication) -> Result<bool> {
        let mut media = self.media.borrow_mut();
        if media.is_empty() { return Ok(false); }
        let evaluated = self.evaluated.borrow();
        let state = &evaluated.as_ref().expect("the frame's table").state;
        let collect = |items: &[usize], drawing_only: bool| {
            let mut paints = std::collections::HashSet::new();
            for &i in items {
                if state.shown[i] && (!drawing_only || state.drawing[i]) {
                    let node = state.nodes[i].as_deref().unwrap_or(&self.items[i].node);
                    visit_image_paints(node, &state.values[i], |src| { paints.insert(src.to_owned()); });
                }
            }
            paints
        };
        let mut paints = collect(media.reader_items(), true);
        if publication == MediaPublication::Missing {
            // Only a painted mask needs its entry at open. A collapsed mask may share
            // the image of a drawing row elsewhere in the document.
            let retained = self.retained.borrow();
            let masks = collect(retained.as_ref().expect("accepted rows").mask_paint_items(), false);
            paints.retain(|src| masks.contains(src));
        }
        media.update(&self.plan, &self.samples.borrow(), &request.exact()?, publication, |src| paints.contains(src))
    }

    /// Write what a frame evaluated into the installed scene, from the evaluated table: the frame's values are in
    /// the scene when it returns, and the scene's stamps are the account of what changed. A frame that evaluated
    /// some items writes their rows; one that evaluated every item, or changed something no item names, or an
    /// item's form, brings the scene whole (`Retained::update`).
    ///
    /// Two cases wait for the plan, and leave `stale` set for it: there is no scene yet (the first install), and a
    /// whole write of a canvas with worlds. A world's picture is drawn after the frame is evaluated and is part of
    /// how its item is lowered, so lowering before the plan has it would lower the canvas twice.
    fn write_rows(&self, changed: &Changed) -> Result<()> {
        let items = match changed {
            Changed::Nothing => return Ok(()),
            Changed::Items(items, true) => Some(&items[..]),
            Changed::Items(_, false) | Changed::Every => None,
        };
        let mut retained = self.retained.borrow_mut();
        // Until the scene is written it is owed everything: a failure below, or a wait for the plan, leaves it so.
        let stale = self.stale.replace(true);
        let Some(scene) = retained.as_mut() else { return Ok(()) };
        let items = items.filter(|_| !stale);
        if items.is_none() && !self.worlds.is_empty() { return Ok(()); }
        let _profile = moda_engine::prof::span("canvas.rows");
        let evaluated = self.evaluated.borrow();
        let state = &evaluated.as_ref().expect("the frame's table").state;
        let mut page = self.page.clone();
        page.background = state.background;
        let node = |i: usize| state.nodes[i].as_deref().unwrap_or(&self.items[i].node);
        let written = match items {
            Some(items) => scene.write(&page, node, &state.values, items)?,
            None => false,
        };
        if !written {
            if !self.worlds.is_empty() { return Ok(()); }
            let nodes: Vec<&crate::canvas_draw::Node> = (0..self.items.len()).map(node).collect();
            scene.update(&page, &nodes, &state.values)?;
        }
        self.stale.set(false);
        Ok(())
    }

    /// Evaluate into the table. Returns the frame and what of the table it changed.
    fn evaluate(&self, request: &vork_time::Request) -> Result<(Frame, Changed)> {
        let _frame_profile = moda_engine::prof::span("canvas.evaluate");
        let time = request.numeric()?;
        let mut samples = self.samples.borrow_mut();
        {
            let _sample = moda_engine::prof::span("canvas.evaluate.sample");
            self.plan.sample_request_into(time, request, &mut samples)?;
        }
        let samples = &*samples;
        // The tracks whose sample or weight changed, control tracks included, are the frame's account of what
        // changed: none means the evaluated state stands (the reuse test), and the frame's kind is the largest
        // kind among those that draw; `None` when none does. Continuously advancing clocks are not among them.
        let changed_tracks = samples.changed();
        let (same, change) = match self.evaluated.borrow().as_ref() {
            None => (false, Some(ChangeKind::Structure)),
            Some(old) => {
                let kind = changed_tracks.iter().map(|&track| self.kinds[track as usize]).max().flatten();
                // A node arriving or leaving on its clock changes which nodes draw.
                let same_presence = !self.plan.has_clocks() || (old.present.len() == samples.clocks.len()
                    && old.present.iter().zip(&samples.clocks).all(|(a,b)| *a == b.is_some()));
                if !same_presence || kind >= Some(ChangeKind::Content) {
                    crate::canvas_host_memory::require_serial_growth()?;
                }
                if same_presence { (changed_tracks.is_empty(), kind) } else { (false, Some(ChangeKind::Structure)) }
            }
        };
        let mut counts = self.changes.get();
        counts[change.map_or(0, |kind| kind as usize + 1)] += 1;
        self.changes.set(counts);
        let scene_instant = request.exact()?;
        // The image graph's images follow scene time (a movie shown under a node's clock is not also one of
        // them: `validate_media_clocks`), as a world's compositor images do.
        let images = self.graph_images.iter().map(|(name, time)| {
            let base = time.as_ref().map(|time| vork_time::exact::Germ::point(time.exact())).unwrap_or_else(|| scene_instant.clone());
            Ok((name.clone(), self.plan.image_time_exact(samples, &scene_instant, name, base)?))
        }).collect::<Result<Vec<_>>>()?;
        // World clocks advance even when the canvas around them stands.
        let worlds = |shown: &[bool]| -> Vec<Option<vork_time::Request>> {
            self.worlds.iter().map(|world| shown[world.item]
                .then(|| self.plan.local_instant(samples, world.item, &scene_instant)).flatten().map(vork_time::Request::Exact)).collect()
        };
        if same {
            // The table stands, and the scene is owed nothing new.
            let evaluated = self.evaluated.borrow();
            let state = &evaluated.as_ref().expect("the state that stands").state;
            return Ok((Frame { request: request.clone(), worlds: worlds(&state.shown), images }, Changed::Nothing));
        }
        // Do not publish a partially evaluated state after an error.
        let previous = self.evaluated.borrow_mut().take();
        let mut styled = self.styles.borrow_mut();
        let mut specs = self.text_specs.borrow_mut();
        // Which nodes draw, and the page, stand until a track of that kind changes.
        let (background, shown, present) = match previous {
            Some(mut old) if change < Some(ChangeKind::Structure) => {
                let _profile = moda_engine::prof::span("canvas.evaluate.changed");
                // What changed, named by the tracks that changed: each one's item, evaluated again in place.
                let (mut items, bound) = self.changed_items(samples);
                if let Some(same_draws) = self.evaluate_changed(samples, &mut old.state, &mut items, &mut styled, &mut specs)? {
                    #[cfg(debug_assertions)]
                    self.assert_whole(samples, &old.state, &old.present, &mut styled, &mut specs)?;
                    let worlds = worlds(&old.state.shown);
                    if !same_draws { self.text.borrow_mut().take(); }
                    *self.evaluated.borrow_mut() = Some(old);
                    return Ok((Frame { request: request.clone(), worlds, images }, Changed::Items(items, bound)));
                }
                // A scale reached or left zero: which nodes draw changed after all, and every item is evaluated.
                crate::canvas_host_memory::require_serial_growth()?;
                (old.state.background, std::mem::take(&mut old.state.shown), old.present)
            }
            previous => {
                let mut present = previous.map(|old| old.present).unwrap_or_default();
                present.clear();
                if self.plan.has_clocks() { present.extend(samples.clocks.iter().map(Option::is_some)); }
                let shown = self.shown(samples);
                (self.background(samples), shown, present)
            }
        };
        let _nodes_profile = moda_engine::prof::span("canvas.evaluate.nodes");
        let count = self.items.len();
        let mut state = FrameState { background, nodes: Vec::with_capacity(count), values: Vec::with_capacity(count), draws: TextDrawRows::new(count),
            shown, drawing: Vec::with_capacity(count) };
        for i in 0..count {
            // Items are parents first.
            let parent_drawing = self.rest[i].parent.is_none_or(|p| state.drawing[p]);
            state.nodes.push(None);
            state.values.push(self.unevaluated(i));
            let mut draws = Vec::new();
            let drawing = self.evaluate_item(samples, i, state.shown[i], parent_drawing, &mut styled, &mut specs,
                &mut state.nodes[i], &mut state.values[i], &mut draws)?;
            if !draws.is_empty() { state.draws.replace(i, &mut draws); }
            state.drawing.push(drawing);
        }
        self.check_references(&state.nodes)?;
        let worlds = worlds(&state.shown);
        // Every item was evaluated: the text builder is owed its batches.
        self.text.borrow_mut().take();
        *self.evaluated.borrow_mut() = Some(Evaluated { present, state });
        Ok((Frame { request: request.clone(), worlds, images }, Changed::Every))
    }

    /// The values an item's slot holds before its first evaluation writes them.
    fn unevaluated(&self, i: usize) -> crate::canvas_draw::Values {
        crate::canvas_draw::Values { style_revision: 0, transform: self.rest[i].transform.clone(), visible: false, opacity: 0., path: None, texts: 0,
            text_composition: Vec::new(), text_revision: 0, text_paint_revisions: Default::default(), world: None }
    }

    /// Item `i`'s tracks, ascending: the order they blend in.
    fn tracks_of(&self, i: usize) -> &[u32] {
        &self.item_tracks[self.item_tracks_first[i] as usize..self.item_tracks_first[i + 1] as usize]
    }

    /// The page's background at this sample.
    fn background(&self, samples: &crate::animation::Samples) -> Paint {
        let mut background = self.page.background;
        for &track in &self.page_tracks {
            let Some((_, crate::animation::Sample::Style(value), weight, _)) = self.plan.sampled_at(samples, track as usize) else { continue };
            if weight == 0. { continue; }
            let crate::animation::style::StyleValue::Paint(crate::animation::style::Paint::Color(color)) = value.as_ref() else { unreachable!("canvas backgrounds key CSS colours") };
            background = Some(crate::animation::style::color_mix(background.unwrap_or([0.; 4]), *color, weight));
        }
        background
    }

    /// Which nodes draw at this sample: each item's visibility through its visibility tracks, false when the node
    /// is absent on its clock (§8 rule 1) or its parent does not draw. Items are parents first.
    fn shown(&self, samples: &crate::animation::Samples) -> Vec<bool> {
        let mut shown: Vec<bool> = Vec::with_capacity(self.items.len());
        for (i, item) in self.items.iter().enumerate() {
            let mut visible = item.visible;
            for &track in self.tracks_of(i) {
                if self.kinds[track as usize] != Some(ChangeKind::Structure) { continue; }
                if let Some((TrackTarget::Node { property: v2::NodeProperty::Visibility(v2::Ray::All), .. }, crate::animation::Sample::Bool(value), weight, additive)) = self.plan.sampled_at(samples, track as usize) {
                    if weight >= 0.5 && !additive { visible = *value; }
                }
            }
            let draws = visible && !self.plan.absent(samples, i) && self.rest[i].parent.is_none_or(|p| shown[p]);
            shown.push(draws);
        }
        shown
    }

    /// The items the tracks the sampler reports changed target, each once, parents before children, with what of
    /// it those tracks change (`track_parts`); and whether every one of those tracks draws through an item of
    /// this canvas.
    fn changed_items(&self, samples: &crate::animation::Samples) -> (Vec<(usize, u8)>, bool) {
        let tracks = samples.changed();
        let mut marks = self.item_marks.borrow_mut();
        let mut items: Vec<(usize, u8)> = Vec::with_capacity(tracks.len());
        let mut bound = true;
        // Items are parents first, so ascending is parents first; tracks authored in node order name them so.
        let mut ascending = true;
        for &track in tracks {
            let i = track as usize;
            // A control track draws nothing itself: the tracks it drives changed weight, and are named themselves.
            if self.kinds.get(i).copied().flatten().is_none() { continue; }
            let Some(item) = self.track_items.get(i).copied().flatten() else { bound = false; continue };
            let what = self.track_parts[i];
            if marks[item] == 0 {
                ascending &= items.last().is_none_or(|&(last, _)| last < item);
                items.push((item, what));
                marks[item] = items.len() as u32;
            } else {
                items[marks[item] as usize - 1].1 |= what;
            }
        }
        for (item, _) in &items { marks[*item] = 0; }
        if !ascending { items.sort_unstable_by_key(|&(item, _)| item); }
        (items, bound)
    }

    /// Bring `state` to this sample by evaluating `items` alone, parents before children: each one's node and values
    /// are written in place, its text batches replace the ones the state held when they differ, and no other item
    /// is visited. An item reads whether its
    /// parent draws from the state, so a changed parent is brought to this sample before its changed child. Returns whether the state still draws the text
    /// batches it drew; `None` when an item's scale reached or left zero, which changes what draws below it: the
    /// caller evaluates every item.
    fn evaluate_changed(&self, samples: &crate::animation::Samples, state: &mut FrameState, items: &mut [(usize, u8)],
        styled: &mut [Option<StyledItem>], specs: &mut [Option<TextSpecs>]) -> Result<Option<bool>> {
        let zero = |t: &v2::Transform| t.scale[0] == 0. || t.scale[1] == 0.;
        let mut draws: Vec<Draw> = Vec::new();
        let mut same_draws = true;
        use crate::canvas_draw::change;
        // Placement tracks only affect these local columns. A zero-scale transition is the
        // exception: it changes descendant drawing and requires a full evaluation.
        // Ranges are disjoint because changed_items sorts and coalesces row indices.
        let plan = &self.plan;
        let (first, tracks, parts, rest) = (&self.item_tracks_first, &self.item_tracks, &self.track_parts, &self.rest);
        let shown = &state.shown;
        let holds = moda_scene::rows::visit_changed(&mut state.values, items, |edit| edit.0, |v, edit| {
            let (i, what) = *edit;
            if what & (change::REST | change::FILTER) != 0 || !shown[i] { return true; }
            let was_zero = zero(&v.transform);
            let mut transform = rest[i].transform.clone();
            let mut opacity = rest[i].opacity;
            for &track in &tracks[first[i] as usize..first[i + 1] as usize] {
                let part = parts[track as usize];
                if part & what == 0 { continue; }
                if let Some((_, crate::animation::Sample::Numeric(value), weight, additive)) = plan.sampled_at(samples, track as usize) {
                    if weight != 0. { blend_placement(part, &mut transform, &mut opacity, value, weight, additive); }
                }
            }
            if what & change::TRANSLATION != 0 { v.transform.translation = transform.translation; }
            if what & change::SCALE != 0 { v.transform.scale = transform.scale; }
            if what & change::ROTATION != 0 { v.transform.rotation = transform.rotation; }
            if what & change::OPACITY != 0 { v.opacity = opacity; }
            if v.transform.translation[2] != 0. || v.transform.scale[2] != 1. ||
                (what & change::ROTATION != 0 && {
                    let q = Quat::from_array(v.transform.resolved_rotation()).normalize();
                    !(q.x.abs() < 1e-5 && q.y.abs() < 1e-5)
                }) {
                // The full evaluator reports the authored node and invalid coordinate.
                return false;
            }
            if zero(&v.transform) != was_zero { return false; }
            true
        });
        if !holds { return Ok(None); }
        for &(i, what) in items.iter() {
            if what & (change::REST | change::FILTER) == 0 { continue; }
            draws.clear();
            let parent_drawing = self.rest[i].parent.is_none_or(|p| state.drawing[p]);
            let was_zero = zero(&state.values[i].transform);
            #[cfg(debug_assertions)]
            let (old_node, old) = (state.nodes[i].clone(), state.values[i].clone());
            // The item's values are written where the state holds them.
            let drawing = self.evaluate_item(samples, i, state.shown[i], parent_drawing, styled, specs,
                &mut state.nodes[i], &mut state.values[i], &mut draws)?;
            if zero(&state.values[i].transform) != was_zero { return Ok(None); }
            // The tracks' account names what of the item changed: the scene writes the named parts alone.
            #[cfg(debug_assertions)]
            {
                use crate::canvas_draw::change;
                let values = &state.values[i];
                let same_node = match (&old_node, &state.nodes[i]) {
                    (None, None) => true,
                    (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                    _ => false,
                };
                let rest = !same_node || old.style_revision != values.style_revision || old.visible != values.visible || old.path != values.path
                    || old.texts != values.texts || old.text_revision != values.text_revision || old.text_paint_revisions != values.text_paint_revisions || old.text_composition != values.text_composition;
                debug_assert!(drawing == state.drawing[i] && (old.transform == values.transform || what & change::TRANSFORM != 0)
                    && (old.opacity == values.opacity || what & change::OPACITY != 0) && (!rest || what & (change::REST | change::FILTER) != 0),
                    "item {:?} changed in a way its changed tracks do not name (named {what:#b})", self.items[i].name);
            }
            let _ = (what, drawing);
            // Only a text node has batches.
            let same = !self.rest[i].text || (state.draws[i].len() == draws.len() && state.draws[i].iter().zip(&draws)
                .all(|(a, b)| Arc::ptr_eq(&a.spec, &b.spec) && Arc::ptr_eq(&a.layout, &b.layout)));
            if !same {
                // The item owns its batches: replacing them moves no other item's.
                state.draws.replace(i, &mut draws);
                same_draws = false;
            }
        }
        Ok(Some(same_draws))
    }

    /// The test of a state evaluated in place: it equals a state evaluated whole at this sample. Nothing of the
    /// state is taken on trust: which nodes draw, the background and the clocks' presence are made again from the
    /// samples, and every item is evaluated from those. The style and text caches are shared with the evaluation
    /// under test, and cannot hide a stale item: they are keyed by the item's sampled fields, so an item whose
    /// samples differ from the ones its kept node or batches were made from is made again, as a new node or new
    /// batches, and differs here.
    #[cfg(debug_assertions)]
    fn assert_whole(&self, samples: &crate::animation::Samples, state: &FrameState, present: &[bool],
        styled: &mut [Option<StyledItem>], specs: &mut [Option<TextSpecs>]) -> Result<()> {
        debug_assert!(!self.plan.has_clocks() || (present.len() == samples.clocks.len()
            && present.iter().zip(&samples.clocks).all(|(a, b)| *a == b.is_some())), "a node arrived or left on its clock");
        debug_assert!(state.background == self.background(samples), "the background changed");
        let shown = self.shown(samples);
        debug_assert!(shown == state.shown, "which nodes draw changed");
        let mut all_drawing: Vec<bool> = Vec::with_capacity(self.items.len());
        let mut draws: Vec<Draw> = Vec::new();
        for i in 0..self.items.len() {
            draws.clear();
            let parent_drawing = self.rest[i].parent.is_none_or(|p| all_drawing[p]);
            let (mut node, mut values) = (None, self.unevaluated(i));
            // A world's picture is given by a plan and kept in the table; no evaluation makes or changes it.
            values.world = state.values[i].world.clone();
            let drawing = self.evaluate_item(samples, i, shown[i], parent_drawing, styled, specs, &mut node, &mut values, &mut draws)?;
            all_drawing.push(drawing);
            let same_node = match (&state.nodes[i], &node) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            };
            let same_draws = draws.len() == state.draws[i].len() && draws.iter().zip(&state.draws[i])
                .all(|(a, b)| a.item == b.item && a.sub == b.sub && Arc::ptr_eq(&a.spec, &b.spec) && Arc::ptr_eq(&a.layout, &b.layout));
            debug_assert!(same_node && same_draws && values == state.values[i] && drawing == state.drawing[i],
                "item {:?} differs from its evaluation at this sample: no changed track names it", self.items[i].name);
        }
        Ok(())
    }

    /// One item at this sample, written where the caller holds it: `node` (its node when style tracks restyle
    /// it), `values` (its animated values) and, appended to `draws`, its text batches. Returns whether it draws
    /// (its scale and every ancestor's is not zero). The item's own tracks blend from its rest values in track
    /// order. It reads no other item's values: `shown` (whether the node draws at all) and `parent_drawing` are
    /// the caller's, from the evaluated state. Every frame evaluates an item through this function: every item on
    /// the first frame and when which nodes draw changed, else the items the changed tracks name. Nothing is
    /// built for what an item does not have: an item that only moves reads its rest values and its tracks, and
    /// writes its transform and opacity.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_item(&self, samples: &crate::animation::Samples, i: usize, shown: bool, parent_drawing: bool,
        styled: &mut [Option<StyledItem>], specs: &mut [Option<TextSpecs>],
        node: &mut Option<Arc<crate::canvas_draw::Node>>, values: &mut crate::canvas_draw::Values, draws: &mut Vec<Draw>) -> Result<bool> {
        let item = &self.items[i];
        let at_rest = &self.rest[i];
        let mut transform = at_rest.transform.clone();
        let mut opacity = at_rest.opacity;
        let mut path = at_rest.path;
        // Made when a track writes one.
        let mut fields: Option<StyleFields> = None;
        let mut units: Option<UnitValues> = None;
        for &track in self.tracks_of(i) {
            let Some((target, sample, weight, additive)) = self.plan.sampled_at(samples, track as usize) else { continue };
            if weight == 0. { continue; }
            match (target, sample) {
                (TrackTarget::Node { property, .. }, crate::animation::Sample::Style(value)) => {
                    let key = match property { v2::NodeProperty::Style(p) => StyleField::Property(*p), v2::NodeProperty::Fill => StyleField::Fill, v2::NodeProperty::Stroke => StyleField::Stroke, _ => unreachable!() };
                    if key == REFERENCE_FIELDS.filter && weight >= 1. {
                        fields.get_or_insert_with(StyleFields::default).set(key, value.clone());
                        continue;
                    }
                    let field = key.name();
                    let kind=target.style_interpolation().unwrap();
                    let rest=fields.as_ref().and_then(|f|f.get(key)).cloned().map(Ok).unwrap_or_else(|| {
                        let mut value=self.authored(i)?.field(key).clone();
                        if field=="filter" && value.is_null() {value=serde_json::json!("none");}
                        crate::animation::style::StyleValue::parse(kind,&value).map(Arc::new)
                    })?;
                    fields.get_or_insert_with(StyleFields::default).set(key,Arc::new(rest.mix(value,weight,kind)));
                }
                (TrackTarget::Text { unit, index, property, .. }, crate::animation::Sample::Style(value)) => {
                    let piece=units.get_or_insert_with(UnitValues::default).write((*unit, *index));
                    if let crate::animation::style::StyleValue::Origin(origin)=value.as_ref() {
                        let rest=piece.origin.unwrap_or([(0.5,0.);2]);
                        piece.origin=Some(std::array::from_fn(|i|(rest[i].0+(origin[i].0-rest[i].0)*weight,rest[i].1+(origin[i].1-rest[i].1)*weight)));
                        continue;
                    }
                    let slot=match property {UnitProperty::Fill=>&mut piece.fill,UnitProperty::Stroke=>&mut piece.stroke,_=>unreachable!()};
                    slot.push(UnitPaint::Css(value.clone(),weight));
                }
                (TrackTarget::Node { property: property @ (v2::NodeProperty::Fill | v2::NodeProperty::Stroke), .. }, crate::animation::Sample::Numeric(value)) => {
                    let key = if *property == v2::NodeProperty::Fill { StyleField::Fill } else { StyleField::Stroke };
                    let field = key.name();
                    let rest=match fields.as_ref().and_then(|f|f.get(key)) {Some(v)=>v.json(),None=>self.authored(i)?.field(key).clone()};
                    let mut color=rest.as_str().and_then(|v|paint(field,v).ok().flatten()).unwrap_or([0.;4]);
                    crate::animation::blend(&mut color,value,weight,additive,crate::animation::BlendKind::Add);
                    fields.get_or_insert_with(StyleFields::default).set(key,Arc::new(crate::animation::style::StyleValue::Paint(crate::animation::style::Paint::Color(color.map(|v|v.clamp(0.,1.))))));
                }
                (TrackTarget::Node { node, property }, crate::animation::Sample::Numeric(value)) => {
                    let t = &mut transform;
                    let add = crate::animation::BlendKind::Add;
                    match property {
                        v2::NodeProperty::Style(p) => {
                            use crate::animation::style::StyleValue;
                            let key = StyleField::Property(*p);
                            let field = key.name();
                            let rest=fields.as_ref().and_then(|f|f.get(key)).cloned().map(Ok).unwrap_or_else(||StyleValue::parse(p.interpolation(),self.authored(i)?.field(key)).map(Arc::new))?;
                            let mut number=value[0];
                            let sample=if additive {
                                let base=rest.json().as_f64().with_context(||format!("node {node:?}: {field} cannot combine from a keyword"))? as f32;
                                number=base+number*weight;StyleValue::Json(serde_json::json!(number))
                            } else {rest.mix(&StyleValue::Json(serde_json::json!(number)),weight,p.interpolation())};
                            fields.get_or_insert_with(StyleFields::default).set(key,Arc::new(sample));
                        }
                        v2::NodeProperty::Translation | v2::NodeProperty::Scale | v2::NodeProperty::Rotation | v2::NodeProperty::Opacity =>
                            blend_placement(self.track_parts[track as usize], t, &mut opacity, value, weight, additive),
                        v2::NodeProperty::StrokeWidth if at_rest.text => {
                            let authored_width=self.authored(i)?.field(StyleField::StrokeWidth).as_f64().unwrap();
                            let mut width=fields.as_ref().and_then(|f|f.get(StyleField::StrokeWidth)).and_then(|s|s.json().as_f64()).unwrap_or(authored_width) as f32;
                            crate::animation::blend(std::slice::from_mut(&mut width),value,weight,additive,add);
                            fields.get_or_insert_with(StyleFields::default).set(StyleField::StrokeWidth,Arc::new(crate::animation::style::StyleValue::Json(serde_json::json!(width))));
                        }
                        property => {
                            let Some(path) = &mut path else { continue };
                            match property {
                                v2::NodeProperty::StrokeWidth => crate::animation::blend(std::slice::from_mut(&mut path.stroke_width), value, weight, additive, add),
                                v2::NodeProperty::StrokeDashoffset => crate::animation::blend(std::slice::from_mut(&mut path.stroke_dashoffset), value, weight, additive, add),
                                v2::NodeProperty::FillOpacity => crate::animation::blend(std::slice::from_mut(&mut path.fill_opacity), value, weight, additive, add),
                                v2::NodeProperty::StrokeOpacity => crate::animation::blend(std::slice::from_mut(&mut path.stroke_opacity), value, weight, additive, add),
                                _ => {}
                            }
                        }
                    }
                }
                (TrackTarget::Text { unit, index, property, .. }, crate::animation::Sample::Numeric(value)) => {
                    if !at_rest.text { continue; }
                    let piece = units.get_or_insert_with(UnitValues::default).write((*unit, *index));
                    match property {
                        UnitProperty::Translation => crate::animation::blend(&mut piece.translation, value, weight, additive, crate::animation::BlendKind::Add),
                        UnitProperty::Scale => crate::animation::blend(&mut piece.scale, value, weight, additive, crate::animation::BlendKind::Multiply),
                        UnitProperty::Rotation => crate::animation::blend(&mut piece.rotation, value, weight, additive, crate::animation::BlendKind::Rotation),
                        UnitProperty::Opacity => crate::animation::blend(std::slice::from_mut(&mut piece.opacity), value, weight, additive, crate::animation::BlendKind::Add),
                        UnitProperty::Fill | UnitProperty::Stroke => {
                            let slot=if *property==UnitProperty::Fill {&mut piece.fill}else{&mut piece.stroke};
                            slot.push(UnitPaint::Numeric(value.as_slice().try_into().expect("validated RGBA"),weight,additive));
                        }
                        UnitProperty::TransformOrigin => {}
                    }
                }
                // Visibility is which nodes draw (`shown`).
                _ => {}
            }
        }
        // Only a text node has kept batches.
        let text_revision = |specs: &[Option<TextSpecs>]| if at_rest.text { specs[i].as_ref().map_or(0, |c| c.content_revision) } else { 0 };
        // A node that does not draw (hidden, or absent on its clock: §8 rule 1) has no picture.
        if !shown {
            if node.is_some() { *node = None; }
            values.style_revision = 0;
            values.transform = at_rest.transform.clone();
            values.visible = false;
            values.opacity = 0.;
            values.path = None;
            values.texts = 0;
            values.text_composition.clear();
            values.text_revision = text_revision(&*specs);
            values.text_paint_revisions = specs[i].as_ref().map_or_else(Default::default, |c| c.paint_revisions.clone());
            return Ok(false);
        }
        // Style tracks restyle the node: it is parsed again when a content field differs from the one kept.
        let restyled = fields.is_some();
        if let Some(fields) = fields {
            if styled[i].as_ref().is_none_or(|cached|cached.fields!=fields) {
                let content_changed = styled[i].as_ref().map_or_else(
                    || fields.content(true).next().is_some(),
                    |old| old.fields.content(true).ne(fields.content(true)),
                );
                let node=if content_changed {
                    // The node as authored, with each field a track wrote in it: written into the typed node, field by field.
                    let mut node=self.authored(i)?.node.clone();
                    for (field,value) in &fields.0 {if *field != REFERENCE_FIELDS.path_data {node.set_field(field.name(),&value.json())?;}}
                    Some(node)
                } else {None};
                // The kept text when only its paints differ: the fields other than fill and stroke are the ones it
                // was made from, and Moda's vector lane draws it (the degraded tier rasters its colours).
                let kept=styled[i].as_ref().filter(|_|node.is_some()).and_then(|old| {
                    old.text.clone().filter(|text| !text.degraded() && old.fields.content(false).eq(fields.content(false)))
                });
                let mut text=match (&node,kept) {
                    (Some(node),Some(kept))=>Some(Arc::new(repainted_text(node,&kept,&self.images,&self.base,&fields,&self.media.borrow())?)),
                    (Some(node),None)=>build_text(node,&self.images,&self.base,&self.font_lists,&self.fonts,item.text.as_ref().map_or_else(TextTracks::default, |text| text.tracks.clone()),&fields,&self.media.borrow())?.map(Arc::new),
                    (None,_)=>styled[i].as_ref().and_then(|old|old.text.clone()),
                };
                if let Some(text)=&mut text {Arc::make_mut(text).clip_source=item.text.as_ref().is_some_and(|t|t.clip_source);}
                let mut draw=styled[i].as_ref().filter(|_|!content_changed).map_or_else(||item.node.clone(),|old|(*old.draw).clone());
                if let Some(sample) = &node {
                    draw.mask_type = sample.mask_type;
                    draw.blend = sample.mix_blend_mode;
                    draw.isolation = sample.isolation;
                }
                if let Some(node)=node.as_ref().filter(|node|matches!(node.kind,NodeKind::Path{..})) {
                    let mut path=crate::canvas_draw::sampled_path_item(&self.images,&self.base,node,fields.get(REFERENCE_FIELDS.path_data).map(|v|v.as_ref()),&self.media.borrow())?;
                    for (field,value) in &fields.0 {if let crate::animation::style::StyleValue::Paint(p)=value.as_ref(){path.set_paint(&self.images,&self.base,field.name(),p,&self.media.borrow())?;}}
                    draw.kind=crate::canvas_draw::Kind::Path(Arc::new(path));
                }
                let reference=|field:StyleField,rest:Option<usize>| fields.get(field).map_or(rest,|value|value.json().as_str().and_then(|name|self.by_name.get(name).copied()));
                draw.clip_path=reference(REFERENCE_FIELDS.clip_path,item.node.clip_path);
                draw.mask=reference(REFERENCE_FIELDS.mask,item.node.mask);
                if let Some(text)=&text {draw.kind=crate::canvas_draw::Kind::Text{size:text.size};}
                draw.filter=if let Some(crate::animation::style::StyleValue::Filter(filter))=fields.get(REFERENCE_FIELDS.filter).map(|v|v.as_ref()) {filter.clone()} else {item.node.filter.clone()};
                let revision=styled[i].as_ref().map_or(1,|c|c.revision+1);
                if content_changed && self.media.borrow().publish_assets {
                    let text_paints: Vec<_> = text.iter().flat_map(|t| t.fill.iter().chain(&t.stroke)).collect();
                    crate::canvas_draw::install_images(&[&draw], &text_paints)?;
                }
                let text_revision=styled[i].as_ref().map_or(0,|old|old.text_revision)+u64::from(content_changed && text.is_some());
                styled[i]=Some(StyledItem{fields,draw:Arc::new(draw),text,revision,text_revision});
            }
            // The restyled path's values, with the four keyed numbers the tracks blended.
            if let crate::canvas_draw::Kind::Path(p)=&styled[i].as_ref().unwrap().draw.kind {
                let old=path;
                path=Some(p.values);
                if let (Some(old),Some(p))=(old,&mut path) {
                    p.stroke_width=old.stroke_width;p.stroke_dashoffset=old.stroke_dashoffset;p.fill_opacity=old.fill_opacity;p.stroke_opacity=old.stroke_opacity;
                }
            }
        }
        let style = if restyled { styled[i].as_ref() } else { None };
        let q = Quat::from_array(transform.resolved_rotation()).normalize();
        ensure!(transform.translation[2] == 0. && transform.scale[2] == 1., "node {:?}: translation z and scale z are not built in a 2D canvas; key x and y", item.name);
        ensure!(q.x.abs() < 1e-5 && q.y.abs() < 1e-5, "node {:?}: a rotation off the canvas's Z axis is not built in a 2D canvas", item.name);
        let mut texts = 0;
        let drawing = transform.scale[0] != 0. && transform.scale[1] != 0. && parent_drawing;
        let text = if at_rest.text { style.and_then(|s| s.text.as_deref()).or(item.text.as_ref()) } else { None };
        if let Some(text) = text {
            if drawing {
                // The text node consumes its own samples, cached-spec hits included.
                let keyed = units.unwrap_or_default();
                for ((unit,index),_) in &keyed.0 {
                    let count=text.units[unit].len();
                    ensure!(*index<count,"node {:?}: {} has {count} pieces at this time; index {index} is past the end",item.name,unit.segment());
                }
                let slot = &mut specs[i];
                let style_revision = style.map_or(0, |s| s.text_revision);
                if slot.as_ref().is_none_or(|c| c.units != keyed || c.style_revision != Some(style_revision)) {
                    let (text_specs, mut composition) = if text.degraded() && !text.degraded_paints.is_empty() {
                        let text_specs = text.degraded_paints.iter().map(|(_, html)| {
                            let mut spec = text.spec.clone(); spec.html = html.clone().into(); spec
                        }).collect();
                        let parts = text.degraded_paints.iter().enumerate().map(|(sub, (paint, _))|
                            crate::canvas_draw::TextPart::Paint { sub, paint: paint.json().unwrap(), size: text.size, bounds: [0., 0., text.size[0], text.size[1]] }).collect();
                        (text_specs, parts)
                    } else if text.degraded() { (vec![text.spec.clone()], Vec::new()) } else { self.batches(&item.name, text, &keyed)? };
                    fn coverage_bounds(parts: &mut [crate::canvas_draw::TextPart], specs: &[moda_text::ops::TextNodeSpec],
                        fonts: &Fonts, item: usize, layout: &Arc<Laid>) -> Result<()> {
                        for part in parts {
                            match part {
                                crate::canvas_draw::TextPart::Paint { sub, bounds, .. } => {
                                    let draw = Draw { item, sub: *sub, spec: Arc::new(specs[*sub].clone()),
                                        model: glam::Mat4::IDENTITY.to_cols_array(), layout: layout.clone() };
                                    if let Some(batch) = fonts.build(&draw, 0., [f32::MIN, f32::MIN, f32::MAX, f32::MAX], 1.)? {
                                        if let Some(b) = batch.local_bounds() { *bounds = b; }
                                    }
                                }
                                crate::canvas_draw::TextPart::Group { children, .. } => coverage_bounds(children, specs, fonts, item, layout)?,
                                _ => {}
                            }
                        }
                        Ok(())
                    }
                    coverage_bounds(&mut composition, &text_specs, &self.fonts, i, &text.layout)?;
                    let revision = slot.as_ref().map_or(1, |cached| cached.revision + 1);
                    let clip = if text.clip_source {
                        let mut geometry = text.clone();
                        geometry.fill = vec![PaintSource::Color([1.; 4]); text.code_points];
                        geometry.stroke = vec![PaintSource::None; text.code_points];
                        geometry.tracks.paint_units = Default::default();
                        geometry.spec.fill_override = Some("white".into());
                        let clip_units = UnitValues(keyed.0.iter().map(|(key, value)| {
                            let mut value = value.clone();
                            value.opacity = 1.; value.fill.clear(); value.stroke.clear();
                            (*key, value)
                        }).collect());
                        Some(Arc::new(self.batches(&item.name, &geometry, &clip_units)?.0.remove(0)))
                    } else { None };
                    let same_geometry = !text.degraded() && slot.as_ref().is_some_and(|old|
                        old.specs.len() == text_specs.len()
                        && old.specs.iter().zip(&text_specs).all(|(a,b)| a.same_paint_geometry(b))
                        && match (&old.clip, &clip) { (None,None) => true, (Some(a),Some(b)) => a == b, _ => false });
                    let content_revision = slot.as_ref().map_or(1, |old| old.content_revision + u64::from(!same_geometry));
                    let paint_revisions = text_specs.iter().chain(clip.as_deref()).enumerate().map(|(sub, spec)| {
                        let Some(old) = slot.as_ref() else { return revision; };
                        let previous = old.specs.get(sub).or_else(|| (sub == old.specs.len()).then_some(old.clip.as_ref()).flatten());
                        if previous.is_some_and(|p| p.as_ref() == spec) { old.paint_revisions[sub] } else { revision }
                    }).collect();
                    *slot = Some(TextSpecs { paint_revisions, revision, content_revision, style_revision: Some(style_revision), units: keyed, composition, clip, specs: text_specs.into_iter().map(Arc::new).collect() });
                }
                let cached = slot.as_ref().unwrap();
                for (sub, spec) in cached.specs.iter().enumerate() {
                    draws.push(Draw { item: i, sub, spec: spec.clone(), model: glam::Mat4::IDENTITY.to_cols_array(), layout: text.layout.clone() });
                    texts += 1;
                }
                if let Some(clip) = &cached.clip {
                    draws.push(Draw { item: i, sub: texts, spec: clip.clone(),
                        model: glam::Mat4::IDENTITY.to_cols_array(), layout: text.layout.clone() });
                }
            }
        }
        if node.is_some() || style.is_some() { *node = style.map(|s| s.draw.clone()); }
        values.style_revision = style.map_or(0, |s| s.revision);
        values.transform = transform;
        values.visible = true;
        values.opacity = opacity;
        values.path = path;
        values.texts = texts;
        values.text_revision = text_revision(&*specs);
        let batches = specs[i].as_ref().filter(|_| drawing);
        values.text_paint_revisions = batches.map_or_else(Default::default, |c| c.paint_revisions.clone());
        match batches {
            Some(cached) => values.text_composition.clone_from(&cached.composition),
            None => values.text_composition.clear(),
        }
        Ok(drawing)
    }

    /// A clip-path or mask a style track names must not lead back to its own node.
    fn check_references(&self, nodes: &[Option<Arc<crate::canvas_draw::Node>>]) -> Result<()> {
        if !nodes.iter().enumerate().any(|(i,n)|n.as_ref().is_some_and(|n|n.clip_path!=self.items[i].node.clip_path || n.mask!=self.items[i].node.mask)) { return Ok(()); }
        let mut states=vec![0u8;self.items.len()];
        // Each installed child/reference edge is visited once. Scanning
        // the document for each node made this cycle check quadratic.
        let mut pending = Vec::new();
        for root in 0..self.items.len() {
            if states[root] == 2 { continue; }
            pending.push((root, false));
            while let Some((i, exit)) = pending.pop() {
                if exit { states[i] = 2; continue; }
                ensure!(states[i] != 1, "clip-path/mask dependency cycle at node {:?}", self.items[i].name);
                if states[i] == 2 { continue; }
                states[i] = 1;
                pending.push((i, true));
                pending.extend(self.children.of(i).iter().rev().map(|&child| (child, false)));
                let node = nodes[i].as_deref().unwrap_or(&self.items[i].node);
                pending.extend([node.mask, node.clip_path].into_iter().flatten().map(|target| (target, false)));
            }
        }
        Ok(())
    }

    /// One text node's Moda specs at an instant. Every code point's transform is its glyph's, then its word's,
    /// then its line's, each about its own origin; its opacity their product; its fill and stroke the innermost
    /// piece's that keys them, else its own. Text in one fill (its runs' colours) and one stroke on every code point (or none) is one
    /// spec; otherwise each stroke colour is a spec drawing only the outlines of its code points, and each fill
    /// colour a spec drawing only the fill of its code points, fills first.
    fn batches(&self, name: &str, text: &TextItem, units: &UnitValues) -> Result<(Vec<moda_text::ops::TextNodeSpec>, Vec<crate::canvas_draw::TextPart>)> {
        let n = text.code_points;
        let mut base = text.spec.clone();
        let paint_bounds=[0.,0.,text.size[0],text.size[1]];
        let apply_paints=|layers:&[UnitPaint],mut paint:PaintSource|->Result<PaintSource> {
            for layer in layers {
                paint=match layer {
                    UnitPaint::Css(value,weight)=>{
                        let crate::animation::style::StyleValue::Paint(value)=value.as_ref() else {unreachable!()};
                        crate::canvas_draw::blend_paint(&paint,&crate::canvas_draw::resolve_animated_paint(&self.images,&self.base,paint_bounds,value,&self.media.borrow())?,*weight,paint_bounds)?
                    },
                    UnitPaint::Numeric(value,weight,additive)=>{
                        let mut color=paint.colour().unwrap_or([0.;4]);
                        crate::animation::blend(&mut color,value,*weight,*additive,crate::animation::BlendKind::Add);
                        PaintSource::Color(color.map(|c|c.clamp(0.,1.)))
                    }
                };
            }
            Ok(paint)
        };
        let mut unit_paints = Vec::with_capacity(units.0.len());
        for ((unit, index), value) in &units.0 {
            let (unit, index) = (*unit, *index);
            let first=text.units[&unit].get(index).and_then(|cps|cps.first()).copied();
            let paint=|layers:&[UnitPaint],rest:&[PaintSource]|->Result<Option<PaintSource>> {
                if layers.is_empty(){Ok(None)}else{apply_paints(layers,first.map_or(PaintSource::None,|cp|rest[cp].clone())).map(Some)}
            };
            unit_paints.push((paint(&value.fill,&text.fill)?,paint(&value.stroke,&text.stroke)?));
        }
        let mut fills = text.fill.clone();
        let mut strokes = text.stroke.clone();
        if text.tracks.pieces {
            let mut xforms = Vec::with_capacity(n);
            for cp in 0..n {
                let (mut m, mut b, mut opacity) = (DMat2::IDENTITY, DVec2::ZERO, 1f32);
                for level in LEVELS {
                    let Some(index) = text.piece_of[&level][cp] else { continue };
                    let Some(value) = units.get(&(level, index)) else { continue };
                    let l = linear(value).with_context(|| format!("node {name:?}: {} {index}", level.segment()))?;
                    let pivot = text.boxes[&level].get(index).copied().flatten().map_or(text.centre, |b| {
                        let [(fx, px), (fy, py)] = value.origin.unwrap_or([(0.5, 0.), (0.5, 0.)]);
                        DVec2::new(b[0] + f64::from(fx) * (b[2] - b[0]) + f64::from(px), b[1] + f64::from(fy) * (b[3] - b[1]) + f64::from(py))
                    });
                    // This level's map, l·(p − P) + P + t, after the inner levels' m·p + b.
                    m = l * m;
                    b = l * (b - pivot) + pivot + DVec2::new(f64::from(value.translation[0]), f64::from(value.translation[1]));
                    opacity *= value.opacity.clamp(0., 1.);
                }
                for level in LEVELS.into_iter().rev() {
                    let Some((fill,stroke)) = text.piece_of[&level][cp].and_then(|index| units.position(&(level, index)).ok().map(|slot| &unit_paints[slot])) else {continue};
                    if let Some(fill)=fill {fills[cp]=fill.clone();}
                    if let Some(stroke)=stroke {strokes[cp]=stroke.clone();}
                }
                // Moda maps each glyph p to lin·(p − W) + W + off, W the centre of all the node's ink.
                let off = b + (m - DMat2::IDENTITY) * text.centre;
                xforms.push(moda_text::vector::ops::GlyphXform {
                    lin: [m.x_axis.x as f32, m.y_axis.x as f32, m.x_axis.y as f32, m.y_axis.y as f32],
                    off: [off.x as f32, off.y as f32],
                    opacity,
                });
            }
            base.unit_transforms = (xforms).into();
            if n > 0 { base.unit_groups = (vec![0; n]).into(); }
        }
        if text.stroke_width <= 0. { strokes.iter_mut().for_each(|s| *s = PaintSource::None); }
        if self.media.borrow().publish_assets { crate::canvas_draw::install_images(&[],&fills.iter().chain(&strokes).collect::<Vec<_>>())?; }
        if fills.iter().chain(&strokes).any(|p| matches!(p, PaintSource::Moda(_))) {
            return Ok(self.painted_batches(text, units, base, fills, strokes));
        }
        let fills: Vec<_> = fills.iter().map(PaintSource::colour).collect();
        let strokes: Vec<_> = strokes.iter().map(PaintSource::colour).collect();
        // Keep every previously supported frame on its original batch path. Only the formerly
        // refused fill/stroke overlap needs unit pictures, composed by Moda's opacity groups.
        let compose = base.unit_transforms.iter().enumerate().any(|(cp, xform)| {
            xform.opacity > 0. && xform.opacity < 1.
                && quantised(fills[cp]).is_some() && quantised(strokes[cp]).is_some()
        });
        let finish = |specs| {
            if compose { self.compose_units(text, units, specs, &[]) } else { (specs, Vec::new()) }
        };
        let rgba = |q: [u8; 4]| q.map(|c| f32::from(c) / 255.);
        let stroke_keyed = text.tracks.paint_units[1].is_some() && text.stroke_width > 0.;
        if text.tracks.paint_units[0].is_none() && !stroke_keyed && (strokes.iter().all(Option::is_none) || text.paint_ranges(true).nth(1).is_none()) {
            if let Some(colour) = strokes.first().copied().and_then(quantised) {
                base.stroke_color = Some(css(rgba(colour)));
                base.stroke_width = Some(text.stroke_width);
            }
            return Ok(finish(vec![base]));
        }
        if base.unit_transforms.is_empty() {
            base.unit_transforms = Arc::new(vec![moda_text::vector::ops::GlyphXform::IDENTITY; n]);
        }
        let mut specs = Vec::new();
        if text.tracks.paint_units[0].is_some() {
            for range in text.paint_ranges(false) {
                let mut spec = base.clone();
                spec.fill_override = Some(css(rgba(quantised(fills[range.start]).unwrap_or([0; 4]))));
                spec.unit_range = Some(range);
                specs.push(spec);
            }
        } else {
            specs.push(base.clone());
        }
        let no_fill = Arc::new(vec![0.; n]);
        for range in text.paint_ranges(true) {
            let colour = quantised(strokes[range.start]);
            if colour.is_none() && !stroke_keyed { continue; }
            let mut spec = base.clone();
            spec.stroke_color = Some(css(rgba(colour.unwrap_or([0; 4]))));
            spec.stroke_width = Some(text.stroke_width);
            spec.unit_fill = no_fill.clone();
            spec.unit_range = Some(range);
            specs.push(spec);
        }
        Ok(finish(specs))
    }

    /// Paint each distinct run/glyph coverage with the same paint the path renderer consumes.
    fn painted_batches(&self, text: &TextItem, units: &UnitValues,
        mut base: moda_text::ops::TextNodeSpec, fills: Vec<PaintSource>, strokes: Vec<PaintSource>)
        -> (Vec<moda_text::ops::TextNodeSpec>, Vec<crate::canvas_draw::TextPart>) {
        let n = text.code_points;
        if base.unit_transforms.is_empty() {
            base.unit_transforms = Arc::new(vec![moda_text::vector::ops::GlyphXform::IDENTITY; n]);
        }
        let mut specs = Vec::new();
        let mut paints = Vec::new();
        for (stroke, sources) in [(false, fills), (true, strokes)] {
            let keyed = text.tracks.paint_units[usize::from(stroke)].is_some() && (!stroke || text.stroke_width > 0.);
            let coverage = Arc::new(vec![if stroke { 0. } else { 1. }; n]);
            for range in text.paint_ranges(stroke) {
                let source = &sources[range.start];
                if matches!(source, PaintSource::None) && !keyed { continue; }
                let mut spec = base.clone();
                spec.fill_override = Some("white".into());
                spec.unit_fill = coverage.clone();
                spec.stroke_color = stroke.then(|| "white".into());
                spec.stroke_width = stroke.then_some(text.stroke_width);
                spec.unit_range = Some(range);
                paints.push(source.json().or_else(|| PaintSource::Color([0.; 4]).json()));
                specs.push(spec);
            }
        }
        self.compose_units(text, units, specs, &paints)
    }

    /// Keep independently faded unit pictures separate; range batches share their full layout.
    /// A word/line contains its glyph pictures; its opacity applies after those children compose.
    /// Paint order inside each picture remains all fills followed by all strokes.
    fn compose_units(&self, text: &TextItem, units: &UnitValues,
        mut specs: Vec<moda_text::ops::TextNodeSpec>, paints: &[Option<moda_scene::value::Value>]) -> (Vec<moda_text::ops::TextNodeSpec>, Vec<crate::canvas_draw::TextPart>) {
        use crate::canvas_draw::TextPart;
        // All paint ranges share the same transforms. Lift fractional opacity
        // into the groups once, preserving the zero-opacity glyphs.
        if let Some(first) = specs.first() {
            let mut transforms = first.unit_transforms.clone();
            if transforms.iter().any(|x| x.opacity > 0. && x.opacity < 1.) {
                for x in Arc::make_mut(&mut transforms) { if x.opacity > 0. { x.opacity = 1.; } }
                for spec in &mut specs { spec.unit_transforms = transforms.clone(); }
            }
        }
        let paths: Vec<Vec<_>> = (0..text.code_points).map(|cp| LEVELS.into_iter().rev().filter_map(|level| {
            let index = text.piece_of[&level][cp]?;
            let opacity = units.get(&(level, index))?.opacity.clamp(0., 1.);
            (opacity > 0. && opacity < 1.).then_some((level, index, opacity))
        }).collect()).collect();
        fn emit(paths: &[Vec<(TextUnit, usize, f32)>], range: std::ops::Range<usize>, depth: usize,
            specs: &[moda_text::ops::TextNodeSpec], paints: &[Option<moda_scene::value::Value>], size: [f64; 2], out: &mut Vec<moda_text::ops::TextNodeSpec>) -> Vec<TextPart> {
            let mut parts = Vec::new();
            let mut start = range.start;
            while start < range.end {
                let key = paths[start].get(depth);
                let end = (start + 1..range.end).find(|&cp| paths[cp].get(depth) != key).unwrap_or(range.end);
                if let Some(&(_, _, opacity)) = key {
                    parts.push(TextPart::Group { opacity, children: emit(paths, start..end, depth + 1, specs, paints, size, out) });
                } else {
                    for (index, spec) in specs.iter().enumerate() {
                        let mut spec = spec.clone();
                        let range = spec.unit_range.as_ref().map_or(start..end, |r| r.start.max(start)..r.end.min(end));
                        if range.is_empty() { continue; }
                        spec.unit_range = Some(range.clone());
                        if spec.unit_transforms[range].iter().any(|x| x.opacity > 0.) {
                            parts.push(match paints.get(index).and_then(Option::as_ref) {
                                Some(paint) => TextPart::Paint { sub: out.len(), paint: paint.clone(), size, bounds: [0., 0., size[0], size[1]] },
                                None => TextPart::Batch(out.len()),
                            });
                            out.push(spec);
                        }
                    }
                }
                start = end;
            }
            parts
        }
        let mut out = Vec::new();
        let composition = emit(&paths, 0..paths.len(), 0, &specs, paints, text.size, &mut out);
        (out, composition)
    }

    /// Bring the installed scene to the evaluated table and plan it through Moda's retained geometry caches. Text
    /// batches keep their geometry until their content or AA/raster scale changes; Moda places them anew.
    /// `worlds` are the pictures of the canvas's worlds at the frame the table stands at.
    fn plan(&self, worlds: &[Option<(String, [u32; 2])>]) -> Result<crate::canvas_draw::Plan> {
        let mut plan = self.plan_for_draw(worlds)?;
        plan.materialize(self.retained.borrow_mut().as_mut().context("canvas scene")?)?;
        Ok(plan)
    }

    fn plan_for_draw(&self, worlds: &[Option<(String, [u32; 2])>]) -> Result<crate::canvas_draw::Plan> {
        let mut evaluated = self.evaluated.borrow_mut();
        let evaluated = evaluated.as_mut().context("internal: a canvas planned before a frame was evaluated")?;
        self.bring(&mut evaluated.state, worlds)
    }

    /// Publish the evaluated frame to GPU workers. Release the producer's old
    /// reference before patching so a finished consumer costs no snapshot copy.
    fn shared_plan(&self, engine: &crate::canvas_draw::Engine) -> Result<crate::canvas_draw::SharedPlan> {
        let scene = self.retained.borrow().as_ref().expect("parallel frames follow a serial draw").picture_scene();
        if !self.stale.get() && let Some(plan) = engine.published(scene) { return Ok(plan); }
        engine.forget_published();
        let plan = self.plan_for_draw(&[])?;
        engine.publish(&plan).ok_or_else(|| wgpu::optional_allocations::Refused.into())
    }

    /// Give the scene its worlds' pictures, bring it whole to `state` if it is owed that, and plan it.
    fn bring(&self, state: &mut FrameState, worlds: &[Option<(String, [u32; 2])>]) -> Result<crate::canvas_draw::Plan> {
        let _profile = moda_engine::prof::span("canvas.plan");
        let update_profile = moda_engine::prof::span("canvas.plan.update");
        let mut page=self.page.clone();page.background=state.background;
        // Until the plan is made the scene is owed everything, and the text builder is not kept: a failure below
        // leaves it so.
        let stale = self.stale.replace(true);
        let text = self.text.borrow_mut().take();
        let mut retained = self.retained.borrow_mut();
        // A world's picture is a new one each frame, so its item's row is written here. The table keeps the
        // picture: a later frame that moves the world's item writes its row from the table, and lowering the
        // canvas again reads it there.
        let mut pictures: Vec<(usize, u8)> = Vec::new();
        for (i, world) in self.worlds.iter().enumerate() {
            let picture = worlds.get(i).cloned().flatten();
            if state.values[world.item].world != picture {
                state.values[world.item].world = picture;
                pictures.push((world.item, crate::canvas_draw::change::REST));
            }
        }
        let state = &*state;
        let node = |i: usize| state.nodes[i].as_deref().unwrap_or(&self.items[i].node);
        let written = match retained.as_mut() {
            Some(scene) if !stale => pictures.is_empty() || scene.write(&page, node, &state.values, &pictures)?,
            _ => false,
        };
        if !written {
            let nodes: Vec<&crate::canvas_draw::Node> = (0..self.items.len()).map(node).collect();
            match retained.as_mut() {
                Some(scene) => scene.update(&page, &nodes, &state.values)?,
                None => *retained = Some(crate::canvas_draw::Retained::new(&page, &nodes, &state.values)?),
            }
        }
        drop(update_profile);
        // A frame that changed no text batch draws the ones planned: their models come from the plan.
        let text = match text {
            Some(text) => text,
            None => {
                let fonts = self.fonts.clone();
                let draws = state.draws.clone();
                let text = std::rc::Rc::new(move |item: usize, sub: usize, model: [f32; 16], z: f64, clip: [f32; 4], scale: f64, opacity: f64, kept: Option<&moda_text_draw::TextQuadBatch>| {
                    let mut placed = draws.get(item).get(sub).context("a text batch the frame did not plan")?.clone();
                    placed.model = model;
                    Arc::make_mut(&mut placed.spec).opacity = opacity;
                    match kept {
                        Some(kept) => fonts.repaint(&placed, kept),
                        None => fonts.build(&placed, z, clip, scale),
                    }
                });
                text as crate::canvas_draw::CanvasTextBuilder
            }
        };
        let result = crate::canvas_draw::plan_for_draw(retained.as_mut().unwrap(), &page, text.clone())?;
        *self.text.borrow_mut() = Some(text);
        self.stale.set(false);
        Ok(result)
    }

    /// A test's copy of the evaluated table at `request`.
    #[cfg(test)]
    fn snapshot(&self, request: &vork_time::Request) -> Result<Snapshot> {
        self.frame(request)?;
        let serial = self.snapshots.get() + 1;
        self.snapshots.set(serial);
        Ok(Snapshot { state: self.evaluated.borrow().as_ref().expect("a frame was evaluated").state.clone(), document: self.id, serial, edited: false })
    }

    /// Plan a test's snapshot. The document's own table when the snapshot is this document's latest and unedited; else the
    /// scene is brought whole to the snapshot's state, and is then owed the whole table again.
    #[cfg(test)]
    fn plan_snapshot(&self, snapshot: &Snapshot, worlds: &[Option<(String, [u32; 2])>]) -> Result<crate::canvas_draw::Plan> {
        if !snapshot.edited && snapshot.document == self.id && snapshot.serial == self.snapshots.get() { return self.plan(worlds); }
        let mut state = snapshot.state.clone();
        self.stale.set(true);
        self.text.borrow_mut().take();
        let plan = self.bring(&mut state, worlds);
        self.stale.set(true);
        self.text.borrow_mut().take();
        plan
    }
}

impl Fonts {
    /// The accepted layout of `spec`'s text, made once for its content (`LaidKey`, with `layout_html` the text
    /// without its colours) on these fonts: Moda's vector
    /// layout, or, for text its vector lane declines, its degraded tier's lines and box.
    fn lay_out(&self, spec: &moda_text::ops::TextNodeSpec, layout_html: Arc<str>, here: &str, drawn: &dyn Fn() -> Result<String>) -> Result<Arc<Laid>> {
        let key: LaidKey = (layout_html, spec.width.map(f64::to_bits), spec.auto_width);
        {
            let layouts = self.layouts.borrow();
            if let Some(laid) = layouts.by_content.get(&key).and_then(std::sync::Weak::upgrade) { return Ok(laid); }
        }
        let _profile = moda_engine::prof::span("canvas.text.layout");
        // Without a width the box is the widest line, so text-align places lines within it.
        let fit = spec.width.is_none();
        let outlines = moda_text::vector::layout_counts()[1];
        let uncoloured = moda_text::ops::TextNodeSpec { html: key.0.clone(), ..spec.clone() };
        let laid = match moda_text::vector::accept_vector_layout(&uncoloured, &self.store, &self.host.catalog, moda_text::vector::VectorPlacement::VectorParity, fit, Some(&self.host)) {
            Ok(vector) => Laid {
                metrics: vector.lines().iter().map(|l| [l.advance, l.baseline]).collect(),
                width: vector.box_width(), height: vector.height(),
                lines: vector.lines().iter().map(|line| line.source.iter().flat_map(|source| source.values()).flatten().collect()).collect(),
                vector: Some(vector),
            },
            Err(moda_text::vector::VectorDecline { reason, cause, families, outlines: true }) =>
                bail!("{here}: Moda's vector text lane declined this text ({reason} ({cause}; families {families:?}))"),
            Err(_) => {
                let engine = self.host.canvas_engine();
                let mut spec = spec.clone();
                spec.html = drawn()?.into();
                let mut layout = engine.layout_json(&spec).map_err(|error| anyhow::anyhow!("{here}: {error}"))?;
                if fit {
                    spec.width = Some(vork_std::moda::widest_line(&layout));
                    layout = engine.layout_json(&spec).map_err(|error| anyhow::anyhow!("{here}: {error}"))?;
                }
                let lines = || layout["paragraphs"].as_array().into_iter().flatten().flat_map(|p| p["lines"].as_array().into_iter().flatten());
                Laid {
                    metrics: lines().map(|l| Ok([l["segments"].as_array().context("missing line segments")?.iter().map(|s| s["width"].as_f64().context("missing segment advance")).sum::<Result<f64>>()?, l["baselineY"].as_f64().context("missing baseline")?])).collect::<Result<_>>()?,
                    vector: None, width: spec.width.unwrap_or(0.),
                    height: layout["totalHeightPx"].as_f64().unwrap_or_else(|| lines()
                        .map(|line| line["lineTopY"].as_f64().unwrap_or(0.) + line["lineHeightPx"].as_f64().unwrap_or(0.)).fold(0., f64::max)),
                    lines: lines().map(|line| line["segments"].as_array().into_iter().flatten()
                        .flat_map(|segment| segment["sourceIndices"].as_array().into_iter().flatten())
                        .filter_map(|v| v.as_u64().map(|i| i as usize)).collect()).collect(),
                }
            }
        };
        if moda_engine::prof::on() {
            moda_engine::prof::add_work("canvas.text.outlines", 0, moda_text::vector::layout_counts()[1] - outlines);
        }
        let laid = Arc::new(laid);
        let mut layouts = self.layouts.borrow_mut();
        if layouts.by_content.len() >= layouts.sweep_at {
            layouts.by_content.retain(|_, laid| laid.strong_count() > 0);
            layouts.sweep_at = 2 * layouts.by_content.len() + 1;
        }
        layouts.by_content.insert(key, Arc::downgrade(&laid));
        Ok(laid)
    }

    /// Recolour an unchanged vector op stream. Effects/emoji keep their ordinary preparation path.
    fn repaint(&self, draw: &Draw, kept: &moda_text_draw::TextQuadBatch) -> Result<Option<Arc<moda_text_draw::TextQuadBatch>>> {
        // A paint edit can reveal an image that emitted no op in the previous batch.
        if draw.layout.vector.as_ref().is_none_or(|layout| layout.has_emojis()) || kept.degraded.is_some() || !kept.ops.is_empty()
            || !draw.spec.effects.is_empty() || draw.spec.gradient.is_some() { return Ok(None); }
        let Some(vector) = &kept.vector else { return Ok(None) };
        if !vector.emojis.is_empty() { return Ok(None); }
        let stroke = draw.spec.stroke_color.as_deref().map(|css|
            moda_text::color::css_color_to_normalized_rgba(css).unwrap_or([0., 0., 0., 1.]).map(|c| c as f32));
        let fill = draw.spec.fill_override.as_deref().filter(|s| !s.is_empty()).map(|css|
            moda_text::color::css_color_to_normalized_rgba(css).unwrap_or([0., 0., 0., 1.]).map(|c| c as f32));
        // Unnamed analytic fills use the node override. Unnamed primitives may instead carry their
        // own decoration colour, so they retain ordinary preparation.
        if vector.ops.iter().any(|op| op.stroke_px == 0. && op.paint.is_none()
            && (fill.is_none() || op.pipeline != moda_text::vector::VectorOpPipeline::Analytic)) { return Ok(None); }
        let mut batch = kept.clone();
        for op in &mut batch.vector.as_mut().unwrap().ops {
            if op.stroke_px > 0. {
                let Some(color) = stroke else { return Ok(None) };
                op.color = color;
            } else {
                let Some(color) = op.paint.and_then(|p| draw.spec.run_colors.get(p as usize)).copied().or(fill)
                    else { return Ok(None) };
                op.color = color;
            }
            op.opacity = draw.spec.opacity as f32;
        }
        // This replacement has a new picture identity; it was not built through the content cache.
        batch.content_key = None;
        Ok(Some(Arc::new(batch)))
    }

    /// Moda's batch for one draw of its layout: its vector lane's, or, for text its degraded tier draws, its raster.
    fn build(&self, draw: &Draw, z: f64, clip: [f32; 4], scale: f64) -> Result<Option<Arc<moda_text_draw::TextQuadBatch>>> {
        let outcome = match &draw.layout.vector {
            None => {
                let engine = self.host.canvas_engine();
                moda_text_draw::build_degraded_batch(engine, &draw.spec, draw.model, z, clip, scale)
                    .with_context(|| format!("node {:?}: Moda's degraded text lane", draw.spec.node_id))?
            }
            Some(layout) => match moda_text_draw::build_vector_batch(layout, &self.store, &self.host, &draw.spec, draw.model, z, clip, scale)? {
                Ok(outcome) => outcome,
                Err(reason) => bail!("node {:?}: Moda's vector text lane declined this text ({reason})", draw.spec.node_id),
            },
        };
        match outcome {
            moda_text_draw::TextBuildOutcome::Item(batch) => Ok(Some(batch)),
            moda_text_draw::TextBuildOutcome::Empty => Ok(None),
            moda_text_draw::TextBuildOutcome::Degraded { reason } => bail!("node {:?}: {reason}", draw.spec.node_id),
        }
    }
}

/// A piece's rotation and scale as a 2×2 map (rotation about the canvas's Z axis after a scale).
fn linear(value: &UnitValue) -> Result<DMat2> {
    let [_, _, tz] = value.translation;
    let [sx, sy, sz] = value.scale;
    let q = Quat::from_array(value.rotation).normalize();
    ensure!(tz == 0. && sz == 1., "translation z and scale z are not built in a 2D canvas; key x and y");
    ensure!(q.x.abs() < 1e-5 && q.y.abs() < 1e-5, "a rotation off the canvas's Z axis is not built in a 2D canvas");
    let angle = 2. * f64::from(q.z).atan2(f64::from(q.w));
    let (sin, cos) = angle.sin_cos();
    let (sx, sy) = (f64::from(sx), f64::from(sy));
    Ok(DMat2::from_cols(DVec2::new(sx * cos, sx * sin), DVec2::new(-sy * sin, sy * cos)))
}

struct OutputCommand {
    encoder: wgpu::CommandEncoder,
    prefix: Option<moda_engine::scratch::FramePrefix>,
}
impl std::ops::Deref for OutputCommand {
    type Target = wgpu::CommandEncoder;
    fn deref(&self) -> &Self::Target { &self.encoder }
}
impl std::ops::DerefMut for OutputCommand {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.encoder }
}
impl OutputCommand {
    fn submit(self, queue: &wgpu::Queue) -> wgpu::SubmissionIndex {
        let output = self.encoder.finish();
        let _profile = moda_engine::prof::span("canvas.output.submit");
        match self.prefix {
            Some(prefix) => prefix.submit(output),
            None => queue.submit([output]),
        }
    }
}

/// The GPU side: Moda's engine, which draws each frame's plan into its own 4x MSAA target and resolve, and the
/// straight-alpha frame the outputs read: video encoders on the GPU, the exposure report reduced on the GPU
/// (`display_stats`), PNG files through a readback.
struct Target {
    engine: crate::canvas_draw::Engine,
    device: wgpu::Device,
    size: [u32; 2],
    /// The last frame Moda drew: associated (premultiplied) colour, Rgba8Unorm or (`Document::float`) Rgba16Float.
    drawn: RefCell<Option<wgpu::Texture>>,
    frame_format: wgpu::TextureFormat,
    /// `finish` pipelines by destination format and by how the destination is written.
    finish: std::cell::RefCell<HashMap<(wgpu::TextureFormat, Finish), wgpu::ComputePipeline>>,
    /// Creates `finish` pipelines through the persistent pipeline cache.
    compiler: crate::pipelines::Compiler,
    stride: u32,
    pictures: Pictures,
    /// The view stages this canvas's renders have asked for, and the one that finishes this render's frames
    /// (`set_display`).
    display: RefCell<Stages>,
    /// Whether the drawn picture is float (`Document::float`).
    float: bool,
    /// The stage that writes the canvas's picture as a reader takes it (`Pictures::reader_stage`), built when a
    /// render first writes that picture to EXR (`set_raw`).
    picture: RefCell<Option<crate::display_program::PictureStage>>,
}
/// A root canvas's view stages, each built when a render first encodes for its display and kept with the
/// canvas: a float canvas has one for any display, an 8-bit canvas for a display other than the one its
/// encoding is the code values of (sRGB's for an sRGB canvas, Display P3's for a Display P3 canvas). A render
/// to Display P3 builds no sRGB stage beside it, and a session that goes back and forth between displays
/// builds each once.
#[derive(Default)]
struct Stages {
    built: Vec<(crate::output_device::Target, crate::display_program::PictureStage)>,
    /// The stage for the display in use; `None` when the picture needs none.
    current: Option<usize>,
}
impl Stages {
    fn stage(&self) -> Option<&crate::display_program::PictureStage> {
        self.current.map(|index| &self.built[index].1)
    }
}
/// How `Target::finish` writes a destination.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Finish {
    /// An 8-bit picture: its associated bytes as straight alpha, with the rounding of an 8-bit divide.
    Unassociate,
    /// A frame the view stage already finished: its code values as they are.
    Copy,
}
/// Installed world pictures shared by standalone canvases and material pictures, and the canvas's image graph
/// and view stage.
pub(crate) struct Pictures {
    /// The canvas's image graph (`composition.compositor`), run over each drawn picture before any view.
    graph: Option<crate::compositor::CanvasGraph>,
    /// The view stage of a canvas a world reads (`floating`, `reader_stage`): it writes that world's float
    /// picture. A root canvas's stage belongs to its `Target`, which builds it for the display a render
    /// encodes for.
    view: Option<crate::display_program::PictureStage>,
    worlds: RefCell<Vec<crate::canvas_world::World>>,
    pool: moda_engine::pixel::pool::TexturePool,
    convert: crate::canvas_world::Convert,
}
impl Pictures {
    /// `floating`: the picture is read by a world, so an authored view delivers it floating point.
    pub(crate) fn new(renderer: &Renderer, document: &Document, base: &Path, floating: bool) -> Result<Self> {
        document.media.borrow_mut().prepare_gpu(renderer)?;
        let worlds = document.worlds.iter().map(|world| crate::canvas_world::World::install(renderer, world.scene.clone(), base, &document.opened)
            .with_context(|| format!("node {:?}", document.items[world.item].name))).collect::<Result<Vec<_>>>()?;
        let view = if floating { Self::reader_stage(renderer, document, base)? } else { None };
        let graph = document.view.as_ref().map(|scene| crate::compositor::CanvasGraph::new(renderer, scene, base, &document.opened)).transpose()?.flatten();
        Ok(Self { graph, view, worlds: RefCell::new(worlds), pool: moda_engine::pixel::pool::TexturePool::new(&renderer.device, moda_engine::pixel::pool::TexturePool::budget_from_env()),
            convert: crate::canvas_world::Convert::into_canvas(document.working) })
    }
    fn worker(renderer: &Renderer, document: &Prepared,
        programs: Option<Arc<crate::compositor::CanvasGraphPrograms>>) -> Result<Self> {
        let graph = programs.map(|programs| crate::compositor::CanvasGraph::from_programs(renderer,
            document.view.as_ref().expect("a graph has an installed scene"), programs)).transpose()?;
        Ok(Self { graph, view: None, worlds: Default::default(),
            pool: moda_engine::pixel::pool::TexturePool::new(&renderer.device, moda_engine::pixel::pool::TexturePool::budget_from_env()),
            convert: crate::canvas_world::Convert::into_canvas(document.working) })
    }

    /// The stage that writes the canvas's picture as a reader takes it: its view's code values for sRGB,
    /// floating point, neither clamped nor rounded. Every picture that leaves a canvas has passed its view; a
    /// world reads the canvas so, and an EXR stores it so, whatever encoding the canvas blends in. A canvas
    /// whose view is the identity for sRGB (the default view of an sRGB canvas) has no stage: its drawn
    /// picture is that picture.
    fn reader_stage(renderer: &Renderer, document: &Prepared, base: &Path) -> Result<Option<crate::display_program::PictureStage>> {
        let reader = crate::output_device::Target::default();
        document.view.as_ref().filter(|scene| !crate::display_program::Prepared::identity(scene, reader))
            .map(|scene| crate::display_program::PictureStage::new(renderer, scene, base, reader, true, &document.opened)).transpose()
    }
    /// The canvas's picture at `frame` before its view: what Moda drew, then the canvas's image graph over it.
    /// `errors`: the shader diagnostics of the frame the picture is drawn for, where the graph records a result
    /// that is not finite; without them the graph's own word (`CanvasGraph::errors`).
    fn draw(&self, renderer: &Renderer, document: &Document, frame: &Frame, output_index: Option<u32>, engine: &crate::canvas_draw::Engine, errors: Option<&wgpu::Buffer>,
        recording: Option<moda_engine::scratch::OutputRecording<'_>>, planned: impl FnOnce()) -> Result<(wgpu::Texture, Option<moda_engine::scratch::FramePrefix>)> {
        let _gpu = moda_engine::prof::capturing().then(moda_engine::gpu_profile::resume);
        // Each present world is drawn by the 3D renderer at its local time, and its frame crosses into a pool
        // texture Moda's engine samples as an image: on the GPU, in queue order before Moda's drawing.
        let worlds_profile = (!frame.worlds.is_empty()).then(|| moda_engine::prof::span("canvas.worlds"));
        let mut captures = Vec::new();
        let mut pictures = vec![None; frame.worlds.len()];
        let mut stored = Vec::new();
        let mut worlds = self.worlds.borrow_mut();
        // Release every hidden camera before admitting any newly visible one,
        // independent of authored world order. Pictures already submitted keep
        // their native resources through their consumers.
        for (world, request) in worlds.iter_mut().zip(&frame.worlds) {
            if request.is_none() { world.hide(); }
        }
        for ((world, request), picture) in worlds.iter_mut().zip(&frame.worlds).zip(&mut pictures) {
            let Some(request) = request else { continue };
            let [w, h] = world.size;
            let output = world.draw(renderer, request).map_err(device)?;
            let texture = self.pool.acquire(w, h, moda_engine::pixel::image::Storage::F16).map_err(|error| device(anyhow::anyhow!("a world's picture: {error}")))?;
            let mut encoder = renderer.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("world picture") });
            self.convert.encode(renderer, &mut encoder, output, texture.view(), texture.texture().format(), crate::canvas_world::Crossing::ToCanvas);
            #[cfg(feature = "frame-profile")]
            crate::frame_profile::count("queue_submissions", 1);
            renderer.queue.submit([encoder.finish()]);
            let capture = moda_engine::external_texture::CaptureTexture::new(texture);
            stored.push((capture.key().to_string(), [w, h], capture.upload_id));
            *picture = Some((capture.key().to_string(), [w, h]));
            captures.push(capture);
        }
        if !frame.worlds.is_empty() { crate::canvas_draw::install_captures(&stored); }
        drop(worlds_profile);
        // Evaluation has already written changed rows. A world capture is a further
        // scene input installed by plan(), so it must go through that step first.
        let held = if pictures.is_empty() && !document.stale.get() {
            document.retained.borrow().as_ref().zip(document.text.borrow().as_ref())
                .and_then(|(scene, text)| engine.picture(scene.picture_scene(), &std::rc::Rc::downgrade(text)))
        } else { None };
        let drawn = if let Some(texture) = held { planned(); texture } else {
            engine.forget_published();
            let mut plan = document.plan_for_draw(&pictures).map_err(scene_error)?;
            planned();
            let _profile = moda_engine::prof::span("canvas.draw");
            engine.render_resident(&mut plan, document.retained.borrow_mut().as_mut().expect("planned scene")).map_err(device)?
        };
        drop(captures);
        self.composite(renderer, drawn, frame, output_index, errors, recording.and_then(|r| r.finish()))
    }

    fn composite(&self, renderer: &Renderer, drawn: wgpu::Texture, frame: &Frame, output_index: Option<u32>, errors: Option<&wgpu::Buffer>,
        mut prefix: Option<moda_engine::scratch::FramePrefix>) -> Result<(wgpu::Texture, Option<moda_engine::scratch::FramePrefix>)> {
        let picture = if let Some(graph) = &self.graph {
            let _profile = moda_engine::prof::span("canvas.compositor");
            graph.draw(renderer, &drawn, &frame.request, output_index, &frame.images, errors, prefix.as_mut())?
        } else {
            drawn
        };
        Ok((picture, prefix))
    }
}

/// One frame in flight: its finished pixels stay on the GPU until acceptance and encoder release.
struct Slot {
    work: crate::canvas_parallel::FrameWork,
    frame: wgpu::Texture,
    stats: crate::display_stats::Stats,
    /// The slot's entry in the submission ring, as a 3D frame slot has: the encoders' surfaces, the frame they
    /// still hold, the PNG staging and the credit that bounds its consumers.
    ring: crate::ring::Entry,
    /// Whether this render reads pixels back into the entry's staging (PNG files).
    readback: bool,
    /// This frame's copies of the error words of the picture's steps: the image graph's, the frame's view
    /// stage's, then the EXR picture's view stage's (four bytes each; a step the frame has not leaves its word
    /// zero), read with the frame's other results.
    picture_errors: Option<wgpu::Buffer>,
    /// This render's read of the canvas's picture after its view, for EXR.
    raw: Option<RawRead>,
    diagnostic_gpu: Option<moda_engine::gpu_profile::PendingFrame>,
    diagnostic_time: Option<moda_engine::gpu_profile::GpuTime>,
    diagnostic_host: moda_engine::prof::HostTimings,
}
/// A slot's read of the canvas's picture after its view, as a reader takes it: associated sRGB code values,
/// RGBA16F or (an 8-bit sRGB canvas, whose drawn picture it is) RGBA8.
struct RawRead {
    /// Where the reader's floating stage writes the picture; without one the drawn picture is read.
    picture: Option<wgpu::Texture>,
    format: wgpu::TextureFormat,
    stride: u32,
    readback: wgpu::Buffer,
}
struct NativeCanvasFrame {
    diagnostic_gpu: Option<moda_engine::gpu_profile::PendingFrame>,
    surfaces: Vec<Surface>,
    frame: Option<usize>,
}
type CanvasJob = (u32, f32, Frame, crate::canvas_draw::SharedPlan, Slot);
type CanvasCompletion = Result<(usize, Slot, Option<wgpu::SubmissionIndex>)>;
type CanvasWorkers = Vec<(std::sync::mpsc::SyncSender<CanvasJob>, std::sync::mpsc::Receiver<CanvasCompletion>)>;

fn stop_canvas_workers(pool: CanvasWorkers, joins: Vec<std::thread::ScopedJoinHandle<'_, ()>>,
    slots: &mut [Option<Slot>], renderer: &Renderer) -> Result<()> {
    if pool.is_empty() { return Ok(()); }
    let (senders, receivers): (Vec<_>, Vec<_>) = pool.into_iter().unzip();
    drop(senders);
    for join in joins { join.join().map_err(|panic| panicked("canvas frame worker", panic))?; }
    for receiver in receivers {
        for result in receiver.try_iter() {
            let (frame, slot, _) = result?;
            let entry = frame % slots.len();
            slots[entry] = Some(slot);
        }
    }
    // Retire GPU consumers before crediting any removed worker's memory.
    crate::gpu_memory::wait(&renderer.device, &renderer.queue, &renderer.errors, None)
}

impl Target {
    fn new(renderer: &Renderer, size: [u32; 2], document: &Document, base: &Path) -> Result<Self> {
        let limit = renderer.device.limits().max_texture_dimension_2d;
        ensure!(size[0] <= limit && size[1] <= limit,
            crate::gpu_memory::device_limit("render resolution", format_args!("{}x{} pixels", size[0], size[1]), format_args!("max_texture_dimension_2d={limit} pixels per axis"), "reduce render.resolution"));
        let engine = crate::canvas_draw::Engine::new(renderer, document.float, document.working).map_err(device)?;
        Ok(Self::with_pictures(renderer, size, document.float, Pictures::new(renderer, document, base, false)?, engine))
    }

    fn with_pictures(renderer: &Renderer, size: [u32; 2], float: bool, pictures: Pictures, engine: crate::canvas_draw::Engine) -> Self {
        let stride = (size[0] * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        Self { engine, device: renderer.device.clone(), size, drawn: RefCell::new(None), frame_format: renderer.output_format(), finish: Default::default(), compiler: renderer.compiler.clone(), stride,
            pictures, display: Default::default(), float, picture: Default::default() }
    }

    /// Prepare this render's EXR picture: the canvas's picture after its view, as a reader takes it
    /// (`Pictures::reader_stage`). Its floating stage is built once and kept with the canvas; a canvas with
    /// none (the default view of an sRGB canvas) has its drawn picture read as it is, 8-bit or float.
    fn set_raw(&self, renderer: &Renderer, document: &Prepared, base: &Path) -> Result<()> {
        let mut picture = self.picture.borrow_mut();
        if picture.is_none() { *picture = Pictures::reader_stage(renderer, document, base)?; }
        Ok(())
    }

    /// Encode this render's frames for `display`. An 8-bit canvas's picture is the code values of the display
    /// its encoding names (sRGB, Display P3): for that display its default view is the identity and it needs
    /// no stage. A float canvas is finished by its view stage for any display, and an 8-bit canvas for any
    /// other display than its own: the canvas's view runs in the display stage with that display as its
    /// target, over the drawn picture, as a world's view does. `scene` is the canvas's root scene.
    fn set_display(&self, renderer: &Renderer, document: &Prepared, scene: &Scene, base: &Path, display: crate::output_device::Target) -> Result<()> {
        let mut stages = self.display.borrow_mut();
        let scene = document.view.as_ref().unwrap_or(scene);
        let current = if !document.float && crate::display_program::Prepared::identity(scene, display) {
            None
        } else if let Some(index) = stages.built.iter().position(|(built, _)| *built == display) {
            Some(index)
        } else {
            let stage = crate::display_program::PictureStage::new(renderer, scene, base, display, false, &document.opened)?;
            stages.built.push((display, stage));
            Some(stages.built.len() - 1)
        };
        stages.current = current;
        Ok(())
    }

    /// `readback`: this render reads the 8-bit frame back (PNG files). `raw`: it reads the picture after its
    /// view back (EXR files; `set_raw` came first).
    fn slot(&self, renderer: &Renderer, readback: bool, raw: bool, surfaces: Vec<Surface>) -> Slot {
        let size = wgpu::Extent3d { width: self.size[0], height: self.size[1], depth_or_array_layers: 1 };
        let viewed = self.picture.borrow().is_some();
        let raw = raw.then(|| {
            let format = if viewed || self.float { wgpu::TextureFormat::Rgba16Float } else { wgpu::TextureFormat::Rgba8Unorm };
            let texel = if format == wgpu::TextureFormat::Rgba16Float { 8 } else { 4 };
            let stride = (self.size[0] * texel).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            RawRead {
                picture: viewed.then(|| crate::gpu_memory::texture(&renderer.device, &wgpu::TextureDescriptor {
                    label: Some("canvas viewed picture"), size, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
                })),
                format, stride,
                readback: crate::gpu_memory::buffer(&renderer.device, &wgpu::BufferDescriptor {
                    label: Some("canvas picture readback"), size: u64::from(stride) * u64::from(self.size[1]),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
                }),
            }
        });
        let staging = if readback { u64::from(self.stride) * u64::from(self.size[1]) } else { 0 };
        let mut ring = crate::ring::Entry::staging(&renderer.device, staging);
        ring.encoder_surfaces = surfaces;
        Slot {
            work: Default::default(),
            frame: renderer.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("canvas frame"),
                size: wgpu::Extent3d { width: self.size[0], height: self.size[1], depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: self.frame_format,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }),
            stats: crate::display_stats::Stats::new(&renderer.device),
            ring,
            readback,
            picture_errors: (self.pictures.graph.is_some() || self.display.borrow().stage().is_some() || raw.as_ref().is_some_and(|raw| raw.picture.is_some()))
                .then(|| crate::gpu_memory::buffer(&renderer.device, &wgpu::BufferDescriptor {
                    label: Some("canvas picture errors"), size: 12,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
                })),
            raw,
            diagnostic_gpu: None, diagnostic_time: None, diagnostic_host: Default::default(),
        }
    }

    /// Draw one frame and record the straight-alpha output, its
    /// exposure reduction and, with a readback, the copy of its pixels.
    fn encode(&self, renderer: &Renderer, document: &Document, frame: &Frame, index: u32, slot: &mut Slot, planned: impl FnOnce()) -> Result<OutputCommand> {
        let recording = self.engine.record_for_output();
        let (picture, prefix) = self.pictures.draw(renderer, document, frame, Some(index), &self.engine, None, Some(recording), planned)?;
        self.encode_drawn(renderer, picture, frame, slot, prefix)
    }

    fn encode_drawn(&self, renderer: &Renderer, drawn: wgpu::Texture, frame: &Frame, slot: &mut Slot,
        prefix: Option<moda_engine::scratch::FramePrefix>) -> Result<OutputCommand> {
        *self.drawn.borrow_mut() = Some(drawn);
        let _profile = moda_engine::prof::span("canvas.output.convert_record");
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("canvas frame") });
        self.finish(renderer, &mut encoder, &slot.frame, &frame.request, None)?;
        // Each step's error word travels with this frame, before the next frame's graph and view clear them.
        let display = self.display.borrow();
        if let Some(errors) = &slot.picture_errors {
            if let Some(graph) = &self.pictures.graph { encoder.copy_buffer_to_buffer(graph.errors(), 0, errors, 0, 4); }
            if let Some(view) = display.stage() { encoder.copy_buffer_to_buffer(view.errors(), 0, errors, 4, 4); }
        }
        // The picture after its view, for EXR: the reader's floating stage writes it; without one it is the
        // drawn picture. It is read before the next draw reuses the drawn picture.
        if let Some(raw) = &slot.raw {
            let drawn = self.drawn.borrow();
            let drawn = drawn.as_ref().context("no frame drawn")?;
            let stage = self.picture.borrow();
            let picture = match (stage.as_ref(), &raw.picture) {
                (Some(stage), Some(picture)) => {
                    stage.encode(renderer, &mut encoder, drawn, picture, &frame.request, None)?;
                    if let Some(errors) = &slot.picture_errors { encoder.copy_buffer_to_buffer(stage.errors(), 0, errors, 8, 4); }
                    picture
                }
                _ => drawn,
            };
            ensure!(picture.format() == raw.format, "the canvas's picture changed its format after the render began");
            encoder.copy_texture_to_buffer(picture.as_image_copy(), wgpu::TexelCopyBufferInfo {
                buffer: &raw.readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(raw.stride), rows_per_image: None },
            }, picture.size());
        }
        slot.stats.encode(renderer, &mut encoder, crate::display_stats::Display::Texture(&slot.frame), None);
        if slot.readback {
            encoder.copy_texture_to_buffer(slot.frame.as_image_copy(), wgpu::TexelCopyBufferInfo {
                buffer: &slot.ring.readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.stride), rows_per_image: None },
            }, slot.frame.size());
        }
        Ok(OutputCommand { encoder, prefix })
    }

    /// Write the resolved frame into `destination` (storage-bindable) as straight alpha: the blend leaves associated
    /// colour (src-alpha colour, one alpha), and every output is straight alpha. A native encoder surface takes
    /// storage writes, not copies, so it is written this way too. An 8-bit picture for the canvas's own display is
    /// unassociated with the rounding of an 8-bit divide, for every destination. A float picture, and any
    /// picture for another display (`set_display`), is finished once, into the frame
    /// (`frame` is `None`), at the frame's `request`, by its view stage, where every view's result is clamped
    /// and rounded; an encoder surface then takes that finished `frame`'s code values.
    fn finish(&self, renderer: &Renderer, encoder: &mut wgpu::CommandEncoder, destination: &wgpu::Texture, request: &vork_time::Request, frame: Option<&wgpu::Texture>) -> Result<()> {
        ensure!(destination.usage().contains(wgpu::TextureUsages::STORAGE_BINDING), "the video encoder's surface cannot be written on this device");
        let drawn = self.drawn.borrow();
        let drawn = drawn.as_ref().context("no frame drawn")?;
        let display = self.display.borrow();
        let (kind, source) = match (display.stage(), frame) {
            (None, _) => (Finish::Unassociate, drawn),
            (Some(view), None) => return view.encode(renderer, encoder, drawn, destination, request, None),
            (Some(_), Some(frame)) => (Finish::Copy, frame),
        };
        self.finish_from(encoder, source, destination, kind)
    }

    fn finish_from(&self, encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture, destination: &wgpu::Texture, kind: Finish) -> Result<()> {
        let device = &self.device;
        let source = source.create_view(&Default::default());
        let mut pipelines = self.finish.borrow_mut();
        let pipeline = pipelines.entry((destination.format(), kind)).or_insert_with(|| {
            let format = match destination.format() { wgpu::TextureFormat::Bgra8Unorm => "bgra8unorm", _ => "rgba8unorm" };
            let texel = match kind {
                Finish::Unassociate => "let c = vec4<u32>(round(textureLoad(source, id.xy, 0) * 255.0)); var rgb = c.rgb;\n\
                    if (c.a > 0u && c.a < 255u) { rgb = min((c.rgb * 255u + c.a / 2u) / c.a, vec3<u32>(255u)); }\n\
                    textureStore(destination, id.xy, vec4<f32>(vec3<f32>(rgb), f32(c.a)) / 255.0);",
                Finish::Copy => "textureStore(destination, id.xy, textureLoad(source, id.xy, 0));",
            };
            let module = crate::timing::shader_module(device, wgpu::ShaderModuleDescriptor {
                label: Some("canvas finish"),
                source: wgpu::ShaderSource::Wgsl(format!("@group(0) @binding(0) var source: texture_2d<f32>;\n\
                    @group(0) @binding(1) var destination: texture_storage_2d<{format}, write>;\n\
                    @compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{\n\
                    let size = textureDimensions(source); if (id.x >= size.x || id.y >= size.y) {{ return; }}\n\
                    {texel} }}").into()),
            });
            self.compiler.compute_cached(&wgpu::ComputePipelineDescriptor {
                label: Some("canvas finish"), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
            })
        });
        let destination_view = destination.create_view(&Default::default());
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("canvas finish"), layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&source) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&destination_view) }],
        });
        let timestamps = moda_engine::gpu_profile::pass_timestamps();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("canvas finish"), timestamp_writes: moda_engine::gpu_profile::compute_writes(&timestamps) });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1);
        Ok(())
    }

    /// A slot's frame once its submission is done: its exposure, with a readback its RGBA8 pixels, and with a
    /// raw read the picture after its view as associated float RGBA.
    fn collect(&self, renderer: &Renderer, slot: &mut Slot, submission: wgpu::SubmissionIndex) -> Result<(crate::Exposure, Option<crate::png_output::CanvasPixels>, Option<Vec<[f32; 4]>>)> {
        let (sender, receiver) = std::sync::mpsc::channel();
        let mapped = [slot.stats.pending().context("canvas exposure was not recorded")?].into_iter().chain(slot.readback.then_some(&slot.ring.readback)).chain(&slot.picture_errors).chain(slot.raw.as_ref().map(|raw| &raw.readback)).chain(slot.diagnostic_gpu.iter().flat_map(|gpu| gpu.readbacks())).collect::<Vec<_>>();
        for buffer in &mapped {
            let sender = sender.clone();
            buffer.map_async(wgpu::MapMode::Read, .., move |result| { let _ = sender.send(result); });
        }
        renderer.wait(Some(submission))?;
        for _ in &mapped { receiver.recv().context("canvas readback").map_err(device)?.context("canvas readback").map_err(device)?; }
        slot.diagnostic_time = slot.diagnostic_gpu.as_ref().map(|gpu| gpu.time());
        let picture_errors = slot.picture_errors.as_ref().map(|errors| {
            let codes = { let mapped = errors.get_mapped_range(..); [0, 4, 8].map(|at| u32::from_le_bytes(mapped[at..at + 4].try_into().expect("four bytes"))) };
            errors.unmap();
            codes
        });
        // The frame's completion is proved before its results are read as the scene's: every pixel
        // counted, and nothing recorded by the device. Only then are the picture's error words the scene's.
        let exposure = slot.stats.read(u64::from(self.size[0]) * u64::from(self.size[1]))?.context("canvas exposure was not recorded")?;
        renderer.errors.check()?;
        // Every step's word is read before the frame's error is chosen. A view written for another display
        // is wrong at every frame, so it outranks a non-finite result, which is this frame's refusal, as a
        // world's is: a sequence of per-frame files skips that.
        let codes = picture_errors.into_iter().flatten().filter(|&code| code != 0).collect::<Vec<_>>();
        let wrong_display = codes.contains(&vork_slang::registry::VIEW_DESTINATION);
        let code = if wrong_display { vork_slang::registry::VIEW_DESTINATION } else { codes.first().copied().unwrap_or(0) };
        if let Err(error) = crate::display_program::refuse(code) {
            // The frame delivers nothing: let go of its pixels, so its slot can take another frame.
            if slot.readback { slot.ring.readback.unmap(); }
            if let Some(raw) = &slot.raw { raw.readback.unmap(); }
            return Err(scene_error(if wrong_display { error } else { anyhow::Error::new(crate::FrameRefused(error.to_string())) }));
        }
        let _conversion = moda_engine::prof::span("canvas.output.convert_record");
        let pixels = slot.readback.then_some(&slot.ring.readback).map(|readback| {
            let row_bytes = self.size[0] as usize * 4;
            let stride = self.stride as usize;
            let len = row_bytes * self.size[1] as usize;
            let mapped = readback.get_mapped_range(..);
            let bytes = if stride == row_bytes {
                mapped[..len].to_vec()
            } else {
                let mut pixels = Vec::with_capacity(len);
                for row in mapped.chunks(stride).take(self.size[1] as usize) {
                    pixels.extend_from_slice(&row[..row_bytes]);
                }
                pixels
            };
            drop(mapped);
            readback.unmap();
            crate::png_output::CanvasPixels::new(bytes, slot.frame.format())
        });
        let raw = slot.raw.as_ref().map(|raw| {
            let mut rgba = Vec::with_capacity((self.size[0] * self.size[1]) as usize);
            {
                let mapped = raw.readback.get_mapped_range(..);
                for row in mapped.chunks(raw.stride as usize).take(self.size[1] as usize) {
                    if raw.format == wgpu::TextureFormat::Rgba16Float {
                        rgba.extend(row[..(self.size[0] * 8) as usize].chunks_exact(8).map(|texel|
                            std::array::from_fn(|c| half::f16::from_bits(u16::from_le_bytes([texel[2 * c], texel[2 * c + 1]])).to_f32())));
                    } else {
                        rgba.extend(row[..(self.size[0] * 4) as usize].chunks_exact(4).map(|texel| std::array::from_fn(|c| f32::from(texel[c]) / 255.)));
                    }
                }
            }
            raw.readback.unmap();
            rgba
        });
        Ok((exposure, pixels, raw))
    }
}

/// `vork check` on a canvas: fonts load, images decode, and the first frame lays out and plans with Moda's engine
/// (no GPU).
pub(crate) fn check(scene: &Scene, base: &Path, renderer: &Renderer, opened: &crate::footage::Opened) -> Result<()> {
    // Everything here reads, lays out and installs the document: its refusals are the scene's,
    // except what the device hands back, which keeps its own category.
    check_document(scene, base, renderer, opened).map_err(scene_error)
}
fn check_document(scene: &Scene, base: &Path, renderer: &Renderer, opened: &crate::footage::Opened) -> Result<()> {
    let document = Document::new_with_paints(scene, base, None, opened)?;
    document.install_images()?;
    document.media.borrow_mut().prepare_gpu(renderer)?;
    document.frame(&vork_time::Request::Numeric(0.))?;
    document.plan(&[])?;
    // A float canvas's image graph and view stage: their programs, inputs and images, as a render prepares them.
    if let Some(scene) = &document.view {
        crate::compositor::CanvasGraph::new(renderer, scene, base, &document.opened)?;
        crate::display_program::PictureStage::new(renderer, scene, base, Default::default(), false, &document.opened)?;
    }
    // Each world installs as its own 3D scene.
    for world in &document.worlds {
        renderer.install_with_options(world.scene.clone(), base, crate::InstallOptions { opened: document.opened.clone(), readback: Default::default(), ..Default::default() })
            .with_context(|| format!("node {:?}", document.items[world.item].name))?;
    }
    Ok(())
}

/// Draw one frame of `scene` as a render to `display` does, and keep nothing of it: the document, Moda's
/// drawing, the frame's finish and view stage, its exposure reduction, and the picture after its view that an
/// EXR destination reads (`set_raw`: an authored view's floating stage, a different shader from the frame's).
/// `vork setup` runs it over its canvas catalogue, so the driver pipelines a canvas render creates at its
/// first draw, to PNG, video or EXR, are already in the pipeline cache. It is the render path itself
/// (`Target::encode`), not a list of pipelines kept beside it. Nothing is read back: a pipeline exists once its
/// pass is encoded, and setup has no use for the frame's statistics or pixels, so only what the device itself
/// reports can fail it.
///
/// Release shader preparation has no device that executes anything. What it keeps of a canvas is what was
/// compiled before any drawing, the view's program and its stages for this display and for the EXR picture,
/// so it stops there.
pub(crate) fn warm(scene: &Scene, base: &Path, renderer: &Renderer, display: crate::output_device::Target) -> Result<()> {
    let document = Document::new(scene, base)?;
    let target = Target::new(renderer, scene.render.resolution, &document, base)?;
    target.set_display(renderer, &document, scene, base, display)?;
    target.set_raw(renderer, &document, base)?;
    if renderer.release_preparation { return Ok(()); }
    document.install_images()?;
    let mut slot = target.slot(renderer, true, true, Vec::new());
    let frame = document.frame(&vork_time::Request::Numeric(0.))?;
    let mut command = target.encode(renderer, &document, &frame, 0, &mut slot, || {})?;
    // A clip's native encoder takes the finished frame through `finish` as well, into its own surface: a
    // storage texture in the frame's format. Finishing into one here creates that pipeline (the copy of a
    // frame its view stage finished, or the same unassociate the frame took).
    let surface = crate::gpu_memory::texture(&renderer.device, &wgpu::TextureDescriptor {
        label: Some("setup encoder surface"), size: slot.frame.size(), mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2, format: slot.frame.format(),
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
    });
    target.finish(renderer, &mut command, &surface, &frame.request, Some(&slot.frame))?;
    #[cfg(feature = "frame-profile")]
    crate::frame_profile::count("queue_submissions", 1);
    let submission = command.submit(&renderer.queue);
    renderer.wait(Some(submission))?;
    renderer.errors.check()
}

// Test-only observation of the real scheduler, without timing thresholds or encoder substitutes.
#[cfg(test)]
std::thread_local! {
    static FRAME_EVENTS: RefCell<Option<Vec<(&'static str, usize)>>> = const { RefCell::new(None) };
}
#[cfg(test)]
fn frame_event(event: &'static str, frame: usize) {
    FRAME_EVENTS.with_borrow_mut(|events| {
        if let Some(events) = events { events.push((event, frame)); }
    });
}

/// What a session keeps of a canvas between renders: the document (its text laid out, its fonts loaded, its tracks
/// compiled) and the GPU side (Moda's engine with its pipelines and caches, the installed worlds). It belongs to the
/// renderer that made it and to one resolved scene, which `identity` names; the session proves the scene's files.
pub(crate) struct Kept {
    identity: blake3::Hash,
    document: Document,
    target: Target,
}
impl Kept {
    pub(crate) fn opened(&self) -> &crate::footage::Opened { &self.document.opened }
    /// The resolved scene and the directory its files resolve against: everything the document and the target are
    /// built from. Outputs, frame ranges and destinations are each render's own. The scene's content is hashed
    /// as the typed scene is walked, once (`crate::identity`).
    fn identity(scene: &Scene, base: &Path) -> Result<blake3::Hash> {
        let mut hash = blake3::Hasher::new();
        hash.update(base.as_os_str().as_encoded_bytes());
        hash.update(&[0]);
        crate::identity::write(&mut hash, scene)?;
        Ok(hash.finalize())
    }

    /// Stage the new CPU inputs while the published document still owns its installation. The
    /// first evaluated sample supplies the same row writes as a seek during an export.
    pub(crate) fn revise(&mut self, renderer: &Renderer, scene: &Scene, base: &Path,
        opened: &crate::footage::Opened, unchanged: &BTreeSet<&Path>) -> Result<bool> {
        let old = &self.document;
        if old.view.is_some() || has_worlds(&scene.nodes) { return Ok(false); }
        let next = Document::prepare_document(scene, base, None, opened, Some((old, unchanged)))?;
        if next.view.is_some() || next.page != old.page || next.items.len() != old.items.len()
            || !Arc::ptr_eq(&next.fonts.host, &old.fonts.host) || old.stale.get() {
            return Ok(false);
        }
        let identity = Self::identity(scene, base)?;
        next.media.borrow_mut().prepare_gpu(renderer)?;
        next.install_images()?;
        // Text stamps are local to a bound row. Compare the new specs against that row's
        // previous specs, even when neither document has an animated text style.
        next.text_specs.borrow_mut().clone_from(&old.text_specs.borrow());
        for cached in next.text_specs.borrow_mut().iter_mut().flatten() { cached.style_revision = None; }
        let request = vork_time::Request::Numeric(0.);
        next.evaluate(&request)?;
        let written = {
            let before = old.evaluated.borrow();
            let Some(before) = before.as_ref() else { return Ok(false) };
            let after = next.evaluated.borrow();
            let after = &after.as_ref().expect("evaluated candidate").state;
            let node = |i: usize| after.nodes[i].as_deref().unwrap_or(&next.items[i].node);
            let changes: Vec<_> = before.state.values.iter().zip(&after.values).enumerate().filter_map(|(i, (a, b))| {
                use crate::canvas_draw::change;
                let old_node = before.state.nodes[i].as_deref().unwrap_or(&old.items[i].node);
                if old_node != node(i) { return Some((i, change::ALL)); }
                if a == b { return None; }
                let mut placed = a.clone();
                placed.transform.clone_from(&b.transform);
                placed.opacity = b.opacity;
                Some((i, if placed == *b { change::TRANSFORM | change::OPACITY } else { change::ALL }))
            }).collect();
            let mut page = next.page.clone();
            page.background = after.background;
            let mut rows = old.retained.borrow_mut();
            let Some(rows) = rows.as_mut() else { return Ok(false) };
            rows.write(&page, node, &after.values, &changes)
        };
        // A form change declines before writing a row. On a partial write or plan failure the
        // published table can lower fresh rows under a fresh scene identity on its next look.
        if matches!(written, Ok(false)) { return Ok(false); }
        *next.retained.borrow_mut() = old.retained.borrow_mut().take();
        next.stale.set(false);
        // A matte needs a store entry to plan. The next look chooses the displayed movie frame.
        if let Err(error) = written.and_then(|_| next.publish_frame_media(&request, MediaPublication::Missing))
            .and_then(|bindings_changed| {
                if bindings_changed { next.write_rows(&Changed::Every)?; }
                next.plan(&[])
            }) {
            old.stale.set(true);
            return Err(error);
        }
        self.document = next;
        self.identity = identity;
        Ok(true)
    }
    /// Build what a render of `scene` builds before its first frame, with a render's errors, then evaluate and
    /// plan the first frame as `check` does: a session proves a candidate document this way before it publishes
    /// it. A canvas with worlds is planned by its first render instead, when the worlds' pictures exist; their
    /// scenes are installed here, in the target.
    pub(crate) fn prepare(renderer: &Renderer, scene: &Scene, base: &Path, display: crate::output_device::Target, opened: &crate::footage::Opened) -> Result<Self> {
        let identity = Self::identity(scene, base)?;
        let document = Document::new_with_paints(scene, base, None, opened).map_err(scene_error)?;
        let target = Target::new(renderer, scene.render.resolution, &document, base).map_err(scene_error)?;
        // The view stage for the display the session's renders encode for: an authored view's program, inputs
        // and images are proved here, and the render finds the stage built.
        target.set_display(renderer, &document, scene, base, display).map_err(scene_error)?;
        document.install_images().map_err(scene_error)?;
        document.frame(&vork_time::Request::Numeric(0.)).map_err(scene_error)?;
        if document.worlds.is_empty() {
            document.plan(&[]).map_err(scene_error)?;
        }
        Ok(Self { identity, document, target })
    }
}

/// What the shot entry hands over for a canvas.
pub(crate) struct Request {
    pub sound: Option<crate::audio::MovieSound>,
    pub opened: crate::footage::Opened,
    pub scene: Scene,
    pub base: PathBuf,
    pub out: Vec<PathBuf>,
    pub outputs: Vec<Output>,
    pub still: Option<shot::Still>,
    pub first_frame: usize,
    pub last_frame: usize,
    pub selected: Option<std::ops::RangeInclusive<u32>>,
    pub png_depth: crate::PngDepth,
    pub exr_codec: crate::ExrCodec,
    /// Each destination's EXR storage (`output.color_space`), by path.
    pub raw_colors: BTreeMap<PathBuf, crate::output_device::RawColor>,
    /// The display the render call's destinations share.
    pub display: crate::output_device::Target,
    pub no_write: bool,
    pub gpu_only: bool,
    pub diagnostics: bool,
    pub stop: Option<Arc<std::sync::atomic::AtomicBool>>,
    pub profile_record: serde_json::Value,
    pub export_identity: Option<String>,
    pub device_reused: bool,
    /// A session's document and GPU side from an earlier render on the same renderer, reused when this render
    /// resolves to the same scene.
    pub kept: Option<Kept>,
}

/// Render a canvas to the destinations a 3D shot writes: a PNG still, a directory of frames, EXR frames, MP4 or MOV.
pub(crate) fn render(request: Request, device_worker: std::thread::ScopedJoinHandle<'_, Result<Renderer>>, warnings: &crate::warnings::Collector, wall: Instant) -> Result<RenderedShot> {
    let Request { scene, base, out, outputs, still, first_frame, last_frame, selected, png_depth, exr_codec, raw_colors, display, no_write, gpu_only, diagnostics, stop, profile_record, export_identity, device_reused, sound, kept, opened } = request;
    let _capture = diagnostics.then(|| moda_engine::prof::capture(crate::canvas_timing::category));
    let parse_start = Instant::now();
    let [w, h] = scene.render.resolution;
    let fps = scene.render.fps;
    ensure_usage!(!out.is_empty(), "at least one output destination is required");
    ensure_usage!(outputs.iter().all(|o| *o == Output::Color), "a 2D canvas has one channel, color; depth, motion, object IDs and AOVs are 3D outputs");
    ensure_usage!(png_depth == crate::PngDepth::Eight, "--png-depth 16: a 2D canvas is 8-bit");
    let is_video = |path: &PathBuf| shot::video_destination(path);
    let is_png = |path: &PathBuf| shot::png_destination(path);
    let is_exr = |path: &PathBuf| shot::exr_destination(path);
    for path in &out {
        ensure_usage!(shot::recognized_destination(path),
            "--out {}: unrecognized extension; use .png (one still), .exr (with %06d), .mp4, .mov, or a directory (no extension, or end the path with /)", path.display());
        ensure_usage!(!is_exr(path) || path.file_name().is_some_and(|name| name.to_string_lossy().contains("%06d")), "EXR destination requires a %06d frame placeholder");
        ensure_usage!(!is_png(path) || !shot::has_frame_pattern(path), "--out {}: PNG frame patterns are unsupported; use a directory (every frame as NNNNNN.png)", path.display());
    }
    shot::validate_unique_destinations(&out)?;
    let video_paths: Vec<PathBuf> = out.iter().filter(|p| is_video(p)).cloned().collect();
    let directory_paths: Vec<PathBuf> = out.iter().filter(|p| !is_video(p)).cloned().collect();
    let video = !video_paths.is_empty();
    if directory_paths.iter().any(|path| is_exr(path)) {
        i32::try_from(fps.numerator()).context("EXR rate numerator exceeds i32")?;
        u32::try_from(fps.denominator()).context("EXR rate denominator exceeds u32")?;
    }
    let at = still.as_ref().map(|still| still.time);
    ensure_usage!(at.is_none() || !video, "--at selects one still; a video renders every frame");
    let frames = if at.is_some() { 1 } else { last_frame + 1 - first_frame };
    let still_index = still.as_ref().map(|still| still.index);
    let output_index = |frame: usize| still_index.unwrap_or((first_frame + frame) as u32);
    if video {
        ensure_usage!(!no_write && !gpu_only, "video output cannot be combined with --no-write or --gpu-only");
        ensure_usage!(w % 2 == 0 && h % 2 == 0, "video output requires even dimensions");
    }
    for path in &video_paths {
        ensure_usage!(!path.is_dir(), "video output path is a directory: {}", path.display());
        std::fs::create_dir_all(path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new(".")))?;
    }
    for path in &directory_paths {
        ensure_usage!(!is_png(path) || frames == 1, "--out {}: a PNG file is one still; add --at SECONDS, or use a directory for every frame", path.display());
        if is_png(path) || is_exr(path) {
            std::fs::create_dir_all(path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new(".")))?;
        } else {
            std::fs::create_dir_all(path)?;
        }
    }
    // A kept document and GPU side serve the same resolved scene; any other is dropped here and built again.
    // The scene's identity is what decides that. With nothing kept there is nothing to decide: the scene is
    // hashed by the worker that prepares the encoders, beside its document's construction, not before it.
    let known = match &kept {
        Some(_) => Some(Kept::identity(&scene, &base)?),
        None => None,
    };
    let kept = kept.filter(|kept| Some(kept.identity) == known);
    let install_reused = kept.is_some();
    let (kept_document, kept_target) = kept.map_or((None, None), |kept| (Some(kept.document), Some(kept.target)));
    // Prepare the encoders while the text is laid out.
    let (document, prepared) = std::thread::scope(|scope| {
        // The worker walks the scene to hash it, so it has the stack a scene nested to its limit needs.
        let prepared = vork_scene::scene_thread().spawn_scoped(scope, || {
            let start = diagnostics.then(Instant::now);
            let prepared = video_paths.iter().map(|path| {
                Encoder::prepare(Backend::Auto, w, h, fps, EncodeSettings { diagnostics, primaries: display.video_primaries(), ..EncodeSettings::default() }, path).map_err(device)
            }).collect::<Result<Vec<_>>>();
            let encoder_prepare_ms = start.map_or(0., |s| s.elapsed().as_secs_f64() * 1000.);
            let identity = match known {
                Some(identity) => Ok(identity),
                None => Kept::identity(&scene, &base),
            };
            (prepared, identity, encoder_prepare_ms)
        }).expect("start the encoder preparation worker");
        let document = match kept_document {
            Some(document) => Ok(document),
            None => Document::new_with_paints(&scene, &base, None, &opened).map_err(scene_error),
        };
        (document, prepared.join().map_err(|panic| panicked("encoder preparation worker", panic)))
    });
    let document = document?;
    let scene_parse_ms = parse_start.elapsed().as_secs_f64() * 1000.;
    let (prepared, identity, encoder_prepare_ms) = prepared?;
    let prepared = prepared?;
    let identity = identity?;
    let renderer = device_worker.join().map_err(|panic| panicked("device creation worker", panic))??;
    // Receipts describe this request: timings from earlier renders on a retained device are excluded.
    let pipeline_timings_before: std::collections::BTreeMap<String, f64> =
        if device_reused { renderer.pipeline_timings().into_iter().collect() } else { Default::default() };
    let _gpu_capture = diagnostics.then(moda_engine::gpu_profile::capture_state);
    let gpu_profile = moda_engine::prof::gpu() && !diagnostics;
    if gpu_profile {
        let available = moda_engine::gpu_profile::enable(&renderer.device, &renderer.queue);
        eprintln!("canvas GPU timestamps available: {available}; query resolution serializes frames");
    }
    // Old outputs are removed only after every encoder can be written.
    for prepared in &prepared { prepared.validate_fallback(frames as u64, Some(&renderer.device)).map_err(device)?; }
    let output_run = if no_write || gpu_only { None } else {
        Some(crate::output_run::OutputRun::begin(&out, selected)?)
    };
    let install_start = Instant::now();
    let target = match kept_target {
        Some(target) => target,
        None => Target::new(&renderer, [w, h], &document, &base).map_err(scene_error)?,
    };
    target.set_display(&renderer, &document, &scene, &base, display).map_err(scene_error)?;
    // An 8-bit canvas shows its authored display colours as they are, or (sRGB code values on a wider display)
    // re-expressed with nothing lost. Display P3 code values shown as sRGB are not: the view clips the colours
    // sRGB cannot show.
    let as_authored = !document.float() && (document.working() == v2::WorkingColorSpace::Srgb || document.working().display() == Some(display.display));
    // Images already in Moda's store are not read again.
    document.install_images().map_err(scene_error)?;
    // Sharing belongs only to optional replicas. Serial and nested-world
    // histories retain their ordinary image-cache eviction/destruction path.
    let shared_images = std::sync::Arc::new(moda_engine::paint::draw::SharedImages::default());
    let target_install_ms = install_start.elapsed().as_secs_f64() * 1000.;
    // Frames in flight: the CPU builds the next frames while the GPU draws and the encoders take the earlier ones.
    let forced_workers = std::env::var("VORK_CANVAS_WORKERS").ok().and_then(|v| v.parse::<usize>().ok());
    let independent = document.worlds.is_empty() && document.media.borrow().is_empty();
    let limit = forced_workers.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from).min(crate::constants::CANVAS_DEFAULT_MAX_WORKERS))
        .clamp(1, crate::constants::CANVAS_MAX_WORKERS).min(frames);
    let frame_bytes = (u64::from(w) * 4).div_ceil(256) * 256 * u64::from(h);
    // Native device/local/unified headroom decides whether replication is
    // optional. The encoder validates its fixed serial buffer independently.
    let can_replicate = independent && limit > 1
        && (forced_workers.is_some() || frames >= crate::constants::CANVAS_PARALLEL_MIN_FRAMES);
    let free = can_replicate.then(|| crate::canvas_parallel::headroom(&renderer.device)).flatten().map(|(_, free)| free);
    let serial_capacity = crate::constants::FRAMES_IN_FLIGHT.min(frames);
    // Each destination may retain conversion textures and a native shared ring;
    // readback/output slots include both the 8-bit and float pictures.
    let slot_bytes = frame_bytes.saturating_mul(8 + 3 * video_paths.len() as u64);
    let ring_workers = (renderer.device.limits().max_buffer_size / frame_bytes.max(1) / 2).max(1) as usize;
    let limit = limit.min(ring_workers);
    let full_capacity = serial_capacity.max(limit * 2).min(frames);
    let capacity = crate::canvas_parallel::ring_limit(free, slot_bytes, serial_capacity, full_capacity);
    let worker_limit = if free.is_some() { crate::canvas_parallel::replicas_for_ring(limit, capacity, full_capacity) } else { 1 };
    let mut workers = 1;
    // Native encoders keep the serial ring. Optional output slots and histories
    // are reserved only when main's existing policy admits parallel rendering.
    let mut in_flight = serial_capacity;
    let encoder_open_start = diagnostics.then(Instant::now);
    let mut encoders = prepared.into_iter().map(|prepared| prepared.open(renderer.device.clone(), renderer.queue.clone(), frames as u64, serial_capacity).map_err(device)).collect::<Result<Vec<_>>>()?;
    let completion_handoff = !encoders.is_empty() && encoders.iter().all(Encoder::uses_completion_handoff);
    // Each slot's ring entry on every encoder, installed in frame order so entry indices match the slots.
    let surfaces = (0..in_flight).map(|_| encoders.iter_mut().map(|encoder| encoder.create_surface().map_err(setup)).collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
    let encoder_start_ms = if video { encoder_prepare_ms + encoder_open_start.map_or(0., |s| s.elapsed().as_secs_f64() * 1000.) } else { 0. };
    // PNG files read the 8-bit frame back; EXR files read the picture after its view.
    let pngs = !no_write && !gpu_only && directory_paths.iter().any(|path| !is_exr(path));
    let raws = !no_write && !gpu_only && directory_paths.iter().any(|path| is_exr(path));
    if raws { target.set_raw(&renderer, &document, &base)?; }
    let slot_start = Instant::now();
    let native_frames = RefCell::new(surfaces.into_iter().map(|surfaces| NativeCanvasFrame { surfaces, frame: None, diagnostic_gpu: None }).collect::<Vec<_>>());
    let mut slots: Vec<Option<Slot>> = (0..in_flight).map(|_| Some(target.slot(&renderer, pngs, raws, Vec::new()))).collect();
    let png_writers: Vec<_> = directory_paths.iter().filter(|path| pngs && !is_exr(path)).map(|path|
        crate::png_output::FrameOutputWriter::canvas(path.clone(), w, h, in_flight, output_index(0) as usize, display.display)).collect();
    let raw_writers: Vec<_> = directory_paths.iter().filter(|path| raws && is_exr(path)).map(|path|
        crate::png_output::FrameOutputWriter::canvas_exr(path.clone(), w, h, exr_codec, in_flight, output_index(0) as usize,
            (fps.numerator() as i32, fps.denominator() as u32), raw_colors[path])).collect();
    let install_ms = target_install_ms + slot_start.elapsed().as_secs_f64() * 1000.;
    eprintln!("render: canvas {w}x{h}; {frames} frames; {} text and {} path node(s), {} world(s)", document.items.iter().filter(|item| item.text.is_some()).count(),
        document.items.iter().filter(|item| matches!(item.node.kind, crate::canvas_draw::Kind::Path(_))).count(), document.worlds.len());
    vork_slang::progress::phase("Rendering frames", frames);
    let setup_host = diagnostics.then(moda_engine::prof::take_capture);
    let loop_start = Instant::now();
    let mut metadata_frames = Vec::new();
    let mut frame_exposures = Vec::new();
    let mut samples = Vec::new();
    let mut memory_frames = Vec::new();
    let mut rendered_frames = 0;
    let mut first_frame_ms = 0.;
    let mut interrupted = false;
    let mp4 = video_paths.iter().any(|path| shot::mp4_destination(path));
    let mut refused = shot::Refusals::default();
    let host_memory = (worker_limit > 1).then(|| crate::canvas_host_memory::Guard::new(diagnostics)).flatten().map(Arc::new);
    let optional_budget = host_memory.as_ref().and_then(|host| crate::canvas_parallel::budget(&renderer.device,
        cfg!(not(target_os = "macos")) && renderer.adapter_info.device_type == wgpu::DeviceType::DiscreteGpu, host.clone()));
    if diagnostics && worker_limit > 1 && optional_budget.is_none() {
        eprintln!("canvas concurrency: memory counters unavailable; keep serial");
    }
    let worker_limit = if optional_budget.is_some() { worker_limit } else { 1 };
    let result = std::thread::scope(|scope| -> Result<()> {
        use crate::canvas_parallel::{Resize, Schedule};
        let automatic = worker_limit > 1 && forced_workers.is_none();
        let shared_masks = target.engine.publish_masks(worker_limit > 1 && forced_workers.is_some());
        let spawn_worker = || {
            let (send, receive) = std::sync::mpsc::sync_channel::<CanvasJob>(0);
            let (done, completed) = std::sync::mpsc::channel();
            let renderer = &renderer;
            let prepared = document.prepared.clone();
            let shared_masks = shared_masks.clone();
            let shared_images = shared_images.clone();
            let budget = optional_budget.as_ref().expect("native headroom admits workers").clone();
            let host_memory = host_memory.as_ref().expect("host headroom admits workers");
            let programs = target.pictures.graph.as_ref().map(|graph| graph.programs());
            // A stage shares every installed resource except its mutable error word.
            let (display_stage, raw_stage) = budget.run(|| (
                target.display.borrow().stage().map(|stage| stage.replica(renderer)),
                target.picture.borrow().as_ref().map(|stage| stage.replica(renderer)),
            ))?;
            let finish = target.finish.borrow().clone();
            let draw = target.engine.draw_resources();
            let join = vork_scene::scene_thread().spawn_scoped(scope, move || {
                let _warnings = warnings.enter();
                // A job's output lease survives cancellation of disposable
                // target/planner state. Retrying it never skips an output frame.
                let current = RefCell::new(None::<(usize, Slot)>);
                let run = || -> Result<()> {
                    let pictures = Pictures::worker(renderer, &prepared, programs).map_err(scene_error)?;
                    let engine = moda_engine::Engine::from_draw_resources(draw).map_err(device)?;
                    let mut target = Target::with_pictures(renderer, [w, h], prepared.float, pictures,
                        crate::canvas_draw::Engine::with_engine(renderer, engine));
                    target.engine.share_images(shared_images);
                    *target.finish.get_mut() = finish;
                    *target.picture.get_mut() = raw_stage;
                    if let Some(stage) = display_stage {
                        *target.display.get_mut() = Stages { built: vec![(display, stage)], current: Some(0) };
                    }
                    if gpu_profile { moda_engine::gpu_profile::enable(&renderer.device, &renderer.queue); }
                    let _capture = diagnostics.then(|| moda_engine::prof::capture(crate::canvas_timing::category));
                    let _gpu_capture = diagnostics.then(moda_engine::gpu_profile::capture_state);
                    while let Ok((_frame, _time, instant, plan, slot)) = receive.recv() {
                        current.replace(Some((_frame as usize, slot)));
                        let mut job = current.borrow_mut();
                        let slot = &mut job.as_mut().unwrap().1;
                        #[cfg(feature = "frame-profile")]
                        let _profile_event = crate::frame_profile::event("canvas_submit",
                            output_index(_frame as usize), _time);
                        let draw_start = automatic.then(Instant::now);
                        host_memory.check(_frame as usize)?;
                        budget.refresh()?;
                        let frame_profile = moda_engine::prof::span("canvas.frame");
                        if diagnostics { moda_engine::gpu_profile::begin_frame(&renderer.device, &renderer.queue, slot.diagnostic_gpu.take()); }
                        let recording = target.engine.record_for_output();
                        let drawn = target.engine.draw_shared(&plan, &shared_masks).map_err(device)?;
                        drop(plan);
                        let (drawn, prefix) = target.pictures.composite(renderer, drawn, &instant, Some(output_index(_frame as usize)), None, recording.finish())?;
                        let mut command = target.encode_drawn(renderer, drawn, &instant, slot, prefix)?;
                        host_memory.check(_frame as usize)?;
                        let output_profile = moda_engine::prof::span("canvas.output.release_convert_submit");
                        let convert_profile = moda_engine::prof::span("canvas.output.encoder_convert");
                        for surface in &slot.ring.encoder_surfaces {
                            match &surface.texture {
                                Some(destination) => target.finish(renderer, &mut command, destination, &instant.request, Some(&slot.frame)).map_err(device)?,
                                None => surface.record_copy(&mut command, &slot.frame),
                            }
                        }
                        drop(convert_profile);
                        if diagnostics { slot.diagnostic_gpu = moda_engine::gpu_profile::end_frame(&mut command); }
                        // All consumers of this worker's resolve texture are queued before its next draw.
                        #[cfg(feature = "frame-profile")]
                        crate::frame_profile::count("queue_submissions", 1);
                        let submission = command.submit(&renderer.queue);
                        if let Some(start) = draw_start { slot.work.draw_ms = start.elapsed().as_secs_f64() * 1000.; }
                        drop(output_profile);
                        drop(frame_profile);
                        if diagnostics {
                            for (name, ms) in moda_engine::prof::take_capture() { *slot.diagnostic_host.entry(name).or_default() += ms; }
                        }
                        if gpu_profile { moda_engine::gpu_profile::resolve_into_prof(&renderer.device, &renderer.queue); }
                        drop(job);
                        let (frame, slot) = current.take().unwrap();
                        if done.send(Ok((frame, slot, Some(submission)))).is_err() { break; }
                    }
                    Ok(())
                };
                let result = match budget.run(run) { Ok(result) => result, Err(refused) => Err(refused.into()) };
                if let Err(error) = result {
                    if crate::failure::cause::<wgpu::optional_allocations::Refused>(&error).is_some() {
                        moda_engine::scratch::discard_pending();
                        if let Some((frame, slot)) = current.take() {
                            let _ = done.send(Ok((frame, slot, None)));
                        }
                        // Return every queued lease, then exit after the
                        // export thread disconnects and joins this worker.
                        while let Ok((frame, _, _, _, slot)) = receive.recv() {
                            if done.send(Ok((frame as usize, slot, None))).is_err() { break; }
                        }
                    } else { let _ = done.send(Err(error)); }
                }
            }).map_err(|error| match error.kind() {
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::OutOfMemory => anyhow::Error::new(wgpu::optional_allocations::Refused),
                _ => anyhow::Error::new(error).context("start canvas frame worker"),
            })?;
            Ok::<_, anyhow::Error>(((send, completed), join))
        };
        let mut pool = CanvasWorkers::new();
        let mut joins = Vec::new();

        let mut retry_from = 0;
        let accepted = std::cell::Cell::new(0usize);
        let mut memory_refused = false;
        let mut schedule = Schedule::default();
        let mut admission_limit = worker_limit;
        let mut awaiting_publication = false;
        let mut lane_claims = Vec::new();
        loop {
        let result = (|| -> Result<()> {
        let mut pending: std::collections::VecDeque<(usize, f32, Instant, Option<wgpu::SubmissionIndex>)> = std::collections::VecDeque::new();
        // How many submitted frames wait for acceptance. A ring entry is busy from its frame's submission until
        // every consumer lets go of it. PNG writers take a copy at acceptance, so the whole ring can be on the
        // GPU. A video encoder retains an entry until reuse. Completion-driven native video fills the
        // bounded ring too: neither GPU completion nor VT handoff waits for host acceptance.
        // Every native entry is released before its next conversion can overwrite it.
        let handoff = |frame: usize, slot: &Slot, encoders: &mut [Encoder], parallel: bool, submission: &wgpu::SubmissionIndex| -> Result<Option<moda_engine::gpu_profile::GpuTime>> {
            let handoff_profile = moda_engine::prof::span("canvas.output.handoff");
            let started = Instant::now();
            let mut native = native_frames.borrow_mut();
            let native = &mut native[frame % serial_capacity];
            if let Some(previous) = native.frame.take() {
                #[cfg(test)]
                frame_event("release", previous);
                for (encoder, surface) in encoders.iter_mut().zip(&mut native.surfaces) {
                    encoder.release(previous as u64, surface).map_err(device)?;
                }
            }
            // NVENC reads the completed worker texture directly on the GPU.
            let copy_pixels = parallel && (native.surfaces.iter().any(|surface| surface.texture.is_some())
                || encoders.iter().any(Encoder::uses_pixel_readback));
            let submission = if copy_pixels {
                let convert_profile = moda_engine::prof::span("canvas.output.encoder_convert");
                // Only the export thread writes the encoder's ordered ring.
                // Optional worker pixels never reserve native encoder entries.
                if diagnostics { moda_engine::gpu_profile::begin_frame(&renderer.device, &renderer.queue, native.diagnostic_gpu.take()); }
                let mut copy = renderer.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("canvas encoder handoff") });
                for surface in &native.surfaces {
                    match &surface.texture {
                        Some(destination) => target.finish_from(&mut copy, &slot.frame, destination, Finish::Copy)?,
                        None => surface.record_copy(&mut copy, &slot.frame),
                    }
                }
                if diagnostics { native.diagnostic_gpu = moda_engine::gpu_profile::end_frame(&mut copy); }
                #[cfg(feature = "frame-profile")]
                crate::frame_profile::count("queue_submissions", 1);
                let copy = copy.finish();
                let _submit = moda_engine::prof::span("canvas.output.native_submit");
                let submission = renderer.queue.submit([copy]);
                drop(convert_profile);
                submission
            } else { submission.clone() };
            for (encoder, surface) in encoders.iter_mut().zip(&mut native.surfaces) {
                let source = surface.texture.clone().unwrap_or_else(|| slot.frame.clone());
                encoder.push(frame as u64, source, surface, &submission).map_err(device)?;
            }
            crate::timing::encoded(started);
            drop(handoff_profile);
            native.frame = Some(frame);
            #[cfg(test)]
            frame_event("push", frame);
            if copy_pixels && gpu_profile { moda_engine::gpu_profile::resolve_into_prof(&renderer.device, &renderer.queue); }
            let mut gpu_time = None;
            if let Some(gpu) = native.diagnostic_gpu.as_ref().filter(|_| copy_pixels && diagnostics) {
                let _wait = moda_engine::prof::span("canvas.output.wait_readback");
                let (send, receive) = std::sync::mpsc::channel();
                let buffers = gpu.readbacks().collect::<Vec<_>>();
                for buffer in &buffers {
                    let send = send.clone();
                    buffer.map_async(wgpu::MapMode::Read, .., move |result| { let _ = send.send(result); });
                }
                renderer.wait(Some(submission.clone()))?;
                for _ in buffers { receive.recv().context("canvas handoff timestamps")??; }
                gpu_time = Some(gpu.time());
            }
            Ok(gpu_time)
        };
        let mut collect = |(frame, time, frame_start, submission): (usize, f32, Instant, Option<wgpu::SubmissionIndex>), slots: &mut [Option<Slot>], encoders: &mut [Encoder], pool: &CanvasWorkers, workers: usize, in_flight: usize, schedule: &mut Schedule| -> Result<()> {
            let parallel = !pool.is_empty();
            let submission = match submission {
                Some(submission) => submission,
                None => {
                    let _wait = moda_engine::prof::span("canvas.output.wait_readback");
                    let (completed, slot, submission) = pool[frame % workers - 1].1.recv().context("canvas frame worker stopped")??;
                    ensure!(completed == frame, "canvas worker completion order");
                    slots[frame % in_flight] = Some(slot);
                    submission.ok_or(wgpu::optional_allocations::Refused)?
                }
            };
            let delivery_start = automatic.then(Instant::now);
            // Everything done to accept one frame; `canvas.output.wait_readback` and `canvas.output.deliver` are
            // inside it, so what they leave is this span's own.
            #[cfg(feature = "frame-profile")]
            let _profile_event = crate::frame_profile::event("canvas_collect", output_index(frame), time);
            let _accept_profile = moda_engine::prof::span("canvas.accept");
            let slot = slots[frame % in_flight].as_mut().expect("frame owns its ring entry");
            // A credit bounds raw pixels and compressed output until every destination finishes writing.
            let credit_profile = moda_engine::prof::span("canvas.output.wait_readback");
            let png_lease = (pngs || raws).then(|| std::sync::Arc::new(slot.ring.credit.acquire()));
            drop(credit_profile);
            // Worker submissions arrive in frame order. Arm native completion before
            // any exposure/pixel/timestamp readback can wait for this frame's GPU work.
            let encoder_gpu = if parallel && completion_handoff {
                handoff(frame, slot, encoders, true, &submission)?
            } else { None };
            #[cfg(test)]
            frame_event("collect", frame);
            let collected = {
                let _profile = moda_engine::prof::span("canvas.output.wait_readback");
                target.collect(&renderer, slot, submission.clone())
            };
            let (mut e, rgba, raw) = match collected {
                Ok(collected) => collected,
                // A video cannot skip a frame; per-frame files can, so the rest of the range renders.
                Err(error) if encoders.is_empty() && refused.skip(output_index(frame), &error) => {
                    for writer in png_writers.iter().chain(&raw_writers) { writer.skip(output_index(frame) as usize)?; }
                    vork_slang::progress::completed(rendered_frames + refused.count());
                    accepted.set(frame + 1);
                    return Ok(());
                }
                Err(error) => return Err(error),
            };
            if let Some(gpu) = encoder_gpu { slot.diagnostic_time.get_or_insert_default().append(gpu); }
            // A canvas shown as authored (`as_authored`) has no view that clips between its colours and the
            // display: a full channel is the colour asked for. A float canvas (a world, an authored view, an
            // image graph, linear light) and Display P3 code values shown as sRGB pass a view whose result can
            // run out of display range.
            if as_authored { e.clipped = None; }
            let _profile = moda_engine::prof::span("canvas.output.deliver");
            if let Some(path) = video_paths.iter().find(|path| mp4 && shot::mp4_destination(path)) {
                ensure_usage!(e.coverage == 1.,
                    "--out {}: MP4 needs an opaque picture and this canvas leaves pixels transparent at {time} s; set render.background to an opaque color for MP4, or write .mov (ProRes 4444 keeps transparency) or PNG frames",
                    path.display());
            }
            if !encoders.is_empty() && !completion_handoff {
                if let Some(gpu) = handoff(frame, slot, encoders, parallel, &submission)? {
                    slot.diagnostic_time.get_or_insert_default().append(gpu);
                }
            }
            let handoff_profile = moda_engine::prof::span("canvas.output.handoff");
            if let Some(rgba) = rgba {
                let rgba = std::sync::Arc::new(rgba);
                let lease = png_lease.as_ref().expect("PNG readback has a writer credit");
                for writer in &png_writers {
                    writer.push_canvas(output_index(frame) as usize, rgba.clone(), lease.clone())?;
                }
            }
            if let Some(raw) = raw {
                let raw = std::sync::Arc::new(raw);
                let lease = png_lease.as_ref().expect("picture readback has a writer credit");
                for writer in &raw_writers {
                    writer.push_canvas_float(output_index(frame) as usize, raw.clone(), lease.clone())?;
                }
            }
            drop(handoff_profile);
            drop(_profile);
            drop(_accept_profile);
            if diagnostics && (frame == 0 || frame + 1 == frames) {
                if let Some(memory) = renderer.device.generate_allocator_report() {
                    memory_frames.push(serde_json::json!({"frame": output_index(frame),
                        "allocated_bytes": memory.total_allocated_bytes,
                        "reserved_bytes": memory.total_reserved_bytes}));
                }
            }
            if rendered_frames == 0 { first_frame_ms = loop_start.elapsed().as_secs_f64() * 1000.; }
            metadata_frames.push(serde_json::json!({"index": output_index(frame), "time": time, "camera": null, "exposure": e.json()}));
            frame_exposures.push((output_index(frame) as usize, e));
            let mut sample = serde_json::json!({"frame": output_index(frame), "time": time, "frame_wall_ms": frame_start.elapsed().as_secs_f64() * 1000.});
            if diagnostics {
                for (name, ms) in moda_engine::prof::take_capture() { *slot.diagnostic_host.entry(name).or_default() += ms; }
                sample["breakdown"] = crate::canvas_timing::frame(&slot.diagnostic_host, slot.diagnostic_time);
                sample["host_spans_inclusive_ms"] = serde_json::to_value(&slot.diagnostic_host)?;
                slot.diagnostic_host.clear();
            }
            samples.push(sample);
            rendered_frames += 1;
            accepted.set(frame + 1);
            vork_slang::progress::completed(rendered_frames + refused.count());
            if let Some(start) = delivery_start {
                let now = Instant::now();
                slot.work.output_ms += (now - start).as_secs_f64() * 1000.;
                schedule.observe(frame, slot.work, now);
            }
            Ok(())
        };
        for frame in retry_from..frames {
            if stop.as_ref().is_some_and(|stop| stop.load(std::sync::atomic::Ordering::Relaxed)) && frame > 0 { interrupted = true; break; }
            let mut wanted = workers;
            let host = host_memory.as_deref();
            if !memory_refused && frame > 0 && workers == 1 && (awaiting_publication || forced_workers.is_some() && admission_limit > 1 && frames > 2) {
                if document.retained.borrow().as_ref().is_some_and(|scene| target.engine.published(scene.picture_scene()).is_some()) {
                    wanted = if automatic { 2 } else { admission_limit };
                } else if awaiting_publication {
                    target.engine.publish_masks(false);
                    schedule.cancel_trial(2);
                }
                awaiting_publication = false;
            } else if automatic && !memory_refused {
                match schedule.decide(workers, admission_limit, frames - frame, install_ms, diagnostics) {
                    Resize::Grow if workers == 1 => {
                        // Capture just the next serial draw. A direct placement replay
                        // may have no CPU plan for a replica to consume.
                        target.engine.publish_masks(true);
                        awaiting_publication = true;
                    }
                    Resize::Grow => wanted += 1,
                    Resize::Retire => wanted -= 1,
                    Resize::Keep => {}
                }
            }

            if wanted != workers {
                // Frame/slot/lane indices depend on the current moduli. Accept
                // every old frame before changing either mapping.
                while let Some(old) = pending.pop_front() { collect(old, &mut slots, &mut encoders, &pool, workers, in_flight, &mut schedule)?; }
                let previous = workers;
                let host = host.expect("optional workers have a memory guard");
                if wanted < workers {
                    // Shrink only after every output consumer has released its
                    // entry. Growth keeps the native encoder ring intact; the
                    // normal handoff can release those readers on next reuse.
                    let native_ring = &mut *native_frames.borrow_mut();
                    for previous in frame.saturating_sub(serial_capacity)..frame {
                        let native = &mut native_ring[previous % serial_capacity];
                        if let Some(index) = native.frame.take() {
                            for (encoder, surface) in encoders.iter_mut().zip(&mut native.surfaces) { encoder.release(index as u64, surface).map_err(device)?; }
                        }
                    }
                    stop_canvas_workers(pool.split_off(wanted - 1), joins.split_off(wanted - 1), &mut slots, &renderer)?;
                    workers = wanted;
                    let retained_slots = serial_capacity + crate::canvas_parallel::extra_slots(serial_capacity, capacity, workers);
                    // PNG/EXR writers own independent pixel copies until the
                    // slot credit returns. Keep that memory charged meanwhile.
                    for slot in slots[retained_slots..].iter().flatten() { slot.ring.credit.wait(); }
                    slots.truncate(retained_slots);
                    host.release_workers(lane_claims.pop().expect("one claim per replica"));
                    if workers == 1 { target.engine.publish_masks(false); }
                } else {
                    let (used, free) = crate::canvas_parallel::headroom(&renderer.device).unwrap_or((u64::MAX, 0));
                    wanted = wanted.min((free / used.max(frame_bytes).max(1)) as usize);
                    let sample = host.sample(None).ok_or(wgpu::optional_allocations::Refused)?;
                    let reserve = sample.reserve();
                    let bytes = sample.resident_peak().saturating_add(vork_scene::LOAD_STACK_BYTES as u64);
                    let admitted = (workers + 1..=wanted).rev().find(|&count| {
                        let extra = crate::canvas_parallel::extra_slots(in_flight, capacity, count);
                        host.claim_workers(sample, reserve, bytes.saturating_mul((count - workers) as u64),
                            slot_bytes.saturating_mul(extra as u64))
                    });
                    if diagnostics {
                        eprintln!("canvas concurrency: host admission at frame {frame}; worker_bytes={bytes}, available={}, source={}, reserve={reserve}, wanted={wanted}, admitted={}",
                            sample.available, sample.source, admitted.unwrap_or(workers));
                    }
                    if let Some(count) = admitted {
                        lane_claims.resize(count - 1, bytes);
                        host.watch(sample, reserve);
                        if workers == 1 { target.engine.publish_images(&shared_images); }
                        let extra = crate::canvas_parallel::extra_slots(in_flight, capacity, count);
                        slots.extend(optional_budget.as_ref().unwrap().run(|| (0..extra)
                            .map(|_| Some(target.slot(&renderer, pngs, raws, Vec::new()))).collect::<Vec<_>>())?);
                        for _ in workers..count {
                            let (worker, join) = spawn_worker()?;
                            pool.push(worker);
                            joins.push(join);
                        }
                        workers = count;
                    } else {
                        admission_limit = workers;
                        schedule.cancel_trial(workers + 1);
                        if workers == 1 { target.engine.publish_masks(false); }
                    }
                }
                in_flight = slots.len();
                schedule.resized(in_flight);
                if diagnostics && previous != workers {
                    let reason = if previous < workers { "host memory admitted" } else { "throughput retired lane" };
                    eprintln!("canvas concurrency: {previous} -> {workers} at frame {frame}; {reason}; slots={in_flight}");
                }
            }
            let window = if !pool.is_empty() || encoders.is_empty() || completion_handoff { in_flight } else { crate::constants::FRAMES_IN_FLIGHT.div_ceil(2).min(in_flight) };
            // Frames stay in flight: the oldest is accepted only when the window is full, so the GPU draws it while
            // the next ones are built.
            if pending.len() == window { collect(pending.pop_front().expect("a frame in flight"), &mut slots, &mut encoders, &pool, workers, in_flight, &mut schedule)?; }
            let frame_start = Instant::now();
            // Everything done to build and submit one frame: `canvas.evaluate`, `canvas.worlds`, `canvas.plan`,
            // `canvas.draw`, `canvas.output.convert_record` and `canvas.output.release_convert_submit` are inside
            // it, so a changing frame's time that none of them covers shows as this span's remainder.
            let frame_profile = moda_engine::prof::span("canvas.frame");
            let request = still.as_ref().map_or_else(|| vork_time::Request::Frame(output_index(frame), fps), |still| still.request.clone());
            let time = request.numeric().map_err(scene_error)?;
            let prepared_frame = if !pool.is_empty() {
                host_memory.as_ref().unwrap().check(frame)?;
                Some(crate::canvas_host_memory::with_workers(|| -> Result<_> {
                    let instant = document.frame(&request).map_err(scene_error)?;
                    let plan = document.shared_plan(&target.engine)?;
                    Ok((instant, plan))
                })?)
            } else { None };
            let mut draw_start = (automatic && prepared_frame.is_some()).then(Instant::now);
            if frame % workers != 0 {
                let (instant, plan) = prepared_frame.expect("worker receives a prepared frame");
                let mut slot = slots[frame % in_flight].take().expect("frame owns its ring entry");
                if let Some(start) = draw_start {
                    slot.work.plan_ms = (start - frame_start).as_secs_f64() * 1000.;
                    slot.work.output_ms = 0.;
                }
                drop(frame_profile);
                if diagnostics { slot.diagnostic_host = moda_engine::prof::take_capture(); }
                if pool[frame % workers - 1].0.send((frame as u32, time, instant, plan, slot)).is_err() {
                    // Initialization can fail before a worker accepts its first slot.
                    // Preserve that error rather than replacing it with channel closure.
                    return match pool[frame % workers - 1].1.recv() {
                        Ok(Err(error)) => Err(error),
                        _ => Err(anyhow::anyhow!("canvas frame worker stopped")),
                    };
                }
                pending.push_back((frame, time, frame_start, None));
                continue;
            }
            #[cfg(feature = "frame-profile")]
            let _profile_event = crate::frame_profile::event("canvas_submit", output_index(frame), time);
            let (instant, plan) = match prepared_frame {
                Some((instant, plan)) => (instant, Some(plan)),
                None => (document.frame(&request).map_err(scene_error)?, None),
            };
            let slot = slots[frame % in_flight].as_mut().expect("frame owns its ring entry");
            // Record the draw before waiting for the previous encoder payload in this ring entry.
            #[cfg(test)]
            frame_event("draw", frame);
            if diagnostics { moda_engine::gpu_profile::begin_frame(&renderer.device, &renderer.queue, slot.diagnostic_gpu.take()); }
            let mut command = if let Some(plan) = plan {
                let recording = target.engine.record_for_output();
                let drawn = target.engine.draw_shared(&plan, &shared_masks).map_err(device)?;
                let (drawn, prefix) = target.pictures.composite(&renderer, drawn, &instant, Some(output_index(frame)), None, recording.finish())?;
                target.encode_drawn(&renderer, drawn, &instant, slot, prefix)?
            } else { target.encode(&renderer, &document, &instant, output_index(frame), slot, || {
                if automatic { draw_start = Some(Instant::now()); }
            })? };
            let output_start = automatic.then(Instant::now);
            let output_profile = moda_engine::prof::span("canvas.output.release_convert_submit");
            let release_profile = moda_engine::prof::span("canvas.output.encoder_release");
            let mut native_guard = native_frames.borrow_mut();
            let native = &mut native_guard[frame % serial_capacity];
            if pool.is_empty() && let Some(previous) = native.frame.take() {
                #[cfg(test)]
                frame_event("release", previous);
                for (encoder, surface) in encoders.iter_mut().zip(&mut native.surfaces) { encoder.release(previous as u64, surface).map_err(device)?; }
            }
            drop(release_profile);
            let convert_profile = moda_engine::prof::span("canvas.output.encoder_convert");
            for surface in native.surfaces.iter().filter(|_| pool.is_empty()) {
                match &surface.texture {
                    Some(destination) => target.finish(&renderer, &mut command, destination, &instant.request, Some(&slot.frame)).map_err(device)?,
                    None => surface.record_copy(&mut command, &slot.frame),
                }
            }
            drop(convert_profile);
            drop(native_guard);
            let submit_start = automatic.then(Instant::now);
            if diagnostics { slot.diagnostic_gpu = moda_engine::gpu_profile::end_frame(&mut command); }
            #[cfg(feature = "frame-profile")]
            crate::frame_profile::count("queue_submissions", 1);
            let submission = command.submit(&renderer.queue);
            let submitted = automatic.then(Instant::now);
            // Native completion owns synchronization with VT. Queue it immediately,
            // then build later frames while GPU/codec/writer retain these bounded slots.
            // Readback still validates every exposure before the output can finish.
            if pool.is_empty() && completion_handoff { handoff(frame, slot, &mut encoders, false, &submission)?; }
            drop(output_profile);
            drop(frame_profile);
            if let (Some(draw), Some(output), Some(submit), Some(end)) = (draw_start, output_start, submit_start, submitted) {
                // Serial conversion writes the native ring; with replicas that
                // conversion and ordered handoff both run during collection.
                slot.work.plan_ms = (draw - frame_start).as_secs_f64() * 1000.;
                slot.work.draw_ms = ((output - draw) + (end - submit)).as_secs_f64() * 1000.;
                slot.work.output_ms = ((submit - output) + end.elapsed()).as_secs_f64() * 1000.;
            }
            if diagnostics { slot.diagnostic_host = moda_engine::prof::take_capture(); }
            if gpu_profile && !diagnostics {
                moda_engine::gpu_profile::resolve_into_prof(&renderer.device, &renderer.queue);
            }
            if moda_engine::prof::on() {
                eprintln!("canvas profile frame {frame} cumulative: {}", moda_engine::prof::report_json());
                eprintln!("canvas residency frame {frame} cumulative: {}", target.engine.profile_counters());
                eprintln!("canvas change kinds frame {frame} cumulative: {}", document.change_counters());
                if let Some(retained) = document.retained.borrow().as_ref() {
                    eprintln!("canvas plan cache frame {frame} cumulative: {}", retained.profile_counters());
                }
            }
            pending.push_back((frame, time, frame_start, Some(submission)));
        }
        while let Some(frame) = pending.pop_front() { collect(frame, &mut slots, &mut encoders, &pool, workers, in_flight, &mut schedule)?; }
        Ok(())
        })();
        if result.as_ref().err().is_some_and(|error| crate::failure::cause::<wgpu::optional_allocations::Refused>(error).is_some()) {
            stop_canvas_workers(std::mem::take(&mut pool), std::mem::take(&mut joins), &mut slots, &renderer)?;
            retry_from = accepted.get();
            target.engine.publish_masks(false);
            workers = 1;
            slots.truncate(serial_capacity);
            in_flight = slots.len();
            crate::gpu_memory::wait(&renderer.device, &renderer.queue, &renderer.errors, None)?;
            if let Some(host) = &host_memory { host.release_workers(lane_claims.drain(..).sum()); }
            memory_refused = true;
            if diagnostics { eprintln!("canvas concurrency: memory reservation refused; retry serial at frame {retry_from}"); }
            continue;
        }
        stop_canvas_workers(std::mem::take(&mut pool), std::mem::take(&mut joins), &mut slots, &renderer)?;
        return result;
        }
    });
    target.engine.publish_masks(false);
    // Join all output workers even when a render or another destination failed.
    let png_finished: Vec<_> = png_writers.into_iter().chain(raw_writers).map(|writer| writer.finish()).collect();
    result?;
    png_finished.into_iter().collect::<Result<()>>()?;
    let finish_profile = moda_engine::prof::span("canvas.output.finish");
    let finish_start = Instant::now();
    // Finish every accepted stream before reporting an encoder error.
    let finished: Vec<Result<()>> = encoders.iter_mut().map(|encoder| encoder.finish().map(|_| ()).map_err(device)).collect();
    if !encoders.is_empty() { crate::timing::encoded(finish_start); }
    let encoder_finish_ms = finish_start.elapsed().as_secs_f64() * 1000.;
    finished.into_iter().collect::<Result<()>>()?;
    let frames = if interrupted { rendered_frames } else { frames };
    drop(finish_profile);
    renderer.errors.check().map_err(device)?;
    // Every frame is drawn, so the pipelines this canvas needed all exist (a later frame can be the
    // first to use one): save what it compiled, once (`Renderer::save_pipelines_soon`). A kept-alive
    // worker saves while it waits for the next request, and a one-shot's exit finishes the save
    // after delivery; optional scene specialization never delays completion.
    renderer.save_pipelines_soon();
    // The other frames are written; withholding meta.json leaves the output marked incomplete, and the
    // exit status reports the refusals.
    // An interrupted render's count is the frames it delivered; the refused ones were rendered too.
    refused.finish(if interrupted { frames + refused.count() } else { frames })?;
    if gpu_profile {
        eprintln!("canvas counted GPU work: {}", serde_json::to_string(&moda_engine::gpu_profile::take_counters())?);
    }
    if moda_engine::prof::on() {
        eprintln!("canvas host profile (inclusive process totals): {}", moda_engine::prof::report_json());
    }
    if let Some(sound) = &sound {
        sound.mux(&out, frames as u64)?;
    }
    let mut sidecars = Vec::new();
    let mut exposures = Vec::new();
    let mut pending_metadata = Vec::new();
    let render_settings = serde_json::json!({"profile": profile_record, "resolution": [w, h], "frames": frames, "canvas": true});
    for path in out.iter().filter(|_| output_run.is_some()) {
        let meta_path = shot::sidecar_path(path, "meta.json");
        // The exposure report and the display's name describe the frame encoded for the display: PNG and video.
        // An EXR holds the picture a reader takes, unclamped, and names its colour itself.
        let viewed = !is_exr(path);
        let frame_rows: Vec<serde_json::Value> = metadata_frames.iter().cloned().map(|mut frame| {
            if !viewed { frame["exposure"] = serde_json::Value::Null; }
            frame
        }).collect();
        let mut meta = serde_json::json!({
            "version": 3, "fps": fps, "frames": frame_rows, "object_ids": {},
            "warnings": warnings.snapshot(), "profile": profile_record, "render_settings": render_settings, "canvas": {"size": [w, h]},
        });
        if shot::mp4_destination(path) && let Some(identity) = &export_identity {
            shot::export_identity::record(&mut meta, identity, path)?;
        }
        if let Some(summary) = shot::exposure_summary(&frame_exposures).filter(|_| viewed) {
            meta["exposure"] = summary.clone();
            exposures.push(serde_json::json!({"output": path, "exposure": summary}));
        }
        // Every viewed picture's sidecar names the display it is encoded for, sRGB included, as its PNG's colour
        // chunk and its video stream do. An EXR is not encoded for a display and names its colour itself.
        if viewed {
            meta["output_display"] = serde_json::to_value(display.display)?;
            if !is_video(path) { meta["output_png_depth"] = 8.into(); }
        }
        if let Some(parent) = meta_path.parent() { std::fs::create_dir_all(parent)?; }
        pending_metadata.push((meta_path.clone(), serde_json::value_to_vec_pretty(&meta)?));
        sidecars.push(meta_path);
    }
    let pipeline_creation_ms: Vec<_> = renderer.pipeline_timings().into_iter().filter_map(|(name, ms)| {
        let delta = ms - pipeline_timings_before.get(&name).copied().unwrap_or(0.);
        (delta > 0.).then_some((name, delta))
    }).collect();
    let stages_end = Instant::now();
    let mut report = serde_json::json!({
        "version": 4, "device": renderer.adapter_info.name, "backend": format!("{:?}", renderer.adapter_info.backend),
        "pipeline_cache_supported": renderer.pipeline_cache_supported(), "pipeline_cache_status": renderer.pipeline_cache_status,
        "pipeline_cache_input_bytes": renderer.pipeline_cache_input_bytes, "pipeline_cache_rejection": renderer.pipeline_cache_rejection,
        "pipeline_creation_ms": pipeline_creation_ms,
        "frames": frames, "frame_start": at.is_none().then_some(first_frame), "still_time": at, "resolution": [w, h], "canvas": true,
        "png_write": pngs, "exr_write": raws, "video_write": video, "output": out,
        "encoder_finish_ms": encoder_finish_ms, "scene_parse_ms": scene_parse_ms, "install_ms": install_ms,
        "device_ms": if device_reused { 0. } else { renderer.startup.as_secs_f64() * 1000. },
        "first_frame_ms": first_frame_ms, "ready_ms": (loop_start - wall).as_secs_f64() * 1000.,
        "loop_wall_ms": (stages_end - loop_start).as_secs_f64() * 1000., "wall_ms": (stages_end - wall).as_secs_f64() * 1000.,
    });
    // The completed rows already are JSON values; json! would serialize a
    // borrowed copy of every per-frame diagnostic into a second value tree.
    report["samples"] = serde_json::Value::Array(samples);
    // Canvas painting records no per-pass GPU spans; its compile and encode stages are timed.
    let (stages, pipeline_cache_saves) = crate::timing::stages(&renderer.device, wall, stages_end, crate::timing::StageTotals {
        setup_ms: (loop_start - wall).as_secs_f64() * 1000., first_frame_ms, frames_ms: (stages_end - loop_start).as_secs_f64() * 1000.,
        gpu_spans_ms: Vec::new(), denoise_busy_ms: 0., denoise_wait_ms: 0.,
    });
    report["stages"] = stages;
    report["pipeline_cache_saves"] = serde_json::to_value(pipeline_cache_saves)?;
    report["pipeline_cache_save_pending"] = renderer.compiler.save_pending().into();
    if diagnostics {
        report["breakdown"] = crate::canvas_timing::total(report["samples"].as_array().unwrap(), setup_host.as_ref().unwrap(), encoder_start_ms, encoder_finish_ms);
        report["stages"]["gpu_ms"] = report["breakdown"]["gpu_busy_ms"].clone();
        report["memory_frames"] = memory_frames.into();
        report["retained_gpu_bytes"] = renderer.residency.retained_bytes().into();
    }
    report["render_settings"] = render_settings;
    report["device_reused"] = device_reused.into();
    report["install_reused"] = install_reused.into();
    if diagnostics { eprintln!("render: {workers} canvas frame worker(s)"); }
    report["interrupted"] = interrupted.into();
    report["warnings"] = serde_json::to_value(warnings.snapshot())?;
    report["sidecars"] = serde_json::to_value(sidecars)?;
    report["exposure"] = exposures.into();
    // Every written render reports where its time went, beside its metadata.
    if output_run.is_some() {
        let mut timing_paths = Vec::new();
        for path in &out {
            let timing_path = shot::write_sidecar(path, "timings.json", &serde_json::value_to_vec_pretty(&report)?)?;
            timing_paths.push(timing_path);
        }
        if let Some(sidecars) = report["sidecars"].as_array_mut() {
            sidecars.extend(timing_paths.iter().map(|path| serde_json::json!(path)));
        }
    }
    if let Some(run) = &output_run { run.complete(&pending_metadata)?; }
    #[cfg(feature = "frame-profile")]
    crate::frame_profile::report(&report);
    Ok(RenderedShot::canvas(report, encoders, renderer, Kept { identity, document, target }))
}

#[cfg(test)]
#[path = "canvas_kept_identity_tests.rs"]
mod kept_identity_tests;

#[cfg(test)]
#[path = "canvas_text_qa_tests.rs"]
mod text_qa_tests;

#[cfg(test)]
#[path = "canvas_paint_tests.rs"]
mod paint_tests;

#[cfg(test)]
mod canvas_regressions {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn canvas_png_sequences_finish_with_selected_frames() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-canvas-png-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let scene = root.join("scene.json");
        std::fs::write(&scene, serde_json::to_vec(&json!({
            "version": 2, "composition": {"space": "2d"},
            "render": {"resolution": [32, 32], "fps": 30, "duration": 1, "background": "red"},
            "nodes": [], "animation":[{"tracks":[{"path":["render","background"], "keys":[[0,"red"],[1,"blue"]]}]}]
        }))?)?;
        let pixels = |path: PathBuf| -> Result<Vec<u8>> {
            let mut decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?)).read_info()?;
            assert_eq!(decoder.info().srgb, Some(png::SrgbRenderingIntent::Perceptual));
            let mut bytes = vec![0; decoder.output_buffer_size().unwrap()];
            let info = decoder.next_frame(&mut bytes)?;
            bytes.truncate(info.buffer_size());
            Ok(bytes)
        };
        let rendered = crate::shot::Shot { scene: scene.clone(), out: vec![root.join("a").into(), root.join("b").into()],
            frame_start: Some(3), frame_end: Some(10), diagnostics: true, ..Default::default() }.render()?;
        assert_eq!(rendered.report["frames"], 8);
        let rows = rendered.report["samples"].as_array().unwrap();
        assert_eq!(rows[0]["frame"], 3);
        for row in rows {
            assert!(row["breakdown"]["evaluation_ms"].as_f64().unwrap() > 0.);
            assert!(row["breakdown"]["gpu_busy_ms"].as_f64().unwrap() > 0.);
            assert_eq!(row["breakdown"]["gpu_untimed_passes"], 0);
        }
        let gpu_sum: f64 = rows.iter().map(|row| row["breakdown"]["gpu_busy_ms"].as_f64().unwrap()).sum();
        assert!((rendered.report["breakdown"]["gpu_busy_ms"].as_f64().unwrap() - gpu_sum).abs() < 1e-6);
        for frame in 3..=10 {
            let name = format!("{frame:06}.png");
            assert_eq!(pixels(root.join("a").join(&name))?, pixels(root.join("b").join(&name))?);
        }
        assert_ne!(pixels(root.join("a/000003.png"))?, pixels(root.join("a/000010.png"))?);
        assert!(!root.join("a/000002.png").exists());
        drop(rendered);
        let still = crate::shot::Shot { scene, out: vec![root.join("still.png").into(), root.join("at").into()],
            at: Some("0.2".parse()?), ..Default::default() }.render()?;
        assert!(still.report.get("breakdown").is_none());
        assert!(!moda_engine::prof::capturing());
        assert!(!moda_engine::gpu_profile::enabled());
        let expected = pixels(root.join("a/000006.png"))?;
        assert_eq!(pixels(root.join("still.png"))?, expected);
        assert_eq!(pixels(root.join("at/000006.png"))?, expected);
        drop(still);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// The precision rule (`Document::float`): only a world, an authored view, an image graph or a linear
    /// working space makes a canvas float.
    #[test]
    fn a_canvas_is_float_only_when_a_world_or_a_view_needs_it() -> Result<()> {
        let float = |scene: Value| -> Result<bool> {
            Ok(Document::new(&Scene::from_json(&serde_json::to_vec(&scene)?)?, Path::new("."))?.float())
        };
        let shape = json!([{"type": "path", "d": "M0 0H16V32H0Z", "fill": "red"}]);
        assert!(!float(json!({"version": 2, "composition": {"space": "2d"}, "render": {"resolution": [32, 32]}, "nodes": shape}))?,
            "a plain canvas keeps the 8-bit picture");
        assert!(float(json!({"version": 2, "composition": {"space": "2d", "view": {"program": "std/standard", "inputs": {}}},
            "render": {"resolution": [32, 32]}, "nodes": shape}))?, "an authored view reads float colour");
        assert!(float(json!({"version": 2, "composition": {"space": "2d", "compositor": ["grade"]}, "render": {"resolution": [32, 32]}, "nodes": shape,
            "materials": {"grade": {"program": "grade"}},
            "programs": {"grade": {"source": "import vork; public struct Inputs {}; public float4 composite(CompositeContext ctx, Inputs inputs) { return compositor_sample(ctx.uv); }"}}}))?,
            "an image graph reads float colour");
        assert!(float(json!({"version": 2, "composition": {"space": "2d"}, "render": {"resolution": [32, 32]},
            "buffers": {"p": {"type": "vec3", "data": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}},
            "geometry": {"g": {"attributes": {"POSITION": "p"}}},
            "nodes": [{"name": "shot", "composition": {"space": "3d"}, "children": [
                {"type": "camera", "name": "cam", "translation": [0, 0, 5]}, {"type": "mesh", "name": "tri", "geometry": "g"}]}]}))?,
            "a world's picture arrives floating point");
        // A working space: linear light needs the precision; an encoded space's code values fit eight bits.
        for (space, expected) in [("sRGB", false), ("Display P3", false), ("Linear Rec.709", true), ("Linear Rec.2020", true), ("ACEScg", true)] {
            assert_eq!(float(json!({"version": 2, "composition": {"space": "2d", "color_space": space}, "render": {"resolution": [32, 32]}, "nodes": shape}))?,
                expected, "{space}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn fitted_image_edges_keep_their_pixels_under_pan_and_crop() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let scene = dir.path().join("scene.json");
        let out = dir.path().join("frame.png");
        image::GrayImage::from_pixel(24, 16, image::Luma([255])).save(dir.path().join("white.png"))?;
        for (x, y, view) in [(0, 0, [0, 0, 96, 72]), (5, 3, [0, 0, 96, 72]), (16, 12, [12, 8, 72, 56])] {
            let document = json!({"version": 2, "composition": {"space": "2d", "viewBox": view},
                "render": {"resolution": [view[2], view[3]], "background": "black"}, "images": {"white": {"file": "white.png"}},
                "nodes": [{"type": "path", "d": "M0 0h33v23h-33Z", "translation": [x, y, 0], "fill": {"image": "white", "fit": "contain"}}]});
            std::fs::write(&scene, serde_json::to_vec(&document)?)?;
            crate::shot::Shot { scene: scene.clone(), out: vec![out.clone().into()], ..Default::default() }.render()?;
            let (left, top) = (x - view[0], y - view[1]);
            for (px, py, pixel) in image::open(&out)?.to_rgb8().enumerate_pixels() {
                let covered = (left..left + 33).contains(&px) && (top + 1..=top + 22).contains(&py);
                assert_eq!(pixel.0, [if covered { 255 } else { 0 }; 3], "view {view:?} at ({px},{py})");
            }
        }
        Ok(())
    }

    /// A canvas's image graph runs over its picture, on sRGB code values, before its view: an animated input
    /// follows the frame's time, a later entry reads a named half-resolution step, and a result that is not
    /// finite refuses the frame.
    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn a_canvas_runs_its_image_graph_before_its_view() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-canvas-graph-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let render = |name: &str, at: &str, compositor: Value, programs: Value| -> Result<Vec<u8>> {
            let document = json!({"version": 2, "composition": {"space": "2d", "compositor": compositor},
                "render": {"resolution": [32, 32], "duration": 2, "background": "rgb(128, 64, 32)"}, "nodes": [],
                "materials": {"grade": {"program": "grade"}, "pass": {"program": "pass"}, "add": {"program": "add", "inputs": {"small": {"step": "small"}}}},
                "programs": programs,
                "animation": [{"tracks": [{"path": ["materials", "grade", "inputs", "gain"], "keys": [[0, 0.5], [1, 1.0]]}]}]});
            let scene = root.join(format!("{name}.json"));
            std::fs::write(&scene, serde_json::to_vec(&document)?)?;
            let out = root.join(format!("{name}.png"));
            drop(crate::shot::Shot { scene, out: vec![out.clone().into()], at: Some(at.parse()?), ..Default::default() }.render()?);
            Ok(image::open(&out)?.to_rgba8().into_raw())
        };
        let programs = json!({
            "grade": {"source": "import vork; public struct Inputs { float gain = 1.; }; public float4 composite(CompositeContext ctx, Inputs inputs) { float4 c = compositor_sample(ctx.uv); return float4(c.rgb * inputs.gain, c.a); }"},
            "pass": {"source": "import vork; public struct Inputs {}; public float4 composite(CompositeContext ctx, Inputs inputs) { return compositor_sample(ctx.uv); }"},
            "add": {"source": "import vork; public struct Inputs { Image small; }; public float4 composite(CompositeContext ctx, Inputs inputs) { float4 c = compositor_sample(inputs.small, ctx.uv); return float4(c.rgb * 0.5, c.a); }"}});
        // The graph reads the picture's code values: half of code 128 is code 64, not half its light.
        let half = render("half", "0", json!(["grade"]), programs.clone())?;
        assert!(half.chunks_exact(4).all(|pixel| pixel == [64, 32, 16, 255]), "{:?}", &half[..4]);
        let whole = render("whole", "1", json!(["grade"]), programs.clone())?;
        assert!(whole.chunks_exact(4).all(|pixel| pixel == [128, 64, 32, 255]), "{:?}", &whole[..4]);
        // A named step at half resolution, read by the last entry and resampled to the output.
        let stepped = render("stepped", "1", json!([{"material": "pass", "name": "small", "resolution": 0.5}, "add"]), programs.clone())?;
        assert!(stepped.chunks_exact(4).all(|pixel| pixel == [64, 32, 16, 255]), "{:?}", &stepped[..4]);
        let mut broken = programs;
        broken["grade"] = json!({"source": "import vork; public struct Inputs { float gain = 1.; }; public float4 composite(CompositeContext ctx, Inputs inputs) { float4 c = compositor_sample(ctx.uv); return float4(float3(1. / (c.r - c.r)), c.a); }"});
        let error = format!("{:#}", render("broken", "0", json!(["grade"]), broken).err().context("a non-finite canvas graph rendered")?);
        assert!(error.contains("composition.compositor or composition.view returned a non-finite colour"), "{error}");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// A canvas's EXR is its picture after its view, as a reader takes it. An 8-bit canvas stays 8-bit: its EXR
    /// holds the codes its PNG has. A float canvas's EXR keeps what the 8-bit frame clamps.
    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn a_canvas_writes_its_picture_after_its_view_to_exr() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-canvas-exr-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let render = |name: &str, composition: Value| -> Result<(Vec<u8>, Vec<f32>)> {
            let document = json!({"version": 2, "composition": composition,
                "render": {"resolution": [32, 32], "background": "rgb(240, 64, 32)"}, "nodes": []});
            let scene = root.join(format!("{name}.json"));
            std::fs::write(&scene, serde_json::to_vec(&document)?)?;
            let (png, exr) = (root.join(format!("{name}.png")), root.join(format!("{name}.%06d.exr")));
            drop(crate::shot::Shot { scene, out: vec![png.clone().into(), exr.into()], at: Some("0".parse()?), ..Default::default() }.render()?);
            Ok((image::open(&png)?.to_rgba8().into_raw(), image::open(root.join(format!("{name}.000000.exr")))?.to_rgba32f().into_raw()))
        };
        let (png, exr) = render("plain", json!({"space": "2d"}))?;
        assert!(png.chunks_exact(4).all(|pixel| pixel == [240, 64, 32, 255]), "{:?}", &png[..4]);
        let codes = [240f32 / 255., 64. / 255., 32. / 255., 1.];
        assert!(exr.chunks_exact(4).all(|pixel| pixel.iter().zip(codes).all(|(a, b)| a.to_bits() == b.to_bits())), "{:?}", &exr[..4]);
        // Two stops up: the frame clamps red at code 255; the EXR keeps the view's code value above one.
        let (png, exr) = render("lifted", json!({"space": "2d", "view": {"program": "std/standard", "inputs": {"exposure": 2}}}))?;
        assert!(png.chunks_exact(4).all(|pixel| pixel[0] == 255), "{:?}", &png[..4]);
        assert!(exr.chunks_exact(4).all(|pixel| pixel[0] > 1.5 && pixel[3] == 1.), "{:?}", &exr[..4]);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// A canvas blends in the encoding its composition names. Its colours, its images and a world's picture are
    /// sRGB code values, converted before they are filtered or blended; its image graph works in that encoding;
    /// its view returns the picture to the display; an EXR stores what a reader takes, in the working space by
    /// default.
    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn a_canvas_blends_in_its_working_space() -> Result<()> {
        use v2::WorkingColorSpace as S;
        let root = std::env::temp_dir().join(format!("vork-canvas-spaces-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        image::RgbaImage::from_pixel(8, 8, image::Rgba([255, 0, 0, 128])).save(root.join("red.png"))?;
        // A grey that barely covers, black beside black beside white, a black texel beside a white one, and a
        // one-texel checker of the two.
        image::RgbaImage::from_pixel(8, 8, image::Rgba([128, 128, 128, 1])).save(root.join("faint.png"))?;
        image::RgbaImage::from_fn(3, 1, |x, _| image::Rgba(if x == 2 { [255, 255, 255, 255] } else { [0, 0, 0, 255] })).save(root.join("third.png"))?;
        image::RgbaImage::from_fn(2, 1, |x, _| image::Rgba(if x == 0 { [0, 0, 0, 255] } else { [255, 255, 255, 255] })).save(root.join("ramp.png"))?;
        image::RgbaImage::from_fn(512, 512, |x, y| image::Rgba(if (x + y) % 2 == 0 { [0, 0, 0, 255] } else { [255, 255, 255, 255] })).save(root.join("checker.png"))?;
        // A still at time zero: its PNG's bytes and its EXR's floats.
        let render = |stem: &str, document: &Value, display: Option<v2::OutputDisplay>| -> Result<(Vec<u8>, Vec<f32>)> {
            let scene = root.join(format!("{stem}.json"));
            std::fs::write(&scene, serde_json::to_vec(document)?)?;
            let (png, exr) = (root.join(format!("{stem}.png")), root.join(format!("{stem}.%06d.exr")));
            let frame = v2::OutputDestination { display, ..png.clone().into() };
            drop(crate::shot::Shot { scene, out: vec![frame, exr.into()], at: Some("0".parse()?), ..Default::default() }.render()?);
            Ok((image::open(&png)?.to_rgba8().into_raw(), image::open(root.join(format!("{stem}.000000.exr")))?.to_rgba32f().into_raw()))
        };
        let centre = (16 * 32 + 16) * 4;
        let codes = |colour: [f32; 3]| colour.map(|v| (v.clamp(0., 1.) * 255.).round() as u8);
        let near = |pixel: &[u8], expected: [u8; 3], within: u8| pixel[3] == 255 && pixel[..3].iter().zip(expected).all(|(a, b)| a.abs_diff(b) <= within);
        let masked_group = json!({"version":2,"composition":{"space":"2d"},
            "render":{"resolution":[32,32],"background":"black"},"nodes":[
                {"mask":"cut","children":[
                    {"type":"path","d":"M0 0H32V32H0Z","fill":"red"},
                    {"type":"path","d":"M0 0H32V32H0Z","fill":"blue"}]},
                {"name":"cut","type":"path","d":"M0 0H32V32H0Z","fill":"black","mask-type":"alpha","opacity":0.5}]});
        let (masked, _) = render("group-alpha-mask", &masked_group, None)?;
        assert!(near(&masked[centre..][..4], [0, 0, 128], 1), "group mask: {:?}", &masked[centre..][..4]);
        let mut isolated = masked_group.clone();
        isolated["nodes"][0]["children"] = json!([{"isolation":"isolate","children":masked_group["nodes"][0]["children"]}]);
        isolated["nodes"][1]["fill"] = json!("white");
        isolated["nodes"][1]["mask-type"] = json!("luminance");
        let (masked, _) = render("isolated-luminance-mask", &isolated, None)?;
        assert!(near(&masked[centre..][..4], [0, 0, 128], 1), "isolated mask child: {:?}", &masked[centre..][..4]);
        let decode = |c: f32| if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
        let encode = |c: f32| if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1. / 2.4) - 0.055 };
        let alpha = 128f32 / 255.;
        for (index, (name, space)) in [("sRGB", S::Srgb), ("Display P3", S::DisplayP3), ("Linear Rec.709", S::LinearRec709),
            ("Linear Rec.2020", S::LinearRec2020), ("ACEScg", S::ACEScg)].into_iter().enumerate() {
            let canvas = |fields: Value| -> Value {
                let mut document = json!({"version": 2, "composition": {"space": "2d", "color_space": name}, "render": {"resolution": [32, 32]}});
                for (field, value) in fields.as_object().expect("fields") {
                    if field == "background" { document["render"]["background"] = value.clone(); } else { document[field.as_str()] = value.clone(); }
                }
                document
            };
            // Red at 128/255 over blue, blended in the working encoding: a colour, and an image of that colour.
            let (red, blue) = (S::Srgb.convert_rgb(space, [1., 0., 0.]), S::Srgb.convert_rgb(space, [0., 0., 1.]));
            let blended: [f32; 3] = std::array::from_fn(|i| alpha * red[i] + (1. - alpha) * blue[i]);
            let shown = codes(space.convert_rgb(S::Srgb, blended));
            for (kind, fill) in [("colour", json!("#ff000080")), ("image", json!({"image": "red", "fit": "fill"}))] {
                let mut document = canvas(json!({"background": "blue", "nodes": [{"type": "path", "d": "M0 0H32V32H0Z", "fill": fill}]}));
                if kind == "image" { document["images"] = json!({"red": {"file": "red.png"}}); }
                let (png, exr) = render(&format!("{kind}-{index}"), &document, None)?;
                assert!(near(&png[centre..][..4], shown, 2), "{name} {kind}: {:?} for {shown:?}", &png[centre..][..4]);
                // The EXR stores the reader's picture in the working space: the blend itself.
                let stored = &exr[centre..][..4];
                assert!(stored[3] == 1. && stored[..3].iter().zip(blended).all(|(a, b)| (a - b).abs() < 0.01), "{name} {kind}: {stored:?} for {blended:?}");
                // Written for the display its encoding names, a canvas is its own code values.
                if let Some(own) = space.display().filter(|_| kind == "colour") {
                    let (png, _) = render(&format!("own-{index}"), &document, Some(own))?;
                    assert!(near(&png[centre..][..4], codes(blended), 1), "{name} for its own display: {:?} for {:?}", &png[centre..][..4], codes(blended));
                }
            }
            // An image is resampled in the working encoding. Reduced: a checker of black and white texels drawn
            // sixteen times smaller is their mean. Enlarged: two texels stretched across the canvas mix by the
            // distance between their centres. Each is a mix of code values on an encoded canvas and of light on
            // a linear one. (An sRGB canvas's small still is first enlarged by Moda's bitmap sizing, on its
            // sRGB bytes, which moves the enlarged mix by a code or two.)
            for (image, weight) in [("checker", 0.5f32), ("ramp", (16.5 / 32. - 0.25) / 0.5)] {
                let mut document = canvas(json!({"background": "black",
                    "nodes": [{"type": "path", "d": "M0 0H32V32H0Z", "fill": {"image": image, "fit": "fill"}}]}));
                document["images"] = json!({});
                document["images"][image] = json!({"file": format!("{image}.png")});
                let (png, _) = render(&format!("{image}-{index}"), &document, None)?;
                let mixed = codes([if space.is_linear() { encode(weight) } else { weight }; 3]);
                assert!(near(&png[centre..][..4], mixed, 4), "{name} resampling {image}: {:?} for {mixed:?}", &png[centre..][..4]);
            }
            // A canvas outside sRGB takes an image's decoded texels. A texel's colour survives low coverage:
            // grey 128 at alpha 1/255 over black is that grey's light at that coverage, where its 8-bit
            // associated byte would read as white. And a mip level counts every texel of an odd extent: black,
            // black and white drawn one pixel wide are a third white.
            if space.is_linear() {
                let mut document = canvas(json!({"background": "black", "nodes": [{"type": "path", "d": "M0 0H32V32H0Z", "fill": {"image": "faint", "fit": "fill"}}]}));
                document["images"] = json!({"faint": {"file": "faint.png"}});
                let (png, _) = render(&format!("faint-{index}"), &document, None)?;
                let faint = codes([encode(decode(128. / 255.) / 255.); 3]);
                assert!(near(&png[centre..][..4], faint, 1), "{name} low coverage: {:?} for {faint:?}", &png[centre..][..4]);
            }
            if space != S::Srgb {
                let mut document = canvas(json!({"background": "black", "nodes": [{"type": "path", "d": "M16 0H17V32H16Z", "fill": {"image": "third", "fit": "fill"}}]}));
                document["images"] = json!({"third": {"file": "third.png"}});
                let (png, _) = render(&format!("third-{index}"), &document, None)?;
                let third = codes([if space.is_linear() { encode(1. / 3.) } else { 1. / 3. }; 3]);
                assert!(near(&png[centre..][..4], third, 4), "{name} odd extent: {:?} for {third:?}", &png[centre..][..4]);
            }
            // The image graph works in the working encoding: half of an sRGB canvas's code values, half of a
            // linear canvas's light.
            if space == S::Srgb || space == S::LinearRec709 {
                let mut document = canvas(json!({"background": "rgb(128, 64, 32)", "nodes": [], "materials": {"half": {"program": "half"}},
                    "programs": {"half": {"source": "import vork; public struct Inputs {}; public float4 composite(CompositeContext ctx, Inputs inputs) { float4 c = compositor_sample(ctx.uv); return float4(c.rgb * 0.5, c.a); }"}}}));
                document["composition"]["compositor"] = json!(["half"]);
                let (png, _) = render(&format!("graph-{index}"), &document, None)?;
                let halved = [128f32, 64., 32.].map(|c| c / 255.).map(|c| if space == S::Srgb { c * 0.5 } else { encode(decode(c) * 0.5) });
                assert!(near(&png[centre..][..4], codes(halved), 1), "{name} graph: {:?} for {:?}", &png[centre..][..4], codes(halved));
            }
            // A world's picture is its view's sRGB code values, converted like an image: opaque, it is the same
            // picture in every working space (within the world's own dither).
            let document = canvas(json!({"materials": {"fill": {"program": "fill"}},
                "programs": {"fill": {"source": "import vork; public struct Inputs {}; public float4 composite(CompositeContext ctx, Inputs inputs) { return float4(.5, .25, .125, 1.); }"}},
                "nodes": [{"name": "shot", "composition": {"space": "3d", "compositor": ["fill"], "view": {"program": "std/standard", "inputs": {}}},
                    "children": [{"type": "camera", "name": "cam", "translation": [0, 0, 5]}]}]}));
            let (png, _) = render(&format!("world-{index}"), &document, None)?;
            let world = codes([0.5f32, 0.25, 0.125].map(encode));
            assert!(near(&png[centre..][..4], world, 3), "{name} world: {:?} for {world:?}", &png[centre..][..4]);
            // A luminance mask is the luminance of the mask's picture: black under half-covering white is
            // code 0.5 where the canvas blends sRGB code values, and half the light, code 0.735, where it
            // blends linear light. The masked white then shows through that much, blended the same way.
            if space == S::Srgb || space == S::LinearRec709 {
                let document = canvas(json!({"background": "black", "nodes": [
                    {"type": "path", "d": "M0 0H32V32H0Z", "fill": "white", "mask": "cut"},
                    {"name": "cut", "type": "path", "d": "M12 -20H44V52H12Z", "fill": "black", "stroke": "white", "stroke-width": 16, "stroke-opacity": 0.5}]}));
                let (png, _) = render(&format!("mask-{index}"), &document, None)?;
                let through = if space == S::Srgb { 0.5 } else { encode(encode(0.5)) };
                assert!(near(&png[centre..][..4], codes([through; 3]), 4), "{name} mask: {:?} for {:?}", &png[centre..][..4], codes([through; 3]));
            }
        }
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// What a world that reads a canvas takes: the canvas's picture after its view, as sRGB code values, from
    /// any working space. An sRGB canvas's drawn picture is that picture; another encoding's passes its view.
    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn a_reader_takes_a_canvas_as_srgb_code_values() -> Result<()> {
        use v2::WorkingColorSpace as S;
        let renderer = Renderer::new()?;
        let alpha = 128f32 / 255.;
        for (name, space) in [("sRGB", S::Srgb), ("Display P3", S::DisplayP3), ("Linear Rec.709", S::LinearRec709),
            ("Linear Rec.2020", S::LinearRec2020), ("ACEScg", S::ACEScg)] {
            let scene = Scene::from_json(&serde_json::to_vec(&json!({"version": 2, "composition": {"space": "2d", "color_space": name},
                "render": {"resolution": [32, 32], "background": "blue"},
                "nodes": [{"type": "path", "d": "M0 0H32V32H0Z", "fill": "#ff000080"}]}))?)?;
            let document = Document::new(&scene, Path::new("."))?;
            document.install_images()?;
            let engine = crate::canvas_draw::Engine::new_picture(&renderer, document.float(), document.working())?;
            let pictures = Pictures::new(&renderer, &document, Path::new("."), true)?;
            let texture = document.draw(&engine, &pictures, &renderer, &vork_time::Request::Numeric(0.), None)?;
            let at = (16 * 32 + 16) * 4;
            let pixel: [f32; 4] = if texture.format() == wgpu::TextureFormat::Rgba16Float {
                let texels = moda_engine::readback::read_texture_rgba16f(&renderer.device, &renderer.queue, &texture, 32, 32)?;
                std::array::from_fn(|c| texels[at + c].to_f32())
            } else {
                let texels = moda_engine::readback::read_texture_rgba(&renderer.device, &renderer.queue, &texture, 32, 32)?;
                std::array::from_fn(|c| f32::from(texels[at + c]) / 255.)
            };
            // Only an sRGB canvas with the default view hands over its drawn 8-bit picture.
            assert_eq!(texture.format() == wgpu::TextureFormat::Rgba8Unorm, space == S::Srgb, "{name}");
            let (red, blue) = (S::Srgb.convert_rgb(space, [1., 0., 0.]), S::Srgb.convert_rgb(space, [0., 0., 1.]));
            let blended: [f32; 3] = std::array::from_fn(|i| alpha * red[i] + (1. - alpha) * blue[i]);
            let expected = space.convert_rgb(S::Srgb, blended);
            assert!(pixel[3] == 1. && pixel[..3].iter().zip(expected).all(|(a, b)| (a - b).abs() < 0.01), "{name}: {pixel:?} for {expected:?}");
        }
        Ok(())
    }

    /// A default canvas is its 8-bit picture copied as it is. A canvas with an authored view or a world in it
    /// keeps a float picture, finished once at the 8-bit output by its view stage. Neither is dithered: a
    /// canvas's flat colour stays flat, and only a world's own picture carries dither.
    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn default_canvases_are_copied_and_float_canvases_are_finished_once() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-canvas-float-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let render = |name: &str, composition: Value, extra: Value| -> Result<Vec<u8>> {
            let mut document = json!({"version": 2, "composition": composition,
                "render": {"resolution": [32, 32], "background": "rgb(128, 64, 32)"}, "nodes": []});
            for (field, value) in extra.as_object().expect("fields") { document[field.as_str()] = value.clone(); }
            let scene = root.join(format!("{name}.json"));
            std::fs::write(&scene, serde_json::to_vec(&document)?)?;
            let out = root.join(format!("{name}.png"));
            drop(crate::shot::Shot { scene, out: vec![out.clone().into()], ..Default::default() }.render()?);
            Ok(image::open(&out)?.to_rgba8().into_raw())
        };
        let colour = [128u8, 64, 32, 255];
        let canvas = json!({"space": "2d"});
        let plain = render("plain", canvas.clone(), json!({}))?;
        assert!(plain.chunks_exact(4).all(|p| p == colour), "a default canvas is copied as it is: {:?}", &plain[..4]);
        // std/standard at its defaults is the identity on the canvas's encoding: the same colour through the
        // float picture and the view stage, on every pixel. The stage adds no dither to a canvas.
        let identity = render("identity", json!({"space": "2d", "view": {"program": "std/standard", "inputs": {}}}), json!({}))?;
        assert!(identity.chunks_exact(4).all(|p| p == colour), "{:?}", &identity[..4]);
        // One stop up, in linear light, and still one flat colour.
        let decode = |c: f64| if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
        let encode = |c: f64| if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1. / 2.4) - 0.055 };
        let lifted = [128., 64., 32.].map(|c: f64| (encode(2. * decode(c / 255.)).clamp(0., 1.) * 255.).round() as u8);
        let viewed = render("lifted", json!({"space": "2d", "view": {"program": "std/standard", "inputs": {"exposure": 1}}}), json!({}))?;
        assert!(viewed[3] == 255 && viewed[..3].iter().zip(&lifted).all(|(a, b)| a.abs_diff(*b) <= 1), "{:?} for {lifted:?}", &viewed[..4]);
        assert!(viewed.chunks_exact(4).all(|p| p == &viewed[..4]), "a viewed flat colour stays flat");
        // A view result that cannot be used refuses the frame, as a world's does.
        let broken = render("broken", json!({"space": "2d", "view": {"program": "broken"}}), json!({"programs": {"broken": {"source":
            "struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs i) {return finish_view(float3(1./(ctx.color.x-ctx.color.x)),ctx,1.);}"}}}));
        let error = format!("{:#}", broken.err().context("a non-finite canvas view rendered")?);
        assert!(error.contains("view returned a non-finite colour"), "{error}");
        // A float picture keeps a translucent pixel's straight colour: it is unassociated from float values,
        // where an 8-bit picture has only the rounded associated bytes (128 * 0.02 is 3 of 255, or 153 back).
        let faint = render("faint", json!({"space": "2d", "view": {"program": "std/standard", "inputs": {}}}), json!({
            "render": {"resolution": [32, 32], "background": "transparent"},
            "nodes": [{"type": "path", "d": "M0 0H32V32H0Z", "fill": "rgb(128, 64, 32)", "opacity": 0.02}]}))?;
        let centre = &faint[(16 * 32 + 16) * 4..][..4];
        assert!(centre[3] == 5 && centre[..3].iter().zip(&colour[..3]).all(|(a, b)| a.abs_diff(*b) <= 1), "{centre:?}");
        // A world makes its canvas float without a view: the stage finishes the picture as it is. The
        // canvas's own colour beside the world (the triangle is up and right of the centre) stays exact.
        let world = json!({
            "buffers": {"p": {"type": "vec3", "data": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}},
            "geometry": {"g": {"attributes": {"POSITION": "p"}}},
            "nodes": [{"name": "shot", "composition": {"space": "3d"}, "children": [
                {"type": "camera", "name": "cam", "translation": [0, 0, 5]}, {"type": "mesh", "name": "tri", "geometry": "g"}]}]});
        let still = render("world", canvas.clone(), world.clone())?;
        assert_eq!(still, render("world-again", canvas, world)?);
        assert_eq!(still[31 * 32 * 4..][..4], colour, "the canvas's own colour beside a world");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU, pinned Slang and video encoder"]
    fn canvas_renders_between_encoder_handoff_and_ring_release() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-canvas-overlap-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let scene = root.join("scene.json");
        std::fs::write(&scene, serde_json::to_vec(&json!({
            "version": 2, "composition": {"space": "2d"},
            "render": {"resolution": [192, 108], "fps": 30, "duration": 0.3, "background": "white"},
            "nodes": []
        }))?)?;
        FRAME_EVENTS.with_borrow_mut(|events| *events = Some(Vec::new()));
        let result = crate::shot::Shot { scene, out: vec![root.join("movie.mp4").into()], ..Default::default() }.render();
        let events = FRAME_EVENTS.with_borrow_mut(|events| events.take().unwrap());
        let shot = result?;
        assert_eq!(shot.report["frames"], 9);
        for (end, &(event, frame)) in events.iter().enumerate().filter(|(_, (event, _))| *event == "release") {
            assert_eq!(event, "release");
            let start = events.iter().position(|e| *e == ("push", frame)).expect("release follows handoff");
            assert!(events[start + 1..end].iter().any(|&(event, next)| event == "draw" && next > frame),
                "encoder frame {frame} had no rendering to overlap before release: {events:?}");
        }
        assert!(events.iter().any(|&(event, _)| event == "release"), "exercise ring reuse");
        if cfg!(target_os = "macos") {
            // Handoff must arm GPU completion before host acceptance/readback.
            // Three submissions can occupy the existing ring before the first collection.
            let first_collect = events.iter().position(|e| *e == ("collect", 0)).unwrap();
            for frame in 0..crate::constants::FRAMES_IN_FLIGHT {
                assert!(events[..first_collect].contains(&("draw", frame)), "ring did not fill: {events:?}");
            }
            for frame in 0..9 {
                let push = events.iter().position(|e| *e == ("push", frame)).unwrap();
                let collect = events.iter().position(|e| *e == ("collect", frame)).unwrap();
                assert!(push < collect, "frame {frame} waited for readback before handoff: {events:?}");
            }
        } else {
            for (at, &(_, frame)) in events.iter().enumerate().filter(|(_, (event, frame))| *event == "push" && frame + 1 < 9) {
                assert!(events[..at].contains(&("draw", frame + 1)), "frame {frame} was accepted before frame {} was drawn: {events:?}", frame + 1);
            }
        }
        drop(shot);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    fn document(nodes: Value, animation: Value) -> Document {
        let scene = Scene::from_json(&serde_json::to_vec(&json!({
            "version": 2, "composition": {"space": "2d"},
            "render": {"resolution": [100, 100]}, "nodes": nodes, "animation": animation
        })).unwrap()).unwrap();
        Document::new(&scene, Path::new(".")).unwrap()
    }

    fn drawing(document: &Document) -> Value {
        let frame = document.snapshot(&vork_time::Request::Numeric(0.)).unwrap();
        let nodes: Vec<_> = document.items.iter().map(|item| &item.node).collect();
        crate::canvas_draw::document(&document.page, &nodes, &frame.values).unwrap()
    }

    #[test]
    fn weighted_path_colours_start_from_authored_paint() {
        for field in ["fill", "stroke"] {
            for additive in [false, true] {
                let doc = document(json!([{"name":"p", "type":"path", "d":"M0 0H100V100H0Z",
                    "fill":"red", "stroke":"red"}]), json!([{"weight":0.5, "blend": if additive {"combine"} else {"replace"},
                    "tracks":[{"path":["nodes","p",field], "keys":[[0,[0,0,1,1]],[1,[0,1,0,1]]]}]}]));
                let frame = doc.snapshot(&vork_time::Request::Numeric(0.)).unwrap();
                let path = frame.values[0].path.unwrap();
                let colour = if field == "fill" { path.fill } else { path.stroke };
                assert_eq!(colour, Some([if additive {1.} else {0.5}, 0., 0.5, 1.]), "{field}, additive={additive}");
            }
        }
    }

    #[test]
    fn clip_sources_keep_their_own_clip() {
        for group in [false, true] {
            let mut outer = json!({"name":"outer", "type":"path", "d":"M0 0H100V100H0Z", "clip-path":"inner"});
            if group {
                outer = json!({"name":"outer", "clip-path":"inner", "children":[
                    {"type":"path", "d":"M0 0H100V100H0Z"}]});
            }
            let doc = document(json!([
                {"name":"picture", "type":"path", "d":"M0 0H100V100H0Z", "fill":"red", "clip-path":"outer"},
                outer, {"name":"inner", "type":"path", "d":"M0 0H50V100H0Z"}
            ]), json!([]));
            let drawing = drawing(&doc);
            let masks = drawing["nodes"].as_array().unwrap().iter().filter(|node| node.get("clipMaskNodeId").is_some()).count();
            assert_eq!(masks, 2, "clip source group={group}: {drawing}");
        }
    }

    /// The canvas's working encoding reaches Moda's planner with the page, and the nodes it writes are the
    /// same in every encoding: Moda plans a luminance mask by the encoding its engine blends in.
    #[test]
    fn a_canvas_document_is_the_same_in_every_working_space() {
        let document = |space: &str| {
            let scene = Scene::from_json(&serde_json::to_vec(&json!({
                "version": 2, "composition": {"space": "2d", "color_space": space}, "render": {"resolution": [100, 100]},
                "nodes": [{"name":"picture", "type":"path", "d":"M0 0H100V100H0Z", "fill":"white", "mask":"cut"},
                    {"name":"cut", "type":"path", "d":"M0 0H50V100H0Z", "fill":"black", "stroke":"white", "stroke-opacity":0.5}]
            })).unwrap()).unwrap();
            Document::new(&scene, Path::new(".")).unwrap()
        };
        let srgb = document("sRGB");
        assert!(srgb.page.working.is_none());
        for space in ["Display P3", "Linear Rec.709", "Linear Rec.2020", "ACEScg"] {
            let other = document(space);
            assert_eq!(drawing(&other), drawing(&srgb), "{space}");
            let working = other.page.working.expect("a working colour");
            // Authored sRGB white is white in every encoding, and the two conversions invert each other.
            let white = working.from_srgb.apply([1., 1., 1.]);
            assert!(white.iter().all(|c| (c - 1.).abs() < 1e-5), "{space}: {white:?}");
            let back = working.to_srgb.apply(working.from_srgb.apply([0.75, 0.25, 0.5]));
            for (a, b) in back.into_iter().zip([0.75, 0.25, 0.5]) { assert!((a - b).abs() < 1e-4, "{space}: {back:?}"); }
            assert_eq!(working.from_srgb.encode, !other.working().is_linear(), "{space}");
        }
    }

    #[test]
    fn masked_group_blends_its_whole_picture() {
        let doc = document(json!([
            {"name":"picture","mask":"cut","mix-blend-mode":"hue","isolation":"isolate","children":[
                {"type":"path","d":"M0 0H100V100H0Z","fill":"red"},
                {"type":"path","d":"M0 0H50V100H0Z","fill":"blue"}]},
            {"name":"cut","type":"path","d":"M0 0H100V100H0Z","fill":"black","mask-type":"alpha"}
        ]), json!([]));
        let picture = drawing(&doc);
        let nodes = picture["nodes"].as_array().unwrap();
        let outer = nodes.iter().find(|n| n["id"] == "n0").unwrap();
        assert_eq!(outer["blendMode"], "hue");
        assert_eq!(outer["isolation"], "isolate");
        assert_eq!(outer["children"], json!(["n0.alpha-group"]));
        assert_eq!(nodes.iter().filter(|n| n.get("blendMode").is_some_and(|mode| mode != "normal")).count(), 1);
        assert!(nodes.iter().any(|n| n["maskType"] == "alpha"));
    }

    #[test]
    fn luminance_mask_does_not_paint_referenced_definitions() {
        let doc = document(json!([
            {"name":"picture", "type":"path", "d":"M0 0H100V100H0Z", "fill":"red", "mask":"mask-group"},
            {"name":"mask-group", "children":[
                {"name":"white", "type":"path", "d":"M0 0H100V100H0Z", "fill":"white", "clip-path":"cut"},
                {"name":"cut", "type":"path", "d":"M0 0H50V100H0Z", "fill":"black"}
            ]}
        ]), json!([]));
        let drawing = drawing(&doc);
        let nodes = drawing["nodes"].as_array().unwrap();
        let mask = nodes.iter().find(|node| node["id"] == "n0.luminance").unwrap();
        assert_eq!(mask["children"].as_array().unwrap().len(), 1, "{drawing}");
        assert!(!nodes.iter().any(|node| node["id"] == "n0.luminance/3"), "{drawing}");
    }
}

#[cfg(test)]
#[path = "canvas_retained_tests.rs"]
mod retained_tests;

#[cfg(test)]
#[path = "canvas_style_tests.rs"]
mod style_tests;

#[cfg(test)]
#[path = "canvas_nesting_tests.rs"]
mod nesting_tests;
