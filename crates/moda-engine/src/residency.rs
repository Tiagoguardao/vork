//! residency.rs — Stage-1 retained-mode engine residency (task #95).
//!
//! All caching here is **keyed by content hashes computed from the re-fed
//! scene**, so every cache is *self-invalidating by construction*: there are
//! no invalidation messages to get wrong. Hash discipline (PLAN §2.2):
//! per-node hashes use **deny-list exclusion, never allow-list inclusion** —
//! unknown/new props therefore OVER-invalidate (safe, costs a re-compute)
//! instead of under-invalidating (stale pixels).
//!
//! Layers (PLAN §1):
//! - R1: `ResidentScene` — parse-once-per-sweep resident hydrated Scene +
//!   per-node hash maps (geom / text / full).
//! - R3: per-node tessellation cache (`PlanCaches::tess` / `line_tess`) —
//!   ONE entry per node, evict-and-replace on GeomKey change. Transform-only
//!   changes (x/y/rotation/scale*) do NOT re-key (deliberately better than
//!   the browser, which over-invalidates on x/y — MAP §5.1). `strokeWidth`
//!   IS in the key (the browser's latent stale-geometry hazard, MAP §5.2).
//! - R4: text item cache (full `TextQuadBatch` outcomes keyed by text-hash;
//!   transform/z/clip patched on hit) + the shaping cache inside moda-text.
//! - R6: blend-expansion tessellation rides R3 keyed by source-node hashes.

/// The plan caches' and resident scene's id-keyed maps hash with FxHash (std's SipHash showed up across a
/// first frame's plan build in wasm). Keys are the document's own ids and content hashes.
pub type HashMap<K, V> = std::collections::HashMap<K, V, rustc_hash::FxBuildHasher>;
mod span_pages;
mod parts;
pub(crate) mod weights;
pub use parts::ItemAddresses;
mod output_records;
use output_records::OutputRecords;

use std::hash::Hasher;
use std::sync::Arc;

use moda_scene::types::RawNode;
use moda_scene::Scene;
use moda_scene::value::Value;

use crate::frame::{PlanWarnings, RenderOptions};
use crate::seams::{CoveragePlan, TessellatedGeometry};

// ---------------------------------------------------------------------------
// Hashing (FxHasher — same hasher family as the production fxhash plan)
// ---------------------------------------------------------------------------

type FxHasher = rustc_hash::FxHasher;

/// A process-wide sequence keeps generations distinct across resident scenes.
static NEXT_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_generation() -> u64 {
    NEXT_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// TRANSFORM-ONLY props (PLAN §2.1): model-matrix inputs only — tessellation
/// and text residency survive changes to these.
///
/// `unitTransforms` (KineticText) is deliberately NOT here: the per-glyph
/// transform/opacity are baked into the analytic instance records at build time
/// (not applied as a separate uniform), so a per-frame change MUST invalidate the
/// text cache and rebuild — keeping it out of this set is required for cache
/// safety. The consequence is that animating `unitTransforms` re-shapes + re-packs
/// the whole vector batch every frame; an optimization (patch only the per-glyph
/// record fields on an otherwise-identical text hash, instead of a full rebuild)
/// is deferred to the slice that ships the emitter, where the perf actually bites.
const TRANSFORM_ONLY: &[&str] = &["x", "y", "rotation", "scale", "scaleX", "scaleY"];

/// Known PAINT-ONLY props (PLAN §2.1): rebuilt per frame into instance /
/// uniform data from the resident plan — never tessellation inputs. This is
/// a DENY list for the geom hash: anything NOT listed here (and not
/// transform-only) re-keys tessellation, so unknown props over-invalidate.
///
/// The paint stacks (`fills`, `strokes`, `textFills`) are hashed through
/// [`hash_paint_geometry`]: their COLOURS are paint-only, but the fill's
/// paint TYPE (routing), an image/video placement (the texture quad), and the
/// stroke's width, caps, joins, dash, alignment, width model and arrowheads
/// (baked into stroke triangles) are tessellation inputs.
///
/// NOTE deliberately KEPT in the geom hash (i.e. absent below): `distort` (a
/// distort warp rewrites the tessellated geometry through the local
/// homography), `fillRule`, and every text prop (lines carve label gaps into
/// their body geometry).
///
/// The effect list (`effects`) is hashed through [`hash_paint_geometry`]
/// too: shadows and post-raster layer filters leave path triangles intact.
/// Other entries conservatively re-key tessellation.
const PAINT_ONLY: &[&str] = &["opacity", "blendMode", "vorkPaint"];

/// Effect types that feed only per-frame paint (see [`PAINT_ONLY`]).
const PAINT_ONLY_EFFECT_TYPES: &[&str] = &["drop-shadow", "inner-shadow", "filter-shadow", "layer-blur"];

/// A key that feeds only per-frame paint.
fn is_paint_only(key: &str) -> bool {
    PAINT_ONLY.contains(&key)
}

/// Hash the tessellation-relevant part of a paint stack (see [`PAINT_ONLY`]) into `h`:
/// per fill layer its visibility, paint type and image/video placement plus
/// the recolour/overlay switches (which pick the texture pipeline);
/// per stroke layer everything but its paint's colours (the paint type stays:
/// a gradient stroke routes differently). Returns false, having hashed nothing, for keys that
/// are not a paint stack.
///
/// The part is hashed where it stands: no value is built to be hashed and dropped (one small map
/// per paint layer of every node, on every hash of the node). It is written through the one
/// encoding of a value ([`ValueHash`]), as the part would hash written out as a value with an
/// absent field as `null`.
///
/// Node hashes are what the tessellation and plan caches key on until those caches follow the
/// scene's stamps (renderer plan 2.4); this function goes with them.
fn hash_paint_geometry(key: &str, v: &Value, h: &mut FxHasher) -> bool {
    if !matches!(key, "fills" | "strokes" | "textFills" | "effects") {
        return false;
    }
    let Some(layers) = v.as_array() else { return false };
    let kept = |effect: &Value| {
        !effect.get("type").and_then(Value::as_str).is_some_and(|t| PAINT_ONLY_EFFECT_TYPES.contains(&t))
    };
    let mut h = ValueHash(h);
    h.array(match key {
        "effects" => layers.iter().filter(|effect| kept(*effect)).count(),
        _ => layers.len(),
    });
    for layer in layers {
        let paint = layer.get("paint");
        match key {
            "fills" => {
                let adjust = paint.and_then(|p| p.get("adjust"));
                let flag = |a: &str, b: &str| adjust.and_then(|x| x.get(a)).and_then(|x| x.get(b));
                h.object(5);
                h.field("visible", layer.get("visible"));
                h.field("type", paint.and_then(|p| p.get("type")));
                h.field("placement", paint.and_then(|p| p.get("placement")));
                h.field("overlay", flag("overlay", "enabled"));
                h.field("recolor", flag("recolor", "enabled"));
            }
            "strokes" => {
                // Every field but the layer's opacity and blend mode, in the layer's order, with the paint as
                // its type alone; a layer without a paint has one of type `null`, last.
                let fields = layer.as_object();
                let own = |k: &str| !matches!(k, "opacity" | "blendMode" | "paint");
                h.object(fields.map_or(0, |o| o.iter().filter(|(k, _)| own(k.as_str())).count()) + 1);
                let paint_type = |h: &mut ValueHash<'_>| {
                    h.key("paint");
                    h.object(1);
                    h.field("type", paint.and_then(|p| p.get("type")));
                };
                let mut painted = false;
                for (k, value) in fields.into_iter().flat_map(|o| o.iter()) {
                    match k.as_str() {
                        "opacity" | "blendMode" => {}
                        "paint" => {
                            paint_type(&mut h);
                            painted = true;
                        }
                        k => h.field(k, Some(value)),
                    }
                }
                if !painted {
                    paint_type(&mut h);
                }
            }
            "textFills" => {
                h.object(2);
                h.field("visible", layer.get("visible"));
                h.field("type", paint.and_then(|p| p.get("type")));
            }
            _ => {
                if kept(layer) {
                    h.value(layer);
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod paint_geometry_tests {
    use super::*;
    use moda_scene::value::json;

    fn in_place(key: &str, v: &Value) -> Option<u64> {
        let mut h = FxHasher::default();
        hash_paint_geometry(key, v, &mut h).then(|| h.finish())
    }
    fn of_value(v: &Value) -> u64 {
        let mut h = FxHasher::default();
        hash_value(v, &mut h);
        h.finish()
    }

    /// The part hashed in place is the hash of the part written out as a value: fills, text fills and effects.
    #[test]
    fn the_paint_part_hashes_as_the_value_it_stands_for() {
        let fills = json!([
            {"id": "a", "visible": true, "opacity": 0.5, "paint": {"type": "solid", "color": {"r": 1, "g": 0, "b": 0, "a": 1}}},
            {"id": "b", "paint": {"type": "image", "placement": {"mode": "fill"}, "adjust": {"overlay": {"enabled": true}, "recolor": {"enabled": false}}}},
            {"id": "c"},
            7,
        ]);
        assert_eq!(in_place("fills", &fills), Some(of_value(&json!([
            {"visible": true, "type": "solid", "placement": null, "overlay": null, "recolor": null},
            {"visible": null, "type": "image", "placement": {"mode": "fill"}, "overlay": true, "recolor": false},
            {"visible": null, "type": null, "placement": null, "overlay": null, "recolor": null},
            {"visible": null, "type": null, "placement": null, "overlay": null, "recolor": null},
        ]))));
        let text = json!([{"visible": false, "paint": {"type": "gradient", "stops": []}}, {}]);
        assert_eq!(in_place("textFills", &text), Some(of_value(&json!([
            {"visible": false, "type": "gradient"}, {"visible": null, "type": null},
        ]))));
        let effects = json!([{"type": "blur", "radius": 4}, {"type": PAINT_ONLY_EFFECT_TYPES[0]}, {"radius": 1}]);
        assert_eq!(in_place("effects", &effects), Some(of_value(&json!([{"type": "blur", "radius": 4}, {"radius": 1}]))));
        // Not a paint stack: nothing is hashed.
        assert_eq!(in_place("fills", &json!({"not": "a list"})), None);
        assert_eq!(in_place("points", &json!([])), None);
    }

    /// A stroke layer hashes everything but its colours, its opacity and its blend mode.
    #[test]
    fn a_stroke_layer_hashes_its_shape_and_paint_type_only() {
        let stroke = |color: f64, opacity: f64, width: f64, kind: &str| json!([
            {"id": "s", "width": width, "opacity": opacity, "blendMode": "normal", "paint": {"type": kind, "color": color}},
        ]);
        let base = in_place("strokes", &stroke(1., 1., 2., "solid"));
        assert!(base.is_some());
        assert_eq!(base, in_place("strokes", &stroke(0.25, 0.5, 2., "solid")), "colour and opacity are paint only");
        assert_ne!(base, in_place("strokes", &stroke(1., 1., 3., "solid")), "width is geometry");
        assert_ne!(base, in_place("strokes", &stroke(1., 1., 2., "gradient")), "the paint's type routes the stroke");
        assert_ne!(in_place("strokes", &json!([{"width": 2}])), in_place("strokes", &json!([{"width": 2}, {"width": 2}])));
    }
}

/// The three per-node content hashes (PLAN §2.2/§2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeHashes {
    /// Hash of the ENTIRE node value (R5 fingerprints, R6 keys).
    pub full: u64,
    /// Full minus TRANSFORM-ONLY minus known PAINT-ONLY (the GeomKey).
    pub geom: u64,
    /// Full minus TRANSFORM-ONLY (text caches: paint IS baked into quads).
    pub text: u64,
}

/// Per-node hashes used by the retained planner. Canvas rows compute a hash at its first reader.
pub trait NodeHashSource: Sync {
    fn read_hash(&self, scene: &Scene, id: &str) -> Option<NodeHashes>;
}

impl NodeHashSource for HashMap<String, NodeHashes> {
    fn read_hash(&self, _scene: &Scene, id: &str) -> Option<NodeHashes> {
        self.get(id).copied()
    }
}

#[derive(Default)]
pub struct RowHashes {
    rows: std::sync::OnceLock<Vec<std::sync::Mutex<Option<(u64, NodeHashes)>>>>,
}

impl NodeHashSource for RowHashes {
    fn read_hash(&self, scene: &Scene, id: &str) -> Option<NodeHashes> {
        let Some(row) = scene.row_of(id) else {
            return scene.effective_node(id).map(|node| node_hashes_in(scene, node));
        };
        let rows = self.rows.get_or_init(|| (0..scene.doc().nodes.len()).map(|_| Default::default()).collect());
        let compute = || scene.effective_node(id).map(|node| node_hashes_in(scene, node));
        let Some(slot) = rows.get(row as usize) else { return compute(); };
        let mut cached = slot.lock().unwrap();
        let stamp = scene.stamps().row_revision(row);
        if let Some((built, hashes)) = *cached {
            if built == stamp { return Some(hashes); }
        }
        let hashes = compute()?;
        *cached = Some((stamp, hashes));
        Some(hashes)
    }
}

/// The one encoding of a JSON value into a hasher. Everything that hashes a value, or a part of one where it
/// stands, writes through this, so a value and its parts hash alike and there is one encoding to change.
struct ValueHash<'a>(&'a mut FxHasher);

impl ValueHash<'_> {
    fn null(&mut self) {
        self.0.write_u8(0);
    }
    fn bool(&mut self, b: bool) {
        self.0.write_u8(1);
        self.0.write_u8(b as u8);
    }
    /// Canonical numeric identity: f64 bits (JSON numbers are f64 in every producer we feed from).
    fn number(&mut self, bits: u64) {
        self.0.write_u8(2);
        self.0.write_u64(bits);
    }
    fn string(&mut self, s: &str) {
        self.0.write_u8(3);
        self.0.write(s.as_bytes());
    }
    /// An array of `len` items: the items follow.
    fn array(&mut self, len: usize) {
        self.0.write_u8(4);
        self.0.write_usize(len);
    }
    /// An object of `len` fields: each field's key, then its value, follow.
    fn object(&mut self, len: usize) {
        self.0.write_u8(5);
        self.0.write_usize(len);
    }
    fn key(&mut self, key: &str) {
        self.0.write(key.as_bytes());
    }
    /// A field of an object: its key and its value, `null` when the value is absent.
    fn field(&mut self, key: &str, value: Option<&Value>) {
        self.key(key);
        match value {
            Some(value) => self.value(value),
            None => self.null(),
        }
    }
    fn value(&mut self, v: &Value) {
        match v {
            Value::Null => self.null(),
            Value::Bool(b) => self.bool(*b),
            Value::Number(n) => self.number(n.as_f64().map(f64::to_bits).unwrap_or(0)),
            Value::String(s) => self.string(s),
            Value::Array(a) => {
                self.array(a.len());
                for item in a {
                    self.value(item);
                }
            }
            Value::Object(o) => {
                self.object(o.len());
                // preserve_order Map: iteration order == insertion order — stable
                // for the same JSON; a re-ordered but equal doc only
                // over-invalidates.
                for (k, val) in o {
                    self.field(k.as_str(), Some(val));
                }
            }
        }
    }
}

fn hash_value(v: &Value, h: &mut FxHasher) {
    ValueHash(h).value(v);
}

/// [`hash_value`] of compact points' JSON, from the compact form (so a node hashes the same however it holds
/// its points).
fn hash_compact_points(points: &moda_scene::path_points::CompactPoints, h: &mut FxHasher) {
    let mut h = ValueHash(h);
    let num = |h: &mut ValueHash<'_>, key: &str, v: f64| {
        h.key(key);
        h.number(v.to_bits());
    };
    h.array(points.len());
    for i in 0..points.len() {
        let (kind, c) = points.point(i);
        h.object(kind.keys().len());
        h.key("type");
        h.string(kind.name());
        num(&mut h, "x", c[0]);
        num(&mut h, "y", c[1]);
        for (j, key) in kind.controls().iter().enumerate() {
            h.key(key);
            h.object(2);
            num(&mut h, "x", c[2 + 2 * j]);
            num(&mut h, "y", c[3 + 2 * j]);
        }
    }
}

/// Per-node effective hashes for an already-hydrated [`Scene`] — the map the
/// plan-cache fingerprint and R3/R4 caches key on. Mirrors what
/// [`ResidentScene::update`] builds from JSON, but for a `Scene` constructed
/// any other way (e.g. the mobile FFI's per-frame `Scene::from_document`
/// overlay lane). Hashes the resolved-??-canonical view (`effective_node`),
/// exactly what the plan builder reads.
pub fn node_hashes_for_scene(scene: &Scene) -> HashMap<String, NodeHashes> {
    use rayon::prelude::*;
    let order = scene.node_order();
    let mut hashes = HashMap::with_capacity_and_hasher(order.len() + scene.transient_by_id().len(), Default::default());
    let hash = |id: &String| {
        let node = scene.effective_node(id)?;
        Some((id.clone(), node_hashes_in(scene, node)))
    };
    if order.len() < 2048 {
        hashes.extend(order.iter().filter_map(hash));
    } else {
        let rows: Vec<_> = order.par_iter().with_min_len(512).filter_map(hash).collect();
        hashes.extend(rows);
    }
    for (id, node) in scene.transient_by_id() {
        let node = scene.effective_node(id).unwrap_or(node);
        hashes.insert(id.clone(), node_hashes_in(scene, node));
    }
    hashes
}

/// [`node_hashes`] plus the document state a node's drawing depends on
/// beyond its own fields: the brush definition its `strokeBrush` names
/// (brush plan P8 — a brush-table edit must re-key the strokes that use it,
/// in the tess cache and in every plan fingerprint).
pub fn node_hashes_in(scene: &Scene, node: &RawNode) -> NodeHashes {
    let mut nh = node_hashes(node);
    // A text layer lays out in the document's locale (the language of text that declares none, which picks its CJK
    // faces: `text_draw::document_locale`), so a locale change re-keys it like an edit of its own.
    if node.node_type() == "richtext" {
        if let Some(locale) = crate::text_draw::document_locale(scene) {
            let mix = |x: u64| {
                let mut m = FxHasher::default();
                m.write_u64(x);
                m.write(b"locale");
                m.write(locale.as_bytes());
                m.finish()
            };
            nh.full = mix(nh.full);
            nh.geom = mix(nh.geom);
            nh.text = mix(nh.text);
        }
    }

    // E8.5: the pattern definitions its paints name (a pattern edit re-keys every node that draws it). Only the
    // full hash: the node's own geometry doesn't depend on a pattern (the tile shapes hash their own).

    nh
}

/// Compute all three hashes in one pass over the node's top-level entries.
pub(crate) fn node_hashes(node: &RawNode) -> NodeHashes {
    let mut full = FxHasher::default();
    let mut geom = FxHasher::default();
    let mut text = FxHasher::default();
    for (k, v) in node.fields() {
        let mut entry = FxHasher::default();
        entry.write(k.as_bytes());
        match node.compact_points().filter(|_| k == "points") {
            Some(points) => hash_compact_points(points, &mut entry),
            None => hash_value(v, &mut entry),
        }
        let eh = entry.finish();
        full.write_u64(eh);
        let is_transform = TRANSFORM_ONLY.contains(&k.as_str());
        if !is_transform {
            text.write_u64(eh);
            let mut g = FxHasher::default();
            g.write(k.as_bytes());
            if hash_paint_geometry(k, v, &mut g) {
                geom.write_u64(g.finish());
            } else if !is_paint_only(k) {
                geom.write_u64(eh);
            }
        }
    }

    NodeHashes {
        full: full.finish(),
        geom: geom.finish(),
        text: text.finish(),
    }
}

// ---------------------------------------------------------------------------
// R1 — resident scene
// ---------------------------------------------------------------------------

/// Parse-once-per-sweep residency: the hydrated Scene + per-node hash maps,
/// keyed by the fxhash of the raw scene JSON. `update` re-parses only when
/// the JSON content changed; a multi-page sweep re-feeding the same string
/// hits the resident copy (Stage 0 re-parsed once per page).
pub struct ResidentScene {
    state: Option<Resident>,
    /// Advances whenever the resident content changes. Retained plan indexes record this generation.
    generation: u64,
    /// rust-edit-incremental-z4 (incremental plan patching): the generation of the
    /// most recent change that could alter the DISPLAY-LIST STRUCTURE (z-order,
    /// membership, visibility) — a full `update`/`clear`, a `zIndex` edit, a
    /// visibility flip (`visible`/`hidden`), an `opacity` crossing 0, or any page
    /// patch. A cached plan index built at generation `g` is only incrementally
    /// patchable when `last_structural_gen <= g` (otherwise the display list it
    /// indexed may have changed shape → full rebuild). Conservative: over-bumping
    /// only forces a (correct) full rebuild.
    last_structural_gen: u64,
    /// rust-edit-incremental-z4: a rolling log of `(generation, node_id)` for every
    /// node a non-structural `apply_props` touched, newest at the back. Lets the
    /// render path compute "which nodes changed since generation g" without an O(n)
    /// scan. Bounded by `DIRTY_LOG_CAP`; when it overflows, the window start rises
    /// and any plan index older than it falls back (safe).
    recent_edits: std::collections::VecDeque<(u64, String, bool)>,
    /// Scale RFC stage C: the generation of the most recent change the retained
    /// camera plan cannot patch at all — a full `update`/`clear`, a page patch, or
    /// a patch that failed part-way. Every other change, structural or not, is in
    /// `recent_edits` with its node ids (`dirty_detail`).
    last_hard_gen: u64,
    /// rust-edit-incremental-z4: the oldest generation `recent_edits` still fully
    /// covers. A `dirty_since(g)` query with `g < dirty_window_start_gen` cannot be
    /// answered precisely → treated as structural (full rebuild).
    dirty_window_start_gen: u64,
    /// Scoped re-expansion kill switch (`?wasmHostReexpand=0`). Default ENABLED.
    /// When off, `apply_props` reverts to the earlier behaviour: bail on any node with a
    /// transient expansion so the host heals with a whole-scene re-feed. The switch
    /// lives on `ResidentScene` (not `PlanCaches`) because the decision is taken
    /// inside `apply_props`, which only ever sees the resident scene.
    host_reexpand_enabled: bool,
    /// Scoped re-expansion counters, surfaced through the wasm `residency_stats` JSON:
    /// how many transient-host prop patches took the scoped re-expansion path, and
    /// how many of those turned out STRUCTURAL (the re-expansion changed the
    /// transient id sequence → the plan index is unusable, full rebuild). A
    /// `host_reexpands` that stays 0 while a table drag is running means the lane
    /// is not engaging; a `host_reexpand_structural` that tracks `host_reexpands`
    /// 1:1 means we re-expanded but still paid the rebuild.
    host_reexpands: u64,
    host_reexpand_structural: u64,
}

/// rust-edit-incremental-z4: the answer to "what changed since generation g?".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirtySince {
    /// Nothing changed since `g` — the cached plan is current as-is.
    Unchanged,
    /// A structural change occurred (or the window can't cover `g`) — the cached
    /// plan index is unusable; do a full rebuild.
    Structural,
    /// Exactly these node ids had non-structural property edits since `g`.
    Nodes(Vec<String>),
}

struct Resident {

    scene: Scene,
    /// Effective-node hashes (resolved ?? canonical) + transients — exactly
    /// what the plan builder reads through `effective_node`.
    hashes: HashMap<String, NodeHashes>,
}

impl Default for ResidentScene {
    fn default() -> Self {
        ResidentScene {
            state: None,
            generation: 0,
            last_structural_gen: 0,
            recent_edits: std::collections::VecDeque::new(),
            last_hard_gen: 0,
            dirty_window_start_gen: 0,
            host_reexpand_enabled: true,
            host_reexpands: 0,
            host_reexpand_structural: 0,
        }
    }
}

impl ResidentScene {

    /// rust-edit-incremental-z4: which nodes changed since a plan index was built
    /// at generation `since_gen` — the gate for the incremental patcher. Returns
    /// `Structural` (→ full rebuild) on any display-list-shape change since then,
    /// or when the log can't cover `since_gen`; `Unchanged` when nothing changed;
    /// else the exact set of property-edited node ids.
    pub(crate) fn dirty_since(&self, since_gen: u64) -> DirtySince {
        if since_gen == self.generation {
            return DirtySince::Unchanged;
        }
        if self.last_structural_gen > since_gen || since_gen < self.dirty_window_start_gen {
            return DirtySince::Structural;
        }
        let mut seen = std::collections::HashSet::new();
        let mut nodes = Vec::new();
        for (g, id, _) in &self.recent_edits {
            if *g > since_gen && seen.insert(id.as_str()) {
                nodes.push(id.clone());
            }
        }
        DirtySince::Nodes(nodes)
    }

    pub(crate) fn scene(&self) -> Option<&Scene> {
        self.state.as_ref().map(|r| &r.scene)
    }

    pub(crate) fn hashes(&self) -> Option<&HashMap<String, NodeHashes>> {
        self.state.as_ref().map(|r| &r.hashes)
    }

    /// The current resident content generation.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    #[cfg(test)]
    pub(crate) fn from_scene(scene: Scene) -> ResidentScene {
        let mut r = ResidentScene::default();
        let hashes = node_hashes_for_scene(&scene);
        r.state = Some(Resident {

            scene,
            hashes,

        });
        r.generation = next_generation();
        r.last_structural_gen = r.generation;
        r.last_hard_gen = r.generation;
        r.dirty_window_start_gen = r.generation;
        r
    }

    pub fn clear(&mut self) {
        self.state = None;
        // M1: dropping the resident scene is a content change — bump so a
        // clear → rebuild cycle never collides with the pre-clear generation.
        self.generation = next_generation();
        // rust-edit-incremental-z4: a clear is structural; reset the dirty window.
        self.last_structural_gen = self.generation;
        self.last_hard_gen = self.generation;
        self.recent_edits.clear();
        self.dirty_window_start_gen = self.generation;
    }
}

const TESS_BUDGET_BYTES: usize = 96 * 1024 * 1024;
/// Byte budget for cached text batches.
const TEXT_BUDGET_BYTES: usize = 192 * 1024 * 1024;
/// Byte budget for retained plan data and prepared rows.
pub(crate) const CAMERA_PLAN_BUDGET_BYTES: usize = 128 * 1024 * 1024;
pub struct TessEntry {
    pub key: u64,
    /// `Arc`-shared with the plan's `DrawItem::Body.geometry` — a cache hit is a
    /// pointer-bump, and the plan never deep-copies the vertex stream.
    pub geometry: Arc<TessellatedGeometry>,
    tick: u64,
    /// Brush plan P8: for an outlined stroke, its key WITHOUT the zoom band
    /// (geometry + brush). A live camera frame may draw this entry while the
    /// outline for the current band is still being refined.
    outline_base: Option<u64>,
    /// Brush plan P8: an outlined stroke's second band (same base) — e.g. the
    /// camera's while a page capture / thumbnail renders at its own scale —
    /// so the two don't evict each other every frame.
    outline_alt: Option<(u64, Arc<TessellatedGeometry>)>,

}

pub struct LineTessEntry {
    pub key: u64,
    pub geometries: Vec<TessellatedGeometry>,
    tick: u64,
}

pub struct TextEntry {
    pub key: u64,
    /// The cached batch with STALE z/model/clip/mask — patched on hit.
    pub batch: Box<crate::seams::TextQuadBatch>,
    /// Connector labels only: the carved body text-gap (start_t, end_t)
    /// measured with the batch (font metrics ride the key's generation).
    pub gap: Option<(f64, f64)>,
    tick: u64,
}

/// Per-engine caches used while building retained plans.
#[derive(Default)]
pub struct PlanCaches {
    tick: u64,
    /// The text cache's budget in bytes; 0 = [`TEXT_BUDGET_BYTES`] (tests set a small one).
    pub(crate) text_budget_bytes: usize,
    pub(crate) tess: HashMap<String, TessEntry>,
    tess_bytes: usize,
    /// The frame (tick) whose eviction pass found only the working set left over budget: no further pass that
    /// frame (each one walks and sorts every entry).
    tess_evict_blocked: Option<u64>,

    line_evict_blocked: Option<u64>,
    pub(crate) line_tess: HashMap<String, LineTessEntry>,
    line_tess_bytes: usize,
    pub(crate) text: HashMap<String, TextEntry>,
    text_bytes: usize,
    // Stats (battery part 2 reads these through the wasm stats surface).
    pub tess_hits: u64,
    pub tess_misses: u64,
    pub text_hits: u64,
    pub text_misses: u64,
    pub evictions: u64,
    /// rust-edit-incremental-z4: edits served entirely from the index (no rebuild):
    /// static re-renders (`incr_static_hits`) and structure-preserving patches
    /// (`incr_patch_hits`, touching `incr_patch_nodes` nodes total). `incr_fallbacks`
    /// counts edits that fell back to a full rebuild (structural / ineligible /
    /// shape-verify miss). Surfaced in `stats_json` for telemetry.
    pub incr_static_hits: u64,
    pub incr_patch_hits: u64,
    pub incr_patch_nodes: u64,
    /// Nodes whose certified placement changed without rebuilding their draw items.
    pub incr_placement_nodes: u64,
    pub incr_fallbacks: u64,
    /// rust-edit-incremental-z4: `true` disables the incremental patcher (the
    /// `?wasmIncrementalPlan=0` kill switch), forcing the legacy full-rebuild path.
    /// Config — preserved across `clear()`.
    incremental_disabled: bool,
    /// The canvas root's display list its caller built for the plan in progress
    /// ([`crate::frame::build_camera_box_plan_cached_with_dl`]), with the address of the scene it was built from.
    /// Set and cleared within that one call.
    frame_canvas_dl: Option<(usize, Arc<moda_scene::DisplayList>)>,
    reverse_deps: Option<(u64, Arc<ReverseDeps>)>,
}

/// rust-edit-incremental-z4: everything the incremental patcher needs to rebuild a
/// single node's items and splice them into the cached plan without rebuilding the
/// display list or re-deriving page geometry.
struct PlanIndex {
    /// Per item-producing primary node, in plan-item order.
    spans: span_pages::Spans,
    /// node_id → index into `spans`.
    by_node: SourceLookup,
    page_index: usize,
    page_clip: [f32; 4],
    transparent: bool,
    /// Plan-relevant `RenderOptions` fingerprint the plan was built with — an opts
    /// change (scale/sample/text-engine/…) forces a full rebuild.
    opts_key: u64,
    /// Resident generation the plan reflects. `dirty_since(built_at_generation)`
    /// drives the fast path.
    built_at_generation: u64,
    /// Scene-wide bindings whose edits can change output outside the edited subtree.
    deps: Arc<ReverseDeps>,
    /// Viewport culling: roots of the display-list ranges the build skipped. A
    /// culled node has no span, which the patcher would otherwise read as "draws
    /// nothing" — an edit that touches one (or one of their descendants) falls back.
    culled_roots: weights::RetainedSet<String>,
}

#[derive(Default)]
struct SourceLookup {
    rows: Vec<u32>,
    generated: weights::RetainedMap<Box<str>, usize>,
}
impl SourceLookup {
    fn insert(&mut self, scene: &Scene, id: &str, at: usize) {
        if let Some(row) = scene.row_of(id) {
            if self.rows.is_empty() { self.rows.resize(scene.doc().nodes.len(), u32::MAX); }
            self.rows[row as usize] = at as u32;
        } else {
            self.generated.insert(id.into(), at);
        }
    }
    fn contains(&self, scene: &Scene, id: &str) -> bool { self.get(scene, id).is_some() }
    fn values(&self) -> impl Iterator<Item = usize> + '_ {
        self.rows.iter().filter(|&&value| value != u32::MAX).map(|&value| value as usize)
            .chain(self.generated.values().copied())
    }
    fn remap(&mut self, map: &[usize]) {
        for row in &mut self.rows {
            if *row != u32::MAX { *row = map[*row as usize] as u32; }
        }
        self.generated.retain(|_, value| { *value = map[*value]; *value != usize::MAX });
        if self.rows.iter().all(|&v| v == u32::MAX) { self.rows = Vec::new(); }
    }
    fn bytes(&self) -> usize {
        self.rows.capacity() * std::mem::size_of::<u32>()
            + self.generated.allocation_size()
            + self.generated.keys().map(|id| id.len()).sum::<usize>()
    }
    fn get(&self, scene: &Scene, id: &str) -> Option<usize> {
        if let Some(row) = scene.row_of(id) {
            return self.row(row);
        }
        self.generated.get(id).copied()
    }
    fn row(&self, row: u32) -> Option<usize> {
        self.rows.get(row as usize).copied().filter(|&span| span != u32::MAX).map(|span| span as usize)
    }
}

#[derive(Default)]
struct ReverseDeps {
    blend_related_ids: weights::RetainedSet<String>,
    mask_source_ids: weights::RetainedSet<String>,
    /// Present only if every binding of this source is a group pruned by the ordinary paint
    /// walk. Its visibility selector runs before mask resolution, so those groups read no mask.
    hidden_clip_owners: weights::RetainedMap<String, Vec<u32>>,
}

impl ReverseDeps {
    fn add_mask_source(&mut self, source: &str, hidden_owner: Option<u32>) {
        let first = self.mask_source_ids.insert(source.to_owned());
        match hidden_owner {
            Some(row) if first => { self.hidden_clip_owners.insert(source.to_owned(), vec![row]); }
            Some(row) => {
                if let Some(owners) = self.hidden_clip_owners.get_mut(source) { owners.push(row); }
            }
            None => { self.hidden_clip_owners.remove(source); }
        }
    }

    fn clip_owners_still_hidden(&self, scene: &Scene, source: &str) -> bool {
        self.hidden_clip_owners.get(source).is_some_and(|owners| owners.iter().all(|&row|
            scene.row(row).is_some_and(|owner| scene.hidden_from_paint_order(owner.id()))))
    }

    fn bytes(&self) -> usize {
        weights::set(&self.blend_related_ids) + weights::set(&self.mask_source_ids)
            + self.hidden_clip_owners.allocation_size()
            + self.hidden_clip_owners.iter().map(|(id, rows)| id.capacity() + weights::capacity(rows)).sum::<usize>()
    }
}

/// Whether `id` or one of its ancestors was culled from the indexed plan.
fn under_culled_root(scene: &Scene, index: &PlanIndex, id: &str) -> bool {
    if index.culled_roots.is_empty() {
        return false;
    }
    let mut cur = id;
    for _ in 0..64 {
        if index.culled_roots.contains(cur) {
            return true;
        }
        match scene
            .parent_map()
            .get(cur)
            .or_else(|| scene.transient_parent_of().get(cur))
        {
            Some(p) => cur = p.as_str(),
            None => return false,
        }
    }
    true // runaway chain: be conservative
}

/// A tess entry's bytes, including an outline's second-band slot.
fn tess_entry_bytes(e: &TessEntry) -> usize {
    geometry_bytes(&e.geometry) + e.outline_alt.as_ref().map_or(0, |(_, g)| geometry_bytes(g))
}

fn geometry_bytes(g: &TessellatedGeometry) -> usize {
    g.positions.len() * 4 + g.vertex_types.len() * 4 + 64
}

fn batch_bytes(b: &crate::seams::TextQuadBatch) -> usize {
    // Marginal heap of a cached text batch, by lane. The MSDF lane carries
    // `quads` + the `ops` draw stream; the vector lane (`b.vector`) carries no
    // quads/ops and instead owns a per-node quadratic-curve point buffer plus
    // per-glyph instance records — count those ACTUAL `Vec` payloads.
    //
    // This replaces a flat 64 KiB/vector-node estimate that charged a 3-glyph
    // label the same as a full paragraph. A realistic node is ~5-10 KiB, so the
    // flat estimate saturated the 48 MB text LRU at ~750 nodes and thrashed
    // dense decks: a 1k-richtext deck overflowed the budget, evicting live
    // batches it immediately re-prepared (a periodic full text re-shape, the
    // intermittent-drag-jank spike). Curve points are still duplicated per node
    // — sharing them across nodes (the real RSS win) is future work.
    let mut bytes = 256;
    bytes += b.ops.len() * 512;
    // A shader text fill carries a fixed packed instance
    // (`PackedFill`'s two `[f32; 64]` arrays + colors/bounds + a module id) —
    // ~0.5 KiB, not trivial on a shader-text deck, so count it explicitly.
    // vork: moda-text-draw's TextQuadBatch carries no shader fill (text shader fills are not ported).
    if let Some(v) = &b.vector {
        bytes += vector_batch_bytes(v);
    }
    // A Degraded-tier batch owns its RGBA raster (the node's box at its raster
    // scale). Uncounted, a text cache of them never reached its budget: a 100k-node
    // document planned before its fonts arrived kept 16k rasters, 1.3 GB of wasm
    // heap, against a counted 4 MB.
    if let Some(d) = &b.degraded {
        bytes += d.raster.rgba.len() + std::mem::size_of::<moda_text::degraded::DegradedRaster>();
    }
    bytes
}

/// Marginal heap of a vector text node's draw ops: the per-node curve buffer
/// (6 f32 per quadratic segment), the `VectorOp` structs themselves, and each
/// op's per-glyph instance records + local-space triangle list. Counted as
/// `len * size_of::<T>()` (consistent with `geometry_bytes` and the `quads`
/// term above) so it scales with content rather than being a flat constant.
///
/// Contract: this is *logical* accounting like the rest of the file — it counts
/// the dominant variable-length payloads and excludes `Vec` spare capacity and
/// the small fixed side payloads (`gradient`, `node_id`/key strings). The batch's
/// `fill_shader` is counted by the caller (`batch_bytes`) since it lives on
/// `TextQuadBatch`. Excluded items are bounded per node and the budgets are
/// deliberately conservative, so any drift is small and non-correctness-bearing
/// (eviction only re-computes). If a future `VectorNodeOps` field carries
/// significant heap, add it here.
fn vector_batch_bytes(v: &moda_text::vector::VectorNodeOps) -> usize {
    let mut bytes = v.curve_points.len() * std::mem::size_of::<f32>();
    bytes += v.ops.len() * std::mem::size_of::<moda_text::vector::VectorOp>();
    for op in &v.ops {
        bytes +=
            op.records.len() * std::mem::size_of::<moda_text::vector::ops::VectorAnalyticRecord>();
        bytes += op.tris.len() * std::mem::size_of::<f32>();
    }
    for e in &v.emojis {
        bytes += std::mem::size_of::<moda_text::vector::VectorEmoji>() + e.cluster.len();
    }
    bytes
}

/// Allocations reachable from the page. One inventory covers masks, nested
/// captures, glyph records, texels and meshes shared between occurrences.
fn plan_bytes(plan: &CoveragePlan) -> usize {
    let mut inventory = weights::Allocations::default();
    std::mem::size_of::<CoveragePlan>() + plan.page_id.capacity() + inventory.masks(&plan.masks)
        + inventory.item_vec(&plan.items) + plan.item_order.as_ref().map_or(0, crate::plan_items::ItemOrder::bytes)
}

/// Additive cost for a local replacement. The fitted offscreen plan reconciles
/// changes in shared allocation ownership before making its eviction decision.
pub(crate) fn items_bytes(items: &[crate::seams::DrawItem]) -> usize {
    weights::Allocations::per_reference().items(items)
}

impl PlanCaches {
    /// Advance the frame tick (call once per render sweep) — drives LRU.
    pub fn begin_frame(&mut self) {
        self.tick += 1;
    }

    pub fn clear(&mut self) {
        *self = PlanCaches {
            tess_hits: self.tess_hits,
            tess_misses: self.tess_misses,
            text_hits: self.text_hits,
            text_misses: self.text_misses,
            evictions: self.evictions,
            // Incremental patch statistics survive a cache clear.
            incr_static_hits: self.incr_static_hits,
            incr_patch_hits: self.incr_patch_hits,
            incr_patch_nodes: self.incr_patch_nodes,
            incr_placement_nodes: self.incr_placement_nodes,
            incr_fallbacks: self.incr_fallbacks,
            incremental_disabled: self.incremental_disabled,
            ..PlanCaches::default()
        };
    }

    /// The text cache's byte budget (`text_budget_bytes`, or [`TEXT_BUDGET_BYTES`]). An over-budget insert trims to
    /// 7/8 of it, so eviction scans are amortised.
    fn text_budget(&self) -> usize {
        if self.text_budget_bytes > 0 {
            self.text_budget_bytes
        } else {
            TEXT_BUDGET_BYTES
        }
    }

    /// Put an outlined node's geometry (`key` includes the band; `base` not).
    pub(crate) fn outline_put(
        &mut self,
        node_id: &str,
        key: u64,
        base: u64,
        geometry: Arc<TessellatedGeometry>,
    ) {
        // Keep the previous band of the same outline as the second slot.
        let keep = self.tess.get_mut(node_id).and_then(|e| {
            (e.outline_base == Some(base) && e.key != key)
                .then(|| (e.key, e.geometry.clone()))
        });
        // Anything in flight for another key of this node is now stale.

        self.tess_put(node_id, key, geometry);
        if let Some(e) = self.tess.get_mut(node_id) {
            e.outline_base = Some(base);
            if let Some((_, g)) = &keep {
                self.tess_bytes += geometry_bytes(g);
            }
            e.outline_alt = keep;
        }
    }

    // --- R3: body tessellation -------------------------------------------
    pub(crate) fn tess_get(&mut self, node_id: &str, key: u64) -> Option<Arc<TessellatedGeometry>> {
        match self.tess.get_mut(node_id) {
            Some(e) if e.key == key => {
                e.tick = self.tick;
                self.tess_hits += 1;
                Some(e.geometry.clone()) // Arc pointer-bump, not a vertex-stream copy
            }
            // An outline's second band: swap it to the front (LRU of two).
            Some(e) if e.outline_alt.as_ref().is_some_and(|(k, _)| *k == key) => {
                let (k, g) = e.outline_alt.take().expect("checked");
                let old = std::mem::replace(&mut e.geometry, g);
                e.outline_alt = Some((e.key, old));
                e.key = k;
                e.tick = self.tick;
                self.tess_hits += 1;
                Some(e.geometry.clone())
            }
            _ => {
                self.tess_misses += 1;
                None
            }
        }
    }

    pub(crate) fn tess_put(&mut self, node_id: &str, key: u64, geometry: Arc<TessellatedGeometry>) {
        let bytes = geometry_bytes(&geometry);
        if let Some(old) = self.tess.insert(
            node_id.to_string(),
            TessEntry {
                key,
                geometry,
                tick: self.tick,
                outline_base: None,
                outline_alt: None,

            },
        ) {
            self.tess_bytes = self.tess_bytes.saturating_sub(tess_entry_bytes(&old));
        }
        self.tess_bytes += bytes;
        if self.tess_bytes > TESS_BUDGET_BYTES && self.tess_evict_blocked != Some(self.tick) {
            // Down to a low-water mark, and never the geometry this frame or the last drew: the frame's plan holds
            // those `Arc`s anyway (dropping them frees nothing), and a view larger than the budget (a 20k-layer
            // document at fit) evicted its own geometry mid-frame and re-tessellated all of it every frame.

            let evicted = evict_stale(
                &mut self.tess,
                self.tick,
                |e| e.tick,
                tess_entry_bytes,
                &mut self.tess_bytes,
                TESS_BUDGET_BYTES / 8 * 7,
                |_| false,
            );
            self.evictions += evicted;
            if self.tess_bytes > TESS_BUDGET_BYTES {
                self.tess_evict_blocked = Some(self.tick);
            }
        }
    }

    // --- R3: line tessellation (multi-part) --------------------------------
    pub(crate) fn line_tess_get(
        &mut self,
        node_id: &str,
        key: u64,
    ) -> Option<Vec<TessellatedGeometry>> {
        match self.line_tess.get_mut(node_id) {
            Some(e) if e.key == key => {
                e.tick = self.tick;
                self.tess_hits += 1;
                Some(e.geometries.clone())
            }
            _ => {
                self.tess_misses += 1;
                None
            }
        }
    }

    pub(crate) fn line_tess_put(
        &mut self,
        node_id: &str,
        key: u64,
        geometries: &[TessellatedGeometry],
    ) {
        let bytes: usize = geometries.iter().map(geometry_bytes).sum();
        if let Some(old) = self.line_tess.insert(
            node_id.to_string(),
            LineTessEntry {
                key,
                geometries: geometries.to_vec(),
                tick: self.tick,
            },
        ) {
            self.line_tess_bytes = self
                .line_tess_bytes
                .saturating_sub(old.geometries.iter().map(geometry_bytes).sum());
        }
        self.line_tess_bytes += bytes;
        if self.line_tess_bytes > TESS_BUDGET_BYTES / 4 && self.line_evict_blocked != Some(self.tick) {
            let evicted = evict_stale(
                &mut self.line_tess,
                self.tick,
                |e| e.tick,
                |e| e.geometries.iter().map(geometry_bytes).sum(),
                &mut self.line_tess_bytes,
                TESS_BUDGET_BYTES / 4 / 8 * 7,
                |_| false,
            );
            self.evictions += evicted;
            if self.line_tess_bytes > TESS_BUDGET_BYTES / 4 {
                self.line_evict_blocked = Some(self.tick);
            }
        }
    }

    /// Look for the scale-independent entry before charging one hit or miss to the lookup.
    pub(crate) fn has_text_content(&self, item_id: &str, key: u64) -> bool {
        self.text.get(item_id).is_some_and(|e| e.key == key)
    }

    // --- R4: text item cache ------------------------------------------------
    pub(crate) fn text_get(
        &mut self,
        item_id: &str,
        key: u64,
    ) -> Option<Box<crate::seams::TextQuadBatch>> {
        self.text_get_with_gap(item_id, key).map(|(b, _)| b)
    }

    pub(crate) fn text_get_with_gap(
        &mut self,
        item_id: &str,
        key: u64,
    ) -> Option<(Box<crate::seams::TextQuadBatch>, Option<(f64, f64)>)> {
        match self.text.get_mut(item_id) {
            Some(e) if e.key == key => {
                e.tick = self.tick;
                self.text_hits += 1;
                Some((e.batch.clone(), e.gap))
            }
            _ => {
                self.text_misses += 1;
                None
            }
        }
    }

    pub(crate) fn text_put(
        &mut self,
        item_id: &str,
        key: u64,
        batch: &crate::seams::TextQuadBatch,
    ) {
        self.text_put_with_gap(item_id, key, batch, None)
    }

    pub(crate) fn text_put_with_gap(
        &mut self,
        item_id: &str,
        key: u64,
        batch: &crate::seams::TextQuadBatch,
        gap: Option<(f64, f64)>,
    ) {
        let bytes = batch_bytes(batch);
        if let Some(old) = self.text.insert(
            item_id.to_string(),
            TextEntry {
                key,
                batch: Box::new(batch.clone()),
                gap,
                tick: self.tick,
            },
        ) {
            self.text_bytes = self.text_bytes.saturating_sub(batch_bytes(&old.batch));
        }
        self.text_bytes += bytes;
        let budget = self.text_budget();
        if self.text_bytes > budget {
            // Down to the low-water mark, not just under the budget: `evict_oldest` clones and sorts every key,
            // and trimming only to the budget re-ran it on every insert once text outgrew it (thousands of text
            // nodes: O(n²) per frame).
            let evicted = evict_oldest(
                &mut self.text,
                self.tick,
                |e| e.tick,
                |e| batch_bytes(&e.batch),
                &mut self.text_bytes,
                budget / 8 * 7,
            );
            self.evictions += evicted;
        }
    }

    /// The canvas root's display list the caller built for this plan, if it was built from `scene`.
    pub(crate) fn frame_canvas_dl(&self, scene: &moda_scene::Scene) -> Option<Arc<moda_scene::DisplayList>> {
        self.frame_canvas_dl
            .as_ref()
            .filter(|(at, _)| *at == std::ptr::from_ref(scene) as usize)
            .map(|(_, dl)| dl.clone())
    }

    pub fn stats_json(&self) -> String {
        format!(
            "{{\"tess_entries\":{},\"tess_bytes\":{},\"line_entries\":{},\"line_bytes\":{},\"text_entries\":{},\"text_bytes\":{},\"tess_hits\":{},\"tess_misses\":{},\"text_hits\":{},\"text_misses\":{},\"evictions\":{},\"incr_static_hits\":{},\"incr_patch_hits\":{},\"incr_patch_nodes\":{},\"incr_placement_nodes\":{},\"incr_fallbacks\":{}}}",
            self.tess.len(),
            self.tess_bytes,
            self.line_tess.len(),
            self.line_tess_bytes,
            self.text.len(),
            self.text_bytes,
            self.tess_hits,
            self.tess_misses,
            self.text_hits,
            self.text_misses,
            self.evictions,
            self.incr_static_hits,
            self.incr_patch_hits,
            self.incr_patch_nodes,
            self.incr_placement_nodes,
            self.incr_fallbacks,
        )
    }
}

/// Evict least-recently-used entries until `bytes` <= `budget`. Returns the
/// eviction count. Linear scan — runs only on budget crossings.
/// [`evict_oldest`] of the entries not used this frame or the last (`tick_of(v) + 1 < now`) and not `in_use`: the
/// working set stays, even over `budget`.
fn evict_stale<V>(
    map: &mut HashMap<String, V>,
    now: u64,
    tick_of: impl Fn(&V) -> u64,
    bytes_of: impl Fn(&V) -> usize,
    bytes: &mut usize,
    budget: usize,
    in_use: impl Fn(&str) -> bool,
) -> u64 {
    let mut entries: Vec<(String, u64)> = map
        .iter()
        .filter(|(k, v)| tick_of(v).saturating_add(1) < now && !in_use(k))
        .map(|(k, v)| (k.clone(), tick_of(v)))
        .collect();
    entries.sort_by_key(|(_, t)| *t);
    let mut evicted = 0;
    for (k, _) in entries {
        if *bytes <= budget {
            break;
        }
        if let Some(v) = map.remove(&k) {
            *bytes = bytes.saturating_sub(bytes_of(&v));
            evicted += 1;
        }
    }
    evicted
}

fn evict_oldest<V>(
    map: &mut HashMap<String, V>,
    _now: u64,
    tick_of: impl Fn(&V) -> u64,
    bytes_of: impl Fn(&V) -> usize,
    bytes: &mut usize,
    budget: usize,
) -> u64 {
    let mut entries: Vec<(String, u64)> =
        map.iter().map(|(k, v)| (k.clone(), tick_of(v))).collect();
    entries.sort_by_key(|(_, t)| *t);
    let mut evicted = 0;
    for (k, _) in entries {
        if *bytes <= budget {
            break;
        }
        if let Some(v) = map.remove(&k) {
            *bytes = bytes.saturating_sub(bytes_of(&v));
            evicted += 1;
        }
    }
    evicted
}

/// The content key of one layer capture (`LayerBlurGroup::cache_key`): a hash
/// of everything its pixels are made of, so the retained engine can reuse the
/// filtered result (`layer_blur::cache`) whenever it recurs.
///
/// - each captured node (`ids`: the node, or a subtree root and its
///   descendants): its content hash (the resident map's, or computed), its
///   canvas affine (a parent's move or resize reaches every descendant this
///   way), its effective opacity (ancestors folded in) and clip;
/// - the capture: region, size and filters — the camera's part (the zoom
///   bucket and the capture window the region was cut to) arrives through
///   these;
/// - scene-wide inputs: plan-relevant render options (the curve band
///   included), the asset store and the font host generations.
///
/// `None` without resident hashes (exports and one-shot renders re-render
/// every capture, as before) or when the capture holds a shader-clock
/// consumer (its pixels change every frame).
pub(crate) fn capture_key<'a>(
    scene: &Scene,
    hashes: Option<&dyn NodeHashSource>,
    ids: impl IntoIterator<Item = &'a str>,
    group: &crate::seams::LayerBlurGroup,
    page_clip: [f32; 4],
    opts: &RenderOptions,
) -> Option<u64> {
    let hashes = hashes?;
    if crate::layer_blur::translation::eligible(group) {
        let mut h = FxHasher::default();
        h.write(b"local-solid-capture");
        h.write_u32(group.width_px);
        h.write_u32(group.height_px);
        h.write(format!("{:?}", group.filters.blur).as_bytes());
        h.write_u32(group.filters.blur_border);
        for item in &group.items {
            if let crate::seams::DrawItem::Body { node_id, z, phase, instance, blend_mode, .. } = item {
                h.write(node_id.as_bytes());
                h.write_u64(z.to_bits());
                h.write(format!("{phase:?}{blend_mode:?}").as_bytes());
                h.write(bytemuck::bytes_of(&crate::layer_blur::translation::instance(instance, group.region_canvas)));
            }
        }
        fold_item_geometry(&mut h, &group.items);
        fold_render_options_fingerprint(&mut h, opts);
        return Some(h.finish());
    }
    let mut h = FxHasher::default();
    h.write(b"layer-capture");
    for id in ids {
        h.write(id.as_bytes());
        h.write_u8(0xff);
        match hashes.read_hash(scene, id) {
            Some(nh) => h.write_u64(nh.full),
            None => match scene.effective_node(id) {
                Some(n) => h.write_u64(node_hashes_in(scene, n).full),
                None => h.write_u8(1),
            },
        }
        if let Some(a) = scene.node_canvas_affine(id) {
            for v in [a.a, a.b, a.c, a.d, a.e, a.f] {
                h.write_u64(v.to_bits());
            }
        }
        h.write_u64(scene.ancestor_opacity_product(id).to_bits());
        match crate::effects::effective_page_clip(scene, id, page_clip) {
            Some(c) => c.iter().for_each(|v| h.write_u32(v.to_bits())),
            None => h.write_u8(2),
        }
        // The rounded clip of the innermost rounded clipping frame above: a
        // subtree's descendants (and nested composites) are cut by it inside
        // the capture.
        fold_clip_rrect(&mut h, crate::effects::clip_rounded_rect(scene, id));
    }
    fn fold_group(h: &mut FxHasher, g: &crate::seams::LayerBlurGroup) {
        for v in g.region_canvas {
            h.write_u64(v.to_bits());
        }
        h.write_u32(g.width_px);
        h.write_u32(g.height_px);
        // Includes the kind and shadow blend, as well as filter parameters.
        h.write(format!("{:?}", g.filters).as_bytes());
        // Every descendant capture's geometry and clips affect these pixels. A descendant that has its own
        // content key folds that key: it covers the descendant's filters, geometry and items. A capture can hold
        // a copy of items that include captures of copies (a filter shadow's capture of a node whose child has a
        // filter shadow), so walking into each one would visit a chain of n of them 2^(n-1) times.
        for item in &g.items {
            if let crate::seams::DrawItem::LayerBlurComposite {
                group: nested, page_clip_bounds, clip_rrect, ..
            } = item {
                // A child may reuse local pixels at a different canvas origin.
                // Its placement still changes the enclosing capture's pixels.
                for v in nested.region_canvas { h.write_u64(v.to_bits()); }
                match nested.cache_key {
                    Some(key) => h.write_u64(key),
                    None => fold_group(h, nested),
                }
                page_clip_bounds.iter().for_each(|v| h.write_u32(v.to_bits()));
                fold_clip_rrect(h, *clip_rrect);
            }
        }
    }
    fold_group(&mut h, group);
    // What the captured items draw with: a live frame may plan them from an outline, curve or strand body it is
    // still refining (the previous band's, or a placeholder) and refine them on later frames; the nodes' hashes
    // don't change then, but these do, so a capture of the stand-in isn't served for the refined items.
    fold_item_geometry(&mut h, &group.items);
    fold_render_options_fingerprint(&mut h, opts);
    h.write_i32(crate::tess::curve_quality::current_band());
    h.write_u64(moda_assets::engine_seam::store_generation());
    h.write_u64(crate::text_draw::text_host_generation());
    Some(h.finish())
}

/// The geometry a capture's items draw with, into its content key: each body's tessellation (its `Arc`, vertex
/// count, bounds and a sample of its positions: the same tessellation is the same `Arc` while a cache holds it, and
/// the sample keeps a reused address from passing for another) and each text batch, nested captures' items
/// included. A text batch is named by what it draws: its content key (`TextQuadBatch::content_key`, the key its
/// plan cache keeps it under) and its placement (z, model, clip). A cache hit is a copy in a new allocation, so
/// its address says nothing: named by address, an unchanged text looked new on every rebuilt plan and a capture
/// holding it was rendered again each time. A batch with no content key (built outside a plan cache) has only
/// its address.
fn fold_item_geometry(h: &mut FxHasher, items: &[crate::seams::DrawItem]) {
    for item in items {
        match item {
            crate::seams::DrawItem::Body { geometry, .. } => {
                h.write_usize(Arc::as_ptr(geometry) as usize);
                h.write_u32(geometry.vertex_count);
                geometry.bounds.iter().for_each(|v| h.write_u32(v.to_bits()));
                let p = &geometry.positions;
                for i in [0, p.len() / 3, p.len() / 2, p.len().saturating_sub(1)] {
                    if let Some(v) = p.get(i) {
                        h.write_u32(v.to_bits());
                    }
                }
            }
            crate::seams::DrawItem::Text(batch) => match batch.content_key {
                Some(key) => {
                    h.write_u64(key);
                    h.write_u64(batch.z.to_bits());
                    batch.model_matrix.iter().chain(&batch.page_clip_bounds).for_each(|v| h.write_u32(v.to_bits()));
                }
                None => h.write_usize(Arc::as_ptr(batch) as usize),
            },
            // A nested capture's own content key already holds its items' geometry (see `fold_group`).
            crate::seams::DrawItem::LayerBlurComposite { group, .. } => match group.cache_key {
                Some(key) => h.write_u64(key),
                None => fold_item_geometry(h, &group.items),
            },
            _ => {}
        }
    }
}

/// A capture composite's rounded clip (`seams::ClipRRect`) into a content key.
fn fold_clip_rrect(h: &mut FxHasher, rr: Option<crate::seams::ClipRRect>) {
    match rr {
        Some(r) => {
            h.write_u8(1);
            r.rect
                .iter()
                .chain(r.radii.iter())
                .for_each(|v| h.write_u32(v.to_bits()));
        }
        None => h.write_u8(0),
    }
}

/// Whether a built plan bakes the shader clock anywhere (R5 hit condition).
pub(crate) fn plan_uses_shader_time(plan: &CoveragePlan) -> bool {
    masks_use_shader_time(&plan.masks)
}

fn masks_use_shader_time(masks: &[crate::seams::MaskPlan]) -> bool {
    masks.iter().any(|mask| mask.image.as_ref().and_then(|image| image.rendered.as_ref())
        .is_some_and(|rendered| rendered.shader_clock.get().copied().unwrap_or(true)))
}

/// Hash the options that affect plan content. Rendered masks track their clock separately.
fn fold_render_options_fingerprint(h: &mut FxHasher, opts: &RenderOptions) {
    h.write_u64(opts.scale.to_bits());
    // Brush plan P8: inside a camera build of a scene with outlined strokes,
    // the outline zoom band is plan content (outlines are refit per band).
    // Absent otherwise, so other scenes keep zoom-invariant camera plans.
    if let Some(band) = crate::tess::curve_quality::camera_key_band() {
        h.write(b"outline-band");
        h.write_i32(band);
    }
    h.write_u8(opts.transparent as u8);
    h.write_u8(opts.preserve_page_fill as u8);
    h.write_u32(opts.sample_count);
    h.write_u8(opts.text_masks_available.unwrap_or(true) as u8);
    // Working colour affects the plan for luminance masks.
    h.write_u8(opts.working.is_some() as u8);
    if let Some(working) = opts.working {
        for conversion in [working.from_srgb, working.to_srgb] {
            h.write_u8(conversion.decode as u8);
            h.write_u8(conversion.encode as u8);
            h.write_u8(conversion.matrix.is_some() as u8);
            if let Some(matrix) = conversion.matrix {
                for coefficient in matrix.into_iter().flatten() {
                    h.write_u32(coefficient.to_bits());
                }
            }
        }
    }
    fold_node_filters(h, opts);
}

pub(crate) fn fold_node_filters(h: &mut FxHasher, opts: &RenderOptions) {
    for ids in [&opts.exclude, &opts.isolate] {
        h.write_usize(ids.len());
        for id in ids {
            h.write_usize(id.len());
            h.write(id.as_bytes());
        }
    }
}

/// Publish checked draw spans into disjoint slices. The serial case also covers
/// physical storage reordered by a previous retained unit replacement.
fn write_placed_spans(
    pages: &mut [span_pages::Page], page_base: usize,
    items: &mut [crate::seams::DrawItem], item_base: usize,
    affected: &[u32], placed: &[(std::ops::Range<usize>, crate::frame::Placed)], grain: usize,
) {
    use span_pages::PAGE;
    if affected.len() > grain {
        let pivot = affected[affected.len() / 2] as usize / PAGE;
        let mid = affected.partition_point(|&i| (i as usize / PAGE) < pivot);
        if mid != 0 {
            let page = pivot - page_base;
            let next_item = placed[mid].0.start;
            let (left, right) = pages.split_at_mut(page);
            let (a, b) = items.split_at_mut(next_item - item_base);
            rayon::join(
                || write_placed_spans(left, page_base, a, item_base, &affected[..mid], &placed[..mid], grain),
                || write_placed_spans(right, pivot, b, next_item, &affected[mid..], &placed[mid..], grain),
            );
            return;
        }
    }
    for (&i, (range, value)) in affected.iter().zip(placed) {
        value.apply(&mut items[range.start - item_base..range.end - item_base]);
        if let Some(bits) = value.opacity_bits() {
            let i = i as usize;
            let span = &mut pages[i / PAGE - page_base].write()[i % PAGE];
            span.placement.as_mut().expect("checked placement").opacity_bits = bits;
        }
    }
}

/// A fractional group composites its complete source at StackEnd. Check ownership on this
/// execution range before changing the scalar. Nested filters also carry source-derived keys.
fn patch_group_opacity(
    scene: &Scene, plan: &mut Arc<CoveragePlan>, seg: &crate::frame::CameraSegment,
    sources: &parts::Sources, parts: Option<&parts::PlanParts>, ui: usize, changed: &[u32], written: &mut PlanWritten,
) -> Option<u32> {
    use crate::seams::{BackdropOp, DrawItem};
    let compact;
    let (unit, row) = if let Some(unit) = seg.units.get(ui) {
        (unit, unit.opacity_row()?)
    } else {
        let (source, row) = parts?.opacity_source(ui)?;
        compact = source;
        (&compact, row)
    };
    if changed.binary_search(&row).is_err() { return None; }
    let node = scene.row(row)?;
    if !moda_scene::scene::is_opacity_group(node) { return None; }
    let changed_source = |id: &str| scene.row_of(id).is_some_and(|r| changed.binary_search(&r).is_ok());
    if sources.names(seg, scene, unit).any(|id| id != node.id() && changed_source(id)) { return None; }
    let mut parent = scene.group_parent_of(node.id());
    while let Some(id) = parent {
        if changed_source(id) { return None; }
        parent = scene.group_parent_of(id);
    }
    let value = node.opacity() as f32;
    let at = if unit.item_start == parts::ABSENT {
        let parts = parts?;
        let write = parts.opacity_write(ui, row, value)?;
        let Some(at) = parts.addresses.physical(write.logical) else {
            written.evicted.push(write);
            return Some(row);
        };
        at
    } else { seg.prefix + unit.item_start + unit.item_len - 1 };
    let DrawItem::BackdropSplit(old) = plan.items.get(at)? else { return None };
    let BackdropOp::StackEnd { opacity, .. } = old.op else { return None };
    if old.node_id != node.id() { return None; }
    if opacity != value {
        let DrawItem::BackdropSplit(end) = &mut Arc::make_mut(plan).items[at] else { unreachable!() };
        let BackdropOp::StackEnd { opacity, .. } = &mut end.op else { unreachable!() };
        *opacity = value;
        written.placed.push(crate::frame::ItemWrite::opacity(at));
    }
    Some(row)
}

/// Recompute the target and filter uniforms when the writer certifies that only the
/// Effects field changed. A capture containing another composition needs the ordinary
/// range executor to update its targets, masks or output dependencies.
fn patch_capture_parameters(
    scene: &Scene,
    plan: &mut Arc<CoveragePlan>,
    seg: &crate::frame::CameraSegment,
    ui: usize,
    records: Option<&OutputRecords>,
    dl: &[moda_scene::types::PaintItem],
    changed: &[u32],
    opts: &RenderOptions,
    hashes: &dyn NodeHashSource,
    written: &mut PlanWritten,
) -> Option<()> {
    use crate::seams::{BodyPaint, DrawItem};
    let unit = &seg.units[ui];
    if !unit.details().mask_reads.is_empty()
        || !unit.compositions.is_empty()
        || unit.culled
        || unit.part_culled
        || !unit.details().skipped.is_empty()
        || !unit.details().dropped.is_empty()
        || unit.details().captures.len() > 1
    {
        return None;
    }
    let start = seg.prefix + unit.item_start;
    let outputs = plan.items.get(start..start + unit.item_len)?;
    let (slot, owner, old) = outputs.iter().enumerate().find_map(|(slot, item)| match item {
        DrawItem::LayerBlurComposite {
            node_id,
            group,
            mask: None,
            clip_rrect: None,
            ..
        } => Some((slot, node_id, group)),
        _ => None,
    })?;
    let row = scene.row_of(owner)?;
    if changed.binary_search(&row).is_err()
        || !old.coverage.is_empty()
        || !old.masks.is_empty()
        || !old.items.iter().all(|i| {
            matches!(
                i,
                DrawItem::Body {
                    paint: BodyPaint::Solid,
                    mask: None,
                    ..
                }
            )
        })
        || !outputs
            .iter()
            .enumerate()
            .all(|(i, item)| i == slot || matches!(item, DrawItem::Body { .. }))
        || old.filters.composite_blend != Default::default()
    {
        return None;
    }
    if dl.iter().any(|item| {
        item.node_id != *owner
            && scene
                .row_of(&item.node_id)
                .is_some_and(|row| changed.binary_search(&row).is_ok())
    }) {
        return None;
    }
    let node = scene.effective_node(owner)?;
    let effects = node
        .fields()
        .get_known(moda_scene::value::K::Effects)?
        .as_array()?;
    let [effect] = effects.as_slice() else { return None };
    let same = match effect["type"].as_str()? {
        "layer-blur" => {
            matches!(old.filters.blur, Some(crate::seams::BlurFilter::Uniform { .. }))
                && old.filters.drop_shadows.is_empty()
        }
        "filter-shadow" => old.filters.drop_shadows.len() == 1 && old.filters.blur.is_none(),
        _ => false,
    };
    if !same
        || old.filters.texture.is_some()
        || old.filters.shadow.is_some()
        || old.filters.feather.is_some()
        || !old.filters.noise.is_empty()
        || !old.filters.color_matrices.is_empty()
    {
        return None;
    }
    let (_, ctx) = crate::frame::page_range_contexts(scene, plan, seg, opts);
    let ctx = ctx?;
    let mut skipped = Vec::new();
    let params = if let [recipe] = unit.details().captures.as_slice() {
        if recipe.row != row
            || !recipe.isolations.is_empty()
            || !recipe.strip_masks.is_empty()
            || !recipe.mask_table.is_empty()
        {
            return None;
        }
        crate::layer_blur::plan::compute_subtree_layer_blur_params(
            scene,
            node,
            &ctx,
            records.and_then(|r| r.capture_bounds(row))
                .or_else(|| crate::layer_blur::shadow::all_items_canvas_bounds(&old.items))?,
            &mut skipped,
        )?
    } else {
        if dl.len() != 1 || dl[0].node_id != *owner || node.node_type() == "richtext" {
            return None;
        }
        let drawn = || crate::layer_blur::shadow::all_items_canvas_bounds(&old.items);
        crate::layer_blur::plan::compute_layer_blur_params_over(scene, node, &ctx, drawn, &mut skipped)?
    };
    if !skipped.is_empty() || params.filters.shadows_alone != old.filters.shadows_alone {
        return None;
    }
    let DrawItem::LayerBlurComposite { group, .. } = &mut Arc::make_mut(plan).items[start + slot] else {
        unreachable!()
    };
    let group = Arc::make_mut(group);
    group.region_canvas = params.region_canvas;
    group.extent_canvas = params.extent_canvas;
    group.reach_canvas = params.reach_canvas;
    group.width_px = params.width_px;
    group.height_px = params.height_px;
    group.filters = params.filters;
    group.cache_key = capture_key(
        scene,
        Some(hashes),
        dl.iter()
            .filter(|item| unit.details().captures.is_empty() || item.is_primary_node_item)
            .map(|item| item.node_id.as_str()),
        group,
        seg.page_clip,
        opts,
    );
    written.replaced.push(((start + slot) as u32, DrawItem::Vacant));
    Some(())
}

/// The plan of a host-owned offscreen scene, with its index. The index owns the plan: the host plans through
/// it, and between frames nothing else holds the plan, so a placement or a node prepared again writes the plan's
/// items in place (`Arc::make_mut`: a caller that still holds an earlier state keeps it, and that write copies).
/// Source metadata, emitted spans and GPU record mirrors share the offscreen retention budget.
#[derive(Default)]
pub struct FramePlanIndex {
    parameter_writes: Option<(u64, u64, moda_scene::value::K)>,
    #[cfg(test)]
    test_budget: Option<usize>,
    entry: Option<FrameIndexEntry>,
    /// What the last `place` or `patch` wrote into the plan ([`Self::take_written`]).
    written: PlanWritten,
}

/// The items a plan's owner wrote in place, by item index, between two states of the plan (the engine's
/// `frame::ChangedItems::Written`).
#[derive(Default)]
pub struct PlanWritten {
    /// New output changed the recorded draw layout.
    pub recollect: bool,
    /// A derived mask value changed; recorded passes must bind the current atlas.
    pub masks_changed: bool,
    /// Fields addressed directly to their logical slots while the CPU items are evicted.
    pub evicted: Vec<crate::frame::LogicalWrite>,
    /// Bodies whose placement values were written by their recipe, ascending.
    pub placed: Vec<crate::frame::ItemWrite>,

    /// Items the planner prepared again, each with the item it replaced.
    pub replaced: Vec<(u32, crate::seams::DrawItem)>,
}

impl PlanWritten {
    /// A consumer requested a complete snapshot. All writes now address its
    /// painter-order items, including values previously addressed directly to the packet.
    pub fn materialize(&mut self, addresses: &ItemAddresses) -> Option<()> {
        for at in self.placed.iter_mut().map(|w| &mut w.at).chain(self.replaced.iter_mut().map(|(at, _)| at)) {
            *at = u32::try_from(addresses.logical(*at as usize)?).ok()?;
        }
        self.placed.extend(self.evicted.drain(..).map(|w| match w.value {
            crate::frame::LogicalValue::Matrix(_) => crate::frame::ItemWrite::full(w.logical),
            crate::frame::LogicalValue::CompositeOpacity(_) => crate::frame::ItemWrite::opacity(w.logical),
        }));
        self.placed.sort_unstable_by_key(|w| w.at);
        self.replaced.sort_unstable_by_key(|(at, _)| *at);
        Some(())
    }
}

struct FrameIndexEntry {
    /// Index, frame header and mask state paid before the evictable span payloads.
    fixed_bytes: usize,
    reweigh: bool,
    /// Metadata pages re-expanded during a change are trimmed at the end of its draw.
    span_bytes: usize,
    packet_bytes: Option<usize>,
    pages_trimmed: bool,
    initial: std::sync::Weak<CoveragePlan>,
    parts: Option<parts::PlanParts>,
    pattern_slots: Option<PublicationIndex>,
    mask_observation: (usize, u64),
    unbound_mask_sources: Vec<crate::seams::MaskAssetReads>,
    layout_key: u64,
    generation: u64,
    publication: u64,
    plan: Arc<CoveragePlan>,
    warnings: Arc<PlanWarnings>,
    /// The host's key for the scene state the plan stands at, and the shader clock it was planned for.
    key: u64,
    shader_time_s: f64,
    built_at: u64,
    uses_shader_time: bool,
    bytes: usize,
    /// What following a change in place needs; `None` for a plan that is only held (a plan that reads the
    /// shader clock, or incremental planning switched off).
    patchable: Option<Patchable>,
}

/// Index top-level logical consumers before eviction. Nested capture pixels are
/// replaced through their ordinary draw tree once their root has been selected.
struct PublishedAsset {
    publication: moda_assets::engine_seam::Publication,
    users: weights::RetainedSet<usize>,
    texture_slot: Option<u64>,
}

#[derive(Default)]
struct PublicationIndex {
    assets: weights::RetainedMap<Arc<str>, PublishedAsset>,
    keys: weights::RetainedMap<usize, Vec<Arc<str>>>,
    capture_masks: weights::RetainedMap<usize, Vec<crate::seams::MaskAssetReads>>,
    page_masks: Vec<crate::seams::MaskAssetReads>,
}

impl PublicationIndex {
    fn build(plan: &CoveragePlan, store: &moda_assets::AssetStore) -> Option<Self> {
        let mut inputs = Self::default();
        inputs.page_masks = Self::masks(&plan.masks)?;
        for (at, item) in plan.items.iter().enumerate() { inputs.update(at, item, store)?; }
        Some(inputs)
    }

    fn masks(masks: &[crate::seams::MaskPlan]) -> Option<Vec<crate::seams::MaskAssetReads>> {
        masks.iter().filter_map(|m| m.image.as_ref())
            .map(|image| Some(image.rendered.as_ref()?.asset_reads.get_or_init(Default::default).clone())).collect()
    }

    fn stale(masks: &[crate::seams::MaskAssetReads], store: &moda_assets::AssetStore) -> bool {
        masks.iter().any(|cell| cell.reads.lock().unwrap().as_ref().is_none_or(|reads| !reads.matches(store)))
    }

    fn update(&mut self, at: usize, item: &crate::seams::DrawItem, store: &moda_assets::AssetStore) -> Option<()> {
        let previous = self.keys.remove(&at).unwrap_or_default();
        for key in &previous { self.assets.get_mut(key)?.users.remove(&at); }
        self.capture_masks.remove(&at);
        self.observe(at, item, store)?;
        for key in previous {
            if self.assets.get(&key)?.users.is_empty() { self.assets.remove(&key); }
        }
        Some(())
    }

    fn observe(&mut self, at: usize, item: &crate::seams::DrawItem, store: &moda_assets::AssetStore) -> Option<()> {
        use crate::seams::{DrawItem, BodyPaint};
        match item {
            DrawItem::Body { paint: BodyPaint::Pattern(p), .. } => {
                let key = p.texture.key.as_str();
                if !self.assets.contains_key(key) {
                    self.assets.insert(Arc::from(key), PublishedAsset {
                        publication: moda_assets::engine_seam::Publication::of(store, key),
                        users: Default::default(), texture_slot: p.texture_slot,
                    });
                }
                let key = self.assets.get_key_value(key)?.0.clone();
                if self.assets.get_mut(&key)?.users.insert(at) { self.keys.entry(at).or_default().push(key); }
            }
            DrawItem::LayerBlurComposite { group, .. } => {
                let masks = Self::masks(&group.masks)?;
                if !masks.is_empty() { self.capture_masks.entry(at).or_default().extend(masks); }
                for child in group.items.iter().chain(&group.coverage) { self.observe(at, child, store)?; }
            }
            DrawItem::Text(b) if b.paint_geometry_key.is_none() => return None,
            _ => {}
        }
        Some(())
    }

    fn changed(&mut self, store: &moda_assets::AssetStore) -> Vec<usize> {
        let mut changed: std::collections::BTreeSet<_> = self.capture_masks.iter()
            .filter_map(|(&at, masks)| Self::stale(masks, store).then_some(at)).collect();
        for (key, asset) in &mut self.assets {
            if !asset.publication.matches(store, key) {
                changed.extend(asset.users.iter().copied());
                asset.publication = moda_assets::engine_seam::Publication::of(store, key);
            }
        }
        changed.into_iter().collect()
    }

    fn mask_read_bytes(&self, allocations: &mut weights::Allocations) -> usize {
        self.capture_masks.values().flatten().chain(&self.page_masks).map(|r| allocations.mask_reads(r)).sum()
    }

    fn bytes(&self, allocations: &mut weights::Allocations) -> usize {
        self.assets.allocation_size() + self.keys.allocation_size() + self.capture_masks.allocation_size()
            + self.assets.iter().map(|(k, asset)| k.len() + 2 * std::mem::size_of::<usize>() + asset.users.allocation_size()).sum::<usize>()
            + self.keys.values().map(weights::capacity).sum::<usize>()
            + self.capture_masks.values().map(weights::capacity).sum::<usize>() + weights::capacity(&self.page_masks)
            + self.mask_read_bytes(allocations)
    }
}

/// An immutable scene source can keep its ordinary emitted plan across movie
/// publications. The parent owns its lifetime and replaces it on a source edit.
/// Projection and resource layout are the same inputs as for a page plan.
pub(crate) struct PublishedPlan {
    plan: Arc<CoveragePlan>,
    layout: u64,
    reads: moda_assets::engine_seam::AssetReads,
    inputs: PublicationIndex,
}

impl PublishedPlan {
    pub(crate) fn environment(opts: &RenderOptions) -> u64 {
        layout_frame_key(0, opts)
    }

    pub(crate) fn new(plan: Arc<CoveragePlan>, opts: &RenderOptions, before: u64,
        reads: moda_assets::engine_seam::AssetReads) -> Option<Self> {
        if plan_uses_shader_time(&plan) { return None; }
        let inputs = reads.with_rebound_images(|_| false, |store| {
            if Self::environment(opts) != before { return None; }
            store.map_or_else(|| Some(PublicationIndex::default()), |s| PublicationIndex::build(&plan, s))
        })??;
        Some(Self { plan, layout: before, reads, inputs })
    }

    pub(crate) fn current(&mut self, opts: &RenderOptions, origin: [f64; 2], size: [f64; 2]) -> Option<Arc<CoveragePlan>> {
        if self.plan.camera != origin || self.plan.page_size != size { return None; }
        let Self { reads, plan, inputs, .. } = self;
        // Validate all reads before changing the publication index. Only loaded
        // image bindings in the old layout can be replaced without emission.
        let store = reads.with_rebound_images(|key| inputs.assets.contains_key(key),
            |store| (self.layout == layout_frame_key(0, opts)).then(|| store.cloned()))??;
        if let Some(store) = store {
            for at in inputs.changed(&store) {
                if let Some(item) = replace_publication(&plan.items[at], &store, inputs, false) {
                    Arc::make_mut(plan).items[at] = item;
                }
            }
        }
        Some(plan.clone())
    }

    fn bytes(&self, allocations: &mut weights::Allocations) -> usize {
        allocations.source_plan(&self.plan) + self.inputs.bytes(allocations) + self.reads.retained_bytes()
    }
}

/// Rewrite one selected consumer. The index validated the supported source families at
/// collection, including nested captures. Private GPU slots are shared by publication key.
fn replace_publication(item: &crate::seams::DrawItem, store: &moda_assets::AssetStore,
    inputs: &mut PublicationIndex, restored: bool) -> Option<crate::seams::DrawItem> {
    use crate::seams::{BodyPaint, DrawItem};
    match item {
        DrawItem::Body { paint: BodyPaint::Pattern(p), .. } => {
            if p.texture.state != crate::paint::PatternTextureState::Loaded { return None; }
            let asset = moda_assets::engine_seam::loaded_asset(store, &p.texture.key)?;
            let texture = crate::paint::image_source::loaded_to_pattern_texture(&p.texture.key, asset);
            let binding = &mut inputs.assets.get_mut(p.texture.key.as_str())?.texture_slot;
            if texture.image.same_pixels(&p.texture.image) && binding.is_none_or(|slot| Some(slot) == p.texture_slot)
                && (!restored || p.texture_slot.is_some()) { return None; }
            let mut item = item.clone();
            let DrawItem::Body { paint: BodyPaint::Pattern(p), .. } = &mut item else { unreachable!() };
            static NEXT_SLOT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
            p.texture_slot = Some(*binding.get_or_insert_with(|| NEXT_SLOT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
            p.texture = texture;
            Some(item)
        }
        DrawItem::LayerBlurComposite { group, .. } => {
            let mut changes = |items: &[DrawItem]| items.iter().enumerate().filter_map(|(i, child)|
                replace_publication(child, store, inputs, restored).map(|replaced| (i, replaced))).collect::<Vec<_>>();
            let children = changes(&group.items);
            let coverage = changes(&group.coverage);
            let stale_mask = group.masks.iter().filter_map(|m| m.image.as_ref()).any(|image|
                image.rendered.as_ref().and_then(|r| r.asset_reads.get())
                    .is_none_or(|cell| PublicationIndex::stale(std::slice::from_ref(cell), store)));
            if children.is_empty() && coverage.is_empty() && !stale_mask { return None; }
            let mut item = item.clone();
            let DrawItem::LayerBlurComposite { group, .. } = &mut item else { unreachable!() };
            let group = Arc::make_mut(group);
            for (i, child) in children { group.items[i] = child; }
            for (i, child) in coverage { group.coverage[i] = child; }
            let revision = |key| {
                let mut h = FxHasher::default();
                h.write(b"capture-pixel-publication");
                h.write_u64(key);
                h.write_u64(moda_assets::engine_seam::store_generation());
                h.finish()
            };
            group.cache_key = group.cache_key.map(revision);
            Some(item)
        }
        _ => None,
    }
}

impl FrameIndexEntry {
    fn reads_clock(&self) -> bool {
        self.uses_shader_time && masks_use_shader_time(&self.plan.masks)
    }
}

struct Patchable {
    page_effects: crate::frame::PageEffects,
    index: PlanIndex,
    /// `Some` for a page with captures, cut-outs or groups painting their appearance. Such a unit bakes its
    /// content's placement. A changed input re-plans its recorded range; other plain spans still follow
    /// their recipes (`place`) or are prepared node by node (`patch`).
    units: Option<Units>,
    /// The plan's spans by scene row, made with the plan. `None` for a plan that is not placed by rows.
    rows: Option<PlacedRows>,
}

/// A plan's prepared items by scene row: which span a row's node owns, and what the scene says about following
/// a placement by recipe. It is made with the plan (`FramePlanIndex::seed`) and holds only what stands for the
/// plan's life: a node's span and its place in the scene do not change while the planner prepares nodes again
/// in place (a patch keeps every span's items the same shape, for the same node), and a change of structure
/// plans whole and seeds again. Whether a span holds a recipe right now is not kept here; it is read from the
/// span. Everything a placement would ask the scene by name is asked here once, so a placement frame works on row
/// and span indices alone.
struct PlacedRows {
    /// The scene's hierarchy by row these were read under (`Worlds::hierarchy`); a placement under another one
    /// is declined.
    hierarchy: u64,
    /// The canonical row is already the span's source. This bit certifies that the span
    /// contains only that node's ordinary output and can follow it by placement recipe.
    follows: Vec<bool>,
    solid_groups: Vec<SolidGroup>,

    /// Per row: moving the node is planned whole. It owns a unit, sits above one or inside one; or the build
    /// culled it or a node above it, so it has no span and a move may bring it into the window.
    unit: Vec<bool>,
    /// Where each culled node sits in the scene's parent-before-child order, ascending: a moved row whose range
    /// holds one is planned whole (the culled node under it may move into the window).
    culled: Vec<u32>,
}

/// A contiguous run of ordinary solid occurrences whose GPU matrices at the
/// anchor recover the exact local origins of their direct children.
#[derive(Clone, Copy)]
struct SolidGroup {
    /// Position in hierarchy order, also the lookup key for placement writes.
    position: u32,
    /// Identity wrappers can share a run. Its leaves read this parent's frame.
    leaf_parent: u32,
    first: u32,
    end: u32,
    /// Parent translation used by the last ordinary writes, including evicted slots.
    anchor: Option<[f64; 2]>,
    proof: Option<GroupOrigins>,
}

#[derive(Clone, Copy)]
struct GroupOrigins {
    axes: [OriginAxis; 2],
    /// A refreshed anchor can round a local origin on the wrong side of a binade.
    /// Recover it on the leaves' exact binary lattice only when necessary.
    grid: [f32; 2],
}

/// A child's anchored origin is an exact binary32 value. All origins on this axis
/// are multiples of the finest power of two seen during ordinary preparation.
#[derive(Clone, Copy)]
struct OriginAxis { power: i32, min: f64, max: f64 }

impl Default for OriginAxis {
    fn default() -> Self { Self { power: i32::MAX, min: f64::INFINITY, max: f64::NEG_INFINITY } }
}

impl OriginAxis {
    fn include(&mut self, value: f32) {
        self.min = self.min.min(f64::from(value));
        self.max = self.max.max(f64::from(value));
        let bits = value.abs().to_bits();
        if bits == 0 { return; }
        let exponent = (bits >> 23) as i32;
        let significand = bits & 0x7fffff | (u32::from(exponent != 0) << 23);
        self.power = self.power.min(exponent.max(1) - 150 + significand.trailing_zeros() as i32);
    }

    /// Check the rounding boundaries for every possible result exponent, once per
    /// group. Origins share a binary lattice. When its spacing is at least an
    /// output ULP, adding any origin preserves the translation's distance from
    /// the nearest midpoint. Otherwise every midpoint is on the origin lattice.
    fn permits(self, translation: f64) -> bool {
        let p = translation.abs();
        let farthest = self.min.abs().max(self.max.abs());
        // Keep all split words, midpoints, and error bounds away from f32
        // underflow/overflow, including on adapters that flush subnormals.
        if farthest > 2f64.powi(60) || self.power < -60 || p > 2f64.powi(60)
            || (p != 0. && p < 2f64.powi(-60)) { return false; }
        if p == f64::from(p as f32) || farthest == 0. { return true; }
        let grid = 2f64.powi(self.power);
        let distance = |step: f64| { let rem = p % step; rem.min(step - rem) };
        let nearest = distance(grid);
        if nearest == 0. { return false; }
        let exponent = |x: f64| ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        // All sums lie between the translated extrema. A run on one side of
        // zero cannot cancel. Include neighboring binades for endpoint rounding.
        let lo = self.min + translation;
        let hi = self.max + translation;
        let closest = nearest.max(lo.max(-hi));
        (exponent(closest) - 1..=exponent(lo.abs().max(hi.abs())) + 1).all(|e| {
            let bin = 2f64.powi(e);
            let ulp = bin * f64::from(f32::EPSILON);
            let margin = if grid >= ulp { ulp * 0.5 - distance(ulp) } else { nearest };
            // After TwoSum the large addition's residual is exact. Rounding
            // its correction and splitting p each cost at most two binary32
            // rounding errors in succession. This bound also covers rounding
            // the original sum to binary64 before the ordinary binary32 cast.
            margin > (p + bin) * 2f64.powi(-45)
        })
    }
}

impl SolidGroup {
    fn read(row: u32, scene: &Scene, worlds: &moda_scene::worlds::Worlds,
        index: &PlanIndex, rows: &PlacedRows, plan: &CoveragePlan,
        prepared: &std::collections::BTreeMap<u32, Self>) -> Option<Self> {
        if rows.unit[row as usize] || !worlds.whole(row) || scene.row(row)?.node_type() != "group" { return None; }
        if index.by_node.row(row).is_some_and(|si| index.spans.range(si).1 != 0) { return None; }
        let anchor = worlds.child_translation(row)?;
        let (from, to) = worlds.range(row)?;
        let children = worlds.ordered(from + 1, to);
        // The canvas lowerer can wrap a group's contents in one identity group.
        // Its already-certified run is also this parent's. Walk the leaves only
        // once even when several nested wrappers stand above them.
        let first = *children.first()?;
        if let Some(run) = prepared.get(&first) {
            let (local, parent) = worlds.local_placement(first)?;
            if worlds.range(first)?.1 == to && parent == Some(row)
                && [local.a, local.b, local.c, local.d, local.e, local.f] == [1., 0., 0., 1., 0., 0.] {
                return Some(Self { position: from, ..*run });
            }
        }
        if children.len() < 2 { return None; }
        let mut logical = None;
        let mut end = 0;
        let mut axes = [OriginAxis::default(); 2];
        for &child in children {
            let (local, parent) = worlds.local_placement(child)?;
            if parent != Some(row) { return None; }
            let si = index.by_node.row(child)?;
            if !rows.follows[si] { return None; }
            let (range, _) = index.spans.placement(si)?;
            if range.is_empty() || logical.is_some() && range.start != end { return None; }
            logical.get_or_insert(range.start);
            end = range.end;
            for item in plan.items.get(range)? {
                // Both path vertex stages consume the logical slot in pathPlacement.
                let crate::seams::DrawItem::Body { instance, paint: crate::seams::BodyPaint::Solid | crate::seams::BodyPaint::Program(_),
                    mask: None, blend_mode: crate::seams::BlendMode::Normal, .. } = item else { return None };
                let m = &instance.model_matrix;
                for (i, value) in [local.e, local.f].into_iter().enumerate() {
                    let origin = m[12 + i] - anchor[i] as f32;
                    if m[12 + i] != (anchor[i] + value) as f32 || !origin.is_finite() || f64::from(origin) != value { return None; }
                    axes[i].include(origin);
                }
            }
        }
        Some(Self { position: from, leaf_parent: row, first: u32::try_from(logical?).ok()?,
            end: u32::try_from(end).ok()?, anchor: Some(anchor), proof: Some(GroupOrigins { axes, grid: [0.; 2] }) })
    }

    /// Only placement recipes have run since collection. Every slot therefore
    /// contains the ordinary world matrix at `anchor`. A pure parent move
    /// leaves all local affines as they were at that write. Recheck their origins
    /// once on demand, including when the CPU copies of those matrices were trimmed.
    fn refreshed_origins(&self, worlds: &moda_scene::worlds::Worlds) -> Option<GroupOrigins> {
        let anchor = self.anchor?;
        let (from, to) = worlds.range(self.leaf_parent)?;
        let mut axes = [OriginAxis::default(); 2];
        let mut error = [0.0_f64; 2];
        for &child in worlds.ordered(from + 1, to) {
            let (local, parent) = worlds.local_placement(child)?;
            if parent != Some(self.leaf_parent) { return None; }
            for (i, value) in [local.e, local.f].into_iter().enumerate() {
                let origin = value as f32;
                if !origin.is_finite() || f64::from(origin) != value { return None; }
                axes[i].include(origin);
                let recovered = (anchor[i] + value) as f32 - anchor[i] as f32;
                if !recovered.is_finite() { return None; }
                error[i] = error[i].max((f64::from(recovered) - value).abs());
            }
        }
        let mut grid = [0.; 2];
        for i in 0..2 {
            if error[i] == 0. { continue; }
            let power = axes[i].power;
            if !(-60..=60).contains(&power) { return None; }
            grid[i] = 2f32.powi(power);
            // Strictly inside the cell, so tie-breaking cannot change the result.
            if error[i] >= 0.5 * f64::from(grid[i]) { return None; }
        }
        Some(GroupOrigins { axes, grid })
    }
}

impl PlacedRows {
    /// Placement preserves the original slot layout. Only intersecting groups
    /// need a new anchor. Sparse edits visit their ancestors and the groups
    /// rooted in the edited subtree, without walking sibling groups or leaves.
    fn placed(&mut self, worlds: &moda_scene::worlds::Worlds, ranges: &[(u32, u32)]) {
        if self.solid_groups.is_empty() { return; }
        let refresh = |group: &mut SolidGroup| {
            group.anchor = worlds.child_translation(group.leaf_parent);
            group.proof = None;
        };
        let mut covered = 0;
        for &(from, to) in ranges {
            if to <= covered { continue; }
            covered = to;
            let start = self.solid_groups.partition_point(|g| g.position < from);
            for group in self.solid_groups[start..].iter_mut().take_while(|g| g.position < to) {
                refresh(group);
            }
            let mut row = worlds.ordered(from, to)[0];
            while let Some((_, Some(parent))) = worlds.local_placement(row) {
                let Some((position, _)) = worlds.range(parent) else { break };
                if let Ok(i) = self.solid_groups.binary_search_by_key(&position, |g| g.position) {
                    refresh(&mut self.solid_groups[i]);
                }
                row = parent;
            }
        }
    }

    fn read(scene: &Scene, index: &PlanIndex, units: Option<&Units>, warnings: &PlanWarnings, plan: &CoveragePlan, has_composites: bool) -> Option<Self> {
        // The scene's rows, when its owner keeps the world columns current (`Scene::refresh_worlds`).
        let worlds = scene.world_rows()?;
        let rows = worlds.len();
        // What the build culled, by row: the culled node and everything under it must not move by recipe, and
        // neither must anything above it. Rows beside it are free. A culled node that is not a row (an
        // expansion's part) is reached by no row's range: such a plan is not placed by rows.
        let mut unit = vec![false; rows];
        let mut culled = Vec::with_capacity(index.culled_roots.len());
        for id in &index.culled_roots {
            let (from, to) = worlds.range(scene.row_of(id)?)?;
            culled.push(from);
            for &under in worlds.ordered(from, to) { *unit.get_mut(under as usize)? = true; }
        }
        culled.sort_unstable();
        let mut follows_row = vec![false; index.spans.len()];
        let mut restriction_memo = HashMap::default();
        let warned = !warnings.skipped.is_empty() || !warnings.dropped.is_empty();
        for (si, entry) in index.spans.iter().enumerate() {
            let id = entry.source.id(scene);
            // A span of something that is not a row (a table's cell) is reached by no row's subtree.
            if !matches!(entry.source, crate::frame::SpanSource::Row(_)) { continue; }
            let Some(items) = plan.items.get(entry.start..entry.start + entry.len) else { continue };
            let follows = plain_source_eligible(scene, index, id, &mut restriction_memo)
                && !(warned && (warnings_mention_node(&warnings.skipped, id) || warnings_mention_node(&warnings.dropped, id)))
                // The span names this node's own items (a unit's cut-out shifts the items inside it).
                && items.iter().all(|item| drawitem_node_id(item) == Some(id))
                && units.is_none_or(|units| !units.holds(scene, id));
            follows_row[si] = follows;
        }
        if let Some(units) = units {
            for (row, slot) in unit.iter_mut().enumerate() {
                let id = scene.row(row as u32)?.id();
                // A unit that moves, sits under a node that moves, or holds one, is planned again whole. A moved
                // node is asked directly: one that draws nothing yet has no span to ask.
                *slot |= units.above.contains(id) || units.holds(scene, id) || units.reads_source(id);
            }
        }
        let mut rows = Self { hierarchy: worlds.hierarchy(), follows: follows_row, unit, culled,
            solid_groups: Vec::new() };
        if !warned && !has_composites && plan.camera == [0., 0.] && plan.camera_frame.is_none()
            && plan.region_viewport.is_none() && plan.item_order.is_none() {
            let mut prepared = std::collections::BTreeMap::new();
            for row in worlds.parents().rev() {
                if let Some(group) = SolidGroup::read(row, scene, worlds, index, &rows, plan, &prepared) {
                    prepared.insert(row, group);
                }
            }
            rows.solid_groups = prepared.into_values().collect();
            rows.solid_groups.sort_unstable_by_key(|g| g.position);
        }
        Some(rows)
    }
}

/// The nodes of a page that own a unit, from the scene rather than from what the plan emitted: a unit that
/// draws nothing yet (a background effect outside the target) still splits the plan once it moves into view.
#[derive(Default)]
struct Units {
    sources: parts::Sources,
    compact_sources: bool,
    output_order: Option<crate::plan_items::ItemOrder>,
    outputs: Vec<Option<Box<OutputRecords>>>,
    /// A fading group, a blurred, shadowed or mask-captured node, a background effect, a split blend, a group
    /// painting its appearance.
    owners: weights::RetainedSet<crate::frame::SpanSource>,
    /// Every node above an owner: moving one moves the unit.
    above: weights::RetainedSet<String>,
    segment: Option<crate::frame::CameraSegment>,
    /// Each source node names the outer recorded unit that consumes it.
    by_node: SourceLookup,
    /// Recorded units reading each mask, including reads through descendant masks.
    mask_users: Vec<Vec<usize>>,
    /// Rows read by each derived mask: its source subtree and inherited source inputs.
    mask_inputs: weights::RetainedMap<String, Vec<usize>>,
    /// Conservative whole-scene reads are a scope, not one duplicate edge per scene row.
    global_mask_inputs: Vec<usize>,
    recorded_by_node: SourceLookup,
    hidden_sources: weights::RetainedSet<String>,
}

fn mask_reads_whole_scene(mask: &crate::seams::MaskPlan) -> bool {
    mask.image.as_ref().and_then(|image| image.rendered.as_ref()).is_some_and(|rendered| rendered.cache_key.is_none())
}

impl Units {
    fn of(scene: &Scene, plan: &CoveragePlan, segment: Option<crate::frame::CameraSegment>, spans: &[crate::frame::NodeSpan], effects: &crate::frame::PageEffects, mut budget: usize) -> Self {
        use crate::seams::DrawItem;
        let mut owners: weights::RetainedSet<_> = plan.items.iter().filter_map(|item| match item {
            DrawItem::BackdropSplit(split) => Some(split.node_id.as_str()),
            DrawItem::LayerBlurComposite { node_id, .. } => Some(node_id.as_str()),
            _ => None,
        }).chain(scene.group_appearance_by_id().keys().chain(effects.owners()).map(String::as_str))
            .map(|id| crate::frame::SpanSource::of(scene, id)).collect();
        // Every recorded leaf output has one writer. Ordinary emission can return an empty
        // output before producing a plain span (for example, a null clip); its source interval
        // still belongs to the range executor, just as a capture's interval does.
        if let Some(seg) = &segment {
            let plain: std::collections::HashSet<&str> = spans.iter().map(|span| span.source.id(scene)).collect();
            for item in seg.dl.items.iter().filter(|item| item.is_primary_node_item) {
                if crate::frame::source_has_body(item)
                    && !plain.contains(item.node_id.as_str())
                    && scene.effective_node(&item.node_id).is_some_and(|node| !moda_scene::scene::is_parent_node(node))
                    && !scene.transient_by_parent().contains_key(&item.node_id) {
                    owners.insert(crate::frame::SpanSource::of(scene, &item.node_id));
                }
            }
        }
        let mut above = weights::RetainedSet::default();
        for owner in &owners {
            let mut cursor = owner.id(scene);
            // Stop at a chain another owner already walked.
            while let Some(parent) = scene.parent_map().get(cursor).or_else(|| scene.transient_parent_of().get(cursor)) {
                if !above.insert(parent.to_string()) { break; }
                cursor = parent;
            }
        }
        let mut by_node = SourceLookup::default();
        if let Some(seg) = &segment {
            for (ui, unit) in seg.units.iter().enumerate() {
                let items = &seg.dl.items[unit.dl_start..unit.dl_end];
                if items.iter().any(|item| owners.contains(&crate::frame::SpanSource::of(scene, &item.node_id))) {
                    for item in items { by_node.insert(scene, &item.node_id, ui); }
                }
            }
        }
        let mut mask_users = vec![Vec::new(); plan.masks.len()];
        if let Some(seg) = &segment {
            for (ui, unit) in seg.units.iter().enumerate() {
                for &read in &unit.details().mask_reads {
                    let mut mask = Some(seg.mask_start + read);
                    let mut remaining = plan.masks.len();
                    while let Some(mi) = mask {
                        if remaining == 0 { break; }
                        remaining -= 1;
                        let Some(users) = mask_users.get_mut(mi) else { break };
                        if users.last() != Some(&ui) { users.push(ui); }
                        mask = plan.masks[mi].ancestor;
                    }
                }
            }
        }
        let mut mask_inputs: weights::RetainedMap<String, Vec<usize>> = Default::default();
        let mut global_mask_inputs = Vec::new();
        for (mi, mask) in plan.masks.iter().enumerate() {
            if mask_reads_whole_scene(mask) { global_mask_inputs.push(mi); continue; }
            let mut pending = if mask.source_node_id != mask.owner_node_id { vec![mask.source_node_id.as_str()] } else { Vec::new() };
            if mask.invert { pending.push(mask.owner_node_id.as_str()); }
            let mut inputs = std::collections::HashSet::new();
            while let Some(id) = pending.pop() {
                if !inputs.insert(id) { continue; }
                if let Some(children) = scene.children_by_parent().get(id) { pending.extend(children.iter().map(String::as_str)); }
                if let Some(children) = scene.transient_by_parent().get(id) { pending.extend(children.iter().map(String::as_str)); }
            }
            inputs.insert(mask.source_node_id.as_str());
            let mut cursor = mask.source_node_id.as_str();
            while let Some(parent) = scene.parent_map().get(cursor).or_else(|| scene.transient_parent_of().get(cursor)) {
                if !inputs.insert(parent.as_str()) { break; }
                cursor = parent;
            }
            for id in inputs { mask_inputs.entry(id.to_string()).or_default().push(mi); }
        }
        let mut recorded_by_node = SourceLookup::default();
        let mut hidden_sources = weights::RetainedSet::default();
        if let Some(seg) = &segment {
            for (ui, unit) in seg.units.iter().enumerate() {
                for item in &seg.dl.items[unit.dl_start..unit.dl_end] {
                    recorded_by_node.insert(scene, &item.node_id, ui);
                    if item.fallback_reasons.contains(&"text-mask-source") { hidden_sources.insert(item.node_id.clone()); }
                }
            }
        }
        let outputs = segment.as_ref().map(|seg| seg.units.iter().map(|unit| {
            // Plain source spans already have their writer. Only a range owned by a capture or
            // composition can call its retained output graph; promotion installs that ownership.
            let allowance = budget.checked_sub(std::mem::size_of::<OutputRecords>())?;
            if !seg.dl.items[unit.dl_start..unit.dl_end].first()
                .is_some_and(|item| by_node.contains(scene, &item.node_id)) { return None; }
            let start = seg.prefix + unit.item_start;
            let record = OutputRecords::read(scene, &seg.dl.items[unit.dl_start..unit.dl_end], unit,
                &plan.items[start..start + unit.item_len], seg.prepared.as_ref())?;
            budget = allowance.checked_sub(record.bytes())?;
            Some(Box::new(record))
        }).collect()).unwrap_or_default();
        let output_order = segment.as_ref().and_then(|seg| {
            let ranges = std::iter::once(0..seg.prefix).chain(seg.units.iter().map(|unit| {
                let start = seg.prefix + unit.item_start;
                start..start + unit.item_len
            })).collect();
            crate::plan_items::ItemOrder::new(ranges, plan.items.len())
        });
        Self { sources: Default::default(), compact_sources: false, output_order, outputs, owners, above, segment, by_node, mask_users, mask_inputs, global_mask_inputs, recorded_by_node, hidden_sources }
    }

    /// Leave the execution inputs and the reverse lookup of independent owners. Cold units
    /// still replay with this segment's projection and mask context.
    fn retain_cold_sources(&mut self, scene: &Scene, owners: SourceLookup) {
        let mut segment = self.segment.take().expect("cold units have a segment");
        let mut sources = std::mem::take(&mut self.sources);
        if !self.compact_sources { sources.push(scene, &segment.dl.items); }
        segment.dl = Arc::new(moda_scene::DisplayList { page_id: segment.dl.page_id.clone(),
            page_index: segment.dl.page_index, items: Vec::new() });
        segment.prepared = None;
        segment.units = Vec::new();
        *self = Self { sources, segment: Some(segment), compact_sources: true, by_node: owners, ..Default::default() };
    }

    /// A plain span can reconstruct its primary from its own source anchor. Keep the
    /// display-list intervals that independently execute a composition, mask or page-mode read.
    fn release_plain_sources(&mut self, scene: &Scene) {
        if self.compact_sources { return; }
        let Some(seg) = &mut self.segment else { return; };
        let mut keep = vec![false; seg.units.len()];
        for ui in self.by_node.values().chain(self.mask_users.iter().flatten().copied()) { keep[ui] = true; }
        for id in self.mask_inputs.keys().chain(&self.hidden_sources) {
            if let Some(ui) = self.recorded_by_node.get(scene, id) { keep[ui] = true; }
        }
        let old = std::mem::take(&mut seg.units);
        let mut remap = vec![usize::MAX; old.len()];
        let mut outputs = Vec::new();
        for (i, mut unit) in old.into_iter().enumerate() {
            if !keep[i] && !unit.needs_source_range() && self.global_mask_inputs.is_empty() { continue; }
            remap[i] = seg.units.len();
            let source = &seg.dl.items[unit.dl_start..unit.dl_end];
            let range = self.sources.push(scene, source);
            unit.dl_start = range.start;
            unit.dl_end = range.end;
            seg.units.push(unit);
            outputs.push(self.outputs[i].take());
        }
        self.by_node.remap(&remap);
        for ui in self.mask_users.iter_mut().flatten() { *ui = remap[*ui]; }
        self.recorded_by_node.remap(&remap);
        self.outputs = outputs;
        seg.dl = Arc::new(moda_scene::DisplayList { page_id: seg.dl.page_id.clone(), page_index: seg.dl.page_index, items: Vec::new() });
        seg.prepared = None;
        self.output_order = None;
        self.compact_sources = true;
    }

    /// Packed source intervals suffice to regenerate compound records. Release the optional
    /// prepared output summaries independently of the range's membership and destination.
    fn trim_output_records(&mut self, mut excess: usize) -> usize {
        let mut freed = 0;
        for output in self.outputs.iter_mut().rev() {
            if excess == 0 { break; }
            if let Some(old) = output.take() {
                let bytes = std::mem::size_of::<OutputRecords>() + old.bytes();
                freed += bytes;
                excess = excess.saturating_sub(bytes);
            }
        }
        freed
    }

    fn source_bytes(&self, allocations: &mut weights::Allocations) -> usize {
        use weights::{capacity, names, set};
        let segment = self.segment.as_ref().map_or(0, |s| {
            let unit_heap = s.units.iter().map(|unit| {
                let recipes = std::mem::size_of_val(unit.compositions.as_ref());
                let Some(details) = unit.details.as_deref() else { return recipes };
                let captures = details.captures.iter().map(|capture| capacity(&capture.strip_masks)
                    + capacity(&capture.mask_table) + capacity(&capture.isolations)).sum::<usize>();
                recipes + std::mem::size_of_val(details) + capacity(&details.mask_reads) + capacity(&details.captures)
                    + names(&details.skipped) + names(&details.dropped) + captures
            }).sum::<usize>();
            let primary_heap = s.dl.items.iter().map(|p| p.id.capacity() + p.node_id.capacity() + p.node_type.capacity()
                + capacity(&p.clip_chain) + capacity(&p.effects) + capacity(&p.fallback_reasons)
                + p.blend_mode.as_ref().map_or(0, String::capacity)
                + p.effects.iter().map(|e| e.source_node_id.capacity()).sum::<usize>()
                + p.clip_chain.iter().map(|c| c.owner_node_id.capacity() + c.mask_node_id.as_ref().map_or(0, String::capacity)).sum::<usize>()
            ).sum::<usize>();
            s.mask_scope.bytes() + capacity(&s.units) + unit_heap + capacity(&s.dl.items) + primary_heap
                + s.prepared.as_ref().map_or(0, |p| p.inventory(allocations))
        });
        segment + self.sources.bytes() + self.recorded_by_node.bytes() + self.by_node.bytes()
            + capacity(&self.outputs) + self.outputs.iter().flatten().map(|r| r.bytes() + std::mem::size_of::<OutputRecords>()).sum::<usize>()
            + self.output_order.as_ref().map_or(0, crate::plan_items::ItemOrder::bytes)
            + self.owners.allocation_size() + self.owners.iter().map(|id| match id {
                crate::frame::SpanSource::Generated(name) => name.len(), _ => 0,
            }).sum::<usize>() + set(&self.above) + set(&self.hidden_sources)
            + capacity(&self.mask_users) + self.mask_users.iter().map(capacity).sum::<usize>()
            + self.mask_inputs.allocation_size()
            + self.mask_inputs.iter().map(|(id, users)| id.capacity() + capacity(users)).sum::<usize>()
            + capacity(&self.global_mask_inputs)
    }

    /// A content write can make an existing source subtree start emitting a capture. Its recorded
    /// source interval already exists; coalesce just that interval into one retained execution unit.
    /// Output addresses outside the interval and every unrelated dependency edge stay in place.
    fn promote_sources(&mut self, scene: &Scene, changed: &[u32], at: u64,
        plan: &mut Arc<CoveragePlan>, rows: &mut Option<PlacedRows>, written: &mut PlanWritten,
        index: &PlanIndex, effects: &crate::frame::PageEffects,
    ) -> Option<i64> {
        use moda_scene::stamps::Group;
        let worlds = scene.world_rows()?;
        let placed = rows.as_ref().filter(|read| read.hierarchy == worlds.hierarchy());
        let mut candidates = Vec::new();
        let mut restriction_memo = HashMap::default();
        for &row in changed {
            let id = scene.row(row)?.id();
            if self.by_node.row(row).is_some() { continue; }
            let hidden = scene.hidden_from_paint_order(id);
            if scene.stamps().current(row, Group::Content, at) && !hidden { continue; }
            if !hidden && !effects.owns(id) && !scene.group_appearance_by_id().contains_key(id)
                && index.by_node.row(row).is_some_and(|si| placed.is_some_and(|read| read.follows[si])
                    || plain_source_eligible(scene, index, id, &mut restriction_memo)) { continue; }
            let (from, to) = worlds.range(row)?;
            candidates.push((from, to, id));
        }
        candidates.sort_unstable_by_key(|&(from, to, _)| (from, std::cmp::Reverse(to)));
        let mut bytes = 0i64;
        for (from, to, id) in candidates {
            // An enclosing changed range claims its descendants first, so overlapping changes do not
            // coalesce or copy the same outputs again.
            if self.by_node.contains(scene, id) { continue; }
            if self.compact_sources { return None; }
            let source_rows = worlds.ordered(from, to);
            let mut members: Vec<usize> = source_rows.iter().filter_map(|&row|
                self.recorded_by_node.get(scene, scene.row(row)?.id())).collect();
            members.sort_unstable(); members.dedup();
            let first = *members.first()?;
            let last = *members.last()?;
            let seg = self.segment.as_mut()?;
            let source_start = seg.units.get(first)?.dl_start;
            let source_end = seg.units.get(last)?.dl_end;
            if !seg.dl.items.get(source_start..source_end)?.iter().all(|item|
                scene.row_of(&item.node_id).and_then(|row| worlds.range(row)).is_some_and(|(at, _)| (from..to).contains(&at))) { return None; }
            // The source interval is contiguous in painter order even if previous output growth moved
            // some of its storage. Empty retired unit descriptors may sit between its active members.
            let mut merged = seg.units[first].clone();
            merged.dl_end = source_end;
            merged.z_consumed = 0;
            let mut compositions = Vec::new();
            merged.details = None;
            merged.culled = true; merged.part_culled = false; merged.flags = 0; merged.reads_body_mode = false;
            let mut items = Vec::new();
            for ui in first..=last {
                let unit = &seg.units[ui];
                if unit.dl_start == unit.dl_end { continue; }
                if unit.z != merged.z + merged.z_consumed { return None; }
                merged.z_consumed += unit.z_consumed;
                compositions.extend_from_slice(&unit.compositions);
                if let Some(details) = &unit.details {
                    let dest = merged.details_mut();
                    dest.mask_reads.extend_from_slice(&details.mask_reads);
                    dest.captures.extend_from_slice(&details.captures);
                    dest.skipped.extend_from_slice(&details.skipped);
                    dest.dropped.extend_from_slice(&details.dropped);
                }
                merged.culled &= unit.culled; merged.part_culled |= unit.part_culled;
                merged.flags |= unit.flags; merged.reads_body_mode |= unit.reads_body_mode;
                if unit.item_start == parts::ABSENT { return None; }
                let start = seg.prefix + unit.item_start;
                items.extend_from_slice(plan.items.get(start..start + unit.item_len)?);
                if let Some(record) = self.outputs.get_mut(ui) {
                    bytes -= record.take().as_ref().map_or(0, |r| r.bytes()) as i64;
                }
            }
            merged.compositions = compositions.into_boxed_slice();
            if let Some(details) = &mut merged.details {
                details.mask_reads.sort_unstable();
                details.mask_reads.dedup();
            }
            merged.item_len = items.len();
            let (start, delta) = crate::plan_items::replace_unit(Arc::make_mut(plan), &mut self.output_order,
                first + 1, items, written)?;
            bytes += delta;
            merged.item_start = start.checked_sub(seg.prefix)?;
            for ui in first + 1..=last {
                let (_, delta) = crate::plan_items::replace_unit(Arc::make_mut(plan), &mut self.output_order,
                    ui + 1, Vec::new(), written)?;
                bytes += delta;
                let unit = &mut seg.units[ui];
                unit.dl_end = unit.dl_start; unit.item_len = 0; unit.z_consumed = 0;
                unit.compositions = Box::default();
                unit.details = None;
                unit.reads_body_mode = false;
            }
            // Replace the reached consumers' mask edges, including ancestor-mask dependencies.
            let mut masks = std::collections::HashSet::new();
            for &read in &merged.details().mask_reads {
                let mut next = Some(seg.mask_start + read);
                while let Some(mi) = next {
                    if !masks.insert(mi) { break; }
                    next = plan.masks.get(mi)?.ancestor;
                }
            }
            for mi in masks {
                let users = self.mask_users.get_mut(mi)?;
                users.retain(|ui| *ui < first || *ui > last);
                if !users.contains(&first) { users.push(first); }
            }
            seg.units[first] = merged;
            for item in &seg.dl.items[source_start..source_end] {
                self.by_node.insert(scene, &item.node_id, first);
                self.recorded_by_node.insert(scene, &item.node_id, first);
            }
            self.owners.insert(crate::frame::SpanSource::of(scene, id));
            let mut parent = scene.parent_map().get(id).or_else(|| scene.transient_parent_of().get(id));
            while let Some(id) = parent {
                self.above.insert(id.clone());
                if let Some(rows) = rows.as_mut() {
                    if let Some(row) = scene.row_of(id) { *rows.unit.get_mut(row as usize)? = true; }
                }
                parent = scene.parent_map().get(id).or_else(|| scene.transient_parent_of().get(id));
            }
            if let Some(rows) = rows.as_mut() {
                for &row in source_rows {
                    *rows.unit.get_mut(row as usize)? = true;
                    let span = *index.by_node.rows.get(row as usize)?;
                    if span != u32::MAX { *rows.follows.get_mut(span as usize)? = false; }
                }
            }
        }
        Some(bytes)
    }

    fn reads_source(&self, id: &str) -> bool {
        !self.global_mask_inputs.is_empty() || self.mask_inputs.contains_key(id)
    }

    /// Follow source ancestry to its recorded output graph. A source absent from that graph's reads
    /// can keep empty output while the same paint-order selector still prunes it. Other consumers,
    /// including masks outside this unit, are reached independently below.
    fn keeps_empty_source(&self, scene: &Scene, id: &str, row: u32) -> bool {
        if !scene.placeable(row) || !scene.hidden_from_paint_order(id) { return false; }
        let mut cursor = id;
        for _ in 0..4096 {
            if let Some(unit) = self.by_node.get(scene, cursor) {
                return self.outputs.get(unit).and_then(Option::as_ref)
                    .is_some_and(|record| record.ignores_source(row));
            }
            let Some(parent) = scene.parent_map().get(cursor).or_else(|| scene.transient_parent_of().get(cursor)) else {
                return false;
            };
            cursor = parent;
        }
        false
    }

    /// Re-run only recorded units reached by the changed rows. Their output shape and render-z must
    /// stand; structural changes still go through the cold build. The ordinary node executor is shared.
    #[allow(clippy::too_many_arguments)]
    fn patch(&mut self, scene: &Scene, plan: &mut Arc<CoveragePlan>, hashes: &dyn NodeHashSource,
        opts: &RenderOptions, changed: &[u32], index: &mut PlanIndex, warnings: &PlanWarnings, caches: &mut PlanCaches,
        bytes: usize, written: &mut PlanWritten, built_at: u64, rows: &mut Option<PlacedRows>,
        mut parts: Option<&mut parts::PlanParts>,
        parameter: Option<moda_scene::value::K>, scalar_rows: &mut Vec<u32>,
    ) -> Option<(usize, HashMap<String, usize>, Vec<usize>)> {
        let placed = rows.as_ref().filter(|rows|
            scene.world_rows().is_some_and(|worlds| worlds.hierarchy() == rows.hierarchy));
        let source_rows = placed.map(|rows| rows.unit.as_slice());
        let mut stack = Vec::with_capacity(changed.len());
        for &row in changed { stack.push((scene.row(row)?.id(), false)); }
        let mut visited = std::collections::HashSet::new();
        let mut affected = Vec::new();
        let mut dirty_masks = std::collections::HashSet::new();
        let mut excluded = HashMap::default();
        if !changed.is_empty() {
            for &mi in &self.global_mask_inputs {
                dirty_masks.insert(mi);
                let users = &self.mask_users[mi];
                if users.is_empty() { return None; }
                affected.extend(users.iter().copied());
            }
        }
        while let Some((id, inherited)) = stack.pop() {
            let row = scene.row_of(id);
            // The existing row index records every unit's source subtree and ancestors, including
            // non-emitting rows and mask reads. An unrelated write cannot reach a unit through its children.
            if row.and_then(|row| source_rows?.get(row as usize)).is_some_and(|read| !read) { continue; }
            let propagates = inherited || row.is_none_or(|row| {
                use moda_scene::stamps::Group;
                !scene.stamps().current(row, Group::Placement, built_at)
                    || !scene.stamps().current(row, Group::Content, built_at)
            });
            if !visited.insert((id, propagates)) { continue; }
            let unrecorded = self.holds(scene, id) && !self.by_node.contains(scene, id);
            let empty_source = unrecorded && row.is_some_and(|row| self.keeps_empty_source(scene, id, row));
            if unrecorded && !empty_source { return None; }
            if empty_source { excluded.insert(id.to_string(), usize::MAX); }
            if let Some(masks) = self.mask_inputs.get(id) {
                for &mi in masks {
                    if plan.masks[mi].mode == 3 { dirty_masks.insert(mi); }
                    let users = &self.mask_users[mi];
                    if users.is_empty() { return None; }
                    affected.extend(users.iter().copied());
                }
            }
            if self.reads_source(id) {
                if self.hidden_sources.contains(id) { excluded.insert(id.to_string(), usize::MAX); }
                // A mask source may also draw normally (clip to the layer below). Its own recorded
                // output follows the same executor as its consumers, instead of the plain-span gate.
                if index.by_node.get(scene, id).is_some() { affected.push(self.recorded_by_node.get(scene, id)?); }
            }
            if self.by_node.contains(scene, id) || self.reads_source(id) {
                let mut cursor = Some(id);
                let mut remaining = 4096;
                while let Some(input) = cursor {
                    if remaining == 0 || index.deps.blend_related_ids.contains(input)
                        || (index.deps.mask_source_ids.contains(input) && !self.reads_source(input)
                            && !index.deps.clip_owners_still_hidden(scene, input)) { return None; }
                    remaining -= 1;
                    cursor = scene.parent_map().get(input).or_else(|| scene.transient_parent_of().get(input)).map(String::as_str);
                }
            }
            if let Some(ui) = self.by_node.get(scene, id) { affected.push(ui); }
            // Source ancestry already indexes every mask read below this row. Its own output graph
            // reads none of this hidden source, so there are no child outputs to revisit.
            if empty_source { continue; }
            // Paint values do not change descendant content or placement. Their composition record is
            // already named above; only placement/content writes propagate through the hierarchy.
            if !propagates { continue; }
            if let Some(children) = scene.children_by_parent().get(id) {
                stack.extend(children.iter().map(|id| (id.as_str(), true)));
            }
            if let Some(children) = scene.transient_by_parent().get(id) {
                stack.extend(children.iter().map(|id| (id.as_str(), true)));
            }
        }
        affected.sort_unstable(); affected.dedup();
        if !affected.is_empty() {
            let seg = self.segment.as_ref()?;
            for &ui in &affected {
                // Compact owners were indexed only after checking the first source.
                let Some(unit) = seg.units.get(ui) else { continue; };
                let mut ids = self.sources.names(seg, scene, unit).peekable();
                if ids.peek().is_some_and(|id| !self.by_node.contains(scene, *id)) {
                    for id in ids { excluded.insert(id.to_owned(), ui); }
                }
            }
        }
        // A missing reverse lookup cannot prove that a changed row has no drawn descendants
        // once the flat spans have been released. Recollect an unaccounted-for write.
        if index.spans.len() == 0 && parts.is_some() && changed.iter().any(|&row| self.by_node.row(row).is_none()) { return None; }
        let plain = match plain_patch_spans(scene, index, warnings, changed, Some((&self.by_node, &excluded)), placed) {
            Some(spans) => spans,
            None => { caches.incr_fallbacks = caches.incr_fallbacks.wrapping_add(1); return None; }
        };
        if affected.is_empty() { return Some((bytes, excluded, plain)); }
        let seg = self.segment.as_mut()?;
        let mut bytes = bytes as i64;
        let mut mask_owners = None;
        let mut selected: std::collections::HashSet<usize> = affected.iter().copied().collect();
        let independent = self.global_mask_inputs.is_empty() && !changed.iter().any(|&row|
            scene.row(row).is_some_and(|node| self.mask_inputs.contains_key(node.id())));
        let mut cursor = 0;
        while cursor < affected.len() {
            let ui = affected[cursor];
            cursor += 1;
            if independent && parameter == Some(moda_scene::value::K::Opacity) {
                if let Some(row) = patch_group_opacity(scene, plan, seg, &self.sources, parts.as_deref(), ui, changed, written) {
                    scalar_rows.push(row);
                    continue;
                }
            }
            let cold = seg.units.get(ui)?.item_start == parts::ABSENT;
            if independent && !cold {
                let start = seg.prefix + seg.units[ui].item_start;
                if parameter == Some(moda_scene::value::K::Effects) {
                    let source = self.sources.load(&seg.dl, scene, &seg.units[ui], &index.deps.mask_source_ids)?;
                    if patch_capture_parameters(scene, plan, seg, ui, self.outputs[ui].as_deref(), source.as_ref(),
                        changed, opts, hashes, written).is_some() {
                        bytes -= self.outputs[ui].take().as_deref().map_or(0, OutputRecords::bytes) as i64;
                        continue;
                    }
                }
                if let Some(delta) = self.outputs.get_mut(ui).and_then(Option::as_mut)
                    .and_then(|records| records.patch(scene, changed, plan, start, seg, written, opts, caches, hashes)) {
                    bytes += delta;
                    continue;
                }
            }
            let unit = &seg.units[ui];
            // Offscreen range execution does not cull. It consumes the recorded membership, primary
            // kinds and mask-source roles; values, transforms, clips and effects come from the scene's
            // current rows. Those structural inputs stand under the caller's stamp proof, so no display
            // list is rebuilt for the unit's placement, paint or node-local content writes.
            let source = self.sources.load(&seg.dl, scene, unit, &index.deps.mask_source_ids)?;
            let dl = source.as_ref();
            let before_bytes = seg.prepared.as_ref().map_or(0, |rows| rows.bytes());
            let mut prepared = seg.prepared.take();
            if let Some(rows) = &mut prepared { rows.begin_range(); }
            // Resolve a retained source only if the emitter actually requests its owner.
            let owners = (!unit.details().mask_reads.is_empty()).then(|| &*mask_owners.get_or_insert_with(|| index_mask_owners(&plan.masks)));
            let retained = |owner: &str| {
                let mi = *owners?.get(owner)?;
                let mask = &plan.masks[mi];
                (mask.mode == 3 && !dirty_masks.contains(&mi) && !mask_reads_whole_scene(mask)).then_some(mask)
            };
            let mut build = crate::frame::rebuild_page_range(scene, plan, seg, dl, opts,
                Some((caches, hashes)), unit.z, prepared, &[],
                dl.first().is_some_and(|item| self.by_node.contains(scene, &item.node_id)), Some(&retained))?;
            seg.prepared = build.prepared.take();
            bytes += seg.prepared.as_ref().map_or(0, |rows| rows.bytes()) as i64 - before_bytes as i64;
            let mut mask_writes = Vec::new();
            let (mapping, reads) = validate_unit_output(&mut build, unit, parts.is_some(), self.compact_sources, &plan.masks,
                &mut mask_owners, Some(&mut mask_writes))?;
            // The retained unit owns a source range, not the executor's temporary subdivision.
            // A fade may disappear while its source membership and consumed z remain unchanged.
            let dz = build.end_z as i64 - (unit.z + unit.z_consumed) as i64;
            let mut emitted = unit.clone();
            emitted.z_consumed = build.end_z - unit.z;
            emitted.item_len = build.items.len();
            emitted.compositions = build.units.iter().flat_map(|part| part.compositions.iter().cloned()).collect();
            if emitted.details.is_some() || build.units.iter().any(|part| !part.details().captures.is_empty()) {
                emitted.details_mut().captures = build.units.iter().flat_map(|part| part.details().captures.iter().cloned()).collect();
            }
            emitted.culled = build.units.iter().all(|part| part.culled);
            emitted.part_culled = build.units.iter().any(|part| part.part_culled);
            emitted.flags = build.units.iter().fold(0, |flags, part| flags | part.flags);
            emitted.reads_body_mode = build.units.iter().any(|part| part.reads_body_mode);
            if !seg.prepared.as_mut().is_none_or(|rows| rows.remap_emitted_masks(&mapping)) { return None; }
            for capture in emitted.details.iter_mut().flat_map(|d| &mut d.captures) {
                for id in capture.strip_masks.iter_mut().chain(&mut capture.mask_table) {
                    *id = *mapping.get(*id as usize)? as u64;
                }
            }
            // Membership belongs to the retained dependency index. A changed read set must reseed it,
            // including reads hidden inside captures whose top-level slots can remain unchanged.
            if reads != unit.details().mask_reads {
                // Queries may start or stop reading an existing mask as bounds move. Patch the
                // reverse edges from the executor's actual reads, including inherited mask inputs.
                let ancestors = |reads: &[usize]| -> Option<std::collections::HashSet<usize>> {
                    let mut result = std::collections::HashSet::new();
                    for &read in reads {
                        let mut next = Some(read);
                        while let Some(mi) = next {
                            if !result.insert(mi) { break; }
                            next = plan.masks.get(mi)?.ancestor;
                        }
                    }
                    Some(result)
                };
                let old_reads = ancestors(&unit.details().mask_reads)?;
                let new_reads = ancestors(&reads)?;
                for &mi in old_reads.difference(&new_reads) { self.mask_users.get_mut(mi)?.retain(|&user| user != ui); }
                for &mi in new_reads.difference(&old_reads) { self.mask_users.get_mut(mi)?.push(ui); }
                emitted.details_mut().mask_reads = reads;
            }
            // Mask values are derived inputs. Re-prepare their consumers with the ordinary executor,
            // including units not reached by the original row writes. An ancestor mask is an input of
            // every descendant mask's consumers too. Membership and slots still stand structurally.
            if !mask_writes.is_empty() {
                written.masks_changed = true;
                let plan = Arc::make_mut(plan);
                for (mi, mask) in mask_writes {
                    // Inversion reads the owner's bounds as well as the source: a changed input set
                    // must reseed the reverse index, like a changed source binding or mask-read set.
                    if plan.masks[mi].invert != mask.invert
                        || mask_reads_whole_scene(&plan.masks[mi]) != mask_reads_whole_scene(&mask) { return None; }
                    Arc::make_mut(&mut plan.masks)[mi] = mask;
                    for &consumer in &self.mask_users[mi] {
                        if selected.insert(consumer) { affected.push(consumer); }
                    }
                }
            }
            // Scratch masks were just prepared against this revision. Further
            // consumers of the same source can use the updated table entry.
            for mi in &mapping { dirty_masks.remove(mi); }
            if cold {
                if build.items.len() != unit.item_len || dz != 0 { return None; }
                let (at, delta) = parts.as_deref_mut()?.restore_unit(ui, index, Arc::make_mut(plan), build.items, written)?;
                bytes += delta;
                emitted.item_start = at.checked_sub(seg.prefix)?;
                seg.units[ui] = emitted;
                continue;
            }
            let start = seg.prefix + unit.item_start;
            let old = plan.items.get(start..start + unit.item_len)?;
            let same_slots = items_slots_match(&build.items, old);
            // Plain spans still address their original slots. Owned ranges have their own output index.
            if !same_slots && !dl.first().is_some_and(|item| self.by_node.contains(scene, &item.node_id)) { return None; }
            let resize = build.items.len() != old.len();
            if !resize { bytes += items_bytes(&build.items) as i64 - items_bytes(old) as i64; }
            if dl.first().is_some_and(|item| !self.by_node.contains(scene, &item.node_id)) {
                for item in dl { excluded.insert(item.node_id.clone(), ui); }
            }
            if let Some(record) = self.outputs.get_mut(ui) {
                bytes -= record.as_ref().map_or(0, |r| r.bytes()) as i64;
                if !dl.first().is_some_and(|item| self.by_node.contains(scene, &item.node_id)) {
                    *record = None;
                } else if !record.as_mut().is_some_and(|record| same_slots && record.refresh(scene, &emitted, &build.items, seg.prepared.as_ref())) {
                    *record = OutputRecords::read(scene, dl, &emitted, &build.items, seg.prepared.as_ref()).map(Box::new);
                }
                if record.as_ref().is_some_and(|record| (bytes.max(0) as usize).saturating_add(record.bytes()) > CAMERA_PLAN_BUDGET_BYTES) { *record = None; }
                bytes += record.as_ref().map_or(0, |r| r.bytes()) as i64;
            }
            if resize {
                let (new_start, delta) = crate::plan_items::replace_unit(Arc::make_mut(plan), &mut self.output_order,
                    ui + 1, build.items, written)?;
                bytes += delta;
                emitted.item_start = new_start.checked_sub(seg.prefix)?;
                seg.units[ui] = emitted;
                Self::shift_depths(scene, seg, ui, dz, plan, index, written)?;
                continue;
            }
            let dest = Arc::make_mut(plan).items.get_mut(start..start + unit.item_len)?;
            for (i, (slot, item)) in dest.iter_mut().zip(build.items).enumerate() {
                if same_slots && drawitem_node_id(&item).and_then(|id| scene.row_of(id)).is_some_and(|row|
                    seg.prepared.as_ref().is_some_and(|rows| rows.kept(row))) { continue; }
                written.replaced.push(((start + i) as u32, std::mem::replace(slot, item)));
            }
            seg.units[ui] = emitted;
            Self::shift_depths(scene, seg, ui, dz, plan, index, written)?;
        }
        if let Some(rows) = seg.prepared.as_mut() {
            let without = (bytes.max(0) as usize).saturating_sub(rows.bytes());
            rows.fit_budget(CAMERA_PLAN_BUDGET_BYTES.saturating_sub(without));
            bytes = (without + rows.bytes()) as i64;
        }
        Some((bytes.max(0) as usize, excluded, plain))
    }

    /// Render depth is an ordered derived column. Changed source visibility shifts later depths,
    /// without preparing their geometry, paint or dependency graphs again.
    fn shift_depths(scene: &Scene, seg: &mut crate::frame::CameraSegment, after: usize, dz: i64,
        plan: &mut Arc<CoveragePlan>, index: &mut PlanIndex, written: &mut PlanWritten) -> Option<()> {
        if dz == 0 { return Some(()); }
        seg.z_end = seg.z_end.checked_add_signed(dz)?;
        if seg.z_end > crate::depth::MAX_RENDER_Z { return None; }
        let plan = Arc::make_mut(plan);
        for unit in &mut seg.units[after + 1..] {
            unit.z = unit.z.checked_add_signed(dz)?;
            let start = seg.prefix + unit.item_start;
            for (offset, item) in plan.items.get_mut(start..start + unit.item_len)?.iter_mut().enumerate() {
                if !crate::camera_slot::z_shiftable(item, dz) { return None; }
                let old = item.clone();
                crate::camera_slot::shift_item_z(item, dz);
                written.replaced.push(((start + offset) as u32, old));
            }
            for source in &seg.dl.items[unit.dl_start..unit.dl_end] {
                if !source.is_primary_node_item { continue; }
                if let Some(span) = index.by_node.get(scene, &source.node_id) {
                    index.spans[span].z = index.spans[span].z.checked_add_signed(dz)?;
                }
                if let (Some(prepared), Some(row)) = (seg.prepared.as_mut(), scene.row_of(&source.node_id)) {
                    prepared.shift_source_z(row, dz)?;
                }
            }
        }
        Some(())
    }

    /// Whether `id` is a unit owner or sits under one.
    fn holds(&self, scene: &Scene, id: &str) -> bool {
        let mut cursor = id;
        // A chain longer than any real hierarchy counts as inside one.
        for _ in 0..4096 {
            if self.owners.contains(&crate::frame::SpanSource::of(scene, cursor)) { return true; }
            match scene.parent_map().get(cursor).or_else(|| scene.transient_parent_of().get(cursor)) {
                Some(parent) => cursor = parent,
                None => return false,
            }
        }
        true
    }
}

impl Patchable {
    fn refresh_effects(&mut self, scene: &Scene, opts: &RenderOptions, at: u64, changed: &[u32],
        scalar_rows: &[u32]) -> Option<bool> {
        let mut reweigh = false;
        if scene.stamps().last_composition() > at && changed != scalar_rows {
            let setup = self.page_effects.setup(opts);
            let compact = self.units.as_ref().is_none_or(|units| units.compact_sources);
            let mut pending = Vec::new();
            let mut expanded = std::collections::HashSet::new();
            for &row in changed {
                if scalar_rows.binary_search(&row).is_ok() || scene.stamps().composition_current(row, at) { continue; }
                let root = scene.row(row)?.id();
                pending.push(root);
                while let Some(id) = pending.pop() {
                    let node = scene.effective_node(id)?;
                    if !self.page_effects.mask_stands(node)
                        || self.units.as_ref().and_then(|units| units.segment.as_ref())
                            .is_some_and(|seg| !seg.mask_scope.raster_stands(scene, node)) { return None; }
                    let effects = if scene.hidden_from_paint_order(id) { Default::default() } else {
                        crate::frame::SourceEffects::of(scene, node, self.index.transparent && !opts.preserve_page_fill)
                    };
                    let owned = self.page_effects.owns(id);
                    reweigh |= self.page_effects.replace(id, effects);
                    if owned != self.page_effects.owns(id) && (compact || id != root) { return None; }
                    let children = scene.children_by_parent().get(id);
                    let transients = scene.transient_by_parent().get(id);
                    if (children.is_some() || transients.is_some()) && expanded.insert(id) {
                        pending.extend(children.into_iter().chain(transients).flatten().map(String::as_str));
                    }
                }
            }
            if self.page_effects.setup(opts) != setup { return None; }
        }
        Some(reweigh)
    }
}

impl FrameIndexEntry {
    fn weigh(&mut self) {
        let mut allocations = weights::Allocations::default();
        let plan = &self.plan;
        let index = self.patchable.as_ref().map_or(0, |p| {
            let index = &p.index;
            index.by_node.bytes() + index.deps.bytes()
                + weights::set(&index.culled_roots) + p.page_effects.bytes()
                + p.rows.as_ref().map_or(0, |r| weights::capacity(&r.follows) + weights::capacity(&r.unit) + weights::capacity(&r.culled) + weights::capacity(&r.solid_groups))
                + p.units.as_ref().map_or(0, |u| u.source_bytes(&mut allocations)) + self.span_bytes
        });
        self.fixed_bytes = std::mem::size_of::<Self>() + std::mem::size_of::<CoveragePlan>() + plan.page_id.capacity()
            + weights::names(&self.warnings.skipped) + weights::names(&self.warnings.dropped)
            + plan.item_order.as_ref().map_or(0, crate::plan_items::ItemOrder::bytes) + index
            + self.pattern_slots.as_ref().map_or(0, |i| i.bytes(&mut allocations)) + weights::capacity(&self.unbound_mask_sources) + allocations.masks(&plan.masks);
        self.bytes = self.fixed_bytes + allocations.item_vec(&plan.items) + self.parts.as_ref().map_or(0, |p| p.bytes());
        self.reweigh = false;
    }

    fn trim_pages(&mut self, cap: usize) {
        let Some(patch) = &mut self.patchable else { return; };
        let size = patch.index.spans.trim(cap.saturating_sub(self.fixed_bytes.saturating_sub(self.span_bytes)));
        self.bytes = self.bytes.saturating_sub(self.span_bytes) + size;
        self.fixed_bytes = self.fixed_bytes.saturating_sub(self.span_bytes) + size;
        self.span_bytes = size;
        self.pages_trimmed = true;
    }

    fn fit_cold_units(&mut self, scene: &Scene, budget: usize) -> bool {
        if self.reads_clock() || !self.warnings.is_empty() { return false; }
        let Some(patch) = &mut self.patchable else { return false; };
        let Some(units) = &mut patch.units else { return false; };
        let Some(segment) = &units.segment else { return false; };
        let total = self.parts.as_ref().map_or(self.plan.items.len(), |p| p.addresses.total);
        let cap = budget.saturating_sub(self.packet_bytes.unwrap_or(total.saturating_mul(crate::frame::DESTINATION_BYTES)));
        let Some((parts, plan, owners)) = parts::PlanParts::from_units(scene, &self.plan, segment, &units.sources,
            self.parts.as_ref()) else { return false; };
        units.retain_cold_sources(scene, owners);
        self.parts = Some(parts);
        self.plan = Arc::new(plan);
        patch.index.spans = Vec::new().into();
        patch.index.by_node = Default::default();
        patch.rows = None;
        patch.page_effects.release_owners();
        self.span_bytes = 0;
        self.weigh();
        self.bytes <= cap
    }

    fn fit(&mut self, scene: &Scene, budget: usize, touched: &[u32]) -> bool {
        // The GPU executor can publish a mask's input proof after the planner seeded the entry.
        let observation = self.pattern_slots.as_ref().map_or_else(|| (0, self.unbound_mask_sources.iter()
            .map(|m| m.preparation_revision.load(std::sync::atomic::Ordering::Relaxed)).fold(0u64, u64::wrapping_add)), |i| {
            let revision = i.capture_masks.values().flatten().chain(&i.page_masks).fold(0u64, |sum, m|
                sum.wrapping_add(m.preparation_revision.load(std::sync::atomic::Ordering::Relaxed)));
            (i.mask_read_bytes(&mut Default::default()), revision)
        });
        if self.reweigh || observation != self.mask_observation { self.weigh(); }
        self.mask_observation = observation;
        let total = self.parts.as_ref().map_or(self.plan.items.len(), |p| p.addresses.total);
        let reserve = self.packet_bytes.unwrap_or_else(|| total.saturating_mul(crate::frame::DESTINATION_BYTES));
        let cap = budget.saturating_sub(reserve);
        if self.pages_trimmed || self.bytes > cap { self.trim_pages(cap); }
        if self.bytes <= cap { return true; }
        let patch = self.patchable.as_mut();
        if let Some(patch) = patch {
            if let Some(units) = patch.units.as_mut().filter(|units| !units.compact_sources) {
                units.release_plain_sources(scene);
                for s in patch.index.spans.iter_mut() { s.primary = usize::MAX; }
                // Prepared records may have held the first reference charged for
                // geometry still owned by the plan. Rebuild the common inventory.
                self.weigh();
            }
        }
        let Some(patch) = &mut self.patchable else { return false; };
        if let Some(units) = &mut patch.units {
            let released = units.trim_output_records(self.bytes.saturating_sub(cap));
            self.fixed_bytes = self.fixed_bytes.saturating_sub(released);
            self.bytes = self.bytes.saturating_sub(released);
        }
        if self.bytes <= cap { self.trim_pages(cap); }
        let mut recent = touched;
        while self.bytes > cap {
            let before = (self.plan.items.len(), self.bytes);
            let patch = self.patchable.as_mut().expect("patchable plan");
            let remaining = cap.saturating_sub(self.fixed_bytes);
            let result = match &mut self.parts {
                Some(parts) => parts.fit(scene, &self.warnings, patch, &self.plan, self.pattern_slots.as_ref(), remaining, recent, false),
                None => parts::PlanParts::compact(scene, &self.warnings, &self.plan, patch, self.pattern_slots.as_ref(), remaining).map(|(parts, plan, size)| {
                    self.parts = Some(parts);
                    (plan, size)
                }),
            };
            let Some((plan, size)) = result else { return false; };
            self.plan = Arc::new(plan);
            self.bytes = self.fixed_bytes + size;
            self.trim_pages(cap);
            // Changing physical starts may grow the exact metadata exceptions on a
            // page. Account for those before deciding how many item copies still fit.
            if self.bytes > cap && (self.plan.items.len(), self.bytes) >= before { return false; }
            recent = &[];
        }
        true
    }
}

fn unbound_source_observers(plan: &CoveragePlan) -> Vec<crate::seams::MaskAssetReads> {
    plan.masks.iter().filter_map(|m| m.image.as_ref()?.rendered.as_ref())
        .map(|r| r.asset_reads.get_or_init(Default::default).clone()).collect()
}

impl FramePlanIndex {
    fn budget(&self) -> usize {
        #[cfg(test)]
        if let Some(bytes) = self.test_budget { return bytes; }
        CAMERA_PLAN_BUDGET_BYTES
    }
    /// Take `plan`, made whole of the scene state `key` names, as the scene's plan. Returns the plan it takes the
    /// place of, when one was held: the host hands it to the engine, which compares the two.
    #[allow(clippy::too_many_arguments)]
    pub fn seed(&mut self, scene: &Scene, opts: &RenderOptions,
        plan: &Arc<CoveragePlan>, warnings: &Arc<PlanWarnings>, mut aux: crate::frame::PlanBuildAux,
        generation: u64, key: u64, caches: &mut PlanCaches) -> Option<Arc<CoveragePlan>> {
        let replaced = self.entry.take().and_then(|entry| entry.parts.is_none().then_some(entry.plan));
        self.written = PlanWritten::default();
        let coverage_bytes = plan_bytes(plan);
        let mut bytes = coverage_bytes;
        if crate::prof::on() { eprintln!("coverage plan: {} items, {bytes} estimated bytes", plan.items.len()); }
        let budget = self.budget();
        aux.spans.shrink_to_fit();
        let span_bytes = aux.spans.capacity() * std::mem::size_of::<crate::frame::NodeSpan>();
        let row_bytes = scene.doc().nodes.len() * (std::mem::size_of::<u32>() + 1) + aux.spans.len();
        let index_bytes = span_bytes + row_bytes;
        let recipe_bytes = aux.spans.iter().filter(|s| s.placement.is_some()).count()
            * std::mem::size_of::<crate::frame::TranslationPlacement>();
        let segment_bytes = aux.segment.as_ref().map_or(0, |s| {
            s.units.len() * std::mem::size_of::<crate::frame::PlanUnit>()
                + s.dl.items.capacity() * std::mem::size_of::<moda_scene::types::PaintItem>()
        });
        // The source row, its absolute z and its output range can re-create an evicted
        // display-list primary. Retain the unit execution graph where it supplies dependencies.
        let need_units = aux.has_backdrop || aux.has_layer_blur || !plan.masks.is_empty()
            || !scene.group_appearance_by_id().is_empty();
        let destination_bytes = plan.items.len().saturating_mul(crate::frame::DESTINATION_BYTES);
        // Do not build reverse maps for full headers that already exceed the allowance.
        // Admission uses the same cold-unit fit as a plan that grew after collection.
        let skip_span_index = need_units && segment_bytes > budget && aux.segment.as_ref().is_some_and(|seg| {
            seg.units.iter().filter(|u| u.needs_source_range()).map(|u|
                std::mem::size_of::<crate::frame::PlanUnit>() + std::mem::size_of_val(&*u.compositions)).sum::<usize>() > budget
        });
        if skip_span_index {
            aux.spans = Vec::new();
            aux.page_effects.release_owners();
        }
        if !need_units && bytes + index_bytes + segment_bytes + recipe_bytes > budget {
            aux.segment = None;
        } else if let Some(segment) = aux.segment.as_mut().filter(|_| !skip_span_index) {
            segment.units.shrink_to_fit();
        }
        let mut overhead = index_bytes + if aux.segment.is_some() { segment_bytes } else { 0 };
        bytes += overhead + recipe_bytes;
        if let Some(rows) = aux.segment.as_mut().and_then(|s| s.prepared.as_mut()) {
            rows.fit_budget(budget.saturating_sub(bytes));
            bytes += rows.bytes();
        }
        let uses_shader_time = masks_use_shader_time(&plan.masks);
        let patchable = (!caches.incremental_disabled).then(|| {
            // Source ownership exists independently of today's emitted effects. A currently empty
            // source can reveal, and an ordinary group can begin composing, through the same writer.
            let mut units = aux.segment.take().map(|segment| {
                if skip_span_index { Units { segment: Some(segment), ..Default::default() } }
                else { Units::of(scene, plan, Some(segment), &aux.spans, &aux.page_effects, budget.saturating_sub(bytes)) }
            });
            let input_bytes = coverage_bytes + index_bytes + recipe_bytes;
            if let Some(units) = units.as_mut().filter(|_| !skip_span_index) {
                let mut size = units.source_bytes(&mut weights::Allocations::owned());
                if input_bytes + size + destination_bytes > budget {
                    units.release_plain_sources(scene);
                    size = units.source_bytes(&mut weights::Allocations::owned());
                    for span in &mut aux.spans { span.primary = usize::MAX; }
                }
                overhead = index_bytes + size;
                bytes = input_bytes + size;
            }
            let page_effects = std::mem::take(&mut aux.page_effects);
            let index = build_plan_index(scene, plan.page_index, opts, aux, generation, &mut caches.reverse_deps);
            let rows = (!index.spans.pages.is_empty()).then(|| PlacedRows::read(scene, &index, units.as_ref(), warnings, plan, need_units)).flatten();
            Patchable { page_effects, units, rows, index }
        });
        let inputs = moda_assets::engine_seam::installed().filter(|store| !store.is_empty())
            .and_then(|store| PublicationIndex::build(plan, &store));
        let unbound_mask_sources = if inputs.is_none() { unbound_source_observers(plan) } else { Vec::new() };
        let mut entry = FrameIndexEntry {
            unbound_mask_sources, pattern_slots: inputs, mask_observation: (0, 0),
            layout_key: layout_frame_key(plan.page_index, opts),
            publication: moda_assets::engine_seam::store_generation(), generation,
            fixed_bytes: overhead + recipe_bytes, reweigh: true,
            span_bytes: span_bytes + recipe_bytes, packet_bytes: None, pages_trimmed: false,
            initial: Arc::downgrade(plan), parts: None,
            plan: plan.clone(), warnings: warnings.clone(),
            key, shader_time_s: opts.shader_time_s, built_at: scene.stamps().clock(), uses_shader_time,
            bytes, patchable,
        };
        if (!skip_span_index && entry.fit(scene, budget, &[])) || entry.fit_cold_units(scene, budget) { self.entry = Some(entry); }
        replaced
    }

    /// Apply a complete batch of scene rows and same-layout publications to one revision.
    /// Empty rows are valid only when the scene still stands at the preceding revision.
    pub fn update(&mut self, scene: &Scene, hashes: &dyn NodeHashSource, opts: &RenderOptions,
        generation: u64, key: u64, rows: &[u32], placement_only: bool, caches: &mut PlanCaches,
    ) -> Option<(Arc<CoveragePlan>, Arc<PlanWarnings>)> {
        let entry = self.entry.as_mut()?;
        if entry.key == key || entry.reads_clock() || entry.layout_key != layout_frame_key(entry.plan.page_index, opts)
            || (rows.is_empty() && generation != entry.generation) { return None; }
        let publication = moda_assets::engine_seam::store_generation();
        let pixels = publication != entry.publication;
        // Select logical consumers against the previous revision, before a writer can
        // replace the last occurrence of an asset or move an output into another slot.
        let changed = if pixels {
            Some(entry.pattern_slots.as_mut()?.changed(moda_assets::engine_seam::installed()?.as_ref()))
        } else { None };
        self.written = Default::default();
        let mut placed_only = false;
        if !rows.is_empty() {
            if placement_only {
                placed_only = self.place(scene, opts, generation, key, rows, caches).is_some();
            }
            if !placed_only {
                let result = self.patch(scene, hashes, opts, generation, key, rows, caches);
                let Some(result) = result else { self.entry = None; return None; };
                if self.entry.is_none() { return (!pixels).then_some(result); }
            }
        }
        let mut entry = self.entry.take()?;
        if self.written.masks_changed && entry.pattern_slots.is_none() {
            entry.unbound_mask_sources = unbound_source_observers(&entry.plan);
        }
        if !rows.is_empty() {
            if let Some(inputs) = &mut entry.pattern_slots {
                let store = moda_assets::engine_seam::installed()?;
                let changed = changed.as_deref().unwrap_or_default();
                // Matrix writes preserve asset bindings. Only a patch can replace a binding.
                let placed = if placed_only { &[][..] } else { &self.written.placed[..] };
                let emitted = self.written.replaced.iter().map(|(at, _)| *at as usize)
                    .chain(placed.iter().filter(|w| w.is_full()).map(|w| w.at as usize));
                for at in emitted {
                    let logical = entry.parts.as_ref().map_or(Some(at), |p| p.addresses.logical(at))?;
                    let item = &mut Arc::make_mut(&mut entry.plan).items[at];
                    inputs.update(logical, item, &store)?;
                    // A cached row recipe can contain an older texture. The publication
                    // pass below already visits every consumer selected on the previous revision.
                    if changed.binary_search(&logical).is_err() {
                        if let Some(current) = replace_publication(item, &store, inputs, false) { *item = current; }
                    }
                }
                if self.written.masks_changed {
                    inputs.page_masks = PublicationIndex::masks(&entry.plan.masks)?;
                }
            }
        }
        if let Some(changed) = changed {
            let store = moda_assets::engine_seam::installed()?;
            Self::refresh_consumers(&mut entry, &mut self.written, changed, &store, scene, hashes, opts, caches)?;
        }
        entry.generation = generation;
        entry.publication = publication;
        entry.key = key;
        let served = (entry.plan.clone(), entry.warnings.clone());
        self.entry = Some(entry);
        Some(served)
    }

    fn refresh_consumers(entry: &mut FrameIndexEntry, written: &mut PlanWritten, changed: Vec<usize>,
        store: &moda_assets::AssetStore, scene: &Scene, hashes: &dyn NodeHashSource, opts: &RenderOptions,
        caches: &mut PlanCaches) -> Option<()> {
        let inputs = entry.pattern_slots.as_mut()?;
        written.masks_changed |= PublicationIndex::stale(&inputs.page_masks, store);
        let previous_len = entry.plan.items.len();
        if let Some(parts) = &mut entry.parts {
            let missing: Vec<_> = changed.iter().copied()
                .filter(|&slot| parts.addresses.physical(slot).is_none()).collect();
            if !missing.is_empty() {
                let delta = parts.restore_inputs(&missing, entry.patchable.as_mut()?, Arc::make_mut(&mut entry.plan), scene, opts, caches, hashes, written)?;
                entry.bytes = entry.bytes.saturating_add_signed(delta as isize);
                // A publication may restore a whole span, including sibling slots.
                // Its full writes already contain this revision's world matrices.
                written.evicted.retain(|w| parts.addresses.physical(w.logical).is_none());
            }
        }
        let addresses = entry.parts.as_ref().map(|p| p.addresses.as_ref());
        let restored_from = if written.recollect { previous_len } else { usize::MAX };
        let mut changes = Vec::new();
        for logical in changed {
            let at = addresses.map_or(Some(logical), |map| map.physical(logical))?;
            let source = entry.plan.items.get(at)?;
            if at >= previous_len { inputs.update(logical, source, store)?; }
            if let Some(item) = replace_publication(source, store, inputs, at >= restored_from) {
                changes.push((at, item));
            }
        }
        if !changes.is_empty() {
            let plan = Arc::make_mut(&mut entry.plan);
            for (at, item) in changes {
                written.replaced.push((at as u32, std::mem::replace(&mut plan.items[at], item)));
            }
            // A consumer can also have been emitted or placed this frame. Keep
            // the first old value and upload the complete final record once.
            written.replaced.sort_by_key(|(at, _)| *at);
            written.replaced.dedup_by_key(|(at, _)| *at);
            written.placed.retain(|w| written.replaced.binary_search_by_key(&w.at, |v| v.0).is_err());

        }
        entry.reweigh = true;
        entry.initial = Default::default();
        Some(())
    }

    /// Replay the anchored packet while the only edit since its plan is a translation
    /// of this group. Its ordinary dirty subscriptions remain pending. A later
    /// change to a child, the clip, or the layout sees all intervening writes.
    pub fn translated_group(&mut self, scene: &Scene, opts: &RenderOptions, row: u32)
        -> Option<(Arc<CoveragePlan>, crate::frame::GroupTranslation)> {
        let entry = self.entry.as_mut()?;
        if entry.reads_clock() || entry.publication != moda_assets::engine_seam::store_generation()
            || scene.stamps().last_composition() > entry.built_at { return None; }
        let patch = entry.patchable.as_mut()?;
        let rows = patch.rows.as_mut()?;
        let worlds = scene.world_rows()?;
        if patch.index.opts_key != render_options_key(opts) || rows.hierarchy != worlds.hierarchy() { return None; }
        let position = worlds.range(row)?.0;
        let i = rows.solid_groups.binary_search_by_key(&position, |g| g.position).ok()?;
        let group = &mut rows.solid_groups[i];
        let anchor = group.anchor?;
        let translation = worlds.child_translation(group.leaf_parent)?;
        if group.proof.is_none() { group.proof = group.refreshed_origins(worlds); }
        let proof = group.proof?;
        if !proof.axes.iter().zip(translation).all(|(axis, t)| axis.permits(t)) { return None; }
        Some((entry.plan.clone(), crate::frame::GroupTranslation::new(group.first, group.end,
            anchor.map(|x| x as f32), translation, proof.grid)))
    }

    /// The plan held for the scene state `key` names, when nothing was written since it was made or last
    /// brought there. An offscreen frame has no live clock to thread into a plan, so a plan that reads the
    /// shader clock is the plan of its exact clock only.
    pub fn held(&self, key: u64, shader_time_s: f64) -> Option<(Arc<CoveragePlan>, Arc<PlanWarnings>)> {
        let entry = self.entry.as_ref()?;
        (entry.key == key && (!entry.reads_clock() || entry.shader_time_s == shader_time_s))
            .then(|| (entry.plan.clone(), entry.warnings.clone()))
    }

    /// Release materializations after the host consumes the revision. Use the same accounting
    /// as initial collection when a content edit grows a previously fully resident plan.
    pub fn finish_draw(&mut self, scene: &Scene, touched: &mut [u32], storage: Option<crate::frame::PacketStorage>) -> bool {
        let budget = self.budget();
        let Some(entry) = &mut self.entry else { return false };
        touched.sort_unstable();
        entry.packet_bytes = storage.map(|s| s.destinations.saturating_add(s.recording));
        if entry.fit(scene, budget, touched) { return true; }
        if let Some(s) = storage.filter(|s| s.recording != 0) {
            // Give the ordinary destinations and CPU plan priority. A declined
            // compaction may have interned draw keys while measuring evictable spans.
            entry.packet_bytes = Some(s.destinations);
            entry.reweigh = true;
            if entry.fit(scene, budget, touched) { return false; }
        }
        if entry.fit_cold_units(scene, budget) {
            if let Some(s) = storage {
                let both = s.destinations.saturating_add(s.recording);
                if both > budget.saturating_sub(entry.bytes) { return false; }
                entry.packet_bytes = Some(both);
            }
            return true;
        }
        self.entry = None;
        false
    }

    /// Assemble painter-order output for consumers that read the complete plan. Reuse every
    /// resident span and only run node emission for the outputs that lost their CPU copy.
    pub fn materialize(&self, plan: &Arc<CoveragePlan>, scene: &Scene, hashes: &dyn NodeHashSource,
        opts: &RenderOptions, caches: &mut PlanCaches) -> Option<Arc<CoveragePlan>> {
        let Some(entry) = self.entry.as_ref().filter(|entry| Arc::ptr_eq(&entry.plan, plan))
            else { return None; };
        let Some(parts) = &entry.parts else { return Some(plan.clone()); };
        if parts.addresses.complete() { return Some(plan.clone()); }
        let patch = entry.patchable.as_ref()?;
        if scene.stamps().clock() != entry.built_at
            || entry.layout_key != layout_frame_key(plan.page_index, opts)
            || entry.publication != moda_assets::engine_seam::store_generation() { return None; }
        let index = &patch.index;
        let seg = patch.units.as_ref().and_then(|u| u.segment.as_ref());
        let (backdrop, layer) = seg.map_or((None, None), |s| crate::frame::page_range_contexts(scene, plan, s, opts));
        let mut items = vec![crate::seams::DrawItem::Vacant; parts.addresses.total];
        for (&at, item) in parts.addresses.slots.iter().zip(&plan.items) {
            *items.get_mut(at as usize)? = item.clone();
        }
        parts.materialize_units(patch, plan, scene, opts, caches, hashes, &mut items)?;
        let mut owners = None;
        for (si, span) in index.spans.iter().enumerate().filter(|(_, s)| s.start == parts::ABSENT) {
            let (out, consumed, _) = emit_span(scene, span, opts, index.transparent, index.page_clip,
                backdrop.as_ref(), layer.as_ref(), caches, hashes, &plan.masks, &index.deps.mask_source_ids,
                true, seg, &mut owners)?;
            if out.len() != span.len || consumed != span.z_consumed { return None; }
            let at = parts.start(si);
            for (slot, item) in items.get_mut(at..at + span.len)?.iter_mut().zip(out) { *slot = item; }
        }
        Some(Arc::new(plan.with_items(items)))
    }

    pub fn addresses(&self, plan: &Arc<CoveragePlan>) -> Option<Arc<ItemAddresses>> {
        let entry = self.entry.as_ref()?;
        Arc::ptr_eq(&entry.plan, plan).then(|| entry.parts.as_ref().filter(|p| !p.addresses.complete()).map(|p| p.addresses.clone())).flatten()
    }

    /// Whether drawing a complete CPU snapshot would have to restore missing spans.
    pub fn needs_materialization(&self) -> bool {
        self.entry.as_ref().and_then(|entry| entry.parts.as_ref()).is_some_and(|parts| !parts.addresses.complete())
    }

    /// Estimated owned host bytes, resident physical outputs and total output slots.
    pub fn retained_size(&self) -> (usize, usize, usize) {
        self.entry.as_ref().map_or((0, 0, 0), |e| {
            let resident = e.plan.items.len();
            (e.bytes, resident, e.parts.as_ref().map_or(resident, |p| p.addresses.total))
        })
    }

    pub fn record_budget(&self) -> crate::frame::RecordBudget {
        let budget = self.budget();
        self.entry.as_ref().map_or(crate::frame::RecordBudget::fixed(0), |entry| crate::frame::RecordBudget {
            available: budget.saturating_sub(entry.bytes),
            ceiling: budget.saturating_sub(entry.fixed_bytes),
        })
    }

    pub fn holds(&self, plan: &Arc<CoveragePlan>) -> bool {
        self.entry.as_ref().is_some_and(|entry| Arc::ptr_eq(&entry.plan, plan) || entry.initial.ptr_eq(&Arc::downgrade(plan)))
    }

    /// Follow placement writes without preparing anything: `rows` is the complete batch of scene rows
    /// (`Scene::row_of`; the scene's placement dirty list) whose own transform changed since the last plan, or
    /// (for a node that draws a body itself) whose opacity did. Every item-producing node under them must hold a
    /// placement recipe that admits its new placement (`frame::placed_span_row`); then only those items' matrices
    /// and solid alphas are written, in the plan held here, which then stands at `key`. Otherwise nothing
    /// changes, the plan and index stay, and `None` is returned for the caller to try [`Self::patch`].
    ///
    /// Nothing is looked up by name, and the work follows the rows that move: each moved row is one range of the
    /// scene's parent-before-child order, the ranges of a batch are visited as their union (a moved node under a
    /// moved node is inside its range, not walked again), and each matrix is read from the scene's world column.
    /// The scene's owner keeps that column current (`Scene::refresh_worlds`); a batch the rows do not answer (the
    /// column is not current, something under a moved row is not a row, a derived record stands in for a node)
    /// is declined like any other.
    ///
    /// The caller proves the batch holds nothing else: no content, paint, effect, visibility or group-opacity
    /// change, and no node whose placement feeds another node's fields. Evicted outputs use their original packet slots.
    #[allow(clippy::too_many_arguments)]
    fn place(&mut self, scene: &Scene, opts: &RenderOptions, generation: u64, key: u64, rows: &[u32],
        caches: &mut PlanCaches,
    ) -> Option<()> {
        self.written = PlanWritten::default();
        let entry = self.entry.as_mut()?;
        if entry.reads_clock() { return None; }
        entry.uses_shader_time = false;
        let patchable = entry.patchable.as_mut()?;
        if patchable.index.opts_key != render_options_key(opts) { return None; }
        if !patchable.page_effects.mask_boxes_stand(scene) { return None; }
        let check_composition = scene.stamps().last_composition() > entry.built_at;
        let read = patchable.rows.as_ref()?;
        let worlds = scene.world_rows()?;
        // The rows were read under the structure the plan was made of; a change of structure plans whole.
        if read.hierarchy != worlds.hierarchy() { return None; }
        let index = &mut patchable.index;
        // Each moved row's range, after asking the moved row itself what only it can answer.
        let mut ranges = Vec::with_capacity(rows.len());
        for &row in rows {
            if *read.unit.get(row as usize)? { return None; }
            let span = *index.by_node.rows.get(row as usize)?;
            if check_composition && !scene.stamps().composition_current(row, entry.built_at) {
                // The placement writer can change a leaf's opacity. A parent
                // also changes inherited inputs, handled by the ordinary patch.
                if worlds.is_parent(row) { return None; }
                // The recipe was certified on the exact emitted and canonical path. Moving or
                // fading its one fill cannot introduce a split, mask or filter. The placement
                // pass below still checks visibility, opacity phase, band and ancestor units.
                let plain = span != u32::MAX && index.spans.placement(span as usize)
                    .is_some_and(|(_, recipe)| recipe.plain_fill);
                if !plain {
                    let node = scene.row(row)?;
                    if patchable.units.as_ref().and_then(|units| units.segment.as_ref())
                        .is_some_and(|seg| !seg.mask_scope.raster_stands(scene, node)) { return None; }
                    if patchable.page_effects.owns(node.id()) || (!scene.hidden_from_paint_order(node.id())
                        && crate::frame::SourceEffects::of(scene, node, index.transparent && !opts.preserve_page_fill)
                            != Default::default()) { return None; }
                }
            }
            // A node that draws itself and has no span may have just started to draw. A node that holds
            // children need not draw: a move of it with nothing drawn under it changes nothing.
            if !worlds.is_parent(row) && span == u32::MAX { return None; }
            // Something under it that is not a row is not reached by rows.
            if !worlds.whole(row) { return None; }
            let (from, to) = worlds.range(row)?;
            // A culled node under it may move into the window.
            let next = read.culled.partition_point(|&at| at < from);
            if read.culled.get(next).is_some_and(|&at| at < to) { return None; }
            ranges.push((from, to));
        }
        // The union of the ranges: a range starts inside an earlier one only when it lies wholly inside it.
        ranges.sort_unstable();
        let mut affected: Vec<u32> = Vec::with_capacity(rows.len());
        let mut covered = 0u32;
        for &(from, to) in ranges.iter() {
            if to <= covered { continue; }
            for &under in worlds.ordered(from.max(covered), to) {
                let si = *index.by_node.rows.get(under as usize)?;
                if si != u32::MAX { affected.push(si); }
            }
            covered = to;
        }
        // Decide every span before the first write.
        use rayon::prelude::*;
        let prepare = |&si: &u32| {
            if !*read.follows.get(si as usize)? { return None; }
            let (range, recipe) = index.spans.placement(si as usize)?;
            if range.start == parts::ABSENT {
                let parts = entry.parts.as_ref()?;
                let logical = parts.span_range(si as usize);
                return Some((logical, crate::frame::placed_span_row(scene, &recipe, None, check_composition)?, true));
            }
            let items = entry.plan.items.get(range.clone())?;
            Some((range, crate::frame::placed_span_row(scene, &recipe, Some(items), check_composition)?, false))
        };
        let prepared: Vec<_> = if affected.len() <= 2048 {
            affected.iter().map(prepare).collect()
        } else {
            let grain = affected.len().div_ceil(rayon::current_num_threads()).max(2048);
            affected.par_iter().with_min_len(grain).map(prepare).collect()
        };
        let mut cold = 0;
        for placement in &prepared {
            let (range, _, evicted) = placement.as_ref()?;
            if *evicted { cold += range.len(); }
        }
        let capacity = affected.len().min(entry.plan.items.len());
        let mut resident = Vec::with_capacity(capacity);
        let mut placed = Vec::with_capacity(capacity);
        let mut written = Vec::with_capacity(capacity);
        let mut matrices = Vec::with_capacity(cold);
        let plan_mut = Arc::make_mut(&mut entry.plan);
        for (&si, placement) in affected.iter().zip(prepared) {
            let (range, value, evicted) = placement.expect("validated placement");
            if evicted {
                matrices.extend(range.map(|logical| crate::frame::LogicalWrite { logical, value: crate::frame::LogicalValue::Matrix(value.model) }));
            } else {
                for at in range.clone() { written.push(value.write(at, &plan_mut.items[at])); }
                resident.push(si);
                placed.push((range, value));
            }
        }
        let ordered = resident.windows(2).all(|pair| pair[0] < pair[1])
            && placed.windows(2).all(|pair| pair[0].0.end <= pair[1].0.start);
        let grain = if ordered && resident.len() > 2048 {
            resident.len().div_ceil(rayon::current_num_threads()).max(2048)
        } else { usize::MAX };
        write_placed_spans(&mut index.spans.pages, 0, &mut plan_mut.items, 0, &resident, &placed, grain);
        patchable.rows.as_mut().unwrap().placed(worlds, &ranges);
        self.written.evicted = matrices;
        if !written.is_sorted_by_key(|w| w.at) { written.sort_unstable_by_key(|w| w.at); }
        self.written.placed = written;
        index.built_at_generation = generation;
        entry.generation = generation;
        entry.initial = Default::default();
        entry.key = key;
        entry.built_at = scene.stamps().clock();
        caches.incr_patch_hits = caches.incr_patch_hits.wrapping_add(1);
        caches.incr_patch_nodes = caches.incr_patch_nodes.wrapping_add(affected.len() as u64);
        caches.incr_placement_nodes = caches.incr_placement_nodes.wrapping_add(affected.len() as u64);
        Some(())
    }

    /// What the last successful [`Self::place`] or [`Self::patch`] wrote into the plan, for the engine
    /// (`frame::ChangedItems::Written`). Taken once; anything else that plans clears it.
    pub fn take_written(&mut self) -> PlanWritten {
        std::mem::take(&mut self.written)
    }

    /// The single field written in the host's most recent batch.
    /// A parameter patch requires its endpoints to match the retained and current scene clocks.
    pub fn record_parameter_writes(&mut self, interval: Option<(u64, u64, moda_scene::value::K)>) {
        self.parameter_writes = interval;
    }

    /// `changed` is the complete batch since the last plan, with no bounded
    /// edit-log truncation. The caller proves topology and derived records stand (the scene has no
    /// Structure stamp or whole-scene invalidation). Each span checks its source row's Content and Paint
    /// stamps: current content can follow placement or external text repaint; changed node-local content
    /// uses ordinary node preparation. Cross-node sources and compositing dependencies still decline.
    /// The plan held here is
    /// written in place and then stands at `key`.
    /// A patch that fails part way leaves nothing held: the caller plans whole.
    #[allow(clippy::too_many_arguments)]
    fn patch(&mut self, scene: &Scene, hashes: &dyn NodeHashSource,
        opts: &RenderOptions, generation: u64, key: u64, changed: &[u32], caches: &mut PlanCaches,
    ) -> Option<(Arc<CoveragePlan>, Arc<PlanWarnings>)> {
        let mut sorted;
        let changed = if changed.is_sorted() { changed } else {
            sorted = changed.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            &sorted
        };
        self.written = PlanWritten::default();
        // Resource and option identity stand before following row dependencies. Unit inputs are prepared
        // through their recorded range below; plain spans keep the node-local writer.
        let entry = self.entry.as_mut()?;
        if entry.reads_clock() { return None; }
        entry.uses_shader_time = false;
        let patchable = self.entry.as_ref()?.patchable.as_ref()?;
        if patchable.index.opts_key != render_options_key(opts) { return None; }
        let mut entry = self.entry.take()?;
        let patchable = entry.patchable.as_mut()?;
        let at = entry.built_at;
        let parameter = self.parameter_writes.and_then(|(from, to, field)|
            ((from, to) == (at, scene.stamps().clock())).then_some(field));
        let mut written = PlanWritten::default();
        if !patchable.page_effects.mask_boxes_stand(scene) { return None; }
        // Content changes can promote new unit owners and need the new page setup
        // before execution. A parameter write can first prove which rows change
        // only an existing compositor scalar and leave their source setup intact.
        let defer_effects = parameter == Some(moda_scene::value::K::Opacity)
            && scene.stamps().latest(moda_scene::stamps::Group::Content) <= at;
        if !defer_effects { entry.reweigh |= patchable.refresh_effects(scene, opts, at, changed, &[])?; }
        if scene.stamps().latest(moda_scene::stamps::Group::Content) > at {
            for &row in changed {
                if scene.stamps().current(row, moda_scene::stamps::Group::Content, at) { continue; }
                let node = scene.row(row)?;
                let id = node.id();
                if patchable.units.as_ref().is_some_and(|units| units.by_node.contains(scene, id)
                    || units.recorded_by_node.contains(scene, id) || (units.hidden_sources.contains(id) && units.reads_source(id))) { continue; }
                if moda_scene::scene::is_parent_node(node) || patchable.page_effects.owns(id)
                    || scene.group_appearance_by_id().contains_key(id) { return None; }
            }
        }
        let resident_before = entry.plan.items.len();
        let mut scalar_rows = Vec::new();
        let mut patched_units = HashMap::default();
        let mut admitted_plain = None;
        if let Some(units) = &mut patchable.units {
            let prepared_before = units.segment.as_ref().and_then(|s| s.prepared.as_ref()).map(|r| r.bytes());
            let delta = units.promote_sources(scene, changed, entry.built_at, &mut entry.plan,
                &mut patchable.rows, &mut written, &patchable.index, &patchable.page_effects)?;
            entry.bytes = (entry.bytes as i64 + delta).max(0) as usize;
            let (bytes, excluded, plain) = units.patch(scene, &mut entry.plan, hashes, opts, changed,
                &mut patchable.index, &entry.warnings, caches, entry.bytes, &mut written, entry.built_at, &mut patchable.rows, entry.parts.as_mut(),
                parameter, &mut scalar_rows)?;
            entry.bytes = bytes;
            entry.reweigh |= prepared_before != units.segment.as_ref().and_then(|s| s.prepared.as_ref()).map(|r| r.bytes());
            patched_units = excluded;
            admitted_plain = Some(plain);
        }
        if defer_effects {
            scalar_rows.sort_unstable();
            entry.reweigh |= patchable.refresh_effects(scene, opts, at, changed, &scalar_rows)?;
        }
        if patchable.units.is_none() {
            // These leaf rows already proved source ownership, ancestry and
            // warning eligibility at collection. A hierarchy edit reseeds them.
            if let Some((rows, worlds)) = patchable.rows.as_ref().zip(scene.world_rows())
                .filter(|(rows, w)| rows.hierarchy == w.hierarchy()) {
                admitted_plain = changed.iter().map(|&row| {
                    let si = *patchable.index.by_node.rows.get(row as usize)? as usize;
                    (!worlds.is_parent(row) && worlds.whole(row) && *rows.follows.get(si)?).then_some(si)
                }).collect();
            }
        }
        let excluded = patchable.units.as_ref().map(|units| (&units.by_node, &patched_units));
        let index = &mut patchable.index;
        let changes = (generation != index.built_at_generation).then_some(changed);
        match apply_incremental_patch(index, &mut entry.plan, &entry.warnings,
            changes, generation, true, scene, hashes, caches, opts, entry.bytes,
            Some(&mut written), Some(entry.built_at), excluded, admitted_plain,
            patchable.units.as_ref().and_then(|units| units.segment.as_ref()), entry.parts.as_mut()) {
            IncrementalOutcome::Fallback => None,
            IncrementalOutcome::Served { bytes } => {
                // Re-emission can change coverage, phase and item kinds. Only the
                // placement writer preserves the proof of the solid run's layout.
                if let Some(rows) = &mut patchable.rows { rows.solid_groups.clear(); }
                if let Some(bytes) = bytes { entry.bytes = bytes; }
                entry.reweigh |= !written.replaced.is_empty() || written.masks_changed || entry.plan.items.len() != resident_before
                    || (written.placed.iter().any(|w| w.is_full()) && [moda_scene::stamps::Group::Content, moda_scene::stamps::Group::Paint]
                        .into_iter().any(|group| scene.stamps().latest(group) > at));
                entry.generation = generation;
                entry.initial = Default::default();
                entry.key = key;
                entry.built_at = scene.stamps().clock();
                entry.shader_time_s = opts.shader_time_s;
                // Unchanged body output keeps its emission proof. Only newly written masks can add a
                // pending capture proof; do not walk the whole page again after a small patch.
                if written.masks_changed { entry.uses_shader_time = masks_use_shader_time(&entry.plan.masks); }
                let served = (entry.plan.clone(), entry.warnings.clone());
                self.entry = Some(entry);
                self.written = written;
                Some(served)
            }
        }
    }
}

/// rust-edit-incremental-z4: assemble the per-page plan index from a full build's
/// `PlanBuildAux` + a one-time reverse-dependency scan of the scene.
fn build_plan_index(
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
    aux: crate::frame::PlanBuildAux,
    generation: u64,
    deps_memo: &mut Option<(u64, Arc<ReverseDeps>)>,
) -> PlanIndex {
    let mut by_node = SourceLookup::default();
    if !aux.spans.is_empty() { by_node.rows.resize(scene.doc().nodes.len(), u32::MAX); }
    for (i, span) in aux.spans.iter().enumerate() {
        match &span.source {
            crate::frame::SpanSource::Row(row) => by_node.rows[*row as usize] = i as u32,
            crate::frame::SpanSource::Generated(id) => { by_node.generated.insert(id.clone(), i); }
        }
    }
    // The scan is O(scene) and depends on the scene alone: once per generation (a
    // culled camera plan re-seeds its index on every pan past its window).
    let deps = match deps_memo {
        Some((gen, deps)) if *gen == generation => deps.clone(),
        _ => {
            let deps = Arc::new(scan_reverse_deps(scene));
            *deps_memo = Some((generation, deps.clone()));
            deps
        }
    };
    PlanIndex {
        culled_roots: aux.culled_roots.into_iter().collect(),
        spans: aux.spans.into(),
        by_node,
        page_index,
        page_clip: aux.page_clip,
        transparent: aux.transparent,
        opts_key: render_options_key(opts),
        built_at_generation: generation,
        deps,
    }
}

/// rust-edit-incremental-z4: collect the cross-node reverse dependencies that make
/// an edit's effect NON-local (so a dirty node in either set forces a full
/// rebuild). O(n), run once per full build. `blend_related_ids`: every blend node
/// + the objects and spine whose geometry it renders.
/// `mask_source_ids`: every `clipMaskNodeId` target (a group mask's coverage is
/// that node's geometry), and every clipped layer and its base (clip to the layer below). Connector anchors are deliberately absent — the engine
/// reads pre-baked line points node-locally, so a shape move never silently
/// re-routes a connector at plan time.
fn scan_reverse_deps(scene: &Scene) -> ReverseDeps {
    let mut deps = ReverseDeps::default();
    scene.visit_effective_nodes(|id, node| {
        if node.node_type() == "blend" {
            deps.blend_related_ids.insert(id.to_owned());
            for src in moda_scene::blend::dependency_ids(node) {
                deps.blend_related_ids.insert(src.to_string());
            }
        }
        if let Some(source) = node.str("clipMaskNodeId").filter(|m| !m.is_empty()) {
            let hidden = node.node_type() == "group" && scene.hidden_from_paint_order(id);
            deps.add_mask_source(source, scene.row_of(id).filter(|_| hidden));
        }
        if node.truthy("clipToBelow") {
            // The layer and the base both participate in sibling selection.
            deps.add_mask_source(id, None);
            if let Some(base) = scene.clip_below_base(id) { deps.add_mask_source(base, None); }
        }
    });
    // Generated nodes have no canonical row on which to retain a visibility proof.
    // Keep their source roles conservative without changing display-list membership.
    if !deps.hidden_clip_owners.is_empty() {
        for (id, node) in scene.transient_by_id() {
            if node.node_type() == "blend" {
                for source in moda_scene::blend::dependency_ids(node) {
                    deps.hidden_clip_owners.remove(source);
                }
            }
            if let Some(source) = node.str("clipMaskNodeId") { deps.hidden_clip_owners.remove(source); }
            if node.truthy("clipToBelow") {
                deps.hidden_clip_owners.remove(id);
                if let Some(base) = scene.clip_below_base(id) { deps.hidden_clip_owners.remove(base); }
            }
        }
    }
    deps
}

fn render_options_key(opts: &RenderOptions) -> u64 {
    let mut h = FxHasher::default();
    fold_render_options_fingerprint(&mut h, opts);
    h.finish()
}

/// rust-edit-incremental-z4: does any plan warning belong to node `id`? Warnings are
/// formatted `"{node_id} (...): ..."` / `"{node_id}: ..."` / `"{node_id}:innershadow ..."`,
/// so a node id followed by a space or colon is its prefix. The space/colon delimiter
/// keeps `n1` from matching `n10`'s warnings.
fn warnings_mention_node(skipped: &[String], id: &str) -> bool {
    skipped.iter().any(|w| {
        w.strip_prefix(id)
            // `rest.is_empty()` covers a warning that is EXACTLY the node id (no
            // suffix): `strip_prefix` yields `Some("")`, which neither delimiter
            // check would catch. Treating it as a match keeps the fallback strictly
            // conservative — a warned node never silently takes the patch path.
            .is_some_and(|rest| rest.is_empty() || rest.starts_with(' ') || rest.starts_with(':'))
    })
}

/// rust-edit-incremental-z4: is `id` a DESCENDANT of a blend- or mask-source node?
/// `scan_reverse_deps` records only the DIRECT source ids (a blend's objects and
/// spine, a `clipMaskNodeId` target), so a leaf nested inside a source GROUP is
/// not in those sets and would otherwise look eligible. But editing such a leaf
/// changes its ancestor's rendered geometry, which the blend/mask CONSUMER
/// composites — so the consumer's items (not the leaf's own span) go stale under an
/// in-place patch. Walk the parent chain and bail if any ancestor is a source.
/// Cycle-guarded: corrupt cyclic parent data resolves to "restricted" (fall back),
/// never an infinite loop.
fn has_blend_or_mask_ancestor(scene: &Scene, id: &str, index: &PlanIndex,
    memo: &mut HashMap<String, bool>) -> bool {
    if index.deps.blend_related_ids.is_empty() && index.deps.mask_source_ids.is_empty() { return false; }
    let mut chain = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut cursor = id;
    let restricted = loop {
        if let Some(&known) = memo.get(cursor) { break known; }
        if !seen.insert(cursor) { break true; }
        chain.push(cursor);
        if index.deps.blend_related_ids.contains(cursor) || index.deps.mask_source_ids.contains(cursor) { break true; }
        match scene.group_parent_of(cursor).or_else(|| scene.transient_parent_of().get(cursor)) {
            Some(parent) => cursor = parent,
            None => break false,
        }
    };
    for node in chain { memo.insert(node.to_string(), restricted); }
    restricted
}

/// renderer-perf-more-z6 (subtree/group incremental edits): expand a host-marked
/// dirty node into the set of ITEM-PRODUCING nodes whose DrawItems the edit can
/// actually change, appending each node's SPAN INDEX (an index into `index.spans`)
/// to `out` (deduped via `out_seen`) — span indices, not ids, so the walk allocates
/// nothing and the caller reads the id from the span directly.
///
/// - A dirty LEAF (no children) maps to itself: it must have a cached span, else
///   we cannot know its item shape stayed stable (it might have just started
///   producing items) — return `false` so the caller falls back. This preserves
///   the exact pre-existing single-leaf behaviour.
/// - A dirty PARENT (`group`/`container` with `children`) maps to every
///   item-producing node in its subtree (itself, if it has a body span — a
///   container with a fill/stroke does — plus all descendants present in the
///   cached index). An inherited transform/opacity edit on a parent flows down to
///   each descendant; `rebuild_node_items` re-reads the full ancestor product, so
///   each rebuilt descendant is byte-identical to a from-scratch rebuild. A parent
///   edit only reaches this path when it is transform/opacity-ONLY: `apply_props`
///   classifies any other key on a parent (fill/stroke/`clipMaskNodeId`/effects/…)
///   as STRUCTURAL via its `SAFE_PARENT_KEYS` gate (and `opacity` crossing 0,
///   `zIndex`, `visible`/`hidden` are structural for any node). A transform/opacity
///   edit never changes WHICH items the parent OR a descendant produces, only their
///   values — so the item set in `by_node` is stable and skipping a parent with no
///   span (a group, or a fill-less container) can never drop a needed item. This is
///   what makes an empty-`affected` expansion (a transform/opacity edit on a subtree
///   with no item-producing descendants) a correct no-op rather than a stale serve.
///
/// - A dirty TRANSIENT HOST (a `table`, or a `generated` chart — anything
///   with a `transient_by_parent` entry) maps to its expanded transient children.
///   A host is NOT `is_parent_node` (that is group/container only) and has no span of
///   its own — it is expanded AWAY, never its own draw item — so it contributes
///   nothing itself and everything through its transients. An EMPTY expansion is a
///   correct no-op for the same reason a span-less group is.
///
/// BYTE-EXACTNESS of splicing a RE-EXPANDED host's children into a retained plan.
/// Four points, in order:
///
/// (a) **The transient records are fully re-derived, not mutated.**
///     `apply_props` merges the patch into the host's CANONICAL node and then calls
///     `Scene::reexpand_transient_host`, which re-resolves the host through
///     `variables::resolve_node` and re-runs `expand_table_node` /
///     `expand_generated_graphic_node` from scratch. Expansion is a pure function of
///     the resolved host, and is ancestor-free (a transient inherits ancestor opacity
///     and transform at PAINT time, via `ancestor_opacity_product` /
///     `transform_parent_chain` — never at expansion time). So every transient record
///     is bit-identical to what `Scene::load(patched_json)` would have produced. A CPU
///     parity oracle in `moda-scene` asserts exactly that, over all five transient maps
///     plus `resolved_by_id`.
///
/// (b) **An identical id SEQUENCE means the display list is unchanged in membership
///     AND order,** so every cached plan-index span stays valid. `Scene::visit_z` emits
///     a host's transients by walking `transient_by_parent[host]` IN ORDER, so that
///     vector IS the display-list order for the host's slice. `apply_props` compares
///     `removed` vs `added` as sequences (not sets) and classifies any difference —
///     add, remove, or permutation — as STRUCTURAL → full rebuild. A same-length,
///     same-order re-expansion therefore patches spans that still describe the same
///     items.
///
/// (c) **`rebuild_node_items` re-reads the ancestor transform/opacity fresh** for each
///     rebuilt transient, so a `native-children` generated host's children get the
///     right model matrix even though the host moved and the children's own coords are
///     host-LOCAL. (Table transients bake the host x/y into their own coords instead;
///     re-expansion is what makes that correct, and their re-hash is what makes the
///     fingerprint move — see `reexpand_host_in_place`.)
///
/// (d) **`items_shape_match` still gates the splice** on item count, per-item variant,
///     node id and mask-freedom. So a re-expansion that keeps the id sequence but
///     changes what a transient PRODUCES — the canonical case being a cell whose text
///     went empty↔non-empty, which adds/removes a text item — is rejected and falls
///     back to a full rebuild rather than serving stale pixels. Pinned by a test.
///
/// Bounded, cycle-guarded subtree walk: a corrupt cyclic structure or a walk that
/// would exceed the indexed node count resolves to `false` (fall back), never an
/// infinite loop. Returns `false` (→ full rebuild) on any doubt.
fn collect_affected_item_nodes<'a>(
    scene: &'a Scene,
    index: &PlanIndex,
    warnings: &PlanWarnings,
    root: &'a str,
    out: &mut Vec<usize>,
    out_seen: &mut std::collections::HashSet<usize>,
    walked: &mut std::collections::HashSet<&'a str>,
    excluded: Option<(&SourceLookup, &HashMap<String, usize>)>,
) -> bool {
    if excluded.is_some_and(|nodes| nodes.0.contains(scene, root) || nodes.1.contains_key(root)) { return true; }
    let Some(root_node) = scene.effective_node(root) else {
        return false; // unknown id — be conservative
    };
    // Viewport culling: a node the build culled (or one under a culled parent) has
    // no span; its edit may bring it into the window.
    if under_culled_root(scene, index, root) {
        return false;
    }
    // A dirty leaf must already be an item-producing node; if it has no span the
    // edit may have changed its item shape (e.g. empty → non-empty) and we cannot
    // patch it safely. (Identical to the pre-existing single-node guard.) We collect
    // SPAN INDICES (not ids): zero per-node allocation on this hot subtree-walk path,
    // and the caller reads `index.spans[si].node_id` directly (no second `by_node`
    // lookup).
    // A transient host walks its subtree too (its transient children),
    // even though it is not `is_parent_node` and has no span of its own.
    let walk_subtree = moda_scene::scene::is_parent_node(root_node)
        || scene.transient_by_parent().contains_key(root);
    if !walk_subtree {
        let Some(si) = index.by_node.get(scene, root) else {
            // An absent leaf remains empty under the ordinary visibility selector. Derived-node
            // sources still decline (placeable); mask consumers were handled by the unit index.
            return scene.row_of(root).is_some_and(|row| scene.placeable(row))
                && scene.hidden_from_paint_order(root)
                && !warnings_mention_node(&warnings.skipped, root)
                && !warnings_mention_node(&warnings.dropped, root);
        };
        if out_seen.insert(si) {
            out.push(si);
        }
        return true;
    }
    // Parent: walk the subtree (inclusive) and collect every node that has a span.
    // Cap the walk at the TOTAL scene node count (+slack), not just the item-
    // producing span count: a deeply-nested hierarchy of span-less group/container
    // boundaries is a legitimate subtree but has more nodes than spans, so capping
    // on `spans.len()` would trip a spurious (correctness-safe but slow) fallback.
    // The total node count is the true upper bound; exceeding it means corrupt/
    // cyclic data.
    //
    // TRANSIENTS COUNT. `node_order` is CANONICAL nodes only, and a
    // transient host is one canonical node that expands into hundreds (a 4x3 table
    // → 30, a 20x20 → ~1.6k). Before this term the cap on a two-node page holding
    // one small table was 18 while the walk legitimately visited 31, so EVERY host
    // patch tripped the runaway guard and fell back — silently giving back the whole
    // win with no error anywhere. Canonical + transient is the true upper bound.
    let cap = scene.node_order().len() + scene.transient_by_id().len() + 16;
    let mut stack: Vec<&'a str> = vec![root];
    while let Some(id) = stack.pop() {
        if excluded.is_some_and(|nodes| nodes.0.contains(scene, id) || nodes.1.contains_key(id)) { continue; }
        if !walked.insert(id) {
            continue; // already visited — cycle-safe
        }
        if walked.len() > cap {
            return false; // runaway / corrupt structure → fall back
        }
        if index.culled_roots.contains(id) {
            return false; // a culled descendant may move into the window
        }
        // Self contributes only if it produced items (a pure group boundary does
        // not; a container body / leaf does).
        if let Some(si) = index.by_node.get(scene, id) {
            if out_seen.insert(si) {
                out.push(si);
            }
        }
        if let Some(kids) = scene.children_by_parent().get(id) {
            for k in kids {
                stack.push(k.as_str());
            }
        }
        if let Some(kids) = scene.transient_by_parent().get(id) {
            for k in kids {
                stack.push(k.as_str());
            }
        }
    }
    true
}

/// rust-edit-incremental-z4: outcome of an incremental patch attempt on a
/// taken-out slot's `index` + `plan`.
enum IncrementalOutcome {
    /// The slot serves this render (unchanged or patched in place) — KEEP it.
    /// `bytes` is the patched plan's recomputed marginal heap when a patch ran
    /// (so the caller updates its `incr_*_bytes` accounting), or `None` for a
    /// no-op static re-render (the byte count is unchanged).
    Served { bytes: Option<usize> },
    /// Cannot serve incrementally — the caller full-rebuilds + reseeds.
    Fallback,
}

/// The ordinary span writer's source-role proof, shared with source-interval promotion.
fn plain_source_eligible(scene: &Scene, index: &PlanIndex, id: &str,
    restriction_memo: &mut HashMap<String, bool>) -> bool {
    !index.deps.blend_related_ids.contains(id) && !index.deps.mask_source_ids.contains(id)
        && !has_blend_or_mask_ancestor(scene, id, index, restriction_memo)
}

/// Admit all plain source writes before any unit performs ordinary preparation.
fn plain_patch_spans(
    scene: &Scene, index: &PlanIndex, seed_warnings: &PlanWarnings, dirty: &[u32],
    excluded: Option<(&SourceLookup, &HashMap<String, usize>)>, placed: Option<&PlacedRows>,
) -> Option<Vec<usize>> {
    let mut affected: Vec<usize> = Vec::with_capacity(dirty.len());
    let mut affected_seen: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    let mut walked = std::collections::HashSet::new();
    let worlds = scene.world_rows();
    for &row in dirty {
        // The retained row already certifies an ordinary, warning-free leaf outside
        // all source edges. Parents and generated children still take the subtree walk.
        if let Some(si) = index.by_node.row(row).filter(|&si| placed.is_some_and(|read|
            !read.unit[row as usize] && read.follows[si]) && worlds.is_some_and(|w| w.whole(row) && !w.is_parent(row))) {
            if affected_seen.insert(si) { affected.push(si); }
            continue;
        }
        let id = scene.row(row)?.id();
        if !collect_affected_item_nodes(scene, index, seed_warnings, id, &mut affected, &mut affected_seen, &mut walked, excluded)
        {
            return None;
        }
    }
    // Gather the spans to patch, bailing (full rebuild) on any node that is
    // not fast-path-eligible. Also bail if a node carried a warning in the
    // SEED build (`skipped` OR a fail-closed `dropped` refusal): the patch
    // path returns the seed warnings unchanged, so a node that warned then
    // could leave a stale entry once it stops warning — fall back to keep the
    // host-visible warning set authoritative (the rebuild below independently
    // bails if a patched node produces a NEW warning of either kind).
    // `affected` holds span indices, so the node id comes straight from the
    // span (no second `by_node` lookup, no alloc).
    let mut restriction_memo = HashMap::default();
    for &si in &affected {
        if placed.is_some_and(|read| read.follows[si]) { continue; }
        // Checked span lookup (`.get`, not `[]`) so a desynced index can
        // never panic — any miss just falls back to the full rebuild.
        let Some(span) = index.spans.get(si) else {
            return None;
        };
        let id = span.source.id(scene);
        // Eligibility: the node must not be a blend/mask source nor nested
        // inside one (editing it changes a CONSUMER's composite, not just
        // this node's own span). Container bodies (a parent WITH a span) are
        // allowed — `rebuild_node_items` drives the exact same
        // `append_node_draw_items` route for them, and the post-rebuild
        // shape/mask checks below catch anything unexpected.
        let eligible = plain_source_eligible(scene, index, id, &mut restriction_memo);
        if !eligible
            || warnings_mention_node(&seed_warnings.skipped, id)
            || warnings_mention_node(&seed_warnings.dropped, id)
        {
            return None;
        }
    }
    Some(affected)
}

/// Re-emit one source range using the same primary and node planner as the full walk.
#[allow(clippy::too_many_arguments)]
fn emit_span(scene: &Scene, span: &crate::frame::NodeSpan, opts: &RenderOptions,
    transparent: bool, page_clip: [f32; 4], backdrop: Option<&crate::backdrop::PlanCtx>,
    layer: Option<&crate::backdrop::PlanCtx>, caches: &mut PlanCaches, hashes: &dyn NodeHashSource,
    masks: &[crate::seams::MaskPlan], mask_sources: &weights::RetainedSet<String>,
    content_current: bool, segment: Option<&crate::frame::CameraSegment>,
    mask_owners: &mut Option<HashMap<String, usize>>) -> Option<(Vec<crate::seams::DrawItem>, u64, Option<Box<crate::frame::TranslationPlacement>>)> {
    let id = span.source.id(scene);
    let cached = segment.and_then(|seg| seg.dl.items.get(span.primary))
        .filter(|item| content_current && item.node_id == id);
    let fresh;
    let primary = if let Some(item) = cached { item } else {
        fresh = moda_scene::display_list::display_list_items_for(scene, &[id.to_owned()], span.dl_z, mask_sources);
        fresh.iter().find(|item| item.is_primary_node_item)?
    };
    let mut scratch = crate::effects::PlanMasks::with_text_masks(opts.text_masks_available.unwrap_or(true))
        .with_vector_text_masks(true).in_working_color(opts.working.is_some());
    let (mut items, consumed, warnings, recipe) = crate::frame::rebuild_node_items(scene, primary, opts,
        transparent, page_clip, backdrop, layer, &mut scratch, Some((caches, hashes)), span.z);
    let mut mask_warnings = crate::frame::PlanWarnings::default();
    let local = scratch.into_masks(&mut mask_warnings);
    (warnings.is_empty() && mask_warnings.is_empty()
        && remap_scratch_masks(&mut items, &local, masks, mask_owners)).then_some((items, consumed, recipe))
}

/// Draw items for a run of changed sources, partitioned by their validated spans.
struct EmittedRun {
    spans: std::vec::IntoIter<crate::frame::NodeSpan>,
    items: std::vec::IntoIter<crate::seams::DrawItem>,
}

impl EmittedRun {
    fn prepare(scene: &Scene, index: &PlanIndex, dirty: &[usize], opts: &RenderOptions,
        cache: crate::frame::CacheCtx<'_>, prior: Option<crate::frame::PriorGlyph<'_>>) -> Option<Self> {
        use rayon::prelude::*;
        let anchors: Vec<_> = dirty.iter().map(|&si| index.spans.anchor(si, scene)).collect::<Option<_>>()?;
        let first = *anchors.first()?;
        let _profile = crate::prof::span("canvas.plan.emit_changed_run");
        let grain = dirty.len().div_ceil(rayon::current_num_threads()).max(2048);
        let chunks: Vec<_> = anchors.par_chunks(grain).map(|rows| {
            let sources = rows.iter().map(|&(id, ordinal, _, _)| (ordinal, id));
            moda_scene::display_list::display_list_items_at(scene, sources, &index.deps.mask_source_ids)
        }).collect();
        let dl = moda_scene::DisplayList { page_id: String::new(), page_index: 0, items: chunks.into_iter().flatten().collect() };
        let mut items = Vec::new();
        let mut spans = Vec::new();
        let mut warnings = crate::frame::PlanWarnings::default();
        if dl.items.len() != anchors.len() { return None; }
        let sparse = !anchors.windows(2).all(|p| p[0].2.checked_add(p[0].3) == Some(p[1].2));
        let depths = sparse.then(|| anchors.iter().map(|a| a.2).collect::<Vec<_>>());
        crate::frame::emit_source_run(scene, &dl, opts, index.transparent, index.page_clip, first.2,
            depths.as_deref(), cache, prior, &mut items, &mut warnings, &mut spans)?;
        if !warnings.is_empty() || spans.len() != dirty.len() { return None; }
        if !spans.iter().zip(&anchors).all(|(new, &(id, _, z, consumed))| {
            new.source.id(scene) == id && new.z == z && new.z_consumed == consumed
        }) { return None; }
        Some(Self { spans: spans.into_iter(), items: items.into_iter() })
    }

    fn next(&mut self) -> Option<(Vec<crate::seams::DrawItem>, u64, Option<Box<crate::frame::TranslationPlacement>>)> {
        let span = self.spans.next()?;
        Some((self.items.by_ref().take(span.len).collect(), span.z_consumed, span.placement))
    }
}

/// Patch a retained plan after the caller has checked its identity. Given a
/// taken-out slot's `index` + `plan` + the seed build's warnings, applies
/// the dirty-node patches IN PLACE (via `Arc::make_mut`) and reports whether the
/// slot can serve this render. Callers do the slot-identity / opts / resource-key
/// gate first; this handles only `dirty_since` → eligibility → splice + stats.
#[allow(clippy::too_many_arguments)]
fn apply_incremental_patch(
    index: &mut PlanIndex,
    plan: &mut Arc<CoveragePlan>,
    seed_warnings: &crate::frame::PlanWarnings,
    changes: Option<&[u32]>,
    generation: u64,
    translation_only: bool,
    scene: &Scene,
    hashes: &dyn NodeHashSource,
    caches: &mut PlanCaches,
    opts: &RenderOptions,
    current_bytes: usize,
    // Where to note the items written, for a host that tells the engine what changed (`FramePlanIndex::patch`).
    mut written: Option<&mut PlanWritten>,
    // Host-owned plans use the scene clock to validate each span's content independently.
    built_at: Option<u64>,
    excluded: Option<(&SourceLookup, &HashMap<String, usize>)>,
    admitted: Option<Vec<usize>>,
    page_segment: Option<&crate::frame::CameraSegment>,
    mut parts: Option<&mut parts::PlanParts>,
) -> IncrementalOutcome {
    match changes {
        None => {
            // Static re-render — nothing changed since the slot was built.
            caches.incr_static_hits = caches.incr_static_hits.wrapping_add(1);
            IncrementalOutcome::Served { bytes: None }
        }
        Some(dirty) => {
            // renderer-perf-more-z6: expand each host-marked dirty node into the
            // item-producing nodes whose DrawItems the edit can change — a dirty
            // leaf is itself; a dirty PARENT (group/container) is its whole subtree
            // of item-producers, so a group/multi-node transform-or-opacity drag
            // patches O(subtree) leaves instead of falling back to a full O(n)
            // rebuild. Each expanded node is rebuilt independently below and
            // shape-checked; byte-exactness is proven by `MODA_INCREMENTAL_VERIFY`
            // + the `mutation_replay` GPU gate. Any doubt (unknown id, dirty leaf
            // with no span, cyclic/runaway subtree) → full rebuild.
            let mut to_patch = match admitted {
                Some(spans) => spans,
                None => match plain_patch_spans(scene, index, seed_warnings, dirty, excluded, None) {
                    Some(spans) => spans,
                    None => {
                        caches.incr_fallbacks = caches.incr_fallbacks.wrapping_add(1);
                        return IncrementalOutcome::Fallback;
                    }
                },
            };
            // A derived mask value may reach another unit during execution. Its source slots now
            // belong to that executor too; never apply the plain writer over those replaced outputs.
            if let Some((owners, reached)) = excluded {
                to_patch.retain(|&si| index.spans.get(si).is_some_and(|span|
                    !owners.contains(scene, span.source.id(scene)) && !reached.contains_key(span.source.id(scene))));
            }
            let page_clip = index.page_clip;
            let transparent = index.transparent;
            let (backdrop_ctx, layer_blur_ctx) = page_segment.map_or((None, None), |seg|
                crate::frame::page_range_contexts(scene, plan, seg, opts));
            // `make_mut` is in-place when the slot UNIQUELY owns the Arc (the page
            // editor lane between frames → true O(k)). On the CAMERA lane the
            // engine's prepared-frame holds a second strong ref, so it clones —
            // which is REQUIRED there anyway: a fresh Arc pointer makes the
            // prepared-frame `ptr_eq` correctly MISS and rebuild the GPU buffers,
            // whereas an in-place mutate would alias the same pointer and present
            // stale buffers. Either way correct; the camera clone copies N DrawItems
            // — but since renderer-perf-more-z6 Arc-shared the body geometry, that is
            // O(n) pointer-bumps for the (large) vertex streams plus the smaller
            // per-item `Box<PaintInstance>` + `node_id` String copies, not an O(n)
            // deep vertex copy. It still skips the DL build + fingerprint + per-node
            // instance assembly a full rebuild pays.
            let plan_mut = Arc::make_mut(plan);
            // Delta byte-accounting: accumulate (new span bytes − old span bytes)
            // over the patched spans only, so the slot's retained size is kept
            // accurate in O(patched-items) instead of an O(n) full re-walk per edit
            // (preserves the page lane's O(dirty) cost).
            let mut delta_bytes: i64 = 0;
            // Owner id → index in the plan's mask table, built on the first masked node.
            let mut mask_owners: Option<HashMap<String, usize>> = None;
            use moda_scene::stamps::Group;
            // A group that has no writes since emission stands for every row. Resolve a source row
            // only when a changed group needs its local proof; placement-only frames do neither lookup.
            let content_stands = built_at.is_none_or(|at| scene.stamps().latest(Group::Content) <= at);
            let paint_stands = built_at.is_none_or(|at| scene.stamps().latest(Group::Paint) <= at);
            // Bound the temporary draw list while giving large edits the same worker
            // parallelism available during initial collection. Entries are disjoint plain
            // spans; sorting them changes neither painter order nor packet slots.
            let can_batch = to_patch.len() >= 4096 && backdrop_ctx.is_none() && layer_blur_ctx.is_none()
                && plan_mut.masks.is_empty() && scene.group_appearance_by_id().is_empty();
            if can_batch { to_patch.sort_unstable(); }
            const BATCH: usize = 32 * 1024;
            // When the writer owns every slot of a contiguous interval, the
            // ordinary bulk emitter can supply its final painter-order output.
            // Keep logical packet slots fixed. Solid paint edits can reuse body
            // geometry, and prepared glyphs can update their own destinations.
            let rewrite = can_batch && (!content_stands || !paint_stands) && plan_mut.item_order.is_none();
            let range = rewrite.then(|| {
                if let Some(table) = parts.as_ref() {
                    let range = table.rewrite_range(&to_patch)?;
                    if page_segment.is_some_and(|seg| seg.units.iter().any(|u|
                        seg.prefix.saturating_add(u.item_start).saturating_add(u.item_len) > range.start)) { return None; }
                    return Some(range);
                }
                let start = index.spans.range(to_patch[0]).0;
                let mut end = start;
                for &si in &to_patch {
                    let (from, len) = index.spans.range(si);
                    if from != end { return None; }
                    end = from.checked_add(len)?;
                }
                Some(start..end)
            }).flatten();
            if let Some((range, writes)) = range.zip(written.as_deref_mut()) {
                if let Some(table) = parts.as_ref() { table.expand_rewrite(&mut plan_mut.items); }
                let mut offset = range.start;
                let paint_only = built_at.is_some_and(|at| content_stands
                    && scene.stamps().latest(Group::Placement) <= at && scene.stamps().last_composition() <= at);
                for sources in to_patch.chunks(BATCH) {
                    let prior = |ordinal: usize| {
                        let si = *sources.get(ordinal)?;
                        let (start, len) = index.spans.range(si);
                        let at = parts.as_ref().map_or(start, |table| table.start(si));
                        let crate::seams::DrawItem::Text(old) = plan_mut.items.get(at)? else { return None; };
                        (len == 1 && (content_stands || scene.row_of(&old.node_id).is_some_and(|row|
                            scene.stamps().current(row, Group::Content, built_at.unwrap())))).then_some(old.as_ref())
                    };
                    let Some(run) = EmittedRun::prepare(scene, index, sources, opts, (caches, hashes),
                        translation_only.then_some(&prior))
                        .filter(|run| run.spans.len() == sources.len()) else { return IncrementalOutcome::Fallback; };
                    writes.recollect |= !run.items.as_slice().iter().all(|item| match item {
                        crate::seams::DrawItem::Text(batch) => batch.paint_geometry_key.is_some(),
                        crate::seams::DrawItem::Body { paint: crate::seams::BodyPaint::Solid, mask: None, .. } => paint_only,
                        _ => false,
                    });
                    for (&si, mut next) in sources.iter().zip(run.spans) {
                        let (old_start, len) = index.spans.range(si);
                        let logical = parts.as_ref().map_or(old_start, |table| table.start(si));
                        if next.len != len || offset + next.start != logical {
                            return IncrementalOutcome::Fallback;
                        }
                        next.start += offset;
                        next.primary = usize::MAX;
                        index.spans[si] = next;
                    }
                    let end = offset + run.items.len();
                    let Some(slots) = plan_mut.items.get_mut(offset..end) else { return IncrementalOutcome::Fallback; };
                    for (slot, item) in slots.iter_mut().zip(run.items) { *slot = item; }
                    offset = end;
                }
                if offset != range.end { return IncrementalOutcome::Fallback; }
                // Outside the interval physical addresses now equal logical ones too.
                if let Some(table) = parts.as_deref_mut() {
                    index.spans.remap_starts(|si, _, _| table.start(si));
                    table.finish_rewrite();
                }
                writes.placed.extend(range.map(crate::frame::ItemWrite::full));
                index.built_at_generation = generation;
                caches.incr_patch_hits += 1;
                caches.incr_patch_nodes += to_patch.len() as u64;
                return IncrementalOutcome::Served { bytes: None };
            }
            let mut emitted: Option<EmittedRun> = None;
            let mut batch_end = 0;
            for (i, &si) in to_patch.iter().enumerate() {
                let mut next = emitted.as_mut().and_then(EmittedRun::next);
                let span = &mut index.spans[si];
                let id = span.source.id(scene);
                let row = (!content_stands || !paint_stands).then(|| scene.row_of(id)).flatten();
                let content_current = content_stands || row.is_some_and(|row|
                    scene.stamps().current(row, Group::Content, built_at.unwrap()));
                let paint_current = paint_stands || row.is_some_and(|row|
                    scene.stamps().current(row, Group::Paint, built_at.unwrap()));
                if span.start != parts::ABSENT && translation_only && content_current && span.len == 1 {
                    if let Some(crate::seams::DrawItem::Text(old)) = plan_mut.items.get(span.start) {
                        if let Some(batch) = crate::text_draw::repaint_text(scene, id, old) {
                            let old = std::mem::replace(&mut plan_mut.items[span.start], crate::seams::DrawItem::Text(batch));
                            if let Some(written) = written.as_deref_mut() {
                                written.replaced.push((span.start as u32, old));
                            }
                            continue;
                        }
                    }
                }
                if span.start != parts::ABSENT && translation_only && content_current && paint_current {
                    if let (Some(recipe), Some(items)) = (span.placement.as_mut(),
                        plan_mut.items.get_mut(span.start..span.start + span.len)) {
                        let _placement = crate::prof::span("canvas.plan.placement");
                        // Capture cut-outs can shift a span. Recipes are certified only for the
                        // owner's plain items; generated paint/brush items instead rebuild below,
                        // where items_shape_match compares their actual old/new draw identities.
                        if let Some(value) = items.iter().all(|item| drawitem_node_id(item) == Some(id))
                            .then(|| crate::frame::placed_span(scene, recipe, items)).flatten() {
                            caches.incr_placement_nodes = caches.incr_placement_nodes.wrapping_add(1);
                            if let Some(written) = written.as_deref_mut() {
                                written.placed.extend(items.iter().enumerate().map(|(i, item)| value.write(span.start + i, item)));
                            }
                            crate::frame::place_span(recipe, &value, items);
                            continue;
                        }
                    }
                }
                // Start a batch only after a source actually needs the emitter. A dense
                // run of resident paint or placement writes needs no display list.
                if can_batch && i >= batch_end {
                    batch_end = (i + BATCH).min(to_patch.len());
                    emitted = EmittedRun::prepare(scene, index, &to_patch[i..batch_end], opts, (caches, hashes), None);
                    next = emitted.as_mut().and_then(EmittedRun::next);
                }
                let span = &mut index.spans[si];
                if !content_current { span.primary = usize::MAX; }
                let Some((new_items, z_consumed, recipe)) = next.or_else(|| emit_span(scene, span, opts, transparent, page_clip,
                    backdrop_ctx.as_ref(), layer_blur_ctx.as_ref(), caches, hashes, &plan_mut.masks,
                    &index.deps.mask_source_ids, content_current, page_segment, &mut mask_owners)) else {
                    caches.incr_fallbacks += 1;
                    return IncrementalOutcome::Fallback;
                };
                if span.start == parts::ABSENT {
                    if z_consumed != span.z_consumed { return IncrementalOutcome::Fallback; }
                    let Some(table) = parts.as_deref_mut() else { return IncrementalOutcome::Fallback; };
                    delta_bytes += (items_bytes(&new_items) + 4 * new_items.len()
                        ) as i64;
                    let Some(writes) = written.as_deref_mut() else { return IncrementalOutcome::Fallback; };
                    span.placement = recipe;
                    let paint_geometry_stands = content_current && !paint_current
                        && content_stands && built_at.zip(row).is_some_and(|(at,row)| scene.stamps().latest(Group::Placement) <= at
                            && scene.stamps().composition_current(row, at))
                        && new_items.iter().all(|item| matches!(item, crate::seams::DrawItem::Body {
                            paint: crate::seams::BodyPaint::Solid, mask: None, .. }));
                    if table.restore(si, span, plan_mut, new_items, writes, paint_geometry_stands).is_none() {
                        return IncrementalOutcome::Fallback;
                    }
                    continue;
                }
                let Some(old) = plan_mut.items.get(span.start..span.start + span.len) else {
                    caches.incr_fallbacks = caches.incr_fallbacks.wrapping_add(1);
                    return IncrementalOutcome::Fallback;
                };
                if z_consumed != span.z_consumed
                    || !items_shape_match(&new_items, old)
                {
                    caches.incr_fallbacks = caches.incr_fallbacks.wrapping_add(1);
                    return IncrementalOutcome::Fallback;
                }
                // Accumulate the byte delta while `old` is still borrowed (before the
                // splice below overwrites it).
                delta_bytes += items_bytes(&new_items) as i64 - items_bytes(old) as i64;
                // Dropping the slot on ANY rebuilt shader span (below) is
                // INTENTIONAL conservative simplicity — including an in-place edit of
                // an ALREADY re-threadable shader node, which could in principle be
                // retained. So a shader-node edit serves once then rebuilds; only
                // drags of UNRELATED (non-shader) nodes retain the slot. Refining
                // this to keep the slot on re-threadable in-place edits is a perf
                // optimization deliberately left out of scope.
                let Some(dest) = plan_mut.items.get_mut(span.start..span.start + span.len) else {
                    caches.incr_fallbacks = caches.incr_fallbacks.wrapping_add(1);
                    return IncrementalOutcome::Fallback;
                };
                // MOVE (not clone) the freshly built items into the plan: `new_items`
                // is discarded right after, so deep-cloning their geometry/text
                // batches would be pure waste. `items_shape_match` guaranteed equal
                // lengths, so the zip overwrites every dest slot exactly.
                for (at, (slot, mut item)) in dest.iter_mut().zip(new_items).enumerate() {
                    // A local emit can allocate the same vertices again after the tessellation
                    // cache evicts them. Keep their packet identity across paint-only writes.
                    if let (crate::seams::DrawItem::Body { geometry: a, .. },
                        crate::seams::DrawItem::Body { geometry: b, .. }) = (&*slot, &mut item) {
                        if !Arc::ptr_eq(a, b) && a == b { *b = a.clone(); }
                    }
                    let old = std::mem::replace(slot, item);
                    if let Some(written) = written.as_deref_mut() {
                        written.replaced.push(((span.start + at) as u32, old));
                    }
                }
                // The planner prepared the span again: its recipe describes the new items, or is dropped.
                span.placement = recipe;
            }
            index.built_at_generation = generation;
            caches.incr_patch_hits = caches.incr_patch_hits.wrapping_add(1);
            caches.incr_patch_nodes = caches.incr_patch_nodes.wrapping_add(to_patch.len() as u64);
            // The host trims the new footprint when this revision's draw releases its snapshot.
            let new_bytes = (current_bytes as i64 + delta_bytes).max(0) as usize;
            IncrementalOutcome::Served { bytes: Some(new_bytes) }
        }
    }
}

/// rust-edit-incremental-z4: structural match between a freshly-rebuilt node's items
/// and the cached ones — same count, same per-item variant + node id, and NEITHER
/// side carrying a mask ref (eligible nodes are unmasked; a mask ref means the node
/// is clipped and the scratch-table rebuild can't be spliced safely). Values
/// (geometry/transform/paint) legitimately differ after an edit — those are proven
/// equal to a full rebuild by `MODA_INCREMENTAL_VERIFY` + the mutation_replay gate.
fn items_shape_match(new: &[crate::seams::DrawItem], old: &[crate::seams::DrawItem]) -> bool {
    if new.len() != old.len() {
        return false;
    }
    new.iter().zip(old).all(|(n, o)| {
        std::mem::discriminant(n) == std::mem::discriminant(o)
            && drawitem_node_id(n) == drawitem_node_id(o)
            && drawitem_mask(n) == drawitem_mask(o)
    })
}

/// Slot identity excludes derived mask placement values; every changed input's consumers are prepared again.
fn items_slots_match(new: &[crate::seams::DrawItem], old: &[crate::seams::DrawItem]) -> bool {
    new.len() == old.len() && new.iter().zip(old).all(|(n, o)| {
        std::mem::discriminant(n) == std::mem::discriminant(o)
            && drawitem_node_id(n) == drawitem_node_id(o)
            && crate::camera_slot::item_mask_id(n) == crate::camera_slot::item_mask_id(o)
    })
}

/// Replayed ranges retain their warning and mask contracts. Only a resident unit with
/// relocatable successors may update its count or depth during a patch.
fn validate_unit_output(build: &mut crate::frame::RangeBuild, unit: &crate::frame::PlanUnit,
    fixed_count: bool, fixed_depth: bool, masks: &[crate::seams::MaskPlan],
    owners: &mut Option<HashMap<String, usize>>, writes: Option<&mut Vec<(usize, crate::seams::MaskPlan)>>) -> Option<(Vec<usize>, Vec<usize>)> {
    if (fixed_count && build.items.len() != unit.item_len)
        || (fixed_depth && build.end_z != unit.z + unit.z_consumed)
        || !build.mask_warnings.is_empty()
        || build.units.iter().flat_map(|part| &part.details().skipped).ne(&unit.details().skipped)
        || build.units.iter().flat_map(|part| &part.details().dropped).ne(&unit.details().dropped) { return None; }
    let restoring = writes.is_none();
    let mapping = remap_scratch_masks_record(&mut build.items, &build.masks, masks, owners, writes)?;
    let mut reads: Vec<_> = build.units.iter().flat_map(|p| &p.details().mask_reads)
        .map(|&id| mapping.get(id).copied()).collect::<Option<_>>()?;
    reads.sort_unstable();
    reads.dedup();
    if restoring && reads != unit.details().mask_reads { return None; }
    Some((mapping, reads))
}

/// Re-point a patched node's items from the scratch mask table its rebuild filled onto the plan's table.
/// Every scratch mask must equal the plan's mask for the same owner, field for field, with its ancestor
/// chain mapped the same way (an edited clip owner changes its mask, so the patch falls back and the full
/// rebuild re-plans the mask). False → fall back. `owners` caches the plan's owner → index map.
pub(crate) fn remap_scratch_masks(
    items: &mut [crate::seams::DrawItem],
    scratch: &[crate::seams::MaskPlan],
    plan: &[crate::seams::MaskPlan],
    owners: &mut Option<HashMap<String, usize>>,
) -> bool {
    remap_scratch_masks_record(items, scratch, plan, owners, None).is_some()
}

fn index_mask_owners(plan: &[crate::seams::MaskPlan]) -> HashMap<String, usize> {
    let mut owners = HashMap::with_capacity_and_hasher(plan.len(), Default::default());
    for (index, mask) in plan.iter().enumerate() {
        owners.entry(mask.owner_node_id.clone()).or_insert(index);
    }
    owners
}

fn remap_scratch_masks_record(
    items: &mut [crate::seams::DrawItem],
    scratch: &[crate::seams::MaskPlan],
    plan: &[crate::seams::MaskPlan],
    owners: &mut Option<HashMap<String, usize>>,
    mut writes: Option<&mut Vec<(usize, crate::seams::MaskPlan)>>,
) -> Option<Vec<usize>> {
    if scratch.is_empty() {
        return Some(Vec::new());
    }
    let owners = owners.get_or_insert_with(|| index_mask_owners(plan));
    // Scratch ancestors are built before their dependants (a smaller index), so `map` has them.
    let mut map: Vec<usize> = Vec::with_capacity(scratch.len());
    for m in scratch {
        let Some(&j) = owners.get(&m.owner_node_id) else {
            return None;
        };
        let ancestor = match m.ancestor {
            Some(a) => match map.get(a) {
                Some(&ja) => Some(ja),
                None => return None,
            },
            None => None,
        };
        if ancestor != plan[j].ancestor || m.source_node_id != plan[j].source_node_id { return None; }
        let mut mapped = std::borrow::Cow::Borrowed(m);
        if m.ancestor != ancestor { mapped.to_mut().ancestor = ancestor; }
        if *mapped != plan[j] || mask_reads_whole_scene(&mapped) {
            writes.as_deref_mut()?.push((j, mapped.into_owned()));
        }
        map.push(j);
    }
    for item in items.iter_mut() {
        let Some(id) = crate::camera_slot::item_mask_id(item) else { continue };
        match map.get(id as usize) {
            Some(&j) => crate::camera_slot::set_item_mask_id(item, j as u64),
            None => return None,
        }
    }
    Some(map)
}

pub fn drawitem_node_id(item: &crate::seams::DrawItem) -> Option<&str> {
    use crate::seams::DrawItem;
    match item {
        DrawItem::Vacant | DrawItem::PageFill { .. } => None,
        DrawItem::Body { node_id, .. } => Some(node_id),
        DrawItem::Shadow { node_id, .. } => Some(node_id),
        DrawItem::Text(b) => Some(&b.node_id),
        DrawItem::BackdropSplit(s) => Some(&s.node_id),
        // SDF bodies are unmasked (an active mask disqualifies them),
        // so they're eligible for the incremental shape-match — keyed by node id
        // like any other body.
        DrawItem::SolidShapeSdf { node_id, .. } => Some(node_id),
        DrawItem::GradientShapeSdf { node_id, .. } => Some(node_id),
        // Unreachable through the incremental lanes (layer-blur pages
        // never seed the slot), keyed defensively anyway.
        DrawItem::LayerBlurComposite { node_id, .. } => Some(node_id),
    }
}

fn drawitem_mask(item: &crate::seams::DrawItem) -> Option<&crate::seams::MaskRef> {
    use crate::seams::DrawItem;
    match item {
        DrawItem::Body { mask, .. } => mask.as_ref(),
        // vork: a text batch carries no mask (moda-text-draw's TextQuadBatch; see PROVENANCE.md).
        DrawItem::Text(_) => None,
        DrawItem::LayerBlurComposite { mask, .. } => mask.as_ref(),
        _ => None,
    }
}

/// rust-edit-incremental-z4: dev correctness oracle. When `MODA_INCREMENTAL_VERIFY`
/// is set, rebuild the page plan from scratch and assert the incremental plan's
/// items are byte-for-byte (fingerprint-)identical. Off by default (a no-op) — the
/// shipped proof is the `mutation_replay` GPU gate; this is the fast CPU loop.
fn debug_verify_incremental(
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
    hashes: &dyn NodeHashSource,
    got: &Arc<CoveragePlan>,
) {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let on = *ENABLED.get_or_init(|| std::env::var("MODA_INCREMENTAL_VERIFY").is_ok());
    if !on {
        return;
    }
    let mut fresh_caches = PlanCaches::default();
    let (full, _w, _aux) = crate::frame::build_coverage_plan_indexed(
        scene,
        page_index,
        opts,
        Some((&mut fresh_caches, hashes)),
    )
    .expect("verify: full rebuild");
    // The PAGE lane is non-shader-only (never seeds a shader plan
    // here), so a served page plan carries no re-threaded clock — the raw
    // fingerprint is correct (and strictly tighter than the normalized variant the
    // CAMERA oracle needs).
    let gi = plan_items_fingerprint(got);
    let fi = plan_items_fingerprint(&full);
    if gi != fi {
        let first_div = got.ordered_items().zip(full.ordered_items()).position(|(a, b)| drawitem_fingerprint(a) != drawitem_fingerprint(b));
        panic!(
            "MODA_INCREMENTAL_VERIFY: incremental plan diverged from full rebuild \
             (page {page_index}): incr {} items / full {} items; first divergent item index {:?}",
            got.ordered_items().len(),
            full.ordered_items().len(),
            first_div
        );
    }
}

/// rust-edit-incremental-z4: a deterministic fingerprint of a plan's draw stream
/// (item-by-item, including f32 bits), for the verifier + tests.
fn plan_items_fingerprint(plan: &CoveragePlan) -> u64 {
    let mut h = FxHasher::default();
    h.write_usize(plan.ordered_items().len());
    for it in plan.ordered_items() {
        h.write_u64(drawitem_fingerprint(it));
    }
    h.write_usize(plan.masks.len());
    h.finish()
}

/// rust-edit-incremental-z4: fingerprint one DrawItem via its Debug encoding — a
/// pragmatic deterministic digest that covers all fields (positions, matrices,
/// colors) without per-type PartialEq. Debug is derived on every DrawItem field, so
/// equal Debug ⇒ structurally equal; f32s render losslessly enough for this dev
/// oracle (the shipped proof is the GPU byte gate).
fn drawitem_fingerprint(item: &crate::seams::DrawItem) -> u64 {
    let mut h = FxHasher::default();
    h.write(format!("{item:?}").as_bytes());
    h.finish()
}

/// Callers outside `ResidentScene` must advance `generation` for every scene mutation.
pub fn page_frame_key(generation: u64, page_index: usize, opts: &RenderOptions) -> u64 {
    frame_key_with_assets(generation, page_index, opts, moda_assets::engine_seam::store_generation())
}

fn layout_frame_key(page_index: usize, opts: &RenderOptions) -> u64 {
    frame_key_with_assets(0, page_index, opts, moda_assets::engine_seam::layout_generation())
}

fn frame_key_with_assets(generation: u64, page_index: usize, opts: &RenderOptions, assets: u64) -> u64 {
    let mut h = FxHasher::default();
    h.write(b"page-frame-key");
    h.write_u64(generation);
    h.write_usize(page_index);
    fold_render_options_fingerprint(&mut h, opts);
    h.write_u64(assets);
    h.write_u64(crate::text_draw::text_host_generation());
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    #[test]
    fn render_options_fingerprint_preserves_filter_boundaries() {
        let mut options = RenderOptions { exclude: vec!["ab".into(), "c".into()], ..Default::default() };
        let before = options_fingerprint(&options);
        options.exclude = vec!["a".into(), "bc".into()];
        assert_ne!(before, options_fingerprint(&options));
        let excluded = options_fingerprint(&options);
        options.isolate = std::mem::take(&mut options.exclude);
        assert_ne!(excluded, options_fingerprint(&options));
    }

    #[test]
    fn render_options_fingerprint_includes_both_colour_conversions() {
        use moda_shaders::working_color::{Conversion, WorkingColor};
        let conversion = Conversion { decode: true, encode: false, matrix: None };
        let working = WorkingColor { from_srgb: conversion, to_srgb: conversion };
        let mut options = RenderOptions { working: Some(working), ..Default::default() };
        let before = options_fingerprint(&options);
        let matrix = Some([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        let converted = Conversion { matrix, ..conversion };
        options.working = Some(WorkingColor { from_srgb: converted, ..working });
        let with_matrix = options_fingerprint(&options);
        assert_ne!(before, with_matrix);
        options.working.as_mut().unwrap().from_srgb.matrix.as_mut().unwrap()[0][0] = 0.5;
        assert_ne!(with_matrix, options_fingerprint(&options));
        options.working = Some(WorkingColor { to_srgb: converted, ..working });
        assert_ne!(before, options_fingerprint(&options));
        assert_ne!(with_matrix, options_fingerprint(&options));
        options.working = Some(WorkingColor {
            from_srgb: Conversion { decode: false, ..conversion },
            ..working
        });
        assert_ne!(before, options_fingerprint(&options));
        options.working = Some(WorkingColor {
            to_srgb: Conversion { encode: true, ..conversion },
            ..working
        });
        assert_ne!(before, options_fingerprint(&options));
    }

    fn options_fingerprint(options: &RenderOptions) -> u64 {
        let mut hash = FxHasher::default();
        fold_render_options_fingerprint(&mut hash, options);
        hash.finish()
    }

    /// The colour of a node's top visible solid fill.

    fn raw(v: Value) -> RawNode {
        match v {
            Value::Object(m) => RawNode::new(m),
            _ => panic!("node must be an object"),
        }
    }

    /// An opaque `DrawItem::Body` with a zeroed instance — the shared body shape
    /// for the residency fixtures. `node_id`, `paint`, and `geometry` are the only
    /// values that vary between tests, so they stay caller arguments.
    fn body_item(
        node_id: &str,
        paint: crate::seams::BodyPaint,
        geometry: Arc<crate::seams::TessellatedGeometry>,
    ) -> crate::seams::DrawItem {
        crate::seams::DrawItem::Body {
            node_id: node_id.to_string(),
            z: 0.0,
            phase: crate::seams::Phase::Opaque,
            paint,
            geometry,
            instance: Box::new(bytemuck::Zeroable::zeroed()),
            blend_mode: crate::seams::BlendMode::Normal,
            mask: None,
        }
    }

    /// The shared 100×100 `CoveragePlan` shell. `page_id` and `items` are the only
    /// values that vary between tests; every other field is the fixture default.
    fn coverage_plan(page_id: &str, items: Vec<crate::seams::DrawItem>) -> CoveragePlan {
        CoveragePlan {
            page_id: page_id.to_string(),
            page_index: 0,
            width_px: 100,
            height_px: 100,
            page_size: [100.0, 100.0],
            scale: 1.0,
            transparent: false,
            sample_count: 1,
            camera: [0.0, 0.0],
            items,
            item_order: None,
            masks: Default::default(),
            camera_frame: None,
            clear_rgba: None,
            region_viewport: None,
        }
    }

    /// A vector-lane text batch with `glyphs` per-glyph instance records,
    /// `curve_floats` curve points, and `tris_floats` solid-triangle floats —
    /// the three variable heap payloads `vector_batch_bytes` counts (plus the
    /// fixed-size `VectorOp` struct itself).
    pub(super) fn vector_text_batch(
        node_id: &str,
        glyphs: usize,
        curve_floats: usize,
        tris_floats: usize,
    ) -> crate::seams::TextQuadBatch {
        use moda_text::vector::ops::VectorAnalyticRecord;
        use moda_text::vector::{VectorNodeOps, VectorOp, VectorOpPipeline};
        let op = VectorOp {
            pipeline: VectorOpPipeline::Analytic,
            color: [0.0, 0.0, 0.0, 1.0],
            opacity: 1.0,
            blur_px: 0.0,
            stroke_px: 0.0,
            additive: false,
            tris: vec![0.0_f32; tris_floats].into(),
            records: vec![
                VectorAnalyticRecord {
                    pen_x: 0.0,
                    baseline_y: 0.0,
                    font_size_px: 16.0,
                    em_box: [0.0, 0.0, 1.0, 1.0],
                    first_curve: 0,
                    curve_count: 1,
                    // KineticText per-glyph transform: identity.
                    glyph_lin: [1.0, 0.0, 0.0, 1.0],
                    glyph_off: [0.0, 0.0],
                    glyph_opacity: 1.0,
                    local_aa_px: 0.,
                };
                glyphs
            ].into(),
            gradient: None,
            paint: None,
        };
        crate::seams::TextQuadBatch {
            emoji_engine: None,
            node_id: node_id.to_string(),
            z: 1.0,
            model_matrix: [0.0; 16],
            page_clip_bounds: [0.0; 4],
            ops: Vec::new(),
            degraded: None,
            vector: Some(VectorNodeOps {
                uniform_padding: None, baked_padding_scale: None,
                ops: vec![op],
                curve_points: vec![0.0_f32; curve_floats].into(),
                gradient: None,
                emojis: Vec::new(),
            }),
            distort: None,
            content_key: None, paint_geometry_key: None,
        }
    }

    #[test]
    fn vector_batch_bytes_track_actual_payload_not_flat_estimate() {
        use moda_text::vector::ops::VectorAnalyticRecord;

        // A realistic short label ("hello world" ≈ 10 glyphs, ~7 unique glyphs ×
        // ~30 quad segments × 6 floats ≈ 1260 curve floats) must cost far less
        // than the old flat 64 KiB/node estimate — that estimate saturated the
        // 48 MB text LRU at ~750 nodes and thrashed dense decks.
        let small = batch_bytes(&vector_text_batch("a", 10, 1260, 0));
        assert!(
            small < 16 * 1024,
            "a short vector label should be well under the old 64 KiB flat estimate, got {small}"
        );

        // Cost must SCALE with content, not be flat: a paragraph costs more.
        let big = batch_bytes(&vector_text_batch("b", 200, 60_000, 0));
        assert!(
            big > small,
            "a paragraph ({big}) must cost more than a short label ({small})"
        );

        // Each variable payload is pinned independently at `len * size_of::<T>()`
        // so removing any one term is caught (curve points otherwise dominate).
        let base = batch_bytes(&vector_text_batch("c", 10, 1000, 0));
        let more_curves = batch_bytes(&vector_text_batch("c", 10, 2000, 0));
        assert_eq!(
            more_curves - base,
            1000 * std::mem::size_of::<f32>(),
            "curve-point delta"
        );
        let more_glyphs = batch_bytes(&vector_text_batch("c", 60, 1000, 0));
        assert_eq!(
            more_glyphs - base,
            50 * std::mem::size_of::<VectorAnalyticRecord>(),
            "glyph-record delta"
        );
        let more_tris = batch_bytes(&vector_text_batch("c", 10, 1000, 300));
        assert_eq!(
            more_tris - base,
            300 * std::mem::size_of::<f32>(),
            "tess-triangle delta"
        );
    }

    /// A multi-op vector batch: one `VectorOp` per entry in `op_glyphs`, each
    /// carrying that many instance records (no curves/tris). Lets a test pin the
    /// per-op term and prove the accounting sums ALL ops, not just the first.
    fn vector_text_batch_multi_op(
        node_id: &str,
        op_glyphs: &[usize],
    ) -> crate::seams::TextQuadBatch {
        use moda_text::vector::ops::VectorAnalyticRecord;
        use moda_text::vector::{VectorNodeOps, VectorOp, VectorOpPipeline};
        let rec = VectorAnalyticRecord {
            pen_x: 0.0,
            baseline_y: 0.0,
            font_size_px: 16.0,
            em_box: [0.0, 0.0, 1.0, 1.0],
            first_curve: 0,
            curve_count: 1,
            // KineticText per-glyph transform: identity.
            glyph_lin: [1.0, 0.0, 0.0, 1.0],
            glyph_off: [0.0, 0.0],
            glyph_opacity: 1.0,
            local_aa_px: 0.,
        };
        let ops = op_glyphs
            .iter()
            .map(|&n| VectorOp {
                pipeline: VectorOpPipeline::Analytic,
                color: [0.0, 0.0, 0.0, 1.0],
                opacity: 1.0,
                blur_px: 0.0,
                stroke_px: 0.0,
                additive: false,
                tris: Default::default(),
                records: vec![rec; n].into(),
                gradient: None,
                paint: None,
            })
            .collect();
        let mut batch = vector_text_batch(node_id, 0, 0, 0);
        batch.vector = Some(VectorNodeOps {
            uniform_padding: None, baked_padding_scale: None,
            ops,
            curve_points: Default::default(),
            gradient: None,
            emojis: Vec::new(),
        });
        batch
    }

    #[test]
    fn vector_batch_bytes_sums_all_ops_and_the_per_op_struct() {
        use moda_text::vector::VectorOp;
        // Same total records (60) and zero curves/tris, split 1-op vs 2-op. The
        // ONLY difference is one extra `VectorOp` struct — which pins that term
        // AND proves the loop sums every op (a "first op only" regression would
        // undercount the 2-op batch's records and break this equality).
        let one = batch_bytes(&vector_text_batch_multi_op("m", &[60]));
        let two = batch_bytes(&vector_text_batch_multi_op("m", &[30, 30]));
        assert_eq!(
            two - one,
            std::mem::size_of::<VectorOp>(),
            "second op adds exactly one VectorOp struct; records summed across ops"
        );
    }

    /// A Degraded-tier text batch owning a `w`×`h` RGBA raster.
    fn degraded_text_batch(id: &str, w: u32, h: u32) -> crate::seams::TextQuadBatch {
        let mut b = vector_text_batch(id, 0, 0, 0);
        b.vector = None;
        b.degraded = Some(crate::seams::DegradedQuad {
            raster: Arc::new(moda_text::degraded::DegradedRaster {
                rgba: vec![0; (w * h * 4) as usize],
                width: w,
                height: h,
                logical_w: w as f64 / 2.0,
                logical_h: h as f64 / 2.0,
                quality_scale: 2.0,
                pad_top: 0.0,
                pad_bottom: 0.0,
                fill_alpha: 1.0,
            }),
            offset: [0.0, 0.0],
            opacity: 1.0,
        });
        b
    }

    #[test]
    fn batch_bytes_counts_a_degraded_raster() {
        let b = degraded_text_batch("d", 200, 100);
        assert!(batch_bytes(&b) >= 200 * 100 * 4, "the raster is the batch's heap");
    }

    /// The text budget the eviction tests run at (the real one is larger; batches this size keep them light).
    const TEST_TEXT_BUDGET: usize = 48 * 1024 * 1024;

    #[test]
    fn text_lru_bounds_degraded_rasters_by_their_bytes() {
        // 16k texts planned before their fonts arrived (Degraded-tier rasters,
        // ~80 KB each) kept 1.3 GB in a cache that counted 4 MB. Counted, the
        // cache stays inside its budget.
        let mut caches = PlanCaches::default();
        caches.text_budget_bytes = TEST_TEXT_BUDGET;
        for i in 0..2_000u64 {
            let id = format!("n{i}");
            caches.text_put(&id, i, &degraded_text_batch(&id, 140, 140));
        }
        let held: usize = caches
            .text
            .values()
            .map(|e| e.batch.degraded.as_ref().map_or(0, |d| d.raster.rgba.len()))
            .sum();
        assert!(caches.text_bytes <= TEST_TEXT_BUDGET);
        assert!(held <= TEST_TEXT_BUDGET, "rasters held: {held}");
        assert!(caches.evictions > 0);
    }

    #[test]
    fn text_lru_holds_a_dense_richtext_deck_without_thrash() {
        // Regression guard for the intermittent-drag-jank fix: ~1k short vector
        // text nodes must fit the 48 MB text budget with zero evictions. The old
        // flat 64 KiB estimate evicted past ~750 nodes, forcing a periodic full
        // text re-prepare mid-drag (the ~1s frame spikes).
        let mut caches = PlanCaches::default();
        for i in 0..1024u64 {
            let id = format!("n{i}");
            let batch = vector_text_batch(&id, 12, 1500, 0);
            caches.text_put(&id, i, &batch);
        }
        assert_eq!(caches.text.len(), 1024, "all nodes should stay resident");
        assert_eq!(caches.evictions, 0, "no eviction thrash under the budget");
    }

    #[test]
    fn text_put_keeps_vector_text_bytes_exact_across_replacement() {
        // Replacing a cached vector batch must subtract the OLD size and add the
        // NEW one exactly — both directions. A bug in the subtract/replace path
        // would silently drift `text_bytes` from the retained heap and skew every
        // future eviction decision.
        let mut caches = PlanCaches::default();
        let big = vector_text_batch("n", 200, 60_000, 400);
        let small = vector_text_batch("n", 10, 1_000, 0);

        caches.text_put("n", 1, &big);
        assert_eq!(caches.text_bytes, batch_bytes(&big));

        caches.text_put("n", 2, &small); // larger → smaller
        assert_eq!(caches.text_bytes, batch_bytes(&small));
        assert_eq!(caches.text.len(), 1, "replacement keeps a single entry");

        caches.text_put("n", 3, &big); // smaller → larger
        assert_eq!(caches.text_bytes, batch_bytes(&big));
        assert_eq!(caches.evictions, 0, "a single replaced entry never evicts");
    }

    #[test]
    fn text_eviction_trims_to_the_low_water_mark() {
        // Crossing the budget trims to 7/8 of it (oldest first), so the next inserts don't rescan the cache: an
        // eviction per insert once text outgrew the budget made a frame O(n²) in text nodes.
        const CURVE_FLOATS: usize = 256 * 1024; // ~1 MiB per batch
        let mut caches = PlanCaches::default();
        caches.text_budget_bytes = TEST_TEXT_BUDGET;
        let mut first_trim = None;
        for i in 0..80u64 {
            caches.begin_frame();
            let id = format!("t{i}");
            let before = caches.evictions;
            caches.text_put(&id, i, &vector_text_batch(&id, 1, CURVE_FLOATS, 0));
            if caches.evictions > before && first_trim.is_none() {
                first_trim = Some(i);
                assert!(
                    caches.text_bytes <= (TEST_TEXT_BUDGET / 8 * 7),
                    "an over-budget insert trims to the low-water mark, got {}",
                    caches.text_bytes
                );
                // The next insert fits under the budget again: no rescan.
                let trimmed = caches.evictions;
                let id = format!("t{i}b");
                caches.text_put(&id, i, &vector_text_batch(&id, 1, CURVE_FLOATS, 0));
                assert_eq!(caches.evictions, trimmed, "no eviction right after a trim");
            }
            assert!(caches.text_bytes <= TEST_TEXT_BUDGET);
        }
        first_trim.expect("80 MiB of text crosses the 48 MiB budget");
        // The oldest entries went first; the newest is still cached.
        assert!(!caches.text.contains_key("t0"));
        assert!(caches.text.contains_key("t79"));
    }

    #[test]
    fn text_lru_evicts_when_aggregate_vector_bytes_exceed_budget() {
        // Eviction boundary (distinct from the 1k-small-node no-thrash case): the
        // AGGREGATE of a few vector batches crosses TEST_TEXT_BUDGET and evicts,
        // and `text_bytes` stays within budget afterward. Each batch is well under
        // budget individually (~28 MiB = 7M f32 curve points); two of them
        // (~56 MiB) cross the 48 MiB budget. Kept to two batches to bound peak
        // test memory (each `text_put` clones the batch into the cache).
        const CURVE_FLOATS: usize = 7_000_000;
        let mut caches = PlanCaches::default();
        caches.text_budget_bytes = TEST_TEXT_BUDGET;
        let per = batch_bytes(&vector_text_batch("x", 1, CURVE_FLOATS, 0));
        assert!(
            per < TEST_TEXT_BUDGET,
            "one batch must fit individually ({per})"
        );
        assert!(
            2 * per > TEST_TEXT_BUDGET,
            "two batches must exceed the budget"
        );
        for i in 0..2u64 {
            let id = format!("big{i}");
            caches.text_put(&id, i, &vector_text_batch(&id, 1, CURVE_FLOATS, 0));
        }
        assert!(
            caches.evictions >= 1,
            "aggregate over-budget puts must evict"
        );
        assert!(
            caches.text_bytes <= TEST_TEXT_BUDGET,
            "text_bytes must stay within budget after eviction, got {}",
            caches.text_bytes
        );
    }

    #[test]
    fn hash_is_deterministic() {
        let n = raw(
            json!({"id": "a", "type": "rectangle", "x": 1.0, "width": 10.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}]}),
        );
        assert_eq!(node_hashes(&n), node_hashes(&n));
        let m = raw(
            json!({"id": "a", "type": "rectangle", "x": 1.0, "width": 10.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}]}),
        );
        assert_eq!(node_hashes(&n), node_hashes(&m));
    }

    #[test]
    fn transform_only_props_do_not_change_geom_or_text_hash() {
        let a = raw(
            json!({"id": "a", "type": "rectangle", "x": 1.0, "y": 2.0, "rotation": 0.0, "scale": 1.0, "width": 10.0}),
        );
        let b = raw(
            json!({"id": "a", "type": "rectangle", "x": 99.0, "y": -4.0, "rotation": 45.0, "scale": 2.0, "width": 10.0}),
        );
        let (ha, hb) = (node_hashes(&a), node_hashes(&b));
        assert_eq!(
            ha.geom, hb.geom,
            "x/y/rotation/scale must not re-key tessellation"
        );
        assert_eq!(ha.text, hb.text);
        assert_ne!(
            ha.full, hb.full,
            "full hash must still see transforms (repaint identity)"
        );
    }

    #[test]
    fn paint_only_props_do_not_change_geom_hash_but_do_change_text_hash() {
        let a = raw(
            json!({"id": "a", "type": "rectangle", "width": 10.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}], "opacity": 1.0}),
        );
        let b = raw(
            json!({"id": "a", "type": "rectangle", "width": 10.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}], "opacity": 0.5}),
        );
        let (ha, hb) = (node_hashes(&a), node_hashes(&b));
        assert_eq!(
            ha.geom, hb.geom,
            "fill/opacity must not re-key tessellation"
        );
        assert_ne!(ha.text, hb.text, "paint IS baked into text quads");
        assert_ne!(ha.full, hb.full);
    }

    #[test]
    fn geometry_class_props_re_key() {
        // Stroke width (the browser's latent stale-geometry hazard — MAP §5.2),
        // dash, cap/join, fillRule, stroke align, width, points, type,
        // distort, the fill's paint type: ALL must change the geom hash.
        let rect = |extra: Value, stroke: Value| {
            let mut layer = json!({"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"}, "width": 1.0});
            for (k, v) in stroke.as_object().unwrap() {
                layer[k] = v.clone();
            }
            let mut n = json!({"id": "a", "type": "rectangle", "width": 10.0, "height": 5.0, "strokes": [layer]});
            for (k, v) in extra.as_object().unwrap() {
                n[k] = v.clone();
            }
            n
        };
        let none = json!({});
        let base = rect(none.clone(), none.clone());
        let variants = vec![
            rect(none.clone(), json!({"width": 3.0})),
            rect(none.clone(), json!({"dash": [5.0, 5.0]})),
            rect(none.clone(), json!({"cap": "round"})),
            rect(none.clone(), json!({"join": "bevel"})),
            rect(json!({"fillRule": "evenodd"}), none.clone()),
            rect(none.clone(), json!({"align": "inner"})),
            rect(json!({"width": 12.0}), none.clone()),
            rect(json!({"type": "ellipse"}), none.clone()),
            rect(
                json!({"distort": {"type": "matrix", "matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2, 0.0, 1.0]}}),
                none.clone(),
            ),
            rect(
                json!({"fills": [{"id": "fill-0", "paint": {"type": "image", "placement": {}}}]}),
                none.clone(),
            ),
            // Trim paths + dash offset are geometry inputs (baked into stroke
            // triangles), so they MUST re-key tessellation or animating them shows
            // stale frames. Guards the invariant that they stay out of the
            // TRANSFORM_ONLY / PAINT_ONLY deny-lists.
            rect(json!({"trimPathStart": 0.25}), none.clone()),
            rect(json!({"trimPathEnd": 0.75}), none.clone()),
            rect(json!({"trimPathOffset": 0.5}), none.clone()),
            rect(none.clone(), json!({"dashOffset": 3.0})),
        ];
        let h0 = node_hashes(&raw(base)).geom;
        for v in variants {
            assert_ne!(
                h0,
                node_hashes(&raw(v.clone())).geom,
                "geometry-class prop failed to re-key: {v}"
            );
        }
    }

    #[test]
    fn stroke_paint_route_presence_re_keys_geom_but_its_value_does_not() {
        // Stroke tessellation is gated on a stroke existing, so gaining/losing a
        // stroke layer MUST re-key the tess cache (else a path given its first
        // stroke keeps the cached band-less geometry), while a colour-to-colour
        // edit must NOT. The projection keeps the stroke's paint TYPE (a
        // gradient stroke routes differently), so a type swap re-keys too
        // (conservative; it used to key only on "has a route").
        let path = |paint: Value| {
            raw(json!({"id": "a", "type": "path", "points": [],
                "strokes": [{"id": "stroke-0", "paint": paint, "width": 2}]}))
        };
        let stops = json!([{"offset": 0, "color": "#ff0000"}, {"offset": 1, "color": "#0000ff"}]);
        let none = raw(json!({"id": "a", "type": "path", "points": []}));
        let red = path(json!({"type": "solid", "color": "#ff0000"}));
        let green = path(json!({"type": "solid", "color": "#00ff00"}));
        let gradient = path(
            json!({"type": "linear", "start": {"x": 0, "y": 0}, "end": {"x": 100, "y": 0},
            "stops": stops}),
        );
        let gradient2 = path(
            json!({"type": "linear", "start": {"x": 0, "y": 0}, "end": {"x": 100, "y": 100},
            "stops": [{"offset": 0, "color": "#00ff00"}, {"offset": 1, "color": "#000000"}]}),
        );
        assert_ne!(
            node_hashes(&none).geom,
            node_hashes(&red).geom,
            "gaining a stroke must re-key tessellation"
        );
        assert_ne!(
            node_hashes(&none).geom,
            node_hashes(&gradient).geom,
            "gaining a gradient stroke must re-key tessellation"
        );
        assert_eq!(
            node_hashes(&red).geom,
            node_hashes(&green).geom,
            "a stroke colour edit stays paint-only"
        );
        assert_ne!(
            node_hashes(&red).geom,
            node_hashes(&gradient).geom,
            "swapping a solid stroke for a gradient re-keys (the route changes)"
        );
        assert_eq!(
            node_hashes(&gradient).geom,
            node_hashes(&gradient2).geom,
            "a gradient's placement / stops edit stays paint-only"
        );
    }

    #[test]
    fn keyed_capture_filters_reuse_path_triangles() {
        for effect in ["filter-shadow", "layer-blur"] {
            let hashes = |value| node_hashes(&raw(json!({"id":"p", "type":"path", "effects":[
                {"type":effect, "radius":value, "blur":value, "offsetX":value}
            ]})));
            assert_eq!(hashes(1).geom, hashes(3).geom);
            assert_ne!(hashes(1).full, hashes(3).full);
        }
    }

    #[test]
    fn unknown_props_over_invalidate() {
        let a = raw(json!({"id": "a", "type": "rectangle", "width": 10.0}));
        let b = raw(json!({"id": "a", "type": "rectangle", "width": 10.0, "someFutureProp": 1.0}));
        assert_ne!(
            node_hashes(&a).geom,
            node_hashes(&b).geom,
            "unknown props must over-invalidate (deny-list law)"
        );
    }

    #[test]
    fn page_frame_key_sensitivity() {
        let opts = RenderOptions {
            scale: 2.0,
            transparent: false,
            sample_count: 4,
            text_masks_available: Some(true),
            exclude: vec!["x".to_string()],
            isolate: Vec::new(),
            ..Default::default()
        };
        let base = page_frame_key(7, 0, &opts);
        assert_eq!(base, page_frame_key(7, 0, &opts), "deterministic");
        assert_ne!(base, page_frame_key(8, 0, &opts), "generation");
        assert_ne!(base, page_frame_key(7, 1, &opts), "page index");
        let key = |o: &RenderOptions| page_frame_key(7, 0, o);
        let mut o = opts.clone();
        o.scale = 1.5;
        assert_ne!(base, key(&o), "scale/dpr");
        let mut o = opts.clone();
        o.transparent = true;
        assert_ne!(base, key(&o), "transparent");
        let mut o = opts.clone();
        o.sample_count = 1;
        assert_ne!(base, key(&o), "sample_count");
        let mut o = opts.clone();
        o.text_masks_available = None;
        assert_eq!(base, key(&o), "text masks default to available");
        o.text_masks_available = Some(false);
        assert_ne!(base, key(&o), "text masks disabled");
        let mut o = opts.clone();
        o.exclude = vec!["y".to_string()];
        assert_ne!(base, key(&o), "exclude");
        // `isolate` is plan-relevant exactly as `exclude` is: a cached plan
        // must never be served across two different per-element isolations.
        let mut o = opts.clone();
        o.isolate = vec!["y".to_string()];
        assert_ne!(base, key(&o), "isolate");
        // ...and an `isolate` list must not collide with the SAME ids as an
        // `exclude` list (opposite filters, opposite plans).
        let mut a = opts.clone();
        a.exclude = vec!["z".to_string()];
        let mut b = opts.clone();
        b.isolate = vec!["z".to_string()];
        assert_ne!(key(&a), key(&b), "exclude vs isolate must not collide");
        // Per-id length prefixes: without them the ids concatenate and two
        // different isolations fold to the same key, so one's cached plan
        // would be served for the other's render.
        let mut c = opts.clone();
        c.isolate = vec!["ab".to_string(), "c".to_string()];
        let mut d = opts.clone();
        d.isolate = vec!["a".to_string(), "bc".to_string()];
        assert_ne!(key(&c), key(&d), "isolate id boundaries must be hashed");
        // Rendered masks track their clock separately.
        let mut o = opts.clone();
        o.shader_time_s = 9.0;
        assert_eq!(base, key(&o), "shader_time must NOT move the key");
    }

    #[test]
    fn text_layout_props_change_text_hash() {
        let a = raw(
            json!({"id": "t", "type": "richtext", "htmlContent": "<p>hi</p>", "width": 100.0, "height": 40.0}),
        );
        for v in [
            json!({"id": "t", "type": "richtext", "htmlContent": "<p>ho</p>", "width": 100.0, "height": 40.0}),
            json!({"id": "t", "type": "richtext", "htmlContent": "<p>hi</p>", "width": 80.0, "height": 40.0}),
            json!({"id": "t", "type": "richtext", "htmlContent": "<p>hi</p>", "width": 100.0, "height": 30.0}),
            json!({"id": "t", "type": "richtext", "htmlContent": "<p>hi</p>", "width": 100.0, "height": 40.0, "verticalAlign": "bottom"}),
            json!({"id": "t", "type": "richtext", "htmlContent": "<p>hi</p>", "width": 100.0, "height": 40.0, "paragraphSpacing": 8.0}),
        ] {
            assert_ne!(node_hashes(&a).text, node_hashes(&raw(v)).text);
        }
    }

    // -----------------------------------------------------------------------
    // rust-edit-incremental-z4: incremental plan patching correctness (CPU, no
    // GPU) — the fast oracle complementing the mutation_replay byte-exact gate.
    // -----------------------------------------------------------------------

    fn editor_opts() -> RenderOptions {
        RenderOptions {
            preserve_page_fill: false,
            scale: 1.0,
            transparent: false,
            sample_count: 4,
            shader_time_s: 0.0,
            exclude: Vec::new(),
            isolate: Vec::new(),
            text_masks_available: Some(true),
            working: None,
        }
    }

    fn rect_scene(n: usize) -> String {
        let nodes: Vec<Value> = (0..n)
            .map(|i| {
                json!({
                    "id": format!("r{i}"), "type": "rectangle",
                    "x": (i * 13 % 700) as f64, "y": (i * 7 % 500) as f64,
                    "width": 40.0, "height": 30.0, "rotation": 0, "scale": 1,
                    "cornerRadius": 3, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}]
                })
            })
            .collect();
        let ids: Vec<String> = (0..n).map(|i| format!("r{i}")).collect();
        moda_scene::value::to_string(&json!({
            "nodes": nodes,
            "pages": [{"id": "page-1", "name": "p", "width": 800.0, "height": 600.0,
                       "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}], "nodeIds": ids}],
            "config": {"renderPageClipping": true},
            "canvasRootNodeIds": ids, "layoutMode": "freeform", "variables": [],
        }))
        .unwrap()
    }

    /// Like `rect_scene` but every rect carries a visible stroke on a SHARP rect,
    /// so each body is tessellation-routed (routes only CURVED solid
    /// strokes to the SDF ring; a sharp-rect stroke still tessellates) — used where
    /// a test needs an actual `DrawItem::Body` carrying cacheable tessellated
    /// geometry (plain solid rects are analytic SDF bodies and never hit the tess
    /// cache).
    fn stroked_rect_scene(n: usize) -> String {
        // SHARP rects (no corner radius): a solid stroke on a sharp rect stays on
        // tessellation (routes only CURVED stroked bodies to the SDF
        // ring), so these give the test a `DrawItem::Body` with tess geometry.
        let nodes: Vec<Value> = (0..n)
            .map(|i| {
                json!({
                    "id": format!("r{i}"), "type": "rectangle",
                    "x": (i * 13 % 700) as f64, "y": (i * 7 % 500) as f64,
                    "width": 40.0, "height": 30.0, "rotation": 0, "scale": 1,
                    "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}],
                    "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#222222"}, "width": 2}]
                })
            })
            .collect();
        let ids: Vec<String> = (0..n).map(|i| format!("r{i}")).collect();
        moda_scene::value::to_string(&json!({
            "nodes": nodes,
            "pages": [{"id": "page-1", "name": "p", "width": 800.0, "height": 600.0,
                       "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}], "nodeIds": ids}],
            "config": {"renderPageClipping": true},
            "canvasRootNodeIds": ids, "layoutMode": "freeform", "variables": [],
        }))
        .unwrap()
    }

    #[test]
    fn row_hashes_follow_writes_and_leave_unread_rows_empty() {
        let value: Value = moda_scene::value::from_str(&rect_scene(3)).unwrap();
        let mut scene = Scene::load(moda_scene::extract_canvas_data(&value)).unwrap();
        let hashes = RowHashes::default();
        for id in ["r0", "r1"] {
            assert_eq!(hashes.read_hash(&scene, id), Some(node_hashes_in(&scene, scene.effective_node(id).unwrap())));
        }
        let first = scene.row_of("r0").unwrap();
        let second = scene.row_of("r1").unwrap();
        let unread = scene.row_of("r2").unwrap();
        let old = hashes.read_hash(&scene, "r0");
        scene.set_node_xy("r0", 83., 14.);
        let rows = hashes.rows.get().unwrap();
        assert_eq!(rows[first as usize].lock().unwrap().unwrap().1, old.unwrap());
        assert!(rows[second as usize].lock().unwrap().is_some());
        assert!(rows[unread as usize].lock().unwrap().is_none());
        let updated = hashes.read_hash(&scene, "r0");
        assert_ne!(updated, old);
        assert_eq!(updated, Some(node_hashes_in(&scene, scene.effective_node("r0").unwrap())));
    }

    #[test]
    fn cold_dl_dedup_is_byte_identical() {
        let opts = editor_opts();
        let value: Value = moda_scene::value::from_str(&rect_scene(30)).unwrap();
        let scene = Scene::load(moda_scene::extract_canvas_data(&value)).unwrap();
        let hashes = node_hashes_for_scene(&scene);
        let page_id = scene.doc().pages[0].id().to_string();
        let dl = moda_scene::build_display_list(&scene, &page_id);

        // Plan: build with the shared DL vs let the builder build its own.
        let (without_index, _) = crate::frame::build_coverage_plan_cached_with_dl(
            &scene,
            0,
            &opts,
            Some((&mut PlanCaches::default(), &hashes)),
            Some(dl.clone()),
        )
        .unwrap();
        let (with_dl, _w, _aux) = crate::frame::build_coverage_plan_indexed_with_dl(
            &scene,
            0,
            &opts,
            Some((&mut PlanCaches::default(), &hashes)),
            Some(dl),
        )
        .unwrap();
        let (without_dl, _w, _aux) = crate::frame::build_coverage_plan_indexed(
            &scene,
            0,
            &opts,
            Some((&mut PlanCaches::default(), &hashes)),
        )
        .unwrap();
        assert_eq!(
            plan_items_fingerprint(&with_dl),
            plan_items_fingerprint(&without_dl),
            "prebuilt-DL plan diverged from a self-built-DL plan"
        );
        assert_eq!(
            plan_items_fingerprint(&without_index),
            plan_items_fingerprint(&with_dl),
            "omitting the unused plan index changed the plan"
        );
    }

    /// renderer-perf-more-z6 (review gap 5.3): a tess-cache HIT must hand the plan
    /// the SAME geometry `Arc` (pointer-equal), not a deep copy — that sharing is
    /// the whole point of the Arc-geometry change (Experiment D). Two builds over a
    /// warm cache should yield ptr-equal body geometry; a regression to deep copies
    /// would fail here while still passing every byte-exact test.
    ///
    /// Uses `stroked_rect_scene`: plain solid rects (and now solid-stroked CURVED
    /// rects) are analytic SDF bodies that never populate the tess
    /// cache, so a SHARP stroked rect — which stays on tessellation — is what
    /// gives this test a `DrawItem::Body` to share.
    #[test]
    fn tess_cache_shares_geometry_arc() {
        use crate::seams::DrawItem;
        let opts = editor_opts();
        let value: Value = moda_scene::value::from_str(&stroked_rect_scene(8)).unwrap();
        let scene = Scene::load(moda_scene::extract_canvas_data(&value)).unwrap();
        let hashes = node_hashes_for_scene(&scene);
        let mut caches = PlanCaches::default();

        // First build populates the tess cache; second build hits it.
        let (p1, _w, _a) = crate::frame::build_coverage_plan_indexed(
            &scene,
            0,
            &opts,
            Some((&mut caches, &hashes)),
        )
        .unwrap();
        let (p2, _w, _a) = crate::frame::build_coverage_plan_indexed(
            &scene,
            0,
            &opts,
            Some((&mut caches, &hashes)),
        )
        .unwrap();

        let first_body_geom = |plan: &CoveragePlan| {
            plan.items.iter().find_map(|it| match it {
                DrawItem::Body { geometry, .. } => Some(geometry.clone()),
                _ => None,
            })
        };
        let g1 = first_body_geom(&p1).expect("a body in plan 1");
        let g2 = first_body_geom(&p2).expect("a body in plan 2");
        assert!(
            std::sync::Arc::ptr_eq(&g1, &g2),
            "a tess-cache hit must share the geometry Arc with the plan (no deep copy)"
        );
    }

    // -----------------------------------------------------------------------
    // The `pages` branch of `apply_props`.
    //
    // A page background is DERIVED, not stored: `compute_resolved_nodes`
    // synthesizes a transient `__page_bg_<pageId>` rect from the page's own
    // props and puts it first in the page's z-order. Previously the pages
    // branch wrote the page props and stopped, so the engine held new page props
    // and the OLD background quad — a stale background on the snapshot lane.
    //
    // Every test below asserts on BOTH surfaces the rest of the engine reads:
    // `scene.effective_node("__page_bg_<pageId>")` (what the plan builder paints)
    // and the entry in `hashes` (what the plan/tessellation caches key on). A fix
    // that updates one without the other serves stale pixels from cache.
    // -----------------------------------------------------------------------

}

#[cfg(test)]
mod unit_placement_tests {
    use super::*;
    use moda_scene::value::json;

    fn path(id: &str, x: f64, y: f64, color: &str) -> Value {
        json!({"id": id, "type": "path", "x": x, "y": y, "width": 70, "height": 60, "closed": true,
            "points": [{"type":"move","x":0,"y":0},{"type":"line","x":70,"y":0},{"type":"line","x":70,"y":60},{"type":"line","x":0,"y":60}],
            "fills": [{"id":"fill","paint":{"type":"solid","color":color}}]})
    }

    struct Host { resident: ResidentScene, caches: PlanCaches, index: FramePlanIndex, opts: RenderOptions, generation: u64 }
    impl Host {
        fn new(nodes: Vec<Value>, roots: &[&str]) -> Self {
            let source = json!({"nodes": nodes, "pages": [{"id":"page","width":320,"height":240,"nodeIds":roots,
                "fills":[{"id":"fill","paint":{"type":"solid","color":"#eeeeee"}}]}],
                "config":{"renderPageClipping":true},"layoutMode":"freeform","canvasRootNodeIds":[]});
            Self::from_source(source)
        }
        fn from_source(source: Value) -> Self {
            let resident = ResidentScene::from_scene(Scene::load(&source).unwrap());
            Self { resident, caches: Default::default(), index: Default::default(),
                opts: RenderOptions { sample_count: 4, transparent: false, ..Default::default() }, generation: 0 }
        }
        fn step(&mut self, id: Option<(&str, f64)>) -> (Arc<CoveragePlan>, bool) {
            let mut ids = Vec::new();
            if let Some((id, x)) = id {
                let state = self.resident.state.as_mut().unwrap();
                let row = state.scene.row_of(id).unwrap();
                state.scene.write_row(row, |fields| fields.set("x", json!(x))).unwrap();
                state.hashes = node_hashes_for_scene(&state.scene);
                ids.push(id.to_string());
            }
            self.generation += 1;
            self.caches.begin_frame();
            // A host keeps the scene's world columns current, as a canvas does before it plans.
            self.resident.state.as_mut().unwrap().scene.refresh_worlds();
            let scene = self.resident.scene().unwrap();
            if !ids.is_empty() {
                let rows: Vec<u32> = ids.iter().filter_map(|id| scene.row_of(id)).collect();
                if self.index.place(scene, &self.opts, self.generation, self.generation, &rows, &mut self.caches).is_some() {
                    let plan = self.index.entry.as_ref().unwrap().plan.clone();
                    self.index.take_written();
                    return (plan, true);
                }
            }
            let (plan, warnings, aux) = crate::frame::build_coverage_plan_indexed(scene, 0, &self.opts,
                Some((&mut self.caches, self.resident.hashes().unwrap()))).unwrap();
            let (plan, warnings) = (Arc::new(plan), Arc::new(warnings));
            self.index.seed(scene, &self.opts, &plan, &warnings, aux, self.generation, self.generation, &mut self.caches);
            (plan, false)
        }
        fn fresh(&self) -> String {
            let mut caches = PlanCaches::default();
            let (plan, ..) = crate::frame::build_coverage_plan_indexed(self.resident.scene().unwrap(), 0, &self.opts,
                Some((&mut caches, self.resident.hashes().unwrap()))).unwrap();
            format!("{plan:?}")
        }
    }

    impl Host {
        /// Drive the public edit / miss / reseed sequence and compare the complete
        /// logical plan and warnings with an independent collector after each edit.
        fn edit_and_compare(&mut self, id: &str, field: &str, value: Value) -> (bool, usize) {
            let state = self.resident.state.as_mut().unwrap();
            let row = state.scene.row_of(id).unwrap();
            state.scene.write_row(row, |f| f.set(field, value)).unwrap();
            state.scene.refresh_worlds();
            state.hashes = node_hashes_for_scene(&state.scene);
            self.generation += 1;
            self.caches.begin_frame();
            let at = self.generation;
            let patch = self.index.update(&state.scene, &state.hashes, &self.opts, at, at, &[row], false, &mut self.caches);
            let hit = patch.is_some();
            let (full, warnings) = match patch {
                Some((p, w)) => (self.index.materialize(&p, &state.scene, &state.hashes, &self.opts, &mut self.caches).expect("valid restored plan"), w),
                None => {
                    let (p, w, aux) = crate::frame::build_coverage_plan_indexed(&state.scene, 0, &self.opts, Some((&mut self.caches, &state.hashes))).unwrap();
                    let (p, w) = (Arc::new(p), Arc::new(w));
                    self.index.seed(&state.scene, &self.opts, &p, &w, aux, at, at, &mut self.caches);
                    (p, w)
                }
            };
            let (fresh, expected, fresh_aux) = crate::frame::build_coverage_plan_indexed(&state.scene, 0, &self.opts,
                Some((&mut PlanCaches::default(), &state.hashes))).unwrap();
            let expected_setup = fresh_aux.page_effects.setup(&self.opts);
            if let Some(patchable) = self.index.entry.as_ref().and_then(|e| e.patchable.as_ref()) {
                assert_eq!(patchable.page_effects.setup(&self.opts), expected_setup, "{id}.{field}");
                let contexts = |segment: Option<&crate::frame::CameraSegment>| segment.map_or((None, false),
                    |s| (s.backdrop, s.layer_blur));
                assert_eq!(contexts(patchable.units.as_ref().and_then(|u| u.segment.as_ref())),
                    contexts(fresh_aux.segment.as_ref()), "emitter contexts after {id}.{field}");
            }
            assert_eq!((full.transparent, full.sample_count), (fresh.transparent, fresh.sample_count), "{id}.{field}");
            let splits = full.items.iter().filter(|i| matches!(i, crate::seams::DrawItem::BackdropSplit(_))).count();
            assert_eq!(full.masks, fresh.masks, "mask bindings after {id}.{field}");
            assert_eq!(format!("{warnings:?}"), format!("{expected:?}"), "edit {at}: {id}.{field}");
            assert_eq!(format!("{full:?}"), format!("{fresh:?}"), "edit {at}: {id}.{field}");
            let writes = self.index.take_written();
            let mut touched: Vec<_> = writes.placed.iter().map(|w| w.at).chain(writes.replaced.iter().map(|(at, _)| *at)).collect();
            self.index.finish_draw(&state.scene, &mut touched, None);
            (hit, splits)
        }
    }

    #[test]
    fn publication_after_content_edit_restores_only_that_assets_evicted_consumers() {
        // Installing a store advances the process-wide resource generation and would
        // invalidate plans being exercised concurrently by other tests.
        const CHILD: &str = "MODA_TEST_EVICTED_PUBLICATION";
        if std::env::var_os(CHILD).is_none() {
            let name = std::thread::current().name().unwrap().to_owned();
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", &name]).env(CHILD, "1").status().unwrap();
            assert!(status.success());
            return;
        }
        for move_consumers in [false, true] {
            check_evicted_publication(move_consumers);
        }
        check_compact_unit_publications();
    }

    fn check_compact_unit_publications() {
        use moda_assets::{engine_seam, AssetEntry, AssetStore, PreparedImage};
        let mut store = AssetStore::default();
        let mut publish = |value| {
            let image = PreparedImage::from_premultiplied(vec![value, value, value, 255], 1, 1);
            store.insert("live.png".into(), AssetEntry::Loaded(Arc::new(image)));
            engine_seam::install_store(Arc::new(store.clone()));
        };
        publish(20);
        let mut body = path("body", 0., 0., "#fff");
        body["fills"] = json!([{"id":"image","paint":{"type":"image","src":"live.png"}}]);
        let mut host = Host::new(vec![body, json!({"id":"group","type":"group","opacity":0.75,"children":["body"]})], &["group"]);
        host.step(None);
        let state = host.resident.state.as_ref().unwrap();
        let entry = host.index.entry.as_mut().unwrap();
        assert!(entry.fit_cold_units(&state.scene, usize::MAX));
        let resident = entry.plan.items.len();
        let budget = entry.bytes;
        host.index.test_budget = Some(budget);
        for revision in 2..5 {
            publish(revision as u8);
            let (view, _) = host.index.update(&state.scene, &state.hashes, &host.opts, 1, revision, &[], false,
                &mut host.caches).expect("publication restores the compact range");
            assert!(view.items.len() > resident);
            let full = host.index.materialize(&view, &state.scene, &state.hashes, &host.opts, &mut host.caches).unwrap();
            let fresh = crate::frame::build_coverage_plan_cached(&state.scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &state.hashes))).unwrap().0;
            // Restored image inputs can have an assigned GPU texture slot.
            let texture = |p: &CoveragePlan| p.items.iter().find_map(|item| match item {
                crate::seams::DrawItem::Body { paint: crate::seams::BodyPaint::Pattern(pattern), .. } =>
                    Some(pattern.texture.image.rgba_premultiplied.clone()),
                _ => None,
            }).unwrap();
            assert_eq!(texture(&full), texture(&fresh));
            assert!(host.index.finish_draw(&state.scene, &mut [], Some(crate::frame::PacketStorage {
                destinations: 0, recording: 0,
            })));
            assert_eq!(host.index.retained_size().1, resident);
        }
    }

    fn check_evicted_publication(move_consumers: bool) {
        use moda_assets::{engine_seam, AssetEntry, AssetStore, PreparedImage};
        let image = |v| AssetEntry::Loaded(Arc::new(PreparedImage::from_premultiplied(vec![v, v, v, 255], 1, 1)));
        let mut store = AssetStore::default();
        store.insert("first.png".into(), image(12));
        store.insert("other.png".into(), image(20));
        engine_seam::install_store(Arc::new(store.clone()));
        let observed = |key| {
            let scope = engine_seam::CaptureReads::begin();
            assert!(engine_seam::resolve(key).is_some());
            Arc::new(crate::seams::MaskObservation {
                reads: std::sync::Mutex::new(Some(Arc::new(scope.finish().unwrap()))),
                ..Default::default()
            })
        };
        let mut deferred = PublicationIndex::default();
        deferred.capture_masks.insert(3, vec![observed("first.png")]);
        deferred.capture_masks.insert(4, vec![observed("other.png")]);
        let ids: Vec<_> = (0..512).map(|i| format!("p{i}")).collect();
        let mut nodes: Vec<_> = ids.iter().map(|id| path(id, 2., 3., "#3070e0")).collect();
        for (i, key) in [(510, "first.png"), (511, "other.png")] {
            nodes[i]["fills"] = json!([{"id":"fill","paint":{"type":"image","src":key}}]);
        }
        let roots: Vec<_> = ids.iter().map(String::as_str).collect();
        let mut host = Host::new(nodes, &roots);
        host.index.test_budget = Some(94 * 1024);
        host.step(None);
        assert!(host.edit_and_compare("p0", "fills", json!([{"id":"fill","paint":{"type":"solid","color":"#2200ff"}}])).0);
        let old = host.index.retained_size();
        let entry = host.index.entry.as_ref().unwrap();
        let parts = entry.parts.as_ref().unwrap();
        let by_node = &entry.patchable.as_ref().unwrap().index.by_node;
        let slots = [510, 511].map(|i| parts.start(by_node.get(host.resident.scene().unwrap(), &ids[i]).unwrap()));
        for slot in slots { assert!(parts.addresses.physical(slot).is_none()); }
        store.insert("first.png".into(), image(220));
        assert_eq!(deferred.changed(&store), [3], "only the capture observing the changed publication needs its source");
        engine_seam::install_store(Arc::new(store));
        let state = host.resident.state.as_mut().unwrap();
        let mut moved = Vec::new();
        if move_consumers {
            for id in &ids[510..] {
                let row = state.scene.row_of(id).unwrap();
                state.scene.write_row(row, |fields| fields.set("x", json!(27))).unwrap();
                moved.push(row);
            }
            state.scene.refresh_worlds();
            host.generation += 1;
        }
        let key = page_frame_key(host.generation, 0, &host.opts);
        let (p, _) = host.index.update(&state.scene, &state.hashes, &host.opts, host.generation, key, &moved, move_consumers, &mut host.caches).expect("publication survives content patch");
        let addresses = host.index.addresses(&p).unwrap();
        assert!(addresses.physical(slots[0]).is_some());
        assert!(addresses.physical(slots[1]).is_none());
        assert_eq!(host.index.retained_size().1, old.1 + 1);
        if move_consumers {
            let writes = host.index.take_written();
            assert_eq!(writes.evicted.iter().map(|w| w.logical).collect::<Vec<_>>(), [slots[1]]);
        }
        let full = host.index.materialize(&p, &state.scene, &state.hashes, &host.opts, &mut host.caches).unwrap();
        assert_eq!(format!("{full:?}"), host.fresh());
    }

    #[test]
    fn mixed_row_and_movie_publications_match_fresh_collection() {
        use moda_assets::{engine_seam, AssetEntry, AssetStore, PreparedImage};
        // The installed store is process-wide. A child isolates the publications from other tests.
        const CHILD: &str = "MODA_TEST_MIXED_PUBLICATIONS";
        if std::env::var_os(CHILD).is_none() {
            let name = std::thread::current().name().unwrap().to_owned();
            assert!(std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", &name]).env(CHILD, "1").status().unwrap().success());
            return;
        }
        let mut seed = 92715u32;
        let mut next = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); seed >> 16 };
        let keys = ["a.mp4", "b.mp4"];
        let movie = |n| AssetEntry::Loaded(Arc::new(PreparedImage::from_premultiplied(vec![n, 24, 36, 255], 1, 1)));
        let mut store = AssetStore::default();
        for key in keys { store.insert(key.into(), movie(0)); }
        engine_seam::install_store(Arc::new(store.clone()));
        let image = |id, key| {
            let mut node = path(id, 10., 10., "#ff0000");
            node["fills"] = json!([{"id":"f","paint":{"type":"image","src":key}}]);
            node
        };
        let ids = ["movie", "other", "marker", "still"];
        let mut capture = image("filtered", keys[1]);
        capture["effects"] = json!([{"id":"blur","type":"layer-blur","radius":3}]);
        let padding: Vec<_> = (0..512).map(|i| format!("p{i}")).collect();
        let mut nodes: Vec<_> = padding.iter().map(|id| path(id, 1., 2., "#55aa88")).collect();
        nodes.extend([image(ids[0], keys[0]), image(ids[1], keys[1]), path(ids[2], 10., 10., "#abcdef"),
            path(ids[3], 5., 5., "#55aa88"), capture,
            json!({"id":"clip","type":"group","children":["mask","child"],"clipMaskNodeId":"mask"}),
            image("mask", keys[0]), path("child", 2., 3., "#ee3322")]);
        let roots: Vec<_> = padding.iter().map(String::as_str).chain(ids).chain(["filtered", "clip"]).collect();
        let mut host = Host::new(nodes, &roots);
        let (initial, _) = host.step(None);
        host.index.test_budget = Some(weights::Allocations::default().masks(&initial.masks) + 112 * 1024);
        host.index.finish_draw(host.resident.scene().unwrap(), &mut [], None);
        // The executor resolves this movie while building the isolated mask. Publish
        // its observed read and clock proof before planning the subsequent frame.
        let mask_picture = |plan: &CoveragePlan, retained| {
            let mask = plan.masks.iter().find(|m| m.source_node_id == "mask").unwrap();
            let rendered = mask.image.as_ref().unwrap().rendered.as_ref().unwrap();
            let reads = engine_seam::CaptureReads::begin();
            let picture = rendered.prepare(mask, 1.0, None, 0.0, retained).unwrap();
            assert!(picture.items.iter().any(|item| matches!(item, crate::seams::DrawItem::Body {
                paint: crate::seams::BodyPaint::Pattern(p), .. } if p.texture.key == keys[0])));
            *rendered.asset_reads.get_or_init(Default::default).reads.lock().unwrap() = Some(Arc::new(reads.finish().unwrap()));
            let _ = rendered.shader_clock.set(plan_uses_shader_time(&picture));
            format!("{picture:?}")
        };
        mask_picture(&initial, true);
        assert!(initial.items.iter().any(|i| matches!(i, crate::seams::DrawItem::LayerBlurComposite { .. })));
        drop(initial);
        let mut widths = [1, 1];
        let mut hits = 0;
        let mut dirty_masks = 0;
        let mut filtered = true;
        for step in 0..96 {
            let (_, resident, total) = host.index.retained_size();
            assert!(resident > 0 && resident < total, "step {step} starts with evicted parts");
            let row = next() % 4;
            let which = (next() % 2) as usize;
            let key = keys[which];
            let layout_change = step % 13 == 5;
            if layout_change { widths[which] = 3 - widths[which]; }
            let rgba = [next() as u8, 24, 36, 255].repeat(widths[which]);
            store.insert(key.into(), AssetEntry::Loaded(Arc::new(PreparedImage::from_premultiplied(rgba, widths[which] as u32, 1))));
            engine_seam::install_store(Arc::new(store.clone()));
            let state = host.resident.state.as_mut().unwrap();
            let change_capture = step % 17 == 6;
            let id = if change_capture || row == 3 { "filtered" } else { ids[row as usize] };
            let at = state.scene.row_of(id).unwrap();
            let placement = !change_capture && step % 4 != 0;
            state.scene.write_row(at, |f| if change_capture {
                filtered = !filtered;
                f.set("effects", if filtered { json!([{"id":"blur","type":"layer-blur","radius":3}]) } else { json!([]) });
            } else if placement {
                f.set("x", json!(next() % 200));
            } else if row == 2 {
                f.set("fills", json!([{"id":"fill","paint":{"type":"solid","color":format!("#{:06x}", next())}}]));
            } else {
                f.set("fills", json!([{"id":"f","paint":{"type":"image","src":key}}]));
            }).unwrap();
            state.scene.refresh_worlds();
            state.hashes = node_hashes_for_scene(&state.scene);
            host.generation += 1;
            host.caches.begin_frame();
            let frame = page_frame_key(host.generation, 0, &host.opts);
            let updated = host.index.update(&state.scene, &state.hashes, &host.opts, host.generation,
                frame, &[at], placement, &mut host.caches);
            assert!(!layout_change || updated.is_none(), "a movie resize must collect its new layout");
            let (plan, warnings) = match updated {
                Some((view, warnings)) => {
                    hits += 1;
                    if key == keys[0] {
                        assert!(host.index.written.masks_changed, "movie-fed mask at step {step}");
                        dirty_masks += 1;
                    }
                    (host.index.materialize(&view, &state.scene, &state.hashes, &host.opts, &mut host.caches).unwrap(), warnings)
                }
                None => {
                    let (plan, warnings, aux) = crate::frame::build_coverage_plan_indexed(&state.scene, 0, &host.opts,
                        Some((&mut host.caches, &state.hashes))).unwrap();
                    let (plan, warnings) = (Arc::new(plan), Arc::new(warnings));
                    host.index.seed(&state.scene, &host.opts, &plan, &warnings, aux, host.generation, frame, &mut host.caches);
                    (plan, warnings)
                }
            };
            let (fresh, expected, _) = crate::frame::build_coverage_plan_indexed(&state.scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &state.hashes))).unwrap();
            assert_eq!(format!("{warnings:?}"), format!("{expected:?}"));
            assert_eq!(plan.items.iter().any(|item| matches!(item, crate::seams::DrawItem::LayerBlurComposite { node_id, .. }
                if node_id == "filtered")), filtered, "capture layout at step {step}");
            assert_eq!(mask_picture(&plan, true), mask_picture(&fresh, false), "mask pixels at step {step}");
            assert_eq!(plan.masks, fresh.masks, "mask placement and sources at step {step}");
            // A deferred mask may keep an earlier snapshot of unrelated page rows.
            // Its isolated picture above and the page's ordered draws are observable.
            assert_eq!(format!("{:?}", plan.ordered_items().collect::<Vec<_>>()),
                format!("{:?}", fresh.ordered_items().collect::<Vec<_>>()), "page at step {step}");
            drop(plan);
            let writes = host.index.take_written();
            let mut touched: Vec<_> = writes.placed.iter().map(|w| w.at)
                .chain(writes.replaced.iter().map(|(at, _)| *at)).collect();
            host.index.finish_draw(&state.scene, &mut touched, None);
        }
        assert!(hits > 48 && dirty_masks > 10, "{hits} retained updates, {dirty_masks} mask refreshes");
    }

    #[test]
    fn unsupported_middle_publication_row_recollects_the_frame() {
        use moda_assets::{engine_seam, AssetEntry, AssetStore, PreparedImage};
        use crate::seams::{BodyPaint, DrawItem};
        const CHILD: &str = "MODA_TEST_MIDDLE_PUBLICATION";
        if std::env::var_os(CHILD).is_none() {
            let name = std::thread::current().name().unwrap().to_owned();
            assert!(std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", &name]).env(CHILD, "1").status().unwrap().success());
            return;
        }
        let mut store = AssetStore::default();
        let image = |v| AssetEntry::Loaded(Arc::new(PreparedImage::from_premultiplied(vec![v, 0, 0, 255], 1, 1)));
        store.insert("movie".into(), image(12));
        engine_seam::install_store(Arc::new(store.clone()));
        crate::text_draw::install_text_builder(Some(std::rc::Rc::new(|node, model, z, clip, _| {
            let mut batch = super::tests::vector_text_batch(node.id(), 1, 6, 0);
            batch.model_matrix = model;
            batch.z = z;
            batch.page_clip_bounds = clip;
            batch.paint_geometry_key = (node.str("htmlContent") == Some("prepared")).then_some([1; 4]);
            Ok(crate::text_draw::TextBuildOutcome::Item(Box::new(DrawItem::Text(Arc::new(batch)))))
        })));
        let mut tail = path("last", 1., 2., "#eeeeee");
        tail["fills"] = json!([{"id":"f","paint":{"type":"image","src":"movie"}}]);
        let mut host = Host::new(vec![path("first", 2., 3., "#553322"),
            json!({"id":"middle","type":"richtext","htmlContent":"prepared","width":70,"height":60}), tail],
            &["first","middle","last"]);
        host.step(None);
        assert!(host.index.entry.as_ref().unwrap().pattern_slots.is_some());
        let state = host.resident.state.as_mut().unwrap();
        store.insert("movie".into(), image(237));
        engine_seam::install_store(Arc::new(store));
        let key = page_frame_key(1, 0, &host.opts);
        host.index.update(&state.scene, &state.hashes, &host.opts, 1, key, &[], false, &mut host.caches).unwrap();
        let mut rows = Vec::new();
        for id in ["first", "middle", "last"] {
            let row = state.scene.row_of(id).unwrap();
            rows.push(row);
            state.scene.write_row(row, |f| if id == "middle" { f.set("htmlContent", json!("unindexed")); }
                else { f.set("width", json!(85)); }).unwrap();
        }
        state.scene.refresh_worlds();
        state.hashes = node_hashes_for_scene(&state.scene);
        let key = page_frame_key(2, 0, &host.opts);
        assert!(host.index.update(&state.scene, &state.hashes, &host.opts, 2, key, &rows, false, &mut host.caches).is_none());
        let writes = host.index.take_written();
        let ids: Vec<_> = writes.replaced.iter().map(|(_, old)| drawitem_node_id(old).unwrap()).collect();
        assert_eq!(ids, ["first", "middle", "last"], "unsupported input is between two re-emitted rows");
        assert!(host.index.held(key, 0.).is_none());
        let (whole, _) = host.step(None);
        assert_eq!(format!("{whole:?}"), host.fresh());
        assert!(whole.items.iter().any(|item| matches!(item, DrawItem::Body { paint: BodyPaint::Pattern(p), .. }
            if p.texture.image.rgba_premultiplied.as_slice() == [237, 0, 0, 255])));
    }

    #[test]
    fn retained_weight_counts_gradient_payload_and_shared_mesh_once() {
        let mut node = path("p", 2., 3., "#3070e0");
        node["fills"] = json!([{"id":"f","paint":{"type":"linear","start":{"x":0,"y":0},"end":{"x":100,"y":0},
            "stops":[{"offset":0,"color":"#ff0000"},{"offset":1,"color":"#0000ff"}]}}]);
        let mut host = Host::new(vec![node], &["p"]);
        let (plan, _) = host.step(None);
        let body = plan.items.iter().find(|i| matches!(i, crate::seams::DrawItem::Body { paint: crate::seams::BodyPaint::Gradient(_), .. })).unwrap();
        let mut inventory = weights::Allocations::default();
        let first = inventory.items(std::slice::from_ref(body));
        let second = inventory.items(std::slice::from_ref(body));
        assert!(first > second);
        let mut solid = body.clone();
        if let crate::seams::DrawItem::Body { paint, .. } = &mut solid { *paint = crate::seams::BodyPaint::Solid; }
        assert_eq!(second - inventory.items(&[solid]), std::mem::size_of::<crate::paint::GradientInstance>());
        inventory.removing = true;
        assert_eq!(inventory.items(std::slice::from_ref(body)), second);
    }

    #[test]
    fn partial_pages_regenerate_backdrop_readers_and_analytic_siblings() {
        let ids: Vec<_> = (0..1024).map(|i| format!("p{i}")).collect();
        let mut nodes: Vec<_> = ids.iter().map(|id| path(id, 0., 0., "#3070e0")).collect();
        nodes.push(json!({"id":"rect","type":"rectangle","width":30,"height":20,
            "fills":[{"id":"f","paint":{"type":"solid","color":"#eeeeee"}}]}));
        let roots: Vec<_> = std::iter::once("rect").chain(ids.iter().map(String::as_str)).collect();
        let mut host = Host::new(nodes, &roots);
        host.index.test_budget = Some(178 * 1024);
        host.step(None);
        for i in [0, 1023] {
            let entry = host.index.entry.as_ref().unwrap();
            let table = entry.parts.as_ref().expect("partial before introducing overlay");
            let patch = entry.patchable.as_ref().unwrap();
            let span = patch.index.by_node.get(host.resident.scene().unwrap(), &ids[i]).unwrap();
            assert_eq!(table.addresses.physical(table.start(span)).is_some(), i == 0);
            assert_eq!(host.edit_and_compare(&ids[i], "blendMode", json!("overlay")).1, 2);
            assert!(host.index.entry.as_ref().is_some_and(|e| e.parts.is_some()), "partial before removing overlay");
            assert_eq!(host.edit_and_compare(&ids[i], "blendMode", json!("normal")).1, 0);
        }
    }

    #[test]
    fn blur_and_overlay_transitions_use_fresh_page_setup() {
        for count in [2, 512] {
            for transparent in [false, true] {
                let ids: Vec<_> = (0..count).map(|i| format!("p{i}")).collect();
                let mut nodes: Vec<_> = ids.iter().map(|id| path(id, 0., 0., "#4060c0")).collect();
                nodes.push(json!({"id":"rect","type":"rectangle","width":23,"height":21,
                    "fills":[{"id":"f","paint":{"type":"solid","color":"#eeeeee"}}]}));
                let roots: Vec<_> = std::iter::once("rect").chain(ids.iter().map(String::as_str)).collect();
                let mut host = Host::new(nodes, &roots);
                host.opts.transparent = transparent;
                host.index.test_budget = Some(110 * 1024);
                host.step(None);
                let blur = json!([{"id":"b","type":"background-blur","radius":6}]);
                for id in [ids.first().unwrap(), ids.last().unwrap()] {
                    // Exercise both orders of introducing and removing the two roles.
                    for (field, value) in [("effects", blur.clone()), ("effects", json!([])),
                        ("blendMode", json!("overlay")), ("effects", blur.clone()), ("effects", json!([])),
                        ("blendMode", json!("normal")), ("effects", blur.clone()), ("blendMode", json!("overlay")),
                        ("blendMode", json!("normal")), ("effects", json!([]))] {
                        let (_, resident, total) = host.index.retained_size();
                        assert!(resident > 0 && (resident < total) == (count == 512));
                        host.edit_and_compare(id, field, value);
                    }
                }
            }
        }
    }

    #[test]
    fn a_second_blur_keeps_page_setup_during_local_effect_edits() {
        let blur = json!([{"id":"b","type":"background-blur","radius":6}]);
        for count in [3, 512] {
            let ids: Vec<_> = (0..count).map(|i| format!("p{i}")).collect();
            let mut nodes: Vec<_> = ids.iter().map(|id| path(id, 0., 0., "#3060c0")).collect();
            nodes[0]["effects"] = blur.clone();
            nodes[1]["blendMode"] = json!("overlay");
            let mut host = Host::new(nodes, &ids.iter().map(String::as_str).collect::<Vec<_>>());
            host.index.test_budget = Some(110 * 1024);
            host.step(None);
            for (field,value) in [("effects", blur.clone()), ("effects", json!([])),
                ("blendMode", json!("normal")), ("blendMode", json!("overlay"))] {
                host.edit_and_compare(&ids[1], field, value);
            }
        }
    }

    #[test]
    fn randomized_partial_page_edits_match_fresh_collection() {
        crate::text_draw::install_text_builder(Some(std::rc::Rc::new(|node, model, z, clip, scale| {
            let text = node.json_field("htmlContent").and_then(Value::as_str).unwrap_or("");
            let mut batch = super::tests::vector_text_batch(node.id(), text.len() + 1, 6, 0);
            batch.model_matrix = model;
            batch.z = z;
            batch.page_clip_bounds = clip;
            batch.vector.as_mut().unwrap().uniform_padding = Some((16., scale));
            Ok(crate::text_draw::TextBuildOutcome::Item(Box::new(crate::seams::DrawItem::Text(Arc::new(batch)))))
        })));
        let ids: Vec<_> = (0..512).map(|i| format!("p{i}")).collect();
        let mut nodes: Vec<_> = ids.iter().map(|id| path(id, 0., 0., "#3060c0")).collect();
        nodes.extend([
            json!({"id":"clip","type":"group","children":["mask","child"],"clipMaskNodeId":"mask"}),
            path("mask", 2., 3., "#ffffff"), path("child", 1., 2., "#ff6060"),
            json!({"id":"label","type":"richtext","width":60,"height":20,"htmlContent":"x"}),
        ]);
        let roots: Vec<_> = ids.iter().map(String::as_str).chain(["clip", "label"]).collect();
        let mut host = Host::new(nodes, &roots);
        host.index.test_budget = Some(110 * 1024);
        host.step(None);
        let mut random = 0x8276_915du32;
        let mut next = || { random ^= random << 13; random ^= random >> 17; random ^= random << 5; random };
        let mut sparse_hits = 0;
        let mut families = [0; 7];
        let (mut overlay, mut blur, mut masked) = (false, false, true);
        for _ in 0..80 {
            let i = next() as usize % ids.len();
            let k = next() as usize % families.len();
            families[k] += 1;
            let v = next();
            let (id, field, value) = match k {
                0 => { overlay = !overlay; (ids[511].as_str(), "blendMode", json!(if overlay { "overlay" } else { "normal" })) },
                1 => ("mask", "x", json!(v % 37)),
                2 => (ids[i].as_str(), "fills", json!([{"id":"fill","paint":{"type":"solid","color":format!("#{:06x}",v & 0xffffff)}}])),
                3 => (ids[i].as_str(), "rotation", json!(v % 90)),
                5 => { blur = !blur; (ids[511].as_str(), "effects", if blur {
                    json!([{"id":"b","type":"background-blur","radius":6}]) } else { json!([]) }) },
                6 => { masked = !masked; ("clip", "clipMaskNodeId", if masked { json!("mask") } else { Value::Null }) },
                _ => ("label", "htmlContent", json!("x".repeat((v % 11 + 1) as usize))),
            };
            let (_, resident, total) = host.index.retained_size();
            assert!(resident > 0 && resident < total, "edit {k} must start partially evicted");
            let hit = host.edit_and_compare(id, field, value).0;
            sparse_hits += usize::from(hit);
        }
        crate::text_draw::install_text_builder(None);
        assert!(families.iter().all(|&n| n >= 2) && sparse_hits > 0, "{families:?}, {sparse_hits} partial patches");
    }

    #[test]
    fn parent_translation_declines_the_unresolved_side_of_a_rounding_midpoint() {
        let mut axis = OriginAxis::default();
        axis.include(-5466.09130859375);
        assert!(!axis.permits(11723.878662109377));
        assert!(axis.permits(11723.5));
    }

    #[test]
    fn translation_checks_only_exponents_reached_by_the_run() {
        let mut axis = OriginAxis::default();
        axis.include(36.);
        axis.include(1035.75);
        assert!(axis.permits(-3.9999998));
        axis.include(4.);
        assert!(!axis.permits(-3.9999998));
    }

    #[test]
    fn translating_a_prepared_solid_group_leaves_its_sibling_outside_the_range() {
        let mut host = Host::new(vec![
            path("left", 20.25, 14., "#224466"), path("right", 31., 18.75, "#cc8844"),
            path("fixed", 90., 38., "#112233"),
            json!({"id":"group","type":"group","children":["left","right"]}),
            json!({"id":"viewport","type":"group","children":["group","fixed"]}),
        ], &["viewport"]);
        let (base, _) = host.step(None);
        let scene = &mut host.resident.state.as_mut().unwrap().scene;
        let row = scene.row_of("group").unwrap();
        scene.write_placement(row, Some(moda_scene::scene::NodePlacement { translation: Some([moda_scene::value::Number::from_f64(0.21333334).unwrap(), 0.into()]), rotation: None, scale: None }), None);
        scene.refresh_worlds();
        let (_, movement) = host.index.translated_group(scene, &host.opts, row).expect("prepared run");
        let names: Vec<_> = base.items[movement.range()].iter().map(drawitem_node_id).collect();
        assert_eq!(names, [Some("left"), Some("right")]);
        scene.write_placement(row, Some(moda_scene::scene::NodePlacement { translation: None, rotation: None, scale: Some([(-1).into(), 1.into()]) }), None);
        scene.refresh_worlds();
        assert!(host.index.translated_group(scene, &host.opts, row).is_none());
    }

    #[test]
    fn consecutive_changed_spans_preserve_their_ordinary_outputs_and_depths() {
        let ids: Vec<_> = (0..4096).map(|i| format!("p{i}")).collect();
        let nodes = ids.iter().enumerate().map(|(i, id)| path(id, (i % 64) as f64, 8., "#00a0e0")).collect();
        let mut host = Host::new(nodes, &ids.iter().map(String::as_str).collect::<Vec<_>>());
        let mut doc = host.resident.scene().unwrap().doc().clone();
        for node in &mut doc.nodes { node.compact(); }
        host.resident = ResidentScene::from_scene(Scene::from_document(doc).unwrap());
        let (plan, _) = host.step(None);
        let scene = host.resident.scene().unwrap();
        let index = &host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().index;
        let dirty: Vec<_> = ids.iter().map(|id| index.by_node.get(scene, id).unwrap()).collect();
        let mut batch = EmittedRun::prepare(scene, index, &dirty, &host.opts, (&mut host.caches, host.resident.hashes().unwrap()), None).expect("ordinary worker emission");
        for (i, si) in dirty.into_iter().enumerate() {
            let old = &index.spans[si];
            let (out, z, _) = batch.next().unwrap();
            assert_eq!((out.len(), z), (old.len, old.z_consumed));
            if i % 127 == 0 || i == 4095 {
                assert_eq!(format!("{out:?}"), format!("{:?}", &plan.items[old.start..old.start + old.len]));
            }
        }
        assert!(batch.next().is_none());
        drop(plan);
        host.index.test_budget = Some(600 * 1024);
        host.index.finish_draw(host.resident.scene().unwrap(), &mut [], None);
        for paint_only in [true, false] {
            let (_, kept, total) = host.index.retained_size();
            assert!(kept > 0 && kept < total);
            let state = host.resident.state.as_mut().unwrap();
            let rows: Vec<_> = ids.iter().map(|id| state.scene.row_of(id).unwrap()).collect();
            let points = state.scene.row(rows[0]).unwrap().shared_points().unwrap().clone();
            let fill = json!([{"id":"fill","paint":{"type":"solid","color":"#ff3020"}}]);
            for &row in &rows {
                state.scene.write_row(row, |f| if paint_only { f.set("fills", fill.clone()); } else {
                    f.set_shared_points(points.clone());
                    f.set("closed", json!(false));
                }).unwrap();
            }
            state.scene.refresh_worlds();
            state.hashes = node_hashes_for_scene(&state.scene);
            host.generation += 1;
            let (view, _) = host.index.update(&state.scene, &state.hashes, &host.opts, host.generation,
                host.generation, &rows, false, &mut host.caches).expect("whole range rewrites in painter order");
            assert!(host.index.addresses(&view).is_none());
            assert_eq!(format!("{view:?}"), host.fresh());
            let written = host.index.take_written();
            assert_eq!(written.recollect, !paint_only);
            assert_eq!(written.placed.len(), rows.len());
            host.index.finish_draw(host.resident.scene().unwrap(), &mut [], None);
        }
    }

    #[test]
    fn dense_rewrite_keeps_text_and_solid_paint_destinations() {
        use crate::seams::DrawItem;
        crate::text_draw::install_text_builder(Some(std::rc::Rc::new(|node, model, z, clip, _| {
            let mut batch = super::tests::vector_text_batch(node.id(), 1, 6, 0);
            batch.model_matrix = model;
            batch.z = z;
            batch.page_clip_bounds = clip;
            batch.paint_geometry_key = Some([1; 4]);
            Ok(crate::text_draw::TextBuildOutcome::Item(Box::new(DrawItem::Text(Arc::new(batch)))))
        })));
        let ids: Vec<_> = (0..4096).map(|i| format!("ink{i}")).collect();
        let nodes = ids.iter().enumerate().map(|(i, id)| if i % 512 == 0 {
            json!({"id":id,"type":"richtext","width":60,"height":20,"htmlContent":"ink"})
        } else { path(id, 0., 0., "#3070e0") }).collect();
        let mut host = Host::new(nodes, &ids.iter().map(String::as_str).collect::<Vec<_>>());
        host.index.test_budget = Some(620 * 1024);
        let (original, _) = host.step(None);
        let glyph_slots: Vec<_> = original.items.iter().enumerate().filter_map(|(at, item)|
            matches!(item, DrawItem::Text(_)).then_some(at)).collect();
        assert_eq!(glyph_slots.len(), 8);
        drop(original);
        let (_, kept, total) = host.index.retained_size();
        assert!(kept > 0 && kept < total);
        let state = host.resident.state.as_mut().unwrap();
        let rows: Vec<_> = ids.iter().map(|id| state.scene.row_of(id).unwrap()).collect();
        let fill = json!([{"id":"fill","paint":{"type":"solid","color":"#6699cc"}}]);
        for (i, &row) in rows.iter().enumerate() {
            state.scene.write_row(row, |f| if i % 512 == 0 { f.set("vorkPaint", json!(2)); }
                else { f.set("fills", fill.clone()); }).unwrap();
        }
        state.scene.refresh_worlds();
        state.hashes = node_hashes_for_scene(&state.scene);
        let (view, _) = host.index.update(&state.scene, &state.hashes, &host.opts, 2, 2, &rows, false, &mut host.caches).unwrap();
        assert!(host.index.addresses(&view).is_none());
        assert_eq!(format!("{view:?}"), host.fresh());
        let writes = host.index.take_written();
        assert!(!writes.recollect && writes.replaced.is_empty());
        assert_eq!(writes.placed.len(), rows.len());
        for at in glyph_slots {
            assert!(matches!(view.items[at], DrawItem::Text(_)));
            assert!(writes.placed.binary_search_by_key(&(at as u32), |w| w.at).is_ok());
        }
        crate::text_draw::install_text_builder(None);
    }

    #[test]
    fn sparse_placement_keeps_evicted_sources_at_their_packet_slots() {
        check_sparse_restore(false);
    }

    #[test]
    fn evicted_text_and_paths_share_their_original_packet_slots() {
        crate::text_draw::install_text_builder(Some(std::rc::Rc::new(|node, model, z, clip, scale| {
            let mut batch = super::tests::vector_text_batch(node.id(), 1, 6, 0);
            batch.model_matrix = model;
            batch.z = z;
            batch.page_clip_bounds = clip;
            batch.paint_geometry_key = Some([1; 4]);
            batch.vector.as_mut().unwrap().uniform_padding = Some((16., scale));
            Ok(crate::text_draw::TextBuildOutcome::Item(Box::new(crate::seams::DrawItem::Text(Arc::new(batch)))))
        })));
        check_sparse_restore(true);
        crate::text_draw::install_text_builder(None);
    }

    fn check_sparse_restore(with_text: bool) {
        let ids: Vec<_> = (0..1024).map(|i| format!("ink{i}")).collect();
        let nodes = ids.iter().enumerate().map(|(i, id)| if with_text && i % 2 == 0 {
            json!({"id":id, "type":"richtext", "width":70, "height":60, "htmlContent":"<p>ink</p>"})
        } else { path(id, 0., 0., "#30a0f0") }).collect();
        let mut host = Host::new(nodes, &ids.iter().map(String::as_str).collect::<Vec<_>>());
        let mut doc = host.resident.scene().unwrap().doc().clone();
        for node in &mut doc.nodes { node.compact(); }
        host.resident = ResidentScene::from_scene(Scene::from_document(doc).unwrap());
        host.index.test_budget = Some(if with_text { 420 * 1024 } else { 160 * 1024 });
        host.step(None);
        let initial = host.index.retained_size().1;
        let state = host.resident.state.as_mut().unwrap();
        let entry = host.index.entry.as_ref().unwrap();
        let index = &entry.patchable.as_ref().unwrap().index;
        let mut edits = Vec::new();
        let mut missing = 0;
        for id in ids.iter().step_by(37) {
            let row = state.scene.row_of(id).unwrap();
            let si = index.by_node.get(&state.scene, id).unwrap();
            if index.spans.placement(si).unwrap().0.start == parts::ABSENT { missing += 1; }
            edits.push(row);
            state.scene.write_row(row, |fields| fields.set("x", json!(12))).unwrap();
        }
        assert!(missing > 0 && missing < edits.len(), "{initial} resident items, {missing} of {} edited spans evicted", edits.len());
        state.scene.refresh_worlds();
        state.hashes = node_hashes_for_scene(&state.scene);
        host.index.place(&state.scene, &host.opts, 2, 2, &edits, &mut host.caches).unwrap();

        let view = host.index.entry.as_ref().unwrap().plan.clone();
        assert_eq!(host.index.retained_size().1, initial);
        let writes = host.index.take_written();
        assert!(!writes.recollect && writes.replaced.is_empty());
        assert_eq!(writes.evicted.len(), missing);
        assert_eq!(writes.placed.len() + missing, edits.len());
        let full = host.index.materialize(&view, &state.scene, &state.hashes, &host.opts, &mut host.caches).unwrap();
        assert_eq!(format!("{full:?}"), host.fresh());
    }

    #[test]
    fn batched_primaries_have_the_same_source_and_depth_as_single_source_emission() {
        let ids: Vec<_> = (0..4096).map(|i| format!("rect{i}")).collect();
        let groups: Vec<_> = (0..ids.len()).map(|i| format!("g{i}")).collect();
        let nodes = ids.iter().zip(&groups).flat_map(|(id, group)| [
            json!({"id":group,"type":"group","children":[id]}),
            json!({"id":id,"type":"rectangle","width":8,"height":8,
                "fills":[{"id":"f","paint":{"type":"solid","color":"#204060"}}]})
        ]).collect();
        let mut host = Host::new(nodes, &groups.iter().map(String::as_str).collect::<Vec<_>>());
        host.step(None);
        let scene = host.resident.scene().unwrap();
        let index = &host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().index;
        let dirty: Vec<_> = ids.iter().map(|id| index.by_node.get(scene, id).unwrap()).collect();
        assert!(index.spans[dirty[1]].dl_z > index.spans[dirty[0]].dl_z + 1);
        let run = EmittedRun::prepare(scene, index, &dirty, &host.opts, (&mut host.caches, host.resident.hashes().unwrap()), None).unwrap();
        assert_eq!(run.spans.len(), dirty.len());
        for (i, span) in run.spans.enumerate().step_by(63) {
            let old = &index.spans[dirty[i]];
            assert_eq!(span.source.id(scene), ids[i]);
            assert_eq!((span.z, span.dl_z, span.len), (old.z, old.dl_z, old.len));
        }
    }

    #[test]
    fn cold_spans_reemit_parent_and_paint_at_their_original_slots() {
        let ids: Vec<_> = (0..512).map(|i| format!("child{i}")).collect();
        let mut nodes: Vec<_> = ids.iter().map(|id| path(id, 0., 0., "#30a0f0")).collect();
        nodes.push(json!({"id":"parent","type":"group","children":[ids[510].as_str(),ids[511].as_str()]}));
        let mut roots: Vec<_> = ids[..510].iter().map(String::as_str).collect();
        roots.push("parent");
        let mut host = Host::new(nodes, &roots);
        host.index.test_budget = Some(96 * 1024);
        host.step(None);
        assert!(host.index.entry.as_ref().unwrap().parts.is_some());
        for (id, field, value) in [
            ("parent", "x", json!(7)),
            (ids[511].as_str(), "fills", json!([{"id":"fill","paint":{"type":"solid","color":"#ff3080"}}])),
            ("parent", "x", json!(13)),
        ] {
            let state = host.resident.state.as_mut().unwrap();
            let row = state.scene.row_of(id).unwrap();
            state.scene.write_row(row, |fields| fields.set(field, value)).unwrap();
            state.scene.refresh_worlds();
            state.hashes = node_hashes_for_scene(&state.scene);
            host.generation += 1;
            let (view, _) = host.index.update(&state.scene, &state.hashes, &host.opts,
                host.generation, host.generation, &[row], false, &mut host.caches).expect("patch reached spans");
            let written = host.index.take_written();
            assert!(!written.recollect);
            let mut touched: Vec<u32> = written.placed.iter().map(|w| w.at)
                .chain(written.replaced.iter().map(|(i, _)| *i)).collect();
            let addresses = host.index.addresses(&view).unwrap();
            let slots: Vec<_> = touched.iter().map(|&i| addresses.logical(i as usize).unwrap()).collect();
            assert!(slots.contains(&512) || slots.contains(&513), "{slots:?}");
            let complete = host.index.materialize(&view, &state.scene, &state.hashes, &host.opts, &mut host.caches).unwrap();
            assert_eq!(format!("{complete:?}"), host.fresh());
            host.index.finish_draw(host.resident.scene().unwrap(), &mut touched, None);
            assert!(host.index.entry.as_ref().unwrap().bytes <= host.index.budget());
        }
    }

    #[test]
    fn an_evicted_group_keeps_its_execution_range() {
        let ids: Vec<_> = (0..512).map(|i| format!("ink{i}")).collect();
        let mut nodes: Vec<_> = ids.iter().map(|id| path(id, 0., 0., "#30a0f0")).collect();
        nodes.push(json!({"id":"group","type":"group","opacity":0.55,"children":ids}));
        let mut host = Host::new(nodes, &["group"]);
        let (dense, _) = host.step(None);
        // The output itself cannot fit, regardless of how compact its source index becomes.
        host.index.test_budget = Some(items_bytes(&dense.items) / 2);
        drop(dense);
        host.index.finish_draw(host.resident.scene().unwrap(), &mut [], None);
        let held = host.index.entry.as_ref().expect("retained group");
        assert!(held.parts.as_ref().is_some_and(|p| p.has_cold_units()), "host bytes {}", held.bytes);
        assert!(held.bytes + held.parts.as_ref().unwrap().addresses.total * crate::frame::DESTINATION_BYTES <= host.index.budget());
        {
            let state = host.resident.state.as_ref().unwrap();
            let whole = host.index.materialize(&held.plan, &state.scene, &state.hashes, &host.opts, &mut host.caches).unwrap();
            assert_eq!(format!("{whole:?}"), host.fresh());
        }
        for (id, field, value) in [("ink3", "x", json!(12)), ("group", "opacity", json!(0.7))] {
            let state = host.resident.state.as_mut().unwrap();
            let row = state.scene.row_of(id).unwrap();
            state.scene.write_row(row, |fields| fields.set(field, value)).unwrap();
            state.scene.refresh_worlds();
            state.hashes = node_hashes_for_scene(&state.scene);
            host.generation += 1;
            let (view, _) = host.index.update(&state.scene, &state.hashes, &host.opts,
                host.generation, host.generation, &[row], false, &mut host.caches).unwrap_or_else(|| panic!("patch {id}"));
            let complete = host.index.materialize(&view, &state.scene, &state.hashes, &host.opts, &mut host.caches).unwrap();
            assert_eq!(format!("{complete:?}"), host.fresh(), "after {id}");
            host.index.finish_draw(host.resident.scene().unwrap(), &mut [], None);
        }
    }

    #[test]
    fn evicted_spans_beside_a_mask_keep_its_execution_range() {
        let ids: Vec<_> = (0..512).map(|i| format!("child{i}")).collect();
        let mut nodes: Vec<_> = ids.iter().map(|id| path(id, 0., 0., "#30a0f0")).collect();
        nodes.extend([
            json!({"id":"clip","type":"group","clipMaskNodeId":"mask","children":["inside"]}),
            path("mask", 4., 4., "#ffffff"), path("inside", 12., 12., "#e04080"),
        ]);
        let mut roots: Vec<_> = ids.iter().map(String::as_str).collect();
        roots.splice(500..500, ["clip", "mask"]);
        let mut host = Host::new(nodes, &roots);
        host.index.test_budget = Some(96 * 1024);
        host.step(None);
        assert!(host.index.entry.as_ref().unwrap().parts.is_some());
        for (id, field, value) in [
            ("clip", "x", json!(7)),
            (ids[505].as_str(), "fills", json!([{"id":"fill","paint":{"type":"solid","color":"#ff3080"}}])),
            ("mask", "x", json!(17)),
            ("inside", "x", json!(22)),
        ] {
            let state = host.resident.state.as_mut().unwrap();
            let row = state.scene.row_of(id).unwrap();
            state.scene.write_row(row, |fields| fields.set(field, value)).unwrap();
            state.scene.refresh_worlds();
            state.hashes = node_hashes_for_scene(&state.scene);
            host.generation += 1;
            let (view, _) = host.index.update(&state.scene, &state.hashes, &host.opts,
                host.generation, host.generation, &[row], false, &mut host.caches).unwrap_or_else(|| panic!("patch {id}"));
            let complete = host.index.materialize(&view, &state.scene, &state.hashes, &host.opts, &mut host.caches).unwrap();
            assert_eq!(format!("{complete:?}"), host.fresh(), "after {id}");
            let writes = host.index.take_written();
            let mut touched: Vec<_> = writes.placed.into_iter().map(|w| w.at).chain(writes.replaced.into_iter().map(|(i, _)| i)).collect();
            host.index.finish_draw(host.resident.scene().unwrap(), &mut touched, None);
        }
    }

    #[test]
    fn page_body_mode_changes_recollect_all_readers() {
        let mut host = Host::new(vec![
            json!({"id":"fade","type":"group","opacity":0.6,"children":["child"]}),
            path("child", 0., 0., "#ff3060"), path("path_only", 220., 130., "#eecc30"),
            json!({"id":"plain","type":"rectangle","x":100,"y":20,"width":70,"height":60,
                "fills":[{"id":"fill","paint":{"type":"solid","color":"#30ee80"}}]}),
            json!({"id":"clip","type":"group","clipMaskNodeId":"mask","children":["steady"]}),
            path("mask", 20., 120., "#ffffff"), path("steady", 25., 125., "#2040ee"),
        ], &["fade","plain","clip","mask","path_only"]);
        host.step(None);
        for opacity in [1., 0.7, 1., 0.4] {
            assert!(!host.edit_and_compare("fade", "opacity", json!(opacity)).0);
        }
    }

    #[test]
    fn plain_page_visibility_has_recorded_source_writers() {
        for group in [false, true] {
            let mut source = path("source", 20., 30., "#2030ff");
            source["opacity"] = json!(0.);
            let mut nodes = vec![source, path("after", 50., 60., "#40ff30")];
            let roots = if group {
                nodes.push(json!({"id":"group","type":"group","opacity":1.,"children":["source"]}));
                vec!["group","after"]
            } else { vec!["source","after"] };
            let mut host = Host::new(nodes, &roots);
            host.step(None);
            for (id, opacity) in [("source",0.5),("source",0.),("source",1.),
                (if group { "group" } else { "source" },0.6), ("source",0.8)] {
                let hit = host.edit_and_compare(id, "opacity", json!(opacity)).0;
                assert_eq!(hit, id != "group", "{id} at {opacity}");

            }
        }
    }

    #[test]
    fn visibility_writes_retain_source_membership_and_depth() {
        for inside in [false, true] {
        for initial in [0., 0.6] {
        let mut hidden = path("hidden", 20., 30., "#2030ff");
        hidden["opacity"] = json!(initial);
        let children = if inside { json!(["child", "hidden", "sibling"]) } else { json!(["child", "sibling"]) };
        let roots = if inside { vec!["fade", "after"] } else { vec!["fade", "hidden", "after"] };
        let mut host = Host::new(vec![
            json!({"id":"fade","type":"group","opacity":0.6,"children":children}),
            path("child", 0., 0., "#ff3060"), hidden,
            path("sibling", 45., 25., "#f0ee33"),
            path("after", 50., 60., "#40ff30"),
        ], &roots);
        host.step(None);
        for opacity in [0.5, 0.8, 0., 0.7, 0.] {
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            scene.write_row(scene.row_of("hidden").unwrap(), |fields| fields.set("opacity", json!(opacity))).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene); let clock = scene.stamps().clock();
            let (plan, _) = host.index.update(scene, &hashes, &host.opts, clock, clock,
                &[scene.row_of("hidden").unwrap()], false, &mut host.caches).unwrap_or_else(|| panic!("visibility is a retained value: inside={inside} initial={initial} opacity={opacity}"));
            host.index.take_written();
            let (fresh, _) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"));
        }
            }
        }
    }

    #[test]
    fn group_visibility_retains_descendant_sources() {
        for inside in [false, true] {
        let children = if inside { json!(["child", "hidden_group", "sibling"]) } else { json!(["child", "sibling"]) };
        let roots = if inside { vec!["fade", "after"] } else { vec!["fade", "hidden_group", "after"] };
        let mut host = Host::new(vec![
            json!({"id":"fade","type":"group","opacity":0.6,"children":children}),
            json!({"id":"hidden_group","type":"group","opacity":0.,"children":["a","b"]}),
            path("child", 0., 0., "#ff3060"), path("a", 30., 20., "#2030ff"), path("b", 40., 50., "#2030ff"),
            path("sibling", 45., 25., "#f0ee33"), path("after", 50., 60., "#40ff30"),
        ], &roots);
        host.step(None);
        for opacity in [0.5, 0., 1., 0.7, 0.] {
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            scene.write_row(scene.row_of("hidden_group").unwrap(), |fields| fields.set("opacity", json!(opacity))).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene); let clock = scene.stamps().clock();
            let (plan, _) = host.index.update(scene, &hashes, &host.opts, clock, clock,
                &[scene.row_of("hidden_group").unwrap()], false, &mut host.caches)
                .unwrap_or_else(|| panic!("group visibility: inside={inside} opacity={opacity}"));
            host.index.take_written();
            let (fresh, _) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"));
        }
        }
    }

    #[test]
    fn empty_recorded_leaf_output_keeps_its_writer() {
        let mut host = Host::new(vec![
            json!({"id":"fade","type":"group","opacity":0.6,"children":["child"]}),
            path("child", 0., 0., "#ff3060"),
            json!({"id":"text","type":"richtext","x":1000,"y":20,"width":70,"height":60,
                "clipTextToBounds":true,"text":"outside"}),
        ], &["fade","text"]);
        host.step(None);
        for x in [1010., 1020., 1030.] {
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            scene.write_row(scene.row_of("text").unwrap(), |fields| fields.set("x", json!(x))).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene); let clock = scene.stamps().clock();
            let (plan, _) = host.index.update(scene, &hashes, &host.opts, clock, clock,
                &[scene.row_of("text").unwrap()], false, &mut host.caches).expect("empty recorded output has a writer too");
            assert!(host.index.take_written().replaced.is_empty(), "empty output does not rewrite the standing capture");
            let (fresh, _) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"));
        }
    }

    #[test]
    fn plain_content_uses_the_standing_page_composition_context() {
        let mut host = Host::new(vec![
            json!({"id":"fade","type":"group","opacity":0.6,"children":["child"]}),
            path("child", 0., 0., "#ff3060"),
            json!({"id":"plain","type":"rectangle","x":100,"y":20,"width":70,"height":60,
                "fills":[{"id":"fill","paint":{"type":"solid","color":"#30ee80"}}]}),
        ], &["fade","plain"]);
        host.step(None);
        for step in 0..8 {
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            scene.write_row(scene.row_of("plain").unwrap(), |fields| fields.set("width", json!(71 + step))).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene); let clock = scene.stamps().clock();
            let (plan, _) = host.index.update(scene, &hashes, &host.opts, clock, clock,
                &[scene.row_of("plain").unwrap()], false, &mut host.caches).expect("plain content uses its cold emission context");
            host.index.take_written();
            let (fresh, _) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"));
        }
    }

    #[test]
    fn a_revealed_source_and_capture_content_patch_together() {
        let mut hidden = path("plain", 150., 20., "#30ee80"); hidden["opacity"] = json!(0);
        let mut host = Host::new(vec![
            json!({"id":"capture","type":"group","children":["child"],
                "effects":[{"id":"blur","type":"layer-blur","radius":3}]}),
            path("child", 0., 0., "#ff3060"), hidden,
        ], &["capture","plain"]);
        host.step(None);
        let scene = &mut host.resident.state.as_mut().unwrap().scene;
        scene.write_row(scene.row_of("child").unwrap(), |fields| fields.set("points", json!([
            {"type":"move","x":0,"y":0},{"type":"line","x":63,"y":0},
            {"type":"line","x":48,"y":57},{"type":"line","x":0,"y":60}]))).unwrap();
        scene.write_row(scene.row_of("plain").unwrap(), |fields| fields.set("opacity", json!(1))).unwrap();
        scene.refresh_worlds();
        let hashes = node_hashes_for_scene(scene); let clock = scene.stamps().clock();
        let (plan, _) = host.index.update(scene, &hashes, &host.opts, clock, clock,
            &[scene.row_of("child").unwrap(), scene.row_of("plain").unwrap()], false, &mut host.caches).expect("recorded source membership survives visibility");
        let (fresh, _) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
            Some((&mut PlanCaches::default(), &hashes))).unwrap();
        assert_eq!(format!("{plan:?}"), format!("{fresh:?}"));
    }

    #[test]
    fn mask_updates_do_not_retain_plain_source_preparation_without_an_owner() {
        let mut host = Host::new(vec![
            json!({"id":"clip","type":"group","clipMaskNodeId":"mask","children":["body"]}),
            path("mask", 0., 0., "#ffffff"), path("body", 5., 10., "#ff3060"),
        ], &["clip","mask"]);
        host.step(None);
        let units = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref().unwrap();
        assert!(!units.owners.contains(&crate::frame::SpanSource::of(host.resident.scene().unwrap(), "mask")), "mask-only source belongs to its consumers, not an empty body writer");
        for step in 0..8 {
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            scene.write_row(scene.row_of("mask").unwrap(), |fields| fields.set("x", json!(step + 1))).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene); let clock = scene.stamps().clock();
            let (plan, _) = host.index.update(scene, &hashes, &host.opts, clock, clock,
                &[scene.row_of("mask").unwrap()], false, &mut host.caches).expect("mask source range remains patchable");
            host.index.take_written();
            let (fresh, _) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"));
            assert_eq!(plan.masks, fresh.masks);
            let units = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref().unwrap();
            let rows = units.segment.as_ref().unwrap().prepared.as_ref().unwrap();
            assert!(rows.output(scene.row_of("body").unwrap()).is_none(),
                "the plain span's existing writer does not consume a retained preparation row");
        }
    }

    #[test]
    fn plain_spans_do_not_allocate_unreachable_output_graphs() {
        let mut nodes = vec![json!({"id":"capture","type":"group","children":["child"],
            "effects":[{"id":"blur","type":"layer-blur","radius":3}]}), path("child", 0., 0., "#ffcc20")];
        let mut roots = vec!["capture".to_string()];
        for i in 0..128 { let id = format!("plain-{i}"); nodes.push(path(&id, 10., 20., "#ff3060")); roots.push(id); }
        let mut host = Host::new(nodes, &roots.iter().map(String::as_str).collect::<Vec<_>>());
        host.step(None);
        let units = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref().unwrap();
        let ui = units.by_node.get(host.resident.scene().unwrap(), "capture").unwrap();
        assert!(units.outputs[ui].is_some());
        assert_eq!(units.outputs.iter().filter(|record| record.is_some()).count(), 1,
            "ordinary spans use their existing index, not duplicate output graphs");
    }

    #[test]
    fn hidden_sources_keep_empty_output_and_reveal_never_reuses_absence() {
        for capture in [false, true] {
            for hidden_parent in [false, true] {
                let mut moving = path("moving", 25., 25., "#ee3080");
                if !hidden_parent { moving["opacity"] = json!(0); }
                let mut root = json!({"id":"root","type":"group","children":["left","hidden","right"]});
                if capture { root["effects"] = json!([{"id":"b","type":"layer-blur","radius":3}]); }
                let hidden = json!({"id":"hidden","type":"group","children":["moving"],
                    "opacity":if hidden_parent {0.} else {1.}});
                let mut host = Host::new(vec![root, hidden, moving,
                    path("left", 0., 0., "#2080cc"), path("right", 100., 100., "#80cc20")], &["root"]);
                host.step(None);
                for step in 0..12 {
                    let id = if hidden_parent && step % 2 == 0 { "hidden" } else { "moving" };
                    let scene = &mut host.resident.state.as_mut().unwrap().scene;
                    let row = scene.row_of(id).unwrap();
                    scene.write_row(row, |fields| { fields.set("x", json!(25 + step)); }).unwrap();
                    scene.refresh_worlds();
                    let hashes = node_hashes_for_scene(scene); let clock = scene.stamps().clock();
                    let (plan, warnings) = host.index.update(scene, &hashes, &host.opts, clock, clock,
                        &[scene.row_of(id).unwrap()], false, &mut host.caches).expect("an unread hidden source retains empty output");
                    let written = host.index.take_written();
                    assert!(written.replaced.is_empty() && written.placed.is_empty(), "hidden moves must not rewrite visible output");
                    let (fresh, fresh_warnings) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                        Some((&mut PlanCaches::default(), &hashes))).unwrap();
                    assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "capture={capture}, hidden_parent={hidden_parent}, step={step}");
                    assert_eq!(format!("{warnings:?}"), format!("{fresh_warnings:?}"));
                }
                let scene = &mut host.resident.state.as_mut().unwrap().scene;
                let id = if hidden_parent {"hidden"} else {"moving"};
                scene.write_row(scene.row_of(id).unwrap(), |fields| { fields.set("opacity", json!(1)); }).unwrap();
                scene.refresh_worlds();
                let hashes = node_hashes_for_scene(scene); let clock = scene.stamps().clock();
                if let Some((plan, _)) = host.index.update(scene, &hashes, &host.opts, clock, clock,
                    &[scene.row_of(id).unwrap()], false, &mut host.caches) {
                    let (fresh, _) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                        Some((&mut PlanCaches::default(), &hashes))).unwrap();
                    assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "revealed output cannot retain an empty proof");
                }
            }
        }
    }

    #[test]
    fn content_stamps_rebuild_only_changed_spans_in_a_moving_frame() {
        let mut host = Host::new(vec![path("moving", 10., 10., "#255fe8"),
            path("edited", 100., 10., "#ff7700"),
            json!({"id":"capture","type":"group","children":["child"],
                "effects":[{"id":"blur","type":"layer-blur","radius":3}]}),
            path("child", 200., 20., "#30ee80")], &["moving", "edited", "capture"]);
        host.step(None);
        for step in 1..12 {
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let moving = scene.row_of("moving").unwrap();
            let edited = scene.row_of("edited").unwrap();
            scene.write_row(moving, |fields| { fields.set("x", json!(10 + step)); }).unwrap();
            scene.write_row(edited, |fields| {
                fields.set("points", json!([
                    {"type":"move","x":0,"y":0}, {"type":"line","x":70 + step,"y":0},
                    {"type":"line","x":70,"y":60}, {"type":"line","x":0,"y":60}
                ]));
                fields.set("fills", json!([{"id":"fill","paint":{"type":"solid",
                    "color":if step % 2 == 0 { "#ff7700" } else { "#00aa77" }}}]));
            }).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene);
            let clock = scene.stamps().clock();
            let (patched, _) = host.index.update(scene, &hashes, &host.opts, clock, clock,
                &[scene.row_of("moving").unwrap(), scene.row_of("edited").unwrap()], false, &mut host.caches).expect("node-local edits patch");
            let written = host.index.take_written();
            assert!(!written.placed.is_empty(), "unaffected geometry follows placement");
            assert!(!written.replaced.is_empty(), "changed content is prepared again");
            let units = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref().unwrap();
            assert!(!units.by_node.contains(scene, "edited"),
                "a plain content write retains its existing span ownership");
            assert_eq!(units.outputs.iter().filter(|output| output.is_some()).count(), 1);
            let (fresh, _) = crate::frame::build_coverage_plan(scene, 0, &host.opts).unwrap();
            assert_eq!(format!("{patched:?}"), format!("{fresh:?}"), "step {step}");
        }
    }

    #[test]
    fn changed_capture_units_match_a_fresh_page() {
        for (effect, initial_x) in ["layer-blur", "background-blur", "fade"].into_iter()
            .flat_map(|effect| [0., 10_000.].map(move |x| (effect, x))) {
            let mut unit = json!({"id":"unit","type":"group","x":initial_x,"children":["inside"]});
            if effect == "fade" { unit["opacity"] = json!(0.6); }
            else { unit["effects"] = json!([{"id":"e","type":effect,"radius":5}]); }
            for mover in ["unit", "inside", "above"] {
                let mut host = Host::new(vec![path("free", 10., 10., "#255fe8"),
                    json!({"id":"above","type":"group","children":["unit"]}), unit.clone(),
                    path("inside", 80., 80., "#ff7700")], &["free", "above"]);
                host.step(None);
                for step in 1..8 {
                    let scene = &mut host.resident.state.as_mut().unwrap().scene;
                    let row = scene.row_of(mover).unwrap();
                    scene.write_row(row, |fields| {
                        fields.set("x", json!(step));
                        if mover == "unit" && effect == "fade" {
                            fields.set("opacity", json!(0.2 + step as f64 * 0.05));
                        }
                        if mover == "inside" {
                            fields.set("fills", json!([{"id":"fill","paint":{"type":"solid",
                                "color":if step % 2 == 0 { "#ff7700" } else { "#00aa77" }}}]));
                        }
                    }).unwrap();
                    scene.refresh_worlds();
                    let hashes = node_hashes_for_scene(scene);
                    let clock = scene.stamps().clock();
                    let (plan, _) = host.index.update(scene, &hashes, &host.opts, clock, clock,
                        &[scene.row_of(mover).unwrap()], false, &mut host.caches).expect("affected unit patches");
                    host.index.take_written();
                    let (fresh, _) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                        Some((&mut PlanCaches::default(), &hashes))).unwrap();
                    assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "{effect}/{mover}/{step}");
                }
            }
        }
    }

    #[test]
    fn isolated_opacity_writes_only_its_end_and_preserves_the_next_move() {
        use crate::seams::{BackdropOp, DrawItem};
        let mut host = Host::new(vec![
            json!({"id":"label","type":"group","opacity":0.75,"children":["pill","ink"]}),
            path("pill", 2., 10., "#873877"), path("ink", 8., 12., "#f0f0f0"),
            path("marker", 10., 70., "#885522"),
        ], &["label", "marker"]);
        let (snapshot, _) = host.step(None);
        let original = format!("{snapshot:?}");
        {
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let row = scene.row_of("label").unwrap();
            let before = scene.stamps().clock();
            scene.write_row(row, |fields| fields.set("opacity", json!(0.375))).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene);
            let at = scene.stamps().clock();
            host.index.record_parameter_writes(Some((before, at, moda_scene::value::K::Opacity)));
            host.generation += 1;
            let (patched, _) = host.index.update(scene, &hashes, &host.opts, host.generation,
                host.generation, &[row], false, &mut host.caches).expect("ordinary plan stays patchable");
            let written = host.index.take_written();
            assert!(written.evicted.is_empty() && written.replaced.is_empty());
            let [end] = written.placed.as_slice() else { panic!("expected one scalar write") };
            assert!(matches!(&patched.items[end.at as usize], DrawItem::BackdropSplit(s)
                if matches!(s.op, BackdropOp::StackEnd { opacity: 0.375, .. })));
            let (fresh, _) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{patched:?}"), format!("{fresh:?}"));
        }
        assert!(host.edit_and_compare("label", "x", json!(9.)).0);
        assert!(host.edit_and_compare("marker", "x", json!(13.)).0);
        assert_eq!(format!("{snapshot:?}"), original);
    }

    #[test]
    fn cold_units_without_spans_patch_and_replay_each_revision() {
        use crate::seams::{BackdropOp, DrawItem};
        let groups: Vec<_> = (0..512).map(|i| format!("group{i}")).collect();
        let mut nodes = Vec::new();
        for (i, group) in groups.iter().enumerate() {
            let child = format!("path{i}");
            nodes.push(json!({"id":group,"type":"group","opacity":0.75,"children":[child]}));
            nodes.push(path(&child, 2., 10., "#873877"));
        }
        let roots: Vec<_> = groups.iter().map(String::as_str).collect();
        let mut host = Host::new(nodes, &roots);
        let budget = 56 * 1024;
        host.index.test_budget = Some(budget);
        let (first, _) = host.step(None);
        let (bytes, resident, logical) = host.index.retained_size();
        assert!(bytes > 0 && bytes < budget);
        assert_eq!(logical, first.items.len());
        assert!(resident < 3, "only page output outside the units stays resident");
        let entry = host.index.entry.as_ref().unwrap();
        assert!(entry.parts.as_ref().unwrap().has_cold_units());
        assert_eq!(entry.patchable.as_ref().unwrap().index.spans.len(), 0);
        assert!(host.index.finish_draw(host.resident.scene().unwrap(), &mut [], None));
        let id = &groups[17];
        let end = first.items.iter().position(|item| matches!(item, DrawItem::BackdropSplit(split)
            if split.node_id == *id && matches!(split.op, BackdropOp::StackEnd { .. }))).unwrap();
        let scene = &mut host.resident.state.as_mut().unwrap().scene;
        let row = scene.row_of(id).unwrap();
        for (revision, value) in [(2, 0.25), (3, 0.5), (4, 1.0)] {
            let before = scene.stamps().clock();
            scene.write_row(row, |fields| fields.set("opacity", json!(value))).unwrap();
            scene.refresh_worlds();
            host.index.record_parameter_writes(Some((before, scene.stamps().clock(), moda_scene::value::K::Opacity)));
            let patched = host.index.update(scene, &node_hashes_for_scene(scene), &host.opts, revision,
                revision, &[row], false, &mut host.caches);
            if value == 1.0 {
                assert!(patched.is_none(), "leaving isolation changes the draw layout");
            } else {
                let (view, _) = patched.unwrap();
                assert!(host.index.addresses(&view).unwrap().physical(end).is_none());
                let hashes = node_hashes_for_scene(scene);
                let full = host.index.materialize(&view, scene, &hashes, &host.opts, &mut host.caches).unwrap();
                let (fresh, _, _) = crate::frame::build_coverage_plan_indexed(scene, 0, &host.opts,
                    Some((&mut PlanCaches::default(), &hashes))).unwrap();
                assert_eq!(format!("{full:?}"), format!("{fresh:?}"));
                assert!(matches!(host.index.take_written().evicted.as_slice(), [crate::frame::LogicalWrite {
                    logical, value: crate::frame::LogicalValue::CompositeOpacity(o),
                }] if *logical == end && *o == value as f32));
            }
        }
        host.step(None);
        assert!(!host.edit_and_compare("path2", "x", json!(29.)).0);
        assert!(!host.edit_and_compare(&groups[3], "x", json!(17.)).0);
    }

    #[test]
    fn independent_group_uses_one_scalar_write_with_an_unrelated_page_mask() {
        let source = json!({"nodes":[
            {"id":"group","type":"group","opacity":0.65,"children":["ink"]}, path("ink", 0., 0., "#f00"),
            {"id":"clipped","type":"group","children":["other"],"clipMaskNodeId":"mask"},
            path("other", 0., 0., "#0f0"), path("mask", 0., 0., "#fff")],
            "pages":[{"id":"first","width":320,"height":240,"nodeIds":["group"]},
                     {"id":"second","width":320,"height":240,"nodeIds":["clipped","mask"]}]});
        for compact in [false, true] {
            let mut host = Host::from_source(source.clone());
            host.step(None);
            let state = host.resident.state.as_mut().unwrap();
            let entry = host.index.entry.as_mut().unwrap();
            if compact { assert!(entry.fit_cold_units(&state.scene, usize::MAX)); }
            assert!(entry.patchable.as_ref().unwrap().index.deps.mask_source_ids.contains("mask"));
            let row = state.scene.row_of("group").unwrap();
            let before = state.scene.stamps().clock();
            state.scene.write_row(row, |f| f.set("opacity", json!(0.375))).unwrap();
            state.scene.refresh_worlds();
            state.hashes = node_hashes_for_scene(&state.scene);
            host.index.record_parameter_writes(Some((before, state.scene.stamps().clock(), moda_scene::value::K::Opacity)));
            let (plan, _) = host.index.update(&state.scene, &state.hashes, &host.opts, 2, 2, &[row], false,
                &mut host.caches).expect("unrelated page mask permits the scalar write");
            let writes = host.index.take_written();
            assert_eq!((writes.placed.len(), writes.evicted.len()), if compact { (0, 1) } else { (1, 0) });
            let full = host.index.materialize(&plan, &state.scene, &state.hashes, &host.opts, &mut host.caches).unwrap();
            assert_eq!(format!("{full:?}"), host.fresh());
        }
    }

    #[test]
    fn replayable_blends_keep_destinations_after_releasing_the_recording() {
        let groups: Vec<_> = (0..128).map(|i| format!("blend{i}")).collect();
        let mut nodes = Vec::new();
        for group in &groups {
            let child = format!("{group}-body");
            nodes.push(json!({"id":group,"type":"group","blendMode":"overlay","children":[child]}));
            nodes.push(path(&child, 2., 10., "#873877"));
        }
        let roots: Vec<_> = groups.iter().map(String::as_str).collect();
        let mut host = Host::new(nodes, &roots);
        let (initial, _) = host.step(None);
        assert!(initial.items.iter().any(|i| matches!(i, crate::seams::DrawItem::BackdropSplit(s)
            if matches!(s.op, crate::seams::BackdropOp::BlendEnd { .. }))));
        let scene = host.resident.scene().unwrap();
        let budget = 20 * 1024;
        host.index.test_budget = Some(budget);
        let storage = crate::frame::PacketStorage { recording: budget, destinations: 1024 };
        assert!(!host.index.finish_draw(scene, &mut [], Some(storage)));
        let entry = host.index.entry.as_ref().expect("compact blend plan fits with destinations");
        let patch = entry.patchable.as_ref().unwrap();
        assert_eq!(patch.index.spans.len(), 0);
        assert!(patch.units.as_ref().unwrap().by_node.rows.is_empty());
        assert!(entry.bytes + storage.destinations <= budget);
        let full = host.index.materialize(&entry.plan.clone(), scene, host.resident.hashes().unwrap(),
            &host.opts, &mut host.caches).unwrap();
        assert_eq!(format!("{full:?}"), format!("{initial:?}"));
    }

    #[test]
    fn releasing_an_oversized_recording_preserves_a_partly_evicted_page() {
        crate::text_draw::install_text_builder(Some(std::rc::Rc::new(|node, model, z, clip, _| {
            let mut batch = super::tests::vector_text_batch(node.id(), 512, 512, 0);
            batch.model_matrix = model;
            batch.z = z;
            batch.page_clip_bounds = clip;
            Ok(crate::text_draw::TextBuildOutcome::Item(Box::new(crate::seams::DrawItem::Text(Arc::new(batch)))))
        })));
        let ids: Vec<_> = (0..8).map(|i| format!("ink{i}")).collect();
        let mut nodes: Vec<_> = ids.iter().map(|id| json!({"id":id,"type":"richtext",
            "width":60,"height":20,"htmlContent":id})).collect();
        let groups: Vec<_> = ids.iter().map(|id| format!("{id}-group")).collect();
        for (id, group) in ids.iter().zip(&groups) {
            nodes.push(json!({"id":group,"type":"group","opacity":0.65,"children":[id]}));
        }
        let roots: Vec<_> = groups.iter().map(String::as_str).collect();
        let mut host = Host::new(nodes, &roots);
        let (initial, _) = host.step(None);
        let scene = host.resident.scene().unwrap();
        // First leave an evictable mixture of resident and cold units, then ask
        // for more command space than even evicting the remaining units can free.
        let full = host.index.retained_size().0;
        host.index.test_budget = Some(full);
        let storage = |recording| Some(crate::frame::PacketStorage { destinations: 0, recording });
        assert!(host.index.finish_draw(scene, &mut [], storage(full / 4)));
        let (_, kept, total) = host.index.retained_size();
        assert!(kept < total && kept > 0);
        assert!(!host.index.finish_draw(scene, &mut [], storage(full)));
        assert!(host.index.entry.is_some());
        let held = host.index.entry.as_ref().unwrap().plan.clone();
        let restored = host.index.materialize(&held, scene, host.resident.hashes().unwrap(),
            &host.opts, &mut host.caches).unwrap();
        assert_eq!(format!("{restored:?}"), format!("{initial:?}"));
        assert!(host.edit_and_compare(&groups[0], "x", json!(13.)).0);
        crate::text_draw::install_text_builder(None);
    }

    #[test]
    fn evicted_composite_parameter_uses_its_normal_logical_slot() {
        use crate::seams::{BackdropOp, DrawItem};
        crate::text_draw::install_text_builder(Some(std::rc::Rc::new(|node, model, z, clip, _| {
            let mut batch = super::tests::vector_text_batch(node.id(), 1024, 1024, 0);
            batch.model_matrix = model;
            batch.z = z;
            batch.page_clip_bounds = clip;
            Ok(crate::text_draw::TextBuildOutcome::Item(Box::new(DrawItem::Text(Arc::new(batch)))))
        })));
        let mut host = Host::new(vec![
            json!({"id":"label","type":"group","opacity":0.75,"children":["ink"]}),
            json!({"id":"ink","type":"richtext","width":60,"height":20,"htmlContent":"ink"}),
        ], &["label"]);
        host.index.test_budget = Some(24 * 1024);
        let (previous, _) = host.step(None);
        let end = previous.items.iter().position(|item| matches!(item, DrawItem::BackdropSplit(s)
            if matches!(s.op, BackdropOp::StackEnd { .. }))).unwrap();
        let (_, kept, total) = host.index.retained_size();
        assert!(total == previous.items.len() && kept < total);
        let state = host.resident.state.as_mut().unwrap();
        let row = state.scene.row_of("label").unwrap();
        let before = state.scene.stamps().clock();
        state.scene.write_row(row, |fields| fields.set("opacity", json!(0.25))).unwrap();
        state.scene.refresh_worlds();
        state.hashes = node_hashes_for_scene(&state.scene);
        host.index.record_parameter_writes(Some((before, state.scene.stamps().clock(), moda_scene::value::K::Opacity)));
        let (view, _) = host.index.update(&state.scene, &state.hashes, &host.opts, 2, 2, &[row],
            false, &mut host.caches).expect("cold group patches");
        let mut writes = host.index.take_written();
        assert!(writes.replaced.is_empty() && writes.placed.is_empty() && !writes.recollect);
        assert!(matches!(writes.evicted.as_slice(), [crate::frame::LogicalWrite { logical,
            value: crate::frame::LogicalValue::CompositeOpacity(0.25) }] if *logical == end));
        let map = host.index.addresses(&view).unwrap();
        assert!(map.physical(end).is_none(), "the scalar edit did not restore the source");
        writes.materialize(&map).unwrap();
        assert_eq!(writes.placed[0].at as usize, end);
        let restored = host.index.materialize(&view, &state.scene, &state.hashes, &host.opts, &mut host.caches).unwrap();
        assert_eq!(format!("{restored:?}"), host.fresh());
        crate::text_draw::install_text_builder(None);
    }

    #[test]
    fn filter_only_parameter_patches_match_fresh_capture_inputs() {
        for subtree in [false, true] {
            let mut body = path("body", 0., 40., "#ff7700");
            let mut group = json!({"id":"group","type":"group","children":["body"]});
            let owner = if subtree { "group" } else { "body" };
            (if subtree { &mut group } else { &mut body })["effects"] =
                json!([{"id":"fx","type":"layer-blur","radius":2}]);
            let mut host = Host::new(vec![group, body], &["group"]);
            host.step(None);
            for (step, effects) in [
                json!([{"id":"fx","type":"layer-blur","radius":7.5}]),
                json!([{"id":"fx","type":"filter-shadow","blur":4,"offset":{"x":2,"y":-3},"color":"#ff0080ff"}]),
                json!([{"id":"fx","type":"filter-shadow","blur":9,"offset":{"x":-4,"y":1},"color":"#ff0080ff"}]),
                json!([]),
            ].into_iter().enumerate() {
                let scene = &mut host.resident.state.as_mut().unwrap().scene;
                let before = scene.stamps().clock();
                let row = scene.row_of(owner).unwrap();
                scene.write_row(row, |fields| fields.set("effects", effects)).unwrap();
                scene.refresh_worlds();
                let hashes = node_hashes_for_scene(scene);
                let clock = scene.stamps().clock();
                host.index.record_parameter_writes(Some((before, clock, moda_scene::value::K::Effects)));
                let patched = host.index.patch(scene, &hashes, &host.opts, clock, clock,
                    &[row], &mut host.caches);
                let (fresh, warnings) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                    Some((&mut host.caches, &hashes))).unwrap();
                if step == 0 || step == 2 {
                    assert!(patched.is_some(), "parameter-only edit must patch, subtree={subtree} step={step}");
                }
                if let Some((plan, actual)) = patched {
                    let keys = |p: &CoveragePlan| p.items.iter().filter_map(|item| match item {
                        crate::seams::DrawItem::LayerBlurComposite { group, .. } =>
                            Some((group.cache_key, group.extent_canvas)),
                        _ => None,
                    }).collect::<Vec<_>>();
                    if step == 0 || step == 2 {
                        assert!(keys(&plan).iter().any(|(key, _)| key.is_some()));
                    }
                    assert_eq!(keys(&plan), keys(&fresh), "subtree={subtree} step={step}");
                    assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "subtree={subtree}");
                    assert_eq!(format!("{actual:?}"), format!("{warnings:?}"));
                } else {
                    host.step(None);
                }
            }
        }
    }

    fn check_group_edit_sequence(edits: impl IntoIterator<Item = (&'static str, &'static str, Value)>) {
        let mut host = Host::new(vec![
            json!({"id":"group","type":"group","children":["left","middle","right"],
                "effects":[{"id":"b","type":"layer-blur","radius":3}]}),
            path("left", 0., 0., "#3355ff"), path("middle", 25., 25., "#dd3311"),
            path("right", 100., 100., "#ff5533"), path("free", 150., 0., "#dd3311"),
        ], &["group","free"]);
        host.step(None);
        for (step, (owner, field, value)) in edits.into_iter().enumerate() {
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let before = scene.stamps().clock();
            let row = scene.row_of(owner).unwrap();
            scene.write_row(row, |fields| fields.set(field, value)).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene);
            let clock = scene.stamps().clock();
            host.index.record_parameter_writes((field == "effects").then_some((before, clock, moda_scene::value::K::Effects)));
            let (plan, _) = host.index.patch(scene, &hashes, &host.opts, clock, clock,
                &[row], &mut host.caches)
                .unwrap_or_else(|| panic!("edit {step} of {owner}.{field}"));
            let (fresh, _) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            let capture = |p: &CoveragePlan| p.items.iter().find_map(|i| match i {
                crate::seams::DrawItem::LayerBlurComposite { group, .. } => Some(group.clone()),
                _ => None,
            }).unwrap();
            let (a, b) = (capture(&plan), capture(&fresh));
            assert_eq!(a.extent_canvas, b.extent_canvas, "edit {step} of {owner}.{field}");
            assert_eq!((a.width_px, a.height_px), (b.width_px, b.height_px));
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "edit {step} of {owner}.{field}");
        }
    }

    #[test]
    fn group_radius_after_child_move_uses_current_bounds() {
        check_group_edit_sequence([
            ("right", "x", json!(185)),
            ("group", "effects", json!([{"id":"b","type":"layer-blur","radius":7}])),
        ]);
    }

    #[test]
    fn randomized_group_edits_match_a_fresh_plan() {
        let mut state = 0x84c39c17u32;
        check_group_edit_sequence((0..96).map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = 1 + state % 13;
            match state >> 30 {
                0 => ("right", "x", json!(50 + state % 180)),
                1 => ("left", "fills", json!([{"id":"fill","paint":{"type":"solid",
                    "color": format!("#{:06x}", state & 0xffffff)}}])),
                2 => ("group", "effects", json!([{"id":"b","type":"layer-blur","radius":n}])),
                _ => ("group", "effects", json!([{"id":"b","type":"filter-shadow","blur":n,
                    "offset":{"x":n,"y":-3},"color":"#ff0080ff"}])),
            }
        }));
    }

    #[test]
    fn mixed_nested_unit_writes_replay_like_cold_planning() {
        let mut host = Host::new(vec![
            path("free", 10., 10., "#255fe8"),
            json!({"id":"outer","type":"group","opacity":0.7,"children":["inner","sibling"]}),
            json!({"id":"inner","type":"group","children":["body"],
                "effects":[{"id":"blur","type":"layer-blur","radius":3}]}),
            path("body", 50., 60., "#ff7700"), path("sibling", 140., 50., "#00aa77"),
            json!({"id":"stationary","type":"group","children":["still"],
                "effects":[{"id":"blur","type":"layer-blur","radius":2}]}),
            path("still", 200., 150., "#a070ff"),
        ], &["free", "outer", "stationary"]);
        let (first, _) = host.step(None);
        let stationary = |plan: &CoveragePlan| plan.items.iter().find_map(|item| match item {
            crate::seams::DrawItem::LayerBlurComposite { node_id, group, .. } if node_id == "stationary" => Some(group.clone()),
            _ => None,
        }).expect("stationary capture");
        let kept = stationary(&first);
        drop(first);
        let mut random = 0x312d_ab87_u32;
        for step in 0..48 {
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            let id = ["outer", "inner", "body", "sibling"][((random >> 24) % 4) as usize];
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let row = scene.row_of(id).unwrap();
            scene.write_row(row, |fields| {
                fields.set("x", json!((random % 31) as f64));
                fields.set("y", json!(((random >> 8) % 17) as f64));
                if id == "outer" { fields.set("opacity", json!(0.2 + (random % 60) as f64 / 100.)); }
                if id == "body" || id == "sibling" {
                    fields.set("fills", json!([{"id":"fill","paint":{"type":"solid",
                        "color":if random & 1 == 0 {"#e02060"} else {"#3070d0"}}}]));
                }
            }).unwrap();
            let free = scene.row_of("free").unwrap();
            scene.write_row(free, |fields| { fields.set("x", json!(10 + step)); }).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene);
            let clock = scene.stamps().clock();
            let (plan, warnings) = host.index.update(scene, &hashes, &host.opts, clock, clock,
                &[scene.row_of(id).unwrap(), scene.row_of("free").unwrap()], false, &mut host.caches).expect("nested value writes patch");
            host.index.take_written();
            let (fresh, fresh_warnings) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "step {step}, {id}");
            assert_eq!(format!("{warnings:?}"), format!("{fresh_warnings:?}"));
            assert!(Arc::ptr_eq(&kept, &stationary(&plan)), "unreached units keep their prepared contents");
        }
    }

    #[test]
    fn nested_output_records_patch_leaf_and_composition_inputs_without_range_replay() {
        let mut host = Host::new(vec![
            json!({"id":"outer","type":"group","opacity":0.7,"children":["left","blur"],
                "effects":[{"id":"o","type":"layer-blur","radius":2}]}),
            json!({"id":"blur","type":"group","children":["inner","right"],
                "effects":[{"id":"b","type":"layer-blur","radius":3}]}),
            json!({"id":"inner","type":"group","opacity":0.5,"children":["moving","sibling"]}),
            path("left", 0., 0., "#2080cc"), path("moving", 25., 25., "#ee3080"),
            path("sibling", 40., 40., "#cc8020"), path("right", 100., 100., "#80cc20"),
            path("free", 150., 0., "#ffff00"),
        ], &["outer", "free"]);
        host.step(None);
        for step in 0..24 {
            let units = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref().unwrap();
            assert!(units.outputs.iter().any(Option::is_some), "nested ordinary output certifies its graph");
            let reused = units.segment.as_ref().unwrap().prepared.as_ref().unwrap().reused;
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let id = match step % 4 { 0 => "inner", 1 => "moving", 2 => "right", _ => "outer" };
            let row = scene.row_of(id).unwrap();
            scene.write_row(row, |fields| {
                if id == "inner" || id == "outer" { fields.set("opacity", json!(0.3 + step as f64 / 100.)); }
                else {
                    fields.set("x", json!(if id == "right" { 100. + step as f64 } else { 25. + step as f64 / 4. }));
                    fields.set("fills", json!([{"id":"fill","paint":{"type":"solid",
                        "color":if step % 2 == 0 {"#e02060"} else {"#3070d0"}}}]));
                }
            }).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene); host.generation += 1;
            let (plan, warnings) = host.index.update(scene, &hashes, &host.opts, host.generation, page_frame_key(host.generation, 0, &host.opts),
                &[scene.row_of(id).unwrap()], false, &mut host.caches).expect("nested value writes patch");
            host.index.take_written();
            let units = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref().unwrap();
            assert_eq!(units.segment.as_ref().unwrap().prepared.as_ref().unwrap().reused, reused,
                "step {step}: the range executor must not revisit unchanged children");
            let (fresh, fresh_warnings) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "step {step}, {id}");
            assert_eq!(format!("{warnings:?}"), format!("{fresh_warnings:?}"));
        }
    }

    #[test]
    fn filtered_child_records_patch_without_reassembling_the_capture() {
        let mut host = Host::new(vec![
            json!({"id":"blur","type":"group","children":["left","moving","right"],
                "effects":[{"id":"b","type":"layer-blur","radius":3}]}),
            path("left", 0., 0., "#2080cc"), path("moving", 25., 25., "#ee3080"),
            path("right", 100., 100., "#80cc20"), path("free", 150., 0., "#ffff00"),
        ], &["blur", "free"]);
        host.step(None);
        for step in 0..16 {
            let units = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref().unwrap();
            assert!(units.outputs.iter().any(Option::is_some), "ordinary output must certify its child partition");
            let reused = units.segment.as_ref().unwrap().prepared.as_ref().unwrap().reused;
            let child_storage = host.index.entry.as_ref().unwrap().plan.items.iter().find_map(|item| match item {
                crate::seams::DrawItem::LayerBlurComposite { group, .. } => Some(group.items.as_ptr()), _ => None,
            }).unwrap();
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let id = if step % 3 == 0 { "right" } else { "moving" };
            let row = scene.row_of(id).unwrap();
            scene.write_row(row, |fields| {
                fields.set("x", json!(if id == "right" { 100. + step as f64 } else { 25. + step as f64 / 4. }));
                fields.set("fills", json!([{"id":"fill","paint":{"type":"solid",
                    "color":if step % 2 == 0 {"#e02060"} else {"#3070d0"}}}]));
            }).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene);
            host.generation += 1;
            let (plan, warnings) = host.index.update(scene, &hashes, &host.opts, host.generation, page_frame_key(host.generation, 0, &host.opts),
                &[scene.row_of(id).unwrap()], false, &mut host.caches).expect("filtered child writes patch");
            let written = host.index.take_written();
            assert_eq!(written.replaced.len(), 1);
            assert!(matches!(written.replaced[0].1, crate::seams::DrawItem::Vacant), "nested updates retain their child storage");
            assert_eq!(plan.items.iter().find_map(|item| match item {
                crate::seams::DrawItem::LayerBlurComposite { group, .. } => Some(group.items.as_ptr()), _ => None,
            }), Some(child_storage), "a sparse update must not clone the capture's child vector");
            let units = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref().unwrap();
            assert_eq!(units.segment.as_ref().unwrap().prepared.as_ref().unwrap().reused, reused,
                "the range executor must not revisit unchanged children");
            let (fresh, fresh_warnings) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "step {step}");
            assert_eq!(format!("{warnings:?}"), format!("{fresh_warnings:?}"));
        }
    }

    #[test]
    fn nested_composition_outputs_follow_only_their_source_paint() {
        let mut host = Host::new(vec![
            json!({"id":"outer","type":"group","opacity":0.7,"children":["inner","a"]}),
            json!({"id":"inner","type":"group","opacity":0.5,"children":["b","c"]}),
            path("a", 0., 0., "#2080cc"), path("b", 10., 10., "#ee3080"),
            path("c", 5., 5., "#80cc20"), path("free", 50., 0., "#ffff00"),
        ], &["outer", "free"]);
        host.step(None);
        for step in 0..16 {
            let prior_reused = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref()
                .unwrap().segment.as_ref().unwrap().prepared.as_ref().unwrap().reused;
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let id = if step % 2 == 0 { "inner" } else { "outer" };
            let row = scene.row_of(id).unwrap();
            scene.write_row(row, |fields| fields.set("opacity", json!(0.2 + step as f64 / 40.))).unwrap();
            let free = scene.row_of("free").unwrap();
            scene.write_row(free, |fields| fields.set("x", json!(50 + step))).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene);
            let clock = scene.stamps().clock();
            let (plan, warnings) = host.index.update(scene, &hashes, &host.opts, clock, clock,
                &[scene.row_of(id).unwrap(), scene.row_of("free").unwrap()], false, &mut host.caches).expect("recorded paint outputs patch");
            let written = host.index.take_written();
            assert!(written.replaced.len() <= 3, "only the reached composition and free row write");
            let units = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref().unwrap();
            assert!(units.outputs.iter().any(Option::is_some));
            assert_eq!(units.segment.as_ref().unwrap().prepared.as_ref().unwrap().reused, prior_reused,
                "neither ancestor nor child composition paint visits prepared children");
            let (fresh, fresh_warnings) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "step {step}");
            assert_eq!(format!("{warnings:?}"), format!("{fresh_warnings:?}"));
        }
    }

    #[test]
    fn moved_child_updates_only_its_outputs_and_enclosing_bounds() {
        let mut nodes = vec![json!({"id":"outer","type":"group","opacity":0.7,"children":["inner"]}),
            json!({"id":"inner","type":"group","opacity":0.5,"children":(0..64).map(|i| format!("p{i}")).collect::<Vec<_>>()})];
        nodes.extend((0..64).map(|i| path(&format!("p{i}"), i as f64 * 2., 10., "#4080cc")));
        let mut host = Host::new(nodes, &["outer"]);
        host.step(None);
        for step in 0..24 {
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let id = if step % 2 == 0 { "p0" } else { "p63" };
            let row = scene.row_of(id).unwrap();
            scene.write_row(row, |fields| {
                fields.set("x", json!(20. + (step * 7 % 150) as f64 / 3.));
                fields.set("y", json!(5. + (step * 11 % 90) as f64 / 7.));
                fields.set("rotation", json!(step as f64 * 0.013));
            }).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene);
            let at = scene.stamps().clock();
            let (plan, warnings) = host.index.update(scene, &hashes, &host.opts, at, at,
                &[scene.row_of(id).unwrap()], false, &mut host.caches).expect("recorded child placement patches");
            let written = host.index.take_written();
            assert!(written.replaced.len() <= 5, "one child and two enclosing compositions");
            let units = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref().unwrap();
            assert!(units.outputs.iter().flatten().any(|records| records.has_leaves()));
            assert_eq!(units.segment.as_ref().unwrap().prepared.as_ref().unwrap().reused, 0,
                "no sibling record is visited");
            let (fresh, fresh_warnings) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "step {step}");
            assert_eq!(format!("{warnings:?}"), format!("{fresh_warnings:?}"));
        }
    }

    #[test]
    fn a_changed_page_lowering_input_reprepares_its_recorded_outputs() {
        let mut host = Host::new(vec![
            json!({"id":"fade","type":"group","opacity":0.5,"children":["a"]}),
            path("a", 10., 10., "#ff3040"), path("outside", 120., 20., "#3080ff"),
        ], &["fade", "outside"]);
        host.step(None);
        assert!(!host.edit_and_compare("fade", "opacity", json!(1.)).0);
    }

    #[test]
    fn changing_unit_output_count_keeps_unrelated_output_addresses() {
        let mut host = Host::new(vec![
            path("before", 0., 0., "#3020ff"),
            json!({"id":"fade","type":"group","opacity":0.5,"children":["nested"]}),
            json!({"id":"nested","type":"group","children":["a","b"]}),
            path("a", 10., 10., "#ff3040"), path("b", 20., 20., "#20ff80"),
            path("after", 100., 0., "#3050ff"),
            json!({"id":"standing","type":"group","opacity":0.7,"children":["last"]}),
            path("last", 150., 20., "#ff7030"),
        ], &["before", "fade", "after", "standing"]);
        host.step(None);
        let address = |plan: &CoveragePlan, id: &str| plan.items.iter().position(|item| drawitem_node_id(item) == Some(id)).unwrap();
        let initial = &host.index.entry.as_ref().unwrap().plan;
        let before = address(initial, "before");
        let after = address(initial, "after");
        let mut storage = None;
        for step in 0..25 {
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let id = match step % 5 { 0 | 2 => "fade", 1 | 3 => "nested", _ => "a" };
            let row = scene.row_of(id).unwrap();
            scene.write_row(row, |fields| {
                if id == "a" { fields.set("x", json!(step * 2)); }
                else { fields.set("opacity", json!(if step % 5 == 0 || step % 5 == 3 { 1.0 } else { 0.4 })); }
            }).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene);
            let at = scene.stamps().clock();
            let (plan, warnings) = host.index.update(scene, &hashes, &host.opts, at, at,
                &[scene.row_of(id).unwrap()], false, &mut host.caches).expect("output shape follows its retained source unit");
            assert_eq!(address(&plan, "before"), before);
            assert_eq!(address(&plan, "after"), after);
            if step == 4 { storage = Some(plan.items.len()); }
            if step > 4 { assert_eq!(Some(plan.items.len()), storage, "bounded storage reuses its slots"); }
            let written = host.index.take_written();
            assert!(!written.replaced.iter().any(|(slot, _)| *slot as usize == before || *slot as usize == after));
            let (fresh, fresh_warnings) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "step {step}");
            assert_eq!(format!("{warnings:?}"), format!("{fresh_warnings:?}"));
        }
    }

    #[test]
    fn new_composition_owns_its_existing_source_interval() {
        let mut host = Host::new(vec![
            path("before", 0., 0., "#3020ff"),
            json!({"id":"fade","type":"group","opacity":1.0,"children":["nested"]}),
            json!({"id":"nested","type":"group","children":["a","b"]}),
            path("a", 10., 10., "#ff3040"), path("b", 20., 20., "#20ff80"),
            path("after", 100., 0., "#3050ff"),
            json!({"id":"standing","type":"group","opacity":0.7,"children":["last"]}),
            path("last", 150., 20., "#ff7030"),
        ], &["before", "fade", "after", "standing"]);
        host.step(None);
        let address = |plan: &CoveragePlan, id: &str| plan.items.iter().position(|item| drawitem_node_id(item) == Some(id)).unwrap();
        let initial = &host.index.entry.as_ref().unwrap().plan;
        let before = address(initial, "before");
        let after = address(initial, "after");
        let mut storage = None;
        for step in 0..25 {
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let id = match step % 5 { 0 | 2 => "fade", 1 | 3 => "nested", _ => "a" };
            let row = scene.row_of(id).unwrap();
            scene.write_row(row, |fields| {
                if id == "a" { fields.set("x", json!(step * 2)); }
                else { fields.set("opacity", json!(if step % 5 == 2 || step % 5 == 3 { 1.0 } else { 0.4 })); }
            }).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene);
            let at = scene.stamps().clock();
            let (plan, warnings) = host.index.update(scene, &hashes, &host.opts, at, at,
                &[scene.row_of(id).unwrap()], false, &mut host.caches).expect("output shape follows its retained source unit");
            assert_eq!(address(&plan, "before"), before);
            assert_eq!(address(&plan, "after"), after);
            if step == 4 { storage = Some(plan.items.len()); }
            if step > 4 { assert_eq!(Some(plan.items.len()), storage, "bounded storage reuses its slots"); }
            let written = host.index.take_written();
            if step % 5 == 4 { assert_eq!(written.replaced.len(), 1,
                "after the compositions end, one child still owns one output write"); }
            assert!(!written.replaced.iter().any(|(slot, _)| *slot as usize == before || *slot as usize == after));
            let (fresh, fresh_warnings) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "step {step}");
            assert_eq!(format!("{warnings:?}"), format!("{fresh_warnings:?}"));
        }
    }

    #[test]
    fn child_content_and_paint_reenter_only_their_emission_rows() {
        let mut nodes = vec![json!({"id":"outer","type":"group","opacity":0.7,"children":["inner"]}),
            json!({"id":"inner","type":"group","opacity":0.5,"children":(0..64).map(|i| format!("p{i}")).collect::<Vec<_>>()})];
        nodes.extend((0..64).map(|i| path(&format!("p{i}"), i as f64 * 2., 10., "#4080cc")));
        let mut host = Host::new(nodes, &["outer"]);
        host.step(None);
        for step in 0..24 {
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let id = if step % 2 == 0 { "p0" } else { "p63" };
            let row = scene.row_of(id).unwrap();
            scene.write_row(row, |fields| {
                fields.set("x", json!(20. + (step * 7 % 150) as f64 / 3.));
                fields.set("y", json!(5. + (step * 11 % 90) as f64 / 7.));
                fields.set("rotation", json!(step as f64 * 0.013));
                fields.set("opacity", json!(if step % 4 == 0 { 1.0 } else { 0.45 }));
                if step % 3 == 0 {
                    fields.set("fills", json!([{"id":"fill","paint":{"type":"solid","color":if step % 2 == 0 {"#ef3070"} else {"#20b0e0"}}}]));
                } else if step % 3 == 1 {
                    fields.set("points", json!([{"type":"move","x":0,"y":0},
                        {"type":"line","x":30+step,"y":0},{"type":"line","x":70,"y":60},
                        {"type":"line","x":0,"y":40+step}]));
                }
            }).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene);
            let at = scene.stamps().clock();
            let (plan, warnings) = host.index.update(scene, &hashes, &host.opts, at, at,
                &[scene.row_of(id).unwrap()], false, &mut host.caches).expect("recorded child content patches");
            let written = host.index.take_written();
            assert!(written.replaced.len() <= 5, "one child and two enclosing compositions");
            let units = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref().unwrap();
            assert!(units.outputs.iter().flatten().any(|records| records.has_leaves()));
            assert_eq!(units.segment.as_ref().unwrap().prepared.as_ref().unwrap().reused, 0,
                "no sibling record is visited");
            let (fresh, fresh_warnings) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "step {step}");
            assert_eq!(format!("{warnings:?}"), format!("{fresh_warnings:?}"));
        }
    }

    #[test]
    fn captured_rows_reuse_output_and_follow_ordinary_placement_recipes() {
        let mut nodes = vec![json!({"id":"fade","type":"group","opacity":0.6,"children":["wrapper"]}),
            json!({"id":"wrapper","type":"group","children":(0..40).map(|i| format!("p{i}")).collect::<Vec<_>>()})];
        nodes.extend((0..40).map(|i| path(&format!("p{i}"), i as f64, 0., "#4080cc")));
        let mut host = Host::new(nodes, &["fade"]);
        host.step(None);
        let mut reused = 0;
        for step in 0..12 {
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let id = if step % 3 == 2 { "p7" } else { "fade" };
            let row = scene.row_of(id).unwrap();
            scene.write_row(row, |fields| match step % 3 {
                0 => fields.set("opacity", json!(0.3 + step as f64 / 50.)),
                1 => { fields.set("scaleX", json!(1. + step as f64 / 100.)); fields.set("x", json!(step)); },
                _ => fields.set("fills", json!([{"id":"f","paint":{"type":"solid","color":"#e03070"}}])),
            }).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene);
            host.generation += 1;
            let (plan, warnings) = host.index.update(scene, &hashes, &host.opts, host.generation, page_frame_key(host.generation, 0, &host.opts),
                &[scene.row_of(id).unwrap()], false, &mut host.caches).expect("captured row writes patch");
            let written = host.index.take_written();
            if step % 3 == 0 { assert!(written.replaced.len() < 10, "fade must retain child GPU records"); }
            let prepared = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref()
                .unwrap().segment.as_ref().unwrap().prepared.as_ref().unwrap();
            if step % 3 != 1 { assert_eq!(prepared.reused, reused, "composition and child paint update outputs without visiting siblings"); }
            reused = prepared.reused;
            let (fresh, fresh_warnings) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "step {step}");
            assert_eq!(format!("{warnings:?}"), format!("{fresh_warnings:?}"));
        }
    }

    #[test]
    fn changing_outputs_patch_their_actual_mask_read_edges() {
        let mut host = Host::new(vec![
            json!({"id":"clip","type":"container","width":200,"height":160,"clipContent":true,
                "children":["standing","body"]}),
            path("standing", -5., 0., "#3050ff"),
            json!({"id":"body","type":"container","x":20,"y":20,"width":70,"height":60}),
        ], &["clip"]);
        host.step(None);
        let mut read_sets = std::collections::HashSet::new();
        for step in 0..16 {
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let id = if step % 4 < 2 { "body" } else { "clip" };
            let row = scene.row_of(id).unwrap();
            scene.write_row(row, |fields| {
                fields.set("x", json!(match step % 4 { 0 => 150., 1 => 20., 2 => 5., _ => 0. }));
                if id == "body" {
                    fields.set("fills", if step % 4 == 0 {
                        json!([{"id":"fill","paint":{"type":"solid","color":"#ff3040"}}])
                    } else { json!([]) });
                }
            }).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene);
            let at = scene.stamps().clock();
            let (plan, warnings) = host.index.update(scene, &hashes, &host.opts, at, at,
                &[scene.row_of(id).unwrap()], false, &mut host.caches).expect("changed query reads patch their dependency edges");
            let units = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref().unwrap();
            let ui = units.by_node.get(scene, "body").unwrap();
            read_sets.insert(units.segment.as_ref().unwrap().units[ui].details().mask_reads.clone());
            let (fresh, fresh_warnings) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "step {step}");
            assert_eq!(format!("{warnings:?}"), format!("{fresh_warnings:?}"));
        }
        assert!(read_sets.len() > 1, "the oracle exercises a changed query read set");
    }

    #[test]
    fn nested_clip_tracks_a_generated_source_and_a_base_that_also_draws() {
        let mut host = Host::new(vec![
            json!({"id":"card","type":"group","opacity":0.6,"children":["base","outer"]}),
            path("base", 4., 2., "#909090"),
            json!({"id":"outer","type":"group","clipToBelow":true,"clipMaskNodeId":"vector","children":["vector","inner"]}),
            path("vector", 0., 0., "#ffffff"),
            json!({"id":"inner","type":"group","clipMaskNodeId":"__page_bg_page","children":["ink"]}),
            path("ink", 5., 3., "#4070ef"),
        ], &["card"]);
        let (plan, _) = host.step(None);
        assert_eq!(plan.masks.len(), 3);
        for source in ["base", "vector", "__page_bg_page"] {
            assert!(plan.masks.iter().any(|mask| mask.source_node_id == source));
        }
        for (id, field, value) in [("card","x",12.), ("card","scaleY",0.8),
            ("base","x",8.), ("inner","x",4.), ("card","opacity",0.7)] {
            host.edit_and_compare(id, field, json!(value));
        }
    }

    #[test]
    fn a_hidden_clip_cannot_certify_a_shared_blend_source() {
        let mut host = Host::new(vec![
            json!({"id":"moving","type":"group","opacity":0,"children":["clipped"]}),
            json!({"id":"clipped","type":"group","clipMaskNodeId":"source","children":["source","ink"]}),
            path("source", 0., 0., "#ffffff"), path("ink", 3., 5., "#4060ef"),
            path("end", 20., 3., "#ef6060"),
            json!({"id":"blend","type":"blend","sourceIds":["source","end"],"spacing":"steps","steps":2}),
        ], &["moving", "end", "blend"]);
        host.step(None);
        assert!(!host.edit_and_compare("moving", "x", json!(12.)).0);
        host.edit_and_compare("source", "y", json!(3.));
        host.edit_and_compare("moving", "opacity", json!(0.6));
        host.edit_and_compare("moving", "scaleX", json!(1.5));
    }

    #[test]
    fn moving_a_hidden_clip_tracks_every_referring_group() {
        let mut host = Host::new(vec![
            json!({"id":"moving","type":"group","opacity":0,"children":["inner"]}),
            json!({"id":"inner","type":"group","clipMaskNodeId":"source","children":["source","first"]}),
            json!({"id":"other","type":"group","opacity":0,"clipMaskNodeId":"source","children":["second"]}),
            path("source", 0., 0., "#ffffff"), path("first", 0., 0., "#3060ff"),
            path("second", 4., 2., "#ef5030"),
        ], &["moving", "other"]);
        let (seed, _) = host.step(None);
        assert!(seed.masks.is_empty());
        for (field, value) in [("x", 12.), ("scaleX", 0.75)] {
            assert!(host.edit_and_compare("moving", field, json!(value)).0, "{field}");
        }
        // A different group can start reading the hidden source. A new mask slot requires
        // ordinary collection. Subsequent edits to its source must update the visible output.
        host.edit_and_compare("other", "opacity", json!(0.75));
        assert_eq!(host.index.entry.as_ref().unwrap().plan.masks.len(), 1);
        host.edit_and_compare("source", "x", json!(27.));
        host.edit_and_compare("source", "points", json!([{"type":"move","x":0,"y":0},
            {"type":"line","x":38,"y":5},{"type":"line","x":20,"y":35}]));
        host.edit_and_compare("moving", "opacity", json!(0.6));
        assert_eq!(host.index.entry.as_ref().unwrap().plan.masks.len(), 2);
        host.edit_and_compare("moving", "y", json!(32.));
        host.edit_and_compare("moving", "opacity", json!(0.4));
    }

    #[test]
    fn mask_source_writes_reprepare_recorded_consumers() {
        for captured in [false, true] {
            let mut owner = json!({"id":"owner","type":"group","children":["a","b","source-parent"]});
            if captured { owner["opacity"] = json!(0.6); }
            let mut host = Host::new(vec![owner,
                json!({"id":"a","type":"group","clipMaskNodeId":"source","children":["pa"]}),
                json!({"id":"b","type":"group","clipMaskNodeId":"source","children":["pb"]}),
                json!({"id":"source-parent","type":"group","children":["source"]}),
                path("pa", 0., 0., "#ff3060"), path("pb", 5., 5., "#3080ff"),
                path("source", 0., 0., "#ffffff"),
            ], &["owner"]);
            host.step(None);
            for step in 0..12 {
                let id = if step % 3 == 0 { "source-parent" } else { "source" };
                let scene = &mut host.resident.state.as_mut().unwrap().scene;
                let row = scene.row_of(id).unwrap();
                scene.write_row(row, |fields| { fields.set("x", json!(step + 1)); }).unwrap();
                scene.refresh_worlds();
                let hashes = node_hashes_for_scene(scene);
                let clock = scene.stamps().clock();
                assert!(host.index.place(scene, &host.opts, clock, clock, &[row], &mut host.caches).is_none(),
                    "a source or its ancestor must reach the mask worklist instead of only placing body spans");
                let (plan, warnings) = host.index.update(scene, &hashes, &host.opts, clock, clock,
                    &[scene.row_of(id).unwrap()], false, &mut host.caches).expect("mask source follows its recorded consumers");
                let written = host.index.take_written();
                assert!(written.masks_changed);
                let (fresh, fresh_warnings) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                    Some((&mut PlanCaches::default(), &hashes))).unwrap();
                assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "captured={captured}, step={step}");
                assert_eq!(format!("{warnings:?}"), format!("{fresh_warnings:?}"));
            }
        }
    }

    #[test]
    fn rendered_mask_clock_admission_waits_for_the_prepared_capture() {
        for uses_clock in [false, true] {
            let mut host = Host::new(vec![
                json!({"id":"clip","type":"group","clipMaskNodeId":"source","children":["body"]}),
                json!({"id":"source","type":"group","children":["ink"]}),
                path("body", 0., 0., "#ff3060"), path("ink", 0., 0., "#ffffff"),
            ], &["clip", "source"]);
            let (plan, _) = host.step(None);
            assert!(plan_uses_shader_time(&plan), "an unprepared capture is conservative");
            assert!(host.index.held(1, 1.0).is_none());
            for mask in plan.masks.iter() {
                if let Some(rendered) = mask.image.as_ref().and_then(|image| image.rendered.as_ref()) {
                    rendered.shader_clock.set(uses_clock).unwrap();
                }
            }
            assert_eq!(plan_uses_shader_time(&plan), uses_clock);
            assert_eq!(host.index.held(1, 1.0).is_some(), !uses_clock);
        }
    }

    #[test]
    fn source_subtree_paint_and_placement_match_fresh_masks() {
        let mut host = Host::new(vec![
            json!({"id":"clip","type":"group","clipMaskNodeId":"source","children":["body"]}),
            json!({"id":"source","type":"group","children":["ink"]}),
            path("body", 0., 0., "#ff3060"), path("ink", 0., 0., "#ffffff"),
        ], &["clip", "source"]);
        host.step(None);
        for step in 0..12 {
            // This CPU fixture has only solid paths; GPU integration establishes the same fact
            // from the actual capture executor before admitting the next frame.
            for mask in host.index.entry.as_ref().unwrap().plan.masks.iter() {
                if let Some(rendered) = mask.image.as_ref().and_then(|image| image.rendered.as_ref()) {
                    let _ = rendered.shader_clock.set(false);
                }
            }
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let row = scene.row_of("ink").unwrap();
            scene.write_row(row, |fields| {
                fields.set("x", json!(step + 1));
                fields.set("fills", json!([{"id":"fill","paint":{"type":"solid","color":if step % 2 == 0 {"#ffffff80"} else {"#ffffffc0"}}}]));
            }).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene);
            let clock = scene.stamps().clock();
            let (plan, _) = host.index.update(scene, &hashes, &host.opts, clock, clock,
                &[scene.row_of("ink").unwrap()], false, &mut host.caches).expect("source subtree input patches");
            host.index.take_written();
            let (fresh, _) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "step {step}");
            assert_eq!(plan.masks, fresh.masks);
        }
    }

    #[test]
    fn group_matte_tracks_authored_row_edits() {
        check_group_matte_edits(false);
    }

    #[test]
    fn a_group_matte_tracks_authored_rows_and_movie_pixels_together() {
        const CHILD: &str = "MODA_TEST_MIXED_GROUP_MATTE";
        if std::env::var_os(CHILD).is_none() {
            let name = std::thread::current().name().unwrap().to_owned();
            assert!(std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", &name]).env(CHILD, "1").status().unwrap().success());
            return;
        }
        check_group_matte_edits(true);
    }

    fn check_group_matte_edits(movie: bool) {
        use moda_assets::{engine_seam, AssetEntry, AssetStore, PreparedImage};
        let publish = |value| {
            let mut store = AssetStore::default();
            store.insert("sibling.mp4".into(), AssetEntry::Loaded(Arc::new(
                PreparedImage::from_premultiplied(vec![value, value, value, 255], 1, 1))));
            engine_seam::install_store(Arc::new(store));
        };
        let ids: Vec<_> = (0..96).map(|i| format!("ink{i}")).collect();
        let mut nodes = vec![
            json!({"id":"clip","type":"group","clipMaskNodeId":"source","children":["source","body"]}),
            json!({"id":"source","type":"group","maskType":"luminance","children":ids}),
            path("body", 0., 0., "#ff3060"),
        ];
        nodes.extend(ids.iter().enumerate().map(|(i,id)| path(id, 2.0 * i as f64, 10., "#ffffff")));
        if movie {
            publish(0);
            nodes[3 + 24]["fills"] = json!([{"id":"f","paint":{"type":"image","src":"sibling.mp4"}}]);
        }
        let mut host = Host::new(nodes, &["clip"]);
        let (page, _) = host.step(None);
        let source_of = |p: &CoveragePlan| p.masks[0].image.as_ref().unwrap().rendered.as_ref().unwrap().clone();
        let source = source_of(&page);
        let plan = source.prepare(&page.masks[0], 1., None, 0., true).unwrap();
        source.shader_clock.set(false).unwrap();
        drop((source, plan, page));
        for step in 0..8 {
            if movie { publish(step as u8 + 1); }
            let state = host.resident.state.as_mut().unwrap();
            let row = state.scene.row_of("ink40").unwrap();
            state.scene.write_row(row, |fields| {
                fields.set("x", json!(85 + step));
                fields.set("fills", json!([{"id":"f","paint":{"type":"solid","color": if step % 2 == 0 {"#554421"} else {"#e6c802"}}}]));
            }).unwrap();
            state.scene.refresh_worlds();
            let clock = state.scene.stamps().clock();
            let (page, _) = host.index.update(&state.scene, &RowHashes::default(), &host.opts, clock,
                page_frame_key(clock, 0, &host.opts), &[row, row], false, &mut host.caches).expect("parent patched only its source");
            let source = source_of(&page);
            let plan = source.prepare(&page.masks[0], 1., None, 0., true).unwrap();
            let _ = source.shader_clock.set(plan_uses_shader_time(&plan));
            let (cold, _, _) = crate::frame::build_coverage_plan_indexed(&state.scene, 0, &host.opts, None).unwrap();
            let expected = source_of(&cold).prepare(&cold.masks[0], 1., None, 0., false).unwrap();
            assert_eq!(page.masks, cold.masks, "source identity and bounds at {step}");
            assert_eq!(format!("{plan:?}"), format!("{expected:?}"), "drawn source at {step}");
            assert!(host.index.take_written().masks_changed);
        }
        // Keep an earlier revision while moving a path beyond the old capture
        // window. Its camera and pixels must survive the new collection.
        let previous = host.index.entry.as_ref().unwrap().plan.clone();
        let previous_source = source_of(&previous);
        let view = previous_source.prepare(&previous.masks[0], 1., None, 0., true).unwrap();
        let state = host.resident.state.as_mut().unwrap();
        let row = state.scene.row_of("ink40").unwrap();
        state.scene.write_row(row, |f| f.set("x", json!(300))).unwrap();
        state.scene.refresh_worlds();
        let clock = state.scene.stamps().clock();
        let (page, _) = host.index.update(&state.scene, &RowHashes::default(), &host.opts, clock,
            page_frame_key(clock, 0, &host.opts), &[row], false, &mut host.caches).unwrap();
        let (cold, _, _) = crate::frame::build_coverage_plan_indexed(&state.scene, 0, &host.opts, None).unwrap();
        assert_ne!(page.masks[0].size, previous.masks[0].size);
        assert_eq!(page.masks, cold.masks);
        assert_eq!(format!("{view:?}"), format!("{:?}", previous_source.prepare(&previous.masks[0], 1., None, 0., false).unwrap()));
    }

    fn release_clip_after_first_rendered_mask() -> [Arc<CoveragePlan>; 2] {
        let mut host = Host::new(vec![
            json!({"id":"clip","type":"group","clipMaskNodeId":"matte","children":["matte","body"]}),
            json!({"id":"matte","type":"group","maskType":"vector","width":60,"height":60,"children":["ink","edge"]}),
            json!({"id":"outside","type":"group","hidden":true,"clipMaskNodeId":"edge","children":[]}),
            path("ink", 0., 0., "#ffffff"), path("edge", 90., 0., "#ffffff"), path("body", 60., 0., "#ff3060"),
        ], &["clip", "outside"]);
        let (page, _) = host.step(None);
        assert!(page.masks.iter().all(|mask| mask.image.as_ref().is_none_or(|i| i.rendered.is_none())));

        host.edit_and_compare("matte", "maskType", json!("luminance"));
        let before = host.index.entry.as_ref().unwrap().plan.clone();
        let source = before.masks[0].image.as_ref().unwrap().rendered.as_ref().unwrap();
        let captured = source.prepare(&before.masks[0], 1., None, 0., true).unwrap();
        source.shader_clock.set(plan_uses_shader_time(&captured)).unwrap();
        host.edit_and_compare("outside", "clipMaskNodeId", Value::Null);
        let after = &host.index.entry.as_ref().unwrap().plan;
        let visible = after.masks[0].image.as_ref().unwrap().rendered.as_ref().unwrap();
        assert_ne!(source.cache_key, visible.cache_key);
        let released = visible.prepare(&after.masks[0], 1., None, 0., true).unwrap();
        assert!(released.items.iter().any(|item| drawitem_node_id(item) == Some("edge")));
        assert!(!captured.items.iter().any(|item| drawitem_node_id(item) == Some("edge")));
        [before, after.clone()]
    }

    #[test]
    fn first_rendered_mask_after_a_row_edit_observes_existing_hidden_clip_owners() {
        release_clip_after_first_rendered_mask();
    }

    #[test]
    #[ignore = "requires a GPU"]
    fn first_rendered_mask_reveals_the_released_edge_on_screen() {
        let engine = crate::Engine::new(crate::GpuContext::new(crate::BackendChoice::Auto).unwrap()).unwrap();
        let [before, after] = release_clip_after_first_rendered_mask().map(|plan| {
            let texture = engine.render_retained_plan_to_texture(&plan).unwrap();
            crate::readback::read_texture_rgba(&engine.gpu.device, &engine.gpu.queue, &texture,
                plan.width_px, plan.height_px).unwrap()
        });
        let pixel = |x: usize| (30 * 320 + x) * 4..(30 * 320 + x + 1) * 4;
        assert_ne!(before[pixel(110)], before[pixel(65)], "the edge starts clipped");
        assert_eq!(after[pixel(110)], before[pixel(65)], "the released edge reveals the body's fill");
        assert_eq!(after[pixel(65)], before[pixel(65)], "the original ink stays visible");
    }

    #[test]
    fn outside_hidden_group_can_gain_and_release_a_clip_into_an_existing_matte() {
        let mut host = Host::new(vec![
            json!({"id":"clip","type":"group","clipMaskNodeId":"matte","children":["matte","body"]}),
            json!({"id":"matte","type":"group","maskType":"luminance","children":["ink"]}),
            json!({"id":"outside","type":"group","hidden":true,"children":[]}),
            path("ink", 1., 0., "#ffffff"), path("body", 0., 0., "#ff3060"),
        ], &["clip", "outside"]);
        let mut keys = Vec::new();
        for binding in [Some("ink"), None] {
            let (page, _) = host.step(None);
            let source = page.masks[0].image.as_ref().unwrap().rendered.as_ref().unwrap();
            source.shader_clock.set(false).unwrap();
            keys.push(source.cache_key.unwrap());
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let at = scene.stamps().clock();
            let row = scene.row_of("outside").unwrap();
            scene.write_row(row, |fields| fields.set("clipMaskNodeId", json!(binding))).unwrap();
            scene.refresh_worlds();
            assert!(!scene.stamps().composition_current(row, at));
            let now = scene.stamps().clock();
            assert!(host.index.update(scene, &RowHashes::default(), &host.opts, now, now,
                &[row], false, &mut host.caches).is_none(), "clip membership changed");
        }
        let (page, _) = host.step(None);
        let restored = page.masks[0].image.as_ref().unwrap().rendered.as_ref().unwrap();
        assert_eq!(Some(keys[0]), restored.cache_key);
        assert_ne!(keys[0], keys[1]);
    }

    #[test]
    fn changing_a_mask_source_binding_reseeds_dependencies_even_when_coverage_is_equal() {
        let mut host = Host::new(vec![
            json!({"id":"fade","type":"group","opacity":0.6,"children":["a","b","c","m1","m2"]}),
            json!({"id":"a","type":"group","clipMaskNodeId":"m1","children":["pa"]}),
            json!({"id":"b","type":"group","clipMaskNodeId":"m2","children":["pb"]}),
            json!({"id":"c","type":"group","clipMaskNodeId":"m1","children":["pc"]}),
            path("pa", 0., 0., "#ff3060"), path("pb", 5., 5., "#3080ff"), path("pc", 10., 10., "#30dd60"),
            path("m1", 0., 0., "#ffffff"), path("m2", 0., 0., "#ffffff"),
        ], &["fade"]);
        host.step(None);
        let scene = &mut host.resident.state.as_mut().unwrap().scene;
        let row = scene.row_of("a").unwrap();
        scene.write_row(row, |fields| fields.set("clipMaskNodeId", json!("m2"))).unwrap();
        scene.refresh_worlds();
        let hashes = node_hashes_for_scene(scene);
        let clock = scene.stamps().clock();
        assert!(host.index.update(scene, &hashes, &host.opts, clock, clock,
            &[scene.row_of("a").unwrap()], false, &mut host.caches).is_none(), "changed source membership requires a fresh index");
    }

    #[test]
    fn masked_capture_sources_patch_without_replaying_stationary_children() {
        for (outer_mask, nested) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut root = json!({"id":"capture","type":"group","children":["clip"],
                "effects":[{"id":"blur","type":"layer-blur","radius":3}]});
            let mut clip = json!({"id":"clip","type":"group","children":["left","body","right"]});
            if nested { clip["effects"] = json!([{"id":"inner-blur","type":"layer-blur","radius":2}]); }
            if outer_mask { root["clipMaskNodeId"] = json!("mask"); }
            else { clip["clipMaskNodeId"] = json!("mask"); }
            let prefix = json!({"id":"prefix","type":"container","width":100,"height":100,
                "clipContent":true,"cornerRadius":10,"children":["prefix-body"]});
            let mut host = Host::new(vec![prefix, path("prefix-body", 0., 0., "#cccccc"), root, clip, path("mask", 0., 0., "#ffffff"),
                path("left", 0., 0., "#3090ff"), path("body", 5., 10., "#ff3060"),
                path("right", 120., 100., "#30ee80")], &["prefix","capture","mask"]);
            host.step(None);
            for step in 0..12 {
                let id = if step == 5 { "mask" } else { "body" };
                let (before, storage) = {
                    let units = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref().unwrap();
                    let ui = units.by_node.get(host.resident.scene().unwrap(), "capture").unwrap();
                    assert!(units.outputs[ui].is_some(), "masked capture records its output graph: outer={outer_mask}, nested={nested}, step={step}");
                    (units.segment.as_ref().unwrap().prepared.as_ref().unwrap().reused,
                        units.outputs[ui].as_ref().unwrap().leaf_storage())
                };
                let scene = &mut host.resident.state.as_mut().unwrap().scene;
                scene.write_row(scene.row_of(id).unwrap(), |fields| {
                    fields.set("x", json!(5 + step));
                    if id == "mask" {
                        fields.set("points", json!([{"type":"move","x":0,"y":0},{"type":"line","x":65,"y":0},
                            {"type":"line","x":65,"y":60},{"type":"line","x":0,"y":60}]));
                    }
                    if step % 3 == 0 || id == "mask" { fields.set("fills", json!([{"id":"fill","paint":{"type":"solid","color":"#60bbf0"}}])); }
                }).unwrap();
                scene.refresh_worlds();
                let hashes = node_hashes_for_scene(scene); let clock = scene.stamps().clock();
                let (plan, warnings) = host.index.update(scene, &hashes, &host.opts, clock, clock,
                    &[scene.row_of(id).unwrap()], false, &mut host.caches).expect("masked source patch");
                host.index.take_written();
                let (fresh, fresh_warnings) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                    Some((&mut PlanCaches::default(), &hashes))).unwrap();
                assert_eq!(format!("{plan:?}"), format!("{fresh:?}"), "outer_mask={outer_mask}, nested={nested}, step={step}");
                assert_eq!(plan.masks, fresh.masks);
                assert_eq!(format!("{warnings:?}"), format!("{fresh_warnings:?}"));
                if id == "mask" {
                    let units = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref().unwrap();
                    assert_eq!(storage, units.outputs[units.by_node.get(host.resident.scene().unwrap(), "capture").unwrap()].as_ref().unwrap().leaf_storage(),
                        "new mask content with unchanged child reads must retain the reverse index");
                }
                if id == "body" {
                    let units = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref().unwrap();
                    assert_eq!(before, units.segment.as_ref().unwrap().prepared.as_ref().unwrap().reused,
                        "a changed child must not replay stationary rows");
                }
            }
        }
    }

    #[test]
    fn unit_mask_inputs_include_reads_inside_captures() {
        let mut host = Host::new(vec![
            json!({"id":"capture","type":"group","children":["clip"],
                "effects":[{"id":"blur","type":"layer-blur","radius":3}]}),
            json!({"id":"clip","type":"container","width":180,"height":140,
                "clipContent":true,"cornerRadius":18,"children":["body"]}),
            path("body", 5., 10., "#ff3060"),
        ], &["capture"]);
        let (plan, _) = host.step(None);
        assert!(!plan.masks.is_empty());
        let units = host.index.entry.as_ref().unwrap().patchable.as_ref().unwrap().units.as_ref().unwrap();
        let seg = units.segment.as_ref().unwrap();
        let ui = units.by_node.get(host.resident.scene().unwrap(), "capture").unwrap();
        let unit = &seg.units[ui];
        assert!(!unit.details().mask_reads.is_empty(), "capture records the mask reads of its contents");
        for &mi in &unit.details().mask_reads {
            assert!(units.mask_users[mi].contains(&ui));
        }
        assert!(plan.items[seg.prefix + unit.item_start..seg.prefix + unit.item_start + unit.item_len]
            .iter().all(|item| crate::camera_slot::item_mask_id(item).is_none()),
            "the nested mask is invisible to a top-level draw-item scan");
    }

    #[test]
    fn moving_clip_updates_mask_values_and_all_recorded_consumers() {
        let mut host = Host::new(vec![
            json!({"id":"moving","type":"group","children":["clip"]}),
            json!({"id":"clip","type":"container","width":210,"height":160,"clipContent":true,
                "cornerRadius":18,"children":["fade","plain","nested"]}),
            json!({"id":"fade","type":"group","opacity":0.6,"children":["body"]}),
            path("body", 5., 10., "#ff3060"), path("plain", 150., 60., "#3080ff"),
            json!({"id":"nested","type":"container","x":30,"y":40,"width":130,"height":100,
                "clipContent":true,"cornerRadius":12,"children":["fade2"]}),
            json!({"id":"fade2","type":"group","opacity":0.7,"children":["body2"]}),
            path("body2", 60., 15., "#30dd60"),
        ], &["moving"]);
        let (initial, _) = host.step(None);
        assert!(!initial.masks.is_empty());
        drop(initial);
        for x in [5., 17., -10., 0., 8.5] {
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let row = scene.row_of("moving").unwrap();
            scene.write_row(row, |fields| fields.set("x", json!(x))).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene);
            let clock = scene.stamps().clock();
            let (plan, _) = host.index.update(scene, &hashes, &host.opts, clock, clock,
                &[scene.row_of("moving").unwrap()], false, &mut host.caches).expect("clip move patches its consumers");
            assert!(host.index.take_written().masks_changed);
            let (fresh, warnings) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert!(warnings.is_empty(), "{warnings:?}");
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"));
        }
    }

    #[test]
    fn composition_paint_does_not_invalidate_unchanged_mask_sources() {
        let mut host = Host::new(vec![
            json!({"id":"fade","type":"group","opacity":0.6,"children":["clip"]}),
            json!({"id":"clip","type":"group","children":["mask","body"],"clipMaskNodeId":"mask"}),
            path("mask", 10., 15., "#ffffff"), path("body", 0., 0., "#3070e0"),
        ], &["fade"]);
        host.step(None);
        for opacity in [0.3, 0.7, 0.4] {
            let scene = &mut host.resident.state.as_mut().unwrap().scene;
            let row = scene.row_of("fade").unwrap();
            scene.write_row(row, |fields| fields.set("opacity", json!(opacity))).unwrap();
            scene.refresh_worlds();
            let hashes = node_hashes_for_scene(scene);
            host.generation += 1;
            let (plan, _) = host.index.update(scene, &hashes, &host.opts, host.generation, page_frame_key(host.generation, 0, &host.opts),
                &[scene.row_of("fade").unwrap()], false, &mut host.caches).expect("composition paint does not dirty mask contents");
            host.index.take_written();
            let (fresh, _) = crate::frame::build_coverage_plan_cached(scene, 0, &host.opts,
                Some((&mut PlanCaches::default(), &hashes))).unwrap();
            assert_eq!(format!("{plan:?}"), format!("{fresh:?}"));
        }
    }

    /// A background effect that starts outside the target emits no split and, on an unpainted container, no
    /// span. Moving it or a node above it into view must plan whole; unrelated paths still follow recipes.
    #[test]
    fn a_unit_that_draws_nothing_yet_still_declines_placement_when_it_or_its_ancestor_moves() {
        let nodes = || vec![path("before", 25., 70., "#e02a40"), json!({"id":"ancestor","type":"group","children":["unit"]}),
            json!({"id":"unit","type":"container","x":-500,"y":75,"width":100,"height":100,"children":[],
                "effects":[{"id":"e","type":"background-blur","radius":12}]}),
            path("after", 190., 85., "#255fe8")];
        let splits = |plan: &CoveragePlan| plan.items.iter().filter(|item| matches!(item, crate::seams::DrawItem::BackdropSplit(_))).count();
        for mover in ["unit", "ancestor"] {
            let mut host = Host::new(nodes(), &["before", "ancestor", "after"]);
            let (first, _) = host.step(None);
            assert_eq!(splits(&first), 0, "the effect starts outside the target");
            let (plan, placed) = host.step(Some(("after", 200.)));
            assert!(placed, "an unrelated path follows its recipe beside the unit");
            assert_eq!(format!("{:?}", *plan), host.fresh());
            let (plan, placed) = host.step(Some((mover, if mover == "unit" { 20. } else { 600. })));
            assert!(!placed, "moving {mover} plans whole");
            assert_eq!(splits(&plan), 1, "the effect is in view after moving {mover}");
            assert_eq!(format!("{:?}", *plan), host.fresh());
            let (plan, placed) = host.step(Some(("before", 40.)));
            assert!(placed);
            assert_eq!(format!("{:?}", *plan), host.fresh());
        }
    }

    /// The guard asks the changed nodes, not every unit: its cost does not grow with units that stay put.
    #[test]
    fn many_stationary_units_do_not_block_or_slow_unrelated_placements() {
        let mut nodes = vec![path("free", 10., 10., "#255fe8")];
        let mut roots = vec!["free".to_string()];
        for i in 0..64 {
            let id = format!("unit{i}");
            let mut unit = path(&id, (i % 8) as f64 * 30., 80. + (i / 8) as f64 * 15., "#ffffff70");
            unit["effects"] = json!([{"id":"e","type":"layer-blur","radius":2}]);
            nodes.push(unit);
            roots.push(id);
        }
        let roots: Vec<&str> = roots.iter().map(String::as_str).collect();
        let mut host = Host::new(nodes, &roots);
        host.step(None);
        for x in [20., 35., 50.] {
            let (plan, placed) = host.step(Some(("free", x)));
            assert!(placed);
            assert_eq!(format!("{:?}", *plan), host.fresh());
        }
        let (plan, placed) = host.step(Some(("unit3", 5.)));
        assert!(!placed, "a unit that moves plans whole");
        assert_eq!(format!("{:?}", *plan), host.fresh());
    }
}
