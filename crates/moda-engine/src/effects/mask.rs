//! effects/mask.rs — group/container clip masks. M3d.
//!
//! Faithful port of:
//! - `renderers/webgpu/rendering/systems/WebGPUMaskRenderer.ts`
//!   (`getOrCreateMask` :538-716, `renderCoverage` :724-1171,
//!   `getImageMaskSrc` :314-324, `computeImagePlacement` :362-389)
//! - `renderers/webgpu/utils/MaskContextUtils.ts`
//!   (`getAncestorMaskOwnerIds` :25-44, `getActiveMaskContextForNode`
//!   :51-101, `computeEffectivePageClip` :113-174)
//!
//! Coverage resolution: a mask's coverage texture is rendered at the frame's
//! canvas→pixel scale (the export scale, or the editor camera's zoom × dpr),
//! quantised up to a power of two so zooming re-renders a mask only when it
//! crosses a power of two and panning never does, and capped by the device's
//! texture limit and [`MAX_MASK_TEXELS`]. The coverage draw is 4× MSAA
//! resolved into the r8 texture, so a mask's edge is anti-aliased at output
//! resolution (it used to be a dpr-1 single-sample staircase, magnified — the
//! old web-golden quirk Q-20, dropped).
//!
//! Pinned quirks:
//! - Q-21: image-alpha clipping only for fillable mask shapes with
//!   `fillPriority: 'pattern'` + `fillPatternRepeat: 'no-repeat'`, no
//!   corner-pin perspective; containers/tiling/loading/failed → geometry.
//! - Masked content is ALWAYS transparent-phase
//!   (`WebGPUPathRenderer.ts:634-645`).
//! - The effective page clip additionally intersects ancestor RECTANGLE
//!   group-mask AABBs and container AABBs (coarse hint; exact clipping is
//!   the R8 sample). An empty intersection leaves the plain page clip in
//!   place (production: `computeEffectivePageClip` returns null → no
//!   override).
//!
//! M4 note: image sources currently decode vendored `data:` URIs inline;
//! remote URLs fall back to geometry coverage until moda-assets (M3c) is
//! wired through integration.

use std::collections::HashMap;

use anyhow::{anyhow, Context, Result};
use base64::Engine as _;
use moda_scene::types::RawNode;
use moda_scene::{transform, Scene};

use crate::seams::{MaskDraw, MaskImage, MaskPaint, MaskPlan, MaskRef, MaskText, MaskType};

/// GPU texture views for rendered masks and registered captures.
pub type SurfaceViews = std::collections::HashMap<String, wgpu::TextureView>;
use crate::shape_to_path;
use crate::tess::{self, moda_engine_path_types as pt};

// ---------------------------------------------------------------------------
// Plan side (GPU-free)
// ---------------------------------------------------------------------------

/// The page index keeps the structural reads of isolated cameras alongside its
/// other selection predicates. Several masks on a page share this inventory.
#[derive(Default, Debug, Clone)]
pub(crate) struct MaskPageScope {
    membership: Option<std::sync::Arc<MaskMembership>>,
}

#[derive(Debug, Default)]
struct MaskMembership {
    owners: HashMap<String, Vec<String>>,
    // Rows whose visibility or composition can select a different raster mode
    // for a sibling (backdrop MSAA and analytic shape pipelines).
    raster: HashMap<String, RasterInput>,
    ambiguous: std::collections::HashSet<String>,
    document_context: u64,
    wrap_free: bool,
}

impl MaskMembership {
    fn of(scene: &Scene) -> Self {
        let mut membership = Self::default();
        scene.visit_effective_nodes(|id, node| {
            if let Some(source) = moda_scene::display_list::clip_mask_source_of(scene, id) {
                membership.owners.entry(source).or_default().push(id.to_owned());
            }
            if let Some(input) = RasterInput::of(scene, node) {
                membership.raster.insert(id.to_owned(), input);
            }
        });
        membership.wrap_free = scene.text_flow().wrap_ids.is_empty();
        for owners in membership.owners.values_mut() { owners.sort_unstable(); }
        let mut seen = std::collections::HashSet::new();
        let pages: std::collections::HashSet<_> = scene.doc().pages.iter().map(|page| page.id()).collect();
        // Canvas roots live in their own document list. A canonical group can
        // itself be named "canvas" (the host's viewBox transform group).
        let edges = scene.children_by_parent().iter().map(|(id, children)|
            ((!pages.contains(id.as_str())).then_some(id.as_str()), children));
        for (parent, children) in edges.chain(std::iter::once((None, &scene.doc().canvas_root_node_ids))) {
            for id in children {
                if !seen.insert(id) || scene.parent_map().get(id).map(String::as_str) != parent {
                    membership.ambiguous.insert(id.clone());
                }
            }
        }
        membership.document_context = document_context(scene);
        membership
    }
}

fn observe_camera_mode(scene: &Scene, node: &RawNode, flags: &mut [bool; 3]) {
    for (flag, contribution) in flags.iter_mut().zip(crate::frame::SourceEffects::raster(scene, node, true)) {
        *flag |= contribution;
    }
}

/// A sibling contributes only these camera flags, regardless of its own pixels.
/// When ancestry can change its flags under isolation, the next composition
/// write to its branch asks the ordinary planner again.
#[derive(Debug, PartialEq, Eq)]
enum RasterInput {
    Visible([bool; 3]),
    DependsOnAncestors,
}

impl RasterInput {
    fn of(scene: &Scene, node: &RawNode) -> Option<Self> {
        use crate::backdrop::{blend, plan};
        let blur = plan::has_active_background_blur(node);
        let body = blend::precludes_shape_sdf(scene, node, true);
        let composite = blend::composite_blend_split(node).is_some();
        let grouped = crate::paint::layers::stack_needs_split_at(node, 0.5, true);
        let opaque = crate::paint::layers::stack_needs_split_at(node, 1., true);
        if !(blur || body || composite || grouped || opaque) { return None; }
        if composite || grouped != opaque || scene.hidden_from_paint_order(node.id())
            || scene.transform_parent_chain(node.id()).iter().any(|id|
                scene.node(id).is_some_and(|n| matches!(n.node_type(), "section" | "boolean"))) {
            Some(Self::DependsOnAncestors)
        } else {
            Some(Self::Visible([blur, body, opaque]))
        }
    }
}

/// The same read set decides both the capture key and what can be isolated.
/// Generated geometry, sibling clipping and document text flow keep their
/// ordinary complete-document dependencies.
struct MaskInputs<'a> {
    sources: std::collections::BTreeSet<&'a str>,
    outside_clips: Vec<(&'a str, &'a str)>,
}

impl<'a> MaskInputs<'a> {
    fn read(scene: &'a Scene, root: &'a str, ancestors: &[String], membership: &'a MaskMembership) -> Option<Self> {
        if ancestors.iter().any(|id| {
            membership.ambiguous.contains(id) || scene.node(id).is_none_or(|node| node.str("_transformSourceId").is_some())
                || scene.boolean_source_by_id().contains_key(id) || scene.group_appearance_by_id().contains_key(id)
                || scene.transient_by_parent().contains_key(id)
        }) { return None; }
        let mut sources = std::collections::BTreeSet::new();
        let mut references = Vec::new();
        let mut pending = vec![root];
        while let Some(id) = pending.pop() {
            if !sources.insert(id) { continue; }
            let node = scene.node(id)?;
            if node.node_type() == "blend" { references.extend(moda_scene::blend::dependency_ids(node)); }
            references.extend(node.str("clipMaskNodeId").filter(|id| !id.is_empty()));
            // A clip to the layer below reads preceding siblings. Only the root
            // can read siblings outside this occurrence. Derived geometry and
            // unrevisioned or document-dependent text need their full context.
            if membership.ambiguous.contains(id) || scene.boolean_source_by_id().contains_key(id) || scene.transient_by_parent().contains_key(id)
                || scene.group_appearance_by_id().contains_key(id) || node.str("_transformSourceId").is_some()
                || (id == root && node.truthy("clipToBelow"))
                || (node.node_type() == "richtext" && !crate::text_draw::layout_is_local(scene, node)) {
                return None;
            }
            pending.extend(scene.children_by_parent().get(id).into_iter().flatten().map(String::as_str));
        }
        if references.iter().any(|id| !sources.contains(id)) { return None; }
        let mut outside_clips = Vec::new();
        for id in sources.iter().copied().chain(ancestors.iter().map(String::as_str)) {
            if let Some((target, owners)) = membership.owners.get_key_value(id) {
                for owner in owners {
                    if !sources.contains(owner.as_str()) && !ancestors.contains(owner) {
                        outside_clips.push((target.as_str(), owner.as_str()));
                    }
                }
            }
        }
        Some(Self { sources, outside_clips })
    }

    fn document(&self, scene: &Scene, root: &str, ancestors: &[String], membership: &MaskMembership) -> Option<moda_scene::types::SceneDocument> {
        use moda_scene::types::{is_page_bg_id, SceneDocument};
        let doc = scene.doc();
        if !membership.ambiguous.is_empty() || !self.outside_clips.is_empty()
            || scene.transient_by_id().keys().any(|id| !is_page_bg_id(id) || scene.node(id).is_some()) {
            return None;
        }
        // Keep the branches whose visibility selects a camera pipeline as well
        // as the isolated occurrence. An unrelated blur needs its own row and
        // placement in the tree, even though its body will never be emitted.
        let mut included = self.sources.clone();
        let mut branches = vec![root];
        for id in membership.raster.keys() {
            if self.sources.contains(id.as_str()) || ancestors.contains(id) || is_page_bg_id(id) { continue; }
            branches.push(id);
            let mut pending = vec![id.as_str()];
            while let Some(id) = pending.pop() {
                if included.insert(id) {
                    pending.extend(scene.children_by_parent().get(id).into_iter().flatten().map(String::as_str));
                }
            }
        }
        let mut children = HashMap::<&str, std::collections::BTreeSet<&str>>::new();
        let mut roots = std::collections::BTreeSet::new();
        for mut child in branches {
            while let Some(parent) = scene.parent_map().get(child) {
                let kept = children.entry(parent).or_default();
                if !kept.insert(child) { break; }
                child = parent;
            }
            if !scene.parent_map().contains_key(child) { roots.insert(child); }
        }
        let mut nodes = Vec::with_capacity(included.len() + children.len());
        for id in &included { nodes.push(scene.node(id)?.mask_snapshot()); }
        for (id, kept) in children {
            if included.contains(id) { continue; }
            let node = scene.node(id)?;
            // A resolved boolean or painted grouping reads even its omitted siblings.
            if scene.boolean_source_by_id().contains_key(id) || scene.group_appearance_by_id().contains_key(id) {
                return None;
            }
            let mut fields = node.fields().iter().filter(|(k, _)| *k != "children")
                .map(|(k, v)| (k.clone(), v.clone())).collect::<moda_scene::types::JsonMap>();
            let ordered: Vec<_> = if kept.len() == 1 { kept.into_iter().collect() } else {
                scene.children_by_parent().get(id)?.iter().map(String::as_str).filter(|id| kept.contains(id)).collect()
            };
            fields.insert("children".into(), moda_scene::value::json!(ordered));
            nodes.push(node.with_fields(fields));
        }
        nodes.sort_unstable_by_key(|node| scene.row_of(node.id()));
        // Page order and dimensions position even a source on an otherwise empty page.
        let pages = doc.pages.iter().map(|page| {
            let mut projected = moda_scene::types::RawPage(page.0.iter().filter(|(k, _)| *k != "nodeIds")
                .map(|(k, v)| (k.clone(), v.clone())).collect());
            let on_page: Vec<_> = roots.iter().copied().filter(|id|
                scene.page_id_by_node().get(*id).is_some_and(|p| p == page.id())).collect();
            let ordered = if on_page.len() < 2 { on_page.into_iter().map(str::to_owned).collect() } else {
                scene.children_by_parent().get(page.id()).into_iter().flatten()
                    .filter(|id| on_page.contains(&id.as_str())).cloned().collect()
            };
            projected.set_node_ids(ordered);
            projected
        }).collect();
        let canvas = doc.canvas_root_node_ids.iter().filter(|id| roots.contains(&id.as_str())).cloned().collect();
        Some(SceneDocument {
            nodes, pages, config: doc.config.clone(), canvas_root_node_ids: canvas,
            layout_mode: doc.layout_mode.clone(), grid_columns: doc.grid_columns,
            unknown_envelope_keys: Vec::new(), unknown_node_types: Vec::new(),
        })
    }
}

impl MaskPageScope {
    fn membership(&mut self, scene: &Scene) -> &MaskMembership {
        self.membership.get_or_insert_with(|| std::sync::Arc::new(MaskMembership::of(scene)))
    }

    pub(crate) fn clip_bindings(&self) -> impl Iterator<Item = (&String, &String)> {
        self.membership.iter().flat_map(|m| &m.owners)
            .flat_map(|(target, owners)| owners.iter().map(move |owner| (owner, target)))
    }

    /// The ordinary composition writer visits descendants of changed ancestors.
    /// A previously invisible blur or split can now participate in a capture.
    pub(crate) fn raster_stands(&self, scene: &Scene, node: &RawNode) -> bool {
        self.membership.as_ref().is_none_or(|m| {
            if m.wrap_free != scene.text_flow().wrap_ids.is_empty() { return false; }
            let was = m.raster.get(node.id());
            was != Some(&RasterInput::DependsOnAncestors) && was == RasterInput::of(scene, node).as_ref()
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        self.membership.as_ref().map_or(0, |m| std::mem::size_of::<MaskMembership>()
            + 2 * std::mem::size_of::<usize>() + m.owners.capacity() * std::mem::size_of::<(String, Vec<String>)>()
            + m.owners.iter().map(|(id, owners)| id.capacity() + owners.capacity() * std::mem::size_of::<String>()
                + owners.iter().map(String::capacity).sum::<usize>()).sum::<usize>()
            + m.ambiguous.capacity() * std::mem::size_of::<String>() + m.ambiguous.iter().map(String::capacity).sum::<usize>()
            + m.raster.capacity() * std::mem::size_of::<(String, RasterInput)>() + m.raster.keys().map(String::capacity).sum::<usize>())
    }
}

/// Per-page mask planning state: builds `MaskPlan`s lazily per owner.
pub(crate) type RetainedMaskLookup<'a> = &'a dyn Fn(&str) -> Option<&'a MaskPlan>;

pub struct PlanMasks<'a> {
    pub(crate) scope: MaskPageScope,
    pub masks: Vec<MaskPlan>,
    /// Sources already proved current by the parent unit's ordinary row dependencies.
    pub(crate) retained: Option<RetainedMaskLookup<'a>>,
    owner_to_id: HashMap<String, Option<usize>>,
    /// Actual mask reads made by the shared planner while recording retained units.
    reads: Option<Vec<usize>>,
    /// Fail-closed mask refusals (`PlanWarnings::dropped`): a mask whose
    /// coverage geometry lyon refused. Such a mask is built with ZERO
    /// coverage — it clips everything — never the full-quad fallback, which
    /// would render the owner's children unclipped across the whole box.
    /// Drained into the plan's warnings by `into_masks`.
    dropped: Vec<String>,
    /// Tolerated degradations (`PlanWarnings::skipped`), e.g. a variable-width
    /// stroke mask filled in cells or drawn uniform.
    skipped: Vec<String>,
    /// Whether richtext (text-as-mask) coverage is built
    /// (`RenderOptions::text_masks_available`, default yes). It is built on
    /// every page: the export path built it on page index 0 only (a copied
    /// browser quirk, Q-95), so page 2 of a PDF or a slide deck baked a text
    /// mask's children UNCLIPPED while the editor clipped them.
    text_masks_available: bool,
    /// Better-tier text-mask raster (mask:text-alpha engine fix): when the
    /// frame's text engine is the GA `TextEngine::Vector`, richtext mask
    /// sources render their raster through the vector outline lane
    /// (`moda_text::vector::render_mask_raster`) with the Degraded raster as
    /// the per-node fallback — mirroring the visible-text routing, so one
    /// export no longer draws two different letterform sets for the same
    /// string. `false` (msdf/hybrid engines, plus `Default` for the legacy
    /// tests/tools) keeps the pure Degraded raster.
    vector_text_masks: bool,
    /// Whether the plan is for an engine that blends in another encoding than sRGB
    /// (`RenderOptions::working`). A luminance mask is the luminance of the mask's picture as sRGB. The
    /// shortcut for one shape reduces each of its paints to luminance and blends those, which is the picture's
    /// luminance where the picture blends sRGB code values, and in any encoding where the shape is one layer
    /// of one solid colour (`one_solid_layer`). Otherwise, in a working colour, the mask is its rendered
    /// picture (`build_rendered_mask`), which the engine converts back and reduces once.
    working_color: bool,
    shader_time_s: f64,
}

impl Default for PlanMasks<'_> {
    /// Default: text masks available.
    fn default() -> Self {
        PlanMasks {
            scope: Default::default(),
            masks: Vec::new(),
            retained: None,
            owner_to_id: HashMap::new(),
            reads: None,
            dropped: Vec::new(),
            skipped: Vec::new(),
            text_masks_available: true,
            vector_text_masks: false,
            working_color: false,
            shader_time_s: 0.0,
        }
    }
}

impl PlanMasks<'_> {
    pub(crate) fn record_reads(&mut self) { self.reads.get_or_insert_with(Vec::new); }

    pub(crate) fn reads(&self) -> &[usize] { self.reads.as_deref().unwrap_or(&[]) }

    /// Explicit text-mask availability (`RenderOptions::text_masks_available`).
    pub fn with_text_masks(text_masks_available: bool) -> Self {
        PlanMasks {
            text_masks_available,
            ..Self::default()
        }
    }

    /// Route richtext mask rasters through the vector text lane (with the
    /// Degraded raster as per-node fallback). Callers pass
    /// `opts.text_engine == TextEngine::Vector` — the mask tier follows the
    /// frame's text engine so mask and visible letterforms agree.
    pub fn with_vector_text_masks(mut self, enabled: bool) -> Self {
        self.vector_text_masks = enabled;
        self
    }

    /// Plan for an engine that blends in another encoding than sRGB (`RenderOptions::working`).
    pub fn in_working_color(mut self, working_color: bool) -> Self {
        self.working_color = working_color;
        self
    }

    pub fn at_shader_time(mut self, shader_time_s: f64) -> Self {
        self.shader_time_s = shader_time_s;
        self
    }

    /// The finished mask table; refused masks are reported into
    /// `warnings.dropped` on the way out.
    pub fn into_masks(self, warnings: &mut crate::frame::PlanWarnings) -> Vec<MaskPlan> {
        warnings.dropped.extend(self.dropped);
        warnings.skipped.extend(self.skipped);
        self.masks
    }

    /// A range can reuse the page's clip owners and raster inputs. Only a full
    /// collection can install them in both the segment and its page effects.
    pub(crate) fn into_range_masks(self, page: &MaskPageScope, warnings: &mut crate::frame::PlanWarnings) -> Option<Vec<MaskPlan>> {
        if page.membership.is_none() && self.scope.membership.is_some() { return None; }
        Some(self.into_masks(warnings))
    }

    /// Nearest mask context for a node (`getActiveMaskContextForNode`): the node's own clip to the layer below,
    /// else its parents' — a clipping group or frame, or a clipped parent's base — first with a buildable mask wins
    /// (its coverage already composes ITS ancestors).
    pub fn mask_ref_for_node(&mut self, scene: &Scene, node_id: &str) -> Option<MaskRef> {
        self.mask_ref_walk(scene, node_id, false)
    }

    /// [`Self::mask_ref_for_node`] for drawing a node's body: a clipping frame whose clip changes nothing for the
    /// node ([`container_clip_is_redundant`]) is passed over, so a body that fills its frame exactly is not cut a
    /// second time at the frame's edge. Use `mask_ref_for_node` to decide whether the node is clipped at all.
    pub fn clip_mask_ref_for_node(&mut self, scene: &Scene, node_id: &str) -> Option<MaskRef> {
        self.mask_ref_walk(scene, node_id, true)
    }

    fn mask_ref_walk(
        &mut self,
        scene: &Scene,
        node_id: &str,
        skip_redundant: bool,
    ) -> Option<MaskRef> {
        let mut from = (node_id.to_string(), true);
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        while let Some(owner) = next_mask_owner(scene, &from.0, from.1) {
            if !seen.insert(owner.key()) {
                break; // cycle guard — corrupt parent data must not hang the renderer
            }
            let redundant = skip_redundant
                && match &owner {
                    MaskOwner::Clip(owner_id) => scene
                        .canonical_or_transient(owner_id)
                        .is_some_and(|c| container_clip_is_redundant(scene, c, node_id)),
                    MaskOwner::Below { .. } => false,
                };
            if !redundant {
                if let Some(id) = self.ensure_owner(scene, &owner) {
                    return Some(self.mask_ref_at(id));
                }
            }
            from = owner.continue_from();
        }
        None
    }

    /// The reference to mask `id` of this table.
    fn mask_ref_at(&mut self, id: usize) -> MaskRef {
        if let Some(reads) = &mut self.reads { reads.push(id); }
        let m = &self.masks[id];
        MaskRef {
            mask_id: id as u64,
            origin: [m.origin[0] as f32, m.origin[1] as f32],
            inv_size: [
                if m.size[0] > 0.0 { 1.0 / m.size[0] } else { 0.0 } as f32,
                if m.size[1] > 0.0 { 1.0 / m.size[1] } else { 0.0 } as f32,
            ],
        }
    }

    /// Build (or fetch) the mask plan for `owner`. Returns the index into `self.masks`, or None when the owner
    /// can't produce a mask (missing mask node, degenerate bounds) — matching `getOrCreateMask`'s nulls.
    fn ensure_owner(&mut self, scene: &Scene, owner: &MaskOwner) -> Option<usize> {
        let key = owner.key();
        if let Some(cached) = self.owner_to_id.get(&key) {
            return *cached;
        }
        // Insert a tombstone first so recursion on a (malformed) cyclic
        // ancestor chain terminates.
        self.owner_to_id.insert(key.clone(), None);
        let built = match owner {
            MaskOwner::Clip(id) => self.build_mask(scene, id),
            MaskOwner::Below { clipped, base } => self.build_clip_below(scene, clipped, base),
        };
        self.owner_to_id.insert(key, built);
        built
    }

    /// Clip to the layer below: the clipped layers draw inside their base's rendered alpha (its paint alpha × layer
    /// and node opacity, as an alpha mask of the base), composed with the base's own mask context. A hidden base
    /// hides them too (zero coverage), as in Photoshop.
    fn build_clip_below(&mut self, scene: &Scene, clipped: &str, base_id: &str) -> Option<usize> {
        let key = MaskOwner::Below { clipped: clipped.to_string(), base: base_id.to_string() }.key();
        let ancestor = next_mask_owner(scene, clipped, false).and_then(|o| self.ensure_owner(scene, &o));
        let hidden = scene.canonical_or_transient(base_id).is_some_and(|n| n.truthy("hidden"));
        self.build_mask_of(scene, &key, base_id, MaskType::Alpha, ancestor, hidden)
    }

    fn build_mask(&mut self, scene: &Scene, owner_id: &str) -> Option<usize> {
        let owner = scene.canonical_or_transient(owner_id)?;
        let mask_node_id = match owner.node_type() {
            "group" => owner.str("clipMaskNodeId")?.to_string(),
            "container" => {
                if !owner.truthy("clipContent") {
                    return None;
                }
                owner_id.to_string()
            }
            _ => return None,
        };
        // E7.18: a group's mask masks by its `maskType`; a clipping container
        // clips to its box.
        let mask_type = if owner.node_type() == "group" {
            MaskType::from_node_value(
                scene.effective_node(&mask_node_id)?
                    .str("maskType"),
            )
        } else {
            MaskType::Vector
        };
        // Nearest ancestor mask of the OWNER (recursive — grandparents are
        // already composed into it). `getAncestorMaskOwnerIds(owner)[0]`; a
        // clipped owner's own clip to the layer below counts.
        let ancestor = next_mask_owner(scene, owner_id, true).and_then(|o| self.ensure_owner(scene, &o));
        self.build_mask_of(scene, owner_id, &mask_node_id, mask_type, ancestor, false)
    }

    /// A mask plan keyed `owner_id` whose coverage is `mask_node_id`'s, by `mask_type`, composed with mask
    /// `ancestor`. `empty`: zero coverage (it clips everything away).
    fn build_mask_of(
        &mut self,
        scene: &Scene,
        owner_id: &str,
        mask_node_id: &str,
        mask_type: MaskType,
        ancestor: Option<usize>,
        empty: bool,
    ) -> Option<usize> {
        if let Some(previous) = self.retained.and_then(|get| get(owner_id))
            .filter(|mask| mask.source_node_id == mask_node_id && mask.mask_type == mask_type)
        {
            let mut mask = previous.clone();
            mask.ancestor = ancestor;
            let id = self.masks.len();
            self.masks.push(mask);
            return Some(id);
        }
        let id = self.build_mask_of_inner(scene, owner_id, mask_node_id, mask_type, ancestor, empty)?;
        let node = scene.effective_node(mask_node_id)?;
        let invert = node.truthy("maskInvert");
        let feather = node.f64("maskFeather").filter(|value| value.is_finite()).unwrap_or(0.0).max(0.0);
        if !invert && feather == 0.0 {
            return Some(id);
        }
        let affine = scene.node_canvas_affine(mask_node_id)?;
        let feather = feather * affine.a.hypot(affine.b).max(affine.c.hypot(affine.d));
        let mask = &mut self.masks[id];
        mask.invert = invert;
        mask.feather = feather;
        let mut bounds = Some((mask.origin[0], mask.origin[1], mask.size[0], mask.size[1]));
        if invert {
            let isolate = std::collections::HashSet::from([owner_id]);
            let mut lists: Vec<_> = scene.doc().pages.iter()
                .map(|page| moda_scene::display_list::build_display_list(scene, page.id())).collect();
            lists.push(moda_scene::display_list::build_display_list_for_canvas(scene));
            for item in lists.iter().flat_map(|list| &list.items) {
                if !scene.is_node_isolated(&item.node_id, &isolate) { continue; }
                let visual = crate::cull::item_visual_bounds(scene, item);
                if visual.iter().all(|value| value.is_finite()) {
                    bounds = union_aabb(bounds, Some((visual[0], visual[1], visual[2] - visual[0], visual[3] - visual[1])));
                }
            }
        }
        let (bx, by, bw, bh) = bounds?;
        let padding = (1.5 * feather).max(if invert { 1.0 } else { 0.0 });
        mask.origin = [bx - padding, by - padding];
        mask.size = [bw + 2.0 * padding, bh + 2.0 * padding];
        if mask.mode == 3 {
            mask.model_matrix[12] = mask.origin[0] as f32;
            mask.model_matrix[13] = mask.origin[1] as f32;
            let width = mask.size[0] as f32;
            let height = mask.size[1] as f32;
            if !mask.positions.is_empty() {
                mask.positions = vec![0.0, 0.0, width, 0.0, width, height, 0.0, 0.0, width, height, 0.0, height].into();
            }
            if let Some(image) = &mut mask.image {
                image.dims = [width, height];
            }
        }
        Some(id)
    }

    fn build_mask_of_inner(
        &mut self,
        scene: &Scene,
        owner_id: &str,
        mask_node_id: &str,
        mask_type: MaskType,
        ancestor: Option<usize>,
        empty: bool,
    ) -> Option<usize> {
        let mask_node_id = mask_node_id.to_string();
        let mask_node = scene.effective_node(&mask_node_id)?;
        let source_type = scene.canonical_or_transient(&mask_node_id)
            .map_or(mask_node.node_type(), |node| node.node_type());
        if mask_type != MaskType::Vector
            && ((mask_type == MaskType::Luminance && self.working_color && !one_solid_layer(mask_node))
                || !matches!(source_type, "rectangle" | "ellipse" | "polygon" | "star" | "path" | "line" | "richtext" | "container" | "boolean")
                || mask_reads_store_image(mask_node)
                || crate::layer_blur::plan::needs_layer_capture(mask_node)
                || (mask_node.node_type() == "container" && scene.children_by_parent().get(mask_node.id()).is_some_and(|children| !children.is_empty()))
                || (mask_node.node_type() == "line" && mask_node.str("textContent").is_some_and(|text| !text.is_empty()))
                || crate::effects::shadow::has_drop_shadow_quad(mask_node))
        {
            return self.build_rendered_mask(scene, owner_id, &mask_node_id, mask_type, ancestor, empty);
        }
        let geometry = mask_geometry_positions(mask_node, Some(scene));

        // Canvas-space AABB of the mask node (Scene.getNodeBoundsCanvas),
        // grown to the drawn geometry: a stroke reaches past the outline (and a
        // straight brushed one has no area at all), so the texture covers the
        // stroke too (the visual bounds, BB8; E7.18 masks with the stroke).
        // Richtext keeps its box: its coverage is the raster quad.
        let (bx, by, mut bw, mut bh) = if mask_node.node_type() == "richtext" {
            node_canvas_aabb(scene, mask_node)?
        } else {
            union_aabb(
                node_canvas_aabb(scene, mask_node),
                positions_canvas_aabb(scene, mask_node, &geometry.positions),
            ).or_else(|| {
                // An empty supported outline may not have a control-point box.
                // Keep a zero-area mask at its origin so it still clips content.
                if geometry.box_fallback || !geometry.positions.is_empty() { return None; }
                let affine = scene.node_canvas_affine(mask_node.id())?;
                Some((affine.e, affine.f, 0.0, 0.0))
            })?
        };
        // A mask with no area (a rect keyed to height 0, a collapsed path) covers nothing: it clips everything away,
        // as a mask of zero size does in Figma and Illustrator, rather than leaving the content unclipped.
        let empty = empty || !(bw > 0.0 && bh > 0.0);
        if !(bw > 0.0 && bh > 0.0) {
            if !(bx.is_finite() && by.is_finite()) {
                return None;
            }
            (bw, bh) = (1.0, 1.0);
        }

        // M4d — text-as-mask (renderCoverage :943-952 text branch): a
        // richtext mask source samples the Canvas2D standalone texture's
        // alpha; natively that texture is the Degraded-tier raster
        // (moda_text::degraded). The coverage quad is `(0,0,logicalSize)`
        // at the node affine and `textOrigin` is the node's canvas
        // position — verticalAlign does NOT move the silhouette (top-left
        // anchored; see the quirk ledger). The raster is opaque for
        // uniform-alpha runs (`fill_alpha` is a display-quad concern) —
        // masks clip at full coverage, exactly like the browser texture.
        let text = if mask_node.node_type() == "richtext" {
            if !self.text_masks_available {
                return None;
            }
            match text_mask_source(scene, mask_node, self.vector_text_masks) {
                Ok(t) => t,
                Err(e) => {
                    // Production parity: when the standalone texture isn't
                    // available, getOrCreateMask returns null and children
                    // render UNMASKED this frame (:576-583).
                    eprintln!(
                        "moda-render: warning: text mask {}: {e} (children render unmasked)",
                        mask_node.id()
                    );
                    return None;
                }
            }
        } else {
            None
        };

        // Geometry: same shape→path converters as rendering, tessellated
        // fill+stroke. Empty supported geometry keeps zero coverage.
        // A REFUSED fill (`MaskGeometry::fill_dropped`) is not "empty": the
        // node has geometry the engine could not produce, so it takes ZERO
        // coverage (the pass below clears to 0 and draws no vertices —
        // children clip away entirely) and is reported fail-closed. The
        // full-quad arm is reserved for node kinds with no geometry at all.
        let MaskGeometry {
            mut positions,
            fill_dropped,
            stroke_dropped,
            note,
            box_fallback,
            ..
        } = geometry;
        let mut quad = false;
        if let Some(note) = note {
            self.skipped.push(format!(
                "mask {} (clips {owner_id}): {note}",
                mask_node.id()
            ));
        }
        if let Some(why) = &fill_dropped {
            self.dropped.push(format!(
                "mask {} (clips {owner_id}): fill dropped: {why}",
                mask_node.id()
            ));
            positions.clear();
        }
        if let Some(t) = &text {
            // The texture starts `pad_top` above the node's top (ink over its first line).
            let (w, top) = (t.raster.logical_w as f32, -t.raster.pad_top as f32);
            let h = top + t.raster.logical_h as f32;
            positions = vec![
                0.0, top, w, top, w, h, //
                0.0, top, w, h, 0.0, h,
            ];
        } else if box_fallback && fill_dropped.is_none() {
            quad = true;
            // Only node kinds without an outline use their box. Successful
            // empty tessellation covers nothing. A distort warp moves the
            // box corners — use the
            // warped quad so coverage matches where the node actually paints.
            // Richtext is EXCLUDED here exactly like node_canvas_aabb above:
            // its coverage lane is the unwarped Degraded raster (accepted
            // gap), so its texture bounds AND fallback quad must both
            // stay unwarped or they would disagree with each other.
            let warp = if mask_node.node_type() != "richtext" {
                crate::paint::node_distort(mask_node)
            } else {
                None
            };
            if let Some(d) = warp {
                let c = &d.corners;
                positions = vec![
                    c[0].x as f32,
                    c[0].y as f32,
                    c[1].x as f32,
                    c[1].y as f32,
                    c[2].x as f32,
                    c[2].y as f32,
                    c[0].x as f32,
                    c[0].y as f32,
                    c[2].x as f32,
                    c[2].y as f32,
                    c[3].x as f32,
                    c[3].y as f32,
                ];
            } else {
                let w = mask_node.f64("width").unwrap_or(bw);
                let h = mask_node.f64("height").unwrap_or(bh);
                positions = vec![
                    0.0, 0.0, w as f32, 0.0, w as f32, h as f32, //
                    0.0, 0.0, w as f32, h as f32, 0.0, h as f32,
                ];
            }
        }

        // Coverage draws (E7.18). Text masks sample their raster; a vector
        // mask, a refused one, and the full-quad fallback draw their geometry
        // at the node's opacity (vector: full coverage); an alpha or luminance
        // shape draws each paint layer on its own geometry.
        let node_opacity = mask_node
            .f64("opacity")
            .filter(|o| o.is_finite())
            .unwrap_or(1.0)
            .clamp(0.0, 1.0) as f32;
        let vector = mask_type == MaskType::Vector;
        let count = (positions.len() / 2) as u32;
        let whole = |paint: MaskPaint| {
            vec![MaskDraw {
                first: 0,
                count,
                paint,
            }]
        };
        let white = |a: f32| MaskPaint::Solid([1.0, 1.0, 1.0, a]);
        let mut image = None;
        let draws = if let Some(t) = &text {
            // Vector: the glyph outlines (the raster's coverage); alpha and
            // luminance: the rendered text, with its uniform run alpha.
            let opacity = if vector {
                1.0
            } else {
                node_opacity * t.raster.fill_alpha
            };
            whole(MaskPaint::Source { opacity })
        } else if vector {
            if let Some(why) = &stroke_dropped {
                self.dropped.push(format!("mask {mask_node_id} (clips {owner_id}): stroke dropped: {why}"));
            }
            whole(white(1.0))
        } else if quad || fill_dropped.is_some() {
            whole(white(node_opacity))
        } else {
            let layered = paint_layer_draws(scene, mask_node, node_opacity);
            for why in layered.strokes_dropped {
                self.dropped.push(format!("mask {mask_node_id} (clips {owner_id}): stroke dropped: {why}"));
            }
            if let Some(why) = layered.fill_dropped {
                // The whole geometry tessellated, a layer clone's did not:
                // fail closed as for the combined geometry above.
                self.dropped.push(format!(
                    "mask {} (clips {owner_id}): fill dropped: {why}",
                    mask_node.id()
                ));
                positions.clear();
                Vec::new()
            } else {
                for n in layered.notes {
                    self.skipped
                        .push(format!("mask {} (clips {owner_id}): {n}", mask_node.id()));
                }
                positions = layered.positions;
                image = layered.image;
                layered.draws
            }
        };
        let (draws, mode) = if empty {
            // Zero coverage: the pass clears to 0 and draws nothing.
            positions.clear();
            (Vec::new(), 0)
        } else {
            let mode = if text.is_some() {
                1
            } else if image.is_some() {
                2
            } else {
                0
            };
            (draws, mode)
        };

        let affine = scene.node_canvas_affine(&mask_node_id)?;
        let model_matrix = [
            affine.a as f32,
            affine.b as f32,
            0.0,
            0.0,
            affine.c as f32,
            affine.d as f32,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            affine.e as f32,
            affine.f as f32,
            0.0,
            1.0,
        ];

        let id = self.masks.len();
        self.masks.push(MaskPlan {
            source_node_id: mask_node_id.to_string(),
            owner_node_id: owner_id.to_string(),
            origin: [bx, by],
            size: [bw, bh],
            model_matrix,
            positions: positions.into(),
            mask_type,
            invert: false,
            feather: 0.0,
            draws: draws.into(),
            mode,
            image,
            text,
            ancestor,
        });
        Some(id)
    }
}

impl PlanMasks<'_> {
    fn build_rendered_mask(
        &mut self,
        scene: &Scene,
        owner_id: &str,
        mask_node_id: &str,
        mask_type: MaskType,
        ancestor: Option<usize>,
        empty: bool,
    ) -> Option<usize> {
        let ancestors = scene.transform_parent_chain(mask_node_id);
        let membership = self.scope.membership(scene);
        let document_context = membership.document_context;
        let inputs = MaskInputs::read(scene, mask_node_id, &ancestors, membership);
        let mut document = inputs.as_ref().and_then(|reads| reads.document(scene, mask_node_id, &ancestors, membership))
            .unwrap_or_else(|| scene.doc().mask_snapshot());
        for node in &mut document.nodes {
            if ancestors.iter().any(|id| id == node.id()) {
                let fields = node.fields_mut();
                fields.remove("clipMaskNodeId");
                fields.remove("clipContent");
                fields.remove("clipToBelow");
                fields.remove("effects");
                fields.remove("blendMode");
                fields.insert("opacity".into(), moda_scene::value::json!(1));
            }
        }
        let capture_scene = Scene::from_document(document.clone()).ok()?;
        let isolate = std::collections::HashSet::from([mask_node_id]);
        let mut bounds: Option<(f64, f64, f64, f64)> = None;
        let mut camera_modes = Vec::new();
        let mut lists: Vec<_> = capture_scene.doc().pages.iter()
            .map(|page| {
                if inputs.is_none() {
                    return moda_scene::display_list::build_display_list(&capture_scene, page.id());
                }
                let mut mode = [false; 3];
                let list = moda_scene::display_list::build_page_list(&capture_scene, page.id(), true, |node| {
                    observe_camera_mode(&capture_scene, node, &mut mode);
                });
                camera_modes.push(mode);
                list
            })
            .collect();
        lists.push(moda_scene::display_list::build_display_list_for_canvas(&capture_scene));
        if inputs.is_some() && capture_scene.is_freeform() {
            let mut mode = [false; 3];
            for id in capture_scene.flattened_z_order_for_canvas() {
                if let Some(node) = capture_scene.effective_node(&id) { observe_camera_mode(&capture_scene, node, &mut mode); }
            }
            camera_modes.push(mode);
        }
        for item in lists.iter().flat_map(|list| &list.items) {
            if !capture_scene.is_node_isolated(&item.node_id, &isolate) {
                continue;
            }
            if let Some(node) = capture_scene.effective_node(&item.node_id).filter(|node| node.node_type() == "blend") {
                if let moda_scene::blend::BlendExpansion::Paths { page_index, nodes } = moda_scene::blend::expand_blend_node(
                    &capture_scene, node, &|source| crate::frame::blend_shape_to_abs_points(&capture_scene, source),
                ) {
                    let (px, py) = page_index.map_or((0.0, 0.0), |index| capture_scene.page_position(index));
                    for step in nodes {
                        let geometry = mask_geometry_positions(&step, Some(&capture_scene));
                        if let Some([x0, y0, x1, y1]) = crate::geometry_bounds::positions_bounds(&geometry.positions) {
                            bounds = union_aabb(bounds, Some((x0 + px, y0 + py, x1 - x0, y1 - y0)));
                        }
                    }
                }
                continue;
            }
            if let Some(node) = capture_scene.effective_node(&item.node_id).filter(|node| node.node_type() == "richtext") {
                match crate::text_draw::host_text_bounds(node) {
                    Ok(Some([x0, y0, x1, y1])) => {
                        if let Some(m) = capture_scene.node_canvas_affine(node.id()) {
                            let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                                .map(|(x, y)| (m.a * x + m.c * y + m.e, m.b * x + m.d * y + m.f));
                            let lo_x = corners.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
                            let lo_y = corners.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
                            let hi_x = corners.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
                            let hi_y = corners.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
                            bounds = union_aabb(bounds, Some((lo_x, lo_y, hi_x - lo_x, hi_y - lo_y)));
                            continue;
                        }
                    }
                    Err(error) => { self.dropped.push(format!("mask {mask_node_id}: {error:#}")); return None; }
                    Ok(None) => {}
                }
            }
            let visual = crate::cull::item_visual_bounds(&capture_scene, item);
            let candidate = if visual.iter().all(|value| value.is_finite()) {
                Some((visual[0], visual[1], visual[2] - visual[0], visual[3] - visual[1]))
            } else {
                item.bounds_canvas.as_ref().map(|rect| (rect.x, rect.y, rect.width, rect.height))
            };
            bounds = union_aabb(bounds, candidate);
            if !visual.iter().all(|value| value.is_finite()) {
                if let Some(node) = capture_scene.effective_node(&item.node_id) {
                    let geometry = mask_geometry_positions(node, Some(&capture_scene));
                    bounds = union_aabb(bounds, positions_canvas_aabb(&capture_scene, node, &geometry.positions));
                }
            }
        }
        let (bx, by, bw, bh) = bounds.unwrap_or((0.0, 0.0, 1.0, 1.0));
        // Capture on the canvas pixel grid. Fractional viewports round their texture dimensions
        // upward, then rescale the glyph coverage when sampled back into the mask atlas.
        let (right, bottom) = ((bx + bw).ceil(), (by + bh).ceil());
        let (bx, by) = (bx.floor(), by.floor());
        let bw = (right - bx).max(1.0);
        let bh = (bottom - by).max(1.0);
        let id = self.masks.len();
        let source_fingerprint = match &inputs {
            Some(reads) => fingerprint_rows(scene, reads.sources.iter().copied()),
            None => rendered_mask_fingerprint(scene, mask_node_id),
        };
        let cache_key = inputs.as_ref().and_then(|reads| {
            use std::hash::{Hash, Hasher};
            let mut key = rustc_hash::FxHasher::default();
            document_context.hash(&mut key);
            for id in &ancestors {
                crate::residency::node_hashes_in(scene, scene.effective_node(id)?).full.hash(&mut key);
            }
            source_fingerprint.hash(&mut key);
            reads.outside_clips.hash(&mut key);
            camera_modes.hash(&mut key);
            Some(key.finish())
        });
        self.masks.push(MaskPlan {
            source_node_id: mask_node_id.to_string(),
            owner_node_id: owner_id.to_string(),
            origin: [bx, by],
            size: [bw, bh],
            model_matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, bx as f32, by as f32, 0.0, 1.0],
            positions: if empty { Vec::new() } else { vec![0.0, 0.0, bw as f32, 0.0, bw as f32, bh as f32, 0.0, 0.0, bw as f32, bh as f32, 0.0, bh as f32] }.into(),
            mask_type,
            invert: false,
            feather: 0.0,
            draws: vec![MaskDraw { first: 0, count: if empty { 0 } else { 6 }, paint: MaskPaint::Source { opacity: 1.0 } }].into(),
            mode: 3,
            image: Some(MaskImage {
                rgba: Default::default(), width: 1, height: 1, offset: [0.0; 2],
                dims: [bw as f32, bh as f32], scale: [1.0; 2], rotation_rad: 0.0,
                surface: Some(format!("rendered-mask:{owner_id}")),
                rendered: Some(std::sync::Arc::new(crate::seams::RenderedMask {
                    document: std::sync::Arc::new(document), node_id: mask_node_id.to_string(),
                    shader_time_s: self.shader_time_s, source_fingerprint, cache_key,
                    prepared: Default::default(), shader_clock: Default::default(), asset_reads: Default::default(),
                })),
            }),
            text: None,
            ancestor,
        });
        Some(id)
    }
}

/// Page layout and root membership stand for the lifetime of the parent topology.
fn document_context(scene: &Scene) -> u64 {
    use std::hash::Hasher;
    let mut h = rustc_hash::FxHasher::default();
    h.write(b"rendered-mask");
    let doc = scene.doc();
    h.write(&moda_scene::value::to_vec(&(
        &doc.canvas_root_node_ids, &doc.layout_mode, &doc.grid_columns,
    )).expect("mask capture document inputs"));
    for page in &doc.pages {
        h.write(&moda_scene::value::to_vec(&page.0).expect("mask capture page"));
    }
    h.finish()
}

fn rendered_mask_fingerprint(scene: &Scene, mask_node_id: &str) -> u64 {
    let mut pending = vec![mask_node_id.to_string()];
    let mut sources = std::collections::BTreeSet::new();
    while let Some(node_id) = pending.pop() {
        if !sources.insert(node_id.clone()) {
            continue;
        }
        for children in [scene.children_by_parent().get(&node_id), scene.transient_by_parent().get(&node_id)].into_iter().flatten() {
            pending.extend(children.iter().cloned());
        }
        if let Some(node) = scene.effective_node(&node_id) {
            if node.node_type() == "blend" {
                pending.extend(moda_scene::blend::dependency_ids(node).into_iter().map(str::to_string));
            }
        }
    }
    fingerprint_rows(scene, sources.iter().map(String::as_str))
}

fn fingerprint_rows<'a>(scene: &Scene, sources: impl Iterator<Item = &'a str>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut fingerprint = rustc_hash::FxHasher::default();
    fingerprint.write(&moda_scene::value::to_vec(&&scene.doc().config.raw).expect("mask document inputs"));
    for node_id in sources {
        node_id.hash(&mut fingerprint);
        if let Some(node) = scene.effective_node(&node_id) {
            crate::residency::node_hashes_in(scene, node).full.hash(&mut fingerprint);
        }
        if let Some(transform) = scene.node_canvas_affine(&node_id) {
            for value in [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f] {
                value.to_bits().hash(&mut fingerprint);
            }
        }
    }
    fingerprint.finish()
}

fn is_mask_owner(node: &RawNode) -> bool {
    (node.node_type() == "group" && node.truthy("clipMaskNodeId"))
        || (node.node_type() == "container" && node.truthy("clipContent"))
}

/// What owns a mask context: a clipping group or frame (by id), or a layer clipped to the layer below (its base's
/// alpha; `Scene::clip_below_base`).
#[derive(Debug, Clone)]
enum MaskOwner {
    Clip(String),
    Below { clipped: String, base: String },
}

impl MaskOwner {
    /// The mask table key: the owner's id, or the base's (consecutive clipped layers share one mask).
    fn key(&self) -> String {
        match self {
            MaskOwner::Clip(id) => id.clone(),
            MaskOwner::Below { base, .. } => format!("clip-below:{base}"),
        }
    }

    /// Where the walk goes on past this owner: above a clipping group (its own clip below included), or above a
    /// clipped layer's own clip.
    fn continue_from(&self) -> (String, bool) {
        match self {
            MaskOwner::Clip(id) => (id.clone(), true),
            MaskOwner::Below { clipped, .. } => (clipped.clone(), false),
        }
    }
}

/// The nearest mask owner of `start`: its own clip to the layer below (when `include_self`), else walking its
/// parents (through transient hosts), a clipping group or frame, or a parent clipped to the layer below.
fn next_mask_owner(scene: &Scene, start: &str, include_self: bool) -> Option<MaskOwner> {
    if include_self {
        if let Some(base) = scene.clip_below_base(start) {
            return Some(MaskOwner::Below { clipped: start.to_string(), base: base.clone() });
        }
    }
    let mut cursor = start;
    let mut visited = moda_scene::scene::SeenIds::default();
    while let Some(parent_id) = scene
        .group_parent_of(cursor)
        .or_else(|| scene.transient_parent_of().get(cursor))
        .map(String::as_str)
    {
        if !visited.insert(parent_id) {
            break; // cycle guard
        }
        // Transient groups (an envelope's bent content, E8.17) own masks too.
        if scene.canonical_or_transient(parent_id).is_some_and(is_mask_owner) {
            return Some(MaskOwner::Clip(parent_id.to_string()));
        }
        if let Some(base) = scene.clip_below_base(parent_id) {
            return Some(MaskOwner::Below { clipped: parent_id.to_string(), base: base.clone() });
        }
        cursor = parent_id;
    }
    None
}

/// The canvas-space box a `clipContent` container clips its descendants to, as
/// `[min_x, min_y, max_x, max_y]`, or `None` when `node` is not a clipping
/// container. Every descendant paint is limited to it: bodies sample the
/// container's R8 mask, whose coverage is 0 outside this box (the texture spans
/// exactly `node_canvas_aabb`), and text / shadows / strokes take the effective
/// page clip, which `effective_page_clip` intersects with the same box.
pub fn container_clip_aabb(scene: &Scene, node: &RawNode) -> Option<[f64; 4]> {
    if node.node_type() != "container" || !node.truthy("clipContent") {
        return None;
    }
    // A degenerate box builds no mask (`build_mask` returns None) and the children
    // draw unclipped, so it bounds nothing.
    let (x, y, w, h) = node_canvas_aabb(scene, node)?;
    (x.is_finite() && y.is_finite() && w > 0.0 && h > 0.0 && w.is_finite() && h.is_finite())
        .then_some([x, y, x + w, y + h])
}

/// Whether a `clipContent` container's clip changes nothing for `node`: the container clips to its plain box (square
/// corners, axis-aligned, undistorted) and `node` paints only inside its own box (a fill-only shape: no visible
/// stroke, effect, fill/stroke effect, distort or label text), which lies within the container's. Such a node
/// skips the container's R8 mask and its box in the page clip. Both cut at the box's edge (the mask anti-aliased,
/// the page clip stepped), so on a fractional edge they cut into the node's own anti-aliased edge a second time:
/// two frames that abut at x = 333.33, each filled to its edge, left a light seam between them.
fn container_clip_is_redundant(scene: &Scene, container: &RawNode, node_id: &str) -> bool {
    const EPS: f64 = 1e-3;
    let Some([cx0, cy0, cx1, cy1]) = container_clip_aabb(scene, container) else {
        return false;
    };
    let square = moda_scene::shape_outline::expand_corner_radii(container.json_field("cornerRadius"))
        .iter()
        .all(|r| *r <= 1e-6);
    let axis_aligned = scene
        .node_canvas_affine(container.id())
        .is_some_and(|a| a.b.abs() <= 1e-9 && a.c.abs() <= 1e-9);
    if !square || !axis_aligned || crate::paint::node_distort(container).is_some() {
        return false;
    }
    let Some(node) = scene.effective_node(node_id) else {
        return false;
    };
    // Qualify the source shape, not its derived rendering path: booleans and
    // other composite nodes can resolve to paths but still need their clips.
    let fill_only = scene.canonical_or_transient(node_id).is_some_and(|source| {
        matches!(
            source.node_type(),
            "rectangle" | "ellipse" | "polygon" | "star" | "path" | "container"
        )
    }) && !moda_scene::paint::strokes_visibly(node)
        && !moda_scene::effect::node_effects_json(node)
            .iter()
            .any(moda_scene::effect::is_visible_json)
        && !crate::paint::layers::has_entry_effects(node)
        && crate::paint::node_distort(node).is_none()
        && node.str("textContent").is_none_or(str::is_empty);
    if !fill_only {
        return false;
    }
    node_canvas_aabb(scene, node).is_some_and(|(x, y, w, h)| {
        [x, y, w, h].iter().all(|v| v.is_finite())
            && x >= cx0 - EPS
            && y >= cy0 - EPS
            && x + w <= cx1 + EPS
            && y + h <= cy1 + EPS
    })
}

/// `Scene.getNodeBoundsCanvas` for mask nodes: transformed AABB of the
/// node-local box (w/h shapes) or the path's rendered outline. Distort-warped
/// GEOMETRY nodes use the warped local quad's AABB — their
/// coverage geometry warps, so the texture must cover the warped extent.
/// Richtext keeps the unwarped AABB (its coverage is the unwarped Degraded
/// raster — texture bounds must match the coverage lane).
fn node_canvas_aabb(scene: &Scene, node: &RawNode) -> Option<(f64, f64, f64, f64)> {
    let affine = scene.node_canvas_affine(node.id())?;
    // Richtext mask sources are EXCLUDED from the warped AABB: their coverage
    // is the UNWARPED Degraded raster quad (the accepted Degraded-
    // lane gap, see render_coverage), so the texture bounds must match that
    // unwarped coverage — a warped (contracting) AABB would crop the clip.
    if node.node_type() != "richtext" {
        if let Some(d) = crate::paint::node_distort(node) {
            // Map each warped local corner through the affine, THEN take the
            // AABB — identical to TS getNodeDistortCanvasBounds (and
            // backdrop/plan.rs). AABB-then-affine would over-approximate
            // under rotation.
            let apply = |x: f64, y: f64| {
                (
                    affine.a * x + affine.c * y + affine.e,
                    affine.b * x + affine.d * y + affine.f,
                )
            };
            let corners = [
                apply(d.corners[0].x, d.corners[0].y),
                apply(d.corners[1].x, d.corners[1].y),
                apply(d.corners[2].x, d.corners[2].y),
                apply(d.corners[3].x, d.corners[3].y),
            ];
            let min_x = corners.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
            let max_x = corners
                .iter()
                .map(|c| c.0)
                .fold(f64::NEG_INFINITY, f64::max);
            let min_y = corners.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
            let max_y = corners
                .iter()
                .map(|c| c.1)
                .fold(f64::NEG_INFINITY, f64::max);
            return Some((min_x, min_y, max_x - min_x, max_y - min_y));
        }
    }
    let local: (f64, f64, f64, f64) = if node.node_type() == "path" {
        let b = tess::drawn_node_path_bounds(node)?;
        (b.min_x, b.min_y, b.max_x - b.min_x, b.max_y - b.min_y)
    } else {
        let w = node.f64("width")?;
        let h = node.f64("height")?;
        (0.0, 0.0, w, h)
    };
    let (x0, y0, x1, y1) = transform::transformed_aabb(local, affine);
    Some((x0, y0, x1 - x0, y1 - y0))
}

/// Coverage geometry of a mask node. `fill_dropped` is the fail-closed
/// refusal of any part's fill (a non-finite coordinate, lyon's vertex cap);
/// `positions` then still holds whatever stroke bands tessellated, which
/// `build_mask` discards — a refused mask covers nothing.
#[derive(Default)]
struct MaskGeometry {
    positions: Vec<f32>,
    /// This node kind has no outline implementation and uses its box.
    box_fallback: bool,
    /// Per-vertex kind, parallel to `positions` pairs: 0 fill, 1 stroke band
    /// (the tessellator's `vertex_types`). Empty when unknown.
    vertex_types: Vec<f32>,
    /// The path's geometric bounds `[x0, y0, x1, y1]` (what gradients span,
    /// `paint::gradient_box`), when the node is a path-like shape.
    path_bounds: Option<[f64; 4]>,
    fill_dropped: Option<String>,
    stroke_dropped: Option<String>,
    /// A tolerated degradation to report (`PlanWarnings::skipped`).
    note: Option<String>,
}

/// Canvas AABB `(x, y, w, h)` of node-local mask positions.
fn positions_canvas_aabb(
    scene: &Scene,
    node: &RawNode,
    positions: &[f32],
) -> Option<(f64, f64, f64, f64)> {
    let affine = scene.node_canvas_affine(node.id())?;
    let (mut x0, mut y0, mut x1, mut y1) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for p in positions.chunks_exact(2) {
        let (x, y) = (p[0] as f64, p[1] as f64);
        let (cx, cy) = (
            affine.a * x + affine.c * y + affine.e,
            affine.b * x + affine.d * y + affine.f,
        );
        x0 = x0.min(cx);
        y0 = y0.min(cy);
        x1 = x1.max(cx);
        y1 = y1.max(cy);
    }
    (x0 <= x1 && y0 <= y1).then(|| (x0, y0, x1 - x0, y1 - y0))
}

fn union_aabb(
    a: Option<(f64, f64, f64, f64)>,
    b: Option<(f64, f64, f64, f64)>,
) -> Option<(f64, f64, f64, f64)> {
    match (a, b) {
        (Some(a), Some(b)) => {
            let (x0, y0) = (a.0.min(b.0), a.1.min(b.1));
            let (x1, y1) = ((a.0 + a.2).max(b.0 + b.2), (a.1 + a.3).max(b.1 + b.3));
            Some((x0, y0, x1 - x0, y1 - y0))
        }
        (a, b) => a.or(b),
    }
}

/// Mask node → tessellated coverage positions (fill + stroke vertices) in
/// node-local space — `renderCoverage`'s geometry branch via the same
/// shape→path converters (`tessellatePathNodeForMask`).
fn mask_geometry_positions(node: &RawNode, scene: Option<&Scene>) -> MaskGeometry {
    // Masks outline the same stroke the canvas draws: same miter limit.
    tess::with_miter_limit(node.miter_limit(), || {
        mask_geometry_positions_inner(node, scene)
    })
}

fn mask_geometry_positions_inner(node: &RawNode, scene: Option<&Scene>) -> MaskGeometry {
    // Brush plan (#107): an art / scatter / pattern stroke masks with its
    // placed artwork (plus the path's own fill), as it draws.

    let w = node.f64("width").unwrap_or(0.0);
    let h = node.f64("height").unwrap_or(0.0);
    // Line masks tessellate per PART, mirroring `tessellateLineForMask`
    // (WebGPUMaskRenderer.ts:503-525): no arrows → the integrated line path
    // (`convertLineToPathNode`); ANY arrow → body + arrowhead paths, all
    // positions concatenated. Each part carries its own strokeWidth/cap/join
    // (arrow fills tessellate with strokeWidth 0); strokeAlign defaults to
    // center like the converted PathNode (`pathNode.strokeAlign ?? 'center'`).
    if node.node_type() == "line" {
        use tess::line_to_path as ltp;
        let mut spec = ltp::LineSpec::from_json(|k| node.json_field(k));
        // Brush plan §4.9: a line with a width profile / taper / wave masks with its body's
        // outline (built on the undashed body, dash as arc-length windows), as it draws.

        let parts_of = |spec: &ltp::LineSpec| -> Vec<ltp::LinePart> {
            if spec.has_arrows() {
                let mut v = vec![ltp::convert_line_body(spec)];
                v.extend(ltp::convert_arrow_heads(spec));
                v
            } else {
                vec![ltp::convert_line_to_path(spec)]
            }
        };
        let parts = parts_of(&spec);
        let mut positions: Vec<f32> = Vec::new();
        let mut fill_dropped: Option<String> = None;
        let mut note: Option<String> = None;
        for part in &parts {

            let geometry = tess::tessellate_combined(
                &part.cmds,
                part.stroke_width,
                part.closed,
                pt::line_cap_from_str(part.line_cap.as_deref()),
                pt::line_join_from_str(part.line_join.as_deref()),
                pt::FillRule::NonZero,
                pt::stroke_align_from_str(None),
            );
            if fill_dropped.is_none() {
                fill_dropped = geometry.fill_dropped;
            }
            positions.extend(geometry.positions);
        }
        // A line paints only strokes (its arrowheads fill with the stroke
        // paint), so every vertex is stroke.
        let vertex_types = vec![1.0; positions.len() / 2];
        return MaskGeometry {
            positions,
            vertex_types,
            path_bounds: None,
            box_fallback: false,
            fill_dropped,
            note,
            ..Default::default()
        };
    }
    if node.node_type() == "path" {
        let cmds = tess::path_cmds_for_node(node);
        let style = moda_scene::paint::node_stroke_style(node);
        let scale = scene
            .and_then(|s| s.node_canvas_affine(moda_scene::scene::paint_transform_id(node)))
            .map_or(1.0, |a| tess::curve_quality::affine_max_scale(&a));
        let _node_scale = tess::curve_quality::enter_node_scale(scale);
        let geometry = tess::tessellate_body(
            node,
            &cmds,
            node.path_closure(),
            crate::paint::painted_stroke_width(node),
            pt::fill_rule_from_str(node.str("fillRule")),
            &style,
        );
        return MaskGeometry {
            positions: geometry.positions.clone(),
            vertex_types: geometry.vertex_types.clone(),
            path_bounds: tess::path_bounds(&cmds),
            fill_dropped: geometry.fill_dropped.clone(),
            stroke_dropped: geometry.stroke_dropped.clone(),
            ..Default::default()
        };
    }
    let cmds = match node.node_type() {
        "rectangle" => shape_to_path::rect_outline_cmds(w, h, node.json_field("cornerRadius"), node.json_field("cornerSmoothing"), node.json_field("cornerStyle")),
        "container" => shape_to_path::rect_outline_cmds(w, h, node.json_field("cornerRadius"), node.json_field("cornerSmoothing"), node.json_field("cornerStyle")),
        "ellipse" => shape_to_path::ellipse_to_path_cmds(
            w,
            h,
            node.f64("startAngle"),
            node.f64("sweepAngle"),
            node.f64("innerRadiusRatio"),
        ),
        // M2 converters, same cmds as the render path (frame.rs body arms):
        // `convertStarToPathNode` / `convertPolygonToPathNode` with center
        // (w/2, h/2), rotation 0, `cornerRadius || 0`.
        "star" => tess::shape_paths::star_to_path_cmds(
            w,
            h,
            node.f64("numPoints").unwrap_or(f64::NAN),
            node.f64("innerRadiusRatio"),
            node.f64("cornerRadius")
                .filter(|v| v.is_finite())
                .unwrap_or(0.0),
        ),
        "polygon" => tess::shape_paths::polygon_to_path_cmds(
            w,
            h,
            node.f64("sides").unwrap_or(f64::NAN),
            node.f64("cornerRadius")
                .filter(|v| v.is_finite())
                .unwrap_or(0.0),
        ),
        // Remaining types (image/blend/table/...) have no geometry arm in the
        // browser either (`renderCoverage`'s if-chain falls through with
        // empty vertices) — the full-quad fallback IS production behavior.
        _ => {
            return MaskGeometry {
                positions: Vec::new(),
                box_fallback: true,
                fill_dropped: None,
                ..Default::default()
            }
        }
    };
    // A distort-warped mask source clips to the WARPED silhouette —
    // the same pre-warp the render path applies (frame.rs build_body_item). On
    // warp fallback the unwarped cmds stay while the texture AABB above is
    // warped (a contracting warp then under-covers the unwarped geometry) —
    // acceptable: only sub-3-point degenerate outlines can decline.
    let cmds = match crate::paint::node_distort(node) {
        Some(d) => tess::shape_paths::warp_path_cmds(&cmds, &d.local).unwrap_or(cmds),
        None => cmds,
    };
    // Mask coverage matches the painted silhouette: an unpainted stroke emits no
    // band in the body lane, so it must not widen the mask either.
    let stroke_width = crate::paint::painted_stroke_width(node);
    let style = moda_scene::paint::node_stroke_style(node);

    let geometry = tess::tessellate_combined(
        &cmds,
        stroke_width,
        true,
        pt::line_cap_from_str(style.cap.as_deref()),
        pt::line_join_from_str(style.join.as_deref()),
        pt::fill_rule_from_str(node.str("fillRule")),
        pt::stroke_align_from_str(style.align.as_deref()),
    );
    MaskGeometry {
        positions: geometry.positions,
        vertex_types: geometry.vertex_types,
        path_bounds: tess::path_bounds(&cmds),
        fill_dropped: geometry.fill_dropped,
        ..Default::default()
    }
}

/// Store-backed image paints use the ordinary pattern sampler, including its mip chain
/// and working-colour conversion. Capture that drawn mask just as for a group of paints.
fn mask_reads_store_image(node: &RawNode) -> bool {
    use moda_scene::paint as sp;
    let stored = |paint: &sp::Paint| {
        let sp::Paint::Image(image) = paint else { return false };
        image.src.as_deref().is_some_and(|src| !src.starts_with("data:")
            && !src.starts_with(PIXEL_SINK_SRC_PREFIX)
            && moda_assets::engine_seam::resolve_streaming(src).is_some())
    };
    sp::node_fills(node).iter().any(|f| sp::shown(f.visible, f.opacity) && stored(&f.paint))
        || sp::node_strokes(node).iter().any(|s| sp::shown(s.visible, s.opacity) && s.width > 0. && stored(&s.paint))
}

/// Whether a mask shape's picture is one layer of one solid colour: one visible fill or stroke, a solid paint.
/// Reducing that paint to luminance is then the luminance of the shape's picture in whatever encoding the
/// engine blends in: the picture is that colour, converted there and back. Several layers blend in the
/// engine's encoding before they are reduced; a gradient's picture is interpolated in its own space, which the
/// coverage shader's gradient does not model; and text, an image or a shader is a picture of its own.
fn one_solid_layer(node: &RawNode) -> bool {
    use moda_scene::paint as sp;
    if node.node_type() == "richtext" {
        return false;
    }
    let authored = |paint: &sp::Paint| matches!(paint, sp::Paint::Solid(_));
    let fills: Vec<sp::Fill> = if node.node_type() == "line" {
        Vec::new()
    } else {
        sp::node_fills(node).into_iter().filter(|f| sp::shown(f.visible, f.opacity)).collect()
    };
    let strokes: Vec<sp::Stroke> =
        sp::node_strokes(node).into_iter().filter(|s| sp::shown(s.visible, s.opacity) && s.width > 0.0).collect();
    fills.len() + strokes.len() <= 1
        && fills.iter().all(|fill| authored(&fill.paint))
        && strokes.iter().all(|stroke| authored(&stroke.paint))
}

/// An alpha or luminance mask's coverage draws (E7.18): every visible fill
/// layer on the node's fill geometry, bottom to top, then every visible
/// stroke layer on its stroke band, each as a clone of the node carrying just
/// that layer (as the body lanes draw a paint stack, `paint::layers`). Blend
/// modes are ignored: source-over of the layers' alpha is the stack's alpha
/// whatever the modes (luminance composites the colours source-over too).
struct LayerDraws {
    positions: Vec<f32>,
    draws: Vec<MaskDraw>,
    image: Option<MaskImage>,
    fill_dropped: Option<String>,
    strokes_dropped: Vec<String>,
    notes: Vec<String>,
}

impl LayerDraws {
    fn push(&mut self, positions: &[f32], paint: MaskPaint) {
        if positions.is_empty() {
            return;
        }
        self.draws.push(MaskDraw {
            first: (self.positions.len() / 2) as u32,
            count: (positions.len() / 2) as u32,
            paint,
        });
        self.positions.extend_from_slice(positions);
    }
}

fn paint_layer_draws(scene: &Scene, node: &RawNode, node_opacity: f32) -> LayerDraws {
    use moda_scene::paint as sp;
    let mut out = LayerDraws {
        positions: Vec::new(),
        draws: Vec::new(),
        image: None,
        fill_dropped: None,
        strokes_dropped: Vec::new(),
        notes: Vec::new(),
    };
    let layer_opacity = |o: Option<f64>| o.unwrap_or(1.0).clamp(0.0, 1.0) as f32;
    // A line has strokes only (its arrowheads fill with the stroke paint).
    let fills: Vec<sp::Fill> = if node.node_type() == "line" {
        Vec::new()
    } else {
        sp::node_fills(node)
            .into_iter()
            .filter(|f| sp::shown(f.visible, f.opacity))
            .collect()
    };
    let strokes: Vec<sp::Stroke> = sp::node_strokes(node)
        .into_iter()
        .filter(|s| sp::shown(s.visible, s.opacity) && s.width > 0.0)
        .collect();
    let fill_stroke_width = crate::paint::painted_stroke_width(node);
    // The mask has one source texture: the topmost image fill samples it.
    let image_layer = fills
        .iter()
        .rposition(|f| matches!(f.paint, sp::Paint::Image(_)));
    for (i, fill) in fills.iter().enumerate() {
        let mut m = node.fields().clone();
        sp::set_fills(&mut m, vec![fill.clone()]);
        sp::clear_strokes(&mut m);
        let g = mask_geometry_positions(&node.with_fields(m), Some(scene));
        if g.fill_dropped.is_some() {
            out.fill_dropped = g.fill_dropped;
            return out;
        }
        let opacity = node_opacity * layer_opacity(fill.opacity);
        let paint = match &fill.paint {
            sp::Paint::Image(img) if Some(i) == image_layer => match image_mask_source(node, img) {
                Some(src) => {
                    out.image = Some(src);
                    MaskPaint::Source { opacity }
                }
                None => {
                    out.notes.push(
                        "image fill masks as opaque (a tiled, distorted or undecoded image)".into(),
                    );
                    MaskPaint::Solid([1.0, 1.0, 1.0, opacity])
                }
            },
            paint => layer_paint(paint, opacity, &g, fill_stroke_width, &mut out.notes),
        };
        out.push(&g.positions, paint);
    }
    for stroke in &strokes {
        let mut m = node.fields().clone();
        sp::set_fills(&mut m, Vec::new());
        sp::set_strokes(&mut m, vec![stroke.clone()]);
        let g = mask_geometry_positions(&node.with_fields(m), Some(scene));
        if let Some(why) = &g.stroke_dropped {
            out.strokes_dropped.push(why.clone());
        }
        // The clone's geometry is fill + stroke band: keep the band.
        let band: Vec<f32> = if g.vertex_types.len() * 2 == g.positions.len() {
            g.positions
                .chunks_exact(6)
                .zip(g.vertex_types.chunks_exact(3))
                .filter(|(_, t)| t.iter().all(|t| *t == 1.0))
                .flat_map(|(tri, _)| tri.iter().copied())
                .collect()
        } else {
            g.positions.clone()
        };
        let opacity = node_opacity * layer_opacity(stroke.opacity);
        let paint = layer_paint(&stroke.paint, opacity, &g, stroke.width, &mut out.notes);
        out.push(&band, paint);
    }
    out
}

/// A fill or stroke layer's paint as a coverage paint, alpha × `opacity`.
/// Gradients span the path's geometric bounds, as they draw
/// (`paint::gradient_box`). Paints a mask can't evaluate (a non-top image,
/// video, shaders) mask as opaque, with a note.
fn layer_paint(
    paint: &moda_scene::paint::Paint,
    opacity: f32,
    g: &MaskGeometry,
    stroke_width: f64,
    notes: &mut Vec<String>,
) -> MaskPaint {
    use moda_scene::paint::Paint;
    match paint {
        Paint::Solid(_) => {
            let c = crate::paint::gradient::solid_fill_color(Some(paint));
            MaskPaint::Solid([c.r as f32, c.g as f32, c.b as f32, c.a as f32 * opacity])
        }
        Paint::Linear(_) | Paint::Radial(_) | Paint::Conic(_) | Paint::Diamond(_) => {
            let gp = crate::paint::gradient::fill_gradient_paint(Some(paint), false);
            MaskPaint::Gradient {
                fill_type: gp.fill_type,
                start: gp.gradient_start,
                end: gp.gradient_end,
                rotation: gp.gradient_rotation,
                num_stops: gp.num_stops,
                stops: gp.gradient_stops,
                bounds: crate::paint::gradient_box(
                    g.path_bounds,
                    stroke_width,
                    positions_local_box(&g.positions),
                ),
                opacity,
            }
        }
        other => {
            let kind = match other {
                Paint::Image(_) => "image",
                Paint::Video(_) => "video",

                _ => "shader",
            };
            notes.push(format!("{kind} paint masks as opaque"));
            MaskPaint::Solid([1.0, 1.0, 1.0, opacity])
        }
    }
}

/// Node-local `[x, y, w, h]` of positions (zeros when empty).
fn positions_local_box(positions: &[f32]) -> [f32; 4] {
    let (mut x0, mut y0, mut x1, mut y1) = (
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    );
    for p in positions.chunks_exact(2) {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    if x0 > x1 || y0 > y1 {
        return [0.0; 4];
    }
    [x0, y0, x1 - x0, y1 - y0]
}

/// `getImageMaskSrc` + `computeImagePlacement` + decode, for an alpha or
/// luminance mask's topmost image fill (E7.18: a vector mask ignores the
/// image). Only `data:` URIs decode natively in M3d; anything else masks as
/// an opaque fill over the geometry (production behavior for not-yet-loaded
/// textures).
///
/// MIRRORED IN TWO PLACES — update them together when this gate changes:
/// - `WebGPUMaskRenderer.getImageMaskSrc`
///   (renderers/webgpu/rendering/systems/WebGPUMaskRenderer.ts), the fallback
///   renderer. It diverges deliberately on distort: it falls back for
///   `rectangle` only, where this falls back for any distort-capable type.
/// - `getClipMaskImageAlphaFallback` (core/scene/clipMaskImageAlpha.ts), the
///   authoring-time mirror that warns an agent when a mask will silently clip
///   by geometry instead of by image alpha. It follows THIS gate's distort
///   rule, since this is the primary engine.
/// The src scheme of a pixel sink's image (an image the pixel graph hands back to the stack through the installed
/// asset store); a mask samples its alpha (see [`image_mask_source`]).
pub const PIXEL_SINK_SRC_PREFIX: &str = "moda-pixel-sink://";

fn image_mask_source(node: &RawNode, image: &moda_scene::paint::ImagePaint) -> Option<MaskImage> {
    if matches!(node.node_type(), "container" | "richtext") {
        return None;
    }
    let src = image.src.as_deref()?;
    if image.placement.mode() == "tile" {
        return None;
    }
    // A valid distort falls back to geometry coverage (which IS
    // warped, see mask_geometry_positions) — the image-mask placement math
    // cannot represent the warp. An invalid distort renders unwarped
    // everywhere, so it keeps the image-mask path, consistently.
    if crate::paint::node_distort(node).is_some() {
        return None;
    }
    // A pixel sink's image (`moda-cli` graph_pixels puts a sink back as an image rectangle the size of its domain,
    // #1702): its pixels are in the installed asset store, and a layer clipped to it, or masked by it, follows its
    // drawn alpha (a blurred base clips softly), not the rectangle.
    if src.starts_with(PIXEL_SINK_SRC_PREFIX) {
        let resolved = moda_assets::engine_seam::resolve(src)?;
        if resolved.state != moda_assets::ImageFillState::Loaded {
            return None;
        }
        // Its pixels on the heap, or (a 3D unit kept on the GPU) its registered texture.
        let (rgba, surface) = match resolved.pixels {
            moda_assets::engine_seam::ResolvedPixels::Heap(rgba) => (rgba, None),
            _ if crate::external_texture::capture_texture(src).is_some() => (Default::default(), Some(src.to_string())),
            _ => return None,
        };
        let (w, h) = (resolved.width, resolved.height);
        let (fx, fy) = resolved.scale_factor;
        let placement = image.placement.resolve(
            moda_scene::paint::node_image_box(node).unwrap_or([0.0; 4]),
            Some((w as f64 / fx, h as f64 / fy)),
        );
        return Some(MaskImage {
            rgba,
            width: w,
            height: h,
            offset: [placement.offset.0 as f32, placement.offset.1 as f32],
            dims: [w.max(1) as f32, h.max(1) as f32],
            scale: [(placement.scale.0 / fx) as f32, (placement.scale.1 / fy) as f32],
            rotation_rad: placement.rotation.to_radians() as f32,
            surface,
            rendered: None,
        });
    }
    let (rgba, w, h) = decode_data_uri(src).ok()?;

    // The scale mode resolved against the node's box (Fill / Fit re-fit live).
    let placement = image.placement.resolve(
        moda_scene::paint::node_image_box(node).unwrap_or([0.0; 4]),
        Some((w as f64, h as f64)),
    );
    let offset = [placement.offset.0 as f32, placement.offset.1 as f32];
    let scale = [placement.scale.0 as f32, placement.scale.1 as f32];
    let rotation_rad = placement.rotation.to_radians() as f32;

    Some(MaskImage {
        rgba: rgba.into(),
        width: w,
        height: h,
        offset,
        // Texture dims from the DECODED image (textureManager
        // getTextureDimensions), min 1; scaleFactor 1 for raster sources.
        dims: [w.max(1) as f32, h.max(1) as f32],
        scale,
        rotation_rad,
        surface: None,
        rendered: None,
    })
}

fn decode_data_uri(src: &str) -> Result<(Vec<u8>, u32, u32)> {
    let rest = src
        .strip_prefix("data:")
        .ok_or_else(|| anyhow!("not a data URI"))?;
    let (_, payload) = rest
        .split_once(";base64,")
        .ok_or_else(|| anyhow!("not base64 data URI"))?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .context("base64 decode")?;
    let img = image::load_from_memory(&bytes).context("image decode")?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Ok((rgba.into_raw(), w, h))
}

/// M4d: build the text-alpha mask source for a richtext mask node — the
/// native analogue of `getStandaloneTextureForMask` (raster + the node's
/// canvas position as `originCanvas`).
///
/// Q-104 (amended): the BROWSER always rasterizes the Canvas2D texture for
/// masks under every engine flag (`getStandaloneTextureForMask` is Canvas2D
/// unconditionally) — but natively that Canvas2D stand-in was the cosmic-text
/// Degraded raster, whose letterforms measurably deviate from Chromium's
/// (IoU 0.605 vs the browser golden on `masks/text-alpha`) AND from the
/// native vector lane the SAME node takes as visible text (IoU 0.65
/// mask-vs-visible). That self-inconsistency is a true quality gap, so under
/// the GA vector engine (`vector_masks`, from `opts.text_engine`) the raster
/// is now built from the exact vector glyph outlines
/// (`moda_text::vector::render_mask_raster`) at the same texture geometry
/// (Degraded `canvas_logical_size` sizing and the Q-68 top-left anchor
/// unchanged); ANY vector decline falls back to the Degraded
/// raster — byte-identical to the old behavior. msdf/hybrid engines keep the
/// pure Degraded raster (their comparison target is the default-config
/// golden). This function still never enters `text_draw::build_batch`.
fn text_mask_source(scene: &Scene, node: &RawNode, vector_masks: bool) -> Result<Option<MaskText>> {
    let mut spec = crate::text_draw::node_to_text_spec_in(scene, node);
    // The mask samples the HTML's per-run alphas, not the node's text fill (`moda_text::vector::render_mask_raster`
    // clears it the same way): the Degraded raster now draws `fill_override`, so the fallback below clears it too.
    spec.fill_override = None;
    if spec.html.is_empty() {
        // MIRRORED at authoring time: `getClipMaskTextCoverageCaveat`
        // (core/scene/clipMaskImageAlpha.ts) warns agents that this case
        // full-quads and that non-empty blank markup ("<p></p>") instead
        // rasterizes to zero coverage and hides the children — move that
        // mirror with any change to this gate.
        return Ok(None); // no content → geometry/full-quad coverage
    }
    #[cfg(test)]
    crate::test_fonts();
    let engine = moda_text::degraded::engine().map_err(|e| anyhow!("degraded engine: {e}"))?;
    let vector_raster = if vector_masks {
        vector_text_mask_raster(&spec, &engine)
    } else {
        None
    };
    let raster = match vector_raster {
        Some(r) => r,
        None => engine
            .render_node(&spec)
            .map_err(|e| anyhow!("degraded raster: {e}"))?,
    };
    let affine = scene
        .node_canvas_affine(node.id())
        .ok_or_else(|| anyhow!("no canvas affine"))?;
    let origin_canvas = [affine.e, affine.f - raster.pad_top];
    Ok(Some(MaskText {
        raster: std::sync::Arc::new(raster),
        origin_canvas,
    }))
}

/// Try the vector-lane mask raster; `None` = fall back to the Degraded
/// raster (missing vector store/host, sizing engine without fonts, or a
/// per-node decline from `render_mask_raster` — silhouette-affecting
/// features, non-vector-servable runs).
fn vector_text_mask_raster(
    spec: &moda_text::ops::TextNodeSpec,
    degraded: &moda_text::degraded::DegradedEngine,
) -> Option<moda_text::degraded::DegradedRaster> {
    let (store, host) = crate::text_draw::vector_store_and_host().ok()?;
    // `canvas_logical_size` shapes via cosmic-text and panics on an empty
    // fontdb; without degraded fonts the legacy path errors out anyway.
    if !degraded.has_fonts() {
        return None;
    }
    let (logical_w, logical_h) = degraded.canvas_logical_size(spec);
    match moda_text::vector::render_mask_raster(spec, logical_w, logical_h, &store, &host.catalog) {
        Ok(r) => Some(r),
        Err(reason) => {
            // Same reporting seam as the visible-text vector declines
            // (`try_build_vector_batch`) — resource-report event, not stderr,
            // so the live wasm editor doesn't log on every plan rebuild.
            moda_assets::report::record(moda_assets::report::RuntimeEvent::FontFallbackClass {
                node_id: spec.node_id.clone(),
                reason: format!("mask-vector→degraded: {reason}"),
                // The mask lane's declines are a single bucket: `reason` here is
                // free text (it forwards the vector decision's interpolated
                // reason), and the mask silhouette tier is not what the
                // vector-vs-MSDF share measures.
                cause: "mask-vector→degraded".to_string(),
                families: Vec::new(),
            });
            None
        }
    }
}

/// `computeEffectivePageClip` (`MaskContextUtils.ts:113-174`): plain page
/// clip ∩ AABBs of ancestor RECTANGLE group-mask nodes ∩ container bounds.
///
/// Empty intersection → production returns **null** (`intersectRect`
/// returning null short-circuits the walk, :147-149/:167-168) and the
/// CONSUMERS diverge — Q-113:
/// - shape/line bodies set no `_pageClipBoundsOverride` and keep the plain
///   page clip (`WebGPUShapeSystem.ts:720-721` `if (clip) …`); the R8 mask
///   they sample still produces the exact (empty) coverage;
/// - richtext draws NOTHING — vector `WebGPUTextSystem.ts:663-665`
///   (`if (!effectiveClip) return true`) and MSDF `:883-886`
///   (`if (!effectiveClip) return`), text never samples the R8 mask.
///
/// `None` here = production's null; callers pick their own fallback.
pub fn effective_page_clip(scene: &Scene, node_id: &str, base: [f32; 4]) -> Option<[f32; 4]> {
    effective_page_clip_until(scene, node_id, base, None)
}

/// [`effective_page_clip`] counting only the clipping ancestors up to and
/// including `stop` (a subtree capture's root: its descendants clip as they
/// do inside it, not by what clips the root).
pub fn effective_page_clip_until(
    scene: &Scene,
    node_id: &str,
    base: [f32; 4],
    stop: Option<&str>,
) -> Option<[f32; 4]> {
    let mut current = [
        base[0] as f64,
        base[1] as f64,
        base[2] as f64,
        base[3] as f64,
    ];
    let mut cursor = node_id;
    let mut visited = moda_scene::scene::SeenIds::default();
    // Step THROUGH transient hosts (see mask_ref_for_node).
    while let Some(parent_id) = scene
        .group_parent_of(cursor)
        .or_else(|| scene.transient_parent_of().get(cursor))
        .map(String::as_str)
    {
        if !visited.insert(parent_id) {
            break; // cycle guard — corrupt parent data must not hang the renderer
        }
        let Some(parent) = scene.canonical_or_transient(parent_id) else {
            cursor = parent_id;
            continue;
        };
        let rect: Option<(f64, f64, f64, f64)> =
            if parent.node_type() == "group" && parent.truthy("clipMaskNodeId") {
                parent
                    .str("clipMaskNodeId")
                    .and_then(|mid| scene.effective_node(mid))
                    // A stroked mask reaches past its box (E7.18 masks with
                    // the stroke): no box hint then.
                    .filter(|m| m.node_type() == "rectangle" && !moda_scene::paint::strokes_visibly(m)
                        && !m.truthy("maskInvert") && m.f64("maskFeather").unwrap_or(0.0) <= 0.0
                        && !crate::layer_blur::plan::needs_layer_capture(m)
                        && !crate::effects::shadow::has_drop_shadow_quad(m))
                    .and_then(|m| node_canvas_aabb(scene, m))
            } else if parent.node_type() == "container"
                && parent.truthy("clipContent")
                && !container_clip_is_redundant(scene, parent, node_id)
            {
                node_canvas_aabb(scene, parent)
            } else {
                None
            };
        if let Some((mx, my, mw, mh)) = rect {
            match intersect_rect(current, [mx, my, mw, mh]) {
                Some(r) => current = r,
                None => return None, // production: computeEffectivePageClip → null
            }
        }
        if stop == Some(parent_id) {
            break;
        }
        cursor = parent_id;
    }
    Some([
        current[0] as f32,
        current[1] as f32,
        current[2] as f32,
        current[3] as f32,
    ])
}

/// Whether a clip-mask group above `node_id` clips by more than the box
/// [`effective_page_clip`] takes from it (#2148). Text binds no R8 mask, so the
/// effective clip is all that clips it: exact for a mask that covers its box
/// fully and nothing else (an unstroked, square-cornered, axis-aligned
/// rectangle, opaque in its mask type), and missing every other mask (a path,
/// an ellipse, a rounded or rotated rectangle, a translucent or luminance one).
pub fn has_non_box_clip_mask(scene: &Scene, node_id: &str) -> bool {
    let mut cursor = node_id;
    let mut visited = moda_scene::scene::SeenIds::default();
    while let Some(parent_id) = scene
        .group_parent_of(cursor)
        .or_else(|| scene.transient_parent_of().get(cursor))
        .map(String::as_str)
    {
        if !visited.insert(parent_id) {
            break; // cycle guard
        }
        if let Some(mid) = scene
            .canonical_or_transient(parent_id)
            .filter(|p| p.node_type() == "group" && p.truthy("clipMaskNodeId"))
            .and_then(|p| p.str("clipMaskNodeId"))
        {
            if !scene.effective_node(mid).is_some_and(|m| clip_mask_is_box(scene, m)) {
                return true;
            }
        }
        cursor = parent_id;
    }
    false
}

/// Whether a text body can consume this mask as an ordinary box clip.
pub(crate) fn clip_mask_is_box(scene: &Scene, m: &RawNode) -> bool {
    const EPS: f64 = 1e-6;
    if m.node_type() != "rectangle"
        || m.truthy("maskInvert")
        || m.f64("maskFeather").unwrap_or(0.0) > 0.0
        || moda_scene::paint::strokes_visibly(m)
        || crate::layer_blur::plan::needs_layer_capture(m)
        || crate::effects::shadow::has_drop_shadow_quad(m)
        || moda_scene::shape_outline::expand_corner_radii(m.json_field("cornerRadius"))
            .iter()
            .any(|r| *r > EPS)
        || scene
            .node_canvas_affine(m.id())
            .is_none_or(|a| a.b.abs() > EPS || a.c.abs() > EPS)
    {
        return false;
    }
    match MaskType::from_node_value(m.str("maskType")) {
        MaskType::Vector => true,
        MaskType::Luminance => false,
        MaskType::Alpha => {
            m.f64("opacity").unwrap_or(1.0) >= 1.0
                && moda_scene::paint::node_fills_shared(m).iter().any(|f| {
                    f.visible != Some(false)
                        && f.opacity.unwrap_or(1.0) >= 1.0
                        && matches!(f.paint, moda_scene::paint::Paint::Solid(_))
                        && crate::paint::gradient::solid_fill_color(Some(&f.paint)).a >= 1.0
                })
        }
    }
}

/// The rounded corners of the innermost rounded clipping frame (`container`
/// with `clipContent` and a corner radius) above `node_id`, in canvas space —
/// what a quad clipped only to [`effective_page_clip`]'s box would still
/// paint outside of. `None` when no clipping frame above is rounded, or the
/// innermost rounded one is rotated or sheared (then its box is all the quad
/// is clipped to). Smoothed corners (`cornerSmoothing`) are taken as circular.
pub fn clip_rounded_rect(scene: &Scene, node_id: &str) -> Option<crate::seams::ClipRRect> {
    clip_rounded_rect_until(scene, node_id, None)
}

/// [`clip_rounded_rect`] counting only the clipping frames up to and including `stop` (a capture's root: what
/// it holds is clipped as inside it, [`effective_page_clip_until`]'s twin).
pub fn clip_rounded_rect_until(scene: &Scene, node_id: &str, stop: Option<&str>) -> Option<crate::seams::ClipRRect> {
    const EPS: f64 = 1e-6;
    let mut cursor = node_id;
    let mut visited = moda_scene::scene::SeenIds::default();
    while let Some(parent_id) = scene
        .group_parent_of(cursor)
        .or_else(|| scene.transient_parent_of().get(cursor))
        .map(String::as_str)
    {
        if !visited.insert(parent_id) {
            break;
        }
        if let Some(parent) = scene
            .node(parent_id)
            .filter(|p| p.node_type() == "container" && p.truthy("clipContent"))
        {
            let w = parent.f64("width").unwrap_or(0.0);
            let h = parent.f64("height").unwrap_or(0.0);
            let radii = moda_scene::shape_outline::normalize_rounded_rect_radii(
                w,
                h,
                parent.json_field("cornerRadius"),
            );
            if radii.iter().any(|r| *r > EPS) {
                let a = scene.node_canvas_affine(parent_id)?;
                if a.b.abs() > EPS || a.c.abs() > EPS {
                    return None;
                }
                let (x, y, bw, bh) = node_canvas_aabb(scene, parent)?;
                let (sx, sy) = (a.a.abs(), a.d.abs());
                // Mirrored axes swap which corner lands where.
                let [mut tl, mut tr, mut br, mut bl] = radii;
                if a.a < 0.0 {
                    (tl, tr, br, bl) = (tr, tl, bl, br);
                }
                if a.d < 0.0 {
                    (tl, tr, br, bl) = (bl, br, tr, tl);
                }
                let s = sx.min(sy);
                return Some(crate::seams::ClipRRect {
                    rect: [x as f32, y as f32, bw as f32, bh as f32],
                    radii: [tl, tr, br, bl].map(|r| (r * s) as f32),
                });
            }
        }
        if stop == Some(parent_id) {
            break;
        }
        cursor = parent_id;
    }
    None
}

/// Table cell text overflow: narrow an already-effective clip to the
/// node's OWN canvas AABB, but ONLY for a node carrying a truthy
/// `clipTextToBounds`.
///
/// Richtext is otherwise deliberately NOT clipped to its own rect (the standing
/// invariant restated at `WebGPUTextSystem.ts:974-979`, and in
/// `build_richtext_item`'s doc) — overflowing text is a designed affordance on a
/// free-standing text node. Table CELLS are the exception: a cell's text must not
/// bleed into its neighbours, so `expand_table_node` sets the flag on every
/// cell's richtext content node and the renderer clips there.
///
/// No flag ⇒ `Some(clip)` verbatim. That is the byte-exactness contract: every
/// doc predating the field walks through this untouched, so the render plan is
/// bit-identical (proven by the `mutation_replay` gate).
///
/// The TS fallback renderer carries the mirrored change (same PR):
/// `applyNodeTextBoundsClip` (`MaskContextUtils.ts:137`), applied at BOTH
/// `WebGPUTextSystem.ts` clip sites — the vector arm `:740-747` and the MSDF arm
/// `:980-985`. Both already early-return on a null clip, so `None` needs no new
/// consumer there either.
///
/// RESIDUAL 1 (documented, accepted): the clip is an AXIS-ALIGNED box, so it is
/// LOOSE under rotation — a rotated cell clips to the AABB of its rotated box,
/// not the box itself, and glyphs can still show in the corner slack. This is
/// exactly the pre-existing behavior of container `clipContent` and group rect
/// masks in `effective_page_clip` above (which take `node_canvas_aabb` too), and
/// the TS helper's own note calls the AABB "a coarse axis-aligned clip rect …
/// an optimization, not a correctness mechanism". Tight rotated clipping would
/// need the R8 coverage-mask path, which the MSDF/vector text pipelines do not
/// bind.
///
/// RESIDUAL 2 (unreachable today, so deliberately NOT worked around): TS reads
/// the box from `scene.getNodeBoundsCanvas`, which differs from
/// `node_canvas_aabb` on two axes that a table cell never exercises —
/// (a) DISTORT: `getNodeDistortCanvasBounds` applies to any node type, while
/// `node_canvas_aabb` deliberately keeps richtext UNWARPED (its mask coverage is
/// the unwarped Degraded raster); (b) OUTSIDE STROKE:
/// `expandBoundsForOutsideStroke` grows the TS box, `node_canvas_aabb` does not.
/// `expand_table_node` sets neither `distort` nor a stroke on a cell's content
/// node, so a flagged node with either is not reachable through the feature. A
/// future flag consumer that CAN carry them has to reconcile the two.
pub fn clip_to_node_box(scene: &Scene, node: &RawNode, clip: [f32; 4]) -> Option<[f32; 4]> {
    if !node.truthy("clipTextToBounds") {
        return Some(clip);
    }
    // A node with no resolvable affine/box cannot narrow anything; keep the
    // incoming clip rather than blanking the text, so a malformed node degrades
    // to the (pre-flag) unclipped draw instead of disappearing.
    let Some((bx, by, bw, bh)) = node_canvas_aabb(scene, node) else {
        return Some(clip);
    };
    // Same `intersect_rect` as `effective_page_clip`, so an empty/degenerate
    // intersection returns None with IDENTICAL semantics to the null-clip path
    // (Q-113): the caller consumes z and draws nothing.
    let r = intersect_rect(clip.map(|v| v as f64), [bx, by, bw, bh])?;
    Some(r.map(|v| v as f32))
}

/// `intersectRect` (`utils/geometryHelpers.ts:386-400`) — degenerate
/// (zero-area) intersections return None.
fn intersect_rect(a: [f64; 4], b: [f64; 4]) -> Option<[f64; 4]> {
    let x = a[0].max(b[0]);
    let y = a[1].max(b[1]);
    let x2 = (a[0] + a[2]).min(b[0] + b[2]);
    let y2 = (a[1] + a[3]).min(b[1] + b[3]);
    if x2 <= x || y2 <= y {
        return None;
    }
    Some([x, y, x2 - x, y2 - y])
}

// ---------------------------------------------------------------------------
// Executor side (wgpu)
// ---------------------------------------------------------------------------

/// WGSL `Globals` byte size (`wgsl/mask-coverage.wgsl`: the mask's placement
/// and source, then the draw's paint, E7.18) and the per-draw slot stride.
const MASK_GLOBALS_BYTE_SIZE: u64 = 352;
const MASK_GLOBALS_ALIGNMENT: u64 = 512;

/// MSAA sample count of the coverage draw (resolved into the r8 texture).
/// 4 is guaranteed for r8unorm render + resolve by WebGPU.
pub const MASK_SAMPLE_COUNT: u32 = 4;

/// Texel budget of one coverage texture (16 Mi texels = 16 MB r8, plus a
/// transient 4× MSAA target while it renders). A mask whose texture at the
/// frame's scale would exceed it renders at a lower power of two.
pub const MAX_MASK_TEXELS: u64 = 1 << 24;

/// Where a coverage texture sits in canvas space and how many texels it has.
/// `origin` / `size` are the mask box snapped out to the texel grid (a texel
/// is `1 / scale` canvas units), so at an integer-aligned export the texels
/// line up with output pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoveragePlacement {
    pub origin: [f64; 2],
    pub size: [f64; 2],
    pub tex: [u32; 2],
    /// Texels per canvas unit.
    pub scale: f64,
}

/// The coverage resolution for a frame whose canvas→pixel scale is
/// `px_scale`: the next power of two at or above it (so a mask re-renders
/// only when zoom crosses a power of two), within [1/16, 64].
/// Side of a shared mask atlas target (texels): a live frame's small masks share one coverage pass per atlas.
const MASK_ATLAS_DIM: u32 = 2048;
/// Masks larger than this on a side get a pass (and target) of their own.
const MASK_ATLAS_MAX_SIDE: u32 = 1024;

thread_local! {
    static MASK_ATLAS: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Shared mask atlas passes for live canvas frames rendered on this thread on (default) or off (every mask renders
/// in a pass of its own; A/Bs and tests, which run in parallel on threads of their own). Exports, thumbnails and
/// readbacks always render each mask on its own, exactly.
pub fn set_mask_atlas_enabled(on: bool) {
    MASK_ATLAS.with(|c| c.set(on));
}

fn mask_atlas_enabled() -> bool {
    MASK_ATLAS.with(std::cell::Cell::get)
}

/// Each mask's nesting level among the masks rendered this frame (`of`, by slot), and the level count: a mask
/// composes its ancestor's coverage, so an ancestor rendered the same frame sits a level below.
struct MaskLevels {
    of: Vec<usize>,
    count: usize,
}

fn mask_levels(masks: &[MaskPlan], to_render: &[usize]) -> MaskLevels {
    let mut slot_of: HashMap<usize, usize> = HashMap::new();
    let mut of = vec![0usize; to_render.len()];
    for (slot, &i) in to_render.iter().enumerate() {
        if let Some(&a_slot) = masks[i].ancestor.and_then(|a| slot_of.get(&a)) {
            of[slot] = of[a_slot] + 1;
        }
        slot_of.insert(i, slot);
    }
    let count = of.iter().max().map_or(0, |m| m + 1);
    MaskLevels { of, count }
}

/// One shared atlas target: its level, used size and the masks (slots) drawn into it.
struct MaskAtlasTarget {
    level: usize,
    size: [u32; 2],
    slots: Vec<usize>,
}

/// Where each mask (slot) sits in an atlas (`None`: rendered on its own), and the atlases.
struct MaskAtlasPacking {
    spot: Vec<Option<[u32; 2]>>,
    atlases: Vec<MaskAtlasTarget>,
}

/// Shelf-pack each level's masks into `dim`-sided atlases (in slot order; a mask over `MASK_ATLAS_MAX_SIDE` on a
/// side renders on its own). Rects never overlap, so each mask's texels are its alone. `on: false` packs nothing.
fn pack_mask_atlases(levels: &MaskLevels, sizes: &[[u32; 2]], dim: u32, on: bool) -> MaskAtlasPacking {
    let mut spot = vec![None; sizes.len()];
    let mut atlases: Vec<MaskAtlasTarget> = Vec::new();
    if !on {
        return MaskAtlasPacking { spot, atlases };
    }
    let max_side = MASK_ATLAS_MAX_SIDE.min(dim);
    for level in 0..levels.count {
        let mut current: Option<usize> = None;
        let (mut x, mut y, mut shelf) = (0u32, 0u32, 0u32);
        for slot in (0..sizes.len()).filter(|&s| levels.of[s] == level) {
            let [w, h] = sizes[slot];
            if w > max_side || h > max_side || w == 0 || h == 0 {
                continue;
            }
            if current.is_some() && x + w > dim {
                x = 0;
                y += shelf;
                shelf = 0;
            }
            if current.is_none() || y + h > dim {
                atlases.push(MaskAtlasTarget { level, size: [0, 0], slots: Vec::new() });
                current = Some(atlases.len() - 1);
                (x, y, shelf) = (0, 0, 0);
            }
            let a = &mut atlases[current.expect("an atlas")];
            spot[slot] = Some([x, y]);
            a.slots.push(slot);
            a.size = [a.size[0].max(x + w), a.size[1].max(y + h)];
            x += w;
            shelf = shelf.max(h);
        }
    }
    // An atlas of one mask saves nothing: that mask renders on its own.
    for a in atlases.iter_mut().filter(|a| a.slots.len() < 2) {
        for &slot in &a.slots {
            spot[slot] = None;
        }
        a.slots.clear();
    }
    atlases.retain(|a| !a.slots.is_empty());
    MaskAtlasPacking { spot, atlases }
}

/// Mask coverage passes per command buffer when a frame renders many masks ([`MaskAtlas`] build).
const MASK_PASSES_PER_SUBMIT: usize = 128;
#[cfg(test)]
thread_local! { static MASK_CHUNK: std::cell::Cell<usize> = const { std::cell::Cell::new(MASK_PASSES_PER_SUBMIT) }; }

#[cfg(test)]
pub(crate) fn with_mask_passes_per_submit<T>(limit: usize, work: impl FnOnce() -> T) -> T {
    struct Restore(usize);
    impl Drop for Restore { fn drop(&mut self) { MASK_CHUNK.set(self.0); } }
    let _restore = Restore(MASK_CHUNK.replace(limit.max(1)));
    work()
}

pub fn coverage_scale(px_scale: f64) -> f64 {
    if !px_scale.is_finite() || px_scale <= 0.0 {
        return 1.0;
    }
    px_scale.log2().ceil().clamp(-4.0, 6.0).exp2()
}

/// The texture placement of a mask box `(origin, size)` at coverage scale
/// `scale`, halving the scale until the texture fits `max_dim` per side and
/// [`MAX_MASK_TEXELS`] overall.
pub fn coverage_placement(origin: [f64; 2], size: [f64; 2], scale: f64, max_dim: u32) -> CoveragePlacement {
    let max_dim = max_dim.max(1);
    let mut s = scale;
    loop {
        let x0 = (origin[0] * s).floor();
        let y0 = (origin[1] * s).floor();
        let x1 = ((origin[0] + size[0]) * s).ceil();
        let y1 = ((origin[1] + size[1]) * s).ceil();
        let w = (x1 - x0).max(1.0);
        let h = (y1 - y0).max(1.0);
        let fits = w <= max_dim as f64 && h <= max_dim as f64 && w * h <= MAX_MASK_TEXELS as f64;
        if (fits || s <= 1.0 / 1024.0) && x0.is_finite() && y0.is_finite() {
            let (tw, th) = (w.min(max_dim as f64) as u32, h.min(max_dim as f64) as u32);
            return CoveragePlacement {
                origin: [x0 / s, y0 / s],
                size: [tw as f64 / s, th as f64 / s],
                tex: [tw, th],
                scale: s,
            };
        }
        if !x0.is_finite() || !y0.is_finite() {
            // Degenerate input: one texel over the plan box.
            return CoveragePlacement {
                origin,
                size,
                tex: [1, 1],
                scale: 1.0,
            };
        }
        s *= 0.5;
    }
}

/// GPU-side mask coverage textures + per-mask consumption bind groups.
pub struct MaskAtlas {
    /// One consumption bind group per `MaskPlan` (engine `mask_bgl` layout:
    /// 0 = linear clamp sampler, 1 = r8unorm view, 2 = MaskUniforms).
    bind_groups: Vec<(wgpu::BindGroup, wgpu::Texture)>,
    // This frame's own resources (not in a [`MaskCache`]), kept alive for the frame.
    _textures: Vec<wgpu::Texture>,
    _buffers: Vec<wgpu::Buffer>,
}

/// Retained mask resources across frames (the editor's camera lane; engine retained
/// mode). A mask's coverage is rendered in canvas space (the texture spans the mask
/// node's canvas AABB at the frame's power-of-two coverage scale, see
/// [`coverage_scale`]), so panning re-renders nothing and zooming re-renders a mask
/// only when the zoom crosses a power of two (the scale is part of the key).
///
/// - The coverage pipeline, samplers and 1×1 fallbacks are created once (a render
///   without a cache creates them per frame).
/// - A geometry mask (mode 0) is keyed by everything its texels depend on: texture
///   placement (scale and size), origin and size, model matrix, coverage vertices, and its ancestor's key.
///   Equal keys render equal textures, so a hit binds the stored texture.
/// - A FULL-RECT mask — an axis-aligned rectangle filling its own AABB, with no
///   ancestor or a full-rect ancestor whose box contains it — covers every texel at
///   1.0, so it needs no texture at all: it binds a shared 1×1 white texture with the
///   same origin / size. Sampling either gives exactly 1.0 inside the box, and the
///   consumer returns 0 outside it for both. This is the common frame clip (a screen,
///   a section, a hero band), and the big one: a 1440×1400 frame is a 2 MB texture.
/// - Text- and image-alpha masks (modes 1, 2) are not cached.
///
/// Entries not used by a frame are evicted, least recently used first, once the
/// cache holds more than [`MASK_CACHE_BUDGET_BYTES`] of textures; a single frame's
/// working set is never evicted during that frame.
pub struct MaskCache {
    pipeline: wgpu::RenderPipeline,
    linear_clamp: wgpu::Sampler,
    white_rgba_view: wgpu::TextureView,
    white_r8_view: wgpu::TextureView,
    _white: [wgpu::Texture; 2],
    entries: HashMap<u64, CachedMask>,
    tex_bytes: u64,
    tick: u64,
    /// The last atlas built for a plan (by identity) at a coverage scale, when every mask came from the cache: a
    /// frame of the same plan at the same scale (a pan) binds it again without keying every mask.
    memo: Option<(std::sync::Weak<crate::seams::CoveragePlan>, u64, Vec<(wgpu::BindGroup, wgpu::Texture)>)>,
    /// Counters for benches: masks served from the cache, rendered, and full-rect.
    pub hits: u64,
    pub renders: u64,
    pub full_rect: u64,
    /// Shared atlas passes (a live frame's small masks, several per pass).
    pub atlas_passes: u64,
}

struct CachedMask {
    /// Placement of the retained consumption binding, independent of local texel identity.
    origin: [f64; 2],
    /// `None` for a full-rect mask (binds the shared white texture).
    _texture: Option<wgpu::Texture>,
    view: wgpu::TextureView,
    _uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    bytes: u64,
    tick: u64,
    /// Rendered in a live frame's shared atlas (or composed from such an ancestor): it may differ from a mask
    /// rendered on its own by a few levels at its edge (the atlas viewport's offset changes the rasterizer's float
    /// rounding), so only live frames bind it; an exact frame (an export, a thumbnail, a readback) renders its own.
    approx: bool,
}

/// Retained mask-texture budget (see [`MaskCache`]).
pub const MASK_CACHE_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

impl MaskCache {
    /// `working`: the owning engine's working colour, which a rendered mask's picture is in.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, working: Option<&moda_shaders::WorkingColor>) -> Result<MaskCache> {
        let pipeline = coverage_pipeline(device, working)?;
        // One linear clamp sampler for the coverage source (textSampler), the
        // ancestor coverage (anti-aliased, possibly at another resolution) and
        // consumption (`bindMaskGroup` fallback/clamp sampler).
        let linear_clamp = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        // 1x1 white fallbacks (rgba8 text/image slot; r8 ancestor slot and full-rect masks).
        let white_rgba = upload_texture(
            device,
            queue,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            1,
            &[255, 255, 255, 255],
            4,
            "mask fallback text texture",
        );
        let white_r8 = upload_texture(
            device,
            queue,
            wgpu::TextureFormat::R8Unorm,
            1,
            1,
            &[255],
            1,
            "mask fallback ancestor texture",
        );
        Ok(MaskCache {
            pipeline,
            linear_clamp,
            white_rgba_view: white_rgba.create_view(&Default::default()),
            white_r8_view: white_r8.create_view(&Default::default()),
            _white: [white_rgba, white_r8],
            entries: HashMap::new(),
            tex_bytes: 0,
            tick: 0,
            memo: None,
            hits: 0,
            renders: 0,
            full_rect: 0,
            atlas_passes: 0,
        })
    }

    /// Resident texture bytes and entries.
    pub fn resident(&self) -> (u64, usize) {
        (self.tex_bytes, self.entries.len())
    }

    fn trim(&mut self) {
        if self.tex_bytes <= MASK_CACHE_BUDGET_BYTES {
            return;
        }
        let mut old: Vec<(u64, u64)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.tick != self.tick)
            .map(|(k, e)| (e.tick, *k))
            .collect();
        old.sort_unstable();
        for (_, k) in old {
            if self.tex_bytes <= MASK_CACHE_BUDGET_BYTES {
                break;
            }
            if let Some(e) = self.entries.remove(&k) {
                self.tex_bytes -= e.bytes;
            }
        }
    }
}

/// Whether `m` is a geometry mask that covers its whole texture: an axis-aligned
/// rectangle (two triangles) exactly spanning its canvas AABB.
fn is_full_rect(m: &MaskPlan) -> bool {
    if m.invert || m.feather > 0.0 { return false; }
    if m.mode != 0 || m.positions.len() != 12 || m.image.is_some() || m.text.is_some() {
        return false;
    }
    // One opaque draw over both triangles (E7.18: an alpha mask's paint may be
    // translucent or a gradient; a luminance mask's depends on the colour).
    let opaque = match m.draws.as_slice() {
        [MaskDraw {
            first: 0,
            count: 6,
            paint: MaskPaint::Solid(c),
        }] => c[3] >= 1.0 && (m.mask_type != MaskType::Luminance || c[..3].iter().all(|v| *v >= 1.0)),
        _ => false,
    };
    if !opaque {
        return false;
    }
    let mm = &m.model_matrix;
    // Column-major 2D affine: a = m[0], b = m[1], c = m[4], d = m[5], e = m[12], f = m[13].
    if mm[1] != 0.0 || mm[4] != 0.0 || mm[0] == 0.0 || mm[5] == 0.0 {
        return false;
    }
    let (a, d, e, f) = (mm[0] as f64, mm[5] as f64, mm[12] as f64, mm[13] as f64);
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    let mut area = 0.0;
    let p: Vec<(f64, f64)> = m
        .positions
        .chunks_exact(2)
        .map(|v| (a * v[0] as f64 + e, d * v[1] as f64 + f))
        .collect();
    for (x, y) in &p {
        min = [min[0].min(*x), min[1].min(*y)];
        max = [max[0].max(*x), max[1].max(*y)];
    }
    for t in p.chunks_exact(3) {
        let (x0, y0) = t[0];
        let (x1, y1) = t[1];
        let (x2, y2) = t[2];
        area += ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs() / 2.0;
    }
    let [ox, oy] = m.origin;
    let [w, h] = m.size;
    let tol = 1e-6 * (1.0 + ox.abs().max(oy.abs()).max(w).max(h));
    (min[0] - ox).abs() <= tol
        && (min[1] - oy).abs() <= tol
        && (max[0] - (ox + w)).abs() <= tol
        && (max[1] - (oy + h)).abs() <= tol
        && (area - w * h).abs() <= tol * (w + h)
}

/// Whether `outer`'s canvas box contains `inner`'s.
fn box_contains(outer: &MaskPlan, inner: &MaskPlan) -> bool {
    outer.origin[0] <= inner.origin[0]
        && outer.origin[1] <= inner.origin[1]
        && outer.origin[0] + outer.size[0] >= inner.origin[0] + inner.size[0]
        && outer.origin[1] + outer.size[1] >= inner.origin[1] + inner.size[1]
}

/// Local coverage has no canvas-space consumers while it is drawn. Only
/// whole-pixel translations and integral coverage origins are normalized.
fn local_coverage(m: &MaskPlan, p: &CoveragePlacement) -> bool {
    m.mode == 0 && m.ancestor.is_none() && !m.invert && m.feather == 0.0
        && crate::layer_blur::translation::pixel_translation(&m.model_matrix)
        && p.scale >= 1.0
        && p.origin.iter().all(|v| v.is_finite() && v.fract() == 0.0 && v.abs() < 8_388_608.0)
        && m.draws.iter().all(|d| matches!(d.paint, MaskPaint::Solid(_)))
}

fn coverage_model(m: &MaskPlan, p: &CoveragePlacement) -> [f32; 16] {
    let mut model = m.model_matrix;
    if local_coverage(m, p) {
        for axis in 0..2 {
            model[12 + axis] = (model[12 + axis] as f64 - p.origin[axis]) as f32;
        }
    }
    model
}

/// Cache key for a geometry mask's texels. Local coverage excludes absolute
/// placement; its consumption uniform is rebound when that placement changes.
/// `ancestor_key` includes ancestor placement (`None` = uncacheable ancestor).
fn content_key(
    m: &MaskPlan,
    full_rect: bool,
    ancestor_key: Option<Option<u64>>,
    placement: &CoveragePlacement,
) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    if m.mode != 0 {
        return None;
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    full_rect.hash(&mut h);
    if m.invert || m.feather > 0.0 {
        m.invert.hash(&mut h);
        m.feather.to_bits().hash(&mut h);
    }
    let local = !full_rect && local_coverage(m, placement);
    local.hash(&mut h);
    for axis in 0..2 {
        let origin = m.origin[axis] - if local { placement.origin[axis] } else { 0.0 };
        origin.to_bits().hash(&mut h);
    }
    m.size[0].to_bits().hash(&mut h);
    m.size[1].to_bits().hash(&mut h);
    if !full_rect {
        match ancestor_key {
            None => 0u8.hash(&mut h),
            Some(Some(k)) => {
                1u8.hash(&mut h);
                k.hash(&mut h);
            }
            Some(None) => return None,
        }
        placement.tex.hash(&mut h);
        placement.scale.to_bits().hash(&mut h);
        if !local {
            placement.origin[0].to_bits().hash(&mut h);
            placement.origin[1].to_bits().hash(&mut h);
        }
        for v in &coverage_model(m, placement) {
            v.to_bits().hash(&mut h);
        }
        m.positions.len().hash(&mut h);
        for v in m.positions.iter() {
            v.to_bits().hash(&mut h);
        }
        // E7.18: the mask type and each draw's range and paint.
        (m.mask_type as u8).hash(&mut h);
        m.draws.len().hash(&mut h);
        for d in m.draws.iter() {
            d.first.hash(&mut h);
            d.count.hash(&mut h);
            hash_paint(&d.paint, &mut h);
        }
    }
    Some(h.finish())
}

fn hash_paint(p: &MaskPaint, h: &mut impl std::hash::Hasher) {
    use std::hash::Hash;
    let floats = |h: &mut dyn std::hash::Hasher, v: &[f32]| {
        for x in v {
            h.write_u32(x.to_bits());
        }
    };
    match p {
        MaskPaint::Solid(c) => {
            0u8.hash(h);
            floats(h, c);
        }
        MaskPaint::Source { opacity } => {
            1u8.hash(h);
            floats(h, &[*opacity]);
        }
        MaskPaint::Gradient {
            fill_type,
            start,
            end,
            rotation,
            num_stops,
            stops,
            bounds,
            opacity,
        } => {
            2u8.hash(h);
            floats(h, &[*fill_type, start[0], start[1], end[0], end[1], *rotation, *opacity]);
            num_stops.hash(h);
            floats(h, bounds);
            for st in stops {
                floats(h, st);
            }
        }
    }
}

impl MaskAtlas {
    pub fn is_empty(&self) -> bool {
        self.bind_groups.is_empty()
    }

    pub fn bind_group(&self, mask_id: u64) -> Option<&wgpu::BindGroup> {
        self.binding(mask_id).map(|(group, _)| group)
    }

    pub(crate) fn binding(&self, mask_id: u64) -> Option<&(wgpu::BindGroup, wgpu::Texture)> {
        self.bind_groups.get(mask_id as usize)
    }

    /// Build all coverage textures and record their render passes into
    /// `encoder` (the frame's command encoder, BEFORE the main pass —
    /// mirrors the production mask pre-pass). `mask_bgl` is the engine's
    /// path-pipeline mask bind group layout (group index 2). `px_scale` is the
    /// frame's canvas→pixel scale; coverage renders at [`coverage_scale`] of it,
    /// 4× MSAA. Nothing is retained: every mask renders into a fresh texture.
    pub fn build(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        mask_bgl: &wgpu::BindGroupLayout,
        masks: &[MaskPlan],
        px_scale: f64,
        surfaces: &SurfaceViews, scratch: &crate::scratch::ScratchPool,
        working: Option<&moda_shaders::WorkingColor>,
    ) -> Result<MaskAtlas> {
        if masks.is_empty() {
            return Ok(MaskAtlas::empty());
        }
        let mut local = MaskCache::new(device, queue, working)?;
        Self::build_inner(device, queue, encoder, mask_bgl, masks, px_scale, &mut local, false, false, surfaces, scratch)
    }

    /// [`MaskAtlas::build`] through a retained [`MaskCache`]: masks whose content is
    /// unchanged since an earlier frame bind their stored texture, full-rect masks
    /// bind no texture, and only new masks render. Pixel-identical to `build`, except that a `live` frame (the canvas
    /// present) packs its small masks into shared atlas passes ([`set_mask_atlas_enabled`]), which may differ by a
    /// few levels at a mask's anti-aliased edge.
    #[allow(clippy::too_many_arguments)]
    pub fn build_cached(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        mask_bgl: &wgpu::BindGroupLayout,
        masks: &[MaskPlan],
        px_scale: f64,
        cache: &mut MaskCache,
        live: bool,
        surfaces: &SurfaceViews, scratch: &crate::scratch::ScratchPool,
    ) -> Result<MaskAtlas> {
        if masks.is_empty() {
            return Ok(MaskAtlas::empty());
        }
        cache.tick += 1;
        let atlas = Self::build_inner(device, queue, encoder, mask_bgl, masks, px_scale, cache, true, live, surfaces, scratch)?;
        cache.trim();
        Ok(atlas)
    }

    /// [`MaskAtlas::build_cached`] for the masks of `plan`, where `identity` is that plan's `Arc` (the camera lane's
    /// prepared-frame identity): a frame of the same plan at the same coverage scale reuses the atlas the last one
    /// built when all its masks were cached (their textures stay alive through the bind groups).
    #[allow(clippy::too_many_arguments)]
    pub fn build_cached_for(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        mask_bgl: &wgpu::BindGroupLayout,
        identity: &std::sync::Arc<crate::seams::CoveragePlan>,
        px_scale: f64,
        cache: &mut MaskCache,
        live: bool,
        surfaces: &SurfaceViews, scratch: &crate::scratch::ScratchPool,
    ) -> Result<MaskAtlas> {
        let masks = &identity.masks;
        if masks.is_empty() {
            return Ok(MaskAtlas::empty());
        }
        if let Some((plan, scale, bind_groups)) = &cache.memo {
            if plan.as_ptr() == std::sync::Arc::as_ptr(identity) && *scale == px_scale.to_bits() && live {
                return Ok(MaskAtlas {
                    bind_groups: bind_groups.clone(),
                    _textures: Vec::new(),
                    _buffers: Vec::new(),
                });
            }
        }
        let atlas = Self::build_cached(device, queue, encoder, mask_bgl, masks, px_scale, cache, live, surfaces, scratch)?;
        cache.memo = (live && atlas._textures.is_empty() && atlas._buffers.is_empty()).then(|| {
            (std::sync::Arc::downgrade(identity), px_scale.to_bits(), atlas.bind_groups.clone())
        });
        Ok(atlas)
    }

    /// Whether every mask texture this atlas binds is retained by the cache (none is this frame's own, freed
    /// with the atlas): a recorded pass may bind it on later frames.
    pub fn is_retained(&self) -> bool {
        self._textures.is_empty() && self._buffers.is_empty()
    }

    fn empty() -> MaskAtlas {
        MaskAtlas {
            bind_groups: Vec::new(),
            _textures: Vec::new(),
            _buffers: Vec::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_inner(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        mask_bgl: &wgpu::BindGroupLayout,
        masks: &[MaskPlan],
        px_scale: f64,
        cache: &mut MaskCache,
        retain: bool,
        live: bool,
        surfaces: &SurfaceViews, scratch: &crate::scratch::ScratchPool,
    ) -> Result<MaskAtlas> {
        let n = masks.len();
        let atlas_on = live && mask_atlas_enabled();
        let scale = coverage_scale(px_scale);
        let max_dim = device.limits().max_texture_dimension_2d;
        // Full-rect masks, texture placements and cache keys, ancestors first (an
        // ancestor always has a smaller index).
        let mut full_rect = vec![false; n];
        let mut keys: Vec<Option<u64>> = vec![None; n];
        let mut placements: Vec<CoveragePlacement> = Vec::with_capacity(n);
        for i in 0..n {
            let m = &masks[i];
            // Full-rect masks bind the shared 1×1 white texture in every lane (an
            // export too): a rendered texture's snapped, anti-aliased border would
            // differ from it, and the retained lane must match a fresh build.
            full_rect[i] = is_full_rect(m)
                && m.ancestor.is_none_or(|a| {
                    a < i && full_rect[a] && masks.get(a).is_some_and(|am| box_contains(am, m))
                });
            // A full-rect mask binds the 1×1 white texture over exactly its box.
            let placement = if full_rect[i] {
                CoveragePlacement {
                    origin: m.origin,
                    size: m.size,
                    tex: [1, 1],
                    scale: 1.0,
                }
            } else {
                coverage_placement(m.origin, m.size, scale, max_dim)
            };
            placements.push(placement);
            // Without retention only full-rect masks are keyed (their white-view
            // entry lives in the frame-local cache); everything else renders.
            if retain || full_rect[i] {
                let anc = m.ancestor.map(|a| {
                    if a >= i { return None; }
                    keys[a].map(|key| {
                        // A parent's local texels may stay the same while its
                        // coverage moves relative to this child.
                        use std::hash::Hasher;
                        let mut h = std::collections::hash_map::DefaultHasher::new();
                        h.write_u64(key);
                        for v in placements[a].origin { h.write_u64(v.to_bits()); }
                        h.finish()
                    })
                });
                keys[i] = content_key(m, full_rect[i], anc, &placement);
            }
        }

        // Resolve each mask: cached, full-rect, or rendered this frame.
        enum Source {
            Cached,
            Render(usize),
        }
        let mut sources: Vec<Source> = Vec::with_capacity(n);
        let mut to_render: Vec<usize> = Vec::new();
        for i in 0..n {
            match keys[i].and_then(|k| cache.entries.get_mut(&k)).filter(|e| atlas_on || !e.approx) {
                Some(e) => {
                    e.tick = cache.tick;
                    cache.hits += 1;
                    sources.push(Source::Cached);
                }
                None if full_rect[i] => {
                    // Inserted below with the white view; no pass.
                    sources.push(Source::Cached);
                }
                None => {
                    sources.push(Source::Render(to_render.len()));
                    to_render.push(i);
                }
            }
        }
        let mut textures: Vec<wgpu::Texture> = Vec::new();
        let mut buffers: Vec<wgpu::Buffer> = Vec::new();

        // Full-rect entries not yet cached: the shared white view.
        for i in 0..n {
            if full_rect[i] {
                let k = keys[i].expect("full-rect masks are keyed");
                if !cache.entries.contains_key(&k) {
                    let (uniforms, bind_group) = consumption_group(
                        device,
                        queue,
                        mask_bgl,
                        cache,
                        &masks[i],
                        &placements[i],
                        &cache.white_r8_view,
                    );
                    cache.full_rect += 1;
                    cache.entries.insert(
                        k,
                        CachedMask {
                            origin: placements[i].origin,
                            _texture: None,
                            view: cache.white_r8_view.clone(),
                            _uniforms: uniforms,
                            bind_group,
                            bytes: 0,
                            tick: cache.tick,
                            approx: false,
                        },
                    );
                }
            }
        }

        // r8 coverage targets for the masks rendered this frame, and the transient
        // multisampled targets their coverage draws resolve from.
        let mut render_views: Vec<wgpu::TextureView> = Vec::with_capacity(to_render.len());
        let mut render_textures: Vec<wgpu::Texture> = Vec::with_capacity(to_render.len());
        let mut msaa_views: Vec<Option<wgpu::TextureView>> = Vec::with_capacity(to_render.len());
        // The multisampled target is scratch (cleared at the start of each coverage pass, resolved at its end and
        // never read after): masks of the same size share one. At fit, a card grid's masks are all one size, and
        // creating a target per mask was a large part of a first frame's mask cost.
        let mut msaa_scratch: HashMap<[u32; 2], wgpu::TextureView> = HashMap::new();
        // A live frame's small masks, by nesting level, packed into shared atlas targets: one pass per atlas
        // instead of one per mask (`pack_mask_atlases`).
        let levels = mask_levels(masks, &to_render);
        let packing = pack_mask_atlases(
            &levels,
            &to_render.iter().map(|&i| if masks[i].invert || masks[i].feather > 0.0 { [0, 0] } else { placements[i].tex }).collect::<Vec<_>>(),
            MASK_ATLAS_DIM.min(max_dim),
            atlas_on,
        );
        // Which rendered masks come out approximate: those in an atlas, and those composing an approximate ancestor
        // (ancestors first: an ancestor has a smaller index).
        let mut approx = vec![false; to_render.len()];
        for (slot, &i) in to_render.iter().enumerate() {
            let ancestor_approx = masks[i].ancestor.is_some_and(|a| match sources[a] {
                Source::Render(r) => approx[r],
                Source::Cached => keys[a].and_then(|k| cache.entries.get(&k)).is_some_and(|e| e.approx),
            });
            approx[slot] = packing.spot[slot].is_some() || ancestor_approx;
        }
        for (slot, &i) in to_render.iter().enumerate() {
            let m = &masks[i];
            let [width, height] = placements[i].tex;
            let in_atlas = packing.spot[slot].is_some();
            let size = wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            };
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("MaskTexture:{}", m.owner_node_id)),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: if in_atlas {
                    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST
                } else {
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
                },
                view_formats: &[],
            });
            render_views.push(tex.create_view(&Default::default()));
            render_textures.push(tex);
            if in_atlas {
                msaa_views.push(None);
                continue;
            }
            if let Some(v) = msaa_scratch.get(&[width, height]) {
                msaa_views.push(Some(v.clone()));
                continue;
            }
            let msaa = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("MaskTextureMsaa:{}", m.owner_node_id)),
                size,
                mip_level_count: 1,
                sample_count: MASK_SAMPLE_COUNT,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = msaa.create_view(&Default::default());
            msaa_scratch.insert([width, height], view.clone());
            msaa_views.push(Some(view));
            textures.push(msaa);
        }
        // The coverage view of mask `i` for this frame (an ancestor's, when composing).
        let view_of = |i: usize, cache: &MaskCache| -> wgpu::TextureView {
            match sources[i] {
                Source::Render(r) => render_views[r].clone(),
                Source::Cached => cache.entries[&keys[i].expect("cached masks are keyed")].view.clone(),
            }
        };

        if !to_render.is_empty() {
            cache.renders += to_render.len() as u64;
            // Coverage-source textures: image (mode 2) or Degraded text raster
            // (mode 1 — binding 1 carries the rich-text standalone texture,
            // renderCoverage :1123-1126).
            let mut image_views: Vec<Option<wgpu::TextureView>> = Vec::with_capacity(to_render.len());
            for &i in &to_render {
                let m = &masks[i];
                // Rendered masks and registered captures supply a GPU view.
                if let (None, Some(MaskImage { surface: Some(src), .. })) = (&m.text, &m.image) {
                    let view = match surfaces.get(src) {
                        Some(v) => v.clone(),
                        None => {
                            let tex = upload_texture(device, queue, wgpu::TextureFormat::Rgba8Unorm, 1, 1, &[0, 0, 0, 0], 4, "mask surface placeholder");
                            let v = tex.create_view(&Default::default());
                            textures.push(tex);
                            v
                        }
                    };
                    image_views.push(Some(view));
                    continue;
                }
                let src: Option<(&[u8], u32, u32, &str)> = match (&m.text, &m.image) {
                    (Some(t), _) => Some((
                        t.raster.rgba.as_slice(),
                        t.raster.width,
                        t.raster.height,
                        "mask text source",
                    )),
                    (None, Some(img)) => Some((
                        img.rgba.as_slice(),
                        img.width,
                        img.height,
                        "mask image source",
                    )),
                    (None, None) => None,
                };
                image_views.push(src.map(|(rgba, w, h, label)| {
                    let tex = upload_texture(
                        device,
                        queue,
                        wgpu::TextureFormat::Rgba8Unorm,
                        w,
                        h,
                        rgba,
                        4,
                        label,
                    );
                    let view = tex.create_view(&Default::default());
                    textures.push(tex);
                    view
                }));
            }

            // Packed globals buffer: one slot per coverage draw of the masks rendered
            // this frame (the mask's placement and source, then the draw's paint).
            let draw_slots: Vec<usize> = to_render
                .iter()
                .scan(0usize, |next, &i| {
                    let first = *next;
                    *next += masks[i].draws.len();
                    Some(first)
                })
                .collect();
            let slot_count = to_render.iter().map(|&i| masks[i].draws.len()).sum::<usize>().max(1);
            let globals = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MaskGlobalsUB (Packed)"),
                size: MASK_GLOBALS_ALIGNMENT * slot_count as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            for (slot, &i) in to_render.iter().enumerate() {
                let m = &masks[i];
                let mut buf = [0u8; MASK_GLOBALS_BYTE_SIZE as usize];
                let f = |b: &mut [u8], off: usize, v: f32| b[off..off + 4].copy_from_slice(&v.to_le_bytes());
                let u = |b: &mut [u8], off: usize, v: u32| b[off..off + 4].copy_from_slice(&v.to_le_bytes());
                for (j, v) in coverage_model(m, &placements[i]).iter().enumerate() {
                    f(&mut buf, j * 4, *v);
                }
                let pl = &placements[i];
                let local = local_coverage(m, pl);
                f(&mut buf, 64, if local { 0.0 } else { pl.origin[0] as f32 });
                f(&mut buf, 68, if local { 0.0 } else { pl.origin[1] as f32 });
                f(&mut buf, 72, inv_or_zero(pl.size[0]));
                f(&mut buf, 76, inv_or_zero(pl.size[1]));
                u(&mut buf, 80, m.mode);
                u(&mut buf, 84, (m.ancestor.is_some() && !m.invert && m.feather == 0.0) as u32);
                // M4d: textOrigin @88, textInvSize @96 (renderCoverage
                // :1054-1070) — world-UV against the standalone texture's
                // logical size.
                if let Some(t) = &m.text {
                    f(&mut buf, 88, t.origin_canvas[0] as f32);
                    f(&mut buf, 92, t.origin_canvas[1] as f32);
                    f(&mut buf, 96, inv_or_zero(t.raster.logical_w));
                    f(&mut buf, 100, inv_or_zero(t.raster.logical_h));
                }
                if let Some(anc) = m.ancestor.and_then(|a| placements.get(a)) {
                    f(&mut buf, 104, anc.origin[0] as f32);
                    f(&mut buf, 108, anc.origin[1] as f32);
                    f(&mut buf, 112, inv_or_zero(anc.size[0]));
                    f(&mut buf, 116, inv_or_zero(anc.size[1]));
                }
                // Image placement @120..148; identity defaults when not image.
                let (off, dims, scale, rot) = match &m.image {
                    Some(img) => (img.offset, img.dims, img.scale, img.rotation_rad),
                    None => ([0.0, 0.0], [1.0, 1.0], [1.0, 1.0], 0.0),
                };
                f(&mut buf, 120, off[0]);
                f(&mut buf, 124, off[1]);
                f(&mut buf, 128, dims[0]);
                f(&mut buf, 132, dims[1]);
                f(&mut buf, 136, scale[0]);
                f(&mut buf, 140, scale[1]);
                f(&mut buf, 144, rot);
                u(&mut buf, 148, (m.mask_type == MaskType::Luminance) as u32);
                for (d, draw) in m.draws.iter().enumerate() {
                    let mut buf = buf;
                    write_draw_paint(&mut buf, &draw.paint, m.mode);
                    crate::gpu_profile::count_buffer_write(buf.len());
                    crate::scratch::write_buffer(device, queue, &globals, MASK_GLOBALS_ALIGNMENT * (draw_slots[slot] + d) as u64, &buf);
                }
            }

            // Vertex buffers.
            // One vertex buffer for every mask rendered this frame, each at its own range.
            let mut vranges: Vec<(u64, u64)> = Vec::with_capacity(to_render.len());
            let mut all_positions: Vec<f32> = Vec::new();
            for &i in &to_render {
                let m = &masks[i];
                let start = (all_positions.len() * 4) as u64;
                all_positions.extend_from_slice(&m.positions);
                vranges.push((start, (m.positions.len() * 4) as u64));
            }
            let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MaskGeomVB"),
                size: ((all_positions.len() * 4).max(4) as u64).div_ceil(4) * 4,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if !all_positions.is_empty() {
                crate::gpu_profile::count_buffer_write(all_positions.len() * 4);
                crate::scratch::write_buffer(device, queue, &vbuf, 0, bytemuck::cast_slice(&all_positions));
            }
            drop(all_positions);

            // Coverage passes, ancestors-first (plan guarantees ancestor index <
            // descendant index; a cached or full-rect ancestor's view is ready).
            let globals_bgl = cache.pipeline.get_bind_group_layout(0);
            #[cfg(test)]
            let per_submit = MASK_CHUNK.get();
            #[cfg(not(test))]
            let per_submit = MASK_PASSES_PER_SUBMIT;
            // The passes in level order (a mask's ancestors render first): each level's atlases, then its masks
            // that render on their own.
            enum CoveragePass {
                Atlas(usize),
                Alone(usize),
            }
            let mut work: Vec<CoveragePass> = Vec::new();
            for level in 0..levels.count {
                work.extend(
                    (0..packing.atlases.len())
                        .filter(|&k| packing.atlases[k].level == level)
                        .map(CoveragePass::Atlas),
                );
                work.extend(
                    (0..to_render.len())
                        .filter(|&s| levels.of[s] == level && packing.spot[s].is_none())
                        .map(CoveragePass::Alone),
                );
            }
            // The bind groups mask `slot` draws with: one globals group a draw, its coverage source (masks without
            // a source image share one) and its ancestor (masks under the same ancestor, or none, share one).
            let slot_binds = |slot: usize,
                              shared_bg1: &mut Option<wgpu::BindGroup>,
                              shared_bg2: &mut HashMap<Option<usize>, wgpu::BindGroup>|
             -> (Vec<wgpu::BindGroup>, wgpu::BindGroup, wgpu::BindGroup) {
                let m = &masks[to_render[slot]];
                let bg0: Vec<wgpu::BindGroup> = (0..m.draws.len())
                    .map(|d| {
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("mask globals bg"),
                            layout: &globals_bgl,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &globals,
                                    offset: MASK_GLOBALS_ALIGNMENT * (draw_slots[slot] + d) as u64,
                                    size: wgpu::BufferSize::new(MASK_GLOBALS_BYTE_SIZE),
                                }),
                            }],
                        })
                    })
                    .collect();
                let make_bg1 = |coverage_view: &wgpu::TextureView| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("mask coverage-source bg"),
                        layout: &cache.pipeline.get_bind_group_layout(1),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Sampler(&cache.linear_clamp),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(coverage_view),
                            },
                        ],
                    })
                };
                let bg1 = match image_views[slot].as_ref() {
                    Some(v) => make_bg1(v),
                    None => shared_bg1.get_or_insert_with(|| make_bg1(&cache.white_rgba_view)).clone(),
                };
                let bg2 = shared_bg2
                    .entry(m.ancestor)
                    .or_insert_with(|| {
                        let ancestor_view = m
                            .ancestor
                            .map(|a| view_of(a, cache))
                            .unwrap_or_else(|| cache.white_r8_view.clone());
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("mask ancestor bg"),
                            layout: &cache.pipeline.get_bind_group_layout(2),
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    // Linear: the ancestor's anti-aliased edge composes smoothly.
                                    resource: wgpu::BindingResource::Sampler(&cache.linear_clamp),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(&ancestor_view),
                                },
                            ],
                        })
                    })
                    .clone();
                (bg0, bg1, bg2)
            };
            let draw_mask = |pass: &mut wgpu::RenderPass<'_>,
                             slot: usize,
                             (bg0, bg1, bg2): &(Vec<wgpu::BindGroup>, wgpu::BindGroup, wgpu::BindGroup)| {
                let m = &masks[to_render[slot]];
                pass.set_pipeline(&cache.pipeline);
                pass.set_bind_group(1, bg1, &[]);
                pass.set_bind_group(2, bg2, &[]);
                let (vstart, vlen) = vranges[slot];
                if vlen > 0 {
                    pass.set_vertex_buffer(0, vbuf.slice(vstart..vstart + vlen));
                }
                let vertices = (m.positions.len() / 2) as u32;
                for (draw, bg0) in m.draws.iter().zip(bg0) {
                    let end = (draw.first + draw.count).min(vertices);
                    if draw.first >= end {
                        continue;
                    }
                    pass.set_bind_group(0, bg0, &[]);
                    pass.draw(draw.first..end, 0..1);
                }
            };
            let clear = wgpu::Operations {
                // clearValue {r:0, g:0, b:0, a:1} (renderCoverage).
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                }),
                // Only the resolved coverage is read.
                store: wgpu::StoreOp::Discard,
            };
            // Roll over the caller's active encoder. The first chunk includes any mask
            // source draws, and the page continues in the encoder holding the last chunk.
            let mut shared_bg1: Option<wgpu::BindGroup> = None;
            let mut shared_bg2: HashMap<Option<usize>, wgpu::BindGroup> = HashMap::new();
            let mut atlas_msaa: HashMap<[u32; 2], wgpu::TextureView> = HashMap::new();
            let mut atlas_passes = 0u64;
            for (p, item) in work.iter().enumerate() {
                if p > 0 && p % per_submit == 0 {
                    let next = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("mask coverage chunk"),
                    });
                    let done = std::mem::replace(encoder, next);
                    crate::scratch::submit(queue, [done.finish()]);
                }
                if p % per_submit == 0 {
                    let inputs = work[p..work.len().min(p + per_submit)].iter().flat_map(|pass| {
                        match pass {
                            CoveragePass::Alone(slot) => std::slice::from_ref(slot),
                            CoveragePass::Atlas(index) => &packing.atlases[*index].slots[..],
                        }
                    }).flat_map(|&slot| {
                        let image = image_views[slot].as_ref().map(|v| v.texture().clone());
                        let ancestor = masks[to_render[slot]].ancestor.map(|a| view_of(a, cache).texture().clone());
                        [image, ancestor].into_iter().flatten()
                    }).collect();
                    scratch.record_storage(encoder, queue, 0, inputs);
                }
                match *item {
                    CoveragePass::Alone(slot) => {
                        let binds = slot_binds(slot, &mut shared_bg1, &mut shared_bg2);
                        let profile_ts = crate::gpu_profile::pass_timestamps();
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some(&format!("MaskCoveragePass:{}", masks[to_render[slot]].owner_node_id)),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: msaa_views[slot].as_ref().expect("a mask rendered on its own has a target"),
                                resolve_target: Some(&render_views[slot]),
                                ops: clear,
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: crate::gpu_profile::render_writes(&profile_ts),
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        draw_mask(&mut pass, slot, &binds);
                        drop(pass);
                        let index = to_render[slot];
                        let mask = &masks[index];
                        if mask.invert || mask.feather > 0.0 {
                            let ancestor_view = mask.ancestor.map(|ancestor| view_of(ancestor, cache));
                            super::mask_filter::apply(device, encoder, &render_views[slot], &placements[index], mask,
                                ancestor_view.as_ref(), mask.ancestor.map(|ancestor| &placements[ancestor]));
                        }
                    }
                    CoveragePass::Atlas(k) => {
                        // The atlas's masks in one pass, each through a viewport and scissor over its own rect (on
                        // whole texels, cleared as a target of its own is), then copied out to its own texture.
                        let atlas = &packing.atlases[k];
                        let size = wgpu::Extent3d {
                            width: atlas.size[0],
                            height: atlas.size[1],
                            depth_or_array_layers: 1,
                        };
                        let resolve = device.create_texture(&wgpu::TextureDescriptor {
                            label: Some("MaskAtlas"),
                            size,
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::R8Unorm,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                            view_formats: &[],
                        });
                        let resolve_view = resolve.create_view(&Default::default());
                        let msaa_view = match atlas_msaa.get(&atlas.size) {
                            Some(v) => v.clone(),
                            None => {
                                let msaa = device.create_texture(&wgpu::TextureDescriptor {
                                    label: Some("MaskAtlasMsaa"),
                                    size,
                                    mip_level_count: 1,
                                    sample_count: MASK_SAMPLE_COUNT,
                                    dimension: wgpu::TextureDimension::D2,
                                    format: wgpu::TextureFormat::R8Unorm,
                                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                                    view_formats: &[],
                                });
                                let v = msaa.create_view(&Default::default());
                                atlas_msaa.insert(atlas.size, v.clone());
                                textures.push(msaa);
                                v
                            }
                        };
                        let binds: Vec<_> = atlas
                            .slots
                            .iter()
                            .map(|&slot| slot_binds(slot, &mut shared_bg1, &mut shared_bg2))
                            .collect();
                        {
                            let profile_ts = crate::gpu_profile::pass_timestamps();
                            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("MaskCoverageAtlasPass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &msaa_view,
                                    resolve_target: Some(&resolve_view),
                                    ops: clear,
                                    depth_slice: None,
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: crate::gpu_profile::render_writes(&profile_ts),
                                occlusion_query_set: None,
                                multiview_mask: None,
                            });
                            for (&slot, binds) in atlas.slots.iter().zip(&binds) {
                                let [x, y] = packing.spot[slot].expect("an atlas slot");
                                let [w, h] = placements[to_render[slot]].tex;
                                pass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
                                pass.set_scissor_rect(x, y, w, h);
                                draw_mask(&mut pass, slot, binds);
                            }
                        }
                        for &slot in &atlas.slots {
                            let [x, y] = packing.spot[slot].expect("an atlas slot");
                            let [w, h] = placements[to_render[slot]].tex;
                            encoder.copy_texture_to_texture(
                                wgpu::TexelCopyTextureInfo {
                                    texture: &resolve,
                                    mip_level: 0,
                                    origin: wgpu::Origin3d { x, y, z: 0 },
                                    aspect: wgpu::TextureAspect::All,
                                },
                                wgpu::TexelCopyTextureInfo {
                                    texture: &render_textures[slot],
                                    mip_level: 0,
                                    origin: wgpu::Origin3d::ZERO,
                                    aspect: wgpu::TextureAspect::All,
                                },
                                wgpu::Extent3d {
                                    width: w,
                                    height: h,
                                    depth_or_array_layers: 1,
                                },
                            );
                        }
                        textures.push(resolve);
                        atlas_passes += 1;
                    }
                }
            }
            cache.atlas_passes += atlas_passes;
            buffers.push(globals);
            buffers.push(vbuf);
        }

        // Consumption bind groups (engine mask_bgl). A rendered mask with a key moves
        // into the cache with its bind group; the rest are this frame's own.
        let mut bind_groups = Vec::with_capacity(n);
        let mut rendered = render_textures.into_iter().map(Some).collect::<Vec<_>>();
        for i in 0..n {
            match sources[i] {
                Source::Cached => {
                    let e = &cache.entries[&keys[i].expect("cached masks are keyed")];
                    if !full_rect[i] && local_coverage(&masks[i], &placements[i]) && e.origin != placements[i].origin {
                        let (uniforms, group) = consumption_group(device, queue, mask_bgl, cache,
                            &masks[i], &placements[i], &e.view);
                        buffers.push(uniforms);
                        bind_groups.push((group, e.view.texture().clone()));
                    } else {
                        bind_groups.push((e.bind_group.clone(), e.view.texture().clone()));
                    }
                }
                Source::Render(r) => {
                    let view = render_views[r].clone();
                    let (uniforms, bind_group) = consumption_group(
                        device,
                        queue,
                        mask_bgl,
                        cache,
                        &masks[i],
                        &placements[i],
                        &view,
                    );
                    bind_groups.push((bind_group.clone(), view.texture().clone()));
                    let texture = rendered[r].take().expect("each render slot once");
                    match keys[i] {
                        Some(k) => {
                            let [w, h] = placements[i].tex;
                            let bytes = u64::from(w) * u64::from(h);
                            cache.tex_bytes += bytes;
                            if let Some(old) = cache.entries.insert(
                                k,
                                CachedMask {
                                    origin: placements[i].origin,
                                    _texture: Some(texture),
                                    view,
                                    _uniforms: uniforms,
                                    bind_group,
                                    bytes,
                                    tick: cache.tick,
                                    approx: approx[r],
                                },
                            ) {
                                cache.tex_bytes -= old.bytes;
                            }
                        }
                        None => {
                            textures.push(texture);
                            buffers.push(uniforms);
                        }
                    }
                }
            }
        }

        Ok(MaskAtlas {
            bind_groups,
            _textures: textures,
            _buffers: buffers,
        })
    }
}

/// The consumption uniform + bind group for mask `m` sampled through `view`, placed
/// at `pl`: MaskUniforms { maskActive: u32 @0, maskOrigin: vec2f @8, maskInvSize:
/// vec2f @16 } — 24 bytes used, 32 allocated (`bindMaskGroup`).
fn consumption_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mask_bgl: &wgpu::BindGroupLayout,
    cache: &MaskCache,
    m: &MaskPlan,
    pl: &CoveragePlacement,
    view: &wgpu::TextureView,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PathMaskUniform"),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut buf = [0u8; 32];
    buf[0..4].copy_from_slice(&1u32.to_le_bytes());
    buf[8..12].copy_from_slice(&(pl.origin[0] as f32).to_le_bytes());
    buf[12..16].copy_from_slice(&(pl.origin[1] as f32).to_le_bytes());
    buf[16..20].copy_from_slice(&inv_or_zero(pl.size[0]).to_le_bytes());
    buf[20..24].copy_from_slice(&inv_or_zero(pl.size[1]).to_le_bytes());
    crate::gpu_profile::count_buffer_write(buf.len());
    crate::scratch::write_buffer(device, queue, &uniforms, 0, &buf);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("mask bg:{}", m.owner_node_id)),
        layout: mask_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(&cache.linear_clamp),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniforms,
                    offset: 0,
                    size: wgpu::BufferSize::new(32),
                }),
            },
        ],
    });
    (uniforms, bind_group)
}

impl Drop for MaskAtlas {
    fn drop(&mut self) {
        self.bind_groups.clear();
        for texture in self._textures.drain(..) {
            crate::scratch::destroy(&texture);
        }
        for buffer in self._buffers.drain(..) {
            crate::scratch::destroy(&buffer);
        }
    }
}

/// A coverage draw's paint into its globals slot (offsets per
/// `wgsl/mask-coverage.wgsl`): source @80 (0 paint, 1 text, 2 image),
/// color @160, gradType @176, gradRotation @180, gradStart @184,
/// gradEnd @192, numStops @200, gradBounds @208, stops @224.
fn write_draw_paint(buf: &mut [u8], paint: &MaskPaint, mode: u32) {
    let f = |b: &mut [u8], off: usize, v: f32| b[off..off + 4].copy_from_slice(&v.to_le_bytes());
    let u = |b: &mut [u8], off: usize, v: u32| b[off..off + 4].copy_from_slice(&v.to_le_bytes());
    let color = |b: &mut [u8], c: [f32; 4]| {
        for (k, v) in c.iter().enumerate() {
            f(b, 160 + k * 4, *v);
        }
    };
    match paint {
        MaskPaint::Solid(c) => {
            u(buf, 80, 0);
            color(buf, *c);
        }
        MaskPaint::Source { opacity } => {
            u(buf, 80, mode);
            color(buf, [1.0, 1.0, 1.0, *opacity]);
        }
        MaskPaint::Gradient {
            fill_type,
            start,
            end,
            rotation,
            num_stops,
            stops,
            bounds,
            opacity,
        } => {
            u(buf, 80, 0);
            color(buf, [1.0, 1.0, 1.0, *opacity]);
            f(buf, 176, *fill_type);
            f(buf, 180, *rotation);
            f(buf, 184, start[0]);
            f(buf, 188, start[1]);
            f(buf, 192, end[0]);
            f(buf, 196, end[1]);
            u(buf, 200, *num_stops);
            for (k, v) in bounds.iter().enumerate() {
                f(buf, 208 + k * 4, *v);
            }
            for (j, stop) in stops.iter().enumerate() {
                for (k, v) in stop.iter().enumerate() {
                    f(buf, 224 + j * 16 + k * 4, *v);
                }
            }
        }
    }
}

fn inv_or_zero(v: f64) -> f32 {
    if v > 0.0 {
        (1.0 / v) as f32
    } else {
        0.0
    }
}

/// The unified coverage pipeline from the EXTRACTED `mask-coverage` module
/// (manifest descriptor: r8unorm target, no blend/depth, float32x2 stride-8
/// vertex stream, entry points vs_main/fs_main).
fn coverage_pipeline(device: &wgpu::Device, working: Option<&moda_shaders::WorkingColor>) -> Result<wgpu::RenderPipeline> {
    let src = moda_shaders::module_source("mask-coverage")
        .ok_or_else(|| anyhow!("no extracted WGSL module 'mask-coverage'"))?;
    let manifest = moda_shaders::manifest();
    let meta = manifest
        .modules
        .get("mask-coverage")
        .ok_or_else(|| anyhow!("mask-coverage missing from manifest"))?;
    let desc = meta
        .pipelines
        .first()
        .ok_or_else(|| anyhow!("mask-coverage has no pipeline descriptor"))?;
    let vs_entry = desc
        .vertex
        .as_ref()
        .map(|v| v.entry_point.as_str())
        .unwrap_or("vs_main");
    let fragment = desc
        .fragment
        .as_ref()
        .ok_or_else(|| anyhow!("mask-coverage lacks fragment state"))?;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mask-coverage"),
        source: wgpu::ShaderSource::Wgsl(moda_shaders::working_color::specialize(working, src.source.into())),
    });

    let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mask globals bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let tex_bgl = |label: &str| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        })
    };
    let bgl1 = tex_bgl("mask coverage-source bgl");
    let bgl2 = tex_bgl("mask ancestor bgl");
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mask coverage layout"),
        bind_group_layouts: &[Some(&bgl0), Some(&bgl1), Some(&bgl2)],
        ..Default::default()
    });

    let target_format = fragment.targets[0]
        .as_ref()
        .map(|t| t.format.as_str())
        .unwrap_or("r8unorm");
    anyhow::ensure!(
        target_format == "r8unorm",
        "mask-coverage target format changed in manifest: {target_format}"
    );

    // M5b (docs/productionization.md): env-gated pipeline-compile stats.
    if std::env::var_os("MODA_PIPELINE_STATS").is_some() {
        eprintln!("moda-pipeline-stat: mask-coverage desc=0 msaa={MASK_SAMPLE_COUNT} blend=-");
    }
    Ok(
        moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
            label: Some("Mask Coverage Pipeline (Unified with Ancestor Compositing)"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some(vs_entry),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(&fragment.entry_point),
                compilation_options: Default::default(),
                // E7.18: a mask's paint layers composite source-over: the
                // shader writes (coverage, 0, 0, layer alpha), so
                // `dst = coverage + dst × (1 − alpha)`.
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            // Anti-aliased coverage: 4× MSAA, resolved into the r8 texture.
            multisample: wgpu::MultisampleState {
                count: MASK_SAMPLE_COUNT,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        }),
    )
}

#[allow(clippy::too_many_arguments)]
fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    data: &[u8],
    bytes_per_pixel: u32,
    label: &str,
) -> wgpu::Texture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    crate::gpu_profile::count_texture_write(data.len());
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * bytes_per_pixel),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    tex
}

#[cfg(test)]
mod tests {
    fn full_document(scene: &Scene) -> MaskPageScope {
        let mut membership = MaskMembership::of(scene);
        // An unrelated ambiguous edge forces the complete capture representation.
        membership.ambiguous.insert("outside-the-occurrence".into());
        MaskPageScope { membership: Some(std::sync::Arc::new(membership)) }
    }

    #[test]
    fn isolated_source_rows_match_the_full_camera_on_edits() {
        use moda_scene::value::json;
        let path = |id, x, color| json!({"id":id,"type":"path","x":x,"width":24,"height":16,
            "points":[{"type":"move","x":0,"y":0},{"type":"line","x":24,"y":0},{"type":"line","x":0,"y":16}],
            "closed":true,"fills":[{"id":"f","paint":{"type":"solid","color":color}}]});
        let mut scene = scene(json!({"nodes":[
            {"id":"canvas","type":"group","children":["outside","owner"],"x":5,"rotation":7},
            path("outside", 142, "#ff00ff"),
            {"id":"owner","type":"group","children":["source","body"],"clipMaskNodeId":"source","opacity":0.6},
            {"id":"source","type":"group","children":["a","b"],"maskType":"luminance","opacity":0.8},
            path("a", 2, "#88ff88"), path("b", 16, "#221188"), path("body", 6, "#ffff00")
        ],"pages":[{"id":"p","width":240,"height":180,"nodeIds":["canvas"]}]}));
        let mut scope = MaskPageScope::default();
        let capture = |scene: &Scene, scope: MaskPageScope| {
            let mut collector = PlanMasks { scope, ..Default::default() };
            collector.build_rendered_mask(scene, "owner", "source", MaskType::Luminance, None, false).unwrap();
            (collector.masks.pop().unwrap(), collector.scope)
        };
        for x in [16, -9, 28] {
            scene.write_row(scene.row_of("b").unwrap(), |f| f.set("x", json!(x))).unwrap();
            scene.refresh_worlds();
            let (short, next) = capture(&scene, scope);
            scope = next;
            let (full, _) = capture(&scene, full_document(&scene));
            let (a, b) = (short.image.as_ref().unwrap().rendered.as_ref().unwrap(),
                full.image.as_ref().unwrap().rendered.as_ref().unwrap());
            assert!(a.cache_key.is_some(), "the authored viewBox group owns its children");
            assert_eq!(a.document.nodes.len(), 5);
            assert_eq!(b.document.nodes.len(), 7);
            assert_eq!((short.origin, short.size), (full.origin, full.size));
            for scale in [0.5, 2.0] {
                let scoped = a.prepare(&short, scale, None, 0.0, false).unwrap();
                let reference = b.prepare(&full, scale, None, 0.0, false).unwrap();
                assert_eq!(format!("{scoped:?}"), format!("{reference:?}"));
            }
        }
        // An external mask reference suppresses the same node in the ordinary
        // display list, even when the referring group is hidden.
        let mut doc = scene.doc().clone();
        doc.nodes.push(moda_scene::types::RawNode::from_value(json!({
            "id":"external","type":"group","hidden":true,"children":[],"clipMaskNodeId":"a"
        })).unwrap());
        let external = Scene::from_document(doc).unwrap();
        let (mask, _) = capture(&external, MaskPageScope::default());
        let source = mask.image.unwrap().rendered.unwrap();
        assert_eq!(source.document.nodes.len(), 8);
        let (old, _) = capture(&scene, MaskPageScope::default());
        let former = old.image.unwrap().rendered.unwrap();
        assert_eq!(former.source_fingerprint, source.source_fingerprint);
        assert_ne!(former.cache_key, source.cache_key, "external clip binding must invalidate the capture");
    }

    #[test]
    fn isolated_path_keeps_the_sample_count_of_an_unrelated_backdrop() {
        use moda_scene::value::json;
        let shape = |id| json!({"id":id,"type":"path","width":24,"height":24,
            "points":[{"type":"move","x":0,"y":0},{"type":"line","x":24,"y":5},{"type":"line","x":0,"y":16}],
            "closed":true,"fills":[{"id":"f","paint":{"type":"solid","color":"#ffffff"}}]});
        let mut matte = shape("matte");
        matte["maskType"] = json!("luminance");
        let scene = scene(json!({"nodes":[
            {"id":"clip","type":"group","children":["matte","body"],"clipMaskNodeId":"matte"},matte,shape("body"),
            {"id":"other","type":"rectangle","x":48,"width":20,"height":20,
             "effects":[{"id":"blur","type":"background-blur","radius":4}]}
        ],"pages":[{"id":"p","width":90,"height":50,"nodeIds":["clip","other"]}]}));
        let plan = |scope| {
            let mut masks = PlanMasks { scope, ..Default::default() };
            masks.build_rendered_mask(&scene,"clip","matte",MaskType::Luminance,None,false).unwrap();
            let mask = &masks.masks[0];
            let source = mask.image.as_ref().unwrap().rendered.as_ref().unwrap();
            source.prepare(mask, 2., None, 0., false).unwrap()
        };
        let cold = plan(full_document(&scene));
        let scoped = plan(MaskPageScope::default());
        assert_eq!(cold.sample_count, 1);
        assert_eq!(format!("{cold:?}"), format!("{scoped:?}"));
    }

    #[test]
    fn isolated_shape_group_preserves_page_and_canvas_camera_inputs() {
        use moda_scene::value::json;
        for canvas in [false, true] {
            let roots = json!(["owner", "glass", "unrelated"]);
            let mut value = json!({
                "layoutMode":"freeform",
                "nodes":[
                    {"id":"owner","type":"group","children":["mask","body"],"clipMaskNodeId":"mask"},
                    {"id":"mask","type":"container","width":48,"height":36,"children":["ink"],"maskType":"luminance"},
                    {"id":"ink","type":"ellipse","x":5,"y":4,"width":24,"height":16,
                        "fills":[{"id":"f","paint":{"type":"solid","color":"#e8fef0"}}]},
                    {"id":"body","type":"rectangle","width":48,"height":36},
                    {"id":"glass","type":"rectangle","x":160,"width":12,"height":12,
                        "effects":[{"type":"background-blur","radius":3}]},
                    {"id":"unrelated","type":"rectangle","width":24,"height":24}
                ],
                "pages":[{"id":"first","width":36,"height":18,"nodeIds":[]},
                    {"id":"second","width":90,"height":50,"freeformX":12,"freeformY":32,"nodeIds":[]}],
                "canvasRootNodeIds":[]
            });
            if canvas { value["canvasRootNodeIds"] = roots; } else { value["pages"][1]["nodeIds"] = roots; }
            let capture = |scene: &Scene, scope| {
                let mut masks = PlanMasks { scope, ..Default::default() };
                masks.build_rendered_mask(scene, "owner", "mask", MaskType::Luminance, None, false).unwrap();
                masks.masks.pop().unwrap()
            };
            let mut keys = Vec::new();
            let mut scope = MaskPageScope::default();
            for radius in [3, 8, 0] {
                value["nodes"][4]["effects"][0]["radius"] = json!(radius);
                let scene = Scene::load(&value).unwrap();
                if !keys.is_empty() {
                    assert_eq!(scope.raster_stands(&scene, scene.node("glass").unwrap()), radius > 0);
                }
                scope.membership(&scene);
                let short = capture(&scene, MaskPageScope::default());
                let full = capture(&scene, full_document(&scene));
                let picture = |mask: &MaskPlan| mask.image.as_ref().unwrap().rendered.as_ref().unwrap()
                    .prepare(mask, 2., None, 0., false).unwrap();
                let source = short.image.as_ref().unwrap().rendered.as_ref().unwrap();
                assert!(source.document.nodes.iter().all(|row| row.id() != "unrelated"));
                assert_eq!((short.origin, short.size), (full.origin, full.size));
                let reference = picture(&full);
                assert_eq!(reference.sample_count == 1, radius > 0);
                assert_eq!(format!("{reference:?}"), format!("{:?}", picture(&short)));
                keys.push(source.cache_key.unwrap());
            }
            assert_eq!(keys[0], keys[1], "outside blur radius preserves raster mode");
            assert_ne!(keys[1], keys[2], "removing the last blur changes coverage");
        }
    }

    #[test]
    fn rendered_text_capture_key_tracks_its_sources_not_unrelated_glyphs() {
        let document = moda_scene::value::json!({
            "nodes": [
                {"id":"parent", "type":"group", "children":["mask", "other"]},
                {"id":"mask", "type":"group", "children":["text"]},
                {"id":"text", "type":"richtext", "vorkText":[0,0,1], "width":100, "height":30},
                {"id":"other", "type":"richtext", "vorkText":[1,0,1], "width":100, "height":30}
            ],
            "pages":[{"id":"page", "width":300, "height":200, "nodeIds":["parent"]}],
            "config":{}, "canvasRootNodeIds":[], "layoutMode":"vertical"
        });
        let capture = |value: &moda_scene::value::Value| {
            let scene = Scene::load(value).unwrap();
            let mut masks = PlanMasks { scope: MaskPageScope::default(), ..Default::default() };
            masks.build_rendered_mask(&scene, "parent", "mask", MaskType::Luminance, None, false).unwrap();
            masks.masks.pop().unwrap().image.unwrap().rendered.unwrap()
        };
        let key = |value: &_| capture(value).cache_key;
        assert!(key(&document).is_some(), "standalone native text reads its own occurrence");
        crate::text_draw::install_text_builder(Some(std::rc::Rc::new(|_, _, _, _, _|
            Ok(crate::text_draw::TextBuildOutcome::Empty))));
        let original_capture = capture(&document);
        assert!(original_capture.cache_key.is_some());
        assert!(original_capture.document.nodes.iter().all(|node| node.id() != "other"));
        let original = key(&document);
        assert!(original.is_some());
        let mut changed = document.clone();
        changed["nodes"][3]["vorkText"][2] = moda_scene::value::json!(2);
        assert_eq!(key(&changed), original);
        assert_eq!(capture(&changed).cache_key, original_capture.cache_key);
        for (node, field, value) in [
            (2, "vorkText", moda_scene::value::json!([0,0,2])),
            (0, "x", moda_scene::value::json!(10)),
            (1, "opacity", moda_scene::value::json!(0.5)),
            (2, "visible", moda_scene::value::json!(false)),
        ] {
            let mut changed = document.clone();
            changed["nodes"][node][field] = value;
            assert_ne!(key(&changed), original, "{field}");
            assert_ne!(capture(&changed).cache_key, original_capture.cache_key, "{field}");
        }
        let mut page = document.clone();
        page["pages"][0]["width"] = moda_scene::value::json!(400);
        assert_ne!(key(&page), original, "page clipping/layout participates");
        let mut outside = document.clone();
        outside["nodes"][1]["clipMaskNodeId"] = moda_scene::value::json!("other");
        assert!(key(&outside).is_none(), "external mask dependencies are not eligible");
        let mut native_text = document;
        native_text["nodes"][2].as_object_mut().unwrap().remove("vorkText");
        crate::text_draw::install_text_builder(None);
        assert!(key(&native_text).is_some(), "standalone box text");
        native_text["nodes"][3]["wrapsText"] = moda_scene::value::json!({});
        assert!(key(&native_text).is_none(), "a wrap object can change native layout");
    }

    use super::*;

    fn scene(json: moda_scene::value::Value) -> Scene {
        Scene::load(&json).unwrap()
    }

    #[test]
    fn compound_stroke_masks_match_visible_closure_dash_and_trim() {
        use crate::seams::DrawItem;
        use moda_scene::value::json;

        // M0 0H10 M20 0H30V10Z. The probe lies strictly inside the closing stroke.
        let points = json!([
            {"type":"move","x":0,"y":0}, {"type":"line","x":10,"y":0},
            {"type":"move","x":20,"y":0}, {"type":"line","x":30,"y":0},
            {"type":"line","x":30,"y":10}
        ]);
        let covers = |triangles: &[f32], [x, y]: [f32; 2]| {
            triangles.chunks_exact(6).any(|t| {
                let cross = |a: usize, b: usize| {
                    (t[b] - t[a]) * (y - t[a + 1]) - (t[b + 1] - t[a + 1]) * (x - t[a])
                };
                let edges = [cross(0, 2), cross(2, 4), cross(4, 0)];
                edges.iter().all(|&e| e > 0.0) || edges.iter().all(|&e| e < 0.0)
            })
        };

        for variant in ["solid", "dash", "trim"] {
            let mut stroke = json!({"id":"s","width":1,"paint":{"type":"solid","color":"#ffffff"}});
            if variant == "dash" {
                stroke["dash"] = json!([4, 4]);
                stroke["dashOffset"] = json!(3);
            }
            let mut path = json!({
                "id":"visible","type":"path","x":5,"y":5,
                "points":points,"closed":[false,true],"strokes":[stroke]
            });
            if variant == "trim" {
                path["trimPathStart"] = json!(0.75);
                path["trimPathEnd"] = json!(1.0);
            }
            for (mask_type, expected) in [("alpha", MaskType::Alpha), ("luminance", MaskType::Luminance)] {
                let mut mask = path.clone();
                mask["id"] = json!("mask");
                mask["maskType"] = json!(mask_type);
                let s = scene(json!({
                    "nodes":[path, mask,
                        {"id":"g","type":"group","children":["mask","content"],"clipMaskNodeId":"mask"},
                        {"id":"content","type":"rectangle","width":40,"height":20,
                         "fills":[{"id":"f","paint":{"type":"solid","color":"#ffffff"}}]}
                    ],
                    "pages":[{"id":"p","width":40,"height":20,"nodeIds":["visible","g"]}]
                }));
                let (plan, warnings) = crate::frame::build_coverage_plan(&s, 0, &Default::default()).unwrap();
                assert!(warnings.is_empty(), "{warnings:?}");
                let visible = plan.items.iter().find_map(|item| match item {
                    DrawItem::Body { node_id, geometry, .. } if node_id == "visible" => Some(geometry),
                    _ => None,
                }).expect("visible stroke");
                let band: Vec<_> = visible.positions.chunks_exact(6).zip(visible.vertex_types.chunks_exact(3))
                    .filter(|(_, types)| *types == [1.0; 3]).flat_map(|(tri, _)| tri.iter().copied()).collect();
                let m = &plan.masks[0];
                assert_eq!(m.mask_type, expected);
                assert_eq!(m.draws.len(), 1);
                assert!(matches!(m.draws[0].paint, MaskPaint::Solid([1.0, 1.0, 1.0, 1.0])));
                assert_eq!(m.positions.as_slice(), band, "{variant} {mask_type}");
                if variant == "solid" || variant == "trim" {
                    assert!(covers(&m.positions, [25.0, 5.05]));
                    assert!(!covers(&m.positions, [27.0, 4.0]));
                    assert_eq!(covers(&m.positions, [5.0, 0.1]), variant == "solid");
                }
            }
        }
    }

    #[test]
    fn local_mask_key_preserves_phase_and_content() {
        let s = scene(moda_scene::value::json!({
            "nodes":[
                {"id":"g","type":"group","x":40,"y":30,"children":["m","c"],"clipMaskNodeId":"m"},
                {"id":"m","type":"ellipse","width":60,"height":40,
                 "fills":[{"id":"f","paint":{"type":"solid","color":"#ffffff"}}]},
                {"id":"c","type":"rectangle","width":80,"height":80}
            ],
            "pages":[{"id":"p","width":300,"height":200,"nodeIds":["g"]}]
        }));
        let mut masks = PlanMasks::default();
        let r = masks.mask_ref_for_node(&s, "c").unwrap();
        let mut m = masks.masks[r.mask_id as usize].clone();
        let key = |m: &MaskPlan| {
            let p = coverage_placement(m.origin, m.size, 1., 8192);
            content_key(m, false, None, &p).unwrap()
        };
        assert!(local_coverage(&m, &coverage_placement(m.origin, m.size, 1., 8192)));
        let initial = key(&m);
        m.origin[0] += 23.;
        m.model_matrix[12] += 23.;
        assert_eq!(key(&m), initial);
        m.origin[0] += 0.25;
        m.model_matrix[12] += 0.25;
        assert_ne!(key(&m), initial, "fractional translation redraws");
        m.origin[0] -= 0.25;
        m.model_matrix[12] -= 0.25;
        let previous = &masks.masks[r.mask_id as usize];
        assert!(std::sync::Arc::ptr_eq(&m.positions, &previous.positions));
        std::sync::Arc::make_mut(&mut m.positions)[0] += 0.5;
        assert!(std::sync::Arc::ptr_eq(&m.draws, &previous.draws));
        assert_eq!(key(previous), initial, "an issued frame keeps its geometry");
        assert_ne!(key(&m), initial, "geometry changes redraw");
        m.ancestor = Some(0);
        assert!(!local_coverage(&m, &coverage_placement(m.origin, m.size, 1., 8192)));
    }

    #[test]
    fn empty_supported_mask_geometry_stays_empty() {
        // Coincident opposite windings have a positive box but no nonzero fill.
        for points in [
            moda_scene::value::json!([
                {"type":"move","x":0,"y":0}, {"type":"line","x":100,"y":0},
                {"type":"line","x":100,"y":100}, {"type":"line","x":0,"y":100}, {"type":"line","x":0,"y":0},
                {"type":"move","x":0,"y":0}, {"type":"line","x":0,"y":100},
                {"type":"line","x":100,"y":100}, {"type":"line","x":100,"y":0}, {"type":"line","x":0,"y":0}
            ]),
            moda_scene::value::json!([]),
        ] {
            let s = scene(moda_scene::value::json!({
                "nodes":[
                    {"id":"g","type":"group","children":["m","c"],"clipMaskNodeId":"m"},
                    {"id":"m","type":"path","width":100,"height":100,"closed":true,"points":points},
                    {"id":"c","type":"rectangle","width":100,"height":100}
                ],
                "pages":[{"id":"p","width":100,"height":100,"nodeIds":["g"]}]
            }));
            let geometry = mask_geometry_positions(s.node("m").unwrap(), Some(&s));
            assert!(geometry.positions.is_empty());
            assert!(geometry.fill_dropped.is_none());
            let mut masks = PlanMasks::default();
            let r = masks.mask_ref_for_node(&s, "c").expect("an empty mask must still clip");
            assert!(masks.masks[r.mask_id as usize].positions.is_empty(), "empty coverage became a box");
        }
    }

    fn group_clip_scene() -> Scene {
        scene(moda_scene::value::json!({
            "nodes": [
                {"id":"g","type":"group","x":240.0,"y":160.0,"children":["m","c"],
                 "clipMaskNodeId":"m"},
                {"id":"m","type":"rectangle","x":0.0,"y":0.0,"width":480.0,"height":360.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}}],"cornerRadius":48.0},
                {"id":"c","type":"ellipse","x":-80.0,"y":-60.0,"width":360.0,"height":360.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}]}
            ],
            "pages": [{"id":"p","width":1280.0,"height":800.0,"nodeIds":["g"]}],
            "config": {"renderPageClipping": true}
        }))
    }

    #[test]
    fn clip_contains_ellipse_but_cuts_its_drawn_cubic() {
        let at = |degrees: f64| {
            (1920. * degrees.to_radians().cos(), 1000. + 1920. * degrees.to_radians().sin())
        };
        let (x0, y0) = at(-19.);
        let (x1, y1) = at(70.99);
        let s = scene(moda_scene::value::json!({
            "nodes":[{"id":"clip","type":"container","width":1920.1,"height":3000,
                      "clipContent":true,"children":["p"]},
                     {"id":"p","type":"path","points":[{"type":"move","x":x0,"y":y0},
                      {"type":"arc","x":x1,"y":y1,"rx":1920,"ry":1920,"xAxisRotation":0,"largeArc":false,"sweep":true}]}],
            "pages":[{"id":"page","width":2000,"height":3000,"nodeIds":["clip"]}]
        }));
        assert!(!container_clip_is_redundant(&s, s.node("clip").unwrap(), "p"));
    }

    /// Warped mask bounds must map each warped corner through the
    /// affine THEN take the AABB — matching TS getNodeDistortCanvasBounds.
    /// A 30° rotation (NOT 90°, where the two orderings coincide) pins the
    /// ordering: taking the warped quad's local AABB first and mapping THAT
    /// box would include the image of the (edge, 100) box corner, which lies
    /// strictly outside the quad hull — a strictly larger canvas AABB.
    #[test]
    fn warped_mask_aabb_maps_corners_through_affine() {
        // 100x100 rect, rotated 30°, with a pure projective x-lean: matrix
        // [1,0,0, 0,1,0, 0.3,0,1] contracts the right edge to 100/1.3 ≈ 76.923
        // local px. Warped local corners: (0,0), (e,0), (e,e), (0,100) with
        // e = 100/1.3.
        let s = scene(moda_scene::value::json!({
            "nodes": [
                {"id":"m","type":"rectangle","x":0.0,"y":0.0,"width":100.0,"height":100.0,
                 "rotation":30.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}],
                 "distort":{"type":"matrix","matrix":[1,0,0,0,1,0,0.3,0,1]}}
            ],
            "pages": [{"id":"p","width":800.0,"height":600.0,"nodeIds":["m"]}],
            "config": {"renderPageClipping": true}
        }));
        let node = s.node("m").expect("mask node");
        let (x, y, w, h) = node_canvas_aabb(&s, node).expect("aabb");

        let a = s.node_canvas_affine("m").expect("affine");
        let apply = |px: f64, py: f64| (a.a * px + a.c * py + a.e, a.b * px + a.d * py + a.f);
        let aabb_of = |pts: &[(f64, f64)]| {
            let min_x = pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
            let max_x = pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
            let min_y = pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
            let max_y = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
            (min_x, min_y, max_x - min_x, max_y - min_y)
        };

        // Expected: corners-through-affine, THEN AABB.
        let e = 100.0 / 1.3;
        let quad = [(0.0, 0.0), (e, 0.0), (e, e), (0.0, 100.0)];
        let mapped: Vec<(f64, f64)> = quad.iter().map(|&(px, py)| apply(px, py)).collect();
        let (ex, ey, ew, eh) = aabb_of(&mapped);
        assert!((x - ex).abs() < 1e-9, "x {x} vs {ex}");
        assert!((y - ey).abs() < 1e-9, "y {y} vs {ey}");
        assert!((w - ew).abs() < 1e-9, "w {w} vs {ew}");
        assert!((h - eh).abs() < 1e-9, "h {h} vs {eh}");

        // The wrong ordering (local AABB of the warped quad, then affine)
        // must be STRICTLY larger at 30° — this is what the corner mapping
        // buys us.
        let boxed = [(0.0, 0.0), (e, 0.0), (e, 100.0), (0.0, 100.0)];
        let mapped_box: Vec<(f64, f64)> = boxed.iter().map(|&(px, py)| apply(px, py)).collect();
        let (_, _, ow, oh) = aabb_of(&mapped_box);
        assert!(
            ow * oh > w * h + 1e-6,
            "AABB-then-affine ({ow}x{oh}) must strictly over-approximate corners-then-AABB ({w}x{h})"
        );
    }

    #[test]
    fn mask_ref_uses_canvas_aabb() {
        let s = group_clip_scene();
        let mut pm = PlanMasks::default();
        let r = pm.mask_ref_for_node(&s, "c").expect("masked");
        let m = &pm.masks[r.mask_id as usize];
        // Group at (240,160) on the page; page origin (0,0) for page 0.
        assert_eq!(m.origin, [240.0, 160.0]);
        assert_eq!(m.size, [480.0, 360.0]);
        assert_eq!(m.mode, 0);
        assert!(m.ancestor.is_none());
        assert!((r.inv_size[0] - 1.0 / 480.0).abs() < 1e-9);
        // The coverage texture follows the frame's pixel scale (no dpr-1 pin).
        let at = |px: f64| coverage_placement(m.origin, m.size, coverage_scale(px), 16384);
        assert_eq!(at(1.0).tex, [480, 360]);
        assert_eq!(at(2.0).tex, [960, 720]);
        // 1.5× rounds up to the next power of two; 0.3× down-sizes to 0.5×.
        assert_eq!(at(1.5).tex, [960, 720]);
        assert_eq!(at(0.3).tex, [240, 180]);
    }

    /// A luminance mask is the luminance of the mask's picture as sRGB. In an sRGB plan a lone shape's paints
    /// reduce one by one. In a plan for an engine that blends in another encoding that holds for one layer of
    /// one solid colour; a shape with several layers, or a gradient, is its rendered picture, reduced once.
    /// Masks that read no colour (vector, alpha) keep the direct path.
    #[test]
    fn luminance_masks_beyond_one_solid_layer_are_rendered_in_a_working_colour() {
        let masked = |mask_type: &str, stroked: bool| {
            let strokes = if stroked {
                moda_scene::value::json!([{"id": "stroke-0", "paint": {"type": "solid", "color": "#ffffff"}, "width": 10, "opacity": 0.5}])
            } else {
                moda_scene::value::json!([])
            };
            scene(moda_scene::value::json!({
                "nodes": [
                    {"id":"g","type":"group","x":0.0,"y":0.0,"children":["m","c"],"clipMaskNodeId":"m"},
                    {"id":"m","type":"rectangle","x":0.0,"y":0.0,"width":100.0,"height":100.0,"maskType":mask_type,
                     "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}}], "strokes": strokes},
                    {"id":"c","type":"rectangle","x":0.0,"y":0.0,"width":100.0,"height":100.0,
                     "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}]}
                ],
                "pages": [{"id":"p","width":200.0,"height":200.0,"nodeIds":["g"]}],
                "config": {"renderPageClipping": true}
            }))
        };
        for (mask_type, stroked, working, rendered) in [
            ("luminance", true, false, false), ("luminance", false, true, false), ("luminance", true, true, true),
            ("alpha", true, true, false), ("vector", true, true, false),
        ] {
            let s = masked(mask_type, stroked);
            let mut pm = PlanMasks::default().in_working_color(working);
            let r = pm.mask_ref_for_node(&s, "c").expect("masked");
            let m = &pm.masks[r.mask_id as usize];
            assert_eq!(m.mode == 3, rendered, "{mask_type}, stroked {stroked}, working colour {working}: mode {}", m.mode);
            assert_eq!(m.image.as_ref().is_some_and(|image| image.rendered.is_some()), rendered, "{mask_type}, stroked {stroked}, working colour {working}");
        }
    }

    #[test]
    fn coverage_scale_is_a_power_of_two_at_or_above_the_pixel_scale() {
        assert_eq!(coverage_scale(1.0), 1.0);
        assert_eq!(coverage_scale(2.0), 2.0);
        assert_eq!(coverage_scale(2.01), 4.0);
        assert_eq!(coverage_scale(0.26), 0.5);
        assert_eq!(coverage_scale(1e-9), 1.0 / 16.0);
        assert_eq!(coverage_scale(1e9), 64.0);
        assert_eq!(coverage_scale(f64::NAN), 1.0);
        assert_eq!(coverage_scale(0.0), 1.0);
    }

    #[test]
    fn coverage_placement_snaps_to_the_texel_grid_and_fits_the_limits() {
        // A fractional box snaps out to whole texels at 2×.
        let p = coverage_placement([10.3, 20.6], [100.0, 50.0], 2.0, 16384);
        assert_eq!(p.scale, 2.0);
        assert_eq!(p.origin, [10.0, 20.5]);
        assert_eq!(p.tex, [201, 101]);
        assert_eq!(p.size, [100.5, 50.5]);
        assert!(p.origin[0] + p.size[0] >= 110.3 && p.origin[1] + p.size[1] >= 70.6);
        // Past the per-side device limit, the scale halves until it fits.
        let p = coverage_placement([0.0, 0.0], [5000.0, 10.0], 4.0, 8192);
        assert_eq!((p.scale, p.tex), (1.0, [5000, 10]));
        // Past the texel budget likewise.
        let p = coverage_placement([0.0, 0.0], [4096.0, 4096.0], 2.0, 16384);
        assert_eq!((p.scale, p.tex), (1.0, [4096, 4096]));
        assert!(u64::from(p.tex[0]) * u64::from(p.tex[1]) <= MAX_MASK_TEXELS);
    }

    #[test]
    fn effective_clip_intersects_rect_mask_aabb() {
        let s = group_clip_scene();
        let base = [0.0f32, 0.0, 1280.0, 800.0];
        let clip = effective_page_clip(&s, "c", base);
        assert_eq!(clip, Some([240.0, 160.0, 480.0, 360.0]));
        // Mask source itself is outside the chain walk only if it has no
        // masked ancestors — "m" is a child of g, so it intersects too.
        // Unmasked nodes keep the base clip.
        let clip_g = effective_page_clip(&s, "g", base);
        assert_eq!(clip_g, Some(base));
    }

    #[test]
    fn effective_clip_disjoint_nested_masks_is_none() {
        // Q-113: nested rect masks whose canvas AABBs do NOT intersect →
        // computeEffectivePageClip returns null (intersectRect :147-149).
        // Derived from fuzz s1101-032 (g-…-6 mask x∈[20,178] vs nested
        // g-…-13 mask x∈[216,336]).
        let s = scene(moda_scene::value::json!({
            "nodes": [
                {"id":"outer","type":"group","x":20.0,"y":458.0,
                 "children":["om","inner"],"clipMaskNodeId":"om"},
                {"id":"om","type":"rectangle","x":0.0,"y":0.0,"width":158.0,"height":115.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}]},
                {"id":"inner","type":"group","x":196.0,"y":32.0,
                 "children":["im","t"],"clipMaskNodeId":"im"},
                {"id":"im","type":"rectangle","x":0.0,"y":0.0,"width":120.0,"height":114.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}]},
                {"id":"t","type":"richtext","x":171.0,"y":24.0,"width":220.0,"height":140.0,
                 "htmlContent":"<p>x</p>"}
            ],
            "pages": [{"id":"p","width":700.0,"height":900.0,"nodeIds":["outer"]}],
            "config": {"renderPageClipping": true}
        }));
        let base = [0.0f32, 0.0, 700.0, 900.0];
        assert_eq!(effective_page_clip(&s, "t", base), None);
        // The inner mask's own clip is the outer ∩ page (non-empty).
        assert!(effective_page_clip(&s, "inner", base).is_some());
    }

    // -----------------------------------------------------------------------
    // `clip_to_node_box` (table cell text overflow)
    // -----------------------------------------------------------------------

    /// One richtext node on a page, with whatever extra fields the caller wants
    /// merged onto it (`clipTextToBounds`, `rotation`, …).
    fn richtext_scene(extra: moda_scene::value::Value) -> Scene {
        let mut node = moda_scene::value::json!({
            "id":"t","type":"richtext","x":40.0,"y":30.0,"width":120.0,"height":60.0,
            "htmlContent":"<p>overflowing cell text</p>"
        });
        let obj = node.as_object_mut().expect("node object");
        for (k, v) in extra.as_object().expect("extra object") {
            obj.insert(k.clone(), v.clone());
        }
        scene(moda_scene::value::json!({
            "nodes": [node],
            "pages": [{"id":"p","width":800.0,"height":600.0,"nodeIds":["t"]}],
            "config": {"renderPageClipping": true}
        }))
    }

    /// The byte-exactness contract: without the flag the clip comes back
    /// VERBATIM, so every doc predating the field plans identically. A present-
    /// but-JS-falsy value (`false`) must behave the same — `truthy` is the gate,
    /// not mere presence, mirroring `!!node.clipTextToBounds` in TS.
    #[test]
    fn clip_to_node_box_without_flag_is_identity() {
        let base = [0.0f32, 0.0, 800.0, 600.0];
        for extra in [
            moda_scene::value::json!({}),
            moda_scene::value::json!({"clipTextToBounds": false}),
            moda_scene::value::json!({"clipTextToBounds": moda_scene::value::Value::Null}),
        ] {
            let s = richtext_scene(extra);
            let node = s.node("t").expect("richtext node");
            assert_eq!(clip_to_node_box(&s, node, base), Some(base));
        }
    }

    /// Flagged node whose box is strictly inside the incoming clip: the clip
    /// narrows to the node's canvas box exactly (this is what stops a table
    /// cell's text bleeding into its neighbours).
    #[test]
    fn clip_to_node_box_narrows_to_node_box() {
        let s = richtext_scene(moda_scene::value::json!({"clipTextToBounds": true}));
        let node = s.node("t").expect("richtext node");
        let base = [0.0f32, 0.0, 800.0, 600.0];
        assert_eq!(
            clip_to_node_box(&s, node, base),
            Some([40.0, 30.0, 120.0, 60.0])
        );
        // Composed with the page clip the way frame.rs's richtext arm does it.
        let eff = effective_page_clip(&s, "t", base).expect("page clip");
        assert_eq!(
            clip_to_node_box(&s, node, eff),
            Some([40.0, 30.0, 120.0, 60.0])
        );
    }

    /// DOCUMENTED RESIDUAL: the clip is an axis-aligned box, so a ROTATED node
    /// clips to the AABB of its rotated box — deliberately LOOSE, exactly like
    /// container `clipContent` / group rect masks above (all three go through
    /// `node_canvas_aabb`). Tight rotated clipping would need the R8 coverage
    /// path, which the MSDF/vector text pipelines do not bind.
    ///
    /// Analytic expectation for a 100x100 square at (150,150) rotated 45°:
    /// Moda's TRS rotates about the node's LOCAL ORIGIN (its top-left), not its
    /// centre (`transform2d.ts:19-31` / `nodeTransformUtils.ts:54-66`, ported in
    /// `Scene::node_canvas_affine`), so the local corners map to (0,0),
    /// (±70.71, 70.71), (0, 141.42) and the canvas AABB is
    /// x ∈ [150 − 100/√2, 150 + 100/√2], y ∈ [150, 150 + 100·√2] — a 141.42
    /// square, TWICE the node box's area, and not even concentric with it. That
    /// slack is the residual, asserted here so a future tightening has to
    /// update this test deliberately.
    #[test]
    fn clip_to_node_box_rotated_is_a_loose_aabb() {
        let s = scene(moda_scene::value::json!({
            "nodes": [{"id":"t","type":"richtext","x":150.0,"y":150.0,
                       "width":100.0,"height":100.0,"rotation":45.0,
                       "clipTextToBounds":true,"htmlContent":"<p>x</p>"}],
            "pages": [{"id":"p","width":800.0,"height":600.0,"nodeIds":["t"]}],
            "config": {"renderPageClipping": true}
        }));
        let node = s.node("t").expect("richtext node");
        let base = [0.0f32, 0.0, 800.0, 600.0];
        let got = clip_to_node_box(&s, node, base).expect("non-empty");
        let half_diag = 100.0 / std::f32::consts::SQRT_2;
        let side = 100.0 * std::f32::consts::SQRT_2;
        let expected = [150.0 - half_diag, 150.0, side, side];
        for i in 0..4 {
            assert!(
                (got[i] - expected[i]).abs() < 1e-3,
                "component {i}: {got:?} vs {expected:?}"
            );
        }
        // Loose, not tight: strictly larger than the 100x100 node box.
        assert!(
            got[2] * got[3] > 100.0 * 100.0 + 1.0,
            "rotated clip {got:?} must over-approximate the node box"
        );
    }

    /// Flagged node disjoint from the incoming clip → `None`, which the richtext
    /// arm treats identically to a null effective clip (Q-113): consume z, draw
    /// nothing. Same `intersect_rect`, so degenerate (edge-touching) overlaps are
    /// empty too.
    #[test]
    fn clip_to_node_box_disjoint_is_none() {
        let s = richtext_scene(moda_scene::value::json!({"clipTextToBounds": true}));
        let node = s.node("t").expect("richtext node");
        // Node box is [40,30,120,60]; this clip lies entirely to its right.
        assert_eq!(clip_to_node_box(&s, node, [400.0, 30.0, 100.0, 60.0]), None);
        // Zero-area edge contact (clip starts exactly at the box's right edge)
        // is also None — `intersectRect`'s `x2 <= x` semantics.
        assert_eq!(clip_to_node_box(&s, node, [160.0, 30.0, 100.0, 60.0]), None);
    }

    #[test]
    fn nested_masks_compose_via_ancestor_index() {
        let s = scene(moda_scene::value::json!({
            "nodes": [
                {"id":"outer","type":"group","x":160.0,"y":120.0,
                 "children":["om","inner"],"clipMaskNodeId":"om"},
                {"id":"om","type":"rectangle","x":0.0,"y":0.0,"width":640.0,"height":480.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}],"cornerRadius":120.0},
                {"id":"inner","type":"group","x":120.0,"y":80.0,
                 "children":["im","content"],"clipMaskNodeId":"im"},
                {"id":"im","type":"rectangle","x":0.0,"y":0.0,"width":400.0,"height":320.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}]},
                {"id":"content","type":"ellipse","x":-60.0,"y":-60.0,"width":520.0,
                 "height":440.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#8338ec"}}]}
            ],
            "pages": [{"id":"p","width":1280.0,"height":800.0,"nodeIds":["outer"]}],
            "config": {"renderPageClipping": true}
        }));
        let mut pm = PlanMasks::default();
        let r = pm.mask_ref_for_node(&s, "content").expect("masked");
        let inner = &pm.masks[r.mask_id as usize];
        let anc = inner.ancestor.expect("inner mask composes outer");
        assert!(anc < r.mask_id as usize, "ancestors precede descendants");
        assert_eq!(pm.masks[anc].owner_node_id, "outer");
        assert_eq!(pm.masks[anc].origin, [160.0, 120.0]);
        assert_eq!(inner.origin, [160.0 + 120.0, 120.0 + 80.0]);
    }

    /// A distort-warped GEOMETRY mask clips to the warped
    /// silhouette — the coverage triangles must land inside the warped quad,
    /// not the unwarped box.
    #[test]
    fn warped_mask_coverage_uses_warped_geometry() {
        let warped = RawNode::from_value(moda_scene::value::json!({
            "id":"m","type":"rectangle","width":100.0,"height":100.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}],
            "distort":{"type":"matrix","matrix":[1,0,0,0,1,0,0.3,0,1]}
        }))
        .unwrap();
        let positions = mask_geometry_positions(&warped, None).positions;
        assert!(!positions.is_empty());
        // The x-lean matrix contracts the right edge to 100/1.3 ≈ 76.92 local
        // px — no warped vertex may reach the unwarped x = 100 edge.
        let max_x = positions
            .chunks(2)
            .map(|p| p[0])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (max_x - 100.0 / 1.3).abs() < 1e-3,
            "warped coverage must end at the contracted edge, got {max_x}"
        );
    }

    fn text_mask_scene() -> Scene {
        scene(moda_scene::value::json!({
            "nodes": [
                {"id":"g","type":"group","x":70.0,"y":74.0,"children":["m","c"],
                 "clipMaskNodeId":"m"},
                {"id":"m","type":"richtext","x":0.0,"y":0.0,"width":1920.0,"height":1080.0,
                 "rotation":0,"scale":1,
                 "htmlContent":"<p style=\"font-size: 800px; font-family: &quot;Season Sans&quot;;\">UGC</p>"},
                {"id":"c","type":"rectangle","x":27.0,"y":162.0,"width":1839.0,"height":621.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}]}
            ],
            "pages": [{"id":"p","width":1920.0,"height":1080.0,"nodeIds":["g"]}],
            "config": {"renderPageClipping": true}
        }))
    }

    #[test]
    fn text_mask_builds_on_every_page() {
        // The export path built text-mask coverage on page 0 only (Q-95), so
        // page 2 of a PDF baked the children unclipped; the default builds it.
        let s = text_mask_scene();
        let mut pm = PlanMasks::default();
        let r = pm
            .mask_ref_for_node(&s, "c")
            .expect("text mask builds");
        assert_eq!(pm.masks[r.mask_id as usize].mode, 1, "text coverage mode");
    }

    #[test]
    fn text_mask_override() {
        let s = text_mask_scene();
        let mut pm = PlanMasks::with_text_masks(true);
        let r = pm
            .mask_ref_for_node(&s, "c")
            .expect("the text mask builds");
        assert_eq!(pm.masks[r.mask_id as usize].mode, 1, "text coverage mode");
        // The explicit false override drops it.
        let mut pm_off = PlanMasks::with_text_masks(false);
        assert!(pm_off.mask_ref_for_node(&s, "c").is_none());
    }

    fn inter_text_mask_scene() -> Scene {
        scene(moda_scene::value::json!({
            "nodes": [
                {"id":"g","type":"group","x":180.0,"y":240.0,"children":["m","c"],
                 "clipMaskNodeId":"m"},
                {"id":"m","type":"richtext","x":0.0,"y":0.0,"width":920.0,"height":320.0,
                 "rotation":0,"scale":1,
                 "htmlContent":"<p style=\"font-family: Inter; font-size: 160px; font-weight: 800; color: #000000\">MODA</p>"},
                {"id":"c","type":"rectangle","x":0.0,"y":0.0,"width":920.0,"height":320.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}]}
            ],
            "pages": [{"id":"p","width":1280.0,"height":800.0,"nodeIds":["g"]}],
            "config": {"renderPageClipping": true}
        }))
    }

    /// mask:text-alpha better tier: under the vector engine a vector-servable
    /// mask source renders the VECTOR raster (different letterforms), while a
    /// fallback-class source keeps the byte-identical Degraded raster; the
    /// coverage-quad GEOMETRY (logical size / origin) never changes tier.
    #[test]
    fn vector_arm_swaps_raster_pixels_but_never_geometry() {
        // Vector-servable (Inter): pixels differ, geometry identical.
        let s = inter_text_mask_scene();
        let mut degraded_pm = PlanMasks::default();
        let mut vector_pm = PlanMasks::default().with_vector_text_masks(true);
        let d = degraded_pm
            .mask_ref_for_node(&s, "c")
            .expect("degraded arm");
        let v = vector_pm.mask_ref_for_node(&s, "c").expect("vector arm");
        let dt = degraded_pm.masks[d.mask_id as usize]
            .text
            .as_ref()
            .expect("degraded MaskText");
        let vt = vector_pm.masks[v.mask_id as usize]
            .text
            .as_ref()
            .expect("vector MaskText");
        assert_eq!(dt.raster.logical_w, vt.raster.logical_w, "Q-68 geometry");
        assert_eq!(dt.raster.logical_h, vt.raster.logical_h, "Q-68 geometry");
        assert_eq!(dt.origin_canvas, vt.origin_canvas, "Q-68 anchor");
        assert_eq!(
            degraded_pm.masks[d.mask_id as usize].positions,
            vector_pm.masks[v.mask_id as usize].positions,
            "coverage quad must not change tier"
        );
        assert_ne!(
            dt.raster.rgba, vt.raster.rgba,
            "vector arm must swap in the vector letterforms"
        );

        // Fallback-class (a drop shadow — a silhouette gate): the vector arm
        // declines and the raster is byte-identical to the degraded arm.
        let s = scene(moda_scene::value::json!({
            "nodes": [
                {"id":"g","type":"group","x":180.0,"y":240.0,"children":["m","c"],
                 "clipMaskNodeId":"m"},
                {"id":"m","type":"richtext","x":0.0,"y":0.0,"width":920.0,"height":320.0,
                 "rotation":0,"scale":1,
                 "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000000","blur":4.0,"offset":{"x":0.0,"y":4.0}}],
                 "htmlContent":"<p style=\"font-family: Inter; font-size: 160px; font-weight: 800; color: #000000\">MODA</p>"},
                {"id":"c","type":"rectangle","x":0.0,"y":0.0,"width":920.0,"height":320.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}]}
            ],
            "pages": [{"id":"p","width":1280.0,"height":800.0,"nodeIds":["g"]}],
            "config": {"renderPageClipping": true}
        }));
        let mut degraded_pm = PlanMasks::default();
        let mut vector_pm = PlanMasks::default().with_vector_text_masks(true);
        let d = degraded_pm
            .mask_ref_for_node(&s, "c")
            .expect("degraded arm");
        let v = vector_pm.mask_ref_for_node(&s, "c").expect("vector arm");
        assert_eq!(
            degraded_pm.masks[d.mask_id as usize]
                .text
                .as_ref()
                .unwrap()
                .raster,
            vector_pm.masks[v.mask_id as usize]
                .text
                .as_ref()
                .unwrap()
                .raster,
            "fallback-class mask must keep the Degraded raster byte-identical"
        );
    }

    /// B3 (canvas 27aa35fe p05): polygon mask must produce the hexagon's
    /// tessellated coverage, not the full-quad fallback (which clipped as a
    /// 420x420 square). Browser arm: WebGPUMaskRenderer.ts:968-975.
    #[test]
    fn polygon_mask_uses_polygon_geometry_not_full_quad() {
        let n = RawNode::from_value(moda_scene::value::json!({
            "id":"m","type":"polygon","width":420.0,"height":420.0,
            "sides":6,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}]
        }))
        .unwrap();
        let positions = mask_geometry_positions(&n, None).positions;
        assert!(!positions.is_empty(), "polygon arm must tessellate");
        // A hexagon inscribed in 420x420 has NO vertex at the box corners;
        // the full-quad fallback does. Corner (0,0) must not appear.
        assert!(
            !positions
                .chunks(2)
                .any(|p| p[0].abs() < 1e-3 && p[1].abs() < 1e-3),
            "coverage contains the (0,0) box corner — full-quad fallback leaked"
        );
        // Top vertex (w/2, 0) of the inscribed hexagon must appear.
        assert!(
            positions
                .chunks(2)
                .any(|p| (p[0] - 210.0).abs() < 1e-3 && p[1].abs() < 1e-3),
            "inscribed-hexagon top vertex missing from coverage"
        );
    }

    #[test]
    fn star_mask_uses_star_geometry_not_full_quad() {
        let n = RawNode::from_value(moda_scene::value::json!({
            "id":"m","type":"star","width":200.0,"height":200.0,
            "numPoints":5,"innerRadiusRatio":0.5,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}]
        }))
        .unwrap();
        let positions = mask_geometry_positions(&n, None).positions;
        assert!(!positions.is_empty(), "star arm must tessellate");
        assert!(
            !positions
                .chunks(2)
                .any(|p| p[0].abs() < 1e-3 && p[1].abs() < 1e-3),
            "coverage contains the (0,0) box corner — full-quad fallback leaked"
        );
    }

    #[test]
    fn line_mask_tessellates_stroke_geometry() {
        let n = RawNode::from_value(moda_scene::value::json!({
            "id":"m","type":"line","points":[0.0,0.0,100.0,0.0],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10}]
        }))
        .unwrap();
        let positions = mask_geometry_positions(&n, None).positions;
        assert!(!positions.is_empty(), "line arm must tessellate its stroke");
        // The stroke band straddles the line (y in [-5, 5] for the default
        // center align) — full-quad fallback would sit at y in [0, h].
        assert!(
            positions.chunks(2).any(|p| p[1] < -1.0),
            "expected stroke coverage above the line's y=0 spine"
        );
    }

    #[test]
    fn image_mask_requires_pattern_no_repeat() {
        let top_image_source = |node: &RawNode| match moda_scene::paint::top_fill(node)?.paint {
            moda_scene::paint::Paint::Image(image) => image_mask_source(node, &image),
            _ => None,
        };
        let tiled = RawNode::from_value(moda_scene::value::json!({
            "id":"m","type":"rectangle","width":64.0,"height":64.0,
            "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "data:image/png;base64,xxxx", "placement": {"repeat": "repeat"}}}]
        }))
        .unwrap();
        assert!(
            top_image_source(&tiled).is_none(),
            "Q-21: tiling → geometry"
        );
        let no_pattern = RawNode::from_value(moda_scene::value::json!({
            "id":"m","type":"rectangle","width":64.0,"height":64.0,
            "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}]
        }))
        .unwrap();
        assert!(top_image_source(&no_pattern).is_none());
        // A valid distort falls back to (warped) geometry coverage —
        // the image-mask placement math cannot represent the warp.
        let warped = RawNode::from_value(moda_scene::value::json!({
            "id":"m","type":"rectangle","width":64.0,"height":64.0,
            "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "data:image/png;base64,xxxx", "placement": {"repeat": "no-repeat"}}}],"distort":{"type":"matrix","matrix":[1,0,0,0,1,0,0.3,0.1,1]}
        }))
        .unwrap();
        assert!(top_image_source(&warped).is_none());
    }

}
