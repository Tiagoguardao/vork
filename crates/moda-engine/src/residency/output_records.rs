//! Retained output operations. Scene stamps name changed sources; output edges name their consumers.
//! Bounds keep the ordinary executor's reduction order, including its floating-point round trips.
use super::*;
use crate::backdrop::plan::Rect;
use crate::seams::DrawItem;
use std::collections::{BTreeMap, BTreeSet};

type Address = Vec<usize>;
fn rect_bits(rect: Option<Rect>) -> Option<[u64; 4]> {
    rect.map(|r| [r.x.to_bits(), r.y.to_bits(), r.w.to_bits(), r.h.to_bits()])
}
use crate::layer_blur::shadow::union_canvas_bounds as union;

fn same_paint(a: &crate::seams::BodyPaint, b: &crate::seams::BodyPaint) -> bool {
    use crate::seams::BodyPaint;
    match (a, b) {
        (BodyPaint::Solid, BodyPaint::Solid) => true,
        (BodyPaint::Gradient(a), BodyPaint::Gradient(b)) => {
            a == b
        }

        (BodyPaint::Pattern(a), BodyPaint::Pattern(b)) => {
            let (at, bt) = (&a.texture, &b.texture);
            let (ai, bi) = (&at.image, &bt.image);
            a.instance == b.instance
                && a.clamp_sampler == b.clamp_sampler
                && at.key == bt.key
                && at.state == bt.state
                && at.scale_factor.0.to_bits() == bt.scale_factor.0.to_bits()
                && at.scale_factor.1.to_bits() == bt.scale_factor.1.to_bits()
                && ai.width == bi.width
                && ai.height == bi.height
                && ai.straight == bi.straight
                && ai.external.as_ref().map(|e| (&e.key, e.upload_id))
                    == bi.external.as_ref().map(|e| (&e.key, e.upload_id))
                && (Arc::ptr_eq(&ai.rgba_premultiplied, &bi.rgba_premultiplied)
                    || ai.rgba_premultiplied == bi.rgba_premultiplied)
        }
        _ => false,
    }
}
fn same_mask(a: &Option<crate::seams::MaskRef>, b: &Option<crate::seams::MaskRef>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.mask_id == b.mask_id
                && a.origin.map(f32::to_bits) == b.origin.map(f32::to_bits)
                && a.inv_size.map(f32::to_bits) == b.inv_size.map(f32::to_bits)
        }
        _ => false,
    }
}

/// Immutable geometry/capture identity and packed instance bytes are emission proofs.
/// Other output kinds retain the exact cold admission comparison until they expose the same columns.
fn same_output(a: &DrawItem, b: &DrawItem) -> bool {
    match (a, b) {
        (
            DrawItem::LayerBlurComposite { node_id: an, z: az, group: ag,
                page_clip_bounds: ac, clip_rrect: ar, mask: am },
            DrawItem::LayerBlurComposite { node_id: bn, z: bz, group: bg,
                page_clip_bounds: bc, clip_rrect: br, mask: bm },
        ) if Arc::ptr_eq(ag, bg) => {
            // Clip normalization may change the composite without changing its immutable capture.
            // Compare the composite columns; walking the shared capture would repeat its emission.
            let rounded = |clip: &Option<crate::seams::ClipRRect>| clip.as_ref()
                .map(|r| (r.rect.map(f32::to_bits), r.radii.map(f32::to_bits)));
            an == bn && az.to_bits() == bz.to_bits()
                && ac.map(f32::to_bits) == bc.map(f32::to_bits)
                && rounded(ar) == rounded(br) && same_mask(am, bm)
        }
        (
            DrawItem::Body {
                node_id: an,
                z: az,
                phase: ap,
                paint: ac,
                geometry: ag,
                instance: ai,
                blend_mode: ab,
                mask: am,
            },
            DrawItem::Body {
                node_id: bn,
                z: bz,
                phase: bp,
                paint: bc,
                geometry: bg,
                instance: bi,
                blend_mode: bb,
                mask: bm,
            },
        ) if Arc::ptr_eq(ag, bg) => {
            an == bn
                && az.to_bits() == bz.to_bits()
                && ap == bp
                && ab == bb
                && bytemuck::bytes_of(ai.as_ref()) == bytemuck::bytes_of(bi.as_ref())
                && same_paint(ac, bc)
                && same_mask(am, bm)
        }
        (DrawItem::Text(a), DrawItem::Text(b)) if Arc::ptr_eq(a, b) => true,
        (DrawItem::Text(a), DrawItem::Text(b))
            if a.content_key.is_some() && a.content_key == b.content_key =>
        {
            // The text cache's existing content identity names every non-placement field.
            // Capture clip normalization may clone the batch, but does not change that identity.
            // Clock-bearing output was refused before admission, as for capture content keys.
            let same = a.node_id == b.node_id
                && a.z.to_bits() == b.z.to_bits()
                && a.model_matrix.map(f32::to_bits) == b.model_matrix.map(f32::to_bits)
                && a.page_clip_bounds.map(f32::to_bits) == b.page_clip_bounds.map(f32::to_bits);
            debug_assert!(!same || format!("{a:?}") == format!("{b:?}"),
                "text content identity and placement must certify ordinary output");
            same
        }
        _ => format!("{a:?}") == format!("{b:?}"),
    }
}

/// Each prefix is a derived value. Re-evaluate from the first changed input, stopping as soon as
/// the prefix and all later input values stand. Unlike regrouping a rectangle union, this is exact.
struct OrderedBounds {
    inputs: Vec<Option<Rect>>,
    prefixes: Vec<Option<Rect>>,
}
impl OrderedBounds {
    fn read(inputs: Vec<Option<Rect>>) -> Self {
        let mut value = None;
        let prefixes = inputs
            .iter()
            .map(|&input| {
                value = union(value, input);
                value
            })
            .collect();
        Self { inputs, prefixes }
    }
    fn write(&mut self, updates: impl IntoIterator<Item = (usize, Option<Rect>)>) -> Option<Rect> {
        let mut dirty = BTreeSet::new();
        for (at, value) in updates {
            if rect_bits(self.inputs[at]) != rect_bits(value) {
                self.inputs[at] = value;
                dirty.insert(at);
            }
        }
        while let Some(start) = dirty.pop_first() {
            let mut value = if start == 0 {
                None
            } else {
                self.prefixes[start - 1]
            };
            for at in start..self.inputs.len() {
                dirty.remove(&at);
                value = union(value, self.inputs[at]);
                if rect_bits(value) == rect_bits(self.prefixes[at]) {
                    break;
                }
                self.prefixes[at] = value;
            }
        }
        self.value()
    }
    fn value(&self) -> Option<Rect> {
        self.prefixes.last().copied().flatten()
    }
}

/// Raw extrema form the bounds column for prepared body/text outputs. The shared bounds emitter does
/// the arithmetic; this tree only selects minima/maxima and constructs the final Rect once per consumer.
#[derive(Default)]
struct OutputBounds {
    first: usize,
    tree: Vec<Option<crate::layer_blur::shadow::ItemExtents>>,
}
impl OutputBounds {
    fn join(
        a: Option<crate::layer_blur::shadow::ItemExtents>,
        b: Option<crate::layer_blur::shadow::ItemExtents>,
    ) -> Option<crate::layer_blur::shadow::ItemExtents> {
        match (a, b) {
            (Some(a), Some(b)) => Some(a.union(b)),
            (a, None) => a,
            (None, b) => b,
        }
    }
    fn write(&mut self, slot: usize, item: &crate::seams::DrawItem) {
        let mut at = self.first + slot;
        self.tree[at] = crate::layer_blur::shadow::item_extents(std::slice::from_ref(item));
        while at > 1 {
            at /= 2;
            self.tree[at] = Self::join(self.tree[at * 2], self.tree[at * 2 + 1]);
        }
    }
    fn range(&self, start: usize, end: usize) -> Option<crate::backdrop::plan::Rect> {
        let (mut left, mut right) = (self.first + start, self.first + end);
        let (mut a, mut b) = (None, None);
        while left < right {
            if left & 1 != 0 {
                a = Self::join(a, self.tree[left]);
                left += 1;
            }
            if right & 1 != 0 {
                right -= 1;
                b = Self::join(self.tree[right], b);
            }
            left /= 2;
            right /= 2;
        }
        Self::join(a, b).map(crate::layer_blur::shadow::ItemExtents::rect)
    }
}

/// The page/capture bounds executor reduces body/text extrema first, then exceptional rectangles
/// in painter order. The two columns deliberately retain that order instead of regrouping floats.
#[derive(Default)]
struct Bounds {
    raw: OutputBounds,
    extra: BTreeMap<usize, Rect>,
    items: Vec<Option<Rect>>,
}
impl Bounds {
    fn read(items: &[DrawItem]) -> Option<Self> {
        if !items.iter().all(|item| {
            matches!(
                item,
                DrawItem::Body { .. }
                    | DrawItem::Text(_)
                    | DrawItem::BackdropSplit(_)
                    | DrawItem::LayerBlurComposite { .. }
            )
        }) {
            return None;
        }
        let first = items.len().max(1).next_power_of_two();
        let mut raw = OutputBounds {
            first,
            tree: vec![None; first * 2],
        };
        let mut extra = BTreeMap::new();
        let mut rects = Vec::with_capacity(items.len());
        for (at, item) in items.iter().enumerate() {
            raw.tree[first + at] =
                crate::layer_blur::shadow::item_extents(std::slice::from_ref(item));
            let rect = if matches!(item, DrawItem::Body { .. } | DrawItem::Text(_)) {
                raw.tree[first + at].map(crate::layer_blur::shadow::ItemExtents::rect)
            } else {
                crate::layer_blur::shadow::all_items_canvas_bounds(std::slice::from_ref(item))
            };
            rects.push(rect);
            if matches!(item, DrawItem::LayerBlurComposite { .. }) {
                if let Some(rect) = rect.filter(Rect::finite) {
                    extra.insert(at, rect);
                }
            }
        }
        for at in (1..first).rev() {
            raw.tree[at] = OutputBounds::join(raw.tree[at * 2], raw.tree[at * 2 + 1]);
        }
        Some(Self {
            raw,
            extra,
            items: rects,
        })
    }
    fn range(&self, start: usize, end: usize) -> Option<Rect> {
        self.extra
            .range(start..end)
            .fold(self.raw.range(start, end), |value, (_, &rect)| {
                union(value, Some(rect))
            })
    }
    fn write(&mut self, slot: usize, item: &DrawItem) {
        self.raw.write(slot, item);
        let rect = crate::layer_blur::shadow::all_items_canvas_bounds(std::slice::from_ref(item));
        self.items[slot] = rect;
        if matches!(item, DrawItem::LayerBlurComposite { .. }) {
            if let Some(rect) = rect.filter(Rect::finite) {
                self.extra.insert(slot, rect);
            } else {
                self.extra.remove(&slot);
            }
        }
    }
    fn capture_extent(&mut self, slot: usize, extent: [f64; 4]) {
        let [x, y, w, h] = extent;
        let rect = Rect { x, y, w, h };
        self.items[slot] = Some(rect);
        if rect.finite() {
            self.extra.insert(slot, rect);
        } else {
            self.extra.remove(&slot);
        }
    }
}

fn container<'a>(items: &'a [DrawItem], path: &[usize]) -> Option<&'a [DrawItem]> {
    match path.split_first() {
        None => Some(items),
        Some((&at, tail)) => {
            let DrawItem::LayerBlurComposite { group, .. } = items.get(at)? else {
                return None;
            };
            container(&group.items, tail)
        }
    }
}
fn container_mut<'a>(items: &'a mut [DrawItem], path: &[usize]) -> Option<&'a mut [DrawItem]> {
    match path.split_first() {
        None => Some(items),
        Some((&at, tail)) => {
            let DrawItem::LayerBlurComposite { group, .. } = items.get_mut(at)? else {
                return None;
            };
            container_mut(&mut Arc::make_mut(group).items, tail)
        }
    }
}

struct Leaf {
    inputs: Vec<(u32, u8)>,
    row: u32,
    primary: usize,
    container: usize,
    slots: Vec<usize>,
}
struct Composition {
    recipe: crate::frame::CompositionRecipe,
    container: usize,
    slots: [usize; 2],
    /// Capture-local isolation has its own ordered bounds reduction.
    isolation: Option<OrderedBounds>,
    isolation_inputs: Vec<usize>,
    isolation_slots: HashMap<usize, usize>,
    pending_bounds: Vec<(usize, Option<Rect>)>,
}
struct Capture {
    recipe: crate::frame::CaptureRecipe,
    container: usize,
    parent: usize,
    slot: usize,
}
struct OutputContainer {
    path: Address,
    bounds: Bounds,
    /// Nearest capture's source row, if this is a captured item list.
    root: Option<u32>,
    users: Vec<Vec<usize>>,
    compositions: Vec<usize>,
}

/// A partition of ordinary output into leaf records and the operations that consume them.
/// The graph owns no copies of output vectors or capture pictures.
pub(super) struct OutputRecords {
    at: u64,
    valid: bool,
    containers: Vec<OutputContainer>,
    leaves: Vec<Leaf>,
    compositions: Vec<Composition>,
    captures: Vec<Capture>,
    inputs: Vec<(u32, SourceEdges)>,
    capture_owner: Vec<Option<usize>>,
    isolation_parents: HashMap<usize, Vec<(usize, usize)>>,
}

#[derive(Default)]
struct SourceEdges {
    readers: [Vec<usize>; 3],
    held: u8,
    hidden: bool,
    compositions: Vec<usize>,
}

struct Sources<'a> {
    primary: HashMap<u32, usize>,
    compositions: HashMap<u32, &'a crate::frame::CompositionRecipe>,
    captures: HashMap<u32, &'a crate::frame::CaptureRecipe>,
}

impl OutputRecords {
    pub(super) fn capture_bounds(&self, row: u32) -> Option<Rect> {
        self.captures.iter().find(|c| self.valid && c.recipe.row == row).map(|c| c.recipe.bounds)
    }

    #[cfg(test)]
    pub(super) fn leaf_storage(&self) -> usize { self.leaves.as_ptr() as usize }

    /// No emitted operation reads this source, including through a prepared child's ancestors.
    pub(super) fn ignores_source(&self, row: u32) -> bool {
        self.valid && self.input(row).is_none_or(|edges| edges.held == 0 && edges.compositions.is_empty()
            && edges.readers.iter().all(Vec::is_empty))
    }

    fn input(&self, row: u32) -> Option<&SourceEdges> {
        self.inputs.binary_search_by_key(&row, |(source, _)| *source).ok().map(|i| &self.inputs[i].1)
    }

    pub(super) fn read(
        scene: &Scene,
        dl: &[moda_scene::types::PaintItem],
        unit: &crate::frame::PlanUnit,
        items: &[DrawItem],
        prepared: Option<&crate::prepared_rows::PreparedRows>,
    ) -> Option<Self> {
        if !unit.details().skipped.is_empty()
            || !unit.details().dropped.is_empty()
            || unit.culled
            || unit.part_culled
        {
            return None;
        }
        let prepared = prepared?;
        if unit.compositions.is_empty()
            && unit.details().captures.is_empty()
            && !items
                .first()
                .and_then(drawitem_node_id)
                .and_then(|id| scene.row_of(id))
                .is_some_and(|row| prepared.output(row).is_some())
        {
            return None;
        }
        let mut graph = Self {
            at: scene.stamps().clock(),
            valid: true,
            containers: Vec::new(),
            leaves: Vec::new(),
            compositions: Vec::new(),
            captures: Vec::new(),
            inputs: Vec::new(),
            capture_owner: Vec::new(),
            isolation_parents: HashMap::default(),
        };
        let sources = Sources {
            primary: dl
                .iter()
                .enumerate()
                .filter(|(_, item)| item.is_primary_node_item)
                .filter_map(|(index, item)| {
                    scene
                        .row_of(&item.node_id)
                        .map(|row| (row, unit.dl_start + index))
                })
                .collect(),
            compositions: unit
                .compositions
                .iter()
                .map(|recipe| (recipe.row, recipe))
                .collect(),
            captures: unit
                .details().captures
                .iter()
                .map(|recipe| (recipe.row, recipe))
                .collect(),
        };
        let mut edges = BTreeMap::<u32, SourceEdges>::new();
        graph.read_container(scene, &sources, &mut edges, items, Vec::new(), None, prepared)?;
        let compositions: HashMap<_, _> = unit.compositions.iter().map(|r| (r.row, r)).collect();
        let captures: std::collections::HashSet<_> = unit.details().captures.iter().map(|r| r.row).collect();
        let leaves: std::collections::HashSet<_> = graph.leaves.iter().map(|r| r.row).collect();
        for primary in dl.iter().filter(|item| item.is_primary_node_item) {
            let row = scene.row_of(&primary.node_id)?;
            if scene.hidden_from_paint_order(&primary.node_id) { edges.entry(row).or_default().hidden = true; continue; }
            if leaves.contains(&row) {
                continue;
            }
            if !compositions.contains_key(&row)
                && !captures.contains(&row)
                && !crate::frame::boundary_has_no_body(primary)
            {
                return None;
            }
            let mut source = row;
            loop {
                let groups = if compositions
                    .get(&source)
                    .is_some_and(|r| r.separates_paint())
                {
                    0b1101
                } else {
                    0b1111
                };
                edges.entry(source).or_default().held |= groups;
                let Some(parent) = scene.parent_map().get(scene.row(source)?.id()) else {
                    break;
                };
                let Some(parent) = scene.row_of(parent) else {
                    break;
                };
                source = parent;
            }
        }
        graph.capture_owner = vec![None; graph.containers.len()];
        for (index, capture) in graph.captures.iter().enumerate() {
            graph.capture_owner[capture.container] = Some(index);
        }
        for (index, composition) in graph.compositions.iter().enumerate() {
            edges.entry(composition.recipe.row).or_default().compositions.push(index);
            let container = &mut graph.containers[composition.container];
            container.compositions.push(index);
            for slot in composition.slots[0] + 1..composition.slots[1] {
                container.users[slot].push(index);
            }
            for (input, &token) in composition.isolation_inputs.iter().enumerate() {
                if token >= container.bounds.items.len() {
                    graph
                        .isolation_parents
                        .entry(token - container.bounds.items.len())
                        .or_default()
                        .push((index, input));
                }
            }
        }
        graph.inputs = edges.into_iter().collect();
        Some(graph)
    }
    fn read_container(
        &mut self,
        scene: &Scene,
        sources: &Sources<'_>,
        edges: &mut BTreeMap<u32, SourceEdges>,
        items: &[DrawItem],
        path: Address,
        root: Option<u32>,
        prepared: &crate::prepared_rows::PreparedRows,
    ) -> Option<usize> {
        use crate::seams::BackdropOp;
        let ci = self.containers.len();
        self.containers.push(OutputContainer {
            path: path.clone(),
            bounds: Bounds::default(),
            root,
            users: vec![Vec::new(); items.len()],
            compositions: Vec::new(),
        });
        let mut leaves: HashMap<u32, Vec<usize>> = HashMap::default();
        let mut pairs: HashMap<u32, [usize; 2]> = HashMap::default();
        for (slot, item) in items.iter().enumerate() {
            let row = scene.row_of(drawitem_node_id(item)?)?;
            match item {
                DrawItem::BackdropSplit(split) => {
                    let side = match split.op {
                        BackdropOp::BlendBegin | BackdropOp::GroupBegin => 0,
                        BackdropOp::BlendEnd { .. } | BackdropOp::StackEnd { .. } => 1,
                        _ => return None,
                    };
                    let pair = pairs.entry(row).or_insert([usize::MAX; 2]);
                    if pair[side] != usize::MAX {
                        return None;
                    }
                    pair[side] = slot;
                }
                DrawItem::LayerBlurComposite {
                    group, ..
                } if sources.captures.contains_key(&row) => {
                    if !group.coverage.is_empty()
                        || group.filters.shadows_alone
                    {
                        return None;
                    }
                    let recipe = (*sources.captures.get(&row)?).clone();
                    let mut child_path = path.clone();
                    child_path.push(slot);
                    let child = self.read_container(
                        scene,
                        sources,
                        edges,
                        &group.items,
                        child_path,
                        Some(row),
                        prepared,
                    )?;
                    if rect_bits(self.containers[child].bounds.range(0, group.items.len()))
                        != rect_bits(Some(recipe.bounds))
                    {
                        return None;
                    }
                    self.captures.push(Capture {
                        recipe,
                        container: child,
                        parent: ci,
                        slot,
                    });
                }
                _ => leaves.entry(row).or_default().push(slot),
            }
        }
        for (row, slots) in leaves {
            let (source, inputs) = prepared.output(row)?;
            if source.len() != slots.len() { return None; }
            // Mask identities belong to the capture's namespace only after normalization below.
            let mut normalized = source.to_vec();
            if let Some(root) = root {
                let recipe = sources.captures.get(&root)?;
                crate::frame::capture_child_masks(&mut normalized, &recipe.strip_masks, &recipe.mask_table);
                crate::frame::capture_child_clips(scene, &mut normalized, scene.row(root)?.id());
            }
            if !normalized
                .iter()
                .zip(&slots)
                .all(|(item, &slot)| same_output(item, &items[slot]))
            {
                return None;
            }
            let primary = *sources.primary.get(&row)?;
            let index = self.leaves.len();
            self.leaves.push(Leaf {
                inputs: inputs.to_vec(),
                row,
                primary,
                container: ci,
                slots,
            });
            for &(source, groups) in inputs {
                for group in 0..3 {
                    if groups & (1 << group) != 0 {
                        edges.entry(source).or_default().readers[group].push(index);
                    }
                }
            }
        }
        // Admit source outputs before deriving their bounds. A refused output graph must not
        // assemble a bounds column that the ordinary range executor will immediately discard.
        self.containers[ci].bounds = Bounds::read(items)?;
        // Closing-order recipes are a topological order for nested isolation bounds.
        let mut pairs: Vec<_> = pairs.into_iter().collect();
        pairs.sort_by_key(|(_, slots)| slots[1]);
        let mut by_start: HashMap<usize, usize> = HashMap::default();
        for (row, slots) in pairs {
            if slots.contains(&usize::MAX) || slots[0] >= slots[1] {
                return None;
            }
            let recipe = (*sources.compositions.get(&row)?).clone();
            if !recipe.bounds_match(self.containers[ci].bounds.range(slots[0] + 1, slots[1])) {
                return None;
            }
            let (isolation, isolation_inputs) = if let Some(root) = root {
                let recorded = sources
                    .captures
                    .get(&root)?
                    .isolations
                    .iter()
                    .find(|(pair, _)| *pair == slots)?
                    .1;
                // A nested pair contributes its already-reduced bounds to the enclosing pair.
                let mut values = Vec::new();
                let mut inputs = Vec::new();
                let mut at = slots[0] + 1;
                while at < slots[1] {
                    if let Some(&index) = by_start.get(&at) {
                        let child = &self.compositions[index];
                        values.push(child.isolation.as_ref()?.value());
                        inputs.push(items.len() + index);
                        at = child.slots[1] + 1;
                    } else {
                        values.push(self.containers[ci].bounds.items[at]);
                        inputs.push(at);
                        at += 1;
                    }
                }
                let bounds = OrderedBounds::read(values);
                if rect_bits(bounds.value()) != rect_bits(recorded) {
                    return None;
                }
                (Some(bounds), inputs)
            } else {
                (None, Vec::new())
            };
            let isolation_slots = isolation_inputs
                .iter()
                .enumerate()
                .filter(|(_, token)| **token < items.len())
                .map(|(input, &slot)| (slot, input))
                .collect();
            by_start.insert(slots[0], self.compositions.len());
            self.compositions.push(Composition {
                recipe,
                container: ci,
                slots,
                isolation,
                isolation_inputs,
                isolation_slots,
                pending_bounds: Vec::new(),
            });
        }
        Some(ci)
    }
}

impl OutputRecords {
    fn reach_captures(&self, mut container: usize, dirty: &mut BTreeSet<usize>) {
        while let Some(capture) = self.capture_owner[container] {
            if !dirty.insert(capture) {
                break;
            }
            container = self.captures[capture].parent;
        }
    }
    fn reach_slot(container: &OutputContainer, outputs: &mut [Composition], slot: usize, dirty: &mut BTreeSet<usize>) {
        for &index in &container.users[slot] {
            dirty.insert(index);
            let output = &mut outputs[index];
            if let Some(&input) = output.isolation_slots.get(&slot) {
                output.pending_bounds.push((input, container.bounds.items[slot]));
            }
        }
    }
    pub(super) fn patch(
        &mut self,
        scene: &Scene,
        changed: &[u32],
        plan: &mut Arc<CoveragePlan>,
        start: usize,
        seg: &mut crate::frame::CameraSegment,
        written: &mut PlanWritten,
        opts: &RenderOptions,
        caches: &mut PlanCaches,
        hashes: &dyn NodeHashSource,
    ) -> Option<i64> {
        use moda_scene::stamps::Group;
        if !self.valid {
            return None;
        }
        let mut leaves = Vec::new();
        let mut compositions = BTreeSet::new();
        let mut captures = BTreeSet::new();
        if scene.stamps().latest(Group::Structure) > self.at &&
            changed.iter().any(|&row| !scene.stamps().current(row, Group::Structure, self.at)) {
            return None;
        }
        let mut visit = |row: u32, edges: &SourceEdges| -> Option<()> {
            if edges.hidden && !scene.hidden_from_paint_order(scene.row(row)?.id()) { return None; }
            for group in [Group::Placement, Group::Paint, Group::Content] {
                if scene.stamps().current(row, group, self.at) { continue; }
                if edges.held & (1 << group as u8) != 0 { return None; }
                for &index in &edges.readers[group as usize] { leaves.push((index, group != Group::Placement)); }
                compositions.extend(edges.compositions.iter().copied());
            }
            Some(())
        };
        // A page-wide fade may reach thousands of independent records. Intersect each record's
        // inputs with the changed rows, starting with the shorter of the two ordered slices.
        if changed.len() < self.inputs.len() {
            for &row in changed {
                if let Some(edges) = self.input(row) { visit(row, edges)?; }
            }
        } else {
            for (row, edges) in &self.inputs {
                if changed.binary_search(row).is_ok() { visit(*row, edges)?; }
            }
        }
        if leaves.is_empty() && compositions.is_empty() {
            self.at = scene.stamps().clock();
            return Some(0);
        }
        for &index in &compositions {
            self.reach_captures(self.compositions[index].container, &mut captures);
        }
        self.valid = false;
        let root_len = self.containers[0].bounds.items.len();
        let mut updates = Vec::new();
        let mut writes = Vec::new();
        let mut delta = 0i64;
        leaves.sort_unstable_by_key(|&(index, _)| index);
        leaves.dedup_by(|later, earlier| {
            if later.0 != earlier.0 { return false; }
            earlier.1 |= later.1;
            true
        });
        let prepared = seg.prepared.as_ref()?;
        for (index, prepare) in leaves {
            let leaf = &self.leaves[index];
            let mut items = match (!prepare)
                .then(|| prepared.placed_output(scene, leaf.row))
                .flatten()
            {
                Some(items) => items,
                None => {
                    let (items, update) = prepared.reprepare(
                        scene,
                        leaf.row,
                        seg.dl.items.get(leaf.primary)?,
                        plan,
                        seg,
                        opts,
                        caches,
                        hashes,
                    )?;
                    updates.push((leaf.row, update));
                    items
                }
            };
            let ci = leaf.container;
            if let Some(root) = self.containers[ci].root {
                let capture = self.captures.get(self.capture_owner[ci]?)?;
                crate::frame::capture_child_masks(&mut items, &capture.recipe.strip_masks, &capture.recipe.mask_table);
                crate::frame::capture_child_clips(scene, &mut items, scene.row(root)?.id());
            }
            if items.len() != leaf.slots.len() {
                return None;
            }
            let old = container(
                plan.items.get(start..start + root_len)?,
                &self.containers[ci].path,
            )?;
            for (&slot, item) in leaf.slots.iter().zip(items) {
                if !items_slots_match(
                    std::slice::from_ref(&item),
                    std::slice::from_ref(old.get(slot)?),
                ) {
                    return None;
                }
                delta += items_bytes(std::slice::from_ref(&item)) as i64
                    - items_bytes(std::slice::from_ref(&old[slot])) as i64;
                self.containers[ci].bounds.write(slot, &item);
                Self::reach_slot(&self.containers[ci], &mut self.compositions, slot, &mut compositions);
                writes.push((ci, slot, item));
            }
            self.reach_captures(ci, &mut captures);
        }
        let mut capture_updates = Vec::new();
        while let Some(index) = captures.pop_first() {
            let capture = &self.captures[index];
            let ci = capture.container;
            let bounds = self.containers[ci]
                .bounds
                .range(0, self.containers[ci].bounds.items.len())?;
            let params = if rect_bits(Some(bounds)) != rect_bits(Some(capture.recipe.bounds)) {
                let context = crate::backdrop::PlanCtx::new(
                    scene,
                    plan.width_px,
                    plan.height_px,
                    plan.camera[0],
                    plan.camera[1],
                    plan.page_size[0],
                    plan.page_size[1],
                    seg.page_clip,
                )
                .at_shader_time(opts.shader_time_s);
                let mut skipped = Vec::new();
                let params = crate::layer_blur::plan::compute_subtree_layer_blur_params(
                    scene,
                    scene.row(capture.recipe.row)?,
                    &context,
                    bounds,
                    &mut skipped,
                )?;
                if !skipped.is_empty() {
                    return None;
                }
                self.containers[capture.parent]
                    .bounds
                    .capture_extent(capture.slot, params.extent_canvas);
                Self::reach_slot(
                    &self.containers[capture.parent], &mut self.compositions,
                    capture.slot,
                    &mut compositions,
                );
                // A capture's target rectangle is an input of every isolation output inside it.
                compositions.extend(self.containers[ci].compositions.iter().copied());
                Some(params)
            } else {
                None
            };
            self.reach_captures(capture.parent, &mut captures);
            // Captures are processed innermost first, in their recorded index order.
            match capture_updates.binary_search_by_key(&index, |(key, _)| *key) {
                Ok(at) => capture_updates[at] = (index, (bounds, params)),
                Err(at) => capture_updates.insert(at, (index, (bounds, params))),
            }
        }
        if !compositions.is_empty() && seg.backdrop.is_none() {
            return None;
        }
        let context = crate::backdrop::PlanCtx::new(
            scene,
            plan.width_px,
            plan.height_px,
            plan.camera[0],
            plan.camera[1],
            plan.page_size[0],
            plan.page_size[1],
            seg.page_clip,
        )
        .over_transparent(seg.node_transparent)
        .keeping_shape_sdf(seg.backdrop.unwrap_or(false));
        let mut recipes = Vec::new();
        while let Some(index) = compositions.pop_first() {
            let output = &mut self.compositions[index];
            let ci = output.container;
            let bounds = self.containers[ci]
                .bounds
                .range(output.slots[0] + 1, output.slots[1]);
            let recipe = output.recipe.with_bounds(bounds);
            let mut items = recipe.emit(scene, &context)?;
            let mut pending_parent_value = None;
            if let Some(isolation) = &mut output.isolation {
                let old = rect_bits(isolation.value());
                isolation.write(output.pending_bounds.drain(..));
                let value = isolation.value();
                if rect_bits(value) != old {
                    pending_parent_value = Some(value);
                }
                let capture_index = self.capture_owner[ci]?;
                let capture = &self.captures[capture_index];
                let updated = capture_updates.binary_search_by_key(&capture_index, |(key, _)| *key).ok();
                let target = if let Some((_, (_, Some(params)))) = updated.map(|at| &capture_updates[at]) {
                    (params.region_canvas, params.width_px, params.height_px)
                } else {
                    let parent = container(
                        plan.items.get(start..start + root_len)?,
                        &self.containers[capture.parent].path,
                    )?;
                    let DrawItem::LayerBlurComposite { group, .. } = parent.get(capture.slot)?
                    else {
                        return None;
                    };
                    (group.region_canvas, group.width_px, group.height_px)
                };
                let rect = crate::backdrop::blend::isolation_rect(
                    isolation.value(),
                    target.0,
                    target.1,
                    target.2,
                );
                for item in &mut items {
                    let DrawItem::BackdropSplit(split) = item else {
                        return None;
                    };
                    **split = crate::backdrop::blend::isolation_split(
                        &split.node_id,
                        split.op.clone(),
                        rect,
                    );
                }
            }
            let old = container(
                plan.items.get(start..start + root_len)?,
                &self.containers[ci].path,
            )?;
            for (&slot, item) in output.slots.iter().zip(items) {
                if !items_slots_match(
                    std::slice::from_ref(&item),
                    std::slice::from_ref(old.get(slot)?),
                ) {
                    return None;
                }
                writes.push((ci, slot, item));
            }
            if let Some(value) = pending_parent_value {
                for &(parent, input) in self.isolation_parents.get(&index).into_iter().flatten() {
                    self.compositions[parent].pending_bounds.push((input, value));
                    compositions.insert(parent);
                }
            }
            recipes.push((index, recipe));
        }
        if !updates.is_empty() {
            let prepared = seg.prepared.as_mut()?;
            let before = prepared.bytes();
            prepared.update(updates);
            delta += prepared.bytes() as i64 - before as i64;
        }
        let dest = Arc::make_mut(plan);
        let mut nested = BTreeSet::new();
        for (ci, slot, item) in writes {
            let path = &self.containers[ci].path;
            let items = container_mut(dest.items.get_mut(start..start + root_len)?, path)?;
            let old = std::mem::replace(items.get_mut(slot)?, item);
            if path.is_empty() {
                written.replaced.push(((start + slot) as u32, old));
            } else {
                nested.insert(path[0]);
            }
        }
        for (index, (bounds, params)) in capture_updates {
            let capture = &mut self.captures[index];
            let path = &self.containers[capture.parent].path;
            let items = container_mut(dest.items.get_mut(start..start + root_len)?, path)?;
            let DrawItem::LayerBlurComposite { group, .. } = items.get_mut(capture.slot)? else {
                return None;
            };
            let group = Arc::make_mut(group);
            if let Some(params) = params {
                group.region_canvas = params.region_canvas;
                group.extent_canvas = params.extent_canvas;
                group.reach_canvas = params.reach_canvas;
                group.width_px = params.width_px;
                group.height_px = params.height_px;
                group.filters = crate::seams::LayerFilters {
                    composite_blend: group.filters.composite_blend,
                    shadows_alone: group.filters.shadows_alone,
                    ..params.filters
                };
            }
            group.cache_key = None;
            capture.recipe.bounds = bounds;
            nested.insert(path.first().copied().unwrap_or(capture.slot));
        }
        for slot in nested {
            written
                .replaced
                .push(((start + slot) as u32, DrawItem::Vacant));
        }
        for (index, recipe) in recipes {
            self.compositions[index].recipe = recipe;
        }
        self.at = scene.stamps().clock();
        self.valid = true;
        Some(delta)
    }
}

impl OutputRecords {
    pub(super) fn bytes(&self) -> usize {
        use super::weights::capacity;
        use std::mem::size_of;
        capacity(&self.containers) + self.containers.iter().map(|c| {
            capacity(&c.path) + capacity(&c.bounds.raw.tree) + capacity(&c.bounds.items)
                + c.bounds.extra.len() * (size_of::<(usize, Rect)>() + 3 * size_of::<usize>())
                + capacity(&c.users) + c.users.iter().map(capacity).sum::<usize>() + capacity(&c.compositions)
        }).sum::<usize>()
            + capacity(&self.leaves) + self.leaves.iter().map(|r| capacity(&r.slots) + capacity(&r.inputs)).sum::<usize>()
            + capacity(&self.compositions) + self.compositions.iter().map(|r| {
                capacity(&r.isolation_inputs) + capacity(&r.pending_bounds)
                    + r.isolation_slots.capacity() * (size_of::<(usize, usize)>() + 1)
                    + r.isolation.as_ref().map_or(0, |b| capacity(&b.inputs) + capacity(&b.prefixes))
            }).sum::<usize>()
            + capacity(&self.captures) + self.captures.iter().map(|c| {
                capacity(&c.recipe.strip_masks) + capacity(&c.recipe.mask_table) + capacity(&c.recipe.isolations)
            }).sum::<usize>()
            + capacity(&self.capture_owner) + capacity(&self.inputs)
            + self.inputs.iter().map(|(_, edges)| capacity(&edges.compositions)
                + edges.readers.iter().map(capacity).sum::<usize>()).sum::<usize>()
            + self.isolation_parents.capacity() * (size_of::<(usize, Vec<(usize, usize)>)>() + 1)
            + self.isolation_parents.values().map(capacity).sum::<usize>()
    }
}

impl OutputRecords {
    /// Ordinary range execution changed values, but standing source membership retains its reverse edges.
    pub(super) fn refresh(
        &mut self,
        scene: &Scene,
        unit: &crate::frame::PlanUnit,
        items: &[DrawItem],
        prepared: Option<&crate::prepared_rows::PreparedRows>,
    ) -> bool {
        use moda_scene::stamps::Group;
        self.valid = false;
        if scene.stamps().latest(Group::Structure) > self.at
            || !unit.details().skipped.is_empty()
            || !unit.details().dropped.is_empty()
            || unit.culled
            || unit.part_culled
            || unit.compositions.len() != self.compositions.len()
            || unit.details().captures.len() != self.captures.len()
        {
            return false;
        }
        let Some(prepared) = prepared else { return false; };
        // Ordinary emission supplies the actual dependency membership. Content writes elsewhere,
        // or new values with these same reads, do not invalidate the reverse index.
        for leaf in &self.leaves {
            let Some((output, inputs)) = prepared.output(leaf.row) else { return false; };
            if inputs != leaf.inputs || output.len() != leaf.slots.len() { return false; }
        }
        for output in &mut self.containers {
            let Some(items) = container(items, &output.path) else {
                return false;
            };
            if items.len() != output.bounds.items.len() {
                return false;
            }
            let Some(bounds) = Bounds::read(items) else {
                return false;
            };
            output.bounds = bounds;
        }
        for leaf in &self.leaves {
            let Some(items) = container(items, &self.containers[leaf.container].path) else {
                return false;
            };
            if !leaf.slots.iter().all(|&slot| {
                items
                    .get(slot)
                    .and_then(drawitem_node_id)
                    .and_then(|id| scene.row_of(id))
                    == Some(leaf.row)
            }) {
                return false;
            }
        }
        for capture in &mut self.captures {
            let Some(recipe) = unit.details().captures.iter().find(|r| r.row == capture.recipe.row) else {
                return false;
            };
            let column = &self.containers[capture.container].bounds;
            if rect_bits(column.range(0, column.items.len())) != rect_bits(Some(recipe.bounds)) {
                return false;
            }
            capture.recipe = recipe.clone();
        }
        for index in 0..self.compositions.len() {
            let output = &self.compositions[index];
            let Some(recipe) = unit
                .compositions
                .iter()
                .find(|r| r.row == output.recipe.row)
            else {
                return false;
            };
            let column = &self.containers[output.container].bounds;
            if !recipe.same_inputs(&output.recipe)
                || !recipe.bounds_match(column.range(output.slots[0] + 1, output.slots[1]))
            {
                return false;
            }
            let isolation = if output.isolation.is_some() {
                let Some(capture) = self.capture_owner[output.container] else {
                    return false;
                };
                let Some((_, recorded)) = self.captures[capture]
                    .recipe
                    .isolations
                    .iter()
                    .find(|(slots, _)| *slots == output.slots)
                else {
                    return false;
                };
                let mut values = Vec::with_capacity(output.isolation_inputs.len());
                for &input in &output.isolation_inputs {
                    values.push(if input < column.items.len() {
                        column.items[input]
                    } else {
                        let Some(child) = self
                            .compositions
                            .get(input - column.items.len())
                            .and_then(|r| r.isolation.as_ref())
                        else {
                            return false;
                        };
                        child.value()
                    });
                }
                let bounds = OrderedBounds::read(values);
                if rect_bits(bounds.value()) != rect_bits(*recorded) {
                    return false;
                }
                Some(bounds)
            } else {
                None
            };
            self.compositions[index].recipe = recipe.clone();
            self.compositions[index].isolation = isolation;
            self.compositions[index].pending_bounds.clear();
        }
        self.at = scene.stamps().clock();
        self.valid = true;
        true
    }
}

impl OutputRecords {
    #[cfg(test)]
    pub(super) fn has_leaves(&self) -> bool {
        !self.leaves.is_empty()
    }
}
