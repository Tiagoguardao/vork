//! Node outputs are derived rows, valid under the scene groups they read and their capture context.
//! A changed placement may follow the ordinary emission recipe; content and paint still come from emission.
use moda_scene::{Scene, stamps::Group};
use crate::{effects::PlanMasks, seams::DrawItem, residency::weights::{Allocations, capacity, RetainedMap, RetainedSet}, frame::TranslationPlacement};

#[derive(Debug)]
pub(crate) struct PreparedRows {
    rows: RetainedMap<u32, Prepared>, owned: usize, budget: usize,
    /// An emission result for this range, not a validity rule: these rows returned their existing values.
    kept: RetainedSet<u32>,
    emitted: RetainedSet<u32>,
    pub(crate) reused: u64, pub(crate) placed: u64,
}
impl Clone for PreparedRows {
    fn clone(&self) -> Self {
        let rows = self.rows.clone();
        let owned = rows.values().map(|row| row.inventory(&mut Allocations::owned())).sum();
        Self { rows, owned, budget: self.budget, kept: self.kept.clone(), emitted: self.emitted.clone(),
            reused: self.reused, placed: self.placed }
    }
}
impl Default for PreparedRows {
    fn default() -> Self { Self { rows: Default::default(), owned: 0,
        budget: crate::residency::CAMERA_PLAN_BUDGET_BYTES, kept: Default::default(), emitted: Default::default(), reused: 0, placed: 0 } }
}
#[derive(Debug, Clone)]
pub(crate) struct Prepared {
    at: u64, inputs: Vec<(u32, u8)>, context: Vec<(u32, u8)>,
    z: u64, consumed: u64, items: Vec<DrawItem>, placement: Option<TranslationPlacement>,
    body_mode: Option<bool>,
}
impl Prepared {
    fn inventory(&self, allocations: &mut Allocations) -> usize {
        capacity(&self.inputs) + capacity(&self.context) + allocations.item_vec(&self.items)
    }
}

impl PreparedRows {
    fn tables(&self) -> usize {
        self.rows.allocation_size() + self.kept.allocation_size() + self.emitted.allocation_size()
    }
    pub(crate) fn bytes(&self) -> usize { self.owned + self.tables() }

    pub(crate) fn inventory(&self, allocations: &mut crate::residency::weights::Allocations) -> usize {
        self.tables() + self.rows.values().map(|row| row.inventory(allocations)).sum::<usize>()
    }
    pub(crate) fn shift_source_z(&mut self, row: u32, dz: i64) -> Option<()> {
        let Some(p) = self.rows.get_mut(&row) else { return Some(()); };
        if !p.items.iter().all(|item| crate::camera_slot::z_shiftable(item, dz)) { return None; }
        p.z = p.z.checked_add_signed(dz)?;
        for item in &mut p.items { crate::camera_slot::shift_item_z(item, dz); }
        Some(())
    }

    pub(crate) fn output(&self, row: u32) -> Option<(&[DrawItem], &[(u32, u8)])> {
        let row = self.rows.get(&row)?;
        Some((&row.items, &row.inputs))
    }
    /// Follow the emitted placement recipe without visiting sibling outputs. Geometry and paint retain
    /// their source proofs; the ordinary range executor handles any changed preparation input.
    pub(crate) fn placed_output(&self, scene: &Scene, row: u32) -> Option<Vec<DrawItem>> {
        let p = self.rows.get(&row)?;
        if ![Group::Paint, Group::Content, Group::Structure].into_iter().all(|group|
            p.inputs.iter().all(|&(row, groups)| groups & (1 << group as u8) == 0
                || scene.stamps().current(row, group, p.at))) { return None; }
        let mut recipe = p.placement.clone()?;
        let placed = crate::frame::placed_span_in_capture(scene, &recipe, &p.items, &p.context)?;
        let mut items = p.items.clone();
        crate::frame::place_span(&mut recipe, &placed, &mut items);
        Some(items)
    }
    /// Re-enter ordinary node preparation under the same emitted capture context and z.
    pub(crate) fn reprepare(&self, scene: &Scene, row: u32, primary: &moda_scene::types::PaintItem, plan: &crate::seams::CoveragePlan,
        seg: &crate::frame::CameraSegment, opts: &crate::RenderOptions,
        caches: &mut crate::residency::PlanCaches, hashes: &dyn crate::residency::NodeHashSource)
        -> Option<(Vec<DrawItem>, Prepared)> {
        let old = self.rows.get(&row)?;
        let mut build = crate::frame::rebuild_page_range(scene, plan, seg, std::slice::from_ref(primary), opts,
            Some((caches, hashes)), old.z, Some(Self::default()), &old.context, true, None)?;
        if build.end_z != old.z + old.consumed || !build.mask_warnings.is_empty()
            || build.units.len() != 1 || !build.units[0].details().skipped.is_empty() || !build.units[0].details().dropped.is_empty() { return None; }
        let mut prepared = build.prepared.as_mut()?.rows.remove(&row)?;
        let mut owners = None;
        if !crate::residency::remap_scratch_masks(&mut build.items, &build.masks, &plan.masks, &mut owners)
            || !crate::residency::remap_scratch_masks(&mut prepared.items, &build.masks, &plan.masks, &mut owners) { return None; }
        if prepared.inputs != old.inputs || prepared.context != old.context { return None; }
        Some((build.items, prepared))
    }
    pub(crate) fn update(&mut self, updates: Vec<(u32, Prepared)>) {
        for (row, prepared) in updates { self.insert(row, prepared); }
    }

    fn remove(&mut self, row: u32) {
        if let Some(old) = self.rows.remove(&row) { self.owned -= old.inventory(&mut Allocations::owned()); }
    }

    fn insert(&mut self, row: u32, prepared: Prepared) {
        self.remove(row);
        self.owned += prepared.inventory(&mut Allocations::owned());
        self.rows.insert(row, prepared);
    }
    pub(crate) fn begin_range(&mut self) { self.kept.clear(); self.emitted.clear(); }
    /// Ordinary range emission owns validation; preserve its mapping on retained source outputs too.
    pub(crate) fn remap_emitted_masks(&mut self, mapping: &[usize]) -> bool {
        for row in &self.emitted {
            let Some(record) = self.rows.get_mut(row) else { continue; };
            for item in &mut record.items {
                if let Some(id) = crate::camera_slot::item_mask_id(item) {
                    let Some(&mapped) = mapping.get(id as usize) else { return false; };
                    crate::camera_slot::set_item_mask_id(item, mapped as u64);
                }
            }
        }
        true
    }
    pub(crate) fn kept(&self, row: u32) -> bool { self.kept.contains(&row) }
    pub(crate) fn fit_budget(&mut self, budget: usize) {
        self.budget = budget;
        if self.bytes() > budget {
            self.rows = Default::default();
            self.kept = Default::default();
            self.emitted = Default::default();
            self.owned = 0;
        }
    }
    pub(crate) fn reuse(&mut self, scene: &Scene, row: u32, context: &[(u32, u8)], z: u64,
        body_mode: bool, masks: &mut PlanMasks) -> Option<(Vec<DrawItem>, u64, bool)> {
        let p = self.rows.get_mut(&row)?;
        if p.z != z || p.context != context { return None; }
        if p.body_mode.is_some_and(|read| read != body_mode) { return None; }
        let stands = |group: Group| p.inputs.iter().all(|&(row, groups)|
            groups & (1 << group as u8) == 0 || scene.stamps().current(row, group, p.at));
        // Prove reuse before consulting the mutable mask builder. A declined preparation must not
        // introduce mask records or read edges that ordinary emission would never request.
        if ![Group::Paint, Group::Content, Group::Structure].into_iter().all(stands) { return None; }
        let placed = if !stands(Group::Placement) {
            Some(crate::frame::placed_span_in_capture(scene, p.placement.as_ref()?, &p.items, context)?)
        } else { None };
        // Masks are derived inputs too: a source outside this row's ancestry may have become visible.
        // An empty output cannot consume a mask.
        let node = scene.row(row)?;
        if !p.items.is_empty() && masks.mask_ref_for_node(scene, node.id()).is_some() { return None; }
        if let Some(placed) = placed {
            crate::frame::place_span(p.placement.as_mut()?, &placed, &mut p.items);
            p.at = scene.stamps().clock();
            self.placed += 1;
        } else {
            self.kept.insert(row);
        }
        self.reused += 1;
        Some((p.items.clone(), p.consumed, p.body_mode.is_some()))
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn put(&mut self, scene: &Scene, row: u32, context: &[(u32, u8)], z: u64,
        consumed: u64, items: &[DrawItem], _masks: &PlanMasks, _masks_before: usize, body_mode: Option<bool>,
        placement: impl FnOnce() -> Option<TranslationPlacement>) {
        self.remove(row);
        // Mask values remain owned by the unit's actual read edges and ordinary mask builder.
        // Retain source outputs in that builder's namespace; range completion remaps emitted rows.
        if !scene.placeable(row) { return; }
        let Some(node) = scene.row(row) else { return };
        let mut inputs = Vec::new();
        let mut cursor = node.id();
        loop {
            let Some(r) = scene.row_of(cursor) else { return };
            if !scene.placeable(r) { return; }
            let faded = r != row && context.iter().any(|&(capture, flags)| capture == r && flags & 1 != 0);
            inputs.push((r, 0b1101 | if faded { 0 } else { 0b0010 }));
            let Some(parent) = scene.parent_map().get(cursor) else { break };
            if scene.effective_node(parent).is_none() { break; }
            cursor = parent;
        }
        if self.bytes() >= self.budget { return; }
        let has_mask = items.iter().any(|item| crate::camera_slot::item_mask_id(item).is_some());
        self.insert(row, Prepared { at: scene.stamps().clock(), inputs, context: context.to_vec(),
            z, consumed, items: items.to_vec(), placement: placement(), body_mode });
        if has_mask { self.emitted.insert(row); }
        if self.bytes() > self.budget { self.remove(row); self.emitted.remove(&row); }

    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    #[test]
    fn empty_allocated_tables_are_charged_and_released() {
        let mut prepared = PreparedRows::default();
        prepared.rows.reserve(100_000);
        prepared.kept.reserve(20);
        prepared.emitted.reserve(30);
        let bytes = prepared.inventory(&mut Default::default());
        assert!(bytes >= 100_000 * std::mem::size_of::<Prepared>());
        assert_eq!(prepared.bytes(), bytes);
        prepared.fit_budget(0);
        assert_eq!(prepared.inventory(&mut Default::default()), 0);
    }

    #[test]
    fn stale_preparation_does_not_install_speculative_masks() {
        let mut scene = Scene::load(&json!({"nodes":[
            {"id":"g","type":"group","opacity":0.6,"children":["m","c"]},
            {"id":"m","type":"ellipse","width":60,"height":40,
                "fills":[{"id":"f","paint":{"type":"solid","color":"#ffffff"}}]},
            {"id":"c","type":"path","width":80,"height":80,"closed":true,
                "points":[{"type":"move","x":0,"y":0},{"type":"line","x":80,"y":0},{"type":"line","x":80,"y":80}],
                "fills":[{"id":"f","paint":{"type":"solid","color":"#ff0000"}}]}
        ],"pages":[{"id":"p","width":300,"height":200,"nodeIds":["g"]}]})).unwrap();
        scene.refresh_worlds();
        let (plan, _) = crate::frame::build_coverage_plan(&scene, 0, &crate::RenderOptions::default()).unwrap();
        let items: Vec<_> = plan.items.into_iter().filter(|item|
            matches!(item, DrawItem::Body { node_id, .. } if node_id == "c")).collect();
        assert!(!items.is_empty());
        let row = scene.row_of("c").unwrap();
        let parent = scene.row_of("g").unwrap();
        let context = [(parent, 1)];
        let mut records = PreparedRows::default();
        records.put(&scene, row, &context, 0, 1, &items, &PlanMasks::default(), 0, None, || None);
        assert!(records.output(row).is_some());
        scene.write_row(parent, |fields| fields.set("clipMaskNodeId", json!("m"))).unwrap();
        scene.refresh_worlds();
        let mut masks = PlanMasks::default();
        assert!(records.reuse(&scene, row, &context, 0, true, &mut masks).is_none());
        assert!(masks.masks.is_empty(), "declined reuse must leave ordinary emission's mask table untouched");
        assert!(masks.mask_ref_for_node(&scene, "c").is_some(), "the fixture must have a newly required mask");
    }
}
