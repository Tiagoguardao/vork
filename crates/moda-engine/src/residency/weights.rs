//! Owned CPU allocations reachable from a retained plan. During fitting, remove
//! references from the same inventory that weighed the dense output. Evicting
//! one instance frees its geometry only when the last retained instance leaves.
use super::*;
use crate::seams::*;
use std::mem::{size_of, size_of_val};

pub(crate) fn capacity<T>(v: &Vec<T>) -> usize { v.capacity() * size_of::<T>() }

pub(crate) type RetainedMap<K, V> = hashbrown::HashMap<K, V, rustc_hash::FxBuildHasher>;
pub(crate) type RetainedSet<K> = hashbrown::HashSet<K, rustc_hash::FxBuildHasher>;

pub(super) fn names(v: &Vec<String>) -> usize {
    capacity(v) + v.iter().map(String::capacity).sum::<usize>()
}

pub(super) fn set(v: &RetainedSet<String>) -> usize {
    v.allocation_size() + v.iter().map(String::capacity).sum::<usize>()
}

#[derive(Default)]
enum Shared { #[default] Unique, PerReference, Excluded }

#[derive(Default)]
pub(crate) struct Allocations {
    refs: HashMap<usize, usize>,
    shared: Shared,
    pub removing: bool,
}

impl Allocations {
    pub(super) fn per_reference() -> Self { Self { shared: Shared::PerReference, ..Default::default() } }
    pub(crate) fn owned() -> Self { Self { shared: Shared::Excluded, ..Default::default() } }

    fn charge(&mut self, ptr: usize) -> bool {
        match self.shared {
            Shared::PerReference => return true,
            Shared::Excluded => return false,
            Shared::Unique => {}
        }
        if self.removing {
            let n = self.refs.get_mut(&ptr).expect("allocation measured before eviction");
            *n -= 1;
            if *n != 0 { return false; }
            self.refs.remove(&ptr);
            true
        } else {
            let n = self.refs.entry(ptr).or_default();
            *n += 1;
            *n == 1
        }
    }

    fn shared<T>(&mut self, arc: &Arc<T>, heap: impl FnOnce(&mut Self, &T) -> usize) -> usize {
        if !self.charge(Arc::as_ptr(arc) as usize) { return 0; }
        size_of::<T>() + 2 * size_of::<usize>() + heap(self, arc)
    }

    fn array<T>(&mut self, v: &Arc<Vec<T>>) -> usize { self.shared(v, |_, v| capacity(v)) }

    pub(super) fn mask_reads(&mut self, cell: &MaskAssetReads) -> usize {
        self.shared(cell, |s, cell| cell.reads.lock().unwrap().as_ref().map_or(0,
            |reads| s.shared(reads, |_, reads| reads.retained_bytes())))
    }

    pub(super) fn source_plan(&mut self, plan: &Arc<CoveragePlan>) -> usize {
        self.shared(plan, |s, p| p.page_id.capacity() + s.masks(&p.masks) + s.item_vec(&p.items)
            + p.item_order.as_ref().map_or(0, crate::plan_items::ItemOrder::bytes))
    }

    pub(super) fn masks(&mut self, masks: &Vec<MaskPlan>) -> usize {
        let mut bytes = capacity(masks);
        for mask in masks {
            bytes += mask.source_node_id.capacity() + mask.owner_node_id.capacity() + capacity(&mask.positions) + capacity(&mask.draws);
            for draw in mask.draws.iter() {
                if let crate::seams::MaskPaint::Program { stroke_uv, .. } = &draw.paint {
                    if !stroke_uv.is_empty() && self.charge(stroke_uv.as_ptr() as usize) {
                        bytes += std::mem::size_of_val(stroke_uv.as_ref()) + 2 * size_of::<usize>();
                    }
                }
            }
            if let Some(image) = &mask.image {
                bytes += image.rgba.capacity() + image.surface.as_ref().map_or(0, String::capacity);
                if let Some(rendered) = &image.rendered {
                    bytes += self.shared(rendered, |s, r| r.node_id.capacity() +
                        r.prepared.lock().unwrap().as_ref().map_or(0, |p| p.bytes(s)) +
                        r.asset_reads.get().map_or(0, |reads| s.mask_reads(reads)) + s.shared(&r.document, |s, doc| doc.retained_heap(&mut |p| s.charge(p))));
                }
            }
            if let Some(text) = &mask.text { bytes += self.shared(&text.raster, |_, r| r.rgba.capacity()); }
        }
        bytes
    }

    fn text(&mut self, batch: &TextQuadBatch) -> usize {
        let mut bytes = batch.node_id.capacity() + capacity(&batch.ops);
        for op in &batch.ops { bytes += capacity(&op.verts) + capacity(&op.indices); }
        if let Some(v) = &batch.vector {
            bytes += self.array(&v.curve_points) + capacity(&v.ops) + capacity(&v.emojis);
            for op in &v.ops { bytes += self.array(&op.tris) + self.array(&op.records); }
            for emoji in &v.emojis {
                bytes += emoji.cluster.capacity();
                if let Some(run) = &emoji.shaped {
                    bytes += self.shared(run, |_, run| run.glyph_bytes());
                }
            }
        }
        if let Some(d) = &batch.degraded { bytes += self.shared(&d.raster, |_, r| r.rgba.capacity()); }
        bytes
    }

    pub(super) fn items(&mut self, items: &[DrawItem]) -> usize {
        let mut bytes = size_of_val(items);
        for item in items {
            bytes += match item {
                DrawItem::Vacant | DrawItem::PageFill { .. } => 0,
                DrawItem::Body { node_id, instance, geometry, paint, .. } => {
                    let payload = match paint {
                        BodyPaint::Solid | BodyPaint::Program(_) => 0,
                        BodyPaint::Gradient(g) => size_of_val(&**g),
                        BodyPaint::Pattern(p) => size_of_val(&**p) + p.texture.key.capacity()
                            + self.array(&p.texture.image.rgba_premultiplied)
                            + p.texture.image.external.as_ref().map_or(0, |e| e.key.capacity()),
                    };
                    node_id.capacity() + size_of_val(&**instance) + payload + self.shared(geometry, |_, g| geometry_heap(g))
                }
                DrawItem::Shadow { node_id, instance, .. } => node_id.capacity() + size_of_val(&**instance),
                DrawItem::SolidShapeSdf { node_id, instance, .. } => node_id.capacity() + size_of_val(&**instance),
                DrawItem::GradientShapeSdf { node_id, instance, .. } => node_id.capacity() + size_of_val(&**instance),
                DrawItem::Text(batch) => self.shared(batch, |s, b| s.text(b)),
                DrawItem::BackdropSplit(b) => {
                    let geometry = match &b.coverage { CoverageSource::Geometry(g) => geometry_heap(g), _ => 0 };
                    let op = match &b.op {
                        BackdropOp::Adjust(a) => size_of_val(&**a) + self.array(&a.lut),
                        BackdropOp::Filter(f) => size_of_val(&**f),
                        _ => 0,
                    };
                    size_of_val(&**b) + b.node_id.capacity() + geometry + op
                }
                DrawItem::LayerBlurComposite { node_id, group, .. } => node_id.capacity() + self.shared(group, |s, g| {
                    let filters = &g.filters;
                    s.item_vec(&g.items) + s.item_vec(&g.coverage) + s.masks(&g.masks) + capacity(&filters.color_matrices)
                        + capacity(&filters.drop_shadows) + capacity(&filters.noise)
                        + filters.shadow.as_ref().map_or(0, |v| capacity(&v.shadows))
                }),
            };
        }
        bytes
    }

    pub(crate) fn item_vec(&mut self, items: &Vec<DrawItem>) -> usize {
        self.items(items) + (items.capacity() - items.len()) * size_of::<DrawItem>()
    }
}

fn geometry_heap(g: &TessellatedGeometry) -> usize {
    capacity(&g.positions) + capacity(&g.vertex_types) + capacity(&g.stroke_uv)
        + g.fill_dropped.as_ref().map_or(0, String::capacity) + g.stroke_dropped.as_ref().map_or(0, String::capacity)
}
