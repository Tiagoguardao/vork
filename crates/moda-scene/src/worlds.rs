//! The scene's derived transform columns: per canonical row (a node's index in `doc.nodes`), its accumulated
//! transform, with the hierarchy as integer arrays the columns are computed over.
//!
//! One rule composes every accumulated transform in the scene, parent first:
//!
//! ```text
//! frame(parent)  = frame(grandparent) ∘ local-as-parent(parent)
//! world(node)    = frame(parent) ∘ local(node)
//! ```
//!
//! ([`Placed::under`]). The columns hold that for the rows whose whole transform ancestry is canonical nodes; the
//! walk by name ([`Scene::accumulated_affine`]) computes the same composition, in the same order, for every other
//! node (an expansion's transients, a node a derived record stands in for). So a matrix read from a column and a
//! matrix walked by name are the same `f64`s, and a reader never needs to know which it got.
//!
//! The columns are derived data under the stamps rule ([`crate::stamps`]): [`Scene::refresh_worlds`] brings them
//! up to the scene's clock, recomputing the hierarchy when a structure stamp is newer than it and otherwise only
//! the subtrees of the rows written since (the scene's own dirty lists: a placement mark flows to descendants
//! through the parent-before-child order). They answer only while the clock stands where they were refreshed;
//! after any later write a reader walks by name until the next refresh. Nothing else invalidates them.

use crate::scene::{is_parent_node, HashMap, Scene};
use crate::stamps::{Group, Stamps, Subscription};
use crate::transform::{self, compose};
use crate::types::PlacementField;
use crate::types::{Affine, NodeId, RawNode, Transform2D};

/// A transform with what [`Transform2D`] accumulates beside the matrix (the rotations summed, the scales
/// multiplied): a node's local transform, or one composed under the transforms above it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Placed {
    pub affine: Affine,
    pub rotation: f64,
    pub scale_x: f64,
    pub scale_y: f64,
}

impl Placed {
    /// The node's own local transform (`calculateAccumulatedTransform`, nodeTransformUtils.ts:251-311).
    pub(crate) fn own(node: &RawNode) -> Placed {
        let (scale_x, scale_y) = node.scale_xy();
        Placed::local(node, scale_x, scale_y)
    }

    /// The node's local transform as the transform parent of what is under it:
    /// `parent.type === 'group' ? parent.scaleX||1 : parent.scale||1` (nodeTransformUtils.ts:288-289). A boolean's
    /// canonical node is a group-like transform parent (E7.1).
    pub(crate) fn parent(node: &RawNode, boolean_source: bool) -> Placed {
        let (scale_x, scale_y) = if node.node_type() == "group" || boolean_source {
            let sx = node.placement_number(PlacementField::ScaleX).unwrap_or(1.0);
            let sy = node.placement_number(PlacementField::ScaleY).unwrap_or(1.0);
            (if sx == 0.0 { 1.0 } else { sx }, if sy == 0.0 { 1.0 } else { sy })
        } else {
            let s = node.scale();
            (s, s)
        };
        Placed::local(node, scale_x, scale_y)
    }

    fn local(node: &RawNode, scale_x: f64, scale_y: f64) -> Placed {
        let [x, y] = [node.placement_number(PlacementField::X).unwrap_or(0.0), node.placement_number(PlacementField::Y).unwrap_or(0.0)];
        let rotation = node.rotation();
        Placed {
            affine: transform::from_trsk(Transform2D { x, y, rotation, scale_x, scale_y }, node.skew_x()),
            rotation,
            scale_x,
            scale_y,
        }
    }

    /// This local transform under the accumulated transform above it: the one composition rule. With nothing
    /// above, the local transform is the accumulated one as it is (it is not multiplied by an identity).
    pub(crate) fn under(self, above: Option<&Placed>) -> Placed {
        match above {
            None => self,
            Some(above) => Placed {
                affine: compose(above.affine, self.affine),
                rotation: above.rotation + self.rotation,
                scale_x: above.scale_x * self.scale_x,
                scale_y: above.scale_y * self.scale_y,
            },
        }
    }

    /// What [`Scene::accumulated_affine`] answers: the matrix and the decomposed transform at its origin.
    pub(crate) fn answer(&self) -> (Affine, Transform2D) {
        let origin = transform::apply_point((0.0, 0.0), self.affine);
        (
            self.affine,
            Transform2D { x: origin.0, y: origin.1, rotation: self.rotation, scale_x: self.scale_x, scale_y: self.scale_y },
        )
    }
}

/// No row: a chain that ends, a row that keeps no frame, a row not yet ordered.
const NONE: u32 = u32::MAX;

/// The derived transform columns of one scene, and the hierarchy by row they are computed over. See the module
/// text for the rule; [`Scene::world_rows`] hands it out while it is current.
#[derive(Default)]
pub struct Worlds {
    /// The scene's dirty lists for these columns: rows whose placement, and rows whose content, was written since
    /// the last refresh (an unrestricted write to a row stamps both, and may have changed any field).
    lists: Option<[Subscription; 2]>,
    /// The scene clock the columns were last brought up to. They answer while the clock stands there.
    current_at: Option<u64>,
    /// The scene clock the hierarchy was read at.
    read_at: Option<u64>,
    /// Whether a derived record stood in for any node when the hierarchy was read: there a content write can
    /// change who a node's transform ancestors are, so the hierarchy also falls with a content stamp.
    derived: bool,

    // The hierarchy, by row.
    /// Rows whose whole transform ancestor chain is canonical nodes: the rows the columns answer for.
    plain: Vec<bool>,
    /// Row `r`'s transform parent: the next link of [`Scene::transform_parent_chain`], as a row, or `NONE` where
    /// the chain ends. One link a row.
    link: Vec<u32>,
    /// The position of the row's page.
    page: Vec<(f64, f64)>,
    /// Every row once, each parent before its children: row `r` and everything under it are
    /// `order[at[r]..end[r]]`, `r` first.
    order: Vec<u32>,
    at: Vec<u32>,
    end: Vec<u32>,
    /// Rows under which every node is a row: no expansion, and no child that is not a canonical node.
    whole: Vec<bool>,
    /// Rows that hold children ([`is_parent_node`]).
    parent: Vec<bool>,
    /// Per row, where its frame sits in `frames`, or `NONE`: plain rows that hold children keep one.
    frame_of: Vec<u32>,

    // The columns.
    /// Per plain row, the node's local transform. Readers compose it with its parent's current frame.
    local: Vec<Placed>,
    /// Whether this row hides its subtree before inherited visibility is applied.
    hidden: Vec<bool>,
    visibility_plain: Vec<bool>,
    /// Per frame-keeping row, its accumulated transform as a transform parent.
    frames: Vec<Placed>,
    frame_hidden: Vec<bool>,
    /// Positions in DFS order of rows whose transform other rows inherit.
    parent_positions: Vec<u32>,

    /// Scratch for a refresh: the written rows' ranges of `order`.
    ranges: Vec<(u32, u32)>,
}

impl Worlds {
    /// Subscribe these columns to the scene's writes. Called once, when the scene is made, before any other
    /// consumer subscribes: a scene that replaces another takes over its subscriptions by position
    /// ([`Stamps::adopt_subscriptions`]), and these are the first of their groups in every scene.
    pub(crate) fn subscribe(&mut self, stamps: &mut Stamps) {
        self.lists = Some([stamps.subscribe(Group::Placement), stamps.subscribe(Group::Content)]);
    }

    /// Whether the columns stand for the scene at `clock`.
    pub(crate) fn current(&self, clock: u64) -> bool {
        self.current_at == Some(clock)
    }

    /// Leaves have no dependent world rows. A translation changes only
    /// the origin of their affine, including when their local basis is skewed.
    pub(crate) fn translate_leaf(&mut self, row: usize, xy: [f64; 2]) -> bool {
        if !self.plain[row] || self.frame_of[row] != NONE
            || self.end[row] - self.at[row] != 1 {
            return false;
        }
        self.local[row].affine.e = xy[0];
        self.local[row].affine.f = xy[1];
        true
    }

    pub(crate) fn finish_placements(&mut self, stamps: &mut Stamps) {
        self.current_at = Some(stamps.clock());
        for list in self.lists.iter().flatten() {
            stamps.clear(*list);
        }
    }

    /// The accumulated transform of the node at `row` in its page, when the columns are current at `clock` and
    /// answer for the row.
    pub(crate) fn placed(&self, clock: u64, row: usize) -> Option<Placed> {
        if !self.current(clock) || !*self.plain.get(row)? {
            return None;
        }
        let parent = self.parent_frame(row);
        Some(self.local[row].under(parent.map(|i| &self.frames[i])))
    }

    fn parent_frame(&self, row: usize) -> Option<usize> {
        let parent = self.link[row];
        (parent != NONE).then(|| self.frame_of[parent as usize] as usize)
    }

    pub(crate) fn hidden(&self, clock: u64, row: usize) -> Option<bool> {
        (self.current(clock) && *self.plain.get(row)? && *self.visibility_plain.get(row)?)
            .then(|| self.hidden[row] || self.parent_frame(row).is_some_and(|i| self.frame_hidden[i]))
    }

    /// The position of the page of the node at `row` (of a row the columns answer for).
    pub(crate) fn page(&self, row: usize) -> (f64, f64) {
        self.page.get(row).copied().unwrap_or((0.0, 0.0))
    }

    /// The scene clock the hierarchy was read at: it names one hierarchy. Something made over these rows (a
    /// plan's spans by row) stands while this does.
    pub fn hierarchy(&self) -> u64 {
        self.read_at.unwrap_or(0)
    }

    /// How many rows the columns cover.
    pub fn len(&self) -> usize {
        self.plain.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plain.is_empty()
    }

    /// Parent frames in hierarchy order, with the row whose direct children read each frame.
    pub fn parents(&self) -> impl DoubleEndedIterator<Item = u32> + '_ {
        self.parent_positions.iter().map(|&p| self.order[p as usize])
    }

    /// A pure translation inherited by direct children. The page offset must be zero
    /// so composing a leaf has exactly the same addition order as the draw matrix.
    pub fn child_translation(&self, row: u32) -> Option<[f64; 2]> {
        let i = row as usize;
        if self.page.get(i) != Some(&(0., 0.)) || !self.plain.get(i).copied()? { return None; }
        let a = self.frames.get(*self.frame_of.get(i)? as usize)?.affine;
        ([a.a, a.b, a.c, a.d] == [1., 0., 0., 1.] && a.e.is_finite() && a.f.is_finite())
            .then_some([a.e, a.f])
    }

    /// The local affine and its transform parent for a canonical row.
    pub fn local_placement(&self, row: u32) -> Option<(Affine, Option<u32>)> {
        let i = row as usize;
        self.plain.get(i).copied().filter(|&plain| plain)?;
        Some((self.local[i].affine, (self.link[i] != NONE).then_some(self.link[i])))
    }

    /// Where `row` and everything under it sit in the parent-before-child order: positions `.0..(.1)`. One
    /// row's range lies wholly inside or wholly outside another's.
    pub fn range(&self, row: u32) -> Option<(u32, u32)> {
        let row = row as usize;
        let (at, end) = (*self.at.get(row)?, *self.end.get(row)?);
        (at != NONE).then_some((at, end))
    }

    /// The rows at positions `from..to` of the parent-before-child order.
    pub fn ordered(&self, from: u32, to: u32) -> &[u32] {
        self.order.get(from as usize..to as usize).unwrap_or(&[])
    }

    /// Whether everything under `row` is a row: nothing under it is an expansion's transient (a table's cells, a
    /// chart's parts) or a child that is not a canonical node. Those are found by name.
    pub fn whole(&self, row: u32) -> bool {
        self.whole.get(row as usize).copied().unwrap_or(false)
    }

    /// Whether the row holds children: a move of it is a move of its subtree, and it need not draw itself.
    pub fn is_parent(&self, row: u32) -> bool {
        self.parent.get(row as usize).copied().unwrap_or(false)
    }

    /// Bring the columns up to `scene`.
    fn refresh(&mut self, scene: &Scene) {
        let clock = scene.stamps.clock();
        if self.current(clock) {
            return;
        }
        let derived = self.derived || !scene.resolved_by_id.is_empty() || !scene.boolean_source_by_id.is_empty();
        let stands = self.plain.len() == scene.doc.nodes.len()
            && self.read_at.is_some_and(|at| {
                scene.stamps.latest(Group::Structure) <= at && (!derived || scene.stamps.latest(Group::Content) <= at)
            });
        if stands {
            // The subtrees of the written rows, as the union of their ranges: a range starts inside an earlier
            // one only when it lies wholly inside it, so each inherited frame is refreshed once, after its parent.
            self.ranges.clear();
            for list in self.lists.into_iter().flatten() {
                for &row in scene.stamps.dirty(list) {
                    self.read_local(scene, row as usize);
                    if self.frame_of[row as usize] != NONE {
                        if let Some(range) = self.range(row) { self.ranges.push(range); }
                    }
                }
            }
            self.ranges.sort_unstable();
            let mut covered = 0u32;
            for i in 0..self.ranges.len() {
                let (from, to) = self.ranges[i];
                if to <= covered {
                    continue;
                }
                let first = self.parent_positions.partition_point(|&p| p < from.max(covered));
                let end = self.parent_positions.partition_point(|&p| p < to);
                for i in first..end {
                    let position = self.parent_positions[i] as usize;
                    self.place_parent(scene, self.order[position] as usize);
                }
                covered = to;
            }
        } else {
            self.read_hierarchy(scene);
            for position in 0..self.order.len() {
                let row = self.order[position] as usize;
                self.read_local(scene, row);
                if self.frame_of[row] != NONE { self.place_parent(scene, row); }
            }
        }
        self.current_at = Some(clock);
    }

    fn read_local(&mut self, scene: &Scene, row: usize) {
        if self.plain[row] {
            let node = &scene.doc.nodes[row];
            self.local[row] = Placed::own(node);
            self.hidden[row] = Scene::hides_paint_subtree(node);
        }
    }

    /// An inherited affine and visibility value are computed once for all children.
    /// A leaf's world-space consumers compose its local value at the point of use.
    fn place_parent(&mut self, scene: &Scene, row: usize) {
        let frame = self.frame_of[row] as usize;
        let above = self.parent_frame(row);
        self.frames[frame] = Placed::parent(&scene.doc.nodes[row], false).under(above.map(|i| &self.frames[i]));
        self.frame_hidden[frame] = self.hidden[row] || above.is_some_and(|i| self.frame_hidden[i]);
    }

    /// Read the scene's hierarchy into the row arrays.
    fn read_hierarchy(&mut self, scene: &Scene) {
        let rows = scene.doc.nodes.len();
        let identity = Placed { affine: transform::translate(0.0, 0.0), rotation: 0.0, scale_x: 1.0, scale_y: 1.0 };
        self.read_at = Some(scene.stamps.clock());
        self.derived = !scene.resolved_by_id.is_empty() || !scene.boolean_source_by_id.is_empty();
        fn reset<T: Clone>(column: &mut Vec<T>, rows: usize, value: T) {
            column.clear();
            column.resize(rows, value);
        }
        reset(&mut self.plain, rows, false);
        reset(&mut self.link, rows, NONE);
        reset(&mut self.page, rows, (0.0, 0.0));
        reset(&mut self.at, rows, NONE);
        reset(&mut self.end, rows, 0);
        reset(&mut self.whole, rows, true);
        reset(&mut self.parent, rows, false);
        reset(&mut self.frame_of, rows, NONE);
        reset(&mut self.local, rows, identity);
        reset(&mut self.hidden, rows, false);
        reset(&mut self.visibility_plain, rows, false);
        self.order.clear();
        self.order.reserve(rows);
        self.frames.clear();
        self.frame_hidden.clear();
        self.parent_positions.clear();
        // A repeated id draws its last copy; only that row is the node.
        let canonical = |row: usize| scene.node_index.get(scene.doc.nodes[row].id()) == Some(&row);
        // Each row's first link of `transform_parent_chain`, by the same rules. A parent that holds children
        // (or is a boolean's source) carries its transform; one that does not ends the chain. A link through
        // an expansion (a table's cell, a chart's part) is not a row: such rows are not answered here.
        // Where a derived record stands in for a node, the node that draws is not the canonical one: that row, and
        // every row under it, is not answered. This is decided here, once; the hierarchy is read again when
        // content changes in a scene that has such records, so a reader looks no name up.
        let derived = self.derived;
        let mut answered = vec![false; rows];
        for (row, node) in scene.doc.nodes.iter().enumerate() {
            if !canonical(row) {
                continue;
            }
            let id = node.id();
            self.parent[row] = is_parent_node(node);
            if derived && (scene.resolved_by_id.contains_key(id) || scene.boolean_source_by_id.contains_key(id)) {
                continue;
            }
            match scene.parent_map.get(id) {
                Some(p) => match scene.effective_node(p) {
                    Some(parent) if is_parent_node(parent) || scene.boolean_source_by_id.contains_key(p) => {
                        if let Some(&parent) = scene.node_index.get(p) {
                            self.link[row] = parent as u32;
                            answered[row] = true;
                        }
                    }
                    _ => answered[row] = true,
                },
                None => answered[row] = !scene.transient_parent_of.contains_key(id),
            }
        }
        // A row is plain when every link above it is: decided once a row, following links that are not decided
        // yet and stopping at any that is. (A hierarchy has no cycles; one is refused on load. A walk longer
        // than the rows is treated as one.)
        let mut decided = vec![false; rows];
        let mut path: Vec<usize> = Vec::new();
        for start in 0..rows {
            if decided[start] {
                continue;
            }
            path.clear();
            let mut row = start;
            let plain = loop {
                if decided[row] {
                    break self.plain[row];
                }
                if !answered[row] || path.len() > rows {
                    break false;
                }
                path.push(row);
                match self.link[row] {
                    NONE => break true,
                    parent => row = parent as usize,
                }
            };
            for &row in &path {
                self.plain[row] = plain;
                decided[row] = true;
            }
            decided[row] = true;
        }
        // Each page's position once (the first page of an id, as the walk by name finds it), then each row its
        // page's.
        let mut positions: HashMap<&str, (f64, f64)> = HashMap::default();
        for (i, page) in scene.doc.pages.iter().enumerate() {
            positions.entry(page.id()).or_insert_with(|| scene.page_position(i));
        }
        for (row, node) in scene.doc.nodes.iter().enumerate() {
            if self.plain[row] {
                self.page[row] =
                    scene.page_of(node.id()).and_then(|pid| positions.get(pid.as_str()).copied()).unwrap_or((0.0, 0.0));
            }
        }
        // Parent-before-child order, by the child lists the display walk reads. Rows with a parent are reached
        // from it; whatever is left after that (a row on no list) starts its own range.
        let mut stack: Vec<(usize, usize)> = Vec::new();
        for pass in 0..2 {
            for root in 0..rows {
                if self.at[root] != NONE {
                    continue;
                }
                if pass == 0 && (!canonical(root) || scene.parent_map.contains_key(scene.doc.nodes[root].id())) {
                    continue;
                }
                self.at[root] = self.order.len() as u32;
                self.order.push(root as u32);
                stack.push((root, 0));
                while let Some(&(row, next)) = stack.last() {
                    let id = scene.doc.nodes[row].id();
                    if next == 0 && (!canonical(row) || scene.transient_by_parent.contains_key(id)) {
                        self.whole[row] = false;
                    }
                    // Only the node a name stands for has that name's children.
                    let kids: &[NodeId] =
                        if canonical(row) { scene.children_by_parent.get(id).map(Vec::as_slice).unwrap_or(&[]) } else { &[] };
                    if let Some(kid) = kids.get(next) {
                        stack.last_mut().expect("the entry just read").1 += 1;
                        match scene.node_index.get(kid) {
                            Some(&child) if self.at[child] == NONE => {
                                self.at[child] = self.order.len() as u32;
                                self.order.push(child as u32);
                                stack.push((child, 0));
                            }
                            // Not a row, or a row already placed under another parent.
                            _ => self.whole[row] = false,
                        }
                    } else {
                        self.end[row] = self.order.len() as u32;
                        stack.pop();
                        if let Some(&(above, _)) = stack.last() {
                            if !self.whole[row] {
                                self.whole[above] = false;
                            }
                        }
                    }
                }
            }
        }
        for &row in &self.order {
            let row = row as usize;
            self.visibility_plain[row] = match scene.parent_map.get(scene.doc.nodes[row].id())
                .and_then(|id| scene.node_index.get(id)) {
                Some(&parent) => self.link[row] == parent as u32 && self.visibility_plain[parent],
                None => !scene.transient_parent_of.contains_key(scene.doc.nodes[row].id()),
            };
        }
        // A plain row that holds children keeps a frame for them.
        for (position, &source) in self.order.iter().enumerate() {
            let row = source as usize;
            if self.plain[row] && self.parent[row] {
                self.parent_positions.push(position as u32);
                self.frame_hidden.push(false);
                self.frame_of[row] = self.frames.len() as u32;
                self.frames.push(identity);
            }
        }
        // The columns are computed along the order: a row's transform parent must keep a frame and hold the row
        // inside its range, so that it is computed first and a write to it reaches the row. A document whose
        // child lists and parent links disagree about that is answered by the walk by name alone.
        let agrees = (0..rows).all(|row| {
            !self.plain[row] || match self.link[row] {
                NONE => true,
                parent => {
                    let parent = parent as usize;
                    self.frame_of[parent] != NONE && self.at[parent] < self.at[row] && self.at[row] < self.end[parent]
                }
            }
        });
        if !agrees {
            reset(&mut self.plain, rows, false);
            reset(&mut self.frame_of, rows, NONE);
            self.parent_positions.clear();
        }
    }
}

impl Scene {
    /// Bring the scene's world columns up to its clock ([`crate::worlds`]): the hierarchy again when structure
    /// changed, else the subtrees of the rows written since the last refresh. Until the next write, every
    /// accumulated transform of a row the columns answer for is read from them, and [`Scene::world_rows`] hands
    /// out the hierarchy by row.
    pub fn refresh_worlds(&mut self) {
        if self.worlds.current(self.stamps.clock()) {
            return;
        }
        let mut worlds = std::mem::take(&mut self.worlds);
        worlds.refresh(self);
        for list in worlds.lists.iter().flatten() {
            self.stamps.clear(*list);
        }
        self.worlds = worlds;
    }

    /// The world columns and the hierarchy by row, while they are current: `None` after any write since
    /// [`Scene::refresh_worlds`].
    pub fn world_rows(&self) -> Option<&Worlds> {
        self.worlds.current(self.stamps.clock()).then_some(&self.worlds)
    }

    /// [`Scene::node_canvas_affine`] of the canonical node at `row`, read from the world columns: no name is
    /// looked up. `None` when the columns are not current or do not answer for the row (a derived record stands
    /// in for it or for one of its transform ancestors, or an ancestor is not a canonical node); the walk by name
    /// answers then, with the same composition.
    pub fn row_canvas_affine(&self, row: u32) -> Option<Affine> {
        let placed = self.worlds.placed(self.stamps.clock(), row as usize)?;
        let page = self.worlds.page(row as usize);
        Some(compose(transform::translate(page.0, page.1), placed.affine))
    }

    /// The accumulated transform as a transform parent of everything above the node named `id`, composed parent
    /// first along [`Scene::transform_parent_chain`]; `None` when nothing is above it.
    pub(crate) fn frame_above(&self, id: &str) -> Option<Placed> {
        let mut frame: Option<Placed> = None;
        for pid in self.transform_parent_chain(id).iter().rev() {
            let Some(parent) = self.effective_node(pid) else {
                continue;
            };
            frame = Some(Placed::parent(parent, self.boolean_source_by_id.contains_key(pid)).under(frame.as_ref()));
        }
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::NodePlacement;
    use crate::value::json;

    fn bits(a: &Affine) -> [u64; 6] {
        [a.a, a.b, a.c, a.d, a.e, a.f].map(f64::to_bits)
    }

    /// What the walk by name answers for each of `ids` (the columns are stale), then what the columns answer
    /// once refreshed: the same numbers, bit for bit, by row and by name.
    fn check(scene: &mut Scene, ids: &[&str]) {
        assert!(scene.world_rows().is_none(), "a write leaves the columns unanswering until they are refreshed");
        let walked: Vec<_> = ids.iter().map(|id| scene.node_canvas_affine_and_transform(id).unwrap()).collect();
        scene.refresh_worlds();
        assert!(scene.world_rows().is_some());
        for (id, walked) in ids.iter().zip(&walked) {
            let row = scene.row_of(id).unwrap();
            assert_eq!(bits(&scene.row_canvas_affine(row).unwrap()), bits(&walked.0), "{id} by row");
            let (affine, trs) = scene.node_canvas_affine_and_transform(id).unwrap();
            assert_eq!(bits(&affine), bits(&walked.0), "{id} by name");
            assert_eq!(format!("{trs:?}"), format!("{:?}", walked.1), "{id} transform");
        }
    }

    #[test]
    fn a_fading_body_keeps_current_world_rows_until_visibility_changes() {
        let mut scene = Scene::load(&json!({
            "nodes": [{"id":"p","type":"path","x":17,"y":3}],
            "pages": [{"id":"page","width":32,"height":32,"nodeIds":["p"]}]
        })).unwrap();
        let row = scene.row_of("p").unwrap();
        scene.refresh_worlds();
        let transform = bits(&scene.row_canvas_affine(row).unwrap());
        let moved = scene.stamps.subscribe(Group::Placement);
        for (opacity, current) in [(0.5, true), (0.7, true), (0.0, false), (0.2, false)] {
            let mut edits = vec![crate::scene::PlacementEdit::new(row, opacity)];
            assert!(scene.write_placements(&mut edits, |n| (None, crate::value::Number::from_f64(*n))));
            assert_eq!(scene.stamps.dirty(moved), &[row]);
            scene.stamps.clear(moved);
            assert_eq!(scene.world_rows().is_some(), current);
            scene.refresh_worlds();
            assert_eq!(scene.hidden_from_paint_order("p"), opacity == 0.0);
            assert_eq!(bits(&scene.row_canvas_affine(row).unwrap()), transform);
        }
    }

    #[test]
    fn the_columns_hold_what_the_walk_by_name_composes() {
        let d = json!({"nodes": [
            {"id": "g", "type": "group", "x": 10.5, "y": -3.25, "rotation": 30, "scaleX": 1.5, "scaleY": 0.75, "children": ["h", "a"]},
            {"id": "h", "type": "group", "x": 2, "y": 7, "rotation": -12.5, "scaleX": -1, "scaleY": 2, "skewX": 10, "children": ["b"]},
            {"id": "a", "type": "path", "x": 1, "y": 2, "rotation": 45,
             "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 10, "y": 0}, {"type": "line", "x": 10, "y": 10}], "closed": true},
            {"id": "b", "type": "path", "x": 3.3, "y": 4.4, "scaleX": 2, "scaleY": 0.3,
             "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 5, "y": 0}, {"type": "line", "x": 5, "y": 5}], "closed": true},
            {"id": "c", "type": "path", "x": 9, "y": 9,
             "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 5, "y": 0}, {"type": "line", "x": 5, "y": 5}], "closed": true}],
            "pages": [{"id": "p", "width": 100, "height": 100, "nodeIds": ["g", "c"]}]});
        let mut scene = Scene::load(&d).unwrap();
        let all = ["g", "h", "a", "b", "c"];
        check(&mut scene, &all);
        // Parent first: a node two groups deep is its local transform under (g ∘ h), not g ∘ (h ∘ local).
        {
            let node = |id: &str| scene.node(id).unwrap();
            let above = Placed::parent(node("h"), false).under(Some(&Placed::parent(node("g"), false)));
            let b = Placed::own(node("b")).under(Some(&above));
            assert_eq!(bits(&scene.accumulated_affine("b").unwrap().0), bits(&b.affine));
        }
        // The hierarchy by row: a row and everything under it are one range, parents first.
        {
            let rows = scene.world_rows().unwrap();
            let names = |id: &str| -> Vec<String> {
                let (from, to) = rows.range(scene.row_of(id).unwrap()).unwrap();
                rows.ordered(from, to).iter().map(|&row| scene.row(row).unwrap().id().to_string()).collect()
            };
            assert_eq!(names("g"), ["g", "h", "b", "a"]);
            assert_eq!(names("h"), ["h", "b"]);
            assert_eq!(names("c"), ["c"]);
            let (g, a) = (scene.row_of("g").unwrap(), scene.row_of("a").unwrap());
            assert!(rows.is_parent(g) && !rows.is_parent(a) && rows.whole(g));
        }
        // A typed placement write to an ancestor reaches everything under it, through the same hierarchy.
        let hierarchy = scene.world_rows().unwrap().hierarchy();
        let number = |v: f64| crate::value::Number::from_f64(v).unwrap();
        let g = scene.row_of("g").unwrap();
        scene.write_placement(g, Some(NodePlacement { translation: Some([number(40.25), number(1.0)]), rotation: Some(number(77.0)), scale: Some([number(0.5), number(3.0)]) }), None);
        check(&mut scene, &all);
        // So does a rotation written as a field, and a write to a leaf alone.
        let h = scene.row_of("h").unwrap();
        scene.write_row(h, |node| node.set("rotation", json!(33.5))).unwrap();
        check(&mut scene, &all);
        let b = scene.row_of("b").unwrap();
        scene.write_placement(b, Some(NodePlacement { translation: Some([number(-7.125), number(0.5)]), rotation: Some(number(3.0)), scale: Some([number(1.0), number(1.0)]) }), None);
        check(&mut scene, &all);
        assert_eq!(scene.world_rows().unwrap().hierarchy(), hierarchy, "writes to rows do not read the hierarchy again");
        // A repeated leaf edit can update the world column in place. Compare it
        // against an independent scene which recomposes the entire ancestor chain.
        let mut reference = Scene::load(&d).unwrap();
        let mut batched = Scene::load(&d).unwrap();
        batched.refresh_worlds();
        for ids in [["b", "b", "c"], ["b", "g", "a"]] {
            let mut edits = Vec::new();
            for (i, id) in ids.iter().enumerate() {
                let row = reference.row_of(id).unwrap();
                let placement = NodePlacement {
                    translation: Some([number(i as f64 - 2.5), number(i as f64 * 1.75)]),
                    rotation: None, scale: None,
                };
                reference.write_placement(row, Some(placement.clone()), None);
                edits.push(crate::scene::PlacementEdit::new(row, placement));
            }
            batched.write_placements(&mut edits, |t| (Some(t.clone()), None));
            assert_eq!(batched.world_rows().is_some(), ids[1] != "g");
            batched.refresh_worlds();
            for id in all {
                let row = reference.row_of(id).unwrap();
                assert_eq!(bits(&batched.row_canvas_affine(row).unwrap()),
                    bits(&reference.node_canvas_affine_and_transform(id).unwrap().0));
            }
        }
        // After a whole-node edit too, nothing answers until the next refresh.
        let mut moved = scene.node("a").unwrap().clone();
        moved.set("x", json!(-55.5));
        scene.apply_node_edits(vec![moved], &[], &[], None).unwrap();
        assert!(scene.world_rows().is_none());
        check(&mut scene, &all);
        // A structural edit moves rows: the hierarchy is read again.
        scene.apply_node_edits(vec![], &["g".to_string()], &[], None).unwrap();
        check(&mut scene, &["c"]);
        assert_ne!(scene.world_rows().unwrap().hierarchy(), hierarchy);
    }

    thread_local! {
        /// What the probe below read.
        static READ: std::cell::Cell<Option<f64>> = const { std::cell::Cell::new(None) };
    }

    /// A write stamps what it wrote after deriving from it. A derivation in that window reads the written node:
    /// columns refreshed before the write do not answer it with the transform the node held. (Without the clock
    /// advance in `Scene::derive_written` the probe reads the old position.)
    #[test]
    fn a_derivation_inside_a_write_reads_the_written_transform() {
        let d = json!({"nodes": [
            {"id": "a", "type": "path", "x": 1, "y": 2,
             "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 10, "y": 0}, {"type": "line", "x": 10, "y": 10}], "closed": true}],
            "pages": [{"id": "p", "width": 100, "height": 100, "nodeIds": ["a"]}]});
        fn probe(scene: &Scene) {
            READ.with(|read| read.set(scene.accumulated_affine("a").map(|(affine, _)| affine.e)));
        }
        for whole in [false, true] {
            let mut scene = Scene::load(&d).unwrap();
            scene.refresh_worlds();
            assert_eq!(scene.accumulated_affine("a").unwrap().0.e, 1.0, "warm: read from the column");
            let x = if whole { 70.0 } else { 40.0 };
            crate::scene::DERIVATION_PROBE.with(|slot| slot.set(Some(probe as fn(&Scene))));
            if whole {
                let mut node = scene.node("a").unwrap().clone();
                node.set("x", json!(x));
                scene.apply_node_edits(vec![node], &[], &[], None).unwrap();
            } else {
                let row = scene.row_of("a").unwrap();
                scene.write_row(row, |node| node.set("x", json!(x))).unwrap();
            }
            crate::scene::DERIVATION_PROBE.with(|slot| slot.set(None));
            assert_eq!(READ.with(|read| read.take()), Some(x), "whole-node edit: {whole}");
        }
    }
}
