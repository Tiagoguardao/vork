//! A page of span metadata can be released independently of the items or the packet it
//! addresses. Ordinals, item ranges and render z usually advance together. Keep the exact
//! exceptions to those strides, so reading one row restores at most one page.
use crate::frame::{NodeSpan, SpanSource, TranslationPlacement};
use std::ops::{Index, IndexMut};
use std::sync::OnceLock;

pub(super) const PAGE: usize = 256;
const FIELDS: usize = 7;
type Values = [u64; FIELDS];

struct Seed {
    first: Values,
    step: Values,
    len: usize,
    exceptions: Vec<(u8, u8, u64)>,
    names: Vec<(u8, Box<str>)>,
    recipes: Vec<(u8, Option<TranslationPlacement>)>,
}

fn values(span: &NodeSpan) -> Values {
    [match &span.source { SpanSource::Row(row) => u64::from(*row), SpanSource::Generated(_) => u64::MAX },
        span.start as u64, span.len as u64, span.z, span.z_consumed, span.primary as u64, span.dl_z as u64]
}
impl Seed {
    fn of(spans: &[NodeSpan]) -> Self {
        let first = values(&spans[0]);
        let second = spans.get(1).map_or(first, values);
        let step = std::array::from_fn(|j| second[j].wrapping_sub(first[j]));
        let mut seed = Self { first, step, len: spans.len(), exceptions: Vec::new(), names: Vec::new(), recipes: Vec::new() };
        for (i, span) in spans.iter().enumerate() {
            let current = values(span);
            for (j, value) in current.into_iter().enumerate() {
                if first[j].wrapping_add(step[j].wrapping_mul(i as u64)) != value {
                    seed.exceptions.push((i as u8, j as u8, value));
                }
            }
            if let SpanSource::Generated(name) = &span.source { seed.names.push((i as u8, name.clone())); }
            let recipe = span.placement.as_deref().map(|r| r.for_row(0));
            if seed.recipes.last().and_then(|(_, r)| r.as_ref()) != recipe.as_ref() {
                seed.recipes.push((i as u8, recipe));
            }
        }
        seed
    }
    fn field(&self, row: u8, field: u8) -> u64 {
        let i = self.exceptions.partition_point(|&(r, f, _)| (r, f) < (row, field));
        if let Some(&(r, f, value)) = self.exceptions.get(i) {
            if (r, f) == (row, field) { return value; }
        }
        let j = usize::from(field);
        self.first[j].wrapping_add(self.step[j].wrapping_mul(u64::from(row)))
    }

    fn placement(&self, row: u8) -> Option<(std::ops::Range<usize>, TranslationPlacement)> {
        let run = self.recipes.partition_point(|&(first, _)| first <= row).checked_sub(1)?;
        let recipe = self.recipes[run].1.as_ref()?;
        let source = u32::try_from(self.field(row, 0)).ok()?;
        let start = self.field(row, 1) as usize;
        let end = start.saturating_add(self.field(row, 2) as usize);
        Some((start..end, recipe.for_row(source)))
    }

    /// Compaction only moves item starts. Recompute that column's stride and exceptions
    /// without allocating the source records, names and placement recipes on this page.
    fn remap_starts(&mut self, mut remap: impl FnMut(usize, usize, usize) -> usize) {
        let mut starts = [0u64; PAGE];
        for (i, start) in starts[..self.len].iter_mut().enumerate() {
            *start = remap(i, self.field(i as u8, 1) as usize, self.field(i as u8, 2) as usize) as u64;
        }
        self.first[1] = starts[0];
        self.step[1] = if self.len > 1 { starts[1].wrapping_sub(starts[0]) } else { 0 };
        self.exceptions.retain(|&(_, field, _)| field != 1);
        let old = self.exceptions.len();
        for (i, &start) in starts[..self.len].iter().enumerate() {
            if start != self.first[1].wrapping_add(self.step[1].wrapping_mul(i as u64)) {
                self.exceptions.push((i as u8, 1, start));
            }
        }
        if old != 0 && self.exceptions.len() != old {
            self.exceptions.sort_unstable_by_key(|&(row, field, _)| (row, field));
        }
    }

    fn expand(&self) -> Box<[NodeSpan]> {
        let mut exceptions = self.exceptions.iter().peekable();
        let mut names = self.names.iter();
        let mut recipes = self.recipes.iter().peekable();
        let mut recipe = None;
        (0..self.len).map(|i| {
            let mut v: Values = std::array::from_fn(|j| self.first[j].wrapping_add(self.step[j].wrapping_mul(i as u64)));
            while exceptions.peek().is_some_and(|(at, _, _)| usize::from(*at) == i) {
                let &(_, j, value) = exceptions.next().unwrap();
                v[usize::from(j)] = value;
            }
            if recipes.peek().is_some_and(|(at, _)| usize::from(*at) == i) { recipe = recipes.next().unwrap().1.as_ref(); }
            let source = if v[0] == u64::MAX { SpanSource::Generated(names.next().unwrap().1.clone()) }
                else { SpanSource::Row(v[0] as u32) };
            NodeSpan { source, start: v[1] as usize, len: v[2] as usize, z: v[3], z_consumed: v[4],
                primary: v[5] as usize, dl_z: v[6] as usize, placement: recipe.map(|r| Box::new(r.for_row(v[0] as u32))) }
        }).collect()
    }
    fn bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.exceptions.capacity() * std::mem::size_of::<(u8, u8, u64)>()
            + self.recipes.capacity() * std::mem::size_of::<(u8, Option<TranslationPlacement>)>()
            + self.names.capacity() * std::mem::size_of::<(u8, Box<str>)>()
            + self.names.iter().map(|(_, name)| name.len()).sum::<usize>()
    }
}

pub(super) struct Page {
    expanded: OnceLock<Box<[NodeSpan]>>,
    seed: Option<Box<Seed>>,
    dense_bytes: Option<usize>,
}
impl Page {
    pub(super) fn read(&self) -> &[NodeSpan] {
        self.expanded.get_or_init(|| self.seed.as_ref().expect("span page seed").expand())
    }
    pub(super) fn write(&mut self) -> &mut [NodeSpan] {
        self.read();
        self.seed = None;
        self.dense_bytes = None;
        self.expanded.get_mut().unwrap()
    }
    fn release_recipes(&mut self) {
        if let Some(seed) = &mut self.seed {
            seed.recipes = Vec::new();
            self.expanded.take();
        } else {
            for span in self.write() { span.placement = None; }
        }
    }

    fn trim(&mut self) -> usize {
        if let Some(bytes) = self.dense_bytes { return bytes; }
        if self.seed.is_none() {
            let spans = self.expanded.get().unwrap();
            let bytes = std::mem::size_of_val(&**spans) + spans.iter().map(|s| {
                usize::from(s.placement.is_some()) * std::mem::size_of::<TranslationPlacement>()
                    + match &s.source { SpanSource::Generated(name) => name.len(), _ => 0 }
            }).sum::<usize>();
            let seed = Seed::of(spans);
            if seed.bytes() >= bytes { self.dense_bytes = Some(bytes); return bytes; }
            self.seed = Some(Box::new(seed));
        }
        self.expanded.take();
        self.seed.as_ref().unwrap().bytes()
    }
}
pub(super) struct Spans {
    pub(super) pages: Vec<Page>,
    len: usize,
}
impl From<Vec<NodeSpan>> for Spans {
    fn from(spans: Vec<NodeSpan>) -> Self {
        let len = spans.len();
        let mut it = spans.into_iter();
        let mut pages = Vec::with_capacity(len.div_ceil(PAGE));
        while it.len() > 0 {
            pages.push(Page { expanded: OnceLock::from(it.by_ref().take(PAGE).collect::<Box<[_]>>()), seed: None, dense_bytes: None });
        }
        Self { pages, len }
    }
}
impl Spans {
    pub(super) fn len(&self) -> usize { self.len }
    pub(super) fn get(&self, i: usize) -> Option<&NodeSpan> { self.pages.get(i / PAGE)?.read().get(i % PAGE) }
    pub(super) fn range(&self, i: usize) -> (usize, usize) {
        let page = &self.pages[i / PAGE];
        let row = i % PAGE;
        if let Some(spans) = page.expanded.get() {
            return (spans[row].start, spans[row].len);
        }
        let seed = page.seed.as_ref().expect("span page seed");
        (seed.field(row as u8, 1) as usize, seed.field(row as u8, 2) as usize)
    }

    pub(super) fn remap_starts(&mut self, mut remap: impl FnMut(usize, usize, usize) -> usize) {
        for (p, page) in self.pages.iter_mut().enumerate() {
            let first = p * PAGE;
            if let Some(seed) = &mut page.seed {
                seed.remap_starts(|i, start, len| remap(first + i, start, len));
                page.expanded.take();
            } else {
                for (i, span) in page.write().iter_mut().enumerate() {
                    span.start = remap(first + i, span.start, span.len);
                }
            }
        }
    }
    /// A matrix edit reads the range and recipe of one span. The page can stay compressed.
    pub(super) fn placement(&self, i: usize) -> Option<(std::ops::Range<usize>, TranslationPlacement)> {
        let page = self.pages.get(i / PAGE)?;
        let row = i % PAGE;
        if let Some(spans) = page.expanded.get() {
            let span = spans.get(row)?;
            return Some((span.start..span.start.saturating_add(span.len), span.placement.as_deref()?.clone()));
        }
        page.seed.as_ref()?.placement(row as u8)
    }
    /// The display-list and draw depths for one ordinary emission. Reading an evicted
    /// source must not allocate the other 255 span recipes on its page.
    pub(super) fn anchor<'a>(&'a self, i: usize, scene: &'a moda_scene::Scene) -> Option<(&'a str, usize, u64, u64)> {
        let page = self.pages.get(i / PAGE)?;
        let row = (i % PAGE) as u8;
        if let Some(spans) = page.expanded.get() {
            let s = spans.get(row as usize)?;
            return Some((s.source.id(scene), s.dl_z, s.z, s.z_consumed));
        }
        let seed = page.seed.as_ref()?;
        let source = u32::try_from(seed.field(row, 0)).ok()?;
        Some((scene.row(source)?.id(), seed.field(row, 6) as usize, seed.field(row, 3), seed.field(row, 4)))
    }

    pub(super) fn iter(&self) -> impl DoubleEndedIterator<Item = &NodeSpan> {
        self.pages.iter().flat_map(|p| p.read())
    }
    pub(super) fn iter_mut(&mut self) -> impl Iterator<Item = &mut NodeSpan> {
        self.pages.iter_mut().flat_map(|p| p.write())
    }
    pub(super) fn trim(&mut self, limit: usize) -> usize {
        let mut bytes = self.pages.capacity() * std::mem::size_of::<Page>()
            + self.pages.iter_mut().map(Page::trim).sum::<usize>();
        for page in self.pages.iter_mut().rev() {
            if bytes <= limit { break; }
            let before = page.trim();
            page.release_recipes();
            bytes = bytes - before + page.trim();
        }
        bytes
    }
}
impl Index<usize> for Spans {
    type Output = NodeSpan;
    fn index(&self, i: usize) -> &NodeSpan { &self.pages[i / PAGE].read()[i % PAGE] }
}
impl IndexMut<usize> for Spans {
    fn index_mut(&mut self, i: usize) -> &mut NodeSpan { &mut self.pages[i / PAGE].write()[i % PAGE] }
}
