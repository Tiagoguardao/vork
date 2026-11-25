//! Stable output addresses and a separate painter-order column for retained units.
use crate::seams::{CoveragePlan, DrawItem};
use std::ops::Range;

#[derive(Clone)]
pub(crate) struct ItemOrder {
    pub(crate) ranges: Vec<Range<usize>>,
    pub(crate) capacities: Vec<usize>,
    pub(crate) len: usize,
}
impl ItemOrder {
    pub(crate) fn new(ranges: Vec<Range<usize>>, storage_len: usize) -> Option<Self> {
        let mut end = 0;
        for range in &ranges {
            if range.start != end || range.end < range.start {
                return None;
            }
            end = range.end;
        }
        if end != storage_len {
            return None;
        }
        let capacities = ranges.iter().map(Range::len).collect();
        Some(Self {
            ranges,
            capacities,
            len: storage_len,
        })
    }
    pub(crate) fn bytes(&self) -> usize {
        self.ranges.capacity() * std::mem::size_of::<Range<usize>>()
            + self.capacities.capacity() * std::mem::size_of::<usize>()
    }
}

#[derive(Clone)]
pub(crate) struct PlanItems<'a> {
    plan: &'a CoveragePlan,
    range: usize,
    at: usize,
    remaining: usize,
}
impl<'a> PlanItems<'a> {
    pub(crate) fn new(plan: &'a CoveragePlan) -> Self {
        Self {
            plan,
            range: 0,
            at: 0,
            remaining: plan
                .item_order
                .as_ref()
                .map_or(plan.items.len(), |order| order.len),
        }
    }
}
impl<'a> Iterator for PlanItems<'a> {
    type Item = (usize, &'a DrawItem);
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let index = if let Some(order) = &self.plan.item_order {
            loop {
                let range = &order.ranges[self.range];
                if self.at < range.len() {
                    break range.start + self.at;
                }
                self.range += 1;
                self.at = 0;
            }
        } else {
            self.at
        };
        self.at += 1;
        self.remaining -= 1;
        Some((index, &self.plan.items[index]))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl ExactSizeIterator for PlanItems<'_> {}

/// Replace one unit without moving any other unit's physical output addresses.
/// Reserved slots grow geometrically; shrinking and regrowing reuses that storage.
pub(crate) fn replace_unit(
    plan: &mut CoveragePlan,
    seed: &mut Option<ItemOrder>,
    unit: usize,
    items: Vec<DrawItem>,
    written: &mut crate::residency::PlanWritten,
) -> Option<(usize, i64)> {
    if plan.item_order.is_none() {
        plan.item_order = seed.take();
    }
    let order = plan.item_order.as_mut()?;
    let old = order.ranges.get(unit)?.clone();
    let capacity = *order.capacities.get(unit)?;
    let count = items.len();
    let old_bytes = crate::residency::items_bytes(&plan.items[old.clone()]);
    let new_bytes = crate::residency::items_bytes(&items);
    let vacant_bytes = crate::residency::items_bytes(&[DrawItem::Vacant]);
    let start;
    let added;
    if count > capacity {
        let next = count.checked_next_power_of_two()?;
        start = plan.items.len();
        added = next;
        plan.items
            .resize_with(start.checked_add(next)?, || DrawItem::Vacant);
        for at in old.clone() {
            written.replaced.push((
                u32::try_from(at).ok()?,
                std::mem::replace(&mut plan.items[at], DrawItem::Vacant),
            ));
        }
        order.capacities[unit] = next;
    } else {
        start = old.start;
        added = 0;
        for at in start + count..old.end {
            written.replaced.push((
                u32::try_from(at).ok()?,
                std::mem::replace(&mut plan.items[at], DrawItem::Vacant),
            ));
        }
    }
    for (offset, item) in items.into_iter().enumerate() {
        let at = start + offset;
        written.replaced.push((
            u32::try_from(at).ok()?,
            std::mem::replace(&mut plan.items[at], item),
        ));
    }
    order.len = order.len - old.len() + count;
    order.ranges[unit] = start..start + count;
    let delta = new_bytes as i64 - old_bytes as i64
        + (added as i64 + old.len() as i64 - count as i64) * vacant_bytes as i64;
    Some((start, delta))
}
