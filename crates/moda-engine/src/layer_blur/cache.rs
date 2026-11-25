//! The retained engine's layer-capture cache: filtered capture results (a
//! layer blur's, a shadow's) kept across frames under their content key
//! (`LayerBlurGroup::cache_key`, `residency::capture_key`), so a frame whose
//! captured nodes are unchanged composites the stored result instead of
//! re-rendering and re-filtering every capture.
//!
//! The key covers everything the result's pixels are made of — the captured
//! nodes' content, placement, opacity and clip, the capture's region, size
//! and filters (the camera's zoom bucket and capture window reach it through
//! these), and the scene-wide resource generations — so there is no explicit
//! invalidation: an edit, a drag preview, a zoom past a bucket or a font load
//! simply asks for a new key, and entries nobody asks for age out.
//!
//! Bounded by a byte budget (the results' texels: RGBA8, or RGBA16F after a
//! uniform blur), evicting the least recently used entries first. One engine's device only: dropped with the
//! engine's retained GPU state (`Engine::clear_retained_gpu`).

use std::collections::HashMap;

/// Default byte budget of the cached results.
pub const DEFAULT_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// Approximate consecutive-frame membership for capture admission. Every
/// key contributes, regardless of traversal order or scene size. Membership
/// selects what to keep. Only exact content keys serve pixels from the cache.
#[derive(Default)]
pub(crate) struct CaptureHistory {
    previous: Box<[u64]>,
    current: Box<[u64]>,
}
impl CaptureHistory {
    // Two 32 KiB filters, rather than a distinct-key cutoff. At 8,192 unique
    // keys/frame, three probes have about 0.07% false positives. Saturation
    // admits more scratch results to the existing 256 MiB LRU; it never drops
    // later keys or changes the exact keys used to serve cached pictures.
    const BITS: usize = 256 * 1024;

    pub(crate) fn begin_frame(&mut self) {
        std::mem::swap(&mut self.previous, &mut self.current);
        self.current.fill(0);
    }

    pub(crate) fn observe(&mut self, key: u64) -> bool {
        if self.current.is_empty() {
            self.previous = vec![0; Self::BITS / 64].into_boxed_slice();
            self.current = vec![0; Self::BITS / 64].into_boxed_slice();
        }
        // Cache keys are already content hashes. Double hashing supplies three
        // probes; the odd step also covers the whole power-of-two bit domain.
        let step = key.rotate_left(32) | 1;
        let mut present = true;
        for probe in 0..3 {
            let bit = key.wrapping_add(step.wrapping_mul(probe)) as usize & (Self::BITS - 1);
            let word = bit / 64;
            let mask = 1u64 << (bit % 64);
            present &= self.previous[word] & mask != 0;
            self.current[word] |= mask;
        }
        present
    }
}

struct Entry<V> {
    view: V,
    allocation: Option<wgpu::Texture>,
    mask_assets: Option<std::sync::Arc<moda_assets::engine_seam::AssetReads>>,
    texels: u64,
    last_used: u64,
}

/// Filtered capture results by content key, LRU within a byte budget. `V` is the result
/// (a texture view; the bookkeeping is tested with a plain value).
pub struct CaptureCache<V = wgpu::TextureView> {
    pub(crate) ledger: Option<std::sync::Arc<crate::scratch::CaptureLedger>>,
    pub(crate) history: CaptureHistory,
    entries: HashMap<u64, Entry<V>>,
    bytes: u64,
    budget: u64,
    /// Number of cached keys retaining each physical backing texture.
    owners: HashMap<wgpu::Texture, usize>,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl<V> Default for CaptureCache<V> {
    fn default() -> Self {
        CaptureCache {
            ledger: None,
            history: CaptureHistory::default(),
            entries: HashMap::new(),
            owners: HashMap::new(),
            bytes: 0,
            budget: DEFAULT_BUDGET_BYTES,
            clock: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }
}

impl CaptureCache<wgpu::TextureView> {
    pub(crate) fn owns(&self, texture: &wgpu::Texture) -> bool { self.owners.contains_key(texture) }

    pub fn put(&mut self, key: u64, view: wgpu::TextureView, width: u32, height: u32) {
        let texture = view.texture().clone();
        debug_assert!(width <= texture.width() && height <= texture.height());
        let bytes = crate::scratch::texture_bytes(&texture);
        self.keep_allocation(key, view, Some(texture), bytes, None);
    }

    /// Keep the drawn mask and the resource identities that validate it, evicting the
    /// least recently used entries past the budget. A result larger than the
    /// whole budget is not kept.
    pub fn put_mask(&mut self, key: u64, view: wgpu::TextureView, reads: std::sync::Arc<moda_assets::engine_seam::AssetReads>) {
        let texture = view.texture().clone();
        let bytes = crate::scratch::texture_bytes(&texture);
        let reads = (!reads.is_empty()).then_some(reads);
        self.keep_allocation(key, view, Some(texture), bytes, reads);
    }

}

impl<V: Clone> CaptureCache<V> {
    /// Spare space for optional source images, without displacing filtered results.
    pub(crate) fn available_bytes(&self) -> u64 {
        self.budget.saturating_sub(self.bytes)
    }

    /// An exact hit, or the key to publish after rendering an admitted miss.
    pub(crate) fn lookup(&mut self, key: u64, owner: Option<&str>, required: bool) -> Result<V, Option<u64>> {
        use std::hash::BuildHasher;
        let repeated = self.history.observe(key);
        let hit = self.get(key);
        // Repeated content ends the owner's run of changes, even on a miss. A returning
        // hit (A / X1 / A) still counts as a change for the next miss.
        let first = owner.filter(|_| !repeated).is_some_and(|node| {
            !self.history.observe(rustc_hash::FxBuildHasher.hash_one(("filtered capture owner", node)))
        });
        hit.ok_or((first || repeated || required).then_some(key))
    }

    /// The stored result for `key`, marked as just used.
    pub fn get(&mut self, key: u64) -> Option<V> {
        self.clock += 1;
        match self.entries.get_mut(&key).filter(|e| e.mask_assets.as_ref().is_none_or(|reads| reads.current())) {
            Some(e) => {
                e.last_used = self.clock;
                self.hits += 1;
                Some(e.view.clone())
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    fn keep_allocation(&mut self, key: u64, view: V, allocation: Option<wgpu::Texture>, texels: u64,
        mask_assets: Option<std::sync::Arc<moda_assets::engine_seam::AssetReads>>) {
        let metadata = mask_assets.as_deref().map_or(0, |r| r.retained_bytes() + std::mem::size_of_val(r)) as u64;
        if texels + metadata > self.budget {
            return;
        }
        self.remove(key);
        self.clock += 1;
        let first = allocation.as_ref().is_none_or(|texture| {
            let owners = self.owners.entry(texture.clone()).or_default();
            *owners += 1;
            if *owners == 1 {
                if let Some(ledger) = &self.ledger { ledger.cache(texture, true); }
            }
            *owners == 1
        });
        self.bytes += metadata + if first { texels } else { 0 };
        self.entries.insert(key, Entry { view, allocation, mask_assets, texels, last_used: self.clock });
        self.evict_to_budget(Some(key));
    }

    /// Drop least recently used entries, never `keep`, until the budget holds what is left.
    fn evict_to_budget(&mut self, keep: Option<u64>) {
        while self.bytes > self.budget {
            let Some((&oldest, _)) = self
                .entries
                .iter()
                .filter(|(k, _)| Some(**k) != keep)
                .min_by_key(|(_, e)| e.last_used)
            else {
                break;
            };
            self.remove(oldest);
            self.evictions += 1;
        }
    }

    /// Release one reconstructible result under device-wide memory pressure.
    pub(crate) fn evict_oldest(&mut self) -> bool {
        let Some((&key, _)) = self.entries.iter().min_by_key(|(_, e)| e.last_used) else { return false };
        self.remove(key);
        self.evictions += 1;
        true
    }

    fn remove(&mut self, key: u64) {
        let Some(entry) = self.entries.remove(&key) else { return };
        let metadata = entry.mask_assets.as_deref().map_or(0, |r| r.retained_bytes() + std::mem::size_of_val(r)) as u64;
        self.bytes -= metadata;
        if let Some(texture) = &entry.allocation {
            let owners = self.owners.get_mut(texture).expect("held allocation");
            *owners -= 1;
            if *owners > 0 { return; }
            self.owners.remove(texture);
            if let Some(ledger) = &self.ledger { ledger.cache(texture, false); }
        }
        self.bytes -= entry.texels;
    }

    /// Change the byte budget (evicting down to it).
    pub fn set_budget(&mut self, budget: u64) {
        self.budget = budget;
        self.evict_to_budget(None);
    }

    /// Drop every entry (the counters stay).
    pub fn clear(&mut self) {
        self.history = CaptureHistory::default();
        self.entries.clear();
        if let Some(ledger) = &self.ledger {
            for t in self.owners.keys() { ledger.cache(t, false); }
        }
        self.owners.clear();
        self.bytes = 0;
    }

    /// (entries, bytes) held; the bytes are the retained textures and mask metadata.
    pub fn resident(&self) -> (usize, u64) {
        (self.entries.len(), self.bytes)
    }

}

#[cfg(test)]
mod tests {
    use super::{CaptureCache, CaptureHistory};

    #[test]
    fn filtered_lookup_renders_once_per_hold_and_skips_changing_content() {
        let mut cache = CaptureCache::default();
        cache.set_budget(4);
        // A returning hit between new values still indicates animation.
        for (key, expected) in [(1, Err(Some(1))), (2, Err(None)), (1, Ok(1)), (3, Err(None)),
            (3, Err(Some(3))), (3, Ok(3)), (4, Err(Some(4))), (4, Ok(4)), (5, Err(Some(5))), (5, Ok(5))] {
            cache.history.begin_frame();
            let result = cache.lookup(key, Some("card"), false);
            assert_eq!(result, expected);
            if let Err(Some(key)) = result { cache.keep_allocation(key, key, None, 4, None); }
        }
        assert!(cache.evict_oldest());
        cache.history.begin_frame();
        assert_eq!(cache.lookup(5, None, false), Err(Some(5))); // last frame's hit was observed
        assert_eq!(cache.lookup(6, None, true), Err(Some(6)));
    }

    fn rendered_frames(keys: impl IntoIterator<Item = u64>) -> Vec<usize> {
        let mut cache = CaptureCache::default();
        let mut draws = Vec::new();
        for (frame, key) in keys.into_iter().enumerate() {
            cache.history.begin_frame();
            if let Err(admitted) = cache.lookup(key, Some("card"), false) {
                draws.push(frame);
                if let Some(key) = admitted { cache.keep_allocation(key, (), None, 4, None); }
            }
        }
        draws
    }

    #[test]
    fn motion_then_twos_recovers_on_the_first_repeated_miss() {
        let keys = (0..3).chain((3..23).flat_map(|key| [key, key]));
        let expected: Vec<_> = (0..5).chain((5..43).step_by(2)).collect();
        assert_eq!(rendered_frames(keys), expected);
    }

    #[test]
    fn one_odd_frame_between_holds_does_not_poison_later_pairs() {
        assert_eq!(rendered_frames([0, 0, 1, 2, 2, 3, 3, 4, 4]), [0, 2, 3, 4, 5, 7]);
    }

    #[test]
    fn mask_capture_admission_covers_keys_beyond_the_old_cutoff_in_any_order() {
        use std::hash::BuildHasher;
        let keys: Vec<_> = (0..20_000u64).map(|key| rustc_hash::FxBuildHasher.hash_one(key)).collect();
        let mut history = CaptureHistory::default();
        history.begin_frame();
        for &key in &keys {
            assert!(!history.observe(key));
            assert!(!history.observe(key), "a second request this frame does not qualify");
        }
        history.begin_frame();
        assert!(!history.observe(123456789), "fresh mask content stays in scratch");
        for &key in keys.iter().rev() {
            assert!(history.observe(key), "every repeated key qualifies, including late keys");
        }
        assert_eq!(history.current.len() + history.previous.len(), 8192);
        history.begin_frame();
        history.begin_frame();
        for &key in &keys { assert!(!history.observe(key), "an empty frame clears membership"); }
        // Full filters can only relax admission; cache lookup still uses the
        // original full content key. Neither membership nor storage grows.
        history.previous.fill(u64::MAX);
        assert!(history.observe(u64::MAX));
        assert_eq!(history.current.len() + history.previous.len(), 8192);
    }

    #[test]
    fn pressure_evicts_lru_within_budget() {
        let mut cache = CaptureCache::default();
        cache.set_budget(8);
        cache.keep_allocation(1, 11, None, 4, None);
        cache.keep_allocation(2, 22, None, 4, None);
        assert_eq!(cache.get(1), Some(11));
        cache.keep_allocation(3, 33, None, 4, None);
        assert_eq!(cache.get(2), None);
        assert_eq!(cache.resident(), (2, 8));
        assert!(cache.evict_oldest());
        assert_eq!(cache.get(1), None);
        cache.keep_allocation(3, 44, None, 6, None);
        cache.keep_allocation(3, 55, None, 9, None);
        assert_eq!(cache.get(3), Some(44));
        assert_eq!(cache.resident(), (1, 6));
        cache.set_budget(0);
        assert!(!cache.evict_oldest());
        assert_eq!(cache.resident(), (0, 0));
    }
}
