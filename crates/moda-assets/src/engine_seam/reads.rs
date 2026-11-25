//! Asset inputs observed while planning and drawing a deferred capture. The isolated
//! camera planner resolves paints synchronously. A nested capture contributes its
//! inputs even when its own pixels come from the capture cache.
use super::{slot, store_generation};
use crate::{AssetEntry, AssetStore, ExternalImageMeta, PreparedImage};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Arc, Weak};

#[derive(Default)]
struct Names {
    keys: BTreeMap<String, Read>,
    whole_store: bool,
}

thread_local! {
    static ACTIVE: RefCell<Vec<Names>> = const { RefCell::new(Vec::new()) };
}

// Reading pixels while constructing another value makes rebinding the texture
// insufficient. Multiple reads of one key keep their strongest dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Read { Binding, Pixels }

pub(super) fn observe(key: &str, read: Read) {
    ACTIVE.with_borrow_mut(|active| {
        for names in active {
            match names.keys.get_mut(key) {
                Some(old) if read == Read::Pixels => *old = read,
                Some(_) => {}
                None => { names.keys.insert(key.to_owned(), read); }
            }
        }
    });
}

pub(super) fn observe_store() {
    ACTIVE.with_borrow_mut(|active| {
        for names in active { names.whole_store = true; }
    });
}

/// Bound observation to this thread. Early returns unwind exactly one nested scope.
pub struct CaptureReads {
    generation: u64,
    thread: PhantomData<Rc<()>>,
}

impl CaptureReads {
    pub fn begin() -> Self {
        ACTIVE.with_borrow_mut(|active| active.push(Names::default()));
        Self { generation: store_generation(), thread: PhantomData }
    }

    pub fn finish(self) -> Option<AssetReads> {
        let guard = slot().read().expect("asset store lock");
        // An intervening publication could mix two snapshots in one picture.
        if self.generation != store_generation() { return None; }
        let names = ACTIVE.with_borrow_mut(|active| std::mem::take(active.last_mut().unwrap()));
        Some(AssetReads {
            entries: names.keys.into_iter().map(|(key, read)| {
                let state = State::of(guard.as_deref(), &key);
                (key, state, read)
            }).collect(),
            generation: names.whole_store.then_some(self.generation),
        })
    }
}

impl Drop for CaptureReads {
    fn drop(&mut self) { ACTIVE.with_borrow_mut(|active| { active.pop(); }); }
}

#[derive(Debug)]
pub struct Publication(State);

impl Publication {
    pub fn of(store: &AssetStore, key: &str) -> Self { Self(State::of(Some(store), key)) }
    pub fn matches(&self, store: &AssetStore, key: &str) -> bool { self.0.matches(Some(store), key) }
}

#[derive(Debug)]
enum State {
    NoStore,
    Missing,
    Failed,
    Heap(Weak<PreparedImage>),
    External(ExternalImageMeta),
    Device(ExternalImageMeta),
}

impl State {
    fn of(store: Option<&AssetStore>, key: &str) -> Self {
        let Some(store) = store else { return Self::NoStore };
        match store.entries.get(key) {
            None => Self::Missing,
            Some(AssetEntry::Failed) => Self::Failed,
            Some(AssetEntry::Loaded(image)) => Self::Heap(Arc::downgrade(image)),
            Some(AssetEntry::LoadedExternal(meta)) => Self::External(meta.clone()),
            Some(AssetEntry::DeviceFrame(meta, _)) => Self::Device(meta.clone()),
        }
    }

    fn matches(&self, store: Option<&AssetStore>, key: &str) -> bool {
        let Some(store) = store else { return matches!(self, Self::NoStore) };
        match (self, store.entries.get(key)) {
            (Self::Missing, None) | (Self::Failed, Some(AssetEntry::Failed)) => true,
            (Self::Heap(stamp), Some(AssetEntry::Loaded(image))) => stamp.as_ptr() == Arc::as_ptr(image),
            (Self::External(stamp), Some(AssetEntry::LoadedExternal(meta)))
            | (Self::Device(stamp), Some(AssetEntry::DeviceFrame(meta, _))) => stamp == meta,
            _ => false,
        }
    }
}

#[derive(Default, Debug)]
pub struct AssetReads {
    entries: Vec<(String, State, Read)>,
    // A host callback may ask for the entire store instead of resolving a key.
    generation: Option<u64>,
}

impl AssetReads {
    /// Used only to decide whether a result recurred often enough to keep. Actual
    /// cache hits check each immutable resource with `current`.
    pub fn hash_for_admission(&self, h: &mut impl std::hash::Hasher) {
        use std::hash::Hash;
        self.generation.hash(h);
        for (key, state, _) in &self.entries {
            key.hash(h);
            std::mem::discriminant(state).hash(h);
            match state {
                State::Heap(stamp) => stamp.as_ptr().hash(h),
                State::Device(meta) | State::External(meta) => {
                    (meta.width, meta.height, meta.scale_factor.0.to_bits(), meta.scale_factor.1.to_bits(), meta.upload_id).hash(h);
                }
                _ => {}
            }
        }
    }

    pub fn is_empty(&self) -> bool { self.entries.is_empty() && self.generation.is_none() }

    pub fn retained_bytes(&self) -> usize {
        self.entries.capacity() * std::mem::size_of::<(String, State, Read)>()
            + self.entries.iter().map(|(key, _, _)| key.capacity()).sum::<usize>()
    }

    /// Check against the store captured by a publication update, without recording a render read.
    pub fn matches(&self, store: &AssetStore) -> bool {
        self.matches_store(Some(store))
    }

    fn matches_store(&self, store: Option<&AssetStore>) -> bool {
        self.generation.is_none_or(|g| g == store_generation())
            && self.entries.iter().all(|(key, state, _)| state.matches(store, key))
    }

    /// Reuse prepared output while its caller updates the named, same-layout
    /// image bindings. Every other observed input must still have its old value.
    /// Enclosing captures observe the current publications, including stream misses.
    pub fn with_rebound_images<T>(&self, rebound: impl Fn(&str) -> bool,
        update: impl FnOnce(Option<&Arc<AssetStore>>) -> T) -> Option<T> {
        let guard = slot().read().expect("asset store lock");
        let store = guard.as_deref();
        if self.generation.is_some_and(|g| g != store_generation()) ||
            self.entries.iter().any(|(key, state, read)|
                (*read != Read::Binding || !rebound(key)) && !state.matches(store, key)) {
            return None;
        }
        if self.generation.is_some() { observe_store(); }
        for (key, _, read) in &self.entries { observe(key, *read); }
        Some(update(guard.as_ref()))
    }

    pub fn current(&self) -> bool {
        if self.entries.is_empty() && self.generation.is_none() { return true; }
        let guard = slot().read().expect("asset store lock");
        if !self.matches_store(guard.as_deref()) { return false; }
        if self.generation.is_some() { observe_store(); }
        // A cached picture already contains these pixels, even if its plan
        // originally observed them through a replaceable texture binding.
        for (key, _, _) in &self.entries { observe(key, Read::Pixels); }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn rebinding_preserves_other_readers_of_the_same_pixels() {
        let _global = crate::report::tests::serial_globals();
        let entry = |v| crate::AssetEntry::Loaded(Arc::new(
            crate::PreparedImage::from_premultiplied(vec![v; 4], 1, 1)));
        let mut store = AssetStore::default();
        store.insert("shared".into(), entry(12));
        install_store(Arc::new(store.clone()));
        let capture = |both| {
            let reads = CaptureReads::begin();
            resolve_pattern_binding("shared").unwrap();
            if both { resolve("shared").unwrap(); }
            reads.finish().unwrap()
        };
        let binding = capture(false);
        let mixed = capture(true);
        store.insert("shared".into(), entry(64));
        install_store(Arc::new(store));
        let outer = CaptureReads::begin();
        assert!(binding.with_rebound_images(|_| true, |_| ()).is_some());
        assert!(mixed.with_rebound_images(|_| true, |_| ()).is_none());
        let outer = outer.finish().unwrap();
        assert!(outer.current());
        assert!(!binding.current());
        clear_store();
    }

    #[test]
    fn cached_child_reads_follow_publications_including_a_missing_stream() {
        let _global = crate::report::tests::serial_globals();
        clear_store();
        let poster = Arc::new(crate::PreparedImage::from_premultiplied(vec![255; 4], 1, 1));
        let mut store = AssetStore::default();
        store.insert("poster".into(), crate::AssetEntry::Loaded(poster.clone()));
        install_store(Arc::new(store.clone()));

        let child = CaptureReads::begin();
        assert!(resolve_streaming("video:clip").is_none());
        assert!(resolve("poster").is_some());
        let child = child.finish().unwrap();

        let parent = CaptureReads::begin();
        assert!(child.current());
        let parent = parent.finish().unwrap();
        store.insert("elsewhere".into(), crate::AssetEntry::Failed);
        install_store(Arc::new(store.clone()));
        assert!(parent.current());
        store.insert("video:clip".into(), crate::AssetEntry::Loaded(poster));
        install_store(Arc::new(store));
        assert!(!parent.current());
        assert!(!child.current());
        clear_store();
    }
}
