//! Ownership of physical capture storage. An image has exactly one charge even
//! when several layers, pages or command encoders still refer to its backing.
use super::{texture_bytes, SubmittedStorage};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

pub(crate) type Ledger = Arc<CaptureLedger>;

/// Most queries need only the total uncovered bytes. Publish it under the same lock
/// as the ownership transition so render threads never sum or traverse submissions.
#[derive(Default)]
pub(crate) struct CaptureLedger {
    state: Mutex<Accounts>,
    outstanding: AtomicU64,
    completed: AtomicU64,
}

impl CaptureLedger {
    pub(super) fn accounts(&self) -> std::sync::MutexGuard<'_, Accounts> {
        self.state.lock().unwrap()
    }

    pub(super) fn edit(&self) -> Update<'_> {
        Update {
            state: self.accounts(),
            published: &self.outstanding,
        }
    }

    pub(super) fn bytes(&self) -> u64 {
        self.outstanding.load(Ordering::Acquire)
    }

    pub(super) fn completed(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    /// Release busy bytes on the same fence that makes the pool slots reusable.
    /// A canvas with no further capture requests must also retire its generation buckets.
    pub(super) fn fence(self: &Arc<Self>, queue: &wgpu::Queue, generation: u64) {
        let weak = Arc::downgrade(self);
        queue.on_submitted_work_done(move || {
            if let Some(ledger) = weak.upgrade() {
                let mut state = ledger.accounts();
                state.complete(generation);
                ledger.completed.store(state.done, Ordering::Release);
            }
        });
    }

    pub(crate) fn cache(&self, texture: &wgpu::Texture, add: bool) {
        self.edit().owner(texture, Owner::Cache, add);
    }
}

pub(super) struct Update<'a> {
    state: std::sync::MutexGuard<'a, Accounts>,
    published: &'a AtomicU64,
}

impl std::ops::Deref for Update<'_> {
    type Target = Accounts;
    fn deref(&self) -> &Accounts {
        &self.state
    }
}

impl std::ops::DerefMut for Update<'_> {
    fn deref_mut(&mut self) -> &mut Accounts {
        &mut self.state
    }
}

impl Drop for Update<'_> {
    fn drop(&mut self) {
        let (queued, open) = self.state.outstanding(0);
        self.published.store(queued + open, Ordering::Release);
    }
}

#[derive(Clone, Copy)]
pub(super) enum Owner {
    Cache,
    Page,
    Snapshot,
    OpenRead,
    SubmittedRead,
    OpenPrivate,
    SubmittedPrivate,
}

#[derive(Default)]
struct Backing {
    bytes: u64,
    // Some(0) is a released pool slot. A generation is busy until its fence.
    pooled: Option<u64>,
    owners: [usize; 7],
    // A required outer-page allocation stays outside filter admission, even
    // when only an encoder or submission owns it.
    required_page: bool,
}

impl Backing {
    fn owner(&mut self, owner: Owner, add: bool) {
        let count = &mut self.owners[owner as usize];
        if add { *count += 1; } else { *count -= 1; }
    }

    fn charge(&self) -> Option<usize> {
        if self.pooled.is_some() {
            return None;
        }
        self.owners.iter().position(|&n| n != 0).map(|owner| {
            if self.required_page { REQUIRED_PAGE } else { owner }
        })
    }
}

const REQUIRED_PAGE: usize = 7;

#[derive(Default)]
pub(super) struct Accounts {
    backings: HashMap<wgpu::Texture, Backing>,
    buffers: HashMap<wgpu::Buffer, Backing>,
    totals: [u64; REQUIRED_PAGE + 1],
    busy: u64,
    generations: BTreeMap<u64, u64>,
    done: u64,
    next: u64,
    submissions: BTreeMap<u64, (wgpu::SubmissionIndex, SubmittedStorage)>,
}

impl Accounts {
    fn update<K: Clone + Eq + std::hash::Hash>(
        backings: &mut HashMap<K, Backing>, totals: &mut [u64; REQUIRED_PAGE + 1], key: &K,
        bytes: impl FnOnce() -> u64, change: impl FnOnce(&mut Backing),
    ) {
        let entry = backings.entry(key.clone()).or_insert_with(|| Backing { bytes: bytes(), ..Default::default() });
        if let Some(i) = entry.charge() { totals[i] -= entry.bytes; }
        change(entry);
        if let Some(i) = entry.charge() {
            totals[i] += entry.bytes;
        } else if entry.pooled.is_none() {
            backings.remove(key);
        }
    }

    fn texture(&mut self, texture: &wgpu::Texture, change: impl FnOnce(&mut Backing)) {
        Self::update(&mut self.backings, &mut self.totals, texture, || texture_bytes(texture), change);
    }

    pub(super) fn owner(&mut self, texture: &wgpu::Texture, owner: Owner, add: bool) {
        self.texture(texture, |b| b.owner(owner, add));
    }

    pub(super) fn buffer_owner(&mut self, buffer: &wgpu::Buffer, owner: Owner, add: bool) {
        Self::update(&mut self.buffers, &mut self.totals, buffer, || buffer.size(), |b| b.owner(owner, add));
    }

    pub(super) fn snapshot_texture(&mut self, texture: &wgpu::Texture, required_page: bool) {
        self.texture(texture, |b| {
            b.required_page = required_page;
            b.owner(Owner::Snapshot, true);
        });
    }

    pub(super) fn snapshot_buffer(&mut self, buffer: &wgpu::Buffer, required_page: bool) {
        Self::update(&mut self.buffers, &mut self.totals, buffer, || buffer.size(), |b| {
            b.required_page = required_page;
            b.owner(Owner::Snapshot, true);
        });
    }

    #[cfg(test)]
    pub(crate) fn required_page_bytes(&self) -> u64 {
        self.totals[REQUIRED_PAGE]
    }

    pub(super) fn busy_bytes(&mut self, gen: u64, bytes: u64, add: bool) {
        if gen <= self.done {
            return;
        }
        let count = self.generations.entry(gen).or_default();
        if add {
            *count += bytes;
            self.busy += bytes;
        } else {
            *count -= bytes;
            self.busy -= bytes;
        }
    }

    pub(super) fn pool(&mut self, texture: &wgpu::Texture, generation: Option<u64>) {
        let bytes = texture_bytes(texture);
        if let Some(gen) = self.backings.get(texture).and_then(|b| b.pooled) {
            self.busy_bytes(gen, bytes, false);
        }
        self.texture(texture, |b| b.pooled = generation);
        if let Some(gen) = generation {
            self.busy_bytes(gen, bytes, true);
        }
    }

    fn complete(&mut self, done: u64) {
        self.done = self.done.max(done);
        while self
            .generations
            .first_key_value()
            .is_some_and(|(&gen, _)| gen <= done)
        {
            self.busy -= self.generations.pop_first().unwrap().1;
        }
    }

    pub(super) fn room_used(&self, scheduled: &std::collections::HashSet<&wgpu::Texture>) -> u64 {
        let scheduled: u64 = scheduled
            .iter()
            .filter_map(|t| self.backings.get(*t))
            .filter(|b| {
                b.pooled.is_some_and(|gen| gen > self.done)
                    || b.charge() == Some(Owner::Page as usize)
            })
            .map(|b| b.bytes)
            .sum();
        self.busy + self.totals[Owner::Page as usize] + self.snapshot_bytes() - scheduled
    }

    pub(super) fn snapshot_bytes(&self) -> u64 {
        self.totals[Owner::Snapshot as usize]
    }

    pub(super) fn outstanding(&self, private_room: u64) -> (u64, u64) {
        let [_, _, _, open, submitted, open_private, submitted_private, _] = self.totals;
        (
            submitted + submitted_private.saturating_sub(private_room.saturating_sub(open_private)),
            open + open_private.saturating_sub(private_room),
        )
    }

    fn storage(&mut self, storage: &SubmittedStorage, submitted: bool, add: bool) {
        let read = if submitted {
            Owner::SubmittedRead
        } else {
            Owner::OpenRead
        };
        let private = if submitted {
            Owner::SubmittedPrivate
        } else {
            Owner::OpenPrivate
        };
        for buffer in &storage.buffers {
            self.buffer_owner(buffer, read, add);
        }
        for texture in &storage.published {
            self.owner(texture, read, add);
        }
        for texture in &storage.private {
            self.owner(texture, private, add);
        }
        // Submitted transient buffers can overlap the parent's private attachments.
        let bucket = if submitted || storage.prefix {
            private
        } else {
            read
        };
        if add {
            self.totals[bucket as usize] += storage.bytes;
        } else {
            self.totals[bucket as usize] -= storage.bytes;
        }
    }

    pub(super) fn oldest(&self) -> Option<wgpu::SubmissionIndex> {
        self.submissions
            .first_key_value()
            .map(|(_, (index, _))| index.clone())
    }
}

/// The encoder (or a detached frame prefix) owns this charge until submission
/// moves it to a fence. Discarding a command buffer releases its open reads.
pub(super) struct OpenStorage {
    ledger: Weak<CaptureLedger>,
    storage: Option<SubmittedStorage>,
}

impl OpenStorage {
    pub(super) fn new(ledger: &Ledger, storage: SubmittedStorage) -> Self {
        ledger.edit().storage(&storage, false, true);
        Self {
            ledger: Arc::downgrade(ledger),
            storage: Some(storage),
        }
    }

    pub(super) fn add_read(&mut self, ledger: &Ledger, texture: &wgpu::Texture) {
        if self.storage.as_mut().unwrap().published.insert(texture.clone()) {
            ledger.edit().owner(texture, Owner::OpenRead, true);
        }
    }

    pub(super) fn attach(self, encoder: &wgpu::CommandEncoder, queue: &wgpu::Queue) {
        let queue = queue.clone();
        encoder.on_submitted(move |index| self.submit(&queue, index));
    }

    pub(super) fn from_prefix(ledger: &Ledger, storage: SubmittedStorage) -> Self {
        ledger.edit().totals[Owner::OpenPrivate as usize] += storage.bytes;
        Self {
            ledger: Arc::downgrade(ledger),
            storage: Some(storage),
        }
    }

    pub(super) fn submit(mut self, queue: &wgpu::Queue, index: wgpu::SubmissionIndex) {
        let storage = self.storage.take().unwrap();
        let Some(ledger) = self.ledger.upgrade() else {
            return;
        };
        let token = {
            let mut l = ledger.edit();
            // Add first, so a backing that changes owners keeps its entry and size.
            l.storage(&storage, true, true);
            l.storage(&storage, false, false);
            l.next += 1;
            let token = l.next;
            l.submissions.insert(token, (index, storage));
            token
        };
        let ledger = self.ledger.clone();
        queue.on_submitted_work_done(move || {
            if let Some(ledger) = ledger.upgrade() {
                let mut l = ledger.edit();
                if let Some((_, storage)) = l.submissions.remove(&token) {
                    l.storage(&storage, true, false);
                }
            }
        });
    }
}

impl Drop for OpenStorage {
    fn drop(&mut self) {
        if let (Some(storage), Some(ledger)) = (&self.storage, self.ledger.upgrade()) {
            ledger.edit().storage(storage, false, false);
        }
    }
}
