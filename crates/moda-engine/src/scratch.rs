//! GPU buffers and textures a render makes and drops (uniforms, vertex streams, a layer capture's target and
//! filter passes), kept and handed out again once the GPU is done with them. Every engine's renders use it: a
//! retained engine keeps its outermost frame's buffers and targets itself, and its captures take theirs here.
//!
//! An export renders thousands of small captures a frame (a shadow per shard of a motion-blurred fracture, the
//! effects of a diorama's planes), and every one created its buffers and textures afresh: on a paravirtual GPU each
//! creation is a host round trip (shatter-1 on the macOS runner: 7.8 M buffers and 1.8 M textures, 154 s of 998 s).
//!
//! A resource is taken inside an outermost render (one generation) and goes back to the pool when that render's
//! work is done on the GPU (`Queue::on_submitted_work_done`, which a later poll runs). Keys are exact: the same
//! size, format and usage, so a reused resource is indistinguishable from a fresh one to everything that writes it
//! in full before reading it (every caller here: a buffer is written whole or bound at the length written, a
//! texture is a render target cleared or covered by its pass). Outside a render nothing is pooled. What rests
//! unused is kept up to the bytes a generation takes, most recently taken first ([`evict_resting`]). What must
//! outlive its generation (a capture result an engine keeps for later frames) is withdrawn ([`ScratchPool::keep`]).
//!
//! A bind group over pooled resources is kept under what it binds, its layout and those resources
//! ([`ScratchPool::bind`]), for as long as the pool holds them all: a frame that takes the same resources for the
//! same passes as an earlier one makes no bind group. A pooled texture has one default view
//! ([`ScratchPool::view`]), so a bind group names it the same way every time.
//!
//! The first dirty retained capture starts a shared encoder. Source draws, filters and the outer pass return it
//! unfinished so the following stage can append commands. Submitting an independently recorded command buffer
//! seals the encoder to preserve ordering. The outer frame submits the recorded work, and operations that need
//! completed GPU results flush before waiting. Scratch acquired by an outstanding capture stays in the frame
//! generation until submission completes.
//!
//! The writes a held command buffer reads ([`write_buffer`]) are held with it. `Queue::write_buffer` makes a staging
//! buffer for every call, and the queue frees each once its submission is done: on Metal a kernel round trip apiece
//! (shatter-1 on the macOS runner: 13% of the render thread). Held writes are copied into one staging buffer, and a
//! command buffer at the front of the submission copies them into place, in the order they were made. They land
//! where the queue's own writes would, before every command buffer of that submission.

mod ledger;
use ledger::{Ledger, OpenStorage, Owner};
pub(crate) use ledger::CaptureLedger;
use moda_text_draw::device::estimated_texture_bytes;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

thread_local! {
    static UPLOAD_PAGES: RefCell<Vec<UploadPage>> = const { RefCell::new(Vec::new()) };
    static WRITE_SIZES: RefCell<HashMap<wgpu::Queue, u64>> = RefCell::new(HashMap::new());
    static RECORDING: RefCell<Option<(wgpu::Queue, wgpu::CommandEncoder)>> = const { RefCell::new(None) };
    /// Renders open on this thread that hold their command buffers, and the command buffers they have submitted so far.
    static HELD: RefCell<(u32, Vec<(wgpu::Queue, wgpu::CommandBuffer)>)> = const { RefCell::new((0, Vec::new())) };
    /// The writes held with them ([`write_buffer`]), in the order they were made, and their bytes end to end.
    static WRITES: RefCell<(Vec<HeldWrite>, Vec<u8>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// A frame's immutable filter and composite parameters. Bind groups can retain a range after
/// submission. A later frame allocates its own page, so replay reads the original values.
struct UploadPage {
    pool: usize,
    lasting: bool,
    usage: wgpu::BufferUsages,
    queue: wgpu::Queue,
    buffer: wgpu::Buffer,
    bytes: Vec<u8>,
}

const UPLOAD_PAGE_BYTES: u64 = 64 * 1024;

/// Layout used both by admission and by upload_range. The destination buffer
/// is charged on its first range. Every new byte written (including alignment
/// padding) then needs staging until the queue has consumed it.
#[derive(Clone, PartialEq, Default)]
pub(crate) struct UploadLayout {
    pages: Vec<(u64, u64)>,
    alignment: u64,
    pub(crate) additional: u64,
}
impl UploadLayout {
    pub(crate) fn empty(&self) -> Self { Self { alignment: self.alignment, ..Default::default() } }

    pub(crate) fn occupied(&self) -> u64 { self.pages.iter().map(|(used, capacity)| used + capacity).sum() }

    fn offset(used: u64, alignment: u64) -> u64 { used.next_multiple_of(alignment.max(1)) }

    pub(crate) fn add(&mut self, size: u64) {
        if size == 0 { return; }
        let alignment = self.alignment;
        if let Some((used, _)) = self.pages.iter_mut().rfind(|(used, cap)| Self::offset(*used, alignment) + size <= *cap) {
            let end = Self::offset(*used, alignment) + size;
            self.additional += end - *used;
            *used = end;
        } else {
            let capacity = size.max(UPLOAD_PAGE_BYTES);
            self.pages.push((size, capacity));
            self.additional += capacity + size;
        }
    }
}

fn open_staging(queue: &wgpu::Queue) -> u64 {
    WRITE_SIZES.with(|sizes| sizes.borrow().get(queue).copied().unwrap_or(0))
}

/// One held [`write_buffer`]: `len` bytes from `at` in [`WRITES`]' bytes, for `buffer` at `offset`.
struct HeldWrite {
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffer: wgpu::Buffer,
    offset: u64,
    at: usize,
    len: usize,
}

/// `queue.write_buffer(buffer, offset, data)`: held with the command buffers while a render is open on this thread
/// (they are the ones that read it), at once otherwise.
pub(crate) fn write_buffer(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
    if data.is_empty() || HELD.with(|h| h.borrow().0 == 0) {
        queue.write_buffer(buffer, offset, data);
        return;
    }
    WRITE_SIZES.with(|s| *s.borrow_mut().entry(queue.clone()).or_default() += data.len() as u64);
    WRITES.with(|w| {
        let mut w = w.borrow_mut();
        let at = w.1.len();
        w.1.extend_from_slice(data);
        w.0.push(HeldWrite { device: device.clone(), queue: queue.clone(), buffer: buffer.clone(), offset, at, len: data.len() });
    });
}

/// The held writes as command buffers, one per queue: each copies that queue's writes out of one staging buffer, in
/// order. Sizes and offsets are multiples of 4, as `write_buffer` requires.
fn held_writes() -> Vec<(wgpu::Queue, wgpu::CommandBuffer)> {
    let _uploads = crate::prof::span("fx.finish_uploads");
    WRITE_SIZES.with(|s| s.borrow_mut().clear());
    let (writes, bytes) = WRITES.with(|w| std::mem::take(&mut *w.borrow_mut()));
    let mut out: Vec<(wgpu::Queue, wgpu::CommandBuffer)> = Vec::new();
    let mut queues: Vec<&wgpu::Queue> = Vec::new();
    for w in &writes {
        if !queues.contains(&&w.queue) {
            queues.push(&w.queue);
        }
    }
    for queue in queues {
        let mine: Vec<&HeldWrite> = writes.iter().filter(|w| w.queue == *queue).collect();
        let device = &mine[0].device;
        let size: usize = mine.iter().map(|w| w.len).sum();
        let mut data = Vec::with_capacity(size);
        for w in &mine {
            data.extend_from_slice(&bytes[w.at..w.at + w.len]);
        }
        let staging = wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor {
            label: Some("held writes"), contents: &data, usage: wgpu::BufferUsages::COPY_SRC,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("held writes") });
        let mut at = 0u64;
        for w in &mine {
            encoder.copy_buffer_to_buffer(&staging, at, &w.buffer, w.offset, w.len as u64);
            at += w.len as u64;
        }
        out.push((queue.clone(), encoder.finish()));
    }
    out
}

/// Continue the current frame's encoder after a capture or filter pass.
pub(crate) fn encoder(device: &wgpu::Device, queue: &wgpu::Queue, desc: &wgpu::CommandEncoderDescriptor<'_>) -> wgpu::CommandEncoder {
    let previous = RECORDING.with(|slot| slot.take());
    if let Some((owner, encoder)) = previous {
        if owner == *queue { return encoder; }
        submit(&owner, [encoder.finish()]);
    }
    device.create_command_encoder(desc)
}

pub(crate) fn end_recording(queue: &wgpu::Queue, encoder: wgpu::CommandEncoder) {
    if HELD.with(|h| h.borrow().0 > 0) {
        seal_recording();
        RECORDING.with(|slot| slot.replace(Some((queue.clone(), encoder))));
    } else {
        submit(queue, [encoder.finish()]);
    }
}

fn seal_recording() {
    if let Some((queue, encoder)) = RECORDING.with(|slot| slot.take()) {
        let _finish = crate::prof::span("fx.finish_encoder");
        let buffer = encoder.finish();
        HELD.with(|h| h.borrow_mut().1.push((queue, buffer)));
    }
}

/// Submit `buffers` to `queue`: held until the outermost holding render on this thread ends while one is open, at
/// once otherwise.
pub(crate) fn submit<I: IntoIterator<Item = wgpu::CommandBuffer>>(queue: &wgpu::Queue, buffers: I) {
    seal_recording();
    let mut buffers: Vec<wgpu::CommandBuffer> = buffers.into_iter().collect();
    HELD.with(|h| {
        let mut h = h.borrow_mut();
        if h.0 > 0 {
            h.1.extend(buffers.drain(..).map(|b| (queue.clone(), b)));
        }
    });
    if !buffers.is_empty() {
        let _submit = crate::prof::span("fx.submit");
        queue.submit(buffers);
    }
}

/// Submit everything held on this thread, in order (before a wait for the GPU, and when the outermost render ends).
pub fn flush() {
    if std::thread::panicking() { discard_pending(); return; }
    let _ = submit_runs(take_recorded(), None);
}

fn take_recorded() -> Vec<(wgpu::Queue, wgpu::CommandBuffer)> {
    // A queue's held writes go first, where its own writes would land.
    seal_recording();
    let pages = UPLOAD_PAGES.with(|pages| pages.take());
    for page in pages {
        page.queue.write_buffer(&page.buffer, 0, &page.bytes);
    }
    let mut held = held_writes();
    held.extend(HELD.with(|h| std::mem::take(&mut h.borrow_mut().1)));
    held
}

/// Submit contiguous commands for each queue in order. A caller limiting transient
/// memory can retire its own last submission without waiting for later device work.
fn submit_runs(
    held: Vec<(wgpu::Queue, wgpu::CommandBuffer)>,
    track: Option<&wgpu::Queue>,
) -> Option<wgpu::SubmissionIndex> {
    let mut held = held.into_iter().peekable();
    let mut last = None;
    while let Some((queue, first)) = held.next() {
        let run = std::iter::once(first).chain(std::iter::from_fn(|| {
            held.next_if(|(owner, _)| owner == &queue).map(|(_, buffer)| buffer)
        }));
        let _submit = crate::prof::span("fx.submit");
        let index = queue.submit(run);
        if track == Some(&queue) {
            last = Some(index);
        }
    }
    last
}

// A frame can inspect the last dependent tree while its output owner publishes the fence.
type TreeSubmission = Arc<std::sync::Mutex<Option<(u64, wgpu::SubmissionIndex)>>>;
const CAPTURE_PAGES_PER_SEGMENT: usize = 8;
const MAX_CAPTURE_SEGMENTS: usize = 8;

/// Scratch evictions and published images whose last ordered consumer is this submission.
#[derive(Default)]
struct SubmittedStorage {
    buffers: Vec<wgpu::Buffer>,
    prefix: bool,
    bytes: u64,
    published: std::collections::HashSet<wgpu::Texture>,
    private: std::collections::HashSet<wgpu::Texture>,
}

/// The finished canvas commands and their frame-arena fence. The output encoder follows these
/// commands in a single queue submission. Dropping the prefix still publishes the canvas work
/// on a recoverable output error, because the retained renderer can reuse that picture.
pub struct FramePrefix {
    queue: wgpu::Queue,
    buffers: Vec<wgpu::CommandBuffer>,
    completion: Option<(Ledger, u64)>,
    reusable_tree: Option<TreeSubmission>,
    storage: Option<OpenStorage>,
}

impl FramePrefix {
    /// Append a consumer of this canvas before its final output conversion.
    pub fn append(&mut self, commands: wgpu::CommandBuffer) {
        self.buffers.push(commands);
    }

    pub fn submit(mut self, output: wgpu::CommandBuffer) -> wgpu::SubmissionIndex {
        let _profile = crate::prof::span("fx.submit");
        self.buffers.push(output);
        let index = self.queue.submit(std::mem::take(&mut self.buffers));
        if let (Some(fence), Some((_, generation))) = (&self.reusable_tree, &self.completion) {
            *fence.lock().unwrap() = Some((*generation, index.clone()));
        }
        self.register_completion(index.clone());
        index
    }

    fn register_completion(&mut self, index: wgpu::SubmissionIndex) {
        if let Some(storage) = self.storage.take() { storage.submit(&self.queue, index); }
        if let Some((ledger, generation)) = self.completion.take() {
            ledger.fence(&self.queue, generation);
        }
    }
}

impl Drop for FramePrefix {
    fn drop(&mut self) {
        if self.completion.is_some() && !std::thread::panicking() {
            let index = self.queue.submit(std::mem::take(&mut self.buffers));
            self.register_completion(index);
        }
    }
}

/// Captures the outer draw's commands for an immediately following output encoder on the same queue.
pub struct OutputRecording<'a> { pool: &'a ScratchPool, owner: bool }
impl OutputRecording<'_> {
    pub fn finish(self) -> Option<FramePrefix> { self.owner.then(|| self.pool.output_prefix.take()).flatten() }
}
impl Drop for OutputRecording<'_> {
    fn drop(&mut self) {
        if self.owner {
            self.pool.output_requested.set(false);
            drop(self.pool.output_prefix.take());
        }
    }
}

/// Cancel a disposable render without allocating staging or submitting partial
/// commands. Drop handles outside TLS borrows because their cleanup can inspect it.
pub fn discard_pending() {
    WRITE_SIZES.with(|s| s.borrow_mut().clear());
    UPLOAD_PAGES.with(|pages| pages.take());
    RECORDING.with(|slot| slot.take());
    let held = HELD.with(|held| std::mem::take(&mut *held.borrow_mut()));
    let writes = WRITES.with(|writes| std::mem::take(&mut *writes.borrow_mut()));
    drop(held); drop(writes);
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;
    #[test]
    fn quota_unwind_discards_a_deferred_render_without_another_allocation() {
        struct FlushOnDrop;
        impl Drop for FlushOnDrop { fn drop(&mut self) { flush(); } }
        let budget = wgpu::optional_allocations::Budget::new(65536, || Some(65536));
        assert!(budget.run(|| {
            HELD.with(|held| held.borrow_mut().0 = 1);
            WRITES.with(|writes| writes.borrow_mut().1.extend_from_slice(&[1, 2, 3, 4]));
            let _flush = FlushOnDrop;
            std::panic::resume_unwind(Box::new(wgpu::optional_allocations::Refused));
        }).is_err());
        HELD.with(|held| assert_eq!(held.borrow().0, 0));
        WRITES.with(|writes| assert!(writes.borrow().0.is_empty() && writes.borrow().1.is_empty()));
    }
}

/// A GPU resource an explicit `destroy` frees at once.
pub(crate) trait Destroy {
    fn destroy_now(&self);
}

impl Destroy for wgpu::Texture {
    fn destroy_now(&self) {
        self.destroy();
    }
}

impl Destroy for wgpu::Buffer {
    fn destroy_now(&self) {
        self.destroy();
    }
}

/// `resource.destroy()`, except while this thread holds command buffers ([`submit`]) or writes ([`write_buffer`]): one
/// of them may use it, so it is only dropped (freed once the GPU is done with it).
/// Destructors can run after either TLS key is destroyed, or while scratch is mutably borrowed.
/// Neither case proves that pending work is empty: leave release to wgpu's handle ownership.
pub(crate) fn destroy<R: Destroy>(resource: &R) {
    if HELD.try_with(|h| h.try_borrow().is_ok_and(|h| h.0 == 0 && h.1.is_empty())).unwrap_or(false)
        && WRITES.try_with(|w| w.try_borrow().is_ok_and(|w| w.0.is_empty())).unwrap_or(false)
    {
        resource.destroy_now();
    }
}

/// One more `name` in the export profile (`MODA_RENDER_PROFILE`), for a pooled take: `reused`, or made because no
/// resource had its key (`new_key`) or every one with it was still in use on the GPU (`all_busy`).
fn count(name: &'static str) {
    if crate::prof::on() {
        crate::prof::add_work(name, 0, 1);
    }
}

/// How many generations an unused bind group is kept before it is dropped.
const KEEP_GENERATIONS: u64 = 64;

/// The native allocation size is shared by every view and clone of this backing.
pub(crate) fn texture_bytes(texture: &wgpu::Texture) -> u64 {
    texture.binding_size().unwrap_or_else(|| estimated_texture_bytes(texture.size(), texture.dimension(),
        texture.format(), texture.mip_level_count(), texture.sample_count()))
}

/// Drop the resting resources of `map` beyond `budget` bytes, least recently taken first. Resting: last taken
/// before generation `done`, the newest finished one. Whole generations are kept, newest first, while they fit.
fn evict_resting<K: std::hash::Hash + Eq, T>(
    map: &mut HashMap<K, VecDeque<Slot<T>>>,
    done: u64,
    budget: u64,
    released: bool,
    bytes: impl Fn(&T) -> u64,
    mut dropped: impl FnMut(&Slot<T>),
) {
    let mut by_gen: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
    for list in map.values() {
        for slot in list.iter().filter(|s| s.gen < done || (released && s.free)) {
            *by_gen.entry(slot.gen).or_default() += bytes(&slot.item);
        }
    }
    // Everything last taken at or before `cutoff` goes (generations count from 1).
    let (mut kept, mut cutoff) = (0u64, 0u64);
    for (&gen, &size) in by_gen.iter().rev() {
        if kept + size > budget {
            cutoff = gen;
            break;
        }
        kept += size;
    }
    if cutoff == 0 {
        return;
    }
    for list in map.values_mut() {
        list.retain(|slot| {
            let goes = slot.gen <= cutoff && (slot.gen < done || (released && slot.free));
            if goes {
                dropped(slot);
            }
            !goes
        });
    }
    map.retain(|_, list| !list.is_empty());
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct TexKey {
    w: u32,
    h: u32,
    layers: u32,
    mips: u32,
    samples: u32,
    dim: u8,
    format: wgpu::TextureFormat,
    usage: u32,
}

impl TexKey {
    pub(crate) fn capture(width: u32, height: u32, layers: u32, samples: u32,
        format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> Self {
        Self { w: width, h: height, layers, mips: 1, samples, dim: wgpu::TextureDimension::D2 as u8,
            format, usage: usage.bits() }
    }
    /// The key `texture` is pooled under, if it is.
    fn of(texture: &wgpu::Texture) -> TexKey {
        TexKey {
            w: texture.width(),
            h: texture.height(),
            layers: texture.depth_or_array_layers(),
            mips: texture.mip_level_count(),
            samples: texture.sample_count(),
            dim: texture.dimension() as u8,
            format: texture.format(),
            usage: texture.usage().bits(),
        }
    }
}

/// One pooled resource: the generation that last took it, and whether it has been handed back since
/// ([`ScratchPool::release`]: free now, whatever the GPU has finished, because whatever uses it next is queued
/// behind what used it last).
struct Slot<T> {
    gen: u64,
    free: bool,
    item: T,
}

/// What the pool knows of a texture it holds beside its slot: its default view, made when first asked for
/// ([`ScratchPool::view`]), and the kept bind groups that name it (the binding map owns their keys: a key names
/// every resource of its group, so an owning reference here would keep a withdrawn texture alive through another).
#[derive(Default)]
struct HeldTexture {
    array_view: Option<wgpu::TextureView>,
    view: Option<wgpu::TextureView>,
    layers: Vec<Option<wgpu::TextureView>>,
    bindings: Vec<Weak<BindKey>>,
}

/// One resource of a bind group ([`ScratchPool::bind`]): a buffer bound whole, a texture view or a sampler.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum Bound {
    Buffer(wgpu::Buffer),
    Range(wgpu::Buffer, u64, wgpu::BufferSize),
    View(wgpu::TextureView),
    Sampler(wgpu::Sampler),
}
impl Bound {
    pub(crate) fn vertex_slice(&self) -> wgpu::BufferSlice<'_> {
        match self {
            Self::Range(buffer,offset,size) => buffer.slice(*offset..offset + size.get()),
            Self::Buffer(buffer) => buffer.slice(..),
            _ => unreachable!("a vertex upload is a buffer"),
        }
    }
}


/// What a kept bind group binds: its layout and resources, binding `i` the `i`th.
type BindKey = (wgpu::BindGroupLayout, Vec<Bound>);

/// A position in the open generation's takes ([`ScratchPool::mark`]).
#[derive(Clone, Copy)]
pub(crate) struct Scope(usize, usize);

/// A kept bind group and the generation that last used it.
struct Binding {
    gen: u64,
    group: wgpu::BindGroup,
}

#[derive(Default)]
pub struct ScratchPool {
    headroom: Option<Box<dyn Fn() -> Option<u64>>>,
    reclaim: Option<Box<dyn Fn(u64)>>,
    allocation_credit: Cell<Option<u64>>,
    headroom_sampled: Cell<bool>,
    depth: Cell<u32>,
    recording_captures: Cell<bool>,
    /// Removed arena slots whose last use is still in the recorded prefix.
    retiring_textures: RefCell<std::collections::HashSet<wgpu::Texture>>,
    capture_spare: Cell<u64>,
    texture_prices: RefCell<HashMap<TexKey, (u64, u64)>>,
    capture_requests: RefCell<HashMap<TexKey, usize>>,
    pub(crate) ledger: Ledger,
    recorded_pages: Cell<usize>,
    dependent_renders: Cell<usize>,
    previous_tree: TreeSubmission,
    output_requested: Cell<bool>,
    output_prefix: RefCell<Option<FramePrefix>>,
    gen: Cell<u64>,
    /// Per key, the pooled resources: those handed back ([`ScratchPool::release`]) at the front, then the rest in
    /// the order they were last taken. The front alone says whether any is free, and a take is the front moved to
    /// the back or a new one at the back.
    buffers: RefCell<HashMap<(u64, u32), VecDeque<Slot<wgpu::Buffer>>>>,
    textures: RefCell<HashMap<TexKey, VecDeque<Slot<wgpu::Texture>>>>,
    /// The buffers and the textures the open generation has taken and not handed back, in order: a position in
    /// each is a scope ([`ScratchPool::mark`], [`ScratchPool::release`]).
    scope: RefCell<(Vec<wgpu::Buffer>, Vec<wgpu::Texture>)>,
    /// The distinct bytes of buffers and textures the open generation has taken: what may rest in the pool once
    /// it ends ([`evict_resting`]).
    taken: Cell<[u64; 2]>,
    /// Every buffer and texture in the queues above, for a direct answer to "does the pool hold this?", each with
    /// the kept bind groups that name it (weakly: the binding map owns their keys).
    held_buffers: RefCell<HashMap<wgpu::Buffer, Vec<Weak<BindKey>>>>,
    held_textures: RefCell<HashMap<wgpu::Texture, HeldTexture>>,
    /// The buffer of each usage that stands for "no data" ([`ScratchPool::empty`]). Nothing writes or reads it, so
    /// every render shares it, in any generation; it counts as held, so its bind group is kept.
    empties: RefCell<HashMap<u32, wgpu::Buffer>>,
    /// Bind groups by what they bind. One is returned only while the pool holds every buffer and texture it names,
    /// and goes with them or once [`KEEP_GENERATIONS`] have not used it.
    bindings: RefCell<HashMap<Arc<BindKey>, Binding>>,
}

/// An independently owned capture target. A recorded command may keep it across
/// scratch generations. Encoder reads take over the charge if its owner drops first.
pub(crate) struct PageTexture {
    pub view: wgpu::TextureView,
    ledger: Ledger,
}

impl Drop for PageTexture {
    fn drop(&mut self) {
        self.ledger.edit().owner(self.view.texture(), Owner::Snapshot, false);
    }
}

pub(crate) struct PageBuffer {
    pub buffer: wgpu::Buffer,
    ledger: Ledger,
}

impl Drop for PageBuffer {
    fn drop(&mut self) { self.ledger.edit().buffer_owner(&self.buffer, Owner::Snapshot, false); }
}

/// Reads accumulated by one page command encoder. Charge a new identity
/// immediately so admission sees it even before this encoder is finished.
#[derive(Default)]
pub(crate) struct EncoderReads(Option<OpenStorage>);

impl EncoderReads {
    pub(crate) fn track(&mut self, pool: &ScratchPool, texture: &wgpu::Texture) {
        if let Some(reads) = &mut self.0 {
            reads.add_read(&pool.ledger, texture);
        } else {
            let storage = SubmittedStorage {
                published: [texture.clone()].into(),
                ..Default::default()
            };
            self.0 = Some(OpenStorage::new(&pool.ledger, storage));
        }
    }

    pub(crate) fn attach(&mut self, encoder: &wgpu::CommandEncoder, queue: &wgpu::Queue) {
        if let Some(reads) = self.0.take() { reads.attach(encoder, queue); }
    }
}

pub(crate) struct PageOwners<'a> {
    pool: &'a ScratchPool,
    views: Vec<wgpu::Texture>,
}

impl PageOwners<'_> {
    pub(crate) fn hold(&mut self, view: &wgpu::TextureView) {
        let texture = view.texture();
        self.pool.ledger.edit().owner(texture, Owner::Page, true);
        self.views.push(texture.clone());
    }
}

impl Drop for PageOwners<'_> {
    fn drop(&mut self) {
        if self.views.is_empty() { return; }
        let mut ledger = self.pool.ledger.edit();
        for texture in &self.views { ledger.owner(texture, Owner::Page, false); }
    }
}

impl ScratchPool {
    /// A render starts. The outermost one opens a generation, after a non-blocking poll so the work that finished
    /// since hands its resources back. `hold`: its command buffers and the writes they read wait for the outermost
    /// holding render on this thread to end ([`submit`]); [`ScratchPool::leave`] takes the same value.
    pub(crate) fn enter(&self, device: &wgpu::Device, hold: bool) {
        if hold {
            HELD.with(|h| h.borrow_mut().0 += 1);
        }
        let d = self.depth.get();
        self.depth.set(d + 1);
        if d > 0 {
            return;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        self.recorded_pages.set(0);
        self.dependent_renders.set(0);
        let gen = self.gen.get() + 1;
        self.gen.set(gen);
        // What rests in the pool (free, and not taken by the newest finished generation or since) is kept up to the
        // bytes the generation that just ended took, most recently taken first: sizes that come back after a while
        // are still here if they fit beside a frame's own working set, and a clip whose sizes never repeat holds
        // about one working set of them, not every size it has drawn.
        *self.scope.borrow_mut() = Default::default();
        let done = self.ledger.completed();
        let [buffer_budget, texture_budget] = self.taken.replace([0, 0]);
        self.trim(done, buffer_budget, texture_budget, false);
        // Reuse-only generations need no native budget query. The first new
        // allocation samples current headroom, after any earlier frame work.
        self.allocation_credit.set(None);
        self.headroom_sampled.set(false);
        if gen % KEEP_GENERATIONS == 0 {
            self.texture_prices.borrow_mut().retain(|_, (_, last)| *last + KEEP_GENERATIONS >= gen);
            let mut bindings = self.bindings.borrow_mut();
            bindings.retain(|key, binding| binding.gen + KEEP_GENERATIONS >= gen && self.holds_all(&key.1));
            drop(bindings);
            for held in self.held_textures.borrow_mut().values_mut() {
                held.bindings.retain(|key| key.strong_count() > 0);
            }
            for keys in self.held_buffers.borrow_mut().values_mut() {
                keys.retain(|key| key.strong_count() > 0);
            }
        }
    }

    pub(crate) fn set_memory_policy(&mut self, headroom: Box<dyn Fn() -> Option<u64>>, reclaim: Box<dyn Fn(u64)>) {
        self.headroom = Some(headroom);
        self.reclaim = Some(reclaim);
    }

    fn trim(&self, done: u64, buffer_budget: u64, texture_budget: u64, released: bool) {
        // An evicted resource's bind groups go with it: a kept group owns what it binds.
        {
            let mut held = self.held_buffers.borrow_mut();
            let mut bindings = self.bindings.borrow_mut();
            evict_resting(&mut self.buffers.borrow_mut(), done, buffer_budget, released, wgpu::Buffer::size, |slot| {
                let buffer = &slot.item;
                if !slot.free { self.ledger.accounts().busy_bytes(slot.gen, buffer.size(), false); }
                for key in held.remove(buffer).iter().flat_map(|keys| keys.iter().filter_map(Weak::upgrade)) {
                    bindings.remove(&key);
                }
            });
        }
        {
            let mut held = self.held_textures.borrow_mut();
            let mut bindings = self.bindings.borrow_mut();
            evict_resting(&mut self.textures.borrow_mut(), done, texture_budget, released, texture_bytes, |slot| {
                let texture = &slot.item;
                self.ledger.edit().pool(texture, None);
                for key in held.remove(texture).iter().flat_map(|h| h.bindings.iter().filter_map(Weak::upgrade)) {
                    bindings.remove(&key);
                }
            });
        }
    }

    pub(crate) fn texture_price(&self, device: &wgpu::Device, key: TexKey) -> u64 {
        let generation = self.gen.get();
        let mut prices = self.texture_prices.borrow_mut();
        let (bytes, last) = prices.entry(key.clone()).or_insert_with(|| {
            let desc = wgpu::TextureDescriptor { label: None,
                size: wgpu::Extent3d { width: key.w, height: key.h, depth_or_array_layers: key.layers },
                dimension: wgpu::TextureDimension::D2, mip_level_count: key.mips,
                format: key.format, usage: wgpu::TextureUsages::from_bits_retain(key.usage),
                sample_count: key.samples, view_formats: &[],
            };
            let bytes = device.texture_binding_size(&desc).unwrap_or_else(|| {
                estimated_texture_bytes(desc.size, desc.dimension, desc.format, desc.mip_level_count, desc.sample_count)
            });
            (bytes, generation)
        });
        *last = generation;
        *bytes
    }

    pub(crate) fn capture_requests(&self, wanted: &HashMap<TexKey, usize>) {
        self.capture_requests.borrow_mut().clone_from(wanted);
    }

    fn trim_capture_slots(&self, request: u64) {
        if !self.recording_captures.get() { return; }
        let done = self.ledger.completed();
        let mut map = self.textures.borrow_mut();
        let resting = |s: &Slot<wgpu::Texture>| s.free || s.gen <= done;
        let wanted = self.capture_requests.borrow();
        let idle: u64 = map.iter().flat_map(|(key, slots)| slots.iter().filter(|s| resting(s))
            .skip(wanted.get(key).copied().unwrap_or(0))).map(|s| texture_bytes(&s.item)).sum();
        if idle == 0 || idle.saturating_add(request) <= self.capture_spare.get() { return; }
        let mut retired = self.retiring_textures.borrow_mut();
        let mut held = self.held_textures.borrow_mut();
        let mut bindings = self.bindings.borrow_mut();
        map.retain(|key, slots| {
            let mut keep = wanted.get(key).copied().unwrap_or(0);
            slots.retain(|s| {
                if !resting(s) { return true; }
                if keep > 0 { keep -= 1; return true; }
                let mut ledger = self.ledger.edit();
                if s.gen > done && retired.insert(s.item.clone()) { ledger.owner(&s.item, Owner::OpenPrivate, true); }
                ledger.pool(&s.item, None);
                if let Some(old) = held.remove(&s.item) {
                    for key in old.bindings.iter().filter_map(Weak::upgrade) { bindings.remove(&key); }
                }
                false
            });
            !slots.is_empty()
        });
    }

    /// Spend measured native headroom only on new allocations, never reuse.
    /// Refresh and release idle storage when that credit is exhausted. A roomy
    /// device queries only generations that allocate, and adds no waits.
    fn prepare_allocation(&self, device: &wgpu::Device, bytes: u64) {
        let Some(headroom) = &self.headroom else { return };
        if let Some(credit) = self.allocation_credit.get().filter(|credit| *credit >= bytes) {
            self.allocation_credit.set(Some(credit - bytes));
            return;
        }
        self.headroom_sampled.set(true);
        let mut free = headroom();
        if free.is_some_and(|free| free < bytes) {
            let _ = device.poll(wgpu::PollType::Poll);
            self.trim(self.ledger.completed().saturating_add(1), 0, 0, true);
            if let Some(reclaim) = &self.reclaim { reclaim(bytes); }
            free = headroom();
            if free.is_some_and(|free| free < bytes) {
                flush();
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                self.trim(self.ledger.completed().saturating_add(1), 0, 0, true);
                free = headroom();
            }
        }
        self.allocation_credit.set(free.map(|free| free.saturating_sub(bytes)));
    }

    /// Let go of everything the pool holds (an engine dropping its retained GPU state): work already submitted
    /// keeps what it uses. Generations go on counting, so a finished one still reports itself correctly.
    pub(crate) fn clear(&self) {
        let mut ledger = self.ledger.edit();
        for slot in self.buffers.borrow().values().flatten().filter(|s| !s.free) {
            ledger.busy_bytes(slot.gen, slot.item.size(), false);
        }
        for texture in self.held_textures.borrow().keys() { ledger.pool(texture, None); }
        drop(ledger);
        self.buffers.borrow_mut().clear();
        self.textures.borrow_mut().clear();
        self.held_buffers.borrow_mut().clear();
        *self.scope.borrow_mut() = Default::default();
        self.empties.borrow_mut().clear();
        self.held_textures.borrow_mut().clear();
        self.bindings.borrow_mut().clear();
    }

    /// Whether the pool holds every buffer and texture of `bound`.
    fn holds_all(&self, bound: &[Bound]) -> bool {
        bound.iter().all(|b| match b {
            Bound::Buffer(buffer) | Bound::Range(buffer, ..) => self.held_buffers.borrow().contains_key(buffer),
            Bound::View(view) => self.held_textures.borrow().contains_key(view.texture()),
            Bound::Sampler(_) => true,
        })
    }

    /// Let the caller submit pending capture commands alongside its output conversion.
    pub(crate) fn record_for_output(&self) -> OutputRecording<'_> {
        let owner = !self.output_requested.get() && self.output_prefix.borrow().is_none();
        if owner { self.output_requested.set(true); }
        OutputRecording { pool: self, owner }
    }

    /// Start recording into the frame encoder on its first capture miss. Clean pages
    /// (including pages replaying their captures) incur no extra plan walk or staging.
    pub(crate) fn has_capture_recording(&self) -> bool { self.recording_captures.get() }

    pub(crate) fn begin_capture(&self) {
        if !self.recording_captures.replace(true) {
            HELD.with(|h| h.borrow_mut().0 += 1);
        }
    }

    /// In-flight captures share the space left in the capture cache.
    /// The native headroom includes resources
    /// held by other engines and the driver. An individually required chain can
    /// still render when it exceeds this room.
    pub(crate) fn capture_room(&self, queue: &wgpu::Queue, spare_cache: u64,
        scheduled: &std::collections::HashSet<&wgpu::Texture>) -> u64 {
        if !self.headroom_sampled.replace(true) {
            self.allocation_credit.set(self.headroom.as_ref().and_then(|measure| measure()));
        }
        // Page owners and pooled resources already occupy the shared capture
        // budget. The encoder ledger accounts for its own outstanding work.
        let used = self.ledger.accounts().room_used(scheduled) + open_staging(queue);
        let room = spare_cache.saturating_sub(used);
        self.allocation_credit.get().map_or(room, |headroom| room.min(headroom))
    }

    pub(crate) fn holds_texture(&self, texture: &wgpu::Texture) -> bool { self.held_textures.borrow().contains_key(texture) }

    #[cfg(test)]
    pub(crate) fn required_page_bytes(&self) -> u64 {
        self.ledger.accounts().required_page_bytes()
    }

    pub(crate) fn snapshot_bytes(&self) -> u64 {
        self.ledger.accounts().snapshot_bytes()
    }

    pub(crate) fn page_buffer(&self, device: &wgpu::Device, desc: &wgpu::util::BufferInitDescriptor<'_>, required_page: bool) -> PageBuffer {
        use wgpu::util::DeviceExt;
        self.prepare_allocation(device, desc.contents.len() as u64);
        let buffer = device.create_buffer_init(desc);
        self.ledger.edit().snapshot_buffer(&buffer, required_page);
        PageBuffer { buffer, ledger: self.ledger.clone() }
    }

    pub(crate) fn page_texture(&self, device: &wgpu::Device, desc: &wgpu::TextureDescriptor<'_>, required_page: bool) -> PageTexture {
        let texture = self.owned_texture(device, desc);
        self.ledger.edit().snapshot_texture(&texture, required_page);
        PageTexture { view: texture.create_view(&Default::default()), ledger: self.ledger.clone() }
    }

    pub(crate) fn page_owners(&self) -> PageOwners<'_> { PageOwners { pool: self, views: Vec::new() } }

    /// Link the backing identities to the consuming encoder. This covers immediate
    /// submission, nested recordings, output prefixes, rollover and discarded work.
    pub(crate) fn record_reads<'a>(&self, encoder: &wgpu::CommandEncoder, queue: &wgpu::Queue,
        textures: impl Iterator<Item = &'a wgpu::Texture>) {
        let published: std::collections::HashSet<_> = textures.cloned().collect();
        if !published.is_empty() { self.record_storage(encoder, queue, 0, published); }
    }

    pub(crate) fn record_storage(&self, encoder: &wgpu::CommandEncoder, queue: &wgpu::Queue, bytes: u64,
        published: std::collections::HashSet<wgpu::Texture>) {
        if bytes == 0 && published.is_empty() { return; }
        self.record_backings(encoder, queue, SubmittedStorage { bytes, published, ..Default::default() });
    }

    pub(crate) fn record_buffers(&self, encoder: &wgpu::CommandEncoder, queue: &wgpu::Queue, buffers: Vec<wgpu::Buffer>) {
        self.record_backings(encoder, queue, SubmittedStorage { buffers, ..Default::default() });
    }

    fn record_backings(&self, encoder: &wgpu::CommandEncoder, queue: &wgpu::Queue, storage: SubmittedStorage) {
        let charge = OpenStorage::new(&self.ledger, storage);
        charge.attach(encoder, queue);
    }

    fn prefix_storage(&self, bytes: u64) -> Option<OpenStorage> {
        let private = self.retiring_textures.take();
        if bytes == 0 && private.is_empty() { return None; }
        // These images were already charged when their pool slots were discarded.
        let storage = SubmittedStorage { bytes, private, prefix: true, ..Default::default() };
        Some(OpenStorage::from_prefix(&self.ledger, storage))
    }

    pub(crate) fn group_allowance(&self, room: u64, page: u64) -> u64 {
        // Bound each new group by its resolved page. Earlier groups can stay
        // in flight as long as all of them fit the shared budget.
        room.saturating_sub(self.ledger.bytes()).min(page)
    }

    pub(crate) fn submit_capture_prefix(&self, queue: &wgpu::Queue) {
        let staging = open_staging(queue);
        if let Some(index) = submit_runs(take_recorded(), Some(queue)) {
            if let Some(storage) = self.prefix_storage(staging) { storage.submit(queue, index); }
        }
    }

    pub(crate) fn start_filter_retirement(&self, queue: &wgpu::Queue, working: u64, room: u64,
        private: u64) {
        let private = self.allocation_credit.get().map_or(private, |native| private.min(native.saturating_sub(working)));
        let (submitted, recording) = self.ledger.accounts().outstanding(private);
        if recording != 0 && working.saturating_add(submitted).saturating_add(recording) > room {
            self.submit_capture_prefix(queue);
        }
    }

    /// A prefix submission drains the current upload pages. Both prices are known before
    /// admission starts. Switch to the fresh-page price at the same point we transfer ownership.
    pub(crate) fn admit_capture(&self, device: &wgpu::Device, queue: &wgpu::Queue, room: u64,
        demand: impl Fn(bool) -> (u64, u64)) -> (u64, bool) {
        let mut drained = false;
        loop {
            let (working, private) = demand(drained);
            let credit = self.allocation_credit.get();
            let private = credit.map_or(private, |n| private.min(n.saturating_sub(working)));
            let capacity = room.saturating_add(private);
            let outstanding = self.ledger.bytes();
            let available = room.saturating_sub(outstanding);
            self.capture_spare.set(credit.map_or(capacity, |n| n.min(capacity))
                .saturating_sub(working).saturating_sub(outstanding));
            if outstanding == 0 || working <= available { return (available, drained); }
            let ledger = self.ledger.accounts();
            let (submitted, recording) = ledger.outstanding(private);
            if (submitted == 0 && recording == 0) || working.saturating_add(submitted).saturating_add(recording) <= room {
                return (available, drained);
            }
            let oldest = ledger.oldest();
            drop(ledger);
            if !drained && (recording != 0 || !self.retiring_textures.borrow().is_empty()) {
                self.submit_capture_prefix(queue);
                drained = true;
            } else if let Some(index) = oldest.filter(|_| submitted != 0 && working.saturating_add(submitted) > room) {
                let _wait = crate::prof::span("fx.capture_allocation_wait");
                if device.poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None }).is_err() {
                    return (available, drained);
                }
            } else { return (available, drained); }
        }
    }

    /// Bound encoder validation on both wide sibling sets and nested capture trees.
    /// Finishing a segment preserves queue order and requires no submission or wait.
    pub(crate) fn finish_capture_page(&self) {
        let depth = self.depth.get();
        if depth < 2 || !self.recording_captures.get() { return; }
        // Only a genuinely deep tree can ask to reuse the preceding tree's fence.
        if depth > 2 && self.dependent_renders.get() < CAPTURE_PAGES_PER_SEGMENT {
            self.dependent_renders.set(self.dependent_renders.get() + 1);
        }
        let pages = self.recorded_pages.get();
        if pages < CAPTURE_PAGES_PER_SEGMENT * (MAX_CAPTURE_SEGMENTS - 1) {
            self.recorded_pages.set(pages + 1);
            if (pages + 1).is_multiple_of(CAPTURE_PAGES_PER_SEGMENT) { seal_recording(); }
        }
    }

    /// A render ends (its work submitted). The outermost one closes its generation: its resources come back once the
    /// queue has done everything submitted so far.
    pub(crate) fn leave(&self, queue: &wgpu::Queue, hold: bool) {
        let captures = self.depth.get() == 1 && self.recording_captures.replace(false);
        let mut deferred = false;
        if hold || captures {
            let open = HELD.with(|h| {
                let mut h = h.borrow_mut();
                h.0 = h.0.saturating_sub(u32::from(hold) + u32::from(captures));
                h.0
            });
            if open == 0 {
                let staging = if captures { open_staging(queue) } else { 0 };
                if captures && self.output_requested.replace(false) && !std::thread::panicking() {
                    let held = take_recorded();
                    if held.iter().all(|(owner, _)| owner == queue) {
                        let buffers = held.into_iter().map(|(_, buffer)| buffer).collect();
                        self.output_prefix.replace(Some(FramePrefix {
                            queue: queue.clone(), buffers,
                            completion: Some((self.ledger.clone(), self.gen.get())),
                            storage: self.prefix_storage(staging),
                            reusable_tree: (self.dependent_renders.get() >= CAPTURE_PAGES_PER_SEGMENT).then(|| self.previous_tree.clone()),
                        }));
                        deferred = true;
                    } else {
                        if let Some(index) = submit_runs(held, Some(queue)) {
                            if let Some(storage) = self.prefix_storage(staging) { storage.submit(queue, index); }
                        }
                    }
                } else if std::thread::panicking() {
                    discard_pending();
                    drop(self.prefix_storage(staging));
                } else if let Some(index) = submit_runs(take_recorded(), Some(queue)) {
                    if let Some(storage) = self.prefix_storage(staging) { storage.submit(queue, index); }
                }
            }
        }
        let d = self.depth.get().saturating_sub(1);
        self.depth.set(d);
        if d > 0 || deferred {
            return;
        }
        self.ledger.fence(queue, self.gen.get());
    }

    fn active(&self) -> bool {
        self.depth.get() > 0
    }

    /// A buffer as `device.create_buffer(desc)` makes it (not mapped at creation).
    pub(crate) fn buffer(&self, device: &wgpu::Device, desc: &wgpu::BufferDescriptor<'_>) -> wgpu::Buffer {
        let mappable = desc.usage.intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE);
        if !self.active() || desc.mapped_at_creation || mappable {
            return device.create_buffer(desc);
        }
        let key = (desc.size, desc.usage.bits());
        let (gen, done) = (self.gen.get(), self.ledger.completed());
        let mut map = self.buffers.borrow_mut();
        let list = map.entry(key).or_default();
        if list.front().is_some_and(|s| s.free || s.gen <= done) {
            let slot = list.pop_front().expect("a front");
            // Reusing a released allocation does not enlarge this frame's working set.
            if slot.gen != gen {
                let [buffers, textures] = self.taken.get();
                self.taken.set([buffers + desc.size, textures]);
            }
            let mut ledger = self.ledger.accounts();
            if !slot.free { ledger.busy_bytes(slot.gen, desc.size, false); }
            ledger.busy_bytes(gen, desc.size, true);
            let item = slot.item;
            list.push_back(Slot { gen, free: false, item: item.clone() });
            if !HELD.with(|held| held.borrow().0 > 0) { self.scope.borrow_mut().0.push(item.clone()); }
            count("scratch.buffer.reused");
            return item;
        }
        count(if list.is_empty() { "scratch.buffer.new_key" } else { "scratch.buffer.all_busy" });
        drop(map);
        self.prepare_allocation(device, desc.size);
        let buf = device.create_buffer(desc);
        let [buffers, textures] = self.taken.get();
        self.taken.set([buffers + desc.size, textures]);
        if !HELD.with(|held| held.borrow().0 > 0) { self.scope.borrow_mut().0.push(buf.clone()); }
        self.buffers.borrow_mut().entry(key).or_default().push_back(Slot { gen, free: false, item: buf.clone() });
        self.ledger.accounts().busy_bytes(gen, desc.size, true);
        self.held_buffers.borrow_mut().insert(buf.clone(), Vec::new());
        buf
    }

    /// Where the open generation's takes stand: the start of a scope ([`ScratchPool::release`]).
    pub(crate) fn mark(&self) -> Scope {
        let scope = self.scope.borrow();
        Scope(scope.0.len(), scope.1.len())
    }

    /// Hand back the buffers and textures taken since `mark`, for the caller whose every command buffer that uses
    /// them has been submitted (a capture, once it is rendered and filtered), all but `result`, the texture
    /// something later still reads (it stays in the scope around this one). The next take of a key may be one of
    /// them, in this generation or any later: the queue runs its writes and command buffers in the order they were
    /// made, so a later write or pass cannot land under an earlier reader. That is how a retained engine's
    /// captures shared its labelled buffers and its attachments; otherwise a frame of a hundred captures holds a
    /// hundred captures' buffers and filter textures to its end, where it needs one capture's at a time. While
    /// command buffers are held on this thread nothing has been submitted, and nothing is handed back.
    pub(crate) fn release(&self, mark: Scope, result: Option<&wgpu::Texture>) {
        let (buffers, textures) = {
            let mut scope = self.scope.borrow_mut();
            let scope = &mut *scope;
            let from = (mark.0.min(scope.0.len()), mark.1.min(scope.1.len()));
            let buffers = scope.0.split_off(from.0);
            let mut textures = scope.1.split_off(from.1);
            if HELD.with(|h| h.borrow().0 > 0) {
                return;
            }
            if let Some(at) = result.and_then(|result| textures.iter().position(|t| t == result)) {
                scope.1.push(textures.swap_remove(at));
            }
            (buffers, textures)
        };
        // Each was taken in this scope, so it is near the back of its queue.
        let mut map = self.buffers.borrow_mut();
        for buffer in buffers {
            let Some(list) = map.get_mut(&(buffer.size(), buffer.usage().bits())) else { continue };
            let at = list.iter().rposition(|s| s.item == buffer);
            if let Some(slot) = at.and_then(|at| list.remove(at)) {
                if !slot.free { self.ledger.accounts().busy_bytes(slot.gen, buffer.size(), false); }
                list.push_front(Slot { free: true, ..slot });
            }
        }
        for texture in textures {
            self.recycle_attachment(&texture);
        }
    }

    /// The last GPU read of this attachment precedes its next clear in the same
    /// ordered command stream. CPU uploads still require the generation fence.
    pub(crate) fn recycle_attachment(&self, texture: &wgpu::Texture) {
        let mut map = self.textures.borrow_mut();
        let Some(list) = map.get_mut(&TexKey::of(texture)) else { return };
        if let Some(at) = list.iter().rposition(|s| s.item == *texture && !s.free) {
            let slot = list.remove(at).unwrap();
            self.ledger.edit().pool(texture, Some(0));
            list.push_front(Slot { free: true, ..slot });
        }
    }

    /// The buffer a render binds where it has no data of `usage` to upload (four bytes, never written, never read:
    /// draws address only what was written). A frame of a hundred captures bound some two thousand of these, each
    /// its own buffer with its own bind group; they are one buffer.
    pub(crate) fn empty(&self, device: &wgpu::Device, label: &str, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let desc = wgpu::BufferDescriptor { label: Some(label), size: 4, usage, mapped_at_creation: false };
        if !self.active() {
            return device.create_buffer(&desc);
        }
        self.empties
            .borrow_mut()
            .entry(usage.bits())
            .or_insert_with(|| {
                let buffer = device.create_buffer(&desc);
                self.held_buffers.borrow_mut().insert(buffer.clone(), Vec::new());
                buffer
            })
            .clone()
    }

    /// A texture as `device.create_texture(desc)` makes it. For a texture its users write whole before they read.
    /// Admit a caller-owned target without putting it in the generation pool.
    /// Resolve images can outlive the render that produced them; their callers
    /// retain ownership while sharing scratch's device-wide admission policy.
    pub(crate) fn owned_texture(&self, device: &wgpu::Device, desc: &wgpu::TextureDescriptor<'_>) -> wgpu::Texture {
        let bytes = estimated_texture_bytes(desc.size, desc.dimension, desc.format, desc.mip_level_count, desc.sample_count);
        self.prepare_allocation(device, bytes);
        device.create_texture(desc)
    }

    pub(crate) fn texture(&self, device: &wgpu::Device, desc: &wgpu::TextureDescriptor<'_>) -> wgpu::Texture {
        if !self.active() || !desc.view_formats.is_empty() {
            return self.owned_texture(device, desc);
        }
        let key = TexKey {
            w: desc.size.width,
            h: desc.size.height,
            layers: desc.size.depth_or_array_layers,
            mips: desc.mip_level_count,
            samples: desc.sample_count,
            dim: desc.dimension as u8,
            format: desc.format,
            usage: desc.usage.bits(),
        };
        let (gen, mut done) = (self.gen.get(), self.ledger.completed());
        let mut map = self.textures.borrow_mut();
        let list = map.entry(key.clone()).or_default();
        // If this dependent tree needs the same attachment as the last tree, retire
        // that frame's fence before allocating another copy of its working set.
        if let Some(slot) = list.front().filter(|slot| self.depth.get() > 2 && self.recording_captures.get()
            && !slot.free && slot.gen > done && slot.gen < gen) {
            let fence = self.previous_tree.lock().unwrap().take_if(|(generation, _)| *generation >= slot.gen);
            if let Some((_, index)) = fence {
                let _wait = crate::prof::span("fx.nested_arena_wait");
                let _ = device.poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None });
                done = self.ledger.completed();
            }
        }
        let tex = if list.front().is_some_and(|s| s.free || s.gen <= done) {
            count("scratch.texture.reused");
            let slot = list.pop_front().expect("a front");
            if slot.gen != gen {
                let [buffers, textures] = self.taken.get();
                self.taken.set([buffers, textures + texture_bytes(&slot.item)]);
            }
            drop(map);
            slot.item
        } else {
            count(if list.is_empty() { "scratch.texture.new_key" } else { "scratch.texture.all_busy" });
            drop(map);
            let bytes = if self.recording_captures.get() && desc.dimension == wgpu::TextureDimension::D2 {
                self.texture_price(device, key.clone())
            } else {
                u64::from(desc.size.width) * u64::from(desc.size.height) * u64::from(desc.size.depth_or_array_layers)
                    * u64::from(desc.sample_count) * u64::from(desc.format.block_copy_size(None).unwrap_or(4))
            };
            self.trim_capture_slots(bytes);
            self.prepare_allocation(device, bytes);
            let tex = device.create_texture(desc);
            let [buffers, textures] = self.taken.get();
            self.taken.set([buffers, textures + texture_bytes(&tex)]);
            self.held_textures.borrow_mut().insert(tex.clone(), HeldTexture::default());
            tex
        };
        self.ledger.edit().pool(&tex, Some(gen));
        self.textures.borrow_mut().entry(key).or_default().push_back(Slot { gen, free: false, item: tex.clone() });
        if !HELD.with(|held| held.borrow().0 > 0) { self.scope.borrow_mut().1.push(tex.clone()); }
        tex
    }

    /// `texture`'s default view (`create_view(&Default::default())`): the one view the pool keeps for a texture it
    /// holds, a new one for any other.
    pub(crate) fn view(&self, texture: &wgpu::Texture) -> wgpu::TextureView {
        match self.held_textures.borrow_mut().get_mut(texture) {
            Some(held) => held.view.get_or_insert_with(|| texture.create_view(&Default::default())).clone(),
            None => texture.create_view(&Default::default()),
        }
    }

    /// The array compositor samples an individual capture through layer zero.
    pub(crate) fn array_view(&self, texture: &wgpu::Texture) -> wgpu::TextureView {
        let make = || texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        match self.held_textures.borrow_mut().get_mut(texture) {
            Some(held) => held.array_view.get_or_insert_with(make).clone(),
            None => make(),
        }
    }

    /// A 2D view of one array layer. The compositor can keep its bindings when a frame reuses the array.
    pub(crate) fn layer_view(&self, texture: &wgpu::Texture, layer: u32) -> wgpu::TextureView {
        let make = || texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        });
        let mut textures = self.held_textures.borrow_mut();
        let Some(held) = textures.get_mut(texture) else { return make() };
        let at = layer as usize;
        if held.layers.len() <= at {
            held.layers.resize_with(at + 1, || None);
        }
        held.layers[at].get_or_insert_with(make).clone()
    }

    /// Withdraw `texture` from the pool: its holder keeps it past this generation (a capture result kept for later
    /// frames), so it is never handed out again. Nothing to do for a texture the pool does not hold.
    pub(crate) fn keep(&self, texture: &wgpu::Texture) {
        let Some(held) = self.held_textures.borrow_mut().remove(texture) else { return };
        self.ledger.edit().pool(texture, None);
        let key = TexKey::of(texture);
        let mut map = self.textures.borrow_mut();
        if let Some(list) = map.get_mut(&key) {
            // A result is kept in the generation that took it, so it is at the back of its queue, behind only what
            // was taken under the same key since.
            if let Some(at) = list.iter().rposition(|s| s.item == *texture) {
                let _ = list.remove(at);
            }
            if list.is_empty() {
                map.remove(&key);
            }
        }
        // The bind groups that name it would keep it alive after its holder lets it go.
        let mut bindings = self.bindings.borrow_mut();
        for key in held.bindings.iter().filter_map(Weak::upgrade) {
            bindings.remove(&key);
        }
    }

    pub(crate) fn upload_layout(&self, queue: &wgpu::Queue, device: &wgpu::Device, usage: wgpu::BufferUsages) -> UploadLayout {
        let alignment = if usage == wgpu::BufferUsages::UNIFORM { device.limits().min_uniform_buffer_offset_alignment }
            else { device.limits().min_storage_buffer_offset_alignment };
        let pool = self as *const Self as usize;
        let usage = usage | wgpu::BufferUsages::COPY_DST;
        let pages = UPLOAD_PAGES.with(|p| p.borrow().iter()
            .filter(|p| p.pool == pool && p.queue == *queue && p.usage == usage && !p.lasting)
            .map(|p| (p.bytes.len() as u64, p.buffer.size())).collect());
        UploadLayout { pages, alignment: u64::from(alignment), additional: 0 }
    }

    pub(crate) fn uniform(&self, device: &wgpu::Device, queue: &wgpu::Queue, label: &str, data: &[f32], lasting: bool) -> Bound {
        self.upload_range(device, queue, label, bytemuck::cast_slice(data), wgpu::BufferUsages::UNIFORM, lasting)
    }

    /// Bind just this allocation. Transient pages return with their generation, while
    /// a composite packet can retain its immutable page beyond the frame.
    pub(crate) fn upload_range(&self, device: &wgpu::Device, queue: &wgpu::Queue,
        label: &str, bytes: &[u8], usage: wgpu::BufferUsages, lasting: bool) -> Bound {
        let size = bytes.len() as u64;
        let alignment = if usage == wgpu::BufferUsages::UNIFORM {
            device.limits().min_uniform_buffer_offset_alignment as usize
        } else if usage == wgpu::BufferUsages::STORAGE {
            device.limits().min_storage_buffer_offset_alignment as usize
        } else { wgpu::COPY_BUFFER_ALIGNMENT as usize };
        let usage = usage | wgpu::BufferUsages::COPY_DST;
        if !HELD.with(|h| h.borrow().0 > 0) {
            let desc = wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false };
            let buffer = if lasting { device.create_buffer(&desc) } else { self.buffer(device, &desc) };
            queue.write_buffer(&buffer, 0, bytes);
            return Bound::Buffer(buffer);
        }
        let pool = self as *const Self as usize;
        let fits = |page: &UploadPage| page.queue == *queue && page.pool == pool
            && page.usage == usage && page.lasting == lasting
            && UploadLayout::offset(page.bytes.len() as u64, alignment as u64) + size <= page.buffer.size();
        let page_index = UPLOAD_PAGES.with(|pages| pages.borrow().iter().rposition(fits));
        let index = page_index.unwrap_or_else(|| {
            let capacity = UPLOAD_PAGE_BYTES.max(size);
            let desc = wgpu::BufferDescriptor { label: Some(label), size: capacity, usage, mapped_at_creation: false };
            let buffer = if lasting {
                self.prepare_allocation(device,capacity);
                device.create_buffer(&desc)
            } else { self.buffer(device,&desc) };
            UPLOAD_PAGES.with(|pages| {
                let mut pages = pages.borrow_mut();
                pages.push(UploadPage { pool, lasting, usage, queue: queue.clone(), buffer, bytes: Vec::new() });
                pages.len() - 1
            })
        });
        UPLOAD_PAGES.with(|pages| {
            let mut pages = pages.borrow_mut();
            let page = &mut pages[index];
            let offset = page.bytes.len().next_multiple_of(alignment);
            WRITE_SIZES.with(|s| *s.borrow_mut().entry(queue.clone()).or_default() += (offset - page.bytes.len()) as u64 + size);
            page.bytes.resize(offset, 0);
            page.bytes.extend_from_slice(bytes);
            Bound::Range(page.buffer.clone(),offset as u64,wgpu::BufferSize::new(size).expect("nonempty allocation"))
        })
    }

    /// A bind group of `layout` with `bound[i]` at binding `i`. While a render is open and the pool holds every
    /// buffer and texture bound, the group is kept under the layout and those resources and returned for them again:
    /// a bind group is what it binds, whatever the buffers hold.
    pub(crate) fn bind(
        &self,
        device: &wgpu::Device,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        bound: &[Bound],
    ) -> wgpu::BindGroup {
        let make = || {
            let entries: Vec<wgpu::BindGroupEntry<'_>> = bound
                .iter()
                .enumerate()
                .map(|(binding, b)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: match b {
                        Bound::Buffer(buffer) => buffer.as_entire_binding(),
                        Bound::Range(buffer, offset, len) => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer, offset: *offset, size: Some(*len),
                        }),
                        Bound::View(view) => wgpu::BindingResource::TextureView(view),
                        Bound::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
                    },
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &entries })
        };
        if !self.active() || !self.holds_all(bound) {
            return make();
        }
        let gen = self.gen.get();
        let key: BindKey = (layout.clone(), bound.to_vec());
        let mut bindings = self.bindings.borrow_mut();
        if let Some(binding) = bindings.get_mut(&key) {
            binding.gen = gen;
            count("scratch.bind.reused");
            return binding.group.clone();
        }
        count("scratch.bind.new");
        let group = make();
        let key = Arc::new(key);
        let mut textures = self.held_textures.borrow_mut();
        let mut buffers = self.held_buffers.borrow_mut();
        for b in bound {
            match b {
                Bound::View(view) => {
                    if let Some(texture) = textures.get_mut(view.texture()) {
                        texture.bindings.push(Arc::downgrade(&key));
                    }
                }
                Bound::Buffer(buffer) | Bound::Range(buffer, ..) => {
                    if let Some(keys) = buffers.get_mut(buffer) {
                        keys.push(Arc::downgrade(&key));
                    }
                }
                Bound::Sampler(_) => {}
            }
        }
        bindings.insert(key, Binding { gen, group: group.clone() });
        group
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    #[test]
    fn admission_charges_each_upload_page_and_its_staging_padding_once() {
        let mut layout = UploadLayout { pages: vec![(12, UPLOAD_PAGE_BYTES)], alignment: 256, additional: 0 };
        layout.add(16);
        assert_eq!(layout.additional, 260);
        layout.add(UPLOAD_PAGE_BYTES);
        assert_eq!(layout.additional, 260 + 2 * UPLOAD_PAGE_BYTES);
        layout.add(32);
        assert_eq!(layout.additional, 260 + 2 * UPLOAD_PAGE_BYTES + 272);
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn array_reads_follow_the_consuming_encoder_and_charge_each_backing_once() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).unwrap();
        let (device, queue) = (&gpu.device, &gpu.queue);
        let pool = ScratchPool::default();
        let desc = wgpu::TextureDescriptor {
            label: None, dimension: wgpu::TextureDimension::D2,
            size: wgpu::Extent3d { width: 1028, height: 1028, depth_or_array_layers: 2 },
            mip_level_count: 1, sample_count: 1, format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
        };
        let texture = device.create_texture(&desc);
        let bytes = texture_bytes(&texture);
        if let Some(required) = device.texture_binding_size(&desc) { assert!(bytes >= required); }
        let left = pool.layer_view(&texture, 0);
        let right = pool.layer_view(&texture, 1);
        let mut page = pool.page_owners();
        page.hold(&left);
        page.hold(&right);
        let room = || pool.capture_room(queue, bytes * 2, &Default::default());
        assert_eq!(room(), bytes);
        // A small next group can overlap a page-sized result in a larger budget.
        assert_eq!(pool.group_allowance(room(), bytes / 2), bytes / 2);
        assert_eq!(pool.capture_room(queue, bytes, &Default::default()), 0);
        let encoder = device.create_command_encoder(&Default::default());
        pool.record_reads(&encoder, queue, [left.texture(), right.texture()].into_iter());
        assert_eq!(pool.group_allowance(room(), bytes), bytes);
        drop(page);
        pool.ledger.cache(&texture, true);
        assert_eq!(pool.ledger.accounts().outstanding(0), (0, 0));
        pool.ledger.cache(&texture, false);
        assert_eq!(pool.ledger.accounts().outstanding(0), (0, bytes));
        pool.ledger.edit().owner(&texture, Owner::OpenPrivate, true);
        assert_eq!(pool.ledger.accounts().outstanding(0), (0, bytes));
        pool.ledger.edit().owner(&texture, Owner::OpenPrivate, false);
        let fence = queue.submit([encoder.finish()]);
        assert_eq!(pool.ledger.accounts().outstanding(0), (bytes, 0));
        device.poll(wgpu::PollType::Wait { submission_index: Some(fence), timeout: None }).unwrap();
        assert_eq!(pool.ledger.accounts().outstanding(0), (0, 0));
        let discarded = device.create_command_encoder(&Default::default());
        pool.record_reads(&discarded, queue, [right.texture()].into_iter());
        drop(discarded);
        assert_eq!(pool.ledger.accounts().outstanding(0), (0, 0));
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn capture_frame_scratch_becomes_reusable_after_output_submission() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).unwrap();
        let (device, queue) = (&gpu.device, &gpu.queue);
        let pool = ScratchPool::default();
        let recording = pool.record_for_output();
        pool.enter(device, false);
        pool.begin_capture();
        end_recording(queue, encoder(device, queue, &Default::default()));
        pool.leave(queue, false);
        let prefix = recording.finish().expect("dirty capture handed to output");
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(pool.ledger.completed(), 0);
        prefix.submit(device.create_command_encoder(&Default::default()).finish());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(pool.ledger.completed(), 1);
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn only_dependent_capture_trees_publish_a_scratch_fence() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).unwrap();
        let (device, queue) = (&gpu.device, &gpu.queue);
        let pool = ScratchPool::default();
        for dependent in [false, true] {
            let recording = pool.record_for_output();
            pool.enter(device, false);
            pool.begin_capture();
            pool.enter(device, false);
            if dependent { pool.enter(device, false); }
            for _ in 0..CAPTURE_PAGES_PER_SEGMENT {
                end_recording(queue, encoder(device, queue, &Default::default()));
                pool.finish_capture_page();
            }
            assert!(RECORDING.with(|r| r.borrow().is_none()), "an eighth capture page closes its encoder at either depth");
            if dependent { pool.leave(queue, false); }
            pool.leave(queue, false);
            pool.leave(queue, false);
            let prefix = recording.finish().expect("capture commands");
            assert!(pool.previous_tree.lock().unwrap().is_none());
            prefix.submit(device.create_command_encoder(&Default::default()).finish());
            let generation = pool.previous_tree.lock().unwrap().as_ref().map(|(generation, _)| *generation);
            assert_eq!(generation, dependent.then_some(pool.gen.get()));
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn caller_owned_targets_are_admitted_without_becoming_reusable_scratch() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU adapter");
        let mut pool = ScratchPool::default();
        let reclaimed = std::rc::Rc::new(Cell::new(0));
        let r = reclaimed.clone();
        pool.set_memory_policy(Box::new(|| Some(0)), Box::new(move |bytes| r.set(r.get() + bytes)));
        let desc = wgpu::TextureDescriptor {
            label: Some("owned target admission"),
            size: wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 4, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
        };
        let a = pool.owned_texture(&gpu.device, &desc);
        assert_eq!(reclaimed.get(), 8 * 8 * 4 * 4);
        assert!(pool.held_textures.borrow().is_empty());
        // Outside a generation, texture() must also consult admission and leave
        // ownership with the caller rather than silently entering the pool.
        let b = pool.texture(&gpu.device, &desc);
        assert_ne!(a, b);
        assert_eq!(reclaimed.get(), 2 * 8 * 8 * 4 * 4);
        assert!(pool.held_textures.borrow().is_empty());
        pool.enter(&gpu.device, false);
        pool.leave(&gpu.queue, false);
        // A pool lifecycle never destroys an externally owned target.
        let _a_view = a.create_view(&Default::default());
        let _b_view = b.create_view(&Default::default());
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn pressure_reclaims_completed_scratch_but_preserves_live_and_reusable_resources() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU adapter");
        let (device, queue) = (&gpu.device, &gpu.queue);
        let mut pool = ScratchPool::default();
        let queried = std::rc::Rc::new(Cell::new(0));
        let reclaimed = std::rc::Rc::new(Cell::new(0));
        let q = queried.clone();
        let r = reclaimed.clone();
        pool.set_memory_policy(Box::new(move || { q.set(q.get() + 1); Some(0) }),
            Box::new(move |bytes| r.set(r.get() + bytes)));
        let mut desc = wgpu::BufferDescriptor { label: Some("pressure contract"), size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false };
        pool.enter(device, false);
        let old = pool.buffer(device, &desc);
        pool.leave(queue, false);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        pool.enter(device, false);
        desc.size = 128;
        let mark = pool.mark();
        let live = pool.buffer(device, &desc);
        assert!(!pool.held_buffers.borrow().contains_key(&old));
        assert!(pool.held_buffers.borrow().contains_key(&live));
        assert!(reclaimed.get() >= 128);
        pool.release(mark, None);
        let queries = queried.get();
        assert_eq!(pool.buffer(device, &desc), live);
        assert_eq!(queried.get(), queries, "reuse consumes no native allocation credit");
        // A released resource from this still-open generation is idle too.
        let released_scope = pool.mark();
        desc.size = 192;
        let released = pool.buffer(device, &desc);
        pool.release(released_scope, None);
        desc.size = 256;
        let _other = pool.buffer(device, &desc);
        assert!(pool.held_buffers.borrow().contains_key(&live), "current generation was evicted");
        assert!(!pool.held_buffers.borrow().contains_key(&released), "released current-generation resource stayed pinned");
        pool.leave(queue, false);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    /// A kept texture is never handed out again; a pooled texture has one view; a bind group over pooled resources
    /// is the same group for the same resources, and goes when one of them leaves the pool.
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn kept_textures_leave_the_pool_and_bind_groups_follow_their_resources() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let (device, queue) = (&gpu.device, &gpu.queue);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pool test"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let desc = wgpu::TextureDescriptor {
            label: Some("pool test"),
            size: wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let pool = ScratchPool::default();
        pool.enter(device, false);
        let (a, b) = (pool.texture(device, &desc), pool.texture(device, &desc));
        assert!(a != b, "one generation handed a texture out twice");
        assert!(pool.view(&a) == pool.view(&a), "a pooled texture has one view");
        let group = |texture: &wgpu::Texture| pool.bind(device, "pool test", &layout, &[Bound::View(pool.view(texture))]);
        assert!(group(&a) == group(&a), "the same resources are the same bind group");
        assert!(group(&a) != group(&b));
        let kept = group(&a);
        // A group over both textures: withdrawing one must leave nothing owning it through the other.
        let pair_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pool test pair"),
            entries: &[0, 1].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }),
        });
        let pair = || pool.bind(device, "pool test pair", &pair_layout, &[Bound::View(pool.view(&a)), Bound::View(pool.view(&b))]);
        assert!(pair() == pair());
        let only_b = group(&b);
        pool.keep(&a);
        assert!(
            pool.bindings
                .borrow()
                .keys()
                .all(|key| !key.1.iter().any(|bound| matches!(bound, Bound::View(view) if *view.texture() == a))),
            "a kept bind group still names a withdrawn texture"
        );
        assert!(group(&b) == only_b, "withdrawing one texture dropped another's bind group");
        assert!(!pool.held_textures.borrow().contains_key(&a) && pool.held_textures.borrow().contains_key(&b));
        assert!(group(&a) != kept, "a bind group outlived its texture's place in the pool");
        pool.leave(queue, false);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        // The finished generation's texture comes back; the kept one never does.
        pool.enter(device, false);
        let c = pool.texture(device, &desc);
        assert!(c == b);
        assert!(pool.texture(device, &desc) != a);
        pool.leave(queue, false);
        // A buffer handed back is the next one taken, in the same generation; one not handed back is not.
        let uniform = wgpu::BufferDescriptor {
            label: Some("pool test"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        pool.enter(device, false);
        let scope = pool.mark();
        let first = pool.buffer(device, &uniform);
        assert!(pool.buffer(device, &uniform) != first, "one generation handed a buffer out twice");
        // Textures too, all but the one named as the scope's result.
        let (result, between) = (pool.texture(device, &desc), pool.texture(device, &desc));
        pool.release(scope, Some(&result));
        let again = pool.buffer(device, &uniform);
        assert!(again == first || pool.buffer(device, &uniform) == first, "a released buffer was not taken again");
        assert!(pool.texture(device, &desc) == between, "a released texture was not taken again");
        assert!(pool.texture(device, &desc) != result, "a scope's result was handed out while still read");
        pool.leave(queue, false);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        // A hundred sequential effects use one physical working set. Repeated
        // releases must not give the resting cache a hundred-frame allowance.
        pool.enter(device, false);
        let scope = pool.mark();
        let buffer = pool.buffer(device, &uniform);
        let texture = pool.texture(device, &desc);
        let working_set = pool.taken.get();
        assert_eq!(working_set, [uniform.size, texture_bytes(&texture)]);
        pool.release(scope, None);
        for _ in 0..100 {
            let scope = pool.mark();
            assert_eq!(pool.buffer(device, &uniform), buffer);
            assert_eq!(pool.texture(device, &desc), texture);
            pool.release(scope, None);
            assert_eq!(pool.taken.get(), working_set);
        }
        pool.leave(queue, false);
        // An evicted buffer takes its bind group with it. A generation that takes nothing leaves no budget for
        // what rests, so once the buffers' generation is finished and another has passed, they go.
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pool test uniform"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        pool.enter(device, false);
        let bound = pool.buffer(device, &uniform);
        let _ = pool.bind(device, "pool test uniform", &uniform_layout, &[Bound::Buffer(bound.clone())]);
        assert!(pool.bindings.borrow().keys().any(|key| key.1.contains(&Bound::Buffer(bound.clone()))));
        pool.leave(queue, false);
        for _ in 0..3 {
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            pool.enter(device, false);
            pool.leave(queue, false);
        }
        assert!(!pool.held_buffers.borrow().contains_key(&bound), "a resting buffer outlived an empty budget");
        assert!(
            !pool.bindings.borrow().keys().any(|key| key.1.contains(&Bound::Buffer(bound.clone()))),
            "a kept bind group still owns an evicted buffer"
        );
        // Cleared, the pool holds nothing and hands nothing it held out again.
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        pool.clear();
        assert!(pool.held_textures.borrow().is_empty() && pool.bindings.borrow().is_empty());
        pool.enter(device, false);
        assert!(pool.texture(device, &desc) != b);
        pool.leave(queue, false);
    }
}

#[cfg(test)]
mod teardown_tests {
    use super::*;

    #[derive(Default)]
    struct Resource(Cell<usize>);
    impl Destroy for Resource {
        fn destroy_now(&self) { self.0.set(self.0.get() + 1); }
    }

    #[test]
    fn destroys_only_when_scratch_can_be_read() {
        let resource = Resource::default();
        destroy(&resource);
        assert_eq!(resource.0.get(), 1);
        HELD.with(|h| {
            let _borrow = h.borrow_mut();
            destroy(&resource);
        });
        WRITES.with(|w| {
            let _borrow = w.borrow_mut();
            destroy(&resource);
        });
        assert_eq!(resource.0.get(), 1);
    }

    struct AtExit {
        held_gone: bool,
        writes_gone: bool,
        checked: Arc<AtomicU64>,
    }
    impl Drop for AtExit {
        fn drop(&mut self) {
            assert_eq!(HELD.try_with(|_| ()).is_err(), self.held_gone);
            assert_eq!(WRITES.try_with(|_| ()).is_err(), self.writes_gone);
            let resource = Resource::default();
            destroy(&resource);
            assert_eq!(resource.0.get(), 0, "unavailable TLS cannot authorize explicit destruction");
            self.checked.fetch_add(1, Ordering::SeqCst);
        }
    }
    thread_local! {
        static AT_EXIT: RefCell<Option<AtExit>> = const { RefCell::new(None) };
    }

    #[test]
    fn destroy_after_either_or_both_scratch_keys_are_gone() {
        for (held_gone, writes_gone) in [(true, false), (false, true), (true, true)] {
            let checked = Arc::new(AtomicU64::new(0));
            let result = checked.clone();
            std::thread::spawn(move || {
                // TLS destructors run in reverse initialization order on our native targets.
                if !held_gone { HELD.with(|_| ()); }
                if !writes_gone { WRITES.with(|_| ()); }
                AT_EXIT.with(|slot| *slot.borrow_mut() = Some(AtExit { held_gone, writes_gone, checked }));
                HELD.with(|_| ());
                WRITES.with(|_| ());
            }).join().unwrap();
            assert_eq!(result.load(Ordering::SeqCst), 1);
        }
    }
}
