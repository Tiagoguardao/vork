//! Engine integration seam (M3c → M3b's `paint::image_source`).
//!
//! M3b shipped the pattern pipeline with a declared seam
//! (`moda-engine/src/paint/image_source.rs` header): "`resolve_pattern_source`
//! should delegate non-data URLs to moda-assets and keep the placeholder
//! mapping". This module is the moda-assets half of that delegation:
//!
//! - the CLI prefetches every pattern source up front
//!   ([`crate::collect_pattern_requests`] + [`crate::prefetch_blocking`]) and
//!   installs the settled store process-wide via [`install_store`];
//! - the engine's `resolve_pattern_source` consults [`resolve`] first and
//!   keeps its inline data-URI decode as the no-store fallback (so engine
//!   unit tests and store-less callers keep working).
//!
//! The store is process-global because the seam function has no store
//! parameter and the M1-frozen `RenderOptions` should not grow one mid-M3;
//! a render process handles one scene (CLI) or can re-install per request.

use std::sync::{Arc, OnceLock, RwLock};

use crate::pattern::ImageFillState;
use crate::AssetStore;

mod reads;
pub use reads::{AssetReads, CaptureReads, Publication};

/// A store-resolved pattern texture in engine-seam shape.
#[derive(Debug, Clone)]
pub struct ResolvedAsset {
    pub state: ImageFillState,
    /// Uploaded texel dimensions.
    pub width: u32,
    pub height: u32,
    /// uploaded / natural per axis — divide `fillPatternScale` by this
    /// (`WebGPUPathRenderer.ts:1338-1345`).
    pub scale_factor: (f64, f64),
    /// Where the resolved pixels live (heap bytes vs an external GPU upload).
    pub pixels: ResolvedPixels,
    /// Whether heap pixels are straight (unassociated) colour (`PreparedImage::straight`).
    pub straight: bool,
}

/// Where a resolved pattern texture's pixels live.
#[derive(Debug, Clone)]
pub enum ResolvedPixels {
    /// Decoded, premultiplied sRGB RGBA8 on the heap (native CLI lane, the
    /// wasm SVG/shader paths, and every placeholder).
    Heap(Arc<Vec<u8>>),
    /// A GPU image publication. Native footage supplies its immutable texture directly.
    /// Browser rasters use the uploader registered for `key`. A fresh `upload_id`
    /// identifies new pixels in either case.
    External { key: String, upload_id: u64, texture: Option<Arc<wgpu::Texture>> },
}

fn slot() -> &'static RwLock<Option<Arc<AssetStore>>> {
    static SLOT: OnceLock<RwLock<Option<Arc<AssetStore>>>> = OnceLock::new();
    SLOT.get_or_init(|| RwLock::new(None))
}

fn transparent_rgba() -> Arc<Vec<u8>> {
    static BYTES: OnceLock<Arc<Vec<u8>>> = OnceLock::new();
    Arc::clone(BYTES.get_or_init(|| Arc::new(vec![0, 0, 0, 0])))
}

/// Install the prefetched store for the current process (CLI: once per run).
///
/// The generation names resource inputs, not the allocation holding the map. Reinstalling a store with
/// the same resources is a no-op, including an immutable snapshot cloned by a frame's capture installer.
/// Changed membership, payload identity, state or external upload metadata advances the generation.
pub fn install_store(store: Arc<AssetStore>) {
    let mut guard = slot().write().expect("asset store lock");
    let same = guard.as_ref().is_some_and(|cur| Arc::ptr_eq(cur, &store) || cur.same_resources(&store));
    if same { return; }
    if !guard.as_ref().is_some_and(|old| old.same_layout(&store)) {
        LAYOUT_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    *guard = Some(store);
    STORE_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Monotonic install counter (task #95): retained-mode caches key on this so
/// an asset settle (install_assets swap) self-invalidates page plans that
/// baked the previous store's pattern states. 0 = never installed.
static STORE_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static LAYOUT_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Advances when a plan can no longer reuse its pattern instances.
pub fn layout_generation() -> u64 {
    LAYOUT_GENERATION.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn store_generation() -> u64 {
    STORE_GENERATION.load(std::sync::atomic::Ordering::Relaxed)
}

/// The installed store, if any (the pixel graph adds its sink images to a copy and re-installs it).
pub fn installed() -> Option<Arc<AssetStore>> {
    reads::observe_store();
    slot().read().expect("asset store lock").clone()
}

/// Atomically replace the live video frames, retaining the settled asset store.
/// Outstanding snapshots remain immutable; only those snapshots require a copy.
/// Every replacement invalidates plans even when the store allocation is reused.
pub fn replace_video_frames(frames: Vec<(String, Arc<crate::PreparedImage>)>) -> bool {
    let mut guard = slot().write().expect("asset store lock");
    if frames.is_empty() && guard.as_ref().is_none_or(|store| !store.entries.keys().any(|key| key.starts_with("video:"))) {
        return false;
    }
    let store = Arc::make_mut(guard.get_or_insert_with(|| Arc::new(AssetStore::default())));
    let frames: std::collections::HashMap<_, _> = frames.into_iter()
        .map(|(id, image)| (format!("video:{id}"), image)).collect();
    let same_layout = frames.iter().all(|(key, image)| store.image(key).is_some_and(|old| old.same_layout(image)));
    let removed = store.clear_prefixed("video:");
    let same_layout = same_layout && frames.len() == removed;
    let changed = removed != 0 || !frames.is_empty();
    for (key, image) in frames {
        store.insert(key, crate::AssetEntry::Loaded(image));
    }
    if !same_layout { LAYOUT_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
    if changed {
        STORE_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    changed
}

/// Remove the installed store.
pub fn clear_store() {
    let mut guard = slot().write().expect("asset store lock");
    if guard.take().is_some() {
        LAYOUT_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        STORE_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Map a `Loaded` store entry (heap OR GPU-resident external) to a
/// `ResolvedAsset`. `None` when `key` is not present as a `Loaded` entry —
/// `fill_state == Loaded` guarantees exactly one of `image`/`external` is set,
/// so a `None` here means the key is Failed/absent. Shared by
/// [`resolve`] (whose placeholder/telemetry handling wraps it) and
/// [`resolve_streaming`].
pub fn loaded_asset(store: &AssetStore, key: &str) -> Option<ResolvedAsset> {
    if let Some(img) = store.image(key) {
        Some(ResolvedAsset {
            state: ImageFillState::Loaded,
            width: img.width,
            height: img.height,
            scale_factor: img.scale_factor,
            // Share the decoded-pixel Arc rather than re-clone the Vec
            // (buffer-churn fix, main e85f7c1f).
            pixels: ResolvedPixels::Heap(Arc::clone(&img.rgba_premul)),
            straight: img.straight,
        })
    } else if let Some(meta) = store.external(key) {
        Some(ResolvedAsset {
            state: ImageFillState::Loaded,
            width: meta.width,
            height: meta.height,
            scale_factor: meta.scale_factor,
            straight: false,
            pixels: ResolvedPixels::External {
                key: key.to_string(),
                upload_id: meta.upload_id,
                texture: match store.entries.get(key) {
                    Some(crate::AssetEntry::DeviceFrame(_, texture)) => Some(texture.clone()),
                    _ => None,
                },
            },
        })
    } else {
        None
    }
}

/// Resolve a per-node streaming texture key (`video:<nodeId>`) against the
/// installed store, returning `Some` ONLY when a live frame is settled there.
///
/// The video-fill poster-vs-streaming rule (see
/// `docs/roadmap/canvas/video-nodes.md`): a `'video'` fill prefers the live
/// frame installed under `video:<nodeId>` by the VideoPlaybackController, and
/// falls back to its poster URL when none has arrived (idle / pre-playback /
/// static export of a never-played video). A miss is therefore the EXPECTED
/// state, not a missing-asset bug — so, unlike [`resolve`], this stays SILENT
/// (records no `PlaceholderBaked` scream-channel event) and returns `None` for
/// any non-`Loaded` state so the caller can resolve the poster instead.
pub fn resolve_streaming(key: &str) -> Option<ResolvedAsset> {
    resolve_streaming_with(key, reads::Read::Pixels)
}

fn resolve_streaming_with(key: &str, usage: reads::Read) -> Option<ResolvedAsset> {
    reads::observe(key, usage);
    let guard = slot().read().expect("asset store lock");
    let store = guard.as_ref()?;
    loaded_asset(store, key)
}

pub fn resolve(src: &str) -> Option<ResolvedAsset> {
    resolve_with(src, reads::Read::Pixels)
}

/// Resolve the texture for an ordinary pattern draw. Its instance depends on
/// resource layout. The publication index can replace same-layout pixels.
pub fn resolve_pattern_binding(src: &str) -> Option<ResolvedAsset> {
    resolve_with(src, reads::Read::Binding)
}

/// The live-frame arm of a pattern draw, with the ordinary poster fallback on a miss.
pub fn resolve_streaming_binding(key: &str) -> Option<ResolvedAsset> {
    resolve_streaming_with(key, reads::Read::Binding)
}

fn resolve_with(src: &str, usage: reads::Read) -> Option<ResolvedAsset> {
    reads::observe(src, usage);
    let guard = slot().read().expect("asset store lock");
    let store = guard.as_ref()?;
    match store.fill_state(src) {
        ImageFillState::Loaded => {
            // `Loaded` covers both heap (`image`) and GPU-resident external
            // (`external`) settles; fill_state guarantees exactly one is present.
            Some(loaded_asset(store, src).unwrap_or_else(|| {
                // Unreachable by construction (fill_state==Loaded ⇒ one of the
                // above). Degrade to the Loading placeholder rather than panic.
                ResolvedAsset {
                    state: ImageFillState::Loading,
                    width: 1,
                    height: 1,
                    scale_factor: (1.0, 1.0),
                    straight: false,
                    pixels: ResolvedPixels::Heap(transparent_rgba()),
                }
            }))
        }
        state => {
            // WF5 r3 scream channel: every placeholder bake at draw time is
            // recorded (deduped). `Loading` without a store entry means the
            // scene scan never requested this source — a collection bug, and
            // still a missing resource in the pixels.
            crate::report::record(crate::report::RuntimeEvent::PlaceholderBaked {
                src: src.to_string(),
                state: match (state, store.entries.contains_key(src)) {
                    (ImageFillState::Failed, _) => "failed".into(),
                    (_, true) => "timed-out".into(),
                    (_, false) => "never-prefetched".into(),
                },
            });
            Some(ResolvedAsset {
                state,
                width: 1,
                height: 1,
                scale_factor: (1.0, 1.0),
                // Shared static transparent Arc — no per-resolve allocation
                // for the placeholder (buffer-churn fix, main e85f7c1f).
                straight: false,
                pixels: ResolvedPixels::Heap(transparent_rgba()),
            })
        }
    }
}

#[cfg(test)]
mod retained_native_regressions {
    use super::*;
    use crate::{AssetEntry, PreparedImage};

    #[test]
    fn resource_generations_follow_inputs_not_store_allocations() {
        let _report = crate::report::tests::serial_globals();
        clear_store();
        let mut store = AssetStore::default();
        store.insert("still".into(), AssetEntry::Loaded(Arc::new(
            PreparedImage::from_premultiplied(vec![255; 4], 1, 1))));
        store.insert("failed".into(), AssetEntry::Failed);
        let external = crate::ExternalImageMeta { width: 10, height: 20, scale_factor: (1., 1.), upload_id: 7 };
        store.insert_external("capture".into(), external.clone());
        install_store(Arc::new(store.clone()));
        let held = installed().unwrap();
        let at = store_generation();
        install_store(Arc::new(store.clone()));
        assert_eq!(store_generation(), at, "a cloned snapshot has no changed resource");
        assert!(Arc::ptr_eq(&held, &installed().unwrap()));

        let mut changes = Vec::new();
        for meta in [
            crate::ExternalImageMeta { upload_id: 8, ..external.clone() },
            crate::ExternalImageMeta { width: 11, ..external.clone() },
            crate::ExternalImageMeta { height: 21, ..external.clone() },
            crate::ExternalImageMeta { scale_factor: (2., 1.), ..external.clone() },
        ] {
            let mut next = store.clone();
            next.insert_external("capture".into(), meta);
            changes.push(next);
        }
        let mut next = store.clone();
        next.remove("failed");
        changes.push(next);
        let mut next = store.clone();
        next.insert("still".into(), AssetEntry::Loaded(Arc::new(
            PreparedImage::from_premultiplied(vec![255; 4], 1, 1))));
        changes.push(next); // A new payload identity writes even when its pixels happen to match.
        let last = changes.len() - 1;
        for (i, next) in changes.into_iter().enumerate() {
            install_store(Arc::new(store.clone()));
            let at = store_generation();
            let layout = layout_generation();
            install_store(Arc::new(next.clone()));
            assert_eq!(layout_generation(), layout + u64::from(i != last));
            assert_eq!(store_generation(), at + 1, "a changed input advances the generation");
            install_store(Arc::new(next));
            assert_eq!(store_generation(), at + 1);
        }
        clear_store();
    }

    #[test]
    fn resolve_maps_store_states_and_none_without_store() {
        let _report = crate::report::tests::serial_globals();
        clear_store();
        assert!(resolve("data:x").is_none());

        let mut store = AssetStore::default();
        store.insert(
            "ok".into(),
            AssetEntry::Loaded(Arc::new(PreparedImage {
                width: 2,
                height: 2,
                natural_width: 1,
                natural_height: 1,
                scale_factor: (2.0, 2.0),
                rgba_premul: Arc::new(vec![1, 2, 3, 4].repeat(4)),
                is_svg: false,
                icc_applied: false,
                coverage_permille: 1000,
                straight: false,
                blank_cause: "",
            })),
        );
        store.insert("dead".into(), AssetEntry::Failed);
        install_store(Arc::new(store));

        let ok = resolve("ok").unwrap();
        assert_eq!(ok.state, ImageFillState::Loaded);
        assert_eq!((ok.width, ok.height), (2, 2));
        assert_eq!(ok.scale_factor, (2.0, 2.0));
        assert!(matches!(ok.pixels, ResolvedPixels::Heap(_)));

        // Churn guard (main e85f7c1f): the decoded-pixel Arc is shared across
        // resolves, not re-cloned — for both a heap entry and the shared
        // transparent placeholder.
        let ok_again = resolve("ok").unwrap();
        match (&ok.pixels, &ok_again.pixels) {
            (ResolvedPixels::Heap(a), ResolvedPixels::Heap(b)) => assert!(Arc::ptr_eq(a, b)),
            _ => panic!("heap raster must resolve to Heap pixels"),
        }

        let dead = resolve("dead").unwrap();
        let dead_again = resolve("dead").unwrap();
        match (&dead.pixels, &dead_again.pixels) {
            (ResolvedPixels::Heap(a), ResolvedPixels::Heap(b)) => assert!(Arc::ptr_eq(a, b)),
            _ => panic!("failed placeholder must resolve to Heap pixels"),
        }
        assert_eq!(resolve("dead").unwrap().state, ImageFillState::Failed);
        // never requested → Loading
        assert_eq!(resolve("unknown").unwrap().state, ImageFillState::Loading);

        clear_store();

        // External (browser-decoded, GPU-resident) raster: Loaded state, no heap
        // bytes, carries the upload key + id for the engine's external uploader.
        let mut store = AssetStore::default();
        store.insert_external(
            "ext".into(),
            crate::ExternalImageMeta {
                width: 512,
                height: 342,
                scale_factor: (5.0, 5.0),
                upload_id: 7,
            },
        );
        install_store(Arc::new(store));
        let ext = resolve("ext").unwrap();
        assert_eq!(ext.state, ImageFillState::Loaded);
        assert_eq!((ext.width, ext.height), (512, 342));
        assert_eq!(ext.scale_factor, (5.0, 5.0));
        match ext.pixels {
            ResolvedPixels::External { key, upload_id, .. } => {
                assert_eq!(key, "ext");
                assert_eq!(upload_id, 7);
            }
            ResolvedPixels::Heap(_) => panic!("external raster must resolve to External pixels"),
        }

        // clear_external_images parity at the seam: clearing the external entry
        // and republishing the store must make resolve stop reporting External
        // (otherwise a render on a reused instance samples a closed bitmap).
        // Mirrors the wasm free fn, which clears pending.store then re-installs.
        let mut cleared = AssetStore::default();
        cleared.insert_external(
            "ext".into(),
            crate::ExternalImageMeta {
                width: 512,
                height: 342,
                scale_factor: (5.0, 5.0),
                upload_id: 7,
            },
        );
        cleared.clear_external();
        install_store(Arc::new(cleared));
        assert_eq!(resolve("ext").unwrap().state, ImageFillState::Loading);
        assert!(matches!(
            resolve("ext").unwrap().pixels,
            ResolvedPixels::Heap(_)
        ));
        clear_store();

        // resolve_streaming (video poster-vs-streaming rule): returns a
        // resolved asset ONLY for a settled (Loaded) key, and None for every
        // other state WITHOUT touching the scream channel (a miss is the
        // expected idle/pre-playback state, not a missing-asset bug).
        assert!(resolve_streaming("video:n1").is_none()); // no store installed

        let mut store = AssetStore::default();
        store.insert_external(
            "video:n1".into(),
            crate::ExternalImageMeta {
                width: 640,
                height: 360,
                scale_factor: (1.0, 1.0),
                upload_id: 42,
            },
        );
        store.insert(
            "video:heap".into(),
            AssetEntry::Loaded(Arc::new(PreparedImage {
                width: 4,
                height: 4,
                natural_width: 4,
                natural_height: 4,
                scale_factor: (1.0, 1.0),
                rgba_premul: Arc::new(vec![1, 2, 3, 4].repeat(16)),
                is_svg: false,
                icc_applied: false,
                coverage_permille: 1000,
                straight: false,
                blank_cause: "",
            })),
        );
        store.insert("video:dead".into(), AssetEntry::Failed);
        install_store(Arc::new(store));

        // Live external frame → External pixels carrying the streaming key + id.
        let streamed = resolve_streaming("video:n1").expect("settled streaming key resolves");
        assert_eq!(streamed.state, ImageFillState::Loaded);
        assert_eq!((streamed.width, streamed.height), (640, 360));
        match streamed.pixels {
            ResolvedPixels::External { key, upload_id, .. } => {
                assert_eq!(key, "video:n1");
                assert_eq!(upload_id, 42);
            }
            ResolvedPixels::Heap(_) => panic!("a live video frame must resolve to External pixels"),
        }
        // A heap-fed live frame (the unsafe-GPU fallback lane) resolves to Heap.
        assert!(matches!(
            resolve_streaming("video:heap").unwrap().pixels,
            ResolvedPixels::Heap(_)
        ));
        // Failed / never-installed streaming keys fall through to None (→ poster).
        assert!(resolve_streaming("video:dead").is_none());
        assert!(resolve_streaming("video:never").is_none());
        clear_store();

        let image = Arc::new(PreparedImage {
            width: 1,
            height: 1,
            natural_width: 1,
            natural_height: 1,
            scale_factor: (1.0, 1.0),
            rgba_premul: Arc::new(vec![1, 2, 3, 255]),
            is_svg: false,
            icc_applied: false,
            coverage_permille: 1000,
            straight: false,
            blank_cause: "",
        });
        assert!(!replace_video_frames(Vec::new()));
        assert!(installed().is_none());
        let mut store = AssetStore::default();
        store.insert("still".into(), AssetEntry::Loaded(image.clone()));
        install_store(Arc::new(store));
        let address = Arc::as_ptr(&installed().unwrap());
        let generation = store_generation();
        assert!(replace_video_frames(vec![("first".into(), image.clone())]));
        assert_eq!(address, Arc::as_ptr(&installed().unwrap()));
        assert_eq!(store_generation(), generation + 1);
        let snapshot = installed().unwrap();
        assert!(replace_video_frames(vec![("second".into(), image)]));
        assert!(snapshot.image("video:first").is_some());
        assert!(snapshot.image("video:second").is_none());
        assert!(resolve_streaming("video:first").is_none());
        assert!(resolve_streaming("video:second").is_some());
        assert!(installed().unwrap().image("still").is_some());
        assert!(replace_video_frames(Vec::new()));
        assert!(resolve_streaming("video:second").is_none());
        let generation = store_generation();
        assert!(!replace_video_frames(Vec::new()));
        assert_eq!(store_generation(), generation);
        clear_store();
    }
}
