//! moda-assets — the moda-render asset layer (PLAN.md M3c).
//!
//! Pipeline: LOCAL source resolution (data URI / vendored file; an http(s)
//! source is a per-asset failure — documents render offline) → bounded
//! parallel prefetch → decode (`image`) → ICC→sRGB (`moxcms`) → premultiply → production
//! sizing policy resize → `rgba8unorm` upload, or resvg rasterization for
//! SVG. Placeholder semantics are state-driven, exactly like
//! `WebGPUTextureManager.getImageFillState`.
//!
//! Public surface the engine/paint tracks consume (SEAMS-compatible):
//! - [`AssetStore`] — post-prefetch lookup: `fill_state` / `dimensions` /
//!   `scale_factor` / `image` (heap-decoded), plus `external` for a
//!   browser-decoded, GPU-resident raster (wasm editor lane —
//!   `insert_external` / `clear_external`, no pixels on the heap).
//! - [`pattern::build_pattern_paint`] → [`pattern::PatternPaint`] — the
//!   per-node pattern instance (416-byte `PathPatternData` ABI) + texture
//!   key + sampler choice.
//! - [`probe::probe_svg`] — usvg per-asset manifest probe.

pub mod constants;
pub mod decode;
pub mod engine_seam;
#[cfg(feature = "native-io")]
pub mod fetch;
pub mod pattern;
pub mod probe;
pub mod report;
pub mod sizing;
pub mod source;
pub mod svg;

use std::collections::HashMap;
use std::sync::Arc;

pub use decode::PreparedImage;
#[cfg(feature = "native-io")]
pub use fetch::{
    fetch_media_bytes_blocking, prefetch_blocking, AssetFailure, BlankAsset, PrefetchConfig,
    PrefetchReport, SvgProbeWarning, REMOTE_URL_NOT_ALLOWED,
};
pub use pattern::{ImageFillState, PatternInstance, PatternPaint};

/// The fontdb byte-decode callers (the wasm editor lane) lay out SVG `<text>` with: the text
/// engine's fonts ([`svg::engine_fontdb`]).
fn default_fontdb() -> Arc<resvg::usvg::fontdb::Database> {
    svg::engine_fontdb()
}

/// Decode pattern-fill bytes that an external host fetched (the wasm
/// editor's browser-fetch feeder) into a [`PreparedImage`] — the same
/// SVG-sniff + decode/ICC/premultiply/sizing pipeline `fetch::load_one`
/// runs after its own fetch. No IO, so not `native-io`-gated.
pub fn decode_pattern_bytes(
    src: &str,
    bytes: &[u8],
    min_raster_size: Option<(u32, u32)>,
) -> anyhow::Result<PreparedImage> {
    anyhow::ensure!(!bytes.is_empty(), "empty asset body for {src}");
    let is_svg = source::is_likely_svg(src) || source::sniff_svg_bytes(bytes);
    let prepared = if is_svg {
        svg::prepare_svg(bytes, min_raster_size, &default_fontdb())
    } else {
        decode::prepare_raster(bytes)
    };
    // A decode that succeeded and paints NOTHING. Recorded on the
    // runtime-event channel (deduped per src, inert unless a report lane armed
    // the collector) so the wasm editor lane — whose fontless SVG rasterization
    // is the main way this happens — is no longer silent about it.
    if let Ok(img) = &prepared {
        if img.coverage_permille == 0 {
            report::record(report::RuntimeEvent::BlankRasterDecoded {
                src: report::sanitized_asset_ref(src),
                coverage_permille: img.coverage_permille,
                cause: img.blank_cause.to_string(),
            });
        }
    }
    prepared
}

/// One asset request from the scene scan ([`collect_pattern_requests`]).
/// Lives here (not in the `native-io`-gated `fetch` module) because the scan
/// itself does no IO: the wasm editor collects requests engine-side and lets
/// the BROWSER fetch the bytes (`renderers/wasm/wasmAssetFeeder.ts`), while
/// the CLI feeds the same requests to [`fetch::prefetch_blocking`].
#[derive(Debug, Clone)]
pub struct PrefetchRequest {
    pub src: String,
    /// Desired min raster size (SVG only; `computeMinRasterSizeForPatternFill`).
    pub min_raster_size: Option<(u32, u32)>,

    /// Requesting node id (WF5 r3 resource report attribution).
    pub node_id: Option<String>,
}

/// The store key of raster `src` prepared without the sizing policy and without 8-bit association: its decoded
/// texels, straight alpha, at their natural size ([`decode::prepare_raster_natural`]). A request for this key
/// fetches `src` and prepares it so; the plain `src` keys the same source as a browser prepares a bitmap. The
/// two are different pixels of one file, so they are two entries.
pub fn natural_key(src: &str) -> String {
    format!("{NATURAL_KEY}{src}")
}

/// The source a [`natural_key`] names; `None` for any other key.
pub fn natural_source(key: &str) -> Option<&str> {
    key.strip_prefix(NATURAL_KEY)
}

const NATURAL_KEY: &str = "moda-natural:";

/// Terminal state of one prefetched asset.
#[derive(Debug, Clone)]
pub enum AssetEntry {
    /// Decoded, premultiplied bytes resident on the heap (the native CLI
    /// lane, and the wasm SVG/shader-image paths).
    Loaded(Arc<PreparedImage>),
    /// A browser-decoded raster uploaded straight to a GPU texture (the wasm
    /// editor lane): the store holds only this metadata,
    /// so NO decoded RGBA `Vec` ever lands on the wasm heap. The pixels are
    /// pulled from the installed `ExternalImageUploader` at bind time, keyed
    /// by the store key.
    LoadedExternal(ExternalImageMeta),
    /// Immutable device-local footage pixels. Snapshots retain the texture.
    DeviceFrame(ExternalImageMeta, Arc<wgpu::Texture>),
    /// Committed failure (remote URL, decode error, missing file) →
    /// rose placeholder.
    Failed,
}

/// Metadata for a [`AssetEntry::LoadedExternal`] raster — exactly what the
/// engine seam consumes to size the texture and pack the pattern instance,
/// carrying NO pixel bytes. The decoded pixels live in the browser (an
/// `ImageBitmap` held by the wasm feeder) and go ImageBitmap → GPU texture via
/// `copy_external_image_to_texture`, never through the wasm linear heap.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalImageMeta {
    /// Uploaded texel dimensions (post sizing policy = the ImageBitmap size).
    pub width: u32,
    pub height: u32,
    /// uploaded / natural per axis (`CachedTexture.scaleFactor`) — the pattern
    /// instance divides `fillPatternScale` by this.
    pub scale_factor: (f64, f64),
    /// Monotonic per-install id — the retained-cache identity for the external
    /// texture. The heap lane keys the texture cache on the decoded-pixel
    /// `Arc` pointer; an external upload has no such `Arc`, so a re-install
    /// bumps this id to self-invalidate the cached texture (MAP §5.7 parity).
    pub upload_id: u64,
}

/// Immutable post-prefetch asset lookup. The native analog of
/// `WebGPUTextureManager`'s cache+state surface, with the async machinery
/// stripped: by render time every source has settled. M4e adds the
/// shader-image side (`ShaderImageResourceManager` analog), keyed by the
/// production resource key (`createImageResourceKey`).
#[derive(Debug, Clone, Default)]
pub struct AssetStore {
    entries: HashMap<String, AssetEntry>,
}

impl AssetStore {
    pub(crate) fn same_layout(&self, other: &Self) -> bool {
        self.entries.len() == other.entries.len() && self.entries.iter().all(|(key, entry)| {
            match (entry, other.entries.get(key)) {
                (AssetEntry::Loaded(a), Some(AssetEntry::Loaded(b))) => Arc::ptr_eq(a, b) || a.same_layout(b),
                (AssetEntry::LoadedExternal(a), Some(AssetEntry::LoadedExternal(b))) => a == b,
                (AssetEntry::DeviceFrame(a, _), Some(AssetEntry::DeviceFrame(b, _))) =>
                    a.width == b.width && a.height == b.height && a.scale_factor == b.scale_factor,
                (AssetEntry::Failed, Some(AssetEntry::Failed)) => true,
                _ => false,
            }
        })
    }

    /// Whether every resource has the same identity. Cloning a store does not write its resources:
    /// decoded payloads are immutable Arcs, and external pixels carry an explicit upload generation.
    pub(crate) fn same_resources(&self, other: &Self) -> bool {
        self.entries.len() == other.entries.len()
            && self.entries.iter().all(|(key, entry)| match (entry, other.entries.get(key)) {
                (AssetEntry::Loaded(a), Some(AssetEntry::Loaded(b))) => Arc::ptr_eq(a, b),
                (AssetEntry::LoadedExternal(a), Some(AssetEntry::LoadedExternal(b))) => {
                    a.width == b.width && a.height == b.height && a.upload_id == b.upload_id
                        && a.scale_factor.0.to_bits() == b.scale_factor.0.to_bits()
                        && a.scale_factor.1.to_bits() == b.scale_factor.1.to_bits()
                }
                (AssetEntry::DeviceFrame(a, _), Some(AssetEntry::DeviceFrame(b, _))) => a == b,
                (AssetEntry::Failed, Some(AssetEntry::Failed)) => true,
                _ => false,
            })
    }

    pub fn extend(&mut self, other: AssetStore) {
        self.entries.extend(other.entries);
    }

    pub fn insert(&mut self, src: String, entry: AssetEntry) {
        self.entries.insert(src, entry);
    }

    /// Insert a browser-decoded, GPU-resident raster (wasm editor lane): the
    /// store carries only `meta`, never the decoded bytes.
    pub fn insert_external(&mut self, src: String, meta: ExternalImageMeta) {
        self.entries.insert(src, AssetEntry::LoadedExternal(meta));
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `getImageFillState` (`WebGPUTextureManager.ts:445-459`): never
    /// requested → Loading; committed failure → Failed. A
    /// GPU-resident external raster is `Loaded` (real pixels, just not on the
    /// heap).
    pub fn fill_state(&self, src: &str) -> ImageFillState {
        match self.entries.get(src) {
            Some(AssetEntry::Loaded(_)) | Some(AssetEntry::LoadedExternal(_)) | Some(AssetEntry::DeviceFrame(_, _)) => {
                ImageFillState::Loaded
            }
            Some(AssetEntry::Failed) => ImageFillState::Failed,
            None => ImageFillState::Loading,
        }
    }

    /// `getTextureDimensions` — uploaded texel dims (heap OR external).
    pub fn dimensions(&self, src: &str) -> Option<(u32, u32)> {
        match self.entries.get(src) {
            Some(AssetEntry::Loaded(i)) => Some((i.width, i.height)),
            Some(AssetEntry::LoadedExternal(m) | AssetEntry::DeviceFrame(m, _)) => Some((m.width, m.height)),
            _ => None,
        }
    }

    /// `getTextureScaleFactor` — uploaded / natural, per axis (heap OR external).
    pub fn scale_factor(&self, src: &str) -> Option<(f64, f64)> {
        match self.entries.get(src) {
            Some(AssetEntry::Loaded(i)) => Some(i.scale_factor),
            Some(AssetEntry::LoadedExternal(m) | AssetEntry::DeviceFrame(m, _)) => Some(m.scale_factor),
            _ => None,
        }
    }

    /// The decoded heap image, if this source settled as heap-resident. An
    /// external (GPU-resident) raster returns `None` here — use
    /// [`AssetStore::external`] for its metadata.
    pub fn image(&self, src: &str) -> Option<&PreparedImage> {
        match self.entries.get(src) {
            Some(AssetEntry::Loaded(img)) => Some(img),
            _ => None,
        }
    }

    /// The metadata for a browser-decoded, GPU-resident raster (wasm editor
    /// lane). `None` for heap/failed/never-requested sources.
    pub fn external(&self, src: &str) -> Option<&ExternalImageMeta> {
        match self.entries.get(src) {
            Some(AssetEntry::LoadedExternal(m) | AssetEntry::DeviceFrame(m, _)) => Some(m),
            _ => None,
        }
    }

    /// Drop every browser-decoded external raster entry, returning how many were
    /// removed. Pairs with the wasm `clear_external_images`: the session's
    /// `ImageBitmap` registry and this metadata must be torn down together. A
    /// retained `LoadedExternal` keeps `fill_state` reporting `Loaded`, so
    /// `collect_pattern_requests` would treat the key as settled and never
    /// re-feed it — and on a reused wasm instance the engine would then resolve
    /// an external texture whose bitmap is gone, rendering transparent. Heap,
    /// shader, and failed entries are untouched (their pixels live in the store,
    /// independent of any bitmap).
    pub fn clear_external(&mut self) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|_, e| !matches!(e, AssetEntry::LoadedExternal(_)));
        before - self.entries.len()
    }

    /// Drop a SINGLE entry by key, returning whether one was present. The
    /// per-key analog of [`AssetStore::clear_external`]: the live-video lane
    /// releases one node's streaming texture (`video:<nodeId>`) on stop / node
    /// delete / page leave without disturbing other videos or static assets.
    /// Reverts the key to `Loading` at the seam (the next `collect`/feed would
    /// re-settle it) — the wasm `release_external_image` pairs this with tearing
    /// down the matching `<video>`/registry entry so the engine can't resolve a
    /// texture whose source element is gone.
    pub fn remove(&mut self, src: &str) -> bool {
        self.entries.remove(src).is_some()
    }

    /// Drop every entry whose key starts with `prefix`, returning how many were
    /// removed. Used by the wasm `clear_external_images` to tear down the
    /// live-video streaming keys (`video:<nodeId>`) on canvas teardown:
    /// HEAP-fed frames settle as `Loaded` (not `LoadedExternal`), so
    /// [`AssetStore::clear_external`] alone leaves them resident in the
    /// process-global store across a canvas switch (the wasm instance is reused).
    /// Generic over the prefix to keep the store unaware of the `video:`
    /// convention.
    pub fn clear_prefixed(&mut self, prefix: &str) -> usize {
        // An empty prefix matches every key (`str::starts_with("")` is always
        // true) and would silently wipe the whole store. No caller passes one
        // today; the guard documents the invariant and trips a future misuse
        // (e.g. a computed prefix from the planned video-export path) in
        // debug/test builds without changing release behavior.
        debug_assert!(
            !prefix.is_empty(),
            "clear_prefixed called with an empty prefix (would wipe the store)"
        );
        if prefix.is_empty() {
            return 0;
        }
        let before = self.entries.len();
        self.entries.retain(|k, _| !k.starts_with(prefix));
        before - self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_entry_accessors() {
        let mut store = AssetStore::default();
        store.insert_external(
            "k".into(),
            ExternalImageMeta {
                width: 256,
                height: 128,
                scale_factor: (2.0, 4.0),
                upload_id: 3,
            },
        );
        // A browser-decoded, GPU-resident raster reports Loaded with its
        // uploaded dims + scale factor, but carries NO heap image (the pixels
        // never entered the wasm heap).
        assert_eq!(store.fill_state("k"), ImageFillState::Loaded);
        assert_eq!(store.dimensions("k"), Some((256, 128)));
        assert_eq!(store.scale_factor("k"), Some((2.0, 4.0)));
        assert!(store.image("k").is_none());
        let meta = store.external("k").expect("external meta present");
        assert_eq!((meta.width, meta.height), (256, 128));
        assert_eq!(meta.upload_id, 3);

        // A heap entry is not external; a never-requested key is neither.
        store.insert(
            "heap".into(),
            AssetEntry::Loaded(Arc::new(PreparedImage {
                width: 4,
                height: 4,
                natural_width: 4,
                natural_height: 4,
                scale_factor: (1.0, 1.0),
                rgba_premul: Arc::new(vec![0; 4 * 4 * 4]),
                is_svg: false,
                icc_applied: false,
                coverage_permille: 1000,
                straight: false,
                blank_cause: "",
            })),
        );
        assert!(store.external("heap").is_none());
        assert!(store.image("heap").is_some());
        assert!(store.external("missing").is_none());
        assert_eq!(store.fill_state("missing"), ImageFillState::Loading);
    }

    #[test]
    fn clear_external_drops_only_external_entries_and_reverts_to_loading() {
        let mut store = AssetStore::default();
        store.insert_external(
            "ext".into(),
            ExternalImageMeta {
                width: 8,
                height: 8,
                scale_factor: (1.0, 1.0),
                upload_id: 1,
            },
        );
        store.insert(
            "heap".into(),
            AssetEntry::Loaded(Arc::new(PreparedImage {
                width: 2,
                height: 2,
                natural_width: 2,
                natural_height: 2,
                scale_factor: (1.0, 1.0),
                rgba_premul: Arc::new(vec![0; 2 * 2 * 4]),
                is_svg: false,
                icc_applied: false,
                coverage_permille: 1000,
                straight: false,
                blank_cause: "",
            })),
        );
        store.insert("failed".into(), AssetEntry::Failed);

        // Mirrors `clear_external_images` teardown on a reused wasm instance:
        // only the external metadata is dropped, and it must revert to Loading
        // so `collect_pattern_requests` re-feeds the now-closed bitmap rather
        // than resolving a bitmap-less external texture (which renders
        // transparent). Heap and failed entries are untouched.
        assert_eq!(store.clear_external(), 1);
        assert_eq!(store.fill_state("ext"), ImageFillState::Loading);
        assert!(store.external("ext").is_none());
        assert!(store.image("heap").is_some());
        assert_eq!(store.fill_state("failed"), ImageFillState::Failed);

        // Idempotent: a second clear removes nothing.
        assert_eq!(store.clear_external(), 0);
    }

    #[test]
    fn clear_prefixed_drops_streaming_keys_heap_and_external_only() {
        let mut store = AssetStore::default();
        // A heap-fed live video frame — `Loaded`, NOT `LoadedExternal`.
        store.insert(
            "video:n1".into(),
            AssetEntry::Loaded(Arc::new(PreparedImage {
                width: 4,
                height: 4,
                natural_width: 4,
                natural_height: 4,
                scale_factor: (1.0, 1.0),
                rgba_premul: Arc::new(vec![0; 4 * 4 * 4]),
                is_svg: false,
                icc_applied: false,
                coverage_permille: 1000,
                straight: false,
                blank_cause: "",
            })),
        );
        // An external live video frame.
        store.insert_external(
            "video:n2".into(),
            ExternalImageMeta {
                width: 8,
                height: 8,
                scale_factor: (1.0, 1.0),
                upload_id: 9,
            },
        );
        // A normal (non-video) image — must survive.
        store.insert(
            "https://cdn/img.png".into(),
            AssetEntry::Loaded(Arc::new(PreparedImage {
                width: 2,
                height: 2,
                natural_width: 2,
                natural_height: 2,
                scale_factor: (1.0, 1.0),
                rgba_premul: Arc::new(vec![0; 2 * 2 * 4]),
                is_svg: false,
                icc_applied: false,
                coverage_permille: 1000,
                straight: false,
                blank_cause: "",
            })),
        );

        // `clear_external` alone would leave the HEAP video frame resident — the
        // canvas-switch leak. `clear_prefixed("video:")` drops the whole streaming
        // key space (heap + external) and nothing else.
        assert_eq!(store.clear_prefixed("video:"), 2);
        assert_eq!(store.fill_state("video:n1"), ImageFillState::Loading);
        assert_eq!(store.fill_state("video:n2"), ImageFillState::Loading);
        assert!(store.image("https://cdn/img.png").is_some());
        assert_eq!(store.clear_prefixed("video:"), 0); // idempotent
    }

    #[test]
    #[cfg(debug_assertions)] // release builds compile the debug_assert (and this test) out.
    #[should_panic(expected = "empty prefix")]
    fn clear_prefixed_rejects_an_empty_prefix() {
        // The guard stops a future caller from silently wiping the whole store
        // with `clear_prefixed("")`; release builds fall through to the no-op.
        let mut store = AssetStore::default();
        store.clear_prefixed("");
    }

}
