//! The ambient faces used by a kept Vork text host. Resolve the same directories
//! as fontdb and freeze the bytes before they reach a shaper or a raster cache.
use super::*;
use std::{cell::RefCell, collections::HashSet, sync::Arc};

type Blobs = Vec<Arc<Vec<u8>>>;

#[derive(Eq, Hash, PartialEq)]
struct Catalog {
    fonts: Blobs,
    baked: Option<BakedCatalog>,
}

fn faces(load: impl FnOnce(&mut fontdb::Database), colour_only: bool) -> Blobs {
    let mut db = fontdb::Database::new();
    load(&mut db);
    let mut seen = HashSet::new();
    let mut blobs = Vec::new();
    for face in db.faces() {
        let path = match &face.source {
            fontdb::Source::File(path) | fontdb::Source::SharedFile(path, _) => path,
            fontdb::Source::Binary(_) => unreachable!("fontdb directory sources are files"),
        };
        if !seen.insert(path) { continue; }
        if let Some(Some(bytes)) = db.with_face_data(face.id, |bytes, _| {
            // A collection can contain both ordinary and colour faces. Keep the
            // blob if any face qualifies, then let the colour reader select faces.
            let count = rustybuzz::ttf_parser::fonts_in_collection(bytes).unwrap_or(1);
            (!colour_only || (0..count).any(|index| colour_glyphs::colour_data(bytes, index)))
                .then(|| Arc::new(bytes.to_vec()))
        }) {
            blobs.push(bytes);
        }
    }
    blobs
}

impl Catalog {
    fn fallback() -> Self {
        let fonts = faces(|db| {
            if let Some(dir) = resolve_degraded_fallback_dir() { db.load_fonts_dir(dir); }
        }, false);
        Self { fonts, baked: read_baked_catalog() }
    }

    fn system_colours() -> Self {
        Self { fonts: faces(|db| db.load_system_fonts(), true), baked: None }
    }

}

#[derive(Clone, Copy)]
enum Faces { Fallback, SystemColours }

impl vork_inputs::Query for Faces {
    type Value = Arc<Catalog>;
    fn key(&self) -> &std::path::Path {
        std::path::Path::new(match self { Self::Fallback => "fallback", Self::SystemColours => "system-colours" })
    }
    fn read(&self) -> std::io::Result<Self::Value> {
        Ok(Arc::new(match self { Self::Fallback => Catalog::fallback(), Self::SystemColours => Catalog::system_colours() }))
    }
}

/// Vector-only hosts never open the ambient catalog. A host that does uses
/// this publication even when it first shapes a later frame on another thread.
#[derive(Clone, Default)]
pub struct FontInputs(Arc<Catalogs>);

#[derive(Default)]
struct Catalogs {
    reads: vork_inputs::Receipt,
    fallback: OnceLock<Arc<Catalog>>,
    system_colours: OnceLock<Arc<Catalog>>,
}

thread_local! {
    static CURRENT: RefCell<Option<FontInputs>> = const { RefCell::new(None) };
}

pub struct FontScope(Option<FontInputs>);
impl Drop for FontScope {
    fn drop(&mut self) { CURRENT.with(|slot| slot.replace(self.0.take())); }
}

impl FontInputs {
    pub fn current() -> Option<Self> { CURRENT.with_borrow(Clone::clone) }
    pub fn enter(&self) -> FontScope { FontScope(CURRENT.with(|slot| slot.replace(Some(self.clone())))) }
    pub fn same(&self, other: &Self) -> bool { Arc::ptr_eq(&self.0, &other.0) }
    pub fn live_stamp(&self) -> blake3::Hash { self.0.reads.live_stamp() }

    pub fn refreshed(&self) -> Self {
        if self.0.reads.valid() { self.clone() } else { Self::default() }
    }

    fn catalog(&self, kind: Faces) -> &Catalog {
        let slot = match kind { Faces::Fallback => &self.0.fallback, Faces::SystemColours => &self.0.system_colours };
        slot.get_or_init(|| {
            let _scope = self.0.reads.enter();
            vork_inputs::read(kind).expect("font discovery supplies the faces available to the text host")
        })
    }

    pub(super) fn system_colours(&self) -> &Blobs {
        &self.catalog(Faces::SystemColours).fonts
    }

    pub(crate) fn engine(&self, root: &std::path::Path, manifest: &serde_json::Value, canvas: bool) -> DegradedEngine {
        // Vork scene hosts declare only their own text faces. The OS adds colour glyphs only.
        debug_assert!(!manifest_allows_system_fonts(manifest));
        let snapshot = self.catalog(Faces::Fallback);
        let mut db = fontdb::Database::new();
        for blob in &snapshot.fonts { db.load_font_source(fontdb::Source::Binary(blob.clone())); }
        let mut engine = compose_loaded(db, snapshot.baked.as_ref().filter(|b| b.dir != root), root, Some(manifest), canvas, false);
        if let Some(emoji) = engine.emoji_fonts.as_mut() {
            emoji.ambient = Some(self.clone());
        }
        engine
    }
}
