//! Colour glyphs are shaped once in an exact face. A raster has its own sampling
//! rectangle at its requested ppem. Neither rectangle nor advance defines the ink.
use super::*;
use std::sync::Arc;

/// A colour-only view of the host's faces. System colour fallback is available
/// even when the scene excludes system text fonts. Its cmap must never become a
/// candidate for ordinary text, so only the colour scaler sees those extra faces.
pub(super) struct EmojiFonts {
    lists: HashMap<String, Vec<(u64, fontdb::ID)>>,
    system_loaded: bool,
    state: OnceLock<Mutex<ColourState>>,
}

struct ColourState {
    authored: HashMap<String, Vec<fontdb::ID>>,
    fallback: Vec<fontdb::ID>,
    font_system: FontSystem,
    swash: SwashCache,
    shapes: crate::lru::ByteLruCache<(fontdb::ID, String), Option<Arc<ShapedEmoji>>>,
    rasters: crate::lru::ByteLruCache<(u64, u32), EmojiRaster>,
    effects: crate::lru::ByteLruCache<(u64, u32, [u32; 9]), Arc<crate::host::RgbaImage>>,
}

fn colour_face(db: &fontdb::Database, id: fontdb::ID) -> bool {
    db.with_face_data(id, |data, index| {
        rustybuzz::ttf_parser::Face::parse(data, index).is_ok_and(|face| {
            let t = face.tables();
            t.cbdt.is_some() || t.colr.is_some() || t.sbix.is_some()
        })
    })
    .unwrap_or(false)
}

impl EmojiFonts {
    pub(super) fn new(
        mut lists: HashMap<String, Vec<(u64, fontdb::ID)>>,
        system_loaded: bool,
    ) -> Self {
        for entries in lists.values_mut() {
            entries.sort_by_key(|&(order, _)| order);
        }
        Self {
            lists,
            system_loaded,
            state: OnceLock::new(),
        }
    }

    fn build(&self, host: &DegradedEngine) -> ColourState {
        let inner = host.inner.lock().unwrap();
        let db = inner.font_system.db();
        let mut colours = fontdb::Database::new();
        let mut ids = HashMap::new();
        let mut ambient = Vec::new();
        let listed: std::collections::HashSet<_> =
            self.lists.values().flatten().map(|entry| entry.1).collect();
        for face in db.faces().filter(|f| colour_face(db, f.id)) {
            let id = colours.push_face_info(face.clone());
            ids.insert(face.id, id);
            if !listed.contains(&face.id) {
                ambient.push(id);
            }
        }
        drop(inner);
        if !self.system_loaded {
            let mut system = fontdb::Database::new();
            system.load_system_fonts();
            ambient.extend(
                system
                    .faces()
                    .filter(|f| colour_face(&system, f.id))
                    .map(|face| colours.push_face_info(face.clone())),
            );
        }
        ambient.sort_by_key(|id| {
            COLOR_EMOJI_FAMILIES
                .iter()
                .position(|name| {
                    colours
                        .face(*id)
                        .unwrap()
                        .families
                        .iter()
                        .any(|(f, _)| f == name)
                })
                .unwrap_or(usize::MAX)
        });
        let authored = self
            .lists
            .iter()
            .map(|(family, entries)| {
                (
                    family.clone(),
                    entries
                        .iter()
                        .filter_map(|(_, id)| ids.get(id).copied())
                        .collect(),
                )
            })
            .collect();
        ColourState {
            authored,
            fallback: ambient,
            font_system: FontSystem::new_with_locale_and_db("en-US".into(), colours),
            swash: SwashCache::new(),
            shapes: crate::lru::ByteLruCache::new(crate::host::emoji_cap_bytes(), 4096),
            rasters: crate::lru::ByteLruCache::new(crate::host::emoji_cap_bytes(), 2048),
            effects: crate::lru::ByteLruCache::new(crate::host::emoji_cap_bytes(), 128),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct ShapedEmoji {
    pub id: u64,
    face: fontdb::ID,
    glyphs: Vec<(u16, f32, f32)>,
    pub advance_em: f64,
    pub ink_em: [f64; 4],
    sampled_size_em: [f64; 2],
    /// Bitmap-only faces draw from their available strikes.
    bitmap_ppem: Option<u32>,
}

#[derive(Clone)]
pub struct EmojiRaster {
    pub image: Arc<crate::host::RgbaImage>,
    pub sampled_em: [f64; 4],
    pub ink_em: [f64; 4],
}

impl ShapedEmoji {
    pub fn glyph_bytes(&self) -> usize {
        self.glyphs.capacity() * std::mem::size_of::<(u16, f32, f32)>()
    }

    pub fn ppem(&self, requested: f64) -> u32 {
        self.fit_ppem(self.bitmap_ppem.map(f64::from).unwrap_or(requested), 0.)
    }

    /// Keep the entire positioned cluster and its effect border within the raster budget.
    pub fn fit_ppem(&self, requested: f64, pad_em: f64) -> u32 {
        let [w, h] = self.sampled_size_em.map(|side| side + 2. * pad_em);
        let limit = ((MAX_RASTER_TEXELS.sqrt() - 4.) / (w * h).sqrt())
            .min(8190. / w.max(h))
            .max(1.);
        requested.ceil().clamp(1., limit).floor() as u32
    }
}

impl DegradedEngine {
    pub fn resolve_emoji(&self, family: &str, cluster: &str) -> Option<Arc<ShapedEmoji>> {
        let fonts = self.emoji_fonts.as_ref()?;
        let mut inner = fonts
            .state
            .get_or_init(|| Mutex::new(fonts.build(self)))
            .lock()
            .unwrap();
        let candidates: Vec<_> = inner
            .authored
            .get(&family.to_lowercase())
            .into_iter()
            .flatten()
            .chain(&inner.fallback)
            .copied()
            .collect();
        let mut try_text = |text: &str| {
            candidates.iter().find_map(|&face| {
                let key = (face, text.to_owned());
                if let Some(hit) = inner.shapes.get(&key) {
                    return hit;
                }
                let run = shape(&mut inner, face, text).map(Arc::new);
                let bytes = text.len()
                    + run.as_ref().map_or(0, |r| r.glyph_bytes());
                inner.shapes.insert(key, run.clone(), bytes as u64);
                run
            })
        };
        try_text(cluster).or_else(|| {
            (cluster.starts_with('🏴') && cluster.ends_with('\u{e007f}'))
                .then(|| try_text("🏴"))
                .flatten()
        })
    }

    pub fn emoji_raster(&self, run: &ShapedEmoji, ppem: u32) -> Option<EmojiRaster> {
        let mut inner = self.emoji_fonts.as_ref()?.state.get()?.lock().unwrap();
        let key = (run.id, ppem);
        if let Some(hit) = inner.rasters.get(&key) {
            return Some(hit);
        }
        let raster = rasterize(&mut inner, run, ppem)?;
        inner
            .rasters
            .insert(key, raster.clone(), raster.image.rgba.len() as u64);
        Some(raster)
    }

    pub(super) fn draw_colour_run(
        &self,
        run: &ShapedEmoji,
        canvas: &mut Canvas,
        origin: (f64, f64),
        size: f64,
        alpha: f64,
    ) {
        // Swash scales bitmap strikes as well as outlines to this paragraph's raster.
        let ppem = run.fit_ppem(size.round(), 0.);
        let Some(cell) = self.emoji_raster(run, ppem) else {
            return;
        };
        let left = (origin.0 + cell.sampled_em[0] * size).round();
        let top = (origin.1 + cell.sampled_em[1] * size).round();
        let img = cell.image;
        // In a very large paragraph the cluster may hit the single-image
        // raster budget. Its sampled texels still span the full logical size.
        let scale = size / f64::from(ppem);
        let visible = |start: f64, length: u32, extent: u32| {
            (start.max(0.) as u32)
                ..((start + f64::from(length) * scale)
                    .round()
                    .clamp(0., f64::from(extent)) as u32)
        };
        let xs = visible(left, img.width, canvas.w);
        for y in visible(top, img.height, canvas.h) {
            let sy = ((f64::from(y) + 0.5 - top) / scale) as u32;
            for x in xs.clone() {
                let sx = ((f64::from(x) + 0.5 - left) / scale) as u32;
                let index =
                    ((sy.min(img.height - 1) * img.width + sx.min(img.width - 1)) * 4) as usize;
                let px = &img.rgba[index..index + 4];
                if px[3] != 0 {
                    canvas.blend(
                        i64::from(x),
                        i64::from(y),
                        [px[0], px[1], px[2], px[3]].map(|c| f64::from(c) / 255.),
                        alpha,
                    );
                }
            }
        }
    }

    pub fn emoji_effect(
        &self,
        run: &ShapedEmoji,
        ppem: u32,
        effect: &crate::vector::EmojiEffect,
    ) -> Option<Arc<crate::host::RgbaImage>> {
        let key = (run.id, ppem, effect.key());
        let state = self.emoji_fonts.as_ref()?.state.get()?;
        if let Some(hit) = state.lock().unwrap().effects.get(&key) {
            return Some(hit);
        }
        let raster = self.emoji_raster(run, ppem)?;
        let image = Arc::new(emoji_effect_raster(&raster.image, effect));
        state
            .lock()
            .unwrap()
            .effects
            .insert(key, image.clone(), image.rgba.len() as u64);
        Some(image)
    }
}

fn shape(inner: &mut ColourState, id: fontdb::ID, text: &str) -> Option<ShapedEmoji> {
    let db = inner.font_system.db();
    let (glyphs, advance_em, bitmap_ppem) = db.with_face_data(id, |data, index| {
        let face = rustybuzz::Face::from_slice(data, index)?;
        let em = face.units_per_em() as f32;
        let mut buffer = rustybuzz::UnicodeBuffer::new();
        buffer.push_str(text);
        // Unconsumed tag characters must fail coverage so the next face can
        // supply the whole flag before Unicode fallback uses the bare flag.
        let flags = if text.ends_with('\u{e007f}') {
            rustybuzz::BufferFlags::PRESERVE_DEFAULT_IGNORABLES
        } else {
            rustybuzz::BufferFlags::REMOVE_DEFAULT_IGNORABLES
        };
        buffer.set_flags(flags);
        buffer.guess_segment_properties();
        let shaped = rustybuzz::shape(&face, &[], buffer);
        let mut glyphs = Vec::new();
        let mut pen = (0, 0);
        for (g, p) in shaped.glyph_infos().iter().zip(shaped.glyph_positions()) {
            if g.glyph_id == 0 {
                return None;
            }
            glyphs.push((
                g.glyph_id as u16,
                (pen.0 + p.x_offset) as f32 / em,
                -(pen.1 + p.y_offset) as f32 / em,
            ));
            pen.0 += p.x_advance;
            pen.1 += p.y_advance;
        }
        let bitmap_ppem = if face.tables().colr.is_none() {
            glyphs
                .iter()
                .filter_map(|g| {
                    face.glyph_raster_image(rustybuzz::ttf_parser::GlyphId(g.0), u16::MAX)
                })
                .map(|i| u32::from(i.pixels_per_em))
                .max()
        } else {
            None
        };
        Some((glyphs, f64::from(pen.0) / f64::from(em), bitmap_ppem))
    })??;
    let mut run = ShapedEmoji {
        id: NEXT_ENGINE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        face: id,
        glyphs,
        advance_em,
        bitmap_ppem,
        sampled_size_em: [1.; 2],
        ink_em: [0.; 4],
    };
    // Verify that every positioned glyph has colour data, including when a
    // face has a colour table whose paint format the scaler cannot execute.
    let ppem = run.ppem(128.);
    let raster = rasterize(inner, &run, ppem)?;
    run.ink_em = raster.ink_em;
    run.sampled_size_em =
        [raster.image.width, raster.image.height].map(|n| f64::from(n) / f64::from(ppem));
    inner.rasters.insert(
        (run.id, ppem),
        raster.clone(),
        raster.image.rgba.len() as u64,
    );
    Some(run)
}

fn rasterize(inner: &mut ColourState, run: &ShapedEmoji, ppem: u32) -> Option<EmojiRaster> {
    let size = ppem as f32;
    let weight = inner.font_system.db().face(run.face)?.weight;
    let mut layers = Vec::new();
    let mut bounds = [i32::MAX, i32::MAX, i32::MIN, i32::MIN];
    for &(gid, x, y) in &run.glyphs {
        let (key, x, y) = cosmic_text::CacheKey::new(
            run.face,
            gid,
            size,
            (x * size, y * size),
            weight,
            cosmic_text::CacheKeyFlags::DISABLE_HINTING,
        );
        let img = inner
            .swash
            .get_image_uncached(&mut inner.font_system, key)?;
        if img.content != SwashContent::Color {
            return None;
        }
        let x = x + img.placement.left;
        let y = y - img.placement.top;
        bounds = [
            bounds[0].min(x),
            bounds[1].min(y),
            bounds[2].max(x + img.placement.width as i32),
            bounds[3].max(y + img.placement.height as i32),
        ];
        layers.push((x, y, img));
    }
    if layers.is_empty() {
        return None;
    }
    // Transparent border for bilinear filtering, independent of the reported ink.
    let [left, top, right, bottom] = bounds;
    let (left, top, right, bottom) = (left - 1, top - 1, right + 1, bottom + 1);
    let (width, height) = ((right - left) as u32, (bottom - top) as u32);
    if f64::from(width) * f64::from(height) > MAX_RASTER_TEXELS {
        return None;
    }
    let mut canvas = Canvas::new(width, height);
    let mut ink = [right, bottom, left, top];
    for (x, y, img) in layers {
        for (i, px) in img.data.chunks_exact(4).enumerate() {
            if px[3] == 0 {
                continue;
            }
            let gx = x + (i as u32 % img.placement.width) as i32;
            let gy = y + (i as u32 / img.placement.width) as i32;
            ink = [
                ink[0].min(gx),
                ink[1].min(gy),
                ink[2].max(gx + 1),
                ink[3].max(gy + 1),
            ];
            canvas.blend(
                i64::from(gx - left),
                i64::from(gy - top),
                [px[0], px[1], px[2], px[3]].map(|c| f64::from(c) / 255.),
                1.,
            );
        }
    }
    if ink[2] <= ink[0] {
        return None;
    }
    let em = |rect: [i32; 4]| rect.map(|v| f64::from(v) / f64::from(ppem));
    Some(EmojiRaster {
        sampled_em: em([left, top, right, bottom]),
        ink_em: em(ink),
        image: Arc::new(crate::host::RgbaImage {
            width,
            height,
            rgba: canvas.data,
        }),
    })
}
