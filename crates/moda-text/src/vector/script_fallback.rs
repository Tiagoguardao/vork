//! Per-script font fallback for the vector engine — port of
//! `renderers/webgpu/text/vector/scriptFallback.ts` (Phase 3;
//! the renamed + generalized `cjkFallback.ts`): coverage split against the
//! primary font, per-script-class re-chunking of uncovered text, and the
//! Noto-class fallback family chains (regional CJK ordering preserved).
//!
//! The coverage probe here is the ORACLE's (`tools/oracle/dump-linebox.mjs`
//! `covers = (font, …rest) => !hasUnsupportedGlyphs(font, text)` — cmap
//! only). Production's probe additionally HarfBuzz-shapes complex-script
//! text to catch shaping-produced `.notdef` (vectorLayoutPipeline.ts:118-126)
//! — unreachable on the vendored catalog (no complex-script family carries a
//! TTF), and deliberately NOT ported: the dump is the parity contract. See
//! the Q-117/Q-118 ledger entries.

use std::rc::Rc;

use crate::catalog::Catalog;
use crate::vector::cjk::{is_cjk_breakable_codepoint, js_whitespace};
use crate::vector::measure::VectorFontStore;
use crate::vector::otjs::OtjsFont;
use crate::vector::unicode_scripts::{classify_codepoint, UnicodeScript};

/// `DEGRADABLE_RESIDUE_RE` (scriptFallback.ts): codepoints it is acceptable to
/// render wrong when NOTHING in the catalog covers them — standalone symbols
/// (`Sm`/`Sk`/`So`), punctuation, separators, format/control/unassigned.
/// Deliberately an ALLOWLIST: a category nobody thought about must default to
/// the safe whole-node path, not to being dropped. Excluded on purpose, each
/// because losing it changes what the text MEANS: letters, numbers (`⑩` `½`
/// `42` are `Nl`/`No`/`Nd` — none of them Alphabetic, which is the trap this
/// replaces), combining marks, and `Sc` currency signs.
fn is_degradable_residue_char(ch: char) -> bool {
    use unicode_properties::general_category::GeneralCategory::*;
    use unicode_properties::UnicodeGeneralCategory;
    matches!(
        ch.general_category(),
        MathSymbol
            | ModifierSymbol
            | OtherSymbol
            | ConnectorPunctuation
            | DashPunctuation
            | OpenPunctuation
            | ClosePunctuation
            | InitialPunctuation
            | FinalPunctuation
            | OtherPunctuation
            | SpaceSeparator
            | LineSeparator
            | ParagraphSeparator
            | Control
            | Format
            | Surrogate
            | PrivateUse
            | Unassigned
    )
}

/// `hasUnsupportedGlyphs` (scriptFallback.ts:34-41): any non-whitespace
/// codepoint mapping to `.notdef` fails; whitespace is exempt (substituted
/// with the space glyph at shaping time, `normalizeTextForShaping`).
pub fn has_unsupported_glyphs(font: &OtjsFont, text: &str) -> bool {
    for ch in text.chars() {
        // Discretionary controls are consumed by wrapping and never need a glyph.
        if js_whitespace(ch) || ch == crate::hyphenate::SOFT_HYPHEN || ch == '\u{200B}' || ch == '\u{200C}' {
            continue;
        }
        if font.char_to_glyph_index(ch) == 0 {
            return true;
        }
    }
    false
}

#[derive(Debug, Clone)]
pub struct CoverageChunk {
    pub text: String,
    pub covered: bool,
}

/// `splitTextByCoverage` (scriptFallback.ts:56-65): maximal runs of uniform
/// coverage; whitespace inherits the preceding chunk's coverage (leading
/// whitespace counts as covered).
pub fn split_text_by_coverage(font: &OtjsFont, text: &str) -> Vec<CoverageChunk> {
    let mut chunks: Vec<CoverageChunk> = Vec::new();
    for ch in text.chars() {
        let covered = if js_whitespace(ch) || ch == crate::hyphenate::SOFT_HYPHEN || ch == '\u{200B}' || ch == '\u{200C}' {
            chunks.last().map(|c| c.covered).unwrap_or(true)
        } else {
            font.char_to_glyph_index(ch) > 0
        };
        match chunks.last_mut() {
            Some(last) if last.covered == covered => last.text.push(ch),
            _ => chunks.push(CoverageChunk {
                text: ch.to_string(),
                covered,
            }),
        }
    }
    chunks
}

/// What picks a CJK fallback face beyond the text itself (`FallbackContext` in scriptFallback.ts): the region of the
/// run's language (`crate::lang`) and whether its primary face is a serif. The default is the order every document
/// had before languages: no region, sans.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FallbackContext {
    pub region: Option<crate::lang::CjkRegion>,
    pub serif: bool,
}

impl FallbackContext {
    /// A run's context: its language (canonical) and its primary family's catalog category.
    pub fn for_run(lang: Option<&str>, primary_family: &str, catalog: &Catalog) -> FallbackContext {
        FallbackContext {
            region: lang.and_then(crate::lang::CjkRegion::of),
            serif: catalog.is_serif_family(primary_family),
        }
    }
}

/// The serif twin of a Noto Sans CJK region face (`Noto Sans JP` → `Noto Serif JP`).
fn serif_twin(family: &'static str) -> Option<&'static str> {
    Some(match family {
        "Noto Sans JP" => "Noto Serif JP",
        "Noto Sans KR" => "Noto Serif KR",
        "Noto Sans SC" => "Noto Serif SC",
        "Noto Sans TC" => "Noto Serif TC",
        _ => return None,
    })
}

/// Push CJK region faces in order; on a serif primary face each one's serif twin goes first.
fn push_cjk(families: &[&'static str], ctx: FallbackContext, ordered: &mut Vec<&'static str>) {
    for f in families {
        if ctx.serif {
            if let Some(serif) = serif_twin(f) {
                push_unique(&[serif], ordered);
            }
        }
        push_unique(&[f], ordered);
    }
}

/// `isCjkScript` (scriptFallback.ts:71-72).
fn is_cjk_script(script: UnicodeScript) -> bool {
    matches!(
        script,
        UnicodeScript::Cjk
            | UnicodeScript::Hiragana
            | UnicodeScript::Katakana
            | UnicodeScript::Korean
    )
}

/// `SCRIPT_FALLBACK_FAMILIES` (scriptFallback.ts:80-109) — per non-CJK
/// script; scripts absent (→ `&[]`) leave the chunk uncovered.
pub(crate) fn script_fallback_families(script: UnicodeScript) -> &'static [&'static str] {
    use UnicodeScript::*;
    match script {
        Arabic => &["Noto Sans Arabic"],
        Hebrew => &["Noto Sans Hebrew"],
        Devanagari => &["Noto Sans Devanagari"],
        Bengali => &["Noto Sans Bengali"],
        Gurmukhi => &["Noto Sans Gurmukhi"],
        Gujarati => &["Noto Sans Gujarati"],
        Oriya => &["Noto Sans Oriya"],
        Tamil => &["Noto Sans Tamil"],
        Telugu => &["Noto Sans Telugu"],
        Kannada => &["Noto Sans Kannada"],
        Malayalam => &["Noto Sans Malayalam"],
        Sinhala => &["Noto Sans Sinhala"],
        Thai => &["Noto Sans Thai"],
        Lao => &["Noto Sans Lao"],
        Khmer => &["Noto Sans Khmer"],
        Myanmar => &["Noto Sans Myanmar"],
        Tibetan => &["Noto Serif Tibetan"],
        Ethiopic => &["Noto Sans Ethiopic"],
        Georgian => &["Noto Sans Georgian"],
        Armenian => &["Noto Sans Armenian"],
        Cyrillic => &["Noto Sans"],
        Greek => &["Noto Sans"],
        // Symbols split into two classes along the two families' cmap boundary
        // (see the comment on the ranges in `unicode_scripts.rs`). Chain order
        // keeps the family that covers the class first. Now the
        // partition no longer decides COVERAGE (the per-codepoint retry spans
        // the whole chain) — it is what keeps the common case one run.
        // STIX Two Math comes last in both: the last resort for symbols
        // neither Noto face has (`⌁` U+2301, `☓` U+2613, …). See
        // `LAST_RESORT_FAMILIES`.
        MathSymbols => &["Noto Sans Math", "Noto Sans Symbols 2", "STIX Two Math"],
        Symbols => &["Noto Sans Symbols 2", "Noto Sans Math", "STIX Two Math"],
        _ => &[],
    }
}

/// `LAST_RESORT_FAMILIES` (scriptFallback.ts): chain families that only draw
/// what the faces before them lack. The whole-piece probe in
/// [`resolve_coverage_chunks`] skips them and the per-codepoint retry reaches
/// them only after every earlier face, so adding one changes only glyphs
/// nothing drew before: `→ ➜` keeps its Noto glyphs even though STIX Two Math
/// alone covers the whole piece.
pub const LAST_RESORT_FAMILIES: &[&str] = &["STIX Two Math"];

/// Whether `family` is a [`LAST_RESORT_FAMILIES`] member (`isLastResortFamily`).
pub fn is_last_resort_family(family: &str) -> bool {
    LAST_RESORT_FAMILIES.contains(&family)
}

fn push_unique(families: &[&'static str], ordered: &mut Vec<&'static str>) {
    for f in families {
        if !ordered.contains(f) {
            ordered.push(f);
        }
    }
}

/// `getFallbackFamiliesForText` (scriptFallback.ts:118-153): non-CJK scripts
/// contribute their families in first-appearance order; CJK keeps the Phase
/// 1a regional ordering (kana → JP first, Hangul → KR first, Han → regional
/// list) appended after; CJK punctuation/fullwidth forms ('unknown' script
/// but CJK-breakable content) get the Han list when nothing else matched.
pub fn get_fallback_families_for_text(text: &str) -> Vec<&'static str> {
    get_fallback_families_for_text_in(text, FallbackContext::default())
}

/// [`get_fallback_families_for_text`] for a run in `ctx`: Han goes to the language's region face first (Japanese,
/// Korean, Simplified or Traditional Chinese; `crate::lang::han_fallback_families`), and on a serif primary face each
/// CJK face's Noto Serif twin comes before it (a twin that isn't installed is skipped).
pub fn get_fallback_families_for_text_in(text: &str, ctx: FallbackContext) -> Vec<&'static str> {
    let mut scripts: Vec<UnicodeScript> = Vec::new();
    let mut has_cjk_content = false;
    let mut ordered: Vec<&'static str> = Vec::new();

    for ch in text.chars() {
        if js_whitespace(ch) {
            continue;
        }
        let cp = ch as u32;
        // Enclosed alphanumerics are meaning-bearing, so they cannot degrade
        // as symbol residue. The JP face covers them, and the Pan-CJK SC face
        // the app ships stands in when JP isn't installed.
        if (0x2460..=0x24ff).contains(&cp) {
            push_unique(&["Noto Sans JP", "Noto Sans SC"], &mut ordered);
        }
        if is_cjk_breakable_codepoint(cp) {
            has_cjk_content = true;
        }
        let script = classify_codepoint(cp);
        if script == UnicodeScript::Unknown || scripts.contains(&script) {
            continue;
        }
        scripts.push(script);
        if !is_cjk_script(script) {
            push_unique(script_fallback_families(script), &mut ordered);
        }
    }

    let han = crate::lang::han_fallback_families(ctx.region);
    let mut cjk_ordered: Vec<&'static str> = Vec::new();
    // Kana and Hangul: their region's face, then the Pan-CJK face the app ships (answering to Noto Sans SC), which
    // draws them when the region's face isn't installed.
    if scripts.contains(&UnicodeScript::Hiragana) || scripts.contains(&UnicodeScript::Katakana) {
        push_cjk(&["Noto Sans JP", "Noto Sans SC"], ctx, &mut cjk_ordered);
    }
    if scripts.contains(&UnicodeScript::Korean) {
        push_cjk(&["Noto Sans KR", "Noto Sans SC"], ctx, &mut cjk_ordered);
    }
    if scripts.contains(&UnicodeScript::Cjk) {
        push_cjk(&han, ctx, &mut cjk_ordered);
    }
    if cjk_ordered.is_empty() && ordered.is_empty() && has_cjk_content {
        push_cjk(&han, ctx, &mut cjk_ordered);
    }
    push_unique(&cjk_ordered, &mut ordered);

    // Last-resort families go after every other family the text asked for
    // (e.g. Noto Sans JP for `⑩` beside `→`).
    let (mut out, last): (Vec<_>, Vec<_>) =
        ordered.into_iter().partition(|f| !is_last_resort_family(f));
    out.extend(last);
    out
}

/// `FallbackVariant` (scriptFallback.ts:155-159) — a fallback candidate
/// snapped to its nearest TTF-backed catalog weight/style.
#[derive(Debug, Clone, PartialEq)]
pub struct FallbackVariant {
    pub family: String,
    pub weight: i64,
    /// 'normal' | 'italic'
    pub style: String,
}

/// `listFallbackVariants` (scriptFallback.ts:166-173): candidate families
/// snapped via `resolveNearestTtfVariant` (NOT the primary path's exact
/// match, Q-103/Q-108); families absent from the catalog are skipped.
pub fn list_fallback_variants(
    text: &str,
    weight: i64,
    style: &str,
    catalog: &Catalog,
) -> Vec<FallbackVariant> {
    list_fallback_variants_in(text, weight, style, catalog, FallbackContext::default())
}

/// [`list_fallback_variants`] for a run in `ctx` ([`get_fallback_families_for_text_in`]).
pub fn list_fallback_variants_in(
    text: &str,
    weight: i64,
    style: &str,
    catalog: &Catalog,
    ctx: FallbackContext,
) -> Vec<FallbackVariant> {
    let mut out = Vec::new();
    for family in get_fallback_families_for_text_in(text, ctx) {
        if let Some(v) = catalog.resolve_nearest_ttf_variant(family, weight, style) {
            out.push(FallbackVariant {
                family: family.to_string(),
                weight: v.weight,
                style: v.style,
            });
        }
    }
    out
}

/// `loadFallbackChain` (scriptFallback.ts:208-221) over the synchronous store:
/// the candidate variants paired with their loaded fonts, in priority order.
/// Unlike the browser's asynchronous loader, an unavailable native face is skipped;
/// it must not prevent a loaded shipped face from drawing the text.
fn load_fallback_chain(
    store: &VectorFontStore,
    catalog: &Catalog,
    text: &str,
    weight: i64,
    style: &str,
    ctx: FallbackContext,
) -> Option<Vec<(Rc<OtjsFont>, FallbackVariant)>> {
    let mut chain = Vec::new();
    for candidate in list_fallback_variants_in(text, weight, style, catalog, ctx) {
        let Some(font) = store.font_for_variant(
            catalog,
            &candidate.family,
            candidate.weight,
            &candidate.style,
            None,
        ) else {
            continue;
        };
        chain.push((font, candidate));
    }
    Some(chain)
}

/// `splitTextByScriptClass` (scriptFallback.ts:233-252): maximal chunks of
/// uniform script class. All CJK scripts collapse to one 'cjk' class;
/// whitespace and script-unknown NON-breakable codepoints are neutral (ride
/// along, a leading run attaches to the first classified chunk); CJK
/// punctuation ('unknown' script but CJK-breakable) is its own 'unknown'
/// class — a real, chunk-splitting class, matched by the TS `=== 'unknown'`
/// neutrality test only when the codepoint is not CJK-breakable.
pub fn split_text_by_script_class(text: &str) -> Vec<String> {
    struct Chunk {
        text: String,
        script: Option<UnicodeScript>,
    }
    let mut chunks: Vec<Chunk> = Vec::new();
    for ch in text.chars() {
        let cp = ch as u32;
        let cls = classify_codepoint(cp);
        let neutral =
            js_whitespace(ch) || (cls == UnicodeScript::Unknown && !is_cjk_breakable_codepoint(cp));
        let script: Option<UnicodeScript> = if neutral {
            None
        } else if is_cjk_script(cls) {
            Some(UnicodeScript::Cjk)
        } else {
            Some(cls)
        };
        match chunks.last_mut() {
            Some(last) if script.is_none() || last.script == script || last.script.is_none() => {
                last.text.push(ch);
                if last.script.is_none() {
                    last.script = script;
                }
            }
            _ => chunks.push(Chunk {
                text: ch.to_string(),
                script,
            }),
        }
    }
    chunks.into_iter().map(|c| c.text).collect()
}

/// One coverage-resolved slice of a run's text (`CoverageResolvedChunk`,
/// scriptFallback.ts:255-268): which variant renders it.
#[derive(Debug, Clone)]
pub struct CoverageResolvedChunk {
    pub text: String,
    pub variant: FallbackVariant,
    /// `degraded` (scriptFallback.ts:262-268): no catalog font covers this
    /// slice; it rides the primary variant, so it draws that face's `.notdef`
    /// — a VISIBLE tofu box on this lane (verified by rendering
    /// `text/symbols.json`), where the TS fallback renderer skips glyph index 0
    /// and leaves a gap instead. Mirrored for port fidelity: the browser lane
    /// warns on it (`vectorLayoutPipeline.reportDegradedChunk`), native has no
    /// telemetry seam here yet, and it reaches no oracle dump either way.
    pub degraded: bool,
}

impl CoverageResolvedChunk {
    fn covered(text: String, variant: FallbackVariant) -> Self {
        Self {
            text,
            variant,
            degraded: false,
        }
    }
}

/// `retryPieceByCodepoint` (scriptFallback.ts:290-337): per-codepoint retry for
/// one script-class piece no single chain family covers. Each codepoint takes
/// the first chain variant that covers it, consecutive codepoints on the same
/// variant merge, whitespace inherits the preceding sub-piece (the
/// `split_text_by_coverage` rule) and a LEADING whitespace run rides onto the
/// first assigned sub-piece. Codepoints nothing covers become `residue`
/// sub-pieces on the primary variant.
fn retry_piece_by_codepoint(
    piece: &str,
    primary_variant: &FallbackVariant,
    chain: &[(Rc<OtjsFont>, FallbackVariant)],
) -> (Vec<CoverageResolvedChunk>, String, bool) {
    let mut chunks: Vec<CoverageResolvedChunk> = Vec::new();
    let mut residue = String::new();
    let mut resolved_any = false;
    let mut inherited: Option<&FallbackVariant> = None;
    let mut pending_leading = String::new();

    for ch in piece.chars() {
        let mut hit = inherited;
        if !js_whitespace(ch) {
            let mut buf = [0u8; 4];
            let ch_str = ch.encode_utf8(&mut buf);
            hit = chain
                .iter()
                .find(|(font, _)| !has_unsupported_glyphs(font, ch_str))
                .map(|(_, variant)| variant);
            inherited = hit;
            match hit {
                Some(_) => resolved_any = true,
                None => residue.push(ch),
            }
        } else if chunks.is_empty() {
            pending_leading.push(ch);
            continue;
        }
        let variant = hit.unwrap_or(primary_variant);
        match chunks.last_mut() {
            Some(last) if last.variant == *variant => {
                last.text.push(ch);
                continue;
            }
            _ => {}
        }
        let mut text = std::mem::take(&mut pending_leading);
        text.push(ch);
        chunks.push(CoverageResolvedChunk {
            text,
            variant: variant.clone(),
            degraded: hit.is_none(),
        });
    }
    // Whitespace-only pieces never reach here from `resolve_coverage_chunks`
    // (whitespace is exempt from the coverage probe); keep the text lossless.
    if !pending_leading.is_empty() {
        chunks.push(CoverageResolvedChunk::covered(
            pending_leading,
            primary_variant.clone(),
        ));
    }
    (chunks, residue, resolved_any)
}

/// `resolveCoverageChunks` (scriptFallback.ts:340-411): covered chunks keep the
/// primary variant; uncovered chunks split by script class and each piece
/// resolves against that class's fallback chain. A piece no single family
/// covers is RE-SPLIT PER CODEPOINT so one uncovered codepoint can
/// no longer decline the whole node. The four guards that keep that strictly
/// additive — whole-piece probe first, complex-script pieces stay
/// all-or-nothing, a retry that resolves nothing still declines, and residue
/// containing an Alphabetic codepoint still declines — are documented in full
/// on the TS original; keep both sides in lockstep.
pub fn resolve_coverage_chunks(
    store: &VectorFontStore,
    catalog: &Catalog,
    primary: &Rc<OtjsFont>,
    primary_variant: &FallbackVariant,
    text: &str,
) -> Option<Vec<CoverageResolvedChunk>> {
    resolve_coverage_chunks_in(store, catalog, primary, primary_variant, text, FallbackContext::default())
}

/// [`resolve_coverage_chunks`] for a run in `ctx`: its CJK pieces fall back to its language's region face
/// ([`FallbackContext`]). With a region, a CJK piece that face doesn't cover whole goes straight to the per-codepoint
/// retry, so the face still draws every character it has and a later face only the rest (`resolveCoverageChunks`).
pub fn resolve_coverage_chunks_in(
    store: &VectorFontStore,
    catalog: &Catalog,
    primary: &Rc<OtjsFont>,
    primary_variant: &FallbackVariant,
    text: &str,
    ctx: FallbackContext,
) -> Option<Vec<CoverageResolvedChunk>> {
    // Fast path: the primary font covers the whole run (the common Latin case).
    if !has_unsupported_glyphs(primary, text) {
        return Some(vec![CoverageResolvedChunk::covered(
            text.to_string(),
            primary_variant.clone(),
        )]);
    }
    let mut chunks = Vec::new();
    for chunk in split_text_by_coverage(primary, text) {
        if chunk.covered {
            chunks.push(CoverageResolvedChunk::covered(
                chunk.text,
                primary_variant.clone(),
            ));
            continue;
        }
        for piece in split_text_by_script_class(&chunk.text) {
            let chain = load_fallback_chain(
                store,
                catalog,
                &piece,
                primary_variant.weight,
                &primary_variant.style,
                ctx,
            )?;
            let region_first =
                ctx.region.is_some() && piece.chars().any(|c| is_cjk_breakable_codepoint(c as u32));
            let whole = chain
                .iter()
                .filter(|(_, variant)| !is_last_resort_family(&variant.family))
                .take(if region_first { 1 } else { usize::MAX })
                .find(|(font, _)| !has_unsupported_glyphs(font, &piece));
            if let Some((_, variant)) = whole {
                chunks.push(CoverageResolvedChunk::covered(piece, variant.clone()));
                continue;
            }
            // Guard 2: splitting a shaped cluster per codepoint would break
            // Arabic joining / Indic matra reordering.
            if piece
                .chars()
                .any(|c| crate::complex_script::codepoint_needs_complex_shaping(c as u32))
            {
                return None;
            }
            let (piece_chunks, residue, resolved_any) =
                retry_piece_by_codepoint(&piece, primary_variant, &chain);
            // Guard 3: the retry saved no neighbor, so keep the safe whole-node
            // path. Guard 4: degrading in place shows the primary face's
            // `.notdef` (a tofu box here, an omitted glyph on the TS fallback
            // renderer) — fine for an isolated symbol, never for anything
            // meaning-bearing.
            if !resolved_any || !residue.chars().all(is_degradable_residue_char) {
                return None;
            }
            chunks.extend(piece_chunks);
        }
    }
    Some(chunks)
}

#[cfg(test)]
#[cfg_attr(not(vork_private_tests), allow(dead_code, unused_imports))]
mod tests {
    use super::*;
    use crate::test_fixtures::store_and_catalog;

    /// The repeated success-path preamble: build the shared store/catalog, fetch
    /// the primary family's 400-normal font, and assemble its `FallbackVariant`.
    /// `message` is the exact panic literal the inlined `.expect` used.
    fn fallback_fixture(
        family: &str,
        message: &str,
    ) -> (
        VectorFontStore,
        Catalog,
        std::rc::Rc<OtjsFont>,
        FallbackVariant,
    ) {
        fallback_fixture_over(store_and_catalog(), family, message)
    }

    /// The same over Moda's oracle corpus, for the tests that wait on its fallback families.
    fn oracle_fallback_fixture(
        family: &str,
        message: &str,
    ) -> (
        VectorFontStore,
        Catalog,
        std::rc::Rc<OtjsFont>,
        FallbackVariant,
    ) {
        fallback_fixture_over(crate::test_fixtures::oracle_store_and_catalog(), family, message)
    }

    fn fallback_fixture_over(
        (store, catalog): (VectorFontStore, Catalog),
        family: &str,
        message: &str,
    ) -> (
        VectorFontStore,
        Catalog,
        std::rc::Rc<OtjsFont>,
        FallbackVariant,
    ) {
        let font = store
            .font_for_variant(&catalog, family, 400, "normal", None)
            .unwrap_or_else(|| panic!("{message}"));
        let primary = FallbackVariant {
            family: family.to_string(),
            weight: 400,
            style: "normal".to_string(),
        };
        (store, catalog, font, primary)
    }

    /// Guard 3: the retry only pays out where it saves NEIGHBORS. A piece in
    /// which NOTHING resolves keeps the whole-node path, where Canvas2D can
    /// still serve the codepoint from a system face — which is what stops the
    /// unclassified blocks (Braille, Enclosed Alphanumerics, Box Drawing) from
    /// regressing from "rendered by a system font" to "omitted".
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn wholly_uncovered_piece_still_declines() {
        let (store, catalog, fraunces, primary) =
            fallback_fixture("Fraunces", "Fraunces 400 vendored");
        // U+254B alone: unclassified block, so no chain, nothing to save. It IS
        // degradable, so only guard 3 can be what declines this.
        assert!(
            resolve_coverage_chunks(&store, &catalog, &fraunces, &primary, "Step ╋").is_none(),
            "a lone uncovered codepoint must keep the safe whole-node path"
        );
        // U+271E alone in its symbols piece: the chain exists but none of its
        // faces (STIX Two Math included) has it, so there is still no neighbor
        // to save. (U+2301 was this case until STIX Two Math joined the chain.)
        assert!(
            resolve_coverage_chunks(&store, &catalog, &fraunces, &primary, "Signal ✞").is_none(),
            "a symbols piece where nothing resolves must still decline"
        );
    }

    /// A script with no fallback family (Thaana → `Unknown`: the font is
    /// vendored under `fixtures/oracle/fonts/noto-sans-thaana/`, but the block
    /// is deliberately absent from `SCRIPT_RANGES`, so nothing maps to it)
    /// stays uncovered — the whole run declines (`None`) as before, so the
    /// fallback coverage is exactly the mapped scripts, nothing wider.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn unmapped_script_stays_uncovered() {
        let (store, catalog, inter, primary) = fallback_fixture("Inter", "Inter 400 vendored");
        // Thaana (U+0780–07BF) — not in SCRIPT_RANGES, so no fallback family is
        // ever consulted (the vendored Noto Sans Thaana is unreachable here).
        assert!(
            resolve_coverage_chunks(&store, &catalog, &inter, &primary, "ދިވެހި").is_none(),
            "Thaana has no fallback family — the run must decline (None)"
        );
    }
}
