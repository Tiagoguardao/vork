//! page_render.rs — port of the page-quad fill resolution the export /
//! screenshot paths bake under every page (`pages_getPageRenderData`,
//! `core/scene/scene.pages.ts:249-270`), plus its helpers
//! `sampleRepresentativeShaderColor` (scene.pages.ts:210-237) and
//! `averageHexColors` (scene.pages.ts:185-208).
//!
//! WHY THIS EXISTS (WF5 v3 grind, shaders/page-background): the browser's
//! export coordinator gets `pageData` from `scene.getPageRenderData()` and
//! bakes `createPageBackgroundData(pageData)` for both the export and the
//! agent-screenshot configs (`renderers/webgpu/export/WebGPUExportCoordinator.ts:267-331,605-650`).
//! For a page whose top background layer is a shader paint, the quad UNDER
//! the transient shader rectangle is NOT a stored colour — it is the
//! "representative" color sampled from the shader params (the average of the
//! first known palette key). A page-background shader whose own `background`
//! param is the default transparent `#00000000` (shape-grid / shape-grid-dots)
//! therefore composites its dots over that representative color, not over
//! the page colour. Rendering the page colour instead (the old native behavior) made
//! same-color dots invisible (canvas d463d8b5 page 4: dots #0B0C10 over page
//! fill #0b0c10 vs the browser's avg(#0B0C10,#00F5FF) ≈ teal #058188 quad).
//! Opaque page-bg shaders mask the divergence, which is why only the
//! shape-grid family exposed it.

use crate::value::Value;

use crate::types::RawPage;

/// `PAGE_DEFAULTS.backgroundColor` (`core/config/canvasDefaults.ts:147`).
pub const PAGE_DEFAULT_BACKGROUND: &str = "#ffffff";

/// The `fill` field of `PageRenderData` (`pageRenderFill`,
/// `core/scene/scene.pages.ts`): the page quad's solid colour. The top
/// background layer decides: a solid's colour string, a shader's
/// representative colour (else the default), and the default under a
/// gradient or image (the page's background rectangle draws over it). A page
/// without a background has no quad: the empty string (the page system's
/// `if (page.fill)` skips it).
/// A colour string that may carry alpha (`#rgba`, `#rrggbbaa`, `rgba()`, `hsla()`, `transparent`).
fn may_be_translucent(color: &str) -> bool {
    let c = color.trim().to_ascii_lowercase();
    let hex = c.strip_prefix('#').map(str::len);
    matches!(hex, Some(4 | 8)) || c.starts_with("rgba") || c.starts_with("hsla") || c == "transparent"
}

pub fn page_render_fill(page: &RawPage) -> String {
    let Some(fill) = crate::paint::top_page_fill(page) else {
        return String::new();
    };
    // A stack of background layers (E3.1 P6) draws on the page's background
    // rectangle, over the default: unless its top layer is an opaque solid
    // (which hides the rest), the quad must not show the top layer under the
    // layers below it.
    let visible = crate::paint::page_fills(page)
        .iter()
        .filter(|f| crate::paint::shown(f.visible, f.opacity))
        .count();
    let top_covers = fill.opacity.unwrap_or(1.0) >= 1.0
        && fill.blend_mode.as_deref().is_none_or(|m| m == "normal")
        && crate::paint::solid_color(&fill.paint).is_some_and(|c| !may_be_translucent(c));
    if visible > 1 && !top_covers {
        return PAGE_DEFAULT_BACKGROUND.to_string();
    }
    match &fill.paint {
        crate::paint::Paint::Solid(s) => s
            .color
            .as_str()
            .map_or_else(|| PAGE_DEFAULT_BACKGROUND.to_string(), str::to_string),

        _ => PAGE_DEFAULT_BACKGROUND.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    fn page(v: Value) -> RawPage {
        RawPage(v.as_object().unwrap().clone())
    }

    /// A page whose background is the one `paint`.
    fn page_with(paint: Value) -> RawPage {
        page(json!({"id": "p", "fills": [{"id": "fill-0", "paint": paint}]}))
    }

    #[test]
    fn a_background_stack_draws_over_the_default() {
        let p = page(json!({"id": "p", "fills": [
            {"id": "a", "paint": {"type": "solid", "color": "#0b0c10"}},
            {"id": "b", "paint": {"type": "solid", "color": "rgba(255,0,0,0.5)"}}]}));
        assert_eq!(page_render_fill(&p), PAGE_DEFAULT_BACKGROUND);
    }

    #[test]
    fn solid_background_uses_its_color() {
        let p = page_with(json!({"type": "solid", "color": "#0b0c10"}));
        assert_eq!(page_render_fill(&p), "#0b0c10");
    }

    #[test]
    fn solid_non_string_color_is_white() {
        // Scene.pages.ts `typeof fill === 'string'` guard (an unresolved
        // variable reference).
        let p = page_with(json!({"type": "solid", "color": {"type": "variable", "variableId": "v1"}}));
        assert_eq!(page_render_fill(&p), "#ffffff");
    }

    #[test]
    fn the_top_visible_layer_decides() {
        let p = page(json!({"id": "p", "fills": [
            {"id": "fill-0", "paint": {"type": "solid", "color": "#111111"}},
            {"id": "fill-1", "paint": {"type": "solid", "color": "#222222"}},
            {"id": "fill-2", "visible": false, "paint": {"type": "solid", "color": "#333333"}}
        ]}));
        assert_eq!(page_render_fill(&p), "#222222");
    }

    #[test]
    fn gradient_and_image_backgrounds_are_white() {
        let stops = json!([{"offset": 0.0, "color": "#0b0c10"}, {"offset": 1.0, "color": "#ffffff"}]);
        for paint in [
            json!({"type": "linear", "start": {"x": 0.0, "y": 50.0}, "end": {"x": 100.0, "y": 50.0}, "stops": stops}),
            json!({"type": "radial", "center": {"x": 50.0, "y": 50.0}, "radiusX": 50.0, "radiusY": 50.0, "rotation": 0.0, "stops": stops}),
            json!({"type": "conic", "center": {"x": 50.0, "y": 50.0}, "angle": 0.0, "stops": stops}),
            json!({"type": "image", "src": "a.png", "placement": {}}),
        ] {
            assert_eq!(page_render_fill(&page_with(paint.clone())), "#ffffff", "paint {paint}");
        }
    }

    #[test]
    fn no_background_has_no_quad() {
        // WebGPUPageSystem.ts `if (page.fill)`: the ENGINE skips the quad for
        // '' (JS falsy). A page with no visible background layer (`fills: []`,
        // every layer hidden) has none [Q5]; the white default of a colour
        // page lives in the document codemod (024-paint-stacks).
        assert_eq!(page_render_fill(&page(json!({"id": "p", "fills": []}))), "");
        let hidden = page(json!({"id": "p", "fills": [
            {"id": "fill-0", "visible": false, "paint": {"type": "solid", "color": "#123456"}}
        ]}));
        assert_eq!(page_render_fill(&hidden), "");
    }

    #[test]
    fn empty_string_color_passes_through() {
        // The rule returns an empty solid colour verbatim (the engine skips it).
        let p = page_with(json!({"type": "solid", "color": ""}));
        assert_eq!(page_render_fill(&p), "");
    }
}
