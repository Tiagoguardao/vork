//! Render-time defaults table (scene-schema.md §15) as typed accessors over
//! `RawNode`. Every value cites its TS source line; the unit tests below pin
//! each one (PLAN M0c: "defaults table with per-value unit tests citing TS
//! source lines").
//!
//! NOTE on creation vs render defaults: values like `strokeWidth` have
//! *creation* defaults (shapes 0, lines/paths 2 — `core/config/shapeDefaults.ts:13,117-118,135-136`)
//! but at render time an ABSENT field simply means "no stroke"
//! (scene-schema.md §3.6). Accessors here implement the RENDER defaults.

use crate::types::PlacementField;
use crate::types::RawNode;
use crate::value::Value;

impl RawNode {
    /// `opacity ?? 1` (nodes.ts:113-159 BaseNodeProperties; displayList.ts:202).
    pub fn opacity(&self) -> f64 {
        self.placement_number(PlacementField::Opacity).unwrap_or(1.0)
    }

    /// `hidden ?? false` (nodes.ts BaseNodeProperties).
    pub fn hidden(&self) -> bool {
        self.bool("hidden").unwrap_or(false)
    }

    /// `locked ?? false` (nodes.ts BaseNodeProperties).
    pub fn locked(&self) -> bool {
        self.bool("locked").unwrap_or(false)
    }

    /// `rotation || 0` (nodeTransformUtils.ts:257 reads `node.rotation || 0`).
    pub fn rotation(&self) -> f64 {
        self.placement_number(PlacementField::Rotation).unwrap_or(0.0)
    }

    /// `skewX` in degrees (`nodeSkewX`, transform2d.ts): 0 when absent or not finite.
    pub fn skew_x(&self) -> f64 {
        match self.placement_number(PlacementField::SkewX) {
            Some(v) if v.is_finite() => v.clamp(-89.0, 89.0),
            _ => 0.0,
        }
    }

    /// Uniform `scale || 1` (nodeTransformUtils.ts:264-268).
    pub fn scale(&self) -> f64 {
        let s = self.placement_number(PlacementField::Scale).unwrap_or(1.0);
        if s == 0.0 {
            1.0 // `node.scale || 1` — JS falsy 0 coerces to 1
        } else {
            s
        }
    }

    /// Group `scaleX || 1` / `scaleY || 1` (nodeTransformUtils.ts:261-263).
    pub fn scale_xy(&self) -> (f64, f64) {
        if self.placement_field(PlacementField::ScaleX).is_some() || self.placement_field(PlacementField::ScaleY).is_some() {
            let sx = self.placement_number(PlacementField::ScaleX).unwrap_or(1.0);
            let sy = self.placement_number(PlacementField::ScaleY).unwrap_or(1.0);
            (
                if sx == 0.0 { 1.0 } else { sx },
                if sy == 0.0 { 1.0 } else { sy },
            )
        } else {
            let s = self.scale();
            (s, s)
        }
    }

    /// `getRoundedRectCornerRadii` clone (nodes.ts:1819-1827): scalar
    /// broadcast; array entries beyond its length read as 0.
    pub fn corner_radii(&self) -> [f64; 4] {
        match self.json_field("cornerRadius") {
            Some(Value::Number(n)) => {
                let r = n.as_f64().unwrap_or(0.0);
                [r, r, r, r]
            }
            Some(Value::Array(a)) => {
                let g = |i: usize| a.get(i).and_then(Value::as_f64).unwrap_or(0.0);
                [g(0), g(1), g(2), g(3)]
            }
            _ => [0.0; 4],
        }
    }

    /// The top stroke's `align ?? 'center'` (scene-schema.md §15).
    pub fn stroke_align(&self) -> String {
        crate::paint::top_stroke(self)
            .and_then(|s| s.align)
            .unwrap_or_else(|| "center".to_string())
    }

    /// `fillRule ?? 'nonzero'` (nodes.ts:804-840 PathNode).
    pub fn fill_rule(&self) -> &str {
        self.str("fillRule").unwrap_or("nonzero")
    }

    /// line `routing ?? 'straight'` (nodes.ts:726-753).
    pub fn line_routing(&self) -> &str {
        self.str("routing").unwrap_or("straight")
    }

    /// `pointerLength` / `pointerWidth` doc default 10 (nodes.ts:726-753;
    /// editor creates with 20 — shapeDefaults.ts:125-126 — but the RENDER
    /// default for an absent field is 10).
    pub fn pointer_length(&self) -> f64 {
        self.f64("pointerLength").unwrap_or(10.0)
    }
    pub fn pointer_width(&self) -> f64 {
        self.f64("pointerWidth").unwrap_or(10.0)
    }

    /// Shape text frame: `textPadding ?? default`, where the DEFAULT is capped at
    /// `floor(min(width, height) * 0.15)` so a badge-sized shape keeps a usable text area
    /// (`getTextPadding`, textInShapeHelpers.ts:284-315; shapeDefaults.ts:186-199). A flat 8
    /// eats the whole inscribed rect of a 36px circle. The cap reaches 8 at ~54px, so larger
    /// shapes are unchanged, and an explicit `textPadding` is always used as given.
    ///
    /// Paths are exempt: they carry no width/height, and their text box comes from the
    /// centroid's distance to the nearest edge rather than from a bounding box.
    pub fn text_frame_padding(&self) -> f64 {
        const BASE: f64 = 8.0;
        if let Some(explicit) = self.f64("textPadding") {
            return explicit;
        }
        if self.node_type() == "path" {
            return BASE;
        }
        // Validate each dimension BEFORE taking the minimum: `f64::min` ignores NaN
        // (`NAN.min(10.0) == 10.0`) where JS `Math.min` propagates it, so folding first and
        // checking after would let a NaN width cap the padding here while `getTextPadding`
        // falls back to 8 — exactly the silent drift this mirror exists to prevent.
        let w = self.f64("width").unwrap_or(0.0);
        let h = self.f64("height").unwrap_or(0.0);
        if !w.is_finite() || !h.is_finite() || w <= 0.0 || h <= 0.0 {
            return BASE;
        }
        BASE.min((w.min(h) * 0.15).floor())
    }

    /// Frame `textHorizontalAlign ?? 'center'` (nodes.ts:416-458).
    pub fn text_frame_h_align(&self) -> &str {
        self.str("textHorizontalAlign").unwrap_or("center")
    }

    /// Frame `textVerticalAlign ?? 'middle'` (nodes.ts:416-458).
    pub fn text_frame_v_align(&self) -> &str {
        self.str("textVerticalAlign").unwrap_or("middle")
    }

    /// Frame `textResizeMode ?? 'no-resize'` (nodes.ts:416-458).
    pub fn text_frame_resize_mode(&self) -> &str {
        self.str("textResizeMode").unwrap_or("no-resize")
    }

    /// richtext `verticalAlign ?? 'top'` (nodes.ts:1063-1128).
    pub fn richtext_vertical_align(&self) -> &str {
        self.str("verticalAlign").unwrap_or("top")
    }

    /// `blendMode ?? 'normal'` (nodes.ts:7; scene-schema.md §3.7).
    pub fn blend_mode(&self) -> &str {
        self.str("blendMode").unwrap_or("normal")
    }

    /// Connector label `textPosition ?? 0.5` (nodes.ts:470-543).
    pub fn connector_text_position(&self) -> f64 {
        self.f64("textPosition").unwrap_or(0.5)
    }

    /// Connector label `textPadding ?? 4` (nodes.ts:470-543).
    pub fn connector_text_padding(&self) -> f64 {
        self.f64("textPadding").unwrap_or(4.0)
    }

    // Drop / inner shadow param defaults (a missing `opacity` is 1, a missing
    // `spread` 0) live on the effect types: `effect::DropShadow::opacity`,
    // `effect::InnerShadow::opacity` / `spread`.

    /// The top stroke's `miterLimit` — absent ⇒ 4, clamped to 1–500.
    pub fn miter_limit(&self) -> f64 {
        crate::paint::top_stroke(self)
            .and_then(|s| s.miter_limit)
            .filter(|v| v.is_finite())
            .map_or(4.0, |v| v.clamp(1.0, 500.0))
    }

}

/// Page defaults.
pub mod page {
    /// `PAGE_DEFAULTS.backgroundColor` (`core/config/canvasDefaults.ts:147`).
    pub const BACKGROUND_COLOR: &str = "#ffffff";
    /// `PAGE_DEFAULTS.pageSpacing` (`core/config/canvasDefaults.ts:152`).
    pub const PAGE_SPACING: f64 = 100.0;
    /// `DEFAULT_PAGES_LAYOUT_SPACING` (`core/config/layoutDefaults.ts:14`),
    /// forced when layoutMode === 'pages' (`core/scene/scene.layout.ts:33-35`).
    pub const PAGES_LAYOUT_SPACING: f64 = 2500.0;
}

/// Rich text default inline style (`utils/richtext/types.ts:30-39`).
pub mod richtext {
    pub const FONT_FAMILY: &str = "sans-serif";
}

#[cfg(test)]
mod tests {
    use crate::types::RawNode;
    use crate::value::json;

    fn node(v: crate::value::Value) -> RawNode {
        RawNode::from_value(v).unwrap()
    }

    #[test]
    fn opacity_defaults_to_1() {
        // nodes.ts BaseNodeProperties `opacity?: number` 0–1, default 1
        // (scene-schema.md §15; displayList.ts:202 `node.opacity ?? 1`).
        assert_eq!(node(json!({"id":"a","type":"rectangle"})).opacity(), 1.0);
        assert_eq!(
            node(json!({"id":"a","type":"rectangle","opacity":0.25})).opacity(),
            0.25
        );
    }

    #[test]
    fn hidden_and_locked_default_false() {
        // nodes.ts:113-159 (scene-schema.md §3.1).
        let n = node(json!({"id":"a","type":"rectangle"}));
        assert!(!n.hidden());
        assert!(!n.locked());
    }

    #[test]
    fn rotation_defaults_to_0_and_scale_to_1() {
        // nodeTransformUtils.ts:257 `node.rotation || 0`;
        // nodeTransformUtils.ts:264-268 `node.scale || 1`.
        let n = node(json!({"id":"a","type":"rectangle"}));
        assert_eq!(n.rotation(), 0.0);
        assert_eq!(n.scale(), 1.0);
        // `scale: 0` is falsy in JS → || 1 (copy the bug).
        assert_eq!(
            node(json!({"id":"a","type":"rectangle","scale":0})).scale(),
            1.0
        );
    }

    #[test]
    fn group_scale_xy_defaults() {
        // nodeTransformUtils.ts:261-263 `node.scaleX || 1` / `node.scaleY || 1`.
        let n = node(json!({"id":"g","type":"group","scaleX":2.0}));
        assert_eq!(n.scale_xy(), (2.0, 1.0));
    }

    #[test]
    fn corner_radius_scalar_array_and_short_array() {
        // nodes.ts:1819-1827 getRoundedRectCornerRadii: short arrays read
        // missing entries as 0 (scene-schema.md §4.1).
        assert_eq!(
            node(json!({"id":"a","type":"rectangle"})).corner_radii(),
            [0.0; 4]
        );
        assert_eq!(
            node(json!({"id":"a","type":"rectangle","cornerRadius":5})).corner_radii(),
            [5.0, 5.0, 5.0, 5.0]
        );
        assert_eq!(
            node(json!({"id":"a","type":"rectangle","cornerRadius":[1,2]})).corner_radii(),
            [1.0, 2.0, 0.0, 0.0]
        );
    }

    #[test]
    fn stroke_align_defaults_to_center() {
        // nodes.ts:378-383 (scene-schema.md §3.6/§15).
        assert_eq!(
            node(json!({"id":"a","type":"rectangle"})).stroke_align(),
            "center"
        );
    }

    #[test]
    fn path_defaults() {
        // nodes.ts:804-840: fillRule 'nonzero', closed false (scene-schema.md §4.5).
        let n = node(json!({"id":"p","type":"path","points":[]}));
        assert_eq!(n.fill_rule(), "nonzero");
        assert!(!n.path_closure().at(0));
    }

    #[test]
    fn line_defaults() {
        // nodes.ts:726-753: routing 'straight'; pointerLength/Width default 10
        // (doc default; creation default is 20 per shapeDefaults.ts:125-126).
        let n = node(json!({"id":"l","type":"line","points":[0,0,1,1]}));
        assert_eq!(n.line_routing(), "straight");
        assert_eq!(n.pointer_length(), 10.0);
        assert_eq!(n.pointer_width(), 10.0);
    }

    #[test]
    fn text_frame_defaults() {
        // nodes.ts:416-458: padding 8, h center, v middle, no-resize
        // (scene-schema.md §3.10/§15).
        let n = node(json!({"id":"a","type":"rectangle","textContent":"<p>x</p>"}));
        assert_eq!(n.text_frame_padding(), 8.0);
        assert_eq!(n.text_frame_h_align(), "center");
        assert_eq!(n.text_frame_v_align(), "middle");
        assert_eq!(n.text_frame_resize_mode(), "no-resize");
    }

    /// The DEFAULT padding is capped at `floor(min(w,h) * 0.15)` so badge-sized shapes keep a
    /// usable text area (textInShapeHelpers.ts:284-315). Must match `getTextPadding` exactly:
    /// a mismatch places shape text differently in the primary and fallback renderers.
    #[test]
    fn text_frame_padding_small_shape_cap() {
        let big = node(json!({"id":"a","type":"ellipse","width":100.0,"height":100.0}));
        assert_eq!(big.text_frame_padding(), 8.0);

        // 54px is the first size where floor(0.15 * side) reaches the flat 8.
        let at_cap = node(json!({"id":"b","type":"ellipse","width":54.0,"height":54.0}));
        assert_eq!(at_cap.text_frame_padding(), 8.0);
        let below_cap = node(json!({"id":"c","type":"ellipse","width":53.0,"height":53.0}));
        assert_eq!(below_cap.text_frame_padding(), 7.0);

        // 36px badge: 5, not 8 — the reported floor-to-8px-numeral case.
        let badge = node(json!({"id":"d","type":"ellipse","width":36.0,"height":36.0}));
        assert_eq!(badge.text_frame_padding(), 5.0);

        // The SHORTER side drives the cap.
        let wide = node(json!({"id":"e","type":"rectangle","width":400.0,"height":20.0}));
        assert_eq!(wide.text_frame_padding(), 3.0);

        // An explicit textPadding is never capped, and 0 stays 0.
        let explicit =
            node(json!({"id":"f","type":"ellipse","width":36.0,"height":36.0,"textPadding":8.0}));
        assert_eq!(explicit.text_frame_padding(), 8.0);
        let zero =
            node(json!({"id":"g","type":"ellipse","width":36.0,"height":36.0,"textPadding":0.0}));
        assert_eq!(zero.text_frame_padding(), 0.0);

        // Paths (no width/height) and degenerate sizes fall back to the flat default.
        let path = node(json!({"id":"h","type":"path","points":[{"x":0.0,"y":0.0}]}));
        assert_eq!(path.text_frame_padding(), 8.0);
        let zero_size = node(json!({"id":"i","type":"rectangle","width":0.0,"height":0.0}));
        assert_eq!(zero_size.text_frame_padding(), 8.0);

        // ONE bad dimension is enough to fall back, matching JS `Math.min` propagation. Folding
        // with `f64::min` first would hide it, since `NAN.min(x) == x`.
        let one_bad = node(json!({"id":"j","type":"rectangle","width":40.0,"height":0.0}));
        assert_eq!(one_bad.text_frame_padding(), 8.0);
        let missing_height = node(json!({"id":"k","type":"rectangle","width":40.0}));
        assert_eq!(missing_height.text_frame_padding(), 8.0);
    }

    #[test]
    fn connector_text_defaults() {
        // nodes.ts:470-543: textPosition 0.5, textPadding 4 (scene-schema.md §4.4).
        let n = node(json!({"id":"l","type":"line","points":[0,0,1,1],"textContent":"hi"}));
        assert_eq!(n.connector_text_position(), 0.5);
        assert_eq!(n.connector_text_padding(), 4.0);
    }

    #[test]
    fn richtext_vertical_align_defaults_top() {
        // nodes.ts:1063-1128 (scene-schema.md §4.9/§15).
        assert_eq!(
            node(json!({"id":"t","type":"richtext","htmlContent":"<p></p>"}))
                .richtext_vertical_align(),
            "top"
        );
    }

    #[test]
    fn blend_mode_defaults_normal() {
        // nodes.ts:7 BlendModeType (scene-schema.md §3.7/§15).
        assert_eq!(
            node(json!({"id":"a","type":"rectangle"})).blend_mode(),
            "normal"
        );
    }

}
