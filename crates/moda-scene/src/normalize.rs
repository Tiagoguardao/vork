//! Node normalization at load — port of `normalizeNodeInPlace`
//! (`core/scene/scene.normalize.ts`) and its helper modules. These repair
//! current-schema input (agent-written, imported or malformed), not old
//! document shapes: there are no migrations (docs/document-format.md). All
//! passes are idempotent in-place mutations, run in the exact TS order.
//!
//! `dash` has no pass: every writer stores `number[]` (the agent paths parse
//! string presets through `parseDashPattern` before storing).
//!
//! The brush / variable-width stroke fields have no pass here: the TS editor
//! normalizes them on write (scene.mutations.ts, Scene.admitNode on insert),
//! Stroke properties are read by the typed paint records.

use crate::types::{JsonMap, RawNode};
use crate::value::{json, Map, Value};

/// Re-home a parsed JSON value at exactly the size it needs. serde_json builds
/// objects and arrays by pushing, so each one keeps the capacity its last
/// doubling left it (an object of 17 keys holds room for 28); across a
/// document's nodes that slack was about a third of the node table. Moves
/// every key and value (no string copies), depth first.
pub fn compact_value(v: &mut Value) {
    match v {
        Value::Object(m) => compact_map(m),
        Value::Array(a) => {
            for x in a.iter_mut() {
                compact_value(x);
            }
            a.shrink_to_fit();
        }
        _ => {}
    }
}

/// [`compact_value`] for an object.
pub fn compact_map(m: &mut JsonMap) {
    let old = std::mem::take(m);
    let mut exact = Map::with_capacity(old.len());
    for (k, mut v) in old {
        compact_value(&mut v);
        exact.insert(k, v);
    }
    *m = exact;
}

/// Entry point — `normalizeNodeInPlace` (scene.normalize.ts:173-230).
pub fn normalize_node_in_place(node: &mut RawNode) {
    normalize_path_points_in_place(node);
    if node.node_type() == "line" {
        normalize_line_anchors_in_place(node);
    }
}

// ---------------------------------------------------------------------------
// 3. path point sanitization (pathHelpers.ts:8-196 via scene.normalize.ts:100-127)
// ---------------------------------------------------------------------------

fn is_finite_xy_point(v: Option<&Value>) -> bool {
    matches!(v, Some(Value::Object(o))
        if is_finite_number(o.get("x")).is_some() && is_finite_number(o.get("y")).is_some())
}

/// `isValidPathPoint` (pathHelpers.ts:14-51).
fn is_valid_path_point(v: &Value) -> bool {
    let Some(o) = v.as_object() else { return false };
    if is_finite_number(o.get("x")).is_none() || is_finite_number(o.get("y")).is_none() {
        return false;
    }
    match o.get("type").and_then(Value::as_str) {
        Some("move") | Some("line") => {}
        Some("curve") => {
            if !is_finite_xy_point(o.get("cp1")) || !is_finite_xy_point(o.get("cp2")) {
                return false;
            }
        }
        Some("quadratic") => {
            if !is_finite_xy_point(o.get("cp")) {
                return false;
            }
        }
        Some("arc") => {
            if is_finite_number(o.get("rx")).is_none()
                || is_finite_number(o.get("ry")).is_none()
                || is_finite_number(o.get("xAxisRotation")).is_none()
                || !matches!(o.get("largeArc"), Some(Value::Bool(_)))
                || !matches!(o.get("sweep"), Some(Value::Bool(_)))
            {
                return false;
            }
        }
        _ => return false,
    }
    if let Some(h) = o.get("outgoingHandle") {
        if !h.is_null() && !is_finite_xy_point(Some(h)) {
            return false;
        }
    }
    true
}

/// `sanitizePathPoints` (pathHelpers.ts:136-196): strip `incomingHandle`,
/// strip malformed `outgoingHandle`, coerce arc 0/1 flags to booleans, drop
/// invalid points.
pub fn sanitize_path_points(value: Option<&Value>) -> (Vec<Value>, bool) {
    let Some(Value::Array(arr)) = value else {
        return (Vec::new(), true);
    };
    let mut out: Vec<Value> = Vec::with_capacity(arr.len());
    let mut changed = false;
    for candidate in arr {
        let Some(rec) = candidate.as_object() else {
            changed = true;
            continue;
        };
        let mut rec = rec.clone();
        let mut item_changed = false;
        if rec.shift_remove("incomingHandle").is_some() {
            item_changed = true;
        }
        if let Some(h) = rec.get("outgoingHandle") {
            if !h.is_null() && !is_finite_xy_point(Some(h)) {
                rec.shift_remove("outgoingHandle");
                item_changed = true;
            }
        }
        if rec.get("type").and_then(Value::as_str) == Some("arc") {
            for flag in ["largeArc", "sweep"] {
                let coerced = match rec.get(flag).and_then(Value::as_f64) {
                    Some(f) if f == 0.0 || f == 1.0 => Some(f == 1.0),
                    _ => None,
                };
                if let Some(b) = coerced {
                    rec.insert(flag.into(), Value::Bool(b));
                    item_changed = true;
                }
            }
        }
        let v = Value::Object(rec);
        if !is_valid_path_point(&v) {
            changed = true;
            continue;
        }
        if item_changed {
            changed = true;
        }
        out.push(v);
    }
    (out, changed)
}

/// Whether [`sanitize_path_points`] would change `value`, without cloning a
/// point: a clean path (every hydrate of every path) skips the rebuild.
fn path_points_need_sanitizing(value: Option<&Value>) -> bool {
    let Some(Value::Array(arr)) = value else {
        return true;
    };
    arr.iter().any(|candidate| {
        let Some(rec) = candidate.as_object() else {
            return true;
        };
        if rec.contains_key("incomingHandle") {
            return true;
        }
        if let Some(h) = rec.get("outgoingHandle") {
            if !h.is_null() && !is_finite_xy_point(Some(h)) {
                return true;
            }
        }
        if rec.get("type").and_then(Value::as_str) == Some("arc")
            && ["largeArc", "sweep"]
                .iter()
                .any(|flag| matches!(rec.get(*flag).and_then(Value::as_f64), Some(f) if f == 0.0 || f == 1.0))
        {
            return true;
        }
        !is_valid_path_point(candidate)
    })
}

fn normalize_path_points_in_place(node: &mut RawNode) {
    // Compact points are canonical (finite anchors and control points of move / line / curve / quadratic points),
    // which is always valid.
    if node.node_type() != "path" || node.is_compact() {
        return;
    }
    if !path_points_need_sanitizing(node.json_field("points")) {
        return;
    }
    let (points, changed) = sanitize_path_points(node.json_field("points"));
    if changed {
        node.set("points", Value::Array(points));
    }
}

// ---------------------------------------------------------------------------
// 11. legacy line anchors (lineAnchorNormalize.ts)
// ---------------------------------------------------------------------------

const KNOWN_PRESETS: [&str; 9] = [
    "top",
    "bottom",
    "left",
    "right",
    "center",
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
];

fn is_finite_integer(v: Option<&Value>) -> Option<f64> {
    is_finite_number(v).filter(|f| f.fract() == 0.0)
}

/// `normalizePerimeterParam` (lineAnchorNormalize.ts:54-69).
fn normalize_perimeter_param(raw: Option<&Value>) -> Option<JsonMap> {
    let o = raw?.as_object()?;
    match o.get("kind").and_then(Value::as_str) {
        Some("scalar") => {
            is_finite_number(o.get("t"))?;
            let mut m = o.clone();
            m.insert("kind".into(), json!("scalar"));
            Some(m)
        }
        Some("edge") => {
            let idx = is_finite_integer(o.get("edgeIndex"))?;
            if idx < 0.0 {
                return None;
            }
            is_finite_number(o.get("localT"))?;
            let mut m = o.clone();
            m.insert("kind".into(), json!("edge"));
            Some(m)
        }
        _ => None,
    }
}

/// `normalizeAnchor` (lineAnchorNormalize.ts:81-161). Returns the normalized
/// anchor or None (caller decides drop-vs-keep per :199-235).
fn normalize_anchor(raw: &Value) -> Option<Value> {
    let a = raw.as_object()?;
    let node_id = a.get("nodeId").and_then(Value::as_str)?.to_string();
    let sub_path = a.get("subPath").and_then(Value::as_str).map(str::to_string);

    let with_subpath = |mut m: JsonMap| -> Value {
        if let Some(sp) = &sub_path {
            m.insert("subPath".into(), json!(sp));
        }
        Value::Object(m)
    };

    match a.get("kind").and_then(Value::as_str) {
        Some("preset") => {
            let preset = a.get("preset").and_then(Value::as_str)?;
            if !KNOWN_PRESETS.contains(&preset) {
                return None;
            }
            let mut m = Map::new();
            m.insert("nodeId".into(), json!(node_id));
            m.insert("kind".into(), json!("preset"));
            m.insert("preset".into(), json!(preset));
            Some(with_subpath(m))
        }
        Some("bbox") => {
            let u = is_finite_number(a.get("u"))?;
            let v = is_finite_number(a.get("v"))?;
            let mut m = Map::new();
            m.insert("nodeId".into(), json!(node_id));
            m.insert("kind".into(), json!("bbox"));
            m.insert("u".into(), json!(u));
            m.insert("v".into(), json!(v));
            Some(with_subpath(m))
        }
        Some("perimeter") => {
            if let Some(param) = normalize_perimeter_param(a.get("param")) {
                let mut m = Map::new();
                m.insert("nodeId".into(), json!(node_id));
                m.insert("kind".into(), json!("perimeter"));
                m.insert("param".into(), Value::Object(param));
                return Some(with_subpath(m));
            }
            // Forward-compat: unknown param kinds preserved unchanged
            // (lineAnchorNormalize.ts:126-144).
            if let Some(Value::Object(p)) = a.get("param") {
                if let Some(kind) = p.get("kind").and_then(Value::as_str) {
                    if kind != "scalar" && kind != "edge" {
                        let mut m = Map::new();
                        m.insert("nodeId".into(), json!(node_id));
                        m.insert("kind".into(), json!("perimeter"));
                        m.insert("param".into(), Value::Object(p.clone()));
                        return Some(with_subpath(m));
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn is_perimeter_kind(raw: Option<&Value>) -> bool {
    matches!(raw, Some(Value::Object(o)) if o.get("kind").and_then(Value::as_str) == Some("perimeter"))
}

/// lineAnchorNormalize.ts:199-235.
pub fn normalize_line_anchors_in_place(node: &mut RawNode) -> bool {
    if node.node_type() != "line" {
        return false;
    }
    let mut changed = false;
    for key in ["startAnchor", "endAnchor"] {
        let raw = node.json_field(key).cloned();
        if is_perimeter_kind(raw.as_ref()) {
            let raw = raw.unwrap();
            if let Some(normalized) = normalize_anchor(&raw) {
                node.set(key, normalized);
                changed = true;
            } else {
                // Malformed known-perimeter payloads are dropped (set
                // undefined in TS → remove the key here).
                node.remove(key);
                changed = true;
            }
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    /// The no-clone pre-check agrees with the sanitizer on every case it handles.
    #[test]
    fn the_path_point_precheck_agrees_with_the_sanitizer() {
        use crate::value::json;
        let cases = [
            json!([{"type":"move","x":0,"y":0},{"type":"line","x":1,"y":2}]),
            json!([{"type":"curve","x":0,"y":0,"cp1":{"x":1,"y":1},"cp2":{"x":2,"y":2}}]),
            json!([{"type":"line","x":0,"y":0,"incomingHandle":{"x":1,"y":1}}]),
            json!([{"type":"line","x":0,"y":0,"outgoingHandle":{"x":"a","y":1}}]),
            json!([{"type":"line","x":0,"y":0,"outgoingHandle":null}]),
            json!([{"type":"arc","x":0,"y":0,"rx":1,"ry":1,"xAxisRotation":0,"largeArc":1,"sweep":false}]),
            json!([{"type":"arc","x":0,"y":0,"rx":1,"ry":1,"xAxisRotation":0,"largeArc":true,"sweep":false}]),
            json!([{"type":"line","x":0}]),
            json!([3, {"type":"line","x":0,"y":0}]),
            json!([]),
            json!("nope"),
        ];
        for c in &cases {
            assert_eq!(
                super::path_points_need_sanitizing(Some(c)),
                super::sanitize_path_points(Some(c)).1,
                "{c}"
            );
        }
        assert!(super::path_points_need_sanitizing(None));
    }

    use super::*;
    use crate::value::json;

    fn node(v: Value) -> RawNode {
        RawNode::from_value(v).unwrap()
    }

    fn normalized(v: Value) -> RawNode {
        let mut n = node(v);
        normalize_node_in_place(&mut n);
        n
    }

    /// Every pass must be idempotent (PLAN M0c: "idempotent in-place passes").
    fn assert_idempotent(v: Value) {
        let mut once = node(v);
        normalize_node_in_place(&mut once);
        let mut twice = once.clone();
        normalize_node_in_place(&mut twice);
        assert_eq!(once, twice);
    }

    // 3. path point sanitization (pathHelpers.ts:136-196)
    #[test]
    fn path_points_sanitized() {
        let n = normalized(json!({"id":"p","type":"path","x":0,"y":0,"points":[
            {"type":"move","x":0,"y":0,"incomingHandle":{"x":1,"y":1}},
            {"type":"line","x":"oops","y":2},
            {"type":"arc","x":5,"y":5,"rx":1,"ry":1,"xAxisRotation":0,"largeArc":1,"sweep":0},
            {"type":"curve","x":9,"y":9,"cp1":{"x":1,"y":1},"cp2":{"x":2,"y":2},"outgoingHandle":"bad"}
        ]}));
        let pts = n.array("points").unwrap();
        assert_eq!(pts.len(), 3); // malformed 'line' dropped
        assert!(!pts[0].as_object().unwrap().contains_key("incomingHandle"));
        assert_eq!(pts[1]["largeArc"], json!(true)); // 0/1 coerced
        assert_eq!(pts[1]["sweep"], json!(false));
        assert!(!pts[2].as_object().unwrap().contains_key("outgoingHandle"));
        assert_idempotent(json!({"id":"p","type":"path","x":0,"y":0,"points":[
            {"type":"move","x":0,"y":0},
            {"type":"arc","x":5,"y":5,"rx":1,"ry":1,"xAxisRotation":0,"largeArc":1,"sweep":0}
        ]}));
    }

    // 11. legacy line anchors (lineAnchorNormalize.ts)
    #[test]
    fn line_perimeter_anchors_validated() {
        let v = json!({"id":"l","type":"line","points":[0,0,1,1],
            "endAnchor":{"nodeId":"b","kind":"perimeter","param":{"kind":"scalar","t":"NaN-ish"}}});
        let n = normalized(v);
        // malformed perimeter dropped (lineAnchorNormalize.ts:211-216)
        assert!(!n.has("endAnchor"));
        // unknown future param kind preserved (lineAnchorNormalize.ts:133-144)
        let v = json!({"id":"l","type":"line","points":[0,0,1,1],
            "endAnchor":{"nodeId":"b","kind":"perimeter","param":{"kind":"bezier","q":1}}});
        let n = normalized(v.clone());
        assert_eq!(
            n.json_field("endAnchor"),
            Some(&json!({"nodeId":"b","kind":"perimeter","param":{"kind":"bezier","q":1}}))
        );
        assert_idempotent(v);
    }

}

fn is_finite_number(v: Option<&Value>) -> Option<f64> { v.and_then(Value::as_f64).filter(|n| n.is_finite()) }
