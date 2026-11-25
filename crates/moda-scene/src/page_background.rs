//! Synthetic page background geometry.
use crate::types::{compute_page_bg_id, RawNode, RawPage};
use crate::{paint, value::{json, Map}};
pub fn create_rectangle_from_page(page: &RawPage) -> RawNode {
    let mut m = Map::new();
    m.insert("id".into(), json!(compute_page_bg_id(page.id())));
    m.insert("type".into(), json!("rectangle"));
    m.insert("x".into(), json!(0.0));
    m.insert("y".into(), json!(0.0));
    m.insert("width".into(), json!(page.width()));
    m.insert("height".into(), json!(page.height()));
    m.insert("rotation".into(), json!(0.0));
    m.insert("scale".into(), json!(1.0));

    m.extend(paint::page_background_fields(page));
    m.insert("locked".into(), json!(true));
    m.insert("opacity".into(), json!(1.0));
    RawNode::new(m)
}
