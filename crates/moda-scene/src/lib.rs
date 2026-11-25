//! Scene hydration, derived geometry and display lists for canvas rendering.


pub mod rows;
pub mod appearance;
pub mod backdrop_kinds;
pub mod blend;
pub mod boolean_node;
pub mod corner_smoothing;
pub mod rect_corners;
pub mod defaults;
pub mod display_list;
pub mod effect;
// Shared distort homography math (TS projectiveTransform.ts mirror) — used by
// the distort readers and the v3 animation tilt/skew bakers.
mod distort;
mod page_background;
pub mod fields;
pub mod flags;
pub mod normalize;
pub mod page_render;
pub mod color;
pub mod paint;
pub mod path_bool;
pub mod path_geom;
pub mod path_length;
pub mod feed_points;
pub mod path_points;
pub mod rectangle_path;
pub mod scene;
pub mod shape_outline;
pub mod stamps;
pub mod transform;
pub mod types;
pub mod value;
pub mod worlds;

pub use display_list::{
    build_display_list, build_display_list_for_canvas, canvas_root_content_bounds,
    canvas_root_frame_size, CANVAS_ROOT_FRAME_MARGIN,
};
pub use path_bool::{
    boolean_paths, boolean_paths_json, pathfinder_paths, pathfinder_paths_json, region_fill_json,
    shape_builder_json,
};
pub use path_geom::fit::fit_points_json;
pub use path_geom::offset_path::offset_path_json;
pub use path_geom::simplify::simplify_path_json;
pub use scene::{extract_canvas_data, extract_canvas_data_owned, load_scene_json, ReexpandOutcome, Scene};
pub use types::{
    AxisStatus, DisplayList, Manifest, PaintItem, PaintKind, PaintPhase, NodeField, RawNode, RawPage,
    SceneDocument,
};
