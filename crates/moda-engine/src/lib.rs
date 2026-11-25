//! moda-engine — the moda-render engine core (PLAN.md M1).
//!
//! Scene JSON → (moda-scene hydration/display list) → `frame::CoveragePlan`
//! → headless wgpu execution → readback → encoded bytes.
//!
//! Module ownership after M1 (PLAN §0 rule 4):
//! - M1 (this milestone): lib.rs, seams.rs (FROZEN), device.rs, frame.rs,
//!   depth.rs, readback.rs, color.rs, shape_to_path.rs
//! - M2: tess/** (replaces the placeholder wholesale)
//! - M3a: text_draw.rs · M3b: paint/** · M3d: effects/** · M3e: backdrop/**
//!
//! See SEAMS.md for the frozen cross-track contract.

pub mod backdrop;
pub mod boolean_operands;
pub(crate) mod camera_slot;
pub use moda_scene::color;
pub mod cull;
/// vork: moda-text-draw's copy (the same file); see PROVENANCE.md.
pub use moda_text_draw::depth;
pub mod device;
pub mod effects;
pub mod export_scale;
pub mod external_texture;
pub mod frame;
mod geometry_bounds;
pub mod gpu_profile;
pub mod layer_blur;
pub mod paint;
pub mod pixel;
pub mod prof;
pub(crate) mod prepared_rows;
mod plan_items;
pub mod readback;
pub mod residency;
pub mod scratch;
pub mod seams;
pub mod shape_to_path;
pub mod solid;
pub mod surface;
pub mod tess;
pub mod text_area;
pub mod text_draw;
pub mod text_geometry;
pub mod text_metrics;
/// vork: moda-text-draw's copy (the same file); see PROVENANCE.md.
pub use moda_text_draw::vec_arena_cache;


pub use boolean_operands::install_scene_seams;
pub use device::{
    clear_device_lost_flag, device_lost_flag, is_device_lost_error, BackendChoice, GpuContext,
};
pub use frame::{Engine, PageRegionPx, RenderOptions};
/// An engine's working colour ([`Engine::new_with_color`]).
pub use moda_shaders::working_color;

/// Initialize both text lanes from bundled fixtures before either global host is built.
#[cfg(test)]
pub(crate) fn test_fonts() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fonts");
        text_draw::set_fonts_dir(dir.clone());
        moda_text::degraded::set_catalog_fonts_dir(dir);
    });
}
