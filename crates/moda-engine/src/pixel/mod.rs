//! Pooled GPU textures for nested world renders.

pub mod image;
pub mod pool;

pub use image::Storage;
pub use pool::{PoolStats, PoolTexture, TexturePool, DEFAULT_BUDGET_BYTES};

/// Texture pool allocation or adoption failed.
#[derive(Debug, Clone, PartialEq)]
pub enum PixelError {
    /// Allocating would take the pool past its byte budget.
    Budget { needed: u64, in_use: u64, budget: u64 },
    /// A domain larger than the device's texture limit.
    TooLarge { w: u32, h: u32, max: u32 },
    /// Invalid texture dimensions, format or usage.
    Invalid(String),
}

impl std::fmt::Display for PixelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PixelError::Budget { needed, in_use, budget } => write!(
                f,
                "the texture pool needs {} MB more GPU memory but {} of its {} MB are in use (fewer or smaller \
                 images or a larger MODA_PIXEL_BUDGET_MB)",
                needed.div_ceil(1 << 20),
                in_use.div_ceil(1 << 20),
                budget >> 20
            ),
            PixelError::TooLarge { w, h, max } => {
                write!(f, "an image of {w}×{h} px is past the GPU's {max} px limit")
            }
            PixelError::Invalid(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for PixelError {}
