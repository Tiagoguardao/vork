/// RGBA storage formats supported by the texture pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Storage {
    F16,
    F32,
}

impl Storage {
    pub const fn format(self) -> wgpu::TextureFormat {
        match self {
            Storage::F16 => wgpu::TextureFormat::Rgba16Float,
            Storage::F32 => wgpu::TextureFormat::Rgba32Float,
        }
    }

    pub const fn bytes_per_pixel(self) -> u64 {
        match self {
            Storage::F16 => 8,
            Storage::F32 => 16,
        }
    }
}
