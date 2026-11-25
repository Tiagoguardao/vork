//! Every faded texel in a scheduled layer has a unique address. Keep the
//! layer intact so packing cannot add another page resolve.
use super::Region;

pub(super) struct Layer {
    pub regions: Vec<Region>,
    pub layout: Layout,
}

pub(super) enum Layout {
    Rect([u32; 4]),
    Packed {
        size: [u32; 2],
        rects: Vec<[u32; 4]>,
    },
}

impl Layout {
    pub fn size(&self) -> [u32; 2] {
        match self {
            Self::Rect([_, _, w, h]) => [*w, *h],
            Self::Packed { size, .. } => *size,
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            Self::Rect(_) => 0,
            Self::Packed { rects, .. } => rects.capacity() * std::mem::size_of::<[u32; 4]>(),
        }
    }

    /// Source origin, source row width and first packed address. The shader
    /// linearizes a pixel relative to the origin, then divides by the pitch of
    /// the physical backing (which may be shared with other layers).
    pub fn address(&self, rect: [u32; 4], first: u32, pitch: u32) -> [u32; 4] {
        match self {
            Self::Rect([x, y, ..]) => [*x, *y, pitch, 0],
            Self::Packed { .. } => [rect[0], rect[1], rect[2], first],
        }
    }

    fn transfers(&self, pitch: u32, mut copy: impl FnMut([u32; 4], [u32; 2])) {
        match self {
            Self::Rect(rect) => copy(*rect, [0, 0]),
            Self::Packed { rects, .. } => {
                let mut address = 0u64;
                for &[x, y, w, h] in rects {
                    let start = address;
                    let end = start + u64::from(w) * u64::from(h);
                    while address < end {
                        let local = (address - start) as u32;
                        let [u, v] = [
                            (address % u64::from(pitch)) as u32,
                            (address / u64::from(pitch)) as u32,
                        ];
                        let col = local % w;
                        let row = local / w;
                        let cols = (w - col).min(pitch - u);
                        let rows = if cols == w && cols == pitch {
                            h - row
                        } else {
                            1
                        };
                        copy([x + col, y + row, cols, rows], [u, v]);
                        address += u64::from(cols) * u64::from(rows);
                    }
                }
            }
        }
    }

    pub fn copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
        dest: &wgpu::Texture,
    ) {
        self.transfers(dest.width(), |[x, y, width, height], [u, v]| {
            let mut src = target.as_image_copy();
            src.origin = wgpu::Origin3d { x, y, z: 0 };
            let mut dst = dest.as_image_copy();
            dst.origin = wgpu::Origin3d { x: u, y: v, z: 0 };
            encoder.copy_texture_to_texture(
                src,
                dst,
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        });
    }
}

pub(super) fn place(regions: Vec<Region>, width: u32, height: u32, layers: &mut Vec<Layer>) {
    let (mut left, mut top, mut right, mut bottom) = (width, height, 0, 0);
    let mut occupied = 0u64;
    for region in regions.iter().filter(|r| r.end_item.is_some()) {
        let [x, y, w, h] = region.rect;
        left = left.min(x);
        top = top.min(y);
        right = right.max(x + w);
        bottom = bottom.max(y + h);
        occupied += u64::from(w) * u64::from(h);
    }
    let w = right.saturating_sub(left);
    let h = bottom.saturating_sub(top);
    let rows = occupied.div_ceil(u64::from(width.max(1)));
    let pitch = occupied.div_ceil(rows.max(1));
    // One resolve either way. Choose the smaller backing for both snapshots.
    // A rectangle that fits in the same space also saves two gather passes.
    let layout = if u64::from(w) * u64::from(h) <= pitch * rows {
        Layout::Rect([left, top, w, h])
    } else {
        Layout::Packed {
            size: [pitch as u32, rows as u32],
            rects: regions
                .iter()
                .filter(|r| r.end_item.is_some())
                .map(|r| r.rect)
                .collect(),
        }
    };
    layers.push(Layer { regions, layout });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backdrop::isolation::Source;

    #[test]
    fn packed_transfers_and_shader_addresses_cover_the_same_texels() {
        let mut rects = vec![[0, 0, 1000, 1], [1000, 10, 1, 1000]];
        for y in (10..940).step_by(65) {
            for x in (0..930).step_by(81) {
                rects.push([x, y, 24 + x % 23, 3 + y % 4]);
            }
        }
        let mut inputs: Vec<_> = rects
            .iter()
            .enumerate()
            .map(|(id, &rect)| Region {
                rect,
                opacity: 0.5,
                end_item: Some(id),
                source: Source::Segment(id),
            })
            .collect();
        inputs.insert(
            1,
            Region {
                rect: [4, 999, 7, 12],
                opacity: 1.,
                end_item: None,
                source: Source::Segment(1),
            },
        );
        let mut layers = Vec::new();
        place(inputs, 1024, 1024, &mut layers);
        assert_eq!(layers.len(), 1);
        let layer = layers.pop().unwrap();
        let [w, h] = layer.layout.size();
        let occupied: u32 = rects.iter().map(|r| r[2] * r[3]).sum();
        assert!(w * h >= occupied && w * h - occupied < h);
        for pitch in [w, w + 17] {
            let mut copied = vec![None; (pitch * h) as usize];
            layer.layout.transfers(pitch, |[sx, sy, sw, sh], [u, v]| {
                for row in 0..sh {
                    for col in 0..sw {
                        assert!(
                            copied[((v + row) * pitch + u + col) as usize]
                                .replace([sx + col, sy + row])
                                .is_none()
                        );
                    }
                }
            });
            let mut first = 0;
            for rect in &rects {
                let [x, y, w, h] = *rect;
                let [ox, oy, stride, base] = layer.layout.address(*rect, first, pitch);
                for cy in y..y + h {
                    for cx in x..x + w {
                        assert_eq!(
                            copied[(base + (cy - oy) * stride + cx - ox) as usize],
                            Some([cx, cy])
                        );
                    }
                }
                first += w * h;
            }
            assert_eq!(copied.iter().flatten().count(), occupied as usize);
        }
    }
}
