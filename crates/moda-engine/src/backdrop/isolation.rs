//! Schedule isolated source-over groups on their existing pixel grid. All draws inside a group
//! still accumulate at full opacity. Regions sharing a pixel must observe painter's order.
use crate::plan_items::PlanItems;
use crate::seams::{BackdropOp, BackdropSplit, DrawItem};
use std::ops::Range;
use std::sync::Arc;

mod snapshot;
use snapshot::{Layer, Layout};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RegionVertex {
    rect: [f32; 4],
    address: [u32; 4],
    atlas: [u32; 3],
}
pub(crate) const VERTEX_BYTES: usize = std::mem::size_of::<RegionVertex>();

#[derive(Clone, Debug)]
pub(crate) enum Source {
    Text(Range<usize>),
    Segment(usize),
}

pub(crate) struct Region {
    pub rect: [u32; 4],
    pub opacity: f32,
    pub end_item: Option<usize>,
    pub source: Source,
}

pub(crate) struct Run {
    pub last_item: usize,
    pub extra_regions: usize,
    pub segments: Vec<usize>,
    pub direct_items: Vec<u32>,
    layers: Vec<Layer>,
}

const TILE: u32 = u16::BITS;

#[derive(Clone, Default)]
struct Tile {
    // The newest layer's exact pixels. Every other pixel precedes it and needs at most 'older'.
    latest: u32,
    older: u32,
    pixels: [u16; TILE as usize],
}

pub(crate) struct Grid {
    touched: Vec<(usize, usize)>,
    stride: usize,
    rows: Vec<Vec<Tile>>,
}

impl Grid {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            stride: width.div_ceil(TILE) as usize,
            rows: vec![Vec::new(); height.div_ceil(TILE) as usize],
            touched: Vec::new(),
        }
    }

    fn reset(&mut self) {
        for (row, col) in self.touched.drain(..) {
            self.rows[row][col] = Tile::default();
        }
    }

    // Each bit names a real pixel. Disjoint source rectangles can always share the first
    // layer, including when they touch the same tile. Keep older layers conservative.
    fn place(&mut self, [x, y, w, h]: [u32; 4]) -> usize {
        let cols = x / TILE..(x + w).div_ceil(TILE);
        let row_indices = y / TILE..(y + h).div_ceil(TILE);
        let row_bits = |col: u32| {
            let left = x.saturating_sub(col * TILE);
            let right = (x + w - col * TILE).min(TILE);
            ((1u32 << right) - (1u32 << left)) as u16
        };
        let pixel_rows = |row: u32| {
            y.saturating_sub(row * TILE) as usize..(y + h - row * TILE).min(TILE) as usize
        };
        let mut layer = 0;
        for row in row_indices.clone() {
            let tiles = &mut self.rows[row as usize];
            if tiles.is_empty() {
                tiles.resize_with(self.stride, Tile::default);
            }
            for col in cols.clone() {
                let tile = &tiles[col as usize];
                if tile.latest <= layer {
                    continue;
                }
                let mask = row_bits(col);
                layer = layer.max(
                    if tile.pixels[pixel_rows(row)]
                        .iter()
                        .any(|bits| bits & mask != 0)
                    {
                        tile.latest
                    } else {
                        tile.older
                    },
                );
            }
        }
        for row in row_indices {
            for col in cols.clone() {
                let tile = &mut self.rows[row as usize][col as usize];
                if tile.latest == 0 {
                    self.touched.push((row as usize, col as usize));
                }
                let next = layer + 1;
                if next < tile.latest {
                    tile.older = tile.older.max(next);
                } else {
                    if next > tile.latest {
                        tile.older = tile.latest;
                        tile.latest = next;
                        tile.pixels.fill(0);
                    }
                    let mask = row_bits(col);
                    for bits in &mut tile.pixels[pixel_rows(row)] {
                        *bits |= mask;
                    }
                }
            }
        }
        layer as usize
    }
}

fn corners([l, t, r, b]: [f32; 4]) -> [f32; 8] {
    [l, t, r, t, l, b, r, b]
}

/// local_bounds walks every glyph quad, decoration, emoji and degraded raster. Account
/// for f32 rounding inside kineticGlyphLocal before the ordinary two affine transforms.
fn text_corners(batch: &crate::seams::TextQuadBatch) -> Option<[f32; 8]> {
    if batch.distort.is_some() {
        return None;
    } // the projective divide can cross the horizon
    let [l, t, r, b] = batch.local_bounds()?;
    let mut size = [l, t, r, b].map(f64::abs).into_iter().fold(1., f64::max);
    for glyph in batch
        .vector
        .iter()
        .flat_map(|v| &v.ops)
        .flat_map(|op| op.records.iter())
    {
        if glyph.glyph_opacity < 0. {
            return None;
        }
        // Deferred padding on static records is at most one em. Other records
        // already include their em-space effects; the transformed box can also
        // grow by local_aa_px after applying the glyph affine.
        let em = f64::from(glyph.em_box.into_iter().map(f32::abs).fold(0., f32::max)) + 1.;
        let point = f64::from(glyph.pen_x.abs().max(glyph.baseline_y.abs()))
            + em * f64::from(glyph.font_size_px.abs());
        let [a, c, b, d] = glyph.glyph_lin.map(|x| f64::from(x).abs());
        let shift = f64::from(glyph.glyph_off[0].abs().max(glyph.glyph_off[1].abs()));
        size = size.max(point * (1. + 2. * (a + c).max(b + d)) + shift + f64::from(glyph.local_aa_px));
    }
    if let Some(q) = &batch.degraded {
        size = size
            .max(f64::from(q.offset[0].abs()) + q.raster.logical_w.abs())
            .max(f64::from(q.offset[1].abs()) + q.raster.logical_h.abs());
    }
    // Each vertex uses fewer than 32 operations before the node transform. Bound
    // absolute errors by the intermediate magnitudes even when terms cancel.
    if size >= f64::from(f32::MAX) / 64. {
        return None;
    }
    let err = size * 64. * f64::from(f32::EPSILON);
    Some(corners([
        ((l - err) as f32).next_down(),
        ((t - err) as f32).next_down(),
        ((r + err) as f32).next_up(),
        ((b + err) as f32).next_up(),
    ]))
}

/// Bound the vertices the ordinary pipelines emit, including effects which grow or move
/// the source quad. Even a projective text warp remains bounded by its fragment clip.
fn intervening_pixels(
    item: &DrawItem,
    vp: &[f32; 20],
    width: u32,
    height: u32,
) -> Option<(f64, [u32; 4])> {
    use crate::seams::BodyPaint;
    let clip = |bounds: &[f32; 4]| {
        let mut rect = [0; 4];
        for (axis, extent) in [width, height].into_iter().enumerate() {
            // rect-clip.wgsl uses the f32 sum, then fma. Include a pixel on each side
            // for sample positions and implementations which lower fma to mul + add.
            let scale = vp[16 + axis];
            let offset = vp[18 + axis];
            let project = |v: f32| {
                let a = f64::from(v) * f64::from(scale);
                let b = f64::from(offset);
                let margin = 1. + (a.abs() + b.abs()) * 4. * f64::from(f32::EPSILON);
                (a + b, margin)
            };
            let (lo, lerr) = project(bounds[axis]);
            let (hi, herr) = project(bounds[axis] + bounds[axis + 2]);
            if !(lo.is_finite() && hi.is_finite()) {
                return None;
            }
            let start = ((lo - lerr).floor() as u32).min(extent);
            let end = ((hi + herr).ceil() as u32).min(extent);
            if end <= start {
                return None;
            }
            rect[axis] = start;
            rect[axis + 2] = end - start;
        }
        // A full-target dependency would serialize both sides anyway. End this run
        // here and leave the draw in the existing page segment.
        (rect != [0, 0, width, height]).then_some(rect)
    };
    let vertex_quad;
    let mut world_offset = [0.; 2];
    let (z, m, coords) = match item {
        DrawItem::Body {
            z,
            geometry,
            instance,
            paint,
            ..
        } => {
            let model = match paint {
                BodyPaint::Solid => &instance.model_matrix,
                BodyPaint::Gradient(g) => &g.model_matrix,
                BodyPaint::Pattern(p) => &p.instance.model_matrix,
            };
            (*z, model, geometry.positions.as_slice())
        }
        DrawItem::SolidShapeSdf { z, .. } | DrawItem::GradientShapeSdf { z, .. } | DrawItem::Shadow { z, .. } => {
            let quad = crate::layer_blur::shadow::painted_quad(item)?;
            vertex_quad = quad.corners();
            world_offset = quad.offset;
            (*z, quad.model, vertex_quad.as_slice())
        }
        DrawItem::Text(batch) => {
            let Some(rect) = text_corners(batch) else {
                return Some((batch.z, clip(&batch.page_clip_bounds)?));
            };
            vertex_quad = rect;
            (batch.z, &batch.model_matrix, vertex_quad.as_slice())
        }
        _ => return None,
    };
    if m[3] != 0. || m[7] != 0. || m[15] != 1. {
        return None;
    }
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for xy in coords.chunks_exact(2) {
        for axis in 0..2 {
            let a = f64::from(m[axis]) * f64::from(xy[0]);
            let b = f64::from(m[4 + axis]) * f64::from(xy[1]);
            let c = f64::from(m[12 + axis]);
            let scale = f64::from(vp[axis * 5]);
            let bias = f64::from(vp[12 + axis]);
            let half = f64::from([width, height][axis]) * 0.5;
            let offset = f64::from(world_offset[axis]);
            let ndc = scale * (a + b + c + offset) + bias;
            let px = half * (1. + if axis == 0 { ndc } else { -ndc });
            // Each term goes through at most eight f32 operations. The extra pixel
            // includes viewport subpixel snapping and all MSAA sample locations.
            let magnitude =
                scale.abs() * (a.abs() + b.abs() + c.abs() + offset.abs()) + bias.abs() + 1.;
            let error = 1. + half * magnitude * (16. * f64::from(f32::EPSILON));
            if !px.is_finite()
                || !error.is_finite()
                || (a.abs() + b.abs() + c.abs() + offset.abs()).max(magnitude) > f64::from(f32::MAX)
            {
                return None;
            }
            lo[axis] = lo[axis].min(px - error);
            hi[axis] = hi[axis].max(px + error);
        }
    }
    let x = (lo[0].floor() as u32).min(width);
    let y = (lo[1].floor() as u32).min(height);
    let right = (hi[0].ceil() as u32).min(width);
    let bottom = (hi[1].ceil() as u32).min(height);
    (x < right && y < bottom).then(|| (z, [x, y, right - x, bottom - y]))
}

impl Run {
    pub fn regions(&self) -> impl Iterator<Item = &Region> { self.layers.iter().flat_map(|layer| &layer.regions) }

    pub fn snapshot_sizes(&self) -> impl Iterator<Item = [u32; 2]> + '_ {
        self.layers.iter().map(|layer| layer.layout.size()).filter(|size| size[0] != 0)
    }

    pub fn recording_bytes(&self) -> usize {
        std::mem::size_of::<Prepared>()
            + self.direct_items.len() * std::mem::size_of::<u32>()
            + self.segments.len() * std::mem::size_of::<(usize, wgpu::RenderBundle)>()
            + self.extra_regions * std::mem::size_of::<(u32, u32)>()
            + self.layers.iter().map(|l| std::mem::size_of::<PreparedLayer>() + l.layout.bytes()
                + l.regions.len() * std::mem::size_of::<([u32; 4], Source)>()).sum::<usize>()
    }

    pub fn collect(
        first_at: usize,
        first: &BackdropSplit,
        mut items: PlanItems<'_>,
        mut text_index: usize,
        depth_band: u64,
        width: u32,
        height: u32,
        max_regions: usize,
        grid: &mut Grid,
        vp: &[f32; 20],
    ) -> Option<Self> {
        let mut begin_at = first_at;
        let mut begin = first;
        let mut result = Self {
            last_item: 0,
            extra_regions: 0,
            segments: Vec::new(),
            direct_items: Vec::new(),
            layers: Vec::new(),
        };
        let mut levels = Vec::<Vec<Region>>::new();
        grid.reset();
        let mut regions = 0;
        // Intervening ordinary draws form one page segment. Preserve its internal phase
        // and z order. Commit it only after another shareable isolation has been found.
        let mut direct = Vec::<(u32, [u32; 4])>::new();
        while max_regions.saturating_sub(regions) >= 4 {
            let [x, y, w, h] = begin.region_px;
            if !matches!(begin.op, BackdropOp::BlendBegin)
                || w == 0
                || h == 0
                || x.checked_add(w).is_none_or(|r| r > width)
                || y.checked_add(h).is_none_or(|b| b > height)
            {
                break;
            }
            let start = text_index;
            let mut only_text = true;
            let mut last_z = f64::NEG_INFINITY;
            let mut nonempty = false;
            let mut next = items.next();
            while let Some((_, item)) = next {
                let z = match item {
                    DrawItem::Text(batch) if batch.distort.is_none() => {
                        text_index += 1;
                        if batch.z < last_z {
                            only_text = false;
                        }
                        batch.z
                    }
                    DrawItem::Body { z, .. }
                    | DrawItem::Shadow { z, .. }
                    | DrawItem::SolidShapeSdf { z, .. }
                    | DrawItem::GradientShapeSdf { z, .. } => {
                        only_text = false;
                        *z
                    }
                    _ => break,
                };
                if z as u64 / crate::depth::MAX_RENDER_Z != depth_band {
                    break;
                }
                nonempty = true;
                last_z = z;
                next = items.next();
            }
            let Some((at, DrawItem::BackdropSplit(end))) = next else {
                break;
            };
            let BackdropOp::StackEnd { mode: 0, opacity } = end.op else {
                break;
            };
            if !nonempty || end.node_id != begin.node_id || end.region_px != begin.region_px {
                break;
            }
            if let Some(&(key, rect)) = direct.first() {
                let [l, t, r, b] = direct.iter().fold(
                    [rect[0], rect[1], 0, 0],
                    |[l, t, r, b], &(_, [x, y, w, h])| {
                        [l.min(x), t.min(y), r.max(x + w), b.max(y + h)]
                    },
                );
                let rect = [l, t, r - l, b - t];
                let level = grid.place(rect);
                if level == levels.len() {
                    levels.push(Vec::new());
                }
                result.segments.push(key as usize);
                result
                    .direct_items
                    .extend(direct.drain(..).map(|(at, _)| at));
                levels[level].push(Region {
                    rect,
                    opacity: 1.,
                    end_item: None,
                    source: Source::Segment(key as usize),
                });
            }
            let source = if only_text {
                Source::Text(start..text_index)
            } else {
                result.segments.push(begin_at);
                Source::Segment(begin_at)
            };
            let first_group = levels.is_empty();
            let mut emit = |level: usize, rect| {
                if level == levels.len() {
                    levels.push(Vec::new());
                }
                regions += 1;
                levels[level].push(Region {
                    rect,
                    opacity,
                    end_item: Some(at),
                    source: source.clone(),
                });
            };
            if first_group {
                grid.place(begin.region_px);
                emit(0, begin.region_px);
            } else {
                // Independent screen tiles may advance at different rates. A chain of overlapping
                // neighbors across the page only orders the pixels each neighbor actually covers.
                // Bound replay of an entire paragraph: a region can cross at most two blocks
                // on either axis. Whole groups that share a level still draw only once.
                // For a chain of touching bounds, a block limits dependencies to B/label
                // neighbours. There are page/B blocks along that axis. Equalising these two
                // spans gives B = sqrt(page*label). Round up in log2 space for aligned seams.
                let page = width.max(height).next_power_of_two().ilog2();
                let label = w.max(h).max(TILE).next_power_of_two().ilog2();
                let block = 1 << (page + label).div_ceil(2);
                let mut pieces = [(0, [0; 4]); 4];
                let mut count = 0;
                for row in y / block..(y + h).div_ceil(block) {
                    for col in x / block..(x + w).div_ceil(block) {
                        let l = x.max(col * block);
                        let t = y.max(row * block);
                        let rect = [
                            l,
                            t,
                            (x + w).min((col + 1) * block) - l,
                            (y + h).min((row + 1) * block) - t,
                        ];
                        let level = grid.place(rect);
                        pieces[count] = (level, rect);
                        count += 1;
                    }
                }
                let pieces = &pieces[..count];
                let level = pieces[0].0;
                if pieces.iter().all(|p| p.0 == level) {
                    emit(level, begin.region_px);
                } else {
                    result.extra_regions += pieces.len() - 1;
                    for &(level, rect) in pieces {
                        emit(level, rect);
                    }
                }
            }
            result.last_item = at;
            let mut next = items.next();
            while let Some((at, item)) = next {
                let Some((z, rect)) = intervening_pixels(item, vp, width, height) else {
                    break;
                };
                if z as u64 / crate::depth::MAX_RENDER_Z != depth_band {
                    break;
                }
                text_index += usize::from(matches!(item, DrawItem::Text(_)));
                direct.push((at as u32, rect));
                next = items.next();
            }
            let Some((next_at, DrawItem::BackdropSplit(split))) = next else {
                break;
            };
            begin_at = next_at;
            begin = split;
        }
        for level in levels { snapshot::place(level, width, height, &mut result.layers); }
        (!result.layers.is_empty()).then_some(result)
    }
}

pub(crate) struct Pipelines {
    clear: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    module: wgpu::ShaderModule,
    gather: std::sync::OnceLock<wgpu::RenderPipeline>,
}

impl Pipelines {
    pub(super) fn new(device: &wgpu::Device, format: wgpu::TextureFormat, samples: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Isolated group"),
            source: wgpu::ShaderSource::Wgsl(super::blend::BLEND_WGSL.into()),
        });
        let pipeline = |entry: &str| {
            moda_text_draw::device::create_render_pipeline(
                device,
                &wgpu::RenderPipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_region"),
                        compilation_options: Default::default(),
                        buffers: &[
                            wgpu::VertexBufferLayout {
                                array_stride: VERTEX_BYTES as u64,
                                step_mode: wgpu::VertexStepMode::Instance,
                                attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Uint32x4],
                            },
                            wgpu::VertexBufferLayout {
                                array_stride: 4,
                                step_mode: wgpu::VertexStepMode::Instance,
                                attributes: &wgpu::vertex_attr_array![2 => Float32],
                            },
                        ],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: Default::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth24Plus,
                        depth_write_enabled: Some(false),
                        depth_compare: Some(wgpu::CompareFunction::Always),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: samples,
                        ..Default::default()
                    },
                    multiview_mask: None,
                    cache: None,
                },
            )
        };
        Self {
            clear: pipeline("fs_clear"),
            composite: pipeline("fs_region"),
            module: shader,
            gather: Default::default(),
        }
    }

    fn gather(&self, device: &wgpu::Device, format: wgpu::TextureFormat) -> &wgpu::RenderPipeline {
        self.gather.get_or_init(|| moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
            label: Some("Gather isolated group pixels"), layout: None,
            vertex: wgpu::VertexState {
                module: &self.module, entry_point: Some("vs_gather"), compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout { array_stride: VERTEX_BYTES as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Uint32x4, 3 => Uint32x3],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.module, entry_point: Some("fs_gather"), compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview_mask: None, cache: None,
        }))
    }

}

pub(crate) struct Gather<'a> {
    pipeline: &'a wgpu::RenderPipeline,
    binding: wgpu::BindGroup,
}

struct Pair {
    backdrop: crate::scratch::PageTexture,
    source: crate::scratch::PageTexture,
    bindings: wgpu::BindGroup,
}

/// Consecutive layers and runs finish reading both images before the next copy.
pub(crate) struct Snapshots(std::collections::BTreeMap<u32, Arc<Pair>>);

impl Snapshots {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, pipeline: &Pipelines,
        stack: &super::blend::BlendStack<'_>, sizes: impl Iterator<Item = [u32; 2]>) -> Self {
        Self(super::blend::snapshot_extents(sizes).into_iter().map(|(class, [w, h])| {
            let backdrop = stack.snapshot(device, format, w, h, "Isolated group backdrop", wgpu::TextureUsages::RENDER_ATTACHMENT);
            let source = stack.snapshot(device, format, w, h, "Isolated group source", wgpu::TextureUsages::RENDER_ATTACHMENT);
            let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Isolated group snapshots"),
                layout: &pipeline.composite.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&backdrop.view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&source.view) },
                ],
            });
            (class, Arc::new(Pair { backdrop, source, bindings }))
        }).collect())
    }

    pub fn textures(&self) -> impl Iterator<Item = &wgpu::Texture> {
        self.0.values().flat_map(|p| [p.backdrop.view.texture(), p.source.view.texture()])
    }

    fn pair(&self, width: u32) -> Option<Arc<Pair>> {
        (width != 0).then(|| self.0.get(&0).unwrap_or_else(|| &self.0[&width.next_power_of_two()]).clone())
    }
}

struct PreparedLayer {
    layout: Layout,
    snapshots: Option<Arc<Pair>>,
    vertices: Range<u32>,
    draws: Range<usize>,
}

pub(crate) struct Prepared {
    vertices: crate::scratch::PageBuffer,
    opacities: crate::scratch::PageBuffer,
    layers: Vec<PreparedLayer>,
    draws: Vec<([u32; 4], Source)>,
    pub segments: std::collections::BTreeMap<usize, wgpu::RenderBundle>,
}

impl Prepared {
    pub fn new(
        device: &wgpu::Device,
        snapshots: &Snapshots,
        scratch: &crate::scratch::ScratchPool,
        admit: Option<&dyn Fn(u64)>,
        run: Run,
        segments: std::collections::BTreeMap<usize, wgpu::RenderBundle>,
        target: &wgpu::Texture,
    ) -> Self {
        let count = run.layers.iter().map(|l| l.regions.len()).sum();
        let mut vertices = Vec::<RegionVertex>::with_capacity(count);
        let mut opacities = Vec::with_capacity(count);
        let mut layers = Vec::with_capacity(run.layers.len());
        let clip_x = |x: u32| (2. * f64::from(x) / f64::from(target.width()) - 1.) as f32;
        let clip_y = |y: u32| (1. - 2. * f64::from(y) / f64::from(target.height())) as f32;
        let mut draws = Vec::with_capacity(count);
        for Layer { regions, layout } in run.layers {
            let pair = snapshots.pair(layout.size()[0]);
            let texture_size = pair.as_ref().map_or([1, 1], |pair| {
                let tex = pair.source.view.texture();
                [tex.width(), tex.height()]
            });
            let start = vertices.len();
            let first_draw = draws.len();
            let mut first_pixel = 0;
            for fade in regions {
                draws.push((fade.rect, fade.source));
                if fade.end_item.is_none() { continue; }
                let [x, y, w, h] = fade.rect;
                let count = w * h;
                vertices.push(RegionVertex {
                    rect: [clip_x(x), clip_y(y), clip_x(x + w), clip_y(y + h)],
                    address: layout.address(fade.rect, first_pixel, texture_size[0]),
                    atlas: [texture_size[0], texture_size[1], count],
                });
                first_pixel += count;
                opacities.push(fade.opacity);
            }
            layers.push(PreparedLayer { snapshots: pair, layout,
                vertices: start as u32..vertices.len() as u32, draws: first_draw..draws.len() });
        }
        // Both mapped-at-creation destinations belong to this recording. Future
        // edits address the same opacity buffer until it leaves the prepared packet.
        if let Some(admit) = admit {
            admit((vertices.len() * VERTEX_BYTES + opacities.len() * std::mem::size_of::<f32>()) as u64);
        }
        let vertices = scratch.page_buffer(device, &wgpu::util::BufferInitDescriptor {
            label: Some("Isolated group regions"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        }, admit.is_none());
        let opacities = scratch.page_buffer(device, &wgpu::util::BufferInitDescriptor {
            label: Some("Isolated group opacities"),
            contents: bytemuck::cast_slice(&opacities),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        }, admit.is_none());
        Self { draws, segments, vertices, opacities, layers }
    }

    pub fn record_buffers(&self, scratch: &crate::scratch::ScratchPool, encoder: &wgpu::CommandEncoder, queue: &wgpu::Queue) {
        scratch.record_buffers(encoder, queue, vec![self.vertices.buffer.clone(), self.opacities.buffer.clone()]);
    }

    pub fn write_opacities(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        first: u32,
        values: &[f32],
    ) {
        crate::scratch::write_buffer(
            device,
            queue,
            &self.opacities.buffer,
            u64::from(first) * 4,
            bytemuck::cast_slice(values),
        );
    }

    pub fn gather_binding<'a>(
        &self,
        device: &wgpu::Device,
        pipelines: &'a Pipelines,
        target: &wgpu::Texture,
    ) -> Option<Gather<'a>> {
        // Keep the transfer path for external, copy-only or sRGB targets.
        // An sRGB sample/store round-trip is not a byte copy.
        if target.format().is_srgb()
            || !target
                .usage()
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
            || !self
                .layers
                .iter()
                .any(|layer| matches!(layer.layout, Layout::Packed { .. }))
        {
            return None;
        }
        let pipeline = pipelines.gather(device, target.format());
        let view = target.create_view(&Default::default());
        let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gather from resolved page"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        Some(Gather { pipeline, binding })
    }

    fn copy(
        &self,
        layer: &PreparedLayer,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
        dest: &wgpu::TextureView,
        gather: Option<&Gather<'_>>,
    ) {
        let Some(gather) = gather.filter(|_| matches!(layer.layout, Layout::Packed { .. })) else {
            layer.layout.copy(encoder, target, dest.texture());
            return;
        };
        let ts = crate::gpu_profile::pass_timestamps();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Gather sparse fade regions"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dest,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: crate::gpu_profile::render_writes(&ts),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(gather.pipeline);
        pass.set_bind_group(0, &gather.binding, &[]);
        pass.set_vertex_buffer(0, self.vertices.buffer.slice(..));
        pass.draw(0..6, layer.vertices.clone());
    }

    pub fn save_backdrop(
        &self,
        layer: usize,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
        gather: Option<&Gather<'_>>,
    ) {
        let layer = &self.layers[layer];
        if let Some(pair) = &layer.snapshots {
            self.copy(layer, encoder, target, &pair.backdrop.view, gather);
        }
    }

    pub fn clear<'a>(
        &'a self,
        layer: usize,
        pipelines: &'a Pipelines,
        pass: &mut crate::gpu_profile::CountedPass<'a>,
    ) {
        if self.layers[layer].vertices.is_empty() {
            return;
        }
        pass.set_pipeline(&pipelines.clear);
        pass.set_vertex_buffer(0, self.vertices.buffer.slice(..));
        pass.set_vertex_buffer(1, self.opacities.buffer.slice(..));
        pass.draw(0..6, self.layers[layer].vertices.clone());
    }

    pub fn save_source(
        &self,
        layer: usize,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
        gather: Option<&Gather<'_>>,
    ) {
        let layer = &self.layers[layer];
        if let Some(pair) = &layer.snapshots {
            self.copy(layer, encoder, target, &pair.source.view, gather);
        }
    }

    pub fn composite<'a>(
        &'a self,
        layer: usize,
        pipelines: &'a Pipelines,
        pass: &mut crate::gpu_profile::CountedPass<'a>,
    ) {
        if self.layers[layer].vertices.is_empty() {
            return;
        }
        pass.set_pipeline(&pipelines.composite);
        pass.set_bind_group(0, &self.layers[layer].snapshots.as_ref().unwrap().bindings, &[]);
        pass.set_vertex_buffer(0, self.vertices.buffer.slice(..));
        pass.set_vertex_buffer(1, self.opacities.buffer.slice(..));
        pass.draw(0..6, self.layers[layer].vertices.clone());
    }

    pub fn draws(&self, layer: usize) -> &[([u32; 4], Source)] {
        &self.draws[self.layers[layer].draws.clone()]
    }

    pub fn resolves(&self, layer: usize) -> bool {
        !self.layers[layer].vertices.is_empty()
    }

    pub fn last_layer(&self) -> usize {
        self.layers.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn snapshot_pair_and_regions_transfer_their_exact_charge_to_the_consuming_encoder() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).unwrap();
        let (device, queue) = (&gpu.device, &gpu.queue);
        let format = wgpu::TextureFormat::Rgba16Float;
        let pool = crate::scratch::ScratchPool::default();
        let demand = std::cell::Cell::new(0);
        let admit = |bytes| demand.set(demand.get() + bytes);
        let stack = crate::backdrop::blend::BlendStack::new(&pool, Some(&admit));
        let snapshots = Snapshots::new(device, format, &Pipelines::new(device, format, 1), &stack,
            [[13, 80], [300, 17], [12, 78], [300, 17]].into_iter());
        let region = pool.page_buffer(device, &wgpu::util::BufferInitDescriptor {
            label: None, contents: &[0u8; 224], usage: wgpu::BufferUsages::VERTEX,
        }, false);
        let bytes = snapshots.textures().map(crate::scratch::texture_bytes).sum::<u64>() + region.buffer.size();
        assert_eq!(snapshots.textures().count(), 4);
        assert!(demand.get() >= (13 * 80 + 300 * 17) * 16);
        let room = || pool.capture_room(queue, bytes, &Default::default());
        assert_eq!(room(), 0);
        let encoder = device.create_command_encoder(&Default::default());
        pool.record_reads(&encoder, queue, snapshots.textures());
        pool.record_buffers(&encoder, queue, vec![region.buffer.clone()]);
        drop(snapshots);
        drop(region);
        assert_eq!(room(), bytes);
        assert_eq!(pool.group_allowance(room(), bytes), 0);
        drop(encoder);
        assert_eq!(pool.group_allowance(room(), bytes), bytes);
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn scalar_reads_cross_encoder_rollover_once_per_backing() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).unwrap();
        let (device, queue) = (&gpu.device, &gpu.queue);
        let pool = crate::scratch::ScratchPool::default();
        let mut stack = crate::backdrop::blend::BlendStack::new(&pool, None);
        let texture = stack.snapshot(device, wgpu::TextureFormat::Rgba16Float, 128, 32, "split", wgpu::TextureUsages::empty());
        let tex = texture.view.texture();
        let bytes = crate::scratch::texture_bytes(tex);
        let first = device.create_command_encoder(&Default::default());
        stack.reads.track(&pool, tex);
        stack.reads.track(&pool, tex);
        stack.reads.attach(&first, queue);
        let second = device.create_command_encoder(&Default::default());
        stack.reads.track(&pool, tex);
        stack.reads.attach(&second, queue);
        drop(texture);
        let available = || pool.group_allowance(bytes, bytes);
        assert_eq!(available(), bytes);
        assert_eq!(pool.required_page_bytes(), bytes);
        let submitted = queue.submit([first.finish()]);
        device.poll(wgpu::PollType::Wait { submission_index: Some(submitted), timeout: None }).unwrap();
        assert_eq!(available(), bytes);
        assert_eq!(pool.required_page_bytes(), bytes, "the second encoder still owns a read");
        drop(second);
        assert_eq!(available(), bytes);
        assert_eq!(pool.required_page_bytes(), 0);
    }

    #[test]
    fn disjoint_regions_share_a_layer_and_every_covered_pixel_stays_ordered() {
        let mut grid = Grid::new(64, 48);
        for x in 0..16 {
            assert_eq!(grid.place([x, 1, 1, 12]), 0);
        }
        let mut pixels = [[0; 64]; 48];
        for row in &mut pixels[1..13] {
            row[..16].fill(1);
        }
        // Compare the conservative schedule with each pixel's actual last writer across tile edges.
        let mut seed = 53u32;
        let mut next = |n| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 12) % n
        };
        for _ in 0..2048 {
            let x = next(64);
            let y = next(48);
            let w = 1 + next(64 - x);
            let h = 1 + next(48 - y);
            let layer = grid.place([x, y, w, h]);
            for row in &mut pixels[y as usize..(y + h) as usize] {
                let touched = &mut row[x as usize..(x + w) as usize];
                assert!(touched.iter().all(|&previous| previous <= layer));
                touched.fill(layer + 1);
            }
        }
    }
}
