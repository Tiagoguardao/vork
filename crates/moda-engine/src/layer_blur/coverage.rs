//! Integrate native raster tiles into a budget-limited capture. The filters read
//! the assembled image, so their kernels cross tile boundaries on the usual grid.
use crate::backdrop::plan::Rect;
use crate::seams::{CaptureRaster, CoveragePlan, DrawItem};

/// At four samples even an RGBA16F tile with depth uses less than 64 MiB.
pub(crate) const TILE: u32 = 1024;

pub(crate) struct Tile {
    pub camera: [f64; 2],
    pub size: [u32; 2],
    pub scissor: [u32; 4],
    /// Native pixel position of the first reduced cell's edge relative to the tile.
    pub first_edge: [f32; 2],
    pub step: [f32; 2],
}

/// The filter's original scale can be non-integral. Tile boundaries can therefore
/// split one destination cell. Both tiles contribute their share of its footprint.
pub(crate) struct Tiles {
    raster: CaptureRaster,
    camera: [f64; 2],
    step: [f64; 2],
    size: [u32; 2],
    x: std::ops::Range<u64>,
    y: std::ops::Range<u64>,
}

impl Tiles {
    pub(crate) fn new(plan: &CoveragePlan, raster: CaptureRaster) -> Self {
        let size = [plan.width_px, plan.height_px];
        let step = std::array::from_fn::<_, 2, _>(|i| {
            plan.page_size[i] * raster.scale / f64::from(size[i])
        });
        // Page clipping is applied by the source draw. In particular a single path
        // may extend far past the page even though only a few native pixels survive.
        let bounds = plan
            .items
            .iter()
            .filter_map(|item| {
                let bounds = super::shadow::all_items_canvas_bounds(std::slice::from_ref(item))?;
                let clip = match item {
                    DrawItem::Body { instance, .. } => instance.page_clip_bounds,
                    DrawItem::SolidShapeSdf { instance, .. } => instance.page_clip_bounds,
                    DrawItem::GradientShapeSdf { instance, .. } => instance.page_clip_bounds,
                    DrawItem::Shadow { instance, .. } => instance.page_clip_bounds,
                    DrawItem::Text(batch) => batch.page_clip_bounds,
                    DrawItem::LayerBlurComposite {
                        page_clip_bounds, ..
                    } => *page_clip_bounds,
                    _ => return Some(bounds),
                };
                let [x, y, w, h] = clip.map(f64::from);
                bounds.intersect(&Rect { x, y, w, h })
            })
            .fold(None, |acc, rect| {
                super::shadow::union_canvas_bounds(acc, Some(rect))
            });
        let range = |axis: usize| {
            let end = (raster.phase[axis] + f64::from(size[axis]) * step[axis]).ceil();
            let (lo, hi) = bounds.map_or((0., 1.), |rect| {
                let (p, span) = [(rect.x, rect.w), (rect.y, rect.h)][axis];
                let left = raster.phase[axis] + (p - plan.camera[axis]) * raster.scale;
                // Include the antialias fringe outside the analytic bounds.
                (
                    (left.floor() - 1.).clamp(0., end - 1.),
                    ((left + span * raster.scale).ceil() + 1.).clamp(1., end),
                )
            });
            lo as u64..(hi as u64).max((lo as u64).saturating_add(1))
        };
        Self {
            raster,
            camera: plan.camera,
            size,
            step,
            x: range(0),
            y: range(1),
        }
    }

    pub(crate) fn attachment_size(&self) -> [u32; 2] {
        [&self.x, &self.y].map(|r| (r.end - r.start).min(u64::from(TILE)) as u32)
    }

    pub(crate) fn scale(&self) -> f64 {
        self.raster.scale
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = Tile> + '_ {
        let Self {
            raster,
            camera,
            step,
            size,
            x,
            y,
        } = self;
        y.clone().step_by(TILE as usize).flat_map(move |top| {
            let bottom = y.end.min(top.saturating_add(TILE.into()));
            (x.clone()).step_by(TILE as usize).map(move |left| {
                let right = x.end.min(left.saturating_add(TILE.into()));
                let start = [left, top];
                let end = [right, bottom];
                let cell = |pixel: u64, i: usize| (pixel as f64 - raster.phase[i]) / step[i];
                let lo = std::array::from_fn::<_, 2, _>(|i| {
                    cell(start[i], i).floor().clamp(0., f64::from(size[i] - 1)) as u32
                });
                let hi = std::array::from_fn::<_, 2, _>(|i| {
                    cell(end[i], i)
                        .ceil()
                        .clamp(f64::from(lo[i] + 1), f64::from(size[i])) as u32
                });
                Tile {
                    camera: std::array::from_fn(|i| {
                        camera[i] + (start[i] as f64 - raster.phase[i]) / raster.scale
                    }),
                    size: [(right - left) as u32, (bottom - top) as u32],
                    scissor: [lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]],
                    first_edge: std::array::from_fn(|i| {
                        (raster.phase[i] + f64::from(lo[i]) * step[i] - start[i] as f64) as f32
                    }),
                    step: step.map(|n| n as f32),
                }
            })
        })
    }
}
