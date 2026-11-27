use super::evaluation::{CpuScene, Pose};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use vork_scene::v2::BufferType;

impl CpuScene {
    pub fn buffer(&self, name: &str, count: bool, range: Option<[usize; 2]>) -> Result<Value> {
        ensure!(
            self.scene.buffers.contains_key(name),
            "ask: unknown buffer {name:?}"
        );
        ensure!(
            !crate::compute::writes_buffer(&self.scene, name),
            "not built yet: this answer needs the GPU program writing buffer {name:?}"
        );
        let buffer = self
            .buffers
            .data
            .get(name)
            .context("not built yet: grid buffer elements")?;
        let capacity = buffer.count();
        if count {
            return Ok(json!({"count":capacity,"capacity":capacity}));
        }
        let [first, count] = range.unwrap_or([0, capacity]);
        ensure!(
            first.checked_add(count).is_some_and(|end| end <= capacity),
            "ask: range exceeds buffer {name:?}; choose a slice within its count"
        );
        let values = buffer
            .words
            .chunks_exact(buffer.width)
            .skip(first)
            .take(count)
            .map(|words| {
                let words = words
                    .iter()
                    .map(|&word| match buffer.element_type {
                        BufferType::I32 => json!(word as i32),
                        BufferType::U32 | BufferType::Uvec4 => json!(word),
                        _ => json!(f32::from_bits(word)),
                    })
                    .collect::<Vec<_>>();
                if buffer.width == 1 {
                    words.into_iter().next().unwrap()
                } else {
                    json!(words)
                }
            })
            .collect::<Vec<_>>();
        Ok(json!(values))
    }

    pub fn image_time(
        &self,
        name: &str,
        pose: &Pose,
        request: &vork_time::Request,
    ) -> Result<vork_time::exact::Germ> {
        let image = self
            .scene
            .images
            .get(name)
            .with_context(|| format!("ask: unknown image {name:?}"))?;
        let time = request.exact()?;
        self.animation.footage_time(&pose.samples, &time, name, image.time.as_ref())?
            .with_context(|| format!("ask: image {name:?} is absent at this time"))
    }

    pub fn image_info(
        &self,
        name: &str,
        pose: &Pose,
        request: &vork_time::Request,
    ) -> Result<Value> {
        let image = self
            .scene
            .images
            .get(name)
            .with_context(|| format!("ask: unknown image {name:?}"))?;
        image_info(image, &self.base, &self.opened, pose.render.fps, StillAxes::Stored, || self.image_time(name, pose, request))
    }
}

/// Still dimensions in the consumer's coordinates. A canvas image question
/// reports the natural paint size. A world reports the stored texture size.
#[derive(Clone, Copy)]
pub(super) enum StillAxes {
    CanvasPaint,
    Stored,
}

/// Read resource metadata. Only sequences need a sampled time to choose the file.
pub(super) fn image_info(
    image: &vork_scene::v2::Image,
    base: &std::path::Path,
    opened: &crate::footage::Opened,
    fps: vork_time::Time,
    still_axes: StillAxes,
    sequence_time: impl FnOnce() -> Result<vork_time::exact::Germ>,
) -> Result<Value> {
    if let Some(render) = &image.render {
        return Ok(json!({"width":render.resolution[0],"height":render.resolution[1]}));
    }
    let path = base.join(image.file.as_ref().context("ask: image has no file")?);
    let (path, axes) = if crate::footage::is_sequence(&path) {
        // Both footage consumers decode frames in stored axes.
        let source = opened.source(&path)?;
        let index = source.frame_index_exact(&sequence_time()?, fps)?;
        (source.files()[index].clone(), StillAxes::Stored)
    } else if crate::footage::is_media(&path) {
        return opened.movie_info(&path);
    } else {
        (path, still_axes)
    };
    let source = opened.files.source(&path);
    let (width, height) = source.dimensions(matches!(axes, StillAxes::CanvasPaint))
        .with_context(|| crate::texture_source::DecodeLocation::image_info(source.name))?;
    Ok(json!({"width":width,"height":height}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_info_uses_the_consumers_axes_for_stills_and_sequence_frames() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let base = directory.path();
        let mut jpeg = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(32, 16).write_to(&mut jpeg, image::ImageFormat::Jpeg)?;
        let mut jpeg = jpeg.into_inner();
        // APP1 Exif. The big-endian TIFF IFD has a single Orientation SHORT set to 6.
        jpeg.splice(2..2, *b"\xff\xe1\x00\x22Exif\0\0MM\0\x2a\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01\0\x06\0\0\0\0\0\0");
        std::fs::write(base.join("frame.01.jpg"), jpeg)?;
        let fps = vork_time::Time::new(24, 1)?;
        let opened = crate::footage::Opened::default();
        let info = |file, axes| {
            let image = serde_json::from_value(json!({"file":file}))?;
            image_info(&image, base, &opened, fps, axes, || vork_time::Request::Numeric(0.).exact())
        };
        let stored = json!({"width":32,"height":16});
        assert_eq!(info("frame.01.jpg", StillAxes::CanvasPaint)?, json!({"width":16,"height":32}));
        assert_eq!(info("frame.01.jpg", StillAxes::Stored)?, stored);
        for axes in [StillAxes::CanvasPaint, StillAxes::Stored] {
            assert_eq!(info("frame.%02d.jpg", axes)?, stored);
        }
        Ok(())
    }
}
