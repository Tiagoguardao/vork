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
        image_info(image, &self.base, &self.opened, pose.render.fps, || self.image_time(name, pose, request))
    }
}

/// Read resource metadata. Only sequences need a sampled time to choose the file.
pub(crate) fn image_info(
    image: &vork_scene::v2::Image,
    base: &std::path::Path,
    opened: &crate::footage::Opened,
    fps: vork_time::Time,
    sequence_time: impl FnOnce() -> Result<vork_time::exact::Germ>,
) -> Result<Value> {
    if let Some(render) = &image.render {
        return Ok(json!({"width":render.resolution[0],"height":render.resolution[1]}));
    }
    let mut path = base.join(image.file.as_ref().context("ask: image has no file")?);
    if crate::footage::is_sequence(&path) {
        let source = opened.source(&path)?;
        let index = source.frame_index_exact(&sequence_time()?, fps)?;
        path = source.files()[index].clone();
    } else if crate::footage::is_media(&path) {
        return opened.movie_info(&path);
    }
    let (width, height) = image::image_dimensions(&path).with_context(|| {
        format!(
            "ask: read image info {}; restore a readable image",
            path.display()
        )
    })?;
    Ok(json!({"width":width,"height":height}))
}
