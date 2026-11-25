//! Per-frame values: node and camera evaluation, uniforms, lights, deformation
//! parameters, packed into one upload per sample.
use super::*;

/// (destination, destination offset, source offset in the packed bytes, size)
pub(crate) type Upload = (wgpu::Buffer, u64, u64, u64);
/// Byte history makes overlapping dirty writes exact without scanning other uploads.
#[derive(Default)]
pub(crate) struct UploadCache {
    bytes: Vec<u8>,
    valid: Vec<bool>,
    stamp: u64,
}
impl UploadCache {
    fn changed(&mut self, offset: usize, bytes: &[u8]) -> bool {
        profile_span!(_profile, "byte_compare");
        let end = offset + bytes.len();
        if self.bytes.len() < end {
            self.bytes.resize(end, 0);
            self.valid.resize(end, false);
        }
        if self.valid[offset..end].iter().all(|v| *v) && self.bytes[offset..end] == *bytes {
            return false;
        }
        self.bytes[offset..end].copy_from_slice(bytes);
        self.valid[offset..end].fill(true);
        true
    }
}

/// Per-slot value and upload history, keyed by the resource it affects.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum ChangeKey {
    Buffer(wgpu::Buffer),
    CameraSelection(wgpu::Buffer),
    LightTree(wgpu::Buffer),
    Subdivision(usize, usize),
    SampleTime(usize),
    ComputeTime(usize),
    DiscreteTime(usize),
    /// Inputs of an install-shared published render image, in view-table order.
    Capture(usize),
    ImageClock(String),
    /// The frame slot's acceleration structures have had their first build.
    RayStructures,
    /// Logical displacement evaluation, independent of its packed control page.
    Displacement(usize),
    DisplacedPositions(usize),
    ComputedGeometry(usize),
    ComputedPositions(usize),
    MaterialAttributes,
    MaterialColumn(usize),
    /// Source publication consumed by an installed shared field index.
    FieldIndex(wgpu::Buffer),
    /// Logical named stream publication, independent of packed GPU page identity.
    Stream(String),
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum LightingKey {
    /// The irradiance volume's lighting inputs (probes::key).
    Probes,
    ProbeGeometry,
    EnvironmentInputs,
    EnvironmentRadiance,
    Fibre,
}
#[derive(Default)]
pub(crate) struct ChangeTracker {
    pub(crate) placement_revision: u64,
    /// Publication generation of render-backed images under the submission lock.
    /// Image-dependent lighting consumes seed publications from this frame.
    pub(crate) image_revision: u64,
    history: std::collections::HashMap<ChangeKey, UploadCache>,
    sample: u64,
    pub(crate) bytes: Vec<u8>,
    pub(crate) uploads: Vec<Upload>,
}
/// Submission-ordered history of retained lighting,
/// independent of upload slots.
#[derive(Default)]
pub(crate) struct LightingTracker {
    history: std::collections::HashMap<LightingKey, UploadCache>,
    sample: u64,
}
impl LightingTracker {
    pub(crate) fn invalidate_probes(&mut self) {
        self.history.remove(&LightingKey::Probes);
        self.history.remove(&LightingKey::ProbeGeometry);
    }

    /// Last input change, retained after the producer's dirty sample. Consumers
    /// may skip submissions and must still observe an intervening update.
    pub(crate) fn revision(&self, key: LightingKey) -> u64 {
        self.history.get(&key).map_or(0, |h| h.stamp)
    }
    pub(crate) fn dirty(&self, key: LightingKey) -> bool {
        self.history
            .get(&key)
            .is_some_and(|h| h.stamp == self.sample)
    }
    pub(crate) fn changed(&mut self, key: LightingKey, bytes: &[u8]) -> bool {
        profile_span!(_profile, "track_change");
        let history = self.history.entry(key).or_default();
        let resized = history.bytes.len() != bytes.len();
        let changed = history.changed(0, bytes);
        history.bytes.truncate(bytes.len());
        history.valid.truncate(bytes.len());
        let changed = resized || changed;
        if changed {
            history.stamp = self.sample;
        }
        changed
    }

}
impl ChangeTracker {
    /// Keep only the inputs whose GPU destinations survive a recording eviction.
    /// Discarding other buffer keys also releases their old view handles.
    pub(crate) fn retain_ray_history(&mut self, objects: &wgpu::Buffer) {
        self.history.retain(|key, _| matches!(key, ChangeKey::RayStructures)
            || matches!(key, ChangeKey::Buffer(buffer) if buffer == objects));
        self.bytes.clear();
        self.uploads.clear();
    }
    pub(crate) fn observe_stream(&mut self, resource: &str, version: u64) {
        self.changed(ChangeKey::Stream(resource.into()), &version.to_le_bytes());
    }
    pub(crate) fn stream_version(&self, resource: &str) -> u64 {
        self.history.get(&ChangeKey::Stream(resource.into())).map_or(0, |h| u64::from_le_bytes(h.bytes.as_slice().try_into().expect("stream version bytes")))
    }

    pub(crate) fn invalidate_captures(&mut self) {
        self.history.retain(|key, _| !matches!(key, ChangeKey::Capture(_)));
    }

    /// Canonical value history, independent of the frame slot's buffer identity.
    /// Validity matters: the first write of zero can replace a nonzero installed value.
    pub(crate) fn append_buffer_inputs(&self, buffer: &wgpu::Buffer, inputs: &mut Vec<u8>) {
        self.append_inputs(ChangeKey::Buffer(buffer.clone()), inputs);
    }
    pub(crate) fn append_inputs(&self, key: ChangeKey, inputs: &mut Vec<u8>) {
        let history = self.history.get(&key);
        inputs.extend_from_slice(&(history.map_or(0, |h| h.bytes.len()) as u64).to_le_bytes());
        if let Some(history) = history {
            inputs.extend_from_slice(&history.bytes);
            inputs.extend(history.valid.iter().map(|v| u8::from(*v)));
        }
    }
    /// Publish a derived resource without inventing a second dirty mechanism.
    pub(crate) fn publish(&mut self, key: ChangeKey) {
        self.history.entry(key).or_default().stamp = self.sample;
    }
    /// Record an external GPU replacement at its encoding boundary. Forget both
    /// the destination CPU byte histories and the derived input key: unchanged
    /// inputs must recompute, and equal old CPU bytes must be uploaded again.
    /// Already queued uploads are retained; callers encode them before replacement.
    pub(crate) fn replace_gpu(&mut self, buffers: &[&wgpu::Buffer], dependent: ChangeKey) {
        for buffer in buffers {
            self.history.remove(&ChangeKey::Buffer((*buffer).clone()));
        }
        self.history.remove(&dependent);
    }
    /// Last sample that published CPU bytes to this buffer. A consumer also
    /// checks dirty() when it may run again inside that same sample.
    pub(crate) fn buffer_revision(&self, buffer: &wgpu::Buffer) -> u64 {
        self.history.get(&ChangeKey::Buffer(buffer.clone())).map_or(0, |history| history.stamp)
    }
    pub(crate) fn buffer_dirty(&self, buffer: &wgpu::Buffer) -> bool {
        self.dirty(ChangeKey::Buffer(buffer.clone()))
    }
    pub(crate) fn dirty(&self, key: ChangeKey) -> bool { self.history.get(&key).is_some_and(|h|h.stamp == self.sample) }
    pub(crate) fn changed(&mut self, key: ChangeKey, bytes: &[u8]) -> bool {
        profile_span!(_profile, "track_change");
        let history = self.history.entry(key).or_default();
        let resized = history.bytes.len() != bytes.len();
        let changed = history.changed(0, bytes);
        history.bytes.truncate(bytes.len());
        history.valid.truncate(bytes.len());
        let changed = resized || changed;
        if changed {
            history.stamp = self.sample;
        }
        changed
    }
    /// Recompute CPU geometry only when its deformation inputs change. The same
    /// history owns both derived bounds and the normal dirty/upload mechanism.
    pub(crate) fn derive_upload<T: Pod + Copy>(
        &mut self, key: ChangeKey, input: &[u8], buffer: &wgpu::Buffer, offset: u64,
        calculate: impl FnOnce() -> Result<(T, Vec<u8>)>,
    ) -> Result<T> {
        if let Some(history) = self.history.get(&key) {
            if history.bytes.len() == input.len() + std::mem::size_of::<T>() && history.bytes[..input.len()] == *input {
                return Ok(bytemuck::pod_read_unaligned(&history.bytes[input.len()..]));
            }
        }
        let (value, bytes) = calculate()?;
        self.upload(buffer, offset, &bytes);
        let history = self.history.entry(key).or_default();
        history.bytes.clear(); history.bytes.extend_from_slice(input); history.bytes.extend_from_slice(bytemuck::bytes_of(&value));
        Ok(value)
    }
    pub(crate) fn upload(&mut self, buffer: &wgpu::Buffer, offset: u64, bytes: &[u8]) {
        profile_span!(_profile, "track_upload");
        let history = self
            .history
            .entry(ChangeKey::Buffer(buffer.clone()))
            .or_default();
        if !history.changed(offset as usize, bytes) {
            return;
        }
        history.stamp = self.sample;
        self.record_upload(buffer, offset, bytes);
    }
    fn record_upload(&mut self, buffer: &wgpu::Buffer, offset: u64, bytes: &[u8]) {
        profile_span!(_profile, "pack_upload");
        #[cfg(feature = "frame-profile")]
        crate::frame_profile::count("dirty_upload_bytes", bytes.len() as u64);
        let source = self.bytes.len() as u64;
        self.bytes.extend_from_slice(bytes);
        // Preserve field/joint granularity without a separate dirty-range detector.
        if let Some((last_buffer, last_offset, last_source, size)) = self.uploads.last_mut() {
            if last_buffer == buffer
                && *last_offset + *size == offset
                && *last_source + *size == source
            {
                *size += bytes.len() as u64;
                return;
            }
        }
        self.uploads
            .push((buffer.clone(), offset, source, bytes.len() as u64));
    }
}

/// One evaluation per keyed node/time; static nodes borrow installed values.
pub(crate) struct SampledNodes<'a> {
    nodes: &'a [EvalNode],
    values: &'a [Option<EvalNode>],
}
pub(crate) fn sample_storage(nodes: &[EvalNode]) -> Vec<Option<EvalNode>> {
    nodes
        .iter()
        .map(|n| {
            n.dynamic.then(|| EvalNode { camera: n.camera.clone(),
                node: n.node.clone(),
                parent: n.parent,
                world: n.world,
                dynamic: n.dynamic,
                embedded: n.embedded,
            })
        })
        .collect()
}
pub(crate) struct OffsetValues {
    samples: animation::Samples,
    nodes: Vec<Option<EvalNode>>,
    worlds: Vec<Mat4>,
    groups: Vec<OffsetGroup>,
}
#[derive(Default)]
struct OffsetGroup {
    offset: f32,
    objects: Vec<usize>,
    deforms: Vec<usize>,
    nodes: Vec<usize>,
    world_order: Vec<usize>,
    assets: Vec<usize>,
}
pub(crate) fn offset_storage(s: &InstalledScene) -> OffsetValues {
    let mut groups = std::collections::BTreeMap::<u32, OffsetGroup>::new();
    for &object in &s.dynamic_objects {
        let root = &s.roots[object];
        let bits = if root.time_offset == 0. {
            0
        } else {
            root.time_offset.to_bits()
        };
        let group = groups.entry(bits).or_default();
        group.offset = root.time_offset;
        group.objects.push(object);
        group.nodes.push(root.node);
    }
    for (index, draw) in s.draws.iter().enumerate() {
        let Some(deform) = draw.deform.as_ref().filter(|deform| deform.leader == index) else {
            continue;
        };
        let root = &s.roots[draw.object as usize];
        let bits = if root.time_offset == 0. {
            0
        } else {
            root.time_offset.to_bits()
        };
        let group = groups.entry(bits).or_default();
        group.offset = root.time_offset;
        group.deforms.push(index);
        group.nodes.push(root.node);
        if let Some(native) = &deform.native {
            group.nodes.push(native.mesh);
            group
                .nodes
                .extend(native.joints.iter().map(|(joint, _)| *joint));
        } else {
            group
                .assets
                .push(deform.asset.expect("glTF deformation asset"));
        }
    }
    for group in groups.values_mut() {
        let mut required = std::collections::BTreeSet::new();
        let mut pending = std::mem::take(&mut group.nodes);
        while let Some(index) = pending.pop() {
            if !required.insert(index) {
                continue;
            }
            if let Some(target) = s.targets.aims[index].and_then(crate::targets::Aim::node) { pending.push(target); }
            if let Some(parent) = s.nodes[index].parent {
                pending.push(parent);
            }
            if let Some((asset, _, root, _)) = s.nodes[index].embedded {
                group.assets.push(asset);
                pending.push(root);
            }
        }
        group.nodes = required.into_iter().collect();
        group.world_order = s.targets.order.iter().copied().filter(|i| group.nodes.binary_search(i).is_ok()).collect();
        group.assets.sort_unstable();
        group.assets.dedup();
    }
    OffsetValues {
        samples: s.animation.samples(),
        nodes: sample_storage(&s.nodes),
        worlds: s.nodes.iter().map(|n| n.world).collect(),
        groups: groups.into_values().collect(),
    }
}
pub(crate) fn sample_frame(
    s: &InstalledScene,
    request: impl Into<vork_time::Request>,
    samples: &mut animation::Samples,
    values: &mut [Option<EvalNode>],
    worlds: &mut [Mat4],
) -> Result<()> {
        profile_span!(_profile, "scene_evaluation");
    let request=request.into();
    let time=request.numeric()?;
    sample_pose(&s.nodes, &s.assets, &s.animation, &s.targets, &s.environment, &request, samples, values, worlds)?;
    let active_camera=s.animation.camera_at_request(&request,s.camera)?;
    for (index,node) in values.iter_mut().enumerate().filter_map(|(index,node)|node.as_mut().map(|node|(index,node))) {
        // Rule 1: an absent camera's program state is not evaluated.
        if s.animation.absent(samples,index) {continue;}
        if let Some(camera)=&mut node.camera {
            camera.time=time;
            for (binding,value) in s.animation.material_values(samples,0.) {
                if binding.slot>=camera.offset && binding.slot<camera.offset+camera.words.len() {
                    let words=programs::sample_words(value,binding);let start=binding.slot-camera.offset;
                    camera.words[start..start+words.len()].copy_from_slice(&words);
                }
            }
            let selected=std::iter::once(s.render.clone()).filter(|_|index==active_camera)
                .chain(s.scene.images.values().filter(|image|image.render.as_ref().is_some_and(|render|
                    render.camera.as_deref().map_or(index==s.camera,|name|name==node.node.name))).filter_map(|image|image.resolved_render(&s.render)));
            for render in selected {
                let calibration=camera.calibration(render.resolution[0] as f32/render.resolution[1] as f32)?;
                ensure!(render.settings.integrator.is_path() || calibration.is_some(),"raster camera must retain projective calibration at every sample");
            }
        }
    }
    Ok(())
}

/// Shared installation and frame pose evaluation; authoring defaults stay untouched.
pub(crate) fn sample_pose(
    originals: &[EvalNode],
    assets: &[Asset],
    animation: &animation::Plan,
    targets: &crate::targets::Plan,
    _environment: &environment::Environment,
    request: impl Into<vork_time::Request>,
    samples: &mut animation::Samples,
    values: &mut [Option<EvalNode>],
    worlds: &mut [Mat4],
) -> Result<()> {
    sample_pose_inputs(originals, assets, animation, targets, request, samples, values, worlds)
}

/// Geometry stages use the same authored pose evaluation at canonical tick times.
pub(crate) fn sample_pose_inputs(
    originals: &[EvalNode],
    assets: &[Asset],
    animation: &animation::Plan,
    targets: &crate::targets::Plan,
    request: impl Into<vork_time::Request>,
    samples: &mut animation::Samples,
    values: &mut [Option<EvalNode>],
    worlds: &mut [Mat4],
) -> Result<()> {
    let request=request.into();
    let time=request.numeric()?;
    animation.sample_request_into(time, &request, samples)?;
    apply_pose_samples(originals, assets, animation, targets, time, samples, values, worlds)
}

/// Apply an already sampled plan, including the frame evaluator's instance offsets.
pub(crate) fn apply_pose_samples(
    originals: &[EvalNode],
    assets: &[Asset],
    animation: &animation::Plan,
    targets: &crate::targets::Plan,
    time: f32,
    samples: &animation::Samples,
    values: &mut [Option<EvalNode>],
    worlds: &mut [Mat4],
) -> Result<()> {
    animation.apply_nodes(samples, values);
    let nodes = SampledNodes::new(originals, values);
    let mut poses = std::collections::HashMap::new();
    for position in 0..originals.len() {
        let index = targets.order.get(position).copied().unwrap_or(position);
        let original = &originals[index];
        let node = &nodes[index];
        // Rule 1: an absent node, an absent asset's imported nodes included, is not evaluated.
        if animation.absent(samples, index) { continue; }
        if let Some((asset, local, root, _)) = original.embedded {
            let pose = poses.entry(asset).or_insert_with(|| assets[asset].pose_with_plan(samples, animation));
            worlds[index] = worlds[root] * pose.worlds[local];
        } else if original.dynamic {
            let world = node_world(&node.node, node.parent.map_or(Mat4::IDENTITY, |p| worlds[p]))?;
            worlds[index] = targets.aim(index, world, worlds, &node.node, |target| animation.absent(samples, target)).map_err(|error| anyhow::anyhow!("{error} at sample time {time}"))?;
        }
    }
    for (index, original) in originals.iter().enumerate() {
        if let Some((asset, local, _, _)) = original.embedded {
            // An absent imported node keeps the visibility its absence gave it: off.
            if animation.absent(samples, index) { continue; }
            if let Some(node) = values[index].as_mut() {
                node.node.visibility = lights::visibility_from_mask(poses[&asset].visibility[local]);
            }
        }
    }
    for &(camera, target) in &targets.focus {
        if animation.absent(samples, camera) { continue; }
        ensure!(!animation.absent(samples, target), "node {:?}: focus_object {:?} is absent (active); focus on a node that exists at sample time {time}",
            originals[camera].node.name, originals[target].node.name);
        if let Some(state) = &mut values[camera].as_mut().expect("focus camera is dynamic").camera {
            state.focus = crate::targets::focus_distance(worlds[camera], worlds[target]);
        }
    }
    Ok(())
}
impl<'a> SampledNodes<'a> {
    fn new(nodes: &'a [EvalNode], values: &'a [Option<EvalNode>]) -> Self {
        Self { nodes, values }
    }
}
impl std::ops::Index<usize> for SampledNodes<'_> {
    type Output = EvalNode;
    fn index(&self, id: usize) -> &EvalNode {
        self.values[id].as_ref().unwrap_or(&self.nodes[id])
    }
}

impl FrameSlot {
    /// Evaluates this sample's time-varying values. Returns the packed upload bytes,
    /// their destinations, the camera view matrix and the frame uniform.
    pub(crate) fn prepare_values(
        &mut self,
        s: &InstalledScene,
        r: &Renderer,
        request: impl Into<vork_time::Request>,
        frame_time: f32,
        shadow_changes: &mut LightingTracker,
    ) -> Result<(Vec<u8>, Vec<Upload>, Mat4, Uniform)> {
        profile_span!(_profile, "prepare_values_inclusive");
        self.sample_request=request.into();
        let time=self.sample_request.numeric()?;
        let exact=!s.footage.is_empty() || s.compute.is_some() || s.animation.has_camera_tracks() || s.environment.has_media() || s.animation.has_discrete_tracks() || s.animation.has_clocks();
        self.previous_request=if self.exposure_cut {self.sample_request.clone()}
            else if exact {self.sample_request.previous(s.render.fps)?}
            else {vork_time::Request::Numeric((time-1./s.render.fps.as_f32()).max(0.))};
        // Queue writes would all execute before a batched submission. Pack each
        // sample's values and copy them immediately before its rendering instead.
        let changes = &mut self.changes;
        changes.sample += 1;
        shadow_changes.sample += 1;
        changes.bytes.clear();
        changes.uploads.clear();
        let camera_index = s.animation.camera_at_request(&self.sample_request,s.camera)?;
        let sample_key=[time.to_bits(),camera_index as u32,0,0];
        let mut discrete_key=Vec::new();
        s.animation.append_discrete_key(&self.sample_request,&mut discrete_key)?;
        if s.animation.has_discrete_tracks() {
            for offset in s.animation.material_offsets().iter().copied().chain(self.offset_values.groups.iter().map(|group|group.offset)).filter(|&offset|offset!=0.) {
                s.animation.append_discrete_offset_key(&self.sample_request,offset,&mut discrete_key)?;
            }
        }
        let discrete_changed=!discrete_key.is_empty() && changes.changed(ChangeKey::DiscreteTime(0),&discrete_key);
        let time_changed = changes.changed(ChangeKey::SampleTime(0), bytemuck::cast_slice(&sample_key)) | discrete_changed;
        // The exposure holds the submission lock for retained lighting and simulation.
        // Nominal controls may use another view.
        self.reuse_values = !time_changed && self.view.sample_index > 0 && !self.view.nominal_sample;
        if time_changed {
            sample_frame(
                s,
                &self.sample_request,
                &mut self.animation_values,
                &mut self.sampled_nodes,
                &mut self.worlds,
            )?;
            s.animation
                .apply_render(&mut self.animation_values, &s.render);
        }
        // Rule 1: an absent camera has no view to render through.
        ensure!(!s.animation.absent(&self.animation_values, camera_index),
            "render.camera: camera {:?} is absent (active); render through a camera that exists at sample time {time}", s.nodes[camera_index].node.name);
        let render = self.animation_values.render.as_ref().unwrap();
        let indirect = s.plan.effects.indirect && render.settings.integrator.indirect().is_some();
        let nodes = SampledNodes::new(&s.nodes, &self.sampled_nodes);
        let cut=self.exposure_cut;
        let previous_time=self.previous_request.numeric()?;
        let next_time=time;
        let next_changed=changes.changed(ChangeKey::SampleTime(2), bytemuck::cast_slice(&sample_key)) | discrete_changed;
        if next_changed {
            sample_frame(s,&self.sample_request,&mut self.next_animation_values,&mut self.next_nodes,&mut self.next_worlds)?;
        }
        // Spatial AA/aperture samples at one time reuse the evaluated pose and render values.
        let previous_key = [previous_time.to_bits(), u32::from(cut),s.animation.camera_at_request(&self.previous_request,s.camera)? as u32,
            0,0];
        discrete_key.clear();
        s.animation.append_discrete_key(&self.previous_request,&mut discrete_key)?;
        if s.animation.has_discrete_tracks() {
            for offset in s.animation.material_offsets().iter().copied().chain(self.offset_values.groups.iter().map(|group|group.offset)).filter(|&offset|offset!=0.) {
                s.animation.append_discrete_offset_key(&self.previous_request,offset,&mut discrete_key)?;
            }
        }
        let previous_discrete_changed=!discrete_key.is_empty() && changes.changed(ChangeKey::DiscreteTime(1),&discrete_key);
        let previous_changed=changes.changed(
            ChangeKey::SampleTime(1), bytemuck::cast_slice(&previous_key)) | previous_discrete_changed;
        // Clamped middle times may repeat while either endpoint still changes.
        self.reuse_values &= !next_changed && !previous_changed;
        if let Some(compute)=&s.compute {
            let current=&self.sample_request;
            for (index,request) in [current,&self.previous_request].into_iter().enumerate() {
                self.reuse_values &= !changes.changed(ChangeKey::ComputeTime(index),bytemuck::cast_slice(&compute.request_key(request)?));
            }
        }
        if previous_changed {
            if cut {
                self.previous_animation_values
                    .clone_from(&self.animation_values);
                for (previous, current) in self.previous_nodes.iter_mut().zip(&self.sampled_nodes) {
                    if let (Some(previous), Some(current)) = (previous, current) {
                        previous.node.clone_from(&current.node);
                        previous.camera.clone_from(&current.camera);
                    }
                }
                self.previous_worlds.clone_from(&self.worlds);
            } else {
                sample_frame(
                    s,
                    &self.previous_request,
                    &mut self.previous_animation_values,
                    &mut self.previous_nodes,
                    &mut self.previous_worlds,
                )?;
            }
        }
        let previous_nodes = SampledNodes::new(&s.nodes, &self.previous_nodes);
        let previous_worlds = &mut self.previous_worlds;
        let camera = &nodes[camera_index];
        let previous_camera = &previous_nodes[camera_index];
        let (near, far, ortho_height) = camera_clip(camera, self.view.width as f32 / self.view.height as f32);
        let eye = self.worlds[camera_index].transform_point3(Vec3::ZERO);
        let view = node_view(camera_index, &nodes, &self.worlds)?;
        let aspect = self.view.width as f32 / self.view.height as f32;
        let physical = physical_camera(camera, aspect);
        let projection = camera_projection(camera, aspect);
        let nominal_view = view;
        let nominal_projection = projection;
        let nominal_vp = projection * view;
        // Decoupled visibility already samples the pixel and lens within a step.
        let camera_sample_count = if self.view.nominal_sample || (s.plan.decoupled && self.view.visibility_samples > 1) {
            1
        } else {
            self.view.sample_count
        };
        let projection = crate::sampling::projection(
                projection,
                self.view.internal,
                self.view.sample_index,
                camera_sample_count,
        );
        let sampled_lens = !self.view.nominal_sample;
        let (projection, view) = crate::sampling::lens(
            projection,
            view,
            physical.as_ref().filter(|_| sampled_lens),
            self.view.sample_index,
            camera_sample_count,
        );
        let camera_sample = camera_sample(self.view.internal, physical.as_ref().filter(|_| sampled_lens),
            self.view.sample_index, camera_sample_count);
        let eye = if sampled_lens && physical.as_ref().is_some_and(|p| p.radius>0.) {
            view.inverse().transform_point3(Vec3::ZERO)
        } else {
            eye
        };
        let vp = projection * view;
        let previous_view = node_view(camera_index, &previous_nodes, previous_worlds)?;
        let previous_projection = camera_projection(previous_camera, aspect);
        let previous_projection = crate::sampling::projection(
                previous_projection,
                self.view.internal,
                self.view.sample_index,
                camera_sample_count,
        );
        let previous_physical = physical_camera(previous_camera, aspect);
        let (previous_projection, previous_view) = crate::sampling::lens(
            previous_projection,
            previous_view,
            previous_physical.as_ref().filter(|_| sampled_lens),
            self.view.sample_index,
            camera_sample_count,
        );
        let previous_vp = previous_projection * previous_view;
        let next_nodes=SampledNodes::new(&s.nodes,&self.next_nodes);
        let next_camera=&next_nodes[camera_index];
        let next_projection=crate::sampling::projection(camera_projection(next_camera,aspect),self.view.internal,self.view.sample_index,camera_sample_count);
        let next_physical=physical_camera(next_camera,aspect);
        let (next_projection,next_view)=crate::sampling::lens(next_projection,node_view(camera_index,&next_nodes,&self.next_worlds)?,next_physical.as_ref().filter(|_|sampled_lens),self.view.sample_index,camera_sample_count);
        let next_vp=next_projection*next_view;
        if self.view.nominal_sample || self.view.sample_index + 1 == self.view.sample_count {
            self.camera_data = crate::CameraRecord::new(
                nominal_view,
                nominal_projection,
                render.resolution,
                ortho_height,
                far,
            );
            if calibration(camera,aspect).is_none() {self.camera_data.k=None;self.camera_data.dist=None;self.camera_data.depth_unit_m=1.;}
        }
        let lens_height=self.view.internal[1] as f32;
        let lens_scale=|active| physical.as_ref().filter(|_| active).map_or(0., |p| p.circle_scale(lens_height));
        let environment = environment_node::sample(s.environment.node, &nodes, &self.worlds)?;
        let environment_visibility = s.environment.node.map_or(constants::RAY_VISIBILITY_ALL, |node| buffers::node_visibility_mask(&nodes, node));
        let environment_constant = environment.constant(s.environment.settings.has_radiance());
        // Camera rays see the world only when it is visible to them: a hidden
        // world leaves the camera background (constant and image) black.
        let camera_world = environment_visibility & constants::RAY_CAMERA != 0;
        let background = match &render.background {
            vork_scene::Background::Color(color) => *color,
            vork_scene::Background::Mode(vork_scene::BackgroundMode::Environment) => {
                if camera_world { environment_constant } else { [0.; 3] }
            }
            vork_scene::Background::Mode(vork_scene::BackgroundMode::Transparent) => [0.; 3],
            // A 2D canvas's background; a 3D world refuses it when the scene loads.
            vork_scene::Background::Css(_) => [0.; 3],
        };
        let environment_links = s.environment.node.map(|node| s.links.illuminates(node));
        // Jittered temporal colour never writes controls; its nominal pass does.
        let controls = true;
        let composite_has_shutter = self.view.composite_steps>1 && !self.view.nominal_sample;
        let composite_steps = if self.view.nominal_sample {1} else {self.view.composite_steps.max(1)};
        let step_iterations = self.view.sample_count as u32 / composite_steps;
        let composite_samples = self.view.visibility_samples * step_iterations;
        let (composite_optics,composite_sensor) = composite_camera(camera,aspect,sampled_lens,composite_samples);
        let uniform = Uniform {
            vp,
            previous_vp,
            next_vp,
            light_vp: Mat4::IDENTITY,
            inverse_vp: vp.inverse(),
            view,
            ao: ao_uniform(indirect),
            projection: [
                projection.y_axis.y,
                projection.x_axis.x,
                reflection_mode(render),
                ((u32::from(self.view.write_color && !self.view.nominal_sample)
                    * constants::OUTPUT_GPU_COLOR)
                    // 16-bit colour also publishes the 8-bit plane that exposure metering reads.
                    | (u32::from((self.view.readback_outputs.color || self.view.readback_outputs.color16) && !self.view.nominal_sample)
                        * constants::OUTPUT_COLOR_PLANE)
                    | (u32::from(self.view.readback_outputs.color16 && !self.view.nominal_sample)
                        * vork_slang::registry::OUTPUT_COLOR_PLANE16)
                    | (u32::from(controls && self.view.readback_outputs.depth_ids)
                        * constants::OUTPUT_DEPTH_IDS)
                    | (u32::from(controls && self.view.readback_outputs.normals)
                        * constants::OUTPUT_NORMALS)
                    | (u32::from(controls && self.view.readback_outputs.motion)
                        * constants::OUTPUT_MOTION)
                    | (u32::from(
                        self.view.readback_outputs.float.color && !self.view.nominal_sample,
                    ) * constants::OUTPUT_FLOAT_COLOR)
                    | (u32::from(controls && self.view.readback_outputs.float.depth)
                        * constants::OUTPUT_FLOAT_DEPTH)
                    | (u32::from(controls && self.view.readback_outputs.float.normals)
                        * constants::OUTPUT_FLOAT_NORMALS)
                    | (u32::from(controls && self.view.readback_outputs.float.motion)
                        * constants::OUTPUT_FLOAT_MOTION)
                    | (crate::compositor::channel_flags(s, &render.compositor))) as f32,
            ],
            aperture: aperture_shape(camera, aspect),
            lens: [physical.as_ref().filter(|_| sampled_lens).map_or(0., crate::sampling::aperture_radius),
                physical.as_ref().map_or(0., |p| p.focus_distance),
                2. / self.view.internal[0] as f32, 2. / self.view.internal[1] as f32],
            temporal: if self.view.nominal_sample {
                [0., 1., self.view.sample_count as f32, if render.settings.integrator.is_path() {vork_slang::registry::DENOISE_GUIDES_PATH as f32} else {0.}]
            } else {
                [self.view.sample_index as f32, self.view.sample_count as f32,
                    self.view.sample_count as f32, if render.settings.integrator.is_path() {vork_slang::registry::DENOISE_GUIDES_PATH as f32} else {0.}]

            },
            environment: [
                if s.environment.settings.has_radiance() {
                    environment.intensity
                } else {
                    0.
                },
                s.environment.initial_filter_scale(),
                5.,
                if render.background.is_environment() && camera_world {
                    1.
                } else {
                    0.
                },
            ],
            environment_rotation: environment.rotation,
            environment_color: [
                environment.color[0],
                environment.color[1],
                environment.color[2],
                0.,
            ],
            environment_mean: [
                s.environment.initial_mean()[0],
                s.environment.initial_mean()[1],
                s.environment.initial_mean()[2],
                f32::from(s.environment.lighting_is_black),
            ],
            scene: [time, self.nominal_frame as f32, f32::from(cut), render.fps.as_f32()],
            shadow_links: [0, 1, 0, 0],
            sampling: {
                // Word meanings: Frame in vork_prelude.slang. Outside decoupled colour,
                // control resolve writes the ambient controls.
                let colour = !self.view.nominal_sample && self.view.color_enabled;
                let mut words=s.sampling.uniform(if colour {self.view.sample_count * self.view.visibility_samples as usize} else {1}, if colour {self.view.visibility_samples as usize} else {1});
                words[3]=if colour && s.plan.decoupled {self.view.visibility_samples} else {0};
                words
            },
            // The shadow and ray sequences cover the render's samples: decoupled slots take
            // map taps and emitter rays for their covered samples within one exposure sample.
            integrator: [0, 0, if self.view.nominal_sample || !self.view.color_enabled {1} else {(self.view.sample_count * self.view.visibility_samples as usize) as u32}, s.shadows().ray_sequence_word(if self.view.nominal_sample || !self.view.color_enabled {1} else {self.view.sample_count * self.view.visibility_samples as usize}) | u32::from(render.settings.integrator.subsurface_method()==Some(vork_scene::v2::SubsurfaceMethod::Thickness))*vork_slang::registry::INTEGRATOR_SUBSURFACE_THICKNESS | u32::from(render.settings.previs)*vork_slang::registry::INTEGRATOR_PREVIS_CENTER],
            environment_links: [
                environment_visibility
                    | if environment_links.is_none_or(|link| link.all) {
                        constants::LIGHT_LINK_ALL
                    } else {
                        0
                    },
                environment_links.map_or(0, |link| link.mask),
                match &render.settings.integrator {
                    vork_scene::Integrator::Raster(_) => {
                        let limit=crate::constants::RASTER_TRANSMISSION_BOUNCES;
                        #[cfg(test)]
                        let limit=r.reference_transport_bounces.unwrap_or(limit);
                        limit
                    },
                    vork_scene::Integrator::Path(_) => 0,
                },
                transmission_word(render),
            ],
            medium: [s.medium_base,s.rays.as_ref().map_or(0,|r|r.emitters),s.environment_base,u32::from(environment_visibility & constants::RAY_VOLUME!=0)],
            display: [f32::from_bits(render.settings.seed),render.resolution[0] as f32,render.resolution[1] as f32,self.view.post.display_base.unwrap_or(0) as f32],
            post: [if s.has_catchers {1.} else {post_mode(render)},
                frame_time,
                // The display stage's dither switch: zero dithers; only tests turn it off (Render::dither).
                if render.dither {0.} else {1.},
                lens_scale(sampled_lens && s.plan.decoupled && self.view.visibility_samples > 1)],
            composite_projection: projection,
            composite_optics,
            composite_sensor,
            composite_shutter: if self.view.nominal_sample {[time;4]} else {self.view.composite_shutter},
            composite_step: [if composite_has_shutter {self.view.sample_index as f32} else {0.},composite_steps as f32,render.settings.samples as f32,step_iterations as f32],
            camera_sample,
            camera_program: crate::camera_program::selection(s,camera_index,camera,aspect),
            field_geometry: {let mut fields=s.rays.as_ref().map_or([0;4],|r|r.implicit);fields[2]=u32::from(!s.implicit_fields.is_empty());fields[3]=0;fields},
            eye: eye.extend(1.).to_array(),
            background,
            light_count: 0.,
            radiance: [
                environment_constant[0],
                environment_constant[1],
                environment_constant[2],
                0.,
            ],
            params: [near, far, 0., 0.],
            flags: [
                if ortho_height.is_some() { 1. } else { 0. },
                0.,
                if render.background.transparent() {
                    1.
                } else {
                    0.
                },
                f32::from(indirect),
            ],
        };
        self.view.post.compositor_inputs.upload(s, frame_time, changes)?;
        if let Some(buffer) = &self.view.post.display_inputs {
            s.display_program.upload(buffer, frame_time, changes);
        }
        changes.upload(&self.view.uniform, 0, bytemuck::bytes_of(&uniform));
        if !self.reuse_values {
        // Animated material inputs write this frame's copy of their records; instances
        // with a time offset have their own record copies, sampled at time + offset.
        if let Some(frame) = &self.materials {
            let mut write = |values: &animation::Samples, offset: f32| {
                for (binding, value) in s.animation.material_values(values, offset) {
                    if binding.storage.is_some() {
                        let bytes = programs::sample_words(value, binding);
                        changes.upload(&frame.records, binding.slot as u64 * 4, bytemuck::cast_slice(&bytes));
                        continue;
                    }
                    let (crate::animation::Sample::Numeric(value), Some((at, width))) =
                        (value, materials::value_location(binding.slot, binding.parameter))
                    else {
                        continue;
                    };
                    changes.upload(&frame.records, at, bytemuck::cast_slice(&value[..width]));
                }
            };
            write(&self.animation_values, 0.);
            for &offset in s.animation.material_offsets() {
                let samples = &mut self.offset_values.samples;
                s.animation.sample_material_request(&self.sample_request,time,offset,samples)?;
                write(samples, offset);
            }
        }
        let mut local_bounds = vec![None; s.draws.len()];
        let mut displacement_spaces=if s.displacement.is_some() {vec![[Mat4::IDENTITY;3];s.draws.len()]} else {Vec::new()};
        let OffsetValues {
            samples: offset_samples,
            nodes: offset_nodes,
            worlds: offset_worlds,
            groups,
        } = &mut self.offset_values;
        for group in groups {
            for (pose,pose_time) in [time,previous_time,next_time].into_iter().enumerate() {
                let current=pose==0;
                let mut poses = std::collections::HashMap::new();
                if group.offset != 0. && (current || !cut) {
                    s.animation.sample_nodes_request(
                        if pose==1 {&self.previous_request}else{&self.sample_request},
                        pose_time, group.offset,
                        offset_samples,
                        &group.nodes,
                        &group.assets,
                    )?;
                    s.animation
                        .apply_nodes_subset(offset_samples, offset_nodes, &group.nodes);
                    let shifted_nodes = SampledNodes::new(&s.nodes, offset_nodes);
                    let order = if group.world_order.is_empty() { &group.nodes } else { &group.world_order };
                    for &index in order {
                        let node = &shifted_nodes[index];
                        if s.animation.absent(offset_samples, index) { continue; }
                        if let Some((asset, local, root, _)) = s.nodes[index].embedded {
                            let pose = poses
                                .entry(asset)
                                .or_insert_with(|| s.assets[asset].pose_sampled(offset_samples));
                            offset_worlds[index] = offset_worlds[root] * pose.worlds[local];
                        } else {
                            let world = node_world(&node.node, node.parent.map_or(Mat4::IDENTITY, |p| offset_worlds[p]))?;
                            offset_worlds[index] = s.targets.aim(index, world, offset_worlds, &node.node, |target| s.animation.absent(offset_samples, target)).map_err(|error| anyhow::anyhow!("{error} at sample time {}", pose_time + group.offset))?;
                        }
                    }
                }
                let (sampled, values, worlds) = if group.offset == 0. {
                    if current {
                        (
                            &self.animation_values,
                            self.sampled_nodes.as_slice(),
                            self.worlds.as_slice(),
                        )
                    } else if pose==1 {
                        (
                            &self.previous_animation_values,
                            self.previous_nodes.as_slice(),
                            self.previous_worlds.as_slice(),
                        )
                    } else {(&self.next_animation_values,self.next_nodes.as_slice(),self.next_worlds.as_slice())}
                } else {
                    (
                        &*offset_samples,
                        offset_nodes.as_slice(),
                        offset_worlds.as_slice(),
                    )
                };
                let sampled_nodes = SampledNodes::new(&s.nodes, values);
                for &object in &group.objects {
                    let root = &s.roots[object];
                    let mut world = root.world(worlds);
                    // Rule 1: an absent object is hidden without evaluating its asset's pose.
                    let absent = s.animation.absent(sampled, root.node);
                    let hidden_scale = absent || buffers::node_collapsed(&sampled_nodes, root.node)
                        || root.imported_pose.is_some_and(|(asset, local)| {
                            poses.entry(asset).or_insert_with(|| s.assets[asset].pose_sampled(sampled)).collapsed[local]
                        });
                    if hidden_scale {
                        world.x_axis = glam::Vec4::ZERO; world.y_axis = glam::Vec4::ZERO; world.z_axis = glam::Vec4::ZERO;
                    }
                    let value = &mut self.object_values[object];
                    if current {
                        value.model = world;
                        value.metadata[1] = if hidden_scale {0} else {
                            buffers::node_visibility_mask(&sampled_nodes, root.node)
                                & if let Some((asset, local)) = root.imported_pose {
                                    poses.entry(asset).or_insert_with(|| s.assets[asset].pose_sampled(sampled)).visibility[local]
                                } else { root.imported_visibility }
                        };
                        value.metadata[1] |= buffers::node_matte_flags(&sampled_nodes,root.node);
                        if value.metadata[1]!=0 {value.metadata[1] |= s.objects[object].metadata[1]&vork_slang::registry::OBJECT_IMPLICIT;}
                    } else if pose==1 {
                        value.previous_model = world;
                    } else {value.next_model=world;}
                }
                let offset = group.offset;
                let offset_bits = offset.to_bits();
                let mut palettes = std::collections::HashMap::new();
                for &draw_id in &group.deforms {
                    let draw = &s.draws[draw_id];
                    // Rule 1: an absent object's deformation is not evaluated; its draw is hidden.
                    if s.animation.absent(sampled, s.roots[draw.object as usize].node) { continue; }
                    let deform = draw.deform.as_ref().unwrap();
                    let key = if let Some(native) = &deform.native {
                        (true, native.mesh, 0, None, offset_bits)
                    } else {
                        (
                            false,
                            deform.asset.expect("glTF deformation asset"),
                            deform.node,
                            deform.skin,
                            offset_bits,
                        )
                    };
                    let (matrices, weights) = palettes.entry(key).or_insert_with(|| {
                        if let Some(native) = &deform.native {
                            let nodes = SampledNodes::new(&s.nodes, values);
                            let NodeKind::Mesh { morph_weights, .. } =
                                &nodes[native.mesh].node.kind
                            else {
                                unreachable!("native deformation belongs to a mesh node")
                            };
                            native.palette(worlds, morph_weights)
                        } else {
                            let asset = deform.asset.expect("glTF deformation asset");
                            let pose = poses
                                .entry(asset)
                                .or_insert_with(|| s.assets[asset].pose_sampled(sampled));
                            (
                                s.assets[asset].matrices(pose, deform.node, deform.skin),
                                pose.weights[deform.node].clone(),
                            )
                        }
                    });
                    if !displacement_spaces.is_empty() && deform.native.is_none() {
                        displacement_spaces[deform.job][pose]=poses[&deform.asset.unwrap()].worlds[deform.node];
                    }
                    let bounds = self.geometry_chain.upload_pose(s,draw_id,pose,matrices,weights,changes)?;
                    if current {local_bounds[deform.job]=bounds;}
                }
            }
        }
        let displacement_bounds = self.geometry_chain.evaluate(r, s, &self.geometry_bounds, changes,
            &self.sample_request, &self.previous_request, &mut self.simulation_checkpoint,
            self.materials.as_ref(), &self.object_values, &displacement_spaces, &self.queries,
            time, previous_time, next_time)?;
        for &object in &s.dynamic_objects {
            if let Some((compute, offset)) = self.compute.as_ref().and_then(|c| c.parent_offset(object).map(|offset| (c, offset))) {
                changes.upload(&compute.parents, offset, bytemuck::bytes_of(&self.object_values[object]));
            } else {
                changes.upload(&self.objects, (object * std::mem::size_of::<GpuObject>()) as u64,
                    bytemuck::bytes_of(&self.object_values[object].gpu()));
            }
        }
        let mut shadow_local_bounds=s.rays.as_ref().filter(|r|r.dynamic_shadow_bounds()).map(|_|Vec::with_capacity(s.draws.len()));
        for (index, draw) in s.draws.iter().enumerate() {
            let displaced=displacement_bounds.get(index).copied().flatten();
            let gpu_bounds=self.deforms[index].as_ref().is_some_and(|g|g.gpu_bounds.is_some());
            let evaluated=if gpu_bounds {None} else {displaced.or_else(||draw.deform.as_ref().and_then(|deform|local_bounds[deform.job]))};
            self.geometry_bounds[index]=displaced;
            if let Some(bounds)=&mut shadow_local_bounds {
                bounds.push(evaluated.unwrap_or_else(||if draw.evaluated() {(Vec3::splat(f32::NEG_INFINITY),Vec3::splat(f32::INFINITY))} else {draw.bounds}));
            }
            if let Some((lo,hi))=displaced {
                let bounds=[lo.x,lo.y,lo.z,hi.x,hi.y,hi.z,1.];
                let records=&self.materials.as_ref().unwrap().records;
                for object in draw.instances.clone() {
                    let at=(self.object_values[object as usize].metadata[2] as u64+6)*4;
                    changes.upload(records,at,bytemuck::cast_slice(&bounds));
                }
            } else if s.rays.is_some() {
                if let (Some((lo,hi)),Some(frame))=(evaluated,self.materials.as_ref()) {
                    // The same current geometry bounds supply ray-origin uncertainty.
                    let object=draw.instances.start as usize;
                    let offset=4*u64::from(self.object_values[object].metadata[2]+6);
                    changes.upload(&frame.records,offset,bytemuck::cast_slice(&[lo.x,lo.y,lo.z,hi.x,hi.y,hi.z]));
                }
            }

        }
        if let (Some(rays),Some(local))=(&self.rays,shadow_local_bounds) {rays.prepare_shadow_bounds(s,&local,changes);}
        {
            let mut lights = s.light_plan.evaluate(
                &nodes,
                &self.worlds,
                &s.links,
            )?;
            if s.rays.as_ref().is_some_and(|rays| rays.opaque) {
                for light in &mut lights {
                    let flags = &mut light.shadow_offsets[constants::LIGHT_RAY_WORD];
                    if *flags & constants::LIGHT_RAY_SHADOWED_BY_ALL != 0 {
                        *flags |= vork_slang::registry::LIGHT_RAY_OPAQUE;
                    }
                }
            }
            for (index,&id) in s.light_plan.nodes.iter().enumerate() {
                if let Some(&address)=s.emitter_profiles.get(&id) {
                    let source=node_view(id,&nodes,&self.worlds)?.inverse();
                    let mut rotation=source.to_scale_rotation_translation().1;
                    if rotation.w<0. {rotation=-rotation;}
                    lights[index].emission=[rotation.x,rotation.y,rotation.z,f32::from_bits(address)];
                }
            }
            self.evaluated_lights.clone_from(&lights);
            if let Some(subset) = &s.main_subset {
                for (light, included) in lights.iter_mut().zip(&subset.lights) {
                    if !included {
                        light.radiance[..3].fill(0.);
                    }
                }
            }
            let mut u = uniform;
            u.light_count = lights.len() as f32;
            u.shadow_links[2] = u32::from(lights::needs_receiver_masks(&lights));
                self.receiver_masks = u.shadow_links[2];
            u.radiance[3] = 1.;
            changes.upload(&self.view.lighting.uniform, 0, bytemuck::bytes_of(&u));
            self.view.lighting.upload_lights(changes, &lights)?;
        }
        } else {
            // Lighting is invariant, but projection/lens/phase/output fields are not.
            let mut u = uniform;
            u.light_count = self.evaluated_lights.len() as f32;
            u.shadow_links[2] = self.receiver_masks;
            u.radiance[3] = 1.;
            changes.upload(&self.view.lighting.uniform, 0, bytemuck::bytes_of(&u));
        }
        if let (Some(p),Some(slot))=(&s.fibre,&self.fibre) {
            slot.prepare(&p.storage(),s,&self.object_values,&self.evaluated_lights,uniform,self.materials.as_ref().map(|m| &m.records),changes,shadow_changes);
        }
        let spatial_camera = (!sampled_lens || physical.as_ref().is_none_or(|p| p.radius==0.))
            && (self.view.sample_count > 1 || self.view.visibility_samples > 1) && !self.view.nominal_sample
            && render.motion_blur.as_ref().is_none_or(|m|m.shutter==0.);
        let selection_vp = if spatial_camera || uniform.post[3]>0. {
            let radius=crate::sampling::lens_selection_radius(uniform.post[3], uniform.lens[1], near, far);
            crate::sampling::selection_projection(nominal_vp, self.view.internal, radius)
        } else { uniform.vp };
        self.view.camera_cull.prepare_exposure(selection_vp, &self.object_values,
            &self.geometry_bounds, changes, self.reuse_values && spatial_camera);
        if !self.view.transparent_order.is_empty() {
            let indirect = self.view.transparent_indirect.is_some();
            let sorted = passes::transparent::sort(
                s,
                &self.object_values,
                &mut self.view.transparent_order,
                view,
                indirect,
            );
            if let (Some(objects), Some(indirect), Some((values, args, runs))) = (
                &self.view.transparent_instances,
                &self.view.transparent_indirect,
                sorted,
            ) {
                // Only the draw-slot mapping changes; the shared object table
                // retains canonical identities for raster shading and ray queries.
                for (i, value) in values {
                    changes.upload(
                        objects,
                        (i * std::mem::size_of::<u32>()) as u64,
                        bytemuck::bytes_of(&value),
                    );
                }
                changes.upload(indirect, 0, bytemuck::cast_slice(&args));
                self.view.transparent_runs = runs;
            }
        }
        Ok((
            std::mem::take(&mut changes.bytes),
            std::mem::take(&mut changes.uploads),
            view,
            uniform,
        ))
    }
    /// Copies this sample's packed values into place at the start of its commands.
    pub(crate) fn copy_values(
        &mut self,
        r: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
        upload_bytes: &[u8],
        uploads: &[Upload],
        batch: bool,
    ) {
        profile_span!(_profile, "copy_values");
        if upload_bytes.is_empty() { return; }
        let size = upload_bytes.len() as u64;
        // Batched commands must retain distinct immutable sources until publication.
        // Streaming commands submit before the next sample and can reuse their source.
        if batch || self.sample_upload.size() < size {
            self.sample_upload = crate::gpu_memory::buffer(&r.device, &wgpu::BufferDescriptor {
                label: Some("sample values"), size,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if batch {
            self.exposure_upload.push((self.sample_upload.clone(), upload_bytes.to_vec()));
        } else {
            r.queue.write_buffer(&self.sample_upload, 0, upload_bytes);
        }
        for (destination, offset, source, size) in uploads {
            encoder.copy_buffer_to_buffer(
                &self.sample_upload,
                *source,
                destination,
                *offset,
                *size,
            );
        }
    }
}

pub(crate) fn deformed_bounds(
    bounds: &[Vec<(Vec3, Vec3)>],
    matrices: &[Mat4],
    weights: &[f32],
) -> (Vec3, Vec3) {
    let mut result = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for (matrix, intervals) in matrices.iter().zip(bounds) {
        let (mut lo, mut hi) = intervals[0];
        if !lo.is_finite() {
            continue;
        }
        for (index, &(dlo, dhi)) in intervals[1..].iter().enumerate() {
            let weight = weights.get(index).copied().unwrap_or(0.);
            lo += (dlo * weight).min(dhi * weight);
            hi += (dlo * weight).max(dhi * weight);
        }
        let transformed = crate::shadow_envelope::transformed(*matrix, (lo, hi));
        result.0 = result.0.min(transformed.0);
        result.1 = result.1.max(transformed.1);
    }
    result
}

fn aperture_shape(node: &EvalNode, aspect: f32) -> [f32;4] {
    let Some(c)=node.camera.as_ref().and_then(|c|c.calibration(aspect).ok().flatten()) else {return [0.,0.,1.,0.]};
    [c[2][2],c[2][3],c[3][0],f32::from(c[0][2]!=0. || c[0][3]!=0.)]
}

/// The pixel jitter and lens offset `sampling::projection` and `sampling::lens`
/// give every pixel of this camera sample (`Uniform::camera_sample`).
fn camera_sample(resolution: [u32; 2], lens: Option<&crate::sampling::Lens>, index: usize, count: usize) -> [f32; 4] {
    let jitter = if count > 1 { crate::sampling::clip_offset(resolution, index, count) } else { [0.; 2] };
    let offset = crate::sampling::lens_offset(lens, index, count);
    [jitter[0], jitter[1], offset.x, offset.y]
}

fn calibration(node: &EvalNode, aspect: f32) -> Option<[[f32;4];4]> {
    node.camera.as_ref()?.calibration(aspect).expect("validated camera calibration")
}
fn composite_camera(node: &EvalNode, aspect: f32, sampled: bool, samples: u32) -> ([f32;4],[f32;4]) {
    let Some(c)=calibration(node,aspect) else {return ([0.;4],[0.;4])};
    let sampled=sampled && c[2][0]>0.;
    ([c[2][0],0.,c[2][1],1.], [c[0][0],c[0][1],f32::from(sampled),if sampled {samples.max(1) as f32} else {1.}])
}
fn physical_camera(node: &EvalNode, aspect: f32) -> Option<crate::sampling::Lens> {
    let c=calibration(node,aspect)?;
    if c[1][2]!=0. {return None;}
    Some(crate::sampling::Lens {radius:c[2][0],screen_scale:[c[0][0],c[0][1]],
        focus_distance:c[2][1],aperture_blades:c[2][2] as u32,aperture_rotation:c[2][3],aperture_ratio:c[3][0]})
}
fn camera_clip(node: &EvalNode, aspect: f32) -> (f32,f32,Option<f32>) {
    calibration(node,aspect).map_or((0.,65535.,None),|c|(c[1][0],c[1][1],(c[1][2]!=0.).then_some(2./c[0][1])))
}
pub(crate) fn camera_projection(node: &EvalNode, aspect: f32) -> Mat4 {
    let Some(c)=calibration(node,aspect) else {return Mat4::IDENTITY;};
    let [near,far,ortho,_]=c[1];
    let mut p=if ortho!=0. {Mat4::orthographic_rh(-1./c[0][0],1./c[0][0],-1./c[0][1],1./c[0][1],near,far)}
        else {Mat4::perspective_rh(2.*(1./c[0][1]).atan(),c[0][1]/c[0][0],near,far)};
    let mut shift=Mat4::IDENTITY;shift.w_axis.x=c[0][2];shift.w_axis.y=c[0][3];p=shift*p;p
}

/// Compose local TRS; the installed target plan applies aiming afterward.
pub(super) fn node_world(node: &Node, parent: Mat4) -> Result<Mat4> {
    Ok(parent * node.transform.matrix())
}

pub(super) fn node_view(
    id: usize,
    nodes: &(impl std::ops::Index<usize, Output = EvalNode> + ?Sized),
    worlds: &[Mat4],
) -> Result<Mat4> {
    let world = worlds[id];
    let eye = world.transform_point3(Vec3::ZERO);
    // A collapsed frame has no recoverable orientation. Compose authored rotations
    // without scale, as for Blender's analytic lights and cameras.
    let orientation = if buffers::node_collapsed(nodes, id) || crate::collapsed(world) {
        let mut chain = Vec::new();
        let mut current = Some(id);
        while let Some(index) = current {
            chain.push(index);
            current = nodes[index].parent;
        }
        let mut frame = Mat4::IDENTITY;
        for index in chain.into_iter().rev() {
            let node = &nodes[index].node;
            if let Some(target) = &node.target {
                let target = match target {
                    vork_scene::AimTarget::Point(point) => Vec3::from_array(*point),
                    vork_scene::AimTarget::Node(name) => {
                        let target = (0..worlds.len()).find(|&i| nodes[i].node.name == *name)
                            .context("view target lost its installed node")?;
                        worlds[target].w_axis.truncate()
                    }
                };
                let origin = worlds[index].w_axis.truncate();
                let direction = (target - origin).try_normalize().context("view target coincides with node origin")?;
                let up = if direction.dot(Vec3::Y).abs() > constants::ORIENTATION_UP_PARALLEL {Vec3::Z} else {Vec3::Y};
                frame = Mat4::look_to_rh(origin, view_axis(origin, direction), up).inverse();
            }
            frame *= Mat4::from_quat(glam::Quat::from_array(node.transform.resolved_rotation()).normalize());
        }
        frame
    } else {world};
    let direction = view_axis(eye, orientation.transform_vector3(Vec3::NEG_Z).normalize());
    let mut up = orientation.transform_vector3(Vec3::Y).normalize();
    if direction.dot(up).abs() > constants::ORIENTATION_UP_PARALLEL {
        up = orientation.transform_vector3(Vec3::Z).normalize();
        if direction.dot(up).abs() > constants::ORIENTATION_UP_PARALLEL {
            up = orientation.transform_vector3(Vec3::X).normalize();
        }
    }
    Ok(Mat4::look_to_rh(eye, direction, up))
}

/// Keep the established eye/target frame when the unit target retains its
/// direction. At large translations its addition can quantize or erase that
/// direction; construct directly once the round-trip exceeds the orientation
/// tolerance instead of accepting an inaccurate or non-finite frame.
pub(super) fn view_axis(eye: Vec3, direction: Vec3) -> Vec3 {
    let axis = (eye + direction) - eye;
    let rounded = axis.normalize_or_zero();
    if rounded.distance_squared(direction) > constants::ORIENTATION_EPSILON {
        direction
    } else {
        rounded
    }
}

#[cfg(test)]
#[path = "values_orientation_tests.rs"]
mod orientation_tests;

#[cfg(test)]
#[path = "values_upload_cache_tests.rs"]
mod upload_cache_tests;

#[cfg(test)]
#[path = "values_deformation_bounds_tests.rs"]
mod deformation_bounds_tests;

/// frame.environment_links.w: environment refraction (bit 0) and the solid
/// glass thickness technique (2 approximate, 0 mesh exits; transmission.slang).
fn transmission_word(render: &vork_scene::Render) -> u32 {
    render.settings.integrator.raster().map_or(0, |r| u32::from(r.refraction == vork_scene::Refraction::Environment))
}
/// A view reads the scene values already evaluated for this shared frame sample.
/// It never samples tracks, poses, material inputs, or shadow state again.
pub(crate) fn capture_uniform(
    s: &InstalledScene,
    frame: &FrameSlot,
    image: &crate::render_images::InstalledImageRender,
    mut u: Uniform,
    sample: u32,
) -> Result<Uniform> {
    let samples = image.exposure_samples;
    let visibility_samples = image.visibility_samples;
    // Rule 1: an absent camera has no view to render through.
    ensure!(!s.animation.absent(&frame.animation_values, image.camera),
        "images.{}.render.camera: camera {:?} is absent (active); a render image renders through a camera that exists at sample time {}",
        image.name, s.nodes[image.camera].node.name, frame.sample_request.numeric()?);
    let nodes = SampledNodes::new(&s.nodes, &frame.sampled_nodes);
    let previous_nodes = SampledNodes::new(&s.nodes, &frame.previous_nodes);
    let node = &nodes[image.camera];
    let previous_node = &previous_nodes[image.camera];
    let render = &image.settings;
    let [w, h] = render.resolution;
    let aspect = w as f32 / h as f32;
    let lens = physical_camera(node, aspect);
    let projection = sampling::projection(
        camera_projection(node, aspect),
        render.resolution,
        sample as usize,
        samples as usize,
    );
    let (projection, view) = sampling::lens(
        projection,
        node_view(image.camera, &nodes, &frame.worlds)?,
        lens.as_ref(),
        sample as usize,
        samples as usize,
    );
    u.camera_program = crate::camera_program::selection(s, image.camera,node,aspect);
    u.camera_sample = camera_sample(render.resolution, lens.as_ref(), sample as usize, samples as usize);
    let previous_projection = sampling::projection(
        camera_projection(previous_node, aspect),
        render.resolution,
        sample as usize,
        samples as usize,
    );
    let previous_lens = physical_camera(previous_node, aspect);
    let (previous_projection, previous_view) = sampling::lens(
        previous_projection,
        node_view(image.camera, &previous_nodes, &frame.previous_worlds)?,
        previous_lens.as_ref(),
        sample as usize,
        samples as usize,
    );
    let (near, far, ortho) = camera_clip(node, aspect);
    u.composite_projection = projection;
    (u.composite_optics,u.composite_sensor) = composite_camera(node,aspect,true,visibility_samples * samples);
    u.composite_shutter = [u.scene[0];4];
    u.composite_step = [0.,1.,render.settings.samples as f32,samples as f32];
    u.display[1]=w as f32;u.display[2]=h as f32;
    u.vp = projection * view;
    u.previous_vp = previous_projection * previous_view;
    // The next pose endpoint is the current sample. It belongs to this capture
    // camera too; inheriting the parent matrix leaks its camera and jitter.
    u.next_vp = u.vp;
    u.inverse_vp = u.vp.inverse();
    u.view = view;
    u.eye = view
        .inverse()
        .transform_point3(Vec3::ZERO)
        .extend(1.)
        .to_array();
    u.params[0] = near;
    u.params[1] = far;
    u.projection[0] = projection.y_axis.y;
    u.projection[1] = projection.x_axis.x;
    u.projection[2] = reflection_mode(render);
    u.projection[3] = (constants::OUTPUT_FLOAT_COLOR | (crate::compositor::channel_flags(s, &render.compositor))) as f32;
    u.sampling = s.sampling.uniform(render.settings.samples as usize, visibility_samples as usize);
    u.sampling[3] = if s.plan.decoupled { visibility_samples } else { 0 };
    u.environment_links[3] = transmission_word(render);
    u.post[2] = 0.;
    let lens_scale=|active| lens.as_ref().filter(|_| active).map_or(0., |p| p.circle_scale(h as f32));
    u.post[3] = lens_scale(s.plan.decoupled && visibility_samples > 1);
    u.integrator[0] = 0;
    u.integrator[1] = 0;
    u.integrator[2] = render.settings.samples;
    u.integrator[3] = s.shadows().ray_sequence_word(render.settings.samples as usize) | u32::from(render.settings.integrator.subsurface_method()==Some(vork_scene::v2::SubsurfaceMethod::Thickness))*vork_slang::registry::INTEGRATOR_SUBSURFACE_THICKNESS;
    u.temporal = [sample as f32, samples as f32, samples as f32, if render.settings.integrator.is_path() {vork_slang::registry::DENOISE_GUIDES_PATH as f32} else {0.}];

    let camera_world = u.environment_links[0] & constants::RAY_CAMERA != 0;
    u.environment[3] =
        f32::from(render.background.is_environment() && camera_world);
    let indirect = s.plan.effects.indirect && render.settings.integrator.indirect().is_some();
    u.ao = ao_uniform(indirect);
    u.flags[3] = f32::from(indirect);
    u.aperture = aperture_shape(node,aspect);
    u.lens = [lens.as_ref().map_or(0., sampling::aperture_radius),
        lens.as_ref().map_or(0., |p| p.focus_distance), 2. / w as f32, 2. / h as f32];
    u.flags[0] = f32::from(ortho.is_some());
    u.flags[2] = f32::from(render.background.transparent());
    u.background = match &render.background {
        ::vork_scene::v2::Background::Color(color) => *color,
        ::vork_scene::v2::Background::Mode(::vork_scene::v2::BackgroundMode::Transparent) => {
            [0.; 3]
        }
        _ if camera_world => [u.radiance[0], u.radiance[1], u.radiance[2]],
        _ => [0.; 3],
    };
    u.shadow_links[2] = u32::from(lights::needs_receiver_masks(&frame.evaluated_lights));
    u.light_count = frame.evaluated_lights.len() as f32;
    u.post[0] = if s.has_catchers {1.} else {post_mode(render)};
    u.radiance[3] = 1.;
    Ok(u)
}

#[cfg(test)]
#[path = "values_capture_lens_tests.rs"]
mod capture_lens_tests;

// These are view choices, unlike the installed union of shader/resources.
fn reflection_mode(render: &vork_scene::Render) -> f32 {
    match render.settings.integrator.reflections() {
        None => constants::REFLECTIONS_OFF,
        Some(vork_scene::Reflections::Environment) => constants::REFLECTIONS_ENVIRONMENT,
        Some(vork_scene::Reflections::Auto | vork_scene::Reflections::Ray) => constants::REFLECTIONS_RAY,
    }
}

/// Fixed diffuse-query budgets, shared with the GI shader.
fn ao_uniform(enabled: bool) -> [f32; 4] {
    [constants::GI_RADIUS_M, if enabled {1.} else {0.}, constants::GI_RAYS_PER_SAMPLE as f32, constants::GI_RESOLUTION_DIVISOR as f32]
}

fn post_mode(render: &vork_scene::Render) -> f32 {
    if !render.compositor.is_empty() || render.settings.integrator.denoiser().is_some() || render.settings.reconstruct {1.}
    else {0.}
}

#[cfg(test)]
#[path = "values_aim_tests.rs"]
mod aim_tests;

#[cfg(test)]
#[path = "values_retained_selection_tests.rs"]
mod retained_selection_tests;
