    use super::UploadCache;
    #[test]
    fn ray_history_preserves_object_deltas_without_retaining_view_uploads() {
        use super::{ChangeKey, ChangeTracker};
        let (device, _) = wgpu::Device::noop(&Default::default());
        let [objects, view] = std::array::from_fn(|_| device.create_buffer(
            &wgpu::BufferDescriptor { label: None, size: 16,
                usage: wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
        let mut changes = ChangeTracker::default();
        changes.sample = 7;
        changes.changed(ChangeKey::RayStructures, &[1]);
        changes.changed(ChangeKey::SampleTime(0), b"previous time");
        changes.upload(&objects, 0, &[1; 16]);
        changes.upload(&view, 0, &[2; 16]);
        changes.retain_ray_history(&objects);
        assert_eq!(changes.history.len(), 2);
        assert!(changes.uploads.is_empty() && changes.bytes.is_empty());
        changes.sample += 1;
        assert!(!changes.changed(ChangeKey::RayStructures, &[1]));
        changes.upload(&objects, 0, &[1; 16]);
        assert!(changes.uploads.is_empty());
        changes.upload(&objects, 4, &[3; 4]);
        assert_eq!(changes.uploads[0].1, 4);
        assert_eq!(changes.uploads[0].3, 4);
        assert!(changes.changed(ChangeKey::SampleTime(0), b"previous time"));
        changes.upload(&view, 0, &[2; 16]);
        assert_eq!(changes.uploads.len(), 2);
    }

    #[test]
    fn external_gpu_replacement_invalidates_outputs_and_derivation_together() {
        use super::{ChangeKey, ChangeTracker};
        let (device, _) = wgpu::Device::noop(&Default::default());
        let [commands, instances, other] = std::array::from_fn(|_| device.create_buffer(
            &wgpu::BufferDescriptor { label: None, size: 16,
                usage: wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
        let key = ChangeKey::CameraSelection(commands.clone());
        let mut changes = ChangeTracker::default();
        changes.sample = 1;
        changes.changed(ChangeKey::CameraSelection(other.clone()), b"other camera");
        changes.upload(&other, 0, &[7; 16]);
        // A replacement can precede the first CPU publication, occur repeatedly
        // in one sample, or follow retained values from an earlier sample.
        for advance in [false, false, true, false] {
            if advance { changes.sample += 1; }
            changes.replace_gpu(&[&commands, &instances], key.clone());
            assert!(changes.changed(key.clone(), b"same camera"));
            assert!(!changes.changed(key.clone(), b"same camera"));
            changes.bytes.clear();
            changes.uploads.clear();
            changes.upload(&commands, 0, &[0; 16]);
            changes.upload(&instances, 4, &[2; 8]);
            assert_eq!(changes.uploads.len(), 2, "equal prior CPU bytes must republish every destination");
            assert!(changes.buffer_dirty(&commands));
            assert!(changes.buffer_dirty(&instances));
            changes.bytes.clear();
            changes.uploads.clear();
            changes.upload(&commands, 4, &[0; 4]);
            changes.upload(&instances, 8, &[2; 4]);
            changes.upload(&other, 0, &[7; 16]);
            assert!(changes.uploads.is_empty(), "CPU history resumes and unrelated outputs stay retained");
            assert!(!changes.changed(ChangeKey::CameraSelection(other.clone()), b"other camera"));
        }
    }

    #[test]
    fn upload_publication_survives_merging_and_drain_until_the_next_sample() {
        let (device, _) = wgpu::Device::noop(&Default::default());
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: 16, usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let other = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: 16, usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut changes = super::ChangeTracker::default();
        changes.sample = 1;
        assert!(!changes.buffer_dirty(&buffer));
        changes.upload(&buffer, 4, &[0; 4]); // First zero write is a publication.
        changes.upload(&buffer, 8, &[1; 4]); // Adjacent ranges merge.
        assert_eq!(changes.uploads.len(), 1);
        assert!(changes.buffer_dirty(&buffer));
        assert!(!changes.buffer_dirty(&other));
        changes.uploads.clear(); // Submission takes the work list, not its history.
        changes.bytes.clear();
        assert!(changes.buffer_dirty(&buffer));
        changes.sample += 1;
        changes.upload(&buffer, 4, &[0; 4]);
        assert!(!changes.buffer_dirty(&buffer));
        assert!(changes.uploads.is_empty());
        changes.upload(&buffer, 6, &[2; 4]); // Partial overlap.
        changes.upload(&buffer, 6, &[2; 4]); // Unchanged write cannot clear dirty.
        assert!(changes.buffer_dirty(&buffer));
        assert_eq!(changes.uploads.len(), 1);
        changes.sample += 1;
        assert!(!changes.buffer_dirty(&buffer));
    }

    #[test]
    fn partial_overlaps_and_first_zero_writes_remain_dirty() {
        let mut cache = UploadCache::default();
        assert!(cache.changed(4, &[1, 2, 3, 4]));
        assert!(!cache.changed(4, &[1, 2, 3, 4]));
        assert!(cache.changed(6, &[0, 0]));
        assert!(!cache.changed(6, &[0, 0]));
        assert!(cache.changed(4, &[1, 2, 3, 4]));
        assert!(cache.changed(0, &[0, 0, 0, 0]));
        assert!(!cache.changed(0, &[0, 0, 0, 0, 1, 2, 3, 4]));
    }
