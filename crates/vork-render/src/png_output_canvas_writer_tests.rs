    use super::*;

    fn read(path: &std::path::Path, expected: &[u8]) {
        let mut decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()))
            .read_info().unwrap();
        assert_eq!(decoder.info().srgb, Some(png::SrgbRenderingIntent::Perceptual));
        let mut pixels = vec![0; decoder.output_buffer_size().unwrap()];
        let info = decoder.next_frame(&mut pixels).unwrap();
        assert_eq!(info.color_type, png::ColorType::Rgba);
        assert_eq!(&pixels[..info.buffer_size()], expected);
    }

    #[test]
    fn canvas_writer_preserves_pixels_indices_and_shared_credits() {
        let root = std::env::temp_dir().join(format!("vork-canvas-writer-{}", std::process::id()));
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::create_dir_all(root.join("b")).unwrap();
        let writers: Vec<_> = ["a", "b"].into_iter().map(|p| FrameOutputWriter::canvas(root.join(p), 2, 1, 2, 7, Default::default())).collect();
        let credits: [Arc<crate::ring::Credit>; 2] = Default::default();
        let pixels = |frame: usize| vec![frame as u8, 34, 56, 78, 90, 123, 210, 255];
        // More frames than credits exercises backpressure and Finish accounting for ready inputs.
        for frame in 7..13 {
            let lease = Arc::new(credits[frame % 2].acquire());
            let rgba = if frame % 2 == 0 {
                CanvasPixels::new(vec![56, 34, frame as u8, 78, 210, 123, 90, 255], wgpu::TextureFormat::Bgra8Unorm)
            } else { CanvasPixels::new(pixels(frame), wgpu::TextureFormat::Rgba8Unorm) };
            let rgba = Arc::new(rgba);
            for writer in &writers { writer.push_canvas(frame, rgba.clone(), lease.clone()).unwrap(); }
        }
        for writer in writers { writer.finish().unwrap(); }
        for destination in ["a", "b"] {
            for frame in 7..13 { read(&root.join(destination).join(format!("{frame:06}.png")), &pixels(frame)); }
        }
        let still = FrameOutputWriter::canvas(root.join("still.png"), 2, 1, 1, 42, Default::default());
        still.push_canvas(42, Arc::new(CanvasPixels::new(pixels(42), wgpu::TextureFormat::Rgba8Unorm)), Arc::new(credits[0].acquire())).unwrap();
        still.finish().unwrap();
        read(&root.join("still.png"), &pixels(42));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn canvas_writer_failure_releases_shared_credit() {
        let root = std::env::temp_dir().join(format!("vork-canvas-writer-failure-{}", std::process::id()));
        std::fs::create_dir_all(root.join("blocked.png")).unwrap();
        let credit: Arc<crate::ring::Credit> = Default::default();
        let writer = FrameOutputWriter::canvas(root.join("blocked.png"), 1, 1, 1, 0, Default::default());
        writer.push_canvas(0, Arc::new(CanvasPixels::new(vec![1, 2, 3, 255], wgpu::TextureFormat::Rgba8Unorm)), Arc::new(credit.acquire())).unwrap();
        assert!(writer.finish().is_err());
        assert_eq!(Arc::strong_count(&credit), 1, "all writer leases were released");
        let writer = FrameOutputWriter::canvas(root.join("invalid.png"), 2, 2, 1, 0, Default::default());
        writer.push_canvas(0, Arc::new(CanvasPixels::new(vec![1], wgpu::TextureFormat::Rgba8Unorm)), Arc::new(credit.acquire())).unwrap();
        assert!(writer.finish().is_err());
        assert_eq!(Arc::strong_count(&credit), 1);
        std::fs::remove_dir_all(root).unwrap();
    }
