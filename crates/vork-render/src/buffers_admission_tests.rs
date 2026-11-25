    #[test]
    fn expanded_work_is_checked_without_allocating_it() {
        let mut total = 0;
        let error = super::admit_triangle_instances(50_000 * crate::points::triangles_per_point(12).unwrap(),
            20_000, &mut total).unwrap_err();
        assert!(error.to_string().contains("reduce geometry density or instance count"));
        let mut total = 0;
        super::admit_triangle_instances(1_000, 20_000, &mut total).unwrap();
        assert_eq!(total, 20_000_000);
        assert!(super::admit_triangle_instances(u64::MAX, 2, &mut total).is_err());
    }
