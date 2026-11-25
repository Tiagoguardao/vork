    use super::*;
    #[test]
    fn two_joint_bounds_include_positive_and_negative_morphs() {
        let bounds = vec![
            vec![(Vec3::ZERO, Vec3::ONE), (Vec3::ZERO, Vec3::X)],
            vec![(Vec3::ZERO, Vec3::ONE), (Vec3::ZERO, Vec3::Y)],
        ];
        let matrices = [
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(2., 0., 0.)),
        ];
        let (lo, hi) = deformed_bounds(&bounds, &matrices, &[0.5]);
        assert_eq!(lo, Vec3::ZERO);
        assert_eq!(hi, Vec3::new(3., 1.5, 1.));
        let (lo, hi) = deformed_bounds(&bounds, &matrices, &[-0.5]);
        assert_eq!(lo, Vec3::new(-0.5, -0.5, 0.));
        assert_eq!(hi, Vec3::new(3., 1., 1.));
    }
