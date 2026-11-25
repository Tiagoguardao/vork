use super::moda_engine_path_types::PathCmd;

/// Local geometric bounds, including point and line boxes.
pub fn path_bounds(cmds: &[PathCmd]) -> Option<[f64; 4]> {
    moda_scene::path_geom::bounds::path_bounds(cmds).map(|b| [b.min_x, b.min_y, b.max_x, b.max_y])
}

/// The path that is drawn, including authored corner rounding and arc-to-cubic conversion.
pub(crate) fn drawn_node_path_bounds(node: &moda_scene::types::RawNode) -> Option<moda_scene::path_geom::bounds::Bounds> {
    moda_scene::path_geom::bounds::drawn_path_bounds(super::path_cmds_for_node(node))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curve_bounds_are_the_extrema_not_the_handles() {
        // A quarter-circle-ish bump: handles reach y = -40, the curve only -30.
        let cmds = [
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Curve {
                x: 100.0,
                y: 0.0,
                cp1: (0.0, -40.0),
                cp2: (100.0, -40.0),
            },
        ];
        let b = path_bounds(&cmds).unwrap();
        assert_eq!([b[0], b[2], b[3]], [0.0, 100.0, 0.0]);
        assert!((b[1] + 30.0).abs() < 1e-9, "{b:?}");
    }

    #[test]
    fn arc_bounds_include_the_bulge() {
        // A half circle of radius 50 from (0, 0) to (100, 0).
        let cmds = [
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Arc {
                x: 100.0,
                y: 0.0,
                rx: 50.0,
                ry: 50.0,
                x_axis_rotation: 0.0,
                large_arc: false,
                sweep: false,
            },
        ];
        let b = path_bounds(&cmds).unwrap();
        assert!((b[0]).abs() < 1e-6 && (b[2] - 100.0).abs() < 1e-6, "{b:?}");
        assert!((b[3] - b[1] - 50.0).abs() < 1e-6, "{b:?}");
    }

    #[test]
    fn empty_path_has_no_bounds() {
        assert_eq!(path_bounds(&[]), None);
    }
}
