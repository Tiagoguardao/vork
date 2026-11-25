//! CSS Filter Effects 1 shorthand functions, applied to unassociated sRGB.
use anyhow::{ensure, Context, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ColorFunction {
    Brightness,
    Contrast,
    Saturate,
    HueRotate,
    Grayscale,
    Sepia,
    Invert,
    Opacity,
}

impl ColorFunction {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "brightness" => Self::Brightness,
            "contrast" => Self::Contrast,
            "saturate" => Self::Saturate,
            "hue-rotate" => Self::HueRotate,
            "grayscale" => Self::Grayscale,
            "sepia" => Self::Sepia,
            "invert" => Self::Invert,
            "opacity" => Self::Opacity,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Brightness => "brightness",
            Self::Contrast => "contrast",
            Self::Saturate => "saturate",
            Self::HueRotate => "hue-rotate",
            Self::Grayscale => "grayscale",
            Self::Sepia => "sepia",
            Self::Invert => "invert",
            Self::Opacity => "opacity",
        }
    }

    pub fn identity(self) -> f64 {
        match self {
            Self::Grayscale | Self::Sepia | Self::Invert | Self::HueRotate => 0.,
            _ => 1.,
        }
    }

    pub fn argument(self, arg: &str) -> Result<f64> {
        let arg = arg.trim();
        if self == Self::HueRotate {
            return if arg.is_empty() {
                Ok(0.)
            } else {
                super::css_angle(arg).context("hue-rotate() takes an angle")
            };
        }
        if arg.is_empty() {
            return Ok(1.);
        }
        let (number, scale) = arg.strip_suffix('%').map_or((arg, 1.), |n| (n, 0.01));
        let value = number
            .parse::<f64>()
            .context("expected a number or percentage")?
            * scale;
        ensure!(
            value.is_finite() && value >= 0.,
            "{}() takes a nonnegative number or percentage",
            self.name()
        );
        Ok(match self {
            Self::Brightness | Self::Contrast | Self::Saturate => value,
            _ => value.min(1.),
        })
    }

    pub fn matrix(self, a: f64) -> [f32; 20] {
        let mut m = [0.; 20];
        m[15] = 1.;
        let diagonal = match self {
            Self::Brightness => a,
            Self::Contrast => {
                m[16..19].fill((1. - a) * 0.5);
                a
            }
            Self::Invert => {
                m[16..19].fill(a);
                1. - 2. * a
            }
            Self::Opacity => {
                m[15] = a;
                1.
            }
            Self::Grayscale | Self::Sepia | Self::Saturate | Self::HueRotate => {
                let (rows, fraction) = match self {
                    Self::Grayscale => ([[0.2126, 0.7152, 0.0722]; 3], a),
                    Self::Sepia => (
                        [
                            [0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131],
                        ],
                        a,
                    ),
                    Self::Saturate => ([[0.213, 0.715, 0.072]; 3], 1. - a),
                    _ => {
                        let (s, c) = a.to_radians().sin_cos();
                        let luma = [0.213, 0.715, 0.072];
                        let sine = [
                            [-0.213, -0.715, 0.928],
                            [0.143, 0.140, -0.283],
                            [-0.787, 0.715, 0.072],
                        ];
                        let rows = std::array::from_fn(|i| {
                            std::array::from_fn(|j| {
                                let identity = f64::from(i == j);
                                luma[j] + c * (identity - luma[j]) + s * sine[i][j]
                            })
                        });
                        (rows, 1.)
                    }
                };
                for (i, row) in rows.iter().enumerate() {
                    for (j, v) in row.iter().enumerate() {
                        m[4 * i + j] = fraction * v;
                    }
                }
                1. - fraction
            }
        };
        for i in [0, 5, 10] {
            m[i] += diagonal;
        }
        m.map(|v| v as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::ColorFunction::*;

    #[test]
    fn argument_units_and_limits() {
        for function in [
            Brightness, Contrast, Saturate, Grayscale, Sepia, Invert, Opacity,
        ] {
            assert_eq!(function.argument("").unwrap(), 1.);
            assert_eq!(
                function.argument("37%").unwrap(),
                function.argument(".37").unwrap()
            );
            for bad in ["-1%", "1px", "NaN", "1 2"] {
                assert!(function.argument(bad).is_err());
            }
        }
        assert_eq!(Brightness.argument("250%").unwrap(), 2.5);
        assert_eq!(Invert.argument("250%").unwrap(), 1.);
        assert_eq!(HueRotate.argument("").unwrap(), 0.);
        assert_eq!(HueRotate.argument("-0.25turn").unwrap(), -90.);
        assert!(HueRotate.argument("90").is_err());
    }

    #[test]
    fn matrices_preserve_straight_alpha_except_for_opacity() {
        let c = [0.8, 0.4, 0.2, 0.5];
        for (f, a, expected) in [
            (Brightness, 0.5, [0.4, 0.2, 0.1, 0.5]),
            (Contrast, 0.5, [0.65, 0.45, 0.35, 0.5]),
            (Invert, 0.25, [0.65, 0.45, 0.35, 0.5]),
            (Opacity, 0.25, [0.8, 0.4, 0.2, 0.125]),
            (Grayscale, 1., [0.4706, 0.4706, 0.4706, 0.5]),
            (Saturate, 0., [0.4708, 0.4708, 0.4708, 0.5]),
            (Sepia, 1., [0.6598, 0.5872, 0.4574, 0.5]),
            (HueRotate, 180., [0.1416, 0.5416, 0.7416, 0.5]),
        ] {
            let m = f.matrix(a);
            for i in 0..4 {
                let value = m[16 + i] + (0..4).map(|j| m[i * 4 + j] * c[j]).sum::<f32>();
                assert!(
                    (value - expected[i]).abs() < 1e-5,
                    "{f:?} channel {i}: {value}"
                );
            }
        }
    }
}
