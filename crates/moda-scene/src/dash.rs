//! Compact SVG repeatable-list interpolation. The sampled lengths stay in authored px.
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DashList {
    Lengths(Arc<[f64]>),
    Mix(Arc<Self>, Arc<Self>, f32),
}

impl DashList {
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Lengths(values) if values.is_empty())
    }

    pub fn mix(&self, other: &Self, weight: f32) -> Self {
        Self::Mix(Arc::new(self.clone()), Arc::new(other.clone()), weight)
    }

    /// Follow exact endpoints before evaluating the repeatable list.
    pub fn active(&self) -> &Self {
        let mut list = self;
        loop {
            list = match list {
                Self::Mix(left, _, 0.) => left,
                Self::Mix(_, right, 1.) => right,
                _ => return list,
            };
        }
    }

    pub fn period(&self) -> Option<u128> {
        fn gcd(mut a: u128, mut b: u128) -> u128 {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        }
        match self.active() {
            Self::Lengths(values) => Some(values.len() as u128),
            Self::Mix(a, b, _) => {
                let (a, b) = (a.period()?, b.period()?);
                if a == 0 || b == 0 {
                    Some(0)
                } else {
                    (a / gcd(a, b)).checked_mul(b)
                }
            }
        }
    }

    pub fn append_key(&self, key: &mut Vec<u64>) {
        match self {
            Self::Lengths(values) => {
                key.extend([0, values.len() as u64]);
                key.extend(values.iter().map(|value| value.to_bits()));
            }
            Self::Mix(a, b, weight) => {
                key.extend([1, u64::from(weight.to_bits())]);
                a.append_key(key);
                b.append_key(key);
            }
        }
    }
}

/// Source lengths, plus the SVG pathLength calibration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SampledDash {
    pub list: DashList,
    pub scale: f64,
}

impl SampledDash {
    pub fn append_key(&self, key: &mut Vec<u64>) {
        key.push(self.scale.to_bits());
        self.list.append_key(key);
    }
}
