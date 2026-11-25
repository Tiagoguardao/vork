//! Spatial source selection; path transport consumes evaluated analytic sources.
//! Bounds/powers are proposal guides, never radiance or visibility certificates.
use super::Light;
use crate::Vec3;
use anyhow::{Result, ensure};
use bytemuck::{Pod, Zeroable};

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Kind {
    Analytic,
    #[cfg(test)]
    Mesh,
    #[cfg(test)]
    Environment = 3,
}
#[derive(Clone, Copy)]
pub(crate) struct Source {
    pub kind: Kind,
    /// Index in the existing light / ray-instance / implicit-field table.
    pub identity: u32,
    pub bounds: (Vec3, Vec3),
    pub power: f64,
    pub infinite: bool,
}
/// Four explicit 16-byte lanes; identical to LightTreeNode in light_tree.slang.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct Node {
    pub lower_power: [f32; 4],
    pub upper_infinite: [f32; 4],
    /// left, right, parent, number of descendant sources. MAX children = leaf.
    pub links: [u32; 4],
    /// kind, installed identity, input source index, reserved.
    pub source: [u32; 4],
}
pub(crate) struct Tree {
    pub nodes: Vec<Node>,
    /// Input source index -> node; continuation-hit PDF replay needs no scan.
    pub leaves: Vec<u32>,
}

/// Current evaluated analytic-source guides, in shared-light identity order.
pub(crate) fn analytic_sources(lights: &[Light]) -> Vec<Source> {
    let mut out = Vec::new();
    for (identity, light) in lights.iter().enumerate() {
        let p = Vec3::from_slice(&light.position);
        let kind = light.position[3] as u32;
        let radius = if kind == 3 {
            (Vec3::from_slice(&light.emitter_u).abs() + Vec3::from_slice(&light.emitter_v).abs())
                * 0.5
        } else {
            Vec3::splat(if kind == 0 { 0. } else { light.params[1] })
        };
        let power = light.radiance[..3]
            .iter()
            .zip(crate::constants::LUMINANCE_REC709)
            .map(|(&v, c)| f64::from(v) * f64::from(c))
            .sum::<f64>();
        out.push(Source {
            kind: Kind::Analytic,
            identity: identity as u32,
            bounds: (p - radius, p + radius),
            power: power.max(0.),
            infinite: kind == 0,
        });
    }
    out
}

impl Tree {
    pub(crate) fn build(sources: &[Source]) -> Result<Self> {
        ensure!(
            sources.len() <= (1 << 24),
            "light tree exceeds the 24-bit uniform source-coordinate capacity"
        );
        for s in sources {
            ensure!(
                s.power.is_finite()
                    && s.power >= 0.
                    && s.bounds.0.is_finite()
                    && s.bounds.1.is_finite()
                    && s.bounds.0.cmple(s.bounds.1).all(),
                "invalid light-tree guide"
            );
        }
        let scale = sources
            .iter()
            .map(|s| s.power)
            .fold(0., f64::max)
            .max(f64::MIN_POSITIVE);
        let mut tree = Self {
            nodes: Vec::with_capacity(sources.len().saturating_mul(2)),
            leaves: vec![0; sources.len()],
        };
        if !sources.is_empty() {
            tree.build_range(
                sources,
                &mut (0..sources.len()).collect::<Vec<_>>(),
                u32::MAX,
                scale,
            );
        }
        Ok(tree)
    }
    fn build_range(
        &mut self,
        sources: &[Source],
        ids: &mut [usize],
        parent: u32,
        scale: f64,
    ) -> u32 {
        let index = self.nodes.len() as u32;
        self.nodes.push(Node::zeroed());
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        let mut power = 0.;
        let mut infinite = false;
        for &id in ids.iter() {
            let s = sources[id];
            lo = lo.min(s.bounds.0);
            hi = hi.max(s.bounds.1);
            power += s.power / scale;
            infinite |= s.infinite;
        }
        // Explicitly discard subnormal guides so CPU and GPU agree even on
        // devices that flush subnormal inputs. The uniform mixture retains these
        // sources; guide quantization changes variance, never source support.
        let power = power as f32;
        let power = if power < f32::MIN_POSITIVE { 0. } else { power };
        let mut node = Node {
            lower_power: lo.extend(power).to_array(),
            upper_infinite: hi.extend(if infinite { 1. } else { 0. }).to_array(),
            links: [u32::MAX, u32::MAX, parent, ids.len() as u32],
            source: [0; 4],
        };
        if ids.len() == 1 {
            let id = ids[0];
            node.source = [sources[id].kind as u32, sources[id].identity, id as u32, 0];
            self.leaves[id] = index;
        } else {
            // Median spatial split gives bounded depth, deterministic topology and
            // O(N log N) build work. Infinite sources form a separate subtree.
            let extent = hi.as_dvec3() - lo.as_dvec3();
            let axis = if extent.x >= extent.y && extent.x >= extent.z {
                0
            } else if extent.y >= extent.z {
                1
            } else {
                2
            };
            let finite = ids.iter().filter(|&&i| !sources[i].infinite).count();
            let split = if finite > 0 && finite < ids.len() {
                ids.sort_unstable_by_key(|&i| (sources[i].infinite, i));
                finite
            } else {
                let mid = ids.len() / 2;
                ids.select_nth_unstable_by(mid, |&a, &b| {
                    let center = |i: usize| {
                        f64::from(sources[i].bounds.0[axis]) + f64::from(sources[i].bounds.1[axis])
                    };
                    center(a).total_cmp(&center(b)).then(a.cmp(&b))
                });
                mid
            };
            let (a, b) = ids.split_at_mut(split);
            node.links[0] = self.build_range(sources, a, index, scale);
            node.links[1] = self.build_range(sources, b, index, scale);
        }
        self.nodes[index as usize] = node;
        index
    }
}
#[cfg(test)]
#[path = "tree_tests.rs"]
mod tests;
