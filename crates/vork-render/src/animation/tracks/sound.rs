//! The sound slice of the animation plan: the node clocks, gain tracks and layer weights that
//! can change what is heard, and nothing a picture alone reads. It is evaluated in blocks of
//! output samples. A block opens with one exact evaluation and closes at the first sample where
//! any of its discrete decisions (existence, holds, wraps, key intervals) can change; between
//! them every clock is a polynomial in the sample index, read exactly at each sample. A clock
//! with constant speeds above it is a line, known from the one evaluation. Under a keyed speed
//! (linear or step) it is the integral of the speed's current piece: its degree is known, so
//! that many more exact evaluations inside the block determine it, and the block closes no
//! later than its fastest possible approach to the next decision. A cubic speed keeps one exact
//! evaluation per sample.
use super::clocks::{ActiveSpec, State};
use super::*;
use vork_audio::SAMPLE_RATE;
use vork_time::exact::{Germ, Progression, Rational, ieee};

fn zero() -> Rational { Rational::from_integer(0.into()) }
fn one() -> Rational { Rational::from_integer(1.into()) }

/// What sound depends on, resolved once from the compiled plan.
pub(crate) struct SoundPlan {
    /// Installed nodes whose clocks sound reads, ascending: audible nodes, their ancestors, the
    /// owners of the layers below and of the tracks keying any of these nodes' existence.
    nodes: Vec<usize>,
    /// Layers with a gain track on an audible node or a weight track on such a layer, in plan order.
    layers: Vec<SoundLayer>,
    /// The audible nodes a gain track keys, ascending.
    keyed: Vec<usize>,
    gain_slots: Vec<Option<usize>>,
    /// Scratch row for a layer whose weight is keyed.
    weight_slots: Vec<Option<usize>>,
    driven: usize,
    /// Every keyed existence in `nodes` reads the scene or its parent, so a block can span samples.
    spans: bool,
}
/// Reused evaluation storage for one mix: clock states by installed node, of which only the
/// plan's nodes are ever written, and each sampled track's last value and weight.
pub(crate) struct SoundScratch {
    states: Vec<State>,
    motions: Vec<Motion>,
    samples: Samples,
    /// The sample up to which the pieces in force are known not to be carried as polynomials
    /// (too short a run, or values past machine integers). No decision changes before it, so
    /// until then each sample is one exact evaluation and nothing else, as without blocks.
    refused_until: i64,
}
impl SoundScratch {
    pub(crate) fn rate_bound(&self, owner: usize) -> Rational { self.motions[owner].rate() }
}

/// How a present node's local time moves until a block's horizon: toward lower or higher
/// values, no faster than `bound` local seconds per scene second. Its value is a polynomial
/// of at most `degree` in scene time and its germ's rate one of at most `rate_degree`, from
/// the speed pieces actually in force. A standing clock has bound zero. While every speed
/// from the scene down is constant the clock is `exact`: a line whose rate is the bound.
#[derive(Clone)]
struct Motion {
    falling: bool,
    bound: Rational,
    degree: usize,
    rate_degree: usize,
    exact: bool,
}
impl Motion {
    fn scene() -> Self { Self { falling: false, bound: one(), degree: 1, rate_degree: 0, exact: true } }
    fn still() -> Self { Self { falling: false, bound: zero(), degree: 0, rate_degree: 0, exact: true } }
    /// The fastest signed rate: the rate itself of an exact clock.
    fn rate(&self) -> Rational { if self.falling { -&self.bound } else { self.bound.clone() } }
    fn moves(&self) -> bool { self.bound != zero() }
}
/// A quantity of a block whose polynomial needs more exact evaluations than the first:
/// `degree` more.
enum Curved {
    /// `clocks[index]`: the local value and rate of `node`, and the product of the raw rates
    /// of the clocks in `chain` (the node's and its ancestors'): sound travels to the node at a
    /// sample only while none of them is zero there.
    Voice { index: usize, node: usize, chain: Vec<usize>, degree: usize },
    /// `layers[index]`: the exact local instant of plan layer `layer` on its owner `node`.
    Layer { index: usize, layer: usize, node: usize, degree: usize },
}
impl Curved {
    fn degree(&self) -> usize { match self { Self::Voice { degree, .. } | Self::Layer { degree, .. } => *degree } }
    fn node(&self) -> usize { match self { Self::Voice { node, .. } | Self::Layer { node, .. } => *node } }
}
struct SoundLayer {
    layer: usize,
    tracks: Vec<SoundTrack>,
}
struct SoundTrack {
    flat: usize,
    /// The audible node whose gain the track keys; a weight track has none.
    gain: Option<usize>,
}
impl SoundPlan {
    #[cfg(test)]
    pub(crate) fn spans(&self) -> bool { self.spans }
    #[cfg(test)]
    pub(crate) fn nodes(&self) -> &[usize] { &self.nodes }
}

/// One run of output samples over which every sound decision stands.
pub(crate) struct SoundBlock {
    first: i64,
    end: i64,
    clocks: Vec<Option<SoundClock>>,
    layers: Vec<BlockLayer>,
}
/// A sounding node's local time along a block.
pub(crate) struct SoundClock {
    first: i64,
    line: Carried<f64>,
    speed: Carried<f64>,
    /// Under a keyed speed: zero at the samples where a clock above stands for an instant.
    gate: Option<Progression>,
    selected: Option<Vec<std::ops::Range<i64>>>,
}
impl SoundClock {
    /// Local seconds at output sample `sample` of the block.
    pub(crate) fn seconds(&self, sample: i64) -> Result<f64> {
        match &self.line { Carried::One(seconds) => Ok(*seconds), Carried::Along(line) => line.seconds(sample - self.first) }
    }
    /// Local seconds per scene second at output sample `sample` of the block.
    pub(crate) fn speed(&self, sample: i64) -> Result<f64> {
        match &self.speed { Carried::One(speed) => Ok(*speed), Carried::Along(line) => line.seconds(sample - self.first) }
    }
    /// Whether the clock is a line with one speed along the block: its extremes are its ends.
    pub(crate) fn straight(&self) -> bool { self.gate.is_none() }
    /// Whether sound travels to the node at output sample `sample` of the block. Sound is
    /// absent at an instant where the node's clock or one above it has no speed, even where
    /// the node's own rate is not zero.
    pub(crate) fn sounds(&self, sample: i64) -> bool {
        self.gate.as_ref().is_none_or(|gate| !gate.is_zero(sample - self.first)) &&
            self.selected.as_ref().is_none_or(|runs| runs.get(runs.partition_point(|r| r.end <= sample))
                .is_some_and(|r| r.start <= sample))
    }

    /// The first source frame if the entire block advances on the output sample grid.
    pub(crate) fn aligned_start(&self) -> Option<i64> {
        match (&self.line, &self.speed, &self.selected) {
            (Carried::Along(line), Carried::One(1.), None) => line.unit_step_start(SAMPLE_RATE),
            _ => None,
        }
    }

    pub(crate) fn windows(&self, whole: std::ops::Range<i64>) -> impl Iterator<Item = std::ops::Range<i64>> + '_ {
        self.selected.is_none().then_some(whole).into_iter().chain(self.selected.iter().flatten().cloned())
    }
}
/// A quantity along a block: the one value its evaluation read, which is all a one-sample
/// block or a constant needs and exactly what an evaluation per sample reads; or the exact
/// polynomial that gives it at each sample.
enum Carried<T> {
    One(T),
    Along(Progression),
}
struct BlockLayer {
    layer: usize,
    /// A node-owned layer's local time along the block, while its owner is present.
    owner: Option<Carried<f32>>,
    tracks: Vec<BlockTrack>,
}
struct BlockTrack {
    flat: usize,
    gain: Option<usize>,
    /// The key interval of a track whose keys are read exactly.
    upper: Option<usize>,
    /// The keyed node is absent (rule 1).
    absent: bool,
}

/// The earliest scene instant after `now` at which a decision of the block can change.
struct Horizon<'a> {
    now: &'a Rational,
    at: Option<Rational>,
}
impl Horizon<'_> {
    fn end(&self, first: i64, last: i64) -> i64 {
        self.at.as_ref().and_then(|at| i64::try_from((at * Rational::from_integer(SAMPLE_RATE.into()))
            .ceil().to_integer()).ok()).unwrap_or(last).clamp(first + 1, last)
    }

    /// `value`, moving at `rate` per scene second, reaches `threshold`. At or before `now` the
    /// exact evaluation has already decided it.
    fn reach(&mut self, value: &Rational, rate: &Rational, threshold: &Rational) {
        if *rate == zero() { return; }
        let at = self.now + (threshold - value) / rate;
        if &at > self.now && self.at.as_ref().is_none_or(|earliest| &at < earliest) { self.at = Some(at); }
    }
}

impl Plan {
    pub(crate) fn keys_image_time(&self, name: &str) -> bool {
        self.layers.iter().flat_map(|l| &l.tracks).any(|t| matches!(&t.target, Some(v2::TrackTarget::ImageTime {image}) if image == name))
    }

    /// The sound slice for `audible` nodes (ascending: the nodes with a file and their
    /// ancestors). `node` resolves a gain track's node name to its installed index.
    pub(crate) fn sound_plan(&self, audible: &[usize], node: impl Fn(&str) -> Result<usize>) -> Result<SoundPlan> {
        let mut gains = Vec::new();
        let mut wanted = vec![false; self.layers.len()];
        for (index, layer) in self.layers.iter().enumerate() {
            for track in &layer.tracks {
                let gain = match &track.target {
                    Some(v2::TrackTarget::Node { node: name, property: v2::NodeProperty::AudioGain }) =>
                        Some(node(name)?).filter(|node| audible.binary_search(node).is_ok()),
                    _ => None,
                };
                wanted[index] |= gain.is_some();
                gains.push(gain);
            }
        }
        // A wanted layer's weight is heard, so the layers whose tracks key it are wanted too.
        loop {
            let before = wanted.iter().filter(|wanted| **wanted).count();
            for (index, layer) in self.layers.iter().enumerate() {
                if layer.tracks.iter().any(|track| track.weight_target.is_some_and(|target| wanted[target])) { wanted[index] = true; }
            }
            if wanted.iter().filter(|wanted| **wanted).count() == before { break; }
        }
        let mut layers = Vec::new();
        let mut flat = 0;
        for (index, layer) in self.layers.iter().enumerate() {
            if wanted[index] {
                let tracks = layer.tracks.iter().enumerate()
                    .filter(|(offset, track)| gains[flat + *offset].is_some() || track.weight_target.is_some_and(|target| wanted[target]))
                    .map(|(offset, _)| SoundTrack { flat: flat + offset, gain: gains[flat + offset] })
                    .collect();
                layers.push(SoundLayer { layer: index, tracks });
            }
            flat += layer.tracks.len();
        }
        let mut stack = audible.to_vec();
        stack.extend(layers.iter().filter_map(|layer| self.layers[layer.layer].owner_node));
        let (nodes, spans) = self.sound_nodes(stack);
        let mut keyed: Vec<usize> = gains.into_iter().flatten().collect();
        keyed.sort_unstable();
        keyed.dedup();
        let mut gain_slots = vec![None; self.clocks.len()];
        for (slot, &node) in keyed.iter().enumerate() { gain_slots[node] = Some(slot); }
        let mut weight_slots = vec![None; self.layers.len()];
        let mut driven = 0;
        for target in layers.iter().flat_map(|layer| &layer.tracks)
            .filter_map(|track| self.track_at(track.flat).weight_target)
        {
            if weight_slots[target].is_none() {
                weight_slots[target] = Some(driven);
                driven += 1;
            }
        }
        Ok(SoundPlan { nodes, layers, keyed, gain_slots, weight_slots, driven, spans })
    }
    fn sound_nodes(&self, mut stack: Vec<usize>) -> (Vec<usize>, bool) {
        let mut needed = vec![false; self.clocks.len()];
        let mut spans = true;
        while let Some(node) = stack.pop() {
            if std::mem::replace(&mut needed[node], true) { continue; }
            let spec = &self.clocks[node];
            stack.extend(spec.parent);
            if let Some(track) = spec.active_track {
                let owner = self.layers[self.layer_of(track)].owner_node;
                spans &= owner.is_none() || owner == spec.parent;
                stack.extend(owner);
            }
        }
        let nodes = (0..needed.len()).filter(|&node| needed[node]).collect();
        (nodes, spans)
    }

    pub(crate) fn sound_ancestors(&self, owner: usize) -> Vec<usize> {
        self.sound_nodes(vec![owner]).0
    }

    pub(crate) fn sound_local(&self, owner: usize, nodes: &[usize], scene: &Germ, scratch: &mut SoundScratch) -> Result<Germ> {
        self.sound_states(nodes, scene, &mut scratch.states, false)?;
        scratch.states[owner].local.clone().context("sounding movie's composition has no clock")
    }

    pub(crate) fn sound_scratch(&self) -> SoundScratch {
        SoundScratch { states: vec![State::absent(); self.clocks.len()], motions: vec![Motion::still(); self.clocks.len()], samples: self.samples(), refused_until: i64::MIN }
    }

    fn sound_horizon(&self, nodes: &[usize], scene: &Germ, scratch: &mut SoundScratch,
        mut spans: bool, scene_motion: &Motion, horizon: &mut Horizon<'_>) -> Result<bool> {
        self.sound_states(nodes, scene, &mut scratch.states, spans)?;
        let (states, motions) = (&scratch.states, &mut scratch.motions);
        // How each clock moves, parents first, and where its decisions can next change.
        for &node in if spans { nodes } else { &[] } {
            let spec = &self.clocks[node];
            // Under an absent parent nothing of this node is decided before the parent returns.
            let (parent, along) = match spec.parent {
                None => (scene, scene_motion.clone()),
                Some(parent) => match &states[parent].local {
                    Some(local) => (local, motions[parent].clone()),
                    None => { motions[node] = Motion::still(); continue }
                },
            };
            let rate = along.rate();
            match (&spec.active, spec.active_track) {
                (_, Some(track)) => {
                    // A spanning plan keys existence from the scene or from the parent.
                    let (owner, rate) = if self.layers[self.layer_of(track)].owner_node.is_none() { (scene, scene_motion.rate()) } else { (parent, rate.clone()) };
                    if let Some(keys) = &self.track_at(track).discrete_keys {
                        if let Some(key) = keys.next(owner, rate > zero()) { horizon.reach(owner.value(), &rate, key); }
                    }
                }
                (ActiveSpec::Window(open, close), None) => {
                    horizon.reach(parent.value(), &rate, open);
                    horizon.reach(parent.value(), &rate, close);
                }
                (ActiveSpec::On(_), None) => {}
            }
            let state = &states[node];
            motions[node] = match (&spec.time, &state.raw) {
                // No clock of its own: the parent's instant.
                (None, _) => along,
                // Absent: it returns at a decision found above.
                (Some(_), None) => Motion::still(),
                (Some(time), Some(raw)) => {
                    // The forward clock rises with the parent no faster than its fastest speed.
                    // A constant speed scales the parent's instant. A keyed one integrates its
                    // current piece over the clock it is read on, which has the parent's degree.
                    let (ceiling, degree, rate_degree) = match spec.speed_track {
                        None => (if time.speed < zero() { -&time.speed } else { time.speed.clone() }, along.degree, along.rate_degree),
                        Some(track) => match self.keyed_speed(spec, track, parent, spec.parent.map(|parent| &states[parent]))? {
                            Some(keyed) => {
                                for edge in &keyed.edges { horizon.reach(parent.value(), &rate, edge); }
                                if keyed.sloped { (keyed.ceiling, along.degree.saturating_mul(2), along.degree) } else { (keyed.ceiling, along.degree, 0) }
                            }
                            None => { spans = false; (zero(), along.degree, along.rate_degree) }
                        },
                    };
                    let forward = Motion { falling: along.falling, bound: &along.bound * ceiling, degree, rate_degree, exact: along.exact && spec.speed_track.is_none() };
                    // It holds below `offset` and from `end`, or wraps between them.
                    let section = &time.section;
                    let wrapped = match &section.end {
                        Some(end) if section.looping => raw.euclidean_loop(&section.offset, end)?,
                        _ => raw.clone(),
                    };
                    horizon.reach(wrapped.value(), &forward.rate(), &section.offset);
                    if let Some(end) = &section.end { horizon.reach(wrapped.value(), &forward.rate(), end); }
                    if state.standing || !forward.moves() { Motion::still() } else { Motion { falling: forward.falling != section.reverse, ..forward } }
                }
            };
        }
        Ok(spans)
    }

    /// The block that starts at output sample `first` and ends at or before `last`
    /// (`first < last`). `voices` are the nodes whose clocks the mixer reads.
    pub(crate) fn sound_block<'a>(&self, sound: &SoundPlan, voices: &[usize], remaps: &[Option<(String, Option<vork_time::Clock>)>], paints: &crate::audio::PaintSelection, first: i64, last: i64, scratch: &mut SoundScratch, movie: impl Fn(usize) -> Option<&'a str>) -> Result<SoundBlock> {
        let hz = Rational::from_integer(SAMPLE_RATE.into());
        let now = Rational::new(first.into(), SAMPLE_RATE.into());
        let scene = Germ::right(now.clone());
        // Whether this block may span samples: then it needs every clock's motion and the
        // horizon. A one-sample block needs neither: its evaluation is the whole answer.
        let remapped = remaps.iter().any(Option::is_some);
        let mut horizon = Horizon { now: &now, at: None };
        let mut spans = self.sound_horizon(&sound.nodes, &scene, scratch,
            !remapped && sound.spans && first >= scratch.refused_until, &Motion::scene(), &mut horizon)?;
        let states = &scratch.states;
        let motions = &scratch.motions;
        let mut curved = Vec::new();
        let mut clocks = Vec::with_capacity(voices.len());
        for (index, &voice) in voices.iter().enumerate() {
            let state = &states[voice];
            let motion = &motions[voice];
            let clock = match &state.local {
                // Every speed above is constant, or the block is one sample: the evaluation
                // itself says whether sound travels, and a constant germ is a held clock.
                Some(local) if (!spans || motion.exact) && state.audible && *local != Germ::point(local.value().clone()) => {
                    let rate = local.rate();
                    // A straight clock is its germ: value and rate.
                    spans = spans && *local == Germ::point(local.value().clone()).add(&Germ::right(zero()).scale(&rate));
                    let line = if spans { Carried::Along(Progression::new(local.value().clone(), &rate / &hz)) } else { Carried::One(local.seconds()?) };
                    Some(SoundClock { first, line, speed: Carried::One(Germ::point(rate).seconds()?), gate: None, selected: None })
                }
                // Under a keyed speed no clock above is held along the block, and each sample
                // asks whether one stands at its instant.
                Some(local) if spans && !motion.exact && motion.moves() => {
                    let mut chain = Vec::new();
                    let mut degree = motion.degree;
                    let mut above = Some(voice);
                    while let Some(node) = above {
                        if states[node].raw.is_some() {
                            chain.push(node);
                            degree = degree.max(chain.iter().fold(0_usize, |sum, node| sum.saturating_add(motions[*node].rate_degree)));
                        }
                        above = self.clocks[node].parent;
                    }
                    let gate = self.sound_gate(&chain, states)?;
                    curved.push(Curved::Voice { index, node: voice, chain, degree });
                    Some(SoundClock { first, line: Carried::One(local.seconds()?), speed: Carried::One(Germ::point(local.rate()).seconds()?), gate: Some(Progression::through(&[gate])), selected: None })
                }
                _ => None,
            };
            clocks.push(clock);
        }
        let mut layers = Vec::with_capacity(sound.layers.len());
        for (index, wanted) in sound.layers.iter().enumerate() {
            let layer = &self.layers[wanted.layer];
            let owner = match layer.owner_node {
                None => Some((&scene, Motion::scene())),
                Some(owner) => states[owner].local.as_ref().map(|local| (local, if spans { motions[owner].clone() } else { Motion::still() })),
            };
            let exact = match owner {
                Some((instant, motion)) => Some((layer.exact_local(instant)?, motion.rate() * ieee(layer.source.speed)?, motion.exact)),
                None => None,
            };
            if let (Some((local, rate, _)), Some((start, end))) = (&exact, layer.loop_span()) {
                horizon.reach(local.value(), rate, start);
                horizon.reach(local.value(), rate, end);
            }
            let mut tracks = Vec::with_capacity(wanted.tracks.len());
            for track in &wanted.tracks {
                let mut upper = None;
                if let (Some((local, rate, _)), Some(keys)) = (&exact, &self.track_at(track.flat).discrete_keys) {
                    if let Some(key) = keys.next(local, *rate > zero()) { horizon.reach(local.value(), rate, key); }
                    upper = Some(keys.upper(local));
                }
                let absent = track.gain.is_some_and(|node| states[node].local.is_none());
                tracks.push(BlockTrack { flat: track.flat, gain: track.gain, upper, absent });
            }
            let owner = match (layer.owner_node, exact) {
                (Some(_), Some((local, rate, true))) if spans => Some(Carried::Along(Progression::new(local.value().clone(), rate / &hz))),
                (Some(node), Some((local, _, exact))) => {
                    if !exact { curved.push(Curved::Layer { index, layer: wanted.layer, node, degree: motions[node].degree }); }
                    Some(Carried::One(local.numeric()?))
                }
                _ => None,
            };
            layers.push(BlockLayer { layer: wanted.layer, owner, tracks });
        }
        // A decision that changes at instant `at` changes for the first sample at or after it.
        let mut end = if spans { horizon.end(first, last) } else { first + 1 };
        // A curved quantity of degree d is the polynomial through its d + 1 first values in the
        // block. Carrying it pays only while reading it is cheaper than evaluating the clock:
        // - finding it takes those evaluations and about d * d exact operations, and saves one
        //   evaluation at each further sample, so the block needs samples left over after the
        //   first d + 1 and at least d * d of them;
        // - it is read in machine integers or not at all: its coefficients carry the sample
        //   rate to the power d in their denominator, and past machine integers every read is
        //   big rational arithmetic on a long polynomial, slower than the evaluation it
        //   replaces. The block is shortened until every polynomial is read that way at every
        //   sample (`Progression::fits_through`); if that leaves too few, the same pieces are
        //   not tried again before the next decision, so a failed attempt costs at most d
        //   evaluations per d * d samples.
        // Any other block is read one exact sample at a time.
        let degree = curved.iter().map(Curved::degree).max().unwrap_or(0);
        let least = (degree as u64).saturating_add(2).max((degree as u64).saturating_mul(degree as u64));
        let affordable = least <= (end - first) as u64 && first >= scratch.refused_until;
        if !curved.is_empty() && !affordable {
            scratch.refused_until = scratch.refused_until.max(end);
            end = first + 1;
        } else if !curved.is_empty() {
            // Per curved quantity: its values at the block's first samples (a voice: then its
            // rates, then its gate).
            let mut values: Vec<[Vec<Rational>; 3]> = curved.iter().map(|_| Default::default()).collect();
            let mut admitted = true;
            for step in 0..=degree {
                // A value past machine integers already rules the polynomial out.
                if !admitted { break; }
                if step > 0 {
                    let instant = Germ::right(Rational::new((first + step as i64).into(), SAMPLE_RATE.into()));
                    self.sound_states(&sound.nodes, &instant, &mut scratch.states, true)?;
                }
                for (curved, values) in curved.iter().zip(&mut values) {
                    if step > curved.degree() { continue; }
                    let local = scratch.states[curved.node()].local.as_ref().context("a clock present at a block's first sample is present inside it")?;
                    match curved {
                        Curved::Voice { chain, .. } => {
                            values[0].push(local.value().clone());
                            values[1].push(local.rate());
                            values[2].push(self.sound_gate(chain, &scratch.states)?);
                        }
                        Curved::Layer { layer, .. } => values[0].push(self.layers[*layer].exact_local(local)?.value().clone()),
                    }
                    admitted &= values.iter().filter_map(|values| values.last()).all(Progression::admits);
                }
            }
            let through: Vec<Vec<Progression>> = if admitted {
                values.iter().map(|values| values.iter().filter(|values| !values.is_empty()).map(|values| Progression::through(values)).collect()).collect()
            } else { Vec::new() };
            let mut span = if admitted { end - first } else { 0 };
            // `fits_through` covers every sample up to the furthest and both readers.
            while span as u64 >= least && !through.iter().flatten().all(|polynomial| polynomial.fits_through(span - 1)) { span /= 2; }
            if span as u64 >= least {
                end = first + span;
                for (curved, through) in curved.iter().zip(through) {
                    let mut through = through.into_iter();
                    match curved {
                        Curved::Voice { index, .. } => {
                            let clock = clocks[*index].as_mut().expect("a curved voice has a clock");
                            clock.line = Carried::Along(through.next().expect("a voice's values"));
                            clock.speed = Carried::Along(through.next().expect("a voice's rates"));
                            clock.gate = through.next();
                        }
                        Curved::Layer { index, .. } => layers[*index].owner = through.next().map(Carried::Along),
                    }
                }
            } else {
                scratch.refused_until = scratch.refused_until.max(end);
                end = first + 1;
            }
        }
        if remapped {
            self.sample_where_request(scene.numeric()?, Some(&vork_time::Request::Exact(scene.clone())), &mut scratch.samples,
                |track| matches!(track.target, Some(v2::TrackTarget::ImageTime {..})))?;
            for (clock, remap) in clocks.iter_mut().zip(remaps) {
                if let (true, Some((name, time))) = (clock.is_some(), remap) {
                    let local = self.footage_time(&scratch.samples, &scene, name, time.as_ref())?
                        .context("present sound has no picture clock")?;
                    *clock = if local.rate() == zero() { None } else {
                        Some(SoundClock { first, line: Carried::One(local.seconds()?), speed: Carried::One(Germ::point(local.rate()).seconds()?), gate: None, selected: None })
                    };
                }
            }
        }
        for (voice, clock) in clocks.iter_mut().enumerate() {
            if let (Some(clock), Some(movie)) = (clock, movie(voice)) {
                clock.selected = paints.windows(voice, movie, self, first, end, scratch)?;
            }
        }
        Ok(SoundBlock { first, end, clocks, layers })
    }
    /// The product of the raw rates of the clocks in `chain`: zero exactly where `clock_state`
    /// finds one of them without speed, which is where it silences everything below.
    fn sound_gate(&self, chain: &[usize], states: &[State]) -> Result<Rational> {
        let mut gate = one();
        for &node in chain {
            gate *= states[node].raw.as_ref().context("a clock running at a block's first sample runs inside it")?.rate();
        }
        Ok(gate)
    }
}

impl SoundBlock {
    /// One past the block's last output sample.
    pub(crate) fn end(&self) -> i64 { self.end }
    /// The clock of `voices[voice]` while it sounds: `None` while it is absent, held or standing.
    pub(crate) fn clock(&self, voice: usize) -> Option<&SoundClock> { self.clocks[voice].as_ref() }
}

mod gain;
pub(crate) use gain::{Gain, GainTile};

/// The picture's content tracks for one item, the weights that drive them, and
/// the clocks those reads depend on. Each item advances on its own horizon.
pub(crate) struct BindingPlan {
    layers: Vec<(usize, Vec<(usize, bool)>)>,
    nodes: Vec<usize>,
    spans: bool,
}

impl BindingPlan {
    pub(crate) fn is_empty(&self) -> bool { self.layers.is_empty() }
}

impl Plan {
    pub(crate) fn binding_plan(&self, item: usize, tracks: impl Iterator<Item = u32>) -> BindingPlan {
        let mut selected: std::collections::BTreeSet<_> = tracks.map(|i| i as usize).collect();
        let mut wanted: std::collections::BTreeSet<_> = selected.iter().map(|&i| self.layer_of(i)).collect();
        // Weight controllers precede the layers they drive in evaluation order.
        for (i, layer) in self.layers.iter().enumerate().rev() {
            for (offset, track) in layer.tracks.iter().enumerate() {
                if track.weight_target.is_some_and(|target| wanted.contains(&target)) {
                    selected.insert(self.layer_starts[i] + offset);
                    wanted.insert(i);
                }
            }
        }
        let mut nodes = vec![item];
        let layers = wanted.into_iter().map(|index| {
            let layer = &self.layers[index];
            nodes.extend(layer.owner_node);
            let start = self.layer_starts[index];
            let tracks = selected.range(start..start + layer.tracks.len()).map(|&flat| {
                nodes.extend(self.subjects[flat]);
                let constant = match &self.track_at(flat).channel {
                    Channel::Numeric(c) => c.times.len() == 1
                        || (c.interpolation == Interpolation::Step && c.values.chunks_exact(c.width).all(|v| v == &c.values[..c.width]))
                        || (c.interpolation == Interpolation::Linear && c.values.iter().all(|&v| v == 0.)),
                    Channel::Style(s) => s.values.windows(2).all(|p| p[0] == p[1]),
                    Channel::Discrete { values, .. } => values.windows(2).all(|p| p[0] == p[1]),
                };
                (flat, constant)
            }).collect();
            (index, tracks)
        }).collect();
        let (nodes, spans) = self.sound_nodes(nodes);
        BindingPlan { layers, nodes, spans }
    }

    pub(crate) fn sample_binding(&self, binding: &BindingPlan, scene: &Germ, first: i64, last: i64,
        scope_rate: Option<&Rational>, scratch: &mut SoundScratch, samples: &mut Samples) -> Result<i64> {
        self.start_sample(samples, binding.layers.iter().map(|(layer, _)| *layer));
        if binding.is_empty() { return Ok(last); }
        let now = Rational::new(first.into(), SAMPLE_RATE.into());
        let mut horizon = Horizon { now: &now, at: None };
        // An enclosing sound block bounds the composition clock until its own next decision.
        let motion = scope_rate.map_or_else(Motion::scene, |rate| Motion {
            falling: rate < &zero(), bound: if rate < &zero() { -rate } else { rate.clone() },
            ..Motion::scene()
        });
        let mut spans = self.sound_horizon(&binding.nodes, scene, scratch, last > first + 1 && binding.spans, &motion, &mut horizon)?;
        let time = scene.numeric()?;
        for (index, tracks) in &binding.layers {
            let layer = &self.layers[*index];
            let owner = match layer.owner_node {
                None => Some(scene),
                Some(node) => scratch.states[node].local.as_ref(),
            };
            let local = owner.map(|t| layer.exact_local(t)).transpose()?;
            let owner_local = local.as_ref().map(Germ::numeric).transpose()?;
            let rate = layer.owner_node.map_or_else(|| motion.rate(), |node| scratch.motions[node].rate()) * ieee(layer.source.speed)?;
            if let (Some(local), Some((a,b))) = (&local, layer.loop_span()) {
                horizon.reach(local.value(), &rate, a);
                horizon.reach(local.value(), &rate, b);
            }
            let numeric = match layer.owner_node {
                Some(_) => owner_local,
                None => Some(layer.source.offset + time * layer.source.speed),
            }.map(|t| layer.looped(t));
            self.sample_layer(*index, self.layer_starts[*index], numeric, tracks.iter().copied(),
                |_, flat, track, constant| {
                    if self.subjects[flat].is_some_and(|node| scratch.states[node].local.is_none()) { return Ok(None); }
                    let local = local.as_ref().expect("a sampled layer has a clock");
                    let upper = track.discrete_keys.as_ref().map(|keys| keys.upper(local));
                    if spans && rate != zero() && !constant {
                        if let Some(keys) = &track.discrete_keys {
                            if let Some(next) = keys.next(local, rate > zero()) { horizon.reach(local.value(), &rate, next); }
                            let i = upper.unwrap();
                            if let Channel::Style(style) = &track.channel
                                && !track.channel.constant_at(i) && style.values[i-1] != style.values[i]
                            {
                                if let Some(times) = &style.clocks
                                    && style.values[i-1].discrete_with(&style.values[i], style.kind)
                                {
                                    let middle = (times[i-1].exact() + times[i].exact()) / Rational::from_integer(2.into());
                                    horizon.reach(local.value(), &rate, &middle);
                                } else { spans = false; }
                            }
                        } else {
                            // Numeric interpolation uses the writer's f32 clock. After its
                            // final key in the travel direction it holds until a clock decision.
                            let t = layer.looped(if layer.owner_node.is_some() { owner_local.unwrap() }
                                else { layer.source.offset + time * layer.source.speed });
                            let times = track.channel.times();
                            spans = !layer.source.looping && if rate > zero() { t >= *times.last().unwrap() } else { t <= times[0] };
                        }
                    }
                    Ok(Some(TrackRead { upper, exact_local: Some(local) }))
                }, samples)?;
        }
        Ok(if spans { horizon.end(first, last) } else { first + 1 })
    }
}
