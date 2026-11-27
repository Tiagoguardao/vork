//! Authored seconds retain their spelling for serialization. Discrete
//! consumers read the exact clock value.
use anyhow::{Context, Result, ensure};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use crate::exact::{Rational, decimal};

#[derive(Clone, Debug, PartialEq)]
pub struct Clock {
    spelling: std::sync::Arc<str>,
    numeric: f64,
    binary: bool,
}
impl Clock {
    pub fn exact(&self) -> Rational {
        if self.binary {
            Rational::from_float(self.numeric).expect("finite buffer clock")
        } else if self.spelling.contains('/') {
            decimal(&self.spelling).expect("validated clock")
        } else {
            simplest_binary64(self.numeric)
        }
    }
    pub fn binary(value:f32)->Self {Self {spelling:value.to_string().into(),numeric:f64::from(value),binary:true}}
    pub fn as_f64(&self) -> f64 { self.numeric }
    pub fn as_f32(&self) -> f32 { self.numeric as f32 }
}
impl std::str::FromStr for Clock {
    type Err = anyhow::Error;
    fn from_str(text: &str) -> Result<Self> {
        let numeric = if text.contains('/') {decimal(text)?.to_f64().unwrap_or(f64::INFINITY)} else {
            let number:serde_json::Number = text.parse()?;
            number.as_f64().context("numeric clock must fit a finite binary64 value")?
        };
        Ok(Self { spelling: text.into(), numeric, binary:false })
    }
}

/// The minimum-denominator rational in this binary64 rounding cell, choosing
/// the smallest absolute numerator if there is a tie.
fn simplest_binary64(value: f64) -> Rational {
    // In this range an integral value is the closest integer to zero in its cell.
    if value.abs() <= (1_u64 << f64::MANTISSA_DIGITS) as f64 && value.fract() == 0. {
        return Rational::from_integer(BigInt::from(value as i64));
    }
    let bits = value.abs().to_bits();
    let exponent = (bits >> 52) as i32;
    let fraction = bits & ((1 << 52) - 1);
    let significand = fraction | (u64::from(exponent != 0) << 52);
    // In quarter-ULPs the adjacent midpoints share a power-of-two denominator.
    // Only a normal power of two above the smallest normal has a closer predecessor.
    let lower = 4 * significand - if fraction == 0 && exponent > 1 { 1 } else { 2 };
    let upper = 4 * significand + 2;
    let shift = 1023 + 52 + 2 - exponent.max(1);
    let closed = significand & 1 == 0;
    let (numerator, denominator) = if shift <= 0 {
        // Both endpoints are integers. Pick the first integer admitted by the cell.
        ((BigInt::from(lower) << -shift) + u8::from(!closed), BigInt::from(1))
    } else if let Some(scale) = 1_u64.checked_shl(shift as u32) {
        let (n, d) = simplest_in_cell(lower, upper, scale, closed);
        (BigInt::from(n), BigInt::from(d))
    } else {
        let scale = num_bigint::BigUint::from(1_u8) << shift;
        let (n, d) = simplest_in_cell(lower.into(), upper.into(), scale, closed);
        (BigInt::from(n), BigInt::from(d))
    };
    Rational::new_raw(if value.is_sign_negative() { -numerator } else { numerator }, denominator)
}

// For a Stern–Brocot bound p/q, the distances to the two endpoints are
// integer cross-products |endpoint*q - scale*p|. Adding the other bound
// subtracts its distances in reverse order. The distances only decrease,
// and ancestors of the cell center have p and q no larger than its terms.
struct FareyBound<T> {
    p: T,
    q: T,
    near: T,
    far: T,
}

impl<T: num_integer::Integer + Clone> FareyBound<T> {
    fn advance(&mut self, other: &Self, closed: bool) -> bool {
        if self.near < other.far || (closed && self.near == other.far) {
            return false;
        }
        // Skip all mediants on this side. An included midpoint ends the search.
        let distance = if closed { self.near.clone() - T::one() } else { self.near.clone() };
        let steps = distance / other.far.clone();
        self.near = self.near.clone() - steps.clone() * other.far.clone();
        self.far = self.far.clone() - steps.clone() * other.near.clone();
        self.p = self.p.clone() + steps.clone() * other.p.clone();
        self.q = self.q.clone() + steps * other.q.clone();
        true
    }
}

fn simplest_in_cell<T: num_integer::Integer + Clone>(lower: T, upper: T, scale: T, closed: bool) -> (T, T) {
    let mut left = FareyBound { p: T::zero(), q: T::one(), near: lower, far: upper };
    let mut right = FareyBound { p: T::one(), q: T::zero(), near: scale.clone(), far: scale };
    while left.advance(&right, closed) || right.advance(&left, closed) {}
    // The first admitted mediant is reduced and has minimum denominator.
    (left.p + right.p, left.q + right.q)
}

impl From<f64> for Clock {
    fn from(value: f64) -> Self { value.to_string().parse().expect("finite clock") }
}
impl From<f32> for Clock {
    fn from(value: f32) -> Self { value.to_string().parse().expect("finite clock") }
}
impl Serialize for Clock {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.binary {let value=self.exact(); return serializer.serialize_str(&format!("{}/{}",value.numer(),value.denom()))}
        if self.spelling.contains('/') { serializer.serialize_str(&self.spelling) }
        else { self.spelling.parse::<serde_json::Number>().map_err(serde::ser::Error::custom)?.serialize(serializer) }
    }
}
impl<'de> Deserialize<'de> for Clock {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        let text = match value {
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::String(s) if s.contains('/') => s,
            _ => return Err(serde::de::Error::custom("clock must be numeric seconds or a rational string")),
        };
        text.parse().map_err(serde::de::Error::custom)
    }
}
impl schemars::JsonSchema for Clock {
    fn schema_name() -> std::borrow::Cow<'static, str> { "Clock".into() }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({"x-clock":true,"anyOf":[{"type":"number"},{"type":"string","pattern":"^-?[0-9]+/[1-9][0-9]*$"}]})
    }
}

/// Internal request identity. Ordinary evaluation lowers directly; only a
/// discrete consumer materializes unbounded arithmetic.
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    Frame(u32, crate::Time),
    Tick(u32, Clock, Clock),
    Authored(Clock),
    Numeric(f32),
    Exact(crate::exact::Germ),
}
impl Request {
    pub fn numeric(&self) -> Result<f32> {
        let value = match self {
            Self::Frame(n, fps) => frame_numeric(*n,*fps)?,
            Self::Tick(..) => return self.exact()?.numeric(),
            Self::Authored(t) => return crate::exact::Germ::point(t.exact()).numeric(),
            Self::Numeric(t) => *t,
            Self::Exact(t) => return t.numeric(),
        };
        ensure!(value.is_finite(), "time exceeds finite numeric evaluation range");
        Ok(value)
    }
    /// Absolute output frame nearest this instant. A point exactly halfway
    /// between two frames belongs to the later frame.
    pub fn nearest_frame(&self, fps: crate::Time) -> Result<u32> {
        use crate::exact::Germ;
        if let Self::Frame(index, rate) = self && *rate == fps {
            return Ok(*index);
        }
        let rate = fps.exact();
        let half = Germ::point(Rational::new(1.into(), 2.into()));
        let index = self.exact()?.scale(&rate).add(&half).floor();
        u32::try_from(index).context("frame index exceeds u32 identity capacity")
    }

    pub fn exact(&self) -> Result<crate::exact::Germ> {
        use crate::exact::{Germ, ieee};
        Ok(match self {
            Self::Frame(n, fps) => Germ::point(Rational::new(num_bigint::BigInt::from(*n) * fps.denominator(), fps.numerator().into())),
            Self::Tick(k,start,dt) => Germ::point(start.exact()+Rational::from_integer((*k).into())*dt.exact()),
            Self::Authored(t) => Germ::point(t.exact()),
            Self::Numeric(t) => Germ::point(ieee(*t)?),
            Self::Exact(t) => t.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nearest_frame_uses_the_exact_instant_and_output_rate() -> Result<()> {
        use crate::exact::Germ;
        for rate in ["24", "30", "23.976", "29.97", "24000/1001", "30000/1001"] {
            let fps = rate.parse()?;
            for lower in [1u32, 16_777_216, u32::MAX - 1] {
                let halfway = Rational::new((2 * u64::from(lower) + 1).into(), 2.into()) / decimal(rate)?;
                for (instant, expected) in [
                    (Germ::left(halfway.clone()), lower),
                    (Germ::point(halfway.clone()), lower + 1),
                    (Germ::right(halfway), lower + 1),
                ] {
                    assert_eq!(Request::Exact(instant).nearest_frame(fps)?, expected);
                }
            }
            assert_eq!(Request::Frame(u32::MAX, fps).nearest_frame(fps)?, u32::MAX);
        }
        let fps = crate::Time::integer(24);
        for (at, index) in [("0.0624999999", 1), ("0.0625", 2), ("0.0625000001", 2)] {
            assert_eq!(Request::Authored(at.parse()?).nearest_frame(fps)?, index);
        }
        assert_eq!(Request::Frame(3, crate::Time::integer(48)).nearest_frame(fps)?, 2);
        assert!(Request::Authored("4294967296".parse()?).nearest_frame(crate::Time::ONE).is_err());
        Ok(())
    }

    #[test]
    fn bounded_frame_lowering_matches_unbounded_rounding() -> Result<()> {
        for rate in ["24", "30000/1001", "60000/1001", "1/9223372036854775807", "9223372036854775807"] {
            let rate=rate.parse()?;
            for n in [0,1,3,7,107892,16777217,u32::MAX] {
                let request=Request::Frame(n,rate);
                assert_eq!(request.numeric()?.to_bits(),request.exact()?.numeric()?.to_bits(),"{n} at {rate}");
            }
        }
        for value in [0.,1.,0.1,-1.] {
            let clock=Clock::binary(value);
            assert_eq!(serde_json::from_str::<Clock>(&serde_json::to_string(&clock)?)?.exact(),clock.exact());
        }
        Ok(())
    }
    #[test]
    fn authored_clock_roundtrips_preserving_its_spelling() -> Result<()> {
        for text in ["0.016666666666666666", "1e-100", "\"1/120\"", "3600"] {
            let clock: Clock=serde_json::from_str(text)?;
            assert_eq!(serde_json::to_string(&clock)?, text);
        }
        let fps:crate::Time="30000/1001".parse()?;
        let clock:Clock=serde_json::from_str("3600")?;
        let count=clock.exact()*decimal(&fps.to_string())?;
        assert_eq!(count.ceil().to_integer(),107893.into());
        for n in [1,5,7,10,17,107892,u32::MAX] {
            let request=Request::Frame(n,fps);
            assert_eq!(request.exact()?.scale(&decimal(&fps.to_string())?).floor(),n.into());
        }
        Ok(())
    }

    #[test]
    fn numeric_clocks_recover_fractions_and_respect_adjacent_values() -> Result<()> {
        let expected = decimal("5/12")?;
        for text in ["0.4166666666666667", "0.41666666666666670", "4166666666666667e-16"] {
            assert_eq!(serde_json::from_str::<Clock>(text)?.exact(), expected);
        }
        assert_eq!(serde_json::from_str::<Clock>("-0.4166666666666667")?.exact(), -&expected);
        let offset: Clock = "0.4166666666666667".parse()?;
        for (adjacent, order) in [
            (offset.as_f64().next_down(), std::cmp::Ordering::Less),
            (offset.as_f64().next_up(), std::cmp::Ordering::Greater),
        ] {
            let clock = Clock::from(adjacent);
            assert_eq!(clock.exact().cmp(&expected), order);
            assert_eq!(clock.exact().to_f64(), Some(adjacent));
        }
        assert_eq!(serde_json::from_str::<Clock>("\"4166666666666667/10000000000000000\"")?.exact(),
            decimal("0.4166666666666667")?);
        Ok(())
    }

    #[test]
    fn numeric_clock_fractions_span_the_native_integer_boundary() {
        // The common dyadic scale needs 63 bits of shift at 2^-9 and 64 just below.
        for (n, d) in [(1, 511), (1, 512), (1, 513), (355, 113), (4181, 6765)] {
            let value = f64::from(n) / f64::from(d);
            let expected = Rational::new(n.into(), d.into());
            assert_eq!(simplest_binary64(value), expected);
            assert_eq!(simplest_binary64(-value), -expected);
        }
    }

    #[test]
    fn numeric_clocks_handle_ties_and_binary64_limits() -> Result<()> {
        // Spacing becomes two at 2^53. An even significand includes integer
        // midpoints and an odd significand excludes them.
        for (input, expected) in [
            ("9007199254740992", "9007199254740992"),
            ("9007199254740994", "9007199254740994"),
            ("9007199254740996", "9007199254740995"),
            ("-9007199254740996", "-9007199254740995"),
        ] {
            assert_eq!(serde_json::from_str::<Clock>(input)?.exact(), decimal(expected)?);
        }
        // For the smallest subnormal, the upper midpoint is excluded. The
        // first unit fraction below it is the simplest in the cell.
        let tiny: Clock = "5e-324".parse()?;
        let denominator = (BigInt::from(1) << 1075_u32) / 3 + 1;
        assert_eq!(tiny.exact(), Rational::new(BigInt::from(1), denominator));
        assert_eq!(serde_json::from_str::<Clock>("-1e-324")?.exact(), decimal("0")?);
        let max = Clock::from(f64::MAX);
        assert!(max.exact().is_integer());
        assert_eq!(max.exact().to_f64(), Some(f64::MAX));
        assert!(serde_json::from_str::<Clock>("1e309").is_err());
        Ok(())
    }
}

impl Request {
    pub fn previous(&self,fps:crate::Time)->Result<Self> {
        if let Self::Frame(n,rate)=self {
            if rate==&fps {return Ok(Self::Frame(n.saturating_sub(1),fps))}
        }
        let value=self.exact()?.sub(&crate::exact::Germ::point(decimal(&fps.to_string())?.recip()));
        Ok(Self::Exact(if value.compare_point(&Rational::from_integer(0.into())).is_lt() {
            crate::exact::Germ::point(Rational::from_integer(0.into()))
        }else{value}))
    }
}

impl From<f32> for Request { fn from(time:f32)->Self {Self::Numeric(time)} }
impl From<&Request> for Request { fn from(time:&Request)->Self {time.clone()} }
impl From<&Clock> for Request { fn from(time:&Clock)->Self {Self::Authored(time.clone())} }

/// The frame ABI bounds n to u32 and rate terms to positive i64. Products
/// therefore use at most 95 bits. Scaling to a 24-bit significand uses at most
/// max(95, 63+24) bits; u128 arithmetic cannot overflow here.
fn frame_numeric(index:u32,rate:crate::Time)->Result<f32> {
    ensure!(rate>crate::Time::ZERO,"frame rate must be positive");
    if index==0 {return Ok(0.)}
    let mut n=u128::from(index)*rate.denominator() as u128;
    let mut d=rate.numerator() as u128;
    let mut exponent=n.ilog2() as i32-d.ilog2() as i32;
    if if exponent>=0 {n<(d<<exponent)}else{(n<<(-exponent))<d} {exponent-=1;}
    let shift=23-exponent;
    if shift>=0 {n<<=shift;}else{d<<=-shift;}
    let mut significand=n/d;
    let remainder=n%d;
    if remainder*2>d || (remainder*2==d && significand&1!=0) {significand+=1;}
    if significand==1<<24 {significand>>=1;exponent+=1;}
    Ok(f32::from_bits(((exponent+127) as u32)<<23 | (significand as u32&0x7fffff)))
}
