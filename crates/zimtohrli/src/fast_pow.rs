use crate::{pow_nsim_table, pow_pp_table};

/// Bits of the f64 nearest sqrt(0.5), the cut point of the range reduction:
/// mantissas at or above sqrt(2) are folded into the next binade so that
/// x = 2^k * m with m in [sqrt(0.5), sqrt(2)).
const REDUCE_OFF: u64 = 0x3fe6a09e667f3bcd;

/// Computes x^PP for the DTW cost exponent PP = 0.32264042946823823,
/// deterministically. x must be zero or a positive normal f64; out-of-domain
/// inputs panic on the table bounds check or (for some subnormals) silently
/// return a wrong value.
///
/// Useful because the DTW path computes a *lot* of these, and this takes up
/// ~half the runtime of delta_norm.
#[inline(always)]
pub(crate) fn pow_pp(x: f64) -> f64 {
    debug_assert!(x == 0.0 || (x.is_sign_positive() && x.is_normal()));

    if x == 0.0 {
        return 0.0;
    }

    // Range reduction: x = 2^k * m with m in [sqrt(0.5), sqrt(2)), so
    // k in [-1022, 1024] for any positive normal x.
    let bits = x.to_bits();
    let tmp = bits.wrapping_sub(REDUCE_OFF);
    let k = (tmp as i64) >> 52;
    let m = f64::from_bits(bits.wrapping_sub(tmp & (0xfff << 52)));
    let scale = f64::from_bits(pow_pp_table::EXP_TABLE[(k + 1022) as usize]);

    // m^PP as a degree-17 minimax polynomial in t = m - 1 (exact by
    // Sterbenz), with constant term exactly 1. Estrin evaluation to shorten
    // the serial dependency chain; C[i] is the coefficient of t^(i+1).
    const C: [f64; 17] = pow_pp_table::POLY;
    let t = m - 1.0;
    let t2 = t * t;
    let t4 = t2 * t2;
    let t8 = t4 * t4;
    let p01 = 1.0 + t * C[0];
    let p23 = C[1] + t * C[2];
    let p45 = C[3] + t * C[4];
    let p67 = C[5] + t * C[6];
    let p89 = C[7] + t * C[8];
    let p1011 = C[9] + t * C[10];
    let p1213 = C[11] + t * C[12];
    let p1415 = C[13] + t * C[14];
    let p1617 = C[15] + t * C[16];
    let q0_3 = p01 + t2 * p23;
    let q4_7 = p45 + t2 * p67;
    let q8_11 = p89 + t2 * p1011;
    let q12_15 = p1213 + t2 * p1415;
    let r0_7 = q0_3 + t4 * q4_7;
    let r8_15 = q8_11 + t4 * q12_15;
    let m_pp = r0_7 + t8 * (r8_15 + t8 * p1617);

    m_pp * scale
}

/// Bits of the f32 nearest sqrt(0.5), the cut point of the f32 range
/// reduction: mantissas at or above sqrt(2) are folded into the next binade
/// so that x = 2^k * m with m in [sqrt(0.5), sqrt(2)).
const REDUCE_OFF_F32: u32 = 0x3f35_04f3;

/// Range reduction shared by the fixed-exponent f32 kernels: splits a
/// positive normal x into x = 2^k * m with m in [sqrt(0.5), sqrt(2)) and
/// k in -126..=128.
#[inline(always)]
fn reduce_f32(x: f32) -> (i32, f32) {
    let bits = x.to_bits();
    let tmp = bits.wrapping_sub(REDUCE_OFF_F32);
    let k = (tmp as i32) >> 23;
    let m = f32::from_bits(bits.wrapping_sub(tmp & (0x1ff << 23)));
    (k, m)
}

/// Scales m^P (m in [sqrt(0.5), sqrt(2))) by 2^(P*k), split into two table
/// entries 2^(P*a) * 2^(P*b) with a + b = k, multiplied as
/// (m^P * 2^(P*a)) * 2^(P*b). A direct 2^(P*k) f32 scale cannot work: with
/// P0 > 1 the range of 2^(P0*k) is 2^-132..2^134, while x^P0 itself stays
/// representable for some of those k (x = 0.71 * 2^122 gives x^P0 = 2^127.6,
/// a normal f32) — and likewise multiplying the two halves first would
/// overflow before m^P < 1 can bring it back. The half-exponent factors and
/// the intermediate product stay normal over all k in -126..=128, and the
/// final multiply rounds into the subnormal/inf range as needed, at a cost
/// of ~1 ulp relative to a single lookup.
#[inline(always)]
fn scale_f32(m_p: f32, table: &'static [u32; 255], k: i32) -> f32 {
    let a = k >> 1;
    let b = k - a;
    (m_p * f32::from_bits(table[(a + 126) as usize])) * f32::from_bits(table[(b + 126) as usize])
}

/// Computes x^P0 for the NSIM intensity exponent P0 = 1.0500187278772866,
/// deterministically. x must be zero or a positive normal f32; out-of-domain
/// inputs panic on the table bounds check or (for some subnormals) silently
/// return a wrong value. (Same contract as [pow_pp], q.v. for the design.)
///
/// Useful because the NSIM aggregation computes one of these per spectrogram
/// element; a generic `powf` was ~2/3 of the `distance_without_dtw` runtime.
#[inline(always)]
pub(crate) fn pow_p0(x: f32) -> f32 {
    debug_assert!(x == 0.0 || (x.is_sign_positive() && x.is_normal()));
    if x == 0.0 {
        return 0.0;
    }

    let (k, m) = reduce_f32(x);

    // m^P0 as a degree-6 minimax polynomial in t = m - 1 (exact by Sterbenz),
    // with constant term exactly 1. Estrin evaluation to shorten the serial
    // dependency chain; C[i] is the coefficient of t^(i+1).
    const C: [f32; 6] = pow_nsim_table::P0_POLY;
    let t = m - 1.0;
    let t2 = t * t;
    let t4 = t2 * t2;
    let q0_1 = 1.0 + t * C[0];
    let q2_3 = C[1] + t * C[2];
    let q4_5 = C[3] + t * C[4];
    let q6 = C[5];
    let m_p = (q0_1 + t2 * q2_3) + t4 * (q4_5 + t2 * q6);
    scale_f32(m_p, &pow_nsim_table::P0_EXP_TABLE, k)
}

/// Computes x^P1 for the NSIM structure exponent P1 = 0.25808223975919764,
/// deterministically. Same domain contract as [pow_p0].
#[inline(always)]
pub(crate) fn pow_p1(x: f32) -> f32 {
    debug_assert!(x == 0.0 || (x.is_sign_positive() && x.is_normal()));
    if x == 0.0 {
        return 0.0;
    }

    let (k, m) = reduce_f32(x);

    // m^P1 as a degree-8 minimax polynomial in t = m - 1; see [pow_p0].
    const C: [f32; 8] = pow_nsim_table::P1_POLY;
    let t = m - 1.0;
    let t2 = t * t;
    let t4 = t2 * t2;
    let t8 = t4 * t4;
    let q0_1 = 1.0 + t * C[0];
    let q2_3 = C[1] + t * C[2];
    let q4_5 = C[3] + t * C[4];
    let q6_7 = C[5] + t * C[6];
    let q8 = C[7];
    let m_p = (q0_1 + t2 * q2_3) + t4 * (q4_5 + t2 * q6_7) + t8 * q8;
    scale_f32(m_p, &pow_nsim_table::P1_EXP_TABLE, k)
}

#[cfg(test)]
mod tests {
    use super::{pow_p0, pow_p1, pow_pp};

    const PP: f64 = 0.32264042946823823;
    const P0: f64 = 1.0500187278772866;
    const P1: f64 = 0.25808223975919764;

    /// Deterministic xorshift.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    fn exact_cases() {
        assert_eq!(pow_pp(0.0), 0.0);
        assert_eq!(pow_pp(1.0), 1.0);
    }

    #[test]
    fn accuracy_sweep() {
        // libm::pow is within 1 ulp, so measuring against it bounds our true
        // error to max_err + ~2^-52.
        let mut rng = Rng(0x243f6a8885a308d3);
        let mut max_err_pp = 0.0f64;
        for _ in 0..2_000_000 {
            // Log-uniform over the full range delta_norm can produce
            // (sums of squared f32 differences: 0 or [2^-298, ~2^80])
            // and beyond, up to the whole normal range.
            let exponent = (rng.next() % 2045) + 1; // biased exp 1..=2045
            let mantissa = rng.next() & 0xf_ffff_ffff_ffff;
            let x = f64::from_bits((exponent << 52) | mantissa);
            let expected = libm::pow(x, PP);
            max_err_pp = max_err_pp.max(((pow_pp(x) - expected) / expected).abs());
        }
        assert!(
            max_err_pp < 1e-15,
            "pow_pp max relative error {max_err_pp:e}"
        );
    }

    #[test]
    fn exact_cases_f32() {
        assert_eq!(pow_p0(0.0), 0.0);
        assert_eq!(pow_p0(1.0), 1.0);
        assert_eq!(pow_p1(0.0), 0.0);
        assert_eq!(pow_p1(1.0), 1.0);
    }

    #[test]
    fn accuracy_sweep_f32() {
        let mut rng = Rng(0x9e3779b97f4a7c15);
        let mut max_rel_err = [0.0f64; 2];
        for _ in 0..2_000_000 {
            // Log-uniform over the positive normal f32 range, far beyond the
            // [~0.6, ~1] domain NSIM actually produces.
            let exponent = (rng.next() % 254) + 1; // biased exp 1..=254
            let mantissa = rng.next() & 0x7f_ffff;
            let x = f32::from_bits(((exponent as u32) << 23) | mantissa as u32);
            for (i, p, kernel) in [
                (0, P0, pow_p0 as fn(f32) -> f32),
                (1, P1, pow_p1 as fn(f32) -> f32),
            ] {
                let expected = libm::pow(x as f64, p);
                let got = kernel(x) as f64;
                if expected >= 2f64.powi(128) {
                    assert!(got.is_infinite(), "x^{p} overflows but got {got}");
                } else if expected >= 2f64.powi(-126) {
                    max_rel_err[i] = max_rel_err[i].max(((got - expected) / expected).abs());
                } else {
                    // Subnormal result: the f32 table entries lose mantissa
                    // bits, so only check the absolute error stays tiny.
                    assert!(
                        (got - expected).abs() < 2f64.powi(-110),
                        "x^{p} subnormal mismatch: got {got}, expected {expected}"
                    );
                }
            }
        }
        assert!(
            max_rel_err[0] < 1e-6,
            "pow_p0 max relative error {:e}",
            max_rel_err[0]
        );
        assert!(
            max_rel_err[1] < 1e-6,
            "pow_p1 max relative error {:e}",
            max_rel_err[1]
        );
    }
}
