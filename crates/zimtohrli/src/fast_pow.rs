use crate::pow_pp_table;

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

#[cfg(test)]
mod tests {
    use super::pow_pp;

    const PP: f64 = 0.32264042946823823;

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
}
