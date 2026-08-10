use crate::pow_table;

/// Bits of the f64 nearest sqrt(0.5), the cut point for reducing a positive
/// normal x to x = 2^k * m with m in [sqrt(0.5), sqrt(2)).
const REDUCE_OFF: u64 = 0x3fe6_a09e_667f_3bcd;

#[inline(always)]
fn reduced_pow(x: f64, coefficients: &[f64; 17]) -> (i32, f64) {
    let bits = x.to_bits();
    let tmp = bits.wrapping_sub(REDUCE_OFF);
    let k = (tmp as i64 >> 52) as i32;
    let m = f64::from_bits(bits.wrapping_sub(tmp & (0xfff << 52)));

    // Degree-17 minimax polynomial for m^p in t = m - 1. The constant
    // coefficient is exactly one. Estrin evaluation keeps the dependency
    // chain short.
    let c = coefficients;
    let t = m - 1.0;
    let t2 = t * t;
    let t4 = t2 * t2;
    let t8 = t4 * t4;
    let p01 = 1.0 + t * c[0];
    let p23 = c[1] + t * c[2];
    let p45 = c[3] + t * c[4];
    let p67 = c[5] + t * c[6];
    let p89 = c[7] + t * c[8];
    let p1011 = c[9] + t * c[10];
    let p1213 = c[11] + t * c[12];
    let p1415 = c[13] + t * c[14];
    let p1617 = c[15] + t * c[16];
    let q0_3 = p01 + t2 * p23;
    let q4_7 = p45 + t2 * p67;
    let q8_11 = p89 + t2 * p1011;
    let q12_15 = p1213 + t2 * p1415;
    let r0_7 = q0_3 + t4 * q4_7;
    let r8_15 = q8_11 + t4 * q12_15;
    let m_p = r0_7 + t8 * (r8_15 + t8 * p1617);
    (k, m_p)
}

#[inline(always)]
fn pow_fixed(x: f64, coefficients: &[f64; 17], exponent_table: &[u64; 2047]) -> f64 {
    debug_assert!(x == 0.0 || (x.is_sign_positive() && x.is_normal()));
    if x == 0.0 {
        return 0.0;
    }
    let (k, m_p) = reduced_pow(x, coefficients);
    m_p * f64::from_bits(exponent_table[(k + 1022) as usize])
}

#[inline(always)]
fn pow_fixed_split(x: f64, coefficients: &[f64; 17], half_exponent_table: &[u64; 1024]) -> f64 {
    debug_assert!(x == 0.0 || (x.is_sign_positive() && x.is_normal()));
    if x == 0.0 {
        return 0.0;
    }
    let (k, m_p) = reduced_pow(x, coefficients);
    let a = k >> 1;
    let b = k - a;
    let scale_a = f64::from_bits(half_exponent_table[(a + 511) as usize]);
    let scale_b = f64::from_bits(half_exponent_table[(b + 511) as usize]);
    (m_p * scale_a) * scale_b
}

/// x^(1/20), specialized for the energy-dependent spreading slope.
#[inline(always)]
pub(crate) fn pow_005(x: f64) -> f64 {
    pow_fixed(x, &pow_table::P005_POLY, &pow_table::P005_EXP_TABLE)
}

/// x^(2/5), specialized for spreading in the loudness domain.
#[inline(always)]
pub(crate) fn pow_04(x: f64) -> f64 {
    pow_fixed(x, &pow_table::P04_POLY, &pow_table::P04_EXP_TABLE)
}

/// x^(3/10), specialized for modulation processing.
#[inline(always)]
pub(crate) fn pow_03(x: f64) -> f64 {
    pow_fixed(x, &pow_table::P03_POLY, &pow_table::P03_EXP_TABLE)
}

/// x^1.71332, specialized for the detection threshold calculation.
#[inline(always)]
pub(crate) fn pow_171332(x: f64) -> f64 {
    pow_fixed_split(
        x,
        &pow_table::P171332_POLY,
        &pow_table::P171332_HALF_EXP_TABLE,
    )
}

#[cfg(test)]
mod tests {
    use super::{pow_03, pow_04, pow_005, pow_171332};

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
        for kernel in [pow_005 as fn(f64) -> f64, pow_04, pow_03, pow_171332] {
            assert_eq!(kernel(0.0), 0.0);
            assert_eq!(kernel(1.0), 1.0);
        }
    }

    #[test]
    fn accuracy_sweep() {
        let mut rng = Rng(0x243f_6a88_85a3_08d3);
        let mut max_relative_error = [0.0f64; 4];
        for _ in 0..1_000_000 {
            let exponent = rng.next() % 2046 + 1;
            let mantissa = rng.next() & 0xf_ffff_ffff_ffff;
            let x = f64::from_bits((exponent << 52) | mantissa);
            for (index, power, kernel) in [
                (0, 0.05, pow_005 as fn(f64) -> f64),
                (1, 0.4, pow_04 as fn(f64) -> f64),
                (2, 0.3, pow_03 as fn(f64) -> f64),
                (3, 1.71332, pow_171332 as fn(f64) -> f64),
            ] {
                let expected = x.powf(power);
                let actual = kernel(x);
                if expected.is_infinite() {
                    assert!(actual.is_infinite());
                } else if expected >= f64::MIN_POSITIVE {
                    let relative_error = ((actual - expected) / expected).abs();
                    max_relative_error[index] = max_relative_error[index].max(relative_error);
                } else {
                    let absolute_error = (actual - expected).abs();
                    assert!(
                        absolute_error <= 16.0 * f64::from_bits(1),
                        "x^{power} subnormal absolute error: {absolute_error:e}"
                    );
                }
            }
        }
        for (power, error) in [0.05, 0.4, 0.3, 1.71332]
            .into_iter()
            .zip(max_relative_error)
        {
            assert!(
                error < 3.0e-15,
                "x^{power} maximum relative error: {error:e}"
            );
        }
    }
}
