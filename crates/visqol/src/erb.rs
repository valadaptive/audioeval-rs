//! Gammatone filter design after Slaney's "An Efficient Implementation of the
//! Patterson-Holdsworth Cochlear Filter Bank", a port of
//! `equivalent_rectangular_bandwidth.cc`.

use std::f64::consts::PI;

use num_complex::Complex64;

pub struct ErbFilters {
    /// Center frequencies, highest first (as produced by the C++).
    pub center_freqs: Vec<f64>,
    /// Per-band biquad coefficients, one entry per band, highest band first:
    /// [A0, A11, A12, A13, A14, A2, B0, B1, B2, gain].
    pub coeffs: Vec<[f64; 10]>,
}

pub fn make_filters(
    sample_rate: f64,
    num_channels: usize,
    low_freq: f64,
    mut high_freq: f64,
) -> ErbFilters {
    if high_freq > sample_rate / 2.0 {
        high_freq = sample_rate / 2.0;
    }
    let cf = calc_uniform_center_freqs(low_freq, high_freq, num_channels);

    const EAR_Q: f64 = 9.26449; // Glasberg and Moore parameters.
    const MIN_BW: f64 = 24.7;
    let order = 1.0;
    let t = 1.0 / sample_rate;

    let mut coeffs = Vec::with_capacity(num_channels);
    for &cf_i in &cf {
        let erb = ((cf_i / EAR_Q).powf(order) + MIN_BW.powf(order)).powf(1.0 / order);
        let b = 1.019 * 2.0 * PI * erb;
        let exp_bt = (b * t).exp();

        let b1 = -2.0 * (2.0 * cf_i * PI * t).cos() / exp_bt;
        let b2 = (-2.0 * b * t).exp();

        let sin_t = (cf_i * 2.0 * PI * t).sin() * t;
        let b_pos = sin_t * 2.0 * (3.0 + 2f64.powf(1.5)).sqrt();
        let b_neg = sin_t * 2.0 * (3.0 - 2f64.powf(1.5)).sqrt();
        let a = (cf_i * 2.0 * PI * t).cos() * 2.0 * t;
        let a11 = -(a / exp_bt + b_pos / exp_bt) / 2.0;
        let a12 = -(a / exp_bt - b_pos / exp_bt) / 2.0;
        let a13 = -(a / exp_bt + b_neg / exp_bt) / 2.0;
        let a14 = -(a / exp_bt - b_neg / exp_bt) / 2.0;

        // Gain requires complex arithmetic.
        let i = Complex64::I;
        let p1 = 2f64.powf(1.5);
        let s1 = (3.0 - p1).sqrt();
        let s2 = (3.0 + p1).sqrt();
        let two_pi_cf_t = 2.0 * cf_i * PI * t;
        let x_exp = (4.0 * i * cf_i * PI * t).exp();
        let x01 = -2.0 * x_exp * t;
        let x02 = 2.0 * (-(b * t) + 2.0 * i * cf_i * PI * t).exp() * t;
        let x_cos = two_pi_cf_t.cos();
        let x_sin = two_pi_cf_t.sin();

        let x1 = x01 + x02 * (x_cos - s1 * x_sin);
        let x2 = x01 + x02 * (x_cos + s1 * x_sin);
        let x3 = x01 + x02 * (x_cos - s2 * x_sin);
        let x4 = x01 + x02 * (x_cos + s2 * x_sin);
        let x5 = -2.0 / (2.0 * b * t).exp() - 2.0 * x_exp + (2.0 * (1.0 + x_exp)) / (b * t).exp();
        let gain = ((x1 * x2 * x3 * x4) / x5.powf(4.0)).norm();

        coeffs.push([t, a11, a12, a13, a14, 0.0, 1.0, b1, b2, gain]);
    }

    ErbFilters {
        center_freqs: cf,
        coeffs,
    }
}

/// Center frequencies uniformly spaced on the ERB scale, highest first.
fn calc_uniform_center_freqs(low_freq: f64, high_freq: f64, num_channels: usize) -> Vec<f64> {
    const EAR_Q: f64 = 9.26449;
    const MIN_BW: f64 = 24.7;

    let a = -(EAR_Q * MIN_BW);
    let b = -(high_freq + EAR_Q * MIN_BW).ln();
    let c = (low_freq + EAR_Q * MIN_BW).ln();
    let d = high_freq + EAR_Q * MIN_BW;
    let e = (b + c) / num_channels as f64;

    (0..num_channels)
        .map(|i| a + ((i + 1) as f64 * e).exp() * d)
        .collect()
}
