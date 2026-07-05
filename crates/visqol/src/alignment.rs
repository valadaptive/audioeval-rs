//! Global signal alignment, a port of `alignment.cc`, `envelope.cc`, and `xcorr.cc`.

use crate::audio_signal::AudioSignal;
use crate::fft;

/// Upper-envelope calculation via the Hilbert transform.
fn calc_upper_env(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    assert!(n > 0, "cannot compute envelope of empty signal");
    let mean = signal.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = signal.iter().map(|&s| s - mean).collect();

    let mut spectrum = fft::rfft(&centered);
    let fft_size = 2 * (spectrum.len() - 1);

    // The C++ version has a quirk that we reproduce faithfully: the analytic
    // signal's one-sided scaling vector is built with indices based on the
    // *original* signal length, but applied to a spectrum whose length is the
    // (possibly larger) power-of-two FFT size. Because the parity-based bound
    // is dead (the FFT size is even) and the n/2 entry always falls inside the
    // 2.0 run, the net effect is exactly one bin: for a non-power-of-two length
    // the padded spectrum's Nyquist bin is weighted 0 instead of MATLAB
    // `hilbert`'s 1.
    let mut scaling = vec![2.0; fft_size / 2 + 1];
    scaling[0] = 1.0;
    scaling[fft_size / 2] = if n == fft_size { 1.0 } else { 0.0 };

    for (bin, &s) in spectrum.iter_mut().zip(&scaling) {
        *bin *= s;
    }

    // The C++ takes the real IFFT of this spectrum (discarding the mirror
    // half), so the "analytic signal" is real-valued and the envelope is the
    // absolute value of that real signal.
    let time = fft::irfft(&mut spectrum);
    time[..n].iter().map(|&v| v.abs() + mean).collect()
}

/// Returns the lag (in samples) at which `signal_2` best aligns to
/// `signal_1`. Positive means `signal_2` is delayed relative to `signal_1`.
fn find_lowest_lag_index(signal_1: &[f64], signal_2: &[f64]) -> i64 {
    let longest = signal_1.len().max(signal_2.len());
    let max_lag = longest - 1;

    // Linear correlation of two length-`longest` signals needs
    // `2 * longest - 1` points; zero-extend both inputs so `rfft` picks the
    // covering power of two (the C++ derives the same count with
    // frexp(2 * len - 1)).
    let padded_len = 2 * longest - 1;
    let mut padded_1 = signal_1.to_vec();
    padded_1.resize(padded_len, 0.0);
    let mut padded_2 = signal_2.to_vec();
    padded_2.resize(padded_len, 0.0);

    let spec_1 = fft::rfft(&padded_1);
    let spec_2 = fft::rfft(&padded_2);
    let mut product: Vec<fft::Complex> = spec_1
        .iter()
        .zip(&spec_2)
        .map(|(a, b)| a * b.conj())
        .collect();
    let corr = fft::irfft(&mut product);

    // The wrapped negative lags sit at the end of the circular correlation,
    // followed (from the front) by the non-negative lags.
    let corrs = corr[corr.len() - max_lag..]
        .iter()
        .chain(&corr[..max_lag + 1]);

    // First maximum, matching std::max_element.
    let mut best_index = 0;
    let mut best_value = f64::NEG_INFINITY;
    for (i, &c) in corrs.enumerate() {
        if c > best_value {
            best_value = c;
            best_index = i;
        }
    }
    best_index as i64 - max_lag as i64
}

/// Aligns the degraded signal to the reference in place by cross-correlating
/// their upper envelopes. Returns the lag in seconds (positive when the
/// degraded signal was delayed / zero-padded).
pub fn globally_align(reference: &AudioSignal, degraded: &mut AudioSignal) -> f64 {
    let ref_env = calc_upper_env(&reference.samples);
    let deg_env = calc_upper_env(&degraded.samples);
    let best_lag = find_lowest_lag_index(&ref_env, &deg_env);

    // Limit the lag to half the reference duration.
    if best_lag == 0 || best_lag.unsigned_abs() as f64 > reference.samples.len() as f64 / 2.0 {
        return 0.0;
    }

    if best_lag < 0 {
        // Degraded leads the reference: truncate its start.
        degraded.samples.drain(..best_lag.unsigned_abs() as usize);
    } else {
        // Degraded trails the reference: prepend zeros.
        let zeros = best_lag as usize;
        let old_len = degraded.samples.len();
        degraded.samples.resize(old_len + zeros, 0.0);
        degraded.samples.copy_within(..old_len, zeros);
        degraded.samples[..zeros].fill(0.0);
    }
    best_lag as f64 / degraded.sample_rate as f64
}

/// Aligns the two signals in place and truncates them to matching lengths,
/// used for per-patch fine alignment. Returns the lag in seconds.
pub fn align_and_truncate(reference: &mut AudioSignal, degraded: &mut AudioSignal) -> f64 {
    let lag = globally_align(reference, degraded);
    let ref_len = reference.samples.len();
    let deg_len = degraded.samples.len();

    if ref_len > deg_len {
        reference.samples.truncate(deg_len);
    } else if ref_len < deg_len {
        // For positive lag the start of the reference is now aligned with the
        // zeros prepended to the degraded signal; truncate that amount from
        // both. (lag is always >= 0 on this branch: negative lag shortens the
        // degraded signal, so it cannot exceed an equal-length reference.)
        let start = ((lag * reference.sample_rate as f64) as i64).max(0) as usize;
        reference.samples.drain(..start);
        degraded.samples.truncate(ref_len);
        degraded.samples.drain(..start);
    }
    lag
}
