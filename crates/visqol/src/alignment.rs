//! Global signal alignment, a port of `alignment.cc`, `envelope.cc`, and `xcorr.cc`.

use std::cell::RefCell;

use realfft::RealFftPlanner;

use crate::audio_signal::AudioSignal;

/// FFT sizes are powers of two with a floor of 32, matching the C++
/// `FftManager`. The C++ runs pffft in single precision; we use `realfft` in
/// double precision, which only affects results below the f32 noise floor.
const MIN_FFT_SIZE: usize = 32;

// The planner caches plans (twiddle tables) by size; keep one per thread so
// the many small per-patch FFTs during fine realignment don't re-plan.
thread_local! {
    static PLANNER: RefCell<RealFftPlanner<f64>> = RefCell::new(RealFftPlanner::new());
}

/// Upper-envelope calculation via the Hilbert transform.
fn calc_upper_env(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    assert!(n > 0, "cannot compute envelope of empty signal");
    let mean = signal.iter().sum::<f64>() / n as f64;

    let fft_size = n.next_power_of_two().max(MIN_FFT_SIZE);
    let (r2c, c2r) =
        PLANNER.with_borrow_mut(|p| (p.plan_fft_forward(fft_size), p.plan_fft_inverse(fft_size)));

    // Mean-centered signal, zero-padded to the FFT size.
    let mut time = vec![0.0; fft_size];
    for (t, &s) in time.iter_mut().zip(signal) {
        *t = s - mean;
    }
    let mut spectrum = r2c.make_output_vec();
    r2c.process(&mut time, &mut spectrum)
        .expect("buffer sizes are correct by construction");

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
    // absolute value of that real signal. The C++ inverse scales by
    // 1/fft_size.
    c2r.process(&mut spectrum, &mut time)
        .expect("buffer sizes are correct by construction");
    let inv = 1.0 / fft_size as f64;
    time[..n].iter().map(|&v| (v * inv).abs() + mean).collect()
}

/// Returns the lag (in samples) at which `signal_2` best aligns to
/// `signal_1`. Positive means `signal_2` is delayed relative to `signal_1`.
fn find_lowest_lag_index(signal_1: &[f64], signal_2: &[f64]) -> i64 {
    let longest = signal_1.len().max(signal_2.len());
    let max_lag = longest - 1;

    // Linear correlation of two length-`longest` signals needs
    // `2 * longest - 1` points (the C++ derives the same count with
    // frexp(2 * len - 1)).
    let fft_size = (2 * longest - 1).next_power_of_two().max(MIN_FFT_SIZE);
    let (r2c, c2r) =
        PLANNER.with_borrow_mut(|p| (p.plan_fft_forward(fft_size), p.plan_fft_inverse(fft_size)));

    // One zero-padded input buffer, refilled for the second transform
    // (`process` uses it as scratch).
    let mut buf = vec![0.0; fft_size];
    buf[..signal_1.len()].copy_from_slice(signal_1);
    let mut spec_1 = r2c.make_output_vec();
    r2c.process(&mut buf, &mut spec_1)
        .expect("buffer sizes are correct by construction");
    buf.fill(0.0);
    buf[..signal_2.len()].copy_from_slice(signal_2);
    let mut spec_2 = r2c.make_output_vec();
    r2c.process(&mut buf, &mut spec_2)
        .expect("buffer sizes are correct by construction");

    for (a, b) in spec_1.iter_mut().zip(&spec_2) {
        *a *= b.conj();
    }
    // The C++ scales the inverse by 1/fft_size; an exact power of two, so
    // omitting it cannot change which lag wins the maximum below.
    c2r.process(&mut spec_1, &mut buf)
        .expect("buffer sizes are correct by construction");
    let corr = buf;

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
