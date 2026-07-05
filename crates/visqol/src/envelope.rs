//! Upper-envelope calculation via the Hilbert transform, a direct port of
//! `envelope.cc`.
//!
//! The C++ version has a quirk that we reproduce faithfully: the analytic
//! signal's one-sided scaling vector is built with indices based on the
//! *original* signal length, but applied to a spectrum whose length is the
//! (possibly larger) power-of-two FFT size. When the length is not a power of
//! two, this zeroes the Nyquist bin and doubles a bin that MATLAB's `hilbert`
//! would treat specially.

use crate::fft::{self, MIN_FFT_SIZE};

pub fn calc_upper_env(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    assert!(n > 0, "cannot compute envelope of empty signal");
    let mean = signal.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = signal.iter().map(|&s| s - mean).collect();

    let fft_size = fft::next_pow_two(n).max(MIN_FFT_SIZE);
    let mut spectrum = fft::rfft(&centered, fft_size);

    // Hilbert scaling as in the C++: scaling[0] = 1, then a value at n/2
    // based on the original length's parity, then indices [1, bound) are
    // set to 2 (overwriting the n/2 entry whenever n/2 < bound).
    let mut scaling = vec![0.0; fft_size / 2 + 1];
    scaling[0] = 1.0;
    let is_odd = n % 2 == 1;
    if n / 2 < scaling.len() {
        scaling[n / 2] = if is_odd { 2.0 } else { 1.0 };
    }
    let bound = if is_odd {
        fft_size.div_ceil(2)
    } else {
        fft_size / 2
    };
    for s in &mut scaling[1..bound] {
        *s = 2.0;
    }

    for (bin, &s) in spectrum.iter_mut().zip(&scaling) {
        *bin *= s;
    }

    // The C++ takes the real IFFT of this spectrum (discarding the mirror
    // half), so the "analytic signal" is real-valued and the envelope is the
    // absolute value of that real signal.
    let time = fft::irfft(&mut spectrum, fft_size);
    time[..n].iter().map(|&v| v.abs() + mean).collect()
}
