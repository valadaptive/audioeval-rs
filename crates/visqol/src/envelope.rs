//! Upper-envelope calculation via the Hilbert transform, a direct port of
//! `envelope.cc`.
//!
//! The C++ version has a quirk that we reproduce faithfully: the analytic
//! signal's one-sided scaling vector is built with indices based on the
//! *original* signal length, but applied to a spectrum whose length is the
//! (possibly larger) power-of-two FFT size. Because the parity-based bound
//! is dead (the FFT size is even) and the n/2 entry always falls inside the
//! 2.0 run, the net effect is exactly one bin: for a non-power-of-two length
//! the padded spectrum's Nyquist bin is weighted 0 instead of MATLAB
//! `hilbert`'s 1.

use crate::fft;

pub fn calc_upper_env(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    assert!(n > 0, "cannot compute envelope of empty signal");
    let mean = signal.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = signal.iter().map(|&s| s - mean).collect();

    let mut spectrum = fft::rfft(&centered);
    let fft_size = 2 * (spectrum.len() - 1);

    // Hilbert scaling with the C++ quirk described in the module docs: the
    // Nyquist bin keeps weight 1 only when the signal length already is the
    // FFT size.
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
