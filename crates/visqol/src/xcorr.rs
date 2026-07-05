//! FFT cross-correlation for global alignment, a port of `xcorr.cc`.

use crate::fft::{self, MIN_FFT_SIZE};

/// Returns the lag (in samples) at which `signal_2` best aligns to
/// `signal_1`. Positive means `signal_2` is delayed relative to `signal_1`.
pub fn find_lowest_lag_index(signal_1: &[f64], signal_2: &[f64]) -> i64 {
    let longest = signal_1.len().max(signal_2.len());
    let max_lag = longest as i64 - 1;

    // The C++ derives the FFT point count with frexp(2 * len - 1): the
    // smallest power of two strictly greater than 2 * len - 1.
    let v = 2 * longest - 1;
    let fft_points = 1usize << (usize::BITS - v.leading_zeros());
    let fft_size = fft_points.max(MIN_FFT_SIZE);

    let spec_1 = fft::rfft(signal_1, fft_size);
    let spec_2 = fft::rfft(signal_2, fft_size);
    let mut product: Vec<fft::Complex> = spec_1
        .iter()
        .zip(&spec_2)
        .map(|(a, b)| a * b.conj())
        .collect();
    let corr = fft::irfft(&mut product, fft_size);

    // corrs = wrapped negative lags followed by non-negative lags.
    let max_lag = max_lag as usize;
    let corrs = corr[fft_points - max_lag..fft_points]
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
