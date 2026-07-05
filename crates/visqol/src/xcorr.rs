//! FFT cross-correlation for global alignment, a port of `xcorr.cc`.

use crate::fft;

/// Returns the lag (in samples) at which `signal_2` best aligns to
/// `signal_1`. Positive means `signal_2` is delayed relative to `signal_1`.
pub fn find_lowest_lag_index(signal_1: &[f64], signal_2: &[f64]) -> i64 {
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
