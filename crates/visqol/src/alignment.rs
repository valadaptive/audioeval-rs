//! Global signal alignment, a port of `alignment.cc`.

use crate::audio_signal::AudioSignal;
use crate::envelope;
use crate::xcorr;

/// Aligns the degraded signal to the reference in place by cross-correlating
/// their upper envelopes. Returns the lag in seconds (positive when the
/// degraded signal was delayed / zero-padded).
pub fn globally_align(reference: &AudioSignal, degraded: &mut AudioSignal) -> f64 {
    let ref_env = envelope::calc_upper_env(&reference.samples);
    let deg_env = envelope::calc_upper_env(&degraded.samples);
    let best_lag = xcorr::find_lowest_lag_index(&ref_env, &deg_env);

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
