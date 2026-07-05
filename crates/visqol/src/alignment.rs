//! Global signal alignment, a port of `alignment.cc`.

use crate::audio_signal::AudioSignal;
use crate::envelope;
use crate::xcorr;

/// Aligns the degraded signal to the reference by cross-correlating their
/// upper envelopes. Returns the aligned degraded signal and the lag in
/// seconds (positive when the degraded signal was delayed / zero-padded).
pub fn globally_align(reference: &AudioSignal, degraded: &AudioSignal) -> (AudioSignal, f64) {
    let ref_env = envelope::calc_upper_env(&reference.samples);
    let deg_env = envelope::calc_upper_env(&degraded.samples);
    let best_lag = xcorr::find_lowest_lag_index(&ref_env, &deg_env);

    // Limit the lag to half the reference duration.
    if best_lag == 0 || best_lag.unsigned_abs() as f64 > reference.samples.len() as f64 / 2.0 {
        return (degraded.clone(), 0.0);
    }

    let new_samples = if best_lag < 0 {
        // Degraded leads the reference: truncate its start.
        degraded.samples[best_lag.unsigned_abs() as usize..].to_vec()
    } else {
        // Degraded trails the reference: prepend zeros.
        let mut padded = vec![0.0; best_lag as usize];
        padded.extend_from_slice(&degraded.samples);
        padded
    };
    let lag_seconds = best_lag as f64 / degraded.sample_rate as f64;
    (
        AudioSignal::new(new_samples, degraded.sample_rate),
        lag_seconds,
    )
}

/// Aligns the two signals and truncates them to matching lengths, used for
/// per-patch fine alignment. Returns (reference, degraded, lag_seconds).
pub fn align_and_truncate(
    reference: &AudioSignal,
    degraded: &AudioSignal,
) -> (AudioSignal, AudioSignal, f64) {
    let (aligned_degraded, lag) = globally_align(reference, degraded);
    let ref_len = reference.samples.len();
    let deg_len = aligned_degraded.samples.len();

    if ref_len > deg_len {
        let new_reference =
            AudioSignal::new(reference.samples[..deg_len].to_vec(), reference.sample_rate);
        (new_reference, aligned_degraded, lag)
    } else if ref_len < deg_len {
        // For positive lag the start of the reference is now aligned with the
        // zeros prepended to the degraded signal; truncate that amount from
        // both. (lag is always >= 0 on this branch: negative lag shortens the
        // degraded signal, so it cannot exceed an equal-length reference.)
        let start = ((lag * reference.sample_rate as f64) as i64).max(0) as usize;
        let new_reference = AudioSignal::new(
            reference.samples[start..ref_len].to_vec(),
            reference.sample_rate,
        );
        let new_degraded = AudioSignal::new(
            aligned_degraded.samples[start..ref_len].to_vec(),
            aligned_degraded.sample_rate,
        );
        (new_reference, new_degraded, lag)
    } else {
        (reference.clone(), aligned_degraded, lag)
    }
}
