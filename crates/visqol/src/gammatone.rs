//! Gammatone spectrogram construction: a port of
//! `gammatone_spectrogram_builder.cc`, `gammatone_filterbank.cc` and
//! `signal_filter.cc`.
//!
//! Each spectrogram cell is the RMS of one gammatone band's output over one
//! Hann-windowed frame; each band is a 4th-order gammatone implemented as four
//! cascaded biquads with state reset at every frame.
//!
//! This is where nearly all of ViSQOL's runtime goes, so the filter bank is
//! organized for data parallelism: bands are processed in coefficient-major
//! groups of `LANES`, with the four cascade stages fused into one pass per
//! frame. Every band still performs exactly the arithmetic (values and
//! operation order) of the straightforward one-biquad-at-a-time formulation,
//! which keeps the output bit-identical while letting the compiler vectorize
//! across bands and the CPU overlap the stages' serial dependency chains.

use crate::analysis_window::AnalysisWindow;
use crate::audio_signal::AudioSignal;
use crate::erb;
use crate::matrix::Matrix;
use crate::spectrogram::Spectrogram;
use crate::{Error, Result};

pub const SPEECH_MODE_MAX_FREQ: f64 = 8000.0;

const STAGES: usize = 4;

pub struct GammatoneSpectrogramBuilder {
    num_bands: usize,
    min_freq: f64,
    speech_mode: bool,
}

/// Filter coefficients for a group of up to `LANES` bands, one band per lane.
/// Each cascade stage is `y = n0*x + z0; z0' = n1*x + z1 - d1*y;
/// z1' = neg_d2*y`, a transposed direct form II biquad with a normalized
/// denominator [1, d1, d2]. Two exact simplifications relative to the C++
/// formulation: the numerator's third tap is omitted because it is always
/// zero in the ERB design (`0*x - d2*y` equals `-(d2*y)`, signed zeros
/// included), and the negation is folded into the stored coefficient
/// (`-(d2*y)` equals `(-d2)*y`, since multiplication XORs the sign bits).
/// Unused lanes have all-zero coefficients and produce all-zero output.
#[derive(Clone, Copy)]
struct BandGroup<const LANES: usize> {
    /// Stage 1's first numerator tap (gain-divided).
    n0_first: [f64; LANES],
    /// Stages 2-4's first numerator tap.
    n0_rest: [f64; LANES],
    /// Each stage's second numerator tap.
    n1: [[f64; LANES]; STAGES],
    d1: [f64; LANES],
    neg_d2: [f64; LANES],
}

impl<const LANES: usize> BandGroup<LANES> {
    fn zeroed() -> Self {
        BandGroup {
            n0_first: [0.0; LANES],
            n0_rest: [0.0; LANES],
            n1: [[0.0; LANES]; STAGES],
            d1: [0.0; LANES],
            neg_d2: [0.0; LANES],
        }
    }

    /// Fills lane `lane` from one row of ERB coefficients
    /// [A0, A11, A12, A13, A14, A2, B0, B1, B2, gain]. The gain divides the
    /// first stage's numerator.
    fn set_lane(&mut self, lane: usize, c: &[f64; 10]) {
        let [a0, a11, a12, a13, a14, _a2, _b0, b1, b2, gain] = *c;
        self.n0_first[lane] = a0 / gain;
        self.n0_rest[lane] = a0;
        for (n1, tap) in self.n1.iter_mut().zip([a11 / gain, a12, a13, a14]) {
            n1[lane] = tap;
        }
        self.d1[lane] = b1;
        self.neg_d2[lane] = -b2;
    }
}

/// Runs `frame` through the four-stage cascade (zero initial conditions) and
/// returns each lane's sum of squared outputs.
///
/// `inline(always)` so the `#[target_feature]` wrappers below compile it
/// with their wider register files enabled.
#[inline(always)]
fn sum_squares_of_filtered<const LANES: usize>(
    group: &BandGroup<LANES>,
    frame: &[f64],
) -> [f64; LANES] {
    let mut z0 = [[0.0; LANES]; STAGES];
    let mut z1 = [[0.0; LANES]; STAGES];
    let mut acc = [0.0; LANES];
    for &x in frame {
        let mut sig = [x; LANES];
        for s in 0..STAGES {
            let n0 = if s == 0 {
                &group.n0_first
            } else {
                &group.n0_rest
            };
            let n1 = &group.n1[s];
            let mut y = [0.0; LANES];
            for l in 0..LANES {
                y[l] = n0[l] * sig[l] + z0[s][l];
            }
            for l in 0..LANES {
                z0[s][l] = n1[l] * sig[l] + z1[s][l] - group.d1[l] * y[l];
                z1[s][l] = group.neg_d2[l] * y[l];
            }
            sig = y;
        }
        for l in 0..LANES {
            acc[l] += sig[l] * sig[l];
        }
    }
    acc
}

fn build_groups<const LANES: usize>(filters: &erb::ErbFilters) -> Vec<BandGroup<LANES>> {
    let num_bands = filters.coeffs.len();
    let mut groups = vec![BandGroup::zeroed(); num_bands.div_ceil(LANES)];
    // Reversed: the C++ flips the ERB coefficient matrix upside down so that
    // band 0 is the lowest frequency.
    for (band, coeffs) in filters.coeffs.iter().rev().enumerate() {
        groups[band / LANES].set_lane(band % LANES, coeffs);
    }
    groups
}

/// Writes each band's RMS-of-filtered-frame into `out`.
#[inline(always)]
fn rms_into<const LANES: usize>(groups: &[BandGroup<LANES>], frame: &[f64], out: &mut [f64]) {
    for (group_idx, group) in groups.iter().enumerate() {
        let sums = sum_squares_of_filtered(group, frame);
        for (lane, &sum) in sums.iter().enumerate() {
            let band = group_idx * LANES + lane;
            if band < out.len() {
                out[band] = (sum / frame.len() as f64).sqrt();
            }
        }
    }
}

impl GammatoneSpectrogramBuilder {
    pub fn new(num_bands: usize, min_freq: f64, speech_mode: bool) -> Self {
        GammatoneSpectrogramBuilder {
            num_bands,
            min_freq,
            speech_mode,
        }
    }

    pub fn build(&self, signal: &AudioSignal, window: &AnalysisWindow) -> Result<Spectrogram> {
        let sample_rate = signal.sample_rate as f64;
        let max_freq = if self.speech_mode {
            SPEECH_MODE_MAX_FREQ
        } else {
            sample_rate / 2.0
        };

        let filters = erb::make_filters(sample_rate, self.num_bands, self.min_freq, max_freq);
        let groups = build_groups::<8>(&filters);

        let hop_size = (window.size as f64 * window.overlap) as usize;
        if signal.samples.len() <= window.size {
            return Err(Error::TooFewSamples {
                samples: signal.samples.len(),
                required: window.size,
            });
        }
        let num_cols = 1 + (signal.samples.len() - window.size) / hop_size;

        let mut out = Matrix::zeros(self.num_bands, num_cols);
        let mut windowed = vec![0.0; window.size];
        for col in 0..num_cols {
            let start = col * hop_size;
            let frame = &signal.samples[start..start + window.size];
            for ((w, &h), &x) in windowed.iter_mut().zip(&window.hann_window).zip(frame) {
                *w = h * x;
            }
            multiversion::multiversion(
                #[inline(always)]
                || rms_into(&groups, &windowed, out.col_mut(col)),
            );
        }

        // Center frequencies ordered lowest to highest.
        let center_freq_bands: Vec<f64> = filters.center_freqs.iter().rev().copied().collect();
        Ok(Spectrogram::new(out, center_freq_bands))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The straightforward formulation: one biquad at a time, stage by stage
    /// over the whole frame, matching the C++ `SignalFilter` exactly.
    fn scalar_reference(coeffs: &[f64; 10], frame: &[f64]) -> f64 {
        let [a0, a11, a12, a13, a14, a2, _b0, b1, b2, gain] = *coeffs;
        let numerators = [
            [a0 / gain, a11 / gain, a2 / gain],
            [a0, a12, a2],
            [a0, a13, a2],
            [a0, a14, a2],
        ];
        let mut signal = frame.to_vec();
        for n in numerators {
            let (mut z0, mut z1) = (0.0, 0.0);
            for x in &mut signal {
                let y = n[0] * *x + z0;
                z0 = n[1] * *x + z1 - b1 * y;
                z1 = n[2] * *x - b2 * y;
                *x = y;
            }
        }
        let mean_sq = signal.iter().map(|&y| y * y).sum::<f64>() / signal.len() as f64;
        mean_sq.sqrt()
    }

    /// A deterministic pseudo-random frame.
    fn test_frame() -> Vec<f64> {
        let mut state = 0x2545F4914F6CDD1Du64;
        (0..960)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }

    /// Whichever kernel the runtime dispatch selects must be bit-identical
    /// to the scalar stage-by-stage formulation.
    #[test]
    fn dispatched_kernel_is_bit_exact() {
        let filters = erb::make_filters(48000.0, 32, 50.0, 24000.0);
        let groups = build_groups::<8>(&filters);
        let frame = test_frame();

        let mut out = vec![0.0; 32];
        multiversion::multiversion(
            #[inline(always)]
            || rms_into(&groups, &frame, &mut out),
        );

        for (band, coeffs) in filters.coeffs.iter().rev().enumerate() {
            let expected = scalar_reference(coeffs, &frame);
            assert_eq!(
                out[band].to_bits(),
                expected.to_bits(),
                "band {band}: {:e} != {expected:e}",
                out[band]
            );
        }
    }
}
