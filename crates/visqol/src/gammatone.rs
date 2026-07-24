//! Gammatone spectrogram construction: a port of
//! `gammatone_spectrogram_builder.cc`, `gammatone_filterbank.cc` and
//! `signal_filter.cc`.
//!
//! Each spectrogram cell is the RMS of one gammatone band's output over one
//! Hann-windowed frame; each band is a 4th-order gammatone implemented as four
//! cascaded biquads with state reset at every frame.

use fearless_simd::{Level, dispatch, f64x8, prelude::*};

use crate::analysis_window::AnalysisWindow;
use crate::audio_signal::AudioSignal;
use crate::erb::{self, ErbFilters};
use crate::matrix::Matrix;
use crate::spectrogram::Spectrogram;
use crate::{Error, Result};

pub const SPEECH_MODE_MAX_FREQ: f64 = 8000.0;

const STAGES: usize = 4;

pub struct GammatoneSpectrogramBuilder {
    num_bands: usize,
    filters: ErbFilters,
    groups: Vec<BandGroupStored>,
}

#[derive(Clone, Copy)]
struct BandGroupStored {
    /// Product of the four factored-out numerator taps, squared because the
    /// kernel accumulates squared output.
    output_scale_sq: [f64; 8],
    /// Each stage's second numerator tap divided by its first.
    q: [[f64; 8]; STAGES],
    neg_d1: [f64; 8],
    neg_d2: [f64; 8],
}

impl BandGroupStored {
    fn zeroed() -> Self {
        Self {
            output_scale_sq: [0.0; 8],
            q: [[0.0; 8]; STAGES],
            neg_d1: [0.0; 8],
            neg_d2: [0.0; 8],
        }
    }

    /// Fills lane `lane` from one row of ERB coefficients
    /// [A0, A11, A12, A13, A14, A2, B0, B1, B2, gain]. The gain divides the
    /// first stage's numerator.
    fn set_lane(&mut self, lane: usize, c: &[f64; 10]) {
        let [a0, a11, a12, a13, a14, _a2, _b0, b1, b2, gain] = *c;
        let output_scale = (a0 / gain) * a0 * a0 * a0;
        self.output_scale_sq[lane] = output_scale * output_scale;
        for (q, tap) in self.q.iter_mut().zip([a11, a12, a13, a14]) {
            q[lane] = tap / a0;
        }
        self.neg_d1[lane] = -b1;
        self.neg_d2[lane] = -b2;
    }
}

/// Filter coefficients for a group of 8 bands, one band per lane.
///
/// Factoring the first numerator coefficient out of every section changes
/// `(n0 + n1*z^-1) / D` into `n0 * (1 + q*z^-1) / D`. The four constant
/// factors are applied once to the accumulated energy instead of multiplying
/// every sample in every section. The normalized transposed direct-form II
/// update is `y = x + z0; z0' = q*x + z1 - d1*y; z1' = -d2*y`.
///
/// The numerator's third tap is omitted because it is always zero in the ERB
/// design, and denominator negations are folded into the stored coefficients.
/// Unused lanes have a zero output scale.
#[derive(Clone, Copy)]
struct BandGroup<S: Simd> {
    output_scale_sq: f64x8<S>,
    q: [f64x8<S>; STAGES],
    neg_d1: f64x8<S>,
    neg_d2: f64x8<S>,
}

impl<S: Simd> BandGroup<S> {
    fn from_stored(simd: S, stored: &BandGroupStored) -> Self {
        Self {
            output_scale_sq: f64x8::simd_from(simd, stored.output_scale_sq),
            q: stored.q.map(|taps| f64x8::simd_from(simd, taps)),
            neg_d1: f64x8::simd_from(simd, stored.neg_d1),
            neg_d2: f64x8::simd_from(simd, stored.neg_d2),
        }
    }
}

/// Runs `frame` through one four-stage cascade with zero initial conditions.
#[inline(always)]
fn sum_squares_of_filtered<S: Simd>(simd: S, group: &BandGroup<S>, frame: &[f64]) -> f64x8<S> {
    let zero = f64x8::splat(simd, 0.0);
    let mut z0 = [zero; STAGES];
    let mut z1 = [zero; STAGES];
    let mut acc = zero;
    for &x in frame {
        let mut sig = f64x8::splat(simd, x);
        for s in 0..STAGES {
            let y = sig + z0[s];
            z0[s] = group.neg_d1.mul_add(y, group.q[s].mul_add(sig, z1[s]));
            z1[s] = group.neg_d2 * y;
            sig = y;
        }
        acc += sig * sig;
    }
    acc
}

/// Runs `frame` through two independent four-stage cascades (zero initial
/// conditions) and returns each lane's sum of squared outputs. Interleaving two
/// groups gives the CPU enough independent work to hide the recurrence's FMA
/// latency; processing one group at a time leaves much of the execution width
/// idle.
#[inline(always)]
fn sum_squares_of_filtered_pair<S: Simd>(
    simd: S,
    a: &BandGroup<S>,
    b: &BandGroup<S>,
    frame: &[f64],
) -> [f64x8<S>; 2] {
    let zero = f64x8::splat(simd, 0.0);
    let mut z0_a = [zero; STAGES];
    let mut z1_a = [zero; STAGES];
    let mut z0_b = [zero; STAGES];
    let mut z1_b = [zero; STAGES];
    let mut acc_a = zero;
    let mut acc_b = zero;
    for &x in frame {
        let input = f64x8::splat(simd, x);
        let mut sig_a = input;
        let mut sig_b = input;
        for s in 0..STAGES {
            let y_a = sig_a + z0_a[s];
            let y_b = sig_b + z0_b[s];
            z0_a[s] = a.neg_d1.mul_add(y_a, a.q[s].mul_add(sig_a, z1_a[s]));
            z0_b[s] = b.neg_d1.mul_add(y_b, b.q[s].mul_add(sig_b, z1_b[s]));
            z1_a[s] = a.neg_d2 * y_a;
            z1_b[s] = b.neg_d2 * y_b;
            sig_a = y_a;
            sig_b = y_b;
        }
        acc_a += sig_a * sig_a;
        acc_b += sig_b * sig_b;
    }
    [acc_a, acc_b]
}

fn build_groups(filters: &erb::ErbFilters) -> Vec<BandGroupStored> {
    let num_bands = filters.coeffs.len();
    let mut groups = vec![BandGroupStored::zeroed(); num_bands.div_ceil(8)];
    // Reversed: the C++ flips the ERB coefficient matrix upside down so that
    // band 0 is the lowest frequency.
    for (band, coeffs) in filters.coeffs.iter().rev().enumerate() {
        groups[band / 8].set_lane(band % 8, coeffs);
    }
    groups
}

/// Writes each band's RMS-of-filtered-frame into `out`.
#[inline(always)]
fn rms_into<S: Simd>(
    simd: S,
    interleave_groups: bool,
    groups: &[BandGroup<S>],
    frame: &[f64],
    out: &mut [f64],
) {
    let mean_scale = 1.0 / frame.len() as f64;
    let mut group_idx = 0;
    // A pair's live filter state fits comfortably in AVX-512's 32 registers.
    // Narrower x86 levels represent f64x8 with multiple registers and are
    // faster using the single-group kernel than spilling the paired state.
    while interleave_groups && group_idx + 1 < groups.len() {
        let sums =
            sum_squares_of_filtered_pair(simd, &groups[group_idx], &groups[group_idx + 1], frame);
        for (pair_idx, sums) in sums.iter().enumerate() {
            let group = &groups[group_idx + pair_idx];
            for ((lane, &sum), &scale_sq) in
                sums.iter().enumerate().zip(group.output_scale_sq.iter())
            {
                let band = (group_idx + pair_idx) * 8 + lane;
                if band < out.len() {
                    out[band] = (sum * scale_sq * mean_scale).sqrt();
                }
            }
        }
        group_idx += 2;
    }
    while group_idx < groups.len() {
        let sums = sum_squares_of_filtered(simd, &groups[group_idx], frame);
        for ((lane, &sum), &scale_sq) in sums
            .iter()
            .enumerate()
            .zip(groups[group_idx].output_scale_sq.iter())
        {
            let band = group_idx * 8 + lane;
            if band < out.len() {
                out[band] = (sum * scale_sq * mean_scale).sqrt();
            }
        }
        group_idx += 1;
    }
}

impl GammatoneSpectrogramBuilder {
    pub fn new(num_bands: usize, sample_rate: u32, min_freq: f64, speech_mode: bool) -> Self {
        let sample_rate = sample_rate as f64;
        let max_freq = if speech_mode {
            SPEECH_MODE_MAX_FREQ
        } else {
            sample_rate / 2.0
        };

        let filters = erb::make_filters(sample_rate, num_bands, min_freq, max_freq);
        let groups = build_groups(&filters);

        GammatoneSpectrogramBuilder {
            num_bands,
            filters,
            groups,
        }
    }

    pub fn build(
        &self,
        level: Level,
        signal: &AudioSignal,
        window: &AnalysisWindow,
    ) -> Result<Spectrogram> {
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
        let interleave_groups = level.as_avx512().is_some();
        dispatch!(level, simd => {
            let groups = self.groups.iter().map(|group| BandGroup::from_stored(simd, group)).collect::<Vec<_>>();

            for col in 0..num_cols {
                let start = col * hop_size;
                let frame = &signal.samples[start..start + window.size];
                for ((w, &h), &x) in windowed.iter_mut().zip(&window.hann_window).zip(frame) {
                    *w = h * x;
                }
                rms_into(simd, interleave_groups, &groups, &windowed, out.col_mut(col));
            }
        });

        // Center frequencies ordered lowest to highest.
        let center_freq_bands: Vec<f64> = self.filters.center_freqs.iter().rev().copied().collect();
        Ok(Spectrogram::new(out, center_freq_bands))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straightforward, unnormalized SOS cascade used to check the algebraic
    /// reformulation in the optimized kernel.
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
        (signal.iter().map(|&y| y * y).sum::<f64>() / signal.len() as f64).sqrt()
    }

    fn test_frame() -> Vec<f64> {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        (0..3840)
            .map(|i| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let noise = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
                let hann =
                    0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (3840.0 - 1.0)).cos();
                noise * hann
            })
            .collect()
    }

    #[test]
    fn normalized_kernel_matches_sos_cascade() {
        // Exercise both the paired path (32 bands) and single-group tail
        // (21 bands), including the least numerically forgiving low bands.
        for num_bands in [21, 32] {
            let filters = erb::make_filters(48_000.0, num_bands, 50.0, 24_000.0);
            let stored = build_groups(&filters);
            let frame = test_frame();
            let mut actual = vec![0.0; num_bands];
            let level = Level::new();
            let interleave_groups = level.as_avx512().is_some();
            dispatch!(level, simd => {
                let groups = stored
                    .iter()
                    .map(|group| BandGroup::from_stored(simd, group))
                    .collect::<Vec<_>>();
                rms_into(simd, interleave_groups, &groups, &frame, &mut actual);
            });

            for (band, coeffs) in filters.coeffs.iter().rev().enumerate() {
                let expected = scalar_reference(coeffs, &frame);
                let relative_error = ((actual[band] - expected) / expected).abs();
                assert!(
                    relative_error < 1e-10,
                    "{num_bands} bands, band {band}: {} != {expected} ({relative_error:e})",
                    actual[band]
                );
            }
        }
    }
}
