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
    /// Stage 1's first numerator tap (gain-divided).
    n0_first: [f64; 8],
    /// Stages 2-4's first numerator tap.
    n0_rest: [f64; 8],
    /// Each stage's second numerator tap.
    n1: [[f64; 8]; STAGES],
    neg_d1: [f64; 8],
    neg_d2: [f64; 8],
}

impl BandGroupStored {
    fn zeroed() -> Self {
        Self {
            n0_first: [0.0; 8],
            n0_rest: [0.0; 8],
            n1: [[0.0; 8]; STAGES],
            neg_d1: [0.0; 8],
            neg_d2: [0.0; 8],
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
        self.neg_d1[lane] = -b1;
        self.neg_d2[lane] = -b2;
    }
}

/// Filter coefficients for a group of 8 bands, one band per lane.
/// Each cascade stage is `y = n0*x + z0; z0' = n1*x + z1 - d1*y;
/// z1' = neg_d2*y`, a transposed direct form II biquad with a normalized
/// denominator [1, d1, d2]. Two simplifications relative to the C++
/// formulation: the numerator's third tap is omitted because it is always
/// zero in the ERB design (`0*x - d2*y` equals `-(d2*y)`, signed zeros
/// included), and the negation is folded into the stored coefficient
/// (`-(d2*y)` equals `(-d2)*y`, since multiplication XORs the sign bits).
/// Unused lanes have all-zero coefficients and produce all-zero output.
#[derive(Clone, Copy)]
struct BandGroup<S: Simd> {
    /// Stage 1's first numerator tap (gain-divided).
    n0_first: f64x8<S>,
    /// Stages 2-4's first numerator tap.
    n0_rest: f64x8<S>,
    /// Each stage's second numerator tap.
    n1: [f64x8<S>; STAGES],
    neg_d1: f64x8<S>,
    neg_d2: f64x8<S>,
}

impl<S: Simd> BandGroup<S> {
    fn from_stored(simd: S, stored: &BandGroupStored) -> Self {
        Self {
            n0_first: f64x8::simd_from(simd, stored.n0_first),
            n0_rest: f64x8::simd_from(simd, stored.n0_rest),
            n1: stored.n1.map(|taps| f64x8::simd_from(simd, taps)),
            neg_d1: f64x8::simd_from(simd, stored.neg_d1),
            neg_d2: f64x8::simd_from(simd, stored.neg_d2),
        }
    }
}

/// Runs `frame` through the four-stage cascade (zero initial conditions) and
/// returns each lane's sum of squared outputs.
#[inline(always)]
fn sum_squares_of_filtered<S: Simd>(simd: S, group: &BandGroup<S>, frame: &[f64]) -> f64x8<S> {
    let mut z0 = [f64x8::splat(simd, 0.0); STAGES];
    let mut z1 = [f64x8::splat(simd, 0.0); STAGES];
    let mut acc = f64x8::splat(simd, 0.0);
    for &x in frame {
        let mut sig = f64x8::splat(simd, x);
        for s in 0..STAGES {
            let n0 = if s == 0 {
                group.n0_first
            } else {
                group.n0_rest
            };
            let n1 = group.n1[s];
            let y = n0.mul_add(sig, z0[s]);
            z0[s] = group.neg_d1.mul_add(y, n1.mul_add(sig, z1[s]));
            z1[s] = group.neg_d2 * y;
            sig = y;
        }
        acc += sig * sig;
    }
    acc
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
fn rms_into<S: Simd>(simd: S, groups: &[BandGroup<S>], frame: &[f64], out: &mut [f64]) {
    for (group_idx, group) in groups.iter().enumerate() {
        let sums = sum_squares_of_filtered(simd, group, frame);
        for (lane, &sum) in sums.iter().enumerate() {
            let band = group_idx * 8 + lane;
            if band < out.len() {
                out[band] = (sum / frame.len() as f64).sqrt();
            }
        }
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
        dispatch!(level, simd => {
            let groups = self.groups.iter().map(|group| BandGroup::from_stored(simd, group)).collect::<Vec<_>>();

            for col in 0..num_cols {
                let start = col * hop_size;
                let frame = &signal.samples[start..start + window.size];
                for ((w, &h), &x) in windowed.iter_mut().zip(&window.hann_window).zip(frame) {
                    *w = h * x;
                }
                rms_into(simd, &groups, &windowed, out.col_mut(col));
            }
        });

        // Center frequencies ordered lowest to highest.
        let center_freq_bands: Vec<f64> = self.filters.center_freqs.iter().rev().copied().collect();
        Ok(Spectrogram::new(out, center_freq_bands))
    }
}
