//! A pure-Rust implementation of the SEBASS 2f-model for estimating the perceived quality of separated audio sources.
//!
//! The model uses only two PEAQ Basic model output variables: `AvgModDiff1` and `ADB`. Their implementation follows Peter Kabal's PQevalAudio v1r0, which is the PEAQ implementation for which the published 2f-model parameters were fitted.
//!
//! ```no_run
//! use audioeval_2f::TwoFModel;
//!
//! let reference: Vec<Vec<f32>> = vec![vec![0.0; 48_000]; 2];
//! let degraded = reference.clone();
//! let mut model = TwoFModel::new();
//! let result = model.run(&reference, &degraded)?;
//! println!("estimated MUSHRA score: {}", result.mushra_score);
//! # Ok::<(), audioeval_2f::Error>(())
//! ```
//!
//! Input samples are normalized PCM in `[-1, 1]`. Only 48 kHz mono or stereo signals are supported. The signals must already be time-aligned.
//!
//! The returned score is clipped to the MUSHRA range of 0–100; the unclipped regression output and both MOVs are also available in `TwoFResult`.
//!
//! ## Why not PEAQ?
//!
//! Why did I choose only to implement two specific PEAQ model output variables (MOVs), rather than just implementing all of PEAQ Basic? There are a few reasons:
//!
//! - PEAQ's ODG (objective difference grade) output seems to be a poor metric. The individual MOVs may still prove useful, but...
//!
//! - There are currently *no conforming PEAQ implementations*, and the standard is worded vaguely enough that it appears impossible to write one (see [this report](https://www.mmsp.ece.mcgill.ca/Documents/Reports/2002/KabalR2002v2.pdf)).
//!
//!   As mentioned above, this package resolves ambiguities in the PEAQ specification by following the behavior of the MATLAB [PQEvalAudio](https://www.mmsp.ece.mcgill.ca/Documents/Software/index.html) package. Other implementations, such as [EAQUAL](https://github.com/spxnn/eaqual) or [GstPEAQ](https://github.com/HSU-ANT/gstpeaq), interpret the spec differently and produce different results.
//!
//! - It's slower to compute MOVs that we don't need.
//!
//! In the future, I *may* attempt to write a PEAQ implementation that can be configured to match the behavior of other implementations, but the lack of standardization renders it unattractive as an objective metric.

mod constants;
mod fast_pow;
mod pow_table;

use std::sync::Arc;

use constants::{FC, FL, FU, NUM_BANDS};
use fast_pow::{pow_03, pow_04, pow_005, pow_171332};
pub use fearless_simd::Level;
use fearless_simd::{dispatch, prelude::*};
use num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

const SAMPLE_RATE: u32 = 48_000;
const FRAME_SIZE: usize = 2048;
const HOP_SIZE: usize = FRAME_SIZE / 2;
const SPECTRUM_SIZE: usize = FRAME_SIZE / 2 + 1;
const FRAME_RATE: f64 = SAMPLE_RATE as f64 / HOP_SIZE as f64;

/// Output of a 2f-model comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoFResult {
    /// Estimated mean MUSHRA score, clipped to the MUSHRA range `[0, 100]`.
    pub mushra_score: f64,
    /// The unbounded output of the published regression equation.
    pub raw_mushra_score: f64,
    /// PEAQ Basic `AvgModDiff1B` model output variable.
    pub avg_mod_diff1: f64,
    /// PEAQ Basic average distorted blocks (`ADBB`) model output variable.
    pub adb: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidChannelCount { reference: usize, degraded: usize },
    UnequalChannelLengths,
    TooFewSamples,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidChannelCount {
                reference,
                degraded,
            } => write!(
                f,
                "inputs must have the same mono or stereo channel count \
                 (reference: {reference}, degraded: {degraded})"
            ),
            Self::UnequalChannelLengths => {
                write!(f, "all channels within each input must have equal lengths")
            }
            Self::TooFewSamples => write!(f, "reference signal is too short to evaluate"),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Copy)]
struct BandMapping {
    lower_bin: usize,
    upper_bin: usize,
    lower_weight: f64,
    upper_weight: f64,
}

#[derive(Clone)]
struct ChannelState {
    time_energy: [[f64; NUM_BANDS]; 2],
    previous_energy: [[f64; NUM_BANDS]; 2],
    derivative: [[f64; NUM_BANDS]; 2],
    average_energy: [[f64; NUM_BANDS]; 2],
}

impl Default for ChannelState {
    fn default() -> Self {
        Self {
            time_energy: [[0.0; NUM_BANDS]; 2],
            previous_energy: [[0.0; NUM_BANDS]; 2],
            derivative: [[0.0; NUM_BANDS]; 2],
            average_energy: [[0.0; NUM_BANDS]; 2],
        }
    }
}

struct FftBuffers {
    scratch: Vec<Complex<f64>>,
    output: Vec<Complex<f64>>,
}

impl FftBuffers {
    fn new(fft: &dyn RealToComplex<f64>) -> Self {
        Self {
            scratch: fft.make_scratch_vec(),
            output: fft.make_output_vec(),
        }
    }
}

#[derive(Clone, Copy)]
struct FrameMov {
    mod_diff: f64,
    weight: f64,
}

struct Spreading {
    lower_powered: f64,
    lower_sum: [f64; NUM_BANDS],
    upper_slope: [f64; NUM_BANDS],
}

impl Spreading {
    fn new() -> Self {
        const EXPONENT: f64 = 0.4;
        const BAND_WIDTH: f64 = 0.25;
        let lower_slope = 10.0f64.powf(-2.7 * BAND_WIDTH);
        Self {
            lower_powered: lower_slope.powf(EXPONENT),
            lower_sum: std::array::from_fn(|band| {
                (1.0 - lower_slope.powi((band + 1) as i32)) / (1.0 - lower_slope)
            }),
            upper_slope: std::array::from_fn(|band| {
                10.0f64.powf((-2.4 - 23.0 / FC[band]) * BAND_WIDTH)
            }),
        }
    }
}

/// Reusable 2f-model evaluator.
///
/// Construction precomputes the PEAQ filter-bank tables and FFT plan. The
/// per-signal filter memories are reset by every call to [`run`](Self::run).
pub struct TwoFModel {
    pub simd_level: Level,
    fft: Arc<dyn RealToComplex<f64>>,
    window: [f64; FRAME_SIZE],
    outer_middle_ear: [f64; SPECTRUM_SIZE],
    mappings: [BandMapping; NUM_BANDS],
    internal_noise: [f64; NUM_BANDS],
    spreading: Spreading,
    spread_normalization: [f64; NUM_BANDS],
    time_a: [f64; NUM_BANDS],
    time_b: [f64; NUM_BANDS],
    modulation_a: [f64; NUM_BANDS],
    modulation_b: [f64; NUM_BANDS],
    noise_powered: [f64; NUM_BANDS],
}

impl Default for TwoFModel {
    fn default() -> Self {
        Self::new()
    }
}

impl TwoFModel {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(FRAME_SIZE);
        let simd_level = Level::new();

        let mut window = [0.0; FRAME_SIZE];
        let normalized_frequency = 1019.5 / SAMPLE_RATE as f64;
        let bin_width = 1.0 / FRAME_SIZE as f64;
        let bin = (normalized_frequency / bin_width).floor();
        let offset = ((bin + 1.0) * bin_width - normalized_frequency)
            .min(normalized_frequency - bin * bin_width)
            * (FRAME_SIZE - 1) as f64;
        let peak_gain = (std::f64::consts::PI * offset).sin()
            / (std::f64::consts::PI * offset * (1.0 - offset * offset));
        // PQevalAudio scales normalized WAV data by 32768 before applying a
        // window whose gain contains 1/32768. Cancel those factors here.
        let level_gain = 10.0f64.powf(92.0 / 20.0) / (peak_gain * (FRAME_SIZE - 1) as f64 / 4.0);
        for (i, value) in window.iter_mut().enumerate() {
            *value = level_gain
                * 0.5
                * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (FRAME_SIZE - 1) as f64).cos());
        }

        let mut outer_middle_ear = [0.0; SPECTRUM_SIZE];
        for (bin, weight) in outer_middle_ear.iter_mut().enumerate().skip(1) {
            let frequency_khz = bin as f64 * SAMPLE_RATE as f64 / FRAME_SIZE as f64 / 1000.0;
            let attenuation = -2.184 * frequency_khz.powf(-0.8)
                + 6.5 * (-0.6 * (frequency_khz - 3.3).powi(2)).exp()
                - 0.001 * frequency_khz.powf(3.6);
            *weight = 10.0f64.powf(attenuation / 10.0);
        }

        let mappings = std::array::from_fn(|band| {
            let df = SAMPLE_RATE as f64 / FRAME_SIZE as f64;
            let lower_bin = (0..SPECTRUM_SIZE)
                .find(|&k| (k as f64 + 0.5) * df > FL[band])
                .unwrap();
            let upper_bin = (0..SPECTRUM_SIZE)
                .rev()
                .find(|&k| (k as f64 - 0.5) * df < FU[band])
                .unwrap();
            let lower_weight = (FU[band].min((lower_bin as f64 + 0.5) * df)
                - FL[band].max((lower_bin as f64 - 0.5) * df))
                / df;
            let upper_weight = if lower_bin == upper_bin {
                0.0
            } else {
                (FU[band].min((upper_bin as f64 + 0.5) * df)
                    - FL[band].max((upper_bin as f64 - 0.5) * df))
                    / df
            };
            BandMapping {
                lower_bin,
                upper_bin,
                lower_weight,
                upper_weight,
            }
        });

        let internal_noise =
            std::array::from_fn(|i| 10.0f64.powf(1.456 * (FC[i] / 1000.0).powf(-0.8) / 10.0));
        let spreading = Spreading::new();
        let spread_normalization =
            dispatch!(simd_level, simd => spread_raw(simd, &[1.0; NUM_BANDS], &spreading));
        let (time_a, time_b) = time_constants(0.030, 0.008);
        let (modulation_a, modulation_b) = time_constants(0.050, 0.008);
        let noise_powered = internal_noise.map(|energy| energy.powf(0.3));

        Self {
            simd_level,
            fft,
            window,
            outer_middle_ear,
            mappings,
            internal_noise,
            spreading,
            spread_normalization,
            time_a,
            time_b,
            modulation_a,
            modulation_b,
            noise_powered,
        }
    }

    /// Evaluates aligned normalized PCM channels at 48 kHz.
    ///
    /// `R` and `D` can be `Vec<f32>`, slices, or other types implementing
    /// `AsRef<[f32]>`. Reference and degraded lengths may differ, matching
    /// PQevalAudio's zero-extension behavior.
    pub fn run<R: AsRef<[f32]>, D: AsRef<[f32]>>(
        &self,
        reference: &[R],
        degraded: &[D],
    ) -> Result<TwoFResult, Error> {
        let mut fft_buffers = FftBuffers::new(self.fft.as_ref());

        if reference.is_empty() || reference.len() > 2 || reference.len() != degraded.len() {
            return Err(Error::InvalidChannelCount {
                reference: reference.len(),
                degraded: degraded.len(),
            });
        }
        let reference_len = reference[0].as_ref().len();
        let degraded_len = degraded[0].as_ref().len();
        if reference
            .iter()
            .any(|ch| ch.as_ref().len() != reference_len)
            || degraded.iter().any(|ch| ch.as_ref().len() != degraded_len)
        {
            return Err(Error::UnequalChannelLengths);
        }

        let Some((data_start, data_end)) = data_boundaries(reference) else {
            return Err(Error::TooFewSamples);
        };
        let first_frame = data_start / HOP_SIZE;
        let end_numerator = data_end + 1;
        if end_numerator < HOP_SIZE {
            return Err(Error::TooFewSamples);
        }
        let last_frame = (end_numerator - HOP_SIZE) / HOP_SIZE;
        if last_frame < first_frame {
            return Err(Error::TooFewSamples);
        }
        let frame_count = last_frame - first_frame + 1;
        let mut states = vec![ChannelState::default(); reference.len()];
        let mut channel_mod_diff = vec![Vec::with_capacity(frame_count); reference.len()];
        let mut channel_weight = vec![Vec::with_capacity(frame_count); reference.len()];
        let mut probabilities = Vec::with_capacity(frame_count);
        let mut steps = Vec::with_capacity(frame_count);

        // Frames preceding the data boundary warm up PEAQ's filter memories.
        for processing_frame in 0..(first_frame + frame_count) {
            let mut frame_movs = Vec::with_capacity(reference.len());
            for channel in 0..reference.len() {
                let mov = self.process_frame(
                    reference[channel].as_ref(),
                    degraded[channel].as_ref(),
                    processing_frame * HOP_SIZE,
                    &mut states[channel],
                    &mut fft_buffers,
                );
                frame_movs.push(mov);
            }
            if processing_frame < first_frame {
                continue;
            }
            for (channel, mov) in frame_movs.iter().enumerate() {
                channel_mod_diff[channel].push(mov.mod_diff);
                channel_weight[channel].push(mov.weight);
            }
            let mut probability_complement = 1.0;
            let mut frame_steps = 0.0;
            for band in 0..NUM_BANDS {
                let probability = frame_movs
                    .iter()
                    .map(|mov| mov.probability_at(band))
                    .fold(0.0, f64::max);
                let band_steps = frame_movs
                    .iter()
                    .map(|mov| mov.steps_at(band))
                    .fold(0.0, f64::max);
                probability_complement *= 1.0 - probability;
                frame_steps += band_steps;
            }
            probabilities.push(1.0 - probability_complement);
            steps.push(frame_steps);
        }

        let delay = 24usize.saturating_sub(first_frame); // ceil(0.5 * 48000/1024)
        if delay >= frame_count {
            return Err(Error::TooFewSamples);
        }
        let avg_mod_diff1 = channel_mod_diff
            .iter()
            .zip(&channel_weight)
            .map(|(values, weights)| weighted_average(&values[delay..], &weights[delay..]))
            .sum::<f64>()
            / reference.len() as f64;
        let adb = average_distorted_blocks(&probabilities, &steps);
        let raw_mushra_score = estimate_mushra(avg_mod_diff1, adb);

        Ok(TwoFResult {
            mushra_score: raw_mushra_score.clamp(0.0, 100.0),
            raw_mushra_score,
            avg_mod_diff1,
            adb,
        })
    }

    fn process_frame(
        &self,
        reference: &[f32],
        degraded: &[f32],
        offset: usize,
        state: &mut ChannelState,
        fft_buffers: &mut FftBuffers,
    ) -> DetailedFrameMov {
        let ref_spectrum = self.spectrum(fft_buffers, reference, offset);
        let deg_spectrum = self.spectrum(fft_buffers, degraded, offset);
        let excitation = [
            self.excitation(&ref_spectrum),
            self.excitation(&deg_spectrum),
        ];
        let mut smoothed = [[0.0; NUM_BANDS]; 2];
        for signal in 0..2 {
            for band in 0..NUM_BANDS {
                state.time_energy[signal][band] = self.time_a[band]
                    * state.time_energy[signal][band]
                    + self.time_b[band] * excitation[signal][band];
                smoothed[signal][band] =
                    state.time_energy[signal][band].max(excitation[signal][band]);
            }
        }

        let mut modulation = [[0.0; NUM_BANDS]; 2];
        for signal in 0..2 {
            for band in 0..NUM_BANDS {
                let powered = pow_03(excitation[signal][band]);
                state.derivative[signal][band] = self.modulation_a[band]
                    * state.derivative[signal][band]
                    + self.modulation_b[band]
                        * FRAME_RATE
                        * (powered - state.previous_energy[signal][band]).abs();
                state.average_energy[signal][band] = self.modulation_a[band]
                    * state.average_energy[signal][band]
                    + self.modulation_b[band] * powered;
                state.previous_energy[signal][band] = powered;
                modulation[signal][band] = state.derivative[signal][band]
                    / (1.0 + state.average_energy[signal][band] / 0.3);
            }
        }

        let mut sum_mod_diff = 0.0;
        let mut weight = 0.0;
        for (((&reference_mod, &degraded_mod), &reference_level), &noise) in modulation[0]
            .iter()
            .zip(&modulation[1])
            .zip(&state.average_energy[0])
            .zip(&self.noise_powered)
        {
            sum_mod_diff += (reference_mod - degraded_mod).abs() / (1.0 + reference_mod);
            weight += reference_level / (reference_level + 100.0 * noise);
        }

        let (probability, steps) = probability_of_detection(&smoothed);
        DetailedFrameMov {
            summary: FrameMov {
                mod_diff: 100.0 / NUM_BANDS as f64 * sum_mod_diff,
                weight,
            },
            probability,
            steps,
        }
    }

    fn spectrum(
        &self,
        fft_buffers: &mut FftBuffers,
        samples: &[f32],
        offset: usize,
    ) -> [f64; SPECTRUM_SIZE] {
        let mut fft_input: [f64; FRAME_SIZE] = std::array::from_fn(|i| {
            samples.get(offset + i).copied().unwrap_or(0.0) as f64 * self.window[i]
        });
        self.fft
            .process_with_scratch(
                &mut fft_input,
                &mut fft_buffers.output,
                &mut fft_buffers.scratch,
            )
            .expect("preallocated real FFT buffers have the correct lengths");
        std::array::from_fn(|i| fft_buffers.output[i].norm_sqr())
    }

    fn excitation(&self, spectrum: &[f64; SPECTRUM_SIZE]) -> [f64; NUM_BANDS] {
        let weighted: [f64; SPECTRUM_SIZE] =
            std::array::from_fn(|i| spectrum[i] * self.outer_middle_ear[i]);
        let grouped = std::array::from_fn(|band| {
            let mapping = self.mappings[band];
            let mut energy = mapping.lower_weight * weighted[mapping.lower_bin];
            for value in &weighted[mapping.lower_bin + 1..mapping.upper_bin] {
                energy += value;
            }
            energy += mapping.upper_weight * weighted[mapping.upper_bin];
            energy.max(1e-12) + self.internal_noise[band]
        });
        let spread =
            dispatch!(self.simd_level, simd => spread_raw(simd, &grouped, &self.spreading));
        std::array::from_fn(|i| spread[i] / self.spread_normalization[i])
    }
}

// The instantaneous PD arrays are kept separately from the two scalar MOVs.
// This wrapper supplies the small accessors used by channel collapsing.
struct DetailedFrameMov {
    summary: FrameMov,
    probability: [f64; NUM_BANDS],
    steps: [f64; NUM_BANDS],
}

impl std::ops::Deref for DetailedFrameMov {
    type Target = FrameMov;
    fn deref(&self) -> &Self::Target {
        &self.summary
    }
}

impl DetailedFrameMov {
    fn probability_at(&self, band: usize) -> f64 {
        self.probability[band]
    }
    fn steps_at(&self, band: usize) -> f64 {
        self.steps[band]
    }
}

fn time_constants(time_at_100_hz: f64, minimum_time: f64) -> ([f64; NUM_BANDS], [f64; NUM_BANDS]) {
    let a = std::array::from_fn(|i| {
        let time = minimum_time + (100.0 / FC[i]) * (time_at_100_hz - minimum_time);
        (-1.0 / (FRAME_RATE * time)).exp()
    });
    (a, a.map(|value| 1.0 - value))
}

#[inline(always)]
fn spread_raw<S: Simd>(
    _simd: S,
    energy: &[f64; NUM_BANDS],
    tables: &Spreading,
) -> [f64; NUM_BANDS] {
    let mut upper_powered = [0.0; NUM_BANDS];
    let mut normalized_powered = [0.0; NUM_BANDS];
    for band in 0..NUM_BANDS {
        let energy_dependent = tables.upper_slope[band] * pow_005(energy[band]);
        let upper_sum =
            (1.0 - positive_powi(energy_dependent, NUM_BANDS - band)) / (1.0 - energy_dependent);
        let normalized = energy[band] / (tables.lower_sum[band] + upper_sum - 1.0);
        upper_powered[band] = pow_04(energy_dependent);
        normalized_powered[band] = pow_04(normalized);
    }

    let mut spread = [0.0; NUM_BANDS];
    spread[NUM_BANDS - 1] = normalized_powered[NUM_BANDS - 1];
    for band in (0..NUM_BANDS - 1).rev() {
        spread[band] = tables.lower_powered * spread[band + 1] + normalized_powered[band];
    }
    for band in 0..NUM_BANDS - 1 {
        let slope = upper_powered[band];
        let slope2 = slope * slope;
        let slope3 = slope2 * slope;
        let slope4 = slope2 * slope2;
        let slope5 = slope4 * slope;
        let slope6 = slope4 * slope2;
        let slope7 = slope4 * slope3;
        let slope8 = slope4 * slope4;
        let mut value = normalized_powered[band];
        let mut chunks = spread[band + 1..].chunks_exact_mut(8);
        for targets in &mut chunks {
            targets[0] += value * slope;
            targets[1] += value * slope2;
            targets[2] += value * slope3;
            targets[3] += value * slope4;
            targets[4] += value * slope5;
            targets[5] += value * slope6;
            targets[6] += value * slope7;
            targets[7] += value * slope8;
            value *= slope8;
        }
        for target in chunks.into_remainder() {
            value *= slope;
            *target += value;
        }
    }
    // The final exponent is exactly 1 / 0.4 = 2.5, for which multiplication
    // and a square root are much cheaper than a generic pow implementation.
    spread.map(|value| value * value * value.sqrt())
}

#[inline(always)]
fn positive_powi(mut base: f64, mut exponent: usize) -> f64 {
    debug_assert!(exponent > 0);
    let mut result = 1.0;
    while exponent > 1 {
        if exponent & 1 != 0 {
            result *= base;
        }
        exponent >>= 1;
        base *= base;
    }
    result * base
}

fn probability_of_detection(
    energy: &[[f64; NUM_BANDS]; 2],
) -> ([f64; NUM_BANDS], [f64; NUM_BANDS]) {
    const C: [f64; 5] = [-0.198719, 0.0550197, -0.00102438, 5.05622e-6, 9.01033e-11];
    let mut probability = [0.0; NUM_BANDS];
    let mut steps = [0.0; NUM_BANDS];
    for band in 0..NUM_BANDS {
        let reference_db = 10.0 * energy[0][band].log10();
        let degraded_db = 10.0 * energy[1][band].log10();
        let difference = reference_db - degraded_db;
        let reference_louder = difference > 0.0;
        let level = if reference_louder {
            0.3 * reference_db + 0.7 * degraded_db
        } else {
            degraded_db
        };
        let threshold = if level > 0.0 {
            5.95072 * pow_171332(6.39468 / level)
                + C[0]
                + level * (C[1] + level * (C[2] + level * (C[3] + level * C[4])))
        } else {
            1e30
        };
        let ratio = difference / threshold;
        let ratio_squared = ratio * ratio;
        let ratio_fourth = ratio_squared * ratio_squared;
        let powered_difference = if reference_louder {
            ratio_fourth
        } else {
            ratio_fourth * ratio_squared
        };
        probability[band] = 1.0 - (-powered_difference).exp2();
        steps[band] = difference.trunc().abs() / threshold;
    }
    (probability, steps)
}

fn weighted_average(values: &[f64], weights: &[f64]) -> f64 {
    let weighted_sum = values.iter().zip(weights).map(|(x, w)| x * w).sum::<f64>();
    weighted_sum / weights.iter().sum::<f64>()
}

fn average_distorted_blocks(probabilities: &[f64], steps: &[f64]) -> f64 {
    let mut distorted = 0usize;
    let mut step_sum = 0.0;
    for (&probability, &frame_steps) in probabilities.iter().zip(steps) {
        if probability > 0.5 {
            distorted += 1;
            step_sum += frame_steps;
        }
    }
    if distorted == 0 {
        0.0
    } else if step_sum > 0.0 {
        (step_sum / distorted as f64).log10()
    } else {
        -0.5
    }
}

/// Applies the published 2f-model regression to its two PEAQ MOVs.
pub fn estimate_mushra(avg_mod_diff1: f64, adb: f64) -> f64 {
    let modulation_term = -0.0282 * avg_mod_diff1 - 0.8628;
    56.1345 / (1.0 + modulation_term * modulation_term) - 27.1451 * adb + 86.3515
}

fn data_boundaries<R: AsRef<[f32]>>(channels: &[R]) -> Option<(usize, usize)> {
    const WINDOW: usize = 5;
    const THRESHOLD: f64 = 200.0 / 32768.0;
    let len = channels[0].as_ref().len();
    if len == 0 {
        return None;
    }
    // PQdataBoundary collapses channels with `max`; a silent channel is -1
    // and therefore does not hide a boundary found in another channel.
    let start = channels
        .iter()
        .filter_map(|channel| boundary_start(channel.as_ref(), WINDOW, THRESHOLD))
        .max();
    let end = channels
        .iter()
        .filter_map(|channel| boundary_end(channel.as_ref(), WINDOW, THRESHOLD))
        .max();
    match (start, end) {
        (Some(start), Some(end)) => Some((start, end)),
        (None, None) => Some((0, 0)), // PQevalAudio's all-silent convention.
        _ => None,
    }
}

fn boundary_start(samples: &[f32], length: usize, threshold: f64) -> Option<usize> {
    if samples.is_empty() {
        return None;
    }
    let width = length.min(samples.len());
    let mut sum = samples[..width].iter().map(|x| x.abs() as f64).sum::<f64>();
    if sum > threshold {
        return Some(0);
    }
    if samples.len() < length {
        return None;
    }
    for start in 1..=samples.len() - length {
        sum += samples[start + length - 1].abs() as f64 - samples[start - 1].abs() as f64;
        if sum > threshold {
            return Some(start);
        }
    }
    None
}

fn boundary_end(samples: &[f32], length: usize, threshold: f64) -> Option<usize> {
    if samples.is_empty() {
        return None;
    }
    let width = length.min(samples.len());
    let mut end = samples.len() - 1;
    let mut sum = samples[samples.len() - width..]
        .iter()
        .map(|x| x.abs() as f64)
        .sum::<f64>();
    if sum > threshold {
        return Some(end);
    }
    if samples.len() < length {
        return None;
    }
    while end >= length {
        sum += samples[end - length].abs() as f64 - samples[end].abs() as f64;
        end -= 1;
        if sum > threshold {
            return Some(end);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimized_spreading_matches_direct_formula() {
        fn direct(energy: &[f64; NUM_BANDS], tables: &Spreading) -> [f64; NUM_BANDS] {
            let mut upper_powered = [0.0; NUM_BANDS];
            let mut normalized_powered = [0.0; NUM_BANDS];
            for band in 0..NUM_BANDS {
                let energy_dependent = tables.upper_slope[band] * energy[band].powf(0.05);
                let upper_sum = (1.0 - energy_dependent.powi((NUM_BANDS - band) as i32))
                    / (1.0 - energy_dependent);
                let normalized = energy[band] / (tables.lower_sum[band] + upper_sum - 1.0);
                upper_powered[band] = energy_dependent.powf(0.4);
                normalized_powered[band] = normalized.powf(0.4);
            }
            let mut spread = [0.0; NUM_BANDS];
            spread[NUM_BANDS - 1] = normalized_powered[NUM_BANDS - 1];
            for band in (0..NUM_BANDS - 1).rev() {
                spread[band] = tables.lower_powered * spread[band + 1] + normalized_powered[band];
            }
            for band in 0..NUM_BANDS - 1 {
                let mut value = normalized_powered[band];
                for target in spread.iter_mut().skip(band + 1) {
                    value *= upper_powered[band];
                    *target += value;
                }
            }
            spread.map(|value| value.powf(2.5))
        }

        let level = Level::new();
        let tables = Spreading::new();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..100 {
            let energy = std::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let exponent = 1013 + state % 51; // Approximately 1e-3 .. 1e12.
                let mantissa = state & 0xf_ffff_ffff_ffff;
                f64::from_bits((exponent << 52) | mantissa)
            });
            let expected = direct(&energy, &tables);
            let actual = dispatch!(level, simd => spread_raw(simd, &energy, &tables));
            for (actual, expected) in actual.into_iter().zip(expected) {
                let relative_error = ((actual - expected) / expected).abs();
                assert!(relative_error < 2e-13, "relative error {relative_error:e}");
            }
        }
    }

    #[test]
    fn published_regression_example() {
        let score = estimate_mushra(23.6545, 1.92561);
        assert!((score - 50.885_035_396_818_12).abs() < 1e-12);
    }

    #[test]
    fn boundary_search() {
        let signal = [0.0, 0.0, 0.0, 0.004, 0.004, 0.0, 0.0];
        assert_eq!(boundary_start(&signal, 5, 200.0 / 32768.0), Some(0));
        assert_eq!(boundary_end(&signal, 5, 200.0 / 32768.0), Some(6));
    }
}
