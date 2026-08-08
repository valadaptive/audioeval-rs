//! A pure-Rust port of Google's [Zimtohrli](https://github.com/google/zimtohrli), a perceptual audio evaluation metric.
//!
//! ```no_run
//! use zimtohrli::Zimtohrli;
//!
//! // This should be 48kHz PCM audio in [-1, 1]. If your audio is not 48kHz, you must resample it. Zimtohrli is not amplitude-invariant.
//! let reference: &[f32];
//! let degraded: &[f32];
//!
//! # reference = &[0.0; 48000];
//! # degraded = &[0.0; 48000];
//!
//! let zimt = Zimtohrli::default();
//!
//! // Construct spectrograms from PCM audio.
//! let mut reference_spec = zimt.analyze(reference);
//! let mut degraded_spec = zimt.analyze(degraded);
//!
//! // Compute the Zimtohrli distance (0.0 = identical, 1.0 = maximally different) between two spectrograms.
//! let distance = zimt.distance(&mut reference_spec, &mut degraded_spec);
//!
//! // If you know the two signals are already time-aligned, you can skip the DTW (dynamic time warping) step for much faster, and potentially more accurate, results.
//! // This requires the reference and degraded audio to have the same length.
//! let distance = zimt.distance_without_dtw(&mut reference_spec, &mut degraded_spec);
//!
//! // You can also map the distance to a "very approximate" mean opinion score (MOS).
//! let mos = Zimtohrli::mos_from_distance(distance);
//! println!("MOS: {mos}");
//! ```
//!
//! ## Usage notes
//!
//! Inputs are expected to be 48kHz mono signals. You must use something like [rubato](https://crates.io/crates/rubato) to resample the input before passing it in.
//!
//! Zimtohrli does not contain any binaural metrics. The Zimtohrli CLI handles stereo/multichannel audio by evaluating each channel's distance separately, and computing the root-mean-squared distance:
//!
//! ```no_run
//! # use zimtohrli::Zimtohrli;
//!
//! let reference: &[&[f32]];
//! let degraded: &[&[f32]];
//!
//! # reference = &[&[0.0; 48000]];
//! # degraded = &[&[0.0; 48000]];
//!
//! let zimt = Zimtohrli::default();
//! let mut sum_of_squares = 0.0;
//!
//! for (ref_channel, deg_channel) in reference.iter().zip(degraded.iter()) {
//!     let mut reference_spec = zimt.analyze(ref_channel);
//!     let mut degraded_spec = zimt.analyze(deg_channel);
//!
//!     let distance = zimt.distance(&mut reference_spec, &mut degraded_spec);
//!     sum_of_squares += distance * distance;
//! }
//!
//! let rms_distance = (sum_of_squares / reference.len() as f32).sqrt();
//! let mos = Zimtohrli::mos_from_distance(rms_distance);
//! println!("MOS: {mos}");
//! ```
//!
//! ## Conformance and exactness
//!
//! This crate matches the results of the original C++ `zimtohrli` repo (as of [this commit](https://github.com/google/zimtohrli/tree/67c28b1b5b78297a38ec01681863ab114c1e9841)) to 1e-5.
//!
//! Matching the C++ original *exactly* is impossible, since the C++ code itself is nondeterministic: it is compiled with "fast math" flags, and uses libm functions which may return different results across different platforms.
//!
//! However, this crate *is* entirely deterministic, and its own results should be bit-identical across all platforms. Rather than using the system libm, it uses the Rust `libm` crate for math. This crate does perform runtime CPU feature detection (see below), but intentionally foregoes non-deterministic optimizations dependent on platform features.
//!
//! ## Performance
//!
//! Some rough benchmarks on my Ryzen 7 7700X put this crate around 35-40% faster than the original C++ version on full analysis (spectrogram creation + distance calculation, not counting I/O or resampling). This goes for analysis both with and without the DTW step.
//!
//! The original C++ code does not perform any runtime CPU feature detection, whereas this crate does. When benchmarking, the original code was compiled with `-march=x86-64-v3`. For some reason, `x86-64-v4` was slower.
//!
//! If DTW is required, but conformance with the original C++ version is *not*, you may use the [`Zimtohrli::dtw_band_radius`] option to reduce the search radius considered during the DTW alignment step. This is an extension of the API exclusive to this crate.

use fearless_simd::{Level, Simd, dispatch, f32x16, prelude::*};
use libm::{cosf, exp, expf, logf, pow, sinf};
use std::{
    f32::{self, consts::PI},
    ops::{Index, IndexMut},
};

mod fast_pow;
mod pow_nsim_table;
mod pow_pp_table;
use fast_pow::{pow_p0, pow_p1, pow_pp};

const SAMPLE_RATE: f32 = 48000.0;

const NUM_ROTATORS: usize = 128;

/// Converts energy values in frequency channels to loudness in dB using
/// psychoacoustic weighting factors for each frequency band. Applies
/// frequency-dependent gain correction and logarithmic scaling.
fn loudness_db(channels: &mut [f32; NUM_ROTATORS]) {
    static MUL: [f32; NUM_ROTATORS] = [
        0.69111, 0.68478, 0.68763, 0.68845, 0.68595, 0.68576, 0.68883, 0.68932, 0.68713, 0.69239,
        0.68762, 0.68928, 0.68449, 0.69143, 0.69494, 0.69796, 0.69697, 0.70122, 0.72878, 0.79911,
        0.85713, 0.88063, 0.88563, 0.87561, 0.81948, 0.70435, 0.63479, 0.58382, 0.52065, 0.48390,
        0.46452, 0.47952, 0.52686, 0.63677, 0.75972, 0.89449, 0.97411, 1.01874, 1.01105, 0.99306,
        0.93613, 0.92825, 0.93149, 0.98687, 1.05782, 1.16461, 1.25028, 1.30768, 1.31484, 1.28574,
        1.23002, 1.15336, 1.08800, 1.01472, 0.94610, 0.91856, 0.87797, 0.85825, 0.82836, 0.82198,
        0.81394, 0.82724, 0.84235, 0.86009, 0.88276, 0.89349, 0.92543, 0.94822, 0.98526, 0.99730,
        1.00532, 1.02506, 1.03689, 1.04897, 1.05307, 1.05817, 1.05174, 1.04766, 1.03553, 1.03437,
        1.03238, 1.05164, 1.08115, 1.13753, 1.21037, 1.31175, 1.44154, 1.52549, 1.60840, 1.67304,
        1.71593, 1.72853, 1.76630, 1.70865, 1.68923, 1.65506, 1.57241, 1.51275, 1.37840, 1.28644,
        1.23809, 1.21714, 1.30432, 1.30430, 1.33396, 1.34255, 1.33987, 1.35309, 1.35169, 1.35219,
        1.35385, 1.35851, 1.34995, 1.20201, 1.17218, 1.19284, 1.23571, 1.34281, 1.16209, 0.89999,
        0.89264, 1.08696, 0.78787, 0.78445, 1.12917, 0.65317, 1.02086, 1.11196,
    ];

    const BASE_NOISE: f32 = 766068.03396368888;
    static BASE_NOISE_SLOPE: [f32; 32] = [
        -427.1872751241109,
        -370.2893289163535,
        -357.01506023770378,
        -301.28879097655118,
        -216.78500670398833,
        -168.07806679629724,
        -168.71805754864141,
        -159.53956835871321,
        -268.72445005379404,
        -311.16419962879075,
        -277.03504398276948,
        -288.39213525341091,
        -305.32237068568082,
        -258.6335011904703,
        -254.78634459132866,
        -181.46038594163568,
        -93.950223670617163,
        -88.818104801961908,
        -26.156023442931389,
        -38.752447643769138,
        -47.906764099227942,
        -21.676071849485375,
        10.884646488419072,
        21.595865980708961,
        -52.559415237056015,
        -57.62886752507012,
        -80.132855392693315,
        -84.248190048411175,
        -87.193989053900296,
        -134.86546270102167,
        -146.23587896776439,
        -211.30970199319108,
    ];
    let mut noise = BASE_NOISE;
    for k in 0..NUM_ROTATORS {
        channels[k] = logf(channels[k] + noise) * MUL[k];
        noise += BASE_NOISE_SLOPE[k >> 2];
    }
}

/// Ear drum and other receiving mass-spring objects are modeled through the
/// Resonator. Resonator is a non-linear process and does complex spectral
/// shifting of energy.
#[derive(Default)]
struct Resonator {
    acc0: f32,
    acc1: f32,
}

impl Resonator {
    /// Resonate and attenuate.
    fn update(&mut self, signal: f32) -> f32 {
        // These parameters relate to a population of ear drums.
        const MUL_0: f32 = 0.97018703367139569;
        const MUL_1: f32 = -0.02209312182872265;

        self.acc0 = (MUL_0 * self.acc0) + (MUL_1 * self.acc1) + signal;
        self.acc1 += self.acc0;
        self.acc0
    }
}

/// Computes dot product of two 32-element float arrays.
#[inline(always)]
fn dot32(a: &[f32; 32], b: &[f32; 32]) -> f32 {
    // The four fixed accumulators expose lane parallelism without changing the
    // final reduction order between SIMD levels.
    let mut sum = [0.0f32; 4];
    for i in (0..32).step_by(4) {
        sum[0] += a[i] * b[i];
        sum[1] += a[i + 1] * b[i + 1];
        sum[2] += a[i + 2] * b[i + 2];
        sum[3] += a[i + 3] * b[i + 3];
    }
    (sum[0] + sum[1]) + (sum[2] + sum[3])
}

// Center frequencies of the filter bank, plus one frequency in both ends.
static FREQ: [f32; NUM_ROTATORS + 2] = [
    17.858, 24.349, 33.199, 42.359, 51.839, 61.651, 71.805, 82.315, 93.192, 104.449, 116.099,
    128.157, 140.636, 153.552, 166.919, 180.754, 195.072, 209.890, 225.227, 241.099, 257.527,
    274.528, 292.124, 310.336, 329.183, 348.690, 368.879, 389.773, 411.398, 433.778, 456.941,
    480.914, 505.725, 531.403, 557.979, 585.484, 613.950, 643.411, 673.902, 705.459, 738.119,
    771.921, 806.905, 843.111, 880.584, 919.366, 959.503, 1001.04, 1044.03, 1088.53, 1134.58,
    1182.24, 1231.57, 1282.62, 1335.46, 1390.14, 1446.73, 1505.31, 1565.93, 1628.67, 1693.60,
    1760.80, 1830.35, 1902.34, 1976.84, 2053.94, 2133.74, 2216.33, 2301.81, 2390.27, 2481.83,
    2576.58, 2674.65, 2776.15, 2881.19, 2989.91, 3102.43, 3218.88, 3339.40, 3464.14, 3593.23,
    3726.84, 3865.12, 4008.23, 4156.35, 4309.64, 4468.30, 4632.49, 4802.43, 4978.31, 5160.34,
    5348.72, 5543.70, 5745.49, 5954.34, 6170.48, 6394.18, 6625.70, 6865.32, 7113.31, 7369.97,
    7635.61, 7910.53, 8195.06, 8489.53, 8794.30, 9109.73, 9436.18, 9774.04, 10123.7, 10485.6,
    10860.1, 11247.8, 11648.9, 12064.2, 12493.9, 12938.7, 13399.0, 13875.3, 14368.4, 14878.7,
    15406.8, 15953.4, 16519.1, 17104.5, 17710.4, 18337.6, 18986.6, 19658.3, 20352.7,
];

/// Returns the center frequency in Hz for filter bank channel `i`. The 128
/// channels are spaced to match human auditory perception, with finer
/// resolution at lower frequencies.
fn freq(i: usize) -> f32 {
    FREQ[i + 1]
}

/// Calculates the effective bandwidth in Hz for filter bank channel `i`. Uses
/// geometric mean spacing between adjacent channels.
fn calculate_bandwidth_in_hz(i: usize) -> f32 {
    // TODO: the C++ version appears to do all math in float precision but returns a double.
    // FREQ[i] is freq(i - 1) without the usize underflow at i == 0.
    (freq(i + 1) * freq(i)).sqrt() - (FREQ[i] * freq(i)).sqrt()
}

/// Core signal processing engine using rotating phasors (Goertzel-like
/// algorithm) for efficient frequency analysis. Implements the Zimtohrli/Tabuli
/// filterbank.
struct EnergyOutput<'a> {
    values: &'a mut [f32],
    step: usize,
    stride: usize,
    weight: f32,
    has_next: bool,
}

struct Rotators {
    /// Four arrays of rotators, with memory layout for up to 128-way
    /// simd-parallel. [0..1] is real and imag for rotation speed [2..3] is real
    /// and imag for a frequency rotator of length sqrt(gain[i])
    rot: [[f32; NUM_ROTATORS]; 4],
    /// [0..1] is for real and imag of 1st leaking accumulation
    /// [2..3] is for real and imag of 2nd leaking accumulation
    /// [4..5] is for real and imag of 3rd leaking accumulation
    accu: [[f32; NUM_ROTATORS]; 6],
    window: [f32; NUM_ROTATORS],
    gain: [f32; NUM_ROTATORS],
}

impl Rotators {
    /// Renormalizes the rotating phasors to prevent numerical drift.
    /// Called periodically during signal processing.
    fn occasionally_renormalize(&mut self) {
        for i in 0..NUM_ROTATORS {
            let norm = self.gain[i]
                / (self.rot[2][i] * self.rot[2][i] + self.rot[3][i] * self.rot[3][i]).sqrt();
            self.rot[2][i] *= norm;
            self.rot[3][i] *= norm;
        }
    }

    /// Updates all rotators and accumulators with a new signal sample, then
    /// adds their energy to the current output step (and optionally the next).
    #[inline(always)]
    fn increment_all<S: Simd>(&mut self, simd: S, signal: f32, output: EnergyOutput<'_>) {
        let EnergyOutput {
            values: out,
            step: out_ix,
            stride: out_stride,
            weight,
            has_next: has_next_output,
        } = output;
        let signal = f32x16::splat(simd, signal);
        let weight = f32x16::splat(simd, weight);
        let next_weight = f32x16::splat(simd, 1.0) - weight;
        for i in (0..NUM_ROTATORS).step_by(16) {
            let range = i..i + 16;
            let w = f32x16::from_slice(simd, &self.window[range.clone()]);
            let mut acc0 = f32x16::from_slice(simd, &self.accu[0][range.clone()]) * w;
            let mut acc1 = f32x16::from_slice(simd, &self.accu[1][range.clone()]) * w;
            let mut acc2 = f32x16::from_slice(simd, &self.accu[2][range.clone()]) * w;
            let mut acc3 = f32x16::from_slice(simd, &self.accu[3][range.clone()]) * w;
            let mut acc4 = f32x16::from_slice(simd, &self.accu[4][range.clone()]) * w;
            let mut acc5 = f32x16::from_slice(simd, &self.accu[5][range.clone()]) * w;

            // Preserve the original update order: later accumulators consume
            // the already-updated value from the preceding stage.
            acc2 += acc0;
            acc3 += acc1;
            acc4 += acc2;
            acc5 += acc3;
            let a = f32x16::from_slice(simd, &self.rot[2][range.clone()]);
            let b = f32x16::from_slice(simd, &self.rot[3][range.clone()]);
            acc0 += a * signal;
            acc1 += b * signal;
            let rot0 = f32x16::from_slice(simd, &self.rot[0][range.clone()]);
            let rot1 = f32x16::from_slice(simd, &self.rot[1][range.clone()]);
            let new_a = (rot0 * a) - (rot1 * b);
            let new_b = (rot0 * b) + (rot1 * a);

            acc0.store_slice(&mut self.accu[0][range.clone()]);
            acc1.store_slice(&mut self.accu[1][range.clone()]);
            acc2.store_slice(&mut self.accu[2][range.clone()]);
            acc3.store_slice(&mut self.accu[3][range.clone()]);
            acc4.store_slice(&mut self.accu[4][range.clone()]);
            acc5.store_slice(&mut self.accu[5][range.clone()]);
            new_a.store_slice(&mut self.rot[2][range.clone()]);
            new_b.store_slice(&mut self.rot[3][range.clone()]);

            // Consume acc4/acc5 while they are still in registers rather than
            // loading them again in a second frequency-channel loop.
            let energy = (acc4 * acc4) + (acc5 * acc5);
            let current_start = out_ix * out_stride + i;
            let current_range = current_start..current_start + 16;
            let current = f32x16::from_slice(simd, &out[current_range.clone()]);
            if has_next_output {
                let next_start = current_start + out_stride;
                let next_range = next_start..next_start + 16;
                let next = f32x16::from_slice(simd, &out[next_range.clone()]);
                (next + next_weight * energy).store_slice(&mut out[next_range]);
                (current + weight * energy).store_slice(&mut out[current_range]);
            } else {
                (current + energy).store_slice(&mut out[current_range]);
            }
        }
    }

    #[inline(always)]
    fn filter_and_downsample_inner<S: Simd>(
        simd: S,
        input: &[f32],
        out: &mut [f32],
        out_shape0: usize,
        out_stride: usize,
        downsample: usize,
    ) {
        const KHZ_TO_RAD: f32 = 2.0 * PI / SAMPLE_RATE;
        const WINDOW: f64 = 0.9996073584827937;
        const BANDWIDTH_MAGIC: f64 = 0.73227703638356523;

        // A big value for normalization. Ideally 1.0, but this works better
        // for an unknown reason even if the base noise level is adapted similarly.
        const SCALE: f64 = 931912404783.44507;

        let gainer = (SCALE / downsample as f64).sqrt() as f32;
        let mut rotators = Rotators {
            rot: [[0.0; _]; _],
            accu: [[0.0; _]; _],
            window: [0.0; _],
            gain: [0.0; _],
        };
        for i in 0..NUM_ROTATORS {
            let bandwidth = calculate_bandwidth_in_hz(i);
            rotators.window[i] = pow(WINDOW, bandwidth as f64 * BANDWIDTH_MAGIC) as f32;
            let window_m1 = 1.0 - rotators.window[i];
            let f = freq(i) * KHZ_TO_RAD;
            rotators.gain[i] = gainer * (window_m1 * window_m1 * window_m1) * freq(i) / bandwidth;
            rotators.rot[0][i] = cosf(f);
            rotators.rot[1][i] = -sinf(f);
            rotators.rot[2][i] = rotators.gain[i];
            // rotators.rot[3][i] is already 0.0
        }

        for zz in 0..out_shape0 {
            for k in 0..NUM_ROTATORS {
                out[zz * out_stride + k] = 0.0;
            }
        }

        let downsample_window = (0..downsample)
            .map(|i| {
                (1.0 / (1.0
                    + exp(
                        8.0246040186567118 * ((2.0 / downsample as f64) * (i as f64 + 0.5) - 1.0)
                    ))) as f32
            })
            .collect::<Vec<f32>>();

        let mut resonator = Resonator::default();
        let mut out_ix = 0;

        const KERNEL_SIZE: usize = 32;

        static RESO_KERNEL: [f32; KERNEL_SIZE] = [
            -0.0076247065632976318,
            0.0039104155534537069,
            0.0006684663662401936,
            0.0071559704794996589,
            -0.0027931528839390098,
            0.0001368658992949717,
            -0.0065802540559526824,
            -0.006574266432654235,
            0.0034740030608061525,
            0.0030263702264320012,
            -0.0029378401470635364,
            0.0034368516858611412,
            0.0020915727560313845,
            -0.001541122014895714,
            0.0033152434154573407,
            0.0015489639154823477,
            -0.012691890416423556,
            -0.00027840484849307723,
            -0.0010427818083574192,
            -0.0087889956707155811,
            -0.0066266333272295289,
            -0.00080043637110705163,
            -0.0072998536521213225,
            0.0036816757141278035,
            -0.00031555808271841742,
            0.00099264355318687508,
            -0.0012897138783731826,
            0.0013771982014390573,
            0.0070121198631592861,
            -0.0016488166452599629,
            -0.00727301918260589,
            0.010964231292090421,
        ];
        static LINEAR_KERNEL: [f32; KERNEL_SIZE] = [
            -0.19947158175459692,
            0.020092596724127186,
            -0.065549345816240306,
            0.059315467827374985,
            0.24679907672434401,
            -0.14582584331716622,
            -0.083626881941168935,
            0.31874018187263292,
            0.22397287387339976,
            0.036279108994617872,
            -0.13919343535956649,
            0.04950990842192754,
            -0.027271514202057801,
            -0.00099846257278084238,
            -0.10798654028268029,
            -0.10489917207275569,
            -0.095906755569884164,
            -0.21168952706515187,
            0.83249555081867532,
            0.58484205043268755,
            -0.21828800943250842,
            0.080106893472851701,
            0.93016317182367492,
            -0.49663918345960828,
            -1.6197347842868257,
            -0.18383066061195377,
            0.6236802270978099,
            1.1976849288800944,
            -0.70212522492743401,
            0.90598962344860279,
            -0.0018858573753579057,
            -0.41452533138089309,
        ];

        let mut in_ix = 0;
        let mut dix = 0;
        while in_ix + KERNEL_SIZE < input.len() {
            let weight = downsample_window[dix];
            let in_slice: &[f32; KERNEL_SIZE] =
                input[in_ix..in_ix + KERNEL_SIZE].try_into().unwrap();
            rotators.increment_all(
                simd,
                resonator.update(dot32(in_slice, &RESO_KERNEL)) + dot32(in_slice, &LINEAR_KERNEL),
                EnergyOutput {
                    values: out,
                    step: out_ix,
                    stride: out_stride,
                    weight,
                    has_next: out_ix + 1 < out_shape0,
                },
            );

            dix += 1;
            if dix == downsample || in_ix + KERNEL_SIZE + 1 == input.len() {
                // NB: the parentheses around `&mut out[..]` matter: without
                // them, `try_into` resolves to the by-value `TryFrom<&[f32]>
                // for [f32; N]` impl and loudness_db mutates a temporary copy
                // of the row instead of the spectrogram itself.
                loudness_db(
                    (&mut out[out_stride * out_ix..out_stride * out_ix + NUM_ROTATORS])
                        .try_into()
                        .unwrap(),
                );

                out_ix += 1;
                if out_ix >= out_shape0 {
                    break;
                }
                dix = 0;
                rotators.occasionally_renormalize();
            }

            in_ix += 1;
        }
    }

    fn filter_and_downsample(
        level: Level,
        input: &[f32],
        out: &mut [f32],
        out_shape0: usize,
        out_stride: usize,
        downsample: usize,
    ) {
        dispatch!(level, simd => {
            Self::filter_and_downsample_inner(simd, input, out, out_shape0, out_stride, downsample)
        })
    }
}

/// A simple buffer of float samples describing a spectrogram with a given
/// number of steps and feature dimensions.
///
///
/// The values buffer is populated like:
/// [
///   [sample0_dim0, sample0_dim1, ..., sample0_dimn],
///   [sample1_dim0, sample1_dim1, ..., sample1_dimn],
///   ...,
///   [samplem_dim0, samplem_dim1, ..., samplem_dimn],
/// ]
pub struct Spectrogram {
    pub num_steps: usize,
    pub num_dims: usize,
    pub values: Box<[f32]>,
}

impl Spectrogram {
    fn new(num_steps: usize, num_dims: usize) -> Self {
        Self {
            num_steps,
            num_dims,
            values: vec![0.0; num_steps * num_dims].into_boxed_slice(),
        }
    }

    /// Returns the maximum absolute value across all spectrogram values.
    fn max(&self) -> f32 {
        // TODO: the C++ version does some weird strided thing that just seems totally unnecessary
        self.values
            .iter()
            .map(|v| v.abs())
            .reduce(f32::max)
            .unwrap_or_default()
    }

    /// Multiplies all spectrogram values by the given factor.
    fn rescale(&mut self, f: f32) {
        for v in self.values.iter_mut() {
            *v *= f;
        }
    }
}

impl Index<usize> for Spectrogram {
    type Output = [f32];

    fn index(&self, n: usize) -> &Self::Output {
        &self.values[n * self.num_dims..(n + 1) * self.num_dims]
    }
}

impl IndexMut<usize> for Spectrogram {
    fn index_mut(&mut self, n: usize) -> &mut Self::Output {
        &mut self.values[n * self.num_dims..(n + 1) * self.num_dims]
    }
}

/// Computes windowed mean values over a 2D spectrogram using efficient
/// prefix sum computation. Used by NSIM to compute local statistics.
/// num_steps: number of time steps
/// num_channels: number of frequency channels
/// step_window: window size in time dimension
/// channel_window: window size in frequency dimension
/// input_loader: function(step, channel) that loads input values
fn window_mean(
    num_steps: usize,
    num_channels: usize,
    step_window: usize,
    channel_window: usize,
    mut input_loader: impl FnMut(usize, usize) -> f32,
) -> Spectrogram {
    let mut tmp_a = Spectrogram::new(num_steps, num_channels);
    let mut tmp_b = Spectrogram::new(num_steps, num_channels);

    // Populate tmp_b with prefix sums across the step axis.
    let channel_prefix_sum_data = &mut tmp_b[0];
    for (channel_index, item) in channel_prefix_sum_data.iter_mut().enumerate() {
        *item = input_loader(0, channel_index);
    }

    for step_index in 1..num_steps {
        for channel_index in 0..num_channels {
            tmp_b[step_index][channel_index] =
                input_loader(step_index, channel_index) + tmp_b[step_index - 1][channel_index];
        }
    }

    // Populate tmp_a with windowed sums across the step axis using the prefix
    // sums in tmp_b.
    // 1: Copy the step_window first rows from tmp_b to tmp_a.
    tmp_a.values[..step_window * num_channels]
        .copy_from_slice(&tmp_b.values[..step_window * num_channels]);

    // 2: Compute windowed sums by subtracting prefix sums from each other.
    for step_index in step_window..num_steps {
        let channel_window_sum_data = &mut tmp_a[step_index];
        let curr_window_sum_data = &tmp_b[step_index];
        let prev_window_sum_data = &tmp_b[step_index - step_window];
        for channel_index in 0..num_channels {
            channel_window_sum_data[channel_index] =
                curr_window_sum_data[channel_index] - prev_window_sum_data[channel_index];
        }
    }

    // Populate tmp_b with prefix sums across the channel axis of the windowed
    // sums across the step axis in tmp_a.
    for step_index in 0..num_steps {
        let channel_window_sum_data = &tmp_a[step_index];
        let step_prefix_sum_data = &mut tmp_b[step_index];
        step_prefix_sum_data[0] = channel_window_sum_data[0];
        for channel_index in 1..num_channels {
            step_prefix_sum_data[channel_index] =
                step_prefix_sum_data[channel_index - 1] + channel_window_sum_data[channel_index];
        }

        // Populate tmp_a with windowed sums across steps-and-channels axes using
        // the "prefix sums across the channel axis and windowed sums across the
        // step axis" of tmp_b.
        let step_prefix_sum_data = &tmp_b[step_index];
        let step_window_sum_data = &mut tmp_a[step_index];
        step_window_sum_data[..channel_window]
            .copy_from_slice(&step_prefix_sum_data[..channel_window]);
        for channel_index in channel_window..num_channels {
            step_window_sum_data[channel_index] = step_prefix_sum_data[channel_index]
                - step_prefix_sum_data[channel_index - channel_window];
        }
    }

    // Divide all windowed sums by step_window * channel_window to make them mean
    // values.
    let reciprocal = 1.0 / (step_window * channel_window) as f32;
    for step_index in 0..num_steps {
        let result_data = &mut tmp_a[step_index];
        for result in result_data.iter_mut() {
            *result *= reciprocal;
        }
    }

    tmp_a
}

// Represents how the two spectrograms are aligned in time.
trait Alignment {
    const IS_PRE_ALIGNED: bool;
    fn step_a(&self, step: usize) -> usize;
    fn step_b(&self, step: usize) -> usize;
    fn num_steps(&self, spectrogram: &Spectrogram) -> usize;
}

/// Assume the signals are already perfectly aligned.
struct PreAligned;

impl Alignment for PreAligned {
    const IS_PRE_ALIGNED: bool = true;
    fn step_a(&self, step: usize) -> usize {
        step
    }

    fn step_b(&self, step: usize) -> usize {
        step
    }

    fn num_steps(&self, spectrogram: &Spectrogram) -> usize {
        spectrogram.num_steps
    }
}

/// The warp path calculated by DTW, mapping frames between a and b.
struct Warped(Vec<[usize; 2]>);

impl Alignment for Warped {
    const IS_PRE_ALIGNED: bool = false;
    fn step_a(&self, step: usize) -> usize {
        self.0[step][0]
    }

    fn step_b(&self, step: usize) -> usize {
        self.0[step][1]
    }

    fn num_steps(&self, _spectrogram: &Spectrogram) -> usize {
        self.0.len()
    }
}

/// Performs the NSIM similarity computation using the specified alignment.
fn nsim<A: Alignment>(
    a: &Spectrogram,
    b: &Spectrogram,
    alignment: A,
    step_window: usize,
    channel_window: usize,
) -> f32 {
    assert_eq!(a.num_dims, b.num_dims);

    if A::IS_PRE_ALIGNED {
        assert_eq!(a.num_steps, b.num_steps);
    }

    let num_channels = a.num_dims;
    let num_steps = alignment.num_steps(a);

    if num_steps == 0 || num_channels == 0 || step_window == 0 || channel_window == 0 {
        return 0.0;
    }

    let step_window = step_window.min(num_steps);
    let channel_window = channel_window.min(num_channels);

    let mean_a = window_mean(
        num_steps,
        num_channels,
        step_window,
        channel_window,
        |step_index, channel_index| a[alignment.step_a(step_index)][channel_index],
    );
    let mean_b = window_mean(
        num_steps,
        num_channels,
        step_window,
        channel_window,
        |step_index, channel_index| b[alignment.step_b(step_index)][channel_index],
    );

    // NB: This computes (value - mean) using the mean computed for the window
    // at the same position as the value, so that each value gets a different mean
    // subtracted.
    let var_a = window_mean(
        num_steps,
        num_channels,
        step_window,
        channel_window,
        |step_index, channel_index| {
            let delta = a[alignment.step_a(step_index)][channel_index]
                - mean_a[alignment.step_a(step_index)][channel_index];
            delta * delta
        },
    );
    let var_b = window_mean(
        num_steps,
        num_channels,
        step_window,
        channel_window,
        |step_index, channel_index| {
            let delta = b[alignment.step_b(step_index)][channel_index]
                - mean_b[alignment.step_b(step_index)][channel_index];
            delta * delta
        },
    );
    let cov = window_mean(
        num_steps,
        num_channels,
        step_window,
        channel_window,
        #[inline(always)]
        |step_index, channel_index| {
            let delta_a = a[alignment.step_a(step_index)][channel_index]
                - mean_a[alignment.step_a(step_index)][channel_index];
            let delta_b = b[alignment.step_b(step_index)][channel_index]
                - mean_b[alignment.step_b(step_index)][channel_index];
            delta_a * delta_b
        },
    );

    // nsim-inspired ad hoc aggregation
    // main changes:
    // The aggregation tries to be more L1 than L2
    // Clamping of structure value
    //
    // These changes were measured to be small improvements on a multi-corpus
    // test.
    const C1: f32 = 26.426389124321354;
    const C3: f32 = 1.9522719384622791;
    const C8: f32 = 0.6325126087671703;
    // The intensity exponent P0 = 1.0500187278772866 and structure exponent
    // P1 = 0.25808223975919764 are baked into the fixed-exponent kernels
    // [pow_p0]/[pow_p1].

    let mut nsim_sum = 0.0f64;
    for step_index in 0..num_steps {
        let mut nsim_accu = 0.0f64;
        for channel_index in 0..num_channels {
            let mean_a_vec = mean_a[step_index][channel_index];
            let mean_b_vec = mean_b[step_index][channel_index];
            let std_a_vec = var_a[step_index][channel_index].sqrt();
            let std_b_vec = var_b[step_index][channel_index].sqrt();
            let cov_vec = cov[step_index][channel_index];
            let intensity = pow_p0(
                (2.0 * (mean_a_vec * mean_b_vec).sqrt() + C1)
                    / (mean_a_vec.abs() + mean_b_vec.abs() + C1),
            );
            let structure_base = (cov_vec + C3) / (std_a_vec * std_b_vec + C3);
            let structure_clamped = structure_base.max(C8);
            let structure = pow_p1(structure_clamped);
            let nsim = intensity * structure;
            nsim_accu += nsim as f64;
        }
        nsim_sum += nsim_accu;
    }

    (nsim_sum / (num_steps * num_channels) as f64).clamp(0.0, 1.0) as f32
}

/// A buffer of double cost values describing the time warp costs between two
/// spectrograms.
///
/// Optionally restricted to a Sakoe-Chiba band around the diagonal: only cells
/// within `radius` steps of the straight line from (0, 0) to
/// (steps_a - 1, steps_b - 1) are stored and computed; everything outside is
/// treated as unreachable (infinite cost). This is not part of the C++
/// original.
struct CostMatrix {
    /// First in-band column of each row (inclusive).
    row_lo: Vec<usize>,
    /// Last in-band column of each row (inclusive).
    row_hi: Vec<usize>,
    /// Stride between stored rows: the widest row's band width.
    width: usize,
    values: Vec<f64>,
}

impl CostMatrix {
    fn new(steps_a: usize, steps_b: usize, band_radius: Option<usize>) -> Self {
        let (row_lo, row_hi) = match band_radius {
            None => (vec![0; steps_a], vec![steps_b - 1; steps_a]),
            Some(radius) => {
                // The band follows the diagonal connecting the two corners, so
                // that inputs of different lengths get a straight-line time
                // mapping plus `radius` steps of allowed drift on both sides.
                let slope = if steps_a > 1 {
                    (steps_b - 1) as f64 / (steps_a - 1) as f64
                } else {
                    0.0
                };
                let mut row_lo = Vec::with_capacity(steps_a);
                let mut row_hi = Vec::with_capacity(steps_a);
                for i in 0..steps_a {
                    let center = i as f64 * slope;
                    row_lo.push((center.floor() as usize).saturating_sub(radius));
                    row_hi.push((center.ceil() as usize + radius).min(steps_b - 1));
                }
                // Keep every in-band cell connected to its neighbors in the
                // rows above and below, even when the diagonal is steep
                // (very different input lengths): the left edge may only
                // advance by one column per row, and the right edge may not
                // retreat. Without this, the DP (and the greedy path tracker,
                // which only moves in +1 steps) could get stranded on a band
                // edge with no reachable in-band neighbor.
                for i in 1..steps_a {
                    row_lo[i] = row_lo[i].min(row_lo[i - 1] + 1);
                    row_hi[i] = row_hi[i].max(row_hi[i - 1]);
                }
                (row_lo, row_hi)
            }
        };

        let width = row_lo
            .iter()
            .zip(row_hi.iter())
            .map(|(lo, hi)| hi - lo + 1)
            .max()
            .unwrap_or(0);
        let mut values = vec![f64::MAX; steps_a * width];
        // row_lo[0] is always 0, so this is cell (0, 0).
        values[0] = 0.0;

        Self {
            row_lo,
            row_hi,
            width,
            values,
        }
    }

    fn get(&self, step_a: usize, step_b: usize) -> f64 {
        let lo = self.row_lo[step_a];
        if step_b < lo || step_b > self.row_hi[step_a] {
            return f64::MAX;
        }
        self.values[step_a * self.width + (step_b - lo)]
    }

    fn set(&mut self, step_a: usize, step_b: usize, value: f64) {
        let lo = self.row_lo[step_a];
        debug_assert!(step_b >= lo && step_b <= self.row_hi[step_a]);
        self.values[step_a * self.width + (step_b - lo)] = value;
    }
}

/// Computes the perceptual distance between two spectrogram frames.
/// Uses p norm with psychoacoustic weighting.
/// Used by DTW to compute frame-to-frame alignment costs.
#[inline(always)]
fn delta_norm(a: &Spectrogram, b: &Spectrogram, step_a: usize, step_b: usize) -> f64 {
    let dims_a = &a[step_a];
    let dims_b = &b[step_b];
    assert_eq!(dims_a.len(), dims_b.len());

    // Multiple accumulators so the f64 sum vectorizes instead of forming one
    // serial dependency chain. This reorders the additions, but the C++ version
    // compiles with `-fassociative-math` anyway.
    const LANES: usize = 8;
    let mut sums = [0.0f64; LANES];
    let mut chunks_a = dims_a.chunks_exact(LANES);
    let mut chunks_b = dims_b.chunks_exact(LANES);
    for (ca, cb) in (&mut chunks_a).zip(&mut chunks_b) {
        for i in 0..LANES {
            let delta = ca[i] - cb[i];
            sums[i] += (delta * delta) as f64;
        }
    }

    let mut result = 0.0f64;
    for (&a, &b) in chunks_a.remainder().iter().zip(chunks_b.remainder().iter()) {
        let delta = a - b;
        result += (delta * delta) as f64;
    }

    result +=
        ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));

    // The exponent PP = 0.32264042946823823 is baked into pow_pp (see
    // fast_pow.rs). Bafflingly, the C++ version defines it as a `const
    // float`, reducing precision for no reason since the calculation is done
    // at double precision anyway.
    pow_pp(result)
}

#[inline(always)]
fn dtw_inner(spec_a: &Spectrogram, spec_b: &Spectrogram, band_radius: Option<usize>) -> Warped {
    // Sanity check that both spectrograms have the same number of feature
    // dimensions.
    assert_eq!(spec_a.num_dims, spec_b.num_dims);

    let mut cost_matrix = CostMatrix::new(spec_a.num_steps, spec_b.num_steps, band_radius);

    // Compute cost as cost as weighted sum of feature dimension norms to each
    // cell.
    // kMul00 value below 1.0 reduces the cost of going in sync, advancing
    // a and b traversal separately is a distance of 1. Purely geometrically
    // sqrt(2) might be a good value, but this works better for an unknown
    // reason (favoring a and b traversing together).
    const MUL_00: f64 = 0.90394786214451761;

    for spec_a_index in 1..spec_a.num_steps {
        let lo = cost_matrix.row_lo[spec_a_index].max(1);
        let hi = cost_matrix.row_hi[spec_a_index];
        for spec_b_index in lo..=hi {
            let cost_at_index = delta_norm(spec_a, spec_b, spec_a_index, spec_b_index);
            let sync_cost = cost_matrix.get(spec_a_index - 1, spec_b_index - 1);
            let bwd_cost = cost_matrix.get(spec_a_index - 1, spec_b_index);
            let fwd_cost = cost_matrix.get(spec_a_index, spec_b_index - 1);
            let unsync_cost = bwd_cost.min(fwd_cost);
            let costmin = (sync_cost + MUL_00 * cost_at_index).min(unsync_cost + cost_at_index);
            cost_matrix.set(spec_a_index, spec_b_index, costmin);
        }
    }

    // Track the cheapest path through the cost matrix.
    let mut result = Vec::new();
    let mut pos = [0; 2];
    result.push(pos);
    while pos[0] + 1 < spec_a.num_steps && pos[1] + 1 < spec_b.num_steps {
        let mut min_cost = f64::MAX;
        for test_pos in [
            [pos[0] + 1, pos[1] + 1],
            [pos[0] + 1, pos[1]],
            [pos[0], pos[1] + 1],
        ] {
            let cost = cost_matrix.get(test_pos[0], test_pos[1]);
            if cost < min_cost {
                min_cost = cost;
                pos = test_pos;
            }
        }
        result.push(pos);
    }

    Warped(result)
}

// Computes the DTW (https://en.wikipedia.org/wiki/Dynamic_time_warping)
// between two arrays.
//
// If `band_radius` is set, only the Sakoe-Chiba band within that many steps of
// the corner-to-corner diagonal is explored (see [CostMatrix]). The result is
// identical to the full DTW whenever the optimal warp path stays inside the
// band; misalignments larger than the band cannot be recovered.
fn dtw(
    level: Level,
    spec_a: &Spectrogram,
    spec_b: &Spectrogram,
    band_radius: Option<usize>,
) -> Warped {
    dispatch!(level, _ => dtw_inner(spec_a, spec_b, band_radius))
}

/// Main class for psychoacoustic audio analysis.
/// Converts audio signals to perceptual spectrograms and computes
/// perceptual distance between audio signals using the Zimtohrli metric.
/// Expected input: 48kHz mono audio with samples in range [-1, 1].
pub struct Zimtohrli {
    /// SIMD feature level.
    pub level: Level,
    /// The window in perceptual_sample_rate time steps when compting the NSIM.
    pub nsim_step_window: usize,
    /// The window in channels when computing the NSIM.
    pub nsim_channel_window: usize,
    pub perceptual_sample_rate: f32,
    /// The reference dB SPL of a sine signal of amplitude 1.
    pub full_scale_sine_db: f32,
    /// Optional Sakoe-Chiba band radius for the DTW in [Self::distance], in
    /// perceptual time steps (roughly [Self::perceptual_sample_rate] steps per
    /// second, ~85 by default).
    ///
    /// This is an extension over the C++ original. `None` (the default)
    /// computes the exact, exhaustive DTW. `Some(radius)` only explores warp
    /// paths within `radius` steps of the straight-line time mapping between
    /// the two inputs, reducing processing time considerably. The result is
    /// identical to the exact DTW as long as the true time misalignment between
    /// the inputs stays within the band.
    pub dtw_band_radius: Option<usize>,
}

impl Default for Zimtohrli {
    fn default() -> Self {
        let high_gamma_band = 85.0;
        let samples_per_perceptual_block = (SAMPLE_RATE / high_gamma_band) as usize;
        let perceptual_sample_rate = SAMPLE_RATE / samples_per_perceptual_block as f32;
        Self {
            level: Level::new(),
            nsim_step_window: 8,
            nsim_channel_window: 5,
            perceptual_sample_rate,
            full_scale_sine_db: 78.3,
            dtw_band_radius: None,
        }
    }
}

impl Zimtohrli {
    /// Analyzes an audio signal and fills the provided spectrogram.
    /// signal: input audio samples at 48kHz, range [-1, 1]
    /// spectrogram: pre-allocated output spectrogram to fill
    pub fn analyze_into(&self, signal: &[f32], spectrogram: &mut Spectrogram) {
        assert_eq!(spectrogram.num_dims, NUM_ROTATORS);
        Rotators::filter_and_downsample(
            self.level,
            signal,
            &mut spectrogram.values,
            spectrogram.num_steps,
            spectrogram.num_dims,
            signal.len() / spectrogram.num_steps,
        );
    }

    /// Analyzes an audio signal and returns a new spectrogram.
    /// signal: input audio samples at 48kHz, range [-1, 1]
    /// Returns: perceptual spectrogram representation
    pub fn analyze(&self, signal: &[f32]) -> Spectrogram {
        let mut spec = Spectrogram::new(self.spectrogram_steps(signal.len()), NUM_ROTATORS);
        self.analyze_into(signal, &mut spec);
        spec
    }

    /// Calculates the number of time steps in the output spectrogram
    /// based on the input signal length and perceptual sample rate.
    pub fn spectrogram_steps(&self, num_samples: usize) -> usize {
        (num_samples as f32 * self.perceptual_sample_rate / SAMPLE_RATE).ceil() as usize
    }

    fn rescale_to_match_energy(spec_a: &mut Spectrogram, spec_b: &mut Spectrogram) {
        assert_eq!(spec_a.num_dims, spec_b.num_dims);

        let max_a = spec_a.max() as f64;
        let max_b = spec_b.max() as f64;
        if max_a != max_b && max_a > 0.0 && max_b > 0.0 {
            // For full correction cora + corb would be 1.0. It is very much
            // unclear why optimization prefers to have overcorrection for
            // distance. Perhaps it softens the error vallay and in combination
            // with the preference of going straight in the path-finding good
            // things happens. (This is pure speculation without trying to
            // obtain evidence about this).
            let mut cora = 0.5828284197882053;
            let mut corb = 0.6310239126768997;

            if max_a > max_b {
                std::mem::swap(&mut cora, &mut corb);
            }
            spec_b.rescale(pow(max_a / max_b, cora) as f32);
            spec_a.rescale(pow(max_b / max_a, corb) as f32);
        }
    }

    /// Computes perceptual distance between two spectrograms.
    /// Uses DTW for time alignment and NSIM for similarity measurement.
    /// Returns: distance in range [0, 1], where 0 = identical, 1 = maximally
    /// different.
    /// Note: both spectrograms may be rescaled to match energy levels.
    pub fn distance(&self, spec_a: &mut Spectrogram, spec_b: &mut Spectrogram) -> f32 {
        assert_eq!(spec_a.num_dims, spec_b.num_dims);

        if spec_a.num_steps == 0 || spec_b.num_steps == 0 {
            return 1.0;
        }

        Self::rescale_to_match_energy(spec_a, spec_b);
        let time_pairs = dtw(self.level, spec_a, spec_b, self.dtw_band_radius);
        1.0 - nsim(
            spec_a,
            spec_b,
            time_pairs,
            self.nsim_step_window,
            self.nsim_channel_window,
        )
    }

    /// Computes perceptual distance between two spectrograms assuming they are
    /// already aligned. Note that this requires `spec_a` and `spec_b` to have
    /// the same length.
    ///
    /// `spec_a` and `spec_a` are the perceptual spectrograms to compare.
    /// `step_window` optionally overrides the default NSIM step window size.
    /// Returns: distance in range [0, 1], where 0 = identical, 1 = maximally
    /// different. Note: both spectrograms may be rescaled to match energy
    /// levels.
    pub fn distance_without_dtw(&self, spec_a: &mut Spectrogram, spec_b: &mut Spectrogram) -> f32 {
        assert_eq!(spec_a.num_dims, spec_b.num_dims);
        assert_eq!(spec_a.num_steps, spec_b.num_steps);

        Self::rescale_to_match_energy(spec_a, spec_b);

        1.0 - nsim(
            spec_a,
            spec_b,
            PreAligned,
            self.nsim_step_window,
            self.nsim_channel_window,
        )
    }

    /// Returns a _very approximate_ mean opinion score based on the
    /// provided Zimtohrli distance.
    pub fn mos_from_distance(distance: f32) -> f32 {
        static MOS_PARAMS: [f32; 3] = [1.000e+00, -6.799e-09, 6.487e+01];

        fn sigmoid(x: f32) -> f32 {
            MOS_PARAMS[0] / (MOS_PARAMS[1] + expf(MOS_PARAMS[2] * x))
        }

        let zero_crossing_reciprocal = 1.0 / sigmoid(0.0);

        1.0 + 4.0 * sigmoid(distance) * zero_crossing_reciprocal
    }
}

#[cfg(test)]
mod rotator_tests {
    use super::*;

    fn scalar_increment_all(
        rotators: &mut Rotators,
        signal: f32,
        out: &mut [f32],
        weight: f32,
        has_next_output: bool,
    ) {
        for i in 0..NUM_ROTATORS {
            let w = rotators.window[i];
            for k in 0..6 {
                rotators.accu[k][i] *= w;
            }
            rotators.accu[2][i] += rotators.accu[0][i];
            rotators.accu[3][i] += rotators.accu[1][i];
            rotators.accu[4][i] += rotators.accu[2][i];
            rotators.accu[5][i] += rotators.accu[3][i];
            let a = rotators.rot[2][i];
            let b = rotators.rot[3][i];
            rotators.accu[0][i] += a * signal;
            rotators.accu[1][i] += b * signal;
            rotators.rot[2][i] = (rotators.rot[0][i] * a) - (rotators.rot[1][i] * b);
            rotators.rot[3][i] = (rotators.rot[0][i] * b) + (rotators.rot[1][i] * a);

            let energy = (rotators.accu[4][i] * rotators.accu[4][i])
                + (rotators.accu[5][i] * rotators.accu[5][i]);
            if has_next_output {
                out[NUM_ROTATORS + i] += (1.0 - weight) * energy;
                out[i] += weight * energy;
            } else {
                out[i] += energy;
            }
        }
    }

    fn initial_rotators() -> Rotators {
        Rotators {
            rot: std::array::from_fn(|row| {
                std::array::from_fn(|i| (1 + row * NUM_ROTATORS + i) as f32 * 0.000_031)
            }),
            accu: std::array::from_fn(|row| {
                std::array::from_fn(|i| (1 + row * NUM_ROTATORS + i) as f32 * -0.000_017)
            }),
            window: std::array::from_fn(|i| 0.97 + i as f32 * 0.000_1),
            gain: std::array::from_fn(|i| 1.0 + i as f32 * 0.01),
        }
    }

    fn assert_increment_matches_scalar(level: Level) {
        let signals = [0.25, -0.75, 0.001, 1.0, -0.125];
        let mut expected = initial_rotators();
        let mut expected_out = [0.0; 2 * NUM_ROTATORS];
        for (i, signal) in signals.into_iter().enumerate() {
            scalar_increment_all(&mut expected, signal, &mut expected_out, 0.25, i < 4);
        }

        let mut actual = initial_rotators();
        let mut actual_out = [0.0; 2 * NUM_ROTATORS];
        dispatch!(level, simd => {
            for (i, signal) in signals.into_iter().enumerate() {
                actual.increment_all(
                    simd,
                    signal,
                    EnergyOutput {
                        values: &mut actual_out,
                        step: 0,
                        stride: NUM_ROTATORS,
                        weight: 0.25,
                        has_next: i < 4,
                    },
                );
            }
        });

        assert_eq!(
            actual.rot.map(|row| row.map(f32::to_bits)),
            expected.rot.map(|row| row.map(f32::to_bits))
        );
        assert_eq!(
            actual.accu.map(|row| row.map(f32::to_bits)),
            expected.accu.map(|row| row.map(f32::to_bits))
        );
        assert_eq!(actual_out.map(f32::to_bits), expected_out.map(f32::to_bits));
    }

    #[test]
    fn simd_increment_is_bit_exact_with_scalar_evaluation_order() {
        assert_increment_matches_scalar(Level::Fallback(fearless_simd::Fallback::new()));
        let detected = Level::new();
        assert_increment_matches_scalar(detected);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some(sse) = detected.as_sse4_2() {
                assert_increment_matches_scalar(Level::Sse4_2(sse));
            }
            if let Some(avx2) = detected.as_avx2() {
                assert_increment_matches_scalar(Level::Avx2(avx2));
            }
        }
    }

    fn filter_at_level(level: Level, input: &[f32]) -> Vec<u32> {
        const STEPS: usize = 8;
        let mut out = vec![0.0; STEPS * NUM_ROTATORS];
        dispatch!(level, simd => {
            Rotators::filter_and_downsample_inner(
                simd,
                input,
                &mut out,
                STEPS,
                NUM_ROTATORS,
                input.len() / STEPS,
            );
        });
        out.into_iter().map(f32::to_bits).collect()
    }

    #[test]
    fn complete_filter_is_bit_exact_across_available_simd_levels() {
        let input = (0..4096)
            .map(|i| ((i * 1049 + 17) % 2001) as f32 / 1000.0 - 1.0)
            .collect::<Vec<_>>();
        let fallback = Level::Fallback(fearless_simd::Fallback::new());
        let expected = filter_at_level(fallback, &input);
        let detected = Level::new();
        assert_eq!(filter_at_level(detected, &input), expected);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some(sse) = detected.as_sse4_2() {
                assert_eq!(filter_at_level(Level::Sse4_2(sse), &input), expected);
            }
            if let Some(avx2) = detected.as_avx2() {
                assert_eq!(filter_at_level(Level::Avx2(avx2), &input), expected);
            }
        }
    }
}
