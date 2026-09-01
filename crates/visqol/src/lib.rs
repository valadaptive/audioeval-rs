//! A fast, pure-Rust port of [ViSQOL](https://github.com/google/visqol) v3.3, an objective, full-reference metric for perceived audio quality.
//!
//! ```ignore
//! use audioeval_visqol::{AudioSignal, Visqol};
//!
//! let reference = AudioSignal::new(ref_samples, 48000);
//! let degraded = AudioSignal::new(deg_samples, 48000);
//! let result = Visqol::audio().run(&reference, &degraded)?;
//! println!("MOS-LQO: {}", result.moslqo);
//! ```
//!
//! The entry points you're most likely to want are:
//! - [`Visqol::audio()`]: for general audio (ViSQOL's default behavior). This expects 48kHz input.
//! - [`Visqol::speech_lattice()`]: for speech (like ViSQOL with `use_speech_scoring` enabled). This expects 16kHz input.
//! - [`Visqol::speech_legacy()`]: for speech, using the older non-lattice model (like ViSQOL with `use_speech_scoring` enabled, but `use_lattice_model` explicitly disabled). This also expects 16kHz input.
//!
//! The inputs are expected to be mono signals. You may use `AudioSignal::from_channels` to downmix by averaging, as with the C++ version.
//!
//! This crate will *not* resample your audio for you. You must use something like [rubato](https://crates.io/crates/rubato) to resample the input before passing it in.
//!
//! ## Mapping from the C++ configuration
//!
//! Here's where each `VisqolConfig.options` field lands:
//!
//! - `use_speech_scoring` + `use_lattice_model`: Choose a constructor. [`Visqol::audio()`] (audio, SVR), [`Visqol::speech_lattice()`] (speech, lattice), or [`Visqol::speech_legacy()`] (speech, exponential fit).
//! - `use_unscaled_speech_mos_mapping`: Passed as a `bool` argument to [`Visqol::speech_legacy()`].
//! - `svr_model_path`: [`Visqol::audio_with_model()`], which is passed an [`SvrModel`]. You may load one via [`SvrModel::from_text()`].
//! - `search_window_radius`: [`Visqol::search_window_radius`].
//! - `allow_unsupported_sample_rates`: [`Visqol::allow_unsupported_sample_rates`].
//! - `output_mos_score` + `detect_voice_activity`: Not supported. Even in the original C++ version, these are unimplemented no-ops.
//!
//! ## Conformance
//!
//! This crate's results are compared against the upstream conformance test suite. We test against [conformance version 333](https://github.com/google/visqol/blob/38d0b01/src/include/conformance.h#L30).
//!
//! The audio and non-lattice speech models match the original C++ implementation to ~13 significant digits. The lattice speech model's scores match to ~6 significant digits. This is well within the [upstream conformance tolerance of 1e-4](https://github.com/google/visqol/blob/38d0b01/python/visqol_lib_py_test.py#L20).
//!
//! Results will likely not be *bit-identical* across machines due to potential rounding differences in libm functions, runtime FFT kernel decisions, and other factors outside this library's control. They are, however, expected to land well within the conformance tests' tolerance.
//!
//! ## Performance
//!
//! This crate is significantly faster than the original C++ implementation of ViSQOL: around **40x faster** in audio mode, and **30-35x faster** in speech mode.
//!
//! I am aware that the performance improvement seems unusually large, and if anybody wants to run their own benchmarks to double-check this figure, feel free to do so. I'm pretty sure that the figures check out, though--the original codebase doesn't seem to have undergone any optimization effort, whereas this codebase has.

mod alignment;
mod analysis_window;
mod audio_signal;
mod erb;
mod gammatone;
mod lattice;
mod matrix;
mod nsim;
mod patches;
mod selector;
mod spectrogram;
mod svr;

use std::borrow::Cow;

pub use audio_signal::AudioSignal;
pub use fearless_simd::Level;
pub use lattice::LatticeModel;
pub use nsim::PatchSimilarityResult;
pub use svr::SvrModel;

use analysis_window::AnalysisWindow;
use gammatone::GammatoneSpectrogramBuilder;
use patches::{create_image_patch_indices, create_vad_patch_indices};

use crate::alignment::FftManager;

#[derive(Debug)]
pub enum Error {
    SampleRateMismatch {
        reference: u32,
        degraded: u32,
    },
    /// Audio mode was given input at a sample rate other than 48 kHz without
    /// [`allow_unsupported_sample_rates`](Visqol::allow_unsupported_sample_rates)
    /// being set.
    UnsupportedSampleRate {
        sample_rate: u32,
    },
    /// A signal is shorter than a single analysis window.
    TooFewSamples {
        samples: usize,
        required: usize,
    },
    /// The reference spectrogram cannot fit a single patch.
    ReferenceSpectrogramTooSmall {
        frames: usize,
        required: usize,
    },
    /// The degraded file was too short, too different, or too misaligned to
    /// score any reference patch.
    DegradedFileTooShort,
    InvalidSVRModel(Cow<'static, str>),
    InvalidLatticeModel(Cow<'static, str>),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::SampleRateMismatch {
                reference,
                degraded,
            } => write!(
                f,
                "input audio signals have different sample rates \
                 (reference: {reference} Hz, degraded: {degraded} Hz)"
            ),
            Error::UnsupportedSampleRate { sample_rate } => write!(
                f,
                "audio mode only supports 48 kHz input, but got {sample_rate} Hz \
                 (set allow_unsupported_sample_rates to override)"
            ),
            Error::TooFewSamples { samples, required } => write!(
                f,
                "too few samples ({samples}) in signal to build spectrogram \
                 ({required} required minimum)"
            ),
            Error::ReferenceSpectrogramTooSmall { frames, required } => write!(
                f,
                "reference spectrum size ({frames} frames) smaller than \
                 minimum patch size ({required} frames)"
            ),
            Error::DegradedFileTooShort => write!(
                f,
                "degraded file was too short, different, or misaligned to \
                 score any of the reference patches"
            ),
            Error::InvalidSVRModel(msg) => write!(f, "invalid SVR model: {msg}"),
            Error::InvalidLatticeModel(msg) => write!(f, "invalid lattice model: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// The complete output of a comparison.
#[derive(Debug)]
pub struct SimilarityResult {
    /// Mean opinion score - listening quality objective, in [1, 5].
    pub moslqo: f64,
    /// Mean NSIM over all patches and frequency bands, in ~[0, 1].
    pub vnsim: f64,
    /// Mean NSIM per frequency band.
    pub fvnsim: Vec<f64>,
    /// Mean of the worst 10% NSIM values per frequency band.
    pub fvnsim10: Vec<f64>,
    /// Pooled NSIM standard deviation per frequency band.
    pub fstdnsim: Vec<f64>,
    /// Mean degraded-spectrogram energy per frequency band.
    pub fvdegenergy: Vec<f64>,
    /// Center frequency of each band, lowest first.
    pub center_freq_bands: Vec<f64>,
    /// Per-patch similarity details.
    pub patch_sims: Vec<PatchSimilarityResult>,
    /// Lag corrected by global alignment, in seconds.
    pub alignment_lag_s: f64,
}

/// Main ViSQOL analysis struct. Creating one of these will likely involve some
/// expensive operations (loading models, creating FFT plans, etc), so you
/// should create one and reuse it across multiple clips.
pub struct Visqol {
    mapper: SimilarityToQualityMapper,
    fft_manager: FftManager,
    speech_mode: bool,
    pub simd_level: fearless_simd::Level,
    /// How many patch-lengths on either side of a reference patch's position
    /// the matching search may look.
    pub search_window_radius: usize,
    pub disable_global_alignment: bool,
    pub disable_realignment: bool,
    /// By default, audio mode rejects any input that isn't 48 kHz (the only
    /// sample rate its model was trained for). Set this to run it anyway, as
    /// with the C++ `allow_unsupported_sample_rates` flag. Speech mode is
    /// unaffected either way.
    pub allow_unsupported_sample_rates: bool,
}

const PATCH_SIZE_AUDIO: usize = 30;
const PATCH_SIZE_SPEECH: usize = 20;
const NUM_BANDS_AUDIO: usize = 32;
const NUM_BANDS_SPEECH: usize = 21;
const MINIMUM_FREQ: f64 = 50.0; // wideband
const OVERLAP: f64 = 0.25;
const DEFAULT_SEARCH_WINDOW_RADIUS: usize = 60;

impl Visqol {
    /// Audio mode (48 kHz input expected) with the default SVR model.
    pub fn audio() -> Self {
        Self::audio_with_model(SvrModel::default_audio_model())
    }

    pub fn audio_with_model(model: SvrModel) -> Self {
        Visqol {
            mapper: SimilarityToQualityMapper::Svr(model),
            speech_mode: false,
            simd_level: fearless_simd::Level::new(),
            search_window_radius: DEFAULT_SEARCH_WINDOW_RADIUS,
            disable_global_alignment: false,
            disable_realignment: false,
            allow_unsupported_sample_rates: false,
            fft_manager: FftManager::default(),
        }
    }

    /// Speech mode (16 kHz input expected) with the deep lattice network
    /// NSIM-to-MOS mapping, the default.
    pub fn speech_lattice() -> Self {
        Visqol {
            mapper: SimilarityToQualityMapper::Lattice(LatticeModel::default_speech_model()),
            speech_mode: true,
            simd_level: fearless_simd::Level::new(),
            search_window_radius: DEFAULT_SEARCH_WINDOW_RADIUS,
            disable_global_alignment: false,
            disable_realignment: false,
            allow_unsupported_sample_rates: false,
            fft_manager: FftManager::default(),
        }
    }

    /// Speech mode (16 kHz input expected): voice-activity-gated patches and
    /// the exponential NSIM-to-MOS mapping (`--use_lattice_model=false`).
    ///
    /// When `use_unscaled_speech_mos_mapping` is `false` (the default), the fit
    /// is rescaled so that a perfect NSIM maps to a MOS of 5.0; when `true`, a
    /// perfect NSIM instead maps to the unscaled ~4.x.
    pub fn speech_legacy(use_unscaled_speech_mos_mapping: bool) -> Self {
        Visqol {
            mapper: SimilarityToQualityMapper::SpeechExponential {
                use_unscaled_speech_mos_mapping,
            },
            speech_mode: true,
            simd_level: fearless_simd::Level::new(),
            search_window_radius: DEFAULT_SEARCH_WINDOW_RADIUS,
            disable_global_alignment: false,
            disable_realignment: false,
            allow_unsupported_sample_rates: false,
            fft_manager: FftManager::default(),
        }
    }

    /// Whether this analysis is in speech mode and hence expects 16kHz input.
    pub fn is_speech_mode(&self) -> bool {
        self.speech_mode
    }

    /// Run an analysis over a reference and degraded signal.
    pub fn run(
        &self,
        ref_signal: &AudioSignal,
        deg_signal: &AudioSignal,
    ) -> Result<SimilarityResult> {
        if ref_signal.sample_rate != deg_signal.sample_rate {
            return Err(Error::SampleRateMismatch {
                reference: ref_signal.sample_rate,
                degraded: deg_signal.sample_rate,
            });
        }

        // Audio mode's model was trained on 48 kHz only. Speech mode is
        // purportedly sample-rate-independent (its bands are pinned to fixed
        // frequencies).
        if !self.speech_mode
            && ref_signal.sample_rate != 48000
            && !self.allow_unsupported_sample_rates
        {
            return Err(Error::UnsupportedSampleRate {
                sample_rate: ref_signal.sample_rate,
            });
        }

        let (mut deg_signal, alignment_lag_s) = if self.disable_global_alignment {
            (deg_signal.as_borrowed(), 0.0)
        } else {
            alignment::globally_align(&self.fft_manager, ref_signal, deg_signal)
        };

        let window = AnalysisWindow::new(ref_signal.sample_rate, OVERLAP);

        let (num_bands, patch_size) = if self.speech_mode {
            (NUM_BANDS_SPEECH, PATCH_SIZE_SPEECH)
        } else {
            (NUM_BANDS_AUDIO, PATCH_SIZE_AUDIO)
        };

        // Stage 1: preprocessing.
        audio_signal::scale_to_match_sound_pressure_level(ref_signal, &mut deg_signal);
        let spect_builder = GammatoneSpectrogramBuilder::new(
            num_bands,
            ref_signal.sample_rate,
            MINIMUM_FREQ,
            self.speech_mode,
        );
        let mut ref_spectrogram = spect_builder.build(self.simd_level, ref_signal, &window)?;
        let mut deg_spectrogram = spect_builder.build(self.simd_level, &deg_signal, &window)?;
        spectrogram::prepare_spectrograms_for_comparison(
            &mut ref_spectrogram,
            &mut deg_spectrogram,
        );

        // Stage 2: feature selection and similarity measure.
        let ref_patch_indices = if self.speech_mode {
            create_vad_patch_indices(&ref_spectrogram.data, ref_signal, &window, patch_size)
        } else {
            create_image_patch_indices(&ref_spectrogram.data, patch_size)?
        };
        let frame_duration =
            (window.size as f64 * window.overlap) as usize as f64 / ref_signal.sample_rate as f64;

        let ref_patches = patches::create_patches_from_indices(
            &ref_spectrogram.data,
            &ref_patch_indices,
            patch_size,
        );
        if ref_patches.is_empty() {
            return Err(Error::DegradedFileTooShort);
        }
        let mut sim_match_info = selector::find_most_optimal_deg_patches(
            self.simd_level,
            &ref_patches,
            &ref_patch_indices,
            &deg_spectrogram.data,
            frame_duration,
            self.search_window_radius,
        )?;

        // Realign the patches as time-domain subsignals starting at the
        // coarse patch times.
        if !self.disable_realignment {
            selector::finely_align_and_recreate_patches(
                self.simd_level,
                &self.fft_manager,
                &mut sim_match_info,
                ref_signal,
                &deg_signal,
                &spect_builder,
                &window,
            )?;
        }

        let num_bands = sim_match_info[0].freq_band_means.len();
        let num_patches = sim_match_info.len();

        // fvnsim: mean similarity per band over all patches.
        let mut fvnsim = vec![0.0; num_bands];
        for patch in &sim_match_info {
            for (acc, &m) in fvnsim.iter_mut().zip(&patch.freq_band_means) {
                *acc += m;
            }
        }
        for v in &mut fvnsim {
            *v /= num_patches as f64;
        }

        // fvnsim10: mean of the worst 10% per band.
        let mut fvnsim10 = vec![0.0; num_bands];
        for (band, out) in fvnsim10.iter_mut().enumerate() {
            let mut band_nsims: Vec<f64> = sim_match_info
                .iter()
                .map(|p| p.freq_band_means[band])
                .collect();
            band_nsims.sort_by(|a, b| a.total_cmp(b));
            let num_in_quantile = ((band_nsims.len() as f64 * 0.10) as usize).max(1);
            *out = band_nsims[..num_in_quantile].iter().sum::<f64>() / num_in_quantile as f64;
        }

        // fvdegenergy: mean degraded energy per band.
        let mut fvdegenergy = vec![0.0; num_bands];
        for patch in &sim_match_info {
            for (acc, &e) in fvdegenergy.iter_mut().zip(&patch.freq_band_deg_energy) {
                *acc += e;
            }
        }
        for v in &mut fvdegenergy {
            *v /= num_patches as f64;
        }

        let fstdnsim = calc_pooled_freq_band_std_devs(&sim_match_info, &fvnsim, frame_duration);

        let mut moslqo = self
            .mapper
            .predict_quality(&fvnsim, &fvnsim10, &fstdnsim, &fvdegenergy);
        let vnsim = fvnsim.iter().sum::<f64>() / fvnsim.len() as f64;

        // Stop totally dissimilar signals from getting a good score: the
        // mapping models were trained on the same content at different
        // qualities and return fairly arbitrary values for unrelated inputs.
        if vnsim < 0.15 {
            moslqo = 1.0;
        }

        Ok(SimilarityResult {
            moslqo,
            vnsim,
            fvnsim,
            fvnsim10,
            fstdnsim,
            fvdegenergy,
            center_freq_bands: ref_spectrogram.center_freq_bands,
            patch_sims: sim_match_info,
            alignment_lag_s,
        })
    }
}

/// Combines the per-patch NSIM means and standard deviations into a pooled
/// per-band standard deviation (see
/// <https://en.wikipedia.org/wiki/Pooled_variance>).
fn calc_pooled_freq_band_std_devs(
    sim_match_info: &[PatchSimilarityResult],
    fvnsim: &[f64],
    frame_duration: f64,
) -> Vec<f64> {
    let num_bands = fvnsim.len();
    let mut contribution = vec![0.0; num_bands];
    let mut total_frame_count = 0i64;
    for patch in sim_match_info {
        let secs_in_patch = patch.ref_patch_end_time - patch.ref_patch_start_time;
        let frame_count = (secs_in_patch / frame_duration).ceil() as i64;
        total_frame_count += frame_count;
        for ((contrib, &stddev), &mean) in contribution
            .iter_mut()
            .zip(&patch.freq_band_stddevs)
            .zip(&patch.freq_band_means)
        {
            // Two separate additions, preserving the C++'s rounding.
            *contrib += (frame_count - 1) as f64 * stddev * stddev;
            *contrib += frame_count as f64 * mean * mean;
        }
    }

    contribution
        .iter()
        .zip(fvnsim)
        .map(|(&c, &mean)| {
            let variance =
                (c - mean * mean * total_frame_count as f64) / (total_frame_count - 1) as f64;
            // Precision issues can push the variance slightly negative.
            variance.max(0.0).sqrt()
        })
        .collect()
}

/// The similarity-to-quality mapping stage.
pub enum SimilarityToQualityMapper {
    /// Audio mode: nu-SVR over the per-band mean similarities.
    Svr(SvrModel),
    /// Speech mode default: deep lattice network over all per-band features.
    Lattice(crate::lattice::LatticeModel),
    /// Speech mode with `--use_lattice_model=false`: exponential fit of mean
    /// NSIM over the TCD-VOIP dataset.
    SpeechExponential {
        use_unscaled_speech_mos_mapping: bool,
    },
}

impl SimilarityToQualityMapper {
    pub fn predict_quality(
        &self,
        fvnsim: &[f64],
        fvnsim10: &[f64],
        fstdnsim: &[f64],
        fvdegenergy: &[f64],
    ) -> f64 {
        match self {
            SimilarityToQualityMapper::Svr(model) => model.predict(fvnsim).clamp(1.0, 5.0),
            SimilarityToQualityMapper::Lattice(model) => {
                model.predict(fvnsim, fvnsim10, fstdnsim, fvdegenergy)
            }
            SimilarityToQualityMapper::SpeechExponential {
                use_unscaled_speech_mos_mapping,
            } => {
                const FIT_PARAMETER_A: f64 = -262.847869;
                const FIT_PARAMETER_B: f64 = 0.0154302525;
                const FIT_PARAMETER_X0: f64 = -361.063949;

                // Oddly, the C++ narrows the scale factor to float.
                const FIT_SCALE: f64 = 1.245063_f32 as f64;

                let nsim_mean = fvnsim.iter().sum::<f64>() / fvnsim.len() as f64;
                let mos =
                    FIT_PARAMETER_A + (FIT_PARAMETER_B * (nsim_mean - FIT_PARAMETER_X0)).exp();
                let scale = if *use_unscaled_speech_mos_mapping {
                    1.0
                } else {
                    FIT_SCALE
                };
                (mos * scale).clamp(1.0, 5.0)
            }
        }
    }
}
