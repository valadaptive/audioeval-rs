//! A pure-Rust port of [ViSQOL](https://github.com/google/visqol) v3
//! (conformance version 333), an objective full-reference metric for
//! perceived audio quality.
//!
//! The pipeline: globally align the degraded signal to the reference by
//! cross-correlating signal envelopes, match their loudness, build gammatone
//! filterbank spectrograms, cut the reference spectrogram into patches, find
//! the best-matching degraded patch for each via a DTW-style search with
//! optional time-domain fine realignment, score each pair with NSIM (an SSIM
//! variant), and map the per-frequency-band similarities to a MOS-LQO score
//! with a mode-dependent model.
//!
//! Differences from the C++ implementation:
//! - FFTs (used for alignment) run in f64 rather than pffft's f32.
//! - The TensorFlow-Lite "lattice" speech-mode mapper is evaluated natively
//!   from its extracted parameters (see [`lattice`](crate::LatticeModel))
//!   rather than through a TFLite runtime.
//!
//! ```no_run
//! use visqol::{AudioSignal, Visqol};
//!
//! let reference = AudioSignal::new(vec![0.0; 48000 * 5], 48000);
//! let degraded = AudioSignal::new(vec![0.0; 48000 * 5], 48000);
//! let result = Visqol::audio().run(&reference, &degraded).unwrap();
//! println!("MOS-LQO: {}", result.moslqo);
//! ```

mod alignment;
mod analysis_window;
mod audio_signal;
mod erb;
mod fft;
mod gammatone;
mod lattice;
mod matrix;
mod nsim;
mod patches;
mod selector;
mod spectrogram;
mod svr;

pub use audio_signal::AudioSignal;
pub use lattice::LatticeModel;
pub use nsim::PatchSimilarityResult;
pub use svr::{SimilarityToQualityMapper, SvrModel};

use analysis_window::AnalysisWindow;
use gammatone::GammatoneSpectrogramBuilder;
use patches::{create_image_patch_indices, create_vad_patch_indices};

#[derive(Debug)]
pub enum Error {
    SampleRateMismatch {
        reference: u32,
        degraded: u32,
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
    InvalidModel(String),
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
            Error::InvalidModel(msg) => write!(f, "invalid SVR model: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// The complete output of a comparison, mirroring the C++
/// `SimilarityResultMsg`.
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

pub struct Visqol {
    mapper: SimilarityToQualityMapper,
    speech_mode: bool,
    /// How many patch-lengths on either side of a reference patch's position
    /// the matching search may look.
    pub search_window_radius: usize,
    pub disable_global_alignment: bool,
    pub disable_realignment: bool,
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
            search_window_radius: DEFAULT_SEARCH_WINDOW_RADIUS,
            disable_global_alignment: false,
            disable_realignment: false,
        }
    }

    /// Speech mode (16 kHz input expected) with the deep lattice network
    /// NSIM-to-MOS mapping, the C++ default (`--use_lattice_model`).
    pub fn speech_lattice() -> Self {
        Visqol {
            mapper: SimilarityToQualityMapper::Lattice(LatticeModel::default_speech_model()),
            speech_mode: true,
            search_window_radius: DEFAULT_SEARCH_WINDOW_RADIUS,
            disable_global_alignment: false,
            disable_realignment: false,
        }
    }

    /// Speech mode (16 kHz input expected): voice-activity-gated patches and
    /// the exponential NSIM-to-MOS mapping (the C++ behavior with
    /// `--use_lattice_model=false`). `scale_to_max_mos` rescales the
    /// fit so that a perfect NSIM maps to a MOS of 5.0 rather than ~4.x
    /// (enabled in the C++ unless `--use_unscaled_speech_mos_mapping`).
    pub fn speech(scale_to_max_mos: bool) -> Self {
        Visqol {
            mapper: SimilarityToQualityMapper::SpeechExponential { scale_to_max_mos },
            speech_mode: true,
            search_window_radius: DEFAULT_SEARCH_WINDOW_RADIUS,
            disable_global_alignment: false,
            disable_realignment: false,
        }
    }

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

        let mut deg_signal = deg_signal.clone();
        let alignment_lag_s = if self.disable_global_alignment {
            0.0
        } else {
            alignment::globally_align(ref_signal, &mut deg_signal)
        };

        let window = AnalysisWindow::new(ref_signal.sample_rate, OVERLAP);

        let (num_bands, patch_size) = if self.speech_mode {
            (NUM_BANDS_SPEECH, PATCH_SIZE_SPEECH)
        } else {
            (NUM_BANDS_AUDIO, PATCH_SIZE_AUDIO)
        };

        // Stage 1: preprocessing.
        let deg_signal = audio_signal::scale_to_match_sound_pressure_level(ref_signal, &deg_signal);
        let spect_builder =
            GammatoneSpectrogramBuilder::new(num_bands, MINIMUM_FREQ, self.speech_mode);
        let mut ref_spectrogram = spect_builder.build(ref_signal, &window)?;
        let mut deg_spectrogram = spect_builder.build(&deg_signal, &window)?;
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
