//! Matching of reference patches to degraded-spectrogram offsets via a
//! DTW-style dynamic program, plus the per-patch time-domain fine
//! realignment. A port of `comparison_patches_selector.cc`.

use std::ops::Range;

use fearless_simd::Level;

use crate::alignment;
use crate::analysis_window::AnalysisWindow;
use crate::audio_signal::AudioSignal;
use crate::gammatone::GammatoneSpectrogramBuilder;
use crate::matrix::Matrix;
use crate::nsim::{self, PatchSimilarityResult};
use crate::spectrogram;
use crate::{Error, Result};

pub fn find_most_optimal_deg_patches(
    level: Level,
    ref_patches: &[Matrix],
    ref_patch_indices: &[usize],
    spectrogram_data: &Matrix,
    frame_duration: f64,
    search_window_radius: usize,
) -> Result<Vec<PatchSimilarityResult>> {
    let num_frames_per_patch = ref_patches[0].cols();
    let num_frames_in_deg_spectro = spectrogram_data.cols();
    let patch_duration = frame_duration * num_frames_per_patch as f64;
    let search_window = (search_window_radius * num_frames_per_patch) as i64;
    let num_patches = calc_max_num_patches(
        ref_patch_indices,
        num_frames_in_deg_spectro,
        num_frames_per_patch,
    );

    if num_patches == 0 {
        return Err(Error::DegradedFileTooShort);
    }

    // Cumulative similarity and backtrace tables. Unvisited cells default to
    // 0.0 / 0, matching the C++'s default-initialized vectors.
    let mut dp = vec![vec![0.0f64; num_frames_in_deg_spectro]; ref_patch_indices.len()];
    let mut backtrace = vec![vec![0i64; num_frames_in_deg_spectro]; ref_patch_indices.len()];

    // Precomputed convolution maps shared by every NSIM cell of the search
    // (see `nsim::similarity_at_offset`).
    let deg_conv = nsim::DegSpectrogramConv::new(spectrogram_data, num_frames_per_patch);
    let mut scratch = nsim::NsimScratch::new(ref_patches[0].rows(), num_frames_per_patch);

    for patch_index in 0..num_patches {
        let ref_patch = &ref_patches[patch_index];
        let ref_conv = nsim::RefPatchConv::new(ref_patch);
        let ref_frame_index = ref_patch_indices[patch_index] as i64;
        let first_offset = (ref_frame_index - search_window).max(0) as usize;
        for slide_offset in first_offset..num_frames_in_deg_spectro {
            if slide_offset as i64 > ref_frame_index + search_window {
                break;
            }
            let mut similarity = if slide_offset + num_frames_per_patch <= num_frames_in_deg_spectro
            {
                nsim::similarity_at_offset(
                    level,
                    ref_patch,
                    &ref_conv,
                    spectrogram_data,
                    &deg_conv,
                    slide_offset,
                    &mut scratch,
                )
            } else {
                // The patch overhangs the end of the spectrogram and gets
                // zero-padded; take the plain path.
                let deg_patch = build_degraded_patch(
                    spectrogram_data,
                    slide_offset..slide_offset + num_frames_per_patch,
                    ref_patch.rows(),
                    num_frames_per_patch,
                );
                nsim::measure_patch_similarity(ref_patch, &deg_patch).similarity
            };

            let mut past_slide_offset: i64 = -1;
            if patch_index > 0 {
                // Highest cumulative similarity achievable up to the previous
                // patch, over that patch's own search window, ending strictly
                // before this offset (two reference patches must not map to
                // the same degraded patch).
                let lower_limit =
                    (ref_patch_indices[patch_index - 1] as i64 - search_window).max(0);
                let mut highest_sim = f64::MIN;
                let mut back_offset = slide_offset as i64 - 1;
                while back_offset >= lower_limit {
                    if dp[patch_index - 1][back_offset as usize] > highest_sim {
                        highest_sim = dp[patch_index - 1][back_offset as usize];
                        past_slide_offset = back_offset;
                    }
                    back_offset -= 1;
                }
                similarity += highest_sim;
                // If this reference patch experienced packet loss, skipping
                // it entirely may score higher than any match.
                if dp[patch_index - 1][slide_offset] > similarity {
                    similarity = dp[patch_index - 1][slide_offset];
                    past_slide_offset = slide_offset as i64;
                }
            }
            dp[patch_index][slide_offset] = similarity;
            backtrace[patch_index][slide_offset] = past_slide_offset;
        }
    }

    // Find the offset of the last patch that maximizes the cumulative score.
    let last_index = num_patches - 1;
    let lower_limit = (ref_patch_indices[last_index] as i64 - search_window).max(0) as usize;
    let mut max_similarity_score = f64::MIN;
    let mut last_offset = 0usize;
    for (slide_offset, &score) in dp[last_index].iter().enumerate().skip(lower_limit) {
        if slide_offset as i64 > ref_patch_indices[last_index] as i64 + search_window {
            break;
        }
        if score > max_similarity_score {
            max_similarity_score = score;
            last_offset = slide_offset;
        }
    }

    // Walk the backtrace, recreating each matched patch.
    let mut best_deg_patches = vec![PatchSimilarityResult::default(); num_patches];
    for patch_index in (0..num_patches).rev() {
        let ref_patch = &ref_patches[patch_index];
        let deg_patch = build_degraded_patch(
            spectrogram_data,
            last_offset..last_offset + ref_patch.cols(),
            ref_patch.rows(),
            ref_patch.cols(),
        );
        let mut result = nsim::measure_patch_similarity(ref_patch, &deg_patch);
        if last_offset as i64 == backtrace[patch_index][last_offset] {
            // No matching patch was found for this reference patch: a null
            // match, distinct from a silent patch.
            result.deg_patch_start_time = 0.0;
            result.deg_patch_end_time = 0.0;
            result.similarity = 0.0;
            result.freq_band_means = vec![0.0; result.freq_band_means.len()];
        } else {
            result.deg_patch_start_time = last_offset as f64 * frame_duration;
            result.deg_patch_end_time = result.deg_patch_start_time + patch_duration;
        }
        result.ref_patch_start_time = ref_patch_indices[patch_index] as f64 * frame_duration;
        result.ref_patch_end_time = result.ref_patch_start_time + patch_duration;
        let prev = backtrace[patch_index][last_offset];
        best_deg_patches[patch_index] = result;
        if patch_index > 0 {
            debug_assert!(
                prev >= 0,
                "backtrace should always be set for inner patches"
            );
            last_offset = prev.max(0) as usize;
        }
    }
    Ok(best_deg_patches)
}

fn calc_max_num_patches(
    ref_patch_indices: &[usize],
    num_frames_in_deg_spectro: usize,
    num_frames_per_patch: usize,
) -> usize {
    let mut num_patches = ref_patch_indices.len();
    // The last patch can start up to half a patch away. Note the wrapping
    // subtraction: the C++ subtracts unsigned values, so a last patch index
    // smaller than half a patch (only possible for single-patch signals)
    // wraps and gets dropped. Reproduced for conformance.
    while num_patches > 0
        && ref_patch_indices[num_patches - 1].wrapping_sub(num_frames_per_patch / 2)
            > num_frames_in_deg_spectro
    {
        num_patches -= 1;
    }
    num_patches
}

/// Extracts the degraded spectrogram columns `window_range`, zero-padding
/// frames past the end of the spectrogram. (The C++ also pads negative frames
/// on the left, but no caller can produce a negative window start since the
/// v3.1.0 dynamic-program rewrite.)
fn build_degraded_patch(
    spectrogram_data: &Matrix,
    window_range: Range<usize>,
    window_height: usize,
    window_width: usize,
) -> Matrix {
    let mut deg_patch = Matrix::zeros(window_height, window_width);
    let last_real_frame = window_range.end.min(spectrogram_data.cols());
    for row_index in 0..spectrogram_data.rows() {
        let mut row = spectrogram_data.row_subset(row_index, window_range.start..last_real_frame);
        if window_range.end > spectrogram_data.cols() {
            row.resize(
                row.len() + (window_range.end - spectrogram_data.cols()),
                0.0,
            );
        }
        deg_patch.set_row(row_index, &row);
    }
    deg_patch
}

/// Extracts `[start_time, end_time)` seconds of audio, zero-padding out-of
/// range regions. Like the C++, the padding is prepended in both cases.
fn slice(signal: &AudioSignal, start_time: f64, end_time: f64) -> AudioSignal {
    let sample_rate = signal.sample_rate as f64;
    let num_samples = signal.samples.len();
    let start_index = ((start_time * sample_rate) as i64).max(0) as usize;
    let end_index = ((end_time * sample_rate) as i64).min(num_samples as i64 - 1) as usize;
    let mut samples = signal.samples[start_index..end_index].to_vec();

    // Add silence for a patch running past the end of the degraded signal...
    let end_time_diff = end_time * sample_rate - num_samples as f64;
    if end_time_diff > 0.0 {
        let mut padded = vec![0.0; end_time_diff as usize];
        padded.append(&mut samples);
        samples = padded;
    }
    // ...or before its start.
    if start_time < 0.0 {
        let mut padded = vec![0.0; (-start_time * sample_rate) as usize];
        padded.append(&mut samples);
        samples = padded;
    }
    AudioSignal::new(samples, signal.sample_rate)
}

/// For each coarsely matched patch pair, realigns the underlying audio with
/// sub-frame precision, rebuilds both spectrograms, rescores, and keeps
/// whichever similarity is higher.
pub fn finely_align_and_recreate_patches(
    level: fearless_simd::Level,
    fft_manager: &alignment::FftManager,
    sim_results: &mut [PatchSimilarityResult],
    ref_signal: &AudioSignal,
    deg_signal: &AudioSignal,
    spect_builder: &GammatoneSpectrogramBuilder,
    window: &AnalysisWindow,
) -> Result<()> {
    for sim_result in sim_results {
        // Skip null matches.
        if sim_result.deg_patch_start_time == sim_result.deg_patch_end_time
            && sim_result.deg_patch_start_time == 0.0
        {
            continue;
        }

        let mut ref_patch_audio = slice(
            ref_signal,
            sim_result.ref_patch_start_time,
            sim_result.ref_patch_end_time,
        );
        let mut deg_patch_audio = slice(
            deg_signal,
            sim_result.deg_patch_start_time,
            sim_result.deg_patch_end_time,
        );
        let lag =
            alignment::align_and_truncate(fft_manager, &mut ref_patch_audio, &mut deg_patch_audio);
        let new_ref_duration = ref_patch_audio.duration();
        let new_deg_duration = deg_patch_audio.duration();

        let mut ref_spectrogram = spect_builder.build(level, &ref_patch_audio, window)?;
        let mut deg_spectrogram = spect_builder.build(level, &deg_patch_audio, window)?;
        spectrogram::prepare_spectrograms_for_comparison(
            &mut ref_spectrogram,
            &mut deg_spectrogram,
        );

        let mut new_sim_result =
            nsim::measure_patch_similarity(&ref_spectrogram.data, &deg_spectrogram.data);
        if new_sim_result.similarity >= sim_result.similarity {
            if lag > 0.0 {
                new_sim_result.ref_patch_start_time = sim_result.ref_patch_start_time + lag;
                new_sim_result.deg_patch_start_time = sim_result.deg_patch_start_time;
            } else {
                new_sim_result.ref_patch_start_time = sim_result.ref_patch_start_time;
                new_sim_result.deg_patch_start_time = sim_result.deg_patch_start_time - lag;
            }
            new_sim_result.ref_patch_end_time =
                new_sim_result.ref_patch_start_time + new_ref_duration;
            new_sim_result.deg_patch_end_time =
                new_sim_result.deg_patch_start_time + new_deg_duration;
            *sim_result = new_sim_result;
        }
    }
    Ok(())
}
