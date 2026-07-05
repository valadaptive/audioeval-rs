//! Reference patch selection: the uniform `ImagePatchCreator` used for audio
//! mode and the voice-activity-gated `VadPatchCreator` used for speech mode.

use crate::analysis_window::AnalysisWindow;
use crate::audio_signal::AudioSignal;
use crate::matrix::Matrix;
use crate::{Error, Result};

pub enum PatchCreator {
    Image,
    Vad,
}

impl PatchCreator {
    pub fn create_ref_patch_indices(
        &self,
        spectrogram: &Matrix,
        ref_signal: &AudioSignal,
        window: &AnalysisWindow,
        patch_size: usize,
    ) -> Result<Vec<usize>> {
        match self {
            PatchCreator::Image => create_image_patch_indices(spectrogram, patch_size),
            PatchCreator::Vad => Ok(create_vad_patch_indices(
                spectrogram,
                ref_signal,
                window,
                patch_size,
            )),
        }
    }
}

pub fn create_patches_from_indices(
    spectrogram: &Matrix,
    patch_indices: &[usize],
    patch_size: usize,
) -> Vec<Matrix> {
    patch_indices
        .iter()
        .map(|&start| spectrogram.get_cols(start, start + patch_size - 1))
        .collect()
}

fn create_image_patch_indices(spectrogram: &Matrix, patch_size: usize) -> Result<Vec<usize>> {
    let spectrum_length = spectrogram.cols();
    let init_patch_index = patch_size / 2;
    if spectrum_length < patch_size + init_patch_index {
        return Err(Error::ReferenceSpectrogramTooSmall {
            frames: spectrum_length,
            required: patch_size + init_patch_index,
        });
    }
    let max_index = if init_patch_index < spectrum_length - patch_size {
        spectrum_length - patch_size
    } else {
        init_patch_index + 1
    };
    Ok((init_patch_index..max_index)
        .step_by(patch_size)
        .map(|i| i - 1)
        .collect())
}

fn create_vad_patch_indices(
    spectrogram: &Matrix,
    ref_signal: &AudioSignal,
    window: &AnalysisWindow,
    patch_size: usize,
) -> Vec<usize> {
    const FRAMES_WITH_VA_THRESHOLD: f64 = 1.0;

    // MiscMath::Normalize: divide by the (signed) maximum element.
    let max = ref_signal
        .samples
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let normalized: Vec<f64> = ref_signal.samples.iter().map(|&s| s / max).collect();

    let frame_size = window.size as f64 * window.overlap;
    let patch_sample_len = (patch_size as f64 * frame_size) as usize;
    let spectrum_length = spectrogram.cols();
    let first_patch_idx = patch_size / 2 - 1;
    let patch_count = (spectrum_length - first_patch_idx) / patch_size;
    let total_sample_count = patch_count * patch_sample_len;

    let vad_res = get_voice_activity(
        &normalized,
        first_patch_idx,
        total_sample_count,
        frame_size as usize,
    );

    let mut patch_indices = Vec::with_capacity(patch_count);
    let mut patch_idx = first_patch_idx;
    for i in 0..patch_count {
        let frames_with_va: f64 = vad_res[i * patch_size..(i + 1) * patch_size].iter().sum();
        if frames_with_va >= FRAMES_WITH_VA_THRESHOLD {
            patch_indices.push(patch_idx);
        }
        patch_idx += patch_size;
    }
    patch_indices
}

fn get_voice_activity(
    signal: &[f64],
    start_sample: usize,
    total_samples: usize,
    frame_len: usize,
) -> Vec<f64> {
    let mut vad = RmsVad::new();
    let patch = &signal[start_sample..start_sample + total_samples];

    let mut frame = Vec::with_capacity(frame_len);
    for &value in patch {
        // Truncating cast after clamping, as in the C++.
        let scaled = (value * 32768.0).clamp(-32768.0, 32767.0) as i16;
        frame.push(scaled);
        if frame.len() == frame_len {
            vad.process_chunk(&frame);
            frame.clear();
        }
    }
    vad.get_vad_results()
}

/// RMS-threshold voice activity detector (`rms_vad.cc`).
struct RmsVad {
    each_chunk_result: Vec<bool>,
}

impl RmsVad {
    const SILENT_CHUNK_COUNT: usize = 3;
    const RMS_THRESHOLD: f64 = 5000.0;

    fn new() -> Self {
        RmsVad {
            each_chunk_result: Vec::new(),
        }
    }

    fn process_chunk(&mut self, chunk: &[i16]) {
        let square: f64 = chunk.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let rms = (square / chunk.len() as f64).sqrt();
        self.each_chunk_result.push(rms >= Self::RMS_THRESHOLD);
    }

    fn get_vad_results(&self) -> Vec<f64> {
        // The first chunks are always marked as active to avoid false
        // negatives; a chunk is only inactive if it and the previous
        // SILENT_CHUNK_COUNT - 1 chunks are all below threshold.
        let mut results = vec![1.0; Self::SILENT_CHUNK_COUNT - 1];
        for i in Self::SILENT_CHUNK_COUNT - 1..self.each_chunk_result.len() {
            let previous_silent = self.each_chunk_result[i + 1 - Self::SILENT_CHUNK_COUNT..i]
                .iter()
                .all(|&active| !active);
            if !self.each_chunk_result[i] && previous_silent {
                results.push(0.0);
            } else {
                results.push(1.0);
            }
        }
        results
    }
}
