//! Spectrogram container and the dB-domain conditioning applied before
//! comparison, ports of `spectrogram.cc` and
//! `MiscAudio::PrepareSpectrogramsForComparison`.

use crate::matrix::Matrix;

const NOISE_FLOOR_RELATIVE_TO_PEAK_DB: f64 = 45.0;
const NOISE_FLOOR_ABSOLUTE_DB: f64 = -45.0;

pub struct Spectrogram {
    pub data: Matrix,
    pub center_freq_bands: Vec<f64>,
}

impl Spectrogram {
    pub fn new(data: Matrix, center_freq_bands: Vec<f64>) -> Self {
        Spectrogram {
            data,
            center_freq_bands,
        }
    }

    fn convert_to_db(&mut self) {
        for v in self.data.data_mut() {
            let abs = if v.abs() == 0.0 {
                f64::EPSILON
            } else {
                v.abs()
            };
            *v = 10.0 * abs.log10();
        }
    }

    fn raise_floor(&mut self, new_floor: f64) {
        for v in self.data.data_mut() {
            *v = v.max(new_floor);
        }
    }

    fn subtract_floor(&mut self, floor: f64) {
        for v in self.data.data_mut() {
            *v -= floor;
        }
    }
}

/// Applies floors and normalization to both spectrograms in tandem.
pub fn prepare_spectrograms_for_comparison(
    reference: &mut Spectrogram,
    degraded: &mut Spectrogram,
) {
    reference.convert_to_db();
    degraded.convert_to_db();

    reference.raise_floor(NOISE_FLOOR_ABSOLUTE_DB);
    degraded.raise_floor(NOISE_FLOOR_ABSOLUTE_DB);

    // Per-frame relative threshold below the louder of the two frames.
    let min_cols = reference.data.cols().min(degraded.data.cols());
    for c in 0..min_cols {
        let max = |m: &Matrix| m.col(c).iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let any_max = max(&reference.data).max(max(&degraded.data));
        let floor_db = any_max - NOISE_FLOOR_RELATIVE_TO_PEAK_DB;
        for v in reference.data.col_mut(c) {
            *v = v.max(floor_db);
        }
        for v in degraded.data.col_mut(c) {
            *v = v.max(floor_db);
        }
    }

    // Normalize to a 0 dB global floor.
    let lowest_floor = reference.data.min().min(degraded.data.min());
    reference.subtract_floor(lowest_floor);
    degraded.subtract_floor(lowest_floor);
}
