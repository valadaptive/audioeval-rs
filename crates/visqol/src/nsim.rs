//! Neurogram Similarity Index Measure: an SSIM variant computed over
//! spectrogram patches. Ports of `neurogram_similiarity_index_measure.cc` and
//! `convolution_2d.cc`.

use crate::matrix::Matrix;

/// Per-patch similarity statistics plus the matched patches' time bounds,
/// mirroring the C++ `PatchSimilarityResult`.
#[derive(Clone, Debug, Default)]
pub struct PatchSimilarityResult {
    /// Mean of `freq_band_means`, a.k.a. NSIM.
    pub similarity: f64,
    /// Mean similarity per frequency band.
    pub freq_band_means: Vec<f64>,
    /// Standard deviation of similarity per frequency band.
    pub freq_band_stddevs: Vec<f64>,
    /// Mean degraded-spectrogram energy per frequency band.
    pub freq_band_deg_energy: Vec<f64>,
    pub ref_patch_start_time: f64,
    pub ref_patch_end_time: f64,
    pub deg_patch_start_time: f64,
    pub deg_patch_end_time: f64,
}

const INTENSITY_RANGE: f64 = 1.0;

/// 3x3 Gaussian window with sigma 0.5 used for the local statistics.
const WINDOW: [f64; 9] = [
    0.0113033910173052,
    0.0838251475442633,
    0.0113033910173052,
    0.0838251475442633,
    0.619485845753726,
    0.0838251475442633,
    0.0113033910173052,
    0.0838251475442633,
    0.0113033910173052,
];

pub fn measure_patch_similarity(ref_patch: &Matrix, deg_patch: &Matrix) -> PatchSimilarityResult {
    let c1 = (0.01 * INTENSITY_RANGE).powi(2);
    let c3 = (0.03 * INTENSITY_RANGE).powi(2) / 2.0;

    let mu_r = conv2d_with_boundary(ref_patch);
    let mu_d = conv2d_with_boundary(deg_patch);

    let pointwise = |a: &Matrix, b: &Matrix, f: fn(f64, f64) -> f64| {
        let mut out = Matrix::zeros(a.rows(), a.cols());
        for ((o, &x), &y) in out.data_mut().iter_mut().zip(a.data()).zip(b.data()) {
            *o = f(x, y);
        }
        out
    };
    let mul = |a: &Matrix, b: &Matrix| pointwise(a, b, |x, y| x * y);

    let ref_mu_sq = mul(&mu_r, &mu_r);
    let deg_mu_sq = mul(&mu_d, &mu_d);
    let mu_r_mu_d = mul(&mu_r, &mu_d);

    let sigma_r_sq = pointwise(
        &conv2d_with_boundary(&mul(ref_patch, ref_patch)),
        &ref_mu_sq,
        |x, y| x - y,
    );
    let sigma_d_sq = pointwise(
        &conv2d_with_boundary(&mul(deg_patch, deg_patch)),
        &deg_mu_sq,
        |x, y| x - y,
    );
    let sigma_r_d = pointwise(
        &conv2d_with_boundary(&mul(ref_patch, deg_patch)),
        &mu_r_mu_d,
        |x, y| x - y,
    );

    let mut sim_map = Matrix::zeros(ref_patch.rows(), ref_patch.cols());
    for i in 0..sim_map.data().len() {
        let intensity =
            (2.0 * mu_r_mu_d.flat(i) + c1) / (ref_mu_sq.flat(i) + deg_mu_sq.flat(i) + c1);
        // Negative variances can occur for silent patches due to precision;
        // the C++ replaces the sqrt with zero in that case.
        let sigma_prod = sigma_r_sq.flat(i) * sigma_d_sq.flat(i);
        let structure_denom = if sigma_prod < 0.0 {
            c3
        } else {
            sigma_prod.sqrt() + c3
        };
        let structure = (sigma_r_d.flat(i) + c3) / structure_denom;
        sim_map.data_mut()[i] = intensity * structure;
    }

    let freq_band_deg_energy = deg_patch.mean_per_row();
    let freq_band_means = sim_map.mean_per_row();
    let freq_band_stddevs = sim_map.stddev_per_row();
    let similarity = freq_band_means.iter().sum::<f64>() / freq_band_means.len() as f64;

    PatchSimilarityResult {
        similarity,
        freq_band_means,
        freq_band_stddevs,
        freq_band_deg_energy,
        ..Default::default()
    }
}

/// "Valid" 2D convolution with the 3x3 window after replicating the matrix's
/// border cells, so the output has the input's dimensions.
fn conv2d_with_boundary(input: &Matrix) -> Matrix {
    let padded = add_matrix_boundary(input);

    let i_r_c = padded.rows();
    let o_r_c = padded.rows() - 3 + 1;
    let o_c_c = padded.cols() - 3 + 1;
    let mut out = Matrix::zeros(o_r_c, o_c_c);

    for o_col in 0..o_c_c {
        for o_row in 0..o_r_c {
            let mut sum = 0.0;
            let mut filter_index = 9;
            for f_col in 0..3 {
                for f_row in 0..3 {
                    filter_index -= 1;
                    let idx = (f_col + o_col) * i_r_c + f_row + o_row;
                    sum += padded.flat(idx) * WINDOW[filter_index];
                }
            }
            out.set(o_row, o_col, sum);
        }
    }
    out
}

/// Pads by one cell on every side, replicating edge rows first and then edge
/// columns (which fills the corners from the row-replicated matrix).
fn add_matrix_boundary(input: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(input.rows() + 2, input.cols() + 2);
    for c in 0..input.cols() {
        for r in 0..input.rows() {
            out.set(r + 1, c + 1, input.at(r, c));
        }
    }
    let row_1 = out.row(1);
    out.set_row(0, &row_1);
    let row_last = out.row(out.rows() - 2);
    out.set_row(out.rows() - 1, &row_last);
    let col_1 = out.col(1).to_vec();
    out.set_col(0, &col_1);
    let col_last = out.col(out.cols() - 2).to_vec();
    out.set_col(out.cols() - 1, &col_last);
    out
}
