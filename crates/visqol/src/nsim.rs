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

    let rows = ref_patch.rows();
    let cols = ref_patch.cols();
    debug_assert_eq!((rows, cols), (deg_patch.rows(), deg_patch.cols()));

    // The five convolutions share one padding buffer, and the three
    // elementwise products share one input buffer.
    let mut padded = Matrix::zeros(rows + 2, cols + 2);
    let mut product = Matrix::zeros(rows, cols);
    let mut mu_r = Matrix::zeros(rows, cols);
    let mut mu_d = Matrix::zeros(rows, cols);
    let mut conv_rr = Matrix::zeros(rows, cols);
    let mut conv_dd = Matrix::zeros(rows, cols);
    let mut conv_rd = Matrix::zeros(rows, cols);

    conv2d_with_boundary_into(ref_patch, &mut padded, &mut mu_r);
    conv2d_with_boundary_into(deg_patch, &mut padded, &mut mu_d);
    let mut conv_of_product = |a: &Matrix, b: &Matrix, out: &mut Matrix| {
        for ((p, &x), &y) in product.data_mut().iter_mut().zip(a.data()).zip(b.data()) {
            *p = x * y;
        }
        conv2d_with_boundary_into(&product, &mut padded, out);
    };
    conv_of_product(ref_patch, ref_patch, &mut conv_rr);
    conv_of_product(deg_patch, deg_patch, &mut conv_dd);
    conv_of_product(ref_patch, deg_patch, &mut conv_rd);

    // Each similarity cell needs only the five convolution values at the same
    // index, so the map is written over the cross-term buffer in place.
    let mut sim_map = conv_rd;
    for i in 0..sim_map.data().len() {
        let mu_r_mu_d = mu_r[i] * mu_d[i];
        let intensity = (2.0 * mu_r_mu_d + c1) / (mu_r[i] * mu_r[i] + mu_d[i] * mu_d[i] + c1);
        let sigma_r_sq = conv_rr[i] - mu_r[i] * mu_r[i];
        let sigma_d_sq = conv_dd[i] - mu_d[i] * mu_d[i];
        let sigma_r_d = sim_map[i] - mu_r_mu_d;
        // Negative variances can occur for silent patches due to precision;
        // the C++ replaces the sqrt with zero in that case.
        let sigma_prod = sigma_r_sq * sigma_d_sq;
        let structure_denom = if sigma_prod < 0.0 {
            c3
        } else {
            sigma_prod.sqrt() + c3
        };
        let structure = (sigma_r_d + c3) / structure_denom;
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
    let mut padded = Matrix::zeros(input.rows() + 2, input.cols() + 2);
    let mut out = Matrix::zeros(input.rows(), input.cols());
    conv2d_with_boundary_into(input, &mut padded, &mut out);
    out
}

/// [`conv2d_with_boundary`] with caller-provided buffers, for use in per-cell
/// loops. `padded` must be `input`'s size plus 2 in both dimensions; `out`
/// must be `input`'s size. Every cell of both buffers is overwritten.
///
/// The nine taps of each output cell are accumulated one at a time in the
/// same descending-filter-index order as the C++, so results are
/// bit-identical to the naive nested-loop form; different output cells are
/// independent, which lets the compiler vectorize down each column.
/// `inline(always)` so the [`multiversion`] wrapper around
/// [`similarity_at_offset`] can compile it for wider register files.
#[inline(always)]
fn conv2d_with_boundary_into(input: &Matrix, padded: &mut Matrix, out: &mut Matrix) {
    add_matrix_boundary_into(input, padded);

    let i_r_c = padded.rows();
    let o_r_c = out.rows();
    let o_c_c = out.cols();

    for o_col in 0..o_c_c {
        let cols = &padded.data()[o_col * i_r_c..(o_col + 3) * i_r_c];
        let (col0, rest) = cols.split_at(i_r_c);
        let (col1, col2) = rest.split_at(i_r_c);
        let out_col = &mut out.col_mut(o_col)[..o_r_c];
        for o_row in 0..o_r_c {
            let mut sum = 0.0;
            sum += col0[o_row] * WINDOW[8];
            sum += col0[o_row + 1] * WINDOW[7];
            sum += col0[o_row + 2] * WINDOW[6];
            sum += col1[o_row] * WINDOW[5];
            sum += col1[o_row + 1] * WINDOW[4];
            sum += col1[o_row + 2] * WINDOW[3];
            sum += col2[o_row] * WINDOW[2];
            sum += col2[o_row + 1] * WINDOW[1];
            sum += col2[o_row + 2] * WINDOW[0];
            out_col[o_row] = sum;
        }
    }
}

/// Pads by one cell on every side, replicating edge rows first and then edge
/// columns (which fills the corners from the row-replicated matrix). Writes
/// every cell of `out`.
#[inline(always)]
fn add_matrix_boundary_into(input: &Matrix, out: &mut Matrix) {
    let rows = input.rows();
    for c in 0..input.cols() {
        let out_col = out.col_mut(c + 1);
        out_col[1..=rows].copy_from_slice(input.col(c));
        out_col[0] = out_col[1];
        out_col[rows + 1] = out_col[rows];
    }
    let padded_rows = rows + 2;
    let last_col = out.cols() - 1;
    let (first, rest) = out.data_mut().split_at_mut(padded_rows);
    first.copy_from_slice(&rest[..padded_rows]);
    let (rest, last) = out.data_mut().split_at_mut(last_col * padded_rows);
    last.copy_from_slice(&rest[(last_col - 1) * padded_rows..]);
}

/// The DTW-style patch search evaluates NSIM between each reference patch and
/// the degraded patch at every candidate offset — thousands of cells, each
/// needing five 3x3 convolutions. Most of that work is shared and can be
/// hoisted (see [`similarity_at_offset`]):
///
/// - `conv(ref)` and `conv(ref²)` depend only on the reference patch: one
///   [`RefPatchConv`] per patch.
/// - `conv(deg)` and `conv(deg²)` depend only on the degraded spectrogram.
///   Because the convolution is local and the boundary is edge replication, a
///   patch's convolution at its *interior* columns reads exactly the same
///   nine values in the same order as the convolution of the whole
///   spectrogram at that column — bit-identical. Only the patch's first and
///   last columns differ (they replicate the patch edge instead of reading
///   the real neighbor column), and those are precomputed per offset in
///   [`DegSpectrogramConv`].
/// - `conv(ref ∘ deg)` depends on both and remains per-cell.
///
/// Only patches lying fully inside the spectrogram are supported; offsets
/// whose patch would be zero-padded past the end must use
/// [`measure_patch_similarity`].
pub struct RefPatchConv {
    mu: Matrix,
    sigma_sq: Matrix,
}

impl RefPatchConv {
    pub fn new(ref_patch: &Matrix) -> Self {
        let mu = conv2d_with_boundary(ref_patch);
        let mut sq = ref_patch.clone();
        for v in sq.data_mut() {
            *v *= *v;
        }
        let mut sigma_sq = conv2d_with_boundary(&sq);
        for (s, &m) in sigma_sq.data_mut().iter_mut().zip(mu.data()) {
            *s -= m * m;
        }
        RefPatchConv { mu, sigma_sq }
    }
}

/// See [`RefPatchConv`]. `mu`/`sigma_sq` are the whole-spectrogram
/// convolution maps, valid for patch-interior columns; `left_*`/`right_*`
/// hold the patch-boundary fix-ups, with column `o` giving the values for
/// the first and last column of the `patch_width`-wide patch starting at
/// offset `o`.
pub struct DegSpectrogramConv {
    mu: Matrix,
    sigma_sq: Matrix,
    left_mu: Matrix,
    left_sigma_sq: Matrix,
    right_mu: Matrix,
    right_sigma_sq: Matrix,
    patch_width: usize,
}

impl DegSpectrogramConv {
    pub fn new(spectrogram: &Matrix, patch_width: usize) -> Self {
        assert!(patch_width >= 3);
        let mu = conv2d_with_boundary(spectrogram);
        let mut sq = spectrogram.clone();
        for v in sq.data_mut() {
            *v *= *v;
        }
        let mut sigma_sq = conv2d_with_boundary(&sq);
        for (s, &m) in sigma_sq.data_mut().iter_mut().zip(mu.data()) {
            *s -= m * m;
        }

        // Boundary fix-ups for every fully-inside offset.
        let rows = spectrogram.rows();
        let num_offsets = (spectrogram.cols() + 1).saturating_sub(patch_width);
        let mut left_mu = Matrix::zeros(rows, num_offsets);
        let mut left_sigma_sq = Matrix::zeros(rows, num_offsets);
        let mut right_mu = Matrix::zeros(rows, num_offsets);
        let mut right_sigma_sq = Matrix::zeros(rows, num_offsets);
        for o in 0..num_offsets {
            // A patch's first column sees source columns [o, o, o + 1]; its
            // last column sees [o + w - 2, o + w - 1, o + w - 1].
            let last = o + patch_width - 1;
            for r in 0..rows {
                let m = edge_conv(spectrogram, r, [o, o, o + 1]);
                let s2 = edge_conv(&sq, r, [o, o, o + 1]);
                left_mu.set(r, o, m);
                left_sigma_sq.set(r, o, s2 - m * m);

                let m = edge_conv(spectrogram, r, [last - 1, last, last]);
                let s2 = edge_conv(&sq, r, [last - 1, last, last]);
                right_mu.set(r, o, m);
                right_sigma_sq.set(r, o, s2 - m * m);
            }
        }

        DegSpectrogramConv {
            mu,
            sigma_sq,
            left_mu,
            left_sigma_sq,
            right_mu,
            right_sigma_sq,
            patch_width,
        }
    }
}

/// One output cell of the boundary convolution, reading the given (clamped)
/// source columns with replicated rows. The product-and-sum order matches
/// `conv2d_with_boundary` exactly.
fn edge_conv(input: &Matrix, row: usize, src_cols: [usize; 3]) -> f64 {
    let mut sum = 0.0;
    let mut filter_index = 9;
    for src_col in src_cols {
        for f_row in 0..3 {
            filter_index -= 1;
            let src_row = (row + f_row).saturating_sub(1).min(input.rows() - 1);
            sum += input.at(src_row, src_col) * WINDOW[filter_index];
        }
    }
    sum
}

/// Reusable buffers for [`similarity_at_offset`].
pub struct NsimScratch {
    product: Matrix,
    padded: Matrix,
    conv: Matrix,
    row_sums: Vec<f64>,
}

impl NsimScratch {
    pub fn new(rows: usize, patch_width: usize) -> Self {
        NsimScratch {
            product: Matrix::zeros(rows, patch_width),
            padded: Matrix::zeros(rows + 2, patch_width + 2),
            conv: Matrix::zeros(rows, patch_width),
            row_sums: vec![0.0; rows],
        }
    }
}

/// NSIM (the `similarity` field of [`measure_patch_similarity`], which see)
/// between `ref_patch` and the degraded patch starting at `offset`, which
/// must lie fully inside the spectrogram. Produces bit-identical results
/// while only performing the cross-term convolution per call.
///
/// This is the DP search's inner loop, so the work is dispatched through
/// [`multiversion`] to vectorize with whatever the CPU offers.
pub fn similarity_at_offset(
    ref_patch: &Matrix,
    ref_conv: &RefPatchConv,
    spectrogram: &Matrix,
    deg_conv: &DegSpectrogramConv,
    offset: usize,
    scratch: &mut NsimScratch,
) -> f64 {
    multiversion::multiversion(
        #[inline(always)]
        || similarity_at_offset_impl(ref_patch, ref_conv, spectrogram, deg_conv, offset, scratch),
    )
}

#[inline(always)]
fn similarity_at_offset_impl(
    ref_patch: &Matrix,
    ref_conv: &RefPatchConv,
    spectrogram: &Matrix,
    deg_conv: &DegSpectrogramConv,
    offset: usize,
    scratch: &mut NsimScratch,
) -> f64 {
    let rows = ref_patch.rows();
    let width = deg_conv.patch_width;
    debug_assert_eq!(ref_patch.cols(), width);
    debug_assert!(offset + width <= spectrogram.cols());

    let c1 = (0.01 * INTENSITY_RANGE).powi(2);
    let c3 = (0.03 * INTENSITY_RANGE).powi(2) / 2.0;

    // conv(ref ∘ deg), the one convolution that changes per cell.
    for c in 0..width {
        let deg_col = &spectrogram.col(offset + c)[..rows];
        let ref_col = &ref_patch.col(c)[..rows];
        let prod_col = &mut scratch.product.col_mut(c)[..rows];
        for r in 0..rows {
            prod_col[r] = ref_col[r] * deg_col[r];
        }
    }
    conv2d_with_boundary_into(&scratch.product, &mut scratch.padded, &mut scratch.conv);

    // The similarity map, reduced per row in the same order as
    // `measure_patch_similarity` (columns outer, then a mean of row means).
    scratch.row_sums.fill(0.0);
    let row_sums = &mut scratch.row_sums[..rows];
    for c in 0..width {
        let (mu_d_col, sigma_d_col) = if c == 0 {
            (
                deg_conv.left_mu.col(offset),
                deg_conv.left_sigma_sq.col(offset),
            )
        } else if c == width - 1 {
            (
                deg_conv.right_mu.col(offset),
                deg_conv.right_sigma_sq.col(offset),
            )
        } else {
            (
                deg_conv.mu.col(offset + c),
                deg_conv.sigma_sq.col(offset + c),
            )
        };
        let mu_d_col = &mu_d_col[..rows];
        let sigma_d_col = &sigma_d_col[..rows];
        let mu_r_col = &ref_conv.mu.col(c)[..rows];
        let sigma_r_col = &ref_conv.sigma_sq.col(c)[..rows];
        let conv_rd_col = &scratch.conv.col(c)[..rows];
        for r in 0..rows {
            let mu_r = mu_r_col[r];
            let mu_d = mu_d_col[r];
            let mu_r_mu_d = mu_r * mu_d;
            let intensity = (2.0 * mu_r_mu_d + c1) / (mu_r * mu_r + mu_d * mu_d + c1);
            let sigma_r_d = conv_rd_col[r] - mu_r_mu_d;
            let sigma_prod = sigma_r_col[r] * sigma_d_col[r];
            let structure_denom = sigma_prod.max(0.0).sqrt() + c3;
            let structure = (sigma_r_d + c3) / structure_denom;
            row_sums[r] += intensity * structure;
        }
    }

    let mut freq_band_sum = 0.0;
    for &row_sum in &scratch.row_sums {
        freq_band_sum += row_sum / width as f64;
    }
    freq_band_sum / rows as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random matrix with dB-spectrogram-like values.
    fn test_matrix(rows: usize, cols: usize, seed: u64) -> Matrix {
        let mut state = seed;
        let mut m = Matrix::zeros(rows, cols);
        for v in m.data_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = (state >> 11) as f64 / (1u64 << 53) as f64 * 45.0;
        }
        m
    }

    /// The precomputed-convolution fast path must be bit-identical to
    /// `measure_patch_similarity` at every fully-inside offset.
    #[test]
    fn similarity_at_offset_is_bit_exact() {
        for (rows, width, cols) in [(32, 30, 97), (21, 20, 20), (32, 30, 30), (8, 3, 11)] {
            let spectrogram = test_matrix(rows, cols, 0x9E3779B97F4A7C15);
            let ref_patch = test_matrix(rows, width, 0xD1B54A32D192ED03);
            let deg_conv = DegSpectrogramConv::new(&spectrogram, width);
            let ref_conv = RefPatchConv::new(&ref_patch);
            let mut scratch = NsimScratch::new(rows, width);

            for offset in 0..=(cols - width) {
                let deg_patch = spectrogram.get_cols(offset..offset + width);
                let expected = measure_patch_similarity(&ref_patch, &deg_patch).similarity;
                let actual = similarity_at_offset(
                    &ref_patch,
                    &ref_conv,
                    &spectrogram,
                    &deg_conv,
                    offset,
                    &mut scratch,
                );
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "rows={rows} width={width} cols={cols} offset={offset}: \
                     {actual:e} != {expected:e}"
                );
            }
        }
    }
}
