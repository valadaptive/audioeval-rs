//! Minimal column-major matrix of `f64`, mirroring the Armadillo-backed
//! `AMatrix` in the C++ implementation. Column-major layout matters: the
//! 2D convolution in the NSIM measure indexes the underlying buffer flat,
//! and spectrogram frames (columns) stay contiguous.

use std::ops::{Index, IndexMut, Range};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::filled(rows, cols, 0.0)
    }

    pub fn filled(rows: usize, cols: usize, value: f64) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    #[inline]
    pub fn at(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < self.rows && col < self.cols);
        self.data[col * self.rows + row]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        debug_assert!(row < self.rows && col < self.cols);
        self.data[col * self.rows + row] = value;
    }

    pub fn col(&self, col: usize) -> &[f64] {
        &self.data[col * self.rows..(col + 1) * self.rows]
    }

    pub fn col_mut(&mut self, col: usize) -> &mut [f64] {
        &mut self.data[col * self.rows..(col + 1) * self.rows]
    }

    pub fn set_row(&mut self, row: usize, values: &[f64]) {
        assert_eq!(values.len(), self.cols);
        for (c, &v) in values.iter().enumerate() {
            self.set(row, c, v);
        }
    }

    /// Copy of columns `start..end`.
    pub fn get_cols(&self, range: Range<usize>) -> Matrix {
        let Range { start, end } = range;
        let cols = end - start;
        Matrix {
            rows: self.rows,
            cols,
            data: self.data[start * self.rows..end * self.rows].to_vec(),
        }
    }

    /// Elements `start..end` of one row.
    pub fn row_subset(&self, row: usize, cols: Range<usize>) -> Vec<f64> {
        cols.map(|c| self.at(row, c)).collect()
    }

    /// Per-row mean across columns (Armadillo `mean(m, 1)`).
    pub fn mean_per_row(&self) -> Vec<f64> {
        let mut means = vec![0.0; self.rows];
        for c in 0..self.cols {
            for (r, m) in means.iter_mut().enumerate() {
                *m += self.at(r, c);
            }
        }
        for m in &mut means {
            *m /= self.cols as f64;
        }
        means
    }

    /// Per-row unbiased standard deviation (Armadillo `stddev(m, 0, 1)`).
    pub fn stddev_per_row(&self) -> Vec<f64> {
        let means = self.mean_per_row();
        let mut stddevs = vec![0.0; self.rows];
        if self.cols < 2 {
            return stddevs;
        }
        for c in 0..self.cols {
            for r in 0..self.rows {
                let d = self.at(r, c) - means[r];
                stddevs[r] += d * d;
            }
        }
        for s in &mut stddevs {
            *s = (*s / (self.cols - 1) as f64).sqrt();
        }
        stddevs
    }

    pub fn min(&self) -> f64 {
        self.data.iter().copied().fold(f64::INFINITY, f64::min)
    }
}

impl Index<usize> for Matrix {
    type Output = f64;

    /// Flat access in column-major order, like Armadillo's `operator()(i)`.
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Matrix {
    /// Flat access in column-major order, like Armadillo's `operator()(i)`.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
