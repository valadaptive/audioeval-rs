//! The default speech-mode similarity-to-quality mapper: a TensorFlow
//! Lattice "calibrated lattice ensemble", evaluated directly from the
//! parameters extracted out of the upstream TFLite flatbuffer (see
//! `models/extract_lattice_model.py`).
//!
//! Structure, mirroring the TFLite graph op for op:
//!
//! 1. Each of the 85 scalar inputs (fvnsim, fvnsim10, fstdnsim, fvdegenergy
//!    per band, plus the quantile parameter tau, fixed at 0.5) runs through a
//!    piecewise-linear calibrator producing one value per lattice that
//!    consumes the feature. fvnsim/fvnsim10 calibrators have a missing-value
//!    branch keyed on *exact f32 equality* with a training-time sentinel
//!    (0.7 and 0.5 respectively) — an upstream quirk reproduced faithfully.
//! 2. 60 lattices, each interpolating over 2^12 corner values with clipped
//!    multilinear weights `1 - min(|v - vertex|, 1)` per dimension.
//! 3. A linear combination of the lattice outputs and one final
//!    piecewise-linear calibrator map to MOS-LQO.
//!
//! All arithmetic is f32 (the C++ narrows the double features when filling
//! the TFLite input tensors) with sequential reductions. The upstream model
//! runs under XNNPack, whose SIMD summation order differs; the resulting
//! discrepancy is below 1e-5, far inside the 1e-4 conformance tolerance.

use std::borrow::Cow;

use crate::{Error, Result};

const NUM_BANDS: usize = 21;
/// Feature vectors are laid out fvnsim | fvnsim10 | fstdnsim | fvdegenergy
/// | tau, matching the extraction script.
const NUM_FEATURES: usize = 4 * NUM_BANDS + 1;
/// The quantile the upstream mapper always requests (the median).
const TAU: f32 = 0.5;

struct Calibrator {
    /// Left keypoint of each linear segment.
    keypoints: Vec<f32>,
    /// Reciprocal segment lengths.
    inv_lengths: Vec<f32>,
    /// Per-unit rows of `1 + keypoints.len()` weights: bias, then one weight
    /// per segment.
    kernel: Vec<f32>,
    n_units: usize,
    /// Sentinel input value and the per-unit outputs it selects.
    missing: Option<(f32, Vec<f32>)>,
}

impl Calibrator {
    /// Appends the `n_units` calibrated values for input `x` to `out`.
    fn evaluate(&self, x: f32, out: &mut Vec<f32>, scratch: &mut Vec<f32>) {
        if let Some((sentinel, missing_vals)) = &self.missing
            && x == *sentinel
        {
            out.extend_from_slice(missing_vals);
            return;
        }
        scratch.clear();
        scratch.push(1.0);
        for (&kp, &il) in self.keypoints.iter().zip(&self.inv_lengths) {
            scratch.push(((x - kp) * il).clamp(0.0, 1.0));
        }
        for row in self.kernel.chunks_exact(scratch.len()) {
            let mut acc = 0.0f32;
            for (&k, &w) in row.iter().zip(&*scratch) {
                acc += k * w;
            }
            out.push(acc);
        }
    }
}

pub struct LatticeModel {
    calibrators: Vec<Calibrator>,
    /// Per lattice, `rank` indices into the concatenated calibrator outputs,
    /// most significant lattice dimension first.
    wiring: Vec<u32>,
    /// Per lattice, `2^rank` corner values.
    corners: Vec<f32>,
    ensemble_weights: Vec<f32>,
    output_keypoints: Vec<f32>,
    output_inv_lengths: Vec<f32>,
    output_kernel: Vec<f32>,
    rank: usize,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

#[inline(always)]
fn invalid_model(msg: impl Into<Cow<'static, str>>) -> Error {
    Error::InvalidLatticeModel(msg.into())
}

impl Reader<'_> {
    fn bytes(&mut self, n: usize) -> Result<&[u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let end = end.ok_or_else(|| invalid_model("truncated lattice model"))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        Ok(self
            .bytes(4 * n)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}

impl LatticeModel {
    /// The default speech-mode model shipped with ViSQOL
    /// (`models/lattice_speech_model.bin`, extracted from the upstream
    /// TFLite flatbuffer).
    pub fn default_speech_model() -> Self {
        Self::from_bytes(include_bytes!("../models/lattice_speech_model.bin"))
            .expect("embedded model is valid")
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        let mut r = Reader { buf, pos: 0 };
        if r.bytes(4)? != b"VQLM" {
            return Err(invalid_model("bad lattice model magic"));
        }
        let version = r.u32()?;
        if version != 1 {
            return Err(invalid_model(format!(
                "unsupported lattice model version {version}"
            )));
        }
        let n_features = r.u32()? as usize;
        let n_lattices = r.u32()? as usize;
        let rank = r.u32()? as usize;
        if n_features != NUM_FEATURES || rank == 0 || rank > 16 {
            return Err(invalid_model(format!(
                "unexpected lattice model shape ({n_features} features, rank {rank})"
            )));
        }

        let mut calibrators = Vec::with_capacity(n_features);
        let mut unit_offsets = Vec::with_capacity(n_features + 1);
        let mut total_units = 0u32;
        for _ in 0..n_features {
            let n_units = r.u32()? as usize;
            let n_kp = r.u32()? as usize;
            let has_missing = r.u32()? != 0;
            let missing_input_value = r.f32()?;
            let keypoints = r.f32s(n_kp)?;
            let inv_lengths = r.f32s(n_kp)?;
            let kernel = r.f32s(n_units * (n_kp + 1))?;
            let missing =
                has_missing.then(|| -> Result<_> { Ok((missing_input_value, r.f32s(n_units)?)) });
            unit_offsets.push(total_units);
            total_units += n_units as u32;
            calibrators.push(Calibrator {
                keypoints,
                inv_lengths,
                kernel,
                n_units,
                missing: missing.transpose()?,
            });
        }

        let mut wiring = Vec::with_capacity(n_lattices * rank);
        for _ in 0..n_lattices * rank {
            let feature = r.u32()? as usize;
            let unit = r.u32()?;
            if feature >= n_features || unit >= calibrators[feature].n_units as u32 {
                return Err(invalid_model("lattice wiring out of range"));
            }
            wiring.push(unit_offsets[feature] + unit);
        }
        let corners = r.f32s(n_lattices << rank)?;
        let ensemble_weights = r.f32s(n_lattices)?;
        let n_out_kp = r.u32()? as usize;
        let output_keypoints = r.f32s(n_out_kp)?;
        let output_inv_lengths = r.f32s(n_out_kp)?;
        let output_kernel = r.f32s(n_out_kp + 1)?;
        if r.pos != buf.len() {
            return Err(invalid_model("trailing lattice model data"));
        }

        Ok(LatticeModel {
            calibrators,
            wiring,
            corners,
            ensemble_weights,
            output_keypoints,
            output_inv_lengths,
            output_kernel,
            rank,
        })
    }

    pub fn predict(
        &self,
        fvnsim: &[f64],
        fvnsim10: &[f64],
        fstdnsim: &[f64],
        fvdegenergy: &[f64],
    ) -> f64 {
        assert!(
            [fvnsim, fvnsim10, fstdnsim, fvdegenergy]
                .iter()
                .all(|v| v.len() == NUM_BANDS),
            "the lattice model requires {NUM_BANDS} frequency bands"
        );

        // Calibrate all features into one flat buffer indexed by `wiring`.
        let mut calibrated = Vec::new();
        let mut scratch = Vec::new();
        let inputs = fvnsim
            .iter()
            .chain(fvnsim10)
            .chain(fstdnsim)
            .chain(fvdegenergy)
            .map(|&v| v as f32)
            .chain([TAU]);
        for (calibrator, x) in self.calibrators.iter().zip(inputs) {
            calibrator.evaluate(x, &mut calibrated, &mut scratch);
        }

        // Interpolate each lattice and sum the weighted outputs.
        let mut corner_weights = vec![0.0f32; 1 << self.rank];
        let mut ensemble = 0.0f32;
        for (lattice, (wiring, corners)) in self
            .wiring
            .chunks_exact(self.rank)
            .zip(self.corners.chunks_exact(1 << self.rank))
            .enumerate()
        {
            corner_weights[0] = 1.0;
            let mut len = 1;
            for &slot in wiring {
                let v = calibrated[slot as usize];
                let w0 = 1.0 - v.abs().min(1.0);
                let w1 = 1.0 - (v - 1.0).abs().min(1.0);
                for j in (0..len).rev() {
                    let a = corner_weights[j];
                    corner_weights[2 * j] = a * w0;
                    corner_weights[2 * j + 1] = a * w1;
                }
                len *= 2;
            }
            let mut dot = 0.0f32;
            for (&c, &w) in corners.iter().zip(&corner_weights) {
                dot += c * w;
            }
            ensemble += self.ensemble_weights[lattice] * dot;
        }

        // Output calibration.
        let mut out = self.output_kernel[0];
        for ((&kp, &il), &k) in self
            .output_keypoints
            .iter()
            .zip(&self.output_inv_lengths)
            .zip(&self.output_kernel[1..])
        {
            out += k * ((ensemble - kp) * il).clamp(0.0, 1.0);
        }
        out as f64
    }
}
