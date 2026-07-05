//! Inference-only libsvm nu-SVR with an RBF kernel, replacing the vendored
//! libsvm used by the C++ for the audio-mode similarity-to-quality mapping.
//! Parses the standard libsvm text model format.

use std::borrow::Cow;

use crate::{Error, Result};

#[inline(always)]
fn invalid_model(msg: impl Into<Cow<'static, str>>) -> Error {
    Error::InvalidSVRModel(msg.into())
}

pub struct SvrModel {
    gamma: f64,
    rho: f64,
    /// (coefficient, dense support vector) pairs.
    support_vectors: Vec<(f64, Vec<f64>)>,
}

impl SvrModel {
    /// Parses a libsvm text model. Only the parameters used by nu-SVR with an
    /// RBF kernel are honored; anything else is rejected.
    pub fn from_text(text: &str) -> Result<Self> {
        let mut gamma = None;
        let mut rho = None;
        let mut lines = text.lines();
        for line in &mut lines {
            let mut parts = line.split_whitespace();
            let key = parts.next().unwrap_or("");
            let value = parts.next();
            match key {
                "svm_type" => {
                    if value != Some("nu_svr") {
                        return Err(invalid_model(format!(
                            "unsupported svm_type {value:?}, expected nu_svr"
                        )));
                    }
                }
                "kernel_type" => {
                    if value != Some("rbf") {
                        return Err(invalid_model(format!(
                            "unsupported kernel_type {value:?}, expected rbf"
                        )));
                    }
                }
                "gamma" => gamma = value.and_then(|v| v.parse().ok()),
                "rho" => rho = value.and_then(|v| v.parse().ok()),
                "nr_class" | "total_sv" => {}
                "SV" => break,
                other => {
                    return Err(invalid_model(format!("unexpected header key {other:?}")));
                }
            }
        }
        let gamma = gamma.ok_or_else(|| invalid_model("missing gamma"))?;
        let rho = rho.ok_or_else(|| invalid_model("missing rho"))?;

        let mut support_vectors = Vec::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let coef: f64 = parts
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| invalid_model("bad SV coefficient"))?;
            let mut sv = Vec::new();
            for part in parts {
                let (index, value) = part
                    .split_once(':')
                    .ok_or_else(|| invalid_model("bad SV feature"))?;
                let index: usize = index
                    .parse()
                    .map_err(|_| invalid_model("bad SV feature index"))?;
                let value: f64 = value
                    .parse()
                    .map_err(|_| invalid_model("bad SV feature value"))?;
                // Feature indices are 1-based and may be sparse.
                if index > sv.len() {
                    sv.resize(index, 0.0);
                }
                sv[index - 1] = value;
            }
            support_vectors.push((coef, sv));
        }
        if support_vectors.is_empty() {
            return Err(invalid_model("no support vectors"));
        }
        Ok(SvrModel {
            gamma,
            rho,
            support_vectors,
        })
    }

    /// The default audio-mode model shipped with ViSQOL
    /// (`model/libsvm_nu_svr_model.txt`).
    pub fn default_audio_model() -> Self {
        Self::from_text(include_str!("../models/libsvm_nu_svr_model.txt"))
            .expect("embedded model is valid")
    }

    pub fn predict(&self, observation: &[f64]) -> f64 {
        let mut sum = 0.0;
        for (coef, sv) in &self.support_vectors {
            let mut dist_sq = 0.0;
            for i in 0..observation.len().max(sv.len()) {
                let a = observation.get(i).copied().unwrap_or(0.0);
                let b = sv.get(i).copied().unwrap_or(0.0);
                dist_sq += (a - b) * (a - b);
            }
            sum += coef * (-self.gamma * dist_sq).exp();
        }
        sum - self.rho
    }
}
