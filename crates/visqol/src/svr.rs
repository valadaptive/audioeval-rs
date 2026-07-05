//! Inference-only libsvm nu-SVR with an RBF kernel, replacing the vendored
//! libsvm used by the C++ for the audio-mode similarity-to-quality mapping.
//! Parses the standard libsvm text model format.

use crate::{Error, Result};

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
                        return Err(Error::InvalidModel(format!(
                            "unsupported svm_type {value:?}, expected nu_svr"
                        )));
                    }
                }
                "kernel_type" => {
                    if value != Some("rbf") {
                        return Err(Error::InvalidModel(format!(
                            "unsupported kernel_type {value:?}, expected rbf"
                        )));
                    }
                }
                "gamma" => gamma = value.and_then(|v| v.parse().ok()),
                "rho" => rho = value.and_then(|v| v.parse().ok()),
                "nr_class" | "total_sv" => {}
                "SV" => break,
                other => {
                    return Err(Error::InvalidModel(format!(
                        "unexpected header key {other:?}"
                    )));
                }
            }
        }
        let gamma = gamma.ok_or_else(|| Error::InvalidModel("missing gamma".into()))?;
        let rho = rho.ok_or_else(|| Error::InvalidModel("missing rho".into()))?;

        let mut support_vectors = Vec::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let coef: f64 = parts
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| Error::InvalidModel("bad SV coefficient".into()))?;
            let mut sv = Vec::new();
            for part in parts {
                let (index, value) = part
                    .split_once(':')
                    .ok_or_else(|| Error::InvalidModel("bad SV feature".into()))?;
                let index: usize = index
                    .parse()
                    .map_err(|_| Error::InvalidModel("bad SV feature index".into()))?;
                let value: f64 = value
                    .parse()
                    .map_err(|_| Error::InvalidModel("bad SV feature value".into()))?;
                // Feature indices are 1-based and may be sparse.
                if index > sv.len() {
                    sv.resize(index, 0.0);
                }
                sv[index - 1] = value;
            }
            support_vectors.push((coef, sv));
        }
        if support_vectors.is_empty() {
            return Err(Error::InvalidModel("no support vectors".into()));
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

/// The similarity-to-quality mapping stage.
pub enum SimilarityToQualityMapper {
    /// Audio mode: nu-SVR over the per-band mean similarities.
    Svr(SvrModel),
    /// Speech mode: exponential fit of mean NSIM over the TCD-VOIP dataset.
    SpeechExponential { scale_to_max_mos: bool },
}

impl SimilarityToQualityMapper {
    pub fn predict_quality(&self, fvnsim: &[f64]) -> f64 {
        match self {
            SimilarityToQualityMapper::Svr(model) => model.predict(fvnsim).clamp(1.0, 5.0),
            SimilarityToQualityMapper::SpeechExponential { scale_to_max_mos } => {
                const FIT_PARAMETER_A: f64 = -262.847869;
                const FIT_PARAMETER_B: f64 = 0.0154302525;
                const FIT_PARAMETER_X0: f64 = -361.063949;
                const FIT_SCALE: f64 = 1.245063;

                let nsim_mean = fvnsim.iter().sum::<f64>() / fvnsim.len() as f64;
                let mos =
                    FIT_PARAMETER_A + (FIT_PARAMETER_B * (nsim_mean - FIT_PARAMETER_X0)).exp();
                // The C++ narrows the scale factor to float.
                let scale = if *scale_to_max_mos {
                    FIT_SCALE as f32 as f64
                } else {
                    1.0
                };
                (mos * scale).clamp(1.0, 5.0)
            }
        }
    }
}
