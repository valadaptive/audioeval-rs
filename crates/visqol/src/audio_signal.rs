//! Mono audio signal input type.

#[derive(Clone, Debug)]
pub struct AudioSignal {
    /// Mono samples, nominally in [-1, 1].
    pub samples: Vec<f64>,
    pub sample_rate: u32,
}

impl AudioSignal {
    pub fn new(samples: Vec<f64>, sample_rate: u32) -> Self {
        AudioSignal {
            samples,
            sample_rate,
        }
    }

    /// Downmix planar channels to mono by averaging, like the C++
    /// `MiscAudio::ToMono`.
    pub fn from_channels<S: AsRef<[f32]>>(channels: &[S], sample_rate: u32) -> Self {
        let num_channels = channels.len();
        assert!(num_channels > 0, "need at least one channel");
        let len = channels.iter().map(|c| c.as_ref().len()).min().unwrap();
        let mut samples = vec![0.0f64; len];
        for channel in channels {
            for (acc, &s) in samples.iter_mut().zip(channel.as_ref()) {
                *acc += s as f64;
            }
        }
        for s in &mut samples {
            *s /= num_channels as f64;
        }
        AudioSignal::new(samples, sample_rate)
    }

    pub fn duration(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }
}

/// Scales the degraded signal so its sound pressure level matches the
/// reference (`MiscAudio::ScaleToMatchSoundPressureLevel`).
pub fn scale_to_match_sound_pressure_level(reference: &AudioSignal, degraded: &mut AudioSignal) {
    let ref_spl = calc_sound_pressure_level(reference);
    let deg_spl = calc_sound_pressure_level(degraded);
    let scale_factor = 10f64.powf((ref_spl - deg_spl) / 20.0);
    for sample in &mut degraded.samples {
        *sample *= scale_factor;
    }
}

fn calc_sound_pressure_level(signal: &AudioSignal) -> f64 {
    const SPL_REFERENCE_POINT: f64 = 0.00002;
    let sum: f64 = signal.samples.iter().map(|&s| s * s).sum();
    let sound_pressure = (sum / signal.samples.len() as f64).sqrt();
    20.0 * (sound_pressure / SPL_REFERENCE_POINT).log10()
}
