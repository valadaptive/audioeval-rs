//! Mono audio signal input type.

use std::{borrow::Cow, ops::Range, slice::SliceIndex};

use crate::audio_signal::private::Sample;

/// A mono audio signal with a given sample rate.
#[derive(Clone, Debug)]
pub struct AudioSignal<'a> {
    /// Mono samples, nominally in [-1, 1].
    pub samples: Cow<'a, [f64]>,
    pub sample_rate: u32,
}

mod private {
    /// Valid type for an input audio sample.
    pub trait Sample {
        fn as_normalized(&self) -> f64;
    }

    impl Sample for i16 {
        fn as_normalized(&self) -> f64 {
            *self as f64 * (1.0 / -(Self::MIN as f64))
        }
    }

    impl Sample for i32 {
        fn as_normalized(&self) -> f64 {
            *self as f64 * (1.0 / -(Self::MIN as f64))
        }
    }

    impl Sample for f32 {
        fn as_normalized(&self) -> f64 {
            *self as f64
        }
    }
    impl Sample for f64 {
        fn as_normalized(&self) -> f64 {
            *self
        }
    }
}

impl<'a> AudioSignal<'a> {
    pub fn new(samples: impl Into<Cow<'a, [f64]>>, sample_rate: u32) -> Self {
        AudioSignal {
            samples: samples.into(),
            sample_rate,
        }
    }

    /// Downmix planar channels to mono by averaging.
    pub fn from_channels<S: Sample, T: AsRef<[S]>>(
        channels: impl IntoIterator<Item = T>,
        sample_rate: u32,
    ) -> Self {
        let mut channels = channels.into_iter();
        let first_channel = channels.next().expect("need at least one channel");
        let first_channel = first_channel.as_ref();
        let mut samples: Vec<f64> = first_channel.iter().map(|s| s.as_normalized()).collect();
        let mut num_channels = 1;
        for channel in channels {
            for (acc, s) in samples.iter_mut().zip(channel.as_ref()) {
                *acc += s.as_normalized();
            }
            num_channels += 1;
        }
        if num_channels > 1 {
            let recip = 1.0 / (num_channels as f64);
            for s in &mut samples {
                *s *= recip;
            }
        }

        AudioSignal::new(samples, sample_rate)
    }

    pub fn duration(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }

    pub(crate) fn samples_mut(&mut self) -> &mut Vec<f64> {
        self.samples.to_mut()
    }

    pub(crate) fn as_borrowed(&self) -> AudioSignal<'_> {
        AudioSignal {
            samples: Cow::Borrowed(self.samples.as_ref()),
            sample_rate: self.sample_rate,
        }
    }

    pub(crate) fn slice(&self, range: impl SliceIndex<[f64], Output = [f64]>) -> AudioSignal<'_> {
        AudioSignal {
            samples: Cow::Borrowed(&self.samples[range]),
            sample_rate: self.sample_rate,
        }
    }

    /// Slices this signal while preserving borrowed storage where possible.
    /// An owned signal remains owned because a result cannot borrow from the
    /// consumed local `AudioSignal` that contained it.
    pub(crate) fn into_slice(self, range: Range<usize>) -> AudioSignal<'a> {
        let Range { start, end } = range;
        assert!(start <= end && end <= self.samples.len());
        let samples = match self.samples {
            Cow::Borrowed(samples) => Cow::Borrowed(&samples[start..end]),
            Cow::Owned(mut samples) => {
                samples.truncate(end);
                samples.drain(..start);
                Cow::Owned(samples)
            }
        };
        AudioSignal {
            samples,
            sample_rate: self.sample_rate,
        }
    }
}

/// Scales the degraded signal so its sound pressure level matches the
/// reference (`MiscAudio::ScaleToMatchSoundPressureLevel`).
pub fn scale_to_match_sound_pressure_level(reference: &AudioSignal, degraded: &mut AudioSignal) {
    let ref_spl = calc_sound_pressure_level(reference);
    let deg_spl = calc_sound_pressure_level(degraded);
    let scale_factor = 10f64.powf((ref_spl - deg_spl) / 20.0);
    for sample in degraded.samples_mut() {
        *sample *= scale_factor;
    }
}

fn calc_sound_pressure_level(signal: &AudioSignal) -> f64 {
    const SPL_REFERENCE_POINT: f64 = 0.00002;
    let sum: f64 = signal.samples.iter().map(|&s| s * s).sum();
    let sound_pressure = (sum / signal.samples.len() as f64).sqrt();
    20.0 * (sound_pressure / SPL_REFERENCE_POINT).log10()
}
