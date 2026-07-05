# visqol

A pure-Rust port of [ViSQOL](https://github.com/google/visqol) v3, an objective, full-reference metric for perceived audio quality.

```rust
use visqol::{AudioSignal, Visqol};

let reference = AudioSignal::new(ref_samples, 48000);
let degraded = AudioSignal::new(deg_samples, 48000);
let result = Visqol::audio().run(&reference, &degraded)?;
println!("MOS-LQO: {}", result.moslqo);
```

The entry points you're most likely to want are:
- `Visqol::audio()`: for general audio (ViSQOL's default behavior). This expects 48kHz input.
- `Visqol::speech_lattice()`: for speech (like ViSQOL with `use_speech_scoring` enabled). This expects 16kHz input.
- `Visqol::speech_legacy()`: for speech, using the older non-lattice model (like ViSQOL with `use_speech_scoring` enabled, but `use_lattice_model` explicitly disabled). This also expects 16kHz input.

The inputs are expected to be mono signals. You may use `AudioSignal::from_channels` to downmix by averaging, as with the C++ version.

This crate will *not* resample your audio for you. You must use something like [rubato](https://crates.io/crates/rubato) to resample the input before passing it in.

## Conformance

This crate's results are compared against the upstream conformance test suite. We test against [conformance version 333](https://github.com/google/visqol/blob/38d0b01/src/include/conformance.h#L30).

The audio and non-lattice speech models match the original C++ implementation to ~13 significant digits. The lattice speech model's scores match to ~6 significant digits. This is well within the [upstream conformance tolerance of 1e-4](https://github.com/google/visqol/blob/38d0b01/python/visqol_lib_py_test.py#L20).

Results will likely not be *bit-identical* across machines due to potential rounding differences in libm functions, runtime FFT kernel decisions, and other factors outside this library's control. They are, however, expected to land well within the conformance tests' tolerance.
