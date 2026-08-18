# audioeval-visqol

<!-- This README is generated via https://crates.io/crates/cargo-rdme. Edit the crate-level docs and regenerate this README via `cargo rdme`. -->

<!-- cargo-rdme start -->

A fast, pure-Rust port of [ViSQOL](https://github.com/google/visqol) v3.3, an objective, full-reference metric for perceived audio quality.

```rust
use audioeval_visqol::{AudioSignal, Visqol};

let reference = AudioSignal::new(ref_samples, 48000);
let degraded = AudioSignal::new(deg_samples, 48000);
let result = Visqol::audio().run(&reference, &degraded)?;
println!("MOS-LQO: {}", result.moslqo);
```

The entry points you're most likely to want are:
- [`Visqol::audio()`](https://docs.rs/audioeval-visqol/latest/audioeval_visqol/struct.Visqol.html#method.audio): for general audio (ViSQOL's default behavior). This expects 48kHz input.
- [`Visqol::speech_lattice()`](https://docs.rs/audioeval-visqol/latest/audioeval_visqol/struct.Visqol.html#method.speech_lattice): for speech (like ViSQOL with `use_speech_scoring` enabled). This expects 16kHz input.
- [`Visqol::speech_legacy()`](https://docs.rs/audioeval-visqol/latest/audioeval_visqol/struct.Visqol.html#method.speech_legacy): for speech, using the older non-lattice model (like ViSQOL with `use_speech_scoring` enabled, but `use_lattice_model` explicitly disabled). This also expects 16kHz input.

The inputs are expected to be mono signals. You may use `AudioSignal::from_channels` to downmix by averaging, as with the C++ version.

This crate will *not* resample your audio for you. You must use something like [rubato](https://crates.io/crates/rubato) to resample the input before passing it in.

### Mapping from the C++ configuration

Here's where each `VisqolConfig.options` field lands:

- `use_speech_scoring` + `use_lattice_model`: Choose a constructor. [`Visqol::audio()`](https://docs.rs/audioeval-visqol/latest/audioeval_visqol/struct.Visqol.html#method.audio) (audio, SVR), [`Visqol::speech_lattice()`](https://docs.rs/audioeval-visqol/latest/audioeval_visqol/struct.Visqol.html#method.speech_lattice) (speech, lattice), or [`Visqol::speech_legacy()`](https://docs.rs/audioeval-visqol/latest/audioeval_visqol/struct.Visqol.html#method.speech_legacy) (speech, exponential fit).
- `use_unscaled_speech_mos_mapping`: Passed as a `bool` argument to [`Visqol::speech_legacy()`](https://docs.rs/audioeval-visqol/latest/audioeval_visqol/struct.Visqol.html#method.speech_legacy).
- `svr_model_path`: [`Visqol::audio_with_model()`](https://docs.rs/audioeval-visqol/latest/audioeval_visqol/struct.Visqol.html#method.audio_with_model), which is passed an [`SvrModel`](https://docs.rs/audioeval-visqol/latest/audioeval_visqol/svr/struct.SvrModel.html). You may load one via [`SvrModel::from_text()`](https://docs.rs/audioeval-visqol/latest/audioeval_visqol/svr/struct.SvrModel.html#method.from_text).
- `search_window_radius`: [`Visqol::search_window_radius`](https://docs.rs/audioeval-visqol/latest/audioeval_visqol/struct.Visqol.html#structfield.search_window_radius).
- `allow_unsupported_sample_rates`: [`Visqol::allow_unsupported_sample_rates`](https://docs.rs/audioeval-visqol/latest/audioeval_visqol/struct.Visqol.html#structfield.allow_unsupported_sample_rates).
- `output_mos_score` + `detect_voice_activity`: Not supported. Even in the original C++ version, these are unimplemented no-ops.

### Conformance

This crate's results are compared against the upstream conformance test suite. We test against [conformance version 333](https://github.com/google/visqol/blob/38d0b01/src/include/conformance.h#L30).

The audio and non-lattice speech models match the original C++ implementation to ~13 significant digits. The lattice speech model's scores match to ~6 significant digits. This is well within the [upstream conformance tolerance of 1e-4](https://github.com/google/visqol/blob/38d0b01/python/visqol_lib_py_test.py#L20).

Results will likely not be *bit-identical* across machines due to potential rounding differences in libm functions, runtime FFT kernel decisions, and other factors outside this library's control. They are, however, expected to land well within the conformance tests' tolerance.

### Performance

This crate is significantly faster than the original C++ implementation of ViSQOL: around **40x faster** in audio mode, and **30-35x faster** in speech mode.

I am aware that the performance improvement seems unusually large, and if anybody . The original codebase doesn't seem to have undergone any optimization effort, whereas this codebase has.

<!-- cargo-rdme end -->
