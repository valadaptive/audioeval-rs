# visqol

A pure-Rust port of [ViSQOL](https://github.com/google/visqol) v3 (conformance
version 333), an objective, full-reference metric for perceived audio quality,
with no Bazel, protobuf, or TensorFlow dependencies.

```rust
use visqol::{AudioSignal, Visqol};

let reference = AudioSignal::new(ref_samples, 48000);
let degraded = AudioSignal::new(deg_samples, 48000);
let result = Visqol::audio().run(&reference, &degraded)?;
println!("MOS-LQO: {}", result.moslqo);
```

Use `Visqol::audio()` for general audio (expects 48 kHz input) and
`Visqol::speech_lattice()` for speech (expects 16 kHz input) — the deep
lattice network mapping, upstream's speech default. `Visqol::speech(true)`
selects the older exponential NSIM-to-MOS fit (upstream's
`--use_lattice_model=false`). Inputs are mono; `AudioSignal::from_channels`
downmixes by averaging, matching the C++.

## Conformance

`tests/conformance.rs` reproduces the entire upstream conformance suite using
the testdata from the `visqol` git submodule. SVR- and exponential-mapped
scores match the C++ implementation's pinned values to ~13 significant digits;
lattice-mapped scores match to ~6 significant digits (the C++ runs the lattice
model in f32 under XNNPack, whose SIMD summation order can't be reproduced
exactly). Both are far inside upstream's own 1e-4 conformance tolerance.

## Differences from the C++ implementation

- The audio-mode SVR mapper reimplements libsvm nu-SVR RBF *inference* only,
  and embeds the default model (`models/libsvm_nu_svr_model.txt`, copied
  unmodified from upstream).
- The speech-mode lattice mapper evaluates the calibrated-lattice-ensemble
  directly from parameters extracted out of the upstream TFLite flatbuffer
  (`models/lattice_speech_model.bin`, see `models/extract_lattice_model.py`),
  instead of depending on a TFLite runtime. Training-time quirks are
  reproduced, including the missing-value sentinels that fire on an fvnsim of
  exactly 0.7 or an fvnsim10 of exactly 0.5 (as f32).
- Some C++ quirks are reproduced deliberately for conformance (grep the
  sources for "C++"): the Hilbert-envelope scaling built from the unpadded
  signal length, silence padding prepended rather than appended when slicing
  patch audio, and the unsigned-wrapping patch-count check that rejects
  single-patch signals.
