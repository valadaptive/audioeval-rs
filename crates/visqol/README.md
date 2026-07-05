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
`Visqol::speech(true)` for speech (expects 16 kHz input). Inputs are mono;
`AudioSignal::from_channels` downmixes by averaging, matching the C++.

## Conformance

`tests/conformance.rs` reproduces every non-lattice case of the upstream
conformance suite using the testdata from the `visqol` git submodule. Scores
match the C++ implementation's pinned values to ~13 significant digits — far
inside upstream's own 1e-4 conformance tolerance. The residual difference
exists because the alignment FFTs run in `f64` here, while the C++ uses
single-precision pffft.

## Differences from the C++ implementation

- The TensorFlow-Lite "lattice" speech-mode mapper (`--use_lattice_model`) is
  not ported; speech mode always uses the default exponential NSIM-to-MOS fit.
- The audio-mode SVR mapper reimplements libsvm nu-SVR RBF *inference* only,
  and embeds the default model (`models/libsvm_nu_svr_model.txt`, copied
  unmodified from upstream).
- Some C++ quirks are reproduced deliberately for conformance (grep the
  sources for "C++"): the Hilbert-envelope scaling built from the unpadded
  signal length, silence padding prepended rather than appended when slicing
  patch audio, and the unsigned-wrapping patch-count check that rejects
  single-patch signals.
