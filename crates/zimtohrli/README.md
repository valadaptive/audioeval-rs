# zimtohrli

<!-- This README is generated via https://crates.io/crates/cargo-rdme. Edit the crate-level docs and regenerate this README via `cargo rdme`. -->

<!-- cargo-rdme start -->

A pure-Rust port of Google's [Zimtohrli](https://github.com/google/zimtohrli), a perceptual audio evaluation metric.

```rust
use audioeval_zimtohrli::Zimtohrli;

// This should be 48kHz PCM audio in [-1, 1]. If your audio is not 48kHz, you must resample it. Zimtohrli is not amplitude-invariant.
let reference: &[f32];
let degraded: &[f32];


let zimt = Zimtohrli::default();

// Construct spectrograms from PCM audio.
let mut reference_spec = zimt.analyze(reference);
let mut degraded_spec = zimt.analyze(degraded);

// Compute the Zimtohrli distance (0.0 = identical, 1.0 = maximally different) between two spectrograms.
let distance = zimt.distance(&mut reference_spec, &mut degraded_spec);

// If you know the two signals are already time-aligned, you can skip the DTW (dynamic time warping) step for much faster, and potentially more accurate, results.
// This requires the reference and degraded audio to have the same length.
let distance = zimt.distance_without_dtw(&mut reference_spec, &mut degraded_spec);

// You can also map the distance to a "very approximate" mean opinion score (MOS).
let mos = Zimtohrli::mos_from_distance(distance);
println!("MOS: {mos}");
```

Audio can also be analyzed incrementally without retaining the whole PCM
signal in memory:

```rust
use audioeval_zimtohrli::{Spectrogram, Zimtohrli};

let chunks: &[&[f32]] = &[&[0.0; 1024], &[0.0; 1024]];
let zimt = Zimtohrli::default();
let mut analyzer = zimt.chunked_analyzer();
let mut frames = Vec::new();
for chunk in chunks {
    analyzer.process(chunk, &mut frames);
}
analyzer.flush(&mut frames);
let spectrogram = Spectrogram::from_frames(frames);
```

### Usage notes

Inputs are expected to be 48kHz mono signals. You must use something like [rubato](https://crates.io/crates/rubato) to resample the input before passing it in.

Zimtohrli does not contain any binaural metrics. The Zimtohrli CLI handles stereo/multichannel audio by evaluating each channel's distance separately, and computing the root-mean-squared distance:

```rust

let reference: &[&[f32]];
let degraded: &[&[f32]];


let zimt = Zimtohrli::default();
let mut sum_of_squares = 0.0;

for (ref_channel, deg_channel) in reference.iter().zip(degraded.iter()) {
    let mut reference_spec = zimt.analyze(ref_channel);
    let mut degraded_spec = zimt.analyze(deg_channel);

    let distance = zimt.distance(&mut reference_spec, &mut degraded_spec);
    sum_of_squares += distance * distance;
}

let rms_distance = (sum_of_squares / reference.len() as f32).sqrt();
let mos = Zimtohrli::mos_from_distance(rms_distance);
println!("MOS: {mos}");
```

### Conformance and exactness

This crate matches the results of the original C++ `zimtohrli` repo (as of [this commit](https://github.com/google/zimtohrli/tree/aad0469673a4aec594d62b82e2b5f95e85b76362)) to 1e-5.

Matching the C++ original *exactly* is impossible, since the C++ code itself is nondeterministic: it is compiled with "fast math" flags, and uses libm functions which may return different results across different platforms.

However, this crate *is* entirely deterministic, and its own results should be bit-identical across all platforms. Rather than using the system libm, it uses the Rust `libm` crate for math. This crate does perform runtime CPU feature detection (see below), but intentionally foregoes non-deterministic optimizations dependent on platform features.

### Performance

Some rough benchmarks on my Ryzen 7 7700X put this crate around 30-35% faster than the original C++ version on full analysis (spectrogram creation + distance calculation, not counting I/O or resampling). This goes for analysis both with and without the DTW step.

The original C++ code does not perform any runtime CPU feature detection, whereas this crate does. When benchmarking, the original code was compiled with `-march=x86-64-v3`. For some reason, `x86-64-v4` was slower.

If DTW is required, but conformance with the original C++ version is *not*, you may use the [`Zimtohrli::dtw_band_radius`](https://docs.rs/audioeval-zimtohrli/latest/audioeval_zimtohrli/struct.Zimtohrli.html#structfield.dtw_band_radius) option to reduce the search radius considered during the DTW alignment step. This is an extension of the API exclusive to this crate.

<!-- cargo-rdme end -->
