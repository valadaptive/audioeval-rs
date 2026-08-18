# audioeval-rs

Fast, pure-Rust implementations of perceptual audio quality metrics.

The goal of this repo is to provide a way to quickly and easily compute many different audio quality metrics (for evaluating codecs, audio source separation algorithms, etc), without wrangling multiple languages and build systems.

## The metrics

I've currently implemented three different audio quality metrics:

- **[Zimtohrli](github.com/google/zimtohrli)**: Google's newest perceptual metric. It seems to correlate quite well with real mean opinion scores.
- **[ViSQOL](https://github.com/google/visqol) v3.3**: A commonly-used metric for both speech and audio, also by Google. Their original C++ implementation appears unoptimized, requires Bazel to build, and depends on an old version of the TFLite runtime; the version in this repo is pure-Rust and ~30-40x faster (yes, really).
- **[The MMS 2f-model](https://www.audiolabs-erlangen.de/resources/2019-WASPAA-SEBASS/#NewModelParams)**: A metric based on PEAQ, and tuned for evaluating audio source separation quality. It seems to perform well overall at evaluating audio degradation.

## Other crates (very WIP)

There's also a crate (`audio-io`) which provides a simplified audio input API, which loads and automatically resamples audio files. Be careful when using it--in particular, [some codecs and containers may not be properly time-aligned](https://github.com/pdeljanov/Symphonia/issues/544).

The CLIs (`audioeval-cli`, `visqol-compare`, `zimtohrli-compare`) are likewise still works in progress.

Finally, the C++ binding crates (`visqol-cpp` and `zimtohrli-cpp`) are used only for benchmarking purposes, and aren't intended to provide general-purpose bindings to the original libraries.

## AI usage

As a general rule: all crates *intended for public use* have (at the very least) gone through significant human review and revision. Many auxiliary crates (e.g. benchmarking and other internal infrastructure) have not.

The `zimtohrli` crate was hand-ported by myself, with AI used to review the code and perform additional optimization.

The `visqol` and `two-f-model` crates were ported from the original codebases (and optimized) via AI, with a lot of manual code cleanups done along the way.

Many optimizations involve custom math kernels for exponentiation (this turned out to be a very common operation). AI was used to write the [Sollya scripts](https://www.sollya.org/) and implement those kernels.

`audioeval-cli` was written by myself. The other CLIs, `audio-io`, and the C++ binding crates were written by AI.
