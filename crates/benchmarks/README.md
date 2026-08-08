# benchmarks

Head-to-head criterion benchmarks of the Rust metric implementations against
the C++ originals.

## Running

```sh
cargo bench -p benchmarks
```

Useful variations:

```sh
cargo bench -p benchmarks -- --test                  # run each benchmark once (just to ensure they work)
cargo bench -p benchmarks -- analyze                 # filter by name
cargo bench -p benchmarks -- --save-baseline main    # compare against a baseline later
cargo bench -p benchmarks -- --baseline main
```

## How it works

- Audio is decoded/resampled once at setup via `audio-io` (the ViSQOL
  conformance subset under `visqol/testdata/`), so codec and resampling cost
  stays out of the measured loops. Benchmarked signals are mono 48 kHz slices.
- The C++ baseline comes from the `zimtohrli-cpp` crate, which compiles the
  single-header original (`zimtohrli/cpp/zimt/zimtohrli.h`) plus a small
  pure-C ABI wrapper via `cc`.
- `distance`/`distance_without_dtw` rescale their input spectrograms in place
  (on both sides), so each iteration gets fresh clones whose cost is excluded
  from the measurement (`iter_batched`).
- Parity between the two implementations is tested by
  `cargo test -p zimtohrli-cpp`. You can run `cargo test -p zimtohrli-cpp --
  --nocapture` to see the actual observed differences.

## Fairness notes

- The C++ side is compiled with `-O3`, with `clang++` preferred when available
  (the upstream filter loop is tuned for clang's auto-vectorizer). Override
  with `CXX`, e.g. `CXX=g++`, and note the compiler when recording results;
  the build script prints the one it picked.
- Both sides build for the baseline target CPU by default. For a best-case
  native comparison: `CXXFLAGS="-march=native" RUSTFLAGS="-C
  target-cpu=native" cargo bench -p benchmarks` (the Rust `zimtohrli` crate
  dispatches SIMD at runtime via `fearless_simd`, so RUSTFLAGS matters less
  for it).

## Extending to ViSQOL

We'll need to add a separate `visqol-cpp` sys crate (building the bazel output
or a prebuilt library) and a `benches/visqol.rs` target next to
`benches/zimtohrli.rs`. Bazel is a bit annoying; we may want to put each C++
library behind its own feature flag.
