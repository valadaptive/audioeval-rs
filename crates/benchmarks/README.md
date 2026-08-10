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
- The C++ baselines come from the `zimtohrli-cpp` and `visqol-cpp` crates;
  see below for how each is built.
- `distance`/`distance_without_dtw` rescale their input spectrograms in place
  (on both sides), so each iteration gets fresh clones whose cost is excluded
  from the measurement (`iter_batched`).
- Parity between the implementations is tested by `cargo test -p
  zimtohrli-cpp` and `cargo test -p visqol-cpp`. Add `-- --nocapture` to see
  the actual observed differences.

## The C++ baselines

### Zimtohrli

The `zimtohrli-cpp` crate compiles the single-header original
(`zimtohrli/cpp/zimt/zimtohrli.h`) plus a small pure-C ABI wrapper via `cc`.
No external setup is required.

### ViSQOL

ViSQOL is a full bazel project (abseil, protobuf, TensorFlow Lite, libsvm,
pffft), so the `visqol-cpp` crate links a prebuilt shared library instead of
compiling anything itself: the vendored checkout in `visqol/` carries a small
pure-C ABI wrapper (`visqol/src/visqol_capi.cc`) with a `//:visqol_capi`
target that bazel links into one self-contained `libvisqol_capi.so`. The
crate's build script runs `bazelisk build //:visqol_capi` (incremental; the
first build fetches and compiles all dependencies and takes a while), stages
the result in its Cargo `OUT_DIR`, and points the linker at it. Cargo supplies
that directory to the dynamic loader when it runs the tests and benchmarks.

Requirements: `bazelisk` on PATH (or set `BAZEL`), and the `visqol` checkout
on a branch containing the capi target.

Environment overrides (also read by the build script):

- `VISQOL_DIR`: path to the visqol checkout (default: `../../visqol`).
- `VISQOL_CAPI_LIB_DIR`: skip bazel and link a prebuilt `libvisqol_capi.so`
  from this directory — e.g. one built with a different compiler or bazel
  `--output_base`.
- `CC`/`CXX`: respected by bazel's toolchain autoconfiguration. Changing the
  compiler invalidates bazel's cache (full rebuild of all dependencies);
  build into a separate `--output_base` to keep the default cache intact.
  Use full paths (`CC=/usr/bin/clang`): on systems where `clang` resolves to
  a ccache symlink, bazel canonicalizes it to `/usr/bin/ccache` and the build
  fails confusingly. As of writing, clang builds the C++ ViSQOL ~15% faster
  and more consistently than gcc; see `visqol-cpp`'s parity test for
  correctness either way.

Note that the C++ ViSQOL is dramatically slower than the Rust port (~40x on
the benchmarked pair), so the `visqol` criterion group is sized for it (10
samples over a long measurement window).

## Fairness notes

- The C++ Zimtohrli is compiled with `-O3`, with `clang++` preferred when
  available (the upstream filter loop is tuned for clang's auto-vectorizer).
  Override with `CXX`, e.g. `CXX=g++`, and note the compiler when recording
  results; the build script prints the one it picked.
- The C++ ViSQOL is compiled by bazel with `-c opt` (the checkout's
  `.bazelrc` default) and the system default compiler; upstream bazel is the
  intended build path, and the README for the port notes that bazel builds
  are the reference configuration. Use `--config=nativeopt`
  (`-O3 -march=native`) and/or `CC=clang` for a best-case native build (see
  above for the cache caveat).
- Both sides build for the baseline target CPU by default. For a best-case
  native comparison: `CXXFLAGS="-march=native" RUSTFLAGS="-C
  target-cpu=native" cargo bench -p benchmarks` (the Rust `zimtohrli` crate
  dispatches SIMD at runtime via `fearless_simd`, so RUSTFLAGS matters less
  for it).


