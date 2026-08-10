# benchmarks

Criterion benchmarks for the audio eval crates, including optional comparisons to the original C++ implementations (when applicable).

I've centralized them in one place to enable the reuse of benchmark data.

You may enable the comparisons by enabling the relevant features:
- The `zimtohrli-cpp` feature for Zimtohrli; it's a single-header library, so it should be easy to compile.
- The `visqol-cpp` feature for ViSQOL; it uses Bazel and is likely substantially more painful to build.
