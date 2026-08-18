//! Benchmark of the Rust (`zimtohrli`) and optionally the original C++
//! (`zimtohrli-cpp`) Zimtohrli implementations.
//!
//! Audio is decoded and resampled once at setup via `audio-io`; only the metric
//! runs inside the benchmark loops. Run with:
//!
//!   cargo bench -p benchmarks
//!
//! The C++ side is built by the `zimtohrli-cpp` build script with clang++ when
//! available (override with CXX; e.g. add CXXFLAGS="-march=native" for a
//! best-case native build). Both `distance` variants rescale their input
//! spectrograms in place, so each iteration receives fresh clones whose cost is
//! excluded from the measurement via `iter_batched`.

use std::hint::black_box;
use std::time::Duration;

use benchmarks::load_corpus_sample;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use audioeval_zimtohrli::Zimtohrli;

fn bench_zimtohrli(c: &mut Criterion) {
    let reference = &load_corpus_sample("ravel48_stereo.wav", 48000, Some(5)).channels[0];
    let degraded =
        &load_corpus_sample("ravel48_stereo_128kbps_opus.wav", 48000, Some(5)).channels[0];

    let rust = Zimtohrli::default();

    let mut group = c.benchmark_group("zimtohrli");
    group
        .sample_size(20)
        .measurement_time(Duration::from_secs(20));

    group.bench_function("analyze/rust", |b| {
        b.iter(|| rust.analyze(black_box(&reference)))
    });

    let rust_spec_ref = rust.analyze(&reference);
    let rust_spec_deg = rust.analyze(&degraded);

    group.bench_function("distance_without_dtw/rust", |b| {
        b.iter_batched(
            || (rust_spec_ref.clone(), rust_spec_deg.clone()),
            |(mut spec_a, mut spec_b)| rust.distance_without_dtw(&mut spec_a, &mut spec_b),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("distance/rust", |b| {
        b.iter_batched(
            || (rust_spec_ref.clone(), rust_spec_deg.clone()),
            |(mut spec_a, mut spec_b)| rust.distance(&mut spec_a, &mut spec_b),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("e2e/rust", |b| {
        b.iter(|| {
            let mut spec_a = rust.analyze(&reference);
            let mut spec_b = rust.analyze(&degraded);
            rust.distance(&mut spec_a, &mut spec_b)
        })
    });
    group.bench_function("e2e_without_dtw/rust", |b| {
        b.iter(|| {
            let mut spec_a = rust.analyze(&reference);
            let mut spec_b = rust.analyze(&degraded);
            rust.distance_without_dtw(&mut spec_a, &mut spec_b)
        })
    });

    #[cfg(feature = "zimtohrli-cpp")]
    {
        let cpp = zimtohrli_cpp::CppZimtohrli::default();
        let cpp_spec_ref = cpp.analyze(&reference);
        let cpp_spec_deg = cpp.analyze(&degraded);
        group.bench_function("analyze/cpp", |b| {
            b.iter(|| cpp.analyze(black_box(&reference)))
        });
        group.bench_function("distance_without_dtw/cpp", |b| {
            b.iter_batched(
                || (cpp_spec_ref.clone(), cpp_spec_deg.clone()),
                |(mut spec_a, mut spec_b)| cpp.distance_without_dtw(&mut spec_a, &mut spec_b),
                BatchSize::SmallInput,
            )
        });
        group.bench_function("distance/cpp", |b| {
            b.iter_batched(
                || (cpp_spec_ref.clone(), cpp_spec_deg.clone()),
                |(mut spec_a, mut spec_b)| cpp.distance(&mut spec_a, &mut spec_b),
                BatchSize::SmallInput,
            )
        });
        group.bench_function("e2e/cpp", |b| {
            b.iter(|| {
                let mut spec_a = cpp.analyze(&reference);
                let mut spec_b = cpp.analyze(&degraded);
                cpp.distance(&mut spec_a, &mut spec_b)
            })
        });
        group.bench_function("e2e_without_dtw/cpp", |b| {
            b.iter(|| {
                let mut spec_a = cpp.analyze(&reference);
                let mut spec_b = cpp.analyze(&degraded);
                cpp.distance_without_dtw(&mut spec_a, &mut spec_b)
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_zimtohrli);
criterion_main!(benches);
