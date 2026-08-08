//! Head-to-head benchmark of the Rust (`zimtohrli`) and original C++
//! (`zimtohrli-cpp`) Zimtohrli implementations.
//!
//! Audio is decoded and resampled once at setup via `audio-io`; only the
//! metric runs inside the benchmark loops. Run with:
//!
//!   cargo bench -p benchmarks
//!
//! The C++ side is built by the `zimtohrli-cpp` build script with clang++
//! when available (override with CXX; e.g. add CXXFLAGS="-march=native" for
//! a best-case native build). Both `distance` variants rescale their input
//! spectrograms in place, so each iteration receives fresh clones whose cost
//! is excluded from the measurement via `iter_batched`.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use zimtohrli::{Spectrogram, Zimtohrli};
use zimtohrli_cpp::CppZimtohrli;

const SAMPLE_RATE: usize = 48_000;
const SECONDS: usize = 5;

/// Loads channel 0 of a file from the ViSQOL conformance subset, truncated to
/// `SECONDS` seconds.
fn load(relative: &str) -> Vec<f32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../visqol/testdata/conformance_testdata_subset")
        .join(relative);
    assert!(
        path.exists(),
        "{path:?} not found; is the visqol checkout present?"
    );
    let file = audio_io::read_audio_file(&path, SAMPLE_RATE).unwrap();
    let channel = file.channels.into_iter().next().unwrap();
    channel[..SAMPLE_RATE * SECONDS].to_vec()
}

fn clone_spec(spec: &Spectrogram) -> Spectrogram {
    Spectrogram {
        num_steps: spec.num_steps,
        num_dims: spec.num_dims,
        values: spec.values.clone(),
    }
}

fn bench_zimtohrli(c: &mut Criterion) {
    let signal_a = load("ravel48_stereo.wav");
    let signal_b = load("ravel48_stereo_128kbps_opus.wav");

    let rust = Zimtohrli::default();
    let cpp = CppZimtohrli::default();

    let mut group = c.benchmark_group("zimtohrli");
    group
        .sample_size(20)
        .measurement_time(Duration::from_secs(20));

    group.bench_function("analyze/rust", |b| {
        b.iter(|| rust.analyze(black_box(&signal_a)))
    });
    group.bench_function("analyze/cpp", |b| {
        b.iter(|| cpp.analyze(black_box(&signal_a)))
    });

    let rust_spec_a = rust.analyze(&signal_a);
    let rust_spec_b = rust.analyze(&signal_b);
    let cpp_spec_a = cpp.analyze(&signal_a);
    let cpp_spec_b = cpp.analyze(&signal_b);

    group.bench_function("distance_without_dtw/rust", |b| {
        b.iter_batched(
            || (clone_spec(&rust_spec_a), clone_spec(&rust_spec_b)),
            |(mut spec_a, mut spec_b)| rust.distance_without_dtw(&mut spec_a, &mut spec_b),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("distance_without_dtw/cpp", |b| {
        b.iter_batched(
            || (cpp_spec_a.clone(), cpp_spec_b.clone()),
            |(mut spec_a, mut spec_b)| cpp.distance_without_dtw(&mut spec_a, &mut spec_b),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("distance/rust", |b| {
        b.iter_batched(
            || (clone_spec(&rust_spec_a), clone_spec(&rust_spec_b)),
            |(mut spec_a, mut spec_b)| rust.distance(&mut spec_a, &mut spec_b),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("distance/cpp", |b| {
        b.iter_batched(
            || (cpp_spec_a.clone(), cpp_spec_b.clone()),
            |(mut spec_a, mut spec_b)| cpp.distance(&mut spec_a, &mut spec_b),
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_zimtohrli);
criterion_main!(benches);
