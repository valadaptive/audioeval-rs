//! Benchmark of the Rust (`visqol`) and optiosnally the original C++
//! (`visqol-cpp`) ViSQOL implementations, in the default 48kHz audio mode.
//!
//! Audio is decoded and downmixed to mono f64 once at setup via `audio-io`;
//! only the metric runs inside the benchmark loops. Run with:
//!
//!   cargo bench -p benchmarks --bench visqol
//!
//! The C++ side is built by bazel from the vendored checkout (see the
//! `visqol-cpp` crate); it is far slower than the Rust port, so the whole group
//! is sized for it (10 samples, long measurement window).

use std::hint::black_box;
use std::time::Duration;

use audioeval_visqol::{AudioSignal, Visqol};
use benchmarks::load_corpus_sample;
use criterion::{Criterion, criterion_group, criterion_main};

const SECONDS: usize = 5;

/// Loads a file from the ViSQOL test data, downmixed to mono f64 and
/// truncated to `SECONDS` seconds.
fn load_mono(relative: &str, sample_rate: usize) -> AudioSignal {
    let file = load_corpus_sample(relative, sample_rate, Some(SECONDS));
    AudioSignal::from_channels(&file.channels, sample_rate as u32)
}

fn bench_visqol(c: &mut Criterion) {
    let reference_audio = load_mono("ravel48_stereo.wav", 48000);
    let degraded_audio = load_mono("ravel48_stereo_128kbps_opus.wav", 48000);

    let reference_speech = load_mono("CA01_01.wav", 16000);
    let degraded_speech = load_mono("transcoded_CA01_01.wav", 16000);

    let rust_audio = Visqol::audio();
    let rust_speech_lattice = Visqol::speech_lattice();
    let rust_speech_legacy = Visqol::speech_legacy(false);

    let mut group = c.benchmark_group("visqol");
    // Sized for the C++ side (~2s per iteration); the Rust side simply gets
    // fewer samples than criterion's default.
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(4))
        .measurement_time(Duration::from_secs(25));

    group.bench_function("audio/rust", |b| {
        b.iter(|| rust_audio.run(black_box(&reference_audio), black_box(&degraded_audio)))
    });
    group.bench_function("speech_lattice/rust", |b| {
        b.iter(|| {
            rust_speech_lattice.run(black_box(&reference_speech), black_box(&degraded_speech))
        })
    });
    group.bench_function("speech_legacy/rust", |b| {
        b.iter(|| rust_speech_legacy.run(black_box(&reference_speech), black_box(&degraded_speech)))
    });

    #[cfg(feature = "visqol-cpp")]
    {
        use visqol_cpp::{CppVisqol, default_lattice_model_path, default_svr_model_path};
        let mut cpp_audio = CppVisqol::audio(&default_svr_model_path()).unwrap();
        let mut cpp_speech_lattice =
            CppVisqol::speech_lattice(&&default_lattice_model_path()).unwrap();
        let mut cpp_speech_legacy = CppVisqol::speech_legacy(false).unwrap();
        group.bench_function("audio/cpp", |b| {
            b.iter(|| {
                cpp_audio.run(
                    black_box(&reference_audio.samples),
                    black_box(&degraded_audio.samples),
                    reference_audio.sample_rate,
                )
            })
        });
        group.bench_function("speech_lattice/cpp", |b| {
            b.iter(|| {
                cpp_speech_lattice.run(
                    black_box(&reference_speech.samples),
                    black_box(&degraded_speech.samples),
                    reference_speech.sample_rate,
                )
            })
        });
        group.bench_function("speech_legacy/cpp", |b| {
            b.iter(|| {
                cpp_speech_legacy.run(
                    black_box(&reference_speech.samples),
                    black_box(&degraded_speech.samples),
                    reference_speech.sample_rate,
                )
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_visqol);
criterion_main!(benches);
