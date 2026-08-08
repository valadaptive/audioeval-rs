//! Head-to-head benchmark of the Rust (`visqol`) and original C++
//! (`visqol-cpp`) ViSQOL implementations, in the default 48kHz audio mode.
//!
//! Audio is decoded and downmixed to mono f64 once at setup via `audio-io`;
//! only the metric runs inside the benchmark loops. Run with:
//!
//!   cargo bench -p benchmarks --bench visqol
//!
//! The C++ side is built by bazel from the vendored checkout (see the
//! `visqol-cpp` crate); it is far slower than the Rust port, so the whole
//! group is sized for it (10 samples, long measurement window).

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use visqol::{AudioSignal, Visqol};
use visqol_cpp::{CppVisqol, default_lattice_model_path, default_svr_model_path};

const SECONDS: usize = 5;

/// Loads a file from the ViSQOL test data, downmixed to mono f64 and
/// truncated to `SECONDS` seconds.
fn load_mono(relative: &str, sample_rate: u32) -> AudioSignal {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../visqol/testdata")
        .join(relative);
    assert!(
        path.exists(),
        "{path:?} not found; is the visqol checkout present?"
    );
    let file = audio_io::read_audio_file(&path, sample_rate as usize).unwrap();
    let duration = (sample_rate as usize * SECONDS).min(file.channels[0].len());
    AudioSignal::from_channels(file.channels.iter().map(|ch| &ch[..duration]), sample_rate)
}

fn bench_visqol(c: &mut Criterion) {
    let reference_audio = load_mono("conformance_testdata_subset/ravel48_stereo.wav", 48000);
    let degraded_audio = load_mono(
        "conformance_testdata_subset/ravel48_stereo_128kbps_opus.wav",
        48000,
    );

    let reference_speech = load_mono("clean_speech/CA01_01.wav", 16000);
    let degraded_speech = load_mono("clean_speech/transcoded_CA01_01.wav", 16000);

    let rust_audio = Visqol::audio();
    let mut cpp_audio = CppVisqol::audio(&default_svr_model_path()).unwrap();

    let rust_speech_lattice = Visqol::speech_lattice();
    let mut cpp_speech_lattice = CppVisqol::speech_lattice(&&default_lattice_model_path()).unwrap();

    let rust_speech_legacy = Visqol::speech_legacy(false);
    let mut cpp_speech_legacy = CppVisqol::speech_legacy(false).unwrap();

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
    group.bench_function("audio/cpp", |b| {
        b.iter(|| {
            cpp_audio.run(
                black_box(&reference_audio.samples),
                black_box(&degraded_audio.samples),
                reference_audio.sample_rate,
            )
        })
    });

    group.bench_function("speech_lattice/rust", |b| {
        b.iter(|| {
            rust_speech_lattice.run(black_box(&reference_speech), black_box(&degraded_speech))
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

    group.bench_function("speech_legacy/rust", |b| {
        b.iter(|| rust_speech_legacy.run(black_box(&reference_speech), black_box(&degraded_speech)))
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

    group.finish();
}

criterion_group!(benches, bench_visqol);
criterion_main!(benches);
