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
use visqol_cpp::{CppVisqol, default_svr_model_path};

const SAMPLE_RATE: u32 = 48_000;
const SECONDS: usize = 5;

/// Loads a file from the ViSQOL conformance subset, downmixed to mono f64 and
/// truncated to `SECONDS` seconds.
fn load_mono(relative: &str) -> Vec<f64> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../visqol/testdata/conformance_testdata_subset")
        .join(relative);
    assert!(
        path.exists(),
        "{path:?} not found; is the visqol checkout present?"
    );
    let file = audio_io::read_audio_file(&path, SAMPLE_RATE as usize).unwrap();
    AudioSignal::from_channels(&file.channels, SAMPLE_RATE).samples
        [..SECONDS * SAMPLE_RATE as usize]
        .to_vec()
}

fn bench_visqol(c: &mut Criterion) {
    let reference = load_mono("ravel48_stereo.wav");
    let degraded = load_mono("ravel48_stereo_128kbps_opus.wav");

    let rust = Visqol::audio();
    let mut cpp = CppVisqol::audio(&default_svr_model_path()).unwrap();

    let rust_ref = AudioSignal::new(reference.clone(), SAMPLE_RATE);
    let rust_deg = AudioSignal::new(degraded.clone(), SAMPLE_RATE);

    let mut group = c.benchmark_group("visqol");
    // Sized for the C++ side (~2s per iteration); the Rust side simply gets
    // fewer samples than criterion's default.
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(4))
        .measurement_time(Duration::from_secs(25));

    group.bench_function("run/rust", |b| {
        b.iter(|| rust.run(black_box(&rust_ref), black_box(&rust_deg)))
    });
    group.bench_function("run/cpp", |b| {
        b.iter(|| cpp.run(black_box(&reference), black_box(&degraded), SAMPLE_RATE))
    });

    group.finish();
}

criterion_group!(benches, bench_visqol);
criterion_main!(benches);
