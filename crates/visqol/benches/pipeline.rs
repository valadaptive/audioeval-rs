//! End-to-end pipeline benchmark on the ravel conformance pair, with audio
//! decoding excluded. Compare runs against a saved baseline with
//! `cargo bench -p visqol -- --save-baseline <name>` /
//! `cargo bench -p visqol -- --baseline <name>`.

use std::path::PathBuf;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use visqol::{AudioSignal, Visqol};

fn load(relative: &str) -> AudioSignal {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../visqol/testdata/conformance_testdata_subset")
        .join(relative);
    assert!(
        path.exists(),
        "{path:?} not found; is the visqol submodule checked out?"
    );
    let file = audio_io::read_audio_file_native(&path).unwrap();
    AudioSignal::from_channels(&file.channels, file.src_sample_rate as u32)
}

fn bench_pipeline(c: &mut Criterion) {
    let reference = load("ravel48_stereo.wav");
    let degraded = load("ravel48_stereo_128kbps_opus.wav");

    let mut group = c.benchmark_group("pipeline");
    group
        .sample_size(20)
        .measurement_time(Duration::from_secs(20));

    group.bench_function("ravel48_opus128", |b| {
        let visqol = Visqol::audio();
        b.iter(|| visqol.run(&reference, &degraded).unwrap())
    });
    group.bench_function("ravel48_opus128_no_realign", |b| {
        let mut visqol = Visqol::audio();
        visqol.disable_realignment = true;
        b.iter(|| visqol.run(&reference, &degraded).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_pipeline);
criterion_main!(benches);
