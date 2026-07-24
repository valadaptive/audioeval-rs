use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use zimtohrli::Zimtohrli;

const SAMPLE_RATE: usize = 48_000;

fn test_signal(seconds: usize, offset: f32) -> Vec<f32> {
    (0..SAMPLE_RATE * seconds)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            0.4 * (440.0 * std::f32::consts::TAU * t + offset).sin()
                + 0.25 * (1733.0 * std::f32::consts::TAU * t).sin()
                + 0.1 * (9973.0 * std::f32::consts::TAU * t + 0.3 * offset).sin()
        })
        .collect()
}

fn bench_pipeline(c: &mut Criterion) {
    let signal_a = test_signal(5, 0.0);
    let signal_b = test_signal(5, 0.02);
    let z = Zimtohrli::default();

    let mut group = c.benchmark_group("zimtohrli");
    group
        .sample_size(20)
        .measurement_time(Duration::from_secs(20));

    group.bench_function("analyze_5s", |b| b.iter(|| z.analyze(black_box(&signal_a))));
    group.bench_function("pair_without_dtw_5s", |b| {
        b.iter(|| {
            let mut spec_a = z.analyze(black_box(&signal_a));
            let mut spec_b = z.analyze(black_box(&signal_b));
            z.distance_without_dtw(&mut spec_a, &mut spec_b)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_pipeline);
criterion_main!(benches);
