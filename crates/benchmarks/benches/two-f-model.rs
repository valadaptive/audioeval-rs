use std::hint::black_box;

use audioeval_2f::TwoFModel;
use benchmarks::load_corpus_sample;
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_two_f(c: &mut Criterion) {
    let model = TwoFModel::new();
    let reference = load_corpus_sample("ravel48_stereo.wav", 48000, Some(5));
    let degraded = load_corpus_sample("ravel48_stereo_128kbps_opus.wav", 48000, Some(5));

    c.bench_function("two_f_eval", |b| {
        b.iter(|| {
            let _ = black_box(model.run(&reference.channels, &degraded.channels));
        });
    });
}

criterion_group!(benches, bench_two_f);
criterion_main!(benches);
