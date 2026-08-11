use zimtohrli::{NUM_CHANNELS, Spectrogram, Zimtohrli};

fn signal(num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            0.4 * (440.0 * std::f32::consts::TAU * t).sin()
                + 0.2 * (1_733.0 * std::f32::consts::TAU * t).sin()
        })
        .collect()
}

fn analyze_in_chunks(z: &Zimtohrli, signal: &[f32], chunk_size: usize) -> Spectrogram {
    let mut analyzer = z.chunked_analyzer();
    let mut frames = Vec::new();
    for chunk in signal.chunks(chunk_size) {
        analyzer.process(chunk, &mut frames);
    }
    analyzer.flush(&mut frames);
    assert_eq!(frames.len(), analyzer.num_steps() * NUM_CHANNELS);
    Spectrogram::from_frames(frames)
}

#[test]
fn arbitrary_chunks_match_batch_analysis_exactly() {
    let z = Zimtohrli::default();
    let signal = signal(5_000);
    let expected = z.analyze(&signal);

    for chunk_size in [1, 7, 31, 32, 100, 564, 4_096, 5_000] {
        let actual = analyze_in_chunks(&z, &signal, chunk_size);
        assert_eq!(actual.num_steps(), expected.num_steps());
        assert_eq!(
            actual.values(),
            expected.values(),
            "chunk size {chunk_size}"
        );
    }
}

#[test]
fn streaming_handles_frame_and_filter_boundaries() {
    let z = Zimtohrli::default();
    for len in [0, 1, 31, 32, 563, 564, 565, 1_128] {
        let signal = signal(len);
        let actual = analyze_in_chunks(&z, &signal, 7);
        let expected = z.analyze(&signal);
        assert_eq!(actual.num_steps(), z.spectrogram_steps(len), "length {len}");
        assert_eq!(actual.values(), expected.values(), "length {len}");
    }
}

#[test]
fn flush_is_idempotent() {
    let z = Zimtohrli::default();
    let mut analyzer = z.chunked_analyzer();
    let mut frames = Vec::new();
    analyzer.process(&signal(1_000), &mut frames);
    analyzer.flush(&mut frames);
    let first_len = frames.len();
    let first_steps = analyzer.num_steps();

    analyzer.flush(&mut frames);
    assert_eq!(frames.len(), first_len);
    assert_eq!(analyzer.num_steps(), first_steps);
}
