use std::path::PathBuf;

pub fn load_corpus_sample(
    relative: &str,
    sample_rate: usize,
    duration: Option<usize>,
) -> audio_io::AudioFile {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_corpus")
        .join(relative);
    assert!(
        path.exists(),
        "{path:?} not found; is the visqol checkout present?"
    );
    let mut file = audio_io::read_audio_file(&path, sample_rate).unwrap();
    if let Some(duration) = duration {
        let duration_samples = (sample_rate * duration).min(file.channels[0].len());
        for ch in &mut file.channels {
            ch.truncate(duration_samples);
        }
    }
    file
}
