//! Parity against the 2f-model scores distributed with SEBASS/SASSEC.
//!
//! The dataset is not redistributable with this crate.
//! Download the SASSEC dataset and reference outputs from https://www.audiolabs-erlangen.de/resources/2019-WASPAA-SEBASS/#NewModelParams,
//! then run:
//!
//! `cargo run -p two-f-model --release --example parity_test -- path/to/SASSEC path/to/ReferenceNumbers.csv`

use std::{env, fs, path::PathBuf};

use audioeval_2f::TwoFModel;

fn main() {
    let argv = env::args_os().collect::<Vec<_>>();
    let root = PathBuf::from(&argv[1]);
    let csv = fs::read_to_string(&argv[2]).unwrap();
    let model = TwoFModel::new();

    for (index, line) in csv.lines().skip(1).enumerate() {
        let mut columns = line.split(';');
        let reference_name = columns.next().unwrap().trim_start_matches("./");
        // For some reason, the anchor folder is renamed to anker_mix in the actual download
        let degraded_name = &columns
            .next()
            .unwrap()
            .trim_start_matches("./")
            .replace("anchor", "anker_mix");
        let expected: f64 = columns.next().unwrap().parse().unwrap();
        let reference = audio_io::read_audio_file_native(&root.join(reference_name)).unwrap();
        let degraded = audio_io::read_audio_file_native(&root.join(degraded_name)).unwrap();
        assert_eq!(reference.src_sample_rate, 48000);
        assert_eq!(degraded.src_sample_rate, 48000);
        let result = model.run(&reference.channels, &degraded.channels).unwrap();
        let error = (result.mushra_score - expected).abs();
        assert!(
            error < 0.001,
            "CSV row {} ({reference_name} vs {degraded_name}): got {}, expected {expected}",
            index + 2,
            result.mushra_score
        );
        eprintln!(
            "{}/182: {:.3} (expected {:.3})",
            index + 1,
            result.mushra_score,
            expected
        );
    }
}
