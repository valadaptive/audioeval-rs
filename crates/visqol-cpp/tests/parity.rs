//! Parity checks between the Rust ViSQOL port and the C++ original, in all
//! three modes: audio (SVR model), speech with the deep lattice model, and
//! legacy speech (exponential NSIM-to-MOS mapping).
//!
//! The Rust implementation is a reimplementation (different floating-point
//! operation order, SIMD), so bitwise equality is not expected — but the
//! observed differences are far below the 1e-6 asserted for audio and legacy
//! speech mode (and upstream's own conformance tolerance of 1e-4).
//!
//! The one exception is the lattice model's moslqo: the pipeline's vnsim
//! agrees to ~1e-14, but the final NSIM-to-MOS mapping runs through the
//! TFLite interpreter's f32 kernels on the C++ side and a from-scratch
//! evaluator over the extracted parameters on the Rust side, which leaves an
//! observed difference of ~3e-5. That moslqo alone gets upstream's 1e-4
//! conformance tolerance.
//!
//! Run with output to see the observed differences:
//!
//!   cargo test -p visqol-cpp -- --nocapture

use std::path::PathBuf;

use audioeval_visqol::{AudioSignal, Visqol};
use visqol_cpp::{CppVisqol, default_lattice_model_path, default_svr_model_path};

/// Loads a file from the upstream testdata directory, downmixed to mono f64
/// exactly once, so both implementations see identical input. `rate`
/// resamples; `None` keeps the file's native rate.
fn load_mono(relative: &str, rate: Option<u32>) -> (AudioSignal<'static>, u32) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../visqol/testdata")
        .join(relative);
    assert!(
        path.exists(),
        "{path:?} not found; is the visqol checkout present?"
    );
    let file = match rate {
        Some(rate) => audio_io::read_audio_file(&path, rate as usize).unwrap(),
        None => audio_io::read_audio_file_native(&path).unwrap(),
    };
    let rate = rate.unwrap_or(file.src_sample_rate as u32);
    (AudioSignal::from_channels(&file.channels, rate), rate)
}

/// Runs both implementations over the same signals and asserts their results
/// agree: vnsim to 1e-6, moslqo to `moslqo_tolerance`.
fn assert_parity(
    mode: &str,
    rust: &Visqol,
    cpp: &mut CppVisqol,
    reference: &AudioSignal,
    degraded: &AudioSignal,
    sample_rate: u32,
    moslqo_tolerance: f64,
) {
    let rust_result = rust.run(reference, degraded).unwrap();
    let cpp_result = cpp
        .run(&reference.samples, &degraded.samples, sample_rate)
        .unwrap();

    println!(
        "[{mode}] moslqo: rust = {}, cpp = {}",
        rust_result.moslqo, cpp_result.moslqo
    );
    println!(
        "[{mode}] vnsim:  rust = {}, cpp = {}",
        rust_result.vnsim, cpp_result.vnsim
    );

    assert!(
        (rust_result.moslqo - cpp_result.moslqo).abs() < moslqo_tolerance,
        "[{mode}] moslqo mismatch: rust = {}, cpp = {}",
        rust_result.moslqo,
        cpp_result.moslqo
    );
    assert!(
        (rust_result.vnsim - cpp_result.vnsim).abs() < 1e-6,
        "[{mode}] vnsim mismatch: rust = {}, cpp = {}",
        rust_result.vnsim,
        cpp_result.vnsim
    );
}

#[test]
fn audio_mode_parity() {
    let (reference, rate) = load_mono(
        "conformance_testdata_subset/ravel48_stereo.wav",
        Some(48_000),
    );
    let (degraded, _) = load_mono(
        "conformance_testdata_subset/ravel48_stereo_128kbps_opus.wav",
        Some(48_000),
    );

    assert_parity(
        "audio",
        &Visqol::audio(),
        &mut CppVisqol::audio(&default_svr_model_path()).unwrap(),
        &reference,
        &degraded,
        rate,
        1e-6,
    );
}

/// Speech mode over the upstream conformance speech pair. The files are 48kHz
/// and neither implementation resamples (speech mode's bands are pinned to
/// fixed frequencies), matching upstream's own conformance tests.
#[test]
fn speech_lattice_mode_parity() {
    let (reference, rate) = load_mono("clean_speech/CA01_01.wav", None);
    let (degraded, _) = load_mono("clean_speech/transcoded_CA01_01.wav", None);

    assert_parity(
        "speech_lattice",
        &Visqol::speech_lattice(),
        &mut CppVisqol::speech_lattice(&default_lattice_model_path()).unwrap(),
        &reference,
        &degraded,
        rate,
        1e-4, // TFLite f32 kernels vs. the Rust extracted-parameter evaluator
    );
}

#[test]
fn speech_legacy_mode_parity() {
    let (reference, rate) = load_mono("clean_speech/CA01_01.wav", None);
    let (degraded, _) = load_mono("clean_speech/transcoded_CA01_01.wav", None);

    assert_parity(
        "speech_legacy",
        &Visqol::speech_legacy(false),
        &mut CppVisqol::speech_legacy(false).unwrap(),
        &reference,
        &degraded,
        rate,
        1e-6,
    );
}
