//! Conformance tests against the scores pinned by the C++ implementation in
//! `tests/conformance_test.cc` / `src/include/conformance.h` (version 333).
//!
//! Requires the `visqol` git submodule (for its testdata) to be checked out.
//! Run with `--release` or the default `opt-level` override; the pipeline is
//! slow without optimization.

use std::path::PathBuf;

use visqol::{AudioSignal, Visqol};

/// C++ conformance tolerance.
const TOLERANCE: f64 = 0.0001;

fn testdata(relative: &str) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../visqol/testdata")
        .join(relative);
    assert!(
        path.exists(),
        "{path:?} not found; is the visqol submodule checked out?"
    );
    path
}

fn load(relative: &str) -> AudioSignal {
    let path = testdata(relative);
    let file = audio_io::read_audio_file_native(&path).unwrap();
    AudioSignal::from_channels(&file.channels, file.src_sample_rate as u32)
}

fn assert_moslqo(visqol: &Visqol, reference: &str, degraded: &str, expected: f64) {
    let result = visqol.run(&load(reference), &load(degraded)).unwrap();
    assert!(
        (result.moslqo - expected).abs() < TOLERANCE,
        "{reference} vs {degraded}: got {}, expected {expected}",
        result.moslqo
    );
}

fn audio_case(reference: &str, degraded: &str, expected: f64) {
    let reference = format!("conformance_testdata_subset/{reference}");
    let degraded_path = if degraded.contains('/') {
        degraded.to_string()
    } else {
        format!("conformance_testdata_subset/{degraded}")
    };
    assert_moslqo(&Visqol::audio(), &reference, &degraded_path, expected);
}

#[test]
fn strauss_lp35() {
    audio_case(
        "strauss48_stereo.wav",
        "strauss48_stereo_lp35.wav",
        1.3888791489130758,
    );
}

#[test]
fn steely_lp7() {
    audio_case(
        "steely48_stereo.wav",
        "steely48_stereo_lp7.wav",
        2.2501683734385183,
    );
}

#[test]
fn sopr_256aac() {
    audio_case(
        "sopr48_stereo.wav",
        "sopr48_stereo_256kbps_aac.wav",
        4.68228969737946,
    );
}

#[test]
fn ravel_128opus() {
    audio_case(
        "ravel48_stereo.wav",
        "ravel48_stereo_128kbps_opus.wav",
        4.465141897255348,
    );
}

#[test]
fn moonlight_128aac() {
    audio_case(
        "moonlight48_stereo.wav",
        "moonlight48_stereo_128kbps_aac.wav",
        4.684292801646114,
    );
}

#[test]
fn harpsichord_96mp3() {
    audio_case(
        "harpsichord48_stereo.wav",
        "harpsichord48_stereo_96kbps_mp3.wav",
        4.22374532766003,
    );
}

#[test]
fn guitar_64aac() {
    audio_case(
        "guitar48_stereo.wav",
        "guitar48_stereo_64kbps_aac.wav",
        4.349722308064298,
    );
}

#[test]
fn glock_48aac() {
    audio_case(
        "glock48_stereo.wav",
        "glock48_stereo_48kbps_aac.wav",
        4.332452943882108,
    );
}

#[test]
fn contrabassoon_24aac() {
    audio_case(
        "contrabassoon48_stereo.wav",
        "contrabassoon48_stereo_24kbps_aac.wav",
        2.346868205375293,
    );
}

#[test]
fn castanets_identity() {
    audio_case(
        "castanets48_stereo.wav",
        "castanets48_stereo.wav",
        4.732101253042348,
    );
}

#[test]
fn guitar_short_degraded_patch() {
    audio_case(
        "guitar48_stereo.wav",
        "short_duration/5_second/guitar48_stereo_5_sec.wav",
        4.314508583690198,
    );
}

#[test]
fn guitar_short_reference_patch() {
    assert_moslqo(
        &Visqol::audio(),
        "short_duration/5_second/guitar48_stereo_5_sec.wav",
        "conformance_testdata_subset/guitar48_stereo.wav",
        4.550791119387646,
    );
}

#[test]
fn speech_ca01_transcoded_lattice() {
    assert_moslqo(
        &Visqol::speech_lattice(),
        "clean_speech/CA01_01.wav",
        "clean_speech/transcoded_CA01_01.wav",
        3.3129234313964844,
    );
}

#[test]
fn speech_ca01_perfect_score_lattice() {
    assert_moslqo(
        &Visqol::speech_lattice(),
        "clean_speech/CA01_01.wav",
        "clean_speech/CA01_01.wav",
        4.505550384521484,
    );
}

#[test]
fn different_audios_lattice() {
    assert_moslqo(
        &Visqol::speech_lattice(),
        "conformance_testdata_subset/guitar48_stereo.wav",
        "clean_speech/CA01_01.wav",
        1.4982070922851562,
    );
}

#[test]
fn bad_degraded_lattice() {
    assert_moslqo(
        &Visqol::speech_lattice(),
        "alignment/reference.wav",
        "alignment/degraded.wav",
        1.19293212890625,
    );
}

#[test]
fn speech_ca01_transcoded_exponential() {
    assert_moslqo(
        &Visqol::speech_legacy(false),
        "clean_speech/CA01_01.wav",
        "clean_speech/transcoded_CA01_01.wav",
        3.374505555111911,
    );
}

#[test]
fn speech_unscaled_perfect_score_exponential() {
    assert_moslqo(
        &Visqol::speech_legacy(true),
        "clean_speech/CA01_01.wav",
        "clean_speech/CA01_01.wav",
        4.015861169223797,
    );
}

#[test]
fn different_audios_exponential() {
    assert_moslqo(
        &Visqol::speech_legacy(false),
        "conformance_testdata_subset/guitar48_stereo.wav",
        "clean_speech/CA01_01.wav",
        1.269675546824064,
    );
}

#[test]
fn bad_degraded_exponential() {
    assert_moslqo(
        &Visqol::speech_legacy(false),
        "alignment/reference.wav",
        "alignment/degraded.wav",
        1.357521678867611,
    );
}
