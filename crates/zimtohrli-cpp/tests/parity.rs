//! Parity checks between the Rust port and the C++ original.
//!
//! The Rust implementation is a reimplementation (different floating-point
//! operation order, fast pow approximation, runtime SIMD dispatch), so
//! bitwise equality is not expected, but results should agree closely. Run
//! with output to see the observed differences:
//!
//!   cargo test -p zimtohrli-cpp -- --nocapture

use audioeval_zimtohrli::Zimtohrli;
use zimtohrli_cpp::CppZimtohrli;

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

#[test]
fn analyze_parity() {
    let signal = test_signal(2, 0.0);

    let rust_spec = Zimtohrli::default().analyze(&signal);
    let cpp_spec = CppZimtohrli::default().analyze(&signal);

    assert_eq!(rust_spec.num_steps(), cpp_spec.num_steps());
    assert_eq!(rust_spec.num_dims(), cpp_spec.num_dims());
    assert_eq!(rust_spec.values().len(), cpp_spec.values().len());

    let peak = rust_spec
        .values()
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()));
    let max_abs_diff = rust_spec
        .values()
        .iter()
        .zip(cpp_spec.values())
        .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
    println!("peak = {peak:e}, max abs diff = {max_abs_diff:e}");

    assert!(
        max_abs_diff <= 1e-4 * peak,
        "spectrogram mismatch: max abs diff {max_abs_diff:e} vs peak {peak:e}"
    );
}

/// Linear-interpolation resample by `scale` (< 1 = slower tempo), giving the
/// DTW something real to align.
fn time_scaled(signal: &[f32], scale: f32) -> Vec<f32> {
    (0..signal.len())
        .map(|i| {
            let pos = i as f32 * scale;
            let idx = (pos as usize).min(signal.len() - 1);
            let next = (idx + 1).min(signal.len() - 1);
            let frac = pos - idx as f32;
            signal[idx] * (1.0 - frac) + signal[next] * frac
        })
        .collect()
}

#[test]
fn distance_parity() {
    let signal_a = test_signal(2, 0.0);
    let signal_b = time_scaled(&test_signal(2, 0.02), 0.99);

    let rust = Zimtohrli::default();
    let cpp = CppZimtohrli::default();

    let rust_distance = rust.distance(&mut rust.analyze(&signal_a), &mut rust.analyze(&signal_b));
    let cpp_distance = cpp.distance(&mut cpp.analyze(&signal_a), &mut cpp.analyze(&signal_b));
    println!("distance: rust = {rust_distance:e}, cpp = {cpp_distance:e}");
    assert!(
        (rust_distance - cpp_distance).abs() <= 1e-5,
        "distance mismatch: rust {rust_distance:e} vs cpp {cpp_distance:e}"
    );

    let rust_aligned =
        rust.distance_without_dtw(&mut rust.analyze(&signal_a), &mut rust.analyze(&signal_b));
    let cpp_aligned =
        cpp.distance_without_dtw(&mut cpp.analyze(&signal_a), &mut cpp.analyze(&signal_b));
    println!("distance_without_dtw: rust = {rust_aligned:e}, cpp = {cpp_aligned:e}");
    assert!(
        (rust_aligned - cpp_aligned).abs() <= 1e-5,
        "distance_without_dtw mismatch: rust {rust_aligned:e} vs cpp {cpp_aligned:e}"
    );
}

#[test]
fn mos_parity() {
    let mut max_abs_diff = 0.0f32;
    for i in 0..=100 {
        let distance = i as f32 / 100.0;
        let diff = (Zimtohrli::mos_from_distance(distance)
            - CppZimtohrli::mos_from_distance(distance))
        .abs();
        max_abs_diff = max_abs_diff.max(diff);
    }
    println!("max abs MOS diff over [0, 1]: {max_abs_diff:e}");
    assert!(max_abs_diff <= 1e-6);
}
