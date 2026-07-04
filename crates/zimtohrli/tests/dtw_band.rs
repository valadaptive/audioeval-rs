use zimtohrli::Zimtohrli;

/// A test signal with enough time structure that DTW alignment matters:
/// a chirp whose frequency wobbles.
fn wobble(seconds: f32) -> Vec<f32> {
    let n = (seconds * 48000.0) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / 48000.0;
            let f = 800.0 + 400.0 * (2.0 * std::f32::consts::PI * 0.7 * t).sin();
            (2.0 * std::f32::consts::PI * f * t).sin() * 0.5
        })
        .collect()
}

fn distance(z: &Zimtohrli, a: &[f32], b: &[f32]) -> f32 {
    let mut spec_a = z.analyze(a);
    let mut spec_b = z.analyze(b);
    z.distance(&mut spec_a, &mut spec_b)
}

#[test]
fn band_covering_everything_matches_exact_dtw() {
    let a = wobble(2.0);
    // Distorted copy.
    let mut b = a.clone();
    for i in 1..b.len() {
        b[i] = 0.7 * b[i] + 0.3 * b[i - 1];
    }

    let exact = Zimtohrli::default();
    let banded = Zimtohrli {
        // Radius larger than the number of steps: the band is the full matrix.
        dtw_band_radius: Some(10_000),
        ..Default::default()
    };
    assert_eq!(distance(&exact, &a, &b), distance(&banded, &a, &b));
}

#[test]
fn band_covering_the_misalignment_matches_exact_dtw() {
    let a = wobble(2.0);
    // b is a delayed by 200ms (~17 steps).
    let mut b = vec![0.0f32; 9600];
    b.extend_from_slice(&a);

    let exact = Zimtohrli::default();
    // 500ms of allowed drift comfortably covers the 200ms delay.
    let banded = Zimtohrli {
        dtw_band_radius: Some((0.5 * exact.perceptual_sample_rate) as usize),
        ..Default::default()
    };
    assert_eq!(distance(&exact, &a, &b), distance(&banded, &a, &b));
}

#[test]
fn band_smaller_than_the_misalignment_changes_the_distance() {
    let a = wobble(2.0);
    let mut b = vec![0.0f32; 9600];
    b.extend_from_slice(&a);

    let exact = Zimtohrli::default();
    let too_narrow = Zimtohrli {
        // 2 steps (~24ms) cannot absorb a 200ms delay, so the constrained
        // warp path differs from the exact one. Note that the resulting
        // distance is not necessarily *larger*: the distance is NSIM along
        // the warp path, not the DTW cost, so constraining the path can move
        // it in either direction.
        dtw_band_radius: Some(2),
        ..Default::default()
    };
    assert_ne!(distance(&too_narrow, &a, &b), distance(&exact, &a, &b));
}

#[test]
fn extreme_shapes_terminate() {
    // Very different lengths (steep diagonal) with a tiny radius, in both
    // orientations, plus tiny inputs: must not hang, panic, or produce NaN.
    let long = wobble(3.0);
    let short = wobble(0.2);
    let z = Zimtohrli {
        dtw_band_radius: Some(1),
        ..Default::default()
    };
    for (a, b) in [
        (&long[..], &short[..]),
        (&short[..], &long[..]),
        (&long[..], &long[..1000]),
        (&short[..600], &short[..]),
    ] {
        let d = distance(&z, a, b);
        assert!(
            d.is_finite(),
            "non-finite distance for {}x{}",
            a.len(),
            b.len()
        );
    }
}
