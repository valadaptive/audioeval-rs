# two-f-model

<!-- This README is generated via https://crates.io/crates/cargo-rdme. Edit the crate-level docs and regenerate this README via `cargo rdme`. -->

<!-- cargo-rdme start -->

A pure-Rust implementation of the SEBASS 2f-model for estimating the perceived quality of separated audio sources.

The model uses only two PEAQ Basic model output variables: `AvgModDiff1` and `ADB`. Their implementation follows Peter Kabal's PQevalAudio v1r0, which is the PEAQ implementation for which the published 2f-model parameters were fitted.

```rust
use two_f_model::TwoFModel;

let reference: Vec<Vec<f32>> = vec![vec![0.0; 48_000]; 2];
let degraded = reference.clone();
let mut model = TwoFModel::new();
let result = model.run(&reference, &degraded, 48_000)?;
println!("estimated MUSHRA score: {}", result.mushra_score);
```

Input samples are normalized PCM in `[-1, 1]`. Only 48 kHz mono or stereo signals are supported. The signals must already be time-aligned.

The returned score is clipped to the MUSHRA range of 0–100; the unclipped regression output and both MOVs are also available in `TwoFResult`.

<!-- cargo-rdme end -->
