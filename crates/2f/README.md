# audioeval-2f

<!-- This README is generated via https://crates.io/crates/cargo-rdme. Edit the crate-level docs and regenerate this README via `cargo rdme`. -->

<!-- cargo-rdme start -->

A pure-Rust implementation of the SEBASS 2f-model for estimating the perceived quality of separated audio sources.

The model uses only two PEAQ Basic model output variables: `AvgModDiff1` and `ADB`. Their implementation follows Peter Kabal's PQevalAudio v1r0, which is the PEAQ implementation for which the published 2f-model parameters were fitted.

```rust
use audioeval_2f::TwoFModel;

let reference: Vec<Vec<f32>> = vec![vec![0.0; 48_000]; 2];
let degraded = reference.clone();
let mut model = TwoFModel::new();
let result = model.run(&reference, &degraded)?;
println!("estimated MUSHRA score: {}", result.mushra_score);
```

Input samples are normalized PCM in `[-1, 1]`. Only 48 kHz mono or stereo signals are supported. The signals must already be time-aligned.

The returned score is clipped to the MUSHRA range of 0–100; the unclipped regression output and both MOVs are also available in `TwoFResult`.

### Why not PEAQ?

Why did I choose only to implement two specific PEAQ model output variables (MOVs), rather than just implementing all of PEAQ Basic? There are a few reasons:

- PEAQ's ODG (objective difference grade) output seems to be a poor metric. The individual MOVs may still prove useful, but...

- There are currently *no conforming PEAQ implementations*, and the standard is worded vaguely enough that it appears impossible to write one (see [this report](https://www.mmsp.ece.mcgill.ca/Documents/Reports/2002/KabalR2002v2.pdf)).

  As mentioned above, this package resolves ambiguities in the PEAQ specification by following the behavior of the MATLAB [PQEvalAudio](https://www.mmsp.ece.mcgill.ca/Documents/Software/index.html) package. Other implementations, such as [EAQUAL](https://github.com/spxnn/eaqual) or [GstPEAQ](https://github.com/HSU-ANT/gstpeaq), interpret the spec differently and produce different results.

- It's slower to compute MOVs that we don't need.

In the future, I *may* attempt to write a PEAQ implementation that can be configured to match the behavior of other implementations, but the lack of standardization renders it unattractive as an objective metric.

<!-- cargo-rdme end -->
