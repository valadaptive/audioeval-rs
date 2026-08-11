//! Profiling harness for the Rust `distance_without_dtw`: replays the
//! benchmark workload in a loop for ~10 seconds so perf/samply can sample it.
//!
//!   cargo build -p benchmarks --example profile_distance --release
//!   perf record -F 2999 --call-graph dwarf \
//!     target/release/examples/profile_distance
//!   perf report
//!
//! or: samply record target/release/examples/profile_distance

use std::path::PathBuf;
use std::time::Instant;

use zimtohrli::Zimtohrli;

const SAMPLE_RATE: usize = 48_000;
const SECONDS: usize = 5;

fn load(relative: &str) -> Vec<f32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../visqol/testdata/conformance_testdata_subset")
        .join(relative);
    let file = audio_io::read_audio_file(&path, SAMPLE_RATE).unwrap();
    file.channels[0][..SAMPLE_RATE * SECONDS].to_vec()
}

fn main() {
    let signal_a = load("ravel48_stereo.wav");
    let signal_b = load("ravel48_stereo_128kbps_opus.wav");

    let z = Zimtohrli::default();
    let spec_a = z.analyze(&signal_a);
    let spec_b = z.analyze(&signal_b);

    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    let mut iterations = 0u64;
    while Instant::now() < deadline {
        std::hint::black_box(z.distance_without_dtw(&mut spec_a.clone(), &mut spec_b.clone()));
        iterations += 1;
    }
    println!("{iterations} iterations");
}
