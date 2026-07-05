use std::path::PathBuf;

use anyhow::{Result, anyhow};
use audio_io::AudioFile;
use clap::{Arg, ArgAction, Command, value_parser};
use zimtohrli::{Spectrogram, Zimtohrli};

fn main() -> Result<()> {
    let default_zimtohrli = zimtohrli::Zimtohrli::default();
    let default_perceptual_sample_rate = default_zimtohrli.perceptual_sample_rate;
    let default_perceptual_sample_rate_str = format!("{}", default_perceptual_sample_rate);

    let matches = Command::new("zimtohrli-compare")
        .about("Compare audio files using the Zimtohrli perceptual metric")
        .arg(
            Arg::new("path_a")
                .long("path_a")
                .short('a')
                .help("file A to compare")
                .required(true)
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("path_b")
                .long("path_b")
                .short('b')
                .help("files B to compare to file A")
                .required(true)
                .action(ArgAction::Append)
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("perceptual_sample_rate")
                .long("perceptual_sample_rate")
                .help("the frequency corresponding to the maximum time resolution, Hz")
                .default_value(default_perceptual_sample_rate_str)
                .value_parser(value_parser!(f32)),
        )
        .arg(
            Arg::new("full_scale_sine_db")
                .long("full_scale_sine_db")
                .help("reference dB SPL for a sine signal of amplitude 1")
                .default_value("80")
                .value_parser(value_parser!(f32)),
        )
        .arg(
            Arg::new("dtw_band_seconds")
                .long("dtw_band_seconds")
                .help(
                    "restrict the DTW to time misalignments of at most this many seconds \
                     (much faster, but results are only reliable if the true misalignment \
                     stays within it); omit for the exact, exhaustive DTW",
                )
                .value_parser(value_parser!(f32)),
        )
        .arg(
            Arg::new("verbose")
                .long("verbose")
                .short('v')
                .help("verbose output")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("output_zimtohrli_distance")
                .long("output_zimtohrli_distance")
                .help("Whether to output the raw Zimtohrli distance instead of a mapped mean opinion score.")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("per_channel")
                .long("per_channel")
                .help("Whether to output the produced metric per channel instead of a single value for all channels.")
                .action(ArgAction::SetTrue),
        )
        .get_matches();

    let path_a = matches.get_one::<PathBuf>("path_a").unwrap();
    let pretty_path_a = path_a.to_string_lossy();
    let path_b: Vec<&PathBuf> = matches.get_many::<PathBuf>("path_b").unwrap().collect();

    if path_b.is_empty() {
        eprintln!("Both path_a and path_b have to be specified.");
        std::process::exit(1);
    }

    let full_scale_sine_db = *matches.get_one::<f32>("full_scale_sine_db").unwrap();
    if full_scale_sine_db < 1.0 {
        eprintln!("Full scale sine dB must be >= 1.");
        std::process::exit(3);
    }

    let perceptual_sample_rate = *matches.get_one::<f32>("perceptual_sample_rate").unwrap();
    let verbose = *matches.get_one::<bool>("verbose").unwrap();
    let output_zimtohrli_distance = *matches
        .get_one::<bool>("output_zimtohrli_distance")
        .unwrap();
    let per_channel = *matches.get_one::<bool>("per_channel").unwrap();

    let get_metric = |distance: f32| {
        if output_zimtohrli_distance {
            distance
        } else {
            Zimtohrli::mos_from_distance(distance)
        }
    };

    let print_file_info = |pretty_path: &str, file: &AudioFile| {
        if !verbose {
            return;
        }

        let format = file.format;
        let src_sample_rate = file.src_sample_rate;
        let num_channels = file.channels.len();

        println!("Loaded {pretty_path} ({src_sample_rate}Hz, {num_channels}ch, codec ID {format})");
    };

    let audio_a = audio_io::read_audio_file(path_a, 48000)?;
    print_file_info(&pretty_path_a, &audio_a);

    let dtw_band_radius = matches
        .get_one::<f32>("dtw_band_seconds")
        .map(|&seconds| (seconds * perceptual_sample_rate).round() as usize);

    let z = Zimtohrli {
        perceptual_sample_rate,
        full_scale_sine_db,
        dtw_band_radius,
        ..Default::default()
    };

    let mut file_a_spectrograms: Vec<Spectrogram> = audio_a
        .channels
        .iter()
        .map(|channel| z.analyze(channel))
        .collect();

    for path in path_b {
        let pretty_path_b = path.to_string_lossy();
        let audio_b = audio_io::read_audio_file(path, 48000)?;
        print_file_info(&pretty_path_b, &audio_b);

        let mut sum_of_squares = 0.0f32;

        if audio_b.channels.len() != audio_a.channels.len() {
            return Err(anyhow!(
                "Reference {pretty_path_a} has {} channels, but {pretty_path_b} has {} channels",
                audio_a.channels.len(),
                audio_b.channels.len()
            ));
        }

        let mut spec_b: Option<Spectrogram> = None;
        for (i, (channel, spec_a)) in audio_b
            .channels
            .iter()
            .zip(file_a_spectrograms.iter_mut())
            .enumerate()
        {
            let spec_b = match spec_b.as_mut() {
                Some(spec) => {
                    z.analyze_into(channel, spec);
                    spec
                }
                None => spec_b.insert(z.analyze(channel)),
            };

            let distance = z.distance(spec_a, spec_b);

            if per_channel {
                println!("{pretty_path_b} ch{i}: {}", get_metric(distance));
            } else {
                sum_of_squares += distance * distance;
            }
        }

        if !per_channel {
            println!(
                "{pretty_path_b}: {}",
                get_metric((sum_of_squares / audio_b.channels.len() as f32).sqrt())
            );
        }
    }

    Ok(())
}
