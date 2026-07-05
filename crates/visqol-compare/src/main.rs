use std::path::PathBuf;

use anyhow::{Context as _, Result};
use clap::{Arg, ArgAction, Command, value_parser};
use visqol::{AudioSignal, SvrModel, Visqol};

fn main() -> Result<()> {
    let matches = Command::new("visqol-compare")
        .about("Compare audio files using the ViSQOL perceptual quality metric")
        .arg(
            Arg::new("reference_file")
                .long("reference_file")
                .short('r')
                .help("reference audio file")
                .required(true)
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("degraded_file")
                .long("degraded_file")
                .short('d')
                .help("degraded audio file to score against the reference")
                .required(true)
                .action(ArgAction::Append)
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("use_speech_mode")
                .long("use_speech_mode")
                .help(
                    "use a wideband model (sensitive up to 8kHz) with voice \
                     activity detection; input is resampled to 16kHz",
                )
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("use_unscaled_speech_mos_mapping")
                .long("use_unscaled_speech_mos_mapping")
                .help(
                    "in speech mode, do not scale the MOS mapping so that a \
                     perfect NSIM score maps to a perfect MOS",
                )
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("similarity_to_quality_model")
                .long("similarity_to_quality_model")
                .help("libsvm nu-SVR model to use in audio mode instead of the built-in one")
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("search_window_radius")
                .long("search_window_radius")
                .help(
                    "how far the algorithm looks for a matching degraded patch, \
                     in units of patch length on either side",
                )
                .default_value("60")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("disable_global_alignment")
                .long("disable_global_alignment")
                .help("disable the initial envelope-based global alignment")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("disable_realignment")
                .long("disable_realignment")
                .help("disable the per-patch time-domain fine realignment")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("verbose")
                .long("verbose")
                .short('v')
                .help("also print vnsim and the per-band similarities")
                .action(ArgAction::SetTrue),
        )
        .get_matches();

    let reference_path = matches.get_one::<PathBuf>("reference_file").unwrap();
    let degraded_paths: Vec<&PathBuf> = matches
        .get_many::<PathBuf>("degraded_file")
        .unwrap()
        .collect();
    let speech_mode = matches.get_flag("use_speech_mode");
    let verbose = matches.get_flag("verbose");

    let mut visqol = if speech_mode {
        Visqol::speech(!matches.get_flag("use_unscaled_speech_mos_mapping"))
    } else {
        match matches.get_one::<PathBuf>("similarity_to_quality_model") {
            Some(path) => {
                let text = std::fs::read_to_string(path)
                    .with_context(|| format!("Error reading model {path:?}"))?;
                Visqol::audio_with_model(SvrModel::from_text(&text)?)
            }
            None => Visqol::audio(),
        }
    };
    visqol.search_window_radius = *matches.get_one::<usize>("search_window_radius").unwrap();
    visqol.disable_global_alignment = matches.get_flag("disable_global_alignment");
    visqol.disable_realignment = matches.get_flag("disable_realignment");

    // Like the C++ CLI, files are used at their native sample rate; ViSQOL
    // only requires that the two rates match.
    let load = |path: &PathBuf| -> Result<AudioSignal> {
        let file = audio_io::read_audio_file_native(path)?;
        let sample_rate = file.src_sample_rate as u32;
        if speech_mode && sample_rate > 16000 {
            eprintln!(
                "WARNING: input audio sample rate is above 16kHz, which may have \
                 undesired effects for speech mode. Consider resampling to 16kHz."
            );
        } else if !speech_mode && sample_rate != 48000 {
            eprintln!(
                "WARNING: input audio does not have the expected sample rate of \
                 48kHz! This may negatively affect the prediction of the MOS-LQO score."
            );
        }
        Ok(AudioSignal::from_channels(&file.channels, sample_rate))
    };

    let reference = load(reference_path)?;
    for path in degraded_paths {
        let degraded = load(path)?;
        let result = visqol.run(&reference, &degraded)?;
        println!("{}: {}", path.to_string_lossy(), result.moslqo);
        if verbose {
            println!("  vnsim: {}", result.vnsim);
            println!("  alignment lag: {}s", result.alignment_lag_s);
            println!("  band Hz: fvnsim fvnsim10 fstdnsim");
            for (i, freq) in result.center_freq_bands.iter().enumerate() {
                println!(
                    "  {freq:9.2}: {:.6} {:.6} {:.6}",
                    result.fvnsim[i], result.fvnsim10[i], result.fstdnsim[i]
                );
            }
        }
    }
    Ok(())
}
