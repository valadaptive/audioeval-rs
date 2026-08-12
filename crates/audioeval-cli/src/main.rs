use std::{
    fmt::Display,
    io::{Write as _, stdout},
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, anyhow};
use audio_io::read_audio_file;
use clap::{
    Arg, ArgAction, Command, ValueEnum,
    builder::{BoolishValueParser, PossibleValue},
    value_parser,
};
use two_f_model::TwoFModel;
use visqol::{AudioSignal, SvrModel, Visqol};
use zimtohrli::Zimtohrli;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Metric {
    TwoF,
    Zimtohrli,
    Visqol,
}

impl ValueEnum for Metric {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::TwoF, Self::Zimtohrli, Self::Visqol]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        match self {
            Self::TwoF => Some(PossibleValue::new("2f").help("The SEBASS 2f-model")),
            Self::Zimtohrli => {
                Some(PossibleValue::new("zimtohrli").help("The Zimtohrli audio metric"))
            }
            Self::Visqol => Some(PossibleValue::new("visqol").help("The ViSQOL audio metric")),
        }
    }
}

struct Metrics {
    degraded_path: PathBuf,
    two_f: Option<f64>,
    zimtohrli: Option<f64>,
    visqol: Option<f64>,
}

struct OrBlank<T>(Option<T>);

impl<T: Display> Display for OrBlank<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(v) => v.fmt(f),
            None => Ok(()),
        }
    }
}

struct ZimtWithOptions {
    zimt: Zimtohrli,
    skip_dtw: bool,
    output_mos: bool,
}

struct TwoFWithOptions {
    two_f: TwoFModel,
    output_raw: bool,
}

fn run_one<'a>(
    reference_path: &Path,
    degraded_paths: impl Iterator<Item = &'a Path> + Clone,
    two_f: Option<&TwoFWithOptions>,
    zimtohrli: Option<&ZimtWithOptions>,
    visqol: Option<&Visqol>,
    fail_fast: bool,
) -> Result<(Vec<Metrics>, bool), anyhow::Error> {
    let need_48k = two_f.is_some()
        || zimtohrli.is_some()
        || visqol.is_some_and(|visqol| !visqol.is_speech_mode());
    let need_16k = visqol.is_some_and(|visqol| visqol.is_speech_mode());

    // It would be nice to use something like LazyCell here, but for returning
    // errors early and pleasing the borrow checker, this will have to do for
    // now
    let reference_48k = need_48k
        .then(|| read_audio_file(reference_path, 48000))
        .transpose()?;
    let reference_16k = need_16k
        .then(|| read_audio_file(reference_path, 16000))
        .transpose()?;

    let reference_zimt_spec = zimtohrli.map(|zimt| {
        reference_48k
            .as_ref()
            .unwrap()
            .channels
            .iter()
            .map(|ch| zimt.zimt.analyze(ch))
            .collect::<Vec<_>>()
    });
    let reference_visqol_signal = visqol.map(|visqol| {
        let reference = if visqol.is_speech_mode() {
            reference_16k.as_ref().unwrap()
        } else {
            reference_48k.as_ref().unwrap()
        };
        AudioSignal::from_channels(
            &reference.channels,
            if visqol.is_speech_mode() {
                16000
            } else {
                48000
            },
        )
    });

    let mut results = Vec::new();
    let mut any_error = false;
    for degraded_path in degraded_paths {
        let deg_res = (|| -> Result<Metrics, anyhow::Error> {
            let degraded_48k = need_48k
                .then(|| read_audio_file(degraded_path, 48000))
                .transpose()?;
            let degraded_16k = need_16k
                .then(|| read_audio_file(degraded_path, 16000))
                .transpose()?;

            let mut metrics = Metrics {
                degraded_path: degraded_path.to_path_buf(),
                two_f: None,
                zimtohrli: None,
                visqol: None,
            };

            if let Some(two_f) = two_f {
                metrics.two_f = match two_f.two_f.run(
                    reference_48k.as_ref().unwrap().channels.as_slice(),
                    degraded_48k.as_ref().unwrap().channels.as_slice(),
                ) {
                    Ok(res) => Some(if two_f.output_raw {
                        res.raw_mushra_score
                    } else {
                        res.mushra_score
                    }),
                    Err(e) => {
                        if fail_fast {
                            return Err(e.into());
                        }
                        eprintln!("{:#}", e);
                        any_error = true;
                        None
                    }
                };
            }
            if let Some(zimt) = zimtohrli {
                let reference_zimt_spec = reference_zimt_spec.as_ref().unwrap();
                let deg_channels = &degraded_48k.as_ref().unwrap().channels;

                let zimt_res = (|| {
                    if deg_channels.len() != reference_zimt_spec.len() {
                        return Err(anyhow!(
                            "channel count mismatch: reference has {} channels but degraded has {}",
                            reference_zimt_spec.len(),
                            deg_channels.len()
                        ));
                    }

                    let mut sum_of_squares = 0.0f64;
                    for (ch, ref_spec) in deg_channels.iter().zip(reference_zimt_spec) {
                        let mut deg_spec = zimt.zimt.analyze(ch);
                        let distance = if zimt.skip_dtw {
                            if ref_spec.values().len() != deg_spec.values().len() {
                                return Err(anyhow!(
                                    "spectrograms must be of equal length when DTW is disabled",
                                ));
                            }
                            zimt.zimt
                                .distance_without_dtw(&mut ref_spec.clone(), &mut deg_spec)
                        } else {
                            zimt.zimt.distance(&mut ref_spec.clone(), &mut deg_spec)
                        } as f64;
                        sum_of_squares += distance * distance;
                    }
                    let aggregate = (sum_of_squares / reference_zimt_spec.len() as f64).sqrt();
                    Ok(if zimt.output_mos {
                        Zimtohrli::mos_from_distance(aggregate as f32) as f64
                    } else {
                        aggregate
                    })
                })();

                metrics.zimtohrli = match zimt_res {
                    Ok(res) => Some(res),
                    Err(e) => {
                        if fail_fast {
                            return Err(e.into());
                        }
                        eprintln!("{:#}", e);
                        any_error = true;
                        None
                    }
                };
            }
            if let Some(visqol) = visqol {
                let degraded = if visqol.is_speech_mode() {
                    AudioSignal::from_channels(&degraded_16k.unwrap().channels, 16000)
                } else {
                    AudioSignal::from_channels(&degraded_48k.unwrap().channels, 48000)
                };

                metrics.visqol =
                    match visqol.run(reference_visqol_signal.as_ref().unwrap(), &degraded) {
                        Ok(res) => Some(res.moslqo),
                        Err(e) => {
                            if fail_fast {
                                return Err(e.into());
                            }
                            eprintln!("{:#}", e);
                            any_error = true;
                            None
                        }
                    };
            }

            Ok(metrics)
        })();

        match deg_res {
            Ok(metrics) => {
                results.push(metrics);
            }
            Err(e) => {
                if fail_fast {
                    return Err(e);
                }
                eprintln!("{:#}", e);
                any_error = true;
            }
        }
    }

    Ok((results, any_error))
}

fn main() -> anyhow::Result<()> {
    let default_zimtohrli = zimtohrli::Zimtohrli::default();
    let default_perceptual_sample_rate = default_zimtohrli.perceptual_sample_rate;
    let default_perceptual_sample_rate_str = format!("{}", default_perceptual_sample_rate);

    let matches = Command::new("audioeval-cli")
        .about("Compare audio files using perceptual metrics")
        .arg(
            Arg::new("input")
                .long("input")
                .short('i')
                .value_names(["REFERENCE", "DEGRADED"])
                .num_args(2..)
                .action(ArgAction::Append)
                .help("input audio files to compare")
                .required(true)
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("metrics")
                .long("metrics")
                .short('m')
                .value_parser(value_parser!(Metric))
                .value_delimiter(',')
                .num_args(1..)
                .required(true)
                .action(ArgAction::Append)
            )
        .arg(
            Arg::new("output_csv")
            .long("output_csv")
            .action(ArgAction::SetTrue)
            .help("output CSV rather than human-readable text")
        )
        .arg(
            Arg::new("fail_fast")
            .long("fail_fast")
            .action(ArgAction::SetTrue)
            .help("fail on the first error and return early")
        )
        .next_help_heading("Zimtohrli")
        .arg(
            Arg::new("zimt_perceptual_sample_rate")
                .long("zimt_perceptual_sample_rate")
                .hide_short_help(true)
                .long_help("the frequency corresponding to the maximum time resolution, Hz")
                .default_value(default_perceptual_sample_rate_str)
                .value_parser(value_parser!(f32)),
        )
        .arg(
            Arg::new("zimt_skip_dtw")
                .long("zimt_skip_dtw")
                .hide_short_help(true)
                .long_help(
                    "disable the DTW (dynamic time warping) step, which improves performance and potentially accuracy if the signals are known not to be time-misaligned",
                )
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("zimt_dtw_band_radius")
                .long("zimt_dtw_band_radius")
                .hide_short_help(true)
                .long_help(
                    "restrict the DTW to time misalignments of at most this many seconds \
                     (much faster, but results are only reliable if the true misalignment \
                     stays within it); omit for the exact, exhaustive DTW",
                )
                .value_parser(value_parser!(f32)),
        )
        .arg(
            Arg::new("zimt_output_mos")
                .long("zimt_output_mos")
                .hide_short_help(true)
                .help("output a mapped mean opinion score instead of the raw Zimtohrli distance value.")
                .action(ArgAction::SetTrue),
        )
        .next_help_heading("ViSQOL")
        .arg(
            Arg::new("visqol_use_speech_mode")
                .long("visqol_use_speech_mode")
                .help(
                    "use a wideband model (sensitive up to 8kHz) with voice \
                     activity detection; input is resampled to 16kHz",
                )
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("visqol_use_lattice_model")
                .long("visqol_use_lattice_model")
                .hide_short_help(true)
                .help(
                    "in speech mode, use a deep lattice network model to map \
                     similarity to quality (more accurate); pass \
                     --visqol_use_lattice_model=false for the exponential fit",
                )
                .value_parser(BoolishValueParser::new())
                .default_value("true"),
        )
        .arg(
            Arg::new("visqol_use_unscaled_speech_mos_mapping")
                .long("visqol_use_unscaled_speech_mos_mapping")
                .hide_short_help(true)
                .help(
                    "in speech mode, do not scale the MOS mapping so that a \
                     perfect NSIM score maps to a perfect MOS",
                )
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("visqol_similarity_to_quality_model")
                .long("visqol_similarity_to_quality_model")
                .hide_short_help(true)
                .help("libsvm nu-SVR model to use in audio mode instead of the built-in one")
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("visqol_search_window_radius")
                .long("visqol_search_window_radius")
                .hide_short_help(true)
                .help(
                    "how far the algorithm looks for a matching degraded patch, \
                     in units of patch length on either side",
                )
                .default_value("60")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("visqol_disable_global_alignment")
                .long("visqol_disable_global_alignment")
                .hide_short_help(true)
                .help("disable the initial envelope-based global alignment")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("visqol_disable_realignment")
                .long("visqol_disable_realignment")
                .hide_short_help(true)
                .help("disable the per-patch time-domain fine realignment")
                .action(ArgAction::SetTrue),
        )
        .next_help_heading("2f")
        .arg(
            Arg::new("2f_unclamped_score")
                .long("2f_unclamped_score")
                .hide_short_help(true)
                .help("output the raw 2f-model score, not clamped between 0-100")
                .action(ArgAction::SetTrue),
        )
        .get_matches();

    let metrics = matches.get_many::<Metric>("metrics").unwrap();
    let mut use_2f = false;
    let mut use_zimtohrli = false;
    let mut use_visqol = false;
    for metric in metrics {
        match metric {
            Metric::TwoF => use_2f = true,
            Metric::Zimtohrli => use_zimtohrli = true,
            Metric::Visqol => use_visqol = true,
        }
    }
    let output_csv = matches.get_flag("output_csv");
    let fail_fast = matches.get_flag("fail_fast");

    let zimtohrli = use_zimtohrli.then(|| {
        let perceptual_sample_rate = *matches
            .get_one::<f32>("zimt_perceptual_sample_rate")
            .unwrap();

        let dtw_band_radius = matches
            .get_one::<f32>("zimt_dtw_band_radius")
            .map(|&seconds| (seconds * perceptual_sample_rate).round() as usize);
        ZimtWithOptions {
            zimt: Zimtohrli {
                perceptual_sample_rate,
                dtw_band_radius,
                ..Default::default()
            },
            skip_dtw: matches.get_flag("zimt_skip_dtw"),
            output_mos: matches.get_flag("zimt_output_mos"),
        }
    });

    let visqol = if use_visqol {
        let speech_mode = matches.get_flag("visqol_use_speech_mode");

        let mut visqol = if speech_mode {
            if matches.get_flag("visqol_use_lattice_model") {
                Visqol::speech_lattice()
            } else {
                Visqol::speech_legacy(matches.get_flag("visqol_use_unscaled_speech_mos_mapping"))
            }
        } else {
            match matches.get_one::<PathBuf>("visqol_similarity_to_quality_model") {
                Some(path) => {
                    let text = std::fs::read_to_string(path)
                        .with_context(|| format!("Error reading model {path:?}"))?;
                    Visqol::audio_with_model(SvrModel::from_text(&text)?)
                }
                None => Visqol::audio(),
            }
        };
        visqol.search_window_radius = *matches
            .get_one::<usize>("visqol_search_window_radius")
            .unwrap();
        visqol.disable_global_alignment = matches.get_flag("visqol_disable_global_alignment");
        visqol.disable_realignment = matches.get_flag("visqol_disable_realignment");

        Some(visqol)
    } else {
        None
    };

    let two_f = use_2f.then(|| TwoFWithOptions {
        two_f: TwoFModel::new(),
        output_raw: matches.get_flag("2f_unclamped_score"),
    });

    let mut stdout = stdout().lock();
    if output_csv {
        write!(stdout, "reference,degraded")?;
        if use_2f {
            write!(stdout, ",2f")?;
        }
        if use_zimtohrli {
            write!(stdout, ",zimtohrli")?;
        }
        if use_visqol {
            write!(stdout, ",visqol")?;
        }
        let _ = writeln!(stdout);
        let _ = stdout.flush();
    }

    let mut any_error = false;
    for mut occ in matches.get_occurrences::<PathBuf>("input").unwrap() {
        let reference = occ.next().unwrap();
        let res = run_one(
            reference,
            occ.map(|p| p.as_path()),
            two_f.as_ref(),
            zimtohrli.as_ref(),
            visqol.as_ref(),
            fail_fast,
        );
        match res {
            Ok((res, any_file_err)) => {
                any_error |= any_file_err;
                if output_csv {
                    for metrics in res {
                        stdout.write_all(reference.as_os_str().as_encoded_bytes())?;
                        write!(stdout, ",")?;
                        stdout.write_all(metrics.degraded_path.as_os_str().as_encoded_bytes())?;
                        if use_2f {
                            write!(stdout, ",{}", OrBlank(metrics.two_f))?;
                        }
                        if use_zimtohrli {
                            write!(stdout, ",{}", OrBlank(metrics.zimtohrli))?;
                        }
                        if use_visqol {
                            write!(stdout, ",{}", OrBlank(metrics.visqol))?;
                        }
                        writeln!(stdout)?;
                    }
                } else {
                    write!(stdout, "Reference: ")?;
                    stdout.write_all(reference.as_os_str().as_encoded_bytes())?;
                    for metrics in res {
                        write!(stdout, "\n  Degraded: ")?;
                        stdout.write_all(metrics.degraded_path.as_os_str().as_encoded_bytes())?;
                        if let Some(two_f) = metrics.two_f {
                            write!(stdout, "\n    2f: {}", two_f)?;
                        }
                        if let Some(zimt) = metrics.zimtohrli {
                            write!(stdout, "\n    Zimtohrli: {}", zimt)?;
                        }
                        if let Some(visqol) = metrics.visqol {
                            write!(stdout, "\n    ViSQOL: {}", visqol)?;
                        }
                        writeln!(stdout)?;
                    }
                }
                stdout.flush()?;
            }
            Err(e) => {
                if fail_fast {
                    return Err(e);
                }
                eprintln!("{:#}", e);
                any_error = true;
            }
        }
    }

    if any_error {
        Err(anyhow!("one or more audio files failed to process"))
    } else {
        Ok(())
    }
}
