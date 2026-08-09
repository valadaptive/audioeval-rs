use std::{env, path::Path};

use two_f_model::TwoFModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let reference_path = args.next().ok_or("usage: score REFERENCE DEGRADED")?;
    let degraded_path = args.next().ok_or("usage: score REFERENCE DEGRADED")?;
    if args.next().is_some() {
        return Err("usage: score REFERENCE DEGRADED".into());
    }
    let reference = audio_io::read_audio_file(Path::new(&reference_path), 48000)?;
    let degraded = audio_io::read_audio_file(Path::new(&degraded_path), 48000)?;
    let result = TwoFModel::new().run(&reference.channels, &degraded.channels)?;
    println!("Estimated mean MUSHRA score: {:.6}", result.mushra_score);
    println!("AvgModDiff1B: {:.9}", result.avg_mod_diff1);
    println!("ADBB: {:.9}", result.adb);
    Ok(())
}
