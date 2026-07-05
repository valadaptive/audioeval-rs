//! Shared audio decoding and resampling for the audioeval metrics.
//!
//! Decodes any format supported by symphonia into planar `f32` channels,
//! resampling to the requested rate with rubato when necessary.

use std::{fs::File, path::Path};

use anyhow::{Context as _, Result, anyhow, bail};
use audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Indexing, Resampler};
use symphonia::core::{
    audio::{Audio as _, AudioBuffer, GenericAudioBufferRef, conv::FromSample, sample::Sample},
    codecs::audio::AudioCodecId,
    formats::{TrackType, probe::Hint},
    io::MediaSourceStream,
};

pub struct AudioFile {
    pub channels: Vec<Vec<f32>>,
    pub format: AudioCodecId,
    pub src_sample_rate: usize,
}

pub fn read_audio_file(path: &Path, rate: usize) -> Result<AudioFile> {
    read_audio_file_inner(path, Some(rate)).with_context(|| format!("Error when reading {path:?}"))
}

/// Reads an audio file at its native sample rate, without resampling.
pub fn read_audio_file_native(path: &Path) -> Result<AudioFile> {
    read_audio_file_inner(path, None).with_context(|| format!("Error when reading {path:?}"))
}

fn append_planar_f32<S>(buf: &AudioBuffer<S>, out: &mut [Vec<f32>])
where
    S: Sample,
    f32: FromSample<S>,
{
    for (dst, plane) in out.iter_mut().zip(buf.iter_planes()) {
        dst.extend(plane.iter().map(|&s| f32::from_sample(s)));
    }
}

fn append_decoded(decoded: &GenericAudioBufferRef<'_>, out: &mut [Vec<f32>]) -> Result<()> {
    if decoded.num_planes() != out.len() {
        bail!("channel count changed mid-stream");
    }
    use GenericAudioBufferRef as G;
    match decoded {
        G::U8(b) => append_planar_f32(b, out),
        G::U16(b) => append_planar_f32(b, out),
        G::U24(b) => append_planar_f32(b, out),
        G::U32(b) => append_planar_f32(b, out),
        G::S8(b) => append_planar_f32(b, out),
        G::S16(b) => append_planar_f32(b, out),
        G::S24(b) => append_planar_f32(b, out),
        G::S32(b) => append_planar_f32(b, out),
        G::F32(b) => append_planar_f32(b, out),
        G::F64(b) => append_planar_f32(b, out),
    }
    Ok(())
}

fn read_audio_file_inner(path: &Path, rate: Option<usize>) -> Result<AudioFile> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(extension);
    }
    let mut format = symphonia::default::get_probe().probe(
        &hint,
        mss,
        Default::default(),
        Default::default(),
    )?;
    let Some(track) = format.default_track(TrackType::Audio) else {
        bail!("No audio track in {path:?}")
    };
    let track_id = track.id;

    let codec_params = track
        .codec_params
        .as_ref()
        .ok_or_else(|| anyhow!("codec parameters missing"))?
        .audio()
        .unwrap();
    let src_rate = codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("sample rate missing"))? as usize;
    let channels = codec_params
        .channels
        .as_ref()
        .ok_or_else(|| anyhow!("channels missing"))?;
    let num_channels = channels.count();
    let mut decoder =
        symphonia::default::get_codecs().make_audio_decoder(codec_params, &Default::default())?;

    let mut out_file = AudioFile {
        channels: vec![Vec::new(); num_channels],
        format: codec_params.codec,
        src_sample_rate: src_rate,
    };
    let out = &mut out_file.channels;

    let rate = rate.unwrap_or(src_rate);
    if src_rate == rate {
        loop {
            let packet = match format.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => break,
                Err(err) => return Err(err.into()),
            };

            format.metadata().skip_to_latest();

            if packet.track_id != track_id {
                continue;
            }

            let decoded = decoder.decode(&packet)?;
            if decoded.is_empty() {
                continue;
            }
            append_decoded(&decoded, out)?;
        }

        return Ok(out_file);
    }

    let mut resampler = rubato::Fft::<f32>::new(
        src_rate,
        rate,
        1024,
        2,
        num_channels,
        rubato::FixedSync::Both,
    )?;

    let mut scratch: Vec<Vec<f32>> = Vec::new();
    let mut pending: Vec<Vec<f32>> = vec![Vec::new(); num_channels];
    let mut chunk = vec![vec![0.0f32; resampler.output_frames_max()]; num_channels];
    let mut total_in = 0usize;

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(err) => return Err(err.into()),
        };

        format.metadata().skip_to_latest();

        if packet.track_id != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet)?;
        if decoded.is_empty() {
            continue;
        }
        decoded.copy_to_vecs_planar(&mut scratch);
        total_in += decoded.frames();
        for (fifo, plane) in pending.iter_mut().zip(&scratch) {
            fifo.extend_from_slice(plane);
        }

        // Feed the resampler every full chunk currently in the FIFO.
        let mut consumed = 0;
        while pending[0].len() - consumed >= resampler.input_frames_next() {
            let input = SequentialSliceOfVecs::new(&pending, num_channels, pending[0].len())?;
            let mut output = SequentialSliceOfVecs::new_mut(
                &mut chunk,
                num_channels,
                resampler.output_frames_max(),
            )?;
            let indexing = Indexing {
                input_offset: consumed,
                output_offset: 0,
                partial_len: None,
                active_channels_mask: None,
            };
            let (n_in, n_out) =
                resampler.process_into_buffer(&input, &mut output, Some(&indexing))?;
            consumed += n_in;
            for (out_ch, chunk_ch) in out.iter_mut().zip(&chunk) {
                out_ch.extend_from_slice(&chunk_ch[..n_out]);
            }
        }
        for fifo in &mut pending {
            fifo.drain(..consumed);
        }
    }

    // Flush: the FIFO tail plus enough zero-padding to push the resampler's
    // delay line out, then trim the latency off the front.
    let delay = resampler.output_delay();
    let expected_out = (total_in as u64 * rate as u64).div_ceil(src_rate as u64) as usize;
    let mut partial_len = pending[0].len();
    while out[0].len() < delay + expected_out {
        let input = SequentialSliceOfVecs::new(&pending, num_channels, pending[0].len())?;
        let mut output = SequentialSliceOfVecs::new_mut(
            &mut chunk,
            num_channels,
            resampler.output_frames_max(),
        )?;
        let indexing = Indexing {
            input_offset: 0,
            output_offset: 0,
            partial_len: Some(partial_len),
            active_channels_mask: None,
        };
        let (_, n_out) = resampler.process_into_buffer(&input, &mut output, Some(&indexing))?;
        partial_len = 0;
        for (out_ch, chunk_ch) in out.iter_mut().zip(&chunk) {
            out_ch.extend_from_slice(&chunk_ch[..n_out]);
        }
    }
    for ch in out {
        ch.drain(..delay);
        ch.truncate(expected_out);
    }

    Ok(out_file)
}
