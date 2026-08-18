//! Uploaded reference-audio decoding and normalization.

use std::io::{Cursor, ErrorKind};

use anyhow::{bail, Context, Result};
use symphonia::core::{
    audio::SampleBuffer,
    codecs::DecoderOptions,
    errors::Error as SymphoniaError,
    formats::FormatOptions,
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
    probe::Hint,
};
use symphonia::default::{get_codecs, get_probe};

/// Channel-major floating-point reference audio for the codec encoder.
pub(crate) struct ReferenceAudio {
    /// Samples arranged as contiguous channels.
    pub(crate) channel_major_samples: Vec<f32>,
    /// Number of audio channels.
    pub(crate) channels: usize,
    /// Samples per channel.
    pub(crate) samples_per_channel: usize,
}

/// Decode uploaded audio and normalize it to the codec sample rate and channels.
pub(crate) fn decode_reference_audio(
    bytes: Vec<u8>,
    filename: Option<&str>,
    target_sample_rate: u32,
    target_channels: usize,
) -> Result<ReferenceAudio> {
    if bytes.is_empty() {
        bail!("prompt_audio must not be empty");
    }
    let source = MediaSourceStream::new(
        Box::new(Cursor::new(bytes)),
        MediaSourceStreamOptions::default(),
    );
    let mut hint = Hint::new();
    if let Some(extension) = filename
        .and_then(|name| name.rsplit_once('.'))
        .map(|(_, extension)| extension)
    {
        hint.with_extension(extension);
    }
    let probed = get_probe()
        .format(
            &hint,
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("unsupported prompt_audio format")?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .context("prompt_audio contains no default audio track")?;
    let track_id = track.id;
    let source_sample_rate = track
        .codec_params
        .sample_rate
        .context("prompt_audio has no sample rate")?;
    let source_channels = track
        .codec_params
        .channels
        .context("prompt_audio has no channel layout")?
        .count();
    let mut decoder = get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("failed to create prompt_audio decoder")?;
    let mut interleaved = Vec::<f32>::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error)) if error.kind() == ErrorKind::UnexpectedEof => {
                break;
            }
            Err(error) => return Err(error).context("failed to read prompt_audio packet"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = decoder
            .decode(&packet)
            .context("failed to decode prompt_audio packet")?;
        let mut samples = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        samples.copy_interleaved_ref(decoded);
        interleaved.extend_from_slice(samples.samples());
    }

    if interleaved.is_empty() || source_channels == 0 {
        bail!("prompt_audio decoded to no samples");
    }
    if !interleaved.len().is_multiple_of(source_channels) {
        bail!("prompt_audio decoded to an incomplete audio frame");
    }
    let source_frames = interleaved.len() / source_channels;
    let converted_channels = convert_channels(
        &interleaved,
        source_frames,
        source_channels,
        target_channels,
    )?;
    let channel_major_samples = converted_channels
        .into_iter()
        .flat_map(|channel| resample_linear(&channel, source_sample_rate, target_sample_rate))
        .collect::<Vec<_>>();
    let samples_per_channel = channel_major_samples
        .len()
        .checked_div(target_channels)
        .context("target channel count must be positive")?;
    if samples_per_channel == 0 {
        bail!("prompt_audio is too short");
    }

    Ok(ReferenceAudio {
        channel_major_samples,
        channels: target_channels,
        samples_per_channel,
    })
}

fn convert_channels(
    interleaved: &[f32],
    frames: usize,
    source_channels: usize,
    target_channels: usize,
) -> Result<Vec<Vec<f32>>> {
    if target_channels == 0 {
        bail!("codec target channel count must be positive");
    }
    if source_channels != target_channels && source_channels != 1 {
        bail!(
            "unsupported prompt_audio channel conversion: {source_channels} -> {target_channels}"
        );
    }
    let mut channels = vec![Vec::with_capacity(frames); target_channels];
    for frame in interleaved.chunks_exact(source_channels) {
        for (channel_index, output) in channels.iter_mut().enumerate() {
            let source_index = if source_channels == 1 {
                0
            } else {
                channel_index
            };
            output.push(frame[source_index]);
        }
    }
    Ok(channels)
}

fn resample_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate || samples.len() < 2 {
        return samples.to_vec();
    }
    let output_length =
        ((samples.len() as u64 * target_rate as u64).div_ceil(source_rate as u64)) as usize;
    let rate_ratio = source_rate as f64 / target_rate as f64;
    (0..output_length)
        .map(|index| {
            let source_position = index as f64 * rate_ratio;
            let left = (source_position.floor() as usize).min(samples.len() - 1);
            let right = (left + 1).min(samples.len() - 1);
            let fraction = (source_position - left as f64) as f32;
            samples[left] + (samples[right] - samples[left]) * fraction
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{convert_channels, resample_linear};

    #[test]
    fn mono_audio_is_repeated_to_stereo() {
        let channels = convert_channels(&[1.0, 2.0], 2, 1, 2).unwrap();
        assert_eq!(channels, vec![vec![1.0, 2.0], vec![1.0, 2.0]]);
    }

    #[test]
    fn linear_resampling_changes_frame_count() {
        let output = resample_linear(&[0.0, 1.0, 0.0, -1.0], 24_000, 48_000);
        assert_eq!(output.len(), 8);
    }
}
