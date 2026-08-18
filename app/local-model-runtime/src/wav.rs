//! PCM and WAV encoding helpers.

const TRAILING_AMPLITUDE_THRESHOLD: f32 = 0.009;
const TRAILING_TAIL_MILLISECONDS: u32 = 40;
const TRAILING_FADE_MILLISECONDS: u32 = 30;

/// Remove codec padding and fade the low-energy tail to suppress end artifacts.
pub(crate) fn trim_and_fade_trailing_audio(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }
    let Some(last_audible_index) = samples
        .iter()
        .rposition(|sample| sample.abs() >= TRAILING_AMPLITUDE_THRESHOLD)
    else {
        return Vec::new();
    };
    let tail_samples =
        usize::try_from(u64::from(sample_rate) * u64::from(TRAILING_TAIL_MILLISECONDS) / 1_000)
            .expect("trailing tail sample count exceeds usize");
    let end_index = samples.len().min(
        last_audible_index
            .saturating_add(1)
            .saturating_add(tail_samples),
    );
    let mut result = samples[..end_index].to_vec();
    let fade_samples =
        usize::try_from(u64::from(sample_rate) * u64::from(TRAILING_FADE_MILLISECONDS) / 1_000)
            .expect("trailing fade sample count exceeds usize");
    if fade_samples == 0 || result.is_empty() {
        return result;
    }

    let quiet_tail_start = (last_audible_index + 1).min(result.len());
    let fade_start = if quiet_tail_start < result.len() {
        quiet_tail_start
    } else {
        result.len().saturating_sub(fade_samples)
    };
    let fade_length = result.len() - fade_start;
    for (offset, sample) in result[fade_start..].iter_mut().enumerate() {
        let gain = (fade_length - offset - 1) as f32 / fade_length as f32;
        *sample *= gain;
    }
    result
}

/// Encode normalized mono floating-point samples as little-endian PCM16.
pub(crate) fn encode_pcm16_mono(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Wrap normalized mono floating-point samples in a PCM16 WAV container.
pub(crate) fn encode_wav_pcm16_mono(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let pcm = encode_pcm16_mono(samples);
    let data_size = u32::try_from(pcm.len()).expect("PCM payload exceeds WAV size limit");
    let riff_size = 36_u32
        .checked_add(data_size)
        .expect("WAV RIFF size overflow");

    let mut bytes = Vec::with_capacity(44 + pcm.len());
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2_u16.to_le_bytes());
    bytes.extend_from_slice(&16_u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_size.to_le_bytes());
    bytes.extend_from_slice(&pcm);
    bytes
}

#[cfg(test)]
mod tests {
    use super::{encode_pcm16_mono, encode_wav_pcm16_mono, trim_and_fade_trailing_audio};

    #[test]
    fn trims_and_fades_codec_tail() {
        let mut samples = vec![0.0; 10];
        samples.push(0.02);
        samples.extend(vec![0.0; 1_000]);
        let trimmed = trim_and_fade_trailing_audio(&samples, 1_000);
        assert_eq!(trimmed.len(), 51);
        assert_eq!(trimmed[10], 0.02);
        assert_eq!(trimmed.last(), Some(&0.0));
    }

    #[test]
    fn fades_frame_limited_audio_without_appending_samples() {
        let samples = vec![0.02; 100];
        let trimmed = trim_and_fade_trailing_audio(&samples, 1_000);
        assert_eq!(trimmed.len(), samples.len());
        assert_eq!(trimmed[69], 0.02);
        assert_eq!(trimmed.last(), Some(&0.0));
    }

    #[test]
    fn pcm_encoding_clamps_samples() {
        let pcm = encode_pcm16_mono(&[-2.0, 0.0, 2.0]);
        assert_eq!(pcm.len(), 6);
        assert_eq!(i16::from_le_bytes([pcm[0], pcm[1]]), -32767);
        assert_eq!(i16::from_le_bytes([pcm[4], pcm[5]]), 32767);
    }

    #[test]
    fn wav_encoding_has_expected_header() {
        let wav = encode_wav_pcm16_mono(&[0.0, 0.25], 48_000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 4);
    }
}
