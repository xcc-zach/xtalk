//! PCM and WAV encoding helpers.

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
    use super::{encode_pcm16_mono, encode_wav_pcm16_mono};

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
