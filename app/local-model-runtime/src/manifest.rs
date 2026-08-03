//! MOSS-TTS-Nano ONNX manifest loading and validation.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

/// Fully resolved model assets required by the native MOSS runtime.
pub(crate) struct ModelAssets {
    /// Browser proof-of-concept manifest supplied with the TTS model.
    pub(crate) manifest: ModelManifest,
    /// TTS prefill graph.
    pub(crate) prefill_path: PathBuf,
    /// TTS autoregressive decode graph.
    pub(crate) decode_step_path: PathBuf,
    /// TTS local sampling graph.
    pub(crate) local_fixed_sampled_frame_path: PathBuf,
    /// Audio-tokenizer full decode graph.
    pub(crate) codec_decode_full_path: PathBuf,
    /// Audio-tokenizer reference-audio encoder graph.
    pub(crate) codec_encode_path: PathBuf,
    /// SentencePiece tokenizer model.
    pub(crate) tokenizer_path: PathBuf,
    /// Ordered autoregressive decode input names after the two scalar inputs.
    pub(crate) decode_past_input_names: Vec<String>,
    /// Ordered autoregressive present-cache output names after global hidden.
    pub(crate) decode_present_output_names: Vec<String>,
    /// PCM sample rate produced by the audio tokenizer.
    pub(crate) sample_rate: u32,
    /// Channel count expected by the audio-tokenizer encoder.
    pub(crate) codec_channels: usize,
}

impl ModelAssets {
    /// Resolve and validate a MOSS ONNX snapshot rooted at `model_root`.
    pub(crate) fn load(model_root: &Path) -> Result<Self> {
        let manifest_path = resolve_manifest_path(model_root)?;
        let manifest_dir = manifest_path
            .parent()
            .context("MOSS manifest has no parent directory")?;
        let manifest: ModelManifest = read_json(&manifest_path)?;

        let tts_meta_path =
            resolve_manifest_relative_path(manifest_dir, &manifest.model_files.tts_meta)?;
        let codec_meta_path =
            resolve_manifest_relative_path(manifest_dir, &manifest.model_files.codec_meta)?;
        let tokenizer_path =
            resolve_manifest_relative_path(manifest_dir, &manifest.model_files.tokenizer_model)?;

        let tts_meta: TtsMeta = read_json(&tts_meta_path)?;
        let codec_meta: CodecMeta = read_json(&codec_meta_path)?;
        let tts_dir = tts_meta_path
            .parent()
            .context("TTS metadata has no parent directory")?;
        let codec_dir = codec_meta_path
            .parent()
            .context("codec metadata has no parent directory")?;

        if tts_meta.onnx.decode_input_names.len() < 3 || tts_meta.onnx.decode_output_names.len() < 2
        {
            bail!("MOSS decode metadata does not contain KV-cache names");
        }
        let decode_past_input_names = tts_meta.onnx.decode_input_names[2..].to_vec();
        let decode_present_output_names = tts_meta.onnx.decode_output_names[1..].to_vec();
        if decode_past_input_names.len() != decode_present_output_names.len() {
            bail!(
                "MOSS decode cache input/output count differs: {} vs {}",
                decode_past_input_names.len(),
                decode_present_output_names.len()
            );
        }

        let assets = Self {
            prefill_path: require_file(tts_dir.join(&tts_meta.files.prefill))?,
            decode_step_path: require_file(tts_dir.join(&tts_meta.files.decode_step))?,
            local_fixed_sampled_frame_path: require_file(
                tts_dir.join(&tts_meta.files.local_fixed_sampled_frame),
            )?,
            codec_decode_full_path: require_file(codec_dir.join(&codec_meta.files.decode_full))?,
            codec_encode_path: require_file(codec_dir.join(&codec_meta.files.encode))?,
            tokenizer_path: require_file(tokenizer_path)?,
            decode_past_input_names,
            decode_present_output_names,
            sample_rate: codec_meta.codec_config.sample_rate,
            codec_channels: codec_meta.codec_config.channels,
            manifest,
        };
        assets.validate()?;
        Ok(assets)
    }

    fn validate(&self) -> Result<()> {
        let config = &self.manifest.tts_config;
        if config.n_vq == 0 {
            bail!("MOSS manifest n_vq must be positive");
        }
        if config.audio_codebook_sizes.len() != config.n_vq {
            bail!(
                "MOSS manifest has {} codebook sizes for n_vq={}",
                config.audio_codebook_sizes.len(),
                config.n_vq
            );
        }
        if self.sample_rate != 48_000 {
            bail!(
                "MOSS codec sample rate must be 48000 Hz, got {}",
                self.sample_rate
            );
        }
        if self.codec_channels == 0 {
            bail!("MOSS codec channel count must be positive");
        }
        if self
            .manifest
            .builtin_voices
            .iter()
            .all(|voice| voice.prompt_audio_codes.is_empty())
        {
            bail!("MOSS manifest contains no built-in voice prompt audio codes");
        }
        Ok(())
    }
}

/// Top-level MOSS browser ONNX manifest.
#[derive(Clone, Deserialize)]
pub(crate) struct ModelManifest {
    pub(crate) model_files: ModelFiles,
    pub(crate) tts_config: TtsConfig,
    pub(crate) prompt_templates: PromptTemplates,
    #[serde(default)]
    pub(crate) generation_defaults: GenerationDefaults,
    #[serde(default)]
    pub(crate) builtin_voices: Vec<BuiltinVoice>,
}

/// Relative metadata and tokenizer paths from the model manifest.
#[derive(Clone, Deserialize)]
pub(crate) struct ModelFiles {
    pub(crate) tts_meta: String,
    pub(crate) codec_meta: String,
    pub(crate) tokenizer_model: String,
}

/// Token and codebook constants used to construct MOSS model rows.
#[derive(Clone, Deserialize)]
pub(crate) struct TtsConfig {
    pub(crate) n_vq: usize,
    pub(crate) audio_pad_token_id: i32,
    pub(crate) audio_start_token_id: i32,
    pub(crate) audio_end_token_id: i32,
    #[serde(default = "default_audio_user_slot_token_id")]
    pub(crate) audio_user_slot_token_id: i32,
    pub(crate) audio_assistant_slot_token_id: i32,
    pub(crate) audio_codebook_sizes: Vec<usize>,
}

/// Static chat-template token sequences supplied with the model.
#[derive(Clone, Deserialize)]
pub(crate) struct PromptTemplates {
    pub(crate) user_prompt_prefix_token_ids: Vec<i32>,
    pub(crate) user_prompt_after_reference_token_ids: Vec<i32>,
    pub(crate) assistant_prompt_prefix_token_ids: Vec<i32>,
}

/// Generation defaults pinned by the exported model.
#[derive(Clone, Deserialize)]
pub(crate) struct GenerationDefaults {
    #[serde(default = "default_max_new_frames")]
    pub(crate) max_new_frames: usize,
}

impl Default for GenerationDefaults {
    fn default() -> Self {
        Self {
            max_new_frames: default_max_new_frames(),
        }
    }
}

/// One built-in voice and its precomputed audio prompt codes.
#[derive(Clone, Deserialize)]
pub(crate) struct BuiltinVoice {
    pub(crate) voice: String,
    #[serde(default)]
    pub(crate) display_name: String,
    #[serde(default)]
    pub(crate) prompt_audio_codes: Vec<Vec<i32>>,
}

#[derive(Deserialize)]
struct TtsMeta {
    files: TtsFiles,
    onnx: TtsOnnxNames,
}

#[derive(Deserialize)]
struct TtsFiles {
    prefill: String,
    decode_step: String,
    local_fixed_sampled_frame: String,
}

#[derive(Deserialize)]
struct TtsOnnxNames {
    decode_input_names: Vec<String>,
    decode_output_names: Vec<String>,
}

#[derive(Deserialize)]
struct CodecMeta {
    files: CodecFiles,
    codec_config: CodecConfig,
}

#[derive(Deserialize)]
struct CodecFiles {
    encode: String,
    decode_full: String,
}

#[derive(Deserialize)]
struct CodecConfig {
    sample_rate: u32,
    channels: usize,
}

fn default_audio_user_slot_token_id() -> i32 {
    8
}

fn default_max_new_frames() -> usize {
    375
}

fn resolve_manifest_path(model_root: &Path) -> Result<PathBuf> {
    let candidates = [
        model_root.join("browser_poc_manifest.json"),
        model_root
            .join("MOSS-TTS-Nano-100M-ONNX")
            .join("browser_poc_manifest.json"),
        model_root
            .join("MOSS-TTS-Nano-ONNX-CPU")
            .join("browser_poc_manifest.json"),
    ];
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .with_context(|| {
            format!(
                "browser_poc_manifest.json not found under {}",
                model_root.display()
            )
        })
}

fn resolve_manifest_relative_path(manifest_dir: &Path, relative: &str) -> Result<PathBuf> {
    let direct = manifest_dir.join(relative);
    if direct.exists() {
        return direct
            .canonicalize()
            .with_context(|| format!("failed to resolve {}", direct.display()));
    }
    let alias = relative
        .replace("MOSS-TTS-Nano-ONNX-CPU", "MOSS-TTS-Nano-100M-ONNX")
        .replace(
            "MOSS-Audio-Tokenizer-Nano-ONNX-CPU",
            "MOSS-Audio-Tokenizer-Nano-ONNX",
        );
    let aliased = manifest_dir.join(alias);
    aliased
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", aliased.display()))
}

fn require_file(path: PathBuf) -> Result<PathBuf> {
    if !path.is_file() {
        bail!("required MOSS model file is missing: {}", path.display());
    }
    Ok(path)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse JSON {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{GenerationDefaults, TtsConfig};

    #[test]
    fn generation_defaults_deserialize_when_absent() {
        let defaults = GenerationDefaults::default();
        assert_eq!(defaults.max_new_frames, 375);
    }

    #[test]
    fn user_slot_token_has_export_compatible_default() {
        let config: TtsConfig = serde_json::from_str(
            r#"{
                "n_vq": 1,
                "audio_pad_token_id": 1024,
                "audio_start_token_id": 6,
                "audio_end_token_id": 7,
                "audio_assistant_slot_token_id": 9,
                "audio_codebook_sizes": [1024]
            }"#,
        )
        .expect("config should deserialize");
        assert_eq!(config.audio_user_slot_token_id, 8);
    }
}
