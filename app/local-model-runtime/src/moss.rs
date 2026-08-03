//! Native MOSS-TTS-Nano inference implemented with ONNX Runtime.

use std::{collections::HashSet, path::Path, time::Instant};

use anyhow::{bail, Context, Result};
use ort::{
    session::{builder::GraphOptimizationLevel, Session, SessionInputValue, SessionOutputs},
    value::{DynValue, Tensor},
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use sentencepiece::SentencePieceProcessor;

use crate::manifest::{BuiltinVoice, ModelAssets, TtsConfig};
use crate::OnnxBackend;
use crate::{audio::ReferenceAudio, wav::trim_and_fade_trailing_audio};

/// Request parameters accepted by the native MOSS inference engine.
pub(crate) struct SynthesisOptions<'a> {
    /// UTF-8 text to synthesize.
    pub(crate) text: &'a str,
    /// Built-in voice identifier.
    pub(crate) voice: &'a str,
    /// Pre-encoded reference-audio codes, overriding the built-in voice.
    pub(crate) prompt_audio_codes: Option<&'a [Vec<i32>]>,
    /// Maximum number of generated audio-token frames.
    pub(crate) max_frames: usize,
    /// Deterministic random seed used by the exported sampling graph.
    pub(crate) seed: u64,
}

/// Synthesized mono floating-point PCM and generation metadata.
pub(crate) struct SynthesisOutput {
    /// Normalized mono samples.
    pub(crate) samples: Vec<f32>,
    /// Sample rate of `samples`.
    pub(crate) sample_rate: u32,
    /// Number of generated audio-token frames.
    pub(crate) generated_frames: usize,
    /// Wall-clock inference duration.
    pub(crate) elapsed_ms: u128,
}

/// Loaded MOSS-TTS-Nano ONNX sessions and tokenizer.
pub(crate) struct MossEngine {
    assets: ModelAssets,
    tokenizer: SentencePieceProcessor,
    prefill_session: Session,
    decode_session: Session,
    local_fixed_frame_session: Session,
    codec_encode_session: Session,
    codec_decode_session: Session,
}

impl MossEngine {
    /// Load all MOSS ONNX sessions from a model snapshot.
    pub(crate) fn load(
        model_root: &Path,
        cpu_threads: usize,
        backend: OnnxBackend,
    ) -> Result<Self> {
        let assets = ModelAssets::load(model_root)?;
        let tokenizer =
            SentencePieceProcessor::open(&assets.tokenizer_path).with_context(|| {
                format!(
                    "failed to load SentencePiece model {}",
                    assets.tokenizer_path.display()
                )
            })?;
        let cpu_threads = cpu_threads.max(1);
        let prefill_session = create_session(&assets.prefill_path, cpu_threads, backend)?;
        let decode_session = create_session(&assets.decode_step_path, cpu_threads, backend)?;
        let local_fixed_frame_session =
            create_session(&assets.local_fixed_sampled_frame_path, cpu_threads, backend)?;
        let codec_encode_session = create_session(&assets.codec_encode_path, cpu_threads, backend)?;
        let codec_decode_session =
            create_session(&assets.codec_decode_full_path, cpu_threads, backend)?;

        Ok(Self {
            assets,
            tokenizer,
            prefill_session,
            decode_session,
            local_fixed_frame_session,
            codec_encode_session,
            codec_decode_session,
        })
    }

    /// Return built-in voices with precomputed reference-audio codes.
    pub(crate) fn voices(&self) -> Vec<(&str, &str)> {
        self.assets
            .manifest
            .builtin_voices
            .iter()
            .filter(|voice| !voice.prompt_audio_codes.is_empty())
            .map(|voice| (voice.voice.as_str(), voice.display_name.as_str()))
            .collect()
    }

    /// Return the maximum frame count supported by the exported model.
    pub(crate) fn max_new_frames(&self) -> usize {
        self.assets.manifest.generation_defaults.max_new_frames
    }

    /// Return the fixed reference-audio sample rate expected by the codec.
    pub(crate) fn reference_sample_rate(&self) -> u32 {
        self.assets.sample_rate
    }

    /// Return the reference-audio channel count expected by the codec.
    pub(crate) fn reference_channels(&self) -> usize {
        self.assets.codec_channels
    }

    /// Synthesize one text request into 48 kHz mono PCM samples.
    pub(crate) fn synthesize(&mut self, options: SynthesisOptions<'_>) -> Result<SynthesisOutput> {
        let text = options.text.trim();
        if text.is_empty() {
            bail!("input text must not be empty");
        }
        let max_frames = options
            .max_frames
            .min(self.assets.manifest.generation_defaults.max_new_frames);
        if max_frames == 0 {
            bail!("max_frames must be positive");
        }

        let started = Instant::now();
        let text_token_ids = self
            .tokenizer
            .encode(text)
            .context("SentencePiece tokenization failed")?
            .into_iter()
            .map(|piece| i32::try_from(piece.id).context("token id exceeds i32"))
            .collect::<Result<Vec<_>>>()?;
        if text_token_ids.is_empty() {
            bail!("tokenizer produced no text tokens");
        }

        let input_rows =
            self.build_input_rows(&text_token_ids, options.voice, options.prompt_audio_codes)?;
        let (mut global_hidden, mut caches, mut past_valid_lengths) =
            self.run_prefill(&input_rows)?;
        let audio_tokens = self.run_decode(
            &mut global_hidden,
            &mut caches,
            &mut past_valid_lengths,
            max_frames,
            options.seed,
        )?;
        let decoded_samples = self.decode_audio_tokens(&audio_tokens)?;
        let samples = trim_and_fade_trailing_audio(&decoded_samples, self.assets.sample_rate);
        if samples.is_empty() {
            bail!("MOSS decoded only inaudible audio");
        }

        Ok(SynthesisOutput {
            samples,
            sample_rate: self.assets.sample_rate,
            generated_frames: audio_tokens.len(),
            elapsed_ms: started.elapsed().as_millis(),
        })
    }

    /// Encode normalized reference audio into MOSS prompt codes.
    pub(crate) fn encode_reference_audio(
        &mut self,
        reference: &ReferenceAudio,
    ) -> Result<Vec<Vec<i32>>> {
        let waveform = Tensor::from_array((
            [1, reference.channels, reference.samples_per_channel],
            reference.channel_major_samples.clone(),
        ))?;
        let input_lengths = Tensor::from_array((
            [1],
            vec![i32::try_from(reference.samples_per_channel)
                .context("reference audio length exceeds i32")?],
        ))?;
        let outputs = self.codec_encode_session.run(ort::inputs! {
            "waveform" => waveform,
            "input_lengths" => input_lengths,
        })?;
        let (codes_shape, codes) = outputs["audio_codes"]
            .try_extract_tensor::<i32>()
            .context("codec audio_codes output is not int32")?;
        if codes_shape.len() != 3 || codes_shape[0] != 1 {
            bail!("unexpected codec audio_codes shape: {codes_shape:?}");
        }
        let frame_capacity =
            usize::try_from(codes_shape[1]).context("invalid audio code frame count")?;
        let quantizers =
            usize::try_from(codes_shape[2]).context("invalid audio code quantizer count")?;
        if quantizers != self.assets.manifest.tts_config.n_vq {
            bail!(
                "codec returned {quantizers} quantizers, expected {}",
                self.assets.manifest.tts_config.n_vq
            );
        }
        let (_, reported_lengths) = outputs["audio_code_lengths"]
            .try_extract_tensor::<i32>()
            .context("codec audio_code_lengths output is not int32")?;
        let frame_count = usize::try_from(
            reported_lengths
                .first()
                .copied()
                .context("codec returned empty audio_code_lengths")?,
        )
        .context("codec returned a negative audio code length")?
        .min(frame_capacity);
        if frame_count == 0 {
            bail!("codec encoded no reference-audio frames");
        }
        Ok((0..frame_count)
            .map(|frame| {
                let start = frame * quantizers;
                codes[start..start + quantizers].to_vec()
            })
            .collect())
    }

    fn build_input_rows(
        &self,
        text_token_ids: &[i32],
        voice: &str,
        prompt_audio_codes: Option<&[Vec<i32>]>,
    ) -> Result<Vec<Vec<i32>>> {
        let config = &self.assets.manifest.tts_config;
        let row_width = config.n_vq + 1;
        let builtin_voice;
        let prompt_audio_codes = if let Some(codes) = prompt_audio_codes {
            if codes.is_empty() {
                bail!("reference audio produced no prompt codes");
            }
            codes
        } else {
            builtin_voice = select_builtin_voice(&self.assets.manifest.builtin_voices, voice)?;
            &builtin_voice.prompt_audio_codes
        };

        let mut prefix_tokens = self
            .assets
            .manifest
            .prompt_templates
            .user_prompt_prefix_token_ids
            .clone();
        prefix_tokens.push(config.audio_start_token_id);

        let mut suffix_tokens = vec![config.audio_end_token_id];
        suffix_tokens.extend_from_slice(
            &self
                .assets
                .manifest
                .prompt_templates
                .user_prompt_after_reference_token_ids,
        );
        suffix_tokens.extend_from_slice(text_token_ids);
        suffix_tokens.extend_from_slice(
            &self
                .assets
                .manifest
                .prompt_templates
                .assistant_prompt_prefix_token_ids,
        );
        suffix_tokens.push(config.audio_start_token_id);

        let mut rows = build_text_rows(&prefix_tokens, config, row_width);
        rows.extend(build_audio_rows(prompt_audio_codes, config, row_width));
        rows.extend(build_text_rows(&suffix_tokens, config, row_width));
        Ok(rows)
    }

    fn run_prefill(&mut self, input_rows: &[Vec<i32>]) -> Result<(DynValue, Vec<DynValue>, i32)> {
        let sequence_length = input_rows.len();
        let row_width = input_rows[0].len();
        let input_ids = input_rows
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
        let attention_mask = vec![1_i32; sequence_length];

        let input_ids_tensor = Tensor::from_array(([1, sequence_length, row_width], input_ids))?;
        let attention_mask_tensor = Tensor::from_array(([1, sequence_length], attention_mask))?;
        let mut outputs = self.prefill_session.run(ort::inputs! {
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        })?;
        let hidden_output = remove_output(&mut outputs, "global_hidden")?;
        let global_hidden = extract_last_hidden(hidden_output)?;
        let caches = remove_outputs(&mut outputs, &self.assets.decode_present_output_names)?;
        let past_valid_lengths =
            i32::try_from(sequence_length).context("prompt sequence exceeds i32")?;
        Ok((global_hidden, caches, past_valid_lengths))
    }

    fn run_decode(
        &mut self,
        global_hidden: &mut DynValue,
        caches: &mut Vec<DynValue>,
        past_valid_lengths: &mut i32,
        max_frames: usize,
        seed: u64,
    ) -> Result<Vec<Vec<i32>>> {
        let config = &self.assets.manifest.tts_config;
        let row_width = config.n_vq + 1;
        let mut audio_tokens = Vec::new();
        let mut seen_tokens = vec![HashSet::<i32>::new(); config.n_vq];
        let mut random = StdRng::seed_from_u64(seed);

        for _ in 0..max_frames {
            let frame = run_local_fixed_sampled_frame(
                &mut self.local_fixed_frame_session,
                global_hidden,
                config,
                &seen_tokens,
                &mut random,
            )?;
            let Some(frame) = frame else {
                break;
            };
            if frame.len() < config.n_vq {
                bail!(
                    "sampling graph returned {} tokens for n_vq={}",
                    frame.len(),
                    config.n_vq
                );
            }

            let mut audio_row = vec![config.audio_pad_token_id; row_width];
            audio_row[0] = config.audio_assistant_slot_token_id;
            for quantizer in 0..config.n_vq {
                let token = frame[quantizer];
                audio_row[quantizer + 1] = token;
                seen_tokens[quantizer].insert(token);
            }
            audio_tokens.push(frame[..config.n_vq].to_vec());

            let input_ids_tensor = Tensor::from_array(([1, 1, row_width], audio_row))?;
            let past_lengths_tensor = Tensor::from_array(([1], vec![*past_valid_lengths]))?;
            let mut inputs: Vec<(String, SessionInputValue<'_>)> = vec![
                (
                    "input_ids".to_owned(),
                    SessionInputValue::from(input_ids_tensor),
                ),
                (
                    "past_valid_lengths".to_owned(),
                    SessionInputValue::from(past_lengths_tensor),
                ),
            ];
            for (name, cache) in self
                .assets
                .decode_past_input_names
                .iter()
                .zip(caches.iter())
            {
                inputs.push((name.clone(), SessionInputValue::from(cache)));
            }

            let mut outputs = self.decode_session.run(inputs)?;
            let next_hidden = extract_last_hidden(remove_output(&mut outputs, "global_hidden")?)?;
            let next_caches =
                remove_outputs(&mut outputs, &self.assets.decode_present_output_names)?;
            *global_hidden = next_hidden;
            *caches = next_caches;
            *past_valid_lengths = past_valid_lengths
                .checked_add(1)
                .context("past_valid_lengths overflow")?;
        }

        if audio_tokens.is_empty() {
            bail!("MOSS generated no audio frames");
        }
        Ok(audio_tokens)
    }

    fn decode_audio_tokens(&mut self, audio_tokens: &[Vec<i32>]) -> Result<Vec<f32>> {
        let num_frames = audio_tokens.len();
        let num_quantizers = self.assets.manifest.tts_config.n_vq;
        let codes = audio_tokens
            .iter()
            .flat_map(|frame| frame.iter().take(num_quantizers).copied())
            .collect::<Vec<_>>();
        let codes_tensor = Tensor::from_array(([1, num_frames, num_quantizers], codes))?;
        let lengths_tensor = Tensor::from_array((
            [1],
            vec![i32::try_from(num_frames).context("frame count exceeds i32")?],
        ))?;
        let outputs = self.codec_decode_session.run(ort::inputs! {
            "audio_codes" => codes_tensor,
            "audio_code_lengths" => lengths_tensor,
        })?;

        let (audio_shape, audio) = outputs["audio"]
            .try_extract_tensor::<f32>()
            .context("codec audio output is not float32")?;
        if audio_shape.len() != 3 || audio_shape[0] != 1 {
            bail!("unexpected codec audio shape: {audio_shape:?}");
        }
        let channels = usize::try_from(audio_shape[1]).context("invalid channel count")?;
        let samples_per_channel =
            usize::try_from(audio_shape[2]).context("invalid sample count")?;
        if channels == 0 {
            bail!("codec returned no audio channels");
        }
        let (_, reported_lengths) = outputs["audio_lengths"]
            .try_extract_tensor::<i32>()
            .context("codec audio_lengths output is not int32")?;
        let reported_length = reported_lengths
            .first()
            .copied()
            .context("codec returned empty audio_lengths")?;
        let length = usize::try_from(reported_length)
            .context("codec returned a negative audio length")?
            .min(samples_per_channel);

        let mut mono = vec![0.0_f32; length];
        for channel in 0..channels {
            let start = channel * samples_per_channel;
            let channel_samples = &audio[start..start + length];
            for (target, sample) in mono.iter_mut().zip(channel_samples) {
                *target += *sample / channels as f32;
            }
        }
        Ok(mono)
    }
}

fn create_session(path: &Path, cpu_threads: usize, backend: OnnxBackend) -> Result<Session> {
    let builder = Session::builder()?;
    let builder = match backend {
        OnnxBackend::Cpu => builder,
        OnnxBackend::Cuda => builder
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
            .map_err(|error| anyhow::anyhow!("{error}"))?,
    };
    let builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let builder = builder
        .with_intra_threads(cpu_threads)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let mut builder = builder
        .with_inter_threads(1)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    builder
        .commit_from_file(path)
        .with_context(|| format!("failed to load ONNX graph {}", path.display()))
}

fn build_text_rows(tokens: &[i32], config: &TtsConfig, row_width: usize) -> Vec<Vec<i32>> {
    tokens
        .iter()
        .map(|token| {
            let mut row = vec![config.audio_pad_token_id; row_width];
            row[0] = *token;
            row
        })
        .collect()
}

fn build_audio_rows(
    audio_codes: &[Vec<i32>],
    config: &TtsConfig,
    row_width: usize,
) -> Vec<Vec<i32>> {
    audio_codes
        .iter()
        .map(|codes| {
            let mut row = vec![config.audio_pad_token_id; row_width];
            row[0] = config.audio_user_slot_token_id;
            for (index, code) in codes.iter().take(config.n_vq).enumerate() {
                row[index + 1] = *code;
            }
            row
        })
        .collect()
}

fn select_builtin_voice<'a>(
    voices: &'a [BuiltinVoice],
    requested: &str,
) -> Result<&'a BuiltinVoice> {
    voices
        .iter()
        .find(|voice| voice.voice == requested && !voice.prompt_audio_codes.is_empty())
        .or_else(|| {
            voices
                .iter()
                .find(|voice| !voice.prompt_audio_codes.is_empty())
        })
        .context("no built-in MOSS voice has prompt audio codes")
}

fn run_local_fixed_sampled_frame(
    session: &mut Session,
    global_hidden: &DynValue,
    config: &TtsConfig,
    seen_tokens: &[HashSet<i32>],
    random: &mut StdRng,
) -> Result<Option<Vec<i32>>> {
    let codebook_size = config.audio_codebook_sizes[0];
    let mut seen_mask = vec![0_i32; config.n_vq * codebook_size];
    for (channel, tokens) in seen_tokens.iter().enumerate() {
        let channel_offset = channel * codebook_size;
        for token in tokens {
            if let Ok(token_index) = usize::try_from(*token) {
                if token_index < codebook_size {
                    seen_mask[channel_offset + token_index] = 1;
                }
            }
        }
    }

    let assistant_random = random.random_range(0.000_001_f32..0.999_999_f32);
    let audio_random = (0..config.n_vq)
        .map(|_| random.random_range(0.000_001_f32..0.999_999_f32))
        .collect::<Vec<_>>();
    let seen_tensor = Tensor::from_array(([1, config.n_vq, codebook_size], seen_mask))?;
    let assistant_tensor = Tensor::from_array(([1], vec![assistant_random]))?;
    let audio_tensor = Tensor::from_array(([1, config.n_vq], audio_random))?;
    let outputs = session.run(ort::inputs! {
        "global_hidden" => global_hidden,
        "repetition_seen_mask" => seen_tensor,
        "assistant_random_u" => assistant_tensor,
        "audio_random_u" => audio_tensor,
    })?;

    let (_, should_continue) = outputs["should_continue"]
        .try_extract_tensor::<i32>()
        .context("should_continue output is not int32")?;
    if should_continue.first().copied().unwrap_or_default() <= 0 {
        return Ok(None);
    }
    let (_, frame) = outputs["frame_token_ids"]
        .try_extract_tensor::<i32>()
        .context("frame_token_ids output is not int32")?;
    Ok(Some(frame.to_vec()))
}

fn extract_last_hidden(value: DynValue) -> Result<DynValue> {
    let (shape, hidden_values) = value
        .try_extract_tensor::<f32>()
        .context("global_hidden output is not float32")?;
    if !(shape.len() == 2 || shape.len() == 3) {
        bail!("unexpected global_hidden shape: {shape:?}");
    }
    let hidden_size = usize::try_from(*shape.last().context("global_hidden shape is empty")?)
        .context("invalid global_hidden size")?;
    if hidden_values.len() < hidden_size {
        bail!("global_hidden output contains too few values");
    }
    let last_hidden = hidden_values[hidden_values.len() - hidden_size..].to_vec();
    Ok(Tensor::from_array(([1, hidden_size], last_hidden))?
        .upcast()
        .into())
}

fn remove_output(outputs: &mut SessionOutputs<'_>, name: &str) -> Result<DynValue> {
    outputs
        .remove(name)
        .with_context(|| format!("ONNX output is missing: {name}"))
}

fn remove_outputs(outputs: &mut SessionOutputs<'_>, names: &[String]) -> Result<Vec<DynValue>> {
    names
        .iter()
        .map(|name| remove_output(outputs, name))
        .collect()
}
