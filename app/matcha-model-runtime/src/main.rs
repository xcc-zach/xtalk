//! Local HTTP service for sherpa-onnx Matcha Chinese-English TTS.

use std::{
    io::{Cursor, Write},
    net::IpAddr,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{bail, Context, Result};
use axum::{
    body::Body,
    extract::State,
    http::{header::CONTENT_TYPE, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsMatchaModelConfig,
    OfflineTtsModelConfig,
};
use tempfile::TempDir;

const ENGINE_ID: &str = "matcha-icefall-zh-en";
const PROTOCOL_VERSION: u16 = 1;
const DEFAULT_OUTPUT_SAMPLE_RATE: i32 = 48_000;
const MAX_TEXT_CHARS: usize = 4_096;

/// Command-line settings for the Matcha HTTP sidecar.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Root containing the Matcha model directory and vocoder.
    #[arg(long)]
    model_root: PathBuf,

    /// Loopback address used by the managed service.
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,

    /// Listening port. Use zero to let the operating system select one.
    #[arg(long, default_value_t = 0)]
    port: u16,

    /// ONNX Runtime execution provider.
    #[arg(long, value_enum, default_value_t = Backend::Cpu)]
    backend: Backend,

    /// Number of native inference threads.
    #[arg(long, default_value_t = 2)]
    num_threads: i32,
}

/// ONNX Runtime backend names accepted by sherpa-onnx.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    /// CPU execution provider.
    Cpu,
    /// CUDA execution provider.
    Cuda,
}

impl Backend {
    /// Return the provider string expected by sherpa-onnx.
    fn provider(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

/// Shared inference state used by HTTP handlers.
#[derive(Clone)]
struct AppState {
    engine: Arc<OfflineTts>,
    native_sample_rate: i32,
    num_speakers: i32,
    _espeak_data_alias: Option<Arc<TempDir>>,
}

/// Matcha engine and filesystem resources that must outlive native inference.
struct CreatedEngine {
    engine: OfflineTts,
    espeak_data_alias: Option<TempDir>,
}

/// OpenAI-compatible speech synthesis request.
#[derive(Debug, Deserialize)]
struct SpeechRequest {
    #[serde(default = "default_model")]
    model: String,
    input: String,
    #[serde(default = "default_voice")]
    voice: String,
    #[serde(default = "default_response_format")]
    response_format: String,
    #[serde(default = "default_speed")]
    speed: f32,
    #[serde(default = "default_sample_rate")]
    sample_rate: i32,
}

/// One built-in voice exposed to compatible clients.
#[derive(Serialize)]
struct VoiceDescription {
    id: String,
    name: String,
}

/// Successfully synthesized samples copied out of the native object.
struct SynthesizedAudio {
    samples: Vec<f32>,
    sample_rate: i32,
}

/// HTTP error with a stable status code and readable message.
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    /// Construct a client-input error.
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    /// Construct an inference failure.
    fn inference(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({"error": self.message}))).into_response()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();
    if args.num_threads <= 0 {
        bail!("--num-threads must be positive");
    }
    let created = create_engine(&args)?;
    let state = AppState {
        native_sample_rate: created.engine.sample_rate(),
        num_speakers: created.engine.num_speakers(),
        engine: Arc::new(created.engine),
        _espeak_data_alias: created.espeak_data_alias.map(Arc::new),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/voices", get(voices))
        .route("/v1/audio/speech", post(synthesize))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind((args.host, args.port))
        .await
        .context("failed to bind Matcha HTTP listener")?;
    let port = listener.local_addr()?.port();

    println!(
        "{}",
        json!({
            "status": "ready",
            "protocol_version": PROTOCOL_VERSION,
            "engine": ENGINE_ID,
            "port": port
        })
    );
    std::io::stdout().flush()?;
    axum::serve(listener, app)
        .await
        .context("Matcha HTTP server failed")?;
    Ok(())
}

/// Create a Matcha engine from one managed model installation.
fn create_engine(args: &Args) -> Result<CreatedEngine> {
    let model_dir = args.model_root.join(ENGINE_ID);
    let acoustic_model = require_file(&model_dir.join("model-steps-3.onnx"))?;
    let vocoder = require_file(&args.model_root.join("vocos-16khz-univ.onnx"))?;
    let lexicon = require_file(&model_dir.join("lexicon.txt"))?;
    let tokens = require_file(&model_dir.join("tokens.txt"))?;
    let data_dir = require_directory(&model_dir.join("espeak-ng-data"))?;
    let (data_dir, espeak_data_alias) = prepare_espeak_data_dir(&data_dir)?;
    let rule_fsts = ["date-zh.fst", "number-zh.fst", "phone-zh.fst"]
        .into_iter()
        .map(|name| require_file(&model_dir.join(name)))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(",");

    let config = OfflineTtsConfig {
        model: OfflineTtsModelConfig {
            matcha: OfflineTtsMatchaModelConfig {
                acoustic_model: Some(acoustic_model.to_string_lossy().into_owned()),
                vocoder: Some(vocoder.to_string_lossy().into_owned()),
                lexicon: Some(lexicon.to_string_lossy().into_owned()),
                tokens: Some(tokens.to_string_lossy().into_owned()),
                data_dir: Some(data_dir.to_string_lossy().into_owned()),
                ..Default::default()
            },
            num_threads: args.num_threads,
            provider: Some(args.backend.provider().to_owned()),
            ..Default::default()
        },
        rule_fsts: Some(rule_fsts),
        max_num_sentences: 1,
        ..Default::default()
    };
    let engine =
        OfflineTts::create(&config).context("sherpa-onnx failed to create the Matcha engine")?;
    Ok(CreatedEngine {
        engine,
        espeak_data_alias,
    })
}

/// Give espeak-ng a whitespace-free path on Unix hosts.
///
/// sherpa-onnx currently falls back to its compiled `/usr/share` path when the
/// configured espeak data path contains whitespace. macOS stores managed
/// models below `Application Support`, so expose only this directory through a
/// process-scoped symbolic link and retain the temporary directory while the
/// native engine is alive.
fn prepare_espeak_data_dir(source: &Path) -> Result<(PathBuf, Option<TempDir>)> {
    if !source.to_string_lossy().chars().any(char::is_whitespace) {
        return Ok((source.to_path_buf(), None));
    }

    #[cfg(unix)]
    {
        let directory = tempfile::Builder::new()
            .prefix("xtalk-matcha-espeak-")
            .tempdir_in("/tmp")
            .context("failed to create a temporary espeak data alias")?;
        let alias = directory.path().join("espeak-ng-data");
        std::os::unix::fs::symlink(source, &alias)
            .context("failed to link the managed espeak data directory")?;
        return Ok((alias, Some(directory)));
    }

    #[cfg(not(unix))]
    Ok((source.to_path_buf(), None))
}

/// Validate that one required model file exists.
fn require_file(path: &Path) -> Result<PathBuf> {
    if !path.is_file() {
        bail!("required Matcha model file is missing: {}", path.display());
    }
    Ok(path.to_path_buf())
}

/// Validate that one required model directory exists.
fn require_directory(path: &Path) -> Result<PathBuf> {
    if !path.is_dir() {
        bail!(
            "required Matcha model directory is missing: {}",
            path.display()
        );
    }
    Ok(path.to_path_buf())
}

/// Return sidecar and native-model health metadata.
async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "engine": ENGINE_ID,
        "protocol_version": PROTOCOL_VERSION,
        "native_sample_rate": state.native_sample_rate,
        "output_sample_rate": DEFAULT_OUTPUT_SAMPLE_RATE,
        "num_speakers": state.num_speakers
    }))
}

/// Return the built-in speaker identifiers.
async fn voices(State(state): State<AppState>) -> Json<serde_json::Value> {
    let count = state.num_speakers.max(1);
    let voices = (0..count)
        .map(|sid| VoiceDescription {
            id: sid.to_string(),
            name: format!("Speaker {sid}"),
        })
        .collect::<Vec<_>>();
    Json(json!({"voices": voices}))
}

/// Synthesize one request into WAV or raw PCM16 audio.
async fn synthesize(
    State(state): State<AppState>,
    Json(request): Json<SpeechRequest>,
) -> Result<Response, ApiError> {
    validate_request(&request, state.num_speakers)?;
    let sid = request
        .voice
        .parse::<i32>()
        .map_err(|_| ApiError::bad_request("voice must be a numeric speaker ID"))?;
    let generation = GenerationConfig {
        speed: request.speed,
        sid,
        silence_scale: 0.2,
        ..Default::default()
    };
    let engine = state.engine.clone();
    let text = request.input.trim().to_owned();
    let generated = tokio::task::spawn_blocking(move || {
        let audio = engine
            .generate_with_config(&text, &generation, None::<fn(&[f32], f32) -> bool>)
            .ok_or_else(|| "sherpa-onnx returned no generated audio".to_owned())?;
        if audio.samples().is_empty() || audio.sample_rate() <= 0 {
            return Err("sherpa-onnx returned empty generated audio".to_owned());
        }
        Ok::<SynthesizedAudio, String>(SynthesizedAudio {
            samples: audio.samples().to_vec(),
            sample_rate: audio.sample_rate(),
        })
    })
    .await
    .map_err(|error| ApiError::inference(format!("Matcha task failed: {error}")))?
    .map_err(ApiError::inference)?;

    let resampled = resample_linear(
        &generated.samples,
        generated.sample_rate,
        request.sample_rate,
    );
    let pcm = pcm16_bytes(&resampled);
    let (content_type, body) = match request.response_format.as_str() {
        "wav" => (
            "audio/wav",
            wav_bytes(&pcm, request.sample_rate).map_err(ApiError::inference)?,
        ),
        "pcm" => ("audio/pcm", pcm),
        _ => unreachable!("response format was validated"),
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, content_type)
        .body(Body::from(body))
        .map_err(|error| ApiError::inference(error.to_string()))
}

/// Validate all request fields before entering native inference.
fn validate_request(request: &SpeechRequest, num_speakers: i32) -> Result<(), ApiError> {
    if request.model != ENGINE_ID {
        return Err(ApiError::bad_request(format!(
            "unsupported model `{}`",
            request.model
        )));
    }
    let text = request.input.trim();
    if text.is_empty() {
        return Err(ApiError::bad_request("input must not be empty"));
    }
    if text.chars().count() > MAX_TEXT_CHARS {
        return Err(ApiError::bad_request(format!(
            "input exceeds the {MAX_TEXT_CHARS}-character limit"
        )));
    }
    if !request.speed.is_finite() || !(0.25..=4.0).contains(&request.speed) {
        return Err(ApiError::bad_request("speed must be between 0.25 and 4.0"));
    }
    if !(8_000..=192_000).contains(&request.sample_rate) {
        return Err(ApiError::bad_request(
            "sample_rate must be between 8000 and 192000",
        ));
    }
    if !matches!(request.response_format.as_str(), "wav" | "pcm") {
        return Err(ApiError::bad_request(
            "response_format must be `wav` or `pcm`",
        ));
    }
    let sid = request
        .voice
        .parse::<i32>()
        .map_err(|_| ApiError::bad_request("voice must be a numeric speaker ID"))?;
    let speaker_count = num_speakers.max(1);
    if sid < 0 || sid >= speaker_count {
        return Err(ApiError::bad_request(format!(
            "voice must be between 0 and {}",
            speaker_count - 1
        )));
    }
    Ok(())
}

/// Resample normalized mono samples with linear interpolation.
fn resample_linear(samples: &[f32], source_rate: i32, target_rate: i32) -> Vec<f32> {
    if samples.is_empty() || source_rate == target_rate {
        return samples.to_vec();
    }
    let output_len =
        ((samples.len() as f64 * target_rate as f64 / source_rate as f64).round() as usize).max(1);
    if samples.len() == 1 {
        return vec![samples[0]; output_len];
    }
    let scale = (samples.len() - 1) as f64 / (output_len.saturating_sub(1).max(1)) as f64;
    (0..output_len)
        .map(|index| {
            let position = index as f64 * scale;
            let left = position.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let fraction = (position - left as f64) as f32;
            samples[left] + (samples[right] - samples[left]) * fraction
        })
        .collect()
}

/// Convert normalized floating-point samples to little-endian PCM16 bytes.
fn pcm16_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Wrap raw mono PCM16 bytes in a complete WAV container.
fn wav_bytes(pcm: &[u8], sample_rate: i32) -> Result<Vec<u8>, String> {
    let mut output = Vec::with_capacity(pcm.len() + 44);
    let cursor = Cursor::new(&mut output);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(cursor, spec).map_err(|error| error.to_string())?;
    for sample in pcm.chunks_exact(2) {
        writer
            .write_sample(i16::from_le_bytes([sample[0], sample[1]]))
            .map_err(|error| error.to_string())?;
    }
    writer.finalize().map_err(|error| error.to_string())?;
    Ok(output)
}

/// Return the managed model name used when the request omits it.
fn default_model() -> String {
    ENGINE_ID.to_owned()
}

/// Return the first Matcha speaker ID.
fn default_voice() -> String {
    "0".to_owned()
}

/// Return the default container used by the Python client.
fn default_response_format() -> String {
    "wav".to_owned()
}

/// Return the neutral speech speed.
fn default_speed() -> f32 {
    1.0
}

/// Return the App-wide audio sample rate.
fn default_sample_rate() -> i32 {
    DEFAULT_OUTPUT_SAMPLE_RATE
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{pcm16_bytes, prepare_espeak_data_dir, resample_linear, wav_bytes};

    #[test]
    fn resampler_preserves_duration() {
        let source = vec![0.25; 16_000];
        let output = resample_linear(&source, 16_000, 48_000);
        assert_eq!(output.len(), 48_000);
        assert!(output.iter().all(|sample| (*sample - 0.25).abs() < 1e-6));
    }

    #[test]
    fn wav_encoder_writes_pcm16_header() {
        let pcm = pcm16_bytes(&[0.0, 0.5, -0.5]);
        let wav = wav_bytes(&pcm, 48_000).expect("encode wav");
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 48_000);
    }

    #[cfg(unix)]
    #[test]
    fn espeak_alias_removes_whitespace_from_native_path() {
        let source_root = tempfile::tempdir().expect("create source root");
        let source = source_root.path().join("Application Support");
        fs::create_dir(&source).expect("create source directory");

        let (path, guard) = prepare_espeak_data_dir(&source).expect("prepare alias");

        assert!(!path.to_string_lossy().chars().any(char::is_whitespace));
        assert_eq!(
            fs::canonicalize(path).unwrap(),
            fs::canonicalize(source).unwrap()
        );
        assert!(guard.is_some());
    }
}
