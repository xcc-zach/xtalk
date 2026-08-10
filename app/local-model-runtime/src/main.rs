//! HTTP entrypoint for XTalk's native local-model runtime.

mod audio;
mod campplus;
mod manifest;
mod moss;
mod refiner;
mod text;
mod wav;

use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result};
use audio::decode_reference_audio;
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Multipart, State},
    http::{header, HeaderValue, Response, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use clap::{Parser, ValueEnum};
use moss::{MossEngine, SynthesisOptions};
use serde::{Deserialize, Serialize};
use serde_json::json;
use text::normalize_for_speech;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
use wav::{encode_pcm16_mono, encode_wav_pcm16_mono};

const DEFAULT_MAX_INPUT_CHARS: usize = 2_048;
const MAX_PROMPT_AUDIO_BYTES: usize = 32 * 1024 * 1024;

#[derive(Parser)]
#[command(author, version, about)]
pub(crate) struct Args {
    /// Native model service to host.
    #[arg(long, value_enum, default_value_t = RuntimeService::MossTtsNano)]
    pub(crate) service: RuntimeService,

    /// Directory containing the selected model snapshot.
    #[arg(long)]
    pub(crate) model_root: PathBuf,

    /// ONNX Runtime dynamic library bundled with the desktop application.
    #[arg(long)]
    pub(crate) ort_dylib: PathBuf,

    /// ONNX Runtime execution backend.
    #[arg(long, value_enum, default_value_t = OnnxBackend::Cpu)]
    pub(crate) backend: OnnxBackend,

    /// Loopback host used by the private HTTP service.
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) host: IpAddr,

    /// HTTP port. Zero requests an OS-assigned port.
    #[arg(long, default_value_t = 0)]
    pub(crate) port: u16,

    /// ONNX Runtime intra-op CPU thread count.
    #[arg(long, default_value_t = 2)]
    pub(crate) cpu_threads: usize,
}

/// Native ONNX service exposed by this sidecar.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum RuntimeService {
    /// Host MOSS-TTS-Nano speech synthesis.
    MossTtsNano,
    /// Host the AgenticASR transcript Refiner.
    AgenticAsrRefiner,
    /// Host CAM++ speaker-embedding extraction.
    #[value(name = "campplus")]
    CampPlus,
}

/// Execution provider selected for all MOSS ONNX sessions.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum OnnxBackend {
    /// Use the ONNX Runtime CPU execution provider.
    Cpu,
    /// Require the ONNX Runtime CUDA execution provider.
    Cuda,
    /// Require the ONNX Runtime CoreML execution provider.
    Coreml,
}

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<MossEngine>>,
    voices: Arc<Vec<VoiceResponse>>,
    max_new_frames: usize,
    sample_rate: u32,
    reference_channels: usize,
}

#[derive(Serialize, Clone)]
struct VoiceResponse {
    id: String,
    name: String,
}

#[derive(Deserialize)]
struct SpeechRequest {
    input: String,
    #[serde(default = "default_voice")]
    voice: String,
    #[serde(default = "default_response_format")]
    response_format: ResponseFormat,
    #[serde(default)]
    max_frames: Option<usize>,
    #[serde(default = "default_seed")]
    seed: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum ResponseFormat {
    Wav,
    Pcm,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    engine: &'static str,
    sample_rate: u32,
}

#[derive(Serialize)]
struct GenerateResponse {
    audio_base64: String,
    sample_rate: u32,
    run_status: String,
    prompt_audio_path: String,
    warmup_status_text: &'static str,
    text_normalization_status_text: &'static str,
    text_chunks: Vec<String>,
    normalized_text: String,
    normalization_method: &'static str,
    text_normalization_language: &'static str,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (
            self.status,
            Json(ErrorBody {
                error: self.message,
            }),
        )
            .into_response()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("xtalk_local_model_runtime=info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();
    if args.host != IpAddr::V4(Ipv4Addr::LOCALHOST)
        && args.host != IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)
    {
        anyhow::bail!("local model runtime must bind to a loopback host");
    }

    ort::init_from(&args.ort_dylib)
        .with_context(|| {
            format!(
                "failed to load ONNX Runtime dynamic library {}",
                args.ort_dylib.display()
            )
        })?
        .commit();
    ort::environment::Environment::current()?.set_log_level(ort::logging::LogLevel::Warning);
    match args.service {
        RuntimeService::MossTtsNano => run_moss(args).await,
        RuntimeService::AgenticAsrRefiner => refiner::run(args).await,
        RuntimeService::CampPlus => campplus::run(args).await,
    }
}

async fn run_moss(args: Args) -> Result<()> {
    info!(model_root = %args.model_root.display(), "loading MOSS ONNX runtime");
    let engine = MossEngine::load(&args.model_root, args.cpu_threads, args.backend)?;
    let voices = engine
        .voices()
        .into_iter()
        .map(|(id, name)| VoiceResponse {
            id: id.to_owned(),
            name: name.to_owned(),
        })
        .collect::<Vec<_>>();
    let max_new_frames = engine.max_new_frames();
    let sample_rate = engine.reference_sample_rate();
    let reference_channels = engine.reference_channels();
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        voices: Arc::new(voices),
        max_new_frames,
        sample_rate,
        reference_channels,
    };

    let router = Router::new()
        .route("/", get(service_info))
        .route("/health", get(health))
        .route("/api/generate", post(generate))
        .route("/v1/voices", get(list_voices))
        .route("/v1/audio/speech", post(synthesize))
        .layer(DefaultBodyLimit::max(MAX_PROMPT_AUDIO_BYTES + 1024 * 1024))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(SocketAddr::new(args.host, args.port))
        .await
        .context("failed to bind local model HTTP listener")?;
    let address = listener
        .local_addr()
        .context("failed to read local model listener address")?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "status": "ready",
            "protocol_version": 1,
            "engine": "moss-tts-nano-onnx",
            "port": address.port()
        }))?
    );
    info!(%address, "MOSS ONNX HTTP service ready");

    axum::serve(listener, router)
        .await
        .context("local model HTTP server failed")
}

async fn service_info() -> Json<serde_json::Value> {
    Json(json!({
        "service": "xtalk-local-model-runtime",
        "engine": "moss-tts-nano-onnx",
        "endpoints": ["/health", "/api/generate", "/v1/voices", "/v1/audio/speech"]
    }))
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        engine: "moss-tts-nano-onnx",
        sample_rate: 48_000,
    })
}

async fn list_voices(State(state): State<AppState>) -> Json<Vec<VoiceResponse>> {
    Json((*state.voices).clone())
}

async fn generate(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<GenerateResponse>, ApiError> {
    let mut text = None;
    let mut prompt_audio = None;
    let mut prompt_filename = None;
    let mut max_frames = state.max_new_frames;
    let mut seed = default_seed();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| ApiError::bad_request(error.to_string()))?
    {
        match field.name() {
            Some("text") => {
                text = Some(
                    field
                        .text()
                        .await
                        .map_err(|error| ApiError::bad_request(error.to_string()))?,
                );
            }
            Some("prompt_audio") => {
                prompt_filename = field.file_name().map(str::to_owned);
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|error| ApiError::bad_request(error.to_string()))?;
                if bytes.len() > MAX_PROMPT_AUDIO_BYTES {
                    return Err(ApiError::bad_request(format!(
                        "prompt_audio exceeds {MAX_PROMPT_AUDIO_BYTES} bytes"
                    )));
                }
                prompt_audio = Some(bytes.to_vec());
            }
            Some("max_new_frames") => {
                let value = field
                    .text()
                    .await
                    .map_err(|error| ApiError::bad_request(error.to_string()))?;
                max_frames = value
                    .parse::<usize>()
                    .map_err(|_| ApiError::bad_request("max_new_frames must be an integer"))?
                    .min(state.max_new_frames);
            }
            Some("seed") => {
                let value = field
                    .text()
                    .await
                    .map_err(|error| ApiError::bad_request(error.to_string()))?;
                seed = if value.trim().is_empty() || value.trim() == "0" {
                    default_seed()
                } else {
                    value
                        .parse::<u64>()
                        .map_err(|_| ApiError::bad_request("seed must be an integer"))?
                };
            }
            _ => {}
        }
    }

    let text = text.unwrap_or_default().trim().to_owned();
    validate_input(&text, max_frames)?;
    let prompt_audio =
        prompt_audio.ok_or_else(|| ApiError::bad_request("prompt_audio is required"))?;
    let display_filename = prompt_filename.clone().unwrap_or_default();
    let engine = Arc::clone(&state.engine);
    let sample_rate = state.sample_rate;
    let reference_channels = state.reference_channels;
    let response_text = normalize_for_speech(&text);
    let response_text_for_task = response_text.clone();
    let (output, text_chunks) = tokio::task::spawn_blocking(move || {
        let reference = decode_reference_audio(
            prompt_audio,
            prompt_filename.as_deref(),
            sample_rate,
            reference_channels,
        )?;
        let mut engine = engine
            .lock()
            .map_err(|_| anyhow::anyhow!("MOSS engine lock is poisoned"))?;
        let prompt_audio_codes = engine.encode_reference_audio(&reference)?;
        synthesize_chunks(
            &mut engine,
            &response_text_for_task,
            "",
            Some(&prompt_audio_codes),
            max_frames,
            seed,
        )
    })
    .await
    .map_err(|error| ApiError::internal(format!("inference task failed: {error}")))?
    .map_err(|error| {
        error!(%error, "MOSS synthesis failed");
        ApiError::internal(error.to_string())
    })?;

    let wav = encode_wav_pcm16_mono(&output.samples, output.sample_rate);
    Ok(Json(GenerateResponse {
        audio_base64: BASE64_STANDARD.encode(wav),
        sample_rate: output.sample_rate,
        run_status: format!(
            "MOSS generation complete: frames={}, inference_ms={}",
            output.generated_frames, output.elapsed_ms
        ),
        prompt_audio_path: display_filename,
        warmup_status_text: "Ready.",
        text_normalization_status_text: "Text normalization is handled by the caller.",
        text_chunks,
        normalized_text: response_text,
        normalization_method: "xtalk-mixed-text-v1",
        text_normalization_language: "multilingual",
    }))
}

async fn synthesize(
    State(state): State<AppState>,
    Json(request): Json<SpeechRequest>,
) -> Result<Response<Body>, ApiError> {
    let input = normalize_for_speech(request.input.trim());
    let max_frames = request
        .max_frames
        .unwrap_or(state.max_new_frames)
        .min(state.max_new_frames);
    validate_input(&input, max_frames)?;

    let voice = request.voice;
    let response_format = request.response_format;
    let seed = request.seed;
    let engine = Arc::clone(&state.engine);
    let (output, _) = tokio::task::spawn_blocking(move || {
        let mut engine = engine
            .lock()
            .map_err(|_| anyhow::anyhow!("MOSS engine lock is poisoned"))?;
        synthesize_chunks(&mut engine, &input, &voice, None, max_frames, seed)
    })
    .await
    .map_err(|error| ApiError::internal(format!("inference task failed: {error}")))?
    .map_err(|error| {
        error!(%error, "MOSS synthesis failed");
        ApiError::internal(error.to_string())
    })?;

    let (content_type, audio_bytes) = match response_format {
        ResponseFormat::Wav => (
            "audio/wav",
            encode_wav_pcm16_mono(&output.samples, output.sample_rate),
        ),
        ResponseFormat::Pcm => ("audio/L16", encode_pcm16_mono(&output.samples)),
    };
    let mut response = Response::new(Body::from(audio_bytes));
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    response.headers_mut().insert(
        "x-audio-sample-rate",
        HeaderValue::from_str(&output.sample_rate.to_string())
            .map_err(|error| ApiError::internal(error.to_string()))?,
    );
    response.headers_mut().insert(
        "x-generated-frames",
        HeaderValue::from_str(&output.generated_frames.to_string())
            .map_err(|error| ApiError::internal(error.to_string()))?,
    );
    response.headers_mut().insert(
        "x-inference-ms",
        HeaderValue::from_str(&output.elapsed_ms.to_string())
            .map_err(|error| ApiError::internal(error.to_string()))?,
    );
    Ok(response)
}

fn validate_input(input: &str, max_frames: usize) -> Result<(), ApiError> {
    if input.is_empty() {
        return Err(ApiError::bad_request("input must not be empty"));
    }
    if input.chars().count() > DEFAULT_MAX_INPUT_CHARS {
        return Err(ApiError::bad_request(format!(
            "input exceeds {DEFAULT_MAX_INPUT_CHARS} characters"
        )));
    }
    if max_frames == 0 {
        return Err(ApiError::bad_request("max_frames must be positive"));
    }
    Ok(())
}

/// Synthesize normalized text one stable chunk at a time and join it at 48 kHz.
fn synthesize_chunks(
    engine: &mut MossEngine,
    text: &str,
    voice: &str,
    prompt_audio_codes: Option<&[Vec<i32>]>,
    max_frames: usize,
    seed: u64,
) -> Result<(moss::SynthesisOutput, Vec<String>)> {
    let chunks = engine.split_text_chunks(text)?;
    let mut samples = Vec::new();
    let mut generated_frames = 0;
    let mut elapsed_ms = 0;
    let sample_rate = engine.reference_sample_rate();
    for (index, chunk) in chunks.iter().enumerate() {
        let output = engine.synthesize(SynthesisOptions {
            text: chunk,
            voice,
            prompt_audio_codes,
            max_frames,
            seed,
        })?;
        generated_frames += output.generated_frames;
        elapsed_ms += output.elapsed_ms;
        samples.extend(output.samples);
        if index + 1 < chunks.len() {
            samples.extend(std::iter::repeat_n(0.0, sample_rate as usize * 400 / 1_000));
        }
    }
    Ok((
        moss::SynthesisOutput {
            samples,
            sample_rate,
            generated_frames,
            elapsed_ms,
        },
        chunks,
    ))
}

fn default_voice() -> String {
    "Junhao".to_owned()
}

fn default_response_format() -> ResponseFormat {
    ResponseFormat::Wav
}

fn default_seed() -> u64 {
    1_234
}
