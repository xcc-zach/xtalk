//! Loopback HTTP service for the managed moss-transcribe.cpp runtime.

use std::{
    collections::{hash_map::Entry, HashMap},
    ffi::{CStr, CString},
    fs::{self, OpenOptions},
    io::Write,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    os::raw::{c_char, c_float, c_int},
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::{Arc, Mutex, OnceLock},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use axum::{
    extract::{DefaultBodyLimit, Multipart, Path as AxumPath, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use clap::{Parser, ValueEnum};
use regex::Regex;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::Mutex as AsyncMutex;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

const MODEL_FILENAME: &str = "moss-transcribe-q4_k.gguf";
const MAX_AUDIO_BYTES: usize = 64 * 1024 * 1024;
const MAX_TEXT_FIELD_BYTES: usize = 256 * 1024;
const DEFAULT_MAX_TOKENS: usize = 2_048;

#[repr(C)]
struct NativeContext {
    _opaque: [u8; 0],
}

#[repr(C)]
struct NativeCancelToken {
    _opaque: [u8; 0],
}

extern "C" {
    fn xtalk_mtd_runtime_available() -> c_int;
    fn xtalk_mtd_backend_name() -> *const c_char;
    fn xtalk_mtd_backend_is_cpu() -> c_int;
    fn xtalk_mtd_load(path: *const c_char) -> *mut NativeContext;
    fn xtalk_mtd_free(context: *mut NativeContext);
    fn xtalk_mtd_cancel_token_new() -> *mut NativeCancelToken;
    fn xtalk_mtd_cancel_token_cancel(token: *mut NativeCancelToken);
    fn xtalk_mtd_cancel_token_free(token: *mut NativeCancelToken);
    fn xtalk_mtd_transcribe_pcm(
        context: *mut NativeContext,
        samples: *const c_float,
        sample_count: c_int,
        sample_rate: c_int,
        instruction: *const c_char,
        decoder_prefix: *const c_char,
        max_new: c_int,
        cancel: *mut NativeCancelToken,
    ) -> *mut c_char;
    fn xtalk_mtd_free_string(value: *mut c_char);
    fn xtalk_mtd_last_error(context: *mut NativeContext) -> *const c_char;
}

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    /// Directory containing the managed GGUF model.
    #[arg(long)]
    model_root: PathBuf,
    /// ggml execution backend.
    #[arg(long, value_enum, default_value_t = Backend::Metal)]
    backend: Backend,
    /// Loopback host used by the private HTTP service.
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,
    /// HTTP port. Zero requests an OS-assigned port.
    #[arg(long, default_value_t = 0)]
    port: u16,
    /// CPU worker threads made available to ggml.
    #[arg(long, default_value_t = 8)]
    threads: usize,
    /// Optional JSONL destination for privacy-safe final speaker labels.
    #[arg(long)]
    event_log: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Cpu,
    Metal,
}

impl Backend {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
}

struct NativeEngine {
    context: NonNull<NativeContext>,
}

unsafe impl Send for NativeEngine {}

impl NativeEngine {
    fn load(path: &Path) -> Result<Self> {
        let path = CString::new(path.to_string_lossy().as_bytes())
            .context("GGUF path contains an embedded NUL byte")?;
        let context = unsafe { xtalk_mtd_load(path.as_ptr()) };
        Ok(Self {
            context: NonNull::new(context).context("failed to load managed MTD GGUF")?,
        })
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        instruction: &str,
        decoder_prefix: &str,
        max_tokens: usize,
        cancel: &CancelToken,
    ) -> Result<String, InferenceError> {
        let instruction = CString::new(instruction)
            .map_err(|_| InferenceError::Failed("instruction contains a NUL byte".to_owned()))?;
        let decoder_prefix = CString::new(decoder_prefix)
            .map_err(|_| InferenceError::Failed("decoder prefix contains a NUL byte".to_owned()))?;
        let output = unsafe {
            xtalk_mtd_transcribe_pcm(
                self.context.as_ptr(),
                samples.as_ptr(),
                samples.len() as c_int,
                sample_rate as c_int,
                instruction.as_ptr(),
                decoder_prefix.as_ptr(),
                max_tokens as c_int,
                cancel.pointer.as_ptr(),
            )
        };
        let Some(output) = NonNull::new(output) else {
            let message = unsafe {
                let pointer = xtalk_mtd_last_error(self.context.as_ptr());
                if pointer.is_null() {
                    "transcription failed".to_owned()
                } else {
                    CStr::from_ptr(pointer).to_string_lossy().into_owned()
                }
            };
            return if message == "request cancelled" {
                Err(InferenceError::Cancelled)
            } else {
                Err(InferenceError::Failed(message))
            };
        };
        let text = unsafe { CStr::from_ptr(output.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        unsafe { xtalk_mtd_free_string(output.as_ptr()) };
        Ok(text)
    }
}

impl Drop for NativeEngine {
    fn drop(&mut self) {
        unsafe { xtalk_mtd_free(self.context.as_ptr()) };
    }
}

struct CancelToken {
    pointer: NonNull<NativeCancelToken>,
}

unsafe impl Send for CancelToken {}
unsafe impl Sync for CancelToken {}

impl CancelToken {
    fn create() -> Result<Self> {
        let pointer = unsafe { xtalk_mtd_cancel_token_new() };
        Ok(Self {
            pointer: NonNull::new(pointer).context("failed to allocate MTD cancellation token")?,
        })
    }

    fn cancel(&self) {
        unsafe { xtalk_mtd_cancel_token_cancel(self.pointer.as_ptr()) };
    }
}

impl Drop for CancelToken {
    fn drop(&mut self) {
        unsafe { xtalk_mtd_cancel_token_free(self.pointer.as_ptr()) };
    }
}

#[derive(Debug, thiserror::Error)]
enum InferenceError {
    #[error("request cancelled")]
    Cancelled,
    #[error("{0}")]
    Failed(String),
}

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<NativeEngine>>,
    requests: Arc<AsyncMutex<HashMap<String, Arc<CancelToken>>>>,
    backend: &'static str,
    event_log: Option<Arc<DiarizationEventLog>>,
}

#[derive(Default)]
struct DecodeRequest {
    request_id: String,
    sample_rate: Option<u32>,
    decoder_prefix: String,
    context_seconds: Option<f64>,
    current_audio_seconds: Option<f64>,
    is_final: bool,
    instruction: String,
    max_tokens: Option<usize>,
    audio: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
struct Segment {
    start_s: f64,
    end_s: f64,
    speaker_id: String,
    text: String,
}

#[derive(Debug, Serialize, PartialEq)]
struct DiarizationLogSegment {
    start_s: f64,
    end_s: f64,
    speaker_id: String,
}

#[derive(Debug, Serialize, PartialEq)]
struct DiarizationLogEvent {
    schema_version: u8,
    timestamp_unix_ms: u64,
    request_id: String,
    is_final: bool,
    backend: String,
    active_speaker_id: Option<String>,
    segments: Vec<DiarizationLogSegment>,
    latency_ms: f64,
}

struct DiarizationEventLog {
    path: PathBuf,
    append_lock: AsyncMutex<()>,
}

impl DiarizationEventLog {
    fn new(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create MTD diarization log directory: {}",
                    parent.display()
                )
            })?;
        }
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("failed to open MTD diarization log: {}", path.display()))?;
        Ok(Self {
            path,
            append_lock: AsyncMutex::new(()),
        })
    }

    async fn append(&self, event: &DiarizationLogEvent) -> Result<()> {
        let _guard = self.append_lock.lock().await;
        let mut line = serde_json::to_vec(event).context("failed to encode MTD diarization log")?;
        line.push(b'\n');
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| {
                format!(
                    "failed to append MTD diarization log: {}",
                    self.path.display()
                )
            })?;
        file.write_all(&line)
            .context("failed to write MTD diarization log")?;
        file.flush()
            .context("failed to flush MTD diarization log")?;
        Ok(())
    }
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

    fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }

    fn cancelled() -> Self {
        Self {
            status: StatusCode::from_u16(499).expect("499 is a valid status"),
            message: "request cancelled".to_owned(),
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
                .unwrap_or_else(|_| EnvFilter::new("xtalk_mtd_model_runtime=info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();
    if args.host != IpAddr::V4(Ipv4Addr::LOCALHOST)
        && args.host != IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)
    {
        anyhow::bail!("MTD runtime must bind to a loopback host");
    }
    if unsafe { xtalk_mtd_runtime_available() } != 1 {
        anyhow::bail!("managed MTD is unavailable in this platform build");
    }
    configure_backend(args.backend, args.threads);
    let model_path = resolve_model_path(&args.model_root)?;
    info!(model = %model_path.display(), backend = args.backend.as_str(), "loading managed MTD model");
    let engine = NativeEngine::load(&model_path)?;
    let native_backend = native_backend_name()?;
    let selected_cpu = unsafe { xtalk_mtd_backend_is_cpu() } == 1;
    let backend_matches = match args.backend {
        Backend::Cpu => selected_cpu,
        Backend::Metal => !selected_cpu,
    };
    if !backend_matches {
        anyhow::bail!(
            "requested {} backend but moss-transcribe.cpp selected {native_backend}",
            args.backend.as_str()
        );
    }
    let event_log = args
        .event_log
        .map(DiarizationEventLog::new)
        .transpose()?
        .map(Arc::new);
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        requests: Arc::new(AsyncMutex::new(HashMap::new())),
        backend: args.backend.as_str(),
        event_log,
    };
    let router = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models_not_found))
        .route("/v1/mtd/decode", post(decode))
        .route("/v1/mtd/requests/{request_id}", delete(cancel_request))
        .layer(DefaultBodyLimit::max(
            MAX_AUDIO_BYTES + MAX_TEXT_FIELD_BYTES * 3,
        ))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(SocketAddr::new(args.host, args.port))
        .await
        .context("failed to bind managed MTD HTTP listener")?;
    let address = listener
        .local_addr()
        .context("failed to read managed MTD listener address")?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "status": "ready",
            "protocol_version": 1,
            "engine": "moss-transcribe.cpp",
            "backend": args.backend.as_str(),
            "port": address.port()
        }))?
    );
    axum::serve(listener, router)
        .await
        .context("managed MTD HTTP server failed")?;
    Ok(())
}

fn configure_backend(backend: Backend, threads: usize) {
    std::env::set_var("MTD_THREADS", threads.max(1).to_string());
    match backend {
        Backend::Cpu => std::env::set_var("MTD_DEVICE", "cpu"),
        Backend::Metal => {
            std::env::set_var("MTD_DEVICE", "metal");
            std::env::set_var("GGML_METAL_NO_RESIDENCY", "1");
        }
    }
}

fn native_backend_name() -> Result<String> {
    let pointer = unsafe { xtalk_mtd_backend_name() };
    if pointer.is_null() {
        anyhow::bail!("moss-transcribe.cpp returned an invalid backend name");
    }
    Ok(unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned())
}

fn resolve_model_path(root: &Path) -> Result<PathBuf> {
    let direct = root.join(MODEL_FILENAME);
    if direct.is_file() {
        return Ok(direct);
    }
    if root.is_file() && root.file_name().and_then(|value| value.to_str()) == Some(MODEL_FILENAME) {
        return Ok(root.to_owned());
    }
    anyhow::bail!("managed MTD model is missing: {}", direct.display())
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "engine": "moss-transcribe.cpp",
        "backend": state.backend
    }))
}

async fn models_not_found() -> StatusCode {
    StatusCode::NOT_FOUND
}

async fn decode(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, ApiError> {
    let request = parse_decode_request(multipart).await?;
    let request_id = request.request_id.clone();
    let cancel =
        Arc::new(CancelToken::create().map_err(|error| ApiError::internal(error.to_string()))?);
    {
        let mut requests = state.requests.lock().await;
        match requests.entry(request_id.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(Arc::clone(&cancel));
            }
            Entry::Occupied(_) => return Err(ApiError::conflict("duplicate request_id")),
        }
    }

    let sample_rate = request.sample_rate.expect("validated sample rate");
    let audio_seconds = request
        .current_audio_seconds
        .unwrap_or(request.audio.len() as f64 / (sample_rate as f64 * 2.0));
    let context_seconds = request.context_seconds.unwrap_or(0.0);
    let samples = pcm16_to_float32(&request.audio);
    let engine = Arc::clone(&state.engine);
    let instruction = request.instruction.clone();
    let decoder_prefix = request.decoder_prefix.clone();
    let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let inference_cancel = Arc::clone(&cancel);
    let started = Instant::now();
    let worker_result = tokio::task::spawn_blocking(move || {
        let mut engine = engine
            .lock()
            .map_err(|_| InferenceError::Failed("MTD engine lock was poisoned".to_owned()))?;
        engine.transcribe(
            &samples,
            sample_rate,
            &instruction,
            &decoder_prefix,
            max_tokens,
            &inference_cancel,
        )
    })
    .await;
    {
        let mut requests = state.requests.lock().await;
        if requests
            .get(&request_id)
            .is_some_and(|active| Arc::ptr_eq(active, &cancel))
        {
            requests.remove(&request_id);
        }
    }
    let result =
        worker_result.map_err(|error| ApiError::internal(format!("MTD worker failed: {error}")))?;
    let suffix = match result {
        Ok(value) => value,
        Err(InferenceError::Cancelled) => return Err(ApiError::cancelled()),
        Err(InferenceError::Failed(message)) => return Err(ApiError::internal(message)),
    };
    let raw_text = join_decoder_prefix(&request.decoder_prefix, &suffix);
    let current_segments =
        crop_current_segments(&parse_segments(&raw_text), context_seconds, audio_seconds);
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    if request.is_final {
        if let Some(event_log) = state.event_log.as_ref() {
            let event = build_diarization_log_event(
                &request_id,
                state.backend,
                &current_segments,
                latency_ms,
            );
            if let Err(error) = event_log.append(&event).await {
                warn!(request_id = %request_id, error = %error, "failed to record final speaker labels");
            }
        }
    }
    Ok(Json(json!({
        "request_id": request_id,
        "raw_text": raw_text,
        "current_segments": current_segments,
        "latency_ms": latency_ms,
        "metrics": {
            "backend": "moss_transcribe_cpp",
            "device": state.backend,
            "is_final": request.is_final,
            "registration_mode": "fixed_decoder_prefix",
            "decoder_prefix_chars": request.decoder_prefix.chars().count(),
            "generated_suffix": suffix
        }
    })))
}

async fn cancel_request(
    AxumPath(request_id): AxumPath<String>,
    State(state): State<AppState>,
) -> StatusCode {
    let token = state.requests.lock().await.get(&request_id).cloned();
    if let Some(token) = token {
        token.cancel();
        StatusCode::ACCEPTED
    } else {
        StatusCode::NOT_FOUND
    }
}

async fn parse_decode_request(mut multipart: Multipart) -> Result<DecodeRequest, ApiError> {
    let mut request = DecodeRequest::default();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| ApiError::bad_request(format!("invalid multipart body: {error}")))?
    {
        let name = field.name().unwrap_or_default().to_owned();
        if name == "audio" {
            let value = field
                .bytes()
                .await
                .map_err(|error| ApiError::bad_request(format!("invalid audio field: {error}")))?;
            if value.len() > MAX_AUDIO_BYTES {
                return Err(ApiError::bad_request("audio field is too large"));
            }
            request.audio = value.to_vec();
            continue;
        }
        let value = field
            .text()
            .await
            .map_err(|error| ApiError::bad_request(format!("invalid {name} field: {error}")))?;
        if value.len() > MAX_TEXT_FIELD_BYTES {
            return Err(ApiError::bad_request(format!("{name} field is too large")));
        }
        match name.as_str() {
            "request_id" => request.request_id = value,
            "sample_rate" => {
                request.sample_rate = Some(
                    value
                        .parse()
                        .map_err(|_| ApiError::bad_request("sample_rate is invalid"))?,
                )
            }
            "decoder_prefix" => request.decoder_prefix = value,
            "context_seconds" => {
                request.context_seconds = Some(
                    value
                        .parse()
                        .map_err(|_| ApiError::bad_request("context_seconds is invalid"))?,
                )
            }
            "current_audio_seconds" => {
                request.current_audio_seconds = Some(
                    value
                        .parse()
                        .map_err(|_| ApiError::bad_request("current_audio_seconds is invalid"))?,
                )
            }
            "is_final" => {
                request.is_final = value
                    .parse()
                    .map_err(|_| ApiError::bad_request("is_final is invalid"))?
            }
            "instruction" => request.instruction = value,
            "max_tokens" => {
                request.max_tokens = Some(
                    value
                        .parse()
                        .map_err(|_| ApiError::bad_request("max_tokens is invalid"))?,
                )
            }
            "temperature" => {}
            _ => {}
        }
    }
    validate_decode_request(&request)?;
    Ok(request)
}

fn validate_decode_request(request: &DecodeRequest) -> Result<(), ApiError> {
    if request.request_id.trim().is_empty() || request.request_id.len() > 256 {
        return Err(ApiError::bad_request("request_id is missing or invalid"));
    }
    let Some(sample_rate) = request.sample_rate else {
        return Err(ApiError::bad_request("sample_rate is missing or invalid"));
    };
    if !(8_000..=192_000).contains(&sample_rate) {
        return Err(ApiError::bad_request(
            "sample_rate is outside the supported range",
        ));
    }
    if request.audio.is_empty() || !request.audio.len().is_multiple_of(2) {
        return Err(ApiError::bad_request(
            "audio must contain complete PCM16 samples",
        ));
    }
    for (name, value) in [
        ("context_seconds", request.context_seconds.unwrap_or(0.0)),
        (
            "current_audio_seconds",
            request.current_audio_seconds.unwrap_or(0.0),
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(ApiError::bad_request(format!("{name} is invalid")));
        }
    }
    let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    if max_tokens == 0 || max_tokens > 8_192 {
        return Err(ApiError::bad_request(
            "max_tokens is outside the supported range",
        ));
    }
    Ok(())
}

fn pcm16_to_float32(pcm16: &[u8]) -> Vec<f32> {
    pcm16
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32_768.0)
        .collect()
}

fn segment_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"(?si)\[(\d+(?:\.\d+)?)\]\s*\[*\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]")
            .expect("segment regex is valid")
    })
}

fn parse_segments(text: &str) -> Vec<Segment> {
    segment_regex()
        .captures_iter(text)
        .filter_map(|capture| {
            let start_s = capture.get(1)?.as_str().parse::<f64>().ok()?;
            let end_s = capture.get(4)?.as_str().parse::<f64>().ok()?;
            (end_s > start_s).then(|| Segment {
                start_s: start_s.max(0.0),
                end_s,
                speaker_id: capture.get(2).unwrap().as_str().to_ascii_uppercase(),
                text: capture
                    .get(3)
                    .unwrap()
                    .as_str()
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" "),
            })
        })
        .collect()
}

fn join_decoder_prefix(prefix: &str, suffix: &str) -> String {
    let prefix = prefix.trim();
    let mut suffix = suffix.trim().to_owned();
    if prefix.is_empty() {
        return suffix;
    }
    if suffix.is_empty() {
        return prefix.to_owned();
    }
    if suffix.starts_with("[S") {
        if let Some(timestamp) = trailing_timestamp(prefix) {
            suffix = format!("[{timestamp}]{suffix}");
        }
    }
    format!("{prefix} {suffix}")
}

fn trailing_timestamp(value: &str) -> Option<&str> {
    let value = value.trim_end();
    let close = value.strip_suffix(']')?;
    let open = close.rfind('[')?;
    let timestamp = &close[open + 1..];
    timestamp.parse::<f64>().ok().map(|_| timestamp)
}

fn crop_current_segments(
    segments: &[Segment],
    context_seconds: f64,
    current_audio_seconds: f64,
) -> Vec<Segment> {
    let current_end = context_seconds + current_audio_seconds;
    segments
        .iter()
        .filter_map(|segment| {
            let start_s = segment.start_s.max(context_seconds);
            let end_s = segment.end_s.min(current_end);
            (end_s > start_s).then(|| Segment {
                start_s: (start_s - context_seconds).max(0.0),
                end_s: (end_s - context_seconds).max(0.0),
                speaker_id: segment.speaker_id.clone(),
                text: segment.text.clone(),
            })
        })
        .collect()
}

fn build_diarization_log_event(
    request_id: &str,
    backend: &str,
    segments: &[Segment],
    latency_ms: f64,
) -> DiarizationLogEvent {
    let timestamp_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;
    let active_speaker_id = segments
        .iter()
        .rev()
        .find(|segment| !segment.text.trim().is_empty())
        .map(|segment| segment.speaker_id.clone());
    DiarizationLogEvent {
        schema_version: 1,
        timestamp_unix_ms,
        request_id: request_id.to_owned(),
        is_final: true,
        backend: backend.to_owned(),
        active_speaker_id,
        segments: segments
            .iter()
            .map(|segment| DiarizationLogSegment {
                start_s: segment.start_s,
                end_s: segment.end_s,
                speaker_id: segment.speaker_id.clone(),
            })
            .collect(),
        latency_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_diarization_log_event, crop_current_segments, join_decoder_prefix, parse_segments,
        pcm16_to_float32, DiarizationEventLog, Segment,
    };
    use std::{
        fs,
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn converts_little_endian_pcm16() {
        assert_eq!(pcm16_to_float32(&[0, 0, 0xff, 0x7f]).len(), 2);
        assert!(pcm16_to_float32(&[0xff, 0x7f])[0] > 0.99);
    }

    #[test]
    fn joins_a_speaker_only_continuation_at_the_prefix_boundary() {
        assert_eq!(
            join_decoder_prefix("[0.00][S01]hello[1.25]", "[S02]world[2.00]"),
            "[0.00][S01]hello[1.25] [1.25][S02]world[2.00]"
        );
    }

    #[test]
    fn parses_and_crops_registered_context() {
        let parsed = parse_segments("[0.00][S01]registered[1.00] [2.00][S02]current words[3.50]");
        assert_eq!(parsed.len(), 2);
        assert_eq!(
            crop_current_segments(&parsed, 2.0, 2.0),
            vec![Segment {
                start_s: 0.0,
                end_s: 1.5,
                speaker_id: "S02".to_owned(),
                text: "current words".to_owned(),
            }]
        );
    }

    #[test]
    fn final_log_records_labels_without_speech_text() {
        let event = build_diarization_log_event(
            "session/2/3/4/final",
            "metal",
            &[
                Segment {
                    start_s: 0.1,
                    end_s: 0.9,
                    speaker_id: "S01".to_owned(),
                    text: "private first phrase".to_owned(),
                },
                Segment {
                    start_s: 1.0,
                    end_s: 1.8,
                    speaker_id: "S02".to_owned(),
                    text: "private last phrase".to_owned(),
                },
            ],
            42.5,
        );
        let encoded = serde_json::to_string(&event).expect("serialize log event");

        assert_eq!(event.active_speaker_id.as_deref(), Some("S02"));
        assert_eq!(event.segments[0].speaker_id, "S01");
        assert!(!encoded.contains("private first phrase"));
        assert!(!encoded.contains("private last phrase"));
    }

    #[tokio::test]
    async fn event_log_serializes_concurrent_appends_as_json_lines() {
        let directory = std::env::temp_dir().join(format!(
            "xtalk-mtd-event-log-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock")
                .as_nanos()
        ));
        let path = directory.join("mtd-diarization.jsonl");
        let log = Arc::new(DiarizationEventLog::new(path.clone()).expect("create event log"));
        let first = build_diarization_log_event("first/final", "cpu", &[], 1.0);
        let second = build_diarization_log_event("second/final", "cpu", &[], 2.0);

        let first_log = Arc::clone(&log);
        let second_log = Arc::clone(&log);
        let (first_result, second_result) =
            tokio::join!(async move { first_log.append(&first).await }, async move {
                second_log.append(&second).await
            },);
        first_result.expect("append first event");
        second_result.expect("append second event");

        let contents = fs::read_to_string(&path).expect("read event log");
        let lines = contents.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 2);
        for line in lines {
            serde_json::from_str::<serde_json::Value>(line).expect("valid JSONL record");
        }
        fs::remove_dir_all(directory).expect("remove event log test directory");
    }
}
