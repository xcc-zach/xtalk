//! CAM++ speaker embeddings backed by the shared native ONNX Runtime.

use std::{
    f32::consts::PI,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use ort::{session::Session, value::Tensor};
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use serde::Serialize;
use serde_json::json;
use tracing::{error, info};

use crate::{moss::create_session, Args, OnnxBackend};

const MODEL_ID: &str = "campplus-cn-common";
const MODEL_FILENAME: &str = "campplus.onnx";
const SAMPLE_RATE: u32 = 16_000;
const FRAME_LENGTH: usize = 400;
const FRAME_SHIFT: usize = 160;
const FFT_SIZE: usize = 512;
const FFT_BINS: usize = FFT_SIZE / 2;
const MEL_BINS: usize = 80;
const EMBEDDING_DIMENSIONS: usize = 192;
const PREEMPHASIS: f32 = 0.97;
const LOW_FREQUENCY_HZ: f32 = 20.0;
const MAX_PCM_BYTES: usize = 32 * 1024 * 1024;

/// Loaded CAM++ graph and deterministic Kaldi-compatible feature extractor.
struct CampPlusEngine {
    session: Session,
    fbank: KaldiFbank,
}

#[derive(Clone)]
struct CampPlusState {
    engine: Arc<Mutex<CampPlusEngine>>,
    backend: &'static str,
}

/// Precomputed constants for 16 kHz, 80-bin Kaldi-style filterbanks.
struct KaldiFbank {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    mel_filters: Vec<Vec<(usize, f32)>>,
}

#[derive(Debug, Serialize)]
struct EmbeddingResponse {
    model: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding: Option<Vec<f32>>,
    latency_ms: f64,
    speech_accepted: bool,
    metrics: EmbeddingMetrics,
}

#[derive(Debug, Serialize)]
struct EmbeddingMetrics {
    engine: &'static str,
    backend: &'static str,
    request_id: String,
    sample_rate: u32,
    samples: usize,
    frames: usize,
    is_final: bool,
    speech_accepted: bool,
}

#[derive(Debug)]
struct CampPlusApiError {
    status: StatusCode,
    message: String,
}

impl CampPlusApiError {
    /// Build a client-visible request validation error.
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    /// Build a server-side inference error.
    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for CampPlusApiError {
    fn into_response(self) -> axum::response::Response {
        (self.status, Json(json!({"error": self.message}))).into_response()
    }
}

impl CampPlusEngine {
    /// Load and warm the pinned CAM++ ONNX graph.
    fn load(args: &Args) -> Result<Self> {
        let model_path = args.model_root.join(MODEL_FILENAME);
        if !model_path.is_file() {
            bail!("CAM++ model is missing: {}", model_path.display());
        }
        let session = create_session(&model_path, args.cpu_threads, args.backend)?;
        let mut engine = Self {
            session,
            fbank: KaldiFbank::new(),
        };
        let warmup = (0..SAMPLE_RATE as usize)
            .map(|index| (index as f32 * 440.0 * 2.0 * PI / SAMPLE_RATE as f32).sin() * 0.1)
            .collect::<Vec<_>>();
        engine
            .embed_samples(&warmup)
            .context("CAM++ warmup failed")?;
        Ok(engine)
    }

    /// Convert normalized mono samples into one L2-normalized embedding.
    fn embed_samples(&mut self, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
        let (features, frames) = self.fbank.extract(samples)?;
        let input = Tensor::from_array(([1, frames, MEL_BINS], features))?;
        let outputs = self.session.run(ort::inputs!["input" => input])?;
        let output = outputs
            .get("output")
            .or_else(|| outputs.get("embedding"))
            .context("CAM++ graph has no embedding output")?;
        let (shape, values) = output
            .try_extract_tensor::<f32>()
            .context("CAM++ embedding output is not float32")?;
        if shape.len() != 2
            || shape[0] != 1
            || shape[1] != EMBEDDING_DIMENSIONS as i64
            || values.len() != EMBEDDING_DIMENSIONS
        {
            bail!("unexpected CAM++ embedding shape: {shape:?}");
        }
        let mut embedding = values.to_vec();
        let norm = embedding
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if !norm.is_finite() || norm <= f32::EPSILON {
            bail!("CAM++ returned a non-finite or zero embedding");
        }
        for value in &mut embedding {
            *value /= norm;
        }
        Ok((embedding, frames))
    }
}

impl KaldiFbank {
    /// Precompute the Povey window, FFT plan, and mel triangles.
    fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let window = (0..FRAME_LENGTH)
            .map(|index| {
                let phase = 2.0 * PI * index as f32 / (FRAME_LENGTH - 1) as f32;
                (0.5 - 0.5 * phase.cos()).powf(0.85)
            })
            .collect();
        Self {
            fft,
            window,
            mel_filters: build_mel_filters(),
        }
    }

    /// Extract mean-normalized log filterbanks in frame-major order.
    fn extract(&self, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
        if samples.len() < FRAME_LENGTH {
            bail!(
                "CAM++ requires at least {FRAME_LENGTH} samples, got {}",
                samples.len()
            );
        }
        let frames = 1 + (samples.len() - FRAME_LENGTH) / FRAME_SHIFT;
        let mut features = vec![0.0_f32; frames * MEL_BINS];
        let mut means = [0.0_f32; MEL_BINS];
        let mut spectrum = vec![Complex32::new(0.0, 0.0); FFT_SIZE];

        for frame_index in 0..frames {
            spectrum.fill(Complex32::new(0.0, 0.0));
            let start = frame_index * FRAME_SHIFT;
            let source = &samples[start..start + FRAME_LENGTH];
            let mean = source.iter().sum::<f32>() / FRAME_LENGTH as f32;
            let mut previous = source[0] - mean;
            spectrum[0].re = previous * (1.0 - PREEMPHASIS) * self.window[0];
            for index in 1..FRAME_LENGTH {
                let current = source[index] - mean;
                spectrum[index].re = (current - PREEMPHASIS * previous) * self.window[index];
                previous = current;
            }
            self.fft.process(&mut spectrum);

            let power = spectrum[..FFT_BINS]
                .iter()
                .map(Complex32::norm_sqr)
                .collect::<Vec<_>>();
            let frame_offset = frame_index * MEL_BINS;
            for (mel_index, filter) in self.mel_filters.iter().enumerate() {
                let energy = filter
                    .iter()
                    .map(|(bin, weight)| power[*bin] * *weight)
                    .sum::<f32>()
                    .max(f32::EPSILON)
                    .ln();
                features[frame_offset + mel_index] = energy;
                means[mel_index] += energy;
            }
        }

        for value in &mut means {
            *value /= frames as f32;
        }
        for frame in features.chunks_exact_mut(MEL_BINS) {
            for (value, mean) in frame.iter_mut().zip(means) {
                *value -= mean;
            }
        }
        Ok((features, frames))
    }
}

/// Build the 20 Hz-to-Nyquist mel triangles used by SpeakerLab.
fn build_mel_filters() -> Vec<Vec<(usize, f32)>> {
    let low_mel = mel_scale(LOW_FREQUENCY_HZ);
    let high_mel = mel_scale(SAMPLE_RATE as f32 / 2.0);
    let mel_step = (high_mel - low_mel) / (MEL_BINS + 1) as f32;
    let fft_bin_width = SAMPLE_RATE as f32 / FFT_SIZE as f32;

    (0..MEL_BINS)
        .map(|index| {
            let left = low_mel + index as f32 * mel_step;
            let middle = left + mel_step;
            let right = middle + mel_step;
            (0..FFT_BINS)
                .filter_map(|bin| {
                    let mel = mel_scale(bin as f32 * fft_bin_width);
                    if mel <= left || mel >= right {
                        return None;
                    }
                    let weight = if mel <= middle {
                        (mel - left) / (middle - left)
                    } else {
                        (right - mel) / (right - middle)
                    };
                    Some((bin, weight))
                })
                .collect()
        })
        .collect()
}

/// Convert frequency in hertz to the Kaldi mel scale.
fn mel_scale(frequency: f32) -> f32 {
    1127.0 * (1.0 + frequency / 700.0).ln()
}

/// Decode little-endian mono PCM16 into normalized samples.
fn decode_pcm16(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.is_empty() {
        bail!("audio must not be empty");
    }
    if !bytes.len().is_multiple_of(2) {
        bail!("audio must contain complete PCM16 samples");
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32_768.0)
        .collect())
}

/// Return whether the snapshot contains any non-zero PCM energy.
fn contains_speech_energy(samples: &[f32]) -> bool {
    samples.iter().any(|sample| sample.abs() > 1.0 / 32_768.0)
}

/// Run the CAM++ HTTP service selected by the desktop managed-model layer.
pub(crate) async fn run(args: Args) -> Result<()> {
    info!(model_root = %args.model_root.display(), "loading CAM++ ONNX runtime");
    let backend = backend_name(args.backend);
    let engine = CampPlusEngine::load(&args)?;
    let state = CampPlusState {
        engine: Arc::new(Mutex::new(engine)),
        backend,
    };
    let router = Router::new()
        .route("/", get(service_info))
        .route("/health", get(health))
        .route("/v1/speaker/embeddings", post(embed))
        .layer(DefaultBodyLimit::max(MAX_PCM_BYTES + 1024 * 1024))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(SocketAddr::new(args.host, args.port))
        .await
        .context("failed to bind CAM++ HTTP listener")?;
    let address = listener
        .local_addr()
        .context("failed to read CAM++ listener address")?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "status": "ready",
            "protocol_version": 1,
            "engine": "campplus-onnx",
            "port": address.port()
        }))?
    );
    info!(%address, backend, "CAM++ ONNX HTTP service ready");

    axum::serve(listener, router)
        .await
        .context("CAM++ HTTP server failed")
}

/// Describe the CAM++ service and its stable endpoint.
async fn service_info() -> Json<serde_json::Value> {
    Json(json!({
        "service": "xtalk-local-model-runtime",
        "engine": "campplus-onnx",
        "endpoints": ["/health", "/v1/speaker/embeddings"]
    }))
}

/// Report readiness without exposing session-local speaker state.
async fn health(State(state): State<CampPlusState>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "engine": "campplus-onnx",
        "model": MODEL_ID,
        "sample_rate": SAMPLE_RATE,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "backend": state.backend
    }))
}

/// Extract one embedding from the complete raw PCM snapshot.
async fn embed(
    State(state): State<CampPlusState>,
    mut multipart: Multipart,
) -> Result<(HeaderMap, Json<EmbeddingResponse>), CampPlusApiError> {
    let mut request_id = None;
    let mut sample_rate = None;
    let mut is_final = None;
    let mut audio = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| CampPlusApiError::bad_request(error.to_string()))?
    {
        match field.name() {
            Some("request_id") => {
                request_id = Some(
                    field
                        .text()
                        .await
                        .map_err(|error| CampPlusApiError::bad_request(error.to_string()))?,
                );
            }
            Some("sample_rate") => {
                let value = field
                    .text()
                    .await
                    .map_err(|error| CampPlusApiError::bad_request(error.to_string()))?;
                sample_rate = Some(value.parse::<u32>().map_err(|_| {
                    CampPlusApiError::bad_request("sample_rate must be an integer")
                })?);
            }
            Some("is_final") => {
                let value = field
                    .text()
                    .await
                    .map_err(|error| CampPlusApiError::bad_request(error.to_string()))?;
                is_final = Some(match value.as_str() {
                    "true" => true,
                    "false" => false,
                    _ => {
                        return Err(CampPlusApiError::bad_request(
                            "is_final must be true or false",
                        ));
                    }
                });
            }
            Some("audio") => {
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|error| CampPlusApiError::bad_request(error.to_string()))?;
                if bytes.len() > MAX_PCM_BYTES {
                    return Err(CampPlusApiError::bad_request(format!(
                        "audio exceeds {MAX_PCM_BYTES} bytes"
                    )));
                }
                audio = Some(bytes.to_vec());
            }
            _ => {}
        }
    }

    let request_id = request_id
        .filter(|value| !value.is_empty())
        .ok_or_else(|| CampPlusApiError::bad_request("request_id is required"))?;
    let sample_rate =
        sample_rate.ok_or_else(|| CampPlusApiError::bad_request("sample_rate is required"))?;
    if sample_rate != SAMPLE_RATE {
        return Err(CampPlusApiError::bad_request(format!(
            "CAM++ requires {SAMPLE_RATE} Hz PCM, got {sample_rate} Hz"
        )));
    }
    let is_final = is_final.ok_or_else(|| CampPlusApiError::bad_request("is_final is required"))?;
    let audio = audio.ok_or_else(|| CampPlusApiError::bad_request("audio is required"))?;
    let samples =
        decode_pcm16(&audio).map_err(|error| CampPlusApiError::bad_request(error.to_string()))?;
    if samples.len() < FRAME_LENGTH {
        return Err(CampPlusApiError::bad_request(format!(
            "audio requires at least {FRAME_LENGTH} samples"
        )));
    }

    let sample_count = samples.len();
    let started = Instant::now();
    let speech_accepted = contains_speech_energy(&samples);
    let (embedding, frames) = if speech_accepted {
        let engine = Arc::clone(&state.engine);
        let (embedding, frames) = tokio::task::spawn_blocking(move || {
            engine
                .lock()
                .map_err(|_| anyhow::anyhow!("CAM++ engine lock is poisoned"))?
                .embed_samples(&samples)
        })
        .await
        .map_err(|error| CampPlusApiError::internal(format!("inference task failed: {error}")))?
        .map_err(|error| {
            error!(%error, "CAM++ embedding inference failed");
            CampPlusApiError::internal(error.to_string())
        })?;
        (Some(embedding), frames)
    } else {
        (None, 0)
    };
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    let metrics = EmbeddingMetrics {
        engine: "onnxruntime",
        backend: state.backend,
        request_id: request_id.clone(),
        sample_rate,
        samples: sample_count,
        frames,
        is_final,
        speech_accepted,
    };
    let response = EmbeddingResponse {
        model: MODEL_ID,
        embedding,
        latency_ms,
        speech_accepted,
        metrics,
    };
    let mut headers = HeaderMap::new();
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        headers.insert("x-request-id", value);
    }
    Ok((headers, Json(response)))
}

/// Return the protocol name for the selected ONNX execution provider.
fn backend_name(backend: OnnxBackend) -> &'static str {
    match backend {
        OnnxBackend::Cpu => "cpu",
        OnnxBackend::Cuda => "cuda",
        OnnxBackend::Coreml => "coreml",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_mel_filters, contains_speech_energy, decode_pcm16, KaldiFbank, FRAME_LENGTH, MEL_BINS,
    };

    #[test]
    fn pcm16_decoder_requires_complete_non_empty_samples() {
        assert!(decode_pcm16(&[]).is_err());
        assert!(decode_pcm16(&[0]).is_err());
        assert_eq!(
            decode_pcm16(&[0, 0, 0xff, 0x7f]).unwrap(),
            [0.0, 32767.0 / 32768.0]
        );
    }

    #[test]
    fn fbank_is_frame_major_and_mean_normalized() {
        let samples = (0..16_000)
            .map(|index| (index as f32 * 0.013).sin() * 0.25)
            .collect::<Vec<_>>();
        let (features, frames) = KaldiFbank::new().extract(&samples).unwrap();
        assert_eq!(features.len(), frames * MEL_BINS);
        for mel in 0..MEL_BINS {
            let mean = (0..frames)
                .map(|frame| features[frame * MEL_BINS + mel])
                .sum::<f32>()
                / frames as f32;
            assert!(mean.abs() < 1.0e-4, "mel {mel} mean was {mean}");
        }
    }

    #[test]
    fn mel_bank_and_energy_gate_match_the_service_contract() {
        let filters = build_mel_filters();
        assert_eq!(filters.len(), MEL_BINS);
        assert!(filters.iter().all(|filter| !filter.is_empty()));
        assert!(!contains_speech_energy(&vec![0.0; FRAME_LENGTH]));
        assert!(contains_speech_energy(&vec![0.5; FRAME_LENGTH]));
    }
}
