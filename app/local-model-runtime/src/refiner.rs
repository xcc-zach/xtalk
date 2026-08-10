//! OpenAI-compatible AgenticASR Refiner backed by an ONNX causal language model.

use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{bail, Context, Result};
use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use ort::{
    session::{Session, SessionInputValue, SessionOutputs},
    value::{DynValue, Tensor},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokenizers::Tokenizer;
use tracing::{error, info};

use crate::{moss::create_session, Args};

const MODEL_ID: &str = "agentic-asr-refiner";
const LAYER_COUNT: usize = 24;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_DIMENSION: usize = 128;
const DEFAULT_MAX_TOKENS: usize = 512;
const MAX_GENERATION_TOKENS: usize = 512;
const MAX_REQUEST_BYTES: usize = 128 * 1024;
const END_OF_TEXT_ID: i64 = 1;
const IM_END_ID: i64 = 130_073;

/// Loaded tokenizer and autoregressive ONNX session.
struct RefinerEngine {
    tokenizer: Tokenizer,
    session: Session,
}

#[derive(Clone)]
struct RefinerState {
    engine: Arc<Mutex<RefinerEngine>>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: &'static str,
    choices: Vec<ChatChoice>,
    usage: TokenUsage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: AssistantMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct AssistantMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct TokenUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug)]
struct RefinerApiError {
    status: StatusCode,
    message: String,
}

impl RefinerApiError {
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

impl IntoResponse for RefinerApiError {
    fn into_response(self) -> axum::response::Response {
        (self.status, Json(json!({"error": self.message}))).into_response()
    }
}

impl RefinerEngine {
    /// Load and warm the pinned AgenticASR Refiner model.
    fn load(args: &Args) -> Result<Self> {
        let tokenizer_path = args.model_root.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            anyhow::anyhow!(
                "failed to load Refiner tokenizer {}: {error}",
                tokenizer_path.display()
            )
        })?;
        let model_path = args.model_root.join("model.onnx");
        let session = create_session(&model_path, args.cpu_threads, args.backend)?;
        let mut engine = Self { tokenizer, session };
        engine
            .generate(
                &[ChatMessage {
                    role: "user".to_owned(),
                    content: "测试".to_owned(),
                }],
                1,
            )
            .context("Refiner warmup failed")?;
        Ok(engine)
    }

    /// Generate one greedy no-thinking refinement response.
    fn generate(&mut self, messages: &[ChatMessage], max_tokens: usize) -> Result<Generation> {
        let prompt = build_prompt(messages)?;
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| anyhow::anyhow!("failed to tokenize Refiner prompt: {error}"))?;
        let prompt_ids = encoding
            .get_ids()
            .iter()
            .map(|token| i64::from(*token))
            .collect::<Vec<_>>();
        if prompt_ids.is_empty() {
            bail!("Refiner prompt tokenized to an empty sequence");
        }

        let mut caches = empty_caches()?;
        let mut input_ids = prompt_ids.clone();
        let mut past_length = 0_usize;
        let mut generated = Vec::<u32>::new();
        let mut stopped = false;

        for _ in 0..max_tokens {
            let (next_token, next_caches) = self.run_step(&input_ids, past_length, &caches)?;
            past_length += input_ids.len();
            caches = next_caches;
            if next_token == END_OF_TEXT_ID || next_token == IM_END_ID {
                stopped = true;
                break;
            }
            generated
                .push(u32::try_from(next_token).context("Refiner produced a negative token ID")?);
            input_ids.clear();
            input_ids.push(next_token);
        }

        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|error| anyhow::anyhow!("failed to decode Refiner response: {error}"))?;
        Ok(Generation {
            text: text.trim().to_owned(),
            prompt_tokens: prompt_ids.len(),
            completion_tokens: generated.len(),
            stopped,
        })
    }

    fn run_step(
        &mut self,
        input_ids: &[i64],
        past_length: usize,
        caches: &[DynValue],
    ) -> Result<(i64, Vec<DynValue>)> {
        let sequence_length = input_ids.len();
        let input_ids_tensor = Tensor::from_array(([1, sequence_length], input_ids.to_vec()))?;
        let attention_mask = vec![1_i64; past_length + sequence_length];
        let attention_mask_tensor =
            Tensor::from_array(([1, attention_mask.len()], attention_mask))?;
        let position_ids = (past_length..past_length + sequence_length)
            .map(|position| i64::try_from(position).context("position ID exceeds i64"))
            .collect::<Result<Vec<_>>>()?;
        let position_ids_tensor = Tensor::from_array(([1, sequence_length], position_ids))?;
        let mut inputs: Vec<(String, SessionInputValue<'_>)> = vec![
            (
                "input_ids".to_owned(),
                SessionInputValue::from(input_ids_tensor),
            ),
            (
                "attention_mask".to_owned(),
                SessionInputValue::from(attention_mask_tensor),
            ),
            (
                "position_ids".to_owned(),
                SessionInputValue::from(position_ids_tensor),
            ),
        ];
        for layer in 0..LAYER_COUNT {
            inputs.push((
                format!("past_key_values.{layer}.key"),
                SessionInputValue::from(&caches[layer * 2]),
            ));
            inputs.push((
                format!("past_key_values.{layer}.value"),
                SessionInputValue::from(&caches[layer * 2 + 1]),
            ));
        }

        let mut outputs = self.session.run(inputs)?;
        let next_token = greedy_last_token(&outputs)?;
        let next_caches = remove_present_caches(&mut outputs)?;
        Ok((next_token, next_caches))
    }
}

struct Generation {
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    stopped: bool,
}

/// Start the private Refiner HTTP server and wait until it exits.
pub(crate) async fn run(args: Args) -> Result<()> {
    info!(model_root = %args.model_root.display(), "loading AgenticASR Refiner ONNX runtime");
    let engine = RefinerEngine::load(&args)?;
    let state = RefinerState {
        engine: Arc::new(Mutex::new(engine)),
    };
    let router = Router::new()
        .route("/", get(service_info))
        .route("/health", get(health))
        .route("/models", get(models))
        .route("/v1/models", get(models))
        .route("/chat/completions", post(chat_completions))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(SocketAddr::new(args.host, args.port))
        .await
        .context("failed to bind Refiner HTTP listener")?;
    let address = listener
        .local_addr()
        .context("failed to read Refiner listener address")?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "status": "ready",
            "protocol_version": 1,
            "engine": "agentic-asr-refiner-onnx",
            "port": address.port()
        }))?
    );
    info!(%address, "AgenticASR Refiner ONNX HTTP service ready");
    axum::serve(listener, router)
        .await
        .context("Refiner HTTP server failed")
}

async fn service_info() -> Json<serde_json::Value> {
    Json(json!({
        "service": "xtalk-local-model-runtime",
        "engine": "agentic-asr-refiner-onnx",
        "endpoints": ["/health", "/v1/models", "/v1/chat/completions"]
    }))
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "protocol_version": 1,
        "engine": "agentic-asr-refiner-onnx"
    }))
}

async fn models() -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "xtalk"}]
    }))
}

async fn chat_completions(
    State(state): State<RefinerState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, RefinerApiError> {
    if request.model != MODEL_ID {
        return Err(RefinerApiError::bad_request(format!(
            "unknown model `{}`; expected `{MODEL_ID}`",
            request.model
        )));
    }
    if request.messages.is_empty() {
        return Err(RefinerApiError::bad_request("messages must not be empty"));
    }
    if request.max_tokens == 0 || request.max_tokens > MAX_GENERATION_TOKENS {
        return Err(RefinerApiError::bad_request(format!(
            "max_tokens must be between 1 and {MAX_GENERATION_TOKENS}"
        )));
    }
    if !request.temperature.is_finite() || request.temperature < 0.0 {
        return Err(RefinerApiError::bad_request(
            "temperature must be a non-negative finite number",
        ));
    }

    let engine = Arc::clone(&state.engine);
    let messages = request.messages;
    let max_tokens = request.max_tokens;
    let generation = tokio::task::spawn_blocking(move || {
        engine
            .lock()
            .map_err(|_| anyhow::anyhow!("Refiner engine lock is poisoned"))?
            .generate(&messages, max_tokens)
    })
    .await
    .map_err(|error| RefinerApiError::internal(format!("inference task failed: {error}")))?
    .map_err(|error| {
        error!(%error, "AgenticASR Refiner inference failed");
        RefinerApiError::internal(error.to_string())
    })?;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let finish_reason = if generation.stopped { "stop" } else { "length" };
    let total_tokens = generation.prompt_tokens + generation.completion_tokens;
    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{created}"),
        object: "chat.completion",
        created,
        model: MODEL_ID,
        choices: vec![ChatChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: generation.text,
            },
            finish_reason,
        }],
        usage: TokenUsage {
            prompt_tokens: generation.prompt_tokens,
            completion_tokens: generation.completion_tokens,
            total_tokens,
        },
    }))
}

fn default_max_tokens() -> usize {
    DEFAULT_MAX_TOKENS
}

fn build_prompt(messages: &[ChatMessage]) -> Result<String> {
    let mut prompt = String::from("<s>");
    for message in messages {
        if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
            bail!("unsupported chat role `{}`", message.role);
        }
        prompt.push_str("<|im_start|>");
        prompt.push_str(&message.role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    Ok(prompt)
}

fn empty_caches() -> Result<Vec<DynValue>> {
    (0..LAYER_COUNT * 2)
        .map(|_| {
            Ok(
                Tensor::from_array(([1, KEY_VALUE_HEADS, 0, HEAD_DIMENSION], Vec::<f32>::new()))?
                    .upcast()
                    .into(),
            )
        })
        .collect()
}

fn greedy_last_token(outputs: &SessionOutputs<'_>) -> Result<i64> {
    let (shape, logits) = outputs["logits"]
        .try_extract_tensor::<f32>()
        .context("Refiner logits output is not float32")?;
    if shape.len() != 3 || shape[0] != 1 {
        bail!("unexpected Refiner logits shape: {shape:?}");
    }
    let vocabulary_size = usize::try_from(shape[2]).context("invalid vocabulary size")?;
    if vocabulary_size == 0 || logits.len() < vocabulary_size {
        bail!("Refiner returned empty logits");
    }
    let last_logits = &logits[logits.len() - vocabulary_size..];
    let (token, _) = last_logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .context("Refiner returned no token logits")?;
    i64::try_from(token).context("token ID exceeds i64")
}

fn remove_present_caches(outputs: &mut SessionOutputs<'_>) -> Result<Vec<DynValue>> {
    let mut caches = Vec::with_capacity(LAYER_COUNT * 2);
    for layer in 0..LAYER_COUNT {
        for kind in ["key", "value"] {
            let name = format!("present.{layer}.{kind}");
            caches.push(
                outputs
                    .remove(&name)
                    .with_context(|| format!("ONNX output is missing: {name}"))?,
            );
        }
    }
    Ok(caches)
}

#[cfg(test)]
mod tests {
    use super::{build_prompt, ChatMessage};

    #[test]
    fn prompt_forces_the_no_thinking_assistant_prefix() {
        let prompt = build_prompt(&[
            ChatMessage {
                role: "system".to_owned(),
                content: "system".to_owned(),
            },
            ChatMessage {
                role: "user".to_owned(),
                content: "raw".to_owned(),
            },
        ])
        .expect("build prompt");
        assert_eq!(
            prompt,
            "<s><|im_start|>system\nsystem<|im_end|>\n<|im_start|>user\nraw<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
