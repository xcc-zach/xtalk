//! Background wake-word lifecycle and sherpa-onnx process supervision.

use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use pinyin::ToPinyin;
use serde::{Deserialize, Serialize};
use tauri::{path::BaseDirectory, AppHandle, Emitter, Manager};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};
use thiserror::Error;
use tokio::{sync::Mutex, time::sleep};

const SIDECAR_NAME: &str = "sherpa-onnx-keyword-spotter-microphone";
const SETTINGS_FILE: &str = "wake-word-settings.json";
const SETTINGS_VERSION: u16 = 1;
const MODEL_RESOURCE: &str = "models/wake-word";
const SHERPA_RUNTIME_RESOURCE: &str = "managed-runtime/ort";
const ENCODER_FILE: &str = "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx";
const DECODER_FILE: &str = "decoder-epoch-13-avg-2-chunk-16-left-64.onnx";
const JOINER_FILE: &str = "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx";
const TOKENS_FILE: &str = "tokens.txt";
const ENGLISH_LEXICON_FILE: &str = "en.phone";
const GENERATED_KEYWORDS_FILE: &str = "wake-word-keywords.txt";
const DEFAULT_WAKE_PHRASE: &str = "";
const DEFAULT_WAKE_THRESHOLD: f32 = 0.05;
const MAX_WAKE_PHRASE_CHARS: usize = 32;
const STATUS_EVENT: &str = "wake-word-status-changed";
const DETECTED_EVENT: &str = "wake-word-detected";
const STARTUP_SETTLE_TIME: Duration = Duration::from_millis(300);
const MAX_KEYWORD_OUTPUT_BYTES: usize = 64 * 1024;

/// Runtime state exposed to the trusted desktop WebView.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WakeWordState {
    /// Background detection is not selected.
    Disabled,
    /// The packaged detector is starting.
    Starting,
    /// The detector owns the microphone and is waiting for the phrase.
    Listening,
    /// Detection is selected but paused while a conversation owns the microphone.
    Paused,
    /// The detector could not start or terminated unexpectedly.
    Error,
}

/// Wake-word settings and live state returned to the trusted WebView.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct NativeWakeWordSettings {
    /// Whether background wake-word detection is selected.
    pub(crate) enabled: bool,
    /// User-selected phrase recognized by the generated keyword file.
    pub(crate) phrase: String,
    /// Minimum acoustic probability required to trigger the wake phrase.
    pub(crate) threshold: f32,
    /// Current detector lifecycle state.
    pub(crate) state: WakeWordState,
    /// Last startup or runtime failure, when one exists.
    pub(crate) last_error: Option<String>,
}

/// Event emitted after the configured wake phrase is detected.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WakeWordDetected {
    /// Phrase reported to the App UI.
    pub(crate) phrase: String,
}

#[derive(Deserialize, Serialize)]
struct PersistedWakeWordSettings {
    version: u16,
    enabled: bool,
    #[serde(default = "default_wake_phrase")]
    phrase: String,
    #[serde(default = "default_wake_threshold")]
    threshold: f32,
}

struct WakeWordRuntimeState {
    child: Option<CommandChild>,
    generation: u64,
    state: WakeWordState,
    last_error: Option<String>,
}

/// Supervises the packaged local keyword-spotting process.
pub(crate) struct WakeWordSupervisor {
    enabled: AtomicBool,
    phrase: Mutex<String>,
    threshold: Mutex<f32>,
    operation_gate: Mutex<()>,
    runtime: Mutex<WakeWordRuntimeState>,
}

impl WakeWordSupervisor {
    /// Loads the user's persisted selection and starts listening when enabled.
    pub(crate) async fn initialize(app: &AppHandle) -> Arc<Self> {
        let settings = load_settings(app).unwrap_or_else(|_| default_settings());
        let enabled = wake_word_is_effectively_enabled(settings.enabled, &settings.phrase);
        if settings.enabled != enabled {
            let _ = persist_settings(app, enabled, &settings.phrase, settings.threshold);
        }
        let supervisor = Arc::new(Self {
            enabled: AtomicBool::new(enabled),
            phrase: Mutex::new(settings.phrase),
            threshold: Mutex::new(settings.threshold),
            operation_gate: Mutex::new(()),
            runtime: Mutex::new(WakeWordRuntimeState {
                child: None,
                generation: 0,
                state: if enabled {
                    WakeWordState::Starting
                } else {
                    WakeWordState::Disabled
                },
                last_error: None,
            }),
        });
        if enabled {
            if let Err(error) = supervisor.start(app).await {
                supervisor.set_error(app, error.to_string()).await;
            }
        }
        supervisor
    }

    /// Returns whether background detection is selected.
    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Returns the persisted selection and current detector state.
    pub(crate) async fn settings(&self) -> NativeWakeWordSettings {
        let phrase = self.phrase.lock().await.clone();
        let threshold = *self.threshold.lock().await;
        let runtime = self.runtime.lock().await;
        NativeWakeWordSettings {
            enabled: self.is_enabled(),
            phrase,
            threshold,
            state: runtime.state,
            last_error: runtime.last_error.clone(),
        }
    }

    /// Enables or disables background detection and persists the selection.
    pub(crate) async fn set_enabled(
        self: &Arc<Self>,
        app: &AppHandle,
        enabled: bool,
        listen_immediately: bool,
    ) -> Result<NativeWakeWordSettings, WakeWordError> {
        let phrase = self.phrase.lock().await.clone();
        let enabled = wake_word_is_effectively_enabled(enabled, &phrase);
        if enabled == self.is_enabled() {
            if enabled && listen_immediately {
                self.resume(app).await?;
            } else if enabled {
                self.pause(app).await;
            }
            return Ok(self.settings().await);
        }

        if enabled {
            if !listen_immediately {
                let phrase = self.phrase.lock().await.clone();
                let threshold = *self.threshold.lock().await;
                WakeWordResources::resolve(app, &phrase, threshold)?;
            }
            self.enabled.store(true, Ordering::Release);
            if listen_immediately {
                if let Err(error) = self.start(app).await {
                    self.enabled.store(false, Ordering::Release);
                    self.set_error(app, error.to_string()).await;
                    return Err(error);
                }
            } else {
                let mut runtime = self.runtime.lock().await;
                runtime.state = WakeWordState::Paused;
                runtime.last_error = None;
                drop(runtime);
                self.emit_status(app).await;
            }
            let phrase = self.phrase.lock().await.clone();
            let threshold = *self.threshold.lock().await;
            persist_settings(app, true, &phrase, threshold)?;
        } else {
            self.enabled.store(false, Ordering::Release);
            self.stop(WakeWordState::Disabled).await;
            let phrase = self.phrase.lock().await.clone();
            let threshold = *self.threshold.lock().await;
            persist_settings(app, false, &phrase, threshold)?;
            self.emit_status(app).await;
        }
        Ok(self.settings().await)
    }

    /// Updates the persisted wake phrase and restarts listening when appropriate.
    pub(crate) async fn set_phrase(
        self: &Arc<Self>,
        app: &AppHandle,
        phrase: String,
        listen_immediately: bool,
    ) -> Result<NativeWakeWordSettings, WakeWordError> {
        if phrase.trim().is_empty() {
            let threshold = *self.threshold.lock().await;
            self.enabled.store(false, Ordering::Release);
            self.stop(WakeWordState::Disabled).await;
            *self.phrase.lock().await = String::new();
            persist_settings(app, false, "", threshold)?;
            self.emit_status(app).await;
            return Ok(self.settings().await);
        }
        let phrase = normalize_wake_phrase(&phrase)?;
        let threshold = *self.threshold.lock().await;
        WakeWordResources::resolve(app, &phrase, threshold)?;
        if phrase == *self.phrase.lock().await {
            return Ok(self.settings().await);
        }

        let enabled = self.is_enabled();
        persist_settings(app, enabled, &phrase, threshold)?;
        if enabled {
            self.stop(WakeWordState::Paused).await;
        }
        *self.phrase.lock().await = phrase;

        if enabled && listen_immediately {
            if let Err(error) = self.start(app).await {
                self.set_error(app, error.to_string()).await;
                return Err(error);
            }
        } else {
            self.emit_status(app).await;
        }
        Ok(self.settings().await)
    }

    /// Updates the acoustic trigger threshold and restarts listening when appropriate.
    pub(crate) async fn set_threshold(
        self: &Arc<Self>,
        app: &AppHandle,
        threshold: f32,
        listen_immediately: bool,
    ) -> Result<NativeWakeWordSettings, WakeWordError> {
        let threshold = normalize_wake_threshold(threshold)?;
        let phrase = self.phrase.lock().await.clone();
        if !phrase.trim().is_empty() {
            WakeWordResources::resolve(app, &phrase, threshold)?;
        }
        if threshold == *self.threshold.lock().await {
            return Ok(self.settings().await);
        }

        let enabled = self.is_enabled();
        persist_settings(app, enabled, &phrase, threshold)?;
        if enabled {
            self.stop(WakeWordState::Paused).await;
        }
        *self.threshold.lock().await = threshold;

        if enabled && listen_immediately {
            if let Err(error) = self.start(app).await {
                self.set_error(app, error.to_string()).await;
                return Err(error);
            }
        } else {
            self.emit_status(app).await;
        }
        Ok(self.settings().await)
    }

    /// Pauses listening and releases the microphone for an XTalk conversation.
    pub(crate) async fn pause(&self, app: &AppHandle) {
        if self.is_enabled() {
            self.stop(WakeWordState::Paused).await;
            self.emit_status(app).await;
        }
    }

    /// Resumes listening after an XTalk conversation releases the microphone.
    pub(crate) async fn resume(
        self: &Arc<Self>,
        app: &AppHandle,
    ) -> Result<NativeWakeWordSettings, WakeWordError> {
        if self.is_enabled() {
            let should_start = self.runtime.lock().await.child.is_none();
            if should_start {
                self.start(app).await?;
            }
        }
        Ok(self.settings().await)
    }

    /// Stops the packaged detector during full application shutdown.
    pub(crate) async fn shutdown(&self) {
        self.stop(WakeWordState::Disabled).await;
    }

    async fn start(self: &Arc<Self>, app: &AppHandle) -> Result<(), WakeWordError> {
        let _operation_guard = self.operation_gate.lock().await;
        if self.runtime.lock().await.child.is_some() {
            return Ok(());
        }

        let phrase = self.phrase.lock().await.clone();
        let threshold = *self.threshold.lock().await;
        let resources = WakeWordResources::resolve(app, &phrase, threshold)?;
        let runtime_dir = app
            .path()
            .resolve(SHERPA_RUNTIME_RESOURCE, BaseDirectory::Resource)?;
        let generation = {
            let mut runtime = self.runtime.lock().await;
            runtime.generation += 1;
            runtime.state = WakeWordState::Starting;
            runtime.last_error = None;
            runtime.generation
        };
        self.emit_status(app).await;

        let args = resources.command_args();
        let command = app.shell().sidecar(SIDECAR_NAME)?.args(args);
        let command =
            crate::managed::configure_library_path(command, &runtime_dir).set_raw_out(true);
        let (events, child) = command.spawn()?;
        {
            let mut runtime = self.runtime.lock().await;
            runtime.child = Some(child);
            runtime.state = WakeWordState::Listening;
        }
        self.emit_status(app).await;

        let app_handle = app.clone();
        let supervisor = Arc::clone(self);
        tauri::async_runtime::spawn(async move {
            supervisor.monitor(app_handle, events, generation).await;
        });

        sleep(STARTUP_SETTLE_TIME).await;
        let runtime = self.runtime.lock().await;
        if runtime.child.is_none() || runtime.state == WakeWordState::Error {
            return Err(WakeWordError::Startup(
                runtime
                    .last_error
                    .clone()
                    .unwrap_or_else(|| "keyword spotter terminated during startup".to_owned()),
            ));
        }
        Ok(())
    }

    async fn stop(&self, state: WakeWordState) {
        let child = {
            let mut runtime = self.runtime.lock().await;
            runtime.state = state;
            runtime.last_error = None;
            runtime.child.take()
        };
        if let Some(child) = child {
            let _ = child.kill();
        }
    }

    async fn monitor(
        self: Arc<Self>,
        app: AppHandle,
        mut events: tauri::async_runtime::Receiver<CommandEvent>,
        generation: u64,
    ) {
        let mut stdout = KeywordOutputBuffer::default();
        let mut stderr = KeywordOutputBuffer::default();
        while let Some(event) = events.recv().await {
            match event {
                CommandEvent::Stdout(chunk) => {
                    if stdout.push(&chunk).is_some() {
                        self.handle_detection(&app, generation).await;
                        return;
                    }
                }
                CommandEvent::Stderr(chunk) => {
                    if stderr.push(&chunk).is_some() {
                        self.handle_detection(&app, generation).await;
                        return;
                    }
                }
                CommandEvent::Terminated(payload) => {
                    let expected = {
                        let mut runtime = self.runtime.lock().await;
                        if runtime.generation != generation {
                            return;
                        }
                        runtime.child = None;
                        matches!(
                            runtime.state,
                            WakeWordState::Paused | WakeWordState::Disabled
                        )
                    };
                    if !expected && self.is_enabled() {
                        self.set_error(
                            &app,
                            format!(
                                "keyword spotter terminated (code {:?}, signal {:?})",
                                payload.code, payload.signal
                            ),
                        )
                        .await;
                    }
                    return;
                }
                CommandEvent::Error(error) => {
                    if self.is_current(generation).await {
                        self.set_error(&app, error).await;
                    }
                    return;
                }
                _ => {}
            }
        }
    }

    async fn is_current(&self, generation: u64) -> bool {
        self.runtime.lock().await.generation == generation
    }

    async fn handle_detection(&self, app: &AppHandle, generation: u64) {
        let child = {
            let mut runtime = self.runtime.lock().await;
            if runtime.generation != generation {
                return;
            }
            runtime.state = WakeWordState::Paused;
            runtime.last_error = None;
            runtime.child.take()
        };
        if let Some(child) = child {
            let _ = child.kill();
        }
        self.emit_status(app).await;
        if let Some(window) = app.get_webview_window("main") {
            let _ = window.unminimize();
            let _ = window.show();
            let _ = window.set_focus();
        }
        let phrase = self.phrase.lock().await.clone();
        let _ = app.emit(DETECTED_EVENT, WakeWordDetected { phrase });
    }

    async fn set_error(&self, app: &AppHandle, error: String) {
        let mut runtime = self.runtime.lock().await;
        runtime.child = None;
        runtime.state = WakeWordState::Error;
        runtime.last_error = Some(error);
        drop(runtime);
        self.emit_status(app).await;
    }

    async fn emit_status(&self, app: &AppHandle) {
        let _ = app.emit(STATUS_EVENT, self.settings().await);
    }
}

struct WakeWordResources {
    encoder: PathBuf,
    decoder: PathBuf,
    joiner: PathBuf,
    tokens: PathBuf,
    keywords: PathBuf,
    threshold: f32,
}

impl WakeWordResources {
    fn resolve(app: &AppHandle, phrase: &str, threshold: f32) -> Result<Self, WakeWordError> {
        let root = app
            .path()
            .resolve(MODEL_RESOURCE, BaseDirectory::Resource)?;
        let tokens = require_resource(&root, TOKENS_FILE)?;
        let english_lexicon = require_resource(&root, ENGLISH_LEXICON_FILE)?;
        let keywords = app.path().app_data_dir()?.join(GENERATED_KEYWORDS_FILE);
        let parent = keywords
            .parent()
            .ok_or(WakeWordError::InvalidSettingsPath)?;
        fs::create_dir_all(parent)?;
        fs::write(
            &keywords,
            build_keyword_definition(
                phrase,
                threshold,
                &fs::read_to_string(&tokens)?,
                &fs::read_to_string(english_lexicon)?,
            )?,
        )?;
        Ok(Self {
            encoder: require_resource(&root, ENCODER_FILE)?,
            decoder: require_resource(&root, DECODER_FILE)?,
            joiner: require_resource(&root, JOINER_FILE)?,
            tokens,
            keywords,
            threshold,
        })
    }

    fn command_args(&self) -> Vec<String> {
        vec![
            format!("--tokens={}", self.tokens.display()),
            format!("--encoder={}", self.encoder.display()),
            format!("--decoder={}", self.decoder.display()),
            format!("--joiner={}", self.joiner.display()),
            format!("--keywords-file={}", self.keywords.display()),
            "--provider=cpu".to_owned(),
            "--num-threads=1".to_owned(),
            "--keywords-score=3.0".to_owned(),
            format!("--keywords-threshold={}", self.threshold),
        ]
    }
}

fn require_resource(root: &Path, filename: &str) -> Result<PathBuf, WakeWordError> {
    let path = root.join(filename);
    if !path.is_file() {
        return Err(WakeWordError::MissingResource(path));
    }
    Ok(path)
}

fn normalize_wake_phrase(phrase: &str) -> Result<String, WakeWordError> {
    let phrase = phrase.split_whitespace().collect::<Vec<_>>().join(" ");
    if phrase.is_empty() {
        return Err(WakeWordError::InvalidPhrase(
            "the wake phrase cannot be empty".to_owned(),
        ));
    }
    if phrase.chars().count() > MAX_WAKE_PHRASE_CHARS {
        return Err(WakeWordError::InvalidPhrase(format!(
            "the wake phrase cannot exceed {MAX_WAKE_PHRASE_CHARS} characters"
        )));
    }
    if let Some(character) = phrase.chars().find(|character| {
        !character.is_whitespace()
            && !character.is_ascii_alphabetic()
            && *character != '\''
            && character.to_pinyin().is_none()
    }) {
        return Err(WakeWordError::InvalidPhrase(format!(
            "unsupported character in wake phrase: {character}"
        )));
    }
    Ok(phrase)
}

fn build_keyword_definition(
    phrase: &str,
    threshold: f32,
    tokens_content: &str,
    english_lexicon_content: &str,
) -> Result<String, WakeWordError> {
    let phrase = normalize_wake_phrase(phrase)?;
    let model_tokens = tokens_content
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .collect::<HashSet<_>>();
    let english_lexicon = parse_english_lexicon(english_lexicon_content);
    let mut tokens = Vec::new();
    let mut english_word = String::new();

    for character in phrase.chars() {
        if character.is_ascii_alphabetic() || character == '\'' {
            english_word.push(character.to_ascii_uppercase());
            continue;
        }
        append_english_word(&mut tokens, &mut english_word, &english_lexicon)?;
        if character.is_whitespace() {
            continue;
        }
        let pinyin = character.to_pinyin().ok_or_else(|| {
            WakeWordError::InvalidPhrase(format!(
                "cannot convert wake phrase character: {character}"
            ))
        })?;
        let plain = pinyin.plain();
        let with_tone = pinyin.with_tone();
        let initial = PINYIN_INITIALS
            .iter()
            .find(|initial| plain.starts_with(**initial))
            .copied()
            .unwrap_or("");
        let final_with_tone = if initial.is_empty() {
            with_tone
        } else if let Some(final_with_tone) = with_tone.strip_prefix(initial) {
            tokens.push(initial.to_owned());
            final_with_tone
        } else {
            with_tone
        };
        if !final_with_tone.is_empty() {
            tokens.push(final_with_tone.to_owned());
        }
    }
    append_english_word(&mut tokens, &mut english_word, &english_lexicon)?;

    for token in &tokens {
        if !model_tokens.contains(token.as_str()) {
            return Err(WakeWordError::UnsupportedModelToken(token.clone()));
        }
    }
    let label = phrase.replace(' ', "_");
    Ok(format!("{} :3.0 #{threshold} @{label}\n", tokens.join(" ")))
}

const PINYIN_INITIALS: [&str; 23] = [
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r",
    "z", "c", "s", "y", "w",
];

fn parse_english_lexicon(content: &str) -> HashMap<String, Vec<String>> {
    let mut lexicon = HashMap::new();
    for line in content.lines() {
        let mut fields = line.split_whitespace();
        let Some(raw_word) = fields.next() else {
            continue;
        };
        let word = raw_word
            .split_once('(')
            .map_or(raw_word, |(base, _)| base)
            .to_ascii_uppercase();
        let phones = fields.map(str::to_owned).collect::<Vec<_>>();
        if !phones.is_empty() {
            lexicon.entry(word).or_insert(phones);
        }
    }
    lexicon
}

fn append_english_word(
    tokens: &mut Vec<String>,
    word: &mut String,
    lexicon: &HashMap<String, Vec<String>>,
) -> Result<(), WakeWordError> {
    if word.is_empty() {
        return Ok(());
    }
    let phones = lexicon
        .get(word)
        .ok_or_else(|| WakeWordError::UnsupportedEnglishWord(word.clone()))?;
    tokens.extend(phones.iter().cloned());
    word.clear();
    Ok(())
}

fn extract_keyword(line: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(line).ok()?;
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    let payload: serde_json::Value = serde_json::from_str(&text[start..=end]).ok()?;
    payload
        .get("keyword")?
        .as_str()
        .map(str::trim)
        .filter(|keyword| !keyword.is_empty())
        .map(str::to_owned)
}

#[derive(Default)]
struct KeywordOutputBuffer {
    bytes: Vec<u8>,
}

impl KeywordOutputBuffer {
    fn push(&mut self, chunk: &[u8]) -> Option<String> {
        self.bytes.extend_from_slice(chunk);
        loop {
            let Some(start) = self.bytes.iter().position(|byte| *byte == b'{') else {
                self.bytes.clear();
                return None;
            };
            if start > 0 {
                self.bytes.drain(..start);
            }
            let Some(end) = self.bytes.iter().position(|byte| *byte == b'}') else {
                if self.bytes.len() > MAX_KEYWORD_OUTPUT_BYTES {
                    self.bytes.clear();
                }
                return None;
            };
            let payload = self.bytes.drain(..=end).collect::<Vec<_>>();
            if let Some(keyword) = extract_keyword(&payload) {
                return Some(keyword);
            }
        }
    }
}

fn default_wake_phrase() -> String {
    DEFAULT_WAKE_PHRASE.to_owned()
}

fn wake_word_is_effectively_enabled(enabled: bool, phrase: &str) -> bool {
    enabled && !phrase.trim().is_empty()
}

fn default_wake_threshold() -> f32 {
    DEFAULT_WAKE_THRESHOLD
}

fn normalize_wake_threshold(threshold: f32) -> Result<f32, WakeWordError> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(WakeWordError::InvalidThreshold);
    }
    Ok(threshold)
}

fn default_settings() -> PersistedWakeWordSettings {
    PersistedWakeWordSettings {
        version: SETTINGS_VERSION,
        enabled: false,
        phrase: default_wake_phrase(),
        threshold: default_wake_threshold(),
    }
}

fn load_settings(app: &AppHandle) -> Result<PersistedWakeWordSettings, WakeWordError> {
    let path = settings_path(app)?;
    if !path.is_file() {
        return Ok(default_settings());
    }
    let settings: PersistedWakeWordSettings = serde_json::from_slice(&fs::read(path)?)?;
    if settings.version != SETTINGS_VERSION {
        return Err(WakeWordError::SettingsVersion);
    }
    Ok(settings)
}

fn persist_settings(
    app: &AppHandle,
    enabled: bool,
    phrase: &str,
    threshold: f32,
) -> Result<(), WakeWordError> {
    let path = settings_path(app)?;
    let parent = path.parent().ok_or(WakeWordError::InvalidSettingsPath)?;
    fs::create_dir_all(parent)?;
    fs::write(
        path,
        serde_json::to_vec_pretty(&PersistedWakeWordSettings {
            version: SETTINGS_VERSION,
            enabled,
            phrase: phrase.to_owned(),
            threshold,
        })?,
    )?;
    Ok(())
}

fn settings_path(app: &AppHandle) -> Result<PathBuf, WakeWordError> {
    Ok(app.path().app_data_dir()?.join(SETTINGS_FILE))
}

/// Failures produced while managing background keyword detection.
#[derive(Debug, Error)]
pub(crate) enum WakeWordError {
    /// A required packaged model file is unavailable.
    #[error("wake-word resource is missing: {0}")]
    MissingResource(PathBuf),
    /// The requested phrase cannot be represented safely.
    #[error("invalid wake phrase: {0}")]
    InvalidPhrase(String),
    /// The acoustic trigger threshold is outside the supported probability range.
    #[error("the wake-word threshold must be a finite number between 0 and 1")]
    InvalidThreshold,
    /// An English word is absent from the packaged pronunciation lexicon.
    #[error("unsupported English word in wake phrase: {0}")]
    UnsupportedEnglishWord(String),
    /// A generated phone is absent from the selected model vocabulary.
    #[error("wake phrase generated an unsupported model token: {0}")]
    UnsupportedModelToken(String),
    /// The persisted settings file has an unsupported version.
    #[error("the saved wake-word settings have an unsupported version")]
    SettingsVersion,
    /// The application-data settings path has no parent directory.
    #[error("the wake-word settings path is invalid")]
    InvalidSettingsPath,
    /// The detector stopped before its startup boundary completed.
    #[error("wake-word detector failed to start: {0}")]
    Startup(String),
    /// Filesystem access failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// JSON settings serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Tauri path resolution failed.
    #[error(transparent)]
    Path(#[from] tauri::Error),
    /// The packaged sidecar could not be launched.
    #[error(transparent)]
    Shell(#[from] tauri_plugin_shell::Error),
}

#[cfg(test)]
mod tests {
    use super::{
        build_keyword_definition, extract_keyword, wake_word_is_effectively_enabled,
        KeywordOutputBuffer, PersistedWakeWordSettings,
    };

    #[test]
    fn extracts_keyword_from_sherpa_display_output() {
        let line = r#"0: {"keyword":"你好小克","tokens":["n"]}"#.as_bytes();

        assert_eq!(extract_keyword(line).as_deref(), Some("你好小克"));
    }

    #[test]
    fn ignores_non_detection_output() {
        assert_eq!(extract_keyword(b"microphone initialized"), None);
        assert_eq!(extract_keyword(br#"{"keyword":""}"#), None);
    }

    #[test]
    fn extracts_first_keyword_without_waiting_for_a_newline() {
        let mut output = KeywordOutputBuffer::default();

        assert_eq!(output.push(b"\r0:{\"key"), None);
        assert_eq!(
            output.push(b"word\":\"\xe4\xbd\xa0\xe5\xa5\xbd\xe5\xb0\x8f\xe5\x85\x8b\","),
            None
        );
        assert_eq!(
            output.push(b"\"tokens\":[\"n\"]}").as_deref(),
            Some("你好小克")
        );
    }

    #[test]
    fn converts_chinese_wake_phrase_to_toned_pinyin_tokens() {
        let tokens = "n 1\nǐ 2\nh 3\nǎo 4\nx 5\niǎo 6\nk 7\nè 8\n";

        assert_eq!(
            build_keyword_definition("你好小克", 0.1, tokens, "").unwrap(),
            "n ǐ h ǎo x iǎo k è :3.0 #0.1 @你好小克\n"
        );
    }

    #[test]
    fn keeps_a_toned_syllabic_nasal_as_one_model_token() {
        let tokens = "ń 1\nh 2\nēng 3\n";

        assert_eq!(
            build_keyword_definition("嗯哼", 0.1, tokens, "").unwrap(),
            "ń h ēng :3.0 #0.1 @嗯哼\n"
        );
    }

    #[test]
    fn converts_english_wake_phrase_with_packaged_lexicon() {
        let tokens = "HH 1\nAH0 2\nL 3\nOW1 4\nW 5\nER1 6\nD 7\n";
        let lexicon = "HELLO HH AH0 L OW1\nWORLD W ER1 L D\n";

        assert_eq!(
            build_keyword_definition("hello world", 0.1, tokens, lexicon).unwrap(),
            "HH AH0 L OW1 W ER1 L D :3.0 #0.1 @hello_world\n"
        );
    }

    #[test]
    fn loads_empty_default_phrase_from_legacy_settings() {
        let settings: PersistedWakeWordSettings =
            serde_json::from_str(r#"{"version":1,"enabled":true}"#).unwrap();

        assert_eq!(settings.phrase, "");
        assert_eq!(settings.threshold, 0.05);
    }

    #[test]
    fn treats_an_empty_wake_phrase_as_disabled() {
        assert!(!wake_word_is_effectively_enabled(true, ""));
        assert!(!wake_word_is_effectively_enabled(true, "   "));
        assert!(wake_word_is_effectively_enabled(true, "hello"));
        assert!(!wake_word_is_effectively_enabled(false, "hello"));
    }
}
