//! Background wake-word lifecycle and sherpa-onnx process supervision.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

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
const KEYWORDS_FILE: &str = "keywords.txt";
const DEFAULT_WAKE_PHRASE: &str = "你好小克";
const STATUS_EVENT: &str = "wake-word-status-changed";
const DETECTED_EVENT: &str = "wake-word-detected";
const STARTUP_SETTLE_TIME: Duration = Duration::from_millis(300);

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
    /// Fixed phrase recognized by the packaged keyword file.
    pub(crate) phrase: &'static str,
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
    pub(crate) phrase: &'static str,
}

#[derive(Deserialize, Serialize)]
struct PersistedWakeWordSettings {
    version: u16,
    enabled: bool,
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
    operation_gate: Mutex<()>,
    runtime: Mutex<WakeWordRuntimeState>,
}

impl WakeWordSupervisor {
    /// Loads the user's persisted selection and starts listening when enabled.
    pub(crate) async fn initialize(app: &AppHandle) -> Arc<Self> {
        let enabled = load_enabled(app).unwrap_or(false);
        let supervisor = Arc::new(Self {
            enabled: AtomicBool::new(enabled),
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
        let runtime = self.runtime.lock().await;
        NativeWakeWordSettings {
            enabled: self.is_enabled(),
            phrase: DEFAULT_WAKE_PHRASE,
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
                WakeWordResources::resolve(app)?;
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
            persist_enabled(app, true)?;
        } else {
            self.enabled.store(false, Ordering::Release);
            self.stop(WakeWordState::Disabled).await;
            persist_enabled(app, false)?;
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

        let resources = WakeWordResources::resolve(app)?;
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
        let command = crate::managed::configure_library_path(command, &runtime_dir);
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
        while let Some(event) = events.recv().await {
            match event {
                CommandEvent::Stdout(line) | CommandEvent::Stderr(line) => {
                    if extract_keyword(&line).is_some() {
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
        let _ = app.emit(
            DETECTED_EVENT,
            WakeWordDetected {
                phrase: DEFAULT_WAKE_PHRASE,
            },
        );
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
}

impl WakeWordResources {
    fn resolve(app: &AppHandle) -> Result<Self, WakeWordError> {
        let root = app
            .path()
            .resolve(MODEL_RESOURCE, BaseDirectory::Resource)?;
        Ok(Self {
            encoder: require_resource(&root, ENCODER_FILE)?,
            decoder: require_resource(&root, DECODER_FILE)?,
            joiner: require_resource(&root, JOINER_FILE)?,
            tokens: require_resource(&root, TOKENS_FILE)?,
            keywords: require_resource(&root, KEYWORDS_FILE)?,
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
            "--keywords-threshold=0.25".to_owned(),
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

fn load_enabled(app: &AppHandle) -> Result<bool, WakeWordError> {
    let path = settings_path(app)?;
    if !path.is_file() {
        return Ok(false);
    }
    let settings: PersistedWakeWordSettings = serde_json::from_slice(&fs::read(path)?)?;
    if settings.version != SETTINGS_VERSION {
        return Err(WakeWordError::SettingsVersion);
    }
    Ok(settings.enabled)
}

fn persist_enabled(app: &AppHandle, enabled: bool) -> Result<(), WakeWordError> {
    let path = settings_path(app)?;
    let parent = path.parent().ok_or(WakeWordError::InvalidSettingsPath)?;
    fs::create_dir_all(parent)?;
    fs::write(
        path,
        serde_json::to_vec_pretty(&PersistedWakeWordSettings {
            version: SETTINGS_VERSION,
            enabled,
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
    use super::extract_keyword;

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
}
