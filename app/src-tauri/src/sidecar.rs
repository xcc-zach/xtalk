//! Lifecycle management for the packaged Python sidecar.

use std::{
    fs,
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{async_runtime::Receiver, path::BaseDirectory, AppHandle, Manager};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};
use thiserror::Error;
use tokio::{
    sync::{watch, Mutex},
    time::{sleep, timeout, Instant},
};

use crate::{
    credentials::{self, CredentialError},
    managed::{inspect_model_config, ManagedError, ManagedModelPlan, ManagedServices},
};

const SIDECAR_NAME: &str = "app-backend";
const DESKTOP_ANONYMOUS_USER_ID: &str = "xtalk-desktop-user";
const PROTOCOL_VERSION: u16 = 1;
const MODEL_CONFIG_SELECTION_FILE: &str = "model-config-selection.json";
const MODEL_CONFIG_SELECTION_VERSION: u16 = 1;
const MAX_MODEL_CONFIG_BYTES: u64 = 1024 * 1024;
const VAD_MODEL_RESOURCE: &str = "models/audio/silero_vad.onnx";
const DESKTOP_VAD_THRESHOLD: f64 = 0.7;
const BUILTIN_TOOLS_RESOURCE: &str = "tools";
const RECOMMENDED_MODEL_CONFIG_RESOURCE: &str = "examples/local_models_matcha.json";
const SIDECAR_RUNTIME_RESOURCE: &str = "app-backend-runtime";
const READY_TIMEOUT: Duration = Duration::from_secs(30);
const HEALTH_TIMEOUT: Duration = Duration::from_secs(5);
const HEALTH_RETRY_INTERVAL: Duration = Duration::from_millis(100);
const HEALTH_REQUEST_TIMEOUT: Duration = Duration::from_secs(1);
const GRACEFUL_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
const FORCED_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);
const CONTROL_REQUEST_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_READY_LINE_BYTES: usize = 4 * 1024;
const MAX_HTTP_RESPONSE_BYTES: usize = 16 * 1024;

/// Validates and inspects a selected model configuration for managed services.
pub(crate) fn inspect_managed_model_config(
    config_path: &Path,
) -> Result<ManagedModelPlan, BackendError> {
    let config_path = validate_model_config_path(config_path)?;
    Ok(inspect_model_config(&config_path)?)
}

/// Connection details returned to the trusted WebView.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct NativeBackendConnection {
    /// The loopback HTTP origin for the current sidecar instance.
    pub(crate) origin: String,
    /// The random token authenticating this application launch.
    pub(crate) launch_token: String,
}

/// Persisted model configuration selected by the desktop user.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct NativeModelConfigSelection {
    /// Canonical external JSON configuration path, when one is selected.
    pub(crate) config_path: Option<PathBuf>,
}

struct BackendSupervisorState {
    config_path: Option<PathBuf>,
    manager: Option<Arc<BackendManager>>,
}

/// Supervises the optional sidecar selected by the desktop user.
pub(crate) struct BackendSupervisor {
    state: Mutex<BackendSupervisorState>,
    operation_gate: Mutex<()>,
    app_close_started: AtomicBool,
}

impl BackendSupervisor {
    /// Loads the persisted selection without starting user-selected services.
    ///
    /// Startup is deferred until the WebView subscribes to managed-model
    /// progress events and explicitly ensures the selected backend is running.
    pub(crate) async fn initialize(app: &AppHandle) -> Arc<Self> {
        let config_path = match resolve_initial_model_config(app) {
            Ok(config_path) => config_path,
            Err(_) => {
                eprintln!("the saved model configuration selection is unavailable");
                None
            }
        };
        Arc::new(Self {
            state: Mutex::new(BackendSupervisorState {
                config_path,
                manager: None,
            }),
            operation_gate: Mutex::new(()),
            app_close_started: AtomicBool::new(false),
        })
    }

    /// Returns the currently selected external model configuration.
    pub(crate) async fn selection(&self) -> NativeModelConfigSelection {
        NativeModelConfigSelection {
            config_path: self.state.lock().await.config_path.clone(),
        }
    }

    /// Returns connection details for the active sidecar.
    pub(crate) async fn connection(&self) -> Result<NativeBackendConnection, BackendError> {
        let manager = self
            .state
            .lock()
            .await
            .manager
            .clone()
            .ok_or(BackendError::Unavailable)?;
        manager.connection()
    }

    /// Starts the selected backend when no healthy instance is running.
    pub(crate) async fn ensure_started(
        &self,
        app: &AppHandle,
    ) -> Result<NativeBackendConnection, BackendError> {
        let _operation_guard = self.operation_gate.lock().await;
        let (config_path, previous_manager) = {
            let mut state = self.state.lock().await;
            if let Some(manager) = state.manager.as_ref() {
                if let Ok(connection) = manager.connection() {
                    return Ok(connection);
                }
            }
            (state.config_path.clone(), state.manager.take())
        };

        if let Some(manager) = previous_manager {
            manager.shutdown().await?;
        }
        let config_path = config_path.ok_or(BackendError::Unavailable)?;
        let manager = BackendManager::start(app, config_path).await?;
        let connection = manager.connection()?;
        self.state.lock().await.manager = Some(manager);
        Ok(connection)
    }

    /// Restarts the sidecar with a validated user-selected configuration.
    pub(crate) async fn apply_model_config(
        &self,
        app: &AppHandle,
        config_path: PathBuf,
    ) -> Result<NativeBackendConnection, BackendError> {
        let config_path = validate_model_config_path(&config_path)?;
        let _operation_guard = self.operation_gate.lock().await;
        let (previous_config_path, previous_manager) = {
            let mut state = self.state.lock().await;
            (state.config_path.clone(), state.manager.take())
        };

        if let Some(manager) = previous_manager {
            manager.shutdown().await?;
        }

        let manager = match BackendManager::start(app, config_path.clone()).await {
            Ok(manager) => manager,
            Err(error) => {
                self.restore_previous_manager(app, previous_config_path)
                    .await;
                return Err(error);
            }
        };
        let connection = match manager.connection() {
            Ok(connection) => connection,
            Err(error) => {
                let _ = manager.shutdown().await;
                self.restore_previous_manager(app, previous_config_path)
                    .await;
                return Err(error);
            }
        };
        if let Err(error) = persist_model_config_selection(app, &config_path) {
            let _ = manager.shutdown().await;
            self.restore_previous_manager(app, previous_config_path)
                .await;
            return Err(error);
        }

        let mut state = self.state.lock().await;
        state.config_path = Some(config_path);
        state.manager = Some(manager);
        Ok(connection)
    }

    /// Restarts the sidecar with the currently selected model configuration.
    pub(crate) async fn restart(
        &self,
        app: &AppHandle,
    ) -> Result<NativeBackendConnection, BackendError> {
        let config_path = self
            .state
            .lock()
            .await
            .config_path
            .clone()
            .ok_or(BackendError::Unavailable)?;
        self.apply_model_config(app, config_path).await
    }

    async fn restore_previous_manager(&self, app: &AppHandle, config_path: Option<PathBuf>) {
        let manager = if let Some(path) = config_path.as_ref() {
            match BackendManager::start(app, path.clone()).await {
                Ok(manager) => Some(manager),
                Err(_) => {
                    eprintln!("the previous app-backend configuration could not be restored");
                    None
                }
            }
        } else {
            None
        };
        let mut state = self.state.lock().await;
        state.config_path = config_path;
        state.manager = manager;
    }

    /// Marks the first main-window close request as accepted.
    pub(crate) fn begin_app_close(&self) -> bool {
        self.app_close_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Stops the active sidecar, when one exists.
    pub(crate) async fn shutdown(&self) -> Result<(), BackendError> {
        let _operation_guard = self.operation_gate.lock().await;
        let manager = self.state.lock().await.manager.take();
        if let Some(manager) = manager {
            manager.shutdown().await?;
        }
        Ok(())
    }
}

/// Owns the sidecar connection details and child-process lifecycle.
pub(crate) struct BackendManager {
    endpoint: String,
    token: String,
    managed_services: ManagedServices,
    child: Mutex<Option<CommandChild>>,
    exit: watch::Receiver<bool>,
    terminated: AtomicBool,
    shutting_down: AtomicBool,
    shutdown_gate: Mutex<()>,
}

impl BackendManager {
    /// Starts the packaged sidecar and completes its readiness handshake.
    async fn start(app: &AppHandle, config_path: PathBuf) -> Result<Arc<Self>, BackendError> {
        let token = generate_launch_token()?;
        let vad_model_path = resolve_required_resource(app, VAD_MODEL_RESOURCE, false)?;
        let builtin_tools_root = resolve_required_resource(app, BUILTIN_TOOLS_RESOURCE, true)?;
        resolve_required_resource(app, SIDECAR_RUNTIME_RESOURCE, true)?;
        let sidecar_directory = sidecar_working_directory()?;
        validate_sidecar_runtime(&sidecar_directory)?;
        let data_dir = app.path().app_data_dir()?;
        std::fs::create_dir_all(&data_dir)?;
        let credential_environment = credentials::sidecar_environment(app).await?;
        let (managed_services, config_overlay) =
            ManagedServices::start(app, &config_path, &data_dir).await?;

        let startup = StartupMessage {
            protocol_version: PROTOCOL_VERSION,
            token: &token,
            config_path,
            data_dir,
            builtin_tools_root,
            origins: allowed_origins(),
            config_fallbacks: build_config_fallbacks(&vad_model_path),
            config_overlay,
            anonymous_user_id: DESKTOP_ANONYMOUS_USER_ID,
        };
        let mut startup_line = serde_json::to_vec(&startup)?;
        startup_line.push(b'\n');

        let mut command = match app.shell().sidecar(SIDECAR_NAME) {
            Ok(command) => command.current_dir(sidecar_directory),
            Err(error) => {
                managed_services.shutdown().await;
                return Err(error.into());
            }
        };
        for (name, value) in credential_environment {
            command = command.env(name, value);
        }
        let (mut events, mut child) = match command.spawn() {
            Ok(spawned) => spawned,
            Err(error) => {
                managed_services.shutdown().await;
                return Err(error.into());
            }
        };

        if let Err(error) = child.write(&startup_line) {
            let _ = child.kill();
            managed_services.shutdown().await;
            return Err(error.into());
        }

        let ready = match receive_ready(&mut events).await {
            Ok(ready) => ready,
            Err(error) => {
                let _ = child.kill();
                managed_services.shutdown().await;
                return Err(error);
            }
        };

        let endpoint = format!("http://127.0.0.1:{}", ready.port);
        if let Err(error) = wait_for_health(ready.port, &token).await {
            let _ = child.kill();
            managed_services.shutdown().await;
            return Err(error);
        }

        let (exit_sender, exit) = watch::channel(false);
        let manager = Arc::new(Self {
            endpoint,
            token,
            managed_services,
            child: Mutex::new(Some(child)),
            exit,
            terminated: AtomicBool::new(false),
            shutting_down: AtomicBool::new(false),
            shutdown_gate: Mutex::new(()),
        });

        let weak_manager = Arc::downgrade(&manager);
        tauri::async_runtime::spawn(monitor_sidecar(events, exit_sender, weak_manager));
        if let Some(failure) = manager.managed_services.failure_receiver() {
            let weak_manager = Arc::downgrade(&manager);
            tauri::async_runtime::spawn(monitor_managed_services(failure, weak_manager));
        }

        Ok(manager)
    }

    /// Returns the current loopback origin and launch token while the sidecar is usable.
    pub(crate) fn connection(&self) -> Result<NativeBackendConnection, BackendError> {
        if self.terminated.load(Ordering::Acquire) || self.shutting_down.load(Ordering::Acquire) {
            return Err(BackendError::Unavailable);
        }
        if !self.managed_services.is_healthy() {
            return Err(BackendError::Unavailable);
        }

        Ok(NativeBackendConnection {
            origin: self.endpoint.clone(),
            launch_token: self.token.clone(),
        })
    }

    /// Requests an authenticated shutdown, then kills the child after a finite grace period.
    pub(crate) async fn shutdown(&self) -> Result<(), BackendError> {
        let _shutdown_guard = self.shutdown_gate.lock().await;
        self.shutting_down.store(true, Ordering::Release);

        if self.terminated.load(Ordering::Acquire) {
            self.child.lock().await.take();
            self.managed_services.shutdown().await;
            return Ok(());
        }

        let endpoint = self.endpoint.clone();
        let token = self.token.clone();
        let request =
            match tokio::task::spawn_blocking(move || send_controlled_shutdown(&endpoint, &token))
                .await
            {
                Ok(result) => result,
                Err(error) => Err(BackendError::ShutdownTask(error.to_string())),
            };

        if let Err(error) = request {
            eprintln!("controlled app-backend shutdown request failed: {error}");
        } else if self.wait_for_exit(GRACEFUL_SHUTDOWN_TIMEOUT).await {
            self.child.lock().await.take();
            self.managed_services.shutdown().await;
            return Ok(());
        }

        let child = self.child.lock().await.take();
        if let Some(child) = child {
            child.kill()?;
        }

        if !self.wait_for_exit(FORCED_SHUTDOWN_TIMEOUT).await {
            self.managed_services.shutdown().await;
            return Err(BackendError::ForcedShutdownTimedOut);
        }

        self.managed_services.shutdown().await;
        Ok(())
    }

    async fn wait_for_exit(&self, duration: Duration) -> bool {
        if self.terminated.load(Ordering::Acquire) {
            return true;
        }

        let mut exit = self.exit.clone();
        let wait = async move {
            while !*exit.borrow() {
                if exit.changed().await.is_err() {
                    break;
                }
            }
            *exit.borrow()
        };

        timeout(duration, wait).await.unwrap_or(false) || self.terminated.load(Ordering::Acquire)
    }
}

#[derive(Serialize)]
struct StartupMessage<'a> {
    protocol_version: u16,
    token: &'a str,
    config_path: PathBuf,
    data_dir: PathBuf,
    builtin_tools_root: PathBuf,
    origins: Vec<String>,
    config_fallbacks: Value,
    config_overlay: Value,
    anonymous_user_id: &'a str,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PersistedModelConfigSelection {
    version: u16,
    config_path: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadyMessage {
    #[serde(rename = "type")]
    kind: String,
    protocol_version: u16,
    port: u16,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HealthMessage {
    status: String,
    protocol_version: u16,
}

struct HttpResponse {
    status: u16,
    body: Vec<u8>,
}

/// Errors raised while starting, probing, or stopping the managed sidecar.
#[derive(Debug, Error)]
pub(crate) enum BackendError {
    #[error("failed to access an application path: {0}")]
    Tauri(#[from] tauri::Error),
    #[error("sidecar I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to encode or decode the sidecar protocol: {0}")]
    Json(#[from] serde_json::Error),
    #[error("failed to start or control app-backend: {0}")]
    Shell(#[from] tauri_plugin_shell::Error),
    #[error("failed to prepare a managed local model: {0}")]
    Managed(#[from] ManagedError),
    #[error("failed to resolve a required App credential: {0}")]
    Credential(#[from] CredentialError),
    #[error("could not generate a secure launch token: {0}")]
    Token(String),
    #[error("a required sidecar file or directory is missing")]
    MissingResource,
    #[error("the selected model configuration is missing or is not a regular file")]
    InvalidModelConfigPath,
    #[error("the selected model configuration exceeds the 1 MiB limit")]
    ModelConfigTooLarge,
    #[error("the selected model configuration root must be a JSON object")]
    InvalidModelConfigRoot,
    #[error("the saved model configuration selection has an unsupported version")]
    UnsupportedModelConfigSelection,
    #[error("the sidecar runtime resource has no parent directory")]
    InvalidResourceLayout,
    #[error("app-backend did not become ready within the startup timeout")]
    ReadyTimedOut,
    #[error("app-backend closed its event stream before readiness")]
    ReadyStreamClosed,
    #[error("app-backend terminated before readiness (code {code:?}, signal {signal:?})")]
    TerminatedBeforeReady {
        code: Option<i32>,
        signal: Option<i32>,
    },
    #[error("app-backend emitted an invalid readiness line: {0}")]
    InvalidReady(String),
    #[error("app-backend did not pass its authenticated health check")]
    HealthCheckFailed,
    #[error("app-backend returned an invalid HTTP response")]
    InvalidHttpResponse,
    #[error("app-backend returned an invalid health response")]
    InvalidHealth,
    #[error("app-backend is stopped or shutting down")]
    Unavailable,
    #[error("the controlled shutdown task failed: {0}")]
    ShutdownTask(String),
    #[error("the controlled shutdown endpoint is invalid")]
    InvalidShutdownEndpoint,
    #[error("the controlled shutdown endpoint returned a non-success HTTP status")]
    ShutdownRejected,
    #[error("app-backend remained alive after the forced shutdown timeout")]
    ForcedShutdownTimedOut,
}

async fn receive_ready(events: &mut Receiver<CommandEvent>) -> Result<ReadyMessage, BackendError> {
    timeout(READY_TIMEOUT, async {
        loop {
            match events.recv().await {
                Some(CommandEvent::Stdout(line)) => return validate_ready_line(&line),
                Some(CommandEvent::Stderr(_)) => {
                    eprintln!("app-backend wrote to stderr before readiness; content suppressed");
                }
                Some(CommandEvent::Error(_)) => {
                    return Err(BackendError::InvalidReady(
                        "process event reader failed".to_owned(),
                    ));
                }
                Some(CommandEvent::Terminated(payload)) => {
                    return Err(BackendError::TerminatedBeforeReady {
                        code: payload.code,
                        signal: payload.signal,
                    });
                }
                None => return Err(BackendError::ReadyStreamClosed),
                Some(_) => {
                    return Err(BackendError::InvalidReady(
                        "unsupported process event".to_owned(),
                    ));
                }
            }
        }
    })
    .await
    .map_err(|_| BackendError::ReadyTimedOut)?
}

fn validate_ready_line(line: &[u8]) -> Result<ReadyMessage, BackendError> {
    if line.is_empty() || line.len() > MAX_READY_LINE_BYTES {
        return Err(BackendError::InvalidReady(
            "line length is outside the allowed range".to_owned(),
        ));
    }

    let ready: ReadyMessage = serde_json::from_slice(line)?;
    if ready.kind != "ready" {
        return Err(BackendError::InvalidReady(
            "message type must be `ready`".to_owned(),
        ));
    }
    if ready.protocol_version != PROTOCOL_VERSION {
        return Err(BackendError::InvalidReady(format!(
            "protocol version {} is unsupported",
            ready.protocol_version
        )));
    }
    if ready.port == 0 {
        return Err(BackendError::InvalidReady(
            "port must be between 1 and 65535".to_owned(),
        ));
    }

    Ok(ready)
}

async fn monitor_sidecar(
    mut events: Receiver<CommandEvent>,
    exit_sender: watch::Sender<bool>,
    manager: std::sync::Weak<BackendManager>,
) {
    while let Some(event) = events.recv().await {
        match event {
            CommandEvent::Terminated(payload) => {
                eprintln!(
                    "app-backend terminated (code {:?}, signal {:?})",
                    payload.code, payload.signal
                );
                break;
            }
            CommandEvent::Stdout(_) => {
                eprintln!("app-backend emitted unexpected stdout after readiness");
            }
            CommandEvent::Stderr(_) => {
                eprintln!("app-backend emitted stderr after readiness; content suppressed");
            }
            CommandEvent::Error(_) => {
                eprintln!("app-backend process event reader failed");
            }
            _ => {
                eprintln!("app-backend emitted an unsupported process event");
            }
        }
    }

    if let Some(manager) = manager.upgrade() {
        manager.terminated.store(true, Ordering::Release);
    }
    let _ = exit_sender.send(true);
}

async fn monitor_managed_services(
    mut failure: watch::Receiver<bool>,
    manager: std::sync::Weak<BackendManager>,
) {
    while !*failure.borrow() {
        if failure.changed().await.is_err() {
            return;
        }
    }
    if let Some(manager) = manager.upgrade() {
        if let Err(error) = manager.shutdown().await {
            eprintln!("failed to stop app-backend after managed service failure: {error}");
        }
    }
}

fn generate_launch_token() -> Result<String, BackendError> {
    let mut bytes = [0_u8; 32];
    getrandom::fill(&mut bytes).map_err(|error| BackendError::Token(error.to_string()))?;

    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut token = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        token.push(HEX[(byte >> 4) as usize] as char);
        token.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(token)
}

fn allowed_origins() -> Vec<String> {
    let origins = vec![
        "tauri://localhost".to_owned(),
        "http://tauri.localhost".to_owned(),
        "https://tauri.localhost".to_owned(),
    ];
    #[cfg(debug_assertions)]
    {
        let mut origins = origins;
        origins.push("http://localhost:1420".to_owned());
        origins
    }
    #[cfg(not(debug_assertions))]
    {
        origins
    }
}

fn build_config_fallbacks(vad_model_path: &Path) -> Value {
    json!({
        "vad": {
            "type": "SileroVAD",
            "params": {
                "model_path": vad_model_path,
                "threshold": DESKTOP_VAD_THRESHOLD,
            },
        },
    })
}

fn resolve_initial_model_config(app: &AppHandle) -> Result<Option<PathBuf>, BackendError> {
    #[cfg(debug_assertions)]
    if let Some(config_path) = std::env::var_os("XTALK_APP_CONFIG_PATH") {
        return validate_model_config_path(&PathBuf::from(config_path)).map(Some);
    }

    load_model_config_selection(app)
}

fn load_model_config_selection(app: &AppHandle) -> Result<Option<PathBuf>, BackendError> {
    let selection_path = model_config_selection_path(app)?;
    if !selection_path.is_file() {
        return Ok(None);
    }

    let selection: PersistedModelConfigSelection =
        serde_json::from_slice(&fs::read(selection_path)?)?;
    if selection.version != MODEL_CONFIG_SELECTION_VERSION {
        return Err(BackendError::UnsupportedModelConfigSelection);
    }
    validate_model_config_path(&selection.config_path).map(Some)
}

fn persist_model_config_selection(app: &AppHandle, config_path: &Path) -> Result<(), BackendError> {
    let selection_path = model_config_selection_path(app)?;
    let parent = selection_path
        .parent()
        .ok_or(BackendError::InvalidResourceLayout)?;
    fs::create_dir_all(parent)?;
    fs::write(
        selection_path,
        serde_json::to_vec_pretty(&PersistedModelConfigSelection {
            version: MODEL_CONFIG_SELECTION_VERSION,
            config_path: config_path.to_path_buf(),
        })?,
    )?;
    Ok(())
}

fn model_config_selection_path(app: &AppHandle) -> Result<PathBuf, BackendError> {
    Ok(app
        .path()
        .app_config_dir()?
        .join(MODEL_CONFIG_SELECTION_FILE))
}

fn validate_model_config_path(path: &Path) -> Result<PathBuf, BackendError> {
    let canonical_path = path
        .canonicalize()
        .map_err(|_| BackendError::InvalidModelConfigPath)?;
    let metadata =
        fs::metadata(&canonical_path).map_err(|_| BackendError::InvalidModelConfigPath)?;
    if !metadata.is_file() {
        return Err(BackendError::InvalidModelConfigPath);
    }
    if metadata.len() > MAX_MODEL_CONFIG_BYTES {
        return Err(BackendError::ModelConfigTooLarge);
    }

    let config: Value = serde_json::from_slice(&fs::read(&canonical_path)?)?;
    if !config.is_object() {
        return Err(BackendError::InvalidModelConfigRoot);
    }
    Ok(canonical_path)
}

fn resolve_required_resource(
    app: &AppHandle,
    resource: &str,
    directory: bool,
) -> Result<PathBuf, BackendError> {
    let path = app.path().resolve(resource, BaseDirectory::Resource)?;
    let exists = if directory {
        path.is_dir()
    } else {
        path.is_file()
    };
    if !exists {
        return Err(BackendError::MissingResource);
    }
    Ok(path)
}

/// Returns the validated bundled recommended model configuration.
pub(crate) fn recommended_model_config_path(app: &AppHandle) -> Result<PathBuf, BackendError> {
    let path = resolve_required_resource(app, RECOMMENDED_MODEL_CONFIG_RESOURCE, false)?;
    validate_model_config_path(&path)
}

fn sidecar_working_directory() -> Result<PathBuf, BackendError> {
    std::env::current_exe()?
        .parent()
        .map(PathBuf::from)
        .ok_or(BackendError::InvalidResourceLayout)
}

fn validate_sidecar_runtime(sidecar_directory: &Path) -> Result<(), BackendError> {
    let sibling_runtime = sidecar_directory.join(SIDECAR_RUNTIME_RESOURCE);
    if is_complete_python_runtime(&sibling_runtime) {
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    {
        let framework_runtime = sidecar_directory
            .parent()
            .ok_or(BackendError::InvalidResourceLayout)?
            .join("Frameworks");
        if is_complete_python_runtime(&framework_runtime) {
            return Ok(());
        }
    }

    Err(BackendError::MissingResource)
}

fn is_complete_python_runtime(path: &Path) -> bool {
    if !path.join("base_library.zip").is_file() {
        return false;
    }

    let Ok(entries) = std::fs::read_dir(path) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|entry| {
        entry
            .file_name()
            .to_str()
            .is_some_and(is_python_runtime_library)
    })
}

fn is_python_runtime_library(name: &str) -> bool {
    #[cfg(target_os = "windows")]
    {
        name.starts_with("python") && name.ends_with(".dll")
    }

    #[cfg(not(target_os = "windows"))]
    {
        name.starts_with("libpython")
    }
}

async fn wait_for_health(port: u16, token: &str) -> Result<(), BackendError> {
    let deadline = Instant::now() + HEALTH_TIMEOUT;

    loop {
        let token = token.to_owned();
        let result = tokio::task::spawn_blocking(move || send_health_probe(port, &token)).await;
        if matches!(result, Ok(Ok(()))) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(BackendError::HealthCheckFailed);
        }
        sleep(HEALTH_RETRY_INTERVAL).await;
    }
}

fn send_health_probe(port: u16, token: &str) -> Result<(), BackendError> {
    let response = send_loopback_request(port, token, "GET", "/health", HEALTH_REQUEST_TIMEOUT)?;
    validate_health_response(response.status, &response.body)
}

fn validate_health_response(status: u16, body: &[u8]) -> Result<(), BackendError> {
    if !(200..300).contains(&status) {
        return Err(BackendError::InvalidHealth);
    }

    let health: HealthMessage = serde_json::from_slice(body)?;
    if health.status != "ok" || health.protocol_version != PROTOCOL_VERSION {
        return Err(BackendError::InvalidHealth);
    }
    Ok(())
}

fn send_controlled_shutdown(endpoint: &str, token: &str) -> Result<(), BackendError> {
    let port = endpoint
        .strip_prefix("http://127.0.0.1:")
        .and_then(|value| value.parse::<u16>().ok())
        .filter(|port| *port != 0)
        .ok_or(BackendError::InvalidShutdownEndpoint)?;
    let response = send_loopback_request(
        port,
        token,
        "POST",
        "/app/api/shutdown",
        CONTROL_REQUEST_TIMEOUT,
    )?;
    if !(200..300).contains(&response.status) {
        return Err(BackendError::ShutdownRejected);
    }
    Ok(())
}

fn send_loopback_request(
    port: u16,
    token: &str,
    method: &str,
    path: &str,
    request_timeout: Duration,
) -> Result<HttpResponse, BackendError> {
    if port == 0 {
        return Err(BackendError::InvalidShutdownEndpoint);
    }

    let address = SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&address, request_timeout)?;
    stream.set_read_timeout(Some(request_timeout))?;
    stream.set_write_timeout(Some(request_timeout))?;

    let request = format!(
        "{method} {path} HTTP/1.1\r\n\
         Host: 127.0.0.1:{port}\r\n\
         X-XTalk-App-Token: {token}\r\n\
         Content-Length: 0\r\n\
         Connection: close\r\n\
         \r\n"
    );
    stream.write_all(request.as_bytes())?;
    stream.flush()?;

    let mut response = Vec::new();
    stream
        .take((MAX_HTTP_RESPONSE_BYTES + 1) as u64)
        .read_to_end(&mut response)?;
    if response.len() > MAX_HTTP_RESPONSE_BYTES {
        return Err(BackendError::InvalidHttpResponse);
    }

    parse_http_response(&response)
}

fn parse_http_response(response: &[u8]) -> Result<HttpResponse, BackendError> {
    let header_end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or(BackendError::InvalidHttpResponse)?;
    let header = std::str::from_utf8(&response[..header_end])
        .map_err(|_| BackendError::InvalidHttpResponse)?;
    let status_line = header
        .lines()
        .next()
        .ok_or(BackendError::InvalidHttpResponse)?;
    let mut status_parts = status_line.split_whitespace();
    let version = status_parts
        .next()
        .ok_or(BackendError::InvalidHttpResponse)?;
    if version != "HTTP/1.1" && version != "HTTP/1.0" {
        return Err(BackendError::InvalidHttpResponse);
    }
    let status = status_parts
        .next()
        .and_then(|value| value.parse::<u16>().ok())
        .filter(|value| (100..=599).contains(value))
        .ok_or(BackendError::InvalidHttpResponse)?;

    Ok(HttpResponse {
        status,
        body: response[header_end + 4..].to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf, process};

    use serde_json::json;

    use super::{
        build_config_fallbacks, parse_http_response, validate_health_response,
        validate_model_config_path, validate_ready_line, BackendError, StartupMessage,
        DESKTOP_ANONYMOUS_USER_ID, DESKTOP_VAD_THRESHOLD, PROTOCOL_VERSION,
    };

    fn temporary_model_config(name: &str, contents: &[u8]) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("xtalk-desktop-{}-{name}.json", process::id()));
        fs::write(&path, contents).expect("test model config must be writable");
        path
    }

    #[test]
    fn accepts_the_exact_ready_protocol() {
        let ready = validate_ready_line(br#"{"type":"ready","protocol_version":1,"port":43127}"#)
            .expect("valid readiness must be accepted");

        assert_eq!(ready.port, 43127);
    }

    #[test]
    fn rejects_zero_port() {
        let error = validate_ready_line(br#"{"type":"ready","protocol_version":1,"port":0}"#)
            .expect_err("zero is not a TCP port");

        assert!(matches!(error, BackendError::InvalidReady(_)));
    }

    #[test]
    fn rejects_an_unknown_protocol_version() {
        let error = validate_ready_line(br#"{"type":"ready","protocol_version":2,"port":43127}"#)
            .expect_err("an unknown protocol must be rejected");

        assert!(matches!(error, BackendError::InvalidReady(_)));
    }

    #[test]
    fn rejects_additional_readiness_fields() {
        let error = validate_ready_line(
            br#"{"type":"ready","protocol_version":1,"port":43127,"host":"0.0.0.0"}"#,
        )
        .expect_err("the sidecar cannot select a non-loopback host");

        assert!(matches!(error, BackendError::Json(_)));
    }

    #[test]
    fn accepts_the_exact_health_protocol() {
        validate_health_response(200, br#"{"status":"ok","protocol_version":1}"#)
            .expect("valid health response must be accepted");
    }

    #[test]
    fn rejects_a_health_protocol_mismatch() {
        let error = validate_health_response(200, br#"{"status":"ok","protocol_version":2}"#)
            .expect_err("an unknown health protocol must be rejected");

        assert!(matches!(error, BackendError::InvalidHealth));
    }

    #[test]
    fn parses_a_bounded_http_response() {
        let response = parse_http_response(
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{\"status\":\"ok\"}",
        )
        .expect("well-formed HTTP response must be accepted");

        assert_eq!(response.status, 200);
        assert_eq!(response.body, br#"{"status":"ok"}"#);
    }

    #[test]
    fn serializes_vad_model_as_a_config_fallback() {
        let vad_model_path = PathBuf::from("packaged-models").join("silero_vad.onnx");
        let expected_model_path =
            serde_json::to_value(&vad_model_path).expect("test path must serialize");
        let startup = StartupMessage {
            protocol_version: PROTOCOL_VERSION,
            token: "test-token",
            config_path: PathBuf::from("sample.json"),
            data_dir: PathBuf::from("app-data"),
            builtin_tools_root: PathBuf::from("resources/tools"),
            origins: vec!["tauri://localhost".to_owned()],
            config_fallbacks: build_config_fallbacks(&vad_model_path),
            config_overlay: json!({}),
            anonymous_user_id: DESKTOP_ANONYMOUS_USER_ID,
        };

        let payload = serde_json::to_value(startup).expect("startup message must serialize");

        assert_eq!(
            payload["config_fallbacks"]["vad"],
            json!({
                "type": "SileroVAD",
                "params": {
                    "model_path": expected_model_path,
                    "threshold": DESKTOP_VAD_THRESHOLD,
                },
            })
        );
        assert_eq!(payload["config_overlay"], json!({}));
        assert_eq!(
            payload["anonymous_user_id"],
            json!(DESKTOP_ANONYMOUS_USER_ID)
        );
    }

    #[test]
    fn accepts_an_external_object_model_config() {
        let path = temporary_model_config(
            "valid-model-config",
            br#"{"service_config":{"enable_persistence":true}}"#,
        );

        let validated = validate_model_config_path(&path).expect("object config must be accepted");

        assert_eq!(
            validated,
            path.canonicalize().expect("test path must canonicalize")
        );
        fs::remove_file(path).expect("test config must be removable");
    }

    #[test]
    fn rejects_a_non_object_model_config() {
        let path = temporary_model_config("array-model-config", br#"[]"#);

        let error = validate_model_config_path(&path).expect_err("array config must be rejected");

        assert!(matches!(error, BackendError::InvalidModelConfigRoot));
        fs::remove_file(path).expect("test config must be removable");
    }
}
