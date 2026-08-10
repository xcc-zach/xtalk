//! Optional local-model installation and process supervision.

use std::{
    fs::{self, File},
    io::{Read, Write},
    net::{Ipv4Addr, SocketAddrV4, TcpListener},
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use bzip2::read::BzDecoder;
use reqwest::{redirect::Policy, Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tauri::{async_runtime::Receiver, path::BaseDirectory, AppHandle, Emitter, Manager};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};
use thiserror::Error;
use tokio::{
    sync::{watch, Mutex},
    time::{sleep, timeout, Instant},
};

const MANIFEST_RESOURCE: &str = "manifests/managed-models.lock.json";
const ONNX_RUNTIME_RESOURCE: &str = "managed-runtime/ort";
const TTS_SIDECAR_NAME: &str = "local-model-runtime";
const MATCHA_TTS_SIDECAR_NAME: &str = "matcha-model-runtime";
const MLX_SIDECAR_NAME: &str = "mlx-model-runtime";
const SHERPA_SIDECAR_NAME: &str = "sherpa-onnx-offline-websocket-server";
const SENSEVOICE_ID: &str = "sensevoice-small";
const SENSEVOICE_MLX_ID: &str = "sensevoice-small-mlx";
const REFINER_ID: &str = "agentic-asr-refiner";
const REFINER_MLX_ID: &str = "agentic-asr-refiner-mlx";
const MOSS_TTS_ID: &str = "moss-tts-nano";
const MOSS_TTS_MLX_ID: &str = "moss-tts-nano-mlx";
const MATCHA_TTS_ID: &str = "matcha-icefall-zh-en";
const MANAGED_ROOT: &str = "managed://";
const SENSEVOICE_URL: &str = "managed://sensevoice-small";
const REFINER_URL: &str = "managed://agentic-asr-refiner";
const MOSS_TTS_URL: &str = "managed://moss-tts-nano";
const MATCHA_TTS_URL: &str = "managed://matcha-icefall-zh-en";
const DEFAULT_MOSS_VOICE_URL: &str = "managed://moss-tts-nano/voices/zh_1.wav";
const INSTALL_MARKER: &str = ".complete.json";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(45);
const STARTUP_POLL_INTERVAL: Duration = Duration::from_millis(100);
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const MAX_READY_LINE_BYTES: usize = 4 * 1024;
const MANAGED_PROGRESS_EVENT: &str = "managed-model-progress";
const HUGGING_FACE_ORIGIN: &str = "https://huggingface.co/";
const HUGGING_FACE_MIRROR_ORIGIN: &str = "https://hf-mirror.com/";

/// Managed model services requested by one external model configuration.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ManagedModelPlan {
    /// Stable managed service identifiers in startup order.
    pub(crate) services: Vec<String>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ManagedModelProgress {
    phase: &'static str,
    service_id: Option<String>,
    service_index: usize,
    service_count: usize,
    completed_bytes: u64,
    total_bytes: u64,
    file_path: Option<String>,
}

/// A set of native model services owned by one Python backend instance.
pub(crate) struct ManagedServices {
    inner: Arc<ManagedServicesInner>,
    active: bool,
}

struct ManagedServicesInner {
    children: Mutex<Vec<CommandChild>>,
    healthy: AtomicBool,
    shutting_down: AtomicBool,
    failure_sender: watch::Sender<bool>,
}

#[derive(Default)]
struct ManagedRequest {
    sensevoice: Option<ManagedBackend>,
    refiner: Option<ManagedBackend>,
    agentic_asr: bool,
    moss_tts: Option<ManagedBackend>,
    matcha_tts: Option<ManagedBackend>,
    moss_voices: Vec<ManagedVoice>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ManagedBackend {
    Auto,
    Cpu,
    Cuda,
    Mlx,
}

#[derive(Clone, Copy)]
enum ManagedServiceKind {
    SenseVoice,
    Refiner,
    MossTts,
    MatchaTts,
}

struct ManagedVoice {
    name: String,
    path: String,
}

struct StartedService {
    port: u16,
    child: CommandChild,
    events: Receiver<CommandEvent>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManagedManifest {
    schema_version: u16,
    services: Vec<ManagedServiceManifest>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManagedServiceManifest {
    id: String,
    version: String,
    files: Vec<ManagedFileManifest>,
    #[serde(default)]
    archives: Vec<ManagedArchiveManifest>,
    #[serde(default)]
    required_paths: Vec<String>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManagedArchiveManifest {
    path: String,
    format: ManagedArchiveFormat,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum ManagedArchiveFormat {
    TarBz2,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManagedFileManifest {
    path: String,
    size: u64,
    sha256: String,
    url: String,
}

#[derive(Deserialize)]
struct ModelReadyMessage {
    status: String,
    protocol_version: u16,
    port: u16,
}

/// Errors raised while installing or supervising optional local models.
#[derive(Debug, Error)]
pub(crate) enum ManagedError {
    #[error("failed to access an application path: {0}")]
    Tauri(#[from] tauri::Error),
    #[error("managed model I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("managed model metadata is invalid: {0}")]
    Json(#[from] serde_json::Error),
    #[error("failed to start a managed model service: {0}")]
    Shell(#[from] tauri_plugin_shell::Error),
    #[error("managed model download failed: {0}")]
    Download(#[from] reqwest::Error),
    #[error(
        "managed model download failed from both Hugging Face and its mirror: primary: {primary}; mirror: {mirror}"
    )]
    DownloadFallback {
        primary: reqwest::Error,
        mirror: reqwest::Error,
    },
    #[error("managed model manifest uses an unsupported schema")]
    UnsupportedManifest,
    #[error("managed model configuration is invalid: {0}")]
    InvalidConfiguration(String),
    #[error("managed model `{0}` is absent from the packaged manifest")]
    MissingManifestService(String),
    #[error("managed model manifest contains an unsafe path")]
    UnsafeManifestPath,
    #[error("managed model file failed size or SHA-256 verification: {0}")]
    VerificationFailed(String),
    #[error("a required managed runtime library is missing")]
    MissingRuntime,
    #[error("managed model service did not become ready")]
    StartupTimedOut,
    #[error("managed model service terminated before readiness: {0}")]
    TerminatedBeforeReady(String),
    #[error("managed model service emitted an invalid readiness message")]
    InvalidReady,
    #[error("the MLX managed runtime is supported only on Apple Silicon macOS")]
    UnsupportedMlxPlatform,
    #[error("the CUDA managed runtime is not available on this device")]
    UnsupportedCudaPlatform,
}

impl ManagedServices {
    /// Installs and starts the local services referenced by one model config.
    pub(crate) async fn start(
        app: &AppHandle,
        config_path: &Path,
        data_dir: &Path,
    ) -> Result<(Self, Value), ManagedError> {
        let request = parse_managed_request(config_path)?;
        let active = request.sensevoice.is_some()
            || request.refiner.is_some()
            || request.moss_tts.is_some()
            || request.matcha_tts.is_some();
        let (failure_sender, _) = watch::channel(false);
        let services = Self {
            inner: Arc::new(ManagedServicesInner {
                children: Mutex::new(Vec::new()),
                healthy: AtomicBool::new(true),
                shutting_down: AtomicBool::new(false),
                failure_sender,
            }),
            active,
        };
        if !active {
            return Ok((services, json!({})));
        }

        let manifest_path = app
            .path()
            .resolve(MANIFEST_RESOURCE, BaseDirectory::Resource)?;
        let manifest: ManagedManifest = serde_json::from_slice(&fs::read(manifest_path)?)?;
        if manifest.schema_version != 1 {
            return Err(ManagedError::UnsupportedManifest);
        }

        let install_root = data_dir.join("models").join("managed");
        fs::create_dir_all(&install_root)?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .timeout(DOWNLOAD_TIMEOUT)
            .user_agent("XTalk-Desktop/0.1")
            .redirect(Policy::custom(|attempt| {
                if attempt.url().scheme() != "https" {
                    attempt.stop()
                } else if attempt.previous().len() >= 10 {
                    attempt.error("too many managed model download redirects")
                } else {
                    attempt.follow()
                }
            }))
            .build()?;

        let result = services
            .start_requested(app, &request, &manifest, &install_root, &client)
            .await;
        match result {
            Ok(overlay) => Ok((services, overlay)),
            Err(error) => {
                services.shutdown().await;
                Err(error)
            }
        }
    }

    async fn start_requested(
        &self,
        app: &AppHandle,
        request: &ManagedRequest,
        manifest: &ManagedManifest,
        install_root: &Path,
        client: &Client,
    ) -> Result<Value, ManagedError> {
        let mut overlay = json!({});
        let service_count = usize::from(request.sensevoice.is_some())
            + usize::from(request.refiner.is_some())
            + usize::from(request.moss_tts.is_some())
            + usize::from(request.matcha_tts.is_some());
        let mut service_index = 0;

        if let Some(requested_backend) = request.sensevoice {
            service_index += 1;
            let backend = resolve_backend(app, requested_backend, ManagedServiceKind::SenseVoice)?;
            let service_manifest = find_service(
                manifest,
                match backend {
                    ManagedBackend::Mlx => SENSEVOICE_MLX_ID,
                    _ => SENSEVOICE_ID,
                },
            )?;
            emit_managed_progress(
                app,
                "checking",
                Some(service_manifest),
                service_index,
                service_count,
                0,
                service_manifest.files.iter().map(|file| file.size).sum(),
                None,
            );
            let model_root = ensure_service_installed(
                client,
                install_root,
                service_manifest,
                app,
                service_index,
                service_count,
            )
            .await?;
            emit_managed_progress(
                app,
                "starting",
                Some(service_manifest),
                service_index,
                service_count,
                0,
                0,
                None,
            );
            let started = match backend {
                ManagedBackend::Mlx => start_mlx(app, SENSEVOICE_ID, &model_root).await?,
                ManagedBackend::Cpu | ManagedBackend::Cuda => {
                    start_sensevoice(app, &model_root, backend).await?
                }
                ManagedBackend::Auto => unreachable!("auto backend must be resolved"),
            };
            let port = started.port;
            self.accept(started).await;
            emit_managed_progress(
                app,
                "ready",
                Some(service_manifest),
                service_index,
                service_count,
                0,
                0,
                None,
            );
            if request.agentic_asr {
                overlay["asr"] = json!({
                    "params": {
                        "asr_base_url": format!("ws://127.0.0.1:{port}"),
                        "asr_mode": "offline"
                    }
                });
            } else {
                overlay["asr"] = json!({
                    "params": {
                        "base_url": format!("ws://127.0.0.1:{port}"),
                        "mode": "offline"
                    }
                });
            }
        }

        if let Some(requested_backend) = request.refiner {
            service_index += 1;
            let backend = resolve_backend(app, requested_backend, ManagedServiceKind::Refiner)?;
            let service_manifest = find_service(
                manifest,
                match backend {
                    ManagedBackend::Mlx => REFINER_MLX_ID,
                    _ => REFINER_ID,
                },
            )?;
            emit_managed_progress(
                app,
                "checking",
                Some(service_manifest),
                service_index,
                service_count,
                0,
                service_manifest.files.iter().map(|file| file.size).sum(),
                None,
            );
            let model_root = ensure_service_installed(
                client,
                install_root,
                service_manifest,
                app,
                service_index,
                service_count,
            )
            .await?;
            emit_managed_progress(
                app,
                "starting",
                Some(service_manifest),
                service_index,
                service_count,
                0,
                0,
                None,
            );
            let started = match backend {
                ManagedBackend::Mlx => start_mlx(app, REFINER_ID, &model_root).await?,
                ManagedBackend::Cpu | ManagedBackend::Cuda => {
                    start_refiner(app, &model_root, backend).await?
                }
                ManagedBackend::Auto => unreachable!("auto backend must be resolved"),
            };
            let port = started.port;
            self.accept(started).await;
            emit_managed_progress(
                app,
                "ready",
                Some(service_manifest),
                service_index,
                service_count,
                0,
                0,
                None,
            );
            overlay["asr"]["params"]["refiner_base_url"] =
                json!(format!("http://127.0.0.1:{port}/v1"));
        }

        if let Some(requested_backend) = request.moss_tts {
            service_index += 1;
            let backend = resolve_backend(app, requested_backend, ManagedServiceKind::MossTts)?;
            let service_manifest = find_service(
                manifest,
                match backend {
                    ManagedBackend::Mlx => MOSS_TTS_MLX_ID,
                    _ => MOSS_TTS_ID,
                },
            )?;
            emit_managed_progress(
                app,
                "checking",
                Some(service_manifest),
                service_index,
                service_count,
                0,
                service_manifest.files.iter().map(|file| file.size).sum(),
                None,
            );
            let model_root = ensure_service_installed(
                client,
                install_root,
                service_manifest,
                app,
                service_index,
                service_count,
            )
            .await?;
            emit_managed_progress(
                app,
                "starting",
                Some(service_manifest),
                service_index,
                service_count,
                0,
                0,
                None,
            );
            let started = match backend {
                ManagedBackend::Mlx => start_mlx(app, MOSS_TTS_ID, &model_root).await?,
                ManagedBackend::Cpu | ManagedBackend::Cuda => {
                    start_moss_tts(app, &model_root, backend).await?
                }
                ManagedBackend::Auto => unreachable!("auto backend must be resolved"),
            };
            let port = started.port;
            self.accept(started).await;
            emit_managed_progress(
                app,
                "ready",
                Some(service_manifest),
                service_index,
                service_count,
                0,
                0,
                None,
            );
            let voices = resolve_moss_voices(&request.moss_voices, &model_root)?;
            overlay["tts"] = json!({
                "params": {
                    "base_url": format!("http://127.0.0.1:{port}"),
                    "voices": voices
                }
            });
        }

        if let Some(requested_backend) = request.matcha_tts {
            service_index += 1;
            let backend = resolve_backend(app, requested_backend, ManagedServiceKind::MatchaTts)?;
            let service_manifest = find_service(manifest, MATCHA_TTS_ID)?;
            emit_managed_progress(
                app,
                "checking",
                Some(service_manifest),
                service_index,
                service_count,
                0,
                service_manifest.files.iter().map(|file| file.size).sum(),
                None,
            );
            let model_root = ensure_service_installed(
                client,
                install_root,
                service_manifest,
                app,
                service_index,
                service_count,
            )
            .await?;
            emit_managed_progress(
                app,
                "starting",
                Some(service_manifest),
                service_index,
                service_count,
                0,
                0,
                None,
            );
            let started = start_matcha_tts(app, &model_root, backend).await?;
            let port = started.port;
            self.accept(started).await;
            emit_managed_progress(
                app,
                "ready",
                Some(service_manifest),
                service_index,
                service_count,
                0,
                0,
                None,
            );
            overlay["tts"] = json!({
                "params": {
                    "base_url": format!("http://127.0.0.1:{port}")
                }
            });
        }

        emit_managed_progress(
            app,
            "complete",
            None,
            service_count,
            service_count,
            0,
            0,
            None,
        );
        Ok(overlay)
    }

    async fn accept(&self, started: StartedService) {
        self.inner.children.lock().await.push(started.child);
        let inner = Arc::downgrade(&self.inner);
        tauri::async_runtime::spawn(monitor_service(started.events, inner));
    }

    /// Returns whether every requested managed process is still available.
    pub(crate) fn is_healthy(&self) -> bool {
        self.inner.healthy.load(Ordering::Acquire)
            && !self.inner.shutting_down.load(Ordering::Acquire)
    }

    /// Subscribes to unexpected managed-process termination when services exist.
    pub(crate) fn failure_receiver(&self) -> Option<watch::Receiver<bool>> {
        self.active.then(|| self.inner.failure_sender.subscribe())
    }

    /// Stops all managed model processes in reverse startup order.
    pub(crate) async fn shutdown(&self) {
        self.inner.shutting_down.store(true, Ordering::Release);
        self.inner.healthy.store(false, Ordering::Release);
        let mut children = self.inner.children.lock().await;
        while let Some(child) = children.pop() {
            if let Err(error) = child.kill() {
                eprintln!("failed to stop a managed model service: {error}");
            }
        }
    }
}

/// Inspects an external configuration without installing or starting models.
pub(crate) fn inspect_model_config(config_path: &Path) -> Result<ManagedModelPlan, ManagedError> {
    let request = parse_managed_request(config_path)?;
    let mut services = Vec::new();
    if request.sensevoice.is_some() {
        services.push(SENSEVOICE_ID.to_owned());
    }
    if request.refiner.is_some() {
        services.push(REFINER_ID.to_owned());
    }
    if request.moss_tts.is_some() {
        services.push(MOSS_TTS_ID.to_owned());
    }
    if request.matcha_tts.is_some() {
        services.push(MATCHA_TTS_ID.to_owned());
    }
    Ok(ManagedModelPlan { services })
}

fn parse_managed_request(config_path: &Path) -> Result<ManagedRequest, ManagedError> {
    let config: Value = serde_json::from_slice(&fs::read(config_path)?)?;
    let mut request = ManagedRequest::default();

    let asr_type = model_type(&config, "asr")?;
    if asr_type == Some("AgenticASR") {
        request.agentic_asr = true;
        if let Some(base_url) = model_param(&config, "asr", "asr_base_url")? {
            request.sensevoice = parse_managed_backend(base_url, SENSEVOICE_URL, "ASR")?;
        }
        if let Some(base_url) = model_param(&config, "asr", "refiner_base_url")? {
            request.refiner = parse_managed_backend(base_url, REFINER_URL, "Refiner")?;
        }
        if request.sensevoice.is_some() {
            match model_param(&config, "asr", "asr_mode")? {
                None | Some("offline") => {}
                Some(mode) => {
                    return Err(ManagedError::InvalidConfiguration(format!(
                        "managed AgenticASR requires `asr.params.asr_mode` to be `offline`, got `{mode}`"
                    )));
                }
            }
        }
    } else if let Some(base_url) = model_param(&config, "asr", "base_url")? {
        request.sensevoice = parse_managed_backend(base_url, SENSEVOICE_URL, "ASR")?;
        if request.sensevoice.is_some() {
            require_model_type(&config, "asr", "SherpaOnnxASR")?;
        }
    }

    if let Some(base_url) = model_param(&config, "tts", "base_url")? {
        if is_service_url(base_url, MOSS_TTS_URL) {
            request.moss_tts = parse_managed_backend(base_url, MOSS_TTS_URL, "TTS")?;
            require_model_type(&config, "tts", "MossTTSNano")?;
            request.moss_voices = parse_moss_voices(&config)?;
        } else if is_service_url(base_url, MATCHA_TTS_URL) {
            request.matcha_tts = parse_managed_backend(base_url, MATCHA_TTS_URL, "TTS")?;
            require_model_type(&config, "tts", "SherpaOnnxTTS")?;
        } else if base_url.starts_with(MANAGED_ROOT) {
            return Err(ManagedError::InvalidConfiguration(format!(
                "unsupported managed TTS URL `{base_url}`"
            )));
        }
    }

    Ok(request)
}

fn is_service_url(base_url: &str, service_url: &str) -> bool {
    base_url == service_url || base_url.starts_with(&format!("{service_url}?"))
}

fn parse_managed_backend(
    base_url: &str,
    service_url: &str,
    service_kind: &str,
) -> Result<Option<ManagedBackend>, ManagedError> {
    match base_url {
        value if value == service_url => Ok(Some(ManagedBackend::Auto)),
        value if value == format!("{service_url}?backend=cpu") => Ok(Some(ManagedBackend::Cpu)),
        value if value == format!("{service_url}?backend=cuda") => Ok(Some(ManagedBackend::Cuda)),
        value if value == format!("{service_url}?backend=mlx") => Ok(Some(ManagedBackend::Mlx)),
        value if value.starts_with(MANAGED_ROOT) => Err(ManagedError::InvalidConfiguration(
            format!("unsupported managed {service_kind} URL `{base_url}`"),
        )),
        _ => Ok(None),
    }
}

fn resolve_backend(
    app: &AppHandle,
    requested: ManagedBackend,
    service: ManagedServiceKind,
) -> Result<ManagedBackend, ManagedError> {
    if matches!(service, ManagedServiceKind::MatchaTts) && matches!(requested, ManagedBackend::Mlx)
    {
        return Err(ManagedError::InvalidConfiguration(
            "Matcha TTS supports only CPU and CUDA backends".to_owned(),
        ));
    }
    select_backend(
        requested,
        cuda_is_available(app, service)?,
        !matches!(service, ManagedServiceKind::MatchaTts) && mlx_is_available(),
    )
}

fn select_backend(
    requested: ManagedBackend,
    cuda_available: bool,
    mlx_available: bool,
) -> Result<ManagedBackend, ManagedError> {
    match requested {
        ManagedBackend::Cpu => Ok(ManagedBackend::Cpu),
        ManagedBackend::Cuda if cuda_available => Ok(ManagedBackend::Cuda),
        ManagedBackend::Cuda => Err(ManagedError::UnsupportedCudaPlatform),
        ManagedBackend::Mlx if mlx_available => Ok(ManagedBackend::Mlx),
        ManagedBackend::Mlx => Err(ManagedError::UnsupportedMlxPlatform),
        ManagedBackend::Auto if cuda_available => Ok(ManagedBackend::Cuda),
        ManagedBackend::Auto if mlx_available => Ok(ManagedBackend::Mlx),
        ManagedBackend::Auto => Ok(ManagedBackend::Cpu),
    }
}

fn mlx_is_available() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

fn cuda_is_available(app: &AppHandle, service: ManagedServiceKind) -> Result<bool, ManagedError> {
    if cfg!(target_os = "macos") {
        return Ok(false);
    }
    let _ = service;
    let runtime_dir = app
        .path()
        .resolve(ONNX_RUNTIME_RESOURCE, BaseDirectory::Resource)?;
    let has_provider = fs::read_dir(runtime_dir)?
        .filter_map(Result::ok)
        .any(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .to_ascii_lowercase()
                .contains("onnxruntime_providers_cuda")
        });
    if !has_provider {
        return Ok(false);
    }
    Ok(std::process::Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .is_ok_and(|output| output.status.success()))
}

fn model_param<'a>(
    config: &'a Value,
    section: &str,
    parameter: &str,
) -> Result<Option<&'a str>, ManagedError> {
    let Some(section_value) = config.get(section) else {
        return Ok(None);
    };
    let section_object = section_value.as_object().ok_or_else(|| {
        ManagedError::InvalidConfiguration(format!("`{section}` must be an object"))
    })?;
    let Some(params) = section_object.get("params") else {
        return Ok(None);
    };
    let params_object = params.as_object().ok_or_else(|| {
        ManagedError::InvalidConfiguration(format!("`{section}.params` must be an object"))
    })?;
    match params_object.get(parameter) {
        None => Ok(None),
        Some(value) => value.as_str().map(Some).ok_or_else(|| {
            ManagedError::InvalidConfiguration(format!(
                "`{section}.params.{parameter}` must be a string"
            ))
        }),
    }
}

fn model_type<'a>(config: &'a Value, section: &str) -> Result<Option<&'a str>, ManagedError> {
    let Some(section_value) = config.get(section) else {
        return Ok(None);
    };
    let section_object = section_value.as_object().ok_or_else(|| {
        ManagedError::InvalidConfiguration(format!("`{section}` must be an object"))
    })?;
    match section_object.get("type") {
        None => Ok(None),
        Some(value) => value.as_str().map(Some).ok_or_else(|| {
            ManagedError::InvalidConfiguration(format!("`{section}.type` must be a string"))
        }),
    }
}

fn require_model_type(config: &Value, section: &str, expected: &str) -> Result<(), ManagedError> {
    let actual = config
        .get(section)
        .and_then(Value::as_object)
        .and_then(|value| value.get("type"))
        .and_then(Value::as_str);
    if actual != Some(expected) {
        return Err(ManagedError::InvalidConfiguration(format!(
            "`{section}.type` must be `{expected}` for its managed URL"
        )));
    }
    Ok(())
}

fn parse_moss_voices(config: &Value) -> Result<Vec<ManagedVoice>, ManagedError> {
    let voices = config
        .pointer("/tts/params/voices")
        .and_then(Value::as_array);
    let Some(voices) = voices else {
        return Ok(vec![ManagedVoice {
            name: "zh".to_owned(),
            path: DEFAULT_MOSS_VOICE_URL.to_owned(),
        }]);
    };
    if voices.is_empty() {
        return Err(ManagedError::InvalidConfiguration(
            "`tts.params.voices` must not be empty".to_owned(),
        ));
    }

    voices
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let voice = value.as_object().ok_or_else(|| {
                ManagedError::InvalidConfiguration(format!(
                    "`tts.params.voices[{index}]` must be an object"
                ))
            })?;
            let name = voice.get("name").and_then(Value::as_str).ok_or_else(|| {
                ManagedError::InvalidConfiguration(format!(
                    "`tts.params.voices[{index}].name` must be a string"
                ))
            })?;
            let path = voice.get("path").and_then(Value::as_str).ok_or_else(|| {
                ManagedError::InvalidConfiguration(format!(
                    "`tts.params.voices[{index}].path` must be a string"
                ))
            })?;
            if name.trim().is_empty() || path.trim().is_empty() {
                return Err(ManagedError::InvalidConfiguration(format!(
                    "`tts.params.voices[{index}]` contains an empty name or path"
                )));
            }
            if path.starts_with(MANAGED_ROOT) && path != DEFAULT_MOSS_VOICE_URL {
                return Err(ManagedError::InvalidConfiguration(format!(
                    "unsupported managed voice URL `{path}`"
                )));
            }
            Ok(ManagedVoice {
                name: name.trim().to_owned(),
                path: path.trim().to_owned(),
            })
        })
        .collect()
}

fn resolve_moss_voices(
    voices: &[ManagedVoice],
    model_root: &Path,
) -> Result<Vec<Value>, ManagedError> {
    voices
        .iter()
        .map(|voice| {
            let path = if voice.path == DEFAULT_MOSS_VOICE_URL {
                model_root.join("voices").join("zh_1.wav")
            } else {
                PathBuf::from(&voice.path)
            };
            if !path.is_file() {
                return Err(ManagedError::InvalidConfiguration(format!(
                    "MOSS reference voice is missing: {}",
                    path.display()
                )));
            }
            Ok(json!({"name": voice.name, "path": path}))
        })
        .collect()
}

fn find_service<'a>(
    manifest: &'a ManagedManifest,
    id: &str,
) -> Result<&'a ManagedServiceManifest, ManagedError> {
    manifest
        .services
        .iter()
        .find(|service| service.id == id)
        .ok_or_else(|| ManagedError::MissingManifestService(id.to_owned()))
}

async fn ensure_service_installed(
    client: &Client,
    install_root: &Path,
    manifest: &ManagedServiceManifest,
    app: &AppHandle,
    service_index: usize,
    service_count: usize,
) -> Result<PathBuf, ManagedError> {
    let service_root = install_root.join(&manifest.id).join(&manifest.version);
    let manifest_clone = manifest.clone();
    let verification_root = service_root.clone();
    let verified = tokio::task::spawn_blocking(move || {
        verify_installed_service(&verification_root, &manifest_clone)
    })
    .await
    .map_err(|error| ManagedError::VerificationFailed(error.to_string()))??;
    if verified {
        return Ok(service_root);
    }

    fs::create_dir_all(&service_root)?;
    let _ = fs::remove_file(service_root.join(INSTALL_MARKER));
    let total_bytes = manifest.files.iter().map(|file| file.size).sum();
    let mut completed_bytes = 0;
    for file in &manifest.files {
        download_file(
            client,
            &service_root,
            file,
            app,
            manifest,
            service_index,
            service_count,
            completed_bytes,
            total_bytes,
        )
        .await?;
        completed_bytes = completed_bytes.saturating_add(file.size);
    }
    if !manifest.archives.is_empty() {
        emit_managed_progress(
            app,
            "checking",
            Some(manifest),
            service_index,
            service_count,
            total_bytes,
            total_bytes,
            None,
        );
        let extraction_root = service_root.clone();
        let extraction_manifest = manifest.clone();
        tokio::task::spawn_blocking(move || {
            extract_service_archives(&extraction_root, &extraction_manifest)
        })
        .await
        .map_err(|error| ManagedError::VerificationFailed(error.to_string()))??;
    }
    write_install_marker(&service_root, manifest)?;
    if !verify_installed_service(&service_root, manifest)? {
        return Err(ManagedError::VerificationFailed(manifest.id.clone()));
    }
    Ok(service_root)
}

fn verify_installed_service(
    root: &Path,
    manifest: &ManagedServiceManifest,
) -> Result<bool, ManagedError> {
    let marker = root.join(INSTALL_MARKER);
    if !marker.is_file() {
        return Ok(false);
    }
    let marker_value: Value = serde_json::from_slice(&fs::read(marker)?)?;
    if marker_value.get("id").and_then(Value::as_str) != Some(&manifest.id)
        || marker_value.get("version").and_then(Value::as_str) != Some(&manifest.version)
    {
        return Ok(false);
    }
    for file in &manifest.files {
        let path = safe_join(root, &file.path)?;
        if !path.is_file() || fs::metadata(&path)?.len() != file.size {
            return Ok(false);
        }
        if sha256_file(&path)? != file.sha256 {
            return Ok(false);
        }
    }
    for relative_path in &manifest.required_paths {
        let path = safe_join(root, relative_path)?;
        let Ok(metadata) = fs::symlink_metadata(&path) else {
            return Ok(false);
        };
        if metadata.file_type().is_symlink() {
            return Ok(false);
        }
    }
    verify_extracted_archives(root, manifest)
}

fn verify_extracted_archives(
    root: &Path,
    manifest: &ManagedServiceManifest,
) -> Result<bool, ManagedError> {
    for archive_manifest in &manifest.archives {
        let archive_path = safe_join(root, &archive_manifest.path)?;
        let input = File::open(archive_path)?;
        match archive_manifest.format {
            ManagedArchiveFormat::TarBz2 => {
                let decoder = BzDecoder::new(input);
                let mut archive = tar::Archive::new(decoder);
                for entry in archive.entries()? {
                    let mut entry = entry?;
                    let relative_path = entry.path()?.into_owned();
                    if relative_path.is_absolute()
                        || relative_path
                            .components()
                            .any(|component| !matches!(component, Component::Normal(_)))
                    {
                        return Err(ManagedError::UnsafeManifestPath);
                    }
                    let destination = root.join(relative_path);
                    let Ok(metadata) = fs::symlink_metadata(&destination) else {
                        return Ok(false);
                    };
                    let entry_type = entry.header().entry_type();
                    if entry_type.is_dir() {
                        if !metadata.is_dir() || metadata.file_type().is_symlink() {
                            return Ok(false);
                        }
                    } else if entry_type.is_file() {
                        if !metadata.is_file()
                            || metadata.file_type().is_symlink()
                            || metadata.len() != entry.header().size()?
                            || sha256_file(&destination)? != sha256_reader(&mut entry)?
                        {
                            return Ok(false);
                        }
                    } else {
                        return Err(ManagedError::UnsafeManifestPath);
                    }
                }
            }
        }
    }
    Ok(true)
}

fn extract_service_archives(
    root: &Path,
    manifest: &ManagedServiceManifest,
) -> Result<(), ManagedError> {
    for archive_manifest in &manifest.archives {
        let archive_path = safe_join(root, &archive_manifest.path)?;
        let input = File::open(archive_path)?;
        match archive_manifest.format {
            ManagedArchiveFormat::TarBz2 => {
                let decoder = BzDecoder::new(input);
                let mut archive = tar::Archive::new(decoder);
                for entry in archive.entries()? {
                    let mut entry = entry?;
                    let entry_type = entry.header().entry_type();
                    if !entry_type.is_file() && !entry_type.is_dir() {
                        return Err(ManagedError::UnsafeManifestPath);
                    }
                    if !entry.unpack_in(root)? {
                        return Err(ManagedError::UnsafeManifestPath);
                    }
                }
            }
        }
    }
    Ok(())
}

async fn download_file(
    client: &Client,
    root: &Path,
    manifest: &ManagedFileManifest,
    app: &AppHandle,
    service: &ManagedServiceManifest,
    service_index: usize,
    service_count: usize,
    completed_before: u64,
    total_bytes: u64,
) -> Result<(), ManagedError> {
    if !manifest.url.starts_with("https://") {
        return Err(ManagedError::InvalidConfiguration(
            "managed model downloads must use HTTPS".to_owned(),
        ));
    }
    let destination = safe_join(root, &manifest.path)?;
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let partial = destination.with_extension(format!(
        "{}.part",
        destination
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
    ));
    let response = send_download_request(client, &manifest.url).await?;
    let mut output = File::create(&partial)?;
    let mut response = response;
    let mut digest = Sha256::new();
    let mut written = 0_u64;
    let mut last_progress = Instant::now() - Duration::from_secs(1);
    while let Some(chunk) = response.chunk().await? {
        output.write_all(&chunk)?;
        digest.update(&chunk);
        written = written.saturating_add(chunk.len() as u64);
        if last_progress.elapsed() >= Duration::from_millis(100) || written == manifest.size {
            emit_managed_progress(
                app,
                "downloading",
                Some(service),
                service_index,
                service_count,
                completed_before.saturating_add(written),
                total_bytes,
                Some(&manifest.path),
            );
            last_progress = Instant::now();
        }
    }
    output.sync_all()?;
    drop(output);

    let actual_hash = format!("{:x}", digest.finalize());
    if written != manifest.size || actual_hash != manifest.sha256 {
        let _ = fs::remove_file(&partial);
        return Err(ManagedError::VerificationFailed(manifest.path.clone()));
    }
    if destination.exists() {
        fs::remove_file(&destination)?;
    }
    fs::rename(partial, destination)?;
    Ok(())
}

async fn send_download_request(client: &Client, url: &str) -> Result<Response, ManagedError> {
    let primary_result = client
        .get(url)
        .send()
        .await
        .and_then(Response::error_for_status);
    let primary_error = match primary_result {
        Ok(response) => return Ok(response),
        Err(error) => error,
    };
    let Some(mirror_url) = hugging_face_mirror_url(url) else {
        return Err(ManagedError::Download(primary_error));
    };
    client
        .get(mirror_url)
        .send()
        .await
        .and_then(Response::error_for_status)
        .map_err(|mirror| ManagedError::DownloadFallback {
            primary: primary_error,
            mirror,
        })
}

fn hugging_face_mirror_url(url: &str) -> Option<String> {
    url.strip_prefix(HUGGING_FACE_ORIGIN)
        .map(|path| format!("{HUGGING_FACE_MIRROR_ORIGIN}{path}"))
}

fn emit_managed_progress(
    app: &AppHandle,
    phase: &'static str,
    service: Option<&ManagedServiceManifest>,
    service_index: usize,
    service_count: usize,
    completed_bytes: u64,
    total_bytes: u64,
    file_path: Option<&str>,
) {
    let _ = app.emit(
        MANAGED_PROGRESS_EVENT,
        ManagedModelProgress {
            phase,
            service_id: service.map(|value| value.id.clone()),
            service_index,
            service_count,
            completed_bytes,
            total_bytes,
            file_path: file_path.map(str::to_owned),
        },
    );
}

fn write_install_marker(
    root: &Path,
    manifest: &ManagedServiceManifest,
) -> Result<(), ManagedError> {
    let marker = root.join(INSTALL_MARKER);
    let temporary = root.join(".complete.json.part");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(&json!({
            "schema_version": 1,
            "id": manifest.id,
            "version": manifest.version
        }))?,
    )?;
    fs::rename(temporary, marker)?;
    Ok(())
}

fn safe_join(root: &Path, relative: &str) -> Result<PathBuf, ManagedError> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ManagedError::UnsafeManifestPath);
    }
    Ok(root.join(relative_path))
}

fn sha256_file(path: &Path) -> Result<String, ManagedError> {
    sha256_reader(File::open(path)?)
}

fn sha256_reader(mut input: impl Read) -> Result<String, ManagedError> {
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

async fn start_mlx(
    app: &AppHandle,
    service_id: &str,
    model_root: &Path,
) -> Result<StartedService, ManagedError> {
    let command = app
        .shell()
        .sidecar(MLX_SIDECAR_NAME)?
        .env("LLVM_PROFILE_FILE", "/dev/null")
        .args([
            "--service".to_owned(),
            service_id.to_owned(),
            "--model-root".to_owned(),
            model_root.to_string_lossy().into_owned(),
            "--host".to_owned(),
            "127.0.0.1".to_owned(),
            "--port".to_owned(),
            "0".to_owned(),
        ]);
    let (mut events, child) = command.spawn()?;
    let port = match receive_model_ready(&mut events).await {
        Ok(port) => port,
        Err(error) => {
            let _ = child.kill();
            return Err(error);
        }
    };
    Ok(StartedService {
        port,
        child,
        events,
    })
}

async fn start_moss_tts(
    app: &AppHandle,
    model_root: &Path,
    backend: ManagedBackend,
) -> Result<StartedService, ManagedError> {
    let runtime_dir = app
        .path()
        .resolve(ONNX_RUNTIME_RESOURCE, BaseDirectory::Resource)?;
    let ort_library = find_ort_library(&runtime_dir)?;
    let command = app.shell().sidecar(TTS_SIDECAR_NAME)?.args([
        "--service".to_owned(),
        MOSS_TTS_ID.to_owned(),
        "--model-root".to_owned(),
        model_root.to_string_lossy().into_owned(),
        "--ort-dylib".to_owned(),
        ort_library.to_string_lossy().into_owned(),
        "--backend".to_owned(),
        onnx_backend_name(backend).to_owned(),
        "--host".to_owned(),
        "127.0.0.1".to_owned(),
        "--port".to_owned(),
        "0".to_owned(),
    ]);
    let command = configure_library_path(command, &runtime_dir);
    let (mut events, child) = command.spawn()?;
    let port = match receive_model_ready(&mut events).await {
        Ok(port) => port,
        Err(error) => {
            let _ = child.kill();
            return Err(error);
        }
    };
    Ok(StartedService {
        port,
        child,
        events,
    })
}

async fn start_refiner(
    app: &AppHandle,
    model_root: &Path,
    backend: ManagedBackend,
) -> Result<StartedService, ManagedError> {
    let runtime_dir = app
        .path()
        .resolve(ONNX_RUNTIME_RESOURCE, BaseDirectory::Resource)?;
    let ort_library = find_ort_library(&runtime_dir)?;
    let command = app.shell().sidecar(TTS_SIDECAR_NAME)?.args([
        "--service".to_owned(),
        REFINER_ID.to_owned(),
        "--model-root".to_owned(),
        model_root.to_string_lossy().into_owned(),
        "--ort-dylib".to_owned(),
        ort_library.to_string_lossy().into_owned(),
        "--backend".to_owned(),
        onnx_backend_name(backend).to_owned(),
        "--host".to_owned(),
        "127.0.0.1".to_owned(),
        "--port".to_owned(),
        "0".to_owned(),
    ]);
    let command = configure_library_path(command, &runtime_dir);
    let (mut events, child) = command.spawn()?;
    let port = match receive_model_ready(&mut events).await {
        Ok(port) => port,
        Err(error) => {
            let _ = child.kill();
            return Err(error);
        }
    };
    Ok(StartedService {
        port,
        child,
        events,
    })
}

async fn start_matcha_tts(
    app: &AppHandle,
    model_root: &Path,
    backend: ManagedBackend,
) -> Result<StartedService, ManagedError> {
    let runtime_dir = app
        .path()
        .resolve(ONNX_RUNTIME_RESOURCE, BaseDirectory::Resource)?;
    find_ort_library(&runtime_dir)?;
    let command = app.shell().sidecar(MATCHA_TTS_SIDECAR_NAME)?.args([
        "--model-root".to_owned(),
        model_root.to_string_lossy().into_owned(),
        "--backend".to_owned(),
        onnx_backend_name(backend).to_owned(),
        "--host".to_owned(),
        "127.0.0.1".to_owned(),
        "--port".to_owned(),
        "0".to_owned(),
    ]);
    let command = configure_library_path(command, &runtime_dir);
    let (mut events, child) = command.spawn()?;
    let port = match receive_model_ready(&mut events).await {
        Ok(port) => port,
        Err(error) => {
            let _ = child.kill();
            return Err(error);
        }
    };
    Ok(StartedService {
        port,
        child,
        events,
    })
}

async fn start_sensevoice(
    app: &AppHandle,
    model_root: &Path,
    backend: ManagedBackend,
) -> Result<StartedService, ManagedError> {
    let port = reserve_loopback_port()?;
    let runtime_dir = app
        .path()
        .resolve(ONNX_RUNTIME_RESOURCE, BaseDirectory::Resource)?;
    find_ort_library(&runtime_dir)?;
    let mut command = app.shell().sidecar(SHERPA_SIDECAR_NAME)?.args([
        format!("--port={port}"),
        "--num-work-threads=3".to_owned(),
        "--num-threads=2".to_owned(),
        "--max-batch-size=5".to_owned(),
        format!("--provider={}", onnx_backend_name(backend)),
        "--sense-voice-use-itn=true".to_owned(),
        format!(
            "--sense-voice-model={}",
            model_root.join("model.int8.onnx").display()
        ),
        format!("--tokens={}", model_root.join("tokens.txt").display()),
        format!("--log-file={}", model_root.join("sherpa.log").display()),
    ]);
    command = configure_library_path(command, &runtime_dir);
    let (mut events, child) = command.spawn()?;
    if let Err(error) = wait_for_tcp_ready(&mut events, port).await {
        let _ = child.kill();
        return Err(error);
    }
    Ok(StartedService {
        port,
        child,
        events,
    })
}

fn onnx_backend_name(backend: ManagedBackend) -> &'static str {
    match backend {
        ManagedBackend::Cpu => "cpu",
        ManagedBackend::Cuda => "cuda",
        ManagedBackend::Auto | ManagedBackend::Mlx => {
            unreachable!("only resolved ONNX backends have provider names")
        }
    }
}

fn reserve_loopback_port() -> Result<u16, ManagedError> {
    let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))?;
    Ok(listener.local_addr()?.port())
}

async fn receive_model_ready(events: &mut Receiver<CommandEvent>) -> Result<u16, ManagedError> {
    timeout(STARTUP_TIMEOUT, async {
        let mut stderr = String::new();
        loop {
            match events.recv().await {
                Some(CommandEvent::Stdout(line)) => {
                    if line.is_empty() || line.len() > MAX_READY_LINE_BYTES {
                        continue;
                    }
                    let Ok(ready) = serde_json::from_slice::<ModelReadyMessage>(&line) else {
                        continue;
                    };
                    if ready.status == "ready" && ready.protocol_version == 1 && ready.port != 0 {
                        return Ok(ready.port);
                    }
                    return Err(ManagedError::InvalidReady);
                }
                Some(CommandEvent::Stderr(line)) => {
                    if stderr.len() < MAX_READY_LINE_BYTES {
                        let detail = String::from_utf8_lossy(&line);
                        for character in detail.chars() {
                            if stderr.len() + character.len_utf8() > MAX_READY_LINE_BYTES {
                                break;
                            }
                            stderr.push(character);
                        }
                    }
                }
                Some(CommandEvent::Terminated(_)) | None => {
                    let detail = stderr.trim();
                    return Err(ManagedError::TerminatedBeforeReady(if detail.is_empty() {
                        "no diagnostic output".to_owned()
                    } else {
                        detail.to_owned()
                    }));
                }
                Some(CommandEvent::Error(_)) => return Err(ManagedError::InvalidReady),
                Some(_) => {}
            }
        }
    })
    .await
    .map_err(|_| ManagedError::StartupTimedOut)?
}

async fn wait_for_tcp_ready(
    events: &mut Receiver<CommandEvent>,
    port: u16,
) -> Result<(), ManagedError> {
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    loop {
        if tokio::net::TcpStream::connect((Ipv4Addr::LOCALHOST, port))
            .await
            .is_ok()
        {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(ManagedError::StartupTimedOut);
        }
        tokio::select! {
            event = events.recv() => {
                if matches!(event, Some(CommandEvent::Terminated(_)) | None) {
                    return Err(ManagedError::TerminatedBeforeReady(
                        "no diagnostic output".to_owned(),
                    ));
                }
            }
            _ = sleep(STARTUP_POLL_INTERVAL) => {}
        }
    }
}

async fn monitor_service(
    mut events: Receiver<CommandEvent>,
    inner: std::sync::Weak<ManagedServicesInner>,
) {
    while let Some(event) = events.recv().await {
        if matches!(event, CommandEvent::Terminated(_)) {
            break;
        }
    }
    if let Some(inner) = inner.upgrade() {
        if !inner.shutting_down.load(Ordering::Acquire) {
            inner.healthy.store(false, Ordering::Release);
            inner.failure_sender.send_replace(true);
            eprintln!("a managed model service terminated unexpectedly");
        }
    }
}

fn find_ort_library(directory: &Path) -> Result<PathBuf, ManagedError> {
    #[cfg(target_os = "macos")]
    let candidates: &[&str] = &["libonnxruntime.1.27.0.dylib", "libonnxruntime.dylib"];
    #[cfg(target_os = "linux")]
    let candidates: &[&str] = &["libonnxruntime.so.1.27.0", "libonnxruntime.so"];
    #[cfg(target_os = "windows")]
    let candidates: &[&str] = &["onnxruntime.dll"];

    candidates
        .iter()
        .map(|name| directory.join(name))
        .find(|path| path.is_file())
        .ok_or(ManagedError::MissingRuntime)
}

pub(crate) fn configure_library_path(
    command: tauri_plugin_shell::process::Command,
    runtime_dir: &Path,
) -> tauri_plugin_shell::process::Command {
    #[cfg(target_os = "macos")]
    let key = "DYLD_LIBRARY_PATH";
    #[cfg(target_os = "linux")]
    let key = "LD_LIBRARY_PATH";
    #[cfg(target_os = "windows")]
    let key = "PATH";

    let mut paths = vec![runtime_dir.to_path_buf()];
    if let Some(existing) = std::env::var_os(key) {
        paths.extend(std::env::split_paths(&existing));
    }
    let value =
        std::env::join_paths(paths).unwrap_or_else(|_| runtime_dir.as_os_str().to_os_string());
    command.env(key, value)
}

#[cfg(test)]
mod tests {
    use super::{
        extract_service_archives, hugging_face_mirror_url, inspect_model_config,
        parse_managed_request, resolve_moss_voices, safe_join, select_backend,
        verify_installed_service, write_install_marker, ManagedArchiveFormat,
        ManagedArchiveManifest, ManagedBackend, ManagedServiceManifest, ManagedVoice,
        DEFAULT_MOSS_VOICE_URL, MATCHA_TTS_ID, MOSS_TTS_ID, REFINER_ID, SENSEVOICE_ID,
    };
    use bzip2::{write::BzEncoder, Compression};
    use serde_json::json;
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn create() -> Self {
            let sequence = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "xtalk-managed-test-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create temporary directory");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn safe_join_rejects_parent_components() {
        assert!(safe_join(Path::new("/tmp/models"), "../secret").is_err());
        assert!(safe_join(Path::new("/tmp/models"), "weights/model.onnx").is_ok());
    }

    #[test]
    fn hugging_face_downloads_have_a_same_path_mirror() {
        assert_eq!(
            hugging_face_mirror_url(
                "https://huggingface.co/example/model/resolve/commit/model.safetensors"
            )
            .as_deref(),
            Some("https://hf-mirror.com/example/model/resolve/commit/model.safetensors")
        );
        assert_eq!(
            hugging_face_mirror_url("https://example.com/model.safetensors"),
            None
        );
    }

    #[test]
    fn automatic_backend_prefers_cuda_then_mlx_then_cpu() {
        assert_eq!(
            select_backend(ManagedBackend::Auto, true, true).expect("select CUDA"),
            ManagedBackend::Cuda
        );
        assert_eq!(
            select_backend(ManagedBackend::Auto, false, true).expect("select MLX"),
            ManagedBackend::Mlx
        );
        assert_eq!(
            select_backend(ManagedBackend::Auto, false, false).expect("select CPU"),
            ManagedBackend::Cpu
        );
        assert!(select_backend(ManagedBackend::Cuda, false, true).is_err());
        assert!(select_backend(ManagedBackend::Mlx, true, false).is_err());
    }

    #[test]
    fn parses_both_managed_model_requests() {
        let directory = TestDirectory::create();
        let config_path = directory.path().join("config.json");
        fs::write(
            &config_path,
            serde_json::to_vec(&json!({
                "asr": {
                    "type": "SherpaOnnxASR",
                    "params": {"base_url": "managed://sensevoice-small"}
                },
                "tts": {
                    "type": "MossTTSNano",
                    "params": {
                        "base_url": "managed://moss-tts-nano",
                        "voices": [{
                            "name": "zh",
                            "path": DEFAULT_MOSS_VOICE_URL
                        }]
                    }
                }
            }))
            .expect("serialize config"),
        )
        .expect("write config");

        let request = parse_managed_request(&config_path).expect("parse config");
        assert_eq!(request.sensevoice, Some(ManagedBackend::Auto));
        assert_eq!(request.moss_tts, Some(ManagedBackend::Auto));
        assert_eq!(request.moss_voices.len(), 1);

        let plan = inspect_model_config(&config_path).expect("inspect config");
        assert_eq!(plan.services, [SENSEVOICE_ID, MOSS_TTS_ID]);
    }

    #[test]
    fn parses_explicit_backends_without_changing_logical_service_ids() {
        let directory = TestDirectory::create();
        let config_path = directory.path().join("config.json");
        fs::write(
            &config_path,
            serde_json::to_vec(&json!({
                "asr": {
                    "type": "SherpaOnnxASR",
                    "params": {"base_url": "managed://sensevoice-small?backend=cuda"}
                },
                "tts": {
                    "type": "MossTTSNano",
                    "params": {"base_url": "managed://moss-tts-nano?backend=mlx"}
                }
            }))
            .expect("serialize config"),
        )
        .expect("write config");

        let request = parse_managed_request(&config_path).expect("parse config");
        assert_eq!(request.sensevoice, Some(ManagedBackend::Cuda));
        assert_eq!(request.moss_tts, Some(ManagedBackend::Mlx));
        let plan = inspect_model_config(&config_path).expect("inspect config");
        assert_eq!(plan.services, [SENSEVOICE_ID, MOSS_TTS_ID]);
    }

    #[test]
    fn parses_managed_agentic_asr_services_in_dependency_order() {
        let directory = TestDirectory::create();
        let config_path = directory.path().join("config.json");
        fs::write(
            &config_path,
            serde_json::to_vec(&json!({
                "asr": {
                    "type": "AgenticASR",
                    "params": {
                        "asr_base_url": "managed://sensevoice-small",
                        "refiner_base_url": "managed://agentic-asr-refiner",
                        "asr_mode": "offline"
                    }
                }
            }))
            .expect("serialize config"),
        )
        .expect("write config");

        let request = parse_managed_request(&config_path).expect("parse config");
        assert!(request.agentic_asr);
        assert_eq!(request.sensevoice, Some(ManagedBackend::Auto));
        assert_eq!(request.refiner, Some(ManagedBackend::Auto));
        let plan = inspect_model_config(&config_path).expect("inspect config");
        assert_eq!(plan.services, [SENSEVOICE_ID, REFINER_ID]);
    }

    #[test]
    fn rejects_streaming_mode_for_managed_agentic_sensevoice() {
        let directory = TestDirectory::create();
        let config_path = directory.path().join("config.json");
        fs::write(
            &config_path,
            serde_json::to_vec(&json!({
                "asr": {
                    "type": "AgenticASR",
                    "params": {
                        "asr_base_url": "managed://sensevoice-small",
                        "refiner_base_url": "managed://agentic-asr-refiner?backend=cpu",
                        "asr_mode": "streaming"
                    }
                }
            }))
            .expect("serialize config"),
        )
        .expect("write config");

        assert!(parse_managed_request(&config_path).is_err());
    }

    #[test]
    fn parses_managed_matcha_tts_request() {
        let directory = TestDirectory::create();
        let config_path = directory.path().join("config.json");
        fs::write(
            &config_path,
            serde_json::to_vec(&json!({
                "asr": {
                    "type": "SherpaOnnxASR",
                    "params": {"base_url": "managed://sensevoice-small?backend=cpu"}
                },
                "tts": {
                    "type": "SherpaOnnxTTS",
                    "params": {"base_url": "managed://matcha-icefall-zh-en"}
                }
            }))
            .expect("serialize config"),
        )
        .expect("write config");

        let request = parse_managed_request(&config_path).expect("parse config");
        assert_eq!(request.sensevoice, Some(ManagedBackend::Cpu));
        assert_eq!(request.moss_tts, None);
        assert_eq!(request.matcha_tts, Some(ManagedBackend::Auto));
        let plan = inspect_model_config(&config_path).expect("inspect config");
        assert_eq!(plan.services, [SENSEVOICE_ID, MATCHA_TTS_ID]);
    }

    #[test]
    fn managed_voice_resolves_inside_model_install() {
        let directory = TestDirectory::create();
        let voice_path = directory.path().join("voices").join("zh_1.wav");
        fs::create_dir_all(voice_path.parent().expect("voice parent"))
            .expect("create voice directory");
        fs::write(&voice_path, b"wave").expect("write voice");
        let voices = vec![ManagedVoice {
            name: "zh".to_owned(),
            path: DEFAULT_MOSS_VOICE_URL.to_owned(),
        }];
        let resolved = resolve_moss_voices(&voices, directory.path()).expect("resolve voice");
        assert_eq!(resolved[0]["path"], json!(voice_path));
    }

    #[test]
    fn extracts_pinned_tar_bz2_service_archive() {
        let directory = TestDirectory::create();
        let archive_path = directory.path().join("archives").join("model.tar.bz2");
        fs::create_dir_all(archive_path.parent().expect("archive parent"))
            .expect("create archive directory");
        let output = fs::File::create(&archive_path).expect("create archive");
        let encoder = BzEncoder::new(output, Compression::best());
        let mut archive = tar::Builder::new(encoder);
        let content = b"model";
        let mut header = tar::Header::new_gnu();
        header.set_size(content.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        archive
            .append_data(&mut header, "matcha/model.onnx", &content[..])
            .expect("append archive entry");
        let encoder = archive.into_inner().expect("finish tar archive");
        encoder.finish().expect("finish bzip2 stream");

        let manifest = ManagedServiceManifest {
            id: "matcha".to_owned(),
            version: "test".to_owned(),
            files: Vec::new(),
            archives: vec![ManagedArchiveManifest {
                path: "archives/model.tar.bz2".to_owned(),
                format: ManagedArchiveFormat::TarBz2,
            }],
            required_paths: vec!["matcha/model.onnx".to_owned()],
        };

        extract_service_archives(directory.path(), &manifest).expect("extract archive");
        write_install_marker(directory.path(), &manifest).expect("write marker");
        assert_eq!(
            fs::read(directory.path().join("matcha").join("model.onnx"))
                .expect("read extracted model"),
            content
        );
        assert!(verify_installed_service(directory.path(), &manifest)
            .expect("verify extracted archive"));
        fs::write(
            directory.path().join("matcha").join("model.onnx"),
            b"tampered",
        )
        .expect("tamper extracted model");
        assert!(!verify_installed_service(directory.path(), &manifest)
            .expect("reject tampered archive output"));
    }
}
