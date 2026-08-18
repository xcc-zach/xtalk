//! Tauri application lifecycle for XTalk Desktop.

#![warn(missing_docs)]

mod credentials;
mod managed;
mod sidecar;
mod tools;
mod tray;
mod wake_word;
mod whiteboard;

use std::{path::PathBuf, sync::Arc};

use sidecar::{
    inspect_managed_model_config, BackendSupervisor, NativeBackendConnection,
    NativeModelConfigSelection,
};
use tauri::{Emitter, Manager, RunEvent, State, WindowEvent};
use tools::NativeToolDefinition;
use wake_word::{NativeWakeWordSettings, WakeWordState, WakeWordSupervisor};

/// Runs the XTalk Desktop native shell.
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            apply_tool_changes,
            apply_model_config,
            background_main_window,
            ensure_backend_started,
            get_backend_connection,
            get_credentials,
            get_installed_tools,
            get_managed_model_plan,
            get_model_config_selection,
            get_recommended_model_config,
            get_tool_ui_source,
            install_tool_directory,
            remove_installed_tool,
            save_credential,
            set_tool_enabled,
            delete_credential,
            get_wake_word_settings,
            pause_wake_word,
            resume_wake_word,
            set_wake_word_enabled,
            set_wake_word_phrase,
            set_wake_word_threshold,
            shutdown_backend,
            show_whiteboard_window,
            hide_whiteboard_window,
            set_whiteboard_window_visible,
            is_whiteboard_window_visible
        ])
        .setup(|app| {
            let supervisor =
                tauri::async_runtime::block_on(BackendSupervisor::initialize(app.handle()));
            app.manage(supervisor);
            let wake_word =
                tauri::async_runtime::block_on(WakeWordSupervisor::initialize(app.handle()));
            app.manage(wake_word);
            tray::setup(app)?;
            Ok(())
        })
        .on_window_event(handle_window_event)
        .build(tauri::generate_context!())
        .expect("failed to build XTalk Desktop");
    app.run(handle_run_event);
}

#[tauri::command]
async fn get_backend_connection(
    supervisor: State<'_, Arc<BackendSupervisor>>,
) -> Result<NativeBackendConnection, String> {
    supervisor
        .connection()
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn ensure_backend_started(
    app: tauri::AppHandle,
    supervisor: State<'_, Arc<BackendSupervisor>>,
) -> Result<NativeBackendConnection, String> {
    supervisor
        .ensure_started(&app)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command(rename_all = "camelCase")]
fn get_managed_model_plan(config_path: PathBuf) -> Result<managed::ManagedModelPlan, String> {
    inspect_managed_model_config(&config_path).map_err(|error| error.to_string())
}

#[tauri::command]
async fn get_model_config_selection(
    supervisor: State<'_, Arc<BackendSupervisor>>,
) -> Result<NativeModelConfigSelection, String> {
    Ok(supervisor.selection().await)
}

#[tauri::command]
fn get_recommended_model_config(app: tauri::AppHandle) -> Result<PathBuf, String> {
    sidecar::recommended_model_config_path(&app).map_err(|error| error.to_string())
}

#[tauri::command]
async fn get_credentials(
    app: tauri::AppHandle,
) -> Result<Vec<credentials::NativeCredentialDefinition>, String> {
    credentials::list_credentials(&app)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command(rename_all = "camelCase")]
async fn save_credential(
    app: tauri::AppHandle,
    credential_id: String,
    value: String,
) -> Result<credentials::NativeCredentialDefinition, String> {
    credentials::save_credential(&app, credential_id, value)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command(rename_all = "camelCase")]
async fn delete_credential(
    app: tauri::AppHandle,
    credential_id: String,
) -> Result<credentials::NativeCredentialDefinition, String> {
    credentials::delete_credential(&app, credential_id)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command(rename_all = "camelCase")]
async fn apply_model_config(
    app: tauri::AppHandle,
    supervisor: State<'_, Arc<BackendSupervisor>>,
    config_path: PathBuf,
) -> Result<NativeBackendConnection, String> {
    supervisor
        .apply_model_config(&app, config_path)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn get_installed_tools(app: tauri::AppHandle) -> Result<Vec<NativeToolDefinition>, String> {
    tools::list_installed_tools(&app)
}

#[tauri::command(rename_all = "camelCase")]
fn get_tool_ui_source(
    app: tauri::AppHandle,
    tool_id: String,
) -> Result<tools::NativeToolUiSource, String> {
    tools::read_tool_ui_source(&app, &tool_id)
}

#[tauri::command(rename_all = "camelCase")]
fn install_tool_directory(
    app: tauri::AppHandle,
    source_path: PathBuf,
) -> Result<NativeToolDefinition, String> {
    tools::install_tool_directory(&app, &source_path)
}

#[tauri::command(rename_all = "camelCase")]
async fn set_tool_enabled(
    app: tauri::AppHandle,
    tool_id: String,
    enabled: bool,
) -> Result<NativeToolDefinition, String> {
    if enabled {
        credentials::ensure_tool_can_enable(&app, &tool_id)
            .await
            .map_err(|error| error.to_string())?;
    }
    tools::set_tool_enabled(&app, &tool_id, enabled)
}

#[tauri::command(rename_all = "camelCase")]
fn remove_installed_tool(app: tauri::AppHandle, tool_id: String) -> Result<(), String> {
    tools::remove_installed_tool(&app, &tool_id)
}

#[tauri::command(rename_all = "camelCase")]
async fn apply_tool_changes(
    app: tauri::AppHandle,
    supervisor: State<'_, Arc<BackendSupervisor>>,
) -> Result<NativeBackendConnection, String> {
    supervisor
        .restart(&app)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn shutdown_backend(app: tauri::AppHandle) -> Result<(), String> {
    let supervisor = app.state::<Arc<BackendSupervisor>>().inner().clone();
    supervisor
        .shutdown()
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn get_wake_word_settings(
    supervisor: State<'_, Arc<WakeWordSupervisor>>,
) -> Result<NativeWakeWordSettings, String> {
    Ok(supervisor.settings().await)
}

#[tauri::command(rename_all = "camelCase")]
async fn set_wake_word_enabled(
    app: tauri::AppHandle,
    supervisor: State<'_, Arc<WakeWordSupervisor>>,
    enabled: bool,
    listen_immediately: bool,
) -> Result<NativeWakeWordSettings, String> {
    supervisor
        .inner()
        .set_enabled(&app, enabled, listen_immediately)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command(rename_all = "camelCase")]
async fn set_wake_word_phrase(
    app: tauri::AppHandle,
    supervisor: State<'_, Arc<WakeWordSupervisor>>,
    phrase: String,
    listen_immediately: bool,
) -> Result<NativeWakeWordSettings, String> {
    supervisor
        .inner()
        .set_phrase(&app, phrase, listen_immediately)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command(rename_all = "camelCase")]
async fn set_wake_word_threshold(
    app: tauri::AppHandle,
    supervisor: State<'_, Arc<WakeWordSupervisor>>,
    threshold: f32,
    listen_immediately: bool,
) -> Result<NativeWakeWordSettings, String> {
    supervisor
        .inner()
        .set_threshold(&app, threshold, listen_immediately)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn pause_wake_word(
    app: tauri::AppHandle,
    supervisor: State<'_, Arc<WakeWordSupervisor>>,
) -> Result<NativeWakeWordSettings, String> {
    supervisor.pause(&app).await;
    Ok(supervisor.settings().await)
}

#[tauri::command]
async fn resume_wake_word(
    app: tauri::AppHandle,
    supervisor: State<'_, Arc<WakeWordSupervisor>>,
) -> Result<NativeWakeWordSettings, String> {
    supervisor
        .inner()
        .resume(&app)
        .await
        .map_err(|error| error.to_string())
}

/// Opens the whiteboard without blocking the Windows webview event loop.
#[tauri::command]
async fn show_whiteboard_window(app: tauri::AppHandle) -> Result<bool, String> {
    whiteboard::show_whiteboard_window(&app)
}

#[tauri::command]
fn hide_whiteboard_window(app: tauri::AppHandle) -> Result<bool, String> {
    whiteboard::hide_whiteboard_window(&app)
}

/// Updates whiteboard visibility outside the synchronous Windows IPC handler.
#[tauri::command]
async fn set_whiteboard_window_visible(
    app: tauri::AppHandle,
    visible: bool,
) -> Result<bool, String> {
    whiteboard::set_whiteboard_window_visible(&app, visible)
}

#[tauri::command]
fn is_whiteboard_window_visible(app: tauri::AppHandle) -> Result<bool, String> {
    Ok(whiteboard::is_whiteboard_window_visible(&app))
}

/// Hides the main window after the WebView has restored wake-word listening.
#[tauri::command]
async fn background_main_window(app: tauri::AppHandle) -> Result<(), String> {
    let wake_word = app.state::<Arc<WakeWordSupervisor>>().inner().clone();
    let settings = wake_word.settings().await;
    if !settings.enabled {
        return Err("voice wake must be enabled before entering sleep mode".to_owned());
    }
    if settings.state != WakeWordState::Listening {
        return Err("voice wake must be listening before entering sleep mode".to_owned());
    }
    hide_main_window(&app)?;
    if wake_word.settings().await.state != WakeWordState::Listening {
        tray::show_main_window(&app);
        return Err("voice wake stopped while entering sleep mode".to_owned());
    }
    Ok(())
}

fn request_main_window_backgrounding(app: &tauri::AppHandle) -> Result<(), String> {
    app.emit("app-backgrounding", ())
        .map_err(|error| format!("failed to request app backgrounding: {error}"))
}

fn hide_main_window(app: &tauri::AppHandle) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window is missing".to_owned())?;
    window
        .hide()
        .map_err(|error| format!("failed to hide the main window: {error}"))?;
    if let Err(error) = whiteboard::hide_whiteboard_window(app) {
        eprintln!("failed to hide the whiteboard window: {error}");
    }
    Ok(())
}

fn handle_window_event(window: &tauri::Window, event: &WindowEvent) {
    whiteboard::handle_whiteboard_window_event(window, event);

    let WindowEvent::CloseRequested { api, .. } = event else {
        return;
    };
    if window.label() != "main" {
        return;
    }

    api.prevent_close();

    let app = window.app_handle().clone();
    let wake_word = app.state::<Arc<WakeWordSupervisor>>().inner().clone();
    if wake_word.is_enabled() {
        if let Err(error) = request_main_window_backgrounding(&app) {
            eprintln!("failed to request app backgrounding: {error}");
        }
        return;
    }

    request_app_exit(app);
}

pub(crate) fn request_app_exit(app: tauri::AppHandle) {
    let supervisor = app.state::<Arc<BackendSupervisor>>().inner().clone();
    if !supervisor.begin_app_close() {
        return;
    }

    let wake_word = app.state::<Arc<WakeWordSupervisor>>().inner().clone();
    tauri::async_runtime::spawn(async move {
        wake_word.shutdown().await;
        if let Err(error) = supervisor.shutdown_for_app_close() {
            eprintln!("app-backend shutdown did not complete cleanly: {error}");
        }
        if let Some(window) = app.get_webview_window("main") {
            if let Err(error) = window.destroy() {
                eprintln!("failed to destroy the main window: {error}");
            }
        }
        supervisor.finish_app_close();
        app.exit(0);
    });
}

fn handle_run_event(app: &tauri::AppHandle, event: RunEvent) {
    #[cfg(target_os = "macos")]
    if let RunEvent::Reopen { .. } = &event {
        tray::show_main_window(app);
        return;
    }

    let RunEvent::ExitRequested { api, .. } = event else {
        return;
    };

    let supervisor = app.state::<Arc<BackendSupervisor>>().inner().clone();
    if supervisor.is_app_close_finished() {
        return;
    }

    api.prevent_exit();
    request_app_exit(app.clone());
}
