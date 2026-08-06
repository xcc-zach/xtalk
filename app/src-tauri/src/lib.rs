//! Tauri application lifecycle for XTalk Desktop.

#![warn(missing_docs)]

mod credentials;
mod managed;
mod sidecar;
mod tools;
mod whiteboard;

use std::{path::PathBuf, sync::Arc};

use sidecar::{
    inspect_managed_model_config, BackendSupervisor, NativeBackendConnection,
    NativeModelConfigSelection,
};
use tauri::{Manager, State, WindowEvent};
use tools::NativeToolDefinition;

/// Runs the XTalk Desktop native shell.
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            apply_tool_changes,
            apply_model_config,
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
            Ok(())
        })
        .on_window_event(handle_window_event)
        .run(tauri::generate_context!())
        .expect("failed to run XTalk Desktop");
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
fn show_whiteboard_window(app: tauri::AppHandle) -> Result<bool, String> {
    whiteboard::show_whiteboard_window(&app)
}

#[tauri::command]
fn hide_whiteboard_window(app: tauri::AppHandle) -> Result<bool, String> {
    whiteboard::hide_whiteboard_window(&app)
}

#[tauri::command]
fn set_whiteboard_window_visible(app: tauri::AppHandle, visible: bool) -> Result<bool, String> {
    whiteboard::set_whiteboard_window_visible(&app, visible)
}

#[tauri::command]
fn is_whiteboard_window_visible(app: tauri::AppHandle) -> Result<bool, String> {
    Ok(whiteboard::is_whiteboard_window_visible(&app))
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
    let supervisor = app.state::<Arc<BackendSupervisor>>().inner().clone();
    if !supervisor.begin_app_close() {
        return;
    }

    let window = window.clone();
    tauri::async_runtime::spawn(async move {
        if let Err(error) = supervisor.shutdown().await {
            eprintln!("app-backend shutdown did not complete cleanly: {error}");
        }
        if let Err(error) = window.destroy() {
            eprintln!("failed to destroy the main window: {error}");
        }
        app.exit(0);
    });
}
