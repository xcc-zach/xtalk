//! Tauri application lifecycle for XTalk Desktop.

#![warn(missing_docs)]

mod sidecar;
mod tools;

use std::{path::PathBuf, sync::Arc};

use sidecar::{
    BackendSupervisor, NativeBackendConnection, NativeModelConfigSelection,
    NativeWebSearchSettings,
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
            get_backend_connection,
            get_web_search_settings,
            get_installed_tools,
            get_model_config_selection,
            install_tool_directory,
            remove_installed_tool,
            set_tool_enabled,
            shutdown_backend
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
async fn get_model_config_selection(
    supervisor: State<'_, Arc<BackendSupervisor>>,
) -> Result<NativeModelConfigSelection, String> {
    Ok(supervisor.selection().await)
}

#[tauri::command]
async fn get_web_search_settings(
    app: tauri::AppHandle,
    supervisor: State<'_, Arc<BackendSupervisor>>,
) -> Result<NativeWebSearchSettings, String> {
    supervisor
        .web_search_settings(&app)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command(rename_all = "camelCase")]
async fn apply_model_config(
    app: tauri::AppHandle,
    supervisor: State<'_, Arc<BackendSupervisor>>,
    config_path: PathBuf,
    web_search_enabled: bool,
    web_search_api_key: Option<String>,
) -> Result<NativeBackendConnection, String> {
    supervisor
        .configure_web_search(&app, web_search_enabled, web_search_api_key)
        .await
        .map_err(|error| error.to_string())?;
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
fn install_tool_directory(
    app: tauri::AppHandle,
    source_path: PathBuf,
) -> Result<NativeToolDefinition, String> {
    tools::install_tool_directory(&app, &source_path)
}

#[tauri::command(rename_all = "camelCase")]
fn set_tool_enabled(
    app: tauri::AppHandle,
    tool_id: String,
    enabled: bool,
) -> Result<NativeToolDefinition, String> {
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
    web_search_enabled: bool,
    web_search_api_key: Option<String>,
) -> Result<NativeBackendConnection, String> {
    supervisor
        .configure_web_search(&app, web_search_enabled, web_search_api_key)
        .await
        .map_err(|error| error.to_string())?;
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

fn handle_window_event(window: &tauri::Window, event: &WindowEvent) {
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
