//! Tauri application lifecycle for XTalk Desktop.

#![warn(missing_docs)]

mod sidecar;

use std::{path::PathBuf, sync::Arc};

use sidecar::{BackendSupervisor, NativeBackendConnection, NativeModelConfigSelection};
use tauri::{Manager, State, WindowEvent};

/// Runs the XTalk Desktop native shell.
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            apply_model_config,
            get_backend_connection,
            get_model_config_selection,
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
