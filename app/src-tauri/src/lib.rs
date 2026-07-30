//! Tauri application lifecycle for XTalk Desktop.

#![warn(missing_docs)]

mod sidecar;

use std::sync::Arc;

use sidecar::{BackendManager, NativeBackendConnection};
use tauri::{Manager, State, WindowEvent};

/// Runs the XTalk Desktop native shell.
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            get_backend_connection,
            shutdown_backend
        ])
        .setup(|app| {
            let manager = tauri::async_runtime::block_on(BackendManager::start(app.handle()))
                .map_err(|error| -> Box<dyn std::error::Error> { Box::new(error) })?;
            app.manage(manager);
            Ok(())
        })
        .on_window_event(handle_window_event)
        .run(tauri::generate_context!())
        .expect("failed to run XTalk Desktop");
}

#[tauri::command]
fn get_backend_connection(
    manager: State<'_, Arc<BackendManager>>,
) -> Result<NativeBackendConnection, String> {
    manager.connection().map_err(|error| error.to_string())
}

#[tauri::command]
async fn shutdown_backend(app: tauri::AppHandle) -> Result<(), String> {
    let manager = app.state::<Arc<BackendManager>>().inner().clone();
    manager.shutdown().await.map_err(|error| error.to_string())
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
    let manager = app.state::<Arc<BackendManager>>().inner().clone();
    if !manager.begin_app_close() {
        return;
    }

    let window = window.clone();
    tauri::async_runtime::spawn(async move {
        if let Err(error) = manager.shutdown().await {
            eprintln!("app-backend shutdown did not complete cleanly: {error}");
        }
        if let Err(error) = window.destroy() {
            eprintln!("failed to destroy the main window: {error}");
        }
        app.exit(0);
    });
}
