//! Global whiteboard window management for XTalk Desktop.

use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use tauri::{AppHandle, Emitter, Manager, WebviewUrl, WebviewWindowBuilder, Window, WindowEvent};

/// Stable webview label of the global whiteboard window.
pub const WHITEBOARD_WINDOW_LABEL: &str = "whiteboard";

/// Event emitted when the whiteboard window is hidden, so the main window can
/// keep its dock button in sync even when the user closes the window itself.
pub const WHITEBOARD_WINDOW_HIDDEN_EVENT: &str = "whiteboard-window-hidden";

/// Appends one diagnostic line to the whiteboard window log and stderr.
fn log_whiteboard(app: &AppHandle, message: &str) {
    eprintln!("whiteboard window: {message}");
    let path = app
        .path()
        .app_data_dir()
        .map(|directory| directory.join("whiteboard-window.log"))
        .unwrap_or_else(|_| PathBuf::from("/tmp/xtalk-whiteboard.log"));
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "?".to_string());
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(file, "{timestamp} {message}");
    }
}

/// Opens the global whiteboard window, creating it on first use.
///
/// Returns whether the window is visible afterwards.
pub fn show_whiteboard_window(app: &AppHandle) -> Result<bool, String> {
    log_whiteboard(app, "show_whiteboard_window called");
    let window = match app.get_webview_window(WHITEBOARD_WINDOW_LABEL) {
        Some(window) => window,
        None => {
            log_whiteboard(app, "creating whiteboard window");
            let mut builder = WebviewWindowBuilder::new(
                app,
                WHITEBOARD_WINDOW_LABEL,
                WebviewUrl::App("whiteboard.html".into()),
            )
            .title("XTalk Whiteboard")
            .inner_size(640.0, 760.0)
            .min_inner_size(400.0, 420.0)
            .resizable(true);
            if let Some(main_window) = app.get_webview_window("main") {
                // Keep the whiteboard floating above the XTalk main window.
                match builder.parent(&main_window) {
                    Ok(parented) => builder = parented,
                    Err(error) => {
                        let message =
                            format!("failed to parent whiteboard window: {error}");
                        log_whiteboard(app, &message);
                        return Err(message);
                    }
                }
            }
            match builder.build() {
                Ok(window) => window,
                Err(error) => {
                    let message = format!("failed to create whiteboard window: {error}");
                    log_whiteboard(app, &message);
                    return Err(message);
                }
            }
        }
    };
    if let Err(error) = window.show() {
        let message = format!("failed to show whiteboard window: {error}");
        log_whiteboard(app, &message);
        return Err(message);
    }
    if let Err(error) = window.set_focus() {
        let message = format!("failed to focus whiteboard window: {error}");
        log_whiteboard(app, &message);
        return Err(message);
    }
    log_whiteboard(app, "whiteboard window shown");
    Ok(true)
}

/// Hides the global whiteboard window without destroying its webview.
///
/// Returns whether a whiteboard window existed to hide.
pub fn hide_whiteboard_window(app: &AppHandle) -> Result<bool, String> {
    let Some(window) = app.get_webview_window(WHITEBOARD_WINDOW_LABEL) else {
        log_whiteboard(app, "hide_whiteboard_window called with no window");
        return Ok(false);
    };
    if let Err(error) = window.hide() {
        let message = format!("failed to hide whiteboard window: {error}");
        log_whiteboard(app, &message);
        return Err(message);
    }
    log_whiteboard(app, "whiteboard window hidden");
    Ok(true)
}

/// Sets the global whiteboard window visibility.
///
/// Returns the resulting visibility.
pub fn set_whiteboard_window_visible(app: &AppHandle, visible: bool) -> Result<bool, String> {
    log_whiteboard(app, &format!("set_whiteboard_window_visible called with {visible}"));
    if visible {
        show_whiteboard_window(app)
    } else {
        hide_whiteboard_window(app)
    }
}

/// Returns whether the global whiteboard window is currently visible.
pub fn is_whiteboard_window_visible(app: &AppHandle) -> bool {
    app.get_webview_window(WHITEBOARD_WINDOW_LABEL)
        .and_then(|window| window.is_visible().ok())
        .unwrap_or(false)
}

/// Converts a whiteboard window close request into a hide plus an event.
pub fn handle_whiteboard_window_event(window: &Window, event: &WindowEvent) {
    if window.label() != WHITEBOARD_WINDOW_LABEL {
        return;
    }
    let WindowEvent::CloseRequested { api, .. } = event else {
        return;
    };
    api.prevent_close();
    if let Err(error) = window.hide() {
        eprintln!("failed to hide the whiteboard window: {error}");
    }
    let _ = window
        .app_handle()
        .emit(WHITEBOARD_WINDOW_HIDDEN_EVENT, ());
}
