const APP_COMMANDS: &[&str] = &[
    "apply_tool_changes",
    "apply_model_config",
    "delete_credential",
    "ensure_backend_started",
    "get_backend_connection",
    "get_credentials",
    "get_installed_tools",
    "get_managed_model_plan",
    "get_tool_ui_source",
    "get_model_config_selection",
    "get_recommended_model_config",
    "install_tool_directory",
    "remove_installed_tool",
    "save_credential",
    "set_tool_enabled",
    "get_wake_word_settings",
    "pause_wake_word",
    "resume_wake_word",
    "set_wake_word_enabled",
    "set_wake_word_phrase",
    "shutdown_backend",
    "show_whiteboard_window",
    "hide_whiteboard_window",
    "set_whiteboard_window_visible",
    "is_whiteboard_window_visible",
];

fn main() {
    tauri_build::try_build(
        tauri_build::Attributes::new()
            .app_manifest(tauri_build::AppManifest::new().commands(APP_COMMANDS)),
    )
    .expect("failed to prepare the XTalk Desktop Tauri build");
}
