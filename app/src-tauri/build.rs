const APP_COMMANDS: &[&str] = &[
    "apply_tool_changes",
    "apply_model_config",
    "get_backend_connection",
    "get_installed_tools",
    "get_managed_model_plan",
    "get_tool_ui_source",
    "get_model_config_selection",
    "install_tool_directory",
    "remove_installed_tool",
    "set_tool_enabled",
    "shutdown_backend",
];

fn main() {
    tauri_build::try_build(
        tauri_build::Attributes::new()
            .app_manifest(tauri_build::AppManifest::new().commands(APP_COMMANDS)),
    )
    .expect("failed to prepare the XTalk Desktop Tauri build");
}
