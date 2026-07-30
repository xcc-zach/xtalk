const APP_COMMANDS: &[&str] = &["get_backend_connection", "shutdown_backend"];

fn main() {
    tauri_build::try_build(
        tauri_build::Attributes::new()
            .app_manifest(tauri_build::AppManifest::new().commands(APP_COMMANDS)),
    )
    .expect("failed to prepare the XTalk Desktop Tauri build");
}
