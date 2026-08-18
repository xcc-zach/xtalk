//! Unsupported-platform placeholder for Tauri's target-specific MLX sidecar.

fn main() {
    eprintln!("the MLX managed runtime is supported only on Apple Silicon macOS");
    std::process::exit(1);
}
