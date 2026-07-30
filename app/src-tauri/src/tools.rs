//! Application-data installation and state for developer tool directories.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};
use thiserror::Error;

const TOOL_MANIFEST_FILE: &str = "xtalk_tool.json";
const TOOL_REGISTRY_FILE: &str = "registry.json";
const TOOLS_DIRECTORY: &str = "tools";

/// Developer tool metadata exposed to the trusted WebView.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct NativeToolDefinition {
    id: String,
    display_name: String,
    entrypoint: String,
    enabled: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolManifest {
    display_name: String,
    entrypoint: String,
}

#[derive(Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PersistedToolRegistry {
    tools: Vec<PersistedToolEntry>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PersistedToolEntry {
    id: String,
    enabled: bool,
}

#[derive(Debug, Error)]
enum ToolDirectoryError {
    #[error("failed to access an application path: {0}")]
    Tauri(#[from] tauri::Error),
    #[error("tool directory I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("tool configuration is not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("the selected tool path is not a directory")]
    InvalidDirectory,
    #[error("the selected directory does not contain xtalk_tool.json")]
    MissingManifest,
    #[error("tool display_name must be a non-empty string")]
    InvalidDisplayName,
    #[error("tool entrypoint must use the module:factory format")]
    InvalidEntrypoint,
    #[error("the selected tool is not installed")]
    ToolNotFound,
    #[error("could not generate an installed tool identifier: {0}")]
    Identifier(String),
}

/// Lists developer tools currently copied into application data.
pub(crate) fn list_installed_tools(app: &AppHandle) -> Result<Vec<NativeToolDefinition>, String> {
    let tools_root = tools_root(app).map_err(|error| error.to_string())?;
    list_installed_tools_at(&tools_root).map_err(|error| error.to_string())
}

/// Copies one selected developer tool directory into application data.
pub(crate) fn install_tool_directory(
    app: &AppHandle,
    source_path: &Path,
) -> Result<NativeToolDefinition, String> {
    let tools_root = tools_root(app).map_err(|error| error.to_string())?;
    install_tool_directory_at(&tools_root, source_path).map_err(|error| error.to_string())
}

/// Updates whether one installed developer tool is loaded at sidecar startup.
pub(crate) fn set_tool_enabled(
    app: &AppHandle,
    tool_id: &str,
    enabled: bool,
) -> Result<NativeToolDefinition, String> {
    let tools_root = tools_root(app).map_err(|error| error.to_string())?;
    set_tool_enabled_at(&tools_root, tool_id, enabled).map_err(|error| error.to_string())
}

/// Removes one copied developer tool directory from application data.
pub(crate) fn remove_installed_tool(app: &AppHandle, tool_id: &str) -> Result<(), String> {
    let tools_root = tools_root(app).map_err(|error| error.to_string())?;
    remove_installed_tool_at(&tools_root, tool_id).map_err(|error| error.to_string())
}

fn tools_root(app: &AppHandle) -> Result<PathBuf, ToolDirectoryError> {
    Ok(app.path().app_data_dir()?.join(TOOLS_DIRECTORY))
}

fn list_installed_tools_at(
    tools_root: &Path,
) -> Result<Vec<NativeToolDefinition>, ToolDirectoryError> {
    let registry = load_registry(tools_root)?;
    let mut tools = registry
        .tools
        .iter()
        .map(|entry| definition_for_entry(tools_root, entry))
        .collect::<Result<Vec<_>, _>>()?;
    tools.sort_by(|left, right| {
        left.display_name
            .cmp(&right.display_name)
            .then_with(|| left.id.cmp(&right.id))
    });
    Ok(tools)
}

fn install_tool_directory_at(
    tools_root: &Path,
    source_path: &Path,
) -> Result<NativeToolDefinition, ToolDirectoryError> {
    let source_path = source_path
        .canonicalize()
        .map_err(|_| ToolDirectoryError::InvalidDirectory)?;
    if !source_path.is_dir() {
        return Err(ToolDirectoryError::InvalidDirectory);
    }
    let manifest = read_manifest(&source_path)?;
    let id = generate_tool_id()?;

    fs::create_dir_all(tools_root)?;
    let destination = tools_root.join(&id);
    if let Err(error) = copy_directory(&source_path, &destination) {
        let _ = fs::remove_dir_all(&destination);
        return Err(error);
    }

    let mut registry = load_registry(tools_root)?;
    registry.tools.push(PersistedToolEntry {
        id: id.clone(),
        enabled: true,
    });
    if let Err(error) = persist_registry(tools_root, &registry) {
        let _ = fs::remove_dir_all(&destination);
        return Err(error);
    }

    Ok(NativeToolDefinition {
        id,
        display_name: manifest.display_name,
        entrypoint: manifest.entrypoint,
        enabled: true,
    })
}

fn set_tool_enabled_at(
    tools_root: &Path,
    tool_id: &str,
    enabled: bool,
) -> Result<NativeToolDefinition, ToolDirectoryError> {
    let mut registry = load_registry(tools_root)?;
    let entry = registry
        .tools
        .iter_mut()
        .find(|entry| entry.id == tool_id)
        .ok_or(ToolDirectoryError::ToolNotFound)?;
    entry.enabled = enabled;
    persist_registry(tools_root, &registry)?;

    let entry = registry
        .tools
        .iter()
        .find(|entry| entry.id == tool_id)
        .ok_or(ToolDirectoryError::ToolNotFound)?;
    definition_for_entry(tools_root, entry)
}

fn remove_installed_tool_at(tools_root: &Path, tool_id: &str) -> Result<(), ToolDirectoryError> {
    let mut registry = load_registry(tools_root)?;
    let previous_length = registry.tools.len();
    registry.tools.retain(|entry| entry.id != tool_id);
    if registry.tools.len() == previous_length {
        return Err(ToolDirectoryError::ToolNotFound);
    }

    let directory = tools_root.join(tool_id);
    if directory.is_dir() {
        fs::remove_dir_all(directory)?;
    }
    persist_registry(tools_root, &registry)
}

fn definition_for_entry(
    tools_root: &Path,
    entry: &PersistedToolEntry,
) -> Result<NativeToolDefinition, ToolDirectoryError> {
    let manifest = read_manifest(&tools_root.join(&entry.id))?;
    Ok(NativeToolDefinition {
        id: entry.id.clone(),
        display_name: manifest.display_name,
        entrypoint: manifest.entrypoint,
        enabled: entry.enabled,
    })
}

fn load_registry(tools_root: &Path) -> Result<PersistedToolRegistry, ToolDirectoryError> {
    let path = tools_root.join(TOOL_REGISTRY_FILE);
    if !path.is_file() {
        return Ok(PersistedToolRegistry::default());
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn persist_registry(
    tools_root: &Path,
    registry: &PersistedToolRegistry,
) -> Result<(), ToolDirectoryError> {
    fs::create_dir_all(tools_root)?;
    fs::write(
        tools_root.join(TOOL_REGISTRY_FILE),
        serde_json::to_vec_pretty(registry)?,
    )?;
    Ok(())
}

fn read_manifest(tool_directory: &Path) -> Result<ToolManifest, ToolDirectoryError> {
    let path = tool_directory.join(TOOL_MANIFEST_FILE);
    if !path.is_file() {
        return Err(ToolDirectoryError::MissingManifest);
    }
    let mut manifest: ToolManifest = serde_json::from_slice(&fs::read(path)?)?;
    manifest.display_name = manifest.display_name.trim().to_owned();
    manifest.entrypoint = manifest.entrypoint.trim().to_owned();
    if manifest.display_name.is_empty() {
        return Err(ToolDirectoryError::InvalidDisplayName);
    }
    let Some((module_name, factory_name)) = manifest.entrypoint.split_once(':') else {
        return Err(ToolDirectoryError::InvalidEntrypoint);
    };
    if module_name.is_empty() || factory_name.is_empty() || factory_name.contains(':') {
        return Err(ToolDirectoryError::InvalidEntrypoint);
    }
    Ok(manifest)
}

fn copy_directory(source: &Path, destination: &Path) -> Result<(), ToolDirectoryError> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            copy_directory(&source_path, &destination_path)?;
        } else if file_type.is_file() {
            fs::copy(source_path, destination_path)?;
        }
    }
    Ok(())
}

fn generate_tool_id() -> Result<String, ToolDirectoryError> {
    let mut bytes = [0_u8; 16];
    getrandom::fill(&mut bytes)
        .map_err(|error| ToolDirectoryError::Identifier(error.to_string()))?;

    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut id = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        id.push(HEX[(byte >> 4) as usize] as char);
        id.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(id)
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temporary_directory(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "xtalk-desktop-{name}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("temporary directory must be writable");
        path
    }

    fn write_tool_source(root: &Path) {
        fs::write(
            root.join(TOOL_MANIFEST_FILE),
            r#"{"display_name":"Timer","entrypoint":"timer_tool:create_tools"}"#,
        )
        .expect("manifest must be writable");
        fs::write(
            root.join("timer_tool.py"),
            "def create_tools():\n    return []\n",
        )
        .expect("tool source must be writable");
    }

    #[test]
    fn installs_lists_toggles_and_removes_a_tool_directory() {
        let root = temporary_directory("tool-registry");
        let source = root.join("source");
        let tools_root = root.join("app-data-tools");
        fs::create_dir_all(&source).expect("tool source must be created");
        write_tool_source(&source);

        let installed =
            install_tool_directory_at(&tools_root, &source).expect("tool directory must install");
        assert_eq!(installed.display_name, "Timer");
        assert!(installed.enabled);
        assert!(tools_root
            .join(&installed.id)
            .join("timer_tool.py")
            .is_file());

        let listed = list_installed_tools_at(&tools_root).expect("installed tools must list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].entrypoint, "timer_tool:create_tools");

        let disabled =
            set_tool_enabled_at(&tools_root, &installed.id, false).expect("tool state must update");
        assert!(!disabled.enabled);

        remove_installed_tool_at(&tools_root, &installed.id)
            .expect("installed tool must be removed");
        assert!(list_installed_tools_at(&tools_root)
            .expect("empty registry must list")
            .is_empty());
        fs::remove_dir_all(root).expect("temporary directory must be removable");
    }

    #[test]
    fn rejects_manifest_fields_outside_the_minimal_contract() {
        let root = temporary_directory("tool-manifest");
        fs::write(
            root.join(TOOL_MANIFEST_FILE),
            r#"{"display_name":"Timer","entrypoint":"timer:create_tools","version":"1"}"#,
        )
        .expect("manifest must be writable");

        assert!(matches!(
            read_manifest(&root),
            Err(ToolDirectoryError::Json(_))
        ));
        fs::remove_dir_all(root).expect("temporary directory must be removable");
    }
}
