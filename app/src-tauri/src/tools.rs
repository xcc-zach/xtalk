//! Unified discovery and state for built-in and user-installed tool directories.

use std::{
    collections::BTreeMap,
    fs,
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use tauri::{path::BaseDirectory, AppHandle, Manager};
use thiserror::Error;

const TOOL_MANIFEST_FILE: &str = "xtalk_tool.json";
const TOOL_REGISTRY_FILE: &str = "registry.json";
const BUILTIN_TOOL_CATALOG_FILE: &str = "builtin_tools.json";
const TOOL_PREFERENCES_FILE: &str = "tool_preferences.json";
const BUILTIN_TOOLS_RESOURCE: &str = "tools";
const TOOLS_DIRECTORY: &str = "tools";
const BUILTIN_ID_PREFIX: &str = "builtin:";
const MAX_UI_ENTRYPOINT_BYTES: u64 = 1024 * 1024;
const DEFAULT_UI_UPDATE_EVERY_SECONDS: f64 = 1.0;
const MIN_UI_UPDATE_EVERY_SECONDS: f64 = 0.1;
const MAX_UI_UPDATE_EVERY_SECONDS: f64 = 3600.0;

/// Unified built-in or user tool metadata exposed to the trusted WebView.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct NativeToolDefinition {
    id: String,
    origin: ToolOrigin,
    can_delete: bool,
    display_name: ToolDisplayName,
    entrypoint: String,
    ui: Option<ToolUiConfig>,
    enabled: bool,
}

/// App-owned source of one tool definition.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum ToolOrigin {
    Builtin,
    User,
}

/// Self-contained HTML source for one installed tool UI.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct NativeToolUiSource {
    tool_id: String,
    source: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum ToolDisplayName {
    Text(String),
    Localized(BTreeMap<String, String>),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ToolUiConfig {
    entrypoint: String,
    #[serde(default = "default_ui_update_every_seconds")]
    update_every_s: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolManifest {
    display_name: ToolDisplayName,
    entrypoint: String,
    ui: Option<ToolUiConfig>,
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuiltinToolCatalog {
    version: u16,
    tools: Vec<BuiltinToolEntry>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct BuiltinToolEntry {
    id: String,
    path: String,
    enabled_by_default: bool,
}

#[derive(Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PersistedToolPreferences {
    version: u16,
    builtin: BTreeMap<String, PersistedBuiltinPreference>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PersistedBuiltinPreference {
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
    #[error("tool display_name language dictionary must contain non-empty names")]
    InvalidLocalizedDisplayName,
    #[error("tool entrypoint must use the module:factory format")]
    InvalidEntrypoint,
    #[error("tool ui.entrypoint must name a safe self-contained HTML file")]
    InvalidUiEntrypoint,
    #[error("tool ui.update_every_s must be -1 or between 0.1 and 3600 seconds")]
    InvalidUiUpdateInterval,
    #[error("tool UI entrypoint exceeds the one MiB size limit")]
    UiEntrypointTooLarge,
    #[error("tool UI entrypoint is not valid UTF-8")]
    InvalidUiEncoding,
    #[error("the selected tool is not installed")]
    ToolNotFound,
    #[error("built-in tools cannot be deleted")]
    BuiltinToolImmutable,
    #[error("built-in tool catalog version is not supported")]
    UnsupportedBuiltinCatalog,
    #[error("tool preferences version is not supported")]
    UnsupportedToolPreferences,
    #[error("built-in tool identifiers and paths must be safe unique names")]
    InvalidBuiltinCatalog,
    #[error("installed tool registry contains an unsafe or duplicate identifier")]
    InvalidToolRegistry,
    #[error("could not generate an installed tool identifier: {0}")]
    Identifier(String),
}

/// Lists built-in and user-installed tools available to the App.
pub(crate) fn list_installed_tools(app: &AppHandle) -> Result<Vec<NativeToolDefinition>, String> {
    let tools_root = tools_root(app).map_err(|error| error.to_string())?;
    let builtin_tools_root = builtin_tools_root(app).map_err(|error| error.to_string())?;
    let preferences_path = tool_preferences_path(app).map_err(|error| error.to_string())?;
    list_installed_tools_at(&tools_root, &builtin_tools_root, &preferences_path)
        .map_err(|error| error.to_string())
}

/// Copies one selected user tool directory into application data.
pub(crate) fn install_tool_directory(
    app: &AppHandle,
    source_path: &Path,
) -> Result<NativeToolDefinition, String> {
    let tools_root = tools_root(app).map_err(|error| error.to_string())?;
    install_tool_directory_at(&tools_root, source_path).map_err(|error| error.to_string())
}

/// Updates whether one built-in or user tool is loaded at sidecar startup.
pub(crate) fn set_tool_enabled(
    app: &AppHandle,
    tool_id: &str,
    enabled: bool,
) -> Result<NativeToolDefinition, String> {
    let tools_root = tools_root(app).map_err(|error| error.to_string())?;
    let builtin_tools_root = builtin_tools_root(app).map_err(|error| error.to_string())?;
    let preferences_path = tool_preferences_path(app).map_err(|error| error.to_string())?;
    set_tool_enabled_at(
        &tools_root,
        &builtin_tools_root,
        &preferences_path,
        tool_id,
        enabled,
    )
    .map_err(|error| error.to_string())
}

/// Removes one copied user tool directory from application data.
pub(crate) fn remove_installed_tool(app: &AppHandle, tool_id: &str) -> Result<(), String> {
    if tool_id.starts_with(BUILTIN_ID_PREFIX) {
        return Err(ToolDirectoryError::BuiltinToolImmutable.to_string());
    }
    let tools_root = tools_root(app).map_err(|error| error.to_string())?;
    remove_installed_tool_at(&tools_root, tool_id).map_err(|error| error.to_string())
}

/// Reads one built-in or user tool's self-contained UI entrypoint.
pub(crate) fn read_tool_ui_source(
    app: &AppHandle,
    tool_id: &str,
) -> Result<NativeToolUiSource, String> {
    let tools_root = tools_root(app).map_err(|error| error.to_string())?;
    let builtin_tools_root = builtin_tools_root(app).map_err(|error| error.to_string())?;
    read_tool_ui_source_at(&tools_root, &builtin_tools_root, tool_id)
        .map_err(|error| error.to_string())
}

fn tools_root(app: &AppHandle) -> Result<PathBuf, ToolDirectoryError> {
    Ok(app.path().app_data_dir()?.join(TOOLS_DIRECTORY))
}

fn builtin_tools_root(app: &AppHandle) -> Result<PathBuf, ToolDirectoryError> {
    Ok(app
        .path()
        .resolve(BUILTIN_TOOLS_RESOURCE, BaseDirectory::Resource)?)
}

fn tool_preferences_path(app: &AppHandle) -> Result<PathBuf, ToolDirectoryError> {
    Ok(app.path().app_data_dir()?.join(TOOL_PREFERENCES_FILE))
}

fn list_installed_tools_at(
    tools_root: &Path,
    builtin_tools_root: &Path,
    preferences_path: &Path,
) -> Result<Vec<NativeToolDefinition>, ToolDirectoryError> {
    let registry = load_registry(tools_root)?;
    let mut tools = list_builtin_tools_at(builtin_tools_root, preferences_path)?;
    tools.extend(
        registry
            .tools
            .iter()
            .map(|entry| definition_for_entry(tools_root, entry))
            .collect::<Result<Vec<_>, _>>()?,
    );
    tools.sort_by(|left, right| {
        display_name_sort_key(&left.display_name)
            .cmp(&display_name_sort_key(&right.display_name))
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
        origin: ToolOrigin::User,
        can_delete: true,
        display_name: manifest.display_name,
        entrypoint: manifest.entrypoint,
        ui: manifest.ui,
        enabled: true,
    })
}

fn set_tool_enabled_at(
    tools_root: &Path,
    builtin_tools_root: &Path,
    preferences_path: &Path,
    tool_id: &str,
    enabled: bool,
) -> Result<NativeToolDefinition, ToolDirectoryError> {
    if let Some(identifier) = tool_id.strip_prefix(BUILTIN_ID_PREFIX) {
        let catalog = load_builtin_catalog(builtin_tools_root)?;
        let entry = catalog
            .tools
            .iter()
            .find(|entry| entry.id == identifier)
            .ok_or(ToolDirectoryError::ToolNotFound)?;
        let mut preferences = load_tool_preferences(preferences_path)?;
        preferences.builtin.insert(
            identifier.to_owned(),
            PersistedBuiltinPreference { enabled },
        );
        persist_tool_preferences(preferences_path, &preferences)?;
        return definition_for_builtin_entry(builtin_tools_root, entry, enabled);
    }

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
    if tool_id.starts_with(BUILTIN_ID_PREFIX) {
        return Err(ToolDirectoryError::BuiltinToolImmutable);
    }
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
        origin: ToolOrigin::User,
        can_delete: true,
        display_name: manifest.display_name,
        entrypoint: manifest.entrypoint,
        ui: manifest.ui,
        enabled: entry.enabled,
    })
}

fn list_builtin_tools_at(
    builtin_tools_root: &Path,
    preferences_path: &Path,
) -> Result<Vec<NativeToolDefinition>, ToolDirectoryError> {
    let catalog = load_builtin_catalog(builtin_tools_root)?;
    let preferences = load_tool_preferences(preferences_path)?;
    catalog
        .tools
        .iter()
        .map(|entry| {
            let enabled = preferences
                .builtin
                .get(&entry.id)
                .map(|preference| preference.enabled)
                .unwrap_or(entry.enabled_by_default);
            definition_for_builtin_entry(builtin_tools_root, entry, enabled)
        })
        .collect()
}

fn definition_for_builtin_entry(
    builtin_tools_root: &Path,
    entry: &BuiltinToolEntry,
    enabled: bool,
) -> Result<NativeToolDefinition, ToolDirectoryError> {
    let manifest = read_manifest(&builtin_tools_root.join(&entry.path))?;
    Ok(NativeToolDefinition {
        id: format!("{BUILTIN_ID_PREFIX}{}", entry.id),
        origin: ToolOrigin::Builtin,
        can_delete: false,
        display_name: manifest.display_name,
        entrypoint: manifest.entrypoint,
        ui: manifest.ui,
        enabled,
    })
}

fn load_builtin_catalog(
    builtin_tools_root: &Path,
) -> Result<BuiltinToolCatalog, ToolDirectoryError> {
    let catalog: BuiltinToolCatalog = serde_json::from_slice(&fs::read(
        builtin_tools_root.join(BUILTIN_TOOL_CATALOG_FILE),
    )?)?;
    if catalog.version != 1 {
        return Err(ToolDirectoryError::UnsupportedBuiltinCatalog);
    }
    let mut identifiers = BTreeMap::new();
    let mut paths = BTreeMap::new();
    for entry in &catalog.tools {
        if !is_safe_name(&entry.id)
            || !is_safe_name(&entry.path)
            || identifiers.insert(entry.id.as_str(), ()).is_some()
            || paths.insert(entry.path.as_str(), ()).is_some()
        {
            return Err(ToolDirectoryError::InvalidBuiltinCatalog);
        }
    }
    Ok(catalog)
}

fn load_registry(tools_root: &Path) -> Result<PersistedToolRegistry, ToolDirectoryError> {
    let path = tools_root.join(TOOL_REGISTRY_FILE);
    if !path.is_file() {
        return Ok(PersistedToolRegistry::default());
    }
    let registry: PersistedToolRegistry = serde_json::from_slice(&fs::read(path)?)?;
    let mut identifiers = BTreeMap::new();
    if registry.tools.iter().any(|entry| {
        !is_safe_name(&entry.id) || identifiers.insert(entry.id.as_str(), ()).is_some()
    }) {
        return Err(ToolDirectoryError::InvalidToolRegistry);
    }
    Ok(registry)
}

fn load_tool_preferences(
    preferences_path: &Path,
) -> Result<PersistedToolPreferences, ToolDirectoryError> {
    if !preferences_path.is_file() {
        return Ok(PersistedToolPreferences {
            version: tool_preferences_version(),
            ..PersistedToolPreferences::default()
        });
    }
    let preferences: PersistedToolPreferences =
        serde_json::from_slice(&fs::read(preferences_path)?)?;
    if preferences.version != tool_preferences_version() {
        return Err(ToolDirectoryError::UnsupportedToolPreferences);
    }
    Ok(preferences)
}

fn persist_tool_preferences(
    preferences_path: &Path,
    preferences: &PersistedToolPreferences,
) -> Result<(), ToolDirectoryError> {
    if let Some(parent) = preferences_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(preferences_path, serde_json::to_vec_pretty(preferences)?)?;
    Ok(())
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
    normalize_display_name(&mut manifest.display_name)?;
    manifest.entrypoint = manifest.entrypoint.trim().to_owned();
    let Some((module_name, factory_name)) = manifest.entrypoint.split_once(':') else {
        return Err(ToolDirectoryError::InvalidEntrypoint);
    };
    if module_name.is_empty() || factory_name.is_empty() || factory_name.contains(':') {
        return Err(ToolDirectoryError::InvalidEntrypoint);
    }
    if let Some(ui) = manifest.ui.as_mut() {
        validate_ui_config(tool_directory, ui)?;
    }
    Ok(manifest)
}

fn normalize_display_name(display_name: &mut ToolDisplayName) -> Result<(), ToolDirectoryError> {
    match display_name {
        ToolDisplayName::Text(value) => {
            *value = value.trim().to_owned();
            if value.is_empty() {
                return Err(ToolDirectoryError::InvalidDisplayName);
            }
        }
        ToolDisplayName::Localized(values) => {
            if values.is_empty() {
                return Err(ToolDirectoryError::InvalidLocalizedDisplayName);
            }
            let mut normalized = BTreeMap::new();
            for (language, value) in std::mem::take(values) {
                let language = language.trim().to_ascii_lowercase();
                let value = value.trim().to_owned();
                if language.is_empty()
                    || value.is_empty()
                    || normalized.insert(language, value).is_some()
                {
                    return Err(ToolDirectoryError::InvalidLocalizedDisplayName);
                }
            }
            *values = normalized;
        }
    }
    Ok(())
}

fn validate_ui_config(
    tool_directory: &Path,
    ui: &mut ToolUiConfig,
) -> Result<(), ToolDirectoryError> {
    ui.entrypoint = ui.entrypoint.trim().to_owned();
    let relative = Path::new(&ui.entrypoint);
    if relative.is_absolute()
        || relative.extension().and_then(|value| value.to_str()) != Some("html")
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ToolDirectoryError::InvalidUiEntrypoint);
    }
    let entrypoint = tool_directory.join(relative);
    if !entrypoint.is_file() {
        return Err(ToolDirectoryError::InvalidUiEntrypoint);
    }
    let size = fs::metadata(entrypoint)?.len();
    if size > MAX_UI_ENTRYPOINT_BYTES {
        return Err(ToolDirectoryError::UiEntrypointTooLarge);
    }
    if ui.update_every_s != -1.0
        && (!ui.update_every_s.is_finite()
            || !(MIN_UI_UPDATE_EVERY_SECONDS..=MAX_UI_UPDATE_EVERY_SECONDS)
                .contains(&ui.update_every_s))
    {
        return Err(ToolDirectoryError::InvalidUiUpdateInterval);
    }
    Ok(())
}

fn read_tool_ui_source_at(
    tools_root: &Path,
    builtin_tools_root: &Path,
    tool_id: &str,
) -> Result<NativeToolUiSource, ToolDirectoryError> {
    if let Some(identifier) = tool_id.strip_prefix(BUILTIN_ID_PREFIX) {
        let catalog = load_builtin_catalog(builtin_tools_root)?;
        let entry = catalog
            .tools
            .iter()
            .find(|entry| entry.id == identifier)
            .ok_or(ToolDirectoryError::ToolNotFound)?;
        return read_tool_ui_source_from_directory(&builtin_tools_root.join(&entry.path), tool_id);
    }

    let registry = load_registry(tools_root)?;
    if !registry.tools.iter().any(|entry| entry.id == tool_id) {
        return Err(ToolDirectoryError::ToolNotFound);
    }
    let tool_directory = tools_root.join(tool_id);
    read_tool_ui_source_from_directory(&tool_directory, tool_id)
}

fn read_tool_ui_source_from_directory(
    tool_directory: &Path,
    tool_id: &str,
) -> Result<NativeToolUiSource, ToolDirectoryError> {
    let manifest = read_manifest(&tool_directory)?;
    let ui = manifest.ui.ok_or(ToolDirectoryError::InvalidUiEntrypoint)?;
    let bytes = fs::read(tool_directory.join(ui.entrypoint))?;
    if bytes.len() as u64 > MAX_UI_ENTRYPOINT_BYTES {
        return Err(ToolDirectoryError::UiEntrypointTooLarge);
    }
    let source = String::from_utf8(bytes).map_err(|_| ToolDirectoryError::InvalidUiEncoding)?;
    Ok(NativeToolUiSource {
        tool_id: tool_id.to_owned(),
        source,
    })
}

fn display_name_sort_key(display_name: &ToolDisplayName) -> &str {
    match display_name {
        ToolDisplayName::Text(value) => value,
        ToolDisplayName::Localized(values) => values
            .get("en")
            .or_else(|| values.get("zh"))
            .or_else(|| values.values().next())
            .map(String::as_str)
            .unwrap_or_default(),
    }
}

fn is_safe_name(value: &str) -> bool {
    !value.is_empty()
        && Path::new(value)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
        && !value.contains(['/', '\\', ':'])
}

const fn tool_preferences_version() -> u16 {
    1
}

const fn default_ui_update_every_seconds() -> f64 {
    DEFAULT_UI_UPDATE_EVERY_SECONDS
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

    fn write_tool_ui_source(root: &Path, update_every_s: Option<f64>) {
        fs::create_dir_all(root.join("ui")).expect("UI directory must be created");
        fs::write(
            root.join("ui/index.html"),
            "<!doctype html><title>Timer</title>",
        )
        .expect("UI source must be writable");
        let interval = update_every_s
            .map(|value| format!(r#","update_every_s":{value}"#))
            .unwrap_or_default();
        fs::write(
            root.join(TOOL_MANIFEST_FILE),
            format!(
                r#"{{"display_name":{{"zh":"计时器","en":"Timer"}},"entrypoint":"timer_tool:create_tools","ui":{{"entrypoint":"ui/index.html"{interval}}}}}"#
            ),
        )
        .expect("UI manifest must be writable");
        fs::write(
            root.join("timer_tool.py"),
            "def create_tools():\n    return []\n",
        )
        .expect("tool source must be writable");
    }

    fn write_builtin_catalog(root: &Path) {
        let timer = root.join("timer");
        fs::create_dir_all(&timer).expect("built-in tool directory must be created");
        write_tool_ui_source(&timer, None);
        fs::write(
            root.join(BUILTIN_TOOL_CATALOG_FILE),
            r#"{"version":1,"tools":[{"id":"timer","path":"timer","enabled_by_default":true}]}"#,
        )
        .expect("built-in tool catalog must be writable");
    }

    #[test]
    fn installs_lists_toggles_and_removes_a_tool_directory() {
        let root = temporary_directory("tool-registry");
        let source = root.join("source");
        let tools_root = root.join("app-data-tools");
        let builtin_tools_root = root.join("builtin-tools");
        let preferences_path = root.join(TOOL_PREFERENCES_FILE);
        fs::create_dir_all(&source).expect("tool source must be created");
        fs::create_dir_all(&builtin_tools_root).expect("built-in root must be created");
        fs::write(
            builtin_tools_root.join(BUILTIN_TOOL_CATALOG_FILE),
            r#"{"version":1,"tools":[]}"#,
        )
        .expect("empty built-in catalog must be writable");
        write_tool_source(&source);

        let installed =
            install_tool_directory_at(&tools_root, &source).expect("tool directory must install");
        assert_eq!(
            installed.display_name,
            ToolDisplayName::Text("Timer".to_owned())
        );
        assert!(installed.enabled);
        assert!(tools_root
            .join(&installed.id)
            .join("timer_tool.py")
            .is_file());

        let listed = list_installed_tools_at(&tools_root, &builtin_tools_root, &preferences_path)
            .expect("installed tools must list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].entrypoint, "timer_tool:create_tools");
        assert_eq!(listed[0].origin, ToolOrigin::User);
        assert!(listed[0].can_delete);

        let disabled = set_tool_enabled_at(
            &tools_root,
            &builtin_tools_root,
            &preferences_path,
            &installed.id,
            false,
        )
        .expect("tool state must update");
        assert!(!disabled.enabled);

        remove_installed_tool_at(&tools_root, &installed.id)
            .expect("installed tool must be removed");
        assert!(
            list_installed_tools_at(&tools_root, &builtin_tools_root, &preferences_path)
                .expect("empty registry must list")
                .is_empty()
        );
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

    #[test]
    fn installs_localized_tool_ui_and_reads_its_source() {
        let root = temporary_directory("localized-tool-ui");
        let source = root.join("source");
        let tools_root = root.join("app-data-tools");
        fs::create_dir_all(&source).expect("tool source must be created");
        write_tool_ui_source(&source, None);

        let installed =
            install_tool_directory_at(&tools_root, &source).expect("tool directory must install");

        assert_eq!(
            installed.display_name,
            ToolDisplayName::Localized(BTreeMap::from([
                ("en".to_owned(), "Timer".to_owned()),
                ("zh".to_owned(), "计时器".to_owned()),
            ]))
        );
        assert_eq!(
            installed.ui,
            Some(ToolUiConfig {
                entrypoint: "ui/index.html".to_owned(),
                update_every_s: DEFAULT_UI_UPDATE_EVERY_SECONDS,
            })
        );
        let builtin_tools_root = root.join("builtin-tools");
        fs::create_dir_all(&builtin_tools_root).expect("built-in root must be created");
        fs::write(
            builtin_tools_root.join(BUILTIN_TOOL_CATALOG_FILE),
            r#"{"version":1,"tools":[]}"#,
        )
        .expect("empty built-in catalog must be writable");
        let ui = read_tool_ui_source_at(&tools_root, &builtin_tools_root, &installed.id)
            .expect("installed UI source must be readable");
        assert_eq!(ui.tool_id, installed.id);
        assert_eq!(ui.source, "<!doctype html><title>Timer</title>");

        fs::remove_dir_all(root).expect("temporary directory must be removable");
    }

    #[test]
    fn lists_toggles_and_protects_a_builtin_tool() {
        let root = temporary_directory("builtin-tool");
        let tools_root = root.join("app-data-tools");
        let builtin_tools_root = root.join("builtin-tools");
        let preferences_path = root.join(TOOL_PREFERENCES_FILE);
        write_builtin_catalog(&builtin_tools_root);

        let listed = list_installed_tools_at(&tools_root, &builtin_tools_root, &preferences_path)
            .expect("built-in tools must list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "builtin:timer");
        assert_eq!(listed[0].origin, ToolOrigin::Builtin);
        assert!(!listed[0].can_delete);
        assert!(listed[0].enabled);

        let disabled = set_tool_enabled_at(
            &tools_root,
            &builtin_tools_root,
            &preferences_path,
            "builtin:timer",
            false,
        )
        .expect("built-in preference must update");
        assert!(!disabled.enabled);
        assert!(matches!(
            remove_installed_tool_at(&tools_root, "builtin:timer"),
            Err(ToolDirectoryError::BuiltinToolImmutable)
        ));

        let relisted = list_installed_tools_at(&tools_root, &builtin_tools_root, &preferences_path)
            .expect("built-in preferences must reload");
        assert!(!relisted[0].enabled);
        let ui = read_tool_ui_source_at(&tools_root, &builtin_tools_root, "builtin:timer")
            .expect("built-in UI source must be readable");
        assert_eq!(ui.tool_id, "builtin:timer");

        fs::remove_dir_all(root).expect("temporary directory must be removable");
    }

    #[test]
    fn rejects_unsafe_ui_entrypoint_and_invalid_update_interval() {
        let root = temporary_directory("invalid-tool-ui");
        fs::create_dir_all(root.join("ui")).expect("UI directory must be created");
        fs::write(root.join("ui/index.html"), "<p>Timer</p>").expect("UI source must be writable");
        fs::write(
            root.join(TOOL_MANIFEST_FILE),
            r#"{"display_name":"Timer","entrypoint":"timer:create_tools","ui":{"entrypoint":"../index.html"}}"#,
        )
        .expect("manifest must be writable");
        assert!(matches!(
            read_manifest(&root),
            Err(ToolDirectoryError::InvalidUiEntrypoint)
        ));

        fs::write(
            root.join(TOOL_MANIFEST_FILE),
            r#"{"display_name":"Timer","entrypoint":"timer:create_tools","ui":{"entrypoint":"ui/index.html","update_every_s":0}}"#,
        )
        .expect("manifest must be writable");
        assert!(matches!(
            read_manifest(&root),
            Err(ToolDirectoryError::InvalidUiUpdateInterval)
        ));

        fs::remove_dir_all(root).expect("temporary directory must be removable");
    }
}
