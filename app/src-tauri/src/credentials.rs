//! Cross-platform system credential storage and sidecar environment binding.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
};

use serde::{Deserialize, Serialize};
use tauri::{async_runtime::spawn_blocking, path::BaseDirectory, AppHandle, Manager};
use thiserror::Error;

use crate::tools;

const CREDENTIALS_RESOURCE: &str = "credentials.json";
const CREDENTIAL_REGISTRY_VERSION: u16 = 1;
const SYSTEM_CREDENTIAL_SERVICE: &str = "com.xtalk.desktop.credentials";
const MAX_CREDENTIAL_BYTES: usize = 16 * 1024;

/// One credential status exposed to the trusted desktop WebView.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct NativeCredentialDefinition {
    /// Stable App-owned credential identifier.
    pub(crate) id: String,
    /// Localized human-readable service name.
    pub(crate) display_name: CredentialDisplayName,
    /// Whether an environment variable or system credential is available.
    pub(crate) configured: bool,
    /// Active source selected by the environment-first resolver.
    pub(crate) source: CredentialSource,
    /// Whether the native system credential store could be accessed.
    pub(crate) storage_available: bool,
}

/// Localized or plain display name declared by the App credential registry.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub(crate) enum CredentialDisplayName {
    /// One name shared by every interface language.
    Text(String),
    /// Names keyed by language code.
    Localized(BTreeMap<String, String>),
}

/// Source selected for one credential without exposing its secret.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum CredentialSource {
    /// A supported process environment variable is set.
    Environment,
    /// The platform credential manager contains a saved secret.
    System,
    /// No usable credential is configured.
    Missing,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CredentialRegistry {
    version: u16,
    credentials: Vec<CredentialDefinition>,
    bindings: Vec<CredentialBinding>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CredentialDefinition {
    id: String,
    display_name: CredentialDisplayName,
    environment: Vec<String>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CredentialBinding {
    tool_id: String,
    credential_id: String,
    inject_environment: String,
}

#[derive(Debug, Error)]
pub(crate) enum CredentialError {
    #[error("failed to access an application path: {0}")]
    Tauri(#[from] tauri::Error),
    #[error("credential registry I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("credential registry is not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("the packaged credential registry is invalid")]
    InvalidRegistry,
    #[error("the requested credential is not registered")]
    CredentialNotFound,
    #[error("credential values must be non-empty and at most 16 KiB")]
    InvalidCredentialValue,
    #[error("the system credential store is unavailable")]
    StoreUnavailable,
    #[error("the selected tool requires a configured service credential")]
    RequiredCredentialMissing,
    #[error("a credential operation could not be scheduled")]
    OperationUnavailable,
    #[error("failed to update a dependent tool: {0}")]
    Tool(String),
}

trait CredentialStore: Send + Sync + 'static {
    fn read(&self, credential_id: &str) -> Result<Option<String>, CredentialError>;
    fn save(&self, credential_id: &str, secret: &str) -> Result<(), CredentialError>;
    fn delete(&self, credential_id: &str) -> Result<(), CredentialError>;
}

#[derive(Clone, Copy)]
struct SystemCredentialStore;

#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
impl SystemCredentialStore {
    fn entry(credential_id: &str) -> Result<keyring::Entry, CredentialError> {
        keyring::Entry::new(SYSTEM_CREDENTIAL_SERVICE, credential_id)
            .map_err(|_| CredentialError::StoreUnavailable)
    }
}

#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
impl CredentialStore for SystemCredentialStore {
    fn read(&self, credential_id: &str) -> Result<Option<String>, CredentialError> {
        match Self::entry(credential_id)?.get_password() {
            Ok(secret) if secret.trim().is_empty() => Ok(None),
            Ok(secret) => Ok(Some(secret)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(_) => Err(CredentialError::StoreUnavailable),
        }
    }

    fn save(&self, credential_id: &str, secret: &str) -> Result<(), CredentialError> {
        Self::entry(credential_id)?
            .set_password(secret)
            .map_err(|_| CredentialError::StoreUnavailable)
    }

    fn delete(&self, credential_id: &str) -> Result<(), CredentialError> {
        match Self::entry(credential_id)?.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(_) => Err(CredentialError::StoreUnavailable),
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
impl CredentialStore for SystemCredentialStore {
    fn read(&self, _credential_id: &str) -> Result<Option<String>, CredentialError> {
        Err(CredentialError::StoreUnavailable)
    }

    fn save(&self, _credential_id: &str, _secret: &str) -> Result<(), CredentialError> {
        Err(CredentialError::StoreUnavailable)
    }

    fn delete(&self, _credential_id: &str) -> Result<(), CredentialError> {
        Err(CredentialError::StoreUnavailable)
    }
}

/// Lists registered credentials without returning any secret values.
pub(crate) async fn list_credentials(
    app: &AppHandle,
) -> Result<Vec<NativeCredentialDefinition>, CredentialError> {
    let registry = load_registry(app)?;
    spawn_blocking(move || {
        list_credentials_with(&registry, &SystemCredentialStore, &environment_value)
    })
    .await
    .map_err(|_| CredentialError::OperationUnavailable)?
}

/// Saves one validated secret in the platform credential manager.
pub(crate) async fn save_credential(
    app: &AppHandle,
    credential_id: String,
    secret: String,
) -> Result<NativeCredentialDefinition, CredentialError> {
    let registry = load_registry(app)?;
    let normalized = normalize_secret(secret)?;
    let definition = registered_credential(&registry, &credential_id)?.clone();
    let returned_definition = definition.clone();
    spawn_blocking(move || {
        SystemCredentialStore.save(&credential_id, &normalized)?;
        Ok(status_for_definition(
            &returned_definition,
            &SystemCredentialStore,
            &environment_value,
        ))
    })
    .await
    .map_err(|_| CredentialError::OperationUnavailable)?
}

/// Removes a system credential and disables dependent tools when no environment key remains.
pub(crate) async fn delete_credential(
    app: &AppHandle,
    credential_id: String,
) -> Result<NativeCredentialDefinition, CredentialError> {
    let registry = load_registry(app)?;
    let definition = registered_credential(&registry, &credential_id)?.clone();
    let returned_definition = definition.clone();
    let stored_credential_id = credential_id.clone();
    let status = spawn_blocking(move || -> Result<_, CredentialError> {
        SystemCredentialStore.delete(&stored_credential_id)?;
        Ok(status_for_definition(
            &returned_definition,
            &SystemCredentialStore,
            &environment_value,
        ))
    })
    .await
    .map_err(|_| CredentialError::OperationUnavailable)??;

    if status.source == CredentialSource::Missing {
        for binding in registry
            .bindings
            .iter()
            .filter(|binding| binding.credential_id == credential_id)
        {
            tools::set_tool_enabled(app, &binding.tool_id, false).map_err(CredentialError::Tool)?;
        }
    }
    Ok(status)
}

/// Ensures every credential bound to a tool is available before enabling it.
pub(crate) async fn ensure_tool_can_enable(
    app: &AppHandle,
    tool_id: &str,
) -> Result<(), CredentialError> {
    let registry = load_registry(app)?;
    let definitions = registry
        .bindings
        .iter()
        .filter(|binding| binding.tool_id == tool_id)
        .map(|binding| registered_credential(&registry, &binding.credential_id).cloned())
        .collect::<Result<Vec<_>, _>>()?;
    spawn_blocking(move || {
        for definition in definitions {
            let status =
                status_for_definition(&definition, &SystemCredentialStore, &environment_value);
            if status.source == CredentialSource::Missing {
                return Err(CredentialError::RequiredCredentialMissing);
            }
        }
        Ok(())
    })
    .await
    .map_err(|_| CredentialError::OperationUnavailable)?
}

/// Resolves secrets needed by enabled tools for one sidecar child process.
pub(crate) async fn sidecar_environment(
    app: &AppHandle,
) -> Result<BTreeMap<String, String>, CredentialError> {
    let registry = load_registry(app)?;
    let enabled_bindings = registry
        .bindings
        .iter()
        .map(|binding| {
            tools::is_tool_enabled(app, &binding.tool_id)
                .map(|enabled| enabled.then(|| binding.clone()))
                .map_err(CredentialError::Tool)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    spawn_blocking(move || {
        let mut resolved = BTreeMap::new();
        for binding in enabled_bindings {
            let definition = registered_credential(&registry, &binding.credential_id)?;
            let secret = resolve_secret(definition, &SystemCredentialStore, &environment_value)?
                .ok_or(CredentialError::RequiredCredentialMissing)?;
            resolved.insert(binding.inject_environment, secret);
        }
        Ok(resolved)
    })
    .await
    .map_err(|_| CredentialError::OperationUnavailable)?
}

fn load_registry(app: &AppHandle) -> Result<CredentialRegistry, CredentialError> {
    let path = app
        .path()
        .resolve(CREDENTIALS_RESOURCE, BaseDirectory::Resource)?;
    let registry: CredentialRegistry = serde_json::from_slice(&std::fs::read(path)?)?;
    validate_registry(&registry)?;
    Ok(registry)
}

fn validate_registry(registry: &CredentialRegistry) -> Result<(), CredentialError> {
    if registry.version != CREDENTIAL_REGISTRY_VERSION || registry.credentials.is_empty() {
        return Err(CredentialError::InvalidRegistry);
    }
    let mut credential_ids = BTreeMap::new();
    for definition in &registry.credentials {
        if !is_safe_identifier(&definition.id)
            || !valid_display_name(&definition.display_name)
            || definition.environment.is_empty()
            || definition
                .environment
                .iter()
                .any(|name| !is_environment_name(name))
            || credential_ids.insert(definition.id.as_str(), ()).is_some()
        {
            return Err(CredentialError::InvalidRegistry);
        }
    }
    let mut bindings = BTreeSet::new();
    let mut injected_names = BTreeMap::new();
    for binding in &registry.bindings {
        if !binding.tool_id.starts_with("builtin:")
            || !credential_ids.contains_key(binding.credential_id.as_str())
            || !is_environment_name(&binding.inject_environment)
            || !bindings.insert((
                binding.tool_id.as_str(),
                binding.credential_id.as_str(),
                binding.inject_environment.as_str(),
            ))
            || injected_names
                .insert(
                    binding.inject_environment.as_str(),
                    binding.credential_id.as_str(),
                )
                .is_some_and(|previous| previous != binding.credential_id.as_str())
        {
            return Err(CredentialError::InvalidRegistry);
        }
    }
    Ok(())
}

fn list_credentials_with(
    registry: &CredentialRegistry,
    store: &dyn CredentialStore,
    environment: &dyn Fn(&str) -> Option<String>,
) -> Result<Vec<NativeCredentialDefinition>, CredentialError> {
    Ok(registry
        .credentials
        .iter()
        .map(|definition| status_for_definition(definition, store, environment))
        .collect())
}

fn status_for_definition(
    definition: &CredentialDefinition,
    store: &dyn CredentialStore,
    environment: &dyn Fn(&str) -> Option<String>,
) -> NativeCredentialDefinition {
    let environment_configured = definition
        .environment
        .iter()
        .any(|name| environment(name).is_some());
    let stored = store.read(&definition.id);
    let storage_available = stored.is_ok();
    let source = if environment_configured {
        CredentialSource::Environment
    } else if stored.as_ref().is_ok_and(|secret| secret.is_some()) {
        CredentialSource::System
    } else {
        CredentialSource::Missing
    };
    NativeCredentialDefinition {
        id: definition.id.clone(),
        display_name: definition.display_name.clone(),
        configured: source != CredentialSource::Missing,
        source,
        storage_available,
    }
}

fn resolve_secret(
    definition: &CredentialDefinition,
    store: &dyn CredentialStore,
    environment: &dyn Fn(&str) -> Option<String>,
) -> Result<Option<String>, CredentialError> {
    for name in &definition.environment {
        if let Some(secret) = environment(name) {
            return Ok(Some(secret));
        }
    }
    store.read(&definition.id)
}

fn registered_credential<'a>(
    registry: &'a CredentialRegistry,
    credential_id: &str,
) -> Result<&'a CredentialDefinition, CredentialError> {
    registry
        .credentials
        .iter()
        .find(|definition| definition.id == credential_id)
        .ok_or(CredentialError::CredentialNotFound)
}

fn normalize_secret(secret: String) -> Result<String, CredentialError> {
    let trimmed = secret.trim();
    if trimmed.is_empty() || trimmed.len() > MAX_CREDENTIAL_BYTES {
        return Err(CredentialError::InvalidCredentialValue);
    }
    Ok(trimmed.to_owned())
}

fn environment_value(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .and_then(|value| normalize_secret(value).ok())
}

fn valid_display_name(display_name: &CredentialDisplayName) -> bool {
    match display_name {
        CredentialDisplayName::Text(value) => !value.trim().is_empty(),
        CredentialDisplayName::Localized(values) => {
            !values.is_empty()
                && values.iter().all(|(language, value)| {
                    !language.trim().is_empty() && !value.trim().is_empty()
                })
        }
    }
}

fn is_safe_identifier(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
}

fn is_environment_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Mutex};

    use super::{
        list_credentials_with, normalize_secret, validate_registry, CredentialBinding,
        CredentialDefinition, CredentialDisplayName, CredentialError, CredentialRegistry,
        CredentialSource, CredentialStore,
    };

    #[derive(Default)]
    struct FakeStore {
        values: Mutex<BTreeMap<String, String>>,
        unavailable: bool,
    }

    impl CredentialStore for FakeStore {
        fn read(&self, credential_id: &str) -> Result<Option<String>, CredentialError> {
            if self.unavailable {
                return Err(CredentialError::StoreUnavailable);
            }
            Ok(self
                .values
                .lock()
                .expect("fake store must lock")
                .get(credential_id)
                .cloned())
        }

        fn save(&self, credential_id: &str, secret: &str) -> Result<(), CredentialError> {
            self.values
                .lock()
                .expect("fake store must lock")
                .insert(credential_id.to_owned(), secret.to_owned());
            Ok(())
        }

        fn delete(&self, credential_id: &str) -> Result<(), CredentialError> {
            self.values
                .lock()
                .expect("fake store must lock")
                .remove(credential_id);
            Ok(())
        }
    }

    fn registry() -> CredentialRegistry {
        CredentialRegistry {
            version: 1,
            credentials: vec![CredentialDefinition {
                id: "serper".to_owned(),
                display_name: CredentialDisplayName::Text("Serper".to_owned()),
                environment: vec!["SERPER_API_KEY".to_owned()],
            }],
            bindings: vec![CredentialBinding {
                tool_id: "builtin:web_search".to_owned(),
                credential_id: "serper".to_owned(),
                inject_environment: "SERPER_API_KEY".to_owned(),
            }],
        }
    }

    #[test]
    fn environment_credentials_take_precedence_over_system_storage() {
        let store = FakeStore::default();
        store
            .save("serper", "system-secret")
            .expect("fake secret must save");
        let statuses = list_credentials_with(&registry(), &store, &|name| {
            (name == "SERPER_API_KEY").then(|| "environment-secret".to_owned())
        })
        .expect("credential statuses must resolve");

        assert_eq!(statuses[0].source, CredentialSource::Environment);
        assert!(statuses[0].configured);
        assert!(statuses[0].storage_available);
    }

    #[test]
    fn unavailable_system_storage_is_reported_without_exposing_an_error() {
        let statuses = list_credentials_with(
            &registry(),
            &FakeStore {
                unavailable: true,
                ..FakeStore::default()
            },
            &|_| None,
        )
        .expect("credential statuses must resolve");

        assert_eq!(statuses[0].source, CredentialSource::Missing);
        assert!(!statuses[0].configured);
        assert!(!statuses[0].storage_available);
    }

    #[test]
    fn validates_registry_and_secret_boundaries() {
        assert!(validate_registry(&registry()).is_ok());
        assert!(normalize_secret(" secret ".to_owned()).is_ok());
        assert!(matches!(
            normalize_secret("  ".to_owned()),
            Err(CredentialError::InvalidCredentialValue)
        ));

        let mut invalid = registry();
        invalid.bindings[0].credential_id = "missing".to_owned();
        assert!(matches!(
            validate_registry(&invalid),
            Err(CredentialError::InvalidRegistry)
        ));
    }
}
