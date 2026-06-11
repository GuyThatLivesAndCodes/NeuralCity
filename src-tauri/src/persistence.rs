//! Disk persistence for the entire app state.
//!
//! Metadata (networks, corpora, vocabs, training history) lives in
//! `<app_data_dir>/state.json` — small enough to load atomically.
//!
//! Model weights are stored in separate files:
//!   `<app_data_dir>/models/<network_id>.json`
//!
//! Models are NOT loaded on startup. They are loaded lazily the first time
//! training or inference is requested for a network (see `load_model`).
//! This prevents a large trained model from causing an OOM crash on launch.
//!
//! Writes to state.json are atomic (write tmp → rename). Model files are
//! written the same way.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use neuralcabin_engine::nn::Model;
use neuralcabin_engine::tokenizer::Vocabulary;
use neuralcabin_engine::transformer::TransformerModel;

use crate::models::{Corpus, Network, TrainingRun, VocabularyInfo};
use crate::server::{ServerConfig, ServerRuntime};
use crate::{AppState, VocabEntry};

pub const STATE_FILENAME: &str = "state.json";
pub const FORMAT_VERSION: u32 = 1;

/// Top-level on-disk snapshot. Intentionally excludes model weights — those
/// live in `models/<id>.json` and are loaded lazily.
///
/// Old `state.json` files that contain a `models` field have it silently
/// ignored on load (serde skips unknown fields by default), so upgrading
/// from a build that stored weights inline does not corrupt the workspace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedState {
    pub format_version: u32,
    #[serde(default)]
    pub networks: HashMap<String, Network>,
    #[serde(default)]
    pub corpora: HashMap<String, Corpus>,
    #[serde(default)]
    pub vocabs: HashMap<String, PersistedVocab>,
    #[serde(default)]
    pub training_history: HashMap<String, Vec<TrainingRun>>,
    #[serde(default)]
    pub servers: Vec<ServerConfig>,
}

impl Default for PersistedState {
    fn default() -> Self {
        Self {
            format_version: FORMAT_VERSION,
            networks: HashMap::new(),
            corpora: HashMap::new(),
            vocabs: HashMap::new(),
            training_history: HashMap::new(),
            servers: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedVocab {
    pub info: VocabularyInfo,
    pub vocab: Vocabulary,
}

// ─── Model file helpers ──────────────────────────────────────────────────────

fn model_path(data_dir: &Path, network_id: &str) -> PathBuf {
    data_dir.join("models").join(format!("{network_id}.json"))
}

/// Write a model's weights to its own file. Atomic: write tmp → rename.
pub async fn save_model(model: &Model, network_id: &str, data_dir: &Path) -> Result<(), String> {
    let json = serde_json::to_vec(model)
        .map_err(|e| format!("serialize model {network_id}: {e}"))?;
    let models_dir = data_dir.join("models");
    tokio::fs::create_dir_all(&models_dir).await
        .map_err(|e| format!("create models dir: {e}"))?;
    let path = model_path(data_dir, network_id);
    let tmp  = path.with_extension("json.tmp");
    tokio::fs::write(&tmp, &json).await
        .map_err(|e| format!("write model tmp {network_id}: {e}"))?;
    tokio::fs::rename(&tmp, &path).await
        .map_err(|e| format!("rename model file {network_id}: {e}"))?;
    Ok(())
}

/// Load a model's weights from disk. Returns `Ok(None)` if the file doesn't
/// exist (network not yet trained / weights not yet migrated).
pub async fn load_model(network_id: &str, data_dir: &Path) -> Result<Option<Model>, String> {
    let path = model_path(data_dir, network_id);
    let bytes = match tokio::fs::read(&path).await {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("read model {}: {e}", path.display())),
    };
    let model: Model = serde_json::from_slice(&bytes)
        .map_err(|e| format!("parse model {}: {e}", path.display()))?;
    Ok(Some(model))
}

/// Delete a model's weight file. Silently succeeds if the file doesn't exist.
pub async fn delete_model(network_id: &str, data_dir: &Path) -> Result<(), String> {
    let path = model_path(data_dir, network_id);
    match tokio::fs::remove_file(&path).await {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("delete model {}: {e}", path.display())),
    }
}

/// Returns true if a weight file exists for this network.
pub async fn model_file_exists(network_id: &str, data_dir: &Path) -> bool {
    tokio::fs::try_exists(model_path(data_dir, network_id)).await.unwrap_or(false)
}

// ─── Transformer file helpers ─────────────────────────────────────────────
//
// Stored in a sibling directory so a single network ID never has both a Model
// and a TransformerModel — the network's `kind` chooses which is canonical.

fn transformer_path(data_dir: &Path, network_id: &str) -> PathBuf {
    data_dir.join("transformers").join(format!("{network_id}.json"))
}

pub async fn save_transformer(t: &TransformerModel, network_id: &str, data_dir: &Path) -> Result<(), String> {
    let json = serde_json::to_vec(t)
        .map_err(|e| format!("serialize transformer {network_id}: {e}"))?;
    let dir = data_dir.join("transformers");
    tokio::fs::create_dir_all(&dir).await
        .map_err(|e| format!("create transformers dir: {e}"))?;
    let path = transformer_path(data_dir, network_id);
    let tmp = path.with_extension("json.tmp");
    tokio::fs::write(&tmp, &json).await
        .map_err(|e| format!("write transformer tmp {network_id}: {e}"))?;
    tokio::fs::rename(&tmp, &path).await
        .map_err(|e| format!("rename transformer file {network_id}: {e}"))?;
    Ok(())
}

pub async fn load_transformer(network_id: &str, data_dir: &Path) -> Result<Option<TransformerModel>, String> {
    let path = transformer_path(data_dir, network_id);
    let bytes = match tokio::fs::read(&path).await {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("read transformer {}: {e}", path.display())),
    };
    let model: TransformerModel = serde_json::from_slice(&bytes)
        .map_err(|e| format!("parse transformer {}: {e}", path.display()))?;
    Ok(Some(model))
}

pub async fn delete_transformer(network_id: &str, data_dir: &Path) -> Result<(), String> {
    let path = transformer_path(data_dir, network_id);
    match tokio::fs::remove_file(&path).await {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("delete transformer {}: {e}", path.display())),
    }
}

pub async fn transformer_file_exists(network_id: &str, data_dir: &Path) -> bool {
    tokio::fs::try_exists(transformer_path(data_dir, network_id)).await.unwrap_or(false)
}

// ─── Snapshot / save / load ──────────────────────────────────────────────────

/// Build a `PersistedState` snapshot from the live `AppState`.
/// Model weights are excluded — they are saved separately by `save_to_dir`.
pub async fn snapshot(state: &AppState) -> PersistedState {
    let networks = state.networks.read().await.clone();
    let corpora  = state.corpora.read().await.clone();

    let mut vocabs: HashMap<String, PersistedVocab> = HashMap::new();
    {
        let live = state.vocabs.read().await;
        for (k, e) in live.iter() {
            let inner = e.vocab.read().await.clone();
            vocabs.insert(k.clone(), PersistedVocab { info: e.info.clone(), vocab: inner });
        }
    }

    let training_history = state.training_history.read().await.clone();

    let servers: Vec<ServerConfig> = state.servers.read().await
        .values().map(|rt| rt.config.clone()).collect();

    PersistedState {
        format_version: FORMAT_VERSION,
        networks, corpora, vocabs, training_history, servers,
    }
}

/// Persist the full state: atomically replace `state.json` (no weights), then
/// write a separate `models/<id>.json` for every model whose weights have
/// changed since the last save.
///
/// Each model carries a cheap 64-bit content signature (`weights_signature()`).
/// We compare the in-memory signature against the one recorded when the file
/// was last written (`AppState.saved_sigs`) and skip the clone + JSON
/// serialization + disk write entirely when they match. This keeps routine
/// metadata-only saves (renaming a network, editing a corpus, changing a server
/// setting) from rewriting every large weight file that happens to be resident
/// in memory — the common cause of the "save takes forever" slowdown. The check
/// is content-addressed, so any real weight change always produces a different
/// signature and is always written; nothing is ever silently dropped.
pub async fn save_to_dir(state: &AppState, data_dir: &Path) -> Result<(), String> {
    // 1. Write the metadata snapshot.
    let persisted = snapshot(state).await;
    save_persisted(&persisted, data_dir).await?;

    // 2. Write each changed in-memory model to its own file. Non-fatal per
    //    model so one corrupt model doesn't block saving everything else.
    let models = state.models.read().await;
    for (id, model_arc) in models.iter() {
        let (sig, model) = {
            let guard = model_arc.read().await;
            let sig = guard.weights_signature();
            (sig, if needs_write(state, id, sig).await { Some(guard.clone()) } else { None })
        };
        if let Some(model) = model {
            match save_model(&model, id, data_dir).await {
                Ok(()) => { state.saved_sigs.write().await.insert(id.clone(), sig); }
                Err(e) => eprintln!("[neuralcabin] {e}"),
            }
        }
    }
    drop(models);
    let transformers = state.transformers.read().await;
    for (id, t_arc) in transformers.iter() {
        let (sig, transformer) = {
            let guard = t_arc.read().await;
            let sig = guard.weights_signature();
            (sig, if needs_write(state, id, sig).await { Some(guard.clone()) } else { None })
        };
        if let Some(t) = transformer {
            match save_transformer(&t, id, data_dir).await {
                Ok(()) => { state.saved_sigs.write().await.insert(id.clone(), sig); }
                Err(e) => eprintln!("[neuralcabin] {e}"),
            }
        }
    }
    Ok(())
}

/// True if the weight file for `id` is missing from the signature cache or its
/// recorded signature differs from `sig` — i.e. the weights changed (or were
/// never written) and the file must be (re)written.
async fn needs_write(state: &AppState, id: &str, sig: u64) -> bool {
    state.saved_sigs.read().await.get(id).copied() != Some(sig)
}

pub async fn save_persisted(persisted: &PersistedState, data_dir: &Path) -> Result<(), String> {
    let json = serde_json::to_vec_pretty(persisted)
        .map_err(|e| format!("serialize state: {e}"))?;
    tokio::fs::create_dir_all(data_dir).await
        .map_err(|e| format!("create data dir {}: {e}", data_dir.display()))?;
    let final_path = data_dir.join(STATE_FILENAME);
    let tmp_path   = data_dir.join(format!("{STATE_FILENAME}.tmp"));
    tokio::fs::write(&tmp_path, &json).await
        .map_err(|e| format!("write state tmp: {e}"))?;
    tokio::fs::rename(&tmp_path, &final_path).await
        .map_err(|e| format!("rename state file: {e}"))?;
    Ok(())
}

/// Load `state.json`. Returns `Ok(None)` for a fresh install.
/// Returns an error if the file is corrupt — we never silently wipe data.
///
/// Uses streaming deserialization (`serde_json::from_reader` via a
/// `BufReader`) so that large unknown fields — in particular the legacy
/// `models` blob that old builds wrote inline — are read and discarded in
/// small chunks without ever being fully buffered in RAM.  A 6 GB legacy
/// state.json therefore needs only ~8 KB of working memory to parse,
/// rather than 6 GB.
pub async fn load_from_dir(data_dir: &Path) -> Result<Option<PersistedState>, String> {
    let path = data_dir.join(STATE_FILENAME);

    let result: Result<Option<PersistedState>, String> =
        tokio::task::spawn_blocking(move || {
            let file = match std::fs::File::open(&path) {
                Ok(f) => f,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(e) => return Err(format!("read {}: {e}", path.display())),
            };
            let reader = std::io::BufReader::new(file);
            let persisted: PersistedState = serde_json::from_reader(reader)
                .map_err(|e| format!("parse {}: {e}", path.display()))?;
            Ok(Some(persisted))
        })
        .await
        .map_err(|e| format!("load_from_dir task panicked: {e}"))?;

    match result? {
        Some(p) if p.format_version > FORMAT_VERSION => Err(format!(
            "state.json format_version {} is newer than this build supports ({})",
            p.format_version, FORMAT_VERSION
        )),
        other => Ok(other),
    }
}

/// Populate an empty `AppState` from a loaded snapshot.
/// Models are intentionally NOT restored here — they are loaded lazily.
pub async fn apply(state: &AppState, persisted: PersistedState) {
    *state.networks.write().await = persisted.networks;
    *state.corpora.write().await  = persisted.corpora;

    let mut vocabs = state.vocabs.write().await;
    vocabs.clear();
    for (k, mut pv) in persisted.vocabs {
        pv.vocab.rebuild_index();
        vocabs.insert(k, VocabEntry {
            vocab: Arc::new(RwLock::new(pv.vocab)),
            info: pv.info,
        });
    }

    *state.training_history.write().await = persisted.training_history;

    let mut servers = state.servers.write().await;
    servers.clear();
    for cfg in persisted.servers {
        servers.insert(cfg.id.clone(), ServerRuntime::new(cfg));
    }
    // state.models is intentionally left empty — loaded on demand.
}

/// After loading, cross-check the `trained` flag on each network against
/// whether a model file actually exists. Networks whose weight file is missing
/// are marked untrained so the UI shows the correct state.
pub async fn reset_missing_model_flags(state: &AppState, data_dir: &Path) {
    let mut networks = state.networks.write().await;
    for (id, net) in networks.iter_mut() {
        if !net.trained { continue; }
        let exists = if net.kind == crate::models::kinds::TRANSFORMER {
            transformer_file_exists(id, data_dir).await
        } else {
            model_file_exists(id, data_dir).await
        };
        if !exists {
            eprintln!(
                "[neuralcabin] weights missing for '{}' ('{}'): marking untrained",
                id, net.name
            );
            net.trained = false;
        }
    }
}

/// Return the configured data directory, or None if persistence is disabled.
pub async fn data_dir(state: &AppState) -> Option<PathBuf> {
    state.data_dir.read().await.clone()
}

/// Best-effort save for background tasks that can't propagate errors.
pub async fn save_best_effort(state: &AppState) {
    let Some(dir) = data_dir(state).await else { return; };
    if let Err(e) = save_to_dir(state, &dir).await {
        eprintln!("[neuralcabin] failed to persist state: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AppState;
    use crate::models::{LayerDef, kinds};
    use chrono::Utc;
    use neuralcabin_engine::nn::{LayerSpec, Model};
    use neuralcabin_engine::activations::Activation as EngineActivation;
    use neuralcabin_engine::tokenizer::{TokenizerMode, Vocabulary, VocabularyOptions};

    fn sample_network(id: &str) -> Network {
        Network {
            id: id.to_string(),
            name: format!("net-{id}"),
            kind: kinds::FEEDFORWARD.to_string(),
            seed: 42,
            created_at: Utc::now(),
            trained: false,
            input_dim: 2,
            output_dim: 1,
            layers: vec![
                LayerDef::Linear { in_dim: 2, out_dim: 4 },
                LayerDef::Activation { activation: "relu".to_string() },
                LayerDef::Linear { in_dim: 4, out_dim: 1 },
            ],
            parameter_count: 17,
            hidden_layers: None,
            context_size: None,
            transformer: None,
            pretrained: false,
        }
    }

    fn sample_model() -> Model {
        let specs = vec![
            LayerSpec::Linear { in_dim: 2, out_dim: 4 },
            LayerSpec::Activation(EngineActivation::ReLU),
            LayerSpec::Linear { in_dim: 4, out_dim: 1 },
        ];
        Model::from_specs(2, &specs, 7)
    }

    fn sample_vocab() -> Vocabulary {
        Vocabulary::build(
            TokenizerMode::Char,
            &["hello world", "the quick brown fox"],
            &VocabularyOptions::default(),
        )
    }

    #[tokio::test]
    async fn snapshot_does_not_include_models() {
        let original = AppState::new();
        original.networks.write().await.insert("a".into(), sample_network("a"));
        original.models.write().await.insert("a".into(), Arc::new(RwLock::new(sample_model())));

        let snap = snapshot(&original).await;
        // snapshot() intentionally omits model weights. `"weights"` is a field
        // on LinearLayer inside Model and does not appear anywhere in the
        // metadata-only snapshot (Network uses LayerDef which has no weights).
        let json = serde_json::to_string(&snap).unwrap();
        assert!(!json.contains("\"weights\""), "model weights leaked into state snapshot");
    }

    #[tokio::test]
    async fn apply_does_not_restore_models() {
        let original = AppState::new();
        original.networks.write().await.insert("a".into(), sample_network("a"));
        original.networks.write().await.insert("b".into(), sample_network("b"));
        original.corpora.write().await.insert("a".into(), Corpus {
            network_id: "a".into(),
            kind: kinds::FEEDFORWARD.into(),
            updated_at: Utc::now(),
            feedforward: None,
            text: Some("hello".into()),
            pairs: None,
            stage: Some("pretrain".into()),
        });
        let vocab = sample_vocab();
        original.vocabs.write().await.insert("a".into(), VocabEntry {
            vocab: Arc::new(RwLock::new(vocab.clone())),
            info: VocabularyInfo {
                mode: "char".into(),
                tokens: vocab.tokens.clone(),
                options: crate::models::VocabularyOptions::default(),
                updated_at: Utc::now(),
            },
        });

        let snap = snapshot(&original).await;
        let restored = AppState::new();
        apply(&restored, snap).await;

        assert_eq!(restored.networks.read().await.len(), 2);
        assert!(restored.networks.read().await.contains_key("a"));
        assert!(restored.networks.read().await.contains_key("b"));
        assert_eq!(restored.corpora.read().await.len(), 1);
        assert_eq!(restored.vocabs.read().await.len(), 1);
        // Models are loaded lazily — apply() leaves them empty.
        assert_eq!(restored.models.read().await.len(), 0);
    }

    #[tokio::test]
    async fn model_file_roundtrip() {
        let dir = tempdir();
        let model = sample_model();

        save_model(&model, "net-x", &dir).await.expect("save");
        assert!(dir.join("models").join("net-x.json").exists());

        let loaded = load_model("net-x", &dir).await.expect("load").expect("exists");
        assert_eq!(loaded.input_dim, model.input_dim);
        assert_eq!(loaded.layers.len(), model.layers.len());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn load_model_returns_none_when_missing() {
        let dir = tempdir();
        std::fs::create_dir_all(&dir).unwrap();
        let result = load_model("nonexistent", &dir).await.expect("no error");
        assert!(result.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn delete_model_is_idempotent() {
        let dir = tempdir();
        let model = sample_model();
        save_model(&model, "del-me", &dir).await.expect("save");
        delete_model("del-me", &dir).await.expect("first delete");
        delete_model("del-me", &dir).await.expect("second delete (idempotent)");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn save_and_load_from_disk() {
        let dir = tempdir();
        let original = AppState::new();
        original.networks.write().await.insert("net-1".into(), sample_network("net-1"));
        // Put a model in memory — save_to_dir should write it to a separate file.
        original.models.write().await.insert("net-1".into(), Arc::new(RwLock::new(sample_model())));

        save_to_dir(&original, &dir).await.expect("save");
        assert!(dir.join(STATE_FILENAME).exists(), "state.json must exist");
        assert!(dir.join("models").join("net-1.json").exists(), "model file must exist");

        // state.json must NOT contain model weights.
        let raw = std::fs::read_to_string(dir.join(STATE_FILENAME)).unwrap();
        assert!(!raw.contains("\"models\""), "weights leaked into state.json");

        let loaded = load_from_dir(&dir).await.expect("load").expect("file exists");
        assert_eq!(loaded.format_version, FORMAT_VERSION);
        assert_eq!(loaded.networks.len(), 1);

        let restored = AppState::new();
        apply(&restored, loaded).await;
        assert_eq!(restored.networks.read().await.get("net-1").unwrap().name, "net-net-1");
        // Models are loaded separately.
        assert_eq!(restored.models.read().await.len(), 0);

        let model = load_model("net-1", &dir).await.expect("ok").expect("exists");
        assert_eq!(model.input_dim, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A metadata-only save must NOT rewrite an unchanged model's weight file,
    /// but a save after the weights change must. We detect the (non-)write by
    /// clobbering the file with a sentinel between saves: if the skip logic
    /// works the sentinel survives an unchanged save and is replaced only once
    /// the in-memory model actually differs.
    #[tokio::test]
    async fn unchanged_model_is_not_rewritten() {
        let dir = tempdir();
        let state = AppState::new();
        let arc = Arc::new(RwLock::new(sample_model()));
        state.networks.write().await.insert("net-1".into(), sample_network("net-1"));
        state.models.write().await.insert("net-1".into(), arc.clone());

        // First save writes the real model and records its signature.
        save_to_dir(&state, &dir).await.expect("first save");
        let model_file = dir.join("models").join("net-1.json");
        assert!(model_file.exists());

        // Clobber the file so any rewrite is observable.
        std::fs::write(&model_file, b"SENTINEL").unwrap();

        // Metadata-only change: rename the network, model weights untouched.
        state.networks.write().await.get_mut("net-1").unwrap().name = "renamed".into();
        save_to_dir(&state, &dir).await.expect("metadata save");
        assert_eq!(
            std::fs::read(&model_file).unwrap(), b"SENTINEL",
            "unchanged model weight file was needlessly rewritten"
        );

        // Now actually change the weights → the file must be rewritten.
        arc.write().await.layers.iter_mut().for_each(|l| {
            if let neuralcabin_engine::nn::Layer::Linear(ll) = l { ll.w.data[0] += 1.0; }
        });
        save_to_dir(&state, &dir).await.expect("weights save");
        let after = std::fs::read(&model_file).unwrap();
        assert_ne!(after, b"SENTINEL", "changed model weight file was not rewritten");
        // And it must be a valid model again.
        let reloaded = load_model("net-1", &dir).await.expect("ok").expect("exists");
        assert_eq!(reloaded.input_dim, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn load_returns_none_when_missing() {
        let dir = tempdir();
        let loaded = load_from_dir(&dir).await.expect("ok");
        assert!(loaded.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn load_errors_on_corrupt_file() {
        let dir = tempdir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(STATE_FILENAME), b"not json {").unwrap();
        let result = load_from_dir(&dir).await;
        assert!(result.is_err(), "corrupt file should error, not silently drop state");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn tempdir() -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("neuralcabin-test-{}", uuid::Uuid::new_v4()));
        p
    }
}
