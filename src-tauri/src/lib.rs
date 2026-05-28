mod models;
mod persistence;
mod export;
mod server;

use models::*;
use server::{ServerConfig, ServerPermissions, ServerRuntime};

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use neuralcabin_engine::nn::{Layer, LayerSpec, Model};
use neuralcabin_engine::transformer::{
    TransformerConfig, TransformerModel,
};
use neuralcabin_engine::optimizer::{Optimizer, OptimizerKind};
use neuralcabin_engine::tensor::{SplitMix64, Tensor};
use neuralcabin_engine::tokenizer::{
    TokenizerMode, Vocabulary, VocabularyOptions as EngineVocabularyOptions,
    ASSISTANT_ID, EOS_ID, USER_ID,
};
use neuralcabin_engine::corpus::{
    build_finetuning_tensors, build_pretraining_tensors, encode_context, Pair,
};
use neuralcabin_engine::activations::softmax_rows;
use neuralcabin_engine::{Activation, Loss};

// ─── State ──────────────────────────────────────────────────────────────────

pub struct AppState {
    pub(crate) networks:         Arc<RwLock<HashMap<String, Network>>>,
    pub(crate) models:           Arc<RwLock<HashMap<String, Arc<RwLock<Model>>>>>,
    pub(crate) transformers:     Arc<RwLock<HashMap<String, Arc<RwLock<TransformerModel>>>>>,
    pub(crate) vocabs:           Arc<RwLock<HashMap<String, VocabEntry>>>,
    pub(crate) corpora:          Arc<RwLock<HashMap<String, Corpus>>>,
    pub(crate) trainers:         Arc<RwLock<HashMap<String, TrainerHandle>>>,
    pub(crate) inferrers:        Arc<RwLock<HashMap<String, InferenceHandle>>>,
    pub(crate) training_history: Arc<RwLock<HashMap<String, Vec<TrainingRun>>>>,
    pub(crate) servers:          Arc<RwLock<HashMap<String, ServerRuntime>>>,
    /// Where to write `state.json`. None disables persistence — used by unit
    /// tests that don't go through the Tauri setup hook.
    pub(crate) data_dir:         Arc<RwLock<Option<PathBuf>>>,
}

#[derive(Clone)]
pub(crate) struct VocabEntry {
    pub(crate) vocab: Arc<RwLock<Vocabulary>>,
    pub(crate) info: VocabularyInfo,
}

#[derive(Clone)]
pub(crate) struct TrainerHandle {
    pub(crate) state:  Arc<RwLock<TrainingState>>,
    pub(crate) cancel: Arc<AtomicBool>,
    /// Set together with `cancel` by `abort_training` — tells the training
    /// loop to restore the pre-training model snapshot on the way out.
    pub(crate) rollback: Arc<AtomicBool>,
}

#[derive(Clone)]
struct InferenceHandle {
    cancel: Arc<AtomicBool>,
}

impl AppState {
    pub(crate) fn new() -> Self {
        Self {
            networks:         Arc::new(RwLock::new(HashMap::new())),
            models:           Arc::new(RwLock::new(HashMap::new())),
            transformers:     Arc::new(RwLock::new(HashMap::new())),
            vocabs:           Arc::new(RwLock::new(HashMap::new())),
            corpora:          Arc::new(RwLock::new(HashMap::new())),
            trainers:         Arc::new(RwLock::new(HashMap::new())),
            inferrers:        Arc::new(RwLock::new(HashMap::new())),
            training_history: Arc::new(RwLock::new(HashMap::new())),
            servers:          Arc::new(RwLock::new(HashMap::new())),
            data_dir:         Arc::new(RwLock::new(None)),
        }
    }
}

/// Save the current state to disk. Most command handlers call this after a
/// mutation. Errors are returned to the caller so the UI can surface them —
/// silent persistence failures are how data gets lost.
async fn persist(state: &AppState) -> Result<(), String> {
    let Some(dir) = state.data_dir.read().await.clone() else { return Ok(()); };
    // Timing diagnostics: if "save takes forever" is reported, this log line
    // pins down whether the cost is in snapshotting, JSON encoding, or the
    // disk write itself.
    let started = std::time::Instant::now();
    let result = persistence::save_to_dir(state, &dir).await;
    let elapsed = started.elapsed();
    if elapsed.as_millis() > 500 {
        eprintln!("[neuralcabin] persist took {}ms — investigating slow save",
                  elapsed.as_millis());
    }
    result
}

// ─── Training history helpers ────────────────────────────────────────────────

/// Downsample a loss history to at most `max_points` evenly-spaced values so
/// state.json stays compact even for very long runs.
fn downsample_history(history: &[f32], max_points: usize) -> Vec<f32> {
    if history.len() <= max_points || max_points < 2 {
        return history.to_vec();
    }
    (0..max_points).map(|i| {
        let idx = i * (history.len() - 1) / (max_points - 1);
        history[idx]
    }).collect()
}

#[allow(clippy::too_many_arguments)]
fn build_training_run_record(
    id: &str,
    network_id: &str,
    cfg: &TrainingConfig,
    started_at: DateTime<Utc>,
    epochs_run: usize,
    final_loss: f32,
    elapsed_secs: f32,
    loss_history: &[f32],
    status: &str,
) -> TrainingRun {
    TrainingRun {
        id: id.to_string(),
        network_id: network_id.to_string(),
        started_at,
        finished_at: Utc::now(),
        status: status.to_string(),
        config_summary: TrainingConfigSummary {
            optimizer: cfg.optimizer.kind.clone(),
            lr: cfg.optimizer.lr,
            batch_size: cfg.batch_size,
            epochs: cfg.epochs,
        },
        total_epochs: cfg.epochs,
        epochs_run,
        final_loss,
        elapsed_secs,
        loss_history: downsample_history(loss_history, 500),
    }
}

/// Append a completed run record and flush state to disk. Runs on the
/// background training task, so we reach AppState through the AppHandle.
async fn record_and_persist_run(app: &AppHandle, run: TrainingRun) {
    if let Some(app_state) = app.try_state::<AppState>() {
        let state = app_state.inner();
        state.training_history.write().await
            .entry(run.network_id.clone())
            .or_default()
            .push(run);
        persistence::save_best_effort(state).await;
    }
}

// ─── Lazy model loader ───────────────────────────────────────────────────────

/// Produce the appropriate "model not ready" error for a network kind.
fn model_missing_err(net: &Network) -> String {
    if net.kind == kinds::NEXT_TOKEN {
        "Model not materialized. Build a vocabulary on the Vocabulary tab first.".to_string()
    } else {
        "Model state missing — try recreating the network.".to_string()
    }
}

/// Return the in-memory model for `network_id`, loading it from its weight
/// file on demand if it isn't already cached. Inserts the loaded model into
/// `AppState.models` for subsequent calls.
async fn get_or_load_model(
    state: &AppState,
    net: &Network,
) -> Result<Arc<RwLock<Model>>, String> {
    // Fast path: already in memory (e.g. just trained, or previously loaded).
    if let Some(m) = state.models.read().await.get(&net.id).cloned() {
        return Ok(m);
    }
    // Slow path: load from the per-network weight file.
    let Some(data_dir) = persistence::data_dir(state).await else {
        return Err(model_missing_err(net));
    };
    match persistence::load_model(&net.id, &data_dir).await {
        Ok(Some(model)) => {
            let arc = Arc::new(RwLock::new(model));
            state.models.write().await.insert(net.id.clone(), arc.clone());
            Ok(arc)
        }
        Ok(None) => Err(model_missing_err(net)),
        Err(e)   => Err(format!("failed to load model: {e}")),
    }
}

/// Same as `get_or_load_model` for the transformer kind.
async fn get_or_load_transformer(
    state: &AppState,
    net: &Network,
) -> Result<Arc<RwLock<TransformerModel>>, String> {
    if let Some(m) = state.transformers.read().await.get(&net.id).cloned() {
        return Ok(m);
    }
    let Some(data_dir) = persistence::data_dir(state).await else {
        return Err("Transformer model not loaded — build vocabulary by saving a corpus first.".into());
    };
    match persistence::load_transformer(&net.id, &data_dir).await {
        Ok(Some(model)) => {
            let arc = Arc::new(RwLock::new(model));
            state.transformers.write().await.insert(net.id.clone(), arc.clone());
            Ok(arc)
        }
        Ok(None) => Err("Transformer model not loaded — build vocabulary by saving a corpus first.".into()),
        Err(e)   => Err(format!("failed to load transformer: {e}")),
    }
}

// ─── Parsing helpers ────────────────────────────────────────────────────────

fn parse_activation(name: &str) -> Result<Activation, String> {
    match name.to_ascii_lowercase().as_str() {
        "identity" => Ok(Activation::Identity),
        "relu"     => Ok(Activation::ReLU),
        "sigmoid"  => Ok(Activation::Sigmoid),
        "tanh"     => Ok(Activation::Tanh),
        "softmax"  => Ok(Activation::Softmax),
        other => Err(format!("unknown activation '{other}'")),
    }
}

fn parse_loss(name: &str) -> Result<Loss, String> {
    match name.to_ascii_lowercase().as_str() {
        "mse" | "mean_squared_error" | "meansquarederror" => Ok(Loss::MeanSquaredError),
        "ce"  | "crossentropy" | "cross_entropy"          => Ok(Loss::CrossEntropy),
        other => Err(format!("unknown loss '{other}'")),
    }
}

fn parse_tokenizer_mode(name: &str) -> Result<TokenizerMode, String> {
    match name.to_ascii_lowercase().as_str() {
        "char"     => Ok(TokenizerMode::Char),
        "subword"  => Ok(TokenizerMode::Subword),
        "word"     => Ok(TokenizerMode::Word),
        "advanced" => Ok(TokenizerMode::Advanced),
        other => Err(format!(
            "unknown vocab mode '{other}' (expected char|subword|word|advanced)"
        )),
    }
}

fn build_optimizer(cfg: &OptimizerConfig, _shapes: &[Vec<usize>]) -> Result<Optimizer, String> {
    let kind = match cfg.kind.to_ascii_lowercase().as_str() {
        "adam" => OptimizerKind::Adam {
            lr: cfg.lr,
            beta1: cfg.beta1.unwrap_or(0.9),
            beta2: cfg.beta2.unwrap_or(0.999),
            eps:   cfg.eps.unwrap_or(1e-8),
        },
        "adamw" => OptimizerKind::AdamW {
            lr: cfg.lr,
            beta1: cfg.beta1.unwrap_or(0.9),
            beta2: cfg.beta2.unwrap_or(0.999),
            eps:   cfg.eps.unwrap_or(1e-8),
            weight_decay: cfg.weight_decay.unwrap_or(0.01),
        },
        "lamb" => OptimizerKind::Lamb {
            lr: cfg.lr,
            beta1: cfg.beta1.unwrap_or(0.9),
            beta2: cfg.beta2.unwrap_or(0.999),
            eps:   cfg.eps.unwrap_or(1e-8),
            weight_decay: cfg.weight_decay.unwrap_or(0.0),
        },
        "sgd" => OptimizerKind::Sgd {
            lr: cfg.lr,
            momentum: cfg.momentum.unwrap_or(0.0),
        },
        other => return Err(format!("unknown optimizer '{other}' (expected adam|adamw|lamb|sgd)")),
    };
    // With the Burn migration, the `Optimizer` type alias is just the kind/config.
    // Burn builds the per-step optimizer instance inside `train_step` itself.
    Ok(kind)
}

fn validate_transformer_hparams(hp: &TransformerHParams) -> Result<(), String> {
    if hp.n_ctx == 0    { return Err("transformer n_ctx must be > 0".into()); }
    if hp.n_embd == 0   { return Err("transformer n_embd must be > 0".into()); }
    if hp.n_layers == 0 { return Err("transformer n_layers must be > 0".into()); }
    if hp.n_heads == 0  { return Err("transformer n_heads must be > 0".into()); }
    if hp.n_ff == 0     { return Err("transformer n_ff must be > 0".into()); }
    if !hp.n_embd.is_multiple_of(hp.n_heads) {
        return Err(format!(
            "transformer n_embd ({}) must be divisible by n_heads ({})",
            hp.n_embd, hp.n_heads
        ));
    }
    let head_dim = hp.n_embd / hp.n_heads;
    if !head_dim.is_multiple_of(2) {
        return Err(format!(
            "transformer head_dim (n_embd/n_heads = {head_dim}) must be even for RoPE"
        ));
    }
    Ok(())
}

/// Rebuild a transformer model after the corpus vocabulary changes. Keeps
/// the existing weights if `vocab_size` matches the current model; otherwise
/// reinitializes (since the embedding and LM head depend on vocab size) and
/// marks the network as untrained.
async fn rebuild_transformer_model(
    state: &State<'_, AppState>,
    net: &Network,
    vocab_size: usize,
) -> Result<(), String> {
    let hp = net.transformer.clone()
        .ok_or_else(|| "transformer network missing hparams".to_string())?;

    // Fast path: existing model already at this vocab size.
    if let Some(existing) = state.transformers.read().await.get(&net.id).cloned() {
        if existing.read().await.config.vocab_size == vocab_size {
            return Ok(());
        }
    } else if let Some(dir) = persistence::data_dir(state).await {
        if let Ok(Some(loaded)) = persistence::load_transformer(&net.id, &dir).await {
            if loaded.config.vocab_size == vocab_size {
                state.transformers.write().await
                    .insert(net.id.clone(), Arc::new(RwLock::new(loaded)));
                return Ok(());
            }
        }
    }

    let cfg = TransformerConfig {
        vocab_size,
        n_ctx:     hp.n_ctx,
        n_embd:    hp.n_embd,
        n_layers:  hp.n_layers,
        n_heads:   hp.n_heads,
        n_ff:      hp.n_ff,
        rope_theta: hp.rope_theta,
        rms_eps:    hp.rms_eps,
    };
    let model = TransformerModel::new(cfg, net.seed);
    let parameter_count = model.parameter_count();
    state.transformers.write().await
        .insert(net.id.clone(), Arc::new(RwLock::new(model)));
    if let Some(n) = state.networks.write().await.get_mut(&net.id) {
        n.parameter_count = parameter_count;
        n.trained = false;
    }
    Ok(())
}

fn parse_frozen_keys(keys: &[String]) -> (bool, bool, bool, Vec<usize>) {
    let mut emb = false; let mut out_norm = false; let mut out = false;
    let mut blocks: Vec<usize> = Vec::new();
    for k in keys {
        let k = k.trim();
        if k.eq_ignore_ascii_case("embedding") || k.eq_ignore_ascii_case("token_embd") {
            emb = true;
        } else if k.eq_ignore_ascii_case("output_norm") {
            out_norm = true;
        } else if k.eq_ignore_ascii_case("output") || k.eq_ignore_ascii_case("lm_head") {
            out = true;
        } else if let Some(rest) = k.strip_prefix("block:").or_else(|| k.strip_prefix("block_")) {
            if let Ok(idx) = rest.parse::<usize>() { blocks.push(idx); }
        }
    }
    blocks.sort();
    blocks.dedup();
    (emb, out_norm, out, blocks)
}

/// Parse `linear:N` keys (and case-equivalents) into the indices of the
/// linear sub-layers to freeze, counted by occurrence among `model.layers`.
/// Unknown keys are ignored — e.g. a transformer-only key on a feedforward
/// network is a no-op rather than a hard error.
fn parse_linear_frozen_keys(keys: &[String]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for k in keys {
        let k = k.trim();
        let rest = k.strip_prefix("linear:")
            .or_else(|| k.strip_prefix("linear_"))
            .or_else(|| k.strip_prefix("layer:"));
        if let Some(rest) = rest {
            if let Ok(idx) = rest.parse::<usize>() { out.push(idx); }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn build_layer_specs(layers: &[LayerDef], input_dim: usize) -> Result<(Vec<LayerSpec>, usize), String> {
    let mut specs = Vec::with_capacity(layers.len());
    let mut cur = input_dim;
    for (i, l) in layers.iter().enumerate() {
        match l {
            LayerDef::Linear { in_dim, out_dim } => {
                if *in_dim != cur {
                    return Err(format!(
                        "layer {i}: linear in_dim {in_dim} doesn't match running dim {cur}"
                    ));
                }
                if *out_dim == 0 {
                    return Err(format!("layer {i}: linear out_dim must be > 0"));
                }
                specs.push(LayerSpec::Linear { in_dim: *in_dim, out_dim: *out_dim });
                cur = *out_dim;
            }
            LayerDef::Activation { activation } => {
                specs.push(LayerSpec::Activation(parse_activation(activation)?));
            }
        }
    }
    Ok((specs, cur))
}

// ─── Network commands ───────────────────────────────────────────────────────

#[tauri::command]
async fn create_network(
    app: AppHandle,
    req: CreateNetworkRequest,
) -> Result<Network, String> {
    create_network_internal(&app, req).await
}

/// Create a network using whatever AppState is held by the given Tauri handle.
/// Shared between the Tauri command and the HTTP server.
pub(crate) async fn create_network_internal(
    app: &AppHandle,
    req: CreateNetworkRequest,
) -> Result<Network, String> {
    let state = app.state::<AppState>();
    if !kinds::all().contains(&req.kind.as_str()) {
        return Err(format!(
            "unknown network kind '{}' (expected one of: {:?})", req.kind, kinds::all()
        ));
    }
    if req.layers.is_empty() && req.kind == kinds::FEEDFORWARD {
        return Err("feedforward network must have at least one layer".into());
    }
    let name = req.name.trim().to_string();
    if name.is_empty() { return Err("network name cannot be empty".into()); }

    let id = Uuid::new_v4().to_string();
    let now = Utc::now();

    let network = if req.kind == kinds::FEEDFORWARD {
        let input_dim = req.input_dim.unwrap_or(0);
        if input_dim == 0 {
            return Err("feedforward network requires input_dim > 0".into());
        }
        let (specs, output_dim) = build_layer_specs(&req.layers, input_dim)?;
        let model = Model::from_specs(input_dim, &specs, req.seed);
        let parameter_count = model.parameter_count();
        state.models.write().await.insert(id.clone(), Arc::new(RwLock::new(model)));

        Network {
            id: id.clone(),
            name, kind: req.kind, seed: req.seed, created_at: now, trained: false,
            input_dim, output_dim,
            layers: req.layers, parameter_count,
            hidden_layers: None, context_size: None, transformer: None,
            pretrained: false,
        }
    } else if req.kind == kinds::TRANSFORMER {
        // Transformer: materialise the model now with a vocab placeholder; it
        // gets rebuilt on first corpus save once the real vocab size is known.
        // We accept hparams via `req.transformer` — `req.layers` is ignored.
        let hp = req.transformer.clone()
            .ok_or_else(|| "transformer network requires `transformer` hparams".to_string())?;
        validate_transformer_hparams(&hp)?;
        // Initial placeholder vocab — overwritten when a corpus is attached.
        let placeholder_vocab = 32;
        let cfg = TransformerConfig {
            vocab_size: placeholder_vocab,
            n_ctx:     hp.n_ctx,
            n_embd:    hp.n_embd,
            n_layers:  hp.n_layers,
            n_heads:   hp.n_heads,
            n_ff:      hp.n_ff,
            rope_theta: hp.rope_theta,
            rms_eps:    hp.rms_eps,
        };
        let model = TransformerModel::new(cfg, req.seed);
        let parameter_count = model.parameter_count();
        state.transformers.write().await.insert(
            id.clone(), Arc::new(RwLock::new(model)),
        );
        Network {
            id: id.clone(),
            name, kind: req.kind, seed: req.seed, created_at: now, trained: false,
            input_dim: 0, output_dim: 0,
            layers: Vec::new(), parameter_count,
            hidden_layers: None,
            context_size: Some(hp.n_ctx),
            transformer: Some(hp),
            pretrained: false,
        }
    } else {
        // Next-token (MLP-on-window): only the hidden chain is meaningful at this point.
        let context_size = req.context_size.unwrap_or(0);
        if context_size == 0 {
            return Err("next-token network requires context_size > 0".into());
        }
        Network {
            id: id.clone(),
            name, kind: req.kind, seed: req.seed, created_at: now, trained: false,
            input_dim: 0, output_dim: 0,
            layers: Vec::new(), parameter_count: 0,
            hidden_layers: Some(req.layers),
            context_size: Some(context_size),
            transformer: None,
            pretrained: false,
        }
    };

    state.networks.write().await.insert(id, network.clone());
    persist(&state).await?;
    Ok(network)
}

#[tauri::command]
async fn list_networks(state: State<'_, AppState>) -> Result<Vec<Network>, String> {
    Ok(state.networks.read().await.values().cloned().collect())
}

#[tauri::command]
async fn get_network(state: State<'_, AppState>, id: String) -> Result<Network, String> {
    state.networks.read().await.get(&id).cloned().ok_or_else(|| "Network not found".into())
}

#[tauri::command]
async fn delete_network(app: AppHandle, id: String) -> Result<bool, String> {
    delete_network_internal(&app, &id).await
}

pub(crate) async fn delete_network_internal(app: &AppHandle, id: &str) -> Result<bool, String> {
    let state = app.state::<AppState>();
    state.models.write().await.remove(id);
    state.transformers.write().await.remove(id);
    state.vocabs.write().await.remove(id);
    state.corpora.write().await.remove(id);
    state.training_history.write().await.remove(id);
    if let Some(dir) = persistence::data_dir(&state).await {
        if let Err(e) = persistence::delete_model(id, &dir).await {
            eprintln!("[neuralcabin] {e}");
        }
        if let Err(e) = persistence::delete_transformer(id, &dir).await {
            eprintln!("[neuralcabin] {e}");
        }
    }
    let removed = state.networks.write().await.remove(id).is_some();
    persist(&state).await?;
    Ok(removed)
}

/// Shared wrapper for the server module: fetch (loading from disk if needed)
/// an Arc<RwLock<Model>> for the given network.
pub(crate) async fn get_or_load_model_arc(
    app: &AppHandle,
    net: &Network,
) -> Result<Arc<RwLock<Model>>, String> {
    let state = app.state::<AppState>();
    get_or_load_model(&state, net).await
}

/// Install a model snapshot for a given network. Used by the HTTP upload
/// endpoint to overwrite the existing weights with a remote-supplied set.
pub(crate) async fn install_model_for_network(
    app: &AppHandle,
    net: &Network,
    model: Model,
) {
    let state = app.state::<AppState>();
    state.models.write().await.insert(net.id.clone(), Arc::new(RwLock::new(model)));
    if let Some(n) = state.networks.write().await.get_mut(&net.id) {
        n.trained = true;
    }
    persistence::save_best_effort(&state).await;
}

/// Synchronous inference helper for the HTTP server. For feed-forward networks
/// this runs the model and returns the output vector inline. For next-token
/// networks it runs generation to completion and returns the produced text.
pub(crate) async fn infer_sync(
    app: &AppHandle,
    req: InferRequest,
) -> Result<serde_json::Value, String> {
    let state = app.state::<AppState>();
    let net = state.networks.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "Network not found".to_string())?;
    let model_arc = get_or_load_model(&state, &net).await?;

    if net.kind == kinds::FEEDFORWARD {
        let model = model_arc.read().await;
        let features = req.features
            .ok_or_else(|| "features array required for feedforward inference".to_string())?;
        if features.len() != net.input_dim {
            return Err(format!(
                "features length {} doesn't match input_dim {}",
                features.len(), net.input_dim
            ));
        }
        let x = Tensor::new(vec![1, net.input_dim], features);
        let y = model.predict(&x);
        Ok(serde_json::json!({ "output": y.data }))
    } else if net.kind == kinds::NEXT_TOKEN {
        let context_size = net.context_size
            .ok_or_else(|| "Next-token network missing context_size".to_string())?;
        let entry = state.vocabs.read().await.get(&req.network_id).cloned()
            .ok_or_else(|| "Vocabulary missing".to_string())?;
        let vocab = entry.vocab.read().await.clone();
        let model = model_arc.read().await;
        let prompt = req.prompt.unwrap_or_default();
        let max_new = req.max_new_tokens.unwrap_or(64).clamp(1, 4096);
        let temperature = req.temperature.unwrap_or(0.0).max(0.0);

        let mut ids: Vec<u32> = vocab.encode(&prompt);
        let initial_len = ids.len();
        let mut generated = String::new();
        let mut rng = SplitMix64::new(0xDEAD_BEEF_u64 ^ (max_new as u64).wrapping_mul(0x9E3779B1));

        for _ in 0..max_new {
            let ctx = encode_context(&ids, &vocab, context_size);
            let logits = model.predict(&ctx);
            let probs = softmax_rows(&logits);
            let row = &probs.data[..probs.cols()];
            let chosen = if temperature <= f32::EPSILON {
                argmax_local(row)
            } else {
                sample_with_temperature_local(row, temperature, &mut rng)
            };
            if (chosen as u32) == EOS_ID { break; }
            if (chosen as u32) >= neuralcabin_engine::tokenizer::RESERVED as u32 {
                generated.push_str(vocab.token_of(chosen as u32));
            }
            ids.push(chosen as u32);
        }
        Ok(serde_json::json!({
            "generated": generated,
            "token_count": ids.len() - initial_len,
        }))
    } else {
        Err(format!("unsupported network kind '{}'", net.kind))
    }
}

fn argmax_local(row: &[f32]) -> usize {
    let mut best = 0; let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() { if v > bv { bv = v; best = i; } }
    best
}
fn sample_with_temperature_local(probs: &[f32], temperature: f32, rng: &mut SplitMix64) -> usize {
    let mut logits: Vec<f32> = probs.iter().map(|p| p.max(1e-12).ln() / temperature).collect();
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in logits.iter_mut() { *v = (*v - mx).exp(); sum += *v; }
    for v in logits.iter_mut() { *v /= sum; }
    let r = (rng.next_u64() as f64 / u64::MAX as f64) as f32;
    let mut acc = 0.0_f32;
    for (i, &p) in logits.iter().enumerate() {
        acc += p; if r <= acc { return i; }
    }
    logits.len() - 1
}

/// Start a training run without going through the Tauri command wrapper.
/// Used by the HTTP server's POST /train endpoint.
pub(crate) async fn start_training_internal(
    app: &AppHandle,
    req: StartTrainingRequest,
) -> Result<StartTrainingResponse, String> {
    start_training(app.clone(), app.state::<AppState>(), req).await
}

// ─── Corpus commands ────────────────────────────────────────────────────────

#[tauri::command]
async fn set_corpus(
    state: State<'_, AppState>,
    req: SetCorpusRequest,
) -> Result<CorpusStats, String> {
    let net = state.networks.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "Network not found".to_string())?;

    let mut corpus = Corpus {
        network_id: net.id.clone(),
        kind: net.kind.clone(),
        updated_at: Utc::now(),
        feedforward: None,
        text: None,
        pairs: None,
        stage: None,
    };

    if net.kind == kinds::FEEDFORWARD {
        let ff = req.feedforward.ok_or_else(|| "feedforward corpus required".to_string())?;
        validate_feedforward(&ff, &net)?;
        corpus.feedforward = Some(ff);
    } else if net.kind == kinds::NEXT_TOKEN || net.kind == kinds::TRANSFORMER {
        // Fine-tuning is only meaningful on top of a pre-trained model. If
        // the user hasn't completed pre-training yet, refuse — otherwise we'd
        // happily wipe whatever weights/vocab exist and pretend we fine-tuned
        // a never-trained model, which is what the previous code did.
        if req.stage.as_deref() == Some("finetune") && !net.pretrained {
            return Err(
                "Fine-tuning requires a completed pre-training run first. \
                 Switch to the Pretraining stage, save a text corpus, and run \
                 training to completion before fine-tuning.".into()
            );
        }
        // Merge with the previously-saved corpus so editing one stage doesn't
        // wipe data from the other. The user might cycle pretrain → finetune
        // → edit pretrain → re-pretrain → finetune again without ever
        // re-uploading the pairs. Each stage save only overwrites the fields
        // that belong to *that* stage; the other stage's data is preserved.
        let existing = state.corpora.read().await.get(&net.id).cloned();
        let stage = req.stage.clone();
        match stage.as_deref() {
            Some("pretrain") => {
                // Pretrain save updates text (and stage). Keep existing pairs.
                corpus.text  = req.text.or_else(|| existing.as_ref().and_then(|c| c.text.clone()));
                corpus.pairs = existing.as_ref().and_then(|c| c.pairs.clone());
            }
            Some("finetune") => {
                // Fine-tune save updates pairs. Keep existing text.
                corpus.pairs = req.pairs.or_else(|| existing.as_ref().and_then(|c| c.pairs.clone()));
                corpus.text  = existing.as_ref().and_then(|c| c.text.clone());
            }
            _ => {
                // Stage unspecified: fall back to the previous behaviour but
                // still preserve whichever field the caller didn't supply.
                corpus.text  = req.text.or_else(|| existing.as_ref().and_then(|c| c.text.clone()));
                corpus.pairs = req.pairs.or_else(|| existing.as_ref().and_then(|c| c.pairs.clone()));
            }
        }
        corpus.stage = stage;
    } else {
        return Err(format!("unsupported network kind '{}'", net.kind));
    }

    // Persist the corpus first so subsequent operations see it.
    state.corpora.write().await.insert(net.id.clone(), corpus.clone());

    // For text-based networks, auto-build the vocabulary from the new corpus
    // so the user doesn't have to do it manually. Without this, training
    // would immediately fail with "no vocabulary" after saving a corpus.
    // Fine-tuning must NEVER rebuild the vocabulary or the model — that would
    // discard the pre-trained weights. The vocab established during pretraining
    // is the canonical one; finetune data is encoded against it (any
    // characters/words missing from it become the <unk> token, matching how
    // GPT-style fine-tuning works on a frozen tokenizer).
    let already_has_vocab = state.vocabs.read().await.contains_key(&net.id);
    let preserve_vocab = corpus.stage.as_deref() == Some("finetune") && already_has_vocab;
    if (net.kind == kinds::NEXT_TOKEN || net.kind == kinds::TRANSFORMER) && !preserve_vocab {
        let mode_name = req.vocab_mode.as_deref().unwrap_or("char");
        if let Ok(mode) = parse_tokenizer_mode(mode_name) {
            if !matches!(mode, TokenizerMode::Advanced) {
                let mut texts: Vec<String> = Vec::new();
                if let Some(t) = &corpus.text { texts.push(t.clone()); }
                if let Some(ps) = &corpus.pairs {
                    for p in ps { texts.push(p.input.clone()); texts.push(p.output.clone()); }
                }
                if !texts.is_empty() {
                    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                    let opts = EngineVocabularyOptions::default();
                    let vocab = Vocabulary::build(mode, &refs, &opts);
                    let vocab_size = vocab.size();
                    let info = VocabularyInfo {
                        mode: mode_str(mode),
                        tokens: vocab.tokens.clone(),
                        options: VocabularyOptions::default(),
                        updated_at: Utc::now(),
                    };
                    let entry = VocabEntry {
                        vocab: Arc::new(RwLock::new(vocab)),
                        info,
                    };
                    state.vocabs.write().await.insert(net.id.clone(), entry);
                    if net.kind == kinds::NEXT_TOKEN {
                        rebuild_next_token_model(&state, &net, vocab_size).await?;
                    } else {
                        rebuild_transformer_model(&state, &net, vocab_size).await?;
                    }
                }
            }
        }
    }

    // Recompute stats AFTER any vocab/model rebuild so the response reflects them.
    let net_after = state.networks.read().await.get(&req.network_id).cloned().unwrap_or(net);
    let stats = compute_stats(
        &net_after,
        &corpus,
        &*state.vocabs.read().await,
        &*state.models.read().await,
    ).await;
    persist(&state).await?;
    Ok(stats)
}

fn validate_feedforward(ff: &FeedforwardCorpus, net: &Network) -> Result<(), String> {
    if ff.in_dim != net.input_dim {
        return Err(format!(
            "feedforward in_dim {} doesn't match network input_dim {}",
            ff.in_dim, net.input_dim
        ));
    }
    if ff.out_dim != net.output_dim {
        return Err(format!(
            "feedforward out_dim {} doesn't match network output_dim {}",
            ff.out_dim, net.output_dim
        ));
    }
    if ff.features.len() != ff.rows * ff.in_dim {
        return Err("feedforward features length doesn't match rows × in_dim".into());
    }
    if ff.targets.len() != ff.rows * ff.out_dim {
        return Err("feedforward targets length doesn't match rows × out_dim".into());
    }
    if ff.rows == 0 {
        return Err("feedforward corpus must have at least one row".into());
    }
    Ok(())
}

async fn compute_stats(
    net: &Network,
    corpus: &Corpus,
    vocabs: &HashMap<String, VocabEntry>,
    models: &HashMap<String, Arc<RwLock<Model>>>,
) -> CorpusStats {
    let mut stats = CorpusStats {
        kind: net.kind.clone(),
        stage: corpus.stage.clone(),
        rows: None, in_dim: None, out_dim: None,
        text_chars: None, text_tokens: None, pair_count: None,
        vocab_size: None, vocab_mode: None,
        training_examples: None,
        vocab_ready: false,
        model_ready: models.contains_key(&net.id),
    };

    if let Some(ff) = &corpus.feedforward {
        stats.rows    = Some(ff.rows);
        stats.in_dim  = Some(ff.in_dim);
        stats.out_dim = Some(ff.out_dim);
        stats.training_examples = Some(ff.rows);
    }

    if net.kind == kinds::NEXT_TOKEN || net.kind == kinds::TRANSFORMER {
        if let Some(text) = &corpus.text {
            stats.text_chars = Some(text.chars().count());
        }
        stats.pair_count = corpus.pairs.as_ref().map(|p| p.len());

        if let Some(entry) = vocabs.get(&net.id) {
            stats.vocab_ready = true;
            stats.vocab_mode = Some(entry.info.mode.clone());
            let v = entry.vocab.read().await;
            stats.vocab_size = Some(v.size());
            if let Some(text) = &corpus.text {
                stats.text_tokens = Some(v.encode(text).len());
            }
            let context_size = net.context_size.unwrap_or(0);
            if context_size > 0 {
                let mut total = 0usize;
                if let Some(text) = &corpus.text {
                    let n = v.encode(text).len();
                    if n > context_size { total += n - context_size; }
                }
                if let Some(pairs) = &corpus.pairs {
                    for p in pairs {
                        let len = 1 + v.encode(&p.input).len() + 1
                                + v.encode(&p.output).len() + 1;
                        if len >= 2 { total += len - 1; }
                    }
                }
                stats.training_examples = Some(total);
            }
        }
    }

    stats
}

#[tauri::command]
async fn get_corpus(
    state: State<'_, AppState>,
    network_id: String,
) -> Result<Option<Corpus>, String> {
    Ok(state.corpora.read().await.get(&network_id).cloned())
}

#[tauri::command]
async fn corpus_stats(
    state: State<'_, AppState>,
    network_id: String,
) -> Result<Option<CorpusStats>, String> {
    let net = state.networks.read().await.get(&network_id).cloned()
        .ok_or_else(|| "Network not found".to_string())?;
    let corpora = state.corpora.read().await;
    let Some(corpus) = corpora.get(&network_id) else { return Ok(None); };
    let vocabs = state.vocabs.read().await;
    let models = state.models.read().await;
    let stats = compute_stats(&net, corpus, &vocabs, &models).await;
    Ok(Some(stats))
}

// ─── Vocabulary commands ────────────────────────────────────────────────────

#[tauri::command]
async fn build_vocabulary(
    state: State<'_, AppState>,
    req: BuildVocabularyRequest,
) -> Result<VocabularyInfo, String> {
    let mode = parse_tokenizer_mode(&req.mode)?;
    if matches!(mode, TokenizerMode::Advanced) {
        return Err("use set_advanced_vocabulary for advanced mode".to_string());
    }
    let net = state.networks.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "Network not found".to_string())?;
    if net.kind != kinds::NEXT_TOKEN && net.kind != kinds::TRANSFORMER {
        return Err("vocabularies apply only to text networks (next-token or transformer)".to_string());
    }
    let corpus = state.corpora.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "No corpus attached. Add training data on the Corpus tab first.".to_string())?;

    let mut owned: Vec<String> = Vec::new();
    if let Some(t) = &corpus.text { owned.push(t.clone()); }
    if let Some(ps) = &corpus.pairs {
        for p in ps { owned.push(p.input.clone()); owned.push(p.output.clone()); }
    }
    if owned.is_empty() {
        return Err("corpus is empty — add text or pairs before building a vocabulary".to_string());
    }
    let refs: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();

    let engine_opts = EngineVocabularyOptions {
        subword_merges: req.options.subword_merges,
        word_top_n:     req.options.word_top_n,
    };
    let vocab = Vocabulary::build(mode, &refs, &engine_opts);
    finalize_vocab(state, net, vocab, mode_str(mode), req.options).await
}

#[tauri::command]
async fn set_advanced_vocabulary(
    state: State<'_, AppState>,
    req: SetAdvancedVocabularyRequest,
) -> Result<VocabularyInfo, String> {
    let net = state.networks.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "Network not found".to_string())?;
    if net.kind != kinds::NEXT_TOKEN && net.kind != kinds::TRANSFORMER {
        return Err("vocabularies apply only to text networks (next-token or transformer)".to_string());
    }
    let trimmed: Vec<String> = req.tokens.into_iter()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    if trimmed.is_empty() {
        return Err("advanced vocabulary requires at least one user token".to_string());
    }
    let vocab = Vocabulary::build_advanced(&trimmed);
    finalize_vocab(state, net, vocab, "advanced".into(), VocabularyOptions::default()).await
}

async fn finalize_vocab(
    state: State<'_, AppState>,
    net: Network,
    vocab: Vocabulary,
    mode_label: String,
    options: VocabularyOptions,
) -> Result<VocabularyInfo, String> {
    let tokens = vocab.tokens.clone();
    let info = VocabularyInfo {
        mode: mode_label,
        tokens: tokens.clone(),
        options,
        updated_at: Utc::now(),
    };
    let entry = VocabEntry {
        vocab: Arc::new(RwLock::new(vocab.clone())),
        info: info.clone(),
    };
    state.vocabs.write().await.insert(net.id.clone(), entry);

    // Materialize the model: input = vocab × context, hidden chain, output = vocab.
    if net.kind == kinds::TRANSFORMER {
        rebuild_transformer_model(&state, &net, vocab.size()).await?;
    } else {
        rebuild_next_token_model(&state, &net, vocab.size()).await?;
    }
    persist(&state).await?;
    Ok(info)
}

fn mode_str(m: TokenizerMode) -> String {
    match m {
        TokenizerMode::Char     => "char",
        TokenizerMode::Subword  => "subword",
        TokenizerMode::Word     => "word",
        TokenizerMode::Advanced => "advanced",
    }.into()
}

async fn rebuild_next_token_model(
    state: &State<'_, AppState>,
    net: &Network,
    vocab_size: usize,
) -> Result<(), String> {
    let context_size = net.context_size
        .ok_or_else(|| "next-token network missing context_size".to_string())?;
    let hidden = net.hidden_layers.clone().unwrap_or_default();
    let input_dim = vocab_size * context_size;

    // Compose the full layer chain.
    let mut full: Vec<LayerDef> = Vec::with_capacity(hidden.len() + 2);
    // First linear projects input one-hot window into the first hidden width.
    let first_hidden_width = first_linear_in_dim_target(&hidden).unwrap_or(vocab_size);
    full.push(LayerDef::Linear { in_dim: input_dim, out_dim: first_hidden_width });

    // The user's hidden chain starts AT first_hidden_width. The first user
    // Linear (if any) must already have in_dim == first_hidden_width.
    full.extend(hidden.iter().cloned());

    // Output projection to vocab logits.
    let pre_output_dim = running_output_dim(&full, input_dim)?;
    full.push(LayerDef::Linear { in_dim: pre_output_dim, out_dim: vocab_size });

    let (specs, output_dim) = build_layer_specs(&full, input_dim)?;

    // CRITICAL: if a model already exists with matching architecture, keep its
    // weights. Re-building unconditionally on every set_corpus call wipes out
    // trained weights — the user's report ("training disappears after I save
    // the corpus / reopen the app") was exactly this bug.
    //
    // Models are lazy-loaded, so we check memory first then fall back to the
    // weight file on disk. This preserves trained weights across restarts even
    // though we no longer load models eagerly on startup.
    let existing_arc = state.models.read().await.get(&net.id).cloned()
        .or(None); // just for clarity; check disk below if None

    let existing_arc = if let Some(a) = existing_arc {
        Some(a)
    } else if let Some(dir) = persistence::data_dir(state).await {
        match persistence::load_model(&net.id, &dir).await {
            Ok(Some(m)) => {
                let a = Arc::new(RwLock::new(m));
                state.models.write().await.insert(net.id.clone(), a.clone());
                Some(a)
            }
            _ => None,
        }
    } else {
        None
    };

    if let Some(existing) = existing_arc {
        let m = existing.read().await;
        if m.input_dim == input_dim && layers_match_specs(&m.layers, &specs) {
            return Ok(());
        }
    }

    let model = Model::from_specs(input_dim, &specs, net.seed);
    let parameter_count = model.parameter_count();

    // Persist the rebuilt model + updated network metadata.
    state.models.write().await.insert(net.id.clone(), Arc::new(RwLock::new(model)));
    if let Some(n) = state.networks.write().await.get_mut(&net.id) {
        n.layers = full;
        n.input_dim = input_dim;
        n.output_dim = output_dim;
        n.parameter_count = parameter_count;
        n.trained = false; // dimensions changed; weights are fresh
    }
    Ok(())
}

/// Return true iff every existing layer matches the corresponding spec in shape
/// and activation kind. Used by `rebuild_next_token_model` to avoid clobbering
/// trained weights when nothing about the architecture has changed.
fn layers_match_specs(layers: &[Layer], specs: &[LayerSpec]) -> bool {
    if layers.len() != specs.len() { return false; }
    for (l, s) in layers.iter().zip(specs.iter()) {
        match (l, s) {
            (Layer::Linear(ll), LayerSpec::Linear { in_dim, out_dim }) => {
                if ll.in_dim != *in_dim || ll.out_dim != *out_dim { return false; }
            }
            (Layer::Activation(a1), LayerSpec::Activation(a2)) => {
                if a1 != a2 { return false; }
            }
            _ => return false,
        }
    }
    true
}

/// First Linear's in_dim in the chain, if any.
fn first_linear_in_dim_target(layers: &[LayerDef]) -> Option<usize> {
    for l in layers {
        if let LayerDef::Linear { in_dim, .. } = l { return Some(*in_dim); }
    }
    None
}

/// Output dim of the chain so far, starting from the given input dim.
fn running_output_dim(layers: &[LayerDef], input_dim: usize) -> Result<usize, String> {
    let mut cur = input_dim;
    for (i, l) in layers.iter().enumerate() {
        if let LayerDef::Linear { in_dim, out_dim } = l {
            if *in_dim != cur {
                return Err(format!(
                    "layer {i}: in_dim {in_dim} doesn't match running dim {cur}"
                ));
            }
            cur = *out_dim;
        }
    }
    Ok(cur)
}

#[tauri::command]
async fn get_vocabulary(
    state: State<'_, AppState>,
    network_id: String,
) -> Result<Option<VocabularyInfo>, String> {
    Ok(state.vocabs.read().await.get(&network_id).map(|e| e.info.clone()))
}

#[tauri::command]
async fn tokenize_preview(
    state: State<'_, AppState>,
    network_id: String,
    text: String,
) -> Result<Vec<(u32, String)>, String> {
    let entry = state.vocabs.read().await.get(&network_id).cloned()
        .ok_or_else(|| "no vocabulary on this network".to_string())?;
    let v = entry.vocab.read().await;
    let ids = v.encode(&text);
    Ok(ids.into_iter().map(|id| (id, v.token_of(id).to_string())).collect())
}

// ─── Transformer training ──────────────────────────────────────────────────

async fn start_transformer_training(
    app: &AppHandle,
    state: &State<'_, AppState>,
    net: Network,
    corpus: Corpus,
    req: StartTrainingRequest,
) -> Result<StartTrainingResponse, String> {
    if req.config.epochs == 0 { return Err("epochs must be > 0".into()); }
    let stage = corpus.stage.as_deref().unwrap_or("pretrain").to_string();
    if stage == "finetune" && !net.pretrained {
        return Err(
            "Fine-tuning requires a completed pre-training run on this network. \
             Pre-train first, then come back to fine-tune.".into()
        );
    }
    let entry = state.vocabs.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "Vocabulary missing — save a corpus first".to_string())?;
    let vocab_size = entry.vocab.read().await.size();
    // Only rebuild when not fine-tuning. For fine-tuning we keep the
    // pre-trained weights at all costs — vocabulary size is already fixed.
    if stage != "finetune" {
        rebuild_transformer_model(state, &net, vocab_size).await?;
    }
    let model_arc = get_or_load_transformer(state, &net).await?;

    // Reload network (parameter count may have updated).
    let net = state.networks.read().await.get(&req.network_id).cloned().unwrap_or(net);
    let hp = net.transformer.clone()
        .ok_or_else(|| "transformer network missing hparams".to_string())?;
    let n_ctx = hp.n_ctx;

    // Build (input, target) token sequences from the corpus.
    let vocab = entry.vocab.read().await.clone();
    let sequences = build_transformer_sequences(&corpus, &vocab, n_ctx)?;
    let total_examples = sequences.len();
    if total_examples == 0 {
        return Err(format!("No training sequences produced (need ≥ {} tokens)", n_ctx + 1));
    }
    if net.parameter_count == 0 {
        return Err("Transformer has no parameters — internal state is broken".into());
    }

    let opt_kind = build_optimizer(&req.config.optimizer, &[])?;
    let batch_size = req.config.batch_size.max(1);

    let training_id = Uuid::new_v4().to_string();
    let started_at = Utc::now();
    let cancel   = Arc::new(AtomicBool::new(false));
    let rollback = Arc::new(AtomicBool::new(false));
    let training_state = Arc::new(RwLock::new(TrainingState {
        running: true,
        total_epochs: req.config.epochs,
        ..Default::default()
    }));
    state.trainers.write().await.insert(training_id.clone(), TrainerHandle {
        state: training_state.clone(),
        cancel: cancel.clone(),
        rollback: rollback.clone(),
    });

    let app_clone = app.clone();
    let networks_handle = state.networks.clone();
    let id_clone = training_id.clone();
    let cfg = req.config.clone();
    let net_id = req.network_id.clone();
    let n_embd = hp.n_embd;
    let frozen = cfg.frozen_layers.clone().unwrap_or_default();
    let stage_for_loop = stage.clone();

    tokio::spawn(async move {
        run_transformer_training_loop(
            app_clone, id_clone, net_id, cfg, started_at, model_arc, networks_handle,
            sequences, n_ctx, n_embd, opt_kind, batch_size,
            training_state, cancel, rollback, frozen, stage_for_loop,
        ).await;
    });

    Ok(StartTrainingResponse { training_id, status: "running".into() })
}

/// Turn the corpus into a flat list of `(input[n_ctx], target[n_ctx])`
/// sequences. We use the same vocab/tokenization the MLP next-token path
/// uses, so users can switch architectures without rebuilding the corpus.
type TokenSequence = (Vec<u32>, Vec<u32>);
fn build_transformer_sequences(
    corpus: &Corpus,
    vocab: &Vocabulary,
    n_ctx: usize,
) -> Result<Vec<TokenSequence>, String> {
    use neuralcabin_engine::tokenizer::{ASSISTANT_ID, EOS_ID, USER_ID};
    let stage = corpus.stage.as_deref().unwrap_or("pretrain");
    let mut sequences = Vec::new();

    match stage {
        "pretrain" => {
            let text = corpus.text.as_deref()
                .ok_or_else(|| "Pretraining requires a text corpus".to_string())?;
            let ids = vocab.encode(text);
            if ids.len() < n_ctx + 1 {
                return Err(format!(
                    "Text has {} tokens, need at least {} for n_ctx = {n_ctx}",
                    ids.len(), n_ctx + 1
                ));
            }
            // Sliding window over the entire corpus.
            for start in 0..=(ids.len() - n_ctx - 1) {
                let input  = ids[start..start + n_ctx].to_vec();
                let target = ids[start + 1..start + n_ctx + 1].to_vec();
                sequences.push((input, target));
            }
        }
        "finetune" => {
            let pairs = corpus.pairs.as_ref()
                .ok_or_else(|| "Fine-tuning requires input/output pairs".to_string())?;
            if pairs.is_empty() {
                return Err("Fine-tuning needs at least one input/output pair".into());
            }
            for p in pairs {
                // <user> input <eos> <assistant> output <eos>
                let mut ids = vec![USER_ID];
                ids.extend(vocab.encode(&p.input));
                ids.push(EOS_ID);
                ids.push(ASSISTANT_ID);
                ids.extend(vocab.encode(&p.output));
                ids.push(EOS_ID);

                // Pad or trim to exactly n_ctx + 1, so we can build one (in, tgt)
                // pair per example.
                let needed = n_ctx + 1;
                if ids.len() < needed {
                    while ids.len() < needed { ids.push(EOS_ID); }
                } else if ids.len() > needed {
                    ids.truncate(needed);
                }
                let input:  Vec<u32> = ids[..n_ctx].to_vec();
                let target: Vec<u32> = ids[1..n_ctx + 1].to_vec();
                sequences.push((input, target));
            }
        }
        other => return Err(format!("unknown stage '{other}'")),
    }
    Ok(sequences)
}

#[allow(clippy::too_many_arguments)]
async fn run_transformer_training_loop(
    app: AppHandle,
    training_id: String,
    network_id: String,
    cfg: TrainingConfig,
    started_at: DateTime<Utc>,
    model_arc: Arc<RwLock<TransformerModel>>,
    networks_handle: Arc<RwLock<HashMap<String, Network>>>,
    sequences: Vec<TokenSequence>,
    n_ctx: usize,
    _n_embd: usize,
    opt_kind: neuralcabin_engine::optimizer::OptimizerKind,
    batch_size: usize,
    training_state: Arc<RwLock<TrainingState>>,
    cancel: Arc<AtomicBool>,
    rollback: Arc<AtomicBool>,
    frozen_layers: Vec<String>,
    stage: String,
) {
    let initial_snapshot: TransformerModel = model_arc.read().await.clone();
    let start = std::time::Instant::now();
    let mut loss_history: Vec<f32> = Vec::with_capacity(cfg.epochs);
    let mut rng = SplitMix64::new(cfg.seed.wrapping_add(0x5A5A_A5A5_A5A5_5A5A));
    let mut indices: Vec<usize> = (0..sequences.len()).collect();
    let device = neuralcabin_engine::default_gpu_device();

    // Build the trainer session ONCE per run. The previous implementation
    // re-uploaded every model weight to the GPU, rebuilt the optimizer (so
    // Adam's momentum/variance state was zero every step → effectively SGD
    // with very small effective LR), and reallocated the RoPE/causal-mask
    // tensors on every batch. That overhead, combined with the lost optimizer
    // state, is why training that should take seconds was taking hours.
    //
    // The session keeps the model and optimizer resident on the GPU for the
    // whole run and reuses the precomputed RoPE tables and causal mask.
    let mut session = neuralcabin_engine::TrainerSession::<
        neuralcabin_engine::GpuAutodiffBackend,
    >::new(
        &*model_arc.read().await,
        &opt_kind,
        n_ctx,
        device.clone(),
    );

    // Parse frozen-layer keys once. The session does on-device snapshot/
    // restore around each step using these.
    let (frz_emb, frz_out_norm, frz_out, frz_blocks) = parse_frozen_keys(&frozen_layers);
    let has_frozen = frz_emb || frz_out_norm || frz_out || !frz_blocks.is_empty();

    for epoch in 1..=cfg.epochs {
        if cancel.load(Ordering::Relaxed) {
            // Sync the latest GPU weights back to the CPU model unless we'll
            // roll back to the snapshot anyway.
            if !rollback.load(Ordering::Relaxed) {
                let mut m = model_arc.write().await;
                session.write_back(&mut m);
            }
            transformer_finish_cancelled(
                &app, &training_id, &network_id, &cfg, started_at,
                epoch.saturating_sub(1), &loss_history, start,
                &training_state, &model_arc, &rollback, &initial_snapshot, &networks_handle,
            ).await;
            return;
        }

        // Shuffle sequence indices each epoch.
        for i in (1..indices.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            indices.swap(i, j);
        }

        let mut epoch_loss = 0.0_f32;
        let mut batches = 0_usize;
        for chunk in indices.chunks(batch_size) {
            if cancel.load(Ordering::Relaxed) { break; }
            let b = chunk.len();
            // Flatten this batch into contiguous (b*n_ctx) input/target arrays.
            let mut input_flat  = Vec::with_capacity(b * n_ctx);
            let mut target_flat = Vec::with_capacity(b * n_ctx);
            for &i in chunk {
                input_flat.extend_from_slice(&sequences[i].0);
                target_flat.extend_from_slice(&sequences[i].1);
            }
            let frozen_snap = if has_frozen {
                Some(session.snapshot_frozen(frz_emb, frz_out_norm, frz_out, &frz_blocks))
            } else {
                None
            };
            let loss = session.step(&input_flat, &target_flat, b);
            if let Some(snap) = &frozen_snap {
                session.restore_frozen(snap);
            }
            if !loss.is_finite() {
                let elapsed = start.elapsed().as_secs_f32();
                let run = build_training_run_record(
                    &training_id, &network_id, &cfg, started_at,
                    epoch, loss_history.last().copied().unwrap_or(0.0),
                    elapsed, &loss_history, "error",
                );
                record_and_persist_run(&app, run).await;
                mark_error(&training_state, &training_id, &app,
                    format!("Loss diverged to {loss} at epoch {epoch}")).await;
                return;
            }
            epoch_loss += loss;
            batches += 1;
        }
        let mean_loss = epoch_loss / batches.max(1) as f32;
        loss_history.push(mean_loss);

        // Sync GPU weights to the CPU model once per epoch so the rest of
        // the app (inference, autosave, UI snapshots) sees the latest.
        {
            let mut m = model_arc.write().await;
            session.write_back(&mut m);
        }

        let elapsed = start.elapsed().as_secs_f32();
        {
            let mut s = training_state.write().await;
            s.epoch = epoch; s.last_loss = mean_loss;
            s.loss_history = loss_history.clone(); s.elapsed_secs = elapsed;
        }
        let _ = app.emit("training_update", TrainingUpdate {
            training_id: training_id.clone(),
            epoch, total_epochs: cfg.epochs,
            loss: mean_loss, loss_history: loss_history.clone(),
            elapsed_secs: elapsed,
        });
        tokio::task::yield_now().await;
    }

    let elapsed = start.elapsed().as_secs_f32();
    let final_loss = loss_history.last().copied().unwrap_or(0.0);
    let epochs_completed = loss_history.len();
    {
        let mut s = training_state.write().await;
        s.running = false; s.stopped = true; s.elapsed_secs = elapsed;
    }
    if let Some(net) = networks_handle.write().await.get_mut(&network_id) {
        net.trained = true;
        // A successful pre-training run unlocks fine-tuning. We only set this
        // on completion (not on cancel) — a half-finished pre-train shouldn't
        // count, as the user can always restart it.
        if stage == "pretrain" && epochs_completed > 0 {
            net.pretrained = true;
        }
    }
    let run = build_training_run_record(
        &training_id, &network_id, &cfg, started_at,
        epochs_completed, final_loss, elapsed, &loss_history, "completed",
    );
    record_and_persist_run(&app, run).await;
    let _ = app.emit("training_finished", TrainingFinished {
        training_id,
        status: "completed".into(),
        final_loss, total_epochs: cfg.epochs, elapsed_secs: elapsed,
    });
}

#[allow(clippy::too_many_arguments)]
async fn transformer_finish_cancelled(
    app: &AppHandle,
    training_id: &str,
    network_id: &str,
    cfg: &TrainingConfig,
    started_at: DateTime<Utc>,
    epochs_run: usize,
    loss_history: &[f32],
    start: std::time::Instant,
    training_state: &Arc<RwLock<TrainingState>>,
    model_arc: &Arc<RwLock<TransformerModel>>,
    rollback: &Arc<AtomicBool>,
    initial_snapshot: &TransformerModel,
    networks_handle: &Arc<RwLock<HashMap<String, Network>>>,
) {
    if rollback.load(Ordering::Relaxed) {
        let mut m = model_arc.write().await;
        *m = initial_snapshot.clone();
    }
    let elapsed = start.elapsed().as_secs_f32();
    let final_loss = loss_history.last().copied().unwrap_or(0.0);
    let aborted = rollback.load(Ordering::Relaxed);
    let status = if aborted { "aborted" } else { "cancelled" };
    {
        let mut s = training_state.write().await;
        s.running = false; s.stopped = true; s.cancelled = true;
        s.elapsed_secs = elapsed;
    }
    if !aborted && epochs_run > 0 {
        if let Some(net) = networks_handle.write().await.get_mut(network_id) {
            net.trained = true;
        }
    }
    let run = build_training_run_record(
        training_id, network_id, cfg, started_at,
        epochs_run, final_loss, elapsed, loss_history, status,
    );
    record_and_persist_run(app, run).await;
    let _ = app.emit("training_finished", TrainingFinished {
        training_id: training_id.to_string(),
        status: status.into(),
        final_loss, total_epochs: cfg.epochs, elapsed_secs: elapsed,
    });
}

// ─── Training ───────────────────────────────────────────────────────────────

#[tauri::command]
async fn start_training(
    app: AppHandle,
    state: State<'_, AppState>,
    req: StartTrainingRequest,
) -> Result<StartTrainingResponse, String> {
    let net = state.networks.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "Network not found".to_string())?;
    let corpus = state.corpora.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "No corpus attached. Add training data on the Corpus tab first.".to_string())?;

    // Transformer kind has its own training loop (token sequences, RoPE,
    // attention) so we dispatch before touching `model_arc`.
    if net.kind == kinds::TRANSFORMER {
        return start_transformer_training(&app, &state, net, corpus, req).await;
    }

    let model_arc = get_or_load_model(&state, &net).await?;

    let (x, y, loss_kind, total_examples) = if net.kind == kinds::FEEDFORWARD {
        let ff = corpus.feedforward.as_ref()
            .ok_or_else(|| "Feedforward network requires a feedforward corpus".to_string())?;
        let loss_kind = parse_loss(&req.config.loss)?;
        let x = Tensor::new(vec![ff.rows, ff.in_dim], ff.features.clone());
        let y = Tensor::new(vec![ff.rows, ff.out_dim], ff.targets.clone());
        (x, y, loss_kind, ff.rows)
    } else if net.kind == kinds::NEXT_TOKEN {
        let context_size = net.context_size
            .ok_or_else(|| "Next-token network missing context_size".to_string())?;
        let entry = state.vocabs.read().await.get(&req.network_id).cloned()
            .ok_or_else(|| "Vocabulary missing — build one on the Vocabulary tab first".to_string())?;
        let vocab = entry.vocab.read().await.clone();
        let stage = corpus.stage.as_deref().unwrap_or("pretrain");

        let (x, y) = match stage {
            "pretrain" => {
                let text = corpus.text.as_deref()
                    .ok_or_else(|| "Pretraining stage requires a text corpus".to_string())?;
                build_pretraining_tensors(text, &vocab, context_size)
                    .ok_or_else(|| format!(
                        "Corpus has too few tokens for context_size {context_size}"
                    ))?
            }
            "finetune" => {
                if !net.pretrained {
                    return Err(
                        "Fine-tuning requires a completed pre-training run first.".into()
                    );
                }
                let pairs_in = corpus.pairs.as_ref()
                    .ok_or_else(|| "Fine-tuning stage requires input/output pairs".to_string())?;
                if pairs_in.is_empty() {
                    return Err("Fine-tuning needs at least one input/output pair".into());
                }
                let pairs: Vec<Pair> = pairs_in.iter()
                    .map(|p| Pair { input: p.input.clone(), output: p.output.clone() })
                    .collect();
                let mask = req.config.mask_user_tokens.unwrap_or(true);
                build_finetuning_tensors(&pairs, &vocab, context_size, mask)
                    .ok_or_else(|| "Failed to build fine-tuning tensors".to_string())?
            }
            other => return Err(format!("unknown stage '{other}' (expected 'pretrain' or 'finetune')")),
        };
        let n = x.rows();
        (x, y, Loss::CrossEntropy, n)
    } else {
        return Err(format!("unsupported network kind '{}'", net.kind));
    };

    if total_examples == 0 {
        return Err("No training examples produced from the current corpus".into());
    }
    if req.config.epochs == 0 {
        return Err("epochs must be > 0".into());
    }
    // Refuse to "train" a network with no parameters — that's how next-token
    // networks end up stuck on "epoch 0 / 500" forever when the vocabulary
    // wasn't built. The model has nothing to learn, but the loop runs anyway.
    if net.parameter_count == 0 {
        return Err(
            "Network has no parameters yet. For next-token networks, set a \
             corpus (which builds the vocabulary and materialises the model) \
             before training."
                .into(),
        );
    }
    let batch_size = req.config.batch_size.max(1);

    let training_id = Uuid::new_v4().to_string();
    let started_at = Utc::now();
    let cancel = Arc::new(AtomicBool::new(false));
    let rollback = Arc::new(AtomicBool::new(false));
    let training_state = Arc::new(RwLock::new(TrainingState {
        running: true,
        total_epochs: req.config.epochs,
        ..Default::default()
    }));
    state.trainers.write().await.insert(training_id.clone(), TrainerHandle {
        state: training_state.clone(),
        cancel: cancel.clone(),
        rollback: rollback.clone(),
    });

    let networks_handle = state.networks.clone();
    let id_clone = training_id.clone();
    let cfg = req.config.clone();
    let net_id = req.network_id.clone();
    let stage_for_loop = corpus.stage.clone().unwrap_or_else(|| "pretrain".into());
    let frozen = cfg.frozen_layers.clone().unwrap_or_default();

    tokio::spawn(async move {
        run_training_loop(
            app, id_clone, net_id, cfg, started_at, model_arc, networks_handle,
            x, y, loss_kind, batch_size, training_state, cancel, rollback,
            stage_for_loop, frozen,
        ).await;
        // The TrainerHandle stays in `state.trainers` so callers can still
        // query its final status. Map cleanup happens at app shutdown.
    });

    Ok(StartTrainingResponse { training_id, status: "running".into() })
}

#[tauri::command]
async fn stop_training(
    state: State<'_, AppState>,
    training_id: String,
) -> Result<bool, String> {
    let trainers = state.trainers.read().await;
    let Some(handle) = trainers.get(&training_id).cloned() else { return Ok(false); };
    handle.cancel.store(true, Ordering::Relaxed);
    Ok(true)
}

/// Cancel training AND roll back to the pre-training weights. Used by the
/// "Abort" button — `stop_training` keeps whatever weights training has
/// produced so far, `abort_training` discards them.
#[tauri::command]
async fn abort_training(
    state: State<'_, AppState>,
    training_id: String,
) -> Result<bool, String> {
    let trainers = state.trainers.read().await;
    let Some(handle) = trainers.get(&training_id).cloned() else { return Ok(false); };
    handle.rollback.store(true, Ordering::Relaxed);
    handle.cancel.store(true, Ordering::Relaxed);
    Ok(true)
}

#[tauri::command]
async fn get_training_status(
    state: State<'_, AppState>,
    training_id: String,
) -> Result<TrainingStatusResponse, String> {
    let trainers = state.trainers.read().await;
    let h = trainers.get(&training_id)
        .ok_or_else(|| "Training session not found".to_string())?
        .clone();
    drop(trainers);
    let s = h.state.read().await;
    let status = if s.error.is_some() { "error" }
                 else if s.cancelled  { "cancelled" }
                 else if s.stopped    { "completed" }
                 else                 { "running" };
    Ok(TrainingStatusResponse {
        training_id,
        status: status.into(),
        epoch: s.epoch,
        total_epochs: s.total_epochs,
        last_loss: s.last_loss,
        loss_history: s.loss_history.clone(),
        elapsed_secs: s.elapsed_secs,
    })
}

#[allow(clippy::too_many_arguments)]
async fn run_training_loop(
    app: AppHandle,
    training_id: String,
    network_id: String,
    cfg: TrainingConfig,
    started_at: DateTime<Utc>,
    model_arc: Arc<RwLock<Model>>,
    networks_handle: Arc<RwLock<HashMap<String, Network>>>,
    x: Tensor,
    y: Tensor,
    loss_kind: Loss,
    batch_size: usize,
    training_state: Arc<RwLock<TrainingState>>,
    cancel: Arc<AtomicBool>,
    rollback: Arc<AtomicBool>,
    stage: String,
    frozen_layers: Vec<String>,
) {
    let n = x.rows();
    let in_dim = x.cols();
    let out_dim = y.cols();

    // Snapshot the pre-training weights so that "Abort" can restore them. This
    // is the only point at which we keep a long-lived clone of the model.
    let initial_snapshot: Model = {
        let m = model_arc.read().await;
        m.clone()
    };

    // Validate the optimizer config (shapes are accepted for API symmetry but
    // unused by the Burn path). Surfacing a bad optimizer string here gives the
    // UI a clean error instead of a run that silently does nothing.
    let shapes = {
        let m = model_arc.read().await;
        m.parameter_shapes()
    };
    let opt_kind = match build_optimizer(&cfg.optimizer, &shapes) {
        Ok(o) => o,
        Err(e) => {
            mark_error(&training_state, &training_id, &app, e).await;
            return;
        }
    };

    let start = std::time::Instant::now();
    let mut loss_history: Vec<f32> = Vec::with_capacity(cfg.epochs);
    let mut rng = SplitMix64::new(cfg.seed.wrapping_add(0xA5A5_5A5A_5A5A_A5A5));
    let mut indices: Vec<usize> = (0..n).collect();

    // Instantiate the WGPU device once for the entire training run. On laptops
    // without a discrete GPU, WGPU falls back to an integrated GPU; if neither
    // is available it falls back to a CPU compute pipeline.
    let device = neuralcabin_engine::default_gpu_device();

    // Build the trainer session ONCE per run. Like the transformer path, this
    // keeps the model weights and the optimizer — crucially, its Adam/SGD
    // momentum/variance buffers — resident on the GPU for the whole run. The
    // previous code called `train_step_on_device` per batch, which rebuilt the
    // optimizer from scratch every step (zeroing Adam's moments → effectively
    // LR-scaled SGD) and re-uploaded every weight to the GPU each step. That
    // both crippled convergence and added a CPU↔GPU round trip per batch.
    let mut session = neuralcabin_engine::MlpTrainerSession::<
        neuralcabin_engine::GpuAutodiffBackend,
    >::new(
        &*model_arc.read().await,
        &opt_kind,
        loss_kind,
        device,
    );

    // Parse the frozen linear-layer slots once. The session does on-device
    // snapshot/restore around each step so frozen weights don't move while
    // gradients still flow through them.
    let frozen_indices = parse_linear_frozen_keys(&frozen_layers);
    let has_frozen = !frozen_indices.is_empty();

    /// Run the cleanup path for a cancelled training run. Honours `rollback`
    /// to decide whether to keep or revert the in-progress weights, records
    /// the run in history, persists state, and emits `training_finished`.
    async fn finish_cancelled(
        app: &AppHandle,
        training_id: &str,
        network_id: &str,
        cfg: &TrainingConfig,
        started_at: DateTime<Utc>,
        epochs_run: usize,
        loss_history: &[f32],
        start: std::time::Instant,
        training_state: &Arc<RwLock<TrainingState>>,
        model_arc: &Arc<RwLock<Model>>,
        rollback: &Arc<AtomicBool>,
        initial_snapshot: &Model,
        networks_handle: &Arc<RwLock<HashMap<String, Network>>>,
    ) {
        if rollback.load(Ordering::Relaxed) {
            // Restore the model to its pre-training state. Any progress made
            // during this run is discarded — that's the whole point of Abort.
            let mut m = model_arc.write().await;
            *m = initial_snapshot.clone();
        }
        let elapsed = start.elapsed().as_secs_f32();
        let final_loss = loss_history.last().copied().unwrap_or(0.0);
        let aborted = rollback.load(Ordering::Relaxed);
        let status = if aborted { "aborted" } else { "cancelled" };
        {
            let mut s = training_state.write().await;
            s.running = false; s.stopped = true; s.cancelled = true;
            s.elapsed_secs = elapsed;
        }
        // Stop (not Abort) keeps whatever weights the model has at this point.
        // Those weights are real and the UI should reflect that the network is
        // now trained — otherwise inference and the Networks tab keep showing
        // "not trained" even though the model produces real outputs.
        if !aborted && epochs_run > 0 {
            if let Some(net) = networks_handle.write().await.get_mut(network_id) {
                net.trained = true;
            }
        }
        let run = build_training_run_record(
            training_id, network_id, cfg, started_at,
            epochs_run, final_loss, elapsed, loss_history, status,
        );
        record_and_persist_run(app, run).await;
        let _ = app.emit("training_finished", TrainingFinished {
            training_id: training_id.to_string(),
            status: status.into(),
            final_loss,
            total_epochs: cfg.epochs,
            elapsed_secs: elapsed,
        });
    }

    for epoch in 1..=cfg.epochs {
        if cancel.load(Ordering::Relaxed) {
            // Sync the latest GPU weights back unless we're rolling back anyway.
            if !rollback.load(Ordering::Relaxed) {
                let mut m = model_arc.write().await;
                session.write_back(&mut m);
            }
            finish_cancelled(
                &app, &training_id, &network_id, &cfg, started_at,
                epoch.saturating_sub(1), &loss_history, start,
                &training_state, &model_arc, &rollback, &initial_snapshot,
                &networks_handle,
            ).await;
            return;
        }

        // Shuffle each epoch (deterministic from seed).
        for i in (1..indices.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            indices.swap(i, j);
        }

        let mut epoch_loss = 0.0_f32;
        let mut batches = 0usize;
        for chunk in indices.chunks(batch_size) {
            if cancel.load(Ordering::Relaxed) { break; }
            let mut bx = Vec::with_capacity(chunk.len() * in_dim);
            let mut by = Vec::with_capacity(chunk.len() * out_dim);
            for &i in chunk {
                bx.extend_from_slice(&x.data[i * in_dim..(i + 1) * in_dim]);
                by.extend_from_slice(&y.data[i * out_dim..(i + 1) * out_dim]);
            }
            let bx = Tensor::new(vec![chunk.len(), in_dim], bx);
            let by = Tensor::new(vec![chunk.len(), out_dim], by);
            // Snapshot frozen layers, run the optimiser step on the resident
            // session, then restore — gradients flow through frozen layers but
            // their weights don't move. The session keeps the model on the GPU
            // for the whole run; we sync back to the CPU model once per epoch.
            let frozen_snap = if has_frozen {
                Some(session.snapshot_frozen(&frozen_indices))
            } else {
                None
            };
            let loss = session.step(&bx, &by);
            if let Some(snap) = &frozen_snap {
                session.restore_frozen(snap);
            }
            if !loss.is_finite() {
                let elapsed = start.elapsed().as_secs_f32();
                let run = build_training_run_record(
                    &training_id, &network_id, &cfg, started_at,
                    epoch, loss_history.last().copied().unwrap_or(0.0),
                    elapsed, &loss_history, "error",
                );
                record_and_persist_run(&app, run).await;
                mark_error(&training_state, &training_id, &app,
                    format!("Loss diverged to {loss} at epoch {epoch}")).await;
                return;
            }
            epoch_loss += loss;
            batches += 1;
        }
        let mean_loss = epoch_loss / batches.max(1) as f32;
        loss_history.push(mean_loss);

        // Sync the GPU weights to the CPU model once per epoch so inference,
        // autosave, and UI snapshots see the latest weights.
        {
            let mut m = model_arc.write().await;
            session.write_back(&mut m);
        }

        let elapsed = start.elapsed().as_secs_f32();
        {
            let mut s = training_state.write().await;
            s.epoch = epoch;
            s.last_loss = mean_loss;
            s.loss_history = loss_history.clone();
            s.elapsed_secs = elapsed;
        }
        let _ = app.emit("training_update", TrainingUpdate {
            training_id: training_id.clone(),
            epoch,
            total_epochs: cfg.epochs,
            loss: mean_loss,
            loss_history: loss_history.clone(),
            elapsed_secs: elapsed,
        });

        tokio::task::yield_now().await;
    }

    let elapsed = start.elapsed().as_secs_f32();
    let final_loss = loss_history.last().copied().unwrap_or(0.0);
    let epochs_completed = loss_history.len();
    {
        let mut s = training_state.write().await;
        s.running = false; s.stopped = true;
        s.elapsed_secs = elapsed;
    }
    if let Some(net) = networks_handle.write().await.get_mut(&network_id) {
        net.trained = true;
        if stage == "pretrain" && epochs_completed > 0 {
            net.pretrained = true;
        }
    }
    // Record the run in history and persist everything (trained weights,
    // `trained = true` flag, and the new history entry) in one shot.
    let run = build_training_run_record(
        &training_id, &network_id, &cfg, started_at,
        epochs_completed, final_loss, elapsed, &loss_history, "completed",
    );
    record_and_persist_run(&app, run).await;
    let _ = app.emit("training_finished", TrainingFinished {
        training_id,
        status: "completed".into(),
        final_loss,
        total_epochs: cfg.epochs,
        elapsed_secs: elapsed,
    });
}

#[tauri::command]
async fn get_training_history(
    state: State<'_, AppState>,
    network_id: String,
) -> Result<Vec<TrainingRun>, String> {
    let history = state.training_history.read().await;
    let mut runs = history.get(&network_id).cloned().unwrap_or_default();
    // Return newest first.
    runs.reverse();
    Ok(runs)
}

#[tauri::command]
async fn clear_training_history(
    state: State<'_, AppState>,
    network_id: String,
) -> Result<(), String> {
    state.training_history.write().await.remove(&network_id);
    persist(&state).await?;
    Ok(())
}

async fn mark_error(
    state: &Arc<RwLock<TrainingState>>,
    training_id: &str,
    app: &AppHandle,
    msg: String,
) {
    {
        let mut s = state.write().await;
        s.error = Some(msg.clone());
        s.running = false; s.stopped = true;
    }
    let _ = app.emit("training_error", TrainingError {
        training_id: training_id.to_string(),
        message: msg,
    });
}

// ─── Inference ──────────────────────────────────────────────────────────────

async fn infer_transformer(
    app: &AppHandle,
    state: &State<'_, AppState>,
    net: Network,
    req: InferRequest,
) -> Result<InferResponse, String> {
    use neuralcabin_engine::tokenizer::{ASSISTANT_ID, EOS_ID, USER_ID, RESERVED};
    let hp = net.transformer.clone()
        .ok_or_else(|| "transformer network missing hparams".to_string())?;
    let n_ctx = hp.n_ctx;
    let entry = state.vocabs.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "Vocabulary missing — save a corpus first".to_string())?;
    let model_arc = get_or_load_transformer(state, &net).await?;

    let prompt = req.prompt.unwrap_or_default();
    let max_new = req.max_new_tokens.unwrap_or(64).clamp(1, 4096);
    let temperature = req.temperature.unwrap_or(0.0).max(0.0);
    let chat_mode = state.corpora.read().await
        .get(&req.network_id)
        .and_then(|c| c.stage.clone())
        .map(|s| s == "finetune")
        .unwrap_or(false);
    let messages = req.messages.clone();

    let inference_id = Uuid::new_v4().to_string();
    let cancel = Arc::new(AtomicBool::new(false));
    state.inferrers.write().await.insert(inference_id.clone(), InferenceHandle {
        cancel: cancel.clone(),
    });

    let app_clone = app.clone();
    let inferrers_handle = state.inferrers.clone();
    let id_clone = inference_id.clone();

    tokio::spawn(async move {
        let vocab = entry.vocab.read().await.clone();
        let mut ids: Vec<u32> = Vec::new();
        if let Some(msgs) = messages.as_ref().filter(|_| chat_mode) {
            for m in msgs {
                match m.role.as_str() {
                    "user"      => ids.push(USER_ID),
                    "assistant" => ids.push(ASSISTANT_ID),
                    _ => continue,
                }
                ids.extend(vocab.encode(&m.text));
                ids.push(EOS_ID);
            }
            ids.push(ASSISTANT_ID);
        } else if chat_mode {
            ids.push(USER_ID);
            ids.extend(vocab.encode(&prompt));
            ids.push(EOS_ID);
            ids.push(ASSISTANT_ID);
        } else {
            ids.extend(vocab.encode(&prompt));
        }
        let initial_len = ids.len();
        let mut generated = String::new();
        let mut rng = SplitMix64::new(0xDEAD_BEEF_u64 ^ (max_new as u64).wrapping_mul(0x9E3779B1));
        let device = neuralcabin_engine::default_gpu_device();

        for index in 0..max_new {
            if cancel.load(Ordering::Relaxed) {
                let _ = app_clone.emit("inference_finished", InferenceFinished {
                    inference_id: id_clone.clone(),
                    status: "cancelled".into(),
                    generated: generated.clone(),
                    token_count: ids.len() - initial_len,
                });
                break;
            }
            // Take the last n_ctx tokens; pad-left with EOS if shorter.
            let mut window: Vec<u32> = if ids.len() >= n_ctx {
                ids[ids.len() - n_ctx..].to_vec()
            } else {
                let mut w = vec![EOS_ID; n_ctx - ids.len()];
                w.extend_from_slice(&ids);
                w
            };
            // forward returns logits for every position; we want the LAST one.
            let model = model_arc.read().await;
            let logits = neuralcabin_engine::transformer::forward_logits::<
                neuralcabin_engine::GpuBackend,
            >(&model, &window, &device);
            drop(model);
            let vocab_sz = vocab.size();
            let last_row = &logits[logits.len() - vocab_sz..];
            // softmax for sampling.
            let mut row = last_row.to_vec();
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            for v in row.iter_mut() { *v = (*v - mx).exp(); }
            let s: f32 = row.iter().sum();
            for v in row.iter_mut() { *v /= s; }

            let chosen = if temperature <= f32::EPSILON {
                argmax_local(&row)
            } else {
                sample_with_temperature_local(&row, temperature, &mut rng)
            };
            let prob = row[chosen];
            let token = vocab.token_of(chosen as u32).to_string();
            let visible = if (chosen as u32) < RESERVED as u32 { String::new() } else { token.clone() };
            generated.push_str(&visible);
            let _ = app_clone.emit("inference_token", InferenceToken {
                inference_id: id_clone.clone(),
                index, token: token.clone(), probability: prob,
            });
            if (chosen as u32) == EOS_ID { break; }
            ids.push(chosen as u32);
            // Suppress unused warning while we keep the variable for future
            // KV-cache work — the window is recomputed every step today.
            let _ = &mut window;
            tokio::task::yield_now().await;
        }
        let _ = app_clone.emit("inference_finished", InferenceFinished {
            inference_id: id_clone.clone(),
            status: "completed".into(),
            generated,
            token_count: ids.len() - initial_len,
        });
        inferrers_handle.write().await.remove(&id_clone);
    });

    Ok(InferResponse { output: None, inference_id: Some(inference_id) })
}

#[tauri::command]
async fn infer(
    app: AppHandle,
    state: State<'_, AppState>,
    req: InferRequest,
) -> Result<InferResponse, String> {
    let net = state.networks.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "Network not found".to_string())?;

    if net.kind == kinds::TRANSFORMER {
        return infer_transformer(&app, &state, net, req).await;
    }

    let model_arc = get_or_load_model(&state, &net).await?;

    if net.kind == kinds::FEEDFORWARD {
        let model = model_arc.read().await;
        let features = req.features
            .ok_or_else(|| "features array required for feedforward inference".to_string())?;
        if features.len() != net.input_dim {
            return Err(format!(
                "features length {} doesn't match input_dim {}",
                features.len(), net.input_dim
            ));
        }
        let x = Tensor::new(vec![1, net.input_dim], features);
        let y = model.predict(&x);
        Ok(InferResponse { output: Some(y.data), inference_id: None })
    } else if net.kind == kinds::NEXT_TOKEN {
        let prompt = req.prompt.unwrap_or_default();
        let max_new = req.max_new_tokens.unwrap_or(64).clamp(1, 4096);
        let temperature = req.temperature.unwrap_or(0.0).max(0.0);
        let context_size = net.context_size
            .ok_or_else(|| "Next-token network missing context_size".to_string())?;
        let entry = state.vocabs.read().await.get(&req.network_id).cloned()
            .ok_or_else(|| "Vocabulary missing — build one on the Vocabulary tab first".to_string())?;

        // For fine-tuned chat models the prompt is wrapped in chat turn
        // markers so the model picks up where the training data left off:
        //   <user> {prompt} <eos> <assistant> ...
        // Generation continues until the assistant's own <eos>.
        let chat_mode = state.corpora.read().await
            .get(&req.network_id)
            .and_then(|c| c.stage.clone())
            .map(|s| s == "finetune")
            .unwrap_or(false);

        let messages = req.messages.clone();
        if let Some(ms) = &messages {
            if ms.is_empty() {
                return Err("messages array must not be empty".into());
            }
            if ms.last().map(|m| m.role.as_str()) != Some("user") {
                return Err("the last message must have role 'user'".into());
            }
        }

        let inference_id = Uuid::new_v4().to_string();
        let cancel = Arc::new(AtomicBool::new(false));
        state.inferrers.write().await.insert(inference_id.clone(), InferenceHandle {
            cancel: cancel.clone(),
        });

        let inferrers_handle = state.inferrers.clone();
        let id_clone = inference_id.clone();
        tokio::spawn(async move {
            run_generation(
                app, id_clone.clone(), model_arc, entry.vocab,
                prompt, messages, context_size, max_new, temperature, chat_mode, cancel,
            ).await;
            inferrers_handle.write().await.remove(&id_clone);
        });

        Ok(InferResponse { output: None, inference_id: Some(inference_id) })
    } else {
        Err(format!("unsupported network kind '{}'", net.kind))
    }
}

#[derive(serde::Serialize)]
pub struct InferActivationsResponse {
    /// `layers[i]` is the i-th layer description (e.g. "Linear (2 -> 8)") or
    /// "input" for the very first entry.
    pub layer_names: Vec<String>,
    /// Activation values per layer. `activations[i]` has length `dims[i]`.
    pub activations: Vec<Vec<f32>>,
    pub dims: Vec<usize>,
}

#[tauri::command]
async fn infer_with_activations(
    state: State<'_, AppState>,
    network_id: String,
    features: Vec<f32>,
) -> Result<InferActivationsResponse, String> {
    let net = state.networks.read().await.get(&network_id).cloned()
        .ok_or_else(|| "Network not found".to_string())?;
    if net.kind != kinds::FEEDFORWARD {
        return Err("infer_with_activations is only supported for feedforward networks".into());
    }
    if features.len() != net.input_dim {
        return Err(format!(
            "features length {} doesn't match input_dim {}",
            features.len(), net.input_dim
        ));
    }
    let model_arc = get_or_load_model(&state, &net).await?;
    let model = model_arc.read().await;
    let x = Tensor::new(vec![1, net.input_dim], features);
    let acts = model.predict_with_activations(&x);

    let mut layer_names = Vec::with_capacity(acts.len());
    layer_names.push("input".to_string());
    for l in &model.layers {
        layer_names.push(l.describe());
    }
    let dims: Vec<usize> = acts.iter().map(|t| t.cols()).collect();
    let activations: Vec<Vec<f32>> = acts.into_iter().map(|t| t.data).collect();
    Ok(InferActivationsResponse { layer_names, activations, dims })
}

#[tauri::command]
async fn stop_inference(
    state: State<'_, AppState>,
    inference_id: String,
) -> Result<bool, String> {
    let h = state.inferrers.read().await.get(&inference_id).cloned();
    if let Some(h) = h {
        h.cancel.store(true, Ordering::Relaxed);
        Ok(true)
    } else { Ok(false) }
}

#[allow(clippy::too_many_arguments)]
async fn run_generation(
    app: AppHandle,
    inference_id: String,
    model_arc: Arc<RwLock<Model>>,
    vocab_arc: Arc<RwLock<Vocabulary>>,
    prompt: String,
    messages: Option<Vec<ChatMessage>>,
    context_size: usize,
    max_new_tokens: usize,
    temperature: f32,
    chat_mode: bool,
    cancel: Arc<AtomicBool>,
) {
    let model  = model_arc.read().await;
    let vocab  = vocab_arc.read().await;
    let mut ids: Vec<u32> = Vec::new();
    if let Some(msgs) = messages.as_ref().filter(|_| chat_mode) {
        // Multi-turn chat: replay the full conversation with proper turn
        // markers so the model sees the same shape as its training data.
        // Sequence ends with `<assistant>`, which is where generation begins.
        for m in msgs {
            match m.role.as_str() {
                "user"      => ids.push(USER_ID),
                "assistant" => ids.push(ASSISTANT_ID),
                _ => continue,
            }
            ids.extend(vocab.encode(&m.text));
            ids.push(EOS_ID);
        }
        ids.push(ASSISTANT_ID);
    } else if chat_mode {
        // Single-turn chat: <user> {prompt} <eos> <assistant>
        ids.push(USER_ID);
        ids.extend(vocab.encode(&prompt));
        ids.push(EOS_ID);
        ids.push(ASSISTANT_ID);
    } else {
        ids.extend(vocab.encode(&prompt));
    }
    let initial_len = ids.len();
    let mut generated = String::new();
    let mut rng = SplitMix64::new(0xDEAD_BEEF_u64 ^ (max_new_tokens as u64).wrapping_mul(0x9E3779B1));

    for index in 0..max_new_tokens {
        if cancel.load(Ordering::Relaxed) {
            let _ = app.emit("inference_finished", InferenceFinished {
                inference_id: inference_id.clone(),
                status: "cancelled".into(),
                generated: generated.clone(),
                token_count: ids.len() - initial_len,
            });
            return;
        }

        let ctx = encode_context(&ids, &vocab, context_size);
        let logits = model.predict(&ctx);
        let probs = softmax_rows(&logits);
        let row = &probs.data[..probs.cols()];

        let chosen = if temperature <= f32::EPSILON {
            argmax(row)
        } else {
            sample_with_temperature(row, temperature, &mut rng)
        };
        let prob = row[chosen];
        let token = vocab.token_of(chosen as u32).to_string();

        // Emit one token at a time for streaming UI.
        let visible_token = if (chosen as u32) < neuralcabin_engine::tokenizer::RESERVED as u32 {
            // Don't append reserved tokens to generated text, but still report them.
            String::new()
        } else {
            token.clone()
        };
        generated.push_str(&visible_token);

        let _ = app.emit("inference_token", InferenceToken {
            inference_id: inference_id.clone(),
            index, token: token.clone(), probability: prob,
        });

        if (chosen as u32) == EOS_ID { break; }
        ids.push(chosen as u32);

        // Cooperative yield so the cancel flag can be observed promptly.
        tokio::task::yield_now().await;
    }

    let _ = app.emit("inference_finished", InferenceFinished {
        inference_id: inference_id.clone(),
        status: "completed".into(),
        generated,
        token_count: ids.len() - initial_len,
    });
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv { bv = v; best = i; }
    }
    best
}

fn sample_with_temperature(probs: &[f32], temperature: f32, rng: &mut SplitMix64) -> usize {
    let mut logits: Vec<f32> = probs.iter().map(|p| p.max(1e-12).ln() / temperature).collect();
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in logits.iter_mut() { *v = (*v - mx).exp(); sum += *v; }
    for v in logits.iter_mut() { *v /= sum; }

    let r = (rng.next_u64() as f64 / u64::MAX as f64) as f32;
    let mut acc = 0.0_f32;
    for (i, &p) in logits.iter().enumerate() {
        acc += p;
        if r <= acc { return i; }
    }
    logits.len() - 1
}

// ─── Tauri entry point ──────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let state = AppState::new();
    tauri::Builder::default()
        .manage(state)
        .setup(|app| {
            // Resolve the per-user data directory and load any saved workspace
            // synchronously before the frontend can issue its first command —
            // otherwise `list_networks` would briefly return an empty list and
            // a fast user could overwrite an existing save.
            let data_dir = app.path().app_data_dir()
                .unwrap_or_else(|_| std::env::temp_dir().join("neuralcabin"));
            let state: State<'_, AppState> = app.state();
            let app_state: &AppState = state.inner();
            tauri::async_runtime::block_on(async {
                *app_state.data_dir.write().await = Some(data_dir.clone());
                match persistence::load_from_dir(&data_dir).await {
                    Ok(Some(persisted)) => {
                        persistence::apply(app_state, persisted).await;
                        // Cross-check 'trained' flags: if a network is marked
                        // trained but its weight file is missing, reset the flag
                        // so the UI reflects reality (models are no longer
                        // embedded in state.json — they live in models/<id>.json).
                        persistence::reset_missing_model_flags(app_state, &data_dir).await;
                        // Re-save immediately in the current compact format.
                        // This is a one-time migration: if the user had a large
                        // legacy state.json with inline model weights, this
                        // replaces it with a small metadata-only file so that
                        // the next launch loads in milliseconds.
                        if let Err(e) = persistence::save_to_dir(app_state, &data_dir).await {
                            eprintln!("[neuralcabin] post-load migration save failed: {e}");
                        }
                        eprintln!("[neuralcabin] loaded state from {}", data_dir.display());
                    }
                    Ok(None) => {
                        eprintln!("[neuralcabin] no saved state at {} — starting fresh", data_dir.display());
                    }
                    Err(e) => {
                        // Corrupt file: surface loudly. We don't delete it so
                        // the user can recover by editing or removing it.
                        eprintln!("[neuralcabin] ERROR loading state from {}: {e}", data_dir.display());
                    }
                }
            });
            // Auto-start any servers that were configured with `auto_start`.
            // This runs after the block_on above so app state is fully populated.
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                autostart_saved_servers(&app_handle).await;
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            create_network, list_networks, get_network, delete_network,
            set_corpus, get_corpus, corpus_stats,
            build_vocabulary, set_advanced_vocabulary, get_vocabulary, tokenize_preview,
            start_training, stop_training, abort_training, get_training_status,
            get_training_history, clear_training_history,
            infer, infer_with_activations, stop_inference,
            export_network,
            list_servers, create_server, update_server, delete_server,
            start_server, stop_server, server_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

// ─── Export commands ────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct ExportPayload {
    pub format: String,
    pub filename: String,
    /// Base64-encoded file contents. The frontend turns this into a Blob and
    /// triggers a browser download — no plugin permissions required.
    pub data_b64: String,
    pub size_bytes: usize,
}

#[tauri::command]
async fn export_network(
    state: State<'_, AppState>,
    network_id: String,
    format: String,
) -> Result<ExportPayload, String> {
    use base64::Engine as _;
    let net = state.networks.read().await.get(&network_id).cloned()
        .ok_or_else(|| "Network not found".to_string())?;
    if !export::FORMATS.contains(&format.as_str()) {
        return Err(format!("unknown format '{format}' (expected pytorch|onnx|gguf)"));
    }
    if net.kind == kinds::FEEDFORWARD && format == "gguf" {
        return Err("GGUF export is only meaningful for text-generation networks".into());
    }
    let bytes = if net.kind == kinds::TRANSFORMER {
        let model_arc = get_or_load_transformer(&state, &net).await?;
        let model = model_arc.read().await.clone();
        let tokens: Vec<String> = state.vocabs.read().await.get(&network_id)
            .map(|e| e.info.tokens.clone())
            .ok_or_else(|| "Vocabulary missing for transformer — save a corpus first".to_string())?;
        export::export_transformer(&format, &net, &model, &tokens)?
    } else {
        let model_arc = get_or_load_model(&state, &net).await?;
        let model = model_arc.read().await.clone();
        export::export_model(&format, &net, &model)?
    };
    let ext = export::extension_for(&format);
    let safe_name: String = net.name.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    Ok(ExportPayload {
        format,
        filename: format!("{safe_name}.{ext}"),
        size_bytes: bytes.len(),
        data_b64: base64::engine::general_purpose::STANDARD.encode(&bytes),
    })
}

// ─── Server commands ────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ServerSummary {
    #[serde(flatten)]
    pub config: ServerConfig,
    pub running: bool,
    pub request_count: u64,
    pub last_error: Option<String>,
}

async fn summarize(rt: &ServerRuntime) -> ServerSummary {
    let s = rt.status().await;
    ServerSummary {
        config: rt.config.clone(),
        running: s.running,
        request_count: s.request_count,
        last_error: s.last_error,
    }
}

#[tauri::command]
async fn list_servers(state: State<'_, AppState>) -> Result<Vec<ServerSummary>, String> {
    let servers = state.servers.read().await;
    let mut out = Vec::with_capacity(servers.len());
    for rt in servers.values() { out.push(summarize(rt).await); }
    out.sort_by(|a, b| a.config.created_at.cmp(&b.config.created_at));
    Ok(out)
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct CreateServerRequest {
    pub name: String,
    pub port: u16,
    #[serde(default)] pub localhost_only: Option<bool>,
    #[serde(default)] pub auth_token: Option<String>,
    #[serde(default)] pub permissions: Option<ServerPermissions>,
}

#[tauri::command]
async fn create_server(
    state: State<'_, AppState>,
    req: CreateServerRequest,
) -> Result<ServerSummary, String> {
    if req.name.trim().is_empty() { return Err("server name cannot be empty".into()); }
    if req.port == 0 { return Err("port must be > 0".into()); }

    let mut cfg = server::new_config(req.name.trim().into(), req.port);
    if let Some(v) = req.localhost_only { cfg.localhost_only = v; }
    if let Some(v) = req.auth_token { cfg.auth_token = v; }
    if let Some(v) = req.permissions { cfg.permissions = v; }

    // Port collision check (a server can reuse the port only if not running).
    {
        let servers = state.servers.read().await;
        for rt in servers.values() {
            if rt.config.port == cfg.port && rt.running.load(Ordering::Relaxed) {
                return Err(format!("port {} is in use by '{}'", cfg.port, rt.config.name));
            }
        }
    }

    let rt = ServerRuntime::new(cfg.clone());
    let summary = summarize(&rt).await;
    state.servers.write().await.insert(cfg.id.clone(), rt);
    persist(&state).await?;
    Ok(summary)
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct UpdateServerRequest {
    pub id: String,
    #[serde(default)] pub name: Option<String>,
    #[serde(default)] pub port: Option<u16>,
    #[serde(default)] pub localhost_only: Option<bool>,
    #[serde(default)] pub auth_token: Option<String>,
    #[serde(default)] pub permissions: Option<ServerPermissions>,
    #[serde(default)] pub auto_start: Option<bool>,
}

#[tauri::command]
async fn update_server(
    state: State<'_, AppState>,
    req: UpdateServerRequest,
) -> Result<ServerSummary, String> {
    let mut servers = state.servers.write().await;
    let rt = servers.get_mut(&req.id).ok_or_else(|| "server not found".to_string())?;
    if rt.running.load(Ordering::Relaxed) {
        return Err("stop the server before editing it".into());
    }
    if let Some(n) = req.name { if !n.trim().is_empty() { rt.config.name = n.trim().into(); } }
    if let Some(p) = req.port { if p > 0 { rt.config.port = p; } }
    if let Some(v) = req.localhost_only { rt.config.localhost_only = v; }
    if let Some(t) = req.auth_token { rt.config.auth_token = t; }
    if let Some(p) = req.permissions { rt.config.permissions = p; }
    if let Some(v) = req.auto_start { rt.config.auto_start = v; }
    let summary = summarize(rt).await;
    drop(servers);
    persist(&state).await?;
    Ok(summary)
}

#[tauri::command]
async fn delete_server(state: State<'_, AppState>, id: String) -> Result<bool, String> {
    let mut servers = state.servers.write().await;
    if let Some(mut rt) = servers.remove(&id) {
        let _ = server::stop(&mut rt).await;
        drop(servers);
        persist(&state).await?;
        Ok(true)
    } else {
        Ok(false)
    }
}

#[tauri::command]
async fn start_server(
    app: AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<ServerSummary, String> {
    let mut servers = state.servers.write().await;
    let rt = servers.get_mut(&id).ok_or_else(|| "server not found".to_string())?;
    server::start(app.clone(), rt).await?;
    let summary = summarize(rt).await;
    drop(servers);
    // Auto-start should be on for any server the user explicitly starts; this
    // is what makes the server come back on app restart.
    {
        let mut servers = state.servers.write().await;
        if let Some(rt) = servers.get_mut(&id) { rt.config.auto_start = true; }
    }
    persist(&state).await?;
    Ok(summary)
}

#[tauri::command]
async fn stop_server(state: State<'_, AppState>, id: String) -> Result<ServerSummary, String> {
    let mut servers = state.servers.write().await;
    let rt = servers.get_mut(&id).ok_or_else(|| "server not found".to_string())?;
    server::stop(rt).await?;
    rt.config.auto_start = false;
    let summary = summarize(rt).await;
    drop(servers);
    persist(&state).await?;
    Ok(summary)
}

#[tauri::command]
async fn server_status(state: State<'_, AppState>, id: String) -> Result<ServerSummary, String> {
    let servers = state.servers.read().await;
    let rt = servers.get(&id).ok_or_else(|| "server not found".to_string())?;
    Ok(summarize(rt).await)
}

/// Auto-start servers persisted from a previous session. Only servers that
/// had `auto_start = true` (i.e., were running when state.json was last
/// written) are restarted.
async fn autostart_saved_servers(app: &AppHandle) {
    let state = app.state::<AppState>();
    let ids: Vec<String> = {
        let servers = state.servers.read().await;
        servers.values()
            .filter(|rt| rt.config.auto_start && rt.config.port != 0)
            .map(|rt| rt.config.id.clone())
            .collect()
    };
    for id in ids {
        let mut servers = state.servers.write().await;
        if let Some(rt) = servers.get_mut(&id) {
            if let Err(e) = server::start(app.clone(), rt).await {
                eprintln!("[neuralcabin] autostart server {}: {e}", rt.config.name);
            } else {
                eprintln!("[neuralcabin] autostarted server '{}' on port {}", rt.config.name, rt.config.port);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuralcabin_engine::tensor::SplitMix64;

    #[test]
    fn layers_match_specs_accepts_identical_architectures() {
        let mut rng = SplitMix64::new(1);
        let layers = vec![
            Layer::Linear(neuralcabin_engine::nn::LinearLayer::new(4, 8, &mut rng)),
            Layer::Activation(Activation::ReLU),
            Layer::Linear(neuralcabin_engine::nn::LinearLayer::new(8, 2, &mut rng)),
        ];
        let specs = vec![
            LayerSpec::Linear { in_dim: 4, out_dim: 8 },
            LayerSpec::Activation(Activation::ReLU),
            LayerSpec::Linear { in_dim: 8, out_dim: 2 },
        ];
        assert!(layers_match_specs(&layers, &specs));
    }

    #[test]
    fn layers_match_specs_rejects_dim_change() {
        let mut rng = SplitMix64::new(1);
        let layers = vec![
            Layer::Linear(neuralcabin_engine::nn::LinearLayer::new(4, 8, &mut rng)),
        ];
        // Different in_dim — must rebuild (e.g. vocab size grew).
        let specs = vec![LayerSpec::Linear { in_dim: 5, out_dim: 8 }];
        assert!(!layers_match_specs(&layers, &specs));
    }

    #[test]
    fn layers_match_specs_rejects_activation_change() {
        let layers = vec![Layer::Activation(Activation::ReLU)];
        let specs = vec![LayerSpec::Activation(Activation::Sigmoid)];
        assert!(!layers_match_specs(&layers, &specs));
    }

    #[test]
    fn layers_match_specs_rejects_length_change() {
        let mut rng = SplitMix64::new(1);
        let layers = vec![
            Layer::Linear(neuralcabin_engine::nn::LinearLayer::new(4, 8, &mut rng)),
        ];
        let specs = vec![
            LayerSpec::Linear { in_dim: 4, out_dim: 8 },
            LayerSpec::Activation(Activation::ReLU),
        ];
        assert!(!layers_match_specs(&layers, &specs));
    }

    /// Regression test for the "training disappears after saving the corpus"
    /// bug. The key invariant: when the architecture is unchanged, the helper
    /// must report a match so the caller skips rebuilding (and thus preserves
    /// trained weights).
    #[test]
    fn matching_architectures_preserve_existing_model() {
        // Build a model the way `rebuild_next_token_model` would.
        let specs = vec![
            LayerSpec::Linear { in_dim: 16, out_dim: 8 },
            LayerSpec::Activation(Activation::ReLU),
            LayerSpec::Linear { in_dim: 8, out_dim: 4 },
        ];
        let model = Model::from_specs(16, &specs, 42);

        // Pretend the user re-saves the corpus — same vocab size, same hidden
        // chain. The helper must say "matches" so the model is kept.
        assert!(layers_match_specs(&model.layers, &specs));
    }
}

