//! Model persistence using a small custom JSON envelope.
//!
//! We deliberately avoid ML-specific formats like `safetensors` — the engine is
//! self-contained, so we just lean on `serde_json`.

use crate::loss::Loss;
use crate::nn::Model;
use crate::optimizer::OptimizerKind;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;

/// On-disk container — bumps `format_version` when the layout changes.
/// All v2+ fields are `Option` with `#[serde(default)]` so v1 files still
/// deserialise cleanly (missing fields become `None`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFile {
    pub format_version: u32,
    pub created_with: String,
    pub model: Model,
    pub loss: Option<Loss>,
    pub optimizer: Option<OptimizerKind>,
    pub notes: Option<String>,

    // ── Added in v2: full network state ──────────────────────────────────────
    /// "Simplex" | "NextTokenGen" | "Plugin:<id>:<type>"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network_kind: Option<String>,

    /// Human-readable name of the network (usually the file stem).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network_name: Option<String>,

    /// Init seed used when building the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    // ── Corpus (text networks) ────────────────────────────────────────────────
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus_text_body: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus_text_paths: Option<Vec<String>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus_context_size: Option<usize>,

    // ── Vocabulary ───────────────────────────────────────────────────────────
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_tokens: Option<Vec<String>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_mode: Option<String>,

    // ── Embedding ────────────────────────────────────────────────────────────
    /// "OneHot" | "TfIdf" | "FastText" | "Transformer"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_kind: Option<String>,

    /// Dense embedding dimension (FastText / Transformer modes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed_dim: Option<usize>,
}

pub const FORMAT_VERSION: u32 = 2;

impl ModelFile {
    pub fn wrap(model: Model, loss: Option<Loss>, optimizer: Option<OptimizerKind>) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            created_with: format!("neuralcabin {}", env!("CARGO_PKG_VERSION")),
            model,
            loss,
            optimizer,
            notes: None,
            network_kind: None,
            network_name: None,
            seed: None,
            corpus_text_body: None,
            corpus_text_paths: None,
            corpus_context_size: None,
            vocab_tokens: None,
            vocab_mode: None,
            embedding_kind: None,
            embed_dim: None,
        }
    }
}

pub fn save<P: AsRef<Path>>(path: P, file: &ModelFile) -> io::Result<()> {
    let s = serde_json::to_string_pretty(file)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(path, s)
}

pub fn load<P: AsRef<Path>>(path: P) -> io::Result<ModelFile> {
    let raw = fs::read_to_string(path)?;
    let file: ModelFile = serde_json::from_str(&raw)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    // Accept any version we understand; reject future versions from newer builds.
    if file.format_version > FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported model format_version {} (this build understands up to {})",
                file.format_version, FORMAT_VERSION
            ),
        ));
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::nn::LayerSpec;

    #[test]
    fn save_and_load_roundtrip() {
        let specs = vec![
            LayerSpec::Linear { in_dim: 2, out_dim: 4 },
            LayerSpec::Activation(Activation::ReLU),
            LayerSpec::Linear { in_dim: 4, out_dim: 1 },
        ];
        let model = Model::from_specs(2, &specs, 7);
        let path = std::env::temp_dir().join("neuralcabin_test_model.json");
        save(&path, &ModelFile::wrap(model.clone(), Some(Loss::MeanSquaredError), None)).unwrap();
        let f = load(&path).unwrap();
        assert_eq!(f.format_version, FORMAT_VERSION);
        assert_eq!(f.model.input_dim, 2);
        assert_eq!(f.model.layers.len(), 3);
        assert_eq!(f.model.output_dim(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn v1_file_loads_cleanly() {
        // Build a real v2 file, downgrade it to format_version=1 (stripping new
        // optional fields), and verify it still deserialises without error.
        let specs = vec![
            LayerSpec::Linear { in_dim: 2, out_dim: 4 },
            LayerSpec::Activation(Activation::ReLU),
        ];
        let model = Model::from_specs(2, &specs, 42);
        let mut file = ModelFile::wrap(model, None, None);

        // Downgrade: set version to 1 and strip v2-only fields.
        file.format_version = 1;
        file.network_kind   = None;
        file.vocab_tokens   = None;
        let json = serde_json::to_string(&file).unwrap();

        let loaded: ModelFile = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.format_version, 1);
        assert!(loaded.network_kind.is_none());
        assert!(loaded.vocab_tokens.is_none());
        assert_eq!(loaded.model.input_dim, 2);
    }
}
