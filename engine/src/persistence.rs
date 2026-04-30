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

/// On-disk container — bumps `format_version` if we ever change the layout.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFile {
    pub format_version: u32,
    pub created_with: String,
    pub model: Model,
    pub loss: Option<Loss>,
    pub optimizer: Option<OptimizerKind>,
    pub notes: Option<String>,
}

pub const FORMAT_VERSION: u32 = 1;

impl ModelFile {
    pub fn wrap(model: Model, loss: Option<Loss>, optimizer: Option<OptimizerKind>) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            created_with: format!("neuralcabin {}", env!("CARGO_PKG_VERSION")),
            model,
            loss,
            optimizer,
            notes: None,
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
    if file.format_version != FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported model format_version {} (expected {})",
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
        assert_eq!(f.model.input_dim, 2);
        assert_eq!(f.model.layers.len(), 3);
        assert_eq!(f.model.output_dim(), 1);
        let _ = std::fs::remove_file(&path);
    }
}
