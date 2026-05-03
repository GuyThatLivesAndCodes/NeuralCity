//! Per-network corpus / training-data definition.
//!
//! Each network owns a `Corpus`. For numeric networks (Simplex / plugin) the
//! corpus is a numeric `Dataset`; for next-token-generation networks it is a
//! body of text plus an arbitrary set of uploaded files.

use crate::networks::EmbeddingKind;
use crate::vocab::Vocab;
use neuralcabin_engine::data::{self, Dataset, TaskKind};
use neuralcabin_engine::tensor::Tensor;
use std::fs;
use std::sync::Arc;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CorpusTemplate {
    Xor,
    Sine,
    Spirals,
    Csv,
    Custom,
    Text,
}

impl CorpusTemplate {
    pub fn name(&self) -> &'static str {
        match self {
            CorpusTemplate::Xor => "XOR",
            CorpusTemplate::Sine => "Sine",
            CorpusTemplate::Spirals => "Spirals",
            CorpusTemplate::Csv => "CSV (custom)",
            CorpusTemplate::Custom => "Hand-rolled",
            CorpusTemplate::Text => "Text corpus",
        }
    }
}

/// Numeric / textual corpus container shared across both supported network
/// kinds. Unused fields are simply ignored for the active mode.
#[derive(Clone, Debug)]
pub struct Corpus {
    pub template: CorpusTemplate,

    // Spirals knobs.
    pub spirals_classes: usize,
    pub spirals_per_class: usize,

    // Sine knobs.
    pub sine_n: usize,
    pub sine_noise: f32,

    // CSV knobs.
    pub csv_path: String,
    pub csv_has_header: bool,
    pub csv_num_classes: String,

    // Hand-rolled rows: text-edit grid of "x0,x1,...,y0,y1,..." entries.
    pub custom_rows: String,
    pub custom_input_dim: usize,
    pub custom_output_dim: usize,
    pub custom_classification: bool,

    // Text-mode corpus (next-token-gen).
    pub text_body: String,
    pub text_paths: Vec<String>,
    pub upload_path: String,
    pub context_size: usize,

    // Resolved dataset (numeric mode) and message.
    pub dataset: Option<Arc<Dataset>>,
    pub message: Option<String>,

    // Cached encoded token stream for text mode.
    pub encoded_tokens: Vec<usize>,
}

impl Default for Corpus {
    fn default() -> Self {
        Self {
            template: CorpusTemplate::Xor,
            spirals_classes: 3,
            spirals_per_class: 100,
            sine_n: 200,
            sine_noise: 0.05,
            csv_path: String::new(),
            csv_has_header: true,
            csv_num_classes: String::new(),
            custom_rows: String::new(),
            custom_input_dim: 2,
            custom_output_dim: 1,
            custom_classification: false,
            text_body: String::new(),
            text_paths: Vec::new(),
            upload_path: String::new(),
            context_size: 1,
            dataset: None,
            message: None,
            encoded_tokens: Vec::new(),
        }
    }
}

impl Corpus {
    /// Available numeric templates for non-text networks.
    pub fn numeric_templates() -> &'static [CorpusTemplate] {
        &[
            CorpusTemplate::Xor,
            CorpusTemplate::Sine,
            CorpusTemplate::Spirals,
            CorpusTemplate::Csv,
            CorpusTemplate::Custom,
        ]
    }

    /// Build a numeric `Dataset` from the current template.
    pub fn build_numeric(&mut self, seed: u64) {
        let result: Result<Dataset, String> = match self.template {
            CorpusTemplate::Xor => Ok(data::xor()),
            CorpusTemplate::Sine => Ok(data::sine(self.sine_n, self.sine_noise, seed)),
            CorpusTemplate::Spirals => Ok(data::spirals(self.spirals_per_class, self.spirals_classes, seed)),
            CorpusTemplate::Csv => {
                if self.csv_path.trim().is_empty() {
                    Err("Provide a CSV file path.".into())
                } else {
                    let nc = self.csv_num_classes.trim();
                    let parsed = if nc.is_empty() {
                        Ok(None)
                    } else {
                        nc.parse::<usize>().map(Some).map_err(|e| format!("num_classes: {e}"))
                    };
                    match parsed {
                        Ok(num_classes) => data::load_csv(&self.csv_path, self.csv_has_header, num_classes)
                            .map_err(|e| format!("CSV load failed: {e}")),
                        Err(e) => Err(e),
                    }
                }
            }
            CorpusTemplate::Custom => self.build_custom_rows(),
            CorpusTemplate::Text => Err("Text template is for next-token-gen networks.".into()),
        };
        match result {
            Ok(ds) => {
                let task_str = match ds.task {
                    TaskKind::Regression => "regression".into(),
                    TaskKind::Classification { num_classes } => format!("classification, {num_classes} classes"),
                };
                self.message = Some(format!(
                    "Loaded {} samples · {} features · {} outputs ({task_str}).",
                    ds.n(), ds.n_features(), ds.n_outputs()
                ));
                self.dataset = Some(Arc::new(ds));
            }
            Err(e) => {
                self.message = Some(format!("Error: {e}"));
                self.dataset = None;
            }
        }
    }

    fn build_custom_rows(&mut self) -> Result<Dataset, String> {
        let in_d = self.custom_input_dim.max(1);
        let out_d = self.custom_output_dim.max(1);
        let need = in_d + out_d;
        let mut feats: Vec<f32> = Vec::new();
        let mut labels: Vec<f32> = Vec::new();
        let mut rows = 0usize;
        for (li, line) in self.custom_rows.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() != need {
                return Err(format!(
                    "row {} expects {need} comma-separated values, got {}",
                    li + 1, parts.len()
                ));
            }
            for p in &parts[..in_d] {
                feats.push(p.parse::<f32>().map_err(|e| format!("row {}: {e}", li + 1))?);
            }
            for p in &parts[in_d..] {
                labels.push(p.parse::<f32>().map_err(|e| format!("row {}: {e}", li + 1))?);
            }
            rows += 1;
        }
        if rows == 0 { return Err("no usable rows in custom corpus".into()); }
        let task = if self.custom_classification {
            TaskKind::Classification { num_classes: out_d }
        } else {
            TaskKind::Regression
        };
        Ok(Dataset {
            feature_names: (0..in_d).map(|i| format!("x{i}")).collect(),
            label_names: (0..out_d).map(|i| format!("y{i}")).collect(),
            features: Tensor::new(vec![rows, in_d], feats),
            labels: Tensor::new(vec![rows, out_d], labels),
            task,
        })
    }

    /// Append the contents of the file at `path` to the text corpus.
    pub fn upload_text_file(&mut self, path: &str) -> Result<(), String> {
        let raw = fs::read_to_string(path).map_err(|e| e.to_string())?;
        if !self.text_body.is_empty() && !self.text_body.ends_with('\n') {
            self.text_body.push('\n');
        }
        self.text_body.push_str(&raw);
        self.text_paths.push(path.to_string());
        self.message = Some(format!(
            "Appended {} bytes from {path} · total corpus = {} chars.",
            raw.len(), self.text_body.chars().count()
        ));
        Ok(())
    }

    /// (Re-)tokenise the text body using the supplied vocabulary.
    pub fn retokenise(&mut self, vocab: &Vocab) {
        self.encoded_tokens = vocab.encode(&self.text_body);
        self.message = Some(format!(
            "Encoded {} tokens (vocab size {}, context {}).",
            self.encoded_tokens.len(), vocab.len(), self.context_size
        ));
    }

    /// Build a next-token-prediction `Dataset` using the selected embedding.
    ///
    /// - **OneHot / TfIdf**: input_dim = ctx × vocab_size
    /// - **FastText / Transformer**: input_dim = ctx × embed_dim
    ///
    /// `seed` is used only to initialise the random embedding table for
    /// FastText / Transformer; it must equal `NetworkInstance::seed` so the
    /// same table is used at inference time.
    pub fn build_text_dataset(
        &mut self,
        vocab: &Vocab,
        embedding: EmbeddingKind,
        embed_dim: usize,
        seed: u64,
    ) -> Result<Arc<Dataset>, String> {
        if self.encoded_tokens.is_empty() {
            self.retokenise(vocab);
        }
        let toks = &self.encoded_tokens;
        let v = vocab.len().max(1);
        let ctx = self.context_size.max(1);
        if toks.len() <= ctx {
            return Err(format!("not enough tokens ({}) for context {}", toks.len(), ctx));
        }
        let n = toks.len() - ctx;

        let ds = match embedding {
            EmbeddingKind::OneHot => self.build_onehot(toks, n, ctx, v),
            EmbeddingKind::TfIdf  => self.build_tfidf(toks, n, ctx, v),
            EmbeddingKind::FastText   => self.build_dense(toks, n, ctx, v, embed_dim, seed, false),
            EmbeddingKind::Transformer=> self.build_dense(toks, n, ctx, v, embed_dim, seed, true),
        };

        let arc = Arc::new(ds);
        self.dataset = Some(arc.clone());
        let in_dim = arc.n_features();
        self.message = Some(format!(
            "Built {} training samples · in_dim {} · vocab {} · embedding: {}.",
            n, in_dim, v, embedding.name()
        ));
        Ok(arc)
    }

    // ── One-Hot ───────────────────────────────────────────────────────────────

    fn build_onehot(&self, toks: &[usize], n: usize, ctx: usize, v: usize) -> Dataset {
        let in_dim = ctx * v;
        let mut feats  = vec![0.0_f32; n * in_dim];
        let mut labels = vec![0.0_f32; n * v];
        for i in 0..n {
            for k in 0..ctx {
                let id = toks[i + k].min(v - 1);
                feats[i * in_dim + k * v + id] = 1.0;
            }
            let target = toks[i + ctx].min(v - 1);
            labels[i * v + target] = 1.0;
        }
        make_dataset(n, in_dim, v, feats, labels)
    }

    // ── TF-IDF ────────────────────────────────────────────────────────────────
    // Each token's weight = ln(1 + total_tokens / count[token]).
    // Rare tokens get higher weight; common tokens are downweighted.

    fn build_tfidf(&self, toks: &[usize], n: usize, ctx: usize, v: usize) -> Dataset {
        // Compute IDF from the entire corpus.
        let total = toks.len() as f32;
        let mut counts = vec![0usize; v];
        for &t in toks { counts[t.min(v - 1)] += 1; }
        let idf: Vec<f32> = counts.iter()
            .map(|&c| if c == 0 { 0.0 } else { (1.0 + total / c as f32).ln() })
            .collect();

        let in_dim = ctx * v;
        let mut feats  = vec![0.0_f32; n * in_dim];
        let mut labels = vec![0.0_f32; n * v];
        for i in 0..n {
            for k in 0..ctx {
                let id = toks[i + k].min(v - 1);
                feats[i * in_dim + k * v + id] = idf[id];
            }
            let target = toks[i + ctx].min(v - 1);
            labels[i * v + target] = 1.0;
        }
        make_dataset(n, in_dim, v, feats, labels)
    }

    // ── Dense embeddings (FastText / Transformer) ─────────────────────────────
    // A random embedding matrix E[vocab_size × embed_dim] is seeded from
    // `seed`, so inference later can reconstruct the same table cheaply.
    // For Transformer mode we add sinusoidal positional encoding.

    #[allow(clippy::too_many_arguments)]
    fn build_dense(
        &self,
        toks: &[usize],
        n: usize,
        ctx: usize,
        v: usize,
        embed_dim: usize,
        seed: u64,
        positional: bool,
    ) -> Dataset {
        let e = embed_dim.max(1);
        let emb = random_embedding_table(v, e, seed);

        let in_dim = ctx * e;
        let mut feats  = vec![0.0_f32; n * in_dim];
        let mut labels = vec![0.0_f32; n * v];
        for i in 0..n {
            for k in 0..ctx {
                let id = toks[i + k].min(v - 1);
                let row = &emb[id * e..(id + 1) * e];
                let base = i * in_dim + k * e;
                for d in 0..e {
                    let mut val = row[d];
                    if positional {
                        val += positional_enc(k, d, e);
                    }
                    feats[base + d] = val;
                }
            }
            let target = toks[i + ctx].min(v - 1);
            labels[i * v + target] = 1.0;
        }
        make_dataset(n, in_dim, v, feats, labels)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_dataset(n: usize, in_dim: usize, v: usize, feats: Vec<f32>, labels: Vec<f32>) -> Dataset {
    Dataset {
        feature_names: (0..in_dim).map(|i| format!("f{i}")).collect(),
        label_names:   (0..v).map(|i| format!("tok{i}")).collect(),
        features: Tensor::new(vec![n, in_dim], feats),
        labels:   Tensor::new(vec![n, v], labels),
        task: TaskKind::Classification { num_classes: v },
    }
}

/// Build a (vocab × embed_dim) embedding table using a simple PRNG seeded by
/// `seed`. Values are in [-0.5, 0.5] (Xavier-ish range for small models).
pub fn random_embedding_table(vocab: usize, embed_dim: usize, seed: u64) -> Vec<f32> {
    let n = vocab * embed_dim;
    let mut out = Vec::with_capacity(n);
    let mut s = seed ^ 0x9E3779B97F4A7C15;
    for _ in 0..n {
        // xorshift64
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let f = (s as f32) / (u64::MAX as f32) - 0.5;
        out.push(f);
    }
    out
}

/// Sinusoidal positional encoding PE(pos, dim, embed_dim).
/// Exposed for the inference path in app.rs.
#[inline]
pub fn positional_enc_pub(pos: usize, dim: usize, embed_dim: usize) -> f32 {
    positional_enc(pos, dim, embed_dim)
}

#[inline]
fn positional_enc(pos: usize, dim: usize, embed_dim: usize) -> f32 {
    let d = embed_dim.max(2) as f32;
    let exponent = (dim as f32 / d).floor() * 2.0 / d;
    let denom = 10000.0_f32.powf(exponent);
    if dim.is_multiple_of(2) {
        (pos as f32 / denom).sin()
    } else {
        (pos as f32 / denom).cos()
    }
}
