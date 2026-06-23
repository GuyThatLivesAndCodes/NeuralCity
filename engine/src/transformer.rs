//! Pre-norm decoder-only transformer language model — llama-style architecture.
//!
//! This is the model behind NeuralCabin's `transformer` network kind. The
//! design intentionally mirrors llama 2 / Mistral so the exported GGUF is
//! loadable by llama.cpp and downstream UIs like LM Studio without the
//! "unknown architecture" error our custom MLP GGUF hits:
//!
//! - RMSNorm before each sublayer (pre-norm)
//! - Multi-head causal self-attention with RoPE on Q and K
//! - SwiGLU feed-forward: `silu(x @ Wgate) * (x @ Wup) @ Wdown`
//! - Tied LM head OR separate output projection (we keep them separate to
//!   match the most common llama checkpoints — GGUF reads `output.weight`)
//! - Final RMSNorm before the LM head
//!
//! Weights are stored in CPU `Tensor` form for serde + persistence. Every
//! training step materializes a Burn `Module` on the requested device, runs
//! forward + backward + optimizer, and writes the updated weights back.

use crate::optimizer::OptimizerKind;
use crate::tensor::{SplitMix64, Tensor};
use burn::module::{Module, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{
    Adam, AdamConfig, AdamW, AdamWConfig, GradientsParams, Optimizer as BurnOptimizer, Sgd,
    SgdConfig,
};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Int, Tensor as BT, TensorData};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub n_ctx:     usize,
    pub n_embd:    usize,
    pub n_layers:  usize,
    pub n_heads:   usize,
    pub n_ff:      usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_rms_eps")]
    pub rms_eps: f32,
}
fn default_rope_theta() -> f32 { 10000.0 }
fn default_rms_eps()    -> f32 { 1e-5 }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockWeights {
    pub attn_norm: Tensor,  // (1, n_embd)
    pub wq:        Tensor,  // (n_embd, n_embd)
    pub wk:        Tensor,  // (n_embd, n_embd)
    pub wv:        Tensor,  // (n_embd, n_embd)
    pub wo:        Tensor,  // (n_embd, n_embd)
    pub ffn_norm:  Tensor,  // (1, n_embd)
    pub ffn_gate:  Tensor,  // (n_embd, n_ff)
    pub ffn_up:    Tensor,  // (n_embd, n_ff)
    pub ffn_down:  Tensor,  // (n_ff, n_embd)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransformerModel {
    pub config:      TransformerConfig,
    pub token_embd:  Tensor,         // (vocab, n_embd)
    pub blocks:      Vec<BlockWeights>,
    pub output_norm: Tensor,         // (1, n_embd)
    pub output:      Tensor,         // (n_embd, vocab) — LM head
    pub seed:        u64,
}

impl TransformerModel {
    pub fn new(config: TransformerConfig, seed: u64) -> Self {
        assert!(config.n_embd.is_multiple_of(config.n_heads),
            "n_embd must be divisible by n_heads (got {} / {})", config.n_embd, config.n_heads);
        let mut rng = SplitMix64::new(seed);
        let n_embd = config.n_embd;
        let n_ff   = config.n_ff;
        let vocab  = config.vocab_size;

        // Small init for embeddings; xavier for projections; ones for RMS scales.
        let token_embd = scaled_randn(vec![vocab, n_embd], (1.0 / n_embd as f32).sqrt(), &mut rng);
        let blocks = (0..config.n_layers).map(|_| BlockWeights {
            attn_norm: Tensor::ones(vec![1, n_embd]),
            wq: Tensor::xavier(vec![n_embd, n_embd], n_embd, n_embd, &mut rng),
            wk: Tensor::xavier(vec![n_embd, n_embd], n_embd, n_embd, &mut rng),
            wv: Tensor::xavier(vec![n_embd, n_embd], n_embd, n_embd, &mut rng),
            wo: Tensor::xavier(vec![n_embd, n_embd], n_embd, n_embd, &mut rng),
            ffn_norm: Tensor::ones(vec![1, n_embd]),
            ffn_gate: Tensor::xavier(vec![n_embd, n_ff], n_embd, n_ff, &mut rng),
            ffn_up:   Tensor::xavier(vec![n_embd, n_ff], n_embd, n_ff, &mut rng),
            ffn_down: Tensor::xavier(vec![n_ff, n_embd], n_ff, n_embd, &mut rng),
        }).collect();
        let output_norm = Tensor::ones(vec![1, n_embd]);
        let output = Tensor::xavier(vec![n_embd, vocab], n_embd, vocab, &mut rng);

        Self { config, token_embd, blocks, output_norm, output, seed }
    }

    pub fn parameter_count(&self) -> usize {
        let mut n = self.token_embd.len() + self.output_norm.len() + self.output.len();
        for b in &self.blocks {
            n += b.attn_norm.len() + b.wq.len() + b.wk.len() + b.wv.len() + b.wo.len();
            n += b.ffn_norm.len()  + b.ffn_gate.len() + b.ffn_up.len() + b.ffn_down.len();
        }
        n
    }
}

fn scaled_randn(shape: Vec<usize>, scale: f32, rng: &mut SplitMix64) -> Tensor {
    let n: usize = shape.iter().product();
    let mut data = Vec::with_capacity(n);
    for _ in 0..n { data.push(rng.next_normal() * scale); }
    Tensor::new(shape, data)
}

// ─── Burn module ────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
struct BurnBlock<B: Backend> {
    attn_norm: Param<BT<B, 1>>,
    wq: Param<BT<B, 2>>,
    wk: Param<BT<B, 2>>,
    wv: Param<BT<B, 2>>,
    wo: Param<BT<B, 2>>,
    ffn_norm: Param<BT<B, 1>>,
    ffn_gate: Param<BT<B, 2>>,
    ffn_up:   Param<BT<B, 2>>,
    ffn_down: Param<BT<B, 2>>,
}

#[derive(Module, Debug)]
struct BurnTransformer<B: Backend> {
    token_embd: Param<BT<B, 2>>,
    blocks: Vec<BurnBlock<B>>,
    output_norm: Param<BT<B, 1>>,
    output: Param<BT<B, 2>>,
}

impl<B: Backend> BurnTransformer<B> {
    fn from_model(m: &TransformerModel, device: &B::Device) -> Self {
        let token_embd = Param::from_tensor(m.token_embd.to_burn_2d::<B>(device));
        let blocks = m.blocks.iter().map(|b| BurnBlock {
            attn_norm: Param::from_tensor(burn_1d::<B>(&b.attn_norm, device)),
            wq: Param::from_tensor(b.wq.to_burn_2d::<B>(device)),
            wk: Param::from_tensor(b.wk.to_burn_2d::<B>(device)),
            wv: Param::from_tensor(b.wv.to_burn_2d::<B>(device)),
            wo: Param::from_tensor(b.wo.to_burn_2d::<B>(device)),
            ffn_norm: Param::from_tensor(burn_1d::<B>(&b.ffn_norm, device)),
            ffn_gate: Param::from_tensor(b.ffn_gate.to_burn_2d::<B>(device)),
            ffn_up:   Param::from_tensor(b.ffn_up.to_burn_2d::<B>(device)),
            ffn_down: Param::from_tensor(b.ffn_down.to_burn_2d::<B>(device)),
        }).collect();
        let output_norm = Param::from_tensor(burn_1d::<B>(&m.output_norm, device));
        let output = Param::from_tensor(m.output.to_burn_2d::<B>(device));
        Self { token_embd, blocks, output_norm, output }
    }

    /// Forward pass.
    /// - `tokens`: (batch, seq_len) Int tensor
    /// - `cos`, `sin`: precomputed RoPE tables (T, head_dim/2)
    /// - `mask`: precomputed causal mask (T, T)
    /// - returns logits: (batch, seq_len, vocab)
    ///
    /// Passing the tables in (rather than rebuilding them every call) is what
    /// the `TrainerSession` uses to avoid rebuilding ~seq_len² booleans plus
    /// the trig tables on every training step.
    fn forward(
        &self,
        tokens: BT<B, 2, Int>,
        cfg: &TransformerConfig,
        cos: &BT<B, 2>,
        sin: &BT<B, 2>,
        mask: &BT<B, 2, burn::tensor::Bool>,
    ) -> BT<B, 3> {
        let [batch, seq_len] = tokens.dims();
        let n_embd = cfg.n_embd;
        let n_heads = cfg.n_heads;
        let head_dim = n_embd / n_heads;

        // Embedding lookup: flatten (B,T) -> (B*T), select rows, reshape back.
        let flat = tokens.reshape([batch * seq_len]);
        let embedded: BT<B, 2> = self.token_embd.val().select(0, flat);
        let mut h: BT<B, 3> = embedded.reshape([batch, seq_len, n_embd]);

        for block in &self.blocks {
            // Attention sub-layer.
            let normed = rmsnorm::<B>(&h, &block.attn_norm.val(), cfg.rms_eps);
            // (B, T, n_embd)
            let bt = batch * seq_len;
            let normed_2d: BT<B, 2> = normed.clone().reshape([bt, n_embd]);
            let q_2d = normed_2d.clone().matmul(block.wq.val());
            let k_2d = normed_2d.clone().matmul(block.wk.val());
            let v_2d = normed_2d.matmul(block.wv.val());

            // Reshape to (B, T, H, head_dim) then swap to (B, H, T, head_dim)
            let q = q_2d.reshape([batch, seq_len, n_heads, head_dim]).swap_dims(1, 2);
            let k = k_2d.reshape([batch, seq_len, n_heads, head_dim]).swap_dims(1, 2);
            let v = v_2d.reshape([batch, seq_len, n_heads, head_dim]).swap_dims(1, 2);

            // Apply RoPE to q and k. cos/sin: (T, head_dim/2) — broadcast over (B, H).
            let q = apply_rope::<B>(q, cos, sin);
            let k = apply_rope::<B>(k, cos, sin);

            // Scaled dot-product: (B, H, T, head_dim) x (B, H, head_dim, T)
            let scale = 1.0_f32 / (head_dim as f32).sqrt();
            let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);

            // Apply causal mask: broadcast (T,T) -> (1,1,T,T)
            let mask_b = mask.clone().unsqueeze::<3>().unsqueeze::<4>();
            let scores = scores.mask_fill(mask_b, f32::NEG_INFINITY);

            let attn = softmax(scores, 3);
            let out = attn.matmul(v); // (B, H, T, head_dim)
            // (B, H, T, head_dim) -> (B, T, H, head_dim) -> (B, T, n_embd)
            let out = out.swap_dims(1, 2).reshape([batch, seq_len, n_embd]);
            let out_2d: BT<B, 2> = out.reshape([bt, n_embd]);
            let out_2d = out_2d.matmul(block.wo.val());
            let out: BT<B, 3> = out_2d.reshape([batch, seq_len, n_embd]);
            h = h + out;

            // FFN sub-layer.
            let normed = rmsnorm::<B>(&h, &block.ffn_norm.val(), cfg.rms_eps);
            let normed_2d: BT<B, 2> = normed.reshape([bt, n_embd]);
            let gate = silu(normed_2d.clone().matmul(block.ffn_gate.val()));
            let up = normed_2d.matmul(block.ffn_up.val());
            let ff_in = gate * up;
            let ff_out_2d = ff_in.matmul(block.ffn_down.val());
            let ff_out: BT<B, 3> = ff_out_2d.reshape([batch, seq_len, n_embd]);
            h = h + ff_out;
        }

        // Final norm + LM head.
        let h = rmsnorm::<B>(&h, &self.output_norm.val(), cfg.rms_eps);
        let bt = batch * seq_len;
        let h_2d: BT<B, 2> = h.reshape([bt, n_embd]);
        let logits_2d = h_2d.matmul(self.output.val());
        logits_2d.reshape([batch, seq_len, cfg.vocab_size])
    }

    fn write_back(self, model: &mut TransformerModel) {
        model.token_embd = Tensor::from_burn_2d::<B>(self.token_embd.val());
        let mut block_iter = self.blocks.into_iter();
        for dst in &mut model.blocks {
            let b = block_iter.next().unwrap();
            dst.attn_norm = from_burn_1d::<B>(b.attn_norm.val());
            dst.wq = Tensor::from_burn_2d::<B>(b.wq.val());
            dst.wk = Tensor::from_burn_2d::<B>(b.wk.val());
            dst.wv = Tensor::from_burn_2d::<B>(b.wv.val());
            dst.wo = Tensor::from_burn_2d::<B>(b.wo.val());
            dst.ffn_norm = from_burn_1d::<B>(b.ffn_norm.val());
            dst.ffn_gate = Tensor::from_burn_2d::<B>(b.ffn_gate.val());
            dst.ffn_up   = Tensor::from_burn_2d::<B>(b.ffn_up.val());
            dst.ffn_down = Tensor::from_burn_2d::<B>(b.ffn_down.val());
        }
        model.output_norm = from_burn_1d::<B>(self.output_norm.val());
        model.output = Tensor::from_burn_2d::<B>(self.output.val());
    }
}

// ─── Forward helpers ────────────────────────────────────────────────────────

fn burn_1d<B: Backend>(t: &Tensor, device: &B::Device) -> BT<B, 1> {
    let len = t.data.len();
    BT::<B, 1>::from_data(TensorData::new(t.data.clone(), [len]), device)
}
fn from_burn_1d<B: Backend>(t: BT<B, 1>) -> Tensor {
    let len = t.dims()[0];
    let data: Vec<f32> = t.into_data().convert::<f32>().into_vec().unwrap();
    Tensor::new(vec![1, len], data)
}

/// Root-mean-square normalization, applied along the last dim.
/// `scale` is the learned (n_embd,) scale; `eps` is the variance floor.
/// Works for either (B, T, D) or (B*T, D) — we accept (B, T, D) here.
fn rmsnorm<B: Backend>(x: &BT<B, 3>, scale: &BT<B, 1>, eps: f32) -> BT<B, 3> {
    let dims = x.dims();
    let d = dims[2];
    // mean of x^2 along last dim, keep dims
    let sq = x.clone().powf_scalar(2.0);
    let mean = sq.mean_dim(2); // (B, T, 1)
    let rms = mean.add_scalar(eps).sqrt();
    let normed = x.clone().div(rms); // broadcasts (B, T, 1)
    // scale: (D,) broadcasts to (B, T, D)
    let scale_b: BT<B, 3> = scale.clone().reshape([1, 1, d]);
    normed * scale_b
}

/// Build cos/sin RoPE tables.
/// Returns (cos, sin) each of shape (T, head_dim/2).
fn rope_tables<B: Backend>(
    seq_len: usize,
    head_dim: usize,
    theta: f32,
    device: &B::Device,
) -> (BT<B, 2>, BT<B, 2>) {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(seq_len * half);
    let mut sin = Vec::with_capacity(seq_len * half);
    for t in 0..seq_len {
        for i in 0..half {
            let freq = (theta).powf(-(2.0 * i as f32) / head_dim as f32);
            let angle = t as f32 * freq;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    let c = BT::<B, 2>::from_data(TensorData::new(cos, [seq_len, half]), device);
    let s = BT::<B, 2>::from_data(TensorData::new(sin, [seq_len, half]), device);
    (c, s)
}

/// Apply RoPE to a (B, H, T, head_dim) tensor. Pairs adjacent dims:
///   for i in 0..head_dim/2:
///     (x[..,2i], x[..,2i+1]) = (x[..,2i]*cos - x[..,2i+1]*sin,
///                               x[..,2i]*sin + x[..,2i+1]*cos)
///
/// Cheaper layout (used by GGML / llama): split the last dim into two
/// halves rather than interleaving pairs. We use the "interleaved pair"
/// scheme — it matches the GPTQ / HF reference. GGML reads RoPE the same
/// way as long as we declare the right `rope_dimension_count` (= head_dim)
/// in GGUF metadata.
fn apply_rope<B: Backend>(
    x: BT<B, 4>,
    cos: &BT<B, 2>,   // (T, head_dim/2)
    sin: &BT<B, 2>,   // (T, head_dim/2)
) -> BT<B, 4> {
    let dims = x.dims();
    let batch = dims[0];
    let heads = dims[1];
    let seq   = dims[2];
    let hd    = dims[3];
    let half  = hd / 2;

    // Reshape (B, H, T, hd) -> (B, H, T, half, 2)
    let x5: BT<B, 5> = x.reshape([batch, heads, seq, half, 2]);
    // Split along last dim into two (B, H, T, half) tensors.
    let x_even = x5.clone().slice([0..batch, 0..heads, 0..seq, 0..half, 0..1])
        .reshape([batch, heads, seq, half]);
    let x_odd  = x5.slice([0..batch, 0..heads, 0..seq, 0..half, 1..2])
        .reshape([batch, heads, seq, half]);

    // cos/sin: (T, half) -> broadcast to (1, 1, T, half)
    let cos_b: BT<B, 4> = cos.clone().reshape([1, 1, seq, half]);
    let sin_b: BT<B, 4> = sin.clone().reshape([1, 1, seq, half]);

    let rot_even = x_even.clone().mul(cos_b.clone()) - x_odd.clone().mul(sin_b.clone());
    let rot_odd  = x_even.mul(sin_b) + x_odd.mul(cos_b);

    // Re-interleave: stack into (B, H, T, half, 2), reshape to (B, H, T, hd).
    let stacked: BT<B, 5> = BT::stack::<5>(vec![rot_even, rot_odd], 4);
    stacked.reshape([batch, heads, seq, hd])
}

/// Boolean causal mask of shape (T, T). `mask[i, j] = (j > i)` — True means
/// "this position is a future token; suppress it".
fn causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> BT<B, 2, burn::tensor::Bool> {
    let mut data = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            data.push(j > i);
        }
    }
    BT::<B, 2, burn::tensor::Bool>::from_data(
        TensorData::new(data, [seq_len, seq_len]),
        device,
    )
}

// ─── Training step ──────────────────────────────────────────────────────────

/// One training step on (input_tokens, target_tokens) sequences.
/// Tensors are `(batch * seq_len)` flat slices of token ids.
#[allow(clippy::too_many_arguments)]
pub fn train_step_on_device<B: AutodiffBackend>(
    model: &mut TransformerModel,
    kind: &OptimizerKind,
    _step_count: u64,
    inputs: &[u32],        // (batch, seq_len) flattened
    targets: &[u32],       // (batch, seq_len) flattened
    batch: usize,
    seq_len: usize,
    device: &B::Device,
) -> f32 {
    let cfg = model.config.clone();
    let net: BurnTransformer<B> = BurnTransformer::from_model(model, device);

    // Move tokens to device as Int tensors.
    let input_i64: Vec<i64> = inputs.iter().map(|&t| t as i64).collect();
    let target_i64: Vec<i64> = targets.iter().map(|&t| t as i64).collect();
    let x = BT::<B, 2, Int>::from_data(
        TensorData::new(input_i64, [batch, seq_len]),
        device,
    );
    let y = BT::<B, 2, Int>::from_data(
        TensorData::new(target_i64, [batch, seq_len]),
        device,
    );

    // Forward: (B, T, vocab)
    let head_dim = cfg.n_embd / cfg.n_heads;
    let (cos, sin) = rope_tables::<B>(seq_len, head_dim, cfg.rope_theta, device);
    let mask = causal_mask::<B>(seq_len, device);
    let logits = net.forward(x, &cfg, &cos, &sin, &mask);
    let vocab = cfg.vocab_size;

    // Loss: cross-entropy averaged over (B*T) positions.
    // CrossEntropyLoss in Burn expects (N, C) logits and (N,) class ids.
    let logits_flat = logits.reshape([batch * seq_len, vocab]);
    let targets_flat = y.reshape([batch * seq_len]);
    let loss_module = CrossEntropyLossConfig::new().init(device);
    let loss = loss_module.forward(logits_flat, targets_flat);
    let loss_scalar: f32 = loss.clone().into_scalar().elem();

    // Backward + optimizer step.
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &net);
    let updated = step_with_optimizer::<B>(kind, net, grads);
    updated.write_back(model);

    loss_scalar
}

fn step_with_optimizer<B: AutodiffBackend>(
    kind: &OptimizerKind,
    net: BurnTransformer<B>,
    grads: GradientsParams,
) -> BurnTransformer<B> {
    match *kind {
        OptimizerKind::Sgd { lr, momentum } => {
            let cfg = if momentum > 0.0 {
                SgdConfig::new().with_momentum(Some(burn::optim::momentum::MomentumConfig {
                    momentum: momentum as f64, dampening: 0.0, nesterov: false,
                }))
            } else { SgdConfig::new() };
            let mut opt = cfg.init::<B, BurnTransformer<B>>();
            opt.step(lr as f64, net, grads)
        }
        OptimizerKind::Adam { lr, beta1, beta2, eps } => {
            let cfg = AdamConfig::new().with_beta_1(beta1).with_beta_2(beta2).with_epsilon(eps);
            let mut opt = cfg.init::<B, BurnTransformer<B>>();
            opt.step(lr as f64, net, grads)
        }
        OptimizerKind::AdamW { lr, beta1, beta2, eps, weight_decay }
        | OptimizerKind::Lamb { lr, beta1, beta2, eps, weight_decay } => {
            let cfg = AdamWConfig::new()
                .with_beta_1(beta1).with_beta_2(beta2).with_epsilon(eps)
                .with_weight_decay(weight_decay);
            let mut opt = cfg.init::<B, BurnTransformer<B>>();
            opt.step(lr as f64, net, grads)
        }
    }
}

// ─── Persistent trainer session ─────────────────────────────────────────────

/// Burn optimizer state for a single training run. Built once at the start
/// of training and reused for every step so Adam/AdamW/LAMB actually keep
/// their momentum and variance buffers between steps.
///
/// The previous implementation re-created the optimizer on every call to
/// `train_step_on_device`, which silently turned Adam into LR-scaled SGD with
/// constant zero moments — meaning the network never actually converged in
/// the way the hyperparameters implied, so training ran far longer than it
/// should have. Keeping the optimizer alive is the single biggest speedup.
enum OptVariant<B: AutodiffBackend> {
    Sgd(OptimizerAdaptor<Sgd<B::InnerBackend>, BurnTransformer<B>, B>),
    Adam(OptimizerAdaptor<Adam, BurnTransformer<B>, B>),
    AdamW(OptimizerAdaptor<AdamW, BurnTransformer<B>, B>),
}

/// A long-lived training session for a single `TransformerModel` on a single
/// device. Holds:
///
/// - The Burn module (`BurnTransformer<B>`) so its parameters live on the
///   GPU for the whole run instead of being uploaded every step.
/// - The optimizer, with its momentum/variance state preserved across steps.
/// - Pre-built RoPE cos/sin tables and the causal mask for the run's
///   `seq_len`, so we don't re-fill seq_len² booleans on each step.
///
/// Call `step` per batch; call `write_back` periodically (e.g. once per
/// epoch) to sync the GPU weights to the CPU `TransformerModel` for
/// persistence, checkpointing, or inference.
pub struct TrainerSession<B: AutodiffBackend> {
    /// `Option` because `BurnOptimizer::step` takes the module by value and
    /// returns the updated one — we `take()`, step, and put it back.
    net: Option<BurnTransformer<B>>,
    opt: OptVariant<B>,
    cfg: TransformerConfig,
    device: B::Device,
    lr: f64,
    seq_len: usize,
    cos: BT<B, 2>,
    sin: BT<B, 2>,
    mask: BT<B, 2, burn::tensor::Bool>,
}

/// On-device snapshot of frozen parameters. Cheap to capture (Burn tensors
/// are reference-counted) and avoids a full GPU→CPU→GPU round trip.
pub struct FrozenSnapshotGpu<B: AutodiffBackend> {
    embedding: Option<BT<B, 2>>,
    output_norm: Option<BT<B, 1>>,
    output: Option<BT<B, 2>>,
    blocks: HashMap<usize, BlockSnapshot<B>>,
}

struct BlockSnapshot<B: AutodiffBackend> {
    attn_norm: BT<B, 1>,
    wq: BT<B, 2>,
    wk: BT<B, 2>,
    wv: BT<B, 2>,
    wo: BT<B, 2>,
    ffn_norm: BT<B, 1>,
    ffn_gate: BT<B, 2>,
    ffn_up:   BT<B, 2>,
    ffn_down: BT<B, 2>,
}

impl<B: AutodiffBackend> TrainerSession<B> {
    /// Build a session: upload the model to `device` once, build the
    /// optimizer once, precompute RoPE/mask once.
    pub fn new(
        model: &TransformerModel,
        kind: &OptimizerKind,
        seq_len: usize,
        device: B::Device,
    ) -> Self {
        let cfg = model.config.clone();
        let net = BurnTransformer::<B>::from_model(model, &device);
        let head_dim = cfg.n_embd / cfg.n_heads;
        let (cos, sin) = rope_tables::<B>(seq_len, head_dim, cfg.rope_theta, &device);
        let mask = causal_mask::<B>(seq_len, &device);
        let (opt, lr) = build_opt::<B>(kind);
        Self {
            net: Some(net),
            opt,
            cfg,
            device,
            lr,
            seq_len,
            cos,
            sin,
            mask,
        }
    }

    /// Update the learning rate for subsequent steps. Useful for LR schedules.
    pub fn set_lr(&mut self, lr: f32) { self.lr = lr as f64; }

    /// One training step on a flattened `(batch * seq_len)` input/target
    /// pair. Returns the loss for monitoring.
    pub fn step(&mut self, inputs: &[u32], targets: &[u32], batch: usize) -> f32 {
        let seq_len = self.seq_len;
        debug_assert_eq!(inputs.len(), batch * seq_len);
        debug_assert_eq!(targets.len(), batch * seq_len);

        // Move tokens to device as Int tensors. Small (B*T i64s), one of the
        // few unavoidable per-step CPU→GPU transfers.
        let input_i64: Vec<i64> = inputs.iter().map(|&t| t as i64).collect();
        let target_i64: Vec<i64> = targets.iter().map(|&t| t as i64).collect();
        let x = BT::<B, 2, Int>::from_data(
            TensorData::new(input_i64, [batch, seq_len]),
            &self.device,
        );
        let y = BT::<B, 2, Int>::from_data(
            TensorData::new(target_i64, [batch, seq_len]),
            &self.device,
        );

        let net = self.net.take().expect("session net is always Some between steps");
        let logits = net.forward(x, &self.cfg, &self.cos, &self.sin, &self.mask);
        let vocab = self.cfg.vocab_size;
        let logits_flat = logits.reshape([batch * seq_len, vocab]);
        let targets_flat = y.reshape([batch * seq_len]);
        let loss_module = CrossEntropyLossConfig::new().init(&self.device);
        let loss = loss_module.forward(logits_flat, targets_flat);
        let loss_scalar: f32 = loss.clone().into_scalar().elem();

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &net);

        let lr = self.lr;
        let updated = match &mut self.opt {
            OptVariant::Sgd(o)   => o.step(lr, net, grads),
            OptVariant::Adam(o)  => o.step(lr, net, grads),
            OptVariant::AdamW(o) => o.step(lr, net, grads),
        };
        self.net = Some(updated);
        loss_scalar
    }

    /// Mirror the live GPU parameters back into the CPU `TransformerModel`.
    /// Cheap-ish but not free, so do this once per epoch / on checkpoint,
    /// not per step.
    pub fn write_back(&self, model: &mut TransformerModel) {
        let net = self.net.as_ref().expect("session net is always Some between steps");
        model.token_embd = Tensor::from_burn_2d::<B>(net.token_embd.val());
        for (dst, b) in model.blocks.iter_mut().zip(net.blocks.iter()) {
            dst.attn_norm = from_burn_1d::<B>(b.attn_norm.val());
            dst.wq = Tensor::from_burn_2d::<B>(b.wq.val());
            dst.wk = Tensor::from_burn_2d::<B>(b.wk.val());
            dst.wv = Tensor::from_burn_2d::<B>(b.wv.val());
            dst.wo = Tensor::from_burn_2d::<B>(b.wo.val());
            dst.ffn_norm = from_burn_1d::<B>(b.ffn_norm.val());
            dst.ffn_gate = Tensor::from_burn_2d::<B>(b.ffn_gate.val());
            dst.ffn_up   = Tensor::from_burn_2d::<B>(b.ffn_up.val());
            dst.ffn_down = Tensor::from_burn_2d::<B>(b.ffn_down.val());
        }
        model.output_norm = from_burn_1d::<B>(net.output_norm.val());
        model.output = Tensor::from_burn_2d::<B>(net.output.val());
    }

    /// Snapshot the chosen parameters straight off the device so they can
    /// be restored verbatim after the optimizer step (= frozen weights).
    pub fn snapshot_frozen(
        &self,
        embedding: bool,
        output_norm: bool,
        output: bool,
        block_ids: &[usize],
    ) -> FrozenSnapshotGpu<B> {
        let net = self.net.as_ref().expect("session net is always Some between steps");
        let mut blocks = HashMap::new();
        for &i in block_ids {
            if let Some(b) = net.blocks.get(i) {
                blocks.insert(i, BlockSnapshot {
                    attn_norm: b.attn_norm.val(),
                    wq: b.wq.val(),
                    wk: b.wk.val(),
                    wv: b.wv.val(),
                    wo: b.wo.val(),
                    ffn_norm: b.ffn_norm.val(),
                    ffn_gate: b.ffn_gate.val(),
                    ffn_up:   b.ffn_up.val(),
                    ffn_down: b.ffn_down.val(),
                });
            }
        }
        FrozenSnapshotGpu {
            embedding:   if embedding   { Some(net.token_embd.val()) }  else { None },
            output_norm: if output_norm { Some(net.output_norm.val()) } else { None },
            output:      if output      { Some(net.output.val()) }      else { None },
            blocks,
        }
    }

    /// Re-install a frozen-parameter snapshot. Called after each optimizer
    /// step so frozen layers stay put while non-frozen layers update.
    pub fn restore_frozen(&mut self, snap: &FrozenSnapshotGpu<B>) {
        let net = self.net.as_mut().expect("session net is always Some between steps");
        if let Some(t) = &snap.embedding {
            net.token_embd = Param::from_tensor(t.clone());
        }
        if let Some(t) = &snap.output_norm {
            net.output_norm = Param::from_tensor(t.clone());
        }
        if let Some(t) = &snap.output {
            net.output = Param::from_tensor(t.clone());
        }
        for (&i, b) in &snap.blocks {
            if let Some(dst) = net.blocks.get_mut(i) {
                dst.attn_norm = Param::from_tensor(b.attn_norm.clone());
                dst.wq = Param::from_tensor(b.wq.clone());
                dst.wk = Param::from_tensor(b.wk.clone());
                dst.wv = Param::from_tensor(b.wv.clone());
                dst.wo = Param::from_tensor(b.wo.clone());
                dst.ffn_norm = Param::from_tensor(b.ffn_norm.clone());
                dst.ffn_gate = Param::from_tensor(b.ffn_gate.clone());
                dst.ffn_up   = Param::from_tensor(b.ffn_up.clone());
                dst.ffn_down = Param::from_tensor(b.ffn_down.clone());
            }
        }
    }
}

fn build_opt<B: AutodiffBackend>(
    kind: &OptimizerKind,
) -> (OptVariant<B>, f64) {
    match *kind {
        OptimizerKind::Sgd { lr, momentum } => {
            let cfg = if momentum > 0.0 {
                SgdConfig::new().with_momentum(Some(burn::optim::momentum::MomentumConfig {
                    momentum: momentum as f64, dampening: 0.0, nesterov: false,
                }))
            } else { SgdConfig::new() };
            (OptVariant::Sgd(cfg.init::<B, BurnTransformer<B>>()), lr as f64)
        }
        OptimizerKind::Adam { lr, beta1, beta2, eps } => {
            let cfg = AdamConfig::new().with_beta_1(beta1).with_beta_2(beta2).with_epsilon(eps);
            (OptVariant::Adam(cfg.init::<B, BurnTransformer<B>>()), lr as f64)
        }
        OptimizerKind::AdamW { lr, beta1, beta2, eps, weight_decay }
        | OptimizerKind::Lamb { lr, beta1, beta2, eps, weight_decay } => {
            let cfg = AdamWConfig::new()
                .with_beta_1(beta1).with_beta_2(beta2).with_epsilon(eps)
                .with_weight_decay(weight_decay);
            (OptVariant::AdamW(cfg.init::<B, BurnTransformer<B>>()), lr as f64)
        }
    }
}

// ─── CPU inference ──────────────────────────────────────────────────────────

/// Compute logits at every position for a (single) sequence of `tokens`.
/// Returns a Vec of length `tokens.len() * vocab_size`, row-major.
/// We run through Burn on the requested backend rather than reimplementing
/// the forward pass — this keeps inference and training byte-identical.
pub fn forward_logits<B: Backend>(
    model: &TransformerModel,
    tokens: &[u32],
    device: &B::Device,
) -> Vec<f32> {
    let net: BurnTransformer<B> = BurnTransformer::from_model(model, device);
    let seq_len = tokens.len();
    let input_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
    let x = BT::<B, 2, Int>::from_data(
        TensorData::new(input_i64, [1, seq_len]),
        device,
    );
    let head_dim = model.config.n_embd / model.config.n_heads;
    let (cos, sin) = rope_tables::<B>(seq_len, head_dim, model.config.rope_theta, device);
    let mask = causal_mask::<B>(seq_len, device);
    let logits = net.forward(x, &model.config, &cos, &sin, &mask); // (1, T, vocab)
    let logits = logits.reshape([seq_len * model.config.vocab_size]);
    logits.into_data().convert::<f32>().into_vec().unwrap()
}

/// A reusable autoregressive-inference context for a single `TransformerModel`
/// on a single device.
///
/// Holds the Burn module (so its weights are uploaded to the device **once**
/// instead of on every generated token) plus the RoPE cos/sin tables and the
/// causal mask for a fixed `seq_len`. Generation always feeds a fixed-width
/// window (the model's context length, left-padded when the prompt is short),
/// so these tables never change across steps and are built a single time here.
///
/// The previous generation path called [`forward_logits`] per token, which
/// rebuilt the whole module — re-uploading the embeddings, every block, and the
/// LM head to the GPU on each step — and recomputed the RoPE/mask tables every
/// call. For an N-token completion that is N redundant full-model uploads. This
/// session removes them, the same fix [`TrainerSession`] applies to training.
pub struct InferenceSession<B: Backend> {
    net: BurnTransformer<B>,
    cfg: TransformerConfig,
    device: B::Device,
    seq_len: usize,
    cos: BT<B, 2>,
    sin: BT<B, 2>,
    mask: BT<B, 2, burn::tensor::Bool>,
}

impl<B: Backend> InferenceSession<B> {
    /// Upload `model` to `device` and precompute the RoPE/mask tables for the
    /// fixed window length `seq_len` used throughout the generation run.
    pub fn new(model: &TransformerModel, seq_len: usize, device: B::Device) -> Self {
        let cfg = model.config.clone();
        let net = BurnTransformer::<B>::from_model(model, &device);
        let head_dim = cfg.n_embd / cfg.n_heads;
        let (cos, sin) = rope_tables::<B>(seq_len, head_dim, cfg.rope_theta, &device);
        let mask = causal_mask::<B>(seq_len, &device);
        Self { net, cfg, device, seq_len, cos, sin, mask }
    }

    /// Logits for the **last** position of `tokens`, length `vocab_size`.
    ///
    /// `tokens.len()` must equal the `seq_len` the session was built with. Only
    /// the final row is pulled back from the device, so the host transfer is
    /// `vocab_size` floats rather than `seq_len * vocab_size`.
    pub fn last_logits(&self, tokens: &[u32]) -> Vec<f32> {
        debug_assert_eq!(
            tokens.len(),
            self.seq_len,
            "InferenceSession::last_logits expects a window of the session's seq_len",
        );
        let input_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let x = BT::<B, 2, Int>::from_data(
            TensorData::new(input_i64, [1, self.seq_len]),
            &self.device,
        );
        let logits = self.net.forward(x, &self.cfg, &self.cos, &self.sin, &self.mask);
        // (1, T, vocab) → slice the final position → (vocab,)
        let vocab = self.cfg.vocab_size;
        let last = logits
            .slice([0..1, (self.seq_len - 1)..self.seq_len, 0..vocab])
            .reshape([vocab]);
        last.into_data().convert::<f32>().into_vec().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{CpuAutodiffBackend, CpuBackend};
    use crate::optimizer::OptimizerKind;
    use burn::tensor::backend::Backend;
    use serial_test::serial;

    fn tiny_config(vocab: usize) -> TransformerConfig {
        TransformerConfig {
            vocab_size: vocab,
            n_ctx: 8,
            n_embd: 16,
            n_layers: 2,
            n_heads: 2,
            n_ff: 32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
        }
    }

    #[test]
    fn forward_produces_expected_shape() {
        let cfg = tiny_config(10);
        let model = TransformerModel::new(cfg.clone(), 1);
        let device = <CpuBackend as Backend>::Device::default();
        let tokens: Vec<u32> = (0..8).collect();
        let logits = forward_logits::<CpuBackend>(&model, &tokens, &device);
        assert_eq!(logits.len(), 8 * 10);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    /// `InferenceSession::last_logits` must equal the final position's row of
    /// the full `forward_logits` output — the session is purely an
    /// optimization (upload-once + last-row slice), never a behaviour change.
    #[test]
    fn inference_session_matches_forward_logits() {
        let cfg = tiny_config(10);
        let model = TransformerModel::new(cfg.clone(), 3);
        let device = <CpuBackend as Backend>::Device::default();
        let tokens: Vec<u32> = (0..cfg.n_ctx as u32).collect();

        let full = forward_logits::<CpuBackend>(&model, &tokens, &device);
        let expected_last = &full[full.len() - cfg.vocab_size..];

        let session = InferenceSession::<CpuBackend>::new(&model, tokens.len(), device);
        let got = session.last_logits(&tokens);

        assert_eq!(got.len(), cfg.vocab_size);
        for (a, b) in got.iter().zip(expected_last.iter()) {
            assert!((a - b).abs() < 1e-5, "logit mismatch: {a} vs {b}");
        }
        // Reusing the session for a second call must be stable (weights are
        // resident and not mutated by inference).
        let got2 = session.last_logits(&tokens);
        assert_eq!(got, got2);
    }

    /// Train on a trivial deterministic sequence and check that the loss
    /// decreases. The model is tiny so this is more about plumbing than
    /// learning quality.
    #[test]
    #[serial(autodiff)]
    fn train_step_decreases_loss() {
        let cfg = tiny_config(8);
        let mut model = TransformerModel::new(cfg.clone(), 42);
        let device = <CpuAutodiffBackend as Backend>::Device::default();
        let opt = OptimizerKind::Adam { lr: 0.01, beta1: 0.9, beta2: 0.999, eps: 1e-8 };

        // Train to predict the next token in a repeating sequence 0..8.
        let input:  Vec<u32> = (0..8).collect();
        let target: Vec<u32> = (1..9).map(|x| x % 8).collect();

        let mut losses = Vec::new();
        for step in 1..=40 {
            let l = train_step_on_device::<CpuAutodiffBackend>(
                &mut model, &opt, step, &input, &target, 1, 8, &device,
            );
            losses.push(l);
        }
        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(last < first, "loss did not decrease: {first} → {last}");
    }
}
