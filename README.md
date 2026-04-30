# NeuralCabin

A from-scratch neural network workbench, written in **pure Rust**.

NeuralCabin compiles to a single self-contained executable. Everything from
linear algebra and reverse-mode automatic differentiation through optimisers
and the GUI is built natively in Rust — no Python, no JavaScript, no PyTorch,
TensorFlow, NumPy, `ndarray`, `burn`, or `candle`.

The only external crates we lean on are:

| Where  | Crates                | Why                                              |
|--------|-----------------------|--------------------------------------------------|
| engine | `serde`, `serde_json` | Model save/load (allowed by spec — non-ML).      |
| ui     | `egui`, `eframe`      | Native cross-platform window / immediate-mode UI.|

The neural-network engine itself has **zero numerical dependencies**: tensors,
matmul, activations, losses, autograd and optimisers are all hand-written.

## Repository layout

```
neuralcabin/
├── Cargo.toml          — workspace manifest + the top-level `neuralcabin` binary
├── src/main.rs         — entry point; launches the UI or runs --xor-demo
├── engine/             — neuralcabin-engine crate (no math deps)
│   └── src/
│       ├── lib.rs
│       ├── tensor.rs        Dense Vec<f32> tensors, matmul, transpose, sum_rows…
│       ├── autograd.rs      Reverse-mode tape autodiff (Add, Mul, MatMul, ReLU,
│       │                    Sigmoid, Tanh, MSE, Softmax+CrossEntropy)
│       ├── activations.rs   ReLU / Sigmoid / Tanh / Softmax (eager + tape forms)
│       ├── loss.rs          MeanSquaredError, CrossEntropy
│       ├── optimizer.rs     SGD (with momentum), Adam
│       ├── nn.rs            Linear / Activation layers, Sequential Model
│       ├── data.rs          Tiny CSV loader + canned datasets (XOR, spirals, sine)
│       └── persistence.rs   JSON model checkpoints (custom envelope, version 1)
└── ui/                 — neuralcabin-ui crate (egui workbench)
    └── src/
        ├── lib.rs
        ├── app.rs           Three-panel layout (Architecture / Training / Inference)
        ├── trainer.rs       Background training thread + command channel
        └── plot.rs          Built-from-scratch line plot & 2-D scatter
```

## Building

```bash
# build everything (the engine, the UI library, the binary)
cargo build --release

# run the workbench (default)
cargo run --release

# headless smoke test — trains an MLP on XOR end-to-end and prints predictions
cargo run --release -- --xor-demo
```

The binary is `target/release/neuralcabin`.

The UI requires standard Linux desktop libs (X11 or Wayland + OpenGL); on
macOS and Windows it builds out of the box. If you’re on a headless server,
use `--xor-demo` to verify the engine without launching a window.

## Tests

```bash
cargo test --workspace
```

The engine ships with 13 tests including:

- analytic-vs-numerical gradient check on `MatMul + MSE`,
- standalone `Sigmoid` backward correctness,
- SGD and Adam convergence on a quadratic,
- CSV parser (with and without one-hot encoding),
- model save/load round-trip,
- end-to-end MLP convergence on XOR.

`cargo clippy --workspace --all-targets -- -D warnings` is also clean.

## Using the workbench

The window is split into three panels.

### Architecture (left)

- Set the input dimension and a deterministic init seed.
- Build a stack of `Linear` and `Activation` layers; in-dim is propagated
  automatically as you add layers.
- Click **Build / Reset Model** to compile the spec into trainable weights
  (Xavier-uniform init).
- Save / load model checkpoints to a JSON file.

### Training (centre)

- Pick a dataset: `XOR` (built-in), `Spirals` (configurable classes / density),
  `Sine` (1-D regression with noise), or load a `CSV` file. For CSV
  classification, set `num_classes` to one-hot the last column.
- Choose `Loss` and `Optimizer` (`SGD` with momentum, or `Adam`).
- Set learning rate, epochs, batch size, validation fraction.
- **Train / Pause / Stop** runs the background trainer thread; the loss curve
  updates live (with optional log-Y), and a 2-D dataset scatter is rendered for
  classification problems.

### Inference (right)

- The input fields auto-size to the model's `input_dim`.
- **Predict** runs a single forward pass; for cross-entropy-trained models the
  output is automatically softmax-normalised so you can read it as
  probabilities, and the `argmax` class is highlighted.

## Engine quick reference

```rust
use neuralcabin_engine::{
    nn::{LayerSpec, Model},
    optimizer::{Optimizer, OptimizerKind},
    tensor::Tensor,
    Activation, Loss,
};

let mut model = Model::from_specs(2, &[
    LayerSpec::Linear { in_dim: 2, out_dim: 8 },
    LayerSpec::Activation(Activation::Tanh),
    LayerSpec::Linear { in_dim: 8, out_dim: 1 },
    LayerSpec::Activation(Activation::Sigmoid),
], 42);
let mut opt = Optimizer::new(
    OptimizerKind::Adam { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
    &model.parameter_shapes(),
);
let x = Tensor::new(vec![4, 2], vec![0.,0., 0.,1., 1.,0., 1.,1.]);
let y = Tensor::new(vec![4, 1], vec![0., 1., 1., 0.]);
for _ in 0..2000 {
    model.train_step(&mut opt, Loss::MeanSquaredError, &x, &y);
}
let pred = model.predict(&x);
```

## License

MIT — see `LICENSE.md`.
