# 🧠 NeuralCabin

A **pure Rust neural network workbench** built as a **proper desktop app** — no browser window, no localhost URL. NeuralCabin opens in its own native application window, just like Discord or VS Code.

Powered by [Tauri](https://tauri.app): the Rust backend handles all ML computation and communicates with the React frontend via Tauri IPC. No Python, no PyTorch, no NumPy.

## Quick Start

### Download & install (pre-built)
1. Download the installer for your OS from [Releases](../../releases)
   - Windows: `.msi` installer
   - macOS: `.dmg`
   - Linux: `.AppImage` or `.deb`
2. Install and launch — a native app window opens immediately

### Build from source
```bash
# Prerequisites: Rust (rustup.rs) and Node.js (nodejs.org)

# Install dependencies
npm install
npm --prefix frontend install

# Development mode — opens Tauri window with Vite hot-reload
npm run dev

# Production build — creates installer in src-tauri/target/release/bundle/
npm run tauri -- build
```

Or use the convenience script:
```bash
./start-dev.sh        # Linux/macOS
start-dev.bat         # Windows
```

## Repository Layout

```
neuralcabin/
├── Cargo.toml              — workspace (engine + src-tauri)
├── package.json            — @tauri-apps/cli, dev/build scripts
│
├── engine/                 — Pure Rust ML engine (zero math deps)
│   └── src/
│       ├── tensor.rs       — Dense Vec<f32> tensors, matmul, operations
│       ├── autograd.rs     — Reverse-mode autodiff (tape-based)
│       ├── activations.rs  — ReLU, Sigmoid, Tanh, Softmax
│       ├── loss.rs         — MSE, CrossEntropy
│       ├── optimizer.rs    — SGD (momentum), Adam
│       ├── nn.rs           — Linear/Activation layers, Model
│       └── persistence.rs  — Model checkpoints (JSON)
│
├── src-tauri/              — Tauri desktop app shell (Rust)
│   ├── src/
│   │   ├── main.rs         — entry point
│   │   ├── lib.rs          — Tauri commands + async training loop
│   │   └── models.rs       — shared types (serde)
│   ├── tauri.conf.json     — window config, bundle targets
│   ├── capabilities/       — Tauri permission declarations
│   └── icons/              — app icons for all platforms
│
├── frontend/               — React + TypeScript UI
│   ├── src/
│   │   ├── App.tsx         — Main multi-tab application + plugin host
│   │   ├── api.ts          — Tauri invoke/listen wrappers
│   │   ├── index.css       — Styling (Times New Roman, orange theme)
│   │   ├── tabs/           — Tab components (incl. PluginsTab)
│   │   ├── components/     — Shared UI (NetworkViz, PluginErrorBoundary)
│   │   └── plugins/        — Plugin system
│   │       ├── types.ts       — Plugin API (NetworkType, PluginContext)
│   │       ├── registry.ts    — Install/enable/load + per-network tagging
│   │       ├── storage.ts     — IndexedDB k/v for large plugin payloads
│   │       └── builtins/      — Out-of-the-box plugins (Image Classification)
│   ├── package.json
│   ├── vite.config.ts
│   └── index.html
│
└── start-dev.sh/bat        — Convenience startup scripts
```

## How It Works

NeuralCabin is a [Tauri](https://tauri.app) desktop app:

```
NeuralCabin (native desktop window — no browser needed)
  ├── Rust backend (src-tauri/)
  │     ├── Tauri commands  ←  invoke('create_network', {...})
  │     ├── Tauri events    →  emit('training_update', {...})
  │     └── ML engine       — pure Rust, zero external deps
  └── React frontend (frontend/)
        ├── Calls backend via @tauri-apps/api  invoke()
        └── Receives real-time updates via  listen()
```

No HTTP server, no WebSocket, no localhost URL. The React frontend talks directly to Rust through Tauri's native IPC bridge.

## Features

### 🖥 Native Desktop App
- **Real application window** — not a browser tab, no localhost URL
- **Lightweight** — uses the OS built-in WebView (~10 MB vs ~150 MB for Electron)
- **Cross-platform** — Windows, macOS, Linux installers from CI

### 🎨 React + TypeScript UI
- **Tabs:** Networks, Corpus, Vocabulary, Training, Inference, Plugins, Documentation, Server, Settings
- **Orange Theme:** Warm design with Times New Roman typography
- **Real-time Training:** Live loss curve updated every epoch via Tauri events
- **Network Visualization:** the Inference tab renders feed-forward networks
  as colored neurons per layer with activations from the most recent forward
  pass — works for built-in feed-forward networks and any plugin-contributed
  type that uses the same engine path.

### 🔌 Plugin System
- **New network types contributed by plugins** appear in the same Type
  dropdown as the built-in ones. A plugin owns its create form, corpus UI,
  and inference UI for the types it adds.
- **Plugin runtime:** plain JS/TS modules loaded into the host realm (no
  sandbox — installing a plugin is a trust decision).
- **Built-in:** an **Image Classification** plugin ships enabled. Configure
  image size, RGB vs. grayscale, hidden layers, output activation, and seed;
  build a corpus by uploading images or drawing them on the canvas; predict
  by drawing/uploading at inference time (with an opt-in real-time mode).
  Samples are persisted in IndexedDB so the localStorage quota isn't an
  issue.
- **Marketplace:** a Cloudflare-backed registry of signed plugins is a
  planned follow-up; the Plugins tab already shows the stub.

### 🧠 Pure Rust ML Engine
- **Zero external math dependencies** — tensors, matmul, autograd, optimizers hand-written
- **Optimizers:** Adam, SGD with momentum
- **Loss functions:** MSE, CrossEntropy

## Tests

```bash
cargo test --package neuralcabin-engine
```

The engine ships with 13 tests including gradient checks, optimizer convergence, CSV parsing, model save/load, and end-to-end XOR MLP convergence.

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
