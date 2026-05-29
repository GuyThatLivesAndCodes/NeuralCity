# NeuralCabin: Tauri → .NET Migration

This document tracks the in-progress migration of NeuralCabin from a **Tauri**
desktop app (Rust shell + Rust ML engine + React frontend) to a **native .NET**
desktop application. The goal is feature parity with **no visible change** for
end users: same UI, same workflows, same look and feel.

## Strategy

NeuralCabin's UI talks to its backend through exactly two primitives —
`invoke(cmd, args)` and `listen(event, handler)`. Everything else (tabs,
styling, plugin host, visualizations) is plain React/TypeScript. The migration
therefore keeps the **entire React frontend unchanged** and swaps out what sits
behind those two primitives:

```
            BEFORE (Tauri)                         AFTER (.NET)
  ┌───────────────────────────────┐    ┌───────────────────────────────┐
  │ React UI (frontend/)           │    │ React UI (frontend/) — SAME    │
  │   invoke()/listen()            │    │   invoke()/listen()            │
  │        │  @tauri-apps/api      │    │        │  frontend/src/host    │
  │  ┌─────▼──────────────────┐    │    │  ┌─────▼──────────────────┐    │
  │  │ Tauri IPC bridge        │    │    │  │ Photino webview bridge  │    │
  │  ├─────────────────────────┤    │    │  ├─────────────────────────┤    │
  │  │ Tauri commands (lib.rs) │    │ →  │  │ NeuralCabin.Host (C#)   │    │
  │  │ axum server, export…    │    │    │  │ command registry + svcs │    │
  │  ├─────────────────────────┤    │    │  ├─────────────────────────┤    │
  │  │ Rust ML engine (Burn)   │    │    │  │ engine (port OR interop)│    │
  │  └─────────────────────────┘    │    │  └─────────────────────────┘    │
  └───────────────────────────────┘    └───────────────────────────────┘
```

The **host shim** (`frontend/src/host/index.ts`) detects which shell it is
running in and routes `invoke`/`listen` accordingly. While both shells exist it
defaults to Tauri, so the shipping app is never broken mid-migration.

The native shell uses **[Photino](https://www.tryphotino.io/)** — a thin,
cross-platform native window around the OS webview. It is the closest analogue
to Tauri (no bundled browser, real app window, Windows/macOS/Linux) and lets us
reuse the React build verbatim, served over a custom `app://` scheme so the
plugin system's IndexedDB origin keeps working.

## .NET solution layout (`dotnet/`)

| Project | Role |
|---|---|
| `NeuralCabin.Core` | DTOs + IPC contracts. Mirror `models.rs`/server types **field-for-field** so JSON is byte-compatible with serde. No platform deps. |
| `NeuralCabin.Host` | Backend logic: IPC router, command registry, domain services. Replaces the Tauri command layer. No GUI deps → unit-testable headless. |
| `NeuralCabin.App` | Photino desktop shell. Thin composition root: window + asset server + transport wiring. Replaces `src-tauri/`. |
| `NeuralCabin.Core.Tests` | JSON-contract tests (serde compatibility). |
| `NeuralCabin.Host.Tests` | IPC router + domain-service tests (end-to-end bridge). |

### IPC wire protocol

One JSON string per message, both directions (see
`NeuralCabin.Host/Ipc/IpcMessages.cs` and `frontend/src/host/index.ts`):

```
UI  → host:  { "kind":"invoke",   "id":…, "cmd":…, "args":… }
host → UI:   { "kind":"response", "id":…, "ok":true,  "result":… }
host → UI:   { "kind":"response", "id":…, "ok":false, "error":… }
host → UI:   { "kind":"event",    "event":…, "payload":… }
```

## Status

Legend: ✅ done · 🟡 partial · ⬜ not started

| Area | Status | Notes |
|---|---|---|
| Native window / shell | ✅ | Photino host (`NeuralCabin.App`) builds. GUI run needs a webview runtime + display (same as Tauri). |
| IPC bridge (`invoke`/`emit`) | ✅ | C# router + TS shim, with end-to-end tests. |
| Shared models / DTOs | ✅ | All of `models.rs` + server types ported; JSON contract under test. |
| Frontend host-agnostic seam | ✅ | `frontend/src/host` + one import change in `api.ts`. |
| **Networks** commands | ✅ | `create` (feedforward materializes a model; next-token stores its chain), `list`, `get`, `delete`, persisted. Transformer `create` pending the transformer engine. |
| Events plumbing | ✅ | `training_*` events produced by the feed-forward loop. `inference_*` streaming events pending (next-token/transformer). |
| Corpus | 🟡 | Feed-forward corpus + stats done and persisted. Next-token text/pairs stored; **vocabulary build** pending the tokenizer port. |
| Training loop | 🟡 | Feed-forward fully ported: per-epoch deterministic shuffle, mini-batches, persistent-optimizer session, layer freezing, live events, history, stop/abort+rollback. Next-token/transformer pending. |
| Inference | 🟡 | Feed-forward sync `infer` + `infer_with_activations` done. Next-token/transformer streaming pending. |
| Export (pytorch/onnx/gguf) | ⬜ | Port `src-tauri/export.rs`. |
| Embedded API server | ⬜ | Port `src-tauri/server.rs` (axum → ASP.NET Core minimal API / `HttpListener`). |
| Persistence (`state.json`, `models/`) | 🟡 | `state.json` (networks/corpora/history) + **serde-compatible** model files done. Vocab/transformer/server persistence and reading a legacy Tauri `state.json` pending. |
| **ML engine** | 🟡 | **MLP/feed-forward engine ported to pure C#** — tensors, analytic backprop, SGD/Adam/AdamW, losses, serde-compatible model JSON — validated by finite-difference gradient checks + XOR convergence. **Tokenizer + transformer (RoPE/RMSNorm/attention fwd+bwd) still to port.** |

### Remaining Tauri dependencies

- `src-tauri/` crate (entire Tauri command layer, server, export, persistence)
- `@tauri-apps/cli`, `@tauri-apps/api` (the latter is still the default
  `invoke`/`listen` path in the host shim)
- `tauri.conf.json`, `src-tauri/capabilities/`, `src-tauri/icons/`
- CI: the `build-app` / `release-bundles` jobs still build/bundle via Tauri

## Roadmap (ordered)

- [x] **Engine decision + spike** — chose a pure-C# port (no Rust/Burn). MLP
  engine ported and gradient-checked.
- [x] **Persistence** — `state.json` + serde-compatible `models/<id>.json`.
- [x] **Feed-forward vertical** — create / corpus / train (+events) / infer,
  end-to-end through the bridge.
1. **Tokenizer + Vocabulary** — port `engine/tokenizer.rs`; `build_vocabulary`,
   `set_advanced_vocabulary`, `get_vocabulary`, `tokenize_preview`; materialize
   next-token models once a vocab exists.
2. **Next-token training + streaming inference** (`inference_*` events).
3. **Transformer engine** — RoPE / RMSNorm / attention / SwiGLU forward **and**
   analytic backward; transformer create / train / infer.
4. **Export** (pytorch / onnx / gguf) — port `src-tauri/export.rs`.
5. **Embedded API server** — port `src-tauri/server.rs`.
6. **Cutover** — default the host shim to `dotnet`, ship the Photino bundle from
   CI, then retire `src-tauri/` and the Tauri toolchain.

## Build, test, run

```bash
# Backend (.NET) — builds all projects and runs the test suites
dotnet build dotnet/NeuralCabin.sln
dotnet test  dotnet/NeuralCabin.sln

# Frontend (unchanged) — still builds the same React bundle
npm --prefix frontend run build

# Engine (unchanged) — the canonical Rust test suite still passes
cargo test --package neuralcabin-engine

# Run the native .NET shell (stages the frontend into wwwroot first)
dotnet/stage-frontend.sh
dotnet run --project dotnet/NeuralCabin.App        # needs a desktop + webview runtime
# …or hot-reload against Vite:
#   (term 1) npm --prefix frontend run dev
#   (term 2) NEURALCABIN_DEV_URL=http://localhost:5173 dotnet run --project dotnet/NeuralCabin.App
```

> In Claude Code web sessions the .NET SDK is installed automatically by the
> `SessionStart` hook (`.claude/hooks/ensure-dotnet.sh`) so these commands work
> in a fresh container.
