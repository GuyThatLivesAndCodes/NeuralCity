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
| **Networks** commands | 🟡 | `create` (feedforward + next-token), `list`, `get`, `delete` implemented in-memory. Transformer `create` and **disk persistence** pending. |
| Events plumbing | 🟡 | Channel + emitter implemented; no producers yet (training/inference not ported). |
| Corpus / Vocabulary | ⬜ | Needs tokenizer port (`engine/tokenizer.rs`). |
| Training loop | ⬜ | Needs engine. |
| Inference (sync + streaming) | ⬜ | Needs engine. |
| Export (pytorch/onnx/gguf) | ⬜ | Port `src-tauri/export.rs`. |
| Embedded API server | ⬜ | Port `src-tauri/server.rs` (axum → ASP.NET Core minimal API / `HttpListener`). |
| Persistence (`state.json`, `models/`) | ⬜ | Port `src-tauri/persistence.rs`; must read/write the same on-disk format for a seamless switch. |
| **ML engine** | ⬜ | Hardest piece — now Burn-backed (wgpu/autodiff). Decide: interop with the Rust engine via a native library, or port to a .NET ML stack (TorchSharp / hand-rolled). |

### Remaining Tauri dependencies

- `src-tauri/` crate (entire Tauri command layer, server, export, persistence)
- `@tauri-apps/cli`, `@tauri-apps/api` (the latter is still the default
  `invoke`/`listen` path in the host shim)
- `tauri.conf.json`, `src-tauri/capabilities/`, `src-tauri/icons/`
- CI: the `build-app` / `release-bundles` jobs still build/bundle via Tauri

## Roadmap (ordered)

1. **Engine decision + spike** — interop vs. port. This unblocks training,
   inference, vocab, corpus stats, and transformer creation.
2. **Persistence** — read/write the existing `state.json` + `models/<id>.json`
   so a user's data carries over to the .NET app unchanged.
3. **Corpus + Vocabulary** commands (tokenizer port).
4. **Training** loop + `training_*` events.
5. **Inference** (feed-forward sync, next-token streaming) + `inference_*` events.
6. **Export** (pytorch/onnx/gguf).
7. **Embedded API server**.
8. **Cutover** — default the host shim to `dotnet`, ship the Photino bundle from
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
