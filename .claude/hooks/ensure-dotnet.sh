#!/usr/bin/env bash
# SessionStart hook: make the .NET SDK available for NeuralCabin's Tauri → .NET
# migration work. Claude Code web sessions run in ephemeral containers that do
# not ship .NET, so we install it on demand. Idempotent and best-effort — it
# never blocks or fails the session.
set -u

# Already on PATH? Nothing to do.
if command -v dotnet >/dev/null 2>&1; then
  exit 0
fi

DOTNET_DIR="${DOTNET_ROOT:-$HOME/.dotnet}"

# Installed previously this container but not linked? Just link it.
if [ -x "$DOTNET_DIR/dotnet" ]; then
  ln -sf "$DOTNET_DIR/dotnet" /usr/local/bin/dotnet 2>/dev/null || true
  exit 0
fi

echo "[neuralcabin] installing .NET SDK 8.0 (needed to build/test the .NET migration)…" >&2
tmp="$(mktemp -d)"
if curl -fsSL -o "$tmp/dotnet-install.sh" https://dot.net/v1/dotnet-install.sh 2>/dev/null; then
  bash "$tmp/dotnet-install.sh" --channel 8.0 --install-dir "$DOTNET_DIR" >/dev/null 2>&1 || true
  ln -sf "$DOTNET_DIR/dotnet" /usr/local/bin/dotnet 2>/dev/null || true
fi
rm -rf "$tmp"

if command -v dotnet >/dev/null 2>&1; then
  echo "[neuralcabin] .NET SDK ready ($(dotnet --version 2>/dev/null))." >&2
else
  echo "[neuralcabin] .NET SDK unavailable (offline?). 'dotnet' commands may not work this session." >&2
fi
exit 0
