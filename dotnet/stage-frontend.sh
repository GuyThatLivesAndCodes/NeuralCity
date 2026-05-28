#!/usr/bin/env bash
# Build the React frontend and stage it into the Photino shell's wwwroot, which
# is served over the app:// scheme at runtime. Mirrors how Tauri bundles
# `frontend/dist` into its binary.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"
wwwroot="$here/NeuralCabin.App/wwwroot"

echo "[stage-frontend] building frontend…"
npm --prefix "$repo_root/frontend" run build

echo "[stage-frontend] copying dist → wwwroot…"
rm -rf "$wwwroot"
mkdir -p "$wwwroot"
cp -r "$repo_root/frontend/dist/." "$wwwroot/"

echo "[stage-frontend] done. Run: dotnet run --project \"$here/NeuralCabin.App\""
