# NeuralCabin — .NET shell

The native .NET desktop application that is replacing the Tauri shell. See
[`../MIGRATION.md`](../MIGRATION.md) for the full plan and current status.

## Projects

- **NeuralCabin.Core** — domain models + IPC contracts (serde-compatible JSON).
- **NeuralCabin.Host** — IPC router, command registry, domain services (no GUI).
- **NeuralCabin.App** — Photino desktop shell (window + asset server + bridge).
- **NeuralCabin.Core.Tests / NeuralCabin.Host.Tests** — xUnit suites.

## Develop

```bash
dotnet build NeuralCabin.sln      # build everything
dotnet test  NeuralCabin.sln      # run all tests

# Run the app (stage the React build into NeuralCabin.App/wwwroot first):
./stage-frontend.sh
dotnet run --project NeuralCabin.App
```

The window loads the bundled UI over a custom `app://` scheme. For live
frontend development, point the shell at the Vite dev server instead:

```bash
npm --prefix ../frontend run dev                          # terminal 1
NEURALCABIN_DEV_URL=http://localhost:5173 \
  dotnet run --project NeuralCabin.App                    # terminal 2
```

> Running the window needs an OS webview runtime (WebView2 on Windows,
> WebKitGTK on Linux, WKWebView on macOS) and a display — the same native
> dependency Tauri requires. Building and testing need neither.
