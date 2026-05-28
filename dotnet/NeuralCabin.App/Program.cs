using NeuralCabin.Host;
using Photino.NET;

namespace NeuralCabin.App;

/// <summary>
/// Entry point for the native .NET desktop shell. Opens a Photino window of the
/// same dimensions as the old Tauri window (1280×900), wires the IPC bridge to
/// the ported backend, and loads the React UI.
/// </summary>
public static class Program
{
    [STAThread]
    public static void Main(string[] args)
    {
        var backend = new NeuralCabinBackend();

        // The transport needs the window, but the window's message handler needs
        // the router — break the cycle with a lazy window reference.
        PhotinoWindow? window = null;
        var transport = new PhotinoTransport(() => window);
        var router = backend.CreateRouter(transport);

        // Optional hot-reload against the Vite dev server (mirrors Tauri devUrl).
        var devUrl = Environment.GetEnvironmentVariable("NEURALCABIN_DEV_URL");

        window = new PhotinoWindow()
            .SetTitle("NeuralCabin")
            .SetUseOsDefaultSize(false)
            .SetSize(1280, 900)
            .Center()
            .SetResizable(true)
            .RegisterWebMessageReceivedHandler((object? sender, string message) =>
            {
                // Fire-and-forget: the router replies via SendWebMessage. Our
                // current handlers complete synchronously, so the response is
                // sent on this (UI) thread before returning.
                _ = router.HandleMessageAsync(message);
            });

        if (string.IsNullOrWhiteSpace(devUrl))
        {
            window.RegisterCustomSchemeHandler("app", AssetServer.Handle);
            window.Load(new Uri("app://index.html"));
        }
        else
        {
            window.Load(new Uri(devUrl));
        }

        window.WaitForClose();
    }
}
