using System.Text;

namespace NeuralCabin.App;

/// <summary>
/// Serves the bundled frontend over a custom <c>app://</c> scheme. Using a
/// scheme (rather than <c>file://</c>) gives the webview a stable origin, which
/// the plugin system relies on for IndexedDB — exactly how Tauri serves assets
/// from <c>tauri://localhost</c>. This keeps the migration invisible to the UI.
/// </summary>
public static class AssetServer
{
    private static readonly string WwwRoot =
        Path.Combine(AppContext.BaseDirectory, "wwwroot");

    /// <summary>Photino custom-scheme callback.</summary>
    public static Stream Handle(object sender, string scheme, string url, out string contentType)
    {
        var path = ExtractPath(url);

        // Root request → index.html.
        if (path.Length == 0)
            path = "index.html";

        var fullPath = ResolveWithinRoot(path);

        // Missing asset with no extension → SPA fallback to index.html.
        if (fullPath is null || !File.Exists(fullPath))
        {
            if (!Path.HasExtension(path))
            {
                var indexPath = ResolveWithinRoot("index.html");
                if (indexPath is not null && File.Exists(indexPath))
                    fullPath = indexPath;
            }
        }

        if (fullPath is null || !File.Exists(fullPath))
        {
            contentType = "text/html";
            return PlaceholderPage(path);
        }

        contentType = ContentTypeFor(fullPath);
        // Copy to memory so we don't hold a file handle open in the webview.
        return new MemoryStream(File.ReadAllBytes(fullPath));
    }

    private static string ExtractPath(string url)
    {
        // url looks like "app://index.html/assets/app.js" (host = "index.html").
        // We care only about the path segment; the host is a constant artifact
        // of loading app://index.html.
        if (Uri.TryCreate(url, UriKind.Absolute, out var uri))
            return Uri.UnescapeDataString(uri.AbsolutePath).TrimStart('/');

        var schemeIndex = url.IndexOf("://", StringComparison.Ordinal);
        if (schemeIndex < 0)
            return url.TrimStart('/');
        var rest = url[(schemeIndex + 3)..];
        var slash = rest.IndexOf('/');
        return slash < 0 ? string.Empty : rest[(slash + 1)..].TrimStart('/');
    }

    private static string? ResolveWithinRoot(string relativePath)
    {
        var combined = Path.GetFullPath(Path.Combine(WwwRoot, relativePath));
        var rootPrefix = WwwRoot.EndsWith(Path.DirectorySeparatorChar)
            ? WwwRoot
            : WwwRoot + Path.DirectorySeparatorChar;
        // Guard against path traversal outside the asset root.
        return combined.StartsWith(rootPrefix, StringComparison.Ordinal) || combined == WwwRoot
            ? combined
            : null;
    }

    private static Stream PlaceholderPage(string requested)
    {
        var html = $$"""
            <!doctype html><html><head><meta charset="utf-8"><title>NeuralCabin</title>
            <style>body{font-family:'Times New Roman',serif;background:#1a1410;color:#e8c9a0;
            display:flex;align-items:center;justify-content:center;height:100vh;margin:0;text-align:center}
            code{color:#ff9f43}</style></head>
            <body><div><h1>NeuralCabin (.NET shell)</h1>
            <p>Frontend assets were not found (requested <code>{{requested}}</code>).</p>
            <p>Build and stage the UI:<br><code>npm --prefix frontend run build</code><br>
            then copy <code>frontend/dist</code> into <code>NeuralCabin.App/wwwroot</code>.</p></div></body></html>
            """;
        return new MemoryStream(Encoding.UTF8.GetBytes(html));
    }

    private static string ContentTypeFor(string path) => Path.GetExtension(path).ToLowerInvariant() switch
    {
        ".html" or ".htm" => "text/html",
        ".js" or ".mjs" => "text/javascript",
        ".css" => "text/css",
        ".json" or ".map" => "application/json",
        ".png" => "image/png",
        ".jpg" or ".jpeg" => "image/jpeg",
        ".gif" => "image/gif",
        ".svg" => "image/svg+xml",
        ".ico" => "image/x-icon",
        ".webp" => "image/webp",
        ".woff" => "font/woff",
        ".woff2" => "font/woff2",
        ".ttf" => "font/ttf",
        ".wasm" => "application/wasm",
        _ => "application/octet-stream",
    };
}
