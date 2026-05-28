using System.Text.Json;
using System.Text.Json.Nodes;

namespace NeuralCabin.Host.Ipc;

/// <summary>
/// Wire envelopes for the JSON bridge between the webview and the host. The
/// matching TypeScript lives in <c>frontend/src/host/index.ts</c>.
///
/// Protocol (one JSON string per message, both directions):
///   JS → host:  { "kind":"invoke",   "id":"…", "cmd":"…", "args":{…} }
///   host → JS:  { "kind":"response", "id":"…", "ok":true, "result":… }
///               { "kind":"response", "id":"…", "ok":false,"error":"…" }
///   host → JS:  { "kind":"event",    "event":"…", "payload":… }
/// </summary>
public sealed record InvokeRequest
{
    public string Kind { get; init; } = "invoke";
    public string Id { get; init; } = "";
    public string Cmd { get; init; } = "";
    public JsonElement? Args { get; init; }
}

public sealed record InvokeResponse
{
    public string Kind { get; init; } = "response";
    public string Id { get; init; } = "";
    public bool Ok { get; init; }
    public JsonNode? Result { get; init; }
    public string? Error { get; init; }
}

public sealed record EventEnvelope
{
    public string Kind { get; init; } = "event";
    public string Event { get; init; } = "";
    public JsonNode? Payload { get; init; }
}
