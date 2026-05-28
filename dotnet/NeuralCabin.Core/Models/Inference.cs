using System.Text.Json.Serialization;

namespace NeuralCabin.Core.Models;

/// <summary>A single chat turn (mirror of <c>models::ChatMessage</c>).</summary>
public sealed record ChatMessage
{
    public string Role { get; init; } = ""; // "user" | "assistant"
    public string Text { get; init; } = "";
}

/// <summary>An inference request (mirror of <c>models::InferRequest</c>).</summary>
public sealed record InferRequest
{
    public string NetworkId { get; init; } = "";
    public List<float>? Features { get; init; }
    public string? Prompt { get; init; }
    public int? MaxNewTokens { get; init; }
    public float? Temperature { get; init; }
    public List<ChatMessage>? Messages { get; init; }
}

/// <summary>
/// An inference response (mirror of <c>models::InferResponse</c>). Both fields
/// carry <c>skip_serializing_if = "Option::is_none"</c> in serde, so they must
/// be omitted (not emitted as <c>null</c>) when absent.
/// </summary>
public sealed record InferResponse
{
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<float>? Output { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? InferenceId { get; init; }
}

public sealed record InferenceToken
{
    public string InferenceId { get; init; } = "";
    public int Index { get; init; }
    public string Token { get; init; } = "";
    public float Probability { get; init; }
}

public sealed record InferenceFinished
{
    public string InferenceId { get; init; } = "";
    public string Status { get; init; } = ""; // "completed" | "cancelled"
    public string Generated { get; init; } = "";
    public int TokenCount { get; init; }
}

public sealed record InferenceError
{
    public string InferenceId { get; init; } = "";
    public string Message { get; init; } = "";
}

/// <summary>Per-layer activations for the network visualizer (feed-forward only).</summary>
public sealed record InferActivations
{
    public List<string> LayerNames { get; init; } = new();
    public List<List<float>> Activations { get; init; } = new();
    public List<int> Dims { get; init; } = new();
}
