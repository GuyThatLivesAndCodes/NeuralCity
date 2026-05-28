using System.Text.Json.Serialization;

namespace NeuralCabin.Core.Models;

/// <summary>Network kind discriminators (mirror of <c>models::kinds</c>).</summary>
public static class NetworkKinds
{
    public const string Feedforward = "feedforward";
    public const string NextToken = "next_token";
    public const string Transformer = "transformer";

    public static readonly IReadOnlyList<string> All = new[] { Feedforward, NextToken, Transformer };
}

/// <summary>Decoder-only transformer hyper-parameters.</summary>
public sealed record TransformerHParams
{
    public int NCtx { get; init; }
    public int NEmbd { get; init; }
    public int NLayers { get; init; }
    public int NHeads { get; init; }
    public int NFf { get; init; }
    public float RopeTheta { get; init; } = 10000.0f;
    public float RmsEps { get; init; } = 1e-5f;
}

/// <summary>
/// One layer in a feed-forward chain. Serializes as an internally-tagged union
/// (<c>{ "type": "linear", ... }</c> / <c>{ "type": "activation", ... }</c>),
/// matching serde's <c>#[serde(tag = "type")]</c> on <c>LayerDef</c>.
/// </summary>
[JsonPolymorphic(TypeDiscriminatorPropertyName = "type")]
[JsonDerivedType(typeof(LinearLayer), "linear")]
[JsonDerivedType(typeof(ActivationLayer), "activation")]
public abstract record LayerDef;

public sealed record LinearLayer : LayerDef
{
    public int InDim { get; init; }
    public int OutDim { get; init; }
}

public sealed record ActivationLayer : LayerDef
{
    public string Activation { get; init; } = "";
}

/// <summary>A network definition as surfaced to the UI (mirror of <c>models::Network</c>).</summary>
public sealed record Network
{
    public string Id { get; init; } = "";
    public string Name { get; init; } = "";
    public string Kind { get; init; } = "";
    public long Seed { get; init; }
    public DateTimeOffset CreatedAt { get; init; }
    public bool Trained { get; init; }
    public bool Pretrained { get; init; }

    public int InputDim { get; init; }
    public int OutputDim { get; init; }
    public List<LayerDef> Layers { get; init; } = new();
    public long ParameterCount { get; init; }

    public List<LayerDef>? HiddenLayers { get; init; }
    public int? ContextSize { get; init; }
    public TransformerHParams? Transformer { get; init; }
}

/// <summary>Create-network request (mirror of <c>models::CreateNetworkRequest</c>).</summary>
public sealed record CreateNetworkRequest
{
    public string Name { get; init; } = "";
    public string Kind { get; init; } = "";
    public long Seed { get; init; }
    public List<LayerDef> Layers { get; init; } = new();
    public int? InputDim { get; init; }
    public int? ContextSize { get; init; }
    public TransformerHParams? Transformer { get; init; }
}
