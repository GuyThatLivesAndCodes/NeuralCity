namespace NeuralCabin.Core.Models;

/// <summary>Tunables for vocabulary construction (mirror of <c>models::VocabularyOptions</c>).</summary>
public sealed record VocabularyOptions
{
    public int SubwordMerges { get; init; } = 200;
    public int WordTopN { get; init; } = 500;
}

/// <summary>A built vocabulary (mirror of <c>models::VocabularyInfo</c>).</summary>
public sealed record VocabularyInfo
{
    public string Mode { get; init; } = ""; // "char" | "subword" | "word" | "advanced"
    public List<string> Tokens { get; init; } = new();
    public VocabularyOptions Options { get; init; } = new();
    public DateTimeOffset UpdatedAt { get; init; }
}

public sealed record BuildVocabularyRequest
{
    public string NetworkId { get; init; } = "";
    public string Mode { get; init; } = ""; // "char" | "subword" | "word"
    public VocabularyOptions Options { get; init; } = new();
}

public sealed record SetAdvancedVocabularyRequest
{
    public string NetworkId { get; init; } = "";
    public List<string> Tokens { get; init; } = new();
}
