namespace NeuralCabin.Core.Models;

/// <summary>Tabular feed-forward training data (mirror of <c>models::FeedforwardCorpus</c>).</summary>
public sealed record FeedforwardCorpus
{
    public List<float> Features { get; init; } = new();
    public List<float> Targets { get; init; } = new();
    public int Rows { get; init; }
    public int InDim { get; init; }
    public int OutDim { get; init; }
}

/// <summary>An instruction/response pair for fine-tuning.</summary>
public sealed record FineTunePair
{
    public string Input { get; init; } = "";
    public string Output { get; init; } = "";
}

/// <summary>Corpus attached to a network (mirror of <c>models::Corpus</c>).</summary>
public sealed record Corpus
{
    public string NetworkId { get; init; } = "";
    public string Kind { get; init; } = "";
    public DateTimeOffset UpdatedAt { get; init; }

    public FeedforwardCorpus? Feedforward { get; init; }
    public string? Text { get; init; }
    public List<FineTunePair>? Pairs { get; init; }
    public string? Stage { get; init; } // "pretrain" | "finetune"
}

public sealed record SetCorpusRequest
{
    public string NetworkId { get; init; } = "";
    public FeedforwardCorpus? Feedforward { get; init; }
    public string? Text { get; init; }
    public List<FineTunePair>? Pairs { get; init; }
    public string? Stage { get; init; }
    public string? VocabMode { get; init; } // "char" | "word"
}

/// <summary>Summary statistics for the corpus (mirror of <c>models::CorpusStats</c>).</summary>
public sealed record CorpusStats
{
    public string Kind { get; init; } = "";
    public string? Stage { get; init; }

    public int? Rows { get; init; }
    public int? InDim { get; init; }
    public int? OutDim { get; init; }

    public int? TextChars { get; init; }
    public int? TextTokens { get; init; }
    public int? PairCount { get; init; }
    public int? VocabSize { get; init; }
    public string? VocabMode { get; init; }
    public int? TrainingExamples { get; init; }

    public bool VocabReady { get; init; }
    public bool ModelReady { get; init; }
}
