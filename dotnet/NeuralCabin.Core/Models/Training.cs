namespace NeuralCabin.Core.Models;

/// <summary>Optimizer configuration (mirror of <c>models::OptimizerConfig</c>).</summary>
public sealed record OptimizerConfig
{
    public string Kind { get; init; } = ""; // "adam" | "adamw" | "lamb" | "sgd"
    public float Lr { get; init; }
    public float? Beta1 { get; init; }
    public float? Beta2 { get; init; }
    public float? Eps { get; init; }
    public float? Momentum { get; init; }
    public float? WeightDecay { get; init; }
}

/// <summary>A full training configuration (mirror of <c>models::TrainingConfig</c>).</summary>
public sealed record TrainingConfig
{
    public int Epochs { get; init; }
    public int BatchSize { get; init; }
    public OptimizerConfig Optimizer { get; init; } = new();
    public string Loss { get; init; } = ""; // "mse" | "crossentropy"
    public long Seed { get; init; }
    public bool? MaskUserTokens { get; init; }
    public List<string>? FrozenLayers { get; init; }
}

public sealed record StartTrainingRequest
{
    public string NetworkId { get; init; } = "";
    public TrainingConfig Config { get; init; } = new();
}

public sealed record StartTrainingResponse
{
    public string TrainingId { get; init; } = "";
    public string Status { get; init; } = "";
}

/// <summary>Internal training state snapshot (mirror of <c>models::TrainingState</c>).</summary>
public sealed record TrainingState
{
    public bool Running { get; init; }
    public bool Stopped { get; init; }
    public bool Cancelled { get; init; }
    public int Epoch { get; init; }
    public int TotalEpochs { get; init; }
    public float LastLoss { get; init; }
    public List<float> LossHistory { get; init; } = new();
    public string? Error { get; init; }
    public float ElapsedSecs { get; init; }
}

public sealed record TrainingStatusResponse
{
    public string TrainingId { get; init; } = "";
    public string Status { get; init; } = ""; // "running" | "completed" | "cancelled" | "error"
    public int Epoch { get; init; }
    public int TotalEpochs { get; init; }
    public float LastLoss { get; init; }
    public List<float> LossHistory { get; init; } = new();
    public float ElapsedSecs { get; init; }
}

public sealed record TrainingUpdate
{
    public string TrainingId { get; init; } = "";
    public int Epoch { get; init; }
    public int TotalEpochs { get; init; }
    public float Loss { get; init; }
    public List<float> LossHistory { get; init; } = new();
    public float ElapsedSecs { get; init; }
}

public sealed record TrainingFinished
{
    public string TrainingId { get; init; } = "";
    public string Status { get; init; } = ""; // "completed" | "cancelled"
    public float FinalLoss { get; init; }
    public int TotalEpochs { get; init; }
    public float ElapsedSecs { get; init; }
}

public sealed record TrainingError
{
    public string TrainingId { get; init; } = "";
    public string Message { get; init; } = "";
}

public sealed record TrainingConfigSummary
{
    public string Optimizer { get; init; } = "";
    public float Lr { get; init; }
    public int BatchSize { get; init; }
    public int Epochs { get; init; }
}

/// <summary>A completed training run, retained for history (mirror of <c>models::TrainingRun</c>).</summary>
public sealed record TrainingRun
{
    public string Id { get; init; } = "";
    public string NetworkId { get; init; } = "";
    public DateTimeOffset StartedAt { get; init; }
    public DateTimeOffset FinishedAt { get; init; }
    public string Status { get; init; } = ""; // "completed" | "cancelled" | "aborted" | "error"
    public TrainingConfigSummary ConfigSummary { get; init; } = new();
    public int TotalEpochs { get; init; }
    public int EpochsRun { get; init; }
    public float FinalLoss { get; init; }
    public float ElapsedSecs { get; init; }
    public List<float> LossHistory { get; init; } = new();
}
