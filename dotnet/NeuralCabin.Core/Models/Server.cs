namespace NeuralCabin.Core.Models;

/// <summary>Per-endpoint permission flags for an embedded API server.</summary>
public sealed record ServerPermissions
{
    public bool AllowList { get; init; } = true;
    public bool AllowInference { get; init; } = true;
    public bool AllowExport { get; init; } = true;
    public bool AllowUpload { get; init; }
    public bool AllowTrain { get; init; }
    public bool AllowCreate { get; init; }
    public bool AllowDelete { get; init; }
}

/// <summary>Saved configuration for an embedded API server (mirror of <c>ServerConfig</c>).</summary>
public sealed record ServerConfig
{
    public string Id { get; init; } = "";
    public string Name { get; init; } = "";
    public int Port { get; init; }
    public bool LocalhostOnly { get; init; } = true;
    public string AuthToken { get; init; } = "";
    public ServerPermissions Permissions { get; init; } = new();
    public bool AutoStart { get; init; }
    public DateTimeOffset CreatedAt { get; init; }
}

/// <summary>
/// Runtime view of a server. serde flattens the embedded <c>ServerConfig</c>
/// (<c>#[serde(flatten)]</c>), so this DTO declares every config field inline
/// to reproduce the flat JSON the frontend expects.
/// </summary>
public sealed record ServerSummary
{
    public string Id { get; init; } = "";
    public string Name { get; init; } = "";
    public int Port { get; init; }
    public bool LocalhostOnly { get; init; } = true;
    public string AuthToken { get; init; } = "";
    public ServerPermissions Permissions { get; init; } = new();
    public bool AutoStart { get; init; }
    public DateTimeOffset CreatedAt { get; init; }

    public bool Running { get; init; }
    public long RequestCount { get; init; }
    public string? LastError { get; init; }
}

public sealed record CreateServerRequest
{
    public string Name { get; init; } = "";
    public int Port { get; init; }
    public bool? LocalhostOnly { get; init; }
    public string? AuthToken { get; init; }
    public ServerPermissions? Permissions { get; init; }
}

public sealed record UpdateServerRequest
{
    public string Id { get; init; } = "";
    public string? Name { get; init; }
    public int? Port { get; init; }
    public bool? LocalhostOnly { get; init; }
    public string? AuthToken { get; init; }
    public ServerPermissions? Permissions { get; init; }
    public bool? AutoStart { get; init; }
}
