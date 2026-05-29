using System.Collections.Concurrent;
using NeuralCabin.Core.Models;
using NeuralCabin.Engine;

namespace NeuralCabin.Host.State;

/// <summary>
/// In-memory application state — the .NET analogue of the Tauri
/// <c>AppState</c>. Network metadata, materialized engine models, corpora, and
/// training history live here. Models are stored as immutable snapshots
/// (replaced wholesale, never mutated in place) so inference can read them
/// concurrently with training without locking.
/// </summary>
public sealed class AppState
{
    public ConcurrentDictionary<string, Network> Networks { get; } = new(StringComparer.Ordinal);
    public ConcurrentDictionary<string, Model> Models { get; } = new(StringComparer.Ordinal);
    public ConcurrentDictionary<string, Corpus> Corpora { get; } = new(StringComparer.Ordinal);
    public ConcurrentDictionary<string, List<TrainingRun>> TrainingHistory { get; } = new(StringComparer.Ordinal);

    /// <summary>Where state.json + model files live. Null disables persistence (tests).</summary>
    public string? DataDir { get; set; }

    public Network? GetNetwork(string id) => Networks.TryGetValue(id, out var n) ? n : null;

    public Model? GetModel(string id) => Models.TryGetValue(id, out var m) ? m : null;

    /// <summary>Store an immutable model snapshot (callers pass an owned clone).</summary>
    public void SetModel(string id, Model model) => Models[id] = model;
}
