using System.Text.Json;
using System.Text.Json.Serialization;
using NeuralCabin.Core.Json;
using NeuralCabin.Core.Models;
using NeuralCabin.Engine;

namespace NeuralCabin.Host.State;

/// <summary>
/// On-disk persistence: a metadata-only <c>state.json</c> plus one
/// <c>models/&lt;id&gt;.json</c> per model. Model files use the engine's
/// serde-compatible envelope so they interoperate with files written by the
/// Rust backend.
/// </summary>
public static class Persistence
{
    public const string StateFilename = "state.json";
    public const int StateFormatVersion = 1;

    private static string ModelsDir(string dataDir) => Path.Combine(dataDir, "models");
    private static string ModelPath(string dataDir, string id) => Path.Combine(ModelsDir(dataDir), $"{id}.json");

    // ── State (metadata only — no weights) ───────────────────────────────────

    public static void SaveState(AppState state)
    {
        if (state.DataDir is not { } dir) return;
        Directory.CreateDirectory(dir);

        var persisted = new PersistedState
        {
            FormatVersion = StateFormatVersion,
            Networks = state.Networks.Values.ToList(),
            Corpora = state.Corpora.Values.ToList(),
            TrainingHistory = state.TrainingHistory.ToDictionary(kv => kv.Key, kv => kv.Value),
        };

        // Atomic replace: write tmp then move over the target.
        var json = JsonSerializer.Serialize(persisted, NeuralCabinJson.Options);
        var tmp = Path.Combine(dir, StateFilename + ".tmp");
        File.WriteAllText(tmp, json);
        File.Move(tmp, Path.Combine(dir, StateFilename), overwrite: true);
    }

    public static void LoadState(AppState state)
    {
        if (state.DataDir is not { } dir) return;
        var path = Path.Combine(dir, StateFilename);
        if (!File.Exists(path)) return;

        var persisted = JsonSerializer.Deserialize<PersistedState>(File.ReadAllText(path), NeuralCabinJson.Options);
        if (persisted is null) return;

        foreach (var n in persisted.Networks) state.Networks[n.Id] = n;
        foreach (var c in persisted.Corpora) state.Corpora[c.NetworkId] = c;
        foreach (var (k, v) in persisted.TrainingHistory) state.TrainingHistory[k] = v;

        // Eagerly load any model files that exist (small for feed-forward nets).
        foreach (var n in persisted.Networks)
        {
            var model = LoadModel(dir, n.Id);
            if (model is not null) state.Models[n.Id] = model;
        }
    }

    // ── Model files (serde-compatible envelope) ──────────────────────────────

    public static void SaveModel(string dataDir, string id, Model model)
    {
        Directory.CreateDirectory(ModelsDir(dataDir));
        var file = new ModelFile { Model = model };
        var json = JsonSerializer.Serialize(file, EngineModelJson.Options);
        var path = ModelPath(dataDir, id);
        var tmp = path + ".tmp";
        File.WriteAllText(tmp, json);
        File.Move(tmp, path, overwrite: true);
    }

    public static Model? LoadModel(string dataDir, string id)
    {
        var path = ModelPath(dataDir, id);
        if (!File.Exists(path)) return null;
        var file = JsonSerializer.Deserialize<ModelFile>(File.ReadAllText(path), EngineModelJson.Options);
        return file?.Model;
    }

    public static void DeleteModel(string dataDir, string id)
    {
        var path = ModelPath(dataDir, id);
        if (File.Exists(path)) File.Delete(path);
    }
}

/// <summary>The metadata-only state.json payload.</summary>
public sealed record PersistedState
{
    public int FormatVersion { get; init; } = Persistence.StateFormatVersion;
    public List<Network> Networks { get; init; } = new();
    public List<Corpus> Corpora { get; init; } = new();
    public Dictionary<string, List<TrainingRun>> TrainingHistory { get; init; } = new();
}

/// <summary>
/// Model-file envelope, compatible with the Rust <c>persistence::ModelFile</c>.
/// Only the fields the .NET app needs are declared; unknown fields written by
/// the Rust backend are ignored on read.
/// </summary>
public sealed class ModelFile
{
    public int FormatVersion { get; set; } = 2;
    public string CreatedWith { get; set; } = "neuralcabin-dotnet";
    public Model Model { get; set; } = new();

    [JsonExtensionData]
    public Dictionary<string, JsonElement>? Extra { get; set; }
}
