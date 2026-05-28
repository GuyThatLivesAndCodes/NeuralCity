namespace NeuralCabin.Core.Ipc;

/// <summary>
/// Canonical command names exchanged over the IPC bridge. These are the exact
/// strings the React frontend passes to <c>invoke(name, args)</c> — identical
/// to the Tauri <c>generate_handler!</c> registration in
/// <c>src-tauri/src/lib.rs</c>.
/// </summary>
public static class CommandNames
{
    // Networks
    public const string CreateNetwork = "create_network";
    public const string ListNetworks = "list_networks";
    public const string GetNetwork = "get_network";
    public const string DeleteNetwork = "delete_network";

    // Corpus
    public const string SetCorpus = "set_corpus";
    public const string GetCorpus = "get_corpus";
    public const string CorpusStats = "corpus_stats";

    // Vocabulary
    public const string BuildVocabulary = "build_vocabulary";
    public const string SetAdvancedVocabulary = "set_advanced_vocabulary";
    public const string GetVocabulary = "get_vocabulary";
    public const string TokenizePreview = "tokenize_preview";

    // Training
    public const string StartTraining = "start_training";
    public const string StopTraining = "stop_training";
    public const string AbortTraining = "abort_training";
    public const string GetTrainingStatus = "get_training_status";
    public const string GetTrainingHistory = "get_training_history";
    public const string ClearTrainingHistory = "clear_training_history";

    // Inference
    public const string Infer = "infer";
    public const string InferWithActivations = "infer_with_activations";
    public const string StopInference = "stop_inference";

    // Export
    public const string ExportNetwork = "export_network";

    // Server
    public const string ListServers = "list_servers";
    public const string CreateServer = "create_server";
    public const string UpdateServer = "update_server";
    public const string DeleteServer = "delete_server";
    public const string StartServer = "start_server";
    public const string StopServer = "stop_server";
    public const string ServerStatus = "server_status";

    /// <summary>Every command the frontend may invoke, for registry validation.</summary>
    public static readonly IReadOnlyList<string> All = new[]
    {
        CreateNetwork, ListNetworks, GetNetwork, DeleteNetwork,
        SetCorpus, GetCorpus, CorpusStats,
        BuildVocabulary, SetAdvancedVocabulary, GetVocabulary, TokenizePreview,
        StartTraining, StopTraining, AbortTraining, GetTrainingStatus,
        GetTrainingHistory, ClearTrainingHistory,
        Infer, InferWithActivations, StopInference,
        ExportNetwork,
        ListServers, CreateServer, UpdateServer, DeleteServer,
        StartServer, StopServer, ServerStatus,
    };
}
