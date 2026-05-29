using NeuralCabin.Core.Ipc;
using NeuralCabin.Core.Models;
using NeuralCabin.Host.Commands;
using NeuralCabin.Host.Ipc;
using NeuralCabin.Host.Services;
using NeuralCabin.Host.State;

namespace NeuralCabin.Host;

/// <summary>
/// Composition root for the backend. Wires the domain services into a
/// <see cref="CommandRegistry"/> over a shared <see cref="State.AppState"/>,
/// loads persisted state, and exposes a router factory for the Photino shell
/// (and tests). As each domain is ported it registers its real handlers here
/// before the <see cref="StubCommands"/> fallback.
/// </summary>
public sealed class NeuralCabinBackend
{
    public CommandRegistry Registry { get; }
    public AppState State { get; }
    public NetworkService Networks { get; }
    public CorpusService Corpus { get; }
    public TrainingService Training { get; }
    public InferenceService Inference { get; }

    public NeuralCabinBackend(string? dataDir = null)
    {
        State = new AppState { DataDir = dataDir };
        Persistence.LoadState(State);

        Networks = new NetworkService(State);
        Corpus = new CorpusService(State);
        Training = new TrainingService(State);
        Inference = new InferenceService(State);

        Registry = new CommandRegistry();
        RegisterNetworks();
        RegisterCorpus();
        RegisterTraining();
        RegisterInference();

        // Everything not yet ported fails loudly with a migration-status message.
        StubCommands.RegisterUnimplemented(Registry);
    }

    private void RegisterNetworks()
    {
        Registry.Register(CommandNames.CreateNetwork, ctx =>
            Task.FromResult<object?>(Networks.Create(ctx.Arg<CreateNetworkRequest>("req"))));
        Registry.Register(CommandNames.ListNetworks, _ =>
            Task.FromResult<object?>(Networks.List()));
        Registry.Register(CommandNames.GetNetwork, ctx =>
            Task.FromResult<object?>(Networks.Get(ctx.ArgString("id"))));
        Registry.Register(CommandNames.DeleteNetwork, ctx =>
            Task.FromResult<object?>(Networks.Delete(ctx.ArgString("id"))));
    }

    private void RegisterCorpus()
    {
        Registry.Register(CommandNames.SetCorpus, ctx =>
            Task.FromResult<object?>(Corpus.Set(ctx.Arg<SetCorpusRequest>("req"))));
        Registry.Register(CommandNames.GetCorpus, ctx =>
            Task.FromResult<object?>(Corpus.Get(ctx.ArgString("networkId"))));
        Registry.Register(CommandNames.CorpusStats, ctx =>
            Task.FromResult<object?>(Corpus.StatsFor(ctx.ArgString("networkId"))));
    }

    private void RegisterTraining()
    {
        Registry.Register(CommandNames.StartTraining, ctx =>
            Task.FromResult<object?>(Training.Start(ctx.Arg<StartTrainingRequest>("req"), ctx.Events)));
        Registry.Register(CommandNames.StopTraining, ctx =>
            Task.FromResult<object?>(Training.Stop(ctx.ArgString("trainingId"))));
        Registry.Register(CommandNames.AbortTraining, ctx =>
            Task.FromResult<object?>(Training.Abort(ctx.ArgString("trainingId"))));
        Registry.Register(CommandNames.GetTrainingStatus, ctx =>
            Task.FromResult<object?>(Training.Status(ctx.ArgString("trainingId"))));
        Registry.Register(CommandNames.GetTrainingHistory, ctx =>
            Task.FromResult<object?>(Training.History(ctx.ArgString("networkId"))));
        Registry.Register(CommandNames.ClearTrainingHistory, ctx =>
        {
            Training.ClearHistory(ctx.ArgString("networkId"));
            return Task.FromResult<object?>(null);
        });
    }

    private void RegisterInference()
    {
        Registry.Register(CommandNames.Infer, ctx =>
            Task.FromResult<object?>(Inference.Infer(ctx.Arg<InferRequest>("req"))));
        Registry.Register(CommandNames.InferWithActivations, ctx =>
            Task.FromResult<object?>(Inference.InferWithActivations(
                ctx.ArgString("networkId"), ctx.Arg<List<float>>("features"))));
        Registry.Register(CommandNames.StopInference, ctx =>
            Task.FromResult<object?>(Inference.Stop(ctx.ArgString("inferenceId"))));
    }

    /// <summary>Build a router that dispatches this backend over the given transport.</summary>
    public IpcRouter CreateRouter(IIpcTransport transport) => new(transport, Registry);
}
