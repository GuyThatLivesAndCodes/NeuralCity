using NeuralCabin.Core.Models;
using NeuralCabin.Host.Ipc;
using NeuralCabin.Host.State;
using NeuralCabin.Engine;

namespace NeuralCabin.Host.Services;

/// <summary>
/// Network CRUD over <see cref="AppState"/> — the .NET port of
/// <c>create_network</c> / <c>list_networks</c> / <c>get_network</c> /
/// <c>delete_network</c>. Feed-forward networks materialize an engine
/// <see cref="Model"/> at create time (so parameter counts and training are
/// real); next-token networks store their hidden chain until a vocabulary is
/// built; transformer materialization awaits the transformer engine port.
/// </summary>
public sealed class NetworkService
{
    private readonly AppState _state;

    public NetworkService(AppState state) => _state = state;

    public Network Create(CreateNetworkRequest request)
    {
        if (!NetworkKinds.All.Contains(request.Kind))
            throw new CommandException(
                $"unknown network kind '{request.Kind}' (expected one of: {string.Join(", ", NetworkKinds.All)})");
        if (request.Layers.Count == 0 && request.Kind == NetworkKinds.Feedforward)
            throw new CommandException("feedforward network must have at least one layer");

        var name = (request.Name ?? string.Empty).Trim();
        if (name.Length == 0) throw new CommandException("network name cannot be empty");

        var id = Guid.NewGuid().ToString();
        var now = DateTimeOffset.UtcNow;

        Network network;
        if (request.Kind == NetworkKinds.Feedforward)
        {
            var inputDim = request.InputDim ?? 0;
            if (inputDim == 0) throw new CommandException("feedforward network requires input_dim > 0");

            var (specs, outputDim) = EngineMapping.ToSpecs(request.Layers, inputDim);
            var model = Model.FromSpecs(inputDim, specs, unchecked((ulong)request.Seed));
            _state.SetModel(id, model);
            if (_state.DataDir is { } dir) Persistence.SaveModel(dir, id, model);

            network = new Network
            {
                Id = id, Name = name, Kind = request.Kind, Seed = request.Seed, CreatedAt = now,
                InputDim = inputDim, OutputDim = outputDim,
                Layers = request.Layers, ParameterCount = model.ParameterCount,
            };
        }
        else if (request.Kind == NetworkKinds.Transformer)
        {
            throw new NotYetMigratedException("create_network (transformer)");
        }
        else // next_token
        {
            var contextSize = request.ContextSize ?? 0;
            if (contextSize == 0) throw new CommandException("next-token network requires context_size > 0");

            network = new Network
            {
                Id = id, Name = name, Kind = request.Kind, Seed = request.Seed, CreatedAt = now,
                InputDim = 0, OutputDim = 0,
                Layers = new List<LayerDef>(), ParameterCount = 0,
                HiddenLayers = request.Layers, ContextSize = contextSize,
            };
        }

        _state.Networks[id] = network;
        Persistence.SaveState(_state);
        return network;
    }

    public IReadOnlyList<Network> List() => _state.Networks.Values.ToList();

    public Network Get(string id) => _state.GetNetwork(id) ?? throw new CommandException("Network not found");

    public bool Delete(string id)
    {
        var removed = _state.Networks.TryRemove(id, out _);
        _state.Models.TryRemove(id, out _);
        _state.Corpora.TryRemove(id, out _);
        _state.TrainingHistory.TryRemove(id, out _);
        if (removed && _state.DataDir is { } dir) Persistence.DeleteModel(dir, id);
        Persistence.SaveState(_state);
        return removed;
    }
}
