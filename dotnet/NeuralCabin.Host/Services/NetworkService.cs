using System.Collections.Concurrent;
using NeuralCabin.Core.Models;
using NeuralCabin.Host.Ipc;

namespace NeuralCabin.Host.Services;

/// <summary>
/// In-memory network registry — the .NET port of the network CRUD half of
/// <c>create_network</c> / <c>list_networks</c> / <c>get_network</c> /
/// <c>delete_network</c> in <c>src-tauri/src/lib.rs</c>.
///
/// Feed-forward and next-token networks are fully supported here because their
/// creation is engine-free (next-token only stores the hidden chain; feed-
/// forward parameter counting is a closed-form sum). Transformer creation
/// materializes a model to count parameters and therefore waits on the engine
/// port. Disk persistence (the shared <c>state.json</c>) is a later milestone;
/// see MIGRATION.md.
/// </summary>
public sealed class NetworkService
{
    private static readonly HashSet<string> ValidActivations =
        new(StringComparer.OrdinalIgnoreCase) { "identity", "relu", "sigmoid", "tanh", "softmax" };

    private readonly ConcurrentDictionary<string, Network> _networks = new(StringComparer.Ordinal);

    public Network Create(CreateNetworkRequest request)
    {
        if (!NetworkKinds.All.Contains(request.Kind))
            throw new CommandException(
                $"unknown network kind '{request.Kind}' (expected one of: {string.Join(", ", NetworkKinds.All)})");

        if (request.Layers.Count == 0 && request.Kind == NetworkKinds.Feedforward)
            throw new CommandException("feedforward network must have at least one layer");

        var name = (request.Name ?? string.Empty).Trim();
        if (name.Length == 0)
            throw new CommandException("network name cannot be empty");

        var id = Guid.NewGuid().ToString();
        var now = DateTimeOffset.UtcNow;

        Network network;
        if (request.Kind == NetworkKinds.Feedforward)
        {
            var inputDim = request.InputDim ?? 0;
            if (inputDim == 0)
                throw new CommandException("feedforward network requires input_dim > 0");

            var (parameterCount, outputDim) = ComputeFeedforward(request.Layers, inputDim);
            network = new Network
            {
                Id = id,
                Name = name,
                Kind = request.Kind,
                Seed = request.Seed,
                CreatedAt = now,
                Trained = false,
                Pretrained = false,
                InputDim = inputDim,
                OutputDim = outputDim,
                Layers = request.Layers,
                ParameterCount = parameterCount,
                HiddenLayers = null,
                ContextSize = null,
                Transformer = null,
            };
        }
        else if (request.Kind == NetworkKinds.Transformer)
        {
            // Counting transformer parameters requires building the model, which
            // depends on the not-yet-ported engine. Fail clearly for now.
            throw new NotYetMigratedException("create_network (transformer)");
        }
        else // next_token: only the hidden chain is meaningful until a vocab is built.
        {
            var contextSize = request.ContextSize ?? 0;
            if (contextSize == 0)
                throw new CommandException("next-token network requires context_size > 0");

            network = new Network
            {
                Id = id,
                Name = name,
                Kind = request.Kind,
                Seed = request.Seed,
                CreatedAt = now,
                Trained = false,
                Pretrained = false,
                InputDim = 0,
                OutputDim = 0,
                Layers = new List<LayerDef>(),
                ParameterCount = 0,
                HiddenLayers = request.Layers,
                ContextSize = contextSize,
                Transformer = null,
            };
        }

        _networks[id] = network;
        return network;
    }

    public IReadOnlyList<Network> List() => _networks.Values.ToList();

    public Network Get(string id) =>
        _networks.TryGetValue(id, out var network)
            ? network
            : throw new CommandException("Network not found");

    public bool Delete(string id) => _networks.TryRemove(id, out _);

    /// <summary>
    /// Mirror of <c>build_layer_specs</c> + <c>Model::parameter_count</c>: walk
    /// the chain validating linear dim continuity, summing weight+bias counts,
    /// and returning the final running dimension as the output dim.
    /// </summary>
    private static (long ParameterCount, int OutputDim) ComputeFeedforward(IReadOnlyList<LayerDef> layers, int inputDim)
    {
        long parameterCount = 0;
        var current = inputDim;

        for (var i = 0; i < layers.Count; i++)
        {
            switch (layers[i])
            {
                case LinearLayer linear:
                    if (linear.InDim != current)
                        throw new CommandException(
                            $"layer {i}: linear in_dim {linear.InDim} doesn't match running dim {current}");
                    if (linear.OutDim == 0)
                        throw new CommandException($"layer {i}: linear out_dim must be > 0");
                    parameterCount += (long)linear.InDim * linear.OutDim + linear.OutDim; // weights + bias
                    current = linear.OutDim;
                    break;

                case ActivationLayer activation:
                    if (!ValidActivations.Contains(activation.Activation))
                        throw new CommandException($"unknown activation '{activation.Activation.ToLowerInvariant()}'");
                    break;

                default:
                    throw new CommandException($"layer {i}: unrecognized layer type");
            }
        }

        return (parameterCount, current);
    }
}
