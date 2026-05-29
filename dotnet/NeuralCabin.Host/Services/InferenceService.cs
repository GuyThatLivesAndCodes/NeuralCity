using NeuralCabin.Core.Models;
using NeuralCabin.Host.Ipc;
using NeuralCabin.Host.State;
using NeuralCabin.Engine;

namespace NeuralCabin.Host.Services;

/// <summary>
/// Inference (port of <c>infer</c> / <c>infer_with_activations</c>). Feed-forward
/// is synchronous; next-token / transformer streaming is added in later slices.
/// </summary>
public sealed class InferenceService
{
    private readonly AppState _state;

    public InferenceService(AppState state) => _state = state;

    public InferResponse Infer(InferRequest request)
    {
        var net = _state.GetNetwork(request.NetworkId) ?? throw new CommandException("Network not found");
        if (net.Kind != NetworkKinds.Feedforward)
            throw new NotYetMigratedException($"infer ({net.Kind})");

        var features = request.Features ?? throw new CommandException("Feedforward inference requires `features`");
        if (features.Count != net.InputDim)
            throw new CommandException($"expected {net.InputDim} features, got {features.Count}");

        var model = _state.GetModel(request.NetworkId) ?? throw new CommandException("Model not materialized");
        var output = model.Predict(new Tensor(new[] { 1, net.InputDim }, features.ToArray()));
        return new InferResponse { Output = output.Data.ToList() };
    }

    public InferActivations InferWithActivations(string networkId, IReadOnlyList<float> features)
    {
        var net = _state.GetNetwork(networkId) ?? throw new CommandException("Network not found");
        if (net.Kind != NetworkKinds.Feedforward)
            throw new CommandException("activations are only available for feed-forward networks");
        if (features.Count != net.InputDim)
            throw new CommandException($"expected {net.InputDim} features, got {features.Count}");

        var model = _state.GetModel(networkId) ?? throw new CommandException("Model not materialized");
        var acts = model.PredictWithActivations(new Tensor(new[] { 1, net.InputDim }, features.ToArray()));

        var names = new List<string> { "input" };
        foreach (var layer in model.Layers)
            names.Add(layer switch
            {
                Engine.LinearLayer ll => $"Linear ({ll.InDim} -> {ll.OutDim})",
                Engine.ActivationLayer al => $"Activation: {al.Activation.Name()}",
                _ => "layer",
            });

        return new InferActivations
        {
            LayerNames = names,
            Activations = acts.Select(t => t.Data.ToList()).ToList(),
            Dims = acts.Select(t => t.Cols).ToList(),
        };
    }

    /// <summary>Feed-forward inference is synchronous, so there is nothing to stop.</summary>
    public bool Stop(string inferenceId) => false;
}
