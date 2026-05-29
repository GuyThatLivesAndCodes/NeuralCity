using NeuralCabin.Host.Ipc;
using NeuralCabin.Engine;
using CoreModels = NeuralCabin.Core.Models;

namespace NeuralCabin.Host.Services;

/// <summary>Converts the IPC layer DTOs into engine layer specs (with validation).</summary>
internal static class EngineMapping
{
    /// <summary>
    /// Mirror of Rust <c>build_layer_specs</c>: validate linear-dim continuity
    /// and out_dim &gt; 0, validate activation names, and produce engine specs.
    /// Returns the specs and the running output dimension.
    /// </summary>
    public static (List<LayerSpec> Specs, int OutputDim) ToSpecs(IReadOnlyList<CoreModels.LayerDef> layers, int inputDim)
    {
        var specs = new List<LayerSpec>(layers.Count);
        var cur = inputDim;
        for (var i = 0; i < layers.Count; i++)
        {
            switch (layers[i])
            {
                case CoreModels.LinearLayer lin:
                    if (lin.InDim != cur)
                        throw new CommandException($"layer {i}: linear in_dim {lin.InDim} doesn't match running dim {cur}");
                    if (lin.OutDim == 0)
                        throw new CommandException($"layer {i}: linear out_dim must be > 0");
                    specs.Add(new LinearSpec(lin.InDim, lin.OutDim));
                    cur = lin.OutDim;
                    break;
                case CoreModels.ActivationLayer act:
                    try { specs.Add(new ActivationSpec(Activations.Parse(act.Activation))); }
                    catch (ArgumentException e) { throw new CommandException(e.Message); }
                    break;
                default:
                    throw new CommandException($"layer {i}: unrecognized layer type");
            }
        }
        return (specs, cur);
    }
}
