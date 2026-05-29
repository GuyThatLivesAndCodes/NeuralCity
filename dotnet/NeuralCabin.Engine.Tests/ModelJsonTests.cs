using System.Text.Json;
using NeuralCabin.Engine;
using Xunit;

namespace NeuralCabin.Engine.Tests;

public class ModelJsonTests
{
    [Fact]
    public void Model_round_trips_through_json()
    {
        var model = Model.FromSpecs(2, new LayerSpec[]
        {
            new LinearSpec(2, 4),
            new ActivationSpec(Activation.ReLU),
            new LinearSpec(4, 1),
        }, 7);

        var json = JsonSerializer.Serialize(model, EngineModelJson.Options);
        var back = JsonSerializer.Deserialize<Model>(json, EngineModelJson.Options)!;

        Assert.Equal(2, back.InputDim);
        Assert.Equal(3, back.Layers.Count);
        Assert.Equal(1, back.ComputeOutputDim());
        Assert.Equal(model.ParameterCount, back.ParameterCount);

        var l0 = Assert.IsType<LinearLayer>(back.Layers[0]);
        Assert.Equal(2, l0.InDim);
        Assert.Equal(4, l0.OutDim);
        Assert.Equal(((LinearLayer)model.Layers[0]).W.Data, l0.W.Data);
        Assert.IsType<ActivationLayer>(back.Layers[1]);
        Assert.Equal(Activation.ReLU, ((ActivationLayer)back.Layers[1]).Activation);
    }

    [Fact]
    public void Serializes_in_serde_compatible_shape()
    {
        var model = Model.FromSpecs(2, new LayerSpec[]
        {
            new LinearSpec(2, 2),
            new ActivationSpec(Activation.Tanh),
        }, 1);

        var json = JsonSerializer.Serialize(model, EngineModelJson.Options);

        Assert.Contains("\"input_dim\": 2", json);
        Assert.Contains("\"seed\": 1", json);
        // Externally-tagged layer union, exactly as serde emits it.
        Assert.Contains("\"Linear\"", json);
        Assert.Contains("\"in_dim\": 2", json);
        Assert.Contains("\"out_dim\": 2", json);
        Assert.Contains("\"shape\"", json);
        Assert.Contains("\"data\"", json);
        Assert.Contains("\"Activation\": \"Tanh\"", json);
    }

    [Fact]
    public void Loads_a_hand_written_serde_style_model()
    {
        // The exact shape the Rust backend writes (model field of ModelFile).
        const string json = """
            {
              "input_dim": 2,
              "layers": [
                { "Linear": { "in_dim": 2, "out_dim": 2,
                  "w": { "shape": [2, 2], "data": [0.1, 0.2, 0.3, 0.4] },
                  "b": { "shape": [1, 2], "data": [0.0, 0.0] } } },
                { "Activation": "Sigmoid" }
              ],
              "seed": 42
            }
            """;

        var model = JsonSerializer.Deserialize<Model>(json, EngineModelJson.Options)!;
        Assert.Equal(2, model.InputDim);
        Assert.Equal(42ul, model.Seed);
        Assert.Equal(2, model.Layers.Count);
        var lin = Assert.IsType<LinearLayer>(model.Layers[0]);
        Assert.Equal(new[] { 0.1f, 0.2f, 0.3f, 0.4f }, lin.W.Data);
        Assert.Equal(Activation.Sigmoid, ((ActivationLayer)model.Layers[1]).Activation);
    }
}
