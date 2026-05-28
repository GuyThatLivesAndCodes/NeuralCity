using System.Text.Json;
using NeuralCabin.Core.Json;
using NeuralCabin.Core.Models;
using Xunit;

namespace NeuralCabin.Core.Tests;

/// <summary>
/// Guards the JSON contract with the React frontend. Every assertion here is
/// something the existing Tauri/serde backend already produces; if these pass,
/// the .NET backend's payloads are interchangeable with the Rust ones.
/// </summary>
public class JsonContractTests
{
    private static readonly JsonSerializerOptions Options = NeuralCabinJson.Options;

    [Fact]
    public void Network_serializes_with_snake_case_and_layer_discriminators()
    {
        var network = new Network
        {
            Id = "abc",
            Name = "XOR",
            Kind = NetworkKinds.Feedforward,
            Seed = 42,
            CreatedAt = DateTimeOffset.UnixEpoch,
            InputDim = 2,
            OutputDim = 1,
            ParameterCount = 25,
            Layers = new List<LayerDef>
            {
                new LinearLayer { InDim = 2, OutDim = 8 },
                new ActivationLayer { Activation = "relu" },
                new LinearLayer { InDim = 8, OutDim = 1 },
                new ActivationLayer { Activation = "sigmoid" },
            },
        };

        var json = JsonSerializer.Serialize(network, Options);

        Assert.Contains("\"kind\":\"feedforward\"", json);
        Assert.Contains("\"input_dim\":2", json);
        Assert.Contains("\"output_dim\":1", json);
        Assert.Contains("\"parameter_count\":25", json);
        Assert.Contains("\"created_at\":\"1970-01-01T00:00:00.000000Z\"", json);
        // Internally-tagged layer union, exactly as serde emits it.
        Assert.Contains("\"type\":\"linear\",\"in_dim\":2,\"out_dim\":8", json);
        Assert.Contains("\"type\":\"activation\",\"activation\":\"relu\"", json);
        // Option::None fields without skip_serializing_if are emitted as null.
        Assert.Contains("\"hidden_layers\":null", json);
        Assert.Contains("\"context_size\":null", json);
        Assert.Contains("\"transformer\":null", json);
    }

    [Fact]
    public void Network_round_trips_through_polymorphic_layers()
    {
        var network = new Network
        {
            Id = "n1",
            Name = "Net",
            Kind = NetworkKinds.Feedforward,
            Seed = 7,
            CreatedAt = DateTimeOffset.UtcNow,
            InputDim = 3,
            OutputDim = 2,
            Layers = new List<LayerDef>
            {
                new LinearLayer { InDim = 3, OutDim = 2 },
                new ActivationLayer { Activation = "tanh" },
            },
        };

        var json = JsonSerializer.Serialize(network, Options);
        var back = JsonSerializer.Deserialize<Network>(json, Options)!;

        Assert.Equal("n1", back.Id);
        Assert.Equal(NetworkKinds.Feedforward, back.Kind);
        Assert.Equal(2, back.Layers.Count);
        var linear = Assert.IsType<LinearLayer>(back.Layers[0]);
        Assert.Equal(3, linear.InDim);
        Assert.Equal(2, linear.OutDim);
        var activation = Assert.IsType<ActivationLayer>(back.Layers[1]);
        Assert.Equal("tanh", activation.Activation);
    }

    [Fact]
    public void CreateNetworkRequest_deserializes_from_frontend_json()
    {
        // The exact shape api.ts places inside the `req` argument.
        const string json = """
            {
              "name": "MyNet",
              "kind": "feedforward",
              "seed": 123,
              "input_dim": 4,
              "layers": [
                { "type": "linear", "in_dim": 4, "out_dim": 16 },
                { "type": "activation", "activation": "relu" },
                { "type": "linear", "in_dim": 16, "out_dim": 3 }
              ]
            }
            """;

        var req = JsonSerializer.Deserialize<CreateNetworkRequest>(json, Options)!;

        Assert.Equal("MyNet", req.Name);
        Assert.Equal("feedforward", req.Kind);
        Assert.Equal(123, req.Seed);
        Assert.Equal(4, req.InputDim);
        Assert.Equal(3, req.Layers.Count);
        Assert.IsType<LinearLayer>(req.Layers[0]);
        Assert.IsType<ActivationLayer>(req.Layers[1]);
    }

    [Fact]
    public void TransformerHParams_apply_serde_defaults_when_absent()
    {
        const string json = """
            { "n_ctx": 64, "n_embd": 32, "n_layers": 2, "n_heads": 2, "n_ff": 64 }
            """;

        var hp = JsonSerializer.Deserialize<TransformerHParams>(json, Options)!;

        Assert.Equal(64, hp.NCtx);
        Assert.Equal(32, hp.NEmbd);
        // serde #[serde(default = ...)] equivalents.
        Assert.Equal(10000.0f, hp.RopeTheta);
        Assert.Equal(1e-5f, hp.RmsEps);
    }

    [Fact]
    public void InferResponse_omits_null_fields_like_skip_serializing_if()
    {
        var streaming = new InferResponse { InferenceId = "job-1" };
        var streamingJson = JsonSerializer.Serialize(streaming, Options);
        Assert.Contains("\"inference_id\":\"job-1\"", streamingJson);
        Assert.DoesNotContain("output", streamingJson);

        var feedforward = new InferResponse { Output = new List<float> { 0.1f, 0.9f } };
        var feedforwardJson = JsonSerializer.Serialize(feedforward, Options);
        Assert.Contains("\"output\":[", feedforwardJson);
        Assert.DoesNotContain("inference_id", feedforwardJson);
    }

    [Fact]
    public void ServerSummary_serializes_flat_like_serde_flatten()
    {
        var summary = new ServerSummary
        {
            Id = "s1",
            Name = "api",
            Port = 8080,
            AuthToken = "tok",
            CreatedAt = DateTimeOffset.UnixEpoch,
            Running = true,
            RequestCount = 5,
            LastError = null,
        };

        var json = JsonSerializer.Serialize(summary, Options);

        // Config fields and runtime fields must sit at the same (flat) level.
        Assert.Contains("\"id\":\"s1\"", json);
        Assert.Contains("\"port\":8080", json);
        Assert.Contains("\"localhost_only\":true", json);
        Assert.Contains("\"auth_token\":\"tok\"", json);
        Assert.Contains("\"auto_start\":false", json);
        Assert.Contains("\"running\":true", json);
        Assert.Contains("\"request_count\":5", json);
        Assert.Contains("\"last_error\":null", json);
        // Must NOT be nested under a "config" key.
        Assert.DoesNotContain("\"config\"", json);
    }

    [Fact]
    public void ServerPermissions_defaults_match_rust_defaults()
    {
        // serde defaults: list/inference/export true, the rest false.
        var permissions = new ServerPermissions();
        Assert.True(permissions.AllowList);
        Assert.True(permissions.AllowInference);
        Assert.True(permissions.AllowExport);
        Assert.False(permissions.AllowUpload);
        Assert.False(permissions.AllowTrain);
        Assert.False(permissions.AllowCreate);
        Assert.False(permissions.AllowDelete);
    }
}
