using System.Text.Json;
using NeuralCabin.Core.Ipc;
using NeuralCabin.Host;
using NeuralCabin.Host.Ipc;
using Xunit;

namespace NeuralCabin.Host.Tests;

/// <summary>
/// End-to-end proof that the feed-forward vertical works entirely in .NET,
/// driven through the IPC bridge exactly as the React UI would: create a
/// network, attach an XOR corpus, train it (streaming events), then run
/// inference on the trained weights.
/// </summary>
public class FeedforwardFlowTests
{
    private readonly NeuralCabinBackend _backend = new();
    private readonly FakeTransport _transport = new();
    private readonly IpcRouter _router;

    public FeedforwardFlowTests() => _router = _backend.CreateRouter(_transport);

    private async Task<JsonElement> Invoke(string cmd, object? args)
    {
        var id = Guid.NewGuid().ToString();
        await _router.HandleMessageAsync(JsonSerializer.Serialize(new { kind = "invoke", id, cmd, args }));
        foreach (var raw in _transport.Sent)
        {
            var el = JsonDocument.Parse(raw).RootElement;
            if (el.GetProperty("kind").GetString() == "response" && el.GetProperty("id").GetString() == id)
            {
                Assert.True(el.GetProperty("ok").GetBoolean(),
                    el.TryGetProperty("error", out var e) ? e.GetString() : "invoke failed");
                return el.GetProperty("result");
            }
        }
        throw new Xunit.Sdk.XunitException($"no response for {cmd}");
    }

    [Fact]
    public async Task Create_train_and_infer_xor_end_to_end()
    {
        // 1. Create a feed-forward XOR network.
        var net = await Invoke(CommandNames.CreateNetwork, new
        {
            req = new
            {
                name = "XOR",
                kind = "feedforward",
                seed = 42,
                input_dim = 2,
                layers = new object[]
                {
                    new { type = "linear", in_dim = 2, out_dim = 8 },
                    new { type = "activation", activation = "tanh" },
                    new { type = "linear", in_dim = 8, out_dim = 1 },
                    new { type = "activation", activation = "sigmoid" },
                },
            },
        });
        var networkId = net.GetProperty("id").GetString()!;

        // 2. Attach the XOR corpus.
        await Invoke(CommandNames.SetCorpus, new
        {
            req = new
            {
                network_id = networkId,
                feedforward = new
                {
                    features = new[] { 0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f },
                    targets = new[] { 0f, 1f, 1f, 0f },
                    rows = 4, in_dim = 2, out_dim = 1,
                },
            },
        });

        // 3. Train.
        var start = await Invoke(CommandNames.StartTraining, new
        {
            req = new
            {
                network_id = networkId,
                config = new
                {
                    epochs = 1500, batch_size = 4, loss = "mse", seed = 1,
                    optimizer = new { kind = "adam", lr = 0.05f },
                },
            },
        });
        var trainingId = start.GetProperty("training_id").GetString()!;

        // 4. Poll until the run finishes (≤ 30 s).
        JsonElement status = default;
        for (var i = 0; i < 600; i++)
        {
            status = await Invoke(CommandNames.GetTrainingStatus, new { trainingId });
            var s = status.GetProperty("status").GetString();
            if (s is "completed" or "cancelled" or "error") break;
            await Task.Delay(50);
        }

        Assert.Equal("completed", status.GetProperty("status").GetString());
        Assert.True(status.GetProperty("last_loss").GetSingle() < 0.1f,
            $"loss={status.GetProperty("last_loss").GetSingle()}");

        // 5. Inference on the trained weights must solve XOR.
        Assert.True(await Predict(networkId, 1, 0) > 0.5f);
        Assert.True(await Predict(networkId, 0, 1) > 0.5f);
        Assert.True(await Predict(networkId, 0, 0) < 0.5f);
        Assert.True(await Predict(networkId, 1, 1) < 0.5f);

        // 6. The network is now flagged trained.
        var list = await Invoke(CommandNames.ListNetworks, null);
        Assert.True(list[0].GetProperty("trained").GetBoolean());

        // 7. Training history recorded the completed run.
        var history = await Invoke(CommandNames.GetTrainingHistory, new { networkId });
        Assert.Equal("completed", history[0].GetProperty("status").GetString());
    }

    private async Task<float> Predict(string networkId, float a, float b)
    {
        var res = await Invoke(CommandNames.Infer, new
        {
            req = new { network_id = networkId, features = new[] { a, b } },
        });
        return res.GetProperty("output")[0].GetSingle();
    }
}
