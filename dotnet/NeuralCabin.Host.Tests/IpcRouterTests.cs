using System.Text.Json;
using NeuralCabin.Core.Ipc;
using NeuralCabin.Host;
using NeuralCabin.Host.Ipc;
using Xunit;

namespace NeuralCabin.Host.Tests;

/// <summary>
/// Exercises the full inbound→handler→response pipeline over the exact JSON the
/// frontend shim sends, plus event emission. This is the end-to-end proof that
/// the .NET bridge is a drop-in for Tauri's invoke/emit.
/// </summary>
public class IpcRouterTests
{
    private static (IpcRouter Router, FakeTransport Transport) NewRouter()
    {
        var backend = new NeuralCabinBackend();
        var transport = new FakeTransport();
        return (backend.CreateRouter(transport), transport);
    }

    private static string InvokeMessage(string id, string cmd, object? args) =>
        JsonSerializer.Serialize(new { kind = "invoke", id, cmd, args });

    private static JsonElement ParseLast(FakeTransport transport) =>
        JsonDocument.Parse(transport.Last).RootElement;

    [Fact]
    public async Task Create_network_round_trips_through_the_bridge()
    {
        var (router, transport) = NewRouter();

        // Byte-for-byte the message the React `api.ts` produces for
        // networks.create({ name, kind, seed, input_dim, layers }).
        var args = new
        {
            req = new
            {
                name = "Bridge",
                kind = "feedforward",
                seed = 1,
                input_dim = 2,
                layers = new object[]
                {
                    new { type = "linear", in_dim = 2, out_dim = 4 },
                    new { type = "activation", activation = "relu" },
                    new { type = "linear", in_dim = 4, out_dim = 1 },
                },
            },
        };

        await router.HandleMessageAsync(InvokeMessage("c1", CommandNames.CreateNetwork, args));

        var response = ParseLast(transport);
        Assert.Equal("response", response.GetProperty("kind").GetString());
        Assert.Equal("c1", response.GetProperty("id").GetString());
        Assert.True(response.GetProperty("ok").GetBoolean());

        var result = response.GetProperty("result");
        Assert.Equal("feedforward", result.GetProperty("kind").GetString());
        Assert.Equal(2, result.GetProperty("input_dim").GetInt32());
        Assert.Equal(1, result.GetProperty("output_dim").GetInt32());
        // (2*4 + 4) + (4*1 + 1) = 12 + 5 = 17.
        Assert.Equal(17, result.GetProperty("parameter_count").GetInt32());
        Assert.False(string.IsNullOrEmpty(result.GetProperty("id").GetString()));
    }

    [Fact]
    public async Task List_networks_returns_an_array_with_no_args()
    {
        var (router, transport) = NewRouter();

        await router.HandleMessageAsync(InvokeMessage("l1", CommandNames.ListNetworks, null));

        var response = ParseLast(transport);
        Assert.True(response.GetProperty("ok").GetBoolean());
        Assert.Equal(JsonValueKind.Array, response.GetProperty("result").ValueKind);
    }

    [Fact]
    public async Task Unknown_command_returns_a_clean_error()
    {
        var (router, transport) = NewRouter();

        await router.HandleMessageAsync(InvokeMessage("u1", "no_such_command", null));

        var response = ParseLast(transport);
        Assert.False(response.GetProperty("ok").GetBoolean());
        Assert.Contains("unknown command", response.GetProperty("error").GetString());
    }

    [Fact]
    public async Task Unmigrated_command_reports_migration_status()
    {
        var (router, transport) = NewRouter();

        await router.HandleMessageAsync(InvokeMessage("t1", CommandNames.StartTraining, new { req = new { } }));

        var response = ParseLast(transport);
        Assert.False(response.GetProperty("ok").GetBoolean());
        Assert.Contains("not yet implemented", response.GetProperty("error").GetString());
    }

    [Fact]
    public async Task Get_network_missing_id_returns_error()
    {
        var (router, transport) = NewRouter();

        await router.HandleMessageAsync(InvokeMessage("g1", CommandNames.GetNetwork, new { id = "nope" }));

        var response = ParseLast(transport);
        Assert.False(response.GetProperty("ok").GetBoolean());
        Assert.Contains("Network not found", response.GetProperty("error").GetString());
    }

    [Fact]
    public async Task Malformed_message_is_ignored_without_replying()
    {
        var (router, transport) = NewRouter();

        await router.HandleMessageAsync("this is not json");
        await router.HandleMessageAsync("{\"kind\":\"event\",\"event\":\"x\"}"); // inbound non-invoke

        Assert.Empty(transport.Sent);
    }

    [Fact]
    public void Emit_writes_an_event_envelope()
    {
        var (router, transport) = NewRouter();

        router.Emit(Events.TrainingUpdate, new { training_id = "r1", epoch = 3, loss = 0.5 });

        var envelope = ParseLast(transport);
        Assert.Equal("event", envelope.GetProperty("kind").GetString());
        Assert.Equal("training_update", envelope.GetProperty("event").GetString());
        Assert.Equal("r1", envelope.GetProperty("payload").GetProperty("training_id").GetString());
        Assert.Equal(3, envelope.GetProperty("payload").GetProperty("epoch").GetInt32());
    }
}
