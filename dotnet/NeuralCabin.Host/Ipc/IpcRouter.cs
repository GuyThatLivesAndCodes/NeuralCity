using System.Text.Json;
using System.Text.Json.Nodes;
using NeuralCabin.Core.Json;

namespace NeuralCabin.Host.Ipc;

/// <summary>
/// The IPC dispatcher. Parses inbound <c>invoke</c> messages, routes them to a
/// registered <see cref="CommandHandler"/>, and writes back a structured
/// response. Also implements <see cref="IEventEmitter"/> so handlers can push
/// events. This replaces Tauri's invoke/emit bridge.
/// </summary>
public sealed class IpcRouter : IEventEmitter
{
    private readonly IIpcTransport _transport;
    private readonly CommandRegistry _registry;

    public IpcRouter(IIpcTransport transport, CommandRegistry registry)
    {
        _transport = transport;
        _registry = registry;
    }

    /// <summary>Handle one raw message string received from the webview.</summary>
    public async Task HandleMessageAsync(string raw, CancellationToken cancellation = default)
    {
        InvokeRequest? request;
        try
        {
            request = JsonSerializer.Deserialize<InvokeRequest>(raw, NeuralCabinJson.Options);
        }
        catch (JsonException)
        {
            // Not a message we understand; nothing we can usefully reply to.
            return;
        }

        if (request is null || request.Kind != "invoke" || request.Id.Length == 0)
            return;

        await DispatchAsync(request, cancellation).ConfigureAwait(false);
    }

    private async Task DispatchAsync(InvokeRequest request, CancellationToken cancellation)
    {
        try
        {
            if (!_registry.TryGet(request.Cmd, out var handler))
                throw new CommandException($"unknown command '{request.Cmd}'");

            var context = new CommandContext(request.Args, this, cancellation);
            var result = await handler(context).ConfigureAwait(false);
            SendResponse(request.Id, ok: true, result: result, error: null);
        }
        catch (CommandException ex)
        {
            SendResponse(request.Id, ok: false, result: null, error: ex.Message);
        }
        catch (Exception ex)
        {
            // Unexpected fault: still report a clean error string to the UI
            // rather than leaving the invoke promise pending forever.
            SendResponse(request.Id, ok: false, result: null, error: ex.Message);
        }
    }

    private void SendResponse(string id, bool ok, object? result, string? error)
    {
        var node = result is null ? null : JsonSerializer.SerializeToNode(result, NeuralCabinJson.Options);
        var response = new InvokeResponse { Id = id, Ok = ok, Result = node, Error = error };
        _transport.Send(JsonSerializer.Serialize(response, NeuralCabinJson.Options));
    }

    /// <inheritdoc />
    public void Emit(string eventName, object? payload)
    {
        var node = payload is null ? null : JsonSerializer.SerializeToNode(payload, NeuralCabinJson.Options);
        var envelope = new EventEnvelope { Event = eventName, Payload = node };
        _transport.Send(JsonSerializer.Serialize(envelope, NeuralCabinJson.Options));
    }
}
