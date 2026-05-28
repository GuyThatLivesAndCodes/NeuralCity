using System.Text.Json;
using NeuralCabin.Core.Json;

namespace NeuralCabin.Host.Ipc;

/// <summary>
/// Everything a command handler needs: the invoke arguments, an event emitter
/// for streaming results, and a cancellation token.
///
/// Argument keys are read verbatim as the frontend sends them. Note these are
/// a deliberate mix — <c>req</c>, <c>id</c>, <c>networkId</c>, <c>trainingId</c>,
/// <c>format</c>, etc. — matching the keys the React <c>api.ts</c> passes to
/// <c>invoke</c> (which in turn match Tauri's camelCased parameter names). The
/// snake_case naming policy only applies when deserializing the typed request
/// bodies, not when looking up these top-level keys.
/// </summary>
public sealed class CommandContext
{
    private readonly JsonElement? _args;

    public IEventEmitter Events { get; }
    public CancellationToken Cancellation { get; }

    public CommandContext(JsonElement? args, IEventEmitter events, CancellationToken cancellation)
    {
        _args = args;
        Events = events;
        Cancellation = cancellation;
    }

    public bool TryGetArg(string name, out JsonElement value)
    {
        if (_args is { ValueKind: JsonValueKind.Object } obj && obj.TryGetProperty(name, out var found))
        {
            value = found;
            return true;
        }
        value = default;
        return false;
    }

    /// <summary>Deserialize a named argument into a typed request body.</summary>
    public T Arg<T>(string name)
    {
        if (!TryGetArg(name, out var value))
            throw new CommandException($"missing required argument '{name}'");

        var result = value.Deserialize<T>(NeuralCabinJson.Options);
        if (result is null)
            throw new CommandException($"argument '{name}' could not be parsed");
        return result;
    }

    /// <summary>Read a required string argument (e.g. an id).</summary>
    public string ArgString(string name)
    {
        if (!TryGetArg(name, out var value) || value.ValueKind != JsonValueKind.String)
            throw new CommandException($"missing required string argument '{name}'");
        return value.GetString()!;
    }
}
