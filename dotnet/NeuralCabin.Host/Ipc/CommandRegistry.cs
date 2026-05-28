namespace NeuralCabin.Host.Ipc;

/// <summary>An async command implementation. Returns the value to serialize as
/// the invoke result, or throws <see cref="CommandException"/> on failure.</summary>
public delegate Task<object?> CommandHandler(CommandContext context);

/// <summary>
/// Maps command names to their handlers — the .NET analogue of Tauri's
/// <c>generate_handler!</c> table.
/// </summary>
public sealed class CommandRegistry
{
    private readonly Dictionary<string, CommandHandler> _handlers = new(StringComparer.Ordinal);

    public void Register(string name, CommandHandler handler)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("command name must be non-empty", nameof(name));
        if (!_handlers.TryAdd(name, handler))
            throw new InvalidOperationException($"command '{name}' is already registered");
    }

    public bool Contains(string name) => _handlers.ContainsKey(name);

    public bool TryGet(string name, out CommandHandler handler) => _handlers.TryGetValue(name, out handler!);

    public IReadOnlyCollection<string> Names => _handlers.Keys;
}
