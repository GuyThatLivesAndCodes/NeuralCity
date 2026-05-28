using NeuralCabin.Core.Ipc;
using NeuralCabin.Host.Ipc;

namespace NeuralCabin.Host.Commands;

/// <summary>
/// Registers a clear "not yet migrated" handler for every command that hasn't
/// been ported from the Rust backend. Run this last so already-implemented
/// commands keep their real handlers; the remainder fail loudly instead of
/// being silently absent. Each port simply deletes its entry here by virtue of
/// being registered earlier.
/// </summary>
public static class StubCommands
{
    public static void RegisterUnimplemented(CommandRegistry registry)
    {
        foreach (var command in CommandNames.All)
        {
            if (registry.Contains(command))
                continue;

            var name = command; // capture per-iteration value for the closure
            registry.Register(name, _ => throw new NotYetMigratedException(name));
        }
    }
}
