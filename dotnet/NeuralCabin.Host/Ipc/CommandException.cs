namespace NeuralCabin.Host.Ipc;

/// <summary>
/// A handled command failure. The router turns the message into the
/// <c>error</c> field of the invoke response — exactly how the Tauri commands
/// returned <c>Result&lt;T, String&gt;</c> error strings to the frontend.
/// </summary>
public class CommandException : Exception
{
    public CommandException(string message) : base(message) { }
}

/// <summary>
/// Thrown by commands that have not yet been ported from the Rust backend.
/// Surfaces a clear, actionable message instead of a silent failure while the
/// Tauri → .NET migration is in progress.
/// </summary>
public sealed class NotYetMigratedException : CommandException
{
    public NotYetMigratedException(string command)
        : base($"Command '{command}' is not yet implemented in the .NET backend. " +
               "The Tauri → .NET migration is in progress — see MIGRATION.md for status.")
    { }
}
