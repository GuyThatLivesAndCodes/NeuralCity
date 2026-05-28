using NeuralCabin.Host.Commands;
using NeuralCabin.Host.Ipc;
using NeuralCabin.Host.Services;

namespace NeuralCabin.Host;

/// <summary>
/// Composition root for the backend. Wires the domain services into a
/// <see cref="CommandRegistry"/> and exposes it for the Photino shell (and for
/// tests) to drive through an <see cref="IpcRouter"/>.
///
/// As each domain is ported from Rust, register its real handlers here before
/// the <see cref="StubCommands"/> fallback so the stub is replaced.
/// </summary>
public sealed class NeuralCabinBackend
{
    public CommandRegistry Registry { get; }

    // Domain services (grown as the migration proceeds).
    public NetworkService Networks { get; }

    public NeuralCabinBackend()
    {
        Registry = new CommandRegistry();
        Networks = new NetworkService();

        // ── Implemented domains ──────────────────────────────────────────────
        NetworkCommands.Register(Registry, Networks);

        // ── Everything else: explicit "not yet migrated" until ported ────────
        StubCommands.RegisterUnimplemented(Registry);
    }

    /// <summary>Build a router that dispatches this backend over the given transport.</summary>
    public IpcRouter CreateRouter(IIpcTransport transport) => new(transport, Registry);
}
