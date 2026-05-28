using NeuralCabin.Host.Ipc;
using Photino.NET;

namespace NeuralCabin.App;

/// <summary>
/// Bridges the host's <see cref="IIpcTransport"/> onto a Photino window's
/// <c>SendWebMessage</c>. The window is supplied lazily because the transport
/// is constructed before the window (so the router can be wired into the
/// window's message handler). Sends are serialized with a lock so background
/// event emits don't interleave with invoke responses.
/// </summary>
public sealed class PhotinoTransport : IIpcTransport
{
    private readonly Func<PhotinoWindow?> _window;
    private readonly object _gate = new();

    public PhotinoTransport(Func<PhotinoWindow?> window) => _window = window;

    public void Send(string message)
    {
        var window = _window();
        if (window is null)
            return;

        lock (_gate)
        {
            window.SendWebMessage(message);
        }
    }
}
