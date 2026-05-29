using NeuralCabin.Host.Ipc;

namespace NeuralCabin.Host.Tests;

/// <summary>
/// An <see cref="IIpcTransport"/> that records every outbound message.
/// Thread-safe: background training tasks emit events concurrently with the
/// test thread's invoke responses.
/// </summary>
public sealed class FakeTransport : IIpcTransport
{
    private readonly object _gate = new();
    private readonly List<string> _sent = new();

    public void Send(string message)
    {
        lock (_gate) _sent.Add(message);
    }

    public IReadOnlyList<string> Sent
    {
        get { lock (_gate) return _sent.ToList(); }
    }

    public string Last
    {
        get { lock (_gate) return _sent[^1]; }
    }
}
