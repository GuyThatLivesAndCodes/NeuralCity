using NeuralCabin.Host.Ipc;

namespace NeuralCabin.Host.Tests;

/// <summary>An <see cref="IIpcTransport"/> that records every outbound message.</summary>
public sealed class FakeTransport : IIpcTransport
{
    public List<string> Sent { get; } = new();

    public void Send(string message) => Sent.Add(message);

    public string Last => Sent[^1];
}
