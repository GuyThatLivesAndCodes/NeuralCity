namespace NeuralCabin.Host.Ipc;

/// <summary>
/// Low-level string pipe to the webview. The Photino shell implements this by
/// forwarding to <c>PhotinoWindow.SendWebMessage</c>; tests implement it with a
/// simple in-memory buffer. Implementations must be safe to call concurrently.
/// </summary>
public interface IIpcTransport
{
    void Send(string message);
}
