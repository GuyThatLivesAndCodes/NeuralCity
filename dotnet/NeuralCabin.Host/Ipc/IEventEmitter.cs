namespace NeuralCabin.Host.Ipc;

/// <summary>
/// Pushes an event to the frontend, mirroring Tauri's <c>app.emit(name, payload)</c>.
/// Command handlers receive one of these so long-running work (training,
/// streaming inference) can stream progress back to the React UI.
/// </summary>
public interface IEventEmitter
{
    void Emit(string eventName, object? payload);
}
