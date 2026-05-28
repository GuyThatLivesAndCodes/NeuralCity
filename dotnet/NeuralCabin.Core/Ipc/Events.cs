namespace NeuralCabin.Core.Ipc;

/// <summary>
/// Event channel names pushed from the backend to the frontend. The React UI
/// subscribes via <c>listen(name, handler)</c>; these mirror the Tauri
/// <c>app.emit(name, payload)</c> calls in <c>src-tauri/src/lib.rs</c>.
/// </summary>
public static class Events
{
    public const string TrainingUpdate = "training_update";
    public const string TrainingFinished = "training_finished";
    public const string TrainingError = "training_error";

    public const string InferenceToken = "inference_token";
    public const string InferenceFinished = "inference_finished";
    public const string InferenceError = "inference_error";
}
