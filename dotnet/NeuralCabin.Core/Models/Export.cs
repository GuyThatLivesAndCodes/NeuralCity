namespace NeuralCabin.Core.Models;

/// <summary>Supported model export formats.</summary>
public static class ExportFormats
{
    public const string PyTorch = "pytorch";
    public const string Onnx = "onnx";
    public const string Gguf = "gguf";

    public static readonly IReadOnlyList<string> All = new[] { PyTorch, Onnx, Gguf };
}

/// <summary>
/// A serialized model export (mirror of the <c>ExportPayload</c> command result
/// in <c>lib.rs</c>). The frontend turns <see cref="DataB64"/> into a Blob and
/// triggers a browser-style download — no native file dialog required.
/// </summary>
public sealed record ExportPayload
{
    public string Format { get; init; } = "";
    public string Filename { get; init; } = "";
    public string DataB64 { get; init; } = "";
    public long SizeBytes { get; init; }
}
