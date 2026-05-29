namespace NeuralCabin.Engine;

/// <summary>Activation functions (port of <c>activations::Activation</c>).</summary>
public enum Activation
{
    Identity,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
}

public static class Activations
{
    /// <summary>serde serializes these enum variants by name — note "ReLU" casing.</summary>
    public static string Name(this Activation a) => a switch
    {
        Activation.Identity => "Identity",
        Activation.ReLU => "ReLU",
        Activation.Sigmoid => "Sigmoid",
        Activation.Tanh => "Tanh",
        Activation.Softmax => "Softmax",
        _ => throw new ArgumentOutOfRangeException(nameof(a)),
    };

    public static Activation Parse(string name) => name.ToLowerInvariant() switch
    {
        "identity" => Activation.Identity,
        "relu" => Activation.ReLU,
        "sigmoid" => Activation.Sigmoid,
        "tanh" => Activation.Tanh,
        "softmax" => Activation.Softmax,
        _ => throw new ArgumentException($"unknown activation '{name.ToLowerInvariant()}'"),
    };

    /// <summary>Eager forward (inference): full row-wise softmax for Softmax.</summary>
    public static Tensor ApplyEager(this Activation a, Tensor x) => a switch
    {
        Activation.Identity => x.Clone(),
        Activation.ReLU => Map(x, v => v > 0.0f ? v : 0.0f),
        Activation.Sigmoid => Map(x, Sigmoid),
        Activation.Tanh => Map(x, v => (float)Math.Tanh(v)),
        Activation.Softmax => SoftmaxRows(x),
        _ => throw new ArgumentOutOfRangeException(nameof(a)),
    };

    private static Tensor Map(Tensor x, Func<float, float> f)
    {
        var outData = new float[x.Length];
        for (var i = 0; i < x.Length; i++) outData[i] = f(x.Data[i]);
        return new Tensor((int[])x.Shape.Clone(), outData);
    }

    public static float Sigmoid(float v) => 1.0f / (1.0f + (float)Math.Exp(-v));

    /// <summary>Numerically-stable row-wise softmax for a 2-D tensor.</summary>
    public static Tensor SoftmaxRows(Tensor x)
    {
        if (x.Shape.Length != 2) throw new ArgumentException("softmax_rows expects (batch, classes)");
        int rows = x.Rows, cols = x.Cols;
        var outData = new float[rows * cols];
        for (var i = 0; i < rows; i++)
        {
            var baseIdx = i * cols;
            var m = float.NegativeInfinity;
            for (var j = 0; j < cols; j++) if (x.Data[baseIdx + j] > m) m = x.Data[baseIdx + j];
            var sum = 0.0f;
            for (var j = 0; j < cols; j++)
            {
                var e = (float)Math.Exp(x.Data[baseIdx + j] - m);
                outData[baseIdx + j] = e;
                sum += e;
            }
            for (var j = 0; j < cols; j++) outData[baseIdx + j] /= sum;
        }
        return new Tensor(new[] { rows, cols }, outData);
    }
}
