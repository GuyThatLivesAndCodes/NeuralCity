namespace NeuralCabin.Engine;

/// <summary>Loss functions (port of <c>loss::Loss</c>).</summary>
public enum Loss
{
    MeanSquaredError,
    CrossEntropy,
}

public static class Losses
{
    public static string Name(this Loss l) => l switch
    {
        Loss.MeanSquaredError => "MeanSquaredError",
        Loss.CrossEntropy => "CrossEntropy",
        _ => throw new ArgumentOutOfRangeException(nameof(l)),
    };

    public static Loss Parse(string name) => name.ToLowerInvariant() switch
    {
        "mse" or "meansquarederror" => Loss.MeanSquaredError,
        "crossentropy" or "cross_entropy" => Loss.CrossEntropy,
        _ => throw new ArgumentException($"unknown loss '{name}'"),
    };

    /// <summary>
    /// Eager scalar loss over a batch.
    /// MSE: mean over all elements of (pred-target)².
    /// CE : mean over rows of −Σ_classes target·log(softmax(pred)).
    /// </summary>
    public static float Eval(this Loss l, Tensor output, Tensor target)
    {
        switch (l)
        {
            case Loss.MeanSquaredError:
            {
                var n = output.Length;
                var s = 0.0f;
                for (var i = 0; i < n; i++)
                {
                    var d = output.Data[i] - target.Data[i];
                    s += d * d;
                }
                return s / n;
            }
            case Loss.CrossEntropy:
            {
                var sm = Activations.SoftmaxRows(output);
                var rows = sm.Rows;
                var s = 0.0f;
                for (var i = 0; i < sm.Length; i++)
                    s -= target.Data[i] * (float)Math.Log(Math.Max(sm.Data[i], 1e-12f));
                return s / rows;
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(l));
        }
    }

    /// <summary>
    /// Gradient of the loss w.r.t. the network's final output.
    /// For CrossEntropy the final output is the raw logits (the last activation,
    /// Softmax/Identity, is the identity during training — the loss folds in the
    /// softmax), giving (softmax(logits) − target)/rows.
    /// For MSE it is 2·(pred − target)/N over all elements.
    /// </summary>
    public static float[] Gradient(this Loss l, Tensor output, Tensor target)
    {
        var grad = new float[output.Length];
        switch (l)
        {
            case Loss.MeanSquaredError:
            {
                var n = output.Length;
                var scale = 2.0f / n;
                for (var i = 0; i < n; i++)
                    grad[i] = scale * (output.Data[i] - target.Data[i]);
                return grad;
            }
            case Loss.CrossEntropy:
            {
                var sm = Activations.SoftmaxRows(output);
                var rows = sm.Rows;
                var inv = 1.0f / rows;
                for (var i = 0; i < sm.Length; i++)
                    grad[i] = (sm.Data[i] - target.Data[i]) * inv;
                return grad;
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(l));
        }
    }
}
