namespace NeuralCabin.Engine;

public enum OptimizerType
{
    Sgd,
    Adam,
    AdamW,
    /// <summary>Layer-wise Adaptive Moments — routed to AdamW, matching the Rust engine.</summary>
    Lamb,
}

/// <summary>
/// Optimizer hyper-parameters (port of <c>optimizer::OptimizerKind</c>). State
/// buffers (momentum / variance) live in <see cref="MlpTrainer"/> for the
/// duration of a run, not here.
/// </summary>
public sealed record OptimizerSpec
{
    public OptimizerType Type { get; init; }
    public float Lr { get; init; }
    public float Beta1 { get; init; } = 0.9f;
    public float Beta2 { get; init; } = 0.999f;
    public float Eps { get; init; } = 1e-8f;
    public float Momentum { get; init; }
    public float WeightDecay { get; init; }

    public string Name() => Type switch
    {
        OptimizerType.Sgd => "SGD",
        OptimizerType.Adam => "Adam",
        OptimizerType.AdamW => "AdamW",
        OptimizerType.Lamb => "LAMB",
        _ => throw new ArgumentOutOfRangeException(),
    };

    /// <summary>Build from the IPC optimizer config (kind string + optional fields).</summary>
    public static OptimizerSpec FromConfig(
        string kind, float lr,
        float? beta1 = null, float? beta2 = null, float? eps = null,
        float? momentum = null, float? weightDecay = null)
    {
        var type = kind.ToLowerInvariant() switch
        {
            "sgd" => OptimizerType.Sgd,
            "adam" => OptimizerType.Adam,
            "adamw" => OptimizerType.AdamW,
            "lamb" => OptimizerType.Lamb,
            _ => throw new ArgumentException($"unknown optimizer '{kind}' (expected adam|adamw|lamb|sgd)"),
        };
        return new OptimizerSpec
        {
            Type = type,
            Lr = lr,
            Beta1 = beta1 ?? 0.9f,
            Beta2 = beta2 ?? 0.999f,
            Eps = eps ?? 1e-8f,
            Momentum = momentum ?? 0.0f,
            WeightDecay = weightDecay ?? 0.0f,
        };
    }
}

/// <summary>Per-parameter optimizer state + update step. Internal to a training run.</summary>
internal sealed class ParamOptimizer
{
    private readonly float[] _m;
    private readonly float[] _v;
    private readonly float[] _vel;

    public ParamOptimizer(int size)
    {
        _m = new float[size];
        _v = new float[size];
        _vel = new float[size];
    }

    /// <summary>Update <paramref name="p"/> in place from gradient <paramref name="g"/>.</summary>
    public void Update(float[] p, float[] g, OptimizerSpec s, int t)
    {
        switch (s.Type)
        {
            case OptimizerType.Sgd:
                if (s.Momentum > 0.0f)
                    for (var i = 0; i < p.Length; i++)
                    {
                        _vel[i] = s.Momentum * _vel[i] + g[i];
                        p[i] -= s.Lr * _vel[i];
                    }
                else
                    for (var i = 0; i < p.Length; i++)
                        p[i] -= s.Lr * g[i];
                break;

            case OptimizerType.Adam:
            case OptimizerType.AdamW:
            case OptimizerType.Lamb: // AdamW drop-in
            {
                var b1 = s.Beta1;
                var b2 = s.Beta2;
                var bc1 = 1.0f - (float)Math.Pow(b1, t);
                var bc2 = 1.0f - (float)Math.Pow(b2, t);
                var decoupledWd = s.Type != OptimizerType.Adam ? s.WeightDecay : 0.0f;
                for (var i = 0; i < p.Length; i++)
                {
                    var grad = g[i];
                    _m[i] = b1 * _m[i] + (1.0f - b1) * grad;
                    _v[i] = b2 * _v[i] + (1.0f - b2) * grad * grad;
                    var mhat = _m[i] / bc1;
                    var vhat = _v[i] / bc2;
                    var step = mhat / ((float)Math.Sqrt(vhat) + s.Eps);
                    if (decoupledWd > 0.0f) step += decoupledWd * p[i];
                    p[i] -= s.Lr * step;
                }
                break;
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(s));
        }
    }
}
