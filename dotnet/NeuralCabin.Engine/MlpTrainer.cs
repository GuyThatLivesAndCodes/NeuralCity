namespace NeuralCabin.Engine;

/// <summary>
/// A long-lived feed-forward training session (port of <c>nn::MlpTrainerSession</c>).
/// Holds the model's live weights and per-parameter optimizer state across steps
/// (so Adam/AdamW keep their moment buffers, the fix the Rust session shipped).
/// Backpropagation is analytic — the closed-form gradients for Linear +
/// Identity/ReLU/Sigmoid/Tanh/Softmax layers under MSE / softmax-cross-entropy —
/// replacing Burn's autodiff.
///
/// The trainer mutates <see cref="Model"/>'s weight tensors in place, so the
/// model always reflects the latest step (no separate write-back needed).
/// </summary>
public sealed class MlpTrainer
{
    private readonly Model _model;
    private readonly Loss _loss;
    private readonly OptimizerSpec _opt;

    // Linear layers in order of appearance, with matching optimizer state.
    private readonly List<int> _linearLayerPos = new();      // layer index per linear order
    private readonly Dictionary<int, int> _posToLinearOrder = new();
    private readonly List<ParamOptimizer> _wOpt = new();
    private readonly List<ParamOptimizer> _bOpt = new();
    private readonly HashSet<int> _frozen;

    private int _t;

    public MlpTrainer(Model model, OptimizerSpec opt, Loss loss, IEnumerable<int>? frozenLinearIndices = null)
    {
        _model = model;
        _opt = opt;
        _loss = loss;
        _frozen = frozenLinearIndices is null ? new HashSet<int>() : new HashSet<int>(frozenLinearIndices);

        for (var pos = 0; pos < model.Layers.Count; pos++)
        {
            if (model.Layers[pos] is LinearLayer ll)
            {
                _posToLinearOrder[pos] = _linearLayerPos.Count;
                _linearLayerPos.Add(pos);
                _wOpt.Add(new ParamOptimizer(ll.W.Length));
                _bOpt.Add(new ParamOptimizer(ll.B.Length));
            }
        }
    }

    /// <summary>One optimizer step on a (input, target) batch. Returns the scalar loss.</summary>
    public float Step(Tensor input, Tensor target)
    {
        var layers = _model.Layers;
        var n = layers.Count;
        var linearInput = new Tensor?[n];
        var actInput = new Tensor?[n];
        var actOutput = new Tensor?[n];

        // ── Forward (cache what backward needs) ──────────────────────────────
        var cur = input;
        for (var pos = 0; pos < n; pos++)
        {
            switch (layers[pos])
            {
                case LinearLayer ll:
                    linearInput[pos] = cur;
                    cur = ll.ForwardEager(cur);
                    break;
                case ActivationLayer al:
                    actInput[pos] = cur;
                    cur = TrainForward(al.Activation, cur);
                    actOutput[pos] = cur;
                    break;
            }
        }

        var lossScalar = _loss.Eval(cur, target);

        // ── Backward ─────────────────────────────────────────────────────────
        _t += 1;
        var grad = new Tensor((int[])cur.Shape.Clone(), _loss.Gradient(cur, target));
        for (var pos = n - 1; pos >= 0; pos--)
        {
            switch (layers[pos])
            {
                case ActivationLayer al:
                    grad = ActBackward(al.Activation, grad, actInput[pos]!, actOutput[pos]!);
                    break;
                case LinearLayer ll:
                {
                    var aPrev = linearInput[pos]!;
                    var gradW = aPrev.Transpose().Matmul(grad); // (in, out)
                    var gradB = grad.SumRows();                  // (1, out)
                    var gradPrev = grad.Matmul(ll.W.Transpose()); // (batch, in) — flows even if frozen
                    var order = _posToLinearOrder[pos];
                    if (!_frozen.Contains(order))
                    {
                        _wOpt[order].Update(ll.W.Data, gradW.Data, _opt, _t);
                        _bOpt[order].Update(ll.B.Data, gradB.Data, _opt, _t);
                    }
                    grad = gradPrev;
                    break;
                }
            }
        }

        return lossScalar;
    }

    /// <summary>Training-time activation forward: Softmax is the identity (folded into CE loss).</summary>
    private static Tensor TrainForward(Activation a, Tensor x) => a switch
    {
        Activation.Identity or Activation.Softmax => x,
        Activation.ReLU => MapNew(x, v => v > 0.0f ? v : 0.0f),
        Activation.Sigmoid => MapNew(x, Activations.Sigmoid),
        Activation.Tanh => MapNew(x, v => (float)Math.Tanh(v)),
        _ => throw new ArgumentOutOfRangeException(nameof(a)),
    };

    private static Tensor ActBackward(Activation a, Tensor grad, Tensor preAct, Tensor postAct)
    {
        switch (a)
        {
            case Activation.Identity:
            case Activation.Softmax:
                return grad;
            case Activation.ReLU:
            {
                var outData = new float[grad.Length];
                for (var i = 0; i < grad.Length; i++)
                    outData[i] = preAct.Data[i] > 0.0f ? grad.Data[i] : 0.0f;
                return new Tensor((int[])grad.Shape.Clone(), outData);
            }
            case Activation.Sigmoid:
            {
                var outData = new float[grad.Length];
                for (var i = 0; i < grad.Length; i++)
                {
                    var s = postAct.Data[i];
                    outData[i] = grad.Data[i] * s * (1.0f - s);
                }
                return new Tensor((int[])grad.Shape.Clone(), outData);
            }
            case Activation.Tanh:
            {
                var outData = new float[grad.Length];
                for (var i = 0; i < grad.Length; i++)
                {
                    var th = postAct.Data[i];
                    outData[i] = grad.Data[i] * (1.0f - th * th);
                }
                return new Tensor((int[])grad.Shape.Clone(), outData);
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(a));
        }
    }

    private static Tensor MapNew(Tensor x, Func<float, float> f)
    {
        var outData = new float[x.Length];
        for (var i = 0; i < x.Length; i++) outData[i] = f(x.Data[i]);
        return new Tensor((int[])x.Shape.Clone(), outData);
    }
}
