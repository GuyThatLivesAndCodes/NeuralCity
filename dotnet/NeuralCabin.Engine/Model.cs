namespace NeuralCabin.Engine;

/// <summary>Declarative layer description for model construction (port of <c>nn::LayerSpec</c>).</summary>
public abstract record LayerSpec;

public sealed record LinearSpec(int InDim, int OutDim) : LayerSpec;

public sealed record ActivationSpec(Activation Activation) : LayerSpec;

/// <summary>A materialized layer with weights (port of <c>nn::Layer</c>).</summary>
public abstract class Layer
{
    public abstract int OutputDim(int inputDim);
    public abstract long ParameterCount { get; }
    public abstract Tensor ForwardEager(Tensor x);
    public abstract Layer Clone();
}

public sealed class LinearLayer : Layer
{
    public int InDim { get; }
    public int OutDim { get; }
    public Tensor W { get; set; } // (in_dim, out_dim)
    public Tensor B { get; set; } // (1, out_dim)

    public LinearLayer(int inDim, int outDim, Tensor w, Tensor b)
    {
        InDim = inDim;
        OutDim = outDim;
        W = w;
        B = b;
    }

    public static LinearLayer New(int inDim, int outDim, SplitMix64 rng) =>
        new(inDim, outDim, Tensor.Xavier(inDim, outDim, rng), Tensor.Zeros(1, outDim));

    public override int OutputDim(int inputDim) => OutDim;
    public override long ParameterCount => W.Length + B.Length;

    public override Tensor ForwardEager(Tensor x) => x.Matmul(W).AddBiasRows(B);

    public override Layer Clone() => new LinearLayer(InDim, OutDim, W.Clone(), B.Clone());
}

public sealed class ActivationLayer : Layer
{
    public Activation Activation { get; }

    public ActivationLayer(Activation activation) => Activation = activation;

    public override int OutputDim(int inputDim) => inputDim;
    public override long ParameterCount => 0;
    public override Tensor ForwardEager(Tensor x) => Activation.ApplyEager(x);
    public override Layer Clone() => new ActivationLayer(Activation);
}

/// <summary>A feed-forward model (port of <c>nn::Model</c>).</summary>
public sealed class Model
{
    public int InputDim { get; set; }
    public List<Layer> Layers { get; set; } = new();
    public ulong Seed { get; set; } = 0xC0FFEE;

    public Model() { }

    public Model(int inputDim, List<Layer> layers, ulong seed)
    {
        InputDim = inputDim;
        Layers = layers;
        Seed = seed;
    }

    public static Model FromSpecs(int inputDim, IReadOnlyList<LayerSpec> specs, ulong seed)
    {
        var rng = new SplitMix64(seed);
        var layers = new List<Layer>(specs.Count);
        var cur = inputDim;
        foreach (var spec in specs)
        {
            switch (spec)
            {
                case LinearSpec ls:
                    if (ls.InDim != cur)
                        throw new ArgumentException($"layer in_dim {ls.InDim} doesn't match running dim {cur}");
                    layers.Add(LinearLayer.New(ls.InDim, ls.OutDim, rng));
                    cur = ls.OutDim;
                    break;
                case ActivationSpec acts:
                    layers.Add(new ActivationLayer(acts.Activation));
                    break;
                default:
                    throw new ArgumentException("unknown layer spec");
            }
        }
        return new Model(inputDim, layers, seed);
    }

    public int ComputeOutputDim()
    {
        var d = InputDim;
        foreach (var l in Layers) d = l.OutputDim(d);
        return d;
    }

    public long ParameterCount => Layers.Sum(l => l.ParameterCount);

    public List<int[]> ParameterShapes()
    {
        var shapes = new List<int[]>();
        foreach (var l in Layers)
            if (l is LinearLayer ll)
            {
                shapes.Add((int[])ll.W.Shape.Clone());
                shapes.Add((int[])ll.B.Shape.Clone());
            }
        return shapes;
    }

    /// <summary>Eager CPU forward — used for inference and metric evaluation.</summary>
    public Tensor Predict(Tensor input)
    {
        var x = input;
        foreach (var l in Layers) x = l.ForwardEager(x);
        return x;
    }

    /// <summary>Forward returning the activation after every layer (input first).</summary>
    public List<Tensor> PredictWithActivations(Tensor input)
    {
        var acts = new List<Tensor>(Layers.Count + 1) { input };
        var x = input;
        foreach (var l in Layers)
        {
            x = l.ForwardEager(x);
            acts.Add(x);
        }
        return acts;
    }

    public float EvaluateLoss(Loss loss, Tensor input, Tensor target) => loss.Eval(Predict(input), target);

    public Model Clone() => new(InputDim, Layers.Select(l => l.Clone()).ToList(), Seed);
}
