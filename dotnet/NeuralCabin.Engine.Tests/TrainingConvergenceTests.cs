using NeuralCabin.Engine;
using Xunit;

namespace NeuralCabin.Engine.Tests;

public class TrainingConvergenceTests
{
    private static (Tensor x, Tensor y) Xor()
    {
        var x = new Tensor(new[] { 4, 2 }, new[] { 0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f });
        var y = new Tensor(new[] { 4, 1 }, new[] { 0f, 1f, 1f, 0f });
        return (x, y);
    }

    /// <summary>Port of <c>nn::tests::session_mlp_learns_xor</c>: Adam, persistent state.</summary>
    [Fact]
    public void Mlp_learns_xor_with_adam()
    {
        var model = Model.FromSpecs(2, new LayerSpec[]
        {
            new LinearSpec(2, 8),
            new ActivationSpec(Activation.Tanh),
            new LinearSpec(8, 1),
            new ActivationSpec(Activation.Sigmoid),
        }, 42);

        var opt = OptimizerSpec.FromConfig("adam", 0.05f);
        var trainer = new MlpTrainer(model, opt, Loss.MeanSquaredError);
        var (x, y) = Xor();

        var last = float.PositiveInfinity;
        for (var i = 0; i < 1500; i++) last = trainer.Step(x, y);

        Assert.True(last < 0.05f, $"XOR did not converge: loss={last}");
        // Eager inference path must reproduce the low loss too.
        Assert.True(model.EvaluateLoss(Loss.MeanSquaredError, x, y) < 0.05f);
    }

    /// <summary>Port of <c>nn::tests::session_frozen_linear_stays_put</c>.</summary>
    [Fact]
    public void Frozen_linear_layer_stays_put_while_others_move()
    {
        var specs = new LayerSpec[]
        {
            new LinearSpec(2, 4),
            new ActivationSpec(Activation.Tanh),
            new LinearSpec(4, 1),
        };
        var model = Model.FromSpecs(2, specs, 7);
        var frozenW0 = ((LinearLayer)model.Layers[0]).W.Clone();
        var initialOutW = ((LinearLayer)Model.FromSpecs(2, specs, 7).Layers[2]).W.Clone();

        var opt = OptimizerSpec.FromConfig("sgd", 0.5f);
        var trainer = new MlpTrainer(model, opt, Loss.MeanSquaredError, frozenLinearIndices: new[] { 0 });
        var x = new Tensor(new[] { 2, 2 }, new[] { 0f, 1f, 1f, 0f });
        var y = new Tensor(new[] { 2, 1 }, new[] { 1f, 0f });
        for (var i = 0; i < 10; i++) trainer.Step(x, y);

        var afterW0 = ((LinearLayer)model.Layers[0]).W;
        Assert.Equal(frozenW0.Data, afterW0.Data); // frozen layer unchanged
        var outW = ((LinearLayer)model.Layers[2]).W;
        Assert.NotEqual(initialOutW.Data, outW.Data); // trainable layer moved
    }

    /// <summary>Adam update math: minimize f(x) = (x − 3)² toward x ≈ 3.</summary>
    [Fact]
    public void Adam_descends_on_quadratic()
    {
        var p = new[] { 0.0f };
        var optState = new ParamOptimizer(1);
        var spec = OptimizerSpec.FromConfig("adam", 0.1f);
        for (var t = 1; t <= 600; t++)
        {
            var g = new[] { 2.0f * (p[0] - 3.0f) }; // df/dx
            optState.Update(p, g, spec, t);
        }
        Assert.True(Math.Abs(p[0] - 3.0f) < 5e-2f, $"x = {p[0]}");
    }

    /// <summary>SGD update math on the same quadratic.</summary>
    [Fact]
    public void Sgd_descends_on_quadratic()
    {
        var p = new[] { 0.0f };
        var optState = new ParamOptimizer(1);
        var spec = OptimizerSpec.FromConfig("sgd", 0.1f);
        for (var t = 1; t <= 400; t++)
        {
            var g = new[] { 2.0f * (p[0] - 3.0f) };
            optState.Update(p, g, spec, t);
        }
        Assert.True(Math.Abs(p[0] - 3.0f) < 1e-2f, $"x = {p[0]}");
    }
}
