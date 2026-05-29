using NeuralCabin.Engine;
using Xunit;

namespace NeuralCabin.Engine.Tests;

/// <summary>
/// Validates analytic backprop against numerical (finite-difference) gradients.
/// Trick: one SGD step with lr=1 and momentum=0 sets w_after = w_before − grad,
/// so the gradient the trainer computed is exactly (w_before − w_after). We
/// compare that to a central finite-difference of the eager loss.
/// </summary>
public class BackpropGradientCheckTests
{
    private static (Tensor x, Tensor y) MseBatch()
    {
        var x = new Tensor(new[] { 2, 3 }, new[] { 0.5f, -0.2f, 0.1f, -0.3f, 0.8f, 0.4f });
        var y = new Tensor(new[] { 2, 2 }, new[] { 0.3f, -0.1f, 0.2f, 0.6f });
        return (x, y);
    }

    private static (Tensor x, Tensor y) CeBatch()
    {
        var x = new Tensor(new[] { 2, 3 }, new[] { 0.5f, -0.2f, 0.1f, -0.3f, 0.8f, 0.4f });
        // one-hot targets
        var y = new Tensor(new[] { 2, 2 }, new[] { 1f, 0f, 0f, 1f });
        return (x, y);
    }

    private static void CheckGradients(Loss loss, IReadOnlyList<LayerSpec> specs, Tensor x, Tensor y)
    {
        var model0 = Model.FromSpecs(3, specs, 123);
        var working = model0.Clone();
        var trainer = new MlpTrainer(working, OptimizerSpec.FromConfig("sgd", 1.0f), loss);
        trainer.Step(x, y);

        const float eps = 1e-2f;
        for (var li = 0; li < model0.Layers.Count; li++)
        {
            if (model0.Layers[li] is not LinearLayer baseLin) continue;
            var workLin = (LinearLayer)working.Layers[li];

            CheckParam(loss, model0, x, y, li, isWeight: true, baseLin.W, workLin.W, eps);
            CheckParam(loss, model0, x, y, li, isWeight: false, baseLin.B, workLin.B, eps);
        }
    }

    private static void CheckParam(
        Loss loss, Model model0, Tensor x, Tensor y, int layerIndex, bool isWeight,
        Tensor baseParam, Tensor afterParam, float eps)
    {
        for (var e = 0; e < baseParam.Length; e++)
        {
            var analytic = baseParam.Data[e] - afterParam.Data[e]; // = grad (lr=1)

            var plus = model0.Clone();
            Param(plus, layerIndex, isWeight).Data[e] += eps;
            var lp = plus.EvaluateLoss(loss, x, y);

            var minus = model0.Clone();
            Param(minus, layerIndex, isWeight).Data[e] -= eps;
            var lm = minus.EvaluateLoss(loss, x, y);

            var numeric = (lp - lm) / (2f * eps);
            var tol = 1e-2f + 0.05f * Math.Abs(numeric);
            Assert.True(Math.Abs(analytic - numeric) <= tol,
                $"layer {layerIndex} {(isWeight ? "W" : "b")}[{e}]: analytic={analytic} numeric={numeric}");
        }
    }

    private static Tensor Param(Model m, int layerIndex, bool isWeight)
    {
        var lin = (LinearLayer)m.Layers[layerIndex];
        return isWeight ? lin.W : lin.B;
    }

    [Fact]
    public void Mse_gradients_match_finite_difference()
    {
        var (x, y) = MseBatch();
        CheckGradients(Loss.MeanSquaredError, new LayerSpec[]
        {
            new LinearSpec(3, 4),
            new ActivationSpec(Activation.Tanh),
            new LinearSpec(4, 2),
            new ActivationSpec(Activation.Sigmoid),
        }, x, y);
    }

    [Fact]
    public void CrossEntropy_gradients_match_finite_difference()
    {
        var (x, y) = CeBatch();
        // CE expects raw logits as the final output (no terminal softmax).
        CheckGradients(Loss.CrossEntropy, new LayerSpec[]
        {
            new LinearSpec(3, 5),
            new ActivationSpec(Activation.ReLU),
            new LinearSpec(5, 2),
        }, x, y);
    }
}
