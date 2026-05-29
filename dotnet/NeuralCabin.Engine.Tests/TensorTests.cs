using NeuralCabin.Engine;
using Xunit;

namespace NeuralCabin.Engine.Tests;

public class TensorTests
{
    [Fact]
    public void Matmul_basic()
    {
        var a = new Tensor(new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f });
        var b = new Tensor(new[] { 3, 2 }, new[] { 7f, 8f, 9f, 10f, 11f, 12f });
        var c = a.Matmul(b);
        Assert.Equal(new[] { 2, 2 }, c.Shape);
        Assert.Equal(new[] { 58f, 64f, 139f, 154f }, c.Data); // [[58,64],[139,154]]
    }

    [Fact]
    public void Transpose_roundtrip()
    {
        var a = new Tensor(new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f });
        var t = a.Transpose();
        Assert.Equal(new[] { 3, 2 }, t.Shape);
        Assert.Equal(a.Data, t.Transpose().Data);
    }

    [Fact]
    public void Bias_broadcast_and_sum_rows()
    {
        var x = new Tensor(new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f });
        var bias = new Tensor(new[] { 1, 3 }, new[] { 10f, 20f, 30f });
        var z = x.AddBiasRows(bias);
        Assert.Equal(new[] { 11f, 22f, 33f, 14f, 25f, 36f }, z.Data);
        Assert.Equal(new[] { 5f, 7f, 9f }, x.SumRows().Data);
    }

    [Fact]
    public void Softmax_rows_sum_to_one()
    {
        var x = new Tensor(new[] { 2, 3 }, new[] { 1f, 2f, 3f, -1f, 0f, 1f });
        var s = Activations.SoftmaxRows(x);
        for (var i = 0; i < 2; i++)
        {
            var sum = s.Data[i * 3] + s.Data[i * 3 + 1] + s.Data[i * 3 + 2];
            Assert.True(Math.Abs(sum - 1f) < 1e-6f);
        }
    }

    [Fact]
    public void SplitMix64_is_deterministic_for_a_seed()
    {
        var a = new SplitMix64(42);
        var b = new SplitMix64(42);
        for (var i = 0; i < 100; i++) Assert.Equal(a.NextU64(), b.NextU64());
    }
}
