namespace NeuralCabin.Engine;

/// <summary>
/// Dense, row-major tensor backed by a <c>float[]</c> — the engine's interchange
/// and on-disk type, a port of Rust's <c>tensor::Tensor</c>. Serializes as
/// <c>{ "shape": [...], "data": [...] }</c>, identical to the serde format.
/// </summary>
public sealed class Tensor
{
    public int[] Shape { get; }
    public float[] Data { get; }

    public Tensor(int[] shape, float[] data)
    {
        var n = 1;
        foreach (var d in shape) n *= d;
        if (n != data.Length)
            throw new ArgumentException($"data length {data.Length} does not match shape [{string.Join(",", shape)}]");
        Shape = shape;
        Data = data;
    }

    public static Tensor Zeros(params int[] shape)
    {
        var n = 1;
        foreach (var d in shape) n *= d;
        return new Tensor(shape, new float[n]);
    }

    public int Length => Data.Length;
    public int Rank => Shape.Length;

    public int Rows => Shape.Length == 2 ? Shape[0] : throw new InvalidOperationException("rows() requires a 2-D tensor");
    public int Cols => Shape.Length == 2 ? Shape[1] : throw new InvalidOperationException("cols() requires a 2-D tensor");

    /// <summary>Xavier/Glorot-uniform init using the deterministic PRNG (matches Rust).</summary>
    public static Tensor Xavier(int inDim, int outDim, SplitMix64 rng)
    {
        var limit = (float)Math.Sqrt(6.0 / (inDim + outDim));
        var n = inDim * outDim;
        var data = new float[n];
        for (var i = 0; i < n; i++)
        {
            var u = rng.NextF32() * 2.0f - 1.0f;
            data[i] = u * limit;
        }
        return new Tensor(new[] { inDim, outDim }, data);
    }

    public Tensor Clone() => new((int[])Shape.Clone(), (float[])Data.Clone());

    // ── 2-D linear algebra (the only shapes the engine needs) ──────────────────

    /// <summary>(m×k) · (k×n) = (m×n), ikj loop order for row-major cache behavior.</summary>
    public Tensor Matmul(Tensor other)
    {
        if (Shape.Length != 2 || other.Shape.Length != 2)
            throw new ArgumentException("matmul requires 2-D tensors");
        int m = Shape[0], k = Shape[1], k2 = other.Shape[0], n = other.Shape[1];
        if (k != k2)
            throw new ArgumentException($"cannot matmul [{m},{k}] x [{k2},{n}]");
        var outData = new float[m * n];
        var a = Data;
        var b = other.Data;
        for (var i = 0; i < m; i++)
        {
            var rowO = i * n;
            var rowA = i * k;
            for (var kk = 0; kk < k; kk++)
            {
                var av = a[rowA + kk];
                if (av == 0.0f) continue;
                var rowB = kk * n;
                for (var j = 0; j < n; j++)
                    outData[rowO + j] += av * b[rowB + j];
            }
        }
        return new Tensor(new[] { m, n }, outData);
    }

    public Tensor Transpose()
    {
        if (Shape.Length != 2) throw new InvalidOperationException("transpose requires a 2-D tensor");
        int r = Shape[0], c = Shape[1];
        var outData = new float[r * c];
        for (var i = 0; i < r; i++)
            for (var j = 0; j < c; j++)
                outData[j * r + i] = Data[i * c + j];
        return new Tensor(new[] { c, r }, outData);
    }

    /// <summary>Add a (1×cols) bias broadcast across <paramref name="rows"/> rows of this (rows×cols).</summary>
    public Tensor AddBiasRows(Tensor bias)
    {
        int r = Rows, c = Cols;
        var outData = new float[r * c];
        for (var i = 0; i < r; i++)
            for (var j = 0; j < c; j++)
                outData[i * c + j] = Data[i * c + j] + bias.Data[j];
        return new Tensor(new[] { r, c }, outData);
    }

    /// <summary>Sum a (rows×cols) tensor down its rows → (1×cols).</summary>
    public Tensor SumRows()
    {
        int r = Rows, c = Cols;
        var outData = new float[c];
        for (var i = 0; i < r; i++)
            for (var j = 0; j < c; j++)
                outData[j] += Data[i * c + j];
        return new Tensor(new[] { 1, c }, outData);
    }
}
