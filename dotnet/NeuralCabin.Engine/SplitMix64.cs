namespace NeuralCabin.Engine;

/// <summary>
/// Deterministic SplitMix64 PRNG — a faithful port of <c>tensor::SplitMix64</c>
/// in the Rust engine, including the exact <c>next_f32</c> / <c>next_normal</c>
/// derivations so seeded weight initialization matches across the migration.
/// </summary>
public sealed class SplitMix64
{
    private ulong _state;

    public SplitMix64(ulong seed)
    {
        unchecked
        {
            _state = seed + 0x9E3779B97F4A7C15UL;
        }
    }

    public ulong NextU64()
    {
        unchecked
        {
            _state += 0x9E3779B97F4A7C15UL;
            var z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    /// <summary>24 high bits mapped into [0, 1).</summary>
    public float NextF32() => (NextU64() >> 40) / (float)(1u << 24);

    /// <summary>Box–Muller standard normal sample.</summary>
    public float NextNormal()
    {
        var u1 = Math.Clamp(NextF32(), 1e-7f, 1.0f - 1e-7f);
        var u2 = NextF32();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}
