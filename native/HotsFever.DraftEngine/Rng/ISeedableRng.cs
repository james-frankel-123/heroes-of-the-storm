namespace HotsFever.DraftEngine.Rng;

/// <summary>
/// Source of uniform [0,1) draws for the MCTS opponent sampling. Abstracted so
/// the search can be driven deterministically in test mode: the C# and web
/// engines step through the SAME preset number sequence, so a seeded run is
/// exactly reproducible on both sides (the M1 parity plan's mock-RNG mode).
/// In production, <see cref="SystemRng"/> is used and the sequence is irrelevant.
/// </summary>
public interface ISeedableRng
{
    /// <summary>Next uniform draw in [0, 1) — mirrors JS Math.random().</summary>
    double NextDouble();
}

/// <summary>Production RNG. Non-deterministic, backed by System.Random.</summary>
public sealed class SystemRng : ISeedableRng
{
    private readonly Random _random;
    public SystemRng() => _random = new Random();
    public SystemRng(int seed) => _random = new Random(seed);
    public double NextDouble() => _random.NextDouble();
}

/// <summary>
/// Deterministic test RNG: cycles a preset sequence of draws that both the C#
/// and web engines agree on ahead of time. This is the shared "mock random
/// number generator" used to validate MCTS equivalence — identical draws in
/// identical call order on both sides.
/// </summary>
public sealed class SequenceRng : ISeedableRng
{
    private readonly IReadOnlyList<double> _seq;
    private int _i;

    public SequenceRng(IReadOnlyList<double> sequence)
    {
        if (sequence == null || sequence.Count == 0)
            throw new ArgumentException("sequence must be non-empty", nameof(sequence));
        foreach (var d in sequence)
            if (d < 0.0 || d >= 1.0)
                throw new ArgumentOutOfRangeException(nameof(sequence), d, "draws must be in [0,1)");
        _seq = sequence;
    }

    /// <summary>Number of draws requested so far (useful for parity assertions).</summary>
    public int DrawCount => _i;

    public double NextDouble()
    {
        var v = _seq[_i % _seq.Count];
        _i++;
        return v;
    }
}
