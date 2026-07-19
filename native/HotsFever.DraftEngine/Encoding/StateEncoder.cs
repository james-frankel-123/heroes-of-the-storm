namespace HotsFever.DraftEngine.Encoding;

/// <summary>
/// A draft state as the models see it — the C# mirror of the web engine's
/// `AIDraftState` (src/lib/draft/ai-inference.ts).
/// </summary>
public sealed class AiDraftState
{
    public IReadOnlyList<string> Team0Picks { get; init; } = Array.Empty<string>();
    public IReadOnlyList<string> Team1Picks { get; init; } = Array.Empty<string>();
    public IReadOnlyList<string> Bans { get; init; } = Array.Empty<string>();
    public string Map { get; init; } = "";
    public string Tier { get; init; } = "mid";
    /// <summary>0-15.</summary>
    public int Step { get; init; }
    public bool IsPick { get; init; }
    /// <summary>0 or 1 — whose perspective.</summary>
    public int OurTeam { get; init; }
}

/// <summary>
/// Builds the exact float32 input vectors the three ONNX models expect. This is
/// Layer 1 of parity — these vectors must be byte-identical to the web engine's
/// `encodeState` / `encodeStateForGD` / WP base for the same state.
///
/// Layouts (matching ai-inference.ts):
///   Policy: 290 = heroes(90)x3 [t0,t1,bans] + map(14) + tier(3) + step/15 + pickFlag + ourTeam
///   GD:     289 = policy minus the trailing ourTeam bit
///   WP:     283 = base(197 = t0(90)+t1(90)+map(14)+tier(3)) + enriched(86)
/// </summary>
public static class StateEncoder
{
    public const int PolicyDims = 290;
    public const int GdDims = 289;
    public const int WpDims = 283;
    public const int WpBaseDims = 197;
    public const int EnrichedDims = 86;

    private static void WriteHeroesMultiHot(Span<float> dest, IReadOnlyList<string> heroes)
    {
        // dest is assumed zero-initialized by the caller.
        foreach (var name in heroes)
        {
            int idx = HeroCatalog.HeroIndex(name);
            if (idx >= 0) dest[idx] = 1f;
        }
    }

    /// <summary>Policy model input (290 dims).</summary>
    public static float[] EncodeState(AiDraftState s)
    {
        var input = new float[PolicyDims];
        int off = 0;
        WriteHeroesMultiHot(input.AsSpan(off, HeroCatalog.NumHeroes), s.Team0Picks); off += HeroCatalog.NumHeroes;
        WriteHeroesMultiHot(input.AsSpan(off, HeroCatalog.NumHeroes), s.Team1Picks); off += HeroCatalog.NumHeroes;
        WriteHeroesMultiHot(input.AsSpan(off, HeroCatalog.NumHeroes), s.Bans); off += HeroCatalog.NumHeroes;

        int mapIdx = HeroCatalog.MapIndex(s.Map);
        if (mapIdx >= 0) input[off + mapIdx] = 1f;
        off += HeroCatalog.NumMaps;

        int tierIdx = HeroCatalog.TierIndex(s.Tier);
        if (tierIdx >= 0) input[off + tierIdx] = 1f;
        off += HeroCatalog.NumTiers;

        input[off++] = s.Step / 15.0f;
        input[off++] = s.IsPick ? 1.0f : 0.0f;
        input[off++] = s.OurTeam;
        return input;
    }

    /// <summary>Generic-Draft model input (289 dims) — policy state minus the ourTeam bit.</summary>
    public static float[] EncodeStateForGd(AiDraftState s)
    {
        var full = EncodeState(s);
        var gd = new float[GdDims];
        Array.Copy(full, gd, GdDims);
        return gd;
    }

    /// <summary>
    /// Win-Probability model input (283 dims). The trailing 86 enriched features
    /// require DraftData and are ported in Phase B; until then they stay zero
    /// (matching the web engine's no-draftData branch).
    /// </summary>
    public static float[] EncodeWpBase(IReadOnlyList<string> team0, IReadOnlyList<string> team1, string map, string tier)
    {
        var input = new float[WpDims];
        int off = 0;
        WriteHeroesMultiHot(input.AsSpan(off, HeroCatalog.NumHeroes), team0); off += HeroCatalog.NumHeroes;
        WriteHeroesMultiHot(input.AsSpan(off, HeroCatalog.NumHeroes), team1); off += HeroCatalog.NumHeroes;

        int mapIdx = HeroCatalog.MapIndex(map);
        if (mapIdx >= 0) input[off + mapIdx] = 1f;
        off += HeroCatalog.NumMaps;

        int tierIdx = HeroCatalog.TierIndex(tier);
        if (tierIdx >= 0) input[off + tierIdx] = 1f;
        // remaining (enriched) dims left as zero
        return input;
    }

    /// <summary>90-length mask: 1 for available heroes, 0 for taken.</summary>
    public static float[] BuildValidMask(IEnumerable<string> taken)
    {
        var mask = new float[HeroCatalog.NumHeroes];
        Array.Fill(mask, 1f);
        foreach (var hero in taken)
        {
            int idx = HeroCatalog.HeroIndex(hero);
            if (idx >= 0) mask[idx] = 0f;
        }
        return mask;
    }
}
