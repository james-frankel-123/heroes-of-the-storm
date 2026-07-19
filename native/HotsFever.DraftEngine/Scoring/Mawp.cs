using HotsFever.DraftEngine.Data;

namespace HotsFever.DraftEngine.Scoring;

/// <summary>Player MAWP data — the personal-stats slice used to personalize picks.</summary>
public sealed class PlayerMawpData
{
    /// <summary>battletag → hero → stat.</summary>
    public Dictionary<string, Dictionary<string, PlayerHeroStat>> PlayerStats { get; init; } = new();
    /// <summary>Battletags on our team not yet assigned to a pick.</summary>
    public IReadOnlyList<string> AvailableBattletags { get; init; } = Array.Empty<string>();
}

/// <summary>
/// Momentum-Adjusted Win Percentage player adjustments — a port of the MAWP
/// helpers in ai-inference.ts. For the native client these are typically no-ops
/// (player stats are Postgres-only and left empty), but the logic is ported so
/// it activates once player data is wired in (M3).
/// </summary>
public static class Mawp
{
    private const double Weight = 0.04;
    private const double MinGames = 10;
    private const double ConfidenceThreshold = 30;

    public static double ConfidenceAdjustedMawp(double mawp, double games)
    {
        if (games >= ConfidenceThreshold) return mawp;
        double w = games / ConfidenceThreshold;
        return mawp * w + 50 * (1 - w);
    }

    public readonly record struct Adjustment(double Value, string? Player);

    /// <summary>Best available player for a hero + the MAWP adjustment they confer.</summary>
    public static Adjustment ComputePlayerAdjustment(string hero, PlayerMawpData? pd)
    {
        if (pd == null || pd.AvailableBattletags.Count == 0) return new Adjustment(0, null);

        double bestAdj = 0;
        string? bestPlayer = null;
        foreach (var bt in pd.AvailableBattletags)
        {
            if (!pd.PlayerStats.TryGetValue(bt, out var byHero) ||
                !byHero.TryGetValue(hero, out var st) || st.Games < MinGames) continue;

            double adjMawp = st.Mawp.HasValue
                ? ConfidenceAdjustedMawp(st.Mawp.Value, st.Games)
                : st.Wins / st.Games * 100;

            double delta = (adjMawp - 50) / 100;
            double adj = Weight * delta;

            if (adj > bestAdj || bestPlayer == null) { bestAdj = adj; bestPlayer = bt; }
        }
        return new Adjustment(bestAdj, bestPlayer);
    }

    /// <summary>Aggregate adjustment across all already-picked heroes with an assigned player.</summary>
    public static double ComputeTeamMawpAdjustment(IEnumerable<string> allPickedHeroes, PlayerMawpData? pd)
    {
        if (pd == null) return 0;
        double total = 0;
        foreach (var h in allPickedHeroes) total += ComputePlayerAdjustment(h, pd).Value;
        return total;
    }
}
