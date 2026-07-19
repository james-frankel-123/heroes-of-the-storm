using System.Text.Json;

namespace HotsFever.DraftEngine.Data;

// Mirrors src/lib/draft/types.ts DraftData. All winRate/games are double
// (decayed "games" are fractional effective counts). Player fields have no file
// artifact (Postgres only) and are left empty for the native client, exactly as
// the repo's benchmark/load-draft-data.ts does.

public sealed class HeroStat
{
    public double WinRate { get; set; }
    public double PickRate { get; set; }
    public double BanRate { get; set; }
    public double Games { get; set; }
}

public sealed class WinRateGames
{
    public double WinRate { get; set; }
    public double Games { get; set; }
}

public sealed class PlayerHeroStat
{
    public double Games { get; set; }
    public double Wins { get; set; }
    public double WinRate { get; set; }
    public double? Mawp { get; set; }
}

public sealed class CompositionData
{
    public List<string> Roles { get; set; } = new();
    public double WinRate { get; set; }
    public double Games { get; set; }
    public double Popularity { get; set; }
}

public sealed class DraftData
{
    public Dictionary<string, HeroStat> HeroStats { get; set; } = new();
    /// <summary>map → hero → {winRate, games}</summary>
    public Dictionary<string, Dictionary<string, WinRateGames>> HeroMapWinRates { get; set; } = new();
    /// <summary>heroA → heroB → {winRate, games} ('with')</summary>
    public Dictionary<string, Dictionary<string, WinRateGames>> Synergies { get; set; } = new();
    /// <summary>heroA → heroB → {winRate, games} ('against')</summary>
    public Dictionary<string, Dictionary<string, WinRateGames>> Counters { get; set; } = new();
    /// <summary>battletag → hero → stat. Empty for the native client (Postgres-only).</summary>
    public Dictionary<string, Dictionary<string, PlayerHeroStat>> PlayerStats { get; set; } = new();
    public Dictionary<string, Dictionary<string, Dictionary<string, WinRateGames>>> PlayerMapStats { get; set; } = new();
    public List<CompositionData> Compositions { get; set; } = new();
    public double BaselineCompWR { get; set; }
}

/// <summary>
/// Loads a tier's <see cref="DraftData"/> from the two committed JSON artifacts
/// (src/lib/data/draft-stats-decayed.json + compositions.json) — the same source
/// the web app's getDraftData() reads. Player stats are left empty.
/// </summary>
public static class DraftDataLoader
{
    private static readonly JsonSerializerOptions Opts = new() { PropertyNameCaseInsensitive = true };

    private sealed class StatsFile
    {
        public Dictionary<string, TierStats> Tiers { get; set; } = new();
    }

    private sealed class TierStats
    {
        public Dictionary<string, HeroStat> HeroStats { get; set; } = new();
        public Dictionary<string, Dictionary<string, WinRateGames>> HeroMapWinRates { get; set; } = new();
        public Dictionary<string, Dictionary<string, WinRateGames>> Synergies { get; set; } = new();
        public Dictionary<string, Dictionary<string, WinRateGames>> Counters { get; set; } = new();
    }

    public static DraftData Load(string statsJsonPath, string compositionsJsonPath, string tier)
    {
        var stats = JsonSerializer.Deserialize<StatsFile>(File.ReadAllText(statsJsonPath), Opts)
                    ?? throw new InvalidOperationException("Failed to parse draft-stats JSON");
        if (!stats.Tiers.TryGetValue(tier, out var ts))
            throw new ArgumentException($"Tier '{tier}' not found in stats JSON", nameof(tier));

        var comps = JsonSerializer.Deserialize<Dictionary<string, List<CompositionData>>>(
            File.ReadAllText(compositionsJsonPath), Opts) ?? new();
        comps.TryGetValue(tier, out var compList);
        compList ??= new List<CompositionData>();

        return new DraftData
        {
            HeroStats = ts.HeroStats,
            HeroMapWinRates = ts.HeroMapWinRates,
            Synergies = ts.Synergies,
            Counters = ts.Counters,
            Compositions = compList,
            BaselineCompWR = ComputeBaselineCompWR(compList),
            // PlayerStats / PlayerMapStats intentionally empty (Postgres-only).
        };
    }

    /// <summary>Popularity-weighted mean comp win rate; 50 when empty / zero weight. Mirrors composition.ts.</summary>
    public static double ComputeBaselineCompWR(IReadOnlyList<CompositionData> comps)
    {
        if (comps.Count == 0) return 50;
        double weightedSum = 0, totalWeight = 0;
        foreach (var c in comps)
        {
            weightedSum += c.WinRate * c.Popularity;
            totalWeight += c.Popularity;
        }
        return totalWeight > 0 ? weightedSum / totalWeight : 50;
    }
}
