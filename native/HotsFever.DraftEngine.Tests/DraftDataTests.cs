using HotsFever.DraftEngine.Data;
using Xunit;

namespace HotsFever.DraftEngine.Tests;

/// <summary>Loading the real committed stat tables (src/lib/data/*.json).</summary>
public class DraftDataTests
{
    private static DraftData LoadMid() =>
        DraftDataLoader.Load(TestPaths.StatsJson(), TestPaths.CompositionsJson(), "mid");

    [Fact]
    public void Load_Mid_HasKnownHeroStat()
    {
        var d = LoadMid();
        Assert.True(d.HeroStats.ContainsKey("Muradin"));
        var m = d.HeroStats["Muradin"];
        Assert.Equal(47.776, m.WinRate, 3);
        Assert.Equal(24.569, m.PickRate, 3);
        Assert.Equal(28068.2, m.Games, 1);
    }

    [Fact]
    public void Load_Mid_HasAllTablesPopulated()
    {
        var d = LoadMid();
        Assert.Equal(90, d.HeroStats.Count);
        Assert.NotEmpty(d.HeroMapWinRates);
        Assert.NotEmpty(d.Synergies);
        Assert.NotEmpty(d.Counters);
        Assert.Equal(127, d.Compositions.Count);
        // player stats are Postgres-only → empty for the native client
        Assert.Empty(d.PlayerStats);
    }

    [Fact]
    public void Load_Mid_BaselineCompWR_IsPopularityWeightedMean()
    {
        var d = LoadMid();
        // Real popularity-weighted mean over the mid compositions (regression anchor).
        Assert.Equal(49.9193, d.BaselineCompWR, 4);
    }

    [Fact]
    public void ComputeBaselineCompWR_UnitBehavior()
    {
        // empty → 50
        Assert.Equal(50.0, DraftDataLoader.ComputeBaselineCompWR(new List<CompositionData>()), 6);
        // all-zero popularity → 50 (zero total weight fallback)
        Assert.Equal(50.0, DraftDataLoader.ComputeBaselineCompWR(new List<CompositionData>
        {
            new() { WinRate = 55, Popularity = 0 },
            new() { WinRate = 45, Popularity = 0 },
        }), 6);
        // weighted mean: (60*3 + 50*1) / 4 = 57.5
        Assert.Equal(57.5, DraftDataLoader.ComputeBaselineCompWR(new List<CompositionData>
        {
            new() { WinRate = 60, Popularity = 3 },
            new() { WinRate = 50, Popularity = 1 },
        }), 6);
    }

    [Fact]
    public void Load_AllTiers_Parse()
    {
        foreach (var tier in new[] { "low", "mid", "high" })
        {
            var d = DraftDataLoader.Load(TestPaths.StatsJson(), TestPaths.CompositionsJson(), tier);
            Assert.NotEmpty(d.HeroStats);
        }
    }
}
