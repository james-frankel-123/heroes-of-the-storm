using HotsFever.DraftEngine;
using HotsFever.DraftEngine.Data;
using HotsFever.DraftEngine.Models;
using HotsFever.DraftEngine.Rng;
using HotsFever.DraftEngine.Scoring;
using HotsFever.DraftEngine.Search;
using Xunit;
using Xunit.Abstractions;

namespace HotsFever.DraftEngine.Tests;

/// <summary>
/// The orchestration layer end-to-end: enriched WP features, symmetrized win
/// probability, MAWP, and getAIRecommendations — against the real stat tables.
/// </summary>
public class OrchestrationTests
{
    private readonly ITestOutputHelper _out;
    public OrchestrationTests(ITestOutputHelper o) => _out = o;

    private static readonly double[] Seq =
    {
        0.10, 0.37, 0.62, 0.85, 0.05, 0.50, 0.95, 0.25, 0.70, 0.42,
        0.13, 0.88, 0.31, 0.57, 0.04, 0.66, 0.79, 0.21, 0.48, 0.92,
    };

    private static DraftData LoadMid() =>
        DraftDataLoader.Load(TestPaths.StatsJson(), TestPaths.CompositionsJson(), "mid");

    // ── Enriched features ────────────────────────────────────────────

    [Fact]
    public void EnrichedFeatures_HasCorrectLengthAndRoleCounts()
    {
        var d = LoadMid();
        // team0: Muradin (tank=0), Rehgar (healer=2); team1: Raynor (ranged_aa=3)
        var f = EnrichedFeatures.Compute(
            new[] { "Muradin", "Rehgar" }, new[] { "Raynor" }, "Cursed Hollow", d);

        Assert.Equal(86, f.Length);
        Assert.Equal(1f, f[0]);  // team0 tank count
        Assert.Equal(1f, f[2]);  // team0 healer count
        Assert.Equal(1f, f[9 + 3]); // team1 ranged_aa count (block starts at 9)
        Assert.All(f, x => Assert.True(float.IsFinite(x)));
    }

    // ── Win probability (symmetrized) ────────────────────────────────

    [Fact]
    public void WinProbability_IsSymmetric_AndInRange()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        var d = LoadMid();
        var t0 = new[] { "Muradin", "Raynor", "Malfurion" };
        var t1 = new[] { "Diablo", "Jaina", "Rehgar" };

        float p01 = WinProbability.Get(sessions, t0, t1, "Cursed Hollow", "mid", d);
        float p10 = WinProbability.Get(sessions, t1, t0, "Cursed Hollow", "mid", d);

        Assert.InRange(p01, 0f, 1f);
        // Symmetrization guarantees P(t0,t1) + P(t1,t0) == 1 exactly.
        Assert.Equal(1.0, p01 + p10, 5);
        _out.WriteLine($"P(t0)={p01:F4}, P(t1)={p10:F4}, sum={p01 + p10:F6}");
    }

    // ── MAWP ─────────────────────────────────────────────────────────

    [Fact]
    public void Mawp_EmptyPlayerData_YieldsNoAdjustment()
    {
        Assert.Equal(0.0, Mawp.ComputeTeamMawpAdjustment(new[] { "Muradin" }, null));
        var empty = new PlayerMawpData();
        Assert.Equal(0.0, Mawp.ComputePlayerAdjustment("Muradin", empty).Value);
    }

    [Fact]
    public void Mawp_StrongPlayer_YieldsPositiveAdjustment()
    {
        var pd = new PlayerMawpData
        {
            AvailableBattletags = new[] { "Ace#1234" },
            PlayerStats = new()
            {
                ["Ace#1234"] = new() { ["Muradin"] = new PlayerHeroStat { Games = 50, Wins = 40, WinRate = 80, Mawp = 65 } },
            },
        };
        var adj = Mawp.ComputePlayerAdjustment("Muradin", pd);
        Assert.Equal("Ace#1234", adj.Player);
        Assert.True(adj.Value > 0, "a 65 MAWP (>50) should give a positive adjustment");
        // games (50) >= confidence threshold (30) → no shrinkage → 0.04 * (65-50)/100
        Assert.Equal(0.04 * (65 - 50) / 100.0, adj.Value, 9);
    }

    // ── getAIRecommendations end-to-end ──────────────────────────────

    private static MctsSearch.Input SampleInput() => new()
    {
        Team0Picks = Array.Empty<string>(),
        Team1Picks = Array.Empty<string>(),
        Bans = new[] { "Nazeebo", "Alarak", "Diablo", "Malfurion" },
        TakenHeroes = new[] { "Nazeebo", "Alarak", "Diablo", "Malfurion" },
        Map = "Cursed Hollow",
        Tier = "mid",
        Step = 4,
        OurTeam = 0,
    };

    private static MctsSearch.Options FixedSims(int n) =>
        new() { MinSims = Math.Min(5, n), MaxSims = n, TimeBudgetMs = double.PositiveInfinity };

    [Fact]
    public void GetRecommendations_WithRealData_IsDeterministicAndSane()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        var d = LoadMid();

        var r1 = AiInference.GetRecommendations(sessions, SampleInput(), isBanStep: false,
            new SequenceRng(Seq), FixedSims(32), d);
        var r2 = AiInference.GetRecommendations(sessions, SampleInput(), isBanStep: false,
            new SequenceRng(Seq), FixedSims(32), d);

        Assert.NotEmpty(r1.Recommendations);
        Assert.Equal(r1.Recommendations.Count, r2.Recommendations.Count);
        for (int i = 0; i < r1.Recommendations.Count; i++)
        {
            Assert.Equal(r1.Recommendations[i].Hero, r2.Recommendations[i].Hero);
            Assert.Equal(r1.Recommendations[i].WinProb, r2.Recommendations[i].WinProb, 6);
        }

        var taken = new HashSet<string>(SampleInput().TakenHeroes);
        foreach (var rec in r1.Recommendations)
        {
            Assert.DoesNotContain(rec.Hero, taken);
            Assert.InRange(rec.WinProb, 0.0, 1.0);
        }

        _out.WriteLine($"value={r1.ValueEstimate:F4}, sims={r1.Sims}");
        foreach (var rec in r1.Recommendations.Take(5))
            _out.WriteLine($"  {rec.Hero,-16} prior={rec.Prior:F3} winProb={rec.WinProb:F4}");
    }
}
