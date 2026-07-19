using System.Text.Json;
using System.Text.Json.Serialization;
using HotsFever.DraftEngine.Models;
using HotsFever.DraftEngine.Rng;
using HotsFever.DraftEngine.Search;
using Xunit;
using Xunit.Abstractions;

namespace HotsFever.DraftEngine.Tests;

/// <summary>
/// The parity oracle (MCTS behavioral layer): runs the C# search through the
/// SAME shared mock-RNG sequence + fixed sim count as the real TS engine
/// (tools/oracle/mcts-golden.json) and checks behavioral agreement.
///
/// Not bit-identical: ~1e-6 model divergence occasionally flips a UCB/GD branch
/// and desyncs the shared RNG stream. So we assert the top pick agrees on (nearly)
/// every case and the value estimate is close — "exact in seeded mode, statistical
/// in the wild," per the M1 plan. Skipped if the golden file isn't generated.
/// </summary>
public class MctsBehavioralParityTests
{
    private readonly ITestOutputHelper _out;
    public MctsBehavioralParityTests(ITestOutputHelper o) => _out = o;

    private sealed class Golden
    {
        [JsonPropertyName("_meta")] public Meta MetaInfo { get; set; } = new();
        public double[] RngSequence { get; set; } = Array.Empty<double>();
        public List<Case> Cases { get; set; } = new();
    }
    private sealed class Meta { public int MaxSims { get; set; } = 60; public int MinSims { get; set; } = 5; }
    private sealed class Case
    {
        public CaseMeta Meta { get; set; } = new();
        public double ValueEstimate { get; set; }
        public int Sims { get; set; }
        public List<Rec> Recommendations { get; set; } = new();
    }
    private sealed class CaseMeta
    {
        public string[] Team0Picks { get; set; } = Array.Empty<string>();
        public string[] Team1Picks { get; set; } = Array.Empty<string>();
        public string[] Bans { get; set; } = Array.Empty<string>();
        public string Map { get; set; } = "";
        public string Tier { get; set; } = "mid";
        public int Step { get; set; }
        public int OurTeam { get; set; }
    }
    private sealed class Rec { public string Hero { get; set; } = ""; public double Visits { get; set; } public double Q { get; set; } }

    [Fact]
    public void Mcts_MatchesWebEngine_Behaviorally()
    {
        var path = TestPaths.MctsGolden();
        if (path == null) { _out.WriteLine("SKIP: mcts-golden.json not generated (tools/oracle: npm run gen:mcts)"); return; }
        var g = JsonSerializer.Deserialize<Golden>(File.ReadAllText(path),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;

        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        var options = new MctsSearch.Options { MinSims = g.MetaInfo.MinSims, MaxSims = g.MetaInfo.MaxSims, TimeBudgetMs = double.PositiveInfinity };

        int top1Agree = 0, top3Agree = 0, exactTop5 = 0;
        double maxValueDiff = 0;
        int n = g.Cases.Count;

        foreach (var c in g.Cases)
        {
            var input = new MctsSearch.Input
            {
                Team0Picks = c.Meta.Team0Picks, Team1Picks = c.Meta.Team1Picks, Bans = c.Meta.Bans,
                TakenHeroes = c.Meta.Team0Picks.Concat(c.Meta.Team1Picks).Concat(c.Meta.Bans).ToArray(),
                Map = c.Meta.Map, Tier = c.Meta.Tier, Step = c.Meta.Step, OurTeam = c.Meta.OurTeam,
            };
            // Same shared RNG sequence, fresh per case (mirrors the harness resetting idx=0).
            var result = MctsSearch.Run(sessions, input, new SequenceRng(g.RngSequence), options);

            Assert.Equal(c.Sims, result.Sims); // both run the full fixed sim count

            var webTop = c.Recommendations.Select(r => r.Hero).ToList();
            var csTop = result.Recommendations.Select(r => r.Hero).ToList();

            if (webTop.Count > 0 && csTop.Count > 0 && webTop[0] == csTop[0]) top1Agree++;
            if (webTop.Count > 0 && csTop.Take(3).Contains(webTop[0])) top3Agree++;
            if (webTop.Take(5).SequenceEqual(csTop.Take(5))) exactTop5++;
            maxValueDiff = Math.Max(maxValueDiff, Math.Abs(result.ValueEstimate - c.ValueEstimate));
        }

        _out.WriteLine($"MCTS behavioral parity over {n} cases:");
        _out.WriteLine($"  top-1 agreement: {top1Agree}/{n}");
        _out.WriteLine($"  web top-1 in C# top-3: {top3Agree}/{n}");
        _out.WriteLine($"  exact top-5 order match: {exactTop5}/{n}");
        _out.WriteLine($"  max value-estimate diff: {maxValueDiff:F5}");

        // Statistical thresholds: the dominant arm is robust to ~1e-6 perturbation.
        Assert.True(top1Agree >= (int)Math.Ceiling(0.85 * n),
            $"top-1 agreement {top1Agree}/{n} below 85%");
        Assert.True(top3Agree >= (int)Math.Ceiling(0.95 * n),
            $"web top-1 in C# top-3 {top3Agree}/{n} below 95%");
        Assert.True(maxValueDiff < 0.05, $"max value diff {maxValueDiff} >= 0.05");
    }
}
