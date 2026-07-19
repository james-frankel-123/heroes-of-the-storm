using HotsFever.DraftEngine.Encoding;
using HotsFever.DraftEngine.Models;
using HotsFever.DraftEngine.Rng;
using HotsFever.DraftEngine.Search;
using Xunit;
using Xunit.Abstractions;

namespace HotsFever.DraftEngine.Tests;

public class MctsTests
{
    private readonly ITestOutputHelper _out;
    public MctsTests(ITestOutputHelper o) => _out = o;

    // A fixed preset draw sequence — the "mock RNG" both engines would share in
    // deterministic test mode.
    private static readonly double[] Seq =
    {
        0.10, 0.37, 0.62, 0.85, 0.05, 0.50, 0.95, 0.25, 0.70, 0.42,
        0.13, 0.88, 0.31, 0.57, 0.04, 0.66, 0.79, 0.21, 0.48, 0.92,
    };

    // Mid-draft: 4 bans done (steps 0-3), step 4 is team 0's first pick = our decision.
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

    private static MctsSearch.Options FixedSims(int n) => new()
    {
        MinSims = Math.Min(5, n),
        MaxSims = n,
        TimeBudgetMs = double.PositiveInfinity, // uncapped → runs exactly MaxSims (test mode)
    };

    [Fact]
    public void Mcts_IsDeterministic_UnderSharedSequence()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());

        var r1 = MctsSearch.Run(sessions, SampleInput(), new SequenceRng(Seq), FixedSims(40));
        var r2 = MctsSearch.Run(sessions, SampleInput(), new SequenceRng(Seq), FixedSims(40));

        Assert.Equal(r1.Sims, r2.Sims);
        Assert.Equal(r1.Recommendations.Count, r2.Recommendations.Count);
        Assert.Equal(r1.ValueEstimate, r2.ValueEstimate, 6);
        for (int i = 0; i < r1.Recommendations.Count; i++)
        {
            Assert.Equal(r1.Recommendations[i].Hero, r2.Recommendations[i].Hero);
            Assert.Equal(r1.Recommendations[i].Visits, r2.Recommendations[i].Visits, 6);
            Assert.Equal(r1.Recommendations[i].Q, r2.Recommendations[i].Q, 6);
        }
    }

    [Fact]
    public void Mcts_ProducesSaneRecommendations()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        var input = SampleInput();

        var result = MctsSearch.Run(sessions, input, new SequenceRng(Seq), FixedSims(40));

        Assert.Equal(40, result.Sims); // uncapped time → full sim count
        Assert.NotEmpty(result.Recommendations);
        Assert.InRange(result.ValueEstimate, 0.0, 1.0);

        var taken = new HashSet<string>(input.TakenHeroes);
        foreach (var rec in result.Recommendations)
        {
            Assert.DoesNotContain(rec.Hero, taken);       // never recommends a taken hero
            Assert.InRange(rec.Q, 0.0, 1.0);
            Assert.InRange(rec.Visits, 0.0, 1.0);
        }
        // visits are a normalized distribution over visited children
        Assert.True(result.Recommendations.Sum(r => r.Visits) <= 1.0000001);

        _out.WriteLine($"valueEstimate={result.ValueEstimate:F4}, sims={result.Sims}");
        foreach (var rec in result.Recommendations.Take(5))
            _out.WriteLine($"  {rec.Hero,-16} visits={rec.Visits:F3} q={rec.Q:F4}");
    }

    [Fact]
    public void Mcts_RecommendationsSortedByVisitsDescending()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        var result = MctsSearch.Run(sessions, SampleInput(), new SequenceRng(Seq), FixedSims(40));

        for (int i = 1; i < result.Recommendations.Count; i++)
            Assert.True(result.Recommendations[i - 1].Visits >= result.Recommendations[i].Visits);
    }

    [Fact]
    public void Mcts_TerminalEvaluator_IsUsedForCompleteDrafts()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        int calls = 0;
        // Stub WP evaluator: proves the terminal hook fires and its value flows through.
        float Evaluator(IReadOnlyList<string> t0, IReadOnlyList<string> t1) { calls++; return 0.5f; }

        var result = MctsSearch.Run(sessions, SampleInput(), new SequenceRng(Seq), FixedSims(40), Evaluator);

        Assert.True(calls > 0, "terminal evaluator should fire on complete-draft leaves");
        Assert.NotEmpty(result.Recommendations);
    }
}
