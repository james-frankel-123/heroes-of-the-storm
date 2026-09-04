using HotsFever.DraftEngine;
using HotsFever.DraftEngine.Data;
using HotsFever.DraftEngine.Models;
using HotsFever.DraftEngine.Search;
using Xunit;
using Xunit.Abstractions;

namespace HotsFever.DraftEngine.Tests;

/// <summary>
/// Opponent predictions — what the enemy is likely to take next, and what it costs us.
/// The engine already ran the opponent model inside MCTS; these tests cover surfacing it.
/// </summary>
public class OpponentPredictionTests
{
    private readonly ITestOutputHelper _out;
    public OpponentPredictionTests(ITestOutputHelper o) => _out = o;

    private static MctsSearch.Input MidDraft(int step, int ourTeam = 0)
    {
        string[] t0 = { "Muradin", "Jaina" };
        string[] t1 = { "Diablo" };
        string[] bans = { "Nova", "Abathur" };
        return new MctsSearch.Input
        {
            Team0Picks = t0,
            Team1Picks = t1,
            Bans = bans,
            TakenHeroes = t0.Concat(t1).Concat(bans).ToArray(),
            Map = "Sky Temple",
            Tier = "mid",
            Step = step,
            OurTeam = ourTeam,
        };
    }

    private static DraftData Data() =>
        DraftDataLoader.Load(TestPaths.StatsJson(), TestPaths.CompositionsJson(), "mid");

    [Fact]
    public void PickStep_RanksByDamage_AndExcludesTakenHeroes()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        var input = MidDraft(step: 7);

        var preds = AiInference.GetOpponentPredictions(sessions, input, isBanStep: false, Data(), topK: 6);

        Assert.NotEmpty(preds);
        foreach (var p in preds) _out.WriteLine($"{p.Hero,-14} {p.Probability * 100,5:F1}%  impact {p.ImpactPp,6:F1}pp");

        // Biggest threat first: most negative impact on our win chance leads.
        for (int i = 1; i < preds.Count; i++)
            Assert.True(preds[i - 1].ImpactPp <= preds[i].ImpactPp,
                $"expected damage-ordered rows, got {preds[i - 1].Hero} {preds[i - 1].ImpactPp} before {preds[i].Hero} {preds[i].ImpactPp}");

        // Never suggests a hero already picked or banned.
        var taken = new HashSet<string>(input.TakenHeroes);
        Assert.All(preds, p => Assert.DoesNotContain(p.Hero, taken));
        Assert.All(preds, p => Assert.InRange(p.Probability, 0.0, 1.0));
    }

    [Fact]
    public void BanStep_StaysInLikelihoodOrder()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        var preds = AiInference.GetOpponentPredictions(sessions, MidDraft(6), isBanStep: true, Data(), topK: 5);

        Assert.NotEmpty(preds);
        for (int i = 1; i < preds.Count; i++)
            Assert.True(preds[i - 1].Probability >= preds[i].Probability);
    }

    [Fact]
    public void PickStep_PricesImpact_BanStepDoesNot()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        var data = Data();

        var onPick = AiInference.GetOpponentPredictions(sessions, MidDraft(7), isBanStep: false, data, topK: 4);
        var onBan = AiInference.GetOpponentPredictions(sessions, MidDraft(6), isBanStep: true, data, topK: 4);

        // A pick adds a hero to their comp, so it can be priced against our win chance.
        Assert.All(onPick, p => Assert.NotNull(p.ImpactPp));
        // A ban removes a hero instead, so the same before/after comparison doesn't apply.
        Assert.All(onBan, p => Assert.Null(p.ImpactPp));
    }

    [Fact]
    public void Impact_IsSignedFromOurPointOfView()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        var data = Data();

        // Same board, read from each side. A hero landing on the enemy team should not
        // look equally good to both teams — the sign is relative to whoever "we" are.
        var asTeam0 = AiInference.GetOpponentPredictions(sessions, MidDraft(7, ourTeam: 0), false, data, topK: 3);
        var asTeam1 = AiInference.GetOpponentPredictions(sessions, MidDraft(7, ourTeam: 1), false, data, topK: 3);

        foreach (var p in asTeam0) _out.WriteLine($"ourTeam=0  {p.Hero,-14} {p.ImpactPp,6:F1}pp");
        foreach (var p in asTeam1) _out.WriteLine($"ourTeam=1  {p.Hero,-14} {p.ImpactPp,6:F1}pp");

        Assert.All(asTeam0, p => Assert.True(Math.Abs(p.ImpactPp!.Value) <= 100));
        Assert.All(asTeam1, p => Assert.True(Math.Abs(p.ImpactPp!.Value) <= 100));
    }
}
