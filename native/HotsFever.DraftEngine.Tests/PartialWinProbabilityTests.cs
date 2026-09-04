using HotsFever.DraftEngine.Data;
using HotsFever.DraftEngine.Models;
using Xunit;
using Xunit.Abstractions;

namespace HotsFever.DraftEngine.Tests;

/// <summary>
/// The partial-draft judge exists because the policy value head measured as
/// state-insensitive on incomplete drafts â€” it barely moved as the board changed, so
/// backed-up Q (and every delta shown to the user) was mush. These tests pin the two
/// properties that made it worth wiring in: it RESPONDS to the state, and it is
/// symmetric, so swapping the teams gives exactly the complementary probability.
/// </summary>
public class PartialWinProbabilityTests
{
    private readonly ITestOutputHelper _out;
    public PartialWinProbabilityTests(ITestOutputHelper o) => _out = o;

    private const string Map = "Sky Temple";
    private const string Tier = "mid";

    [Fact]
    public void PartialModel_IsLoaded_AndExposesStepInput()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        Assert.True(sessions.HasPartialWinProbability,
            "partial_wp.onnx should load from the models directory");
    }

    [Fact]
    public void Partial_IsSymmetric()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        var data = DraftDataLoader.Load(TestPaths.StatsJson(), TestPaths.CompositionsJson(), Tier);
        string[] t0 = { "Muradin", "Jaina", "Rehgar" };
        string[] t1 = { "Diablo", "Valla" };

        float p = PartialWinProbability.Get(sessions, t0, t1, Map, Tier, step: 5, data);
        float pSwapped = PartialWinProbability.Get(sessions, t1, t0, Map, Tier, step: 5, data);

        _out.WriteLine($"p(t0)={p:F4}  p(t1)={pSwapped:F4}  sum={p + pSwapped:F6}");
        Assert.InRange(p + pSwapped, 0.999, 1.001);
    }

    [Fact]
    public void Partial_RespondsToTheBoard()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        var data = DraftDataLoader.Load(TestPaths.StatsJson(), TestPaths.CompositionsJson(), Tier);

        // Same step, same map/tier â€” only the heroes differ. A state-insensitive
        // evaluator returns (near) the same number for both; that is the failure
        // mode this model was brought in to fix.
        string[] ourStrong = { "Muradin", "Jaina", "Rehgar" };
        string[] ourWeak = { "Abathur", "Murky", "Nova" };
        string[] enemy = { "Diablo", "Valla" };

        float strong = PartialWinProbability.Get(sessions, ourStrong, enemy, Map, Tier, 5, data);
        float weak = PartialWinProbability.Get(sessions, ourWeak, enemy, Map, Tier, 5, data);

        _out.WriteLine($"strong={strong:F4}  weak={weak:F4}  spread={Math.Abs(strong - weak):F4}");
        Assert.True(Math.Abs(strong - weak) > 0.01,
            $"partial judge should separate clearly different boards, got {strong:F4} vs {weak:F4}");
        Assert.InRange(strong, 0f, 1f);
        Assert.InRange(weak, 0f, 1f);
    }

    [Fact]
    public void Partial_RespondsToDraftStep()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        var data = DraftDataLoader.Load(TestPaths.StatsJson(), TestPaths.CompositionsJson(), Tier);
        string[] t0 = { "Muradin", "Jaina" };
        string[] t1 = { "Diablo" };

        float early = PartialWinProbability.Get(sessions, t0, t1, Map, Tier, step: 2, data);
        float late = PartialWinProbability.Get(sessions, t0, t1, Map, Tier, step: 12, data);

        _out.WriteLine($"step2={early:F4}  step12={late:F4}");
        Assert.True(float.IsFinite(early) && float.IsFinite(late));
        Assert.InRange(early, 0f, 1f);
        Assert.InRange(late, 0f, 1f);
    }
}
