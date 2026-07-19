using HotsFever.DraftEngine.Encoding;
using HotsFever.DraftEngine.Models;
using Xunit;
using Xunit.Abstractions;

namespace HotsFever.DraftEngine.Tests;

/// <summary>
/// Layer 2 (smoke): the three ONNX models load and run under
/// Microsoft.ML.OnnxRuntime and produce well-formed, sane outputs against the
/// C#-built input tensors. The within-margin comparison to onnxruntime-web
/// lands with the oracle harness (Phase C).
/// </summary>
public class ForwardPassTests
{
    private readonly ITestOutputHelper _out;
    public ForwardPassTests(ITestOutputHelper o) => _out = o;

    private static AiDraftState SampleState() => new()
    {
        Team0Picks = new[] { "Muradin", "Raynor" },
        Team1Picks = new[] { "Diablo" },
        Bans = new[] { "Nazeebo", "Alarak" },
        Map = "Cursed Hollow",
        Tier = "mid",
        Step = 6,
        IsPick = true,
        OurTeam = 0,
    };

    private static IEnumerable<string> Taken(AiDraftState s)
        => s.Team0Picks.Concat(s.Team1Picks).Concat(s.Bans);

    [Fact]
    public void Models_LoadAndExposeExpectedIo()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());

        _out.WriteLine("policy inputs:  " + string.Join(", ", sessions.PolicyInputs.Keys));
        _out.WriteLine("policy outputs: " + string.Join(", ", sessions.PolicyOutputs.Keys));
        _out.WriteLine("gd outputs:     " + string.Join(", ", sessions.GdOutputs.Keys));
        _out.WriteLine("wp outputs:     " + string.Join(", ", sessions.WpOutputs.Keys));

        Assert.Contains("state", sessions.PolicyInputs.Keys);
        Assert.Contains("valid_mask", sessions.PolicyInputs.Keys);
        Assert.Contains("policy_logits", sessions.PolicyOutputs.Keys);
        Assert.Contains("value", sessions.PolicyOutputs.Keys);
        Assert.Contains("hero_logits", sessions.GdOutputs.Keys);
        Assert.Contains("win_probability", sessions.WpOutputs.Keys);
    }

    [Fact]
    public void Policy_ForwardPass_ProducesSaneOutput()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        var s = SampleState();
        var state = StateEncoder.EncodeState(s);
        var mask = StateEncoder.BuildValidMask(Taken(s));

        var result = sessions.RunPolicy(state, mask);

        Assert.Equal(HeroCatalog.NumHeroes, result.PolicyLogits.Length);
        Assert.All(result.PolicyLogits, x => Assert.True(float.IsFinite(x)));
        Assert.True(float.IsFinite(result.Value));
        Assert.InRange(result.Value, 0f, 1f);

        // Top-5 by logit among available heroes (sanity read-out).
        var top = result.PolicyLogits
            .Select((logit, i) => (Hero: HeroCatalog.Heroes[i], Logit: logit, Avail: mask[i] > 0))
            .Where(t => t.Avail)
            .OrderByDescending(t => t.Logit)
            .Take(5);
        _out.WriteLine($"value (raw)={result.Value:F4}");
        foreach (var t in top) _out.WriteLine($"  {t.Hero,-16} {t.Logit:F3}");
    }

    [Fact]
    public void GenericDraft_ForwardPass_ProducesSaneOutput()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        var s = SampleState();
        var state = StateEncoder.EncodeStateForGd(s);
        var mask = StateEncoder.BuildValidMask(Taken(s));

        var logits = sessions.RunGenericDraft(state, mask);

        Assert.Equal(HeroCatalog.NumHeroes, logits.Length);
        Assert.All(logits, x => Assert.True(float.IsFinite(x)));
    }

    [Fact]
    public void WinProbability_ForwardPass_ProducesProbability()
    {
        using var sessions = OnnxSessions.FromDirectory(TestPaths.ModelsDir());
        // No enriched features yet (Phase B) → trailing 86 dims are zero.
        var input = StateEncoder.EncodeWpBase(
            new[] { "Muradin", "Raynor", "Malfurion" },
            new[] { "Diablo", "Jaina" },
            "Cursed Hollow", "mid");

        var p = sessions.RunWinProbability(input);
        _out.WriteLine($"win_probability={p:F4}");

        Assert.True(float.IsFinite(p));
        Assert.InRange(p, 0f, 1f);
    }
}
