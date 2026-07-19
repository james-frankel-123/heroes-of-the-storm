using HotsFever.DraftEngine.Encoding;
using Xunit;

namespace HotsFever.DraftEngine.Tests;

/// <summary>
/// Layer 1 parity: the encoding vocabulary and input-vector construction. These
/// are self-contained checks; the byte-for-byte comparison against the web
/// engine's vectors lands with the oracle harness (Phase C).
/// </summary>
public class EncodingTests
{
    [Fact]
    public void Catalog_HasExpectedCounts()
    {
        Assert.Equal(90, HeroCatalog.Heroes.Length);
        Assert.Equal(14, HeroCatalog.Maps.Length);
        Assert.Equal(3, HeroCatalog.SkillTiers.Length);
    }

    [Fact]
    public void Catalog_IsAlphabeticalAndDistinct()
    {
        Assert.Equal(HeroCatalog.Heroes.Length, HeroCatalog.Heroes.Distinct().Count());
        // First and last anchor the alphabetical ordering used at training time.
        Assert.Equal("Abathur", HeroCatalog.Heroes[0]);
        Assert.Equal("Zul'jin", HeroCatalog.Heroes[^1]);
    }

    [Fact]
    public void HeroIndex_KnownHeroes()
    {
        Assert.Equal(0, HeroCatalog.HeroIndex("Abathur"));
        Assert.Equal(5, HeroCatalog.HeroIndex("Anub'arak"));
        Assert.Equal(46, HeroCatalog.HeroIndex("Lúcio"));
        Assert.Equal(-1, HeroCatalog.HeroIndex("Not A Hero"));
    }

    [Fact]
    public void EncodeState_HasCorrectShapeAndMultiHot()
    {
        var s = new AiDraftState
        {
            Team0Picks = new[] { "Abathur", "Muradin" },
            Team1Picks = new[] { "Raynor" },
            Bans = new[] { "Nazeebo" },
            Map = "Cursed Hollow",
            Tier = "mid",
            Step = 6,
            IsPick = true,
            OurTeam = 0,
        };
        var v = StateEncoder.EncodeState(s);

        Assert.Equal(StateEncoder.PolicyDims, v.Length);
        // team0 multi-hot at Abathur(0), Muradin(54)
        Assert.Equal(1f, v[HeroCatalog.HeroIndex("Abathur")]);
        Assert.Equal(1f, v[HeroCatalog.HeroIndex("Muradin")]);
        // team1 block starts at 90
        Assert.Equal(1f, v[90 + HeroCatalog.HeroIndex("Raynor")]);
        // bans block starts at 180
        Assert.Equal(1f, v[180 + HeroCatalog.HeroIndex("Nazeebo")]);
        // map one-hot: 270 + mapIdx
        Assert.Equal(1f, v[270 + HeroCatalog.MapIndex("Cursed Hollow")]);
        // tier one-hot: 284 + tierIdx("mid"=1)
        Assert.Equal(1f, v[284 + HeroCatalog.TierIndex("mid")]);
        // scalars: step/15, pickFlag, ourTeam
        Assert.Equal(6f / 15f, v[287]);
        Assert.Equal(1f, v[288]);
        Assert.Equal(0f, v[289]);
    }

    [Fact]
    public void EncodeStateForGd_DropsOurTeamBit()
    {
        var s = new AiDraftState { Map = "Sky Temple", Tier = "high", Step = 3, IsPick = false, OurTeam = 1 };
        var full = StateEncoder.EncodeState(s);
        var gd = StateEncoder.EncodeStateForGd(s);

        Assert.Equal(StateEncoder.GdDims, gd.Length);
        for (int i = 0; i < StateEncoder.GdDims; i++) Assert.Equal(full[i], gd[i]);
    }

    [Fact]
    public void BuildValidMask_ZerosTakenHeroes()
    {
        var mask = StateEncoder.BuildValidMask(new[] { "Abathur", "Raynor" });
        Assert.Equal(90, mask.Length);
        Assert.Equal(0f, mask[HeroCatalog.HeroIndex("Abathur")]);
        Assert.Equal(0f, mask[HeroCatalog.HeroIndex("Raynor")]);
        Assert.Equal(1f, mask[HeroCatalog.HeroIndex("Muradin")]);
        Assert.Equal(88, mask.Count(x => x == 1f));
    }
}
