using HotsFever.DraftEngine.Data;

namespace HotsFever.DraftEngine.Models;

/// <summary>
/// Computes the 86 enriched WP features — a faithful port of
/// computeEnrichedFeatures in src/lib/draft/ai-inference.ts. Feature groups:
///   role_counts(18) + team_avg_wr(2) + map_delta(2) + pairwise_counters(2) +
///   pairwise_synergies(2) + counter_detail(50) + meta_strength(4) +
///   draft_diversity(2) + comp_wr(4).
/// Intermediate math is done in double (matching JS number semantics) and stored
/// as float32.
/// </summary>
public static class EnrichedFeatures
{
    public const int Dims = 86;
    private const int NumFineRoles = 9;

    // Fine-grained role per hero (must match training/shared.py HERO_ROLE_FINE):
    // 0=tank 1=bruiser 2=healer 3=ranged_aa 4=ranged_mage 5=melee_assassin 6=support_utility 7=varian 8=pusher
    private static readonly Dictionary<string, int> FineRole = new(StringComparer.Ordinal)
    {
        // Tanks
        ["Anub'arak"] = 0, ["Arthas"] = 0, ["Blaze"] = 0, ["Cho"] = 0, ["Diablo"] = 0, ["E.T.C."] = 0, ["Garrosh"] = 0,
        ["Johanna"] = 0, ["Mal'Ganis"] = 0, ["Mei"] = 0, ["Muradin"] = 0, ["Stitches"] = 0, ["Tyrael"] = 0,
        // Bruisers
        ["Artanis"] = 1, ["Chen"] = 1, ["Deathwing"] = 1, ["Dehaka"] = 1, ["D.Va"] = 1, ["Gazlowe"] = 1, ["Hogger"] = 1,
        ["Imperius"] = 1, ["Leoric"] = 1, ["Malthael"] = 1, ["Ragnaros"] = 1, ["Rexxar"] = 1, ["Sonya"] = 1, ["Thrall"] = 1,
        ["Xul"] = 1, ["Yrel"] = 1,
        // Healers
        ["Alexstrasza"] = 2, ["Ana"] = 2, ["Anduin"] = 2, ["Auriel"] = 2, ["Brightwing"] = 2, ["Deckard"] = 2,
        ["Kharazim"] = 2, ["Li Li"] = 2, ["Lt. Morales"] = 2, ["Lúcio"] = 2, ["Malfurion"] = 2, ["Rehgar"] = 2,
        ["Stukov"] = 2, ["Tyrande"] = 2, ["Uther"] = 2, ["Whitemane"] = 2,
        // Ranged AA
        ["Cassia"] = 3, ["Falstad"] = 3, ["Fenix"] = 3, ["Greymane"] = 3, ["Hanzo"] = 3, ["Lunara"] = 3, ["Raynor"] = 3,
        ["Sgt. Hammer"] = 3, ["Sylvanas"] = 3, ["Tracer"] = 3, ["Tychus"] = 3, ["Valla"] = 3, ["Zul'jin"] = 3,
        // Ranged Mage
        ["Chromie"] = 4, ["Gall"] = 4, ["Genji"] = 4, ["Gul'dan"] = 4, ["Jaina"] = 4, ["Junkrat"] = 4, ["Kael'thas"] = 4,
        ["Kel'Thuzad"] = 4, ["Li-Ming"] = 4, ["Mephisto"] = 4, ["Nova"] = 4, ["Orphea"] = 4, ["Probius"] = 4, ["Tassadar"] = 4,
        // Melee Assassin
        ["Alarak"] = 5, ["Illidan"] = 5, ["Kerrigan"] = 5, ["Maiev"] = 5, ["Qhira"] = 5, ["Samuro"] = 5,
        ["The Butcher"] = 5, ["Valeera"] = 5, ["Zeratul"] = 5,
        // Support Utility
        ["Abathur"] = 6, ["Medivh"] = 6, ["Zarya"] = 6,
        // Varian
        ["Varian"] = 7,
        // Pusher
        ["Azmodan"] = 8, ["Nazeebo"] = 8, ["Zagara"] = 8, ["Murky"] = 8, ["The Lost Vikings"] = 8,
    };

    // Fine role → HP official role (for comp lookup).
    private static readonly Dictionary<int, string> HpRoleMap = new()
    {
        [0] = "Tank", [1] = "Bruiser", [2] = "Healer", [3] = "Ranged Assassin",
        [4] = "Ranged Assassin", [5] = "Melee Assassin", [6] = "Support",
        [7] = "Bruiser", [8] = "Ranged Assassin",
    };

    /// <summary>Fine role index (0-8) for a hero, or -1 if unknown. Exposed for UI role labels.</summary>
    public static int FineRoleOf(string hero) => FineRole.TryGetValue(hero, out var r) ? r : -1;

    public static float[] Compute(IReadOnlyList<string> t0, IReadOnlyList<string> t1, string map, DraftData d)
    {
        var f = new float[Dims];
        int off = 0;

        double GetWR(string hero)
        {
            if (d.HeroMapWinRates.TryGetValue(map, out var byHero) &&
                byHero.TryGetValue(hero, out var md) && md.Games >= 50) return md.WinRate;
            return d.HeroStats.TryGetValue(hero, out var hs) ? hs.WinRate : 50;
        }
        double GetOverallWR(string hero) => d.HeroStats.TryGetValue(hero, out var hs) ? hs.WinRate : 50;

        // 1. role_counts (18 = 9 per team)
        foreach (var h in t0) { if (FineRole.TryGetValue(h, out var r)) f[off + r] += 1; }
        off += NumFineRoles;
        foreach (var h in t1) { if (FineRole.TryGetValue(h, out var r)) f[off + r] += 1; }
        off += NumFineRoles;

        // 2. team_avg_wr (2)
        var t0wrs = t0.Select(GetWR).ToList();
        var t1wrs = t1.Select(GetWR).ToList();
        f[off++] = (float)(t0wrs.Count > 0 ? t0wrs.Average() : 50);
        f[off++] = (float)(t1wrs.Count > 0 ? t1wrs.Average() : 50);

        // 3. map_delta (2)
        double t0MapDelta = 0, t1MapDelta = 0;
        foreach (var h in t0)
            if (d.HeroMapWinRates.TryGetValue(map, out var bh) && bh.TryGetValue(h, out var mw) && mw.Games >= 50)
                t0MapDelta += mw.WinRate - GetOverallWR(h);
        foreach (var h in t1)
            if (d.HeroMapWinRates.TryGetValue(map, out var bh) && bh.TryGetValue(h, out var mw) && mw.Games >= 50)
                t1MapDelta += mw.WinRate - GetOverallWR(h);
        f[off++] = (float)t0MapDelta;
        f[off++] = (float)t1MapDelta;

        // 4. pairwise_counters (2)
        double CounterDelta(IReadOnlyList<string> ourH, IReadOnlyList<string> theirH)
        {
            double sum = 0; int count = 0;
            foreach (var a in ourH)
                foreach (var b in theirH)
                {
                    if (!TryPair(d.Counters, a, b, out var dd) || dd.Games < 30) continue;
                    double expected = GetWR(a) + (100 - GetWR(b)) - 50;
                    sum += dd.WinRate - expected; count++;
                }
            return count > 0 ? sum / count : 0;
        }
        f[off++] = (float)CounterDelta(t0, t1);
        f[off++] = (float)CounterDelta(t1, t0);

        // 5. pairwise_synergies (2)
        double SynergyDelta(IReadOnlyList<string> heroes)
        {
            double sum = 0; int count = 0;
            for (int i = 0; i < heroes.Count; i++)
                for (int j = i + 1; j < heroes.Count; j++)
                {
                    if (!TryPair(d.Synergies, heroes[i], heroes[j], out var dd) || dd.Games < 30) continue;
                    double expected = 50 + (GetWR(heroes[i]) - 50) + (GetWR(heroes[j]) - 50);
                    sum += dd.WinRate - expected; count++;
                }
            return count > 0 ? sum / count : 0;
        }
        f[off++] = (float)SynergyDelta(t0);
        f[off++] = (float)SynergyDelta(t1);

        // 6. counter_detail (50 = 25 + 25) — all cross-team pairs, both directions, padded.
        foreach (var a in t0)
            foreach (var b in t1)
            {
                if (TryPair(d.Counters, a, b, out var dd) && dd.Games >= 30)
                    f[off] = (float)(dd.WinRate - (GetWR(a) + (100 - GetWR(b)) - 50));
                off++;
            }
        while (off < 18 + 2 + 2 + 2 + 2 + 25) off++; // pad to 51
        foreach (var b in t1)
            foreach (var a in t0)
            {
                if (TryPair(d.Counters, b, a, out var dd) && dd.Games >= 30)
                    f[off] = (float)(dd.WinRate - (GetWR(b) + (100 - GetWR(a)) - 50));
                off++;
            }
        while (off < 18 + 2 + 2 + 2 + 2 + 50) off++; // pad to 76

        // 7. meta_strength (4) — avg pick_rate & ban_rate per team
        double PickRate(string h) => d.HeroStats.TryGetValue(h, out var hs) ? hs.PickRate : 0;
        double BanRate(string h) => d.HeroStats.TryGetValue(h, out var hs) ? hs.BanRate : 0;
        f[off++] = (float)(t0.Sum(PickRate) / Math.Max(t0.Count, 1));
        f[off++] = (float)(t0.Sum(BanRate) / Math.Max(t0.Count, 1));
        f[off++] = (float)(t1.Sum(PickRate) / Math.Max(t1.Count, 1));
        f[off++] = (float)(t1.Sum(BanRate) / Math.Max(t1.Count, 1));

        // 8. draft_diversity (2) — std dev of hero WRs per team
        f[off++] = (float)StdDev(t0wrs);
        f[off++] = (float)StdDev(t1wrs);

        // 9. comp_wr (4) — composition win rate + log(games) per team
        var (t0cw, t0cg) = GetCompWR(t0, d);
        var (t1cw, t1cg) = GetCompWR(t1, d);
        f[off++] = (float)((t0cw - 50.0) / 10.0);
        f[off++] = (float)(Math.Log(1.0 + t0cg) / 15.0);
        f[off++] = (float)((t1cw - 50.0) / 10.0);
        f[off++] = (float)(Math.Log(1.0 + t1cg) / 15.0);

        return f;
    }

    private static bool TryPair(Dictionary<string, Dictionary<string, WinRateGames>> table,
        string a, string b, out WinRateGames val)
    {
        if (table.TryGetValue(a, out var inner) && inner.TryGetValue(b, out var v)) { val = v; return true; }
        val = null!;
        return false;
    }

    private static double StdDev(IReadOnlyList<double> vals)
    {
        if (vals.Count < 2) return 0;
        double mean = vals.Average();
        double sumSq = 0;
        foreach (var v in vals) sumSq += (v - mean) * (v - mean);
        return Math.Sqrt(sumSq / vals.Count);
    }

    private static (double WinRate, double Games) GetCompWR(IReadOnlyList<string> heroes, DraftData d)
    {
        if (heroes.Count == 0) return (33.0, 0);
        var hpRoles = heroes.Select(h =>
        {
            if (FineRole.TryGetValue(h, out var fr))
                return HpRoleMap.TryGetValue(fr, out var role) ? role : "Ranged Assassin";
            return "Ranged Assassin";
        }).OrderBy(s => s, StringComparer.Ordinal).ToList();
        string key = string.Join(",", hpRoles);

        foreach (var c in d.Compositions)
        {
            if (c.Roles.Count == hpRoles.Count &&
                string.Join(",", c.Roles.OrderBy(s => s, StringComparer.Ordinal)) == key)
                return (c.WinRate, c.Games);
        }
        return (33.0, 0);
    }
}
