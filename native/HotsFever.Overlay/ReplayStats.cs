using Heroes.ReplayParser;
using HotsFever.DraftEngine.Data;
using HotsFever.DraftEngine.Encoding;

namespace HotsFever.Overlay;

/// <summary>
/// Scans local .StormReplay files into personal per-hero stats (the "historic
/// data") that drive MAWP personalization. Runtime-only client concern (M3) —
/// not part of the ported web engine.
/// </summary>
public sealed class ReplayScanResult
{
    public Dictionary<string, Dictionary<string, PlayerHeroStat>> PlayerStats { get; } = new();
    public string? LocalBattletag { get; set; }
    public int ReplaysParsed { get; set; }
}

public static class ReplayStats
{
    // Catalog hero by normalized slug (lowercase, diacritics stripped, non-alnum removed),
    // so replay hero names line up with the engine's roster.
    private static readonly Dictionary<string, string> BySlug = BuildSlugMap();

    public static ReplayScanResult Scan(string replaysDir, int maxReplays = 150)
    {
        var result = new ReplayScanResult();
        var appearances = new Dictionary<string, int>(StringComparer.Ordinal);
        var acc = new Dictionary<string, Dictionary<string, (int games, int wins)>>(StringComparer.Ordinal);

        var files = Directory.EnumerateFiles(replaysDir, "*.StormReplay", SearchOption.AllDirectories)
            .OrderByDescending(File.GetLastWriteTimeUtc)
            .Take(maxReplays);

        foreach (var f in files)
        {
            Replay? replay = null;
            try
            {
                var parsed = DataParser.ParseReplay(f, false, new ParseOptions());
                if (parsed.Item1 == DataParser.ReplayParseResult.Success) replay = parsed.Item2;
            }
            catch { }
            if (replay?.Players == null) continue;
            result.ReplaysParsed++;

            foreach (var p in replay.Players)
            {
                if (p == null || string.IsNullOrEmpty(p.Character)) continue;
                if (p.Team != 0 && p.Team != 1) continue; // skip observers

                string bt = p.BattleTag > 0 ? $"{p.Name}#{p.BattleTag}" : p.Name ?? "";
                if (bt.Length == 0) continue;
                string hero = MapHero(p.Character);

                appearances[bt] = appearances.GetValueOrDefault(bt) + 1;
                if (!acc.TryGetValue(bt, out var byHero)) { byHero = new(); acc[bt] = byHero; }
                var cur = byHero.GetValueOrDefault(hero);
                byHero[hero] = (cur.games + 1, cur.wins + (p.IsWinner ? 1 : 0));
            }
        }

        foreach (var (bt, byHero) in acc)
        {
            var d = new Dictionary<string, PlayerHeroStat>(StringComparer.Ordinal);
            foreach (var (hero, gw) in byHero)
                d[hero] = new PlayerHeroStat
                {
                    Games = gw.games,
                    Wins = gw.wins,
                    WinRate = gw.games > 0 ? 100.0 * gw.wins / gw.games : 0,
                    Mawp = null,
                };
            result.PlayerStats[bt] = d;
        }

        // The local player appears in every replay → most-frequent battletag.
        result.LocalBattletag = appearances.OrderByDescending(kv => kv.Value).Select(kv => kv.Key).FirstOrDefault();
        return result;
    }

    private static string MapHero(string character)
    {
        if (BySlug.TryGetValue(Slug(character), out var name)) return name;
        return character;
    }

    private static Dictionary<string, string> BuildSlugMap()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var h in HeroCatalog.Heroes) map[Slug(h)] = h;
        return map;
    }

    private static string Slug(string s)
    {
        var norm = s.Normalize(System.Text.NormalizationForm.FormD);
        var sb = new System.Text.StringBuilder();
        foreach (var c in norm)
        {
            var cat = System.Globalization.CharUnicodeInfo.GetUnicodeCategory(c);
            if (cat == System.Globalization.UnicodeCategory.NonSpacingMark) continue;
            if (char.IsLetterOrDigit(c)) sb.Append(char.ToLowerInvariant(c));
        }
        return sb.ToString();
    }
}
