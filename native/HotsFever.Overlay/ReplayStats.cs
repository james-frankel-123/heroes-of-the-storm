using System.Collections.Concurrent;
using System.Text.Json;
using Heroes.ReplayParser;
using HotsFever.DraftEngine.Data;
using HotsFever.DraftEngine.Encoding;

namespace HotsFever.Overlay;

/// <summary>
/// Scans ALL local .StormReplay files into personal per-hero stats matching the
/// website: Storm-League-only, with raw Win% AND the momentum-adjusted MAWP
/// (recency + time decay + confidence padding, ported from src/lib/mawp.ts).
/// Cached per-replay so each file is parsed once. Client-only (M3).
/// </summary>
public sealed class ReplayScanResult
{
    public Dictionary<string, Dictionary<string, PlayerHeroStat>> PlayerStats { get; } = new();
    public string? LocalBattletag { get; set; }
    public int ReplaysParsed { get; set; }   // Storm League replays represented
    public int NewlyParsed { get; set; }
}

public sealed class ReplayRec
{
    public string Bt { get; set; } = "";
    public string Hero { get; set; } = "";
    public bool Won { get; set; }
}

/// <summary>Cached per-replay: the match date + each player's result. Empty Recs = parsed but not Storm League.</summary>
public sealed class ReplayEntry
{
    public long Date { get; set; } // match timestamp, UTC ticks
    public List<ReplayRec> Recs { get; set; } = new();
}

public static class ReplayStats
{
    private static readonly Dictionary<string, string> BySlug = BuildSlugMap();

    // MAWP decay constants (mirror src/lib/mawp.ts).
    private const double Ln2 = 0.69314718055994531;
    private const double LambdaG = Ln2 / 30.0; // game-count half-life 30 games
    private const double LambdaT = Ln2 / 90.0; // time half-life 90 days

    public static string DefaultCachePath() => System.IO.Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "HotsFever", "replay-cache-v2.json");

    public static ReplayScanResult ScanCached(string replaysDir, string cachePath, Action<int, int>? progress = null)
    {
        var cache = LoadCache(cachePath);

        var all = System.IO.Directory.EnumerateFiles(replaysDir, "*.StormReplay", System.IO.SearchOption.AllDirectories).ToList();
        var newFiles = all.Where(f => !cache.ContainsKey(f)).ToList();

        int done = 0, total = newFiles.Count;
        var parsed = new ConcurrentDictionary<string, ReplayEntry>();
        Parallel.ForEach(newFiles, new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount - 1) }, f =>
        {
            var entry = ParseFile(f);
            if (entry != null) parsed[f] = entry;
            int d = System.Threading.Interlocked.Increment(ref done);
            if (d % 20 == 0 || d == total) progress?.Invoke(d, total);
        });

        foreach (var kv in parsed) cache[kv.Key] = kv.Value;
        if (parsed.Count > 0) SaveCache(cachePath, cache);

        var result = Aggregate(cache, DateTime.UtcNow);
        result.NewlyParsed = parsed.Count;
        return result;
    }

    private static ReplayEntry? ParseFile(string path)
    {
        try
        {
            var parsed = DataParser.ParseReplay(path, false, new ParseOptions());
            if (parsed.Item1 != DataParser.ReplayParseResult.Success || parsed.Item2 == null) return null;
            var replay = parsed.Item2;

            var entry = new ReplayEntry { Date = replay.Timestamp.Ticks };
            // Storm League only (matches the website's fetch filter).
            if (replay.GameMode != GameMode.StormLeague || replay.Players == null) return entry; // cached as empty

            foreach (var p in replay.Players)
            {
                if (p == null || string.IsNullOrEmpty(p.Character)) continue;
                if (p.Team != 0 && p.Team != 1) continue;
                string bt = p.BattleTag > 0 ? $"{p.Name}#{p.BattleTag}" : p.Name ?? "";
                if (bt.Length == 0) continue;
                entry.Recs.Add(new ReplayRec { Bt = bt, Hero = MapHero(p.Character), Won = p.IsWinner });
            }
            return entry;
        }
        catch { return null; }
    }

    private static ReplayScanResult Aggregate(Dictionary<string, ReplayEntry> cache, DateTime now)
    {
        var appearances = new Dictionary<string, int>(StringComparer.Ordinal);
        // (bt -> hero -> list of (won, dateTicks))
        var matches = new Dictionary<string, Dictionary<string, List<(bool won, long date)>>>(StringComparer.Ordinal);
        int slReplays = 0;

        foreach (var entry in cache.Values)
        {
            if (entry.Recs.Count == 0) continue; // non-Storm-League
            slReplays++;
            foreach (var r in entry.Recs)
            {
                appearances[r.Bt] = appearances.GetValueOrDefault(r.Bt) + 1;
                if (!matches.TryGetValue(r.Bt, out var byHero)) { byHero = new(); matches[r.Bt] = byHero; }
                if (!byHero.TryGetValue(r.Hero, out var list)) { list = new(); byHero[r.Hero] = list; }
                list.Add((r.Won, entry.Date));
            }
        }

        var result = new ReplayScanResult { ReplaysParsed = slReplays };
        foreach (var (bt, byHero) in matches)
        {
            var d = new Dictionary<string, PlayerHeroStat>(StringComparer.Ordinal);
            foreach (var (hero, list) in byHero)
            {
                int games = list.Count;
                int wins = list.Count(m => m.won);
                d[hero] = new PlayerHeroStat
                {
                    Games = games,
                    Wins = wins,
                    WinRate = games > 0 ? 100.0 * wins / games : 0,
                    Mawp = ComputeMawp(list, now) * 100.0,
                };
            }
            result.PlayerStats[bt] = d;
        }

        result.LocalBattletag = appearances.OrderByDescending(kv => kv.Value).Select(kv => kv.Key).FirstOrDefault();
        return result;
    }

    /// <summary>Momentum-adjusted win fraction [0,1] — exact port of computeMAWP (src/lib/mawp.ts).</summary>
    private static double ComputeMawp(List<(bool won, long date)> matches, DateTime now)
    {
        if (matches.Count == 0) return 0.5;
        var sorted = matches.OrderByDescending(m => m.date).ToList(); // newest first
        double weightedSum = 0, weightSum = 0;
        for (int i = 0; i < sorted.Count; i++)
        {
            int rank = i + 1;
            double wGames = rank <= 30 ? 1.0 : Math.Exp(-LambdaG * (rank - 30));
            double days = (now - new DateTime(sorted[i].date, DateTimeKind.Utc)).TotalDays;
            if (days < 0) days = 0;
            double wTime = days <= 180 ? 1.0 : Math.Exp(-LambdaT * (days - 180));
            double outcome = sorted[i].won ? 1.0 : 0.0;
            double effectiveOutcome = outcome * wTime + 0.5 * (1 - wTime);
            weightedSum += wGames * effectiveOutcome;
            weightSum += wGames;
        }
        int n = sorted.Count;
        if (n < 30) { double phantom = 30 - n; weightedSum += phantom * 0.5; weightSum += phantom; }
        return weightSum > 0 ? weightedSum / weightSum : 0.5;
    }

    private static Dictionary<string, ReplayEntry> LoadCache(string path)
    {
        try
        {
            if (System.IO.File.Exists(path))
                return JsonSerializer.Deserialize<Dictionary<string, ReplayEntry>>(System.IO.File.ReadAllText(path)) ?? new();
        }
        catch { }
        return new();
    }

    private static void SaveCache(string path, Dictionary<string, ReplayEntry> cache)
    {
        try
        {
            System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(path)!);
            System.IO.File.WriteAllText(path, JsonSerializer.Serialize(cache));
        }
        catch { }
    }

    private static string MapHero(string character)
        => BySlug.TryGetValue(Slug(character), out var name) ? name : character;

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
            if (System.Globalization.CharUnicodeInfo.GetUnicodeCategory(c) == System.Globalization.UnicodeCategory.NonSpacingMark) continue;
            if (char.IsLetterOrDigit(c)) sb.Append(char.ToLowerInvariant(c));
        }
        return sb.ToString();
    }
}
