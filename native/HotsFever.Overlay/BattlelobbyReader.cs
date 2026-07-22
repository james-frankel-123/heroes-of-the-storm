using System.Diagnostics;
using System.Text.Json;

namespace HotsFever.Overlay;

public sealed class LobbyPlayer
{
    public string Battletag { get; set; } = "";
    public string ToonId { get; set; } = "";
    public int Team { get; set; }      // 0 = blue, 1 = red (by roster index)
    public string Hero { get; set; } = ""; // resolved canonical name, or "" if unresolved
}

public sealed class LobbyInfo
{
    public string Map { get; set; } = "";
    public string GameMode { get; set; } = "";
    public List<LobbyPlayer> Players { get; set; } = new();
    public List<string> BansBlue { get; set; } = new(); // resolved hero names
    public List<string> BansRed { get; set; } = new();
}

/// <summary>
/// Reads the game's replay.server.battlelobby at the loading screen (M0-verified)
/// into real battletags + teams by shelling HeroesDecode's get-pregame-json.
/// Copies the file first (never parses the live original). Picks/bans (4-char
/// attribute IDs) are ignored here — battletags + teams need no hero mapping.
/// </summary>
public static class BattlelobbyReader
{
    private static readonly JsonSerializerOptions Opts = new() { PropertyNameCaseInsensitive = true };

    private sealed class Root
    {
        public MapInfoJson? MapInfo { get; set; }
        public string? GameMode { get; set; }
        public List<PlayerJson>? Players { get; set; }
        public TeamBansJson? TeamBans { get; set; }
    }
    private sealed class MapInfoJson { public string? MapTitle { get; set; } }
    private sealed class PlayerJson
    {
        public string? BattleTagName { get; set; }
        public string? PlayerToonId { get; set; }
        public PlayerHeroJson? PlayerHero { get; set; }
    }
    private sealed class PlayerHeroJson { public string? HeroAttributeId { get; set; } }
    private sealed class TeamBansJson { public List<string>? blue { get; set; } public List<string>? red { get; set; } }

    /// <summary>
    /// The local machine's HotS toon IDs (e.g. "1-Hero-1-11154214"), read from
    /// Documents\Heroes of the Storm\Accounts\&lt;acct&gt;\&lt;toon&gt;\. A lobby player whose
    /// PlayerToonId is in this set is definitively a local account — the reliable
    /// way to find "our" team when the user has multiple accounts.
    /// </summary>
    public static HashSet<string> GetLocalToonIds()
    {
        var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        try
        {
            var acct = System.IO.Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                "Heroes of the Storm", "Accounts");
            if (!System.IO.Directory.Exists(acct)) return set;
            foreach (var accDir in System.IO.Directory.EnumerateDirectories(acct))
                foreach (var toonDir in System.IO.Directory.EnumerateDirectories(accDir))
                {
                    var name = System.IO.Path.GetFileName(toonDir);
                    if (System.Text.RegularExpressions.Regex.IsMatch(name, @"^\d+-Hero-\d+-\d+$"))
                        set.Add(name);
                }
        }
        catch { }
        return set;
    }

    public static LobbyInfo? Read(string battlelobbyPath)
    {
        try
        {
            // CAVEAT (ship blocker): HeroesDecode is invoked from the current dev
            // machine's user-local .NET tool path, with DOTNET_ROOT hardcoded below.
            // Before shipping to other machines we must bundle HeroesDecode (or its
            // parsing logic) with the app and stop assuming this layout. Tracked in
            // the hotsfever-project memory and the roadmap "caveats" note.
            var hd = System.IO.Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".dotnet", "tools", "dotnet-heroes-decode.exe");
            if (!System.IO.File.Exists(hd)) return null;

            var copy = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "hf_bl.battlelobby");
            System.IO.File.Copy(battlelobbyPath, copy, true);
            var outDir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "hf_bljson");
            if (System.IO.Directory.Exists(outDir)) System.IO.Directory.Delete(outDir, true);
            System.IO.Directory.CreateDirectory(outDir);

            var psi = new ProcessStartInfo
            {
                FileName = hd,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            psi.ArgumentList.Add("get-pregame-json");
            psi.ArgumentList.Add("-p"); psi.ArgumentList.Add(copy);
            psi.ArgumentList.Add("-o"); psi.ArgumentList.Add(outDir);
            psi.ArgumentList.Add("--no-json-display");
            psi.Environment["DOTNET_ROOT"] = System.IO.Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Microsoft", "dotnet");

            using var proc = Process.Start(psi);
            if (proc == null) return null;
            if (!proc.WaitForExit(20000)) { try { proc.Kill(); } catch { } return null; }

            var jsonFile = System.IO.Directory.EnumerateFiles(outDir, "*.json").FirstOrDefault();
            if (jsonFile == null) return null;

            var root = JsonSerializer.Deserialize<Root>(System.IO.File.ReadAllText(jsonFile), Opts);
            if (root?.Players == null || root.Players.Count < 10) return null;

            var players = new List<LobbyPlayer>();
            for (int i = 0; i < root.Players.Count; i++)
            {
                var bt = root.Players[i].BattleTagName;
                if (string.IsNullOrEmpty(bt)) continue;
                players.Add(new LobbyPlayer
                {
                    Battletag = bt,
                    ToonId = root.Players[i].PlayerToonId ?? "",
                    Team = i < 5 ? 0 : 1,
                    Hero = HeroAttributeIds.Resolve(root.Players[i].PlayerHero?.HeroAttributeId) ?? "",
                });
            }
            static List<string> Resolve(List<string>? ids) =>
                (ids ?? new()).Select(HeroAttributeIds.Resolve).Where(n => n != null).Select(n => n!).ToList();
            return new LobbyInfo
            {
                Map = root.MapInfo?.MapTitle ?? "",
                GameMode = root.GameMode ?? "",
                Players = players,
                BansBlue = Resolve(root.TeamBans?.blue),
                BansRed = Resolve(root.TeamBans?.red),
            };
        }
        catch { return null; }
    }
}
