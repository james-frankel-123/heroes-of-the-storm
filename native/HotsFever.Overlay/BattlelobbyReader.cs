using Heroes.StormReplayParser;
using Heroes.StormReplayParser.Replay;

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
/// Reads the game's replay.server.battlelobby (written at the loading screen)
/// into real battletags + teams + picks/bans, fully in-process via the
/// Heroes.StormReplayParser library — no external tool, no subprocess, no
/// machine-specific paths, so it ships and runs anywhere. Copies the file first
/// (never parses the live original the game may hold). Picks/bans arrive as
/// 4-char attribute IDs and are mapped to catalog names via HeroAttributeIds.
/// </summary>
public static class BattlelobbyReader
{
    /// <summary>
    /// The local machine's HotS toon IDs (e.g. "1-Hero-1-11154214"), read from
    /// Documents\Heroes of the Storm\Accounts\&lt;acct&gt;\&lt;toon&gt;\. A lobby player whose
    /// ToonId is in this set is definitively a local account — the reliable way
    /// to find "our" team when the user has multiple accounts.
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
            // Copy first, then parse the copy — never lock the live original.
            var copy = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "hf_bl.battlelobby");
            System.IO.File.Copy(battlelobbyPath, copy, true);

            var result = StormReplayPregame.Parse(copy);
            if (result.Status != StormReplayPregameParseStatus.Success) return null;
            var pg = result.ReplayBattleLobby;

            var roster = pg.StormPlayers.ToList();
            if (roster.Count < 10) return null;

            var players = new List<LobbyPlayer>();
            for (int i = 0; i < roster.Count; i++)
            {
                var p = roster[i];
                if (string.IsNullOrEmpty(p.BattleTagName)) continue;
                var t = p.ToonHandle;
                string toonId = t != null ? $"{t.Region}-Hero-{t.Realm}-{t.Id}" : "";
                players.Add(new LobbyPlayer
                {
                    Battletag = p.BattleTagName,
                    ToonId = toonId,
                    Team = i < 5 ? 0 : 1,
                    Hero = HeroAttributeIds.Resolve(p.PlayerHero?.HeroAttributeId) ?? "",
                });
            }

            List<string> ResolveBans(StormTeam team) => pg.GetTeamBans(team)
                .Select(HeroAttributeIds.Resolve)
                .Where(n => n != null).Select(n => n!).ToList();

            return new LobbyInfo
            {
                Map = pg.MapTitle ?? "",
                GameMode = pg.GameMode.ToString(),
                Players = players,
                BansBlue = ResolveBans(StormTeam.Blue),
                BansRed = ResolveBans(StormTeam.Red),
            };
        }
        catch { return null; }
    }
}
