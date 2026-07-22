namespace HotsFever.Overlay;

/// <summary>
/// Maps Blizzard's 4-char hero attribute IDs (as they appear in the pregame
/// battlelobby: Players[].PlayerHero.HeroAttributeId and TeamBans) to the
/// canonical HeroCatalog names used by the draft engine.
///
/// Generated from HeroesToolChest/heroes-data (build 93640): each hero's
/// "attributeId" matched to our catalog via hyperlinkId/name (accent-folded,
/// punctuation-stripped). All 90 catalog heroes covered, attIds unique.
/// The 4-char codes are stable Blizzard identifiers; regenerate only if a new
/// hero ships. Codes are case-sensitive (e.g. "STUK", "DVA0", "L90E").
/// </summary>
public static class HeroAttributeIds
{
    private static readonly Dictionary<string, string> AttIdToHero = new(StringComparer.Ordinal)
    {
        ["Abat"] = "Abathur",
        ["Alar"] = "Alarak",
        ["Alex"] = "Alexstrasza",
        ["Amaz"] = "Cassia",
        ["Andu"] = "Anduin",
        ["Anub"] = "Anub'arak",
        ["Arth"] = "Arthas",
        ["Arts"] = "Artanis",
        ["Auri"] = "Auriel",
        ["Azmo"] = "Azmodan",
        ["Barb"] = "Sonya",
        ["Butc"] = "The Butcher",
        ["CCho"] = "Cho",
        ["Chen"] = "Chen",
        ["Chro"] = "Chromie",
        ["Crus"] = "Johanna",
        ["DEAT"] = "Deathwing",
        ["DECK"] = "Deckard",
        ["Deha"] = "Dehaka",
        ["Demo"] = "Valla",
        ["Diab"] = "Diablo",
        ["Drya"] = "Lunara",
        ["DVA0"] = "D.Va",
        ["Faer"] = "Brightwing",
        ["Fals"] = "Falstad",
        ["FENX"] = "Fenix",
        ["Fire"] = "Blaze",
        ["Gall"] = "Gall",
        ["Garr"] = "Garrosh",
        ["Genj"] = "Genji",
        ["Genn"] = "Greymane",
        ["Guld"] = "Gul'dan",
        ["HANA"] = "Ana",
        ["Hanz"] = "Hanzo",
        ["HMEI"] = "Mei",
        ["HOGG"] = "Hogger",
        ["Illi"] = "Illidan",
        ["IMPE"] = "Imperius",
        ["Jain"] = "Jaina",
        ["Junk"] = "Junkrat",
        ["Kael"] = "Kael'thas",
        ["KelT"] = "Kel'Thuzad",
        ["Kerr"] = "Kerrigan",
        ["L90E"] = "E.T.C.",
        ["Leor"] = "Leoric",
        ["LiLi"] = "Li Li",
        ["Lost"] = "The Lost Vikings",
        ["Luci"] = "Lúcio",
        ["Maie"] = "Maiev",
        ["Malf"] = "Malfurion",
        ["MalG"] = "Mal'Ganis",
        ["MALT"] = "Malthael",
        ["Mdvh"] = "Medivh",
        ["Medi"] = "Lt. Morales",
        ["MEPH"] = "Mephisto",
        ["Monk"] = "Kharazim",
        ["Mura"] = "Muradin",
        ["Murk"] = "Murky",
        ["Necr"] = "Xul",
        ["Nova"] = "Nova",
        ["NXHU"] = "Qhira",
        ["ORPH"] = "Orphea",
        ["Prob"] = "Probius",
        ["Ragn"] = "Ragnaros",
        ["Rayn"] = "Raynor",
        ["Rehg"] = "Rehgar",
        ["Rexx"] = "Rexxar",
        ["Samu"] = "Samuro",
        ["Sgth"] = "Sgt. Hammer",
        ["Stit"] = "Stitches",
        ["STUK"] = "Stukov",
        ["Sylv"] = "Sylvanas",
        ["Tass"] = "Tassadar",
        ["Thra"] = "Thrall",
        ["Tink"] = "Gazlowe",
        ["Tra0"] = "Tracer",
        ["Tych"] = "Tychus",
        ["Tyrd"] = "Tyrande",
        ["Tyrl"] = "Tyrael",
        ["Uthe"] = "Uther",
        ["VALE"] = "Valeera",
        ["Vari"] = "Varian",
        ["WHIT"] = "Whitemane",
        ["Witc"] = "Nazeebo",
        ["Wiza"] = "Li-Ming",
        ["YREL"] = "Yrel",
        ["Zaga"] = "Zagara",
        ["Zary"] = "Zarya",
        ["Zera"] = "Zeratul",
        ["ZULJ"] = "Zul'jin",
    };

    /// <summary>Canonical hero name for a 4-char attribute ID, or null if unknown/empty.</summary>
    public static string? Resolve(string? attId)
        => !string.IsNullOrEmpty(attId) && AttIdToHero.TryGetValue(attId, out var name) ? name : null;
}
