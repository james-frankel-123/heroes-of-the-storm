namespace HotsFever.DraftEngine.Encoding;

/// <summary>
/// The canonical encoding vocabulary — heroes, maps, and skill tiers — and the
/// index maps used to build model input tensors.
///
/// THIS IS THE MODEL CONTRACT. The ordering here must match training/shared.py
/// and the web engine (src/lib/draft/ai-inference.ts) byte-for-byte, or every
/// model input silently shifts. Heroes are alphabetically sorted exactly as in
/// training.
///
/// NOTE (Phase A follow-up): these lists are currently hand-ported. They should
/// be generated from a single shared source (training/shared.py) with a
/// build-time check that fails on drift — see tools/gen-catalog.
/// </summary>
public static class HeroCatalog
{
    /// <summary>
    /// Cho and Gall are two catalog entries but a single draft action — taking either
    /// consumes both. Mirrors expandChoGall in src/lib/draft/engine.ts, which the web
    /// applies when computing unavailable heroes. Without it the search can recommend
    /// a hero that is not actually available, and the board can hold a state the game
    /// cannot produce.
    /// </summary>
    public static IReadOnlyList<string> ExpandChoGall(IEnumerable<string> heroes)
    {
        var set = new HashSet<string>(heroes, StringComparer.Ordinal);
        if (set.Contains("Cho") || set.Contains("Gall"))
        {
            set.Add("Cho");
            set.Add("Gall");
        }
        return set.ToArray();
    }

    /// <summary>90 heroes, alphabetically sorted — must match training exactly.</summary>
    public static readonly string[] Heroes =
    {
        "Abathur", "Alarak", "Alexstrasza", "Ana", "Anduin", "Anub'arak", "Artanis",
        "Arthas", "Auriel", "Azmodan", "Blaze", "Brightwing", "Cassia", "Chen", "Cho",
        "Chromie", "D.Va", "Deathwing", "Deckard", "Dehaka", "Diablo", "E.T.C.",
        "Falstad", "Fenix", "Gall", "Garrosh", "Gazlowe", "Genji", "Greymane",
        "Gul'dan", "Hanzo", "Hogger", "Illidan", "Imperius", "Jaina", "Johanna",
        "Junkrat", "Kael'thas", "Kel'Thuzad", "Kerrigan", "Kharazim", "Leoric",
        "Li Li", "Li-Ming", "Lt. Morales", "Lunara", "Lúcio", "Maiev", "Mal'Ganis",
        "Malfurion", "Malthael", "Medivh", "Mei", "Mephisto", "Muradin", "Murky",
        "Nazeebo", "Nova", "Orphea", "Probius", "Qhira", "Ragnaros", "Raynor",
        "Rehgar", "Rexxar", "Samuro", "Sgt. Hammer", "Sonya", "Stitches", "Stukov",
        "Sylvanas", "Tassadar", "The Butcher", "The Lost Vikings", "Thrall", "Tracer",
        "Tychus", "Tyrael", "Tyrande", "Uther", "Valeera", "Valla", "Varian",
        "Whitemane", "Xul", "Yrel", "Zagara", "Zarya", "Zeratul", "Zul'jin",
    };

    /// <summary>14 maps — must match training/shared.py.</summary>
    public static readonly string[] Maps =
    {
        "Alterac Pass", "Battlefield of Eternity", "Blackheart's Bay",
        "Braxis Holdout", "Cursed Hollow", "Dragon Shire",
        "Garden of Terror", "Hanamura Temple", "Infernal Shrines",
        "Sky Temple", "Tomb of the Spider Queen", "Towers of Doom",
        "Volskaya Foundry", "Warhead Junction",
    };

    /// <summary>3 skill tiers.</summary>
    public static readonly string[] SkillTiers = { "low", "mid", "high" };

    public const int NumHeroes = 90;
    public const int NumMaps = 14;
    public const int NumTiers = 3;

    private static readonly Dictionary<string, int> HeroToIdx = BuildIndex(Heroes);
    private static readonly Dictionary<string, int> MapToIdx = BuildIndex(Maps);
    private static readonly Dictionary<string, int> TierToIdx = BuildIndex(SkillTiers);

    private static Dictionary<string, int> BuildIndex(string[] items)
    {
        var d = new Dictionary<string, int>(items.Length, StringComparer.Ordinal);
        for (int i = 0; i < items.Length; i++) d[items[i]] = i;
        return d;
    }

    /// <summary>Hero index, or -1 if unknown (mirrors the TS `!== undefined` guard).</summary>
    public static int HeroIndex(string hero) => HeroToIdx.TryGetValue(hero, out var i) ? i : -1;
    public static int MapIndex(string map) => MapToIdx.TryGetValue(map, out var i) ? i : -1;
    public static int TierIndex(string tier) => TierToIdx.TryGetValue(tier, out var i) ? i : -1;

    static HeroCatalog()
    {
        // Cheap invariant guards so a bad edit fails loudly at startup.
        if (Heroes.Length != NumHeroes) throw new InvalidOperationException($"Heroes count {Heroes.Length} != {NumHeroes}");
        if (Maps.Length != NumMaps) throw new InvalidOperationException($"Maps count {Maps.Length} != {NumMaps}");
        if (SkillTiers.Length != NumTiers) throw new InvalidOperationException($"Tiers count {SkillTiers.Length} != {NumTiers}");
    }
}
