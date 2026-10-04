/**
 * Model input encoding shared by every browser model (draft policy, generic
 * draft, WP, partial WP). Must match training/shared.py with
 * HOTS_HERO_SET=v2: the original 90 heroes and 14 maps in their original
 * (alphabetical) order, then heroes and maps added later APPENDED, so an old
 * index never changes meaning. Never re-sort these lists.
 */
export const HEROES: readonly string[] = [
  "Abathur","Alarak","Alexstrasza","Ana","Anduin","Anub'arak","Artanis",
  "Arthas","Auriel","Azmodan","Blaze","Brightwing","Cassia","Chen","Cho",
  "Chromie","D.Va","Deathwing","Deckard","Dehaka","Diablo","E.T.C.",
  "Falstad","Fenix","Gall","Garrosh","Gazlowe","Genji","Greymane",
  "Gul'dan","Hanzo","Hogger","Illidan","Imperius","Jaina","Johanna",
  "Junkrat","Kael'thas","Kel'Thuzad","Kerrigan","Kharazim","Leoric",
  "Li Li","Li-Ming","Lt. Morales","Lunara","Lúcio","Maiev","Mal'Ganis",
  "Malfurion","Malthael","Medivh","Mei","Mephisto","Muradin","Murky",
  "Nazeebo","Nova","Orphea","Probius","Qhira","Ragnaros","Raynor",
  "Rehgar","Rexxar","Samuro","Sgt. Hammer","Sonya","Stitches","Stukov",
  "Sylvanas","Tassadar","The Butcher","The Lost Vikings","Thrall","Tracer",
  "Tychus","Tyrael","Tyrande","Uther","Valeera","Valla","Varian",
  "Whitemane","Xul","Yrel","Zagara","Zarya","Zeratul","Zul'jin",
  // appended (patch 2.57)
  "Xal'atath",
]

export const MAPS: readonly string[] = [
  "Alterac Pass", "Battlefield of Eternity", "Blackheart's Bay",
  "Braxis Holdout", "Cursed Hollow", "Dragon Shire",
  "Garden of Terror", "Hanamura Temple", "Infernal Shrines",
  "Sky Temple", "Tomb of the Spider Queen", "Towers of Doom",
  "Volskaya Foundry", "Warhead Junction",
  // appended (ranked rotation since 2026-07-20)
  "Haunted Mines",
]

export const SKILL_TIERS: readonly string[] = ["low", "mid", "high"]

export const NUM_HEROES = HEROES.length // 91
export const NUM_MAPS = MAPS.length // 15
export const NUM_TIERS = SKILL_TIERS.length // 3
/** Policy state: t0 + t1 + bans + map + tier + step + is_pick + our_team */
export const STATE_DIM = NUM_HEROES * 3 + NUM_MAPS + NUM_TIERS + 2 + 1 // 294
/** WP inputs: t0 + t1 + map + tier, then the 86 enriched features */
export const WP_BASE_DIM = NUM_HEROES * 2 + NUM_MAPS + NUM_TIERS // 200
export const ENRICHED_DIM = 86
export const WP_INPUT_DIM = WP_BASE_DIM + ENRICHED_DIM // 286

function indexOf(list: readonly string[]): Record<string, number> {
  const out: Record<string, number> = {}
  list.forEach((x, i) => { out[x] = i })
  return out
}
export const HERO_TO_IDX = indexOf(HEROES)
export const MAP_TO_IDX = indexOf(MAPS)
export const TIER_TO_IDX = indexOf(SKILL_TIERS)
