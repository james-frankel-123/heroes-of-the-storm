/**
 * The v1 /replays listing dropped avg_mmr and league_tier, which the old
 * Replay/Min_id rows carried and our queues store. Both are recoverable:
 *
 *  - avg_mmr was exactly the mean of the players' player_mmr (checked on
 *    4,998 ranked games, 2026-09-30: r = 1.0, mean abs error 6e-5).
 *  - league_tier was the game's tier from avg_mmr, encoded one above the v1
 *    tier ids (Bronze = 2 ... Diamond = 6), with Master stored as null in
 *    Storm League listings and 0 in Quick Match listings. Tiers come from
 *    v1 /mmr/tier (per game type thresholds), cached per whole MMR.
 *
 * Reproducing the old encoding keeps v1-era rows consistent with the rest
 * of replay_fetch_queue / replay_draft_data / qm_games.
 */
import { HeroesProfileApiV2 } from './api-client-v2'
import { isPlayerEntry } from './player-store'

type ListingGameType = 'sl' | 'qm'

const V1_TIER_IDS: Record<string, number> = {
  wood: 0, bronze: 1, silver: 2, gold: 3, platinum: 4, diamond: 5,
}

/** Old listing encoding for a v1 tier name such as "Diamond 3" or "Master". */
export function legacyListingTier(tierName: string, gameType: ListingGameType): number | null {
  const base = tierName.toLowerCase().replace(/\s*\d+$/, '').trim()
  if (base === 'master' || base === 'grand master' || base === 'grandmaster') {
    return gameType === 'qm' ? 0 : null
  }
  const id = V1_TIER_IDS[base]
  return id === undefined ? null : id + 1
}

/** Mean player_mmr over the player entries of a legacy-shaped replay. */
export function avgPlayerMmr(replay: Record<string, any>): number | null {
  const mmrs = Object.values(replay)
    .filter(isPlayerEntry)
    .map(p => Number(p.player_mmr))
    .filter(Number.isFinite)
  return mmrs.length > 0 ? mmrs.reduce((a, b) => a + b, 0) / mmrs.length : null
}

const tierCache = new Map<string, string>()

export async function deriveListingFields(
  api: HeroesProfileApiV2,
  replay: Record<string, any>,
  gameType: ListingGameType,
): Promise<{ avgMmr: number | null; leagueTier: number | null }> {
  const avgMmr = avgPlayerMmr(replay)
  if (avgMmr === null) return { avgMmr: null, leagueTier: null }
  const key = `${gameType}:${Math.floor(avgMmr)}`
  let name = tierCache.get(key)
  if (name === undefined) {
    const d: any = await api.fetch('mmr/tier', { game_type: gameType, mmr: String(Math.floor(avgMmr)) })
    name = String(d.tier ?? '')
    tierCache.set(key, name)
  }
  return { avgMmr, leagueTier: legacyListingTier(name, gameType) }
}
