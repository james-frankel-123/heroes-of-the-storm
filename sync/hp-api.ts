/**
 * Heroes Profile client factory: one switch for the v1 migration.
 *
 *   HP_API=v2  -> new API (heroesprofile.com/api/external/v1, Bearer auth,
 *                 HEROES_PROFILE_V2_API_KEY). Legacy-shaped responses via
 *                 sync/api-client-v2.ts adapters.
 *   otherwise  -> old API (api.heroesprofile.com, retires 2027-01-01).
 *
 * Cutover plan (sync/docs/hp-v1-migration-notes.md): after the dense-window
 * backfill completes and live data is activated on the account, export
 * HP_API=v2 in sync/run-backfills.sh and the daily sync entrypoints.
 * IMPORTANT: never store v2 responses while api.lastDataSource === 'fixture'.
 */
import { HeroesProfileApi } from './api-client'
import { HeroesProfileApiV2 } from './api-client-v2'

export type HpApi = HeroesProfileApi | HeroesProfileApiV2

export function isV2(): boolean {
  return process.env.HP_API === 'v2'
}

export function createHpApi(
  which: 'key1' | 'key2',
  maxCallsPerMinute: number,
  maxRetries = 3,
): HpApi {
  if (isV2()) {
    const key = process.env.HEROES_PROFILE_V2_API_KEY
    if (!key) throw new Error('HP_API=v2 but HEROES_PROFILE_V2_API_KEY is not set')
    // v1 meters per endpoint with one key; key1/key2 distinction collapses.
    return new HeroesProfileApiV2(key, maxCallsPerMinute, maxRetries)
  }
  const key = which === 'key1'
    ? process.env.HEROES_PROFILE_API_KEY
    : process.env.HEROES_PROFILE_API_KEY2
  if (!key) throw new Error(`missing API key for ${which}`)
  return new HeroesProfileApi(key, maxCallsPerMinute, maxRetries)
}
