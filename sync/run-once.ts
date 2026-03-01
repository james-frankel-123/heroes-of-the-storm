/**
 * One-shot resumable sync. Checks the DB before each API call and skips
 * anything that already has data. Writes progress to sync/progress.json
 * so it can be safely restarted without re-fetching.
 *
 * Usage: set -a && source .env && set +a && npx tsx sync/run-once.ts
 */
import { existsSync, readFileSync, writeFileSync } from 'fs'
import { sql } from 'drizzle-orm'
import { HERO_ROLES } from '../src/lib/data/hero-roles'
import { heroPairwiseStats, heroTalentStats } from '../src/lib/db/schema'
import { HeroesProfileApi } from './api-client'
import { computeDerivedStats } from './compute-derived'
import { createDb, SyncDb } from './db'
import { log } from './logger'
import { getCurrentPatch } from './sync-global'
import { syncPlayerData, BattletagConfig } from './sync-players'

const ALL_HEROES = Object.keys(HERO_ROLES)
const PROGRESS_FILE = new URL('progress.json', import.meta.url).pathname

const BATTLETAGS: BattletagConfig[] = [
  { battletag: 'Django#1458', region: 1 },
  { battletag: 'AzmoDonTrump#1139', region: 1 },
  { battletag: 'SirWatsonII#1400', region: 1 },
]

// ── Helpers ──────────────────────────────────────────────────────────

async function pMap<T, R>(
  items: T[],
  fn: (item: T, index: number) => Promise<R>,
  concurrency: number,
): Promise<R[]> {
  const results: R[] = new Array(items.length)
  let nextIndex = 0
  async function worker() {
    while (nextIndex < items.length) {
      const i = nextIndex++
      results[i] = await fn(items[i], i)
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(concurrency, items.length) }, () => worker()),
  )
  return results
}

// ── Progress tracking ────────────────────────────────────────────────

interface Progress {
  heroStats: boolean
  talents: Record<string, boolean>       // keyed by tier: low, mid, high
  matchups: Record<string, boolean>      // keyed by hero name
  playerReplays: Record<string, boolean> // keyed by battletag
  derived: boolean
}

function loadProgress(): Progress {
  if (existsSync(PROGRESS_FILE)) {
    try {
      return JSON.parse(readFileSync(PROGRESS_FILE, 'utf-8'))
    } catch { /* corrupted, start fresh */ }
  }
  return {
    heroStats: false,
    talents: {},
    matchups: {},
    playerReplays: {},
    derived: false,
  }
}

function saveProgress(p: Progress) {
  writeFileSync(PROGRESS_FILE, JSON.stringify(p, null, 2))
}

// ── DB checks ────────────────────────────────────────────────────────

async function hasHeroStats(db: SyncDb): Promise<boolean> {
  const result = await db.execute(
    sql`SELECT count(*) as c FROM hero_stats_aggregate`,
  )
  return Number(result.rows[0].c) >= 250 // ~90 heroes × 3 tiers
}

async function hasTalentsForTier(db: SyncDb, tier: string): Promise<boolean> {
  const result = await db.execute(
    sql`SELECT count(*) as c FROM hero_talent_stats WHERE skill_tier = ${tier}`,
  )
  return Number(result.rows[0].c) > 100 // reasonable threshold
}

async function hasMatchupsForHero(db: SyncDb, hero: string): Promise<boolean> {
  const result = await db.execute(
    sql`SELECT count(*) as c FROM hero_pairwise_stats WHERE hero_a = ${hero}`,
  )
  return Number(result.rows[0].c) > 0
}

// ── Main ─────────────────────────────────────────────────────────────

async function main() {
  const apiKey = process.env.HEROES_PROFILE_API_KEY
  if (!apiKey) { log.error('HEROES_PROFILE_API_KEY required'); process.exit(1) }
  if (!process.env.DATABASE_URL) { log.error('DATABASE_URL required'); process.exit(1) }

  const db = createDb()
  const api = new HeroesProfileApi(apiKey, 55, 2) // max 2 retries to fail fast
  const progress = loadProgress()

  log.info('=== Resumable one-shot sync ===')
  log.info(`Progress file: ${PROGRESS_FILE}`)

  // Get current patch (1 General call, doesn't count toward Hero Data)
  const patch = await getCurrentPatch(api)
  log.info(`Patch: ${patch.version}`)

  // ── 1. Hero Stats (already done from DB check) ──
  if (progress.heroStats || await hasHeroStats(db)) {
    log.info('[SKIP] Hero stats — already in DB')
    progress.heroStats = true
    saveProgress(progress)
  } else {
    log.info('[TODO] Hero stats need syncing — but this costs 3 Hero Data calls')
    log.info('       Run the full sync (npx tsx sync/index.ts) if you want to refresh these')
    // Not implemented here to save calls — they already exist
  }

  // ── 2. Talent Stats (per-hero to avoid bulk endpoint timeout) ──
  const tierMap: Array<[string, string]> = [['low', '1,2'], ['mid', '3,4'], ['high', '5,6']]
  const TALENT_CONCURRENCY = 15

  for (const [tier, leagueTier] of tierMap) {
    if (progress.talents[tier] as any || await hasTalentsForTier(db, tier)) {
      log.info(`[SKIP] Talents tier=${tier} — already in DB`)
      progress.talents[tier] = true
      saveProgress(progress)
      continue
    }

    // Track per-hero progress within each tier
    const talentHeroKey = `talentHeroes_${tier}` as string
    const doneTalentHeroes: Record<string, boolean> = (progress as any)[talentHeroKey] ?? {}

    const missingHeroes = ALL_HEROES.filter(h => !doneTalentHeroes[h])
    if (missingHeroes.length === 0) {
      log.info(`[SKIP] Talents tier=${tier} — all heroes fetched`)
      progress.talents[tier] = true
      saveProgress(progress)
      continue
    }

    log.info(`[FETCH] Talents tier=${tier} per-hero: ${missingHeroes.length} heroes (concurrency=${TALENT_CONCURRENCY})`)
    let ok = 0, fail = 0

    await pMap(missingHeroes, async (hero) => {
      try {
        const raw = await api.getTalentDetails(patch.type, patch.version, leagueTier, hero)
        const rows = parseTalentBulk(raw, tier)
        if (rows.length > 0) {
          await batchInsertTalents(db, rows)
        }
        ok++
        doneTalentHeroes[hero] = true
        ;(progress as any)[talentHeroKey] = doneTalentHeroes
        saveProgress(progress)
        if (ok % 20 === 0) log.info(`  Talents tier=${tier} progress: ${ok}/${missingHeroes.length}`)
      } catch (err) {
        fail++
        log.error(`  FAILED Talents tier=${tier} hero=${hero}: ${err instanceof Error ? err.message : err}`)
        log.error(`  API: Heroes/Talents/Details?hero=${hero}&league_tier=${leagueTier}&timeframe=${patch.version}`)
      }
    }, TALENT_CONCURRENCY)

    log.info(`Talents tier=${tier} done: ${ok} succeeded, ${fail} failed`)
    if (fail === 0) {
      progress.talents[tier] = true
      saveProgress(progress)
    }
  }

  // ── 3. Matchups ──
  const missingMatchups = []
  for (const hero of ALL_HEROES) {
    if (progress.matchups[hero] || await hasMatchupsForHero(db, hero)) {
      progress.matchups[hero] = true
      continue
    }
    missingMatchups.push(hero)
  }
  saveProgress(progress)

  if (missingMatchups.length === 0) {
    log.info('[SKIP] All matchups — already in DB')
  } else {
    log.info(`[FETCH] Matchups for ${missingMatchups.length} heroes (${missingMatchups.length} Hero Data calls)`)
    let ok = 0, fail = 0
    // Sequential to be gentle on rate limits
    for (const hero of missingMatchups) {
      try {
        const raw = await api.getHeroMatchups(hero, patch.type, patch.version)
        const rows = parseMatchups(hero, raw)
        if (rows.length > 0) {
          await batchInsertMatchups(db, rows)
        }
        ok++
        progress.matchups[hero] = true
        saveProgress(progress)
        if (ok % 10 === 0) log.info(`  Matchups progress: ${ok}/${missingMatchups.length}`)
      } catch (err) {
        fail++
        log.error(`  FAILED Matchups hero=${hero}: ${err instanceof Error ? err.message : err}`)
        log.error(`  API: Heroes/Matchups?hero=${hero}&timeframe=${patch.version}`)
      }
    }
    log.info(`Matchups done: ${ok} succeeded, ${fail} failed`)
  }

  // ── 4. Player Replays (incremental — always cheap) ──
  for (const config of BATTLETAGS) {
    if (progress.playerReplays[config.battletag]) {
      log.info(`[SKIP] Player replays ${config.battletag} — already synced this run`)
      continue
    }
    log.info(`[FETCH] Player replays ${config.battletag} (1 Player Data call, incremental)`)
    try {
      await syncPlayerData(api, db, [config])
      progress.playerReplays[config.battletag] = true
      saveProgress(progress)
      log.info(`  ✓ Synced ${config.battletag}`)
    } catch (err) {
      log.error(`  FAILED Player replays ${config.battletag}: ${err instanceof Error ? err.message : err}`)
      log.error(`  API: Player/Replays?battletag=${config.battletag}`)
    }
  }

  // ── 5. Derived Stats (no API calls) ──
  if (progress.derived) {
    log.info('[SKIP] Derived stats — already computed this run')
  } else {
    log.info('[COMPUTE] Derived stats (MAWP, trends) — no API calls')
    try {
      await computeDerivedStats(db, BATTLETAGS.map(b => b.battletag))
      progress.derived = true
      saveProgress(progress)
      log.info('  ✓ Derived stats computed')
    } catch (err) {
      log.error(`  FAILED Derived stats: ${err instanceof Error ? err.message : err}`)
    }
  }

  log.info(`\nDone. Total API calls this run: ${api.getCallCount()}`)
  log.info(`Progress saved to ${PROGRESS_FILE}`)
}

// ── Parsers (duplicated from sync-global to keep this self-contained) ──

function num(val: any, fallback = 0): number {
  if (val === null || val === undefined) return fallback
  const n = typeof val === 'string' ? parseFloat(val) : Number(val)
  return isNaN(n) ? fallback : n
}

function parseTalentBulk(data: any, tier: string) {
  const rows: any[] = []
  if (!data || typeof data !== 'object' || Array.isArray(data)) return rows

  for (const [heroName, heroData] of Object.entries(data)) {
    if (typeof heroData !== 'object' || heroData === null) continue
    for (const [tierStr, talents] of Object.entries(heroData as Record<string, any>)) {
      const talentTier = parseInt(tierStr, 10)
      if (isNaN(talentTier)) continue
      if (typeof talents !== 'object' || talents === null) continue
      for (const [talentName, stats] of Object.entries(talents as Record<string, any>)) {
        if (typeof stats !== 'object' || stats === null) continue
        rows.push({
          hero: heroName,
          skillTier: tier,
          talentTier,
          talentName,
          games: num(stats.games_played ?? stats.games),
          wins: num(stats.wins),
          winRate: num(stats.win_rate ?? stats.winrate),
          pickRate: num(stats.popularity ?? stats.pick_rate, 0),
          updatedAt: new Date(),
        })
      }
    }
  }
  return rows
}

async function batchInsertTalents(db: SyncDb, rows: any[]) {
  for (let i = 0; i < rows.length; i += 100) {
    const batch = rows.slice(i, i + 100)
    await db.insert(heroTalentStats).values(batch).onConflictDoUpdate({
      target: [heroTalentStats.hero, heroTalentStats.skillTier, heroTalentStats.talentTier, heroTalentStats.talentName],
      set: {
        games: sql.raw('excluded.games'),
        wins: sql.raw('excluded.wins'),
        winRate: sql.raw('excluded.win_rate'),
        pickRate: sql.raw('excluded.pick_rate'),
        updatedAt: sql`now()`,
      },
    })
  }
}

function parseMatchups(hero: string, data: any) {
  const rows: any[] = []
  if (!data || typeof data !== 'object') return rows

  let heroMatchups = data
  if (data[hero] && typeof data[hero] === 'object') heroMatchups = data[hero]

  for (const [heroB, md] of Object.entries(heroMatchups)) {
    if (typeof md !== 'object' || md === null) continue
    const m = md as Record<string, any>

    if (m.ally && typeof m.ally === 'object') {
      const w = num(m.ally.wins_with ?? m.ally.wins)
      const l = num(m.ally.losses_with ?? m.ally.losses)
      if (w + l > 0) {
        // Store for all tiers (matchup API doesn't have tier param)
        for (const tier of ['low', 'mid', 'high']) {
          rows.push({
            heroA: hero, heroB, relationship: 'with', skillTier: tier,
            games: w + l, wins: w, winRate: num(m.ally.win_rate_as_ally ?? m.ally.win_rate),
            updatedAt: new Date(),
          })
        }
      }
    }

    if (m.enemy && typeof m.enemy === 'object') {
      const w = num(m.enemy.wins_against ?? m.enemy.wins)
      const l = num(m.enemy.losses_against ?? m.enemy.losses)
      if (w + l > 0) {
        for (const tier of ['low', 'mid', 'high']) {
          rows.push({
            heroA: hero, heroB, relationship: 'against', skillTier: tier,
            games: w + l, wins: w, winRate: num(m.enemy.win_rate_against ?? m.enemy.win_rate),
            updatedAt: new Date(),
          })
        }
      }
    }
  }
  return rows
}

async function batchInsertMatchups(db: SyncDb, rows: any[]) {
  for (let i = 0; i < rows.length; i += 100) {
    const batch = rows.slice(i, i + 100)
    await db.insert(heroPairwiseStats).values(batch).onConflictDoUpdate({
      target: [heroPairwiseStats.heroA, heroPairwiseStats.heroB, heroPairwiseStats.relationship, heroPairwiseStats.skillTier],
      set: {
        games: sql.raw('excluded.games'),
        wins: sql.raw('excluded.wins'),
        winRate: sql.raw('excluded.win_rate'),
        updatedAt: sql`now()`,
      },
    })
  }
}

main().catch(err => {
  log.error('Fatal error', err)
  process.exit(1)
})
