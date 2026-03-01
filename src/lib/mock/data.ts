/**
 * Mock data matching the DB schema for development.
 * Uses realistic HotS stats. Test battletags from spec:
 *   Django#1458      — strong on Ana, Falstad
 *   AzmoDonTrump#1139 — strong on Nazeebo, Azmodan
 *   SirWatsonII#1400  — strong on Malthael, Dehaka
 */

import type {
  HeroStats,
  MapStats,
  HeroMapStats,
  HeroTalentStats,
  HeroPairwiseStats,
  PlayerHeroStats,
  PlayerMatch,
  PlayerHeroMapStats,
  SkillTier,
} from '@/lib/types'
import { HERO_ROLES } from '@/lib/data/hero-roles'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const ALL_HEROES = Object.keys(HERO_ROLES)

const MAPS = [
  'Alterac Pass',
  'Battlefield of Eternity',
  'Braxis Holdout',
  'Cursed Hollow',
  'Dragon Shire',
  'Garden of Terror',
  'Hanamura Temple',
  'Infernal Shrines',
  'Sky Temple',
  'Tomb of the Spider Queen',
  'Towers of Doom',
  'Volskaya Foundry',
]

const TIERS: SkillTier[] = ['low', 'mid', 'high']

/** Deterministic pseudo-random from a string seed */
function seededRand(seed: string): number {
  let h = 0
  for (let i = 0; i < seed.length; i++) {
    h = (Math.imul(31, h) + seed.charCodeAt(i)) | 0
  }
  h = ((h >>> 16) ^ h) * 0x45d9f3b
  h = ((h >>> 16) ^ h) * 0x45d9f3b
  h = (h >>> 16) ^ h
  return (h & 0x7fffffff) / 0x7fffffff
}

/** Range-bound seeded value */
function sr(seed: string, min: number, max: number): number {
  return min + seededRand(seed) * (max - min)
}

function round1(n: number): number {
  return Math.round(n * 10) / 10
}

function round2(n: number): number {
  return Math.round(n * 100) / 100
}

// ---------------------------------------------------------------------------
// Hero Stats Aggregate
// ---------------------------------------------------------------------------

function generateHeroStats(): HeroStats[] {
  const rows: HeroStats[] = []
  for (const hero of ALL_HEROES) {
    for (const tier of TIERS) {
      const seed = `${hero}-${tier}`
      const games = Math.floor(sr(seed + 'g', 200, 8000))
      const winRate = round1(sr(seed + 'wr', 42, 58))
      const wins = Math.round(games * winRate / 100)
      rows.push({
        hero,
        skillTier: tier,
        games,
        wins,
        winRate,
        banRate: round1(sr(seed + 'br', 0, 30)),
        pickRate: round1(sr(seed + 'pr', 1, 15)),
        avgKills: round1(sr(seed + 'k', 2, 9)),
        avgDeaths: round1(sr(seed + 'd', 2, 6)),
        avgAssists: round1(sr(seed + 'a', 4, 14)),
        avgHeroDamage: Math.floor(sr(seed + 'hd', 20000, 70000)),
        avgSiegeDamage: Math.floor(sr(seed + 'sd', 15000, 65000)),
        avgHealing: HERO_ROLES[hero] === 'Healer'
          ? Math.floor(sr(seed + 'hl', 40000, 80000))
          : Math.floor(sr(seed + 'hl', 0, 5000)),
        avgExperience: Math.floor(sr(seed + 'xp', 8000, 16000)),
        avgDamageSoaked: HERO_ROLES[hero] === 'Tank'
          ? Math.floor(sr(seed + 'soak', 30000, 60000))
          : HERO_ROLES[hero] === 'Bruiser'
            ? Math.floor(sr(seed + 'soak', 18000, 40000))
            : Math.floor(sr(seed + 'soak', 5000, 18000)),
        avgMercCaptures: round1(sr(seed + 'merc', 0.5, 4.5)),
        avgSelfHealing: HERO_ROLES[hero] === 'Bruiser' || HERO_ROLES[hero] === 'Melee Assassin'
          ? Math.floor(sr(seed + 'sh', 5000, 20000))
          : HERO_ROLES[hero] === 'Healer'
            ? Math.floor(sr(seed + 'sh', 3000, 12000))
            : Math.floor(sr(seed + 'sh', 0, 5000)),
        avgTimeDead: round1(sr(seed + 'td', 20, 90)),
        patchTag: '2.55.3',
      })
    }
  }
  return rows
}

// ---------------------------------------------------------------------------
// Map Stats Aggregate
// ---------------------------------------------------------------------------

function generateMapStats(): MapStats[] {
  const rows: MapStats[] = []
  for (const map of MAPS) {
    for (const tier of TIERS) {
      rows.push({
        map,
        skillTier: tier,
        games: Math.floor(sr(`${map}-${tier}-mg`, 5000, 25000)),
      })
    }
  }
  return rows
}

// ---------------------------------------------------------------------------
// Hero Map Stats Aggregate
// ---------------------------------------------------------------------------

function generateHeroMapStats(): HeroMapStats[] {
  const rows: HeroMapStats[] = []
  for (const hero of ALL_HEROES) {
    for (const map of MAPS) {
      for (const tier of TIERS) {
        const seed = `${hero}-${map}-${tier}`
        const games = Math.floor(sr(seed + 'g', 20, 800))
        const winRate = round1(sr(seed + 'wr', 38, 64))
        rows.push({
          hero,
          map,
          skillTier: tier,
          games,
          wins: Math.round(games * winRate / 100),
          winRate,
        })
      }
    }
  }
  return rows
}

// ---------------------------------------------------------------------------
// Talent Stats (subset — top heroes only to keep data manageable)
// ---------------------------------------------------------------------------

const TALENT_DATA: Record<string, Record<number, string[]>> = {
  'Ana': {
    1: ['Grenade Calibration', 'Detachable Box Magazine', 'Contact Healing'],
    4: ['Overdose', 'Air Strike', 'Aim Down Sights'],
    7: ['Night Terrors', 'Mind-Numbing Agent', 'Temporary Blindness'],
    10: ['Nano Boost', 'Eye of Horus'],
    13: ['Smelling Salts', 'Speed Serum', 'Purifying Darts'],
    16: ['Sharpshooter', 'Concentrated Doses', 'Dynamic Optics'],
    20: ['Nano Infusion', 'Ballistic Advantage', 'Dynamic Shooting', 'Deadeye'],
  },
  'Falstad': {
    1: ['Seasoned Marksman', 'Wingman', 'Dishonorable Discharge'],
    4: ['Static Shield', 'Updraft', 'Hammer Gains'],
    7: ['Secret Weapon', 'BOOMerang', 'Charged Up'],
    10: ['Hinterland Blast', 'Mighty Gust'],
    13: ['Thunderstrikes', 'Flow Rider', 'Crippling Hammer'],
    16: ['Martial Prowess', 'Aerie Gusts', 'Afterburner'],
    20: ['Call of the Wildhammer', 'Wind Tunnel', 'Epic Mount', 'Nexus Frenzy'],
  },
  'Nazeebo': {
    1: ['Pandemic', 'Widowmakers', 'Thing of the Deep'],
    4: ['Big Voodoo', 'Hexed Crawlers', 'Blood Ritual'],
    7: ['Spirit of Arachyr', 'Dead Rush', 'Toads of Hugeness'],
    10: ['Gargantuan', 'Ravenous Spirit'],
    13: ['Guardian Toads', 'Superstition', 'Ice Block'],
    16: ['Spider Colony', 'Soul Harvest', 'Ring of Poison'],
    20: ['Humongoid', 'Annihilating Spirit', 'Vile Infection', 'Fury of the Storm'],
  },
  'Azmodan': {
    1: ['Greed', 'Gluttony', 'Wrath'],
    4: ['Army of Hell', 'Hellforged Armor', 'Battleborn'],
    7: ['Master of Destruction', 'Bombardment', 'Art of Chaos'],
    10: ['Demonic Invasion', 'Tide of Sin'],
    13: ['Brutish Vanguard', 'Chain of Command', 'Cydaea\'s Kiss'],
    16: ['Total Annihilation', 'Hell Rift', 'Trample'],
    20: ['Siegebreaker', 'Black Pool', 'Pride', 'Fury of the Storm'],
  },
  'Malthael': {
    1: ['On a Pale Horse', 'Death\'s Reach', 'Fear the Reaper'],
    4: ['Die Alone', 'Throwing Shade', 'Black Harvest'],
    7: ['Cold Hand', 'Massacre', 'Touch of Death'],
    10: ['Tormented Souls', 'Last Rites'],
    13: ['Soul Siphon', 'Ethereal Existence', 'Inevitable End'],
    16: ['Soul Collector', 'Mortality', 'Memento Mori'],
    20: ['Reaper of Souls', 'Angel of Death', 'Final Curtain', 'No One Can Stop Death'],
  },
  'Dehaka': {
    1: ['Enduring Swarm', 'Tissue Regeneration', 'Enhanced Agility'],
    4: ['Hero Stalker', 'Lurker Strain', 'One-Who-Collects'],
    7: ['Feeding Frenzy', 'Paralyzing Enzymes', 'Symbiosis'],
    10: ['Isolation', 'Adaptation'],
    13: ['Primal Swarm', 'Primal Rage', 'Swift Pursuit'],
    16: ['Pack Leader', 'Elongated Tongue', 'Tunneling Claws'],
    20: ['Contagion', 'Change Is Survival', 'Essence Claws', 'Apex Predator'],
  },
}

function generateTalentStats(): HeroTalentStats[] {
  const rows: HeroTalentStats[] = []
  for (const [hero, tiers] of Object.entries(TALENT_DATA)) {
    for (const [tierStr, talents] of Object.entries(tiers)) {
      const talentTier = parseInt(tierStr)
      for (const skillTier of TIERS) {
        // Distribute pick rates so they sum ~100%
        const totalTalents = talents.length
        for (let i = 0; i < talents.length; i++) {
          const seed = `${hero}-${talentTier}-${talents[i]}-${skillTier}`
          const rawPick = sr(seed + 'pk', 10, 60)
          const winRate = round1(sr(seed + 'wr', 42, 60))
          const games = Math.floor(sr(seed + 'g', 50, 2000))
          rows.push({
            hero,
            skillTier,
            talentTier,
            talentName: talents[i],
            games,
            wins: Math.round(games * winRate / 100),
            winRate,
            pickRate: round1(rawPick / totalTalents * (i === 0 ? 1.5 : 1)),
          })
        }
      }
    }
  }
  return rows
}

// ---------------------------------------------------------------------------
// Pairwise Stats (curated subset for meaningful synergies/counters)
// ---------------------------------------------------------------------------

interface PairSeed {
  a: string
  b: string
  rel: 'with' | 'against'
  baseWr: number // approximate center win rate
}

const PAIRWISE_SEEDS: PairSeed[] = [
  // Strong synergies
  { a: 'Ana', b: 'Greymane', rel: 'with', baseWr: 56 },
  { a: 'Ana', b: 'Valla', rel: 'with', baseWr: 55 },
  { a: 'Auriel', b: 'Gul\'dan', rel: 'with', baseWr: 57 },
  { a: 'Auriel', b: 'Cho', rel: 'with', baseWr: 58 },
  { a: 'Abathur', b: 'Illidan', rel: 'with', baseWr: 55 },
  { a: 'Abathur', b: 'Tracer', rel: 'with', baseWr: 54 },
  { a: 'Dehaka', b: 'Falstad', rel: 'with', baseWr: 54 },
  { a: 'Uther', b: 'The Butcher', rel: 'with', baseWr: 55 },
  { a: 'Malthael', b: 'Johanna', rel: 'with', baseWr: 53 },
  { a: 'Nazeebo', b: 'Johanna', rel: 'with', baseWr: 54 },
  { a: 'E.T.C.', b: 'Jaina', rel: 'with', baseWr: 55 },
  { a: 'E.T.C.', b: 'Kael\'thas', rel: 'with', baseWr: 54 },
  { a: 'Garrosh', b: 'Kerrigan', rel: 'with', baseWr: 56 },
  { a: 'Maiev', b: 'Diablo', rel: 'with', baseWr: 55 },
  // Counters (A beats B)
  { a: 'Ana', b: 'Muradin', rel: 'against', baseWr: 54 },
  { a: 'Lunara', b: 'Uther', rel: 'against', baseWr: 55 },
  { a: 'Tychus', b: 'Cho', rel: 'against', baseWr: 60 },
  { a: 'Tychus', b: 'Diablo', rel: 'against', baseWr: 55 },
  { a: 'Malthael', b: 'Cho', rel: 'against', baseWr: 62 },
  { a: 'Malthael', b: 'Arthas', rel: 'against', baseWr: 54 },
  { a: 'Falstad', b: 'Chromie', rel: 'against', baseWr: 54 },
  { a: 'Johanna', b: 'Illidan', rel: 'against', baseWr: 56 },
  { a: 'Li-Ming', b: 'The Lost Vikings', rel: 'against', baseWr: 58 },
  { a: 'Anub\'arak', b: 'Li-Ming', rel: 'against', baseWr: 56 },
  { a: 'Brightwing', b: 'The Butcher', rel: 'against', baseWr: 54 },
  { a: 'Zeratul', b: 'Ana', rel: 'against', baseWr: 55 },
  { a: 'Genji', b: 'Ana', rel: 'against', baseWr: 54 },
  { a: 'Dehaka', b: 'Murky', rel: 'against', baseWr: 53 },
]

function generatePairwiseStats(): HeroPairwiseStats[] {
  const rows: HeroPairwiseStats[] = []
  for (const pair of PAIRWISE_SEEDS) {
    for (const tier of TIERS) {
      const seed = `${pair.a}-${pair.b}-${pair.rel}-${tier}`
      const jitter = sr(seed, -3, 3)
      const winRate = round1(Math.max(40, Math.min(65, pair.baseWr + jitter)))
      const games = Math.floor(sr(seed + 'g', 100, 3000))
      rows.push({
        heroA: pair.a,
        heroB: pair.b,
        relationship: pair.rel,
        skillTier: tier,
        games,
        wins: Math.round(games * winRate / 100),
        winRate,
      })
    }
  }
  return rows
}

// ---------------------------------------------------------------------------
// Player Data — test battletags
// ---------------------------------------------------------------------------

interface PlayerProfile {
  battletag: string
  strongHeroes: string[]
  weakHeroes: string[]
  playedHeroes: string[] // broader pool
}

const PLAYERS: PlayerProfile[] = [
  {
    battletag: 'Django#1458',
    strongHeroes: ['Ana', 'Falstad'],
    weakHeroes: ['Murky', 'Probius'],
    playedHeroes: [
      'Ana', 'Falstad', 'Jaina', 'Johanna', 'Rehgar', 'Malfurion',
      'Valla', 'Muradin', 'Greymane', 'Li-Ming', 'Murky', 'Probius',
      'Brightwing', 'Tyrande', 'Raynor', 'Sylvanas',
    ],
  },
  {
    battletag: 'AzmoDonTrump#1139',
    strongHeroes: ['Nazeebo', 'Azmodan'],
    weakHeroes: ['Genji', 'Tracer'],
    playedHeroes: [
      'Nazeebo', 'Azmodan', 'Zagara', 'Xul', 'Sylvanas', 'Jaina',
      'Kael\'thas', 'Johanna', 'Arthas', 'Rehgar', 'Genji', 'Tracer',
      'Gazlowe', 'Ragnaros', 'Sonya', 'Gul\'dan',
    ],
  },
  {
    battletag: 'SirWatsonII#1400',
    strongHeroes: ['Malthael', 'Dehaka'],
    weakHeroes: ['Li-Ming', 'Chromie'],
    playedHeroes: [
      'Malthael', 'Dehaka', 'Sonya', 'Thrall', 'Imperius', 'Artanis',
      'Blaze', 'Yrel', 'Diablo', 'Muradin', 'Rehgar', 'Li-Ming',
      'Chromie', 'Leoric', 'Hogger', 'Ragnaros',
    ],
  },
]

function generatePlayerHeroStats(): PlayerHeroStats[] {
  const rows: PlayerHeroStats[] = []
  for (const player of PLAYERS) {
    for (const hero of player.playedHeroes) {
      const seed = `${player.battletag}-${hero}`
      const isStrong = player.strongHeroes.includes(hero)
      const isWeak = player.weakHeroes.includes(hero)

      const baseWr = isStrong ? sr(seed + 'bwr', 58, 68)
        : isWeak ? sr(seed + 'bwr', 35, 44)
        : sr(seed + 'bwr', 46, 55)

      const games = isStrong
        ? Math.floor(sr(seed + 'g', 80, 250))
        : isWeak
          ? Math.floor(sr(seed + 'g', 10, 30))
          : Math.floor(sr(seed + 'g', 20, 100))

      const winRate = round1(baseWr)
      const wins = Math.round(games * winRate / 100)

      // MAWP slightly different from overall for strong heroes (recent hot streak)
      const mawp = isStrong
        ? round1(baseWr + sr(seed + 'mawp', 1, 6))
        : round1(baseWr + sr(seed + 'mawp', -4, 4))

      const recentWr = round1(mawp + sr(seed + 'rwr', -3, 3))
      const trend = round2(recentWr - winRate)

      rows.push({
        battletag: player.battletag,
        hero,
        games,
        wins,
        winRate,
        mawp,
        avgKills: round1(sr(seed + 'k', 3, 8)),
        avgDeaths: round1(sr(seed + 'd', 2, 5)),
        avgAssists: round1(sr(seed + 'a', 5, 13)),
        recentWinRate: recentWr,
        trend,
      })
    }
  }
  return rows
}

function generatePlayerHeroMapStats(): PlayerHeroMapStats[] {
  const rows: PlayerHeroMapStats[] = []
  for (const player of PLAYERS) {
    // Only strong heroes get per-map breakdowns (keeps data sane)
    for (const hero of player.strongHeroes) {
      for (const map of MAPS) {
        const seed = `${player.battletag}-${hero}-${map}`
        const games = Math.floor(sr(seed + 'g', 5, 40))
        const winRate = round1(sr(seed + 'wr', 50, 75))
        rows.push({
          battletag: player.battletag,
          hero,
          map,
          games,
          wins: Math.round(games * winRate / 100),
          winRate,
        })
      }
    }
  }
  return rows
}

function generatePlayerMatchHistory(): PlayerMatch[] {
  const rows: PlayerMatch[] = []
  const now = new Date()

  for (const player of PLAYERS) {
    // Generate 60 recent matches per player
    for (let i = 0; i < 60; i++) {
      const daysAgo = i * sr(`${player.battletag}-${i}-day`, 0.3, 2.5)
      const gameDate = new Date(now.getTime() - daysAgo * 24 * 60 * 60 * 1000)

      // Weight hero selection toward strong heroes
      const heroPool = player.playedHeroes
      const seed = `${player.battletag}-match-${i}`
      const roll = sr(seed, 0, 1)
      let hero: string
      if (roll < 0.35) {
        hero = player.strongHeroes[Math.floor(sr(seed + 'sh', 0, player.strongHeroes.length))]
      } else {
        hero = heroPool[Math.floor(sr(seed + 'hp', 0, heroPool.length))]
      }

      const map = MAPS[Math.floor(sr(seed + 'map', 0, MAPS.length))]
      const isStrong = player.strongHeroes.includes(hero)
      const winChance = isStrong ? 0.62 : 0.50

      rows.push({
        battletag: player.battletag,
        replayId: `replay-${player.battletag}-${i}`,
        hero,
        map,
        win: sr(seed + 'win', 0, 1) < winChance,
        gameDate,
        gameLength: Math.floor(sr(seed + 'len', 12 * 60, 25 * 60)),
        kills: Math.floor(sr(seed + 'k', 1, 12)),
        deaths: Math.floor(sr(seed + 'd', 0, 8)),
        assists: Math.floor(sr(seed + 'a', 3, 18)),
        heroDamage: Math.floor(sr(seed + 'hd', 15000, 70000)),
        siegeDamage: Math.floor(sr(seed + 'sd', 10000, 55000)),
        healing: HERO_ROLES[hero] === 'Healer'
          ? Math.floor(sr(seed + 'hl', 35000, 75000))
          : Math.floor(sr(seed + 'hl', 0, 5000)),
        experience: Math.floor(sr(seed + 'xp', 6000, 15000)),
        talents: null,
        gameMode: 'Storm League',
        rank: null,
      })
    }
  }

  return rows
}

// ---------------------------------------------------------------------------
// Export singleton instances (generated once on import)
// ---------------------------------------------------------------------------

export const mockHeroStats = generateHeroStats()
export const mockMapStats = generateMapStats()
export const mockHeroMapStats = generateHeroMapStats()
export const mockTalentStats = generateTalentStats()
export const mockPairwiseStats = generatePairwiseStats()
export const mockPlayerHeroStats = generatePlayerHeroStats()
export const mockPlayerHeroMapStats = generatePlayerHeroMapStats()
export const mockPlayerMatchHistory = generatePlayerMatchHistory()

export const mockTrackedBattletags = PLAYERS.map((p) => ({
  battletag: p.battletag,
  region: 1,
  lastSynced: new Date(),
}))

export { ALL_HEROES, MAPS, TIERS }
