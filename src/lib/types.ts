/**
 * Application types — derived from DB schema.
 * Pages import from here, not from drizzle schema directly.
 */

export type SkillTier = 'low' | 'mid' | 'high'

export interface HeroStats {
  hero: string
  skillTier: SkillTier
  games: number
  wins: number
  winRate: number
  banRate: number
  pickRate: number
  avgKills: number
  avgDeaths: number
  avgAssists: number
  avgHeroDamage: number
  avgSiegeDamage: number
  avgHealing: number
  avgExperience: number
  avgDamageSoaked: number
  avgMercCaptures: number
  avgSelfHealing: number
  avgTimeDead: number
  patchTag: string | null
}

export interface MapStats {
  map: string
  skillTier: SkillTier
  games: number
}

export interface HeroMapStats {
  hero: string
  map: string
  skillTier: SkillTier
  games: number
  wins: number
  winRate: number
}

export interface HeroTalentStats {
  hero: string
  skillTier: SkillTier
  talentTier: number // 1, 4, 7, 10, 13, 16, 20
  talentName: string
  games: number
  wins: number
  winRate: number
  pickRate: number
}

export interface HeroPairwiseStats {
  heroA: string
  heroB: string
  relationship: 'with' | 'against'
  skillTier: SkillTier
  games: number
  wins: number
  winRate: number
}

export interface PlayerMatch {
  battletag: string
  replayId: string
  hero: string
  map: string
  win: boolean
  gameDate: Date
  gameLength: number
  kills: number
  deaths: number
  assists: number
  heroDamage: number
  siegeDamage: number
  healing: number
  experience: number
  talents: unknown
  gameMode: string
  rank: string | null
}

export interface PlayerHeroStats {
  battletag: string
  hero: string
  games: number
  wins: number
  winRate: number
  mawp: number | null
  avgKills: number
  avgDeaths: number
  avgAssists: number
  recentWinRate: number | null
  trend: number | null
}

export interface PlayerHeroMapStats {
  battletag: string
  hero: string
  map: string
  games: number
  wins: number
  winRate: number
}

export interface TrackedBattletag {
  battletag: string
  region: number
  lastSynced: Date | null
}
