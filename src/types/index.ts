export interface HeroStats {
  hero: string
  role: string
  wins: number
  losses: number
  games: number
  winRate: number
}

export interface MapStats {
  map: string
  wins: number
  losses: number
  games: number
  winRate: number
  heroes: HeroStats[]
}

export interface PlayerData {
  playerName: string
  totalGames: number
  totalWins: number
  totalLosses: number
  overallWinRate: number
  heroStats: HeroStats[]
  mapStats: MapStats[]
  roleStats: Record<string, { wins: number; games: number; winRate: number }>
}

export interface PlayerMMR {
  mmr: number | null
  league_tier: string | null
  games_played: number
}

export interface PowerPick {
  hero: string
  role: string
  map: string
  winRate: number
  games: number
  wins: number
  losses: number
}

export interface Insight {
  type: 'success' | 'warning' | 'info' | 'tip'
  title: string
  description: string
  icon?: string
}

export type Role =
  | 'Tank'
  | 'Bruiser'
  | 'Healer'
  | 'Ranged Assassin'
  | 'Melee Assassin'
  | 'Support'
  | 'Unknown'

export interface ReplayData {
  replayId: string
  gameType: string
  hero: string
  map: string
  result: 'win' | 'loss'
  date: string
  duration: number
  partyMembers: string[]  // Battletags of ALL party members INCLUDING self
  partySize: number       // 2 = duo, 3 = trio, 4 = quadruple, 5 = quintuple
  kills: number
  deaths: number
  assists: number
  partyMemberHeroes: { [battletag: string]: string }  // Hero each party member played
}

export interface PartyGroup {
  members: string[]       // Sorted array of battletags in the party (including player)
  displayNames: string[]  // Formatted names without #numbers
  partySize: number       // 2, 3, 4, or 5
  totalGames: number
  totalWins: number
  totalLosses: number
  winRate: number
  commonHeroes: { hero: string; games: number; winRate: number }[]
  bestMaps: { map: string; games: number; winRate: number }[]
  membershipKey: string   // Unique identifier: sorted battletags joined with '|'
  memberHeroes: {         // Top heroes for each party member
    [battletag: string]: { hero: string; games: number; winRate: number }[]
  }
  compositions: {         // Win rates by team composition
    composition: string   // e.g., "Tank + Healer + 3 Assassin"
    games: number
    wins: number
    losses: number
    winRate: number
  }[]
}

export interface PartyStats {
  duos: PartyGroup[]       // 2-player parties
  trios: PartyGroup[]      // 3-player parties
  quadruples: PartyGroup[] // 4-player parties
  quintuples: PartyGroup[] // 5-player parties (full premade)
}

export interface ReplayResponse {
  battletag: string
  totalReplays: number
  soloGames: number
  partyGames: number
  partyStats: PartyStats
  replays: ReplayData[]
}
