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
