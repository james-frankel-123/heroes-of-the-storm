// Team composition duo data
export interface DuoStats {
  heroes: string
  games: number
  wins: number
  losses: number
  winRate: number
}

// Parse hero pair from "Hero1 + Hero2" format
export function parseDuoHeroes(duoString: string): [string, string] {
  const heroes = duoString.split(' + ').map(h => h.trim())
  return [heroes[0], heroes[1]]
}

// Get duo win rate for two heroes
export function getDuoWinRate(hero1: string, hero2: string, duoData: DuoStats[]): number | null {
  const duo = duoData.find(d => {
    const [h1, h2] = parseDuoHeroes(d.heroes)
    return (h1 === hero1 && h2 === hero2) || (h1 === hero2 && h2 === hero1)
  })

  return duo && duo.games >= 2 ? duo.winRate : null
}

// Static duo data (from team_compositions.csv)
export const TEAM_COMPOSITIONS: DuoStats[] = [
  { heroes: 'Dehaka + Nazeebo', games: 5, wins: 3, losses: 2, winRate: 60.0 },
  { heroes: 'Falstad + Li-Ming', games: 4, wins: 3, losses: 1, winRate: 75.0 },
  { heroes: 'Diablo + Nazeebo', games: 2, wins: 0, losses: 2, winRate: 0.0 },
  { heroes: 'Falstad + Lúcio', games: 3, wins: 1, losses: 2, winRate: 33.3 },
  { heroes: 'Falstad + Nazeebo', games: 5, wins: 3, losses: 2, winRate: 60.0 },
  { heroes: 'Falstad + Garrosh', games: 5, wins: 3, losses: 2, winRate: 60.0 },
  { heroes: 'Anduin + Stitches', games: 2, wins: 2, losses: 0, winRate: 100.0 },
  { heroes: 'Diablo + Deckard', games: 2, wins: 2, losses: 0, winRate: 100.0 },
  { heroes: 'Dehaka + Stukov', games: 2, wins: 0, losses: 2, winRate: 0.0 },
  { heroes: 'Raynor + Stitches', games: 2, wins: 0, losses: 2, winRate: 0.0 },
  { heroes: 'Diablo + Li-Ming', games: 2, wins: 0, losses: 2, winRate: 0.0 },
  { heroes: 'Valla + Stukov', games: 2, wins: 0, losses: 2, winRate: 0.0 },
  { heroes: 'Anduin + Nazeebo', games: 2, wins: 2, losses: 0, winRate: 100.0 },
]
