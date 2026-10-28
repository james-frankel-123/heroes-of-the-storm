/**
 * Hero matchup rows (hero_pairwise_stats) from the Heroes/Matchups payload. Pure; tested in
 * sync/__tests__/matchup-rows.test.ts. Invariant for every row: wins = round(games * winRate / 100),
 * both from hero A's side.
 */

function num(val: any, fallback = 0): number {
  if (val === null || val === undefined) return fallback
  const n = typeof val === 'string' ? parseFloat(val) : Number(val)
  return isNaN(n) ? fallback : n
}

export interface MatchupRow {
  heroA: string
  heroB: string
  relationship: string
  games: number
  wins: number
  winRate: number
}

/**
 * Parse matchup data from the API.
 * Actual format: { "Ana": { "HeroB": { "ally": { wins_with, losses_with, win_rate_as_ally }, "enemy": { wins_against, losses_against, win_rate_against } }, ... } }
 */
export function parseMatchupData(hero: string, data: any): MatchupRow[] {
  const rows: MatchupRow[] = []
  if (!data || typeof data !== 'object') return rows

  // Unwrap the hero key if present
  let heroMatchups = data
  if (data[hero] && typeof data[hero] === 'object') {
    heroMatchups = data[hero]
  }

  for (const [heroB, matchupData] of Object.entries(heroMatchups)) {
    if (typeof matchupData !== 'object' || matchupData === null) continue
    const md = matchupData as Record<string, any>

    // Parse ally (synergy) data
    if (md.ally && typeof md.ally === 'object') {
      const ally = md.ally
      const winsW = num(ally.wins_with ?? ally.wins)
      const lossesW = num(ally.losses_with ?? ally.losses)
      const gamesW = winsW + lossesW
      const winRateW = num(ally.win_rate_as_ally ?? ally.win_rate)

      if (gamesW > 0) {
        rows.push({
          heroA: hero,
          heroB,
          relationship: 'with',
          games: gamesW,
          wins: winsW,
          winRate: winRateW,
        })
      }
    }

    // Parse enemy (counter) data
    // API returns the OPPONENT's wins/WR, so we invert to get hero A's perspective
    if (md.enemy && typeof md.enemy === 'object') {
      const enemy = md.enemy
      const opponentWins = num(enemy.wins_against ?? enemy.wins)
      const opponentLosses = num(enemy.losses_against ?? enemy.losses)
      const gamesA = opponentWins + opponentLosses
      const opponentWR = num(enemy.win_rate_against ?? enemy.win_rate)

      if (gamesA > 0) {
        // The rate is the field validated against our own replays (check-v1-live.ts,
        // r = -0.966); v1 enemy rows' win counts disagree with it (they come out as A's
        // own wins, so inverting them gave the opponent's). Derive A's wins from the rate
        // so wins / games and winRate always describe the same side.
        const winRateA = Math.round((100 - opponentWR) * 100) / 100
        rows.push({
          heroA: hero,
          heroB,
          relationship: 'against',
          games: gamesA,
          wins: Math.round((gamesA * winRateA) / 100),
          winRate: winRateA,
        })
      }
    }
  }

  return rows
}
