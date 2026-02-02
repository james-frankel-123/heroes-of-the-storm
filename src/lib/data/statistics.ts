import { PlayerData, ReplayData, HeroStats } from '@/types'

// Types for statistics
export interface TimeSeriesPoint {
  date: string
  winRate: number
  games: number
  wins: number
  losses: number
}

export interface Streak {
  type: 'win' | 'loss'
  length: number
  startDate: string
  endDate: string
  current: boolean
}

export interface HeroKDA {
  hero: string
  role: string
  kills: number
  deaths: number
  assists: number
  kda: number
  games: number
  wins: number
  losses: number
  winRate: number
  avgKills: number
  avgDeaths: number
  avgAssists: number
}

export interface TemporalPattern {
  hour: number
  winRate: number
  games: number
  wins: number
  losses: number
}

export interface DayOfWeekPattern {
  day: string
  dayNumber: number
  winRate: number
  games: number
  wins: number
  losses: number
}

export interface StatisticsSummary {
  // Time series
  winRateOverTime: TimeSeriesPoint[]
  gamesOverTime: TimeSeriesPoint[]

  // Streaks
  currentStreak: Streak | null
  longestWinStreak: Streak | null
  longestLossStreak: Streak | null
  allStreaks: Streak[]

  // KDA
  overallKDA: {
    kills: number
    deaths: number
    assists: number
    kda: number
    avgKills: number
    avgDeaths: number
    avgAssists: number
    totalGames: number
  }
  kdaByHero: HeroKDA[]

  // Temporal patterns
  hourlyPerformance: TemporalPattern[]
  dailyPerformance: DayOfWeekPattern[]

  // Consistency metrics
  winRateVariance: number
  consistencyScore: number

  // Recent form
  last10WinRate: number
  last20WinRate: number
  last50WinRate: number
}

/**
 * Calculate time series data from replays
 * Note: Expects replays to already be sorted by date for optimal performance
 */
export function calculateTimeSeries(
  replays: ReplayData[],
  granularity: 'daily' | 'weekly' | 'monthly' = 'daily'
): TimeSeriesPoint[] {
  if (replays.length === 0) return []

  // Group by time period (replays should already be sorted)
  const grouped = new Map<string, { wins: number; losses: number; games: number }>()

  replays.forEach(replay => {
    const date = new Date(replay.date)
    let key: string

    if (granularity === 'daily') {
      key = date.toISOString().split('T')[0] // YYYY-MM-DD
    } else if (granularity === 'weekly') {
      const weekStart = new Date(date)
      weekStart.setDate(date.getDate() - date.getDay()) // Start of week
      key = weekStart.toISOString().split('T')[0]
    } else {
      key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}` // YYYY-MM
    }

    if (!grouped.has(key)) {
      grouped.set(key, { wins: 0, losses: 0, games: 0 })
    }

    const stats = grouped.get(key)!
    stats.games++
    if (replay.result === 'win') {
      stats.wins++
    } else {
      stats.losses++
    }
  })

  // Convert to array and calculate win rates
  return Array.from(grouped.entries())
    .map(([date, stats]) => ({
      date,
      winRate: stats.games > 0 ? (stats.wins / stats.games) * 100 : 0,
      games: stats.games,
      wins: stats.wins,
      losses: stats.losses,
    }))
    .sort((a, b) => a.date.localeCompare(b.date))
}

/**
 * Detect streaks in replay history
 */
export function detectStreaks(replays: ReplayData[]): {
  currentStreak: Streak | null
  longestWinStreak: Streak | null
  longestLossStreak: Streak | null
  allStreaks: Streak[]
} {
  if (replays.length === 0) {
    return {
      currentStreak: null,
      longestWinStreak: null,
      longestLossStreak: null,
      allStreaks: [],
    }
  }

  // Replays should already be sorted descending (most recent first) from parent function
  // No need to re-sort for performance

  const streaks: Streak[] = []
  let currentStreakType: 'win' | 'loss' | null = null
  let currentStreakLength = 0
  let currentStreakStart = ''
  let currentStreakEnd = ''

  // Process replays from most recent to oldest
  replays.forEach((replay, index) => {
    if (currentStreakType === null) {
      // Start new streak
      currentStreakType = replay.result
      currentStreakLength = 1
      currentStreakStart = replay.date
      currentStreakEnd = replay.date
    } else if (currentStreakType === replay.result) {
      // Continue current streak
      currentStreakLength++
      currentStreakStart = replay.date // Older date becomes start
    } else {
      // End current streak and save it
      if (currentStreakLength >= 2) {
        streaks.push({
          type: currentStreakType,
          length: currentStreakLength,
          startDate: currentStreakStart,
          endDate: currentStreakEnd,
          current: index === currentStreakLength // Is this the current streak?
        })
      }

      // Start new streak
      currentStreakType = replay.result
      currentStreakLength = 1
      currentStreakStart = replay.date
      currentStreakEnd = replay.date
    }
  })

  // Don't forget the last streak
  if (currentStreakLength >= 2 && currentStreakType) {
    streaks.push({
      type: currentStreakType,
      length: currentStreakLength,
      startDate: currentStreakStart,
      endDate: currentStreakEnd,
      current: true // Last processed is current
    })
  }

  // Find longest streaks
  const winStreaks = streaks.filter(s => s.type === 'win')
  const lossStreaks = streaks.filter(s => s.type === 'loss')

  const longestWinStreak = winStreaks.length > 0
    ? winStreaks.reduce((max, s) => s.length > max.length ? s : max)
    : null

  const longestLossStreak = lossStreaks.length > 0
    ? lossStreaks.reduce((max, s) => s.length > max.length ? s : max)
    : null

  const currentStreak = streaks.find(s => s.current) || null

  return {
    currentStreak,
    longestWinStreak,
    longestLossStreak,
    allStreaks: streaks,
  }
}

/**
 * Calculate KDA statistics from replays
 */
export function calculateKDAStats(replays: ReplayData[], playerData: PlayerData): {
  overall: {
    kills: number
    deaths: number
    assists: number
    kda: number
    games: number
    avgKills: number
    avgDeaths: number
    avgAssists: number
    totalGames: number
  }
  byHero: HeroKDA[]
} {
  if (replays.length === 0) {
    return {
      overall: {
        kills: 0,
        deaths: 0,
        assists: 0,
        kda: 0,
        games: 0,
        avgKills: 0,
        avgDeaths: 0,
        avgAssists: 0,
        totalGames: 0,
      },
      byHero: [],
    }
  }

  // Overall KDA
  const totalKills = replays.reduce((sum, r) => sum + r.kills, 0)
  const totalDeaths = replays.reduce((sum, r) => sum + r.deaths, 0)
  const totalAssists = replays.reduce((sum, r) => sum + r.assists, 0)
  const games = replays.length

  const overallKDA = totalDeaths > 0 ? (totalKills + totalAssists) / totalDeaths : totalKills + totalAssists

  // KDA by hero
  const heroMap = new Map<string, {
    kills: number
    deaths: number
    assists: number
    games: number
    wins: number
  }>()

  replays.forEach(replay => {
    if (!heroMap.has(replay.hero)) {
      heroMap.set(replay.hero, { kills: 0, deaths: 0, assists: 0, games: 0, wins: 0 })
    }

    const stats = heroMap.get(replay.hero)!
    stats.kills += replay.kills
    stats.deaths += replay.deaths
    stats.assists += replay.assists
    stats.games++
    if (replay.result === 'win') stats.wins++
  })

  const kdaByHero: HeroKDA[] = Array.from(heroMap.entries())
    .map(([hero, stats]) => {
      const kda = stats.deaths > 0
        ? (stats.kills + stats.assists) / stats.deaths
        : stats.kills + stats.assists

      // Get role from playerData
      const heroData = playerData.heroStats.find(h => h.hero === hero)
      const role = heroData?.role || 'Unknown'

      return {
        hero,
        role,
        kills: stats.kills,
        deaths: stats.deaths,
        assists: stats.assists,
        kda,
        games: stats.games,
        wins: stats.wins,
        losses: stats.games - stats.wins,
        winRate: stats.games > 0 ? (stats.wins / stats.games) * 100 : 0,
        avgKills: stats.games > 0 ? stats.kills / stats.games : 0,
        avgDeaths: stats.games > 0 ? stats.deaths / stats.games : 0,
        avgAssists: stats.games > 0 ? stats.assists / stats.games : 0,
      }
    })
    .sort((a, b) => b.kda - a.kda) // Sort by KDA descending

  return {
    overall: {
      kills: totalKills,
      deaths: totalDeaths,
      assists: totalAssists,
      kda: overallKDA,
      games,
      avgKills: games > 0 ? totalKills / games : 0,
      avgDeaths: games > 0 ? totalDeaths / games : 0,
      avgAssists: games > 0 ? totalAssists / games : 0,
      totalGames: games,
    },
    byHero: kdaByHero,
  }
}

/**
 * Calculate temporal patterns (performance by time of day/day of week)
 */
export function calculateTemporalPatterns(replays: ReplayData[]): {
  hourly: TemporalPattern[]
  daily: DayOfWeekPattern[]
} {
  if (replays.length === 0) {
    return { hourly: [], daily: [] }
  }

  // By hour
  const hourlyMap = new Map<number, { wins: number; games: number }>()
  for (let i = 0; i < 24; i++) {
    hourlyMap.set(i, { wins: 0, games: 0 })
  }

  // By day of week
  const dailyMap = new Map<number, { wins: number; games: number }>()
  for (let i = 0; i < 7; i++) {
    dailyMap.set(i, { wins: 0, games: 0 })
  }

  replays.forEach(replay => {
    const date = new Date(replay.date)
    const hour = date.getHours()
    const dayOfWeek = date.getDay()

    // Hour
    const hourStats = hourlyMap.get(hour)!
    hourStats.games++
    if (replay.result === 'win') hourStats.wins++

    // Day
    const dayStats = dailyMap.get(dayOfWeek)!
    dayStats.games++
    if (replay.result === 'win') dayStats.wins++
  })

  const hourly: TemporalPattern[] = Array.from(hourlyMap.entries())
    .map(([hour, stats]) => ({
      hour,
      winRate: stats.games > 0 ? (stats.wins / stats.games) * 100 : 0,
      games: stats.games,
      wins: stats.wins,
      losses: stats.games - stats.wins,
    }))
    .sort((a, b) => a.hour - b.hour) // Keep all 24 hours for heatmap display

  const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
  const daily: DayOfWeekPattern[] = Array.from(dailyMap.entries())
    .map(([dayNumber, stats]) => ({
      day: dayNames[dayNumber],
      dayNumber,
      winRate: stats.games > 0 ? (stats.wins / stats.games) * 100 : 0,
      games: stats.games,
      wins: stats.wins,
      losses: stats.games - stats.wins,
    }))
    .sort((a, b) => a.dayNumber - b.dayNumber) // Keep all 7 days for heatmap display

  return { hourly, daily }
}

/**
 * Calculate consistency score (0-100)
 * Lower variance = higher consistency
 */
export function calculateConsistencyScore(winRates: number[]): number {
  if (winRates.length === 0) return 0

  const mean = winRates.reduce((sum, wr) => sum + wr, 0) / winRates.length
  const variance = winRates.reduce((sum, wr) => sum + Math.pow(wr - mean, 2), 0) / winRates.length
  const stdDev = Math.sqrt(variance)

  // Convert to 0-100 scale (lower std dev = higher score)
  // Assume 0-25% std dev maps to 100-0 consistency score
  const maxStdDev = 25
  const consistencyScore = Math.max(0, Math.min(100, 100 - (stdDev / maxStdDev) * 100))

  return Math.round(consistencyScore)
}

/**
 * Calculate recent form win rates
 * Note: Expects replays to already be sorted descending (newest first)
 */
export function calculateRecentForm(replays: ReplayData[]): {
  last10: number
  last20: number
  last50: number
} {
  // Replays should already be sorted descending (most recent first)

  const calculateWinRate = (games: number): number => {
    const subset = replays.slice(0, games)
    if (subset.length === 0) return 0
    const wins = subset.filter(r => r.result === 'win').length
    return (wins / subset.length) * 100
  }

  return {
    last10: calculateWinRate(10),
    last20: calculateWinRate(20),
    last50: calculateWinRate(50),
  }
}

/**
 * Generate comprehensive statistics from replays and player data
 */
export function generateStatistics(
  replays: ReplayData[],
  playerData: PlayerData
): StatisticsSummary {
  console.log('🔍 Generating statistics for', replays.length, 'replays')

  // PERFORMANCE: Limit to most recent 500 games for faster processing
  // Sort by date descending and take the most recent
  const sortedReplays = [...replays].sort((a, b) =>
    new Date(b.date).getTime() - new Date(a.date).getTime()
  )
  const recentReplays = sortedReplays.slice(0, 500)

  console.log('📊 Processing', recentReplays.length, 'most recent games')

  const winRateOverTime = calculateTimeSeries(recentReplays, 'daily')
  console.log('✅ Time series calculated')

  const gamesOverTime = winRateOverTime // Same data, different visualization focus

  const streakData = detectStreaks(recentReplays)
  console.log('✅ Streaks detected')

  const kdaData = calculateKDAStats(recentReplays, playerData)
  console.log('✅ KDA calculated')

  const temporalData = calculateTemporalPatterns(recentReplays)
  console.log('✅ Temporal patterns calculated')

  const recentForm = calculateRecentForm(recentReplays)
  console.log('✅ Recent form calculated')

  // Calculate win rate variance for consistency
  const winRates = winRateOverTime.map(p => p.winRate)
  const mean = winRates.length > 0 ? winRates.reduce((s, w) => s + w, 0) / winRates.length : 0
  const winRateVariance = winRates.length > 0
    ? winRates.reduce((sum, wr) => sum + Math.pow(wr - mean, 2), 0) / winRates.length
    : 0

  const consistencyScore = calculateConsistencyScore(winRates)
  console.log('✅ Consistency score calculated')

  console.log('🎉 Statistics generation complete!')

  return {
    winRateOverTime,
    gamesOverTime,
    currentStreak: streakData.currentStreak,
    longestWinStreak: streakData.longestWinStreak,
    longestLossStreak: streakData.longestLossStreak,
    allStreaks: streakData.allStreaks,
    overallKDA: kdaData.overall,
    kdaByHero: kdaData.byHero,
    hourlyPerformance: temporalData.hourly,
    dailyPerformance: temporalData.daily,
    winRateVariance,
    consistencyScore,
    last10WinRate: recentForm.last10,
    last20WinRate: recentForm.last20,
    last50WinRate: recentForm.last50,
  }
}
