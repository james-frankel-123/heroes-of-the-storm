import { NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'
import { transformHeroStatsData } from '@/lib/data/transform'

export async function GET() {
  try {
    // Read the data file from the data directory
    const dataPath = path.join(process.cwd(), 'data', 'player-stats.json')
    const fileContents = await fs.readFile(dataPath, 'utf8')
    const rawData = JSON.parse(fileContents)

    // Transform the data
    const playerData = transformHeroStatsData(rawData)

    // Return as an object with player name as key
    return NextResponse.json({
      [playerData.playerName]: playerData,
    })
  } catch (error) {
    console.error('Error reading data:', error)

    // Fallback to mock data if file doesn't exist
    return NextResponse.json({
      'AzmoDonTrump#1139': {
        playerName: 'AzmoDonTrump#1139',
        totalGames: 2766,
        totalWins: 1386,
        totalLosses: 1380,
        overallWinRate: 50.1,
        heroStats: [],
        mapStats: [],
        roleStats: {},
      },
    })
  }
}
