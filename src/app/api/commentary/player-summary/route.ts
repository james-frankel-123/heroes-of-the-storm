import { NextResponse } from 'next/server'
import { PlayerData } from '@/types'

/**
 * Generate a template-based player summary without LLM
 * This provides the same value without API costs or latency
 */
function generateTemplateSummary(playerData: PlayerData): string {
  const playerName = playerData.playerName.split('#')[0]
  const winRate = playerData.overallWinRate.toFixed(1)
  const games = playerData.totalGames

  // Get top hero
  const topHero = playerData.heroStats[0]

  // Get best map
  const bestMap = playerData.mapStats.length > 0
    ? playerData.mapStats[0]
    : null

  // Generate template-based summary based on performance
  if (winRate >= '60.0') {
    if (bestMap && bestMap.winRate >= 65) {
      return `Welcome back, ${playerName}! Your ${topHero.hero} has ${topHero.winRate.toFixed(1)}% win rate, and you're dominating on ${bestMap.map} (${bestMap.winRate.toFixed(1)}%).`
    }
    return `Welcome back, ${playerName}! With ${winRate}% win rate across ${games} games, your ${topHero.hero} performance (${topHero.winRate.toFixed(1)}%) is outstanding.`
  } else if (winRate >= '50.0') {
    return `Welcome back, ${playerName}! You're maintaining a solid ${winRate}% win rate with ${topHero.hero} as your strongest pick (${topHero.winRate.toFixed(1)}%).`
  } else {
    return `Welcome back, ${playerName}! Your ${topHero.hero} shows promise at ${topHero.winRate.toFixed(1)}% win rate—let's build on that foundation.`
  }
}

export async function POST(req: Request) {
  try {
    // Parse request body
    const { playerData } = await req.json() as {
      playerData: PlayerData
    }

    if (!playerData) {
      return NextResponse.json(
        { error: 'Missing playerData' },
        { status: 400 }
      )
    }

    // Generate template-based summary
    const summary = generateTemplateSummary(playerData)

    // Create a readable stream for SSE (maintaining same interface)
    const encoder = new TextEncoder()
    const customReadable = new ReadableStream({
      async start(controller) {
        try {
          // Stream the summary in small chunks to match streaming behavior
          const chunkSize = 5
          for (let i = 0; i < summary.length; i += chunkSize) {
            const chunk = summary.substring(i, i + chunkSize)
            const data = `data: ${chunk}\n\n`
            controller.enqueue(encoder.encode(data))
            // Small delay to maintain streaming feel
            await new Promise(resolve => setTimeout(resolve, 10))
          }
          // Send done signal
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        } catch (error) {
          console.error('Streaming error:', error)
          controller.error(error)
        }
      },
    })

    // Return SSE stream
    return new Response(customReadable, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    })
  } catch (error) {
    console.error('Error generating player summary:', error)
    return NextResponse.json(
      { error: 'Failed to generate summary' },
      { status: 500 }
    )
  }
}
