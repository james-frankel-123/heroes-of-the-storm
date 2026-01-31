import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { PlayerData } from '@/types'

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

    // Get top performing heroes
    const topHeroes = playerData.heroStats
      .slice(0, 3)
      .map(h => `${h.hero} (${h.winRate.toFixed(1)}%, ${h.games} games)`)
      .join(', ')

    // Get best and worst maps
    const sortedMaps = [...playerData.mapStats].sort((a, b) => b.winRate - a.winRate)
    const bestMap = sortedMaps[0]
    const worstMap = sortedMaps[sortedMaps.length - 1]

    // Get role distribution
    const roleStats = Object.entries(playerData.roleStats)
      .map(([role, stats]) => `${role}: ${stats.games} games (${stats.winRate.toFixed(1)}% WR)`)
      .join(', ')

    // Create the OpenAI prompt
    const prompt = `You are a friendly Heroes of the Storm analyst creating a personalized welcome message.

**Player:** ${playerData.playerName.split('#')[0]}

**Overall Stats:**
- Total Games: ${playerData.totalGames}
- Win Rate: ${playerData.overallWinRate.toFixed(1)}%
- Record: ${playerData.totalWins}W - ${playerData.totalLosses}L

**Top Heroes:**
${topHeroes}

**Map Performance:**
- Best: ${bestMap.map} (${bestMap.winRate.toFixed(1)}%)
- Worst: ${worstMap.map} (${worstMap.winRate.toFixed(1)}%)

**Role Distribution:**
${roleStats}

Generate a single, engaging sentence (15-25 words) that:
1. Acknowledges their playstyle or standout stat
2. Mentions a specific strength (hero, map, or role)
3. Feels personalized and motivating

Examples of the tone:
- "Your ${topHeroes.split(' ')[0]} dominance is impressive—keep crushing it on ${bestMap.map}!"
- "A ${playerData.overallWinRate > 55 ? 'strong' : 'solid'} Storm League contender with exceptional ${bestMap.map} performance."

Make it conversational and specific to THEIR data. Don't use generic phrases.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are a Heroes of the Storm analyst. Create personalized, engaging welcome messages. Be specific, concise, and motivating. Avoid generic phrases.'
        },
        {
          role: 'user',
          content: prompt
        }
      ],
      max_tokens: 100, // Short summary
      temperature: 0.8, // More creative
      stream: true,
    })

    // Create a readable stream for SSE
    const encoder = new TextEncoder()
    const customReadable = new ReadableStream({
      async start(controller) {
        try {
          for await (const chunk of stream) {
            const content = chunk.choices[0]?.delta?.content
            if (content) {
              const data = `data: ${content}\n\n`
              controller.enqueue(encoder.encode(data))
            }
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
