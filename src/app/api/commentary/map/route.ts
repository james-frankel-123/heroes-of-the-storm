import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { createMapPayload } from '@/lib/utils/commentary'
import { PlayerData, MapStats } from '@/types'

export async function POST(req: Request) {
  try {
    // Parse request body
    const { mapName, playerData } = await req.json() as {
      mapName: string
      playerData: PlayerData
    }

    if (!mapName || !playerData) {
      return NextResponse.json(
        { error: 'Missing mapName or playerData' },
        { status: 400 }
      )
    }

    // Find the map stats
    const mapStats = playerData.mapStats.find(m => m.map === mapName)

    if (!mapStats) {
      return NextResponse.json(
        { error: `Map "${mapName}" not found in player data` },
        { status: 404 }
      )
    }

    // Create the commentary payload
    const payload = createMapPayload(mapStats, playerData)

    // Create the OpenAI prompt
    const prompt = `You are a professional Heroes of the Storm coach providing personalized map analysis.

Analyze this player's performance on ${payload.map}:

**Map Stats:**
- Games Played: ${payload.games}
- Record: ${payload.wins}W - ${payload.losses}L
- Win Rate: ${payload.winRate.toFixed(1)}%
- Map Ranking: #${payload.playerContext.mapRank} out of ${payload.playerContext.totalMaps} maps

**Top Performing Heroes on This Map:**
${payload.topHeroes.length > 0
  ? payload.topHeroes.map(h => `- ${h.hero}: ${h.winRate.toFixed(1)}% (${h.games} games)`).join('\n')
  : 'Not enough data yet'}

${payload.weakHeroes.length > 0 ? `**Underperforming Heroes on This Map:**
${payload.weakHeroes.map(h => `- ${h.hero}: ${h.winRate.toFixed(1)}% (${h.games} games)`).join('\n')}` : ''}

${payload.highPotentialHeroes.length > 0 ? `**High Potential Heroes (Limited Data):**
${payload.highPotentialHeroes.map(h => `- ${h.hero}: ${h.winRate.toFixed(1)}% (${h.games} games)`).join('\n')}` : ''}

**Player Context:**
- Overall Win Rate: ${payload.playerContext.overallWinRate.toFixed(1)}%
- Total Games: ${payload.playerContext.totalGames}
- Best Overall Heroes: ${payload.playerContext.topHeroesOverall.slice(0, 3).map(h => `${h.hero} (${h.winRate.toFixed(1)}%)`).join(', ')}

Provide a concise analysis (2-3 paragraphs) covering:
1. Map performance assessment: How this map compares to their overall performance
2. Hero recommendations: Which of their heroes work best on this map and why
3. Strategic advice: Map-specific tactics or objectives they should focus on
4. Draft strategy: What types of heroes to prioritize on this map based on their strengths

Keep it practical, specific to this player's data, and action-oriented.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach. Provide concise, actionable map-specific advice. Reference the player\'s actual performance data. Be direct and specific.'
        },
        {
          role: 'user',
          content: prompt
        }
      ],
      max_tokens: OPENAI_CONFIG.maxTokens,
      temperature: OPENAI_CONFIG.temperature,
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
    console.error('Error generating map commentary:', error)
    return NextResponse.json(
      { error: 'Failed to generate commentary' },
      { status: 500 }
    )
  }
}
