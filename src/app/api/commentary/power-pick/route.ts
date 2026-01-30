import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { createPowerPickPayload } from '@/lib/utils/commentary'
import { PlayerData, PowerPick } from '@/types'

export async function POST(req: Request) {
  try {
    // Parse request body
    const { powerPick, playerData } = await req.json() as {
      powerPick: PowerPick
      playerData: PlayerData
    }

    if (!powerPick || !playerData) {
      return NextResponse.json(
        { error: 'Missing powerPick or playerData' },
        { status: 400 }
      )
    }

    // Create the commentary payload
    const payload = createPowerPickPayload(powerPick, playerData)

    // Create the OpenAI prompt
    const prompt = `You are a professional Heroes of the Storm coach analyzing a player's power pick combination.

**Power Pick Analysis:**
${payload.hero} on ${payload.map}
- Win Rate: ${payload.winRate.toFixed(1)}% (${payload.games} games)
- Record: ${payload.wins}W - ${payload.losses}L

**Comparative Performance:**

Hero Overall Performance:
- ${payload.hero} (${payload.playerContext.heroRole}): ${payload.playerContext.heroOverallStats.winRate.toFixed(1)}% across all maps (${payload.playerContext.heroOverallStats.games} games)
- Performance Boost on ${payload.map}: +${(payload.winRate - payload.playerContext.heroOverallStats.winRate).toFixed(1)} percentage points

Map Overall Performance:
- ${payload.map}: ${payload.playerContext.mapOverallStats.winRate.toFixed(1)}% across all heroes (${payload.playerContext.mapOverallStats.games} games)

Role Performance:
- ${payload.playerContext.heroRole}: ${payload.playerContext.rolePerformance.winRate.toFixed(1)}% (${payload.playerContext.rolePerformance.games} games)

${payload.playerContext.otherPowerPicks.length > 0 ? `**Other Power Picks:**
${payload.playerContext.otherPowerPicks.map(pp => `- ${pp.hero} on ${pp.map}: ${pp.winRate.toFixed(1)}%`).join('\n')}` : ''}

**Player Context:**
- Overall Win Rate: ${payload.playerContext.overallWinRate.toFixed(1)}%

Provide an analysis (2-3 paragraphs) that:
1. Explains why this hero-map combination works so well for this player
2. Identifies the specific synergies between the hero's kit and the map objectives/layout
3. Compares this power pick to their general performance to highlight what makes it special
4. Provides strategic advice on maximizing this combination (draft priority, talent choices, playstyle focus)

Be specific to THIS player's performance patterns and reference the comparative stats.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach. Analyze power picks by connecting hero kits to map mechanics and explaining why specific combinations work for individual players. Be analytical and strategic.'
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
    console.error('Error generating power pick commentary:', error)
    return NextResponse.json(
      { error: 'Failed to generate commentary' },
      { status: 500 }
    )
  }
}
