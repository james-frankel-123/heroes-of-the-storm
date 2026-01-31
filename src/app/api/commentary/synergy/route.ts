import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { PlayerData } from '@/types'
import { DuoStats } from '@/lib/data/team-compositions'

export async function POST(req: Request) {
  try {
    // Parse request body
    const { hero1, hero2, synergy, playerData } = await req.json() as {
      hero1: string
      hero2: string
      synergy: DuoStats
      playerData: PlayerData
    }

    if (!hero1 || !hero2 || !synergy || !playerData) {
      return NextResponse.json(
        { error: 'Missing required parameters' },
        { status: 400 }
      )
    }

    // Get individual hero stats
    const hero1Stats = playerData.heroStats.find(h => h.hero === hero1)
    const hero2Stats = playerData.heroStats.find(h => h.hero === hero2)

    // Create the OpenAI prompt
    const prompt = `You are a professional Heroes of the Storm coach analyzing duo synergies.

Analyze this hero duo's performance:

**Duo:** ${hero1} + ${hero2}

**Duo Performance:**
- Games Together: ${synergy.games}
- Win Rate: ${synergy.winRate.toFixed(1)}%
- Record: ${synergy.wins}W - ${synergy.losses}L

${hero1Stats ? `**${hero1} Individual Stats:**
- Games: ${hero1Stats.games}
- Win Rate: ${hero1Stats.winRate.toFixed(1)}%
- Role: ${hero1Stats.role}
` : ''}

${hero2Stats ? `**${hero2} Individual Stats:**
- Games: ${hero2Stats.games}
- Win Rate: ${hero2Stats.winRate.toFixed(1)}%
- Role: ${hero2Stats.role}
` : ''}

**Player Context:**
- Overall Win Rate: ${playerData.overallWinRate.toFixed(1)}%
- Total Games: ${playerData.totalGames}

Provide a concise analysis (2-3 paragraphs) covering:
1. **Synergy Assessment**: How effective is this duo compared to their individual performances and overall win rate?
2. **Why It Works/Doesn't Work**: Explain the strategic synergy (or anti-synergy) between these heroes' kits and playstyles
3. **Recommendations**: ${synergy.winRate >= 50 ? 'When to prioritize this duo in draft' : 'How to improve or whether to avoid this pairing'}

Be specific and actionable. Reference actual game mechanics and synergies.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach specializing in hero synergies and team composition. Provide specific, actionable advice based on actual game mechanics.'
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
    console.error('Error generating synergy commentary:', error)
    return NextResponse.json(
      { error: 'Failed to generate commentary' },
      { status: 500 }
    )
  }
}
