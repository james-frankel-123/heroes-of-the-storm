import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { createHeroPayload } from '@/lib/utils/commentary'
import { PlayerData, HeroStats } from '@/types'

export async function POST(req: Request) {
  try {
    // Parse request body
    const { heroName, playerData } = await req.json() as {
      heroName: string
      playerData: PlayerData
    }

    if (!heroName || !playerData) {
      return NextResponse.json(
        { error: 'Missing heroName or playerData' },
        { status: 400 }
      )
    }

    // Find the hero stats
    const heroStats = playerData.heroStats.find(h => h.hero === heroName)

    if (!heroStats) {
      return NextResponse.json(
        { error: `Hero "${heroName}" not found in player data` },
        { status: 404 }
      )
    }

    // Create the commentary payload
    const payload = createHeroPayload(heroStats, playerData)

    // Create the OpenAI prompt
    const prompt = `You are a professional Heroes of the Storm coach providing personalized analysis.

Analyze this player's performance with ${payload.hero}:

**Hero Stats:**
- Role: ${payload.role}
- Games Played: ${payload.games}
- Record: ${payload.wins}W - ${payload.losses}L
- Win Rate: ${payload.winRate.toFixed(1)}%

**Map Performance:**
${payload.mapPerformance
  .filter(m => m.games >= 3)
  .sort((a, b) => b.winRate - a.winRate)
  .map(m => `- ${m.map}: ${m.winRate.toFixed(1)}% (${m.games} games)`)
  .join('\n') || 'Not enough data per map'}

**Player Context:**
- Overall Win Rate: ${payload.playerContext.overallWinRate.toFixed(1)}%
- Total Games: ${payload.playerContext.totalGames}
- Best Maps: ${payload.playerContext.bestMaps.map(m => `${m.map} (${m.winRate.toFixed(1)}%)`).join(', ')}
${payload.playerContext.topHeroes.length > 0 ? `- Top Heroes: ${payload.playerContext.topHeroes.slice(0, 3).map(h => `${h.hero} (${h.winRate.toFixed(1)}%)`).join(', ')}` : ''}

${payload.playerContext.knownSynergies.length > 0 ? `**Known Team Synergies:**
${payload.playerContext.knownSynergies.slice(0, 3).map(s => `- ${s.hero1} + ${s.hero2}: ${s.winRate.toFixed(1)}% win rate`).join('\n')}` : ''}

Provide a concise, actionable analysis (2-3 paragraphs) covering:
1. Performance assessment: How well they're doing with this hero compared to their overall performance
2. Strengths: Which maps or situations where they excel
3. Improvement areas: Specific, actionable advice for increasing win rate
4. Draft recommendations: When to pick this hero based on their playstyle

Keep it conversational, direct, and focused on actionable insights.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach. Provide concise, actionable advice. Be direct and specific. Avoid generic platitudes.'
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
    console.error('Error generating hero commentary:', error)
    return NextResponse.json(
      { error: 'Failed to generate commentary' },
      { status: 500 }
    )
  }
}
