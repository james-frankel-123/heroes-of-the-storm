import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { createInsightPayload } from '@/lib/utils/commentary'
import { PlayerData, Insight } from '@/types'

export async function POST(req: Request) {
  try {
    // Parse request body
    const { insight, playerData, insightData } = await req.json() as {
      insight: Insight
      playerData: PlayerData
      insightData?: any
    }

    if (!insight || !playerData) {
      return NextResponse.json(
        { error: 'Missing insight or playerData' },
        { status: 400 }
      )
    }

    // Create the commentary payload
    const payload = createInsightPayload(insight, playerData, insightData)

    // Build context based on insight type
    let contextInfo = ''

    if (payload.insightType === 'success' && payload.title.includes('Best Map')) {
      const bestMap = payload.playerData.mapStats.sort((a, b) => b.winRate - a.winRate)[0]
      if (bestMap) {
        const topHeroesOnMap = bestMap.heroes
          ?.filter(h => h.games >= 3)
          .sort((a, b) => b.winRate - a.winRate)
          .slice(0, 3)
        contextInfo = `Best Map: ${bestMap.map} (${bestMap.winRate.toFixed(1)}%, ${bestMap.games} games)
Top Heroes: ${topHeroesOnMap?.map(h => `${h.hero} (${h.winRate.toFixed(1)}%)`).join(', ') || 'N/A'}`
      }
    } else if (payload.insightType === 'success' && payload.title.includes('Best Hero')) {
      const bestHero = payload.playerData.heroStats.sort((a, b) => b.winRate - a.winRate)[0]
      if (bestHero) {
        contextInfo = `Best Hero: ${bestHero.hero} - ${bestHero.role}
Win Rate: ${bestHero.winRate.toFixed(1)}% (${bestHero.games} games)
Record: ${bestHero.wins}W - ${bestHero.losses}L`
      }
    } else if (payload.insightType === 'warning' && payload.title.includes('Weak Role')) {
      const weakRoles = Object.entries(payload.playerData.roleStats)
        .filter(([_, stats]) => stats.winRate < 45 && stats.games >= 50)
        .sort((a, b) => a[1].winRate - b[1].winRate)
      if (weakRoles.length > 0) {
        contextInfo = `Underperforming Roles:\n${weakRoles.map(([role, stats]) =>
          `- ${role}: ${stats.winRate.toFixed(1)}% (${stats.games} games)`
        ).join('\n')}`
      }
    } else if (payload.insightType === 'info' && payload.title.includes('Power Picks')) {
      const powerPicks = payload.playerData.powerPicks.slice(0, 5)
      if (powerPicks.length > 0) {
        contextInfo = `Your Power Picks:\n${powerPicks.map(pp =>
          `- ${pp.hero} on ${pp.map}: ${pp.winRate.toFixed(1)}% (${pp.games} games)`
        ).join('\n')}`
      }
    }

    // Create the OpenAI prompt
    const prompt = `You are a professional Heroes of the Storm coach explaining a key insight about this player's performance.

**Insight:**
Type: ${payload.insightType}
${payload.title}
${payload.description}

**Player Overview:**
- Total Games: ${payload.playerData.totalGames}
- Overall Win Rate: ${payload.playerData.overallWinRate.toFixed(1)}%
- Total Heroes Played: ${payload.playerData.heroStats.length}

${contextInfo ? `**Relevant Details:**
${contextInfo}` : ''}

${payload.playerData.powerPicks.length > 0 ? `**Power Picks Available:**
${payload.playerData.powerPicks.slice(0, 3).map(pp => `- ${pp.hero} on ${pp.map}: ${pp.winRate.toFixed(1)}%`).join('\n')}` : ''}

Provide a personalized explanation (2-3 paragraphs) that:
1. Elaborates on why this insight matters for their specific playstyle
2. Provides concrete examples from their stats to support the insight
3. Offers actionable next steps to leverage this insight or address the issue
4. Connects this insight to their broader performance patterns

Be encouraging if it's a success insight, constructive if it's a warning, and strategic if it's a tip or info.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach. Provide personalized, insight-driven commentary that helps players understand their performance patterns. Be specific, reference their actual data, and offer actionable guidance.'
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
    console.error('Error generating insight commentary:', error)
    return NextResponse.json(
      { error: 'Failed to generate commentary' },
      { status: 500 }
    )
  }
}
