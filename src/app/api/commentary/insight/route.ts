import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { createInsightPayload } from '@/lib/utils/commentary'
import { formatCommentary } from '@/lib/utils/server-commentary'
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

CRITICAL FORMATTING REQUIREMENTS - You MUST output proper markdown:

WRONG FORMAT (DO NOT DO THIS):
## Why This Matters- This insight shows

CORRECT FORMAT (DO THIS):
## Why This Matters

- 1-2 sentences explaining the significance
- Impact on your ranked performance

## Evidence from Your Stats

- 2-3 bullet points with specific data
- Use bold for key numbers
- Compare to benchmarks or patterns

## Action Steps

1. Numbered, concrete steps to take
2. Specific and actionable (not generic advice)
3. Prioritized by impact

MANDATORY RULES:
1. Write ## then the section name, then press ENTER TWICE
2. Then write each bullet point starting with "- " on its own line
3. NEVER write text immediately after ## on the same line
4. Do NOT include the word "markdown" anywhere in your response

Keep concise but detailed. Bold key stats. Be encouraging if it's a success insight, constructive if it's a warning.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach. Provide personalized, insight-driven commentary that helps players understand their performance patterns. Be specific, reference their actual data, and offer actionable guidance.\n\nABSOLUTE REQUIREMENT - YOUR RESPONSE MUST START EXACTLY LIKE THIS:\n\n##<space>Why This Matters\n<blank line>\n-<space>This insight\n\nNEVER write "Why This Matters- This insight" - this is WRONG.\nALWAYS write "## Why This Matters" then blank line then "- This insight" - this is CORRECT.\n\nEach section MUST have ## at the start.\nEach bullet MUST be on its own line starting with "-<space>".\nThere MUST be a blank line between ## headers and bullet points.'
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

    // Collect full response to format it
    let fullResponse = ''
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content
      if (content) {
        fullResponse += content
      }
    }

    // Format the complete response on the server
    const formattedResponse = formatCommentary(fullResponse)

    // Create a readable stream for SSE
    const encoder = new TextEncoder()
    const customReadable = new ReadableStream({
      async start(controller) {
        try {
          const chunkSize = 15
          for (let i = 0; i < formattedResponse.length; i += chunkSize) {
            const chunk = formattedResponse.substring(i, i + chunkSize)
            // JSON encode to preserve newlines in SSE format
            const data = `data: ${JSON.stringify(chunk)}\n\n`
            controller.enqueue(encoder.encode(data))
            await new Promise(resolve => setTimeout(resolve, 20))
          }
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
