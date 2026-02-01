import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { createMapPayload } from '@/lib/utils/commentary'
import { formatCommentary } from '@/lib/utils/server-commentary'
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

CRITICAL FORMATTING REQUIREMENTS - You MUST output proper markdown:

WRONG FORMAT (DO NOT DO THIS):
## Map Performance Summary- Win rate: **${payload.winRate.toFixed(1)}%**

CORRECT FORMAT (DO THIS):
## Map Performance Summary

- Win rate: **${payload.winRate.toFixed(1)}%** (compare to overall **${payload.playerContext.overallWinRate.toFixed(1)}%**)
- Games played: **${payload.games}** (ranked #${payload.playerContext.mapRank} of ${payload.playerContext.totalMaps} maps)
- Overall assessment

## Top Performers

- Your best heroes on this map (3-5 heroes)
- Include win rates in bold
- Brief note on why each works well here

## Heroes to Avoid

- Underperforming heroes on this map (if any)
- Performance metrics with explanations

## Strategic Recommendations

- Hero pool suggestions for this map
- Map-specific strategies to focus on
- Draft priorities based on your strengths

MANDATORY RULES:
1. Write ## then the section name, then press ENTER TWICE
2. Then write each bullet point starting with "- " on its own line
3. NEVER write text immediately after ## on the same line
4. Do NOT include the word "markdown" anywhere in your response

Use bullet points and bold text for stats.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach. Provide concise, actionable map-specific advice. Reference the player\'s actual performance data. Be direct and specific.\n\nABSOLUTE REQUIREMENT - YOUR RESPONSE MUST START EXACTLY LIKE THIS:\n\n##<space>Map Performance Summary\n<blank line>\n-<space>Win rate:\n\nNEVER write "Map Performance Summary- Win rate" - this is WRONG.\nALWAYS write "## Map Performance Summary" then blank line then "- Win rate" - this is CORRECT.\n\nEach section MUST have ## at the start.\nEach bullet MUST be on its own line starting with "-<space>".\nThere MUST be a blank line between ## headers and bullet points.'
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

    // Create a readable stream for SSE with formatted content
    const encoder = new TextEncoder()
    const customReadable = new ReadableStream({
      async start(controller) {
        try {
          // Stream the formatted response in chunks
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
    console.error('Error generating map commentary:', error)
    return NextResponse.json(
      { error: 'Failed to generate commentary' },
      { status: 500 }
    )
  }
}
