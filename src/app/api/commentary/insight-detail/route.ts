import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { formatCommentary } from '@/lib/utils/server-commentary'
import { PlayerData, Insight } from '@/types'

export async function POST(req: Request) {
  try {
    // Parse request body
    const { insight, playerData } = await req.json() as {
      insight: Insight
      playerData: PlayerData
    }

    if (!insight || !playerData) {
      return NextResponse.json(
        { error: 'Missing required parameters' },
        { status: 400 }
      )
    }

    // Get relevant player stats based on insight type
    const topHeroes = playerData.heroStats.slice(0, 5)
      .map(h => `${h.hero}: ${h.winRate.toFixed(1)}% (${h.games} games)`)
      .join('\n')

    const sortedMaps = [...playerData.mapStats].sort((a, b) => b.winRate - a.winRate)
    const bestMaps = sortedMaps.slice(0, 3)
      .map(m => `${m.map}: ${m.winRate.toFixed(1)}% (${m.games} games)`)
      .join('\n')

    // Create the OpenAI prompt
    const prompt = `You are a professional Heroes of the Storm coach providing detailed analysis.

The player received this insight:
**"${insight.title}"**
${insight.description}

**Player Context:**
- Overall Win Rate: ${playerData.overallWinRate.toFixed(1)}%
- Total Games: ${playerData.totalGames}
- Record: ${playerData.totalWins}W - ${playerData.totalLosses}L

**Top Heroes:**
${topHeroes}

**Best Maps:**
${bestMaps}

CRITICAL FORMATTING REQUIREMENTS - You MUST output proper markdown:

WRONG FORMAT (DO NOT DO THIS):
## Deep Dive Analysis- Expanded explanation

CORRECT FORMAT (DO THIS):
## Deep Dive Analysis

- Expanded explanation of the insight
- Why this pattern is significant
- Strategic implications

## Supporting Evidence

- Detailed breakdown of relevant stats
- Comparisons and trends
- Specific examples from your data

## Actionable Recommendations

1. Prioritized action steps
2. Practice recommendations
3. Draft adjustments
4. Mindset or approach shifts

## Expected Impact

- What changes you should see
- Timeline for improvement
- Metrics to track

MANDATORY RULES:
1. Write ## then the section name, then press ENTER TWICE
2. Then write each bullet point starting with "- " on its own line
3. NEVER write text immediately after ## on the same line
4. Do NOT include the word "markdown" anywhere in your response

Use clear sections, bullets, and bold emphasis. Be highly specific with hero names, maps, and game mechanics.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach. Provide detailed, actionable analysis that helps players improve. Be specific and practical.\n\nABSOLUTE REQUIREMENT - YOUR RESPONSE MUST START EXACTLY LIKE THIS:\n\n##<space>Deep Dive Analysis\n<blank line>\n-<space>This pattern\n\nNEVER write "Deep Dive Analysis- This pattern" - this is WRONG.\nALWAYS write "## Deep Dive Analysis" then blank line then "- This pattern" - this is CORRECT.\n\nEach section MUST have ## at the start.\nEach bullet MUST be on its own line starting with "-<space>".\nThere MUST be a blank line between ## headers and bullet points.'
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
    console.error('Error generating insight detail:', error)
    return NextResponse.json(
      { error: 'Failed to generate commentary' },
      { status: 500 }
    )
  }
}
