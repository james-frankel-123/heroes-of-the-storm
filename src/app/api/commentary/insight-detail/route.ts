import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
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

Provide an expanded, actionable analysis (3-4 paragraphs) that:
1. **Explains Why**: Deeper explanation of why this insight matters and what the data reveals
2. **Strategic Impact**: How this affects their ranked performance and draft strategy
3. **Concrete Action Steps**: Specific, numbered steps they can take to capitalize on strengths or address weaknesses
4. **Practice Recommendations**: What to focus on in their next games

Be highly specific with hero names, maps, and game mechanics. Provide actionable advice they can implement immediately.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach. Provide detailed, actionable analysis that helps players improve. Be specific and practical.'
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
    console.error('Error generating insight detail:', error)
    return NextResponse.json(
      { error: 'Failed to generate commentary' },
      { status: 500 }
    )
  }
}
