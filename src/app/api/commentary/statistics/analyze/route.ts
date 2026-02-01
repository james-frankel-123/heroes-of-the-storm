import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { formatCommentary } from '@/lib/utils/server-commentary'
import { PlayerData } from '@/types'

export async function POST(req: Request) {
  try {
    // Parse request body
    const { userQuestion, contexts, playerData, conversationHistory } = await req.json() as {
      userQuestion: string
      contexts: any[]
      playerData: PlayerData
      conversationHistory?: Array<{ question: string; answer: string }>
    }

    if (!userQuestion || !contexts || !playerData) {
      return NextResponse.json(
        { error: 'Missing userQuestion, contexts, or playerData' },
        { status: 400 }
      )
    }

    // Build context description
    const contextsDescription = contexts
      .map((ctx) => {
        let desc = `- ${ctx.label}: ${ctx.value}`
        if (ctx.type) desc += ` (type: ${ctx.type})`
        if (ctx.trend) desc += ` - trending ${ctx.trend}`
        if (ctx.relatedMetrics) {
          desc += `\n  Related: ${JSON.stringify(ctx.relatedMetrics)}`
        }
        return desc
      })
      .join('\n')

    // Build the prompt
    const prompt = `You're analyzing a player's Heroes of the Storm statistics. The player has clicked on specific stats and is asking a question about them.

**Player's Overall Stats:**
- Player: ${playerData.playerName}
- Total Games: ${playerData.totalGames}
- Overall Win Rate: ${playerData.overallWinRate.toFixed(1)}%
- Total Wins: ${playerData.totalWins}
- Total Losses: ${playerData.totalLosses}

**Statistics the Player Clicked On:**
${contextsDescription}

**Player's Question:**
"${userQuestion}"

${conversationHistory && conversationHistory.length > 0 ? `**Previous Conversation:**
${conversationHistory.map((msg) => `Q: ${msg.question}\nA: ${msg.answer}`).join('\n\n')}` : ''}

**Instructions:**
1. Answer the player's question directly and specifically about the statistics they clicked on
2. Provide insights by analyzing the clicked stats in context of their overall performance
3. Give 2-3 actionable recommendations based on the data
4. Be encouraging but honest
5. Use markdown formatting (headers, bullets, bold) but don't mention the word "markdown"
6. Keep it concise (3-4 paragraphs max) unless they ask for more detail

Talk like their coach reviewing their stats together. Use "you" and "your" to make it personal.`

    // Create messages array
    const messages = [
      {
        role: 'system' as const,
        content: 'You are an expert Heroes of the Storm performance analyst and coach. You help players understand their statistics and improve their gameplay through data-driven insights. Be encouraging, specific, and actionable in your advice. Use markdown formatting naturally without mentioning it.',
      },
      {
        role: 'user' as const,
        content: prompt,
      },
    ]

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages,
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
          const chunkSize = 15 // Characters per chunk
          for (let i = 0; i < formattedResponse.length; i += chunkSize) {
            const chunk = formattedResponse.substring(i, i + chunkSize)
            // JSON encode to preserve newlines in SSE format
            const data = `data: ${JSON.stringify(chunk)}\n\n`
            controller.enqueue(encoder.encode(data))
            // Small delay to simulate streaming
            await new Promise((resolve) => setTimeout(resolve, 20))
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
    console.error('Error generating statistics analysis:', error)
    return NextResponse.json(
      { error: 'Failed to generate analysis' },
      { status: 500 }
    )
  }
}
