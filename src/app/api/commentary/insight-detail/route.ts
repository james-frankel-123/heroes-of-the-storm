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
    const prompt = `You are a professional Heroes of the Storm coach having a one-on-one session with your student.

You're discussing this insight about their gameplay:
**"${insight.title}"**
${insight.description}

**Your Student's Stats:**
- Overall Win Rate: ${playerData.overallWinRate.toFixed(1)}%
- Total Games: ${playerData.totalGames}
- Record: ${playerData.totalWins}W - ${playerData.totalLosses}L

**Their Top Heroes:**
${topHeroes}

**Their Best Maps:**
${bestMaps}

CRITICAL FORMATTING REQUIREMENTS - You MUST output proper markdown:

WRONG FORMAT (DO NOT DO THIS):
## Let's Break This Down- Here's what I'm seeing

CORRECT FORMAT (DO THIS):
## Let's Break This Down

- Talk directly to them using "you" and "your"
- Explain what this pattern reveals about their playstyle
- Share why this matters for their improvement
- Be encouraging but honest

## Here's What Your Data Shows

- Point out specific trends in their stats
- Compare their performance to help them understand context
- Use their actual numbers to make it real
- Keep it conversational - like you're sitting next to them

## My Coaching Tips for You

1. Give them 3-4 concrete things to practice or try
2. Include insider tips that only experienced players know
3. Suggest specific techniques or positioning tricks
4. Focus on what will make the biggest difference

## What to Expect

- Tell them what improvement looks like
- Be realistic about timelines (usually 20-30 games)
- Give them specific signs of progress to watch for
- Keep them motivated

MANDATORY RULES:
1. Write ## then the section name, then press ENTER TWICE
2. Then write each bullet point starting with "- " on its own line
3. NEVER write text immediately after ## on the same line
4. Do NOT include the word "markdown" anywhere in your response
5. Always use "you/your" not "the player"
6. Be encouraging and supportive while being constructive
7. Include expert tips and insider knowledge when relevant

Write like you're coaching them in person - friendly, direct, and genuinely helpful.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach talking directly to your student. Use "you" and "your" to make it personal. Be encouraging, supportive, and share insider knowledge. Talk like you\'re having a coaching session, not writing a formal report.\n\nABSOLUTE REQUIREMENT - YOUR RESPONSE MUST START EXACTLY LIKE THIS:\n\n##<space>Let\'s Break This Down\n<blank line>\n-<space>Hey, so looking at\n\nNEVER write "Let\'s Break This Down- Hey" - this is WRONG.\nALWAYS write "## Let\'s Break This Down" then blank line then "- Hey, so looking at" - this is CORRECT.\n\nEach section MUST have ## at the start.\nEach bullet MUST be on its own line starting with "-<space>".\nThere MUST be a blank line between ## headers and bullet points.'
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
