import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { formatCommentary } from '@/lib/utils/server-commentary'
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

CRITICAL FORMATTING REQUIREMENTS - You MUST output proper markdown:

WRONG FORMAT (DO NOT DO THIS):
## Synergy Performance- Combined win rate: **${synergy.winRate.toFixed(1)}%**

CORRECT FORMAT (DO THIS):
## Synergy Performance

- Combined win rate: **${synergy.winRate.toFixed(1)}%** (**${synergy.wins}**W-**${synergy.losses}**L)
- Games played together: **${synergy.games}**
- Performance vs. individual hero averages

## Why This Synergy Works (or Doesn't)

- 2-3 bullet points explaining the synergy
- Kit interactions and complementary abilities
- Timing windows and power spikes

## Draft Recommendations

- When to prioritize this duo in draft
- Maps where this synergy excels
- Team compositions to build around it
- Potential counters to watch for

MANDATORY RULES:
1. Write ## then the section name, then press ENTER TWICE
2. Then write each bullet point starting with "- " on its own line
3. NEVER write text immediately after ## on the same line
4. Do NOT include the word "markdown" anywhere in your response

Bold key statistics and use bullets for clarity. Reference actual game mechanics.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach specializing in hero synergies and team composition. Provide specific, actionable advice based on actual game mechanics.\n\nABSOLUTE REQUIREMENT - YOUR RESPONSE MUST START EXACTLY LIKE THIS:\n\n##<space>Synergy Performance\n<blank line>\n-<space>Combined win rate:\n\nNEVER write "Synergy Performance- Combined" - this is WRONG.\nALWAYS write "## Synergy Performance" then blank line then "- Combined" - this is CORRECT.\n\nEach section MUST have ## at the start.\nEach bullet MUST be on its own line starting with "-<space>".\nThere MUST be a blank line between ## headers and bullet points.'
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
    console.error('Error generating synergy commentary:', error)
    return NextResponse.json(
      { error: 'Failed to generate commentary' },
      { status: 500 }
    )
  }
}
