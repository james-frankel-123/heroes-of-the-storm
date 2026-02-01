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
    const prompt = `You're coaching a player about how well they perform with a specific duo. Talk to them directly.

**Their Duo:** ${hero1} + ${hero2}

**How They Do Together:**
- Games Together: ${synergy.games}
- Win Rate: ${synergy.winRate.toFixed(1)}%
- Record: ${synergy.wins}W - ${synergy.losses}L

${hero1Stats ? `**Their ${hero1} Stats:**
- Games: ${hero1Stats.games}
- Win Rate: ${hero1Stats.winRate.toFixed(1)}%
- Role: ${hero1Stats.role}
` : ''}

${hero2Stats ? `**Their ${hero2} Stats:**
- Games: ${hero2Stats.games}
- Win Rate: ${hero2Stats.winRate.toFixed(1)}%
- Role: ${hero2Stats.role}
` : ''}

**Their Overall Performance:**
- Overall Win Rate: ${playerData.overallWinRate.toFixed(1)}%
- Total Games: ${playerData.totalGames}

CRITICAL FORMATTING REQUIREMENTS - You MUST output proper markdown:

WRONG FORMAT (DO NOT DO THIS):
## How This Duo Works for You- Combined win rate: **${synergy.winRate.toFixed(1)}%**

CORRECT FORMAT (DO THIS):
## How This Duo Works for You

- You're at **${synergy.winRate.toFixed(1)}%** with this combo (**${synergy.wins}**W-**${synergy.losses}**L)
- You've played **${synergy.games}** games together
- Compare that to their individual win rates
- Quick take: [honest assessment of the synergy]

## Why This Pairing ${synergy.winRate >= 50 ? 'Works' : 'Struggles'}

- Explain the kit interactions and ability combos
- Point out complementary strengths
- Mention timing windows and power spikes
- Include insider tips on how to execute combos
- ${synergy.winRate >= 50 ? 'Celebrate what they\'re doing right' : 'Explain why it\'s tough and offer alternatives'}

## My Draft Advice

- When to lock in this duo (or when to avoid it)
- Maps where this synergy shines
- Team comps to build around these two
- Enemy comps that counter you
- Specific strategies to maximize the partnership

MANDATORY RULES:
1. Write ## then the section name, then press ENTER TWICE
2. Then write each bullet point starting with "- " on its own line
3. NEVER write text immediately after ## on the same line
4. Do NOT include the word "markdown" anywhere in your response
5. Always use "you/your" not "the player"
6. Be encouraging even if the synergy isn't great
7. Share advanced combo techniques

Talk like you're reviewing their duo queue replays together.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach helping your student understand their duo synergies. Use "you" and "your" to make it personal. Share advanced combo techniques and positioning tips. Be encouraging about good synergies and constructive about challenging ones.\n\nABSOLUTE REQUIREMENT - YOUR RESPONSE MUST START EXACTLY LIKE THIS:\n\n##<space>How This Duo Works for You\n<blank line>\n-<space>You\'re at **X.X%**\n\nNEVER write "How This Duo Works for You- You\'re at" - this is WRONG.\nALWAYS write "## How This Duo Works for You" then blank line then "- You\'re at **X.X%**" - this is CORRECT.\n\nEach section MUST have ## at the start.\nEach bullet MUST be on its own line starting with "-<space>".\nThere MUST be a blank line between ## headers and bullet points.'
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
