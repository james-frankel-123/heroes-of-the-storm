import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { createHeroPayload } from '@/lib/utils/commentary'
import { formatCommentary } from '@/lib/utils/server-commentary'
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
    const prompt = `You're coaching a player on their ${payload.hero} performance. Talk to them directly.

**Their ${payload.hero} Stats:**
- Role: ${payload.role}
- Games Played: ${payload.games}
- Record: ${payload.wins}W - ${payload.losses}L
- Win Rate: ${payload.winRate.toFixed(1)}%

**Where They Play ${payload.hero}:**
${payload.mapPerformance
  .filter(m => m.games >= 3)
  .sort((a, b) => b.winRate - a.winRate)
  .map(m => `- ${m.map}: ${m.winRate.toFixed(1)}% (${m.games} games)`)
  .join('\n') || 'Not enough data per map yet'}

**Their Overall Performance:**
- Overall Win Rate: ${payload.playerContext.overallWinRate.toFixed(1)}%
- Total Games: ${payload.playerContext.totalGames}
- Best Maps: ${payload.playerContext.bestMaps.map(m => `${m.map} (${m.winRate.toFixed(1)}%)`).join(', ')}
${payload.playerContext.topHeroes.length > 0 ? `- Top Heroes: ${payload.playerContext.topHeroes.slice(0, 3).map(h => `${h.hero} (${h.winRate.toFixed(1)}%)`).join(', ')}` : ''}

${payload.playerContext.knownSynergies.length > 0 ? `**Duo Synergies You've Played:**
${payload.playerContext.knownSynergies.slice(0, 3).map(s => `- ${s.hero1} + ${s.hero2}: ${s.winRate.toFixed(1)}% win rate`).join('\n')}` : ''}

CRITICAL FORMATTING REQUIREMENTS - You MUST output proper markdown:

WRONG FORMAT (DO NOT DO THIS):
## How You're Doing with ${payload.hero}- Your **${payload.winRate.toFixed(1)}%** win rate

CORRECT FORMAT (DO THIS):
## How You're Doing with ${payload.hero}

- Your **${payload.winRate.toFixed(1)}%** win rate compares to your overall **${payload.playerContext.overallWinRate.toFixed(1)}%**
- You've played **${payload.games}** games - that's enough to see real patterns
- Quick take: [give them an honest, encouraging assessment in one line]

## What's Working

- Point out 2-3 things they're doing well with this hero
- Mention specific maps where they crush it
- If they have good duo partners, highlight that
- Be specific and encouraging

## Where You Can Improve

- 1-2 concrete things to work on
- Maps where the win rate drops and why
- Specific mechanics or matchups to practice
- Keep it constructive - focus on growth

## My Draft Advice for You

- When to instalock this hero (be specific about situations)
- Maps where you should definitely pick it
- Maps to maybe avoid or be careful on
- Team comps where this hero will carry for you
- Include insider tips on positioning or combos

MANDATORY RULES:
1. Write ## then the section name, then press ENTER TWICE
2. Then write each bullet point starting with "- " on its own line
3. NEVER write text immediately after ## on the same line
4. Do NOT include the word "markdown" anywhere in your response
5. Always use "you/your" not "the player"
6. Be encouraging but honest
7. Share expert tips and tricks

Talk like you're their coach sitting next to them reviewing their replay. Be supportive and specific.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach having a one-on-one with your student about their hero. Use "you" and "your" to make it personal. Be encouraging, share insider tips, and give them actionable advice they can use in their next game. Talk like a supportive coach, not a statistics report.\n\nABSOLUTE REQUIREMENT - YOUR RESPONSE MUST START EXACTLY LIKE THIS:\n\n##<space>How You\'re Doing with [Hero Name]\n<blank line>\n-<space>Your **X.X%** win rate\n\nNEVER write "How You\'re Doing- Your" - this is WRONG.\nALWAYS write "## How You\'re Doing with [Hero Name]" then blank line then "- Your **X.X%**" - this is CORRECT.\n\nEach section MUST have ## at the start.\nEach bullet MUST be on its own line starting with "-<space>".\nThere MUST be a blank line between ## headers and bullet points.'
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
          const chunkSize = 15 // Characters per chunk
          for (let i = 0; i < formattedResponse.length; i += chunkSize) {
            const chunk = formattedResponse.substring(i, i + chunkSize)
            // JSON encode to preserve newlines in SSE format
            const data = `data: ${JSON.stringify(chunk)}\n\n`
            controller.enqueue(encoder.encode(data))
            // Small delay to simulate streaming
            await new Promise(resolve => setTimeout(resolve, 20))
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
