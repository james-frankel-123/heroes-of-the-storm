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
    const prompt = `You are a professional Heroes of the Storm coach providing personalized analysis.

Analyze this player's performance with ${payload.hero}:

**Hero Stats:**
- Role: ${payload.role}
- Games Played: ${payload.games}
- Record: ${payload.wins}W - ${payload.losses}L
- Win Rate: ${payload.winRate.toFixed(1)}%

**Map Performance:**
${payload.mapPerformance
  .filter(m => m.games >= 3)
  .sort((a, b) => b.winRate - a.winRate)
  .map(m => `- ${m.map}: ${m.winRate.toFixed(1)}% (${m.games} games)`)
  .join('\n') || 'Not enough data per map'}

**Player Context:**
- Overall Win Rate: ${payload.playerContext.overallWinRate.toFixed(1)}%
- Total Games: ${payload.playerContext.totalGames}
- Best Maps: ${payload.playerContext.bestMaps.map(m => `${m.map} (${m.winRate.toFixed(1)}%)`).join(', ')}
${payload.playerContext.topHeroes.length > 0 ? `- Top Heroes: ${payload.playerContext.topHeroes.slice(0, 3).map(h => `${h.hero} (${h.winRate.toFixed(1)}%)`).join(', ')}` : ''}

${payload.playerContext.knownSynergies.length > 0 ? `**Known Team Synergies:**
${payload.playerContext.knownSynergies.slice(0, 3).map(s => `- ${s.hero1} + ${s.hero2}: ${s.winRate.toFixed(1)}% win rate`).join('\n')}` : ''}

CRITICAL FORMATTING REQUIREMENTS - You MUST output proper markdown:

WRONG FORMAT (DO NOT DO THIS):
## Performance Overview- Win rate comparison

CORRECT FORMAT (DO THIS):
## Performance Overview

- Win rate comparison to overall average (use bold for numbers)
- Games played and assessment
- Overall performance summary (one sentence)

## Strengths

- 2-3 bullet points on what's working well with this hero
- Map-specific successes where they excel
- Synergies that amplify this hero

## Areas for Improvement

- 1-2 bullet points on weaknesses or inconsistencies
- Maps where performance dips
- Mechanics or matchups to practice

## Draft Recommendations

- When to prioritize this hero in draft
- Maps to pick/avoid based on their data
- Team compositions where this hero thrives

MANDATORY RULES:
1. Write ## then the section name, then press ENTER TWICE
2. Then write each bullet point starting with "- " on its own line
3. NEVER write text immediately after ## on the same line
4. Do NOT include the word "markdown" anywhere in your response

Keep each section concise and data-driven. Use bold for key stats.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach. Provide concise, actionable advice. Be direct and specific. Avoid generic platitudes.\n\nABSOLUTE REQUIREMENT - YOUR RESPONSE MUST START EXACTLY LIKE THIS:\n\n##<space>Performance Overview\n<blank line>\n-<space>Win rate comparison\n\nNEVER write "Performance Overview- Win rate" - this is WRONG.\nALWAYS write "## Performance Overview" then blank line then "- Win rate" - this is CORRECT.\n\nEach section MUST have ## at the start.\nEach bullet MUST be on its own line starting with "-<space>".\nThere MUST be a blank line between ## headers and bullet points.'
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
