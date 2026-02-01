import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { formatCommentary } from '@/lib/utils/server-commentary'
import { PlayerData, PartyGroup } from '@/types'

export async function POST(req: Request) {
  try {
    const { group, playerData } = await req.json() as {
      group: PartyGroup
      playerData: PlayerData
    }

    if (!group || !playerData) {
      return NextResponse.json(
        { error: 'Missing required parameters' },
        { status: 400 }
      )
    }

    // Prepare context
    const topHeroes = group.commonHeroes.slice(0, 5)
      .map(h => `${h.hero}: ${h.winRate.toFixed(1)}% (${h.games} games)`)
      .join('\n')

    const bestMaps = group.bestMaps.slice(0, 3)
      .map(m => `${m.map}: ${m.winRate.toFixed(1)}% (${m.games} games)`)
      .join('\n')

    // Format all members' heroes
    const allMembersHeroes = group.memberHeroes
      ? Object.entries(group.memberHeroes)
          .map(([battletag, heroes]) => {
            const displayName = battletag.split('#')[0]
            const heroList = heroes.slice(0, 3)
              .map(h => `${h.hero} (${h.winRate.toFixed(0)}% WR, ${h.games}g)`)
              .join(', ')
            return `- ${displayName}: ${heroList}`
          })
          .join('\n')
      : ''

    // Format top compositions
    const topCompositions = group.compositions
      ? group.compositions.slice(0, 3)
          .map(c => `${c.composition}: ${c.winRate.toFixed(0)}% WR (${c.wins}W-${c.losses}L)`)
          .join('\n')
      : ''

    const partyTypeLabel =
      group.partySize === 2 ? 'duo' :
      group.partySize === 3 ? 'trio' :
      group.partySize === 4 ? 'quadruple' :
      'quintuple (full premade)'

    const membersList = group.displayNames.join(', ')

    const prompt = `You are a professional Heroes of the Storm coach analyzing party group performance.

**Party Type:** ${partyTypeLabel}
**Party Members:** ${membersList}
**Games Played:** ${group.totalGames}
**Combined Win Rate:** ${group.winRate.toFixed(1)}%
**Record:** ${group.totalWins}W - ${group.totalLosses}L

**Player's Solo Stats:**
- Overall Win Rate: ${playerData.overallWinRate.toFixed(1)}%
- Total Games: ${playerData.totalGames}

**Top Heroes by Each Member:**
${allMembersHeroes}

**Most Common Hero Combinations:**
${topCompositions}

**Best Maps for This Party:**
${bestMaps}

CRITICAL FORMATTING REQUIREMENTS - You MUST output proper markdown:

WRONG FORMAT (DO NOT DO THIS):
## Party Performance Overview- Win rate: **${group.winRate.toFixed(1)}%**

CORRECT FORMAT (DO THIS):
## Party Performance Overview

- Win rate: **${group.winRate.toFixed(1)}%** (compare to your solo queue **${playerData.overallWinRate.toFixed(1)}%**)
- Games played: **${group.totalGames}** games (**${group.totalWins}**W-**${group.totalLosses}**L)
- Overall assessment of this party's strength

## Team Dynamics

- Each member's preferred heroes and performance
- Notable hero combinations that work well
- Synergy patterns and role distribution

## Best Maps Together

- Top performing maps (with win rates)
- Why these maps work well for this group

## Recommendations

- Should you queue with this party more?
- Hero compositions to prioritize
- Areas to coordinate on
- Strategic advice for maximizing success

MANDATORY RULES:
1. Write ## then the section name, then press ENTER TWICE
2. Then write each bullet point starting with "- " on its own line
3. NEVER write text immediately after ## on the same line
4. Do NOT include the word "markdown" anywhere in your response

Use headers, bullets, and bold text for readability. Be specific and data-driven while maintaining an encouraging tone.`

    // Stream response
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach analyzing party dynamics. Provide data-driven, actionable insights about teammate synergy.\n\nABSOLUTE REQUIREMENT - YOUR RESPONSE MUST START EXACTLY LIKE THIS:\n\n##<space>Party Performance Overview\n<blank line>\n-<space>Win rate:\n\nNEVER write "Party Performance Overview- Win rate" - this is WRONG.\nALWAYS write "## Party Performance Overview" then blank line then "- Win rate" - this is CORRECT.\n\nEach section MUST have ## at the start.\nEach bullet MUST be on its own line starting with "-<space>".\nThere MUST be a blank line between ## headers and bullet points.'
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

    // Create SSE stream
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

    return new Response(customReadable, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    })
  } catch (error) {
    console.error('Error generating party group commentary:', error)
    return NextResponse.json(
      { error: 'Failed to generate commentary' },
      { status: 500 }
    )
  }
}
