import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { formatCommentary } from '@/lib/utils/server-commentary'
import { PlayerData, HeroStats } from '@/types'

interface RoleStats {
  role: string
  wins: number
  games: number
  winRate: number
}

export async function POST(req: Request) {
  try {
    // Parse request body
    const { role, stats, heroesInRole, playerData } = await req.json() as {
      role: string
      stats: RoleStats
      heroesInRole: HeroStats[]
      playerData: PlayerData
    }

    if (!role || !stats || !playerData) {
      return NextResponse.json(
        { error: 'Missing required parameters' },
        { status: 400 }
      )
    }

    const losses = stats.games - stats.wins

    // Get top and bottom performers in this role
    const topHeroes = heroesInRole
      .filter(h => h.games >= 3)
      .slice(0, 3)
      .map(h => `${h.hero} (${h.winRate.toFixed(1)}%, ${h.games} games)`)
      .join(', ')

    const strugglingHeroes = heroesInRole
      .filter(h => h.games >= 3 && h.winRate < 45)
      .slice(0, 3)
      .map(h => `${h.hero} (${h.winRate.toFixed(1)}%, ${h.games} games)`)
      .join(', ')

    // Create the OpenAI prompt
    const prompt = `You are a professional Heroes of the Storm coach analyzing role performance.

Analyze this player's performance with the ${role} role:

**Role Performance:**
- Games Played: ${stats.games}
- Win Rate: ${stats.winRate.toFixed(1)}%
- Record: ${stats.wins}W - ${losses}L

**Heroes Played in This Role:**
${heroesInRole.slice(0, 5).map(h => `- ${h.hero}: ${h.winRate.toFixed(1)}% (${h.games} games)`).join('\n')}

${topHeroes ? `**Top Performers:**
${topHeroes}
` : ''}

${strugglingHeroes ? `**Struggling Heroes:**
${strugglingHeroes}
` : ''}

**Player Context:**
- Overall Win Rate: ${playerData.overallWinRate.toFixed(1)}%
- Total Games: ${playerData.totalGames}
- Role Performance vs Overall: ${(stats.winRate - playerData.overallWinRate).toFixed(1)}% ${stats.winRate > playerData.overallWinRate ? 'above' : 'below'} average

CRITICAL FORMATTING REQUIREMENTS - You MUST output proper markdown:

WRONG FORMAT (DO NOT DO THIS):
## Role Performance Summary- Win rate: **${stats.winRate.toFixed(1)}%**

CORRECT FORMAT (DO THIS):
## Role Performance Summary

- Win rate: **${stats.winRate.toFixed(1)}%** (**${stats.wins}**W-**${losses}**L in **${stats.games}** games)
- Compare to overall **${playerData.overallWinRate.toFixed(1)}%**
- Performance ranking among your roles

## Top Performers

- Your strongest heroes in this role (with win rates in bold)
- Why each succeeds

## Heroes Needing Practice

- Lower win rate heroes in this role (if any)
- Specific areas to improve

## Recommendations

- Hero pool adjustments for this role
- Situations where this role shines for you
- Skills or matchups to focus on

MANDATORY RULES:
1. Write ## then the section name, then press ENTER TWICE
2. Then write each bullet point starting with "- " on its own line
3. NEVER write text immediately after ## on the same line
4. Do NOT include the word "markdown" anywhere in your response

Use bullets, bold stats, and clear sections. Reference actual game mechanics and role responsibilities.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach specializing in role-specific performance analysis. Provide specific, actionable advice based on actual game mechanics and role responsibilities.\n\nABSOLUTE REQUIREMENT - YOUR RESPONSE MUST START EXACTLY LIKE THIS:\n\n##<space>Role Performance Summary\n<blank line>\n-<space>Win rate:\n\nNEVER write "Role Performance Summary- Win rate" - this is WRONG.\nALWAYS write "## Role Performance Summary" then blank line then "- Win rate" - this is CORRECT.\n\nEach section MUST have ## at the start.\nEach bullet MUST be on its own line starting with "-<space>".\nThere MUST be a blank line between ## headers and bullet points.'
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
    console.error('Error generating role commentary:', error)
    return NextResponse.json(
      { error: 'Failed to generate commentary' },
      { status: 500 }
    )
  }
}
