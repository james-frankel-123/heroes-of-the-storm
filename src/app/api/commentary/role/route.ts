import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
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

Provide a concise analysis (2-3 paragraphs) covering:
1. **Performance Assessment**: How well they perform in this role compared to their overall performance
2. **Strengths & Weaknesses**: Which heroes they excel with and which need improvement
3. **Actionable Recommendations**: Specific advice to improve their ${role} performance, including:
   - Hero pool suggestions (expand or focus)
   - Playstyle adjustments for this role
   - Draft considerations

Be specific and actionable. Reference actual game mechanics and role responsibilities.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach specializing in role-specific performance analysis. Provide specific, actionable advice based on actual game mechanics and role responsibilities.'
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
    console.error('Error generating role commentary:', error)
    return NextResponse.json(
      { error: 'Failed to generate commentary' },
      { status: 500 }
    )
  }
}
