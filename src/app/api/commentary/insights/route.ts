import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { PlayerData } from '@/types'

export async function POST(req: Request) {
  try {
    // Parse request body
    const { playerData } = await req.json() as {
      playerData: PlayerData
    }

    if (!playerData) {
      return NextResponse.json(
        { error: 'Missing playerData' },
        { status: 400 }
      )
    }

    // Prepare data summaries
    const topHeroes = playerData.heroStats
      .slice(0, 5)
      .map(h => `${h.hero}: ${h.winRate.toFixed(1)}% WR, ${h.games} games`)
      .join('\n')

    const sortedMaps = [...playerData.mapStats].sort((a, b) => b.winRate - a.winRate)
    const bestMaps = sortedMaps.slice(0, 3)
      .map(m => `${m.map}: ${m.winRate.toFixed(1)}% (${m.games} games)`)
      .join('\n')
    const worstMaps = sortedMaps.slice(-2)
      .map(m => `${m.map}: ${m.winRate.toFixed(1)}% (${m.games} games)`)
      .join('\n')

    const roleStats = playerData.roleStats
      .map(r => `${r.role}: ${r.winRate.toFixed(1)}% WR, ${r.games} games`)
      .join('\n')

    // Create the OpenAI prompt
    const prompt = `You are a Heroes of the Storm analyst generating smart insights for a player dashboard.

**Player Stats:**
- Battletag: ${playerData.playerName}
- Overall Win Rate: ${playerData.overallWinRate.toFixed(1)}%
- Total Games: ${playerData.totalGames}
- Record: ${playerData.totalWins}W - ${playerData.totalLosses}L

**Top Heroes:**
${topHeroes}

**Best Maps:**
${bestMaps}

**Worst Maps:**
${worstMaps}

**Role Performance:**
${roleStats}

Generate exactly 3-4 actionable insights in JSON format. Each insight should be:
1. Specific to the player's data (not generic advice)
2. Actionable (suggest what they can do to improve)
3. One of these types: 'success', 'warning', or 'info'

Format as a JSON array:
[
  {
    "type": "success" | "warning" | "info",
    "title": "Short title (3-6 words)",
    "description": "One sentence explanation with specific numbers/heroes/maps"
  }
]

Focus on:
- Standout performances (high win rate heroes/maps)
- Problem areas (low win rate heroes/maps/roles)
- Patterns (role preferences, map trends)
- Opportunities for improvement

Examples:
- {"type": "success", "title": "Cursed Hollow Specialist", "description": "You have a 72.5% win rate on Cursed Hollow—keep prioritizing this map in ranked."}
- {"type": "warning", "title": "Tank Role Struggles", "description": "Your tank win rate is 15% below your average—consider practicing Muradin or Johanna."}
- {"type": "info", "title": "Assassin Main", "description": "85% of your games are on assassins—expanding your role pool could improve draft flexibility."}

Return ONLY the JSON array, no other text.`

    // Get completion from OpenAI (non-streaming for structured data)
    const completion = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are a Heroes of the Storm analyst. Generate specific, actionable insights based on player data. Return only valid JSON.'
        },
        {
          role: 'user',
          content: prompt
        }
      ],
      max_tokens: 500,
      temperature: 0.7,
      response_format: { type: 'json_object' },
    })

    const responseText = completion.choices[0]?.message?.content || '{"insights": []}'

    // Parse the JSON response
    let insights
    try {
      const parsed = JSON.parse(responseText)
      // Handle both array and object with insights key
      insights = Array.isArray(parsed) ? parsed : (parsed.insights || [])
    } catch (parseError) {
      console.error('Failed to parse OpenAI response:', responseText)
      insights = []
    }

    return NextResponse.json({ insights })
  } catch (error) {
    console.error('Error generating insights:', error)
    return NextResponse.json(
      { error: 'Failed to generate insights' },
      { status: 500 }
    )
  }
}
