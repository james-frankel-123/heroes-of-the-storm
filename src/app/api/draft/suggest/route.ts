import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { PlayerData } from '@/types'

interface DraftSuggestionRequest {
  yourTeam: string[] // Array of hero names already picked
  enemyTeam: string[] // Array of enemy hero names
  bannedHeroes: string[] // Array of banned hero names
  selectedMap: string
  playerData: PlayerData
  activeSlot?: number // Which slot is currently picking (0-4)
}

export async function POST(req: Request) {
  try {
    // Parse request body
    const {
      yourTeam,
      enemyTeam,
      bannedHeroes,
      selectedMap,
      playerData,
      activeSlot
    } = await req.json() as DraftSuggestionRequest

    if (!playerData) {
      return NextResponse.json(
        { error: 'Missing playerData' },
        { status: 400 }
      )
    }

    // Filter out unavailable heroes (already picked or banned)
    const unavailableHeroes = [
      ...yourTeam.filter(h => h && h !== 'Flexible'),
      ...enemyTeam.filter(h => h && h !== ''),
      ...bannedHeroes.filter(h => h && h !== '')
    ]

    // Get available heroes from player's hero pool
    const availableHeroes = playerData.heroStats.filter(
      h => !unavailableHeroes.includes(h.hero)
    )

    // Get map-specific performance if map is selected
    let mapPerformance: { hero: string; winRate: number; games: number }[] = []
    if (selectedMap) {
      const mapStats = playerData.mapStats.find(m => m.map === selectedMap)
      if (mapStats?.heroes) {
        mapPerformance = mapStats.heroes
          .filter(h => !unavailableHeroes.includes(h.hero) && h.games >= 3)
          .map(h => ({ hero: h.hero, winRate: h.winRate, games: h.games }))
          .sort((a, b) => b.winRate - a.winRate)
          .slice(0, 10)
      }
    }

    // Get top overall heroes
    const topHeroes = availableHeroes
      .sort((a, b) => b.winRate - a.winRate)
      .slice(0, 10)
      .map(h => ({ hero: h.hero, role: h.role, winRate: h.winRate, games: h.games }))

    // Analyze current team composition
    const currentTeamRoles = yourTeam
      .filter(h => h && h !== 'Flexible')
      .map(heroName => {
        const heroStats = playerData.heroStats.find(h => h.hero === heroName)
        return heroStats?.role || 'Unknown'
      })

    // Identify role gaps
    const hasRole = (role: string) => currentTeamRoles.includes(role)
    const roleNeeds: string[] = []
    if (!hasRole('Tank')) roleNeeds.push('Tank')
    if (!hasRole('Healer')) roleNeeds.push('Healer')
    if (!hasRole('Ranged Assassin') && !hasRole('Melee Assassin')) roleNeeds.push('Assassin')

    // Get role-specific performance
    const roleStats = Object.entries(playerData.roleStats)
      .map(([role, stats]) => ({ role, winRate: stats.winRate, games: stats.games }))
      .sort((a, b) => b.winRate - a.winRate)

    // Build the prompt
    const prompt = `You are a professional Heroes of the Storm draft coach. Analyze this draft situation and suggest the BEST hero pick for this player RIGHT NOW.

**Current Draft State:**
Map: ${selectedMap || 'Not selected yet'}
Slot Picking: ${activeSlot !== undefined ? `Position ${activeSlot + 1}` : 'Unknown'}

Your Team: ${yourTeam.filter(h => h && h !== 'Flexible').join(', ') || 'None yet'}
Enemy Team: ${enemyTeam.filter(h => h).join(', ') || 'None yet'}
Banned Heroes: ${bannedHeroes.filter(h => h).join(', ') || 'None'}

**Team Composition Analysis:**
Current Roles: ${currentTeamRoles.length > 0 ? currentTeamRoles.join(', ') : 'None'}
Role Needs: ${roleNeeds.length > 0 ? roleNeeds.join(', ') : 'All core roles covered'}

**Player Performance Data:**
Overall Win Rate: ${playerData.overallWinRate.toFixed(1)}%
Total Games: ${playerData.totalGames}

Role Performance (Top 3):
${roleStats.slice(0, 3).map(r => `- ${r.role}: ${r.winRate.toFixed(1)}% (${r.games} games)`).join('\n')}

${selectedMap && mapPerformance.length > 0 ? `Best Heroes on ${selectedMap}:
${mapPerformance.slice(0, 5).map(h => `- ${h.hero}: ${h.winRate.toFixed(1)}% (${h.games} games)`).join('\n')}` : ''}

Top Overall Heroes (Available):
${topHeroes.slice(0, 8).map(h => `- ${h.hero} (${h.role}): ${h.winRate.toFixed(1)}% (${h.games} games)`).join('\n')}

**Your Task:**
Suggest the TOP 3 hero picks for this situation. For each hero:
1. Hero name and role
2. Win rate and experience level with this hero
3. Why it's a good pick right now (team synergy, counters enemy, fills role gap, strong on map)
4. Specific tactical advice for playing this hero in this draft

Format as:
**#1 PICK: [Hero Name]**
[Detailed reasoning - 2-3 sentences]

**#2 PICK: [Hero Name]**
[Detailed reasoning - 2-3 sentences]

**#3 PICK: [Hero Name]**
[Detailed reasoning - 2-3 sentences]

**Draft Strategy Tip:**
[One concise tip about what to prioritize in the next picks or what to watch out for]

Consider:
- Role gaps (critical priority)
- Player's win rates and comfort level with heroes
- Map-specific performance
- Synergies with existing team
- Counters to enemy team
- Player's strongest roles

Be specific and actionable. Reference actual stats.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm draft coach. Provide specific, data-driven hero recommendations. Consider player comfort, team needs, and matchup advantages. Be direct and tactical.'
        },
        {
          role: 'user',
          content: prompt
        }
      ],
      max_tokens: OPENAI_CONFIG.maxTokens,
      temperature: 0.5, // Lower temperature for more consistent draft advice
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
    console.error('Error generating draft suggestions:', error)
    return NextResponse.json(
      { error: 'Failed to generate suggestions' },
      { status: 500 }
    )
  }
}
