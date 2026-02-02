import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { formatCommentary } from '@/lib/utils/server-commentary'
import { PlayerData, HeroStats, MapStats } from '@/types'

interface EnrichedContext {
  basic: string
  enriched: string
}

function enrichContextWithData(ctx: any, playerData: PlayerData): EnrichedContext {
  let basicDesc = `- ${ctx.label}: ${ctx.value}`
  if (ctx.type) basicDesc += ` (type: ${ctx.type})`
  if (ctx.trend) basicDesc += ` - trending ${ctx.trend}`
  if (ctx.relatedMetrics) {
    basicDesc += `\n  Related: ${JSON.stringify(ctx.relatedMetrics)}`
  }

  let enrichedData = ''

  // If this is a hero-related stat, find map performance for that hero
  if (ctx.hero || ctx.type === 'hero-row') {
    const heroName = ctx.hero || ctx.label
    const heroData = playerData.heroStats.find(h => h.hero === heroName)

    if (heroData) {
      // Find maps where this hero was played
      const heroMaps = playerData.mapStats
        .map(mapStat => {
          const heroOnMap = mapStat.heroes?.find(h => h.hero === heroName)
          if (heroOnMap) {
            return {
              map: mapStat.map,
              games: heroOnMap.games,
              winRate: heroOnMap.winRate,
              wins: heroOnMap.wins,
              losses: heroOnMap.losses
            }
          }
          return null
        })
        .filter(Boolean)
        .sort((a, b) => (b?.games || 0) - (a?.games || 0)) // Sort by games played

      if (heroMaps.length > 0) {
        enrichedData += `\n\n**${heroName} Performance by Map:**`
        heroMaps.slice(0, 5).forEach(mapData => {
          if (mapData) {
            enrichedData += `\n  - ${mapData.map}: ${mapData.winRate.toFixed(1)}% WR (${mapData.wins}W-${mapData.losses}L, ${mapData.games} games)`
          }
        })
      }

      enrichedData += `\n**${heroName} Overall:** ${heroData.role} | ${heroData.winRate.toFixed(1)}% WR | ${heroData.games} games`
    }
  }

  // If this is a map-related stat, find heroes played on that map
  if (ctx.map || ctx.type === 'map-metric') {
    const mapName = ctx.map || ctx.label
    const mapData = playerData.mapStats.find(m => m.map === mapName)

    if (mapData && mapData.heroes && mapData.heroes.length > 0) {
      // Sort heroes by games played on this map
      const topHeroesOnMap = [...mapData.heroes]
        .sort((a, b) => b.games - a.games)
        .slice(0, 5)

      enrichedData += `\n\n**Heroes Played on ${mapName}:**`
      topHeroesOnMap.forEach(hero => {
        enrichedData += `\n  - ${hero.hero} (${hero.role}): ${hero.winRate.toFixed(1)}% WR (${hero.wins}W-${hero.losses}L, ${hero.games} games)`
      })

      // Show role distribution on this map
      const roleDistribution: Record<string, { games: number; wins: number }> = {}
      mapData.heroes.forEach(hero => {
        if (!roleDistribution[hero.role]) {
          roleDistribution[hero.role] = { games: 0, wins: 0 }
        }
        roleDistribution[hero.role].games += hero.games
        roleDistribution[hero.role].wins += hero.wins
      })

      enrichedData += `\n\n**Role Distribution on ${mapName}:**`
      Object.entries(roleDistribution)
        .sort((a, b) => b[1].games - a[1].games)
        .forEach(([role, stats]) => {
          const winRate = stats.games > 0 ? (stats.wins / stats.games) * 100 : 0
          enrichedData += `\n  - ${role}: ${stats.games} games, ${winRate.toFixed(1)}% WR`
        })
    }
  }

  return { basic: basicDesc, enriched: enrichedData }
}

export async function POST(req: Request) {
  try {
    // Parse request body
    const { userQuestion, contexts, playerData, conversationHistory } = await req.json() as {
      userQuestion: string
      contexts: any[]
      playerData: PlayerData
      conversationHistory?: Array<{ question: string; answer: string }>
    }

    if (!userQuestion || !contexts || !playerData) {
      return NextResponse.json(
        { error: 'Missing userQuestion, contexts, or playerData' },
        { status: 400 }
      )
    }

    // Build enriched context descriptions
    const enrichedContexts = contexts.map(ctx => enrichContextWithData(ctx, playerData))

    const contextsDescription = enrichedContexts
      .map(({ basic, enriched }) => basic + enriched)
      .join('\n')

    // Build role stats summary
    const roleStatsSummary = Object.entries(playerData.roleStats)
      .filter(([_, stats]) => stats.games >= 5) // Only show roles with 5+ games
      .sort((a, b) => b[1].games - a[1].games)
      .map(([role, stats]) => `  - ${role}: ${stats.winRate.toFixed(1)}% WR (${stats.wins}W-${stats.games - stats.wins}L, ${stats.games} games)`)
      .join('\n')

    // Build the prompt
    const prompt = `You're analyzing a player's Heroes of the Storm statistics. The player has clicked on specific stats and is asking a question about them.

**Player's Overall Stats:**
- Player: ${playerData.playerName}
- Total Games: ${playerData.totalGames}
- Overall Win Rate: ${playerData.overallWinRate.toFixed(1)}%
- Total Wins: ${playerData.totalWins}
- Total Losses: ${playerData.totalLosses}

**Performance by Role:**
${roleStatsSummary}

**Statistics the Player Clicked On:**
${contextsDescription}

**Player's Question:**
"${userQuestion}"

${conversationHistory && conversationHistory.length > 0 ? `**Previous Conversation:**
${conversationHistory.map((msg) => `Q: ${msg.question}\nA: ${msg.answer}`).join('\n\n')}` : ''}

**Instructions:**
1. Answer the player's question directly and specifically about the statistics they clicked on
2. Use the enriched data to identify PATTERNS and CORRELATIONS:
   - If they ask about a hero, analyze their map performance to find where they excel/struggle
   - If they ask about a map, analyze which heroes/roles they play there and correlate with win rates
   - Compare clicked stats against their overall performance and role averages
3. Provide SPECIFIC insights based on the data, NOT assumptions:
   - Reference actual numbers (win rates, game counts, maps, heroes)
   - Explain correlations you observe in the data
   - Avoid statements like "your positioning is good" unless you have data to support it
4. Give 2-3 actionable recommendations based on patterns in the data:
   - Suggest playing more of high-WR heroes/maps/roles
   - Recommend avoiding low-WR combinations unless with more practice
   - Identify gaps (e.g., "you rarely play Tank on this map")
5. Be encouraging but honest - if the data shows a weakness, acknowledge it and suggest improvement
6. Use markdown formatting (headers, bullets, bold) but don't mention the word "markdown"
7. Keep it concise (3-4 paragraphs max) unless they ask for more detail

Talk like their coach reviewing their stats together. Use "you" and "your" to make it personal. Focus on DATA-DRIVEN insights, not assumptions.`

    // Create messages array
    const messages = [
      {
        role: 'system' as const,
        content: 'You are an expert Heroes of the Storm performance analyst and coach. You help players understand their statistics and improve their gameplay through data-driven insights. CRITICAL: Base ALL insights on the provided data. Identify patterns and correlations in win rates across heroes, maps, and roles. Never make assumptions about mechanics, positioning, or skill without supporting data. Be encouraging, specific, and actionable in your advice. Use markdown formatting naturally without mentioning it.',
      },
      {
        role: 'user' as const,
        content: prompt,
      },
    ]

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages,
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
            await new Promise((resolve) => setTimeout(resolve, 20))
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
    console.error('Error generating statistics analysis:', error)
    return NextResponse.json(
      { error: 'Failed to generate analysis' },
      { status: 500 }
    )
  }
}
