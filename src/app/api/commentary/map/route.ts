import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { createMapPayload } from '@/lib/utils/commentary'
import { formatCommentary } from '@/lib/utils/server-commentary'
import { PlayerData, MapStats } from '@/types'
import { getMapKnowledge, hasMapKnowledge, getHeroMapSynergy } from '@/lib/data/map-knowledge'

export async function POST(req: Request) {
  try {
    // Parse request body
    const { mapName, playerData } = await req.json() as {
      mapName: string
      playerData: PlayerData
    }

    if (!mapName || !playerData) {
      return NextResponse.json(
        { error: 'Missing mapName or playerData' },
        { status: 400 }
      )
    }

    // Find the map stats
    const mapStats = playerData.mapStats.find(m => m.map === mapName)

    if (!mapStats) {
      return NextResponse.json(
        { error: `Map "${mapName}" not found in player data` },
        { status: 404 }
      )
    }

    // Create the commentary payload
    const payload = createMapPayload(mapStats, playerData)

    // Get map knowledge for strategic advice
    const mapKnowledge = getMapKnowledge(mapName)

    // Build map knowledge section
    let mapKnowledgeSection = ''
    if (mapKnowledge) {
      mapKnowledgeSection = `

**${mapName} MAP KNOWLEDGE (Use this to provide specific, strategic advice):**

Map Size: ${mapKnowledge.size} (${mapKnowledge.laneCount} lanes)

Objective: ${mapKnowledge.objective}
- Description: ${mapKnowledge.objectiveDescription}
- Strategy: ${mapKnowledge.objectiveStrategy}
- Timing: ${mapKnowledge.objectiveTiming}

Macro Strategy:
- Early Game (pre-10): ${mapKnowledge.macro.earlyGame}
- Mid Game (10-16): ${mapKnowledge.macro.midGame}
- Late Game (16+): ${mapKnowledge.macro.lateGame}

Camp Importance: ${mapKnowledge.campImportance}
- Timing: ${mapKnowledge.campTiming}

Heroes That Excel on This Map:
${mapKnowledge.goodHeroes.map(cat => `- ${cat.category} (${cat.heroes.join(', ')}): ${cat.reason}`).join('\n')}

Heroes That Struggle:
${mapKnowledge.badHeroes.map(cat => `- ${cat.category} (${cat.heroes.join(', ')}): ${cat.reason}`).join('\n')}

Pro Tips:
${mapKnowledge.tips.map(tip => `- ${tip}`).join('\n')}

IMPORTANT: Use this knowledge to explain WHY their hero choices work/don't work based on MAP MECHANICS, not just restate the stats.`
    }

    // Add hero-specific map synergies
    let heroSynergySection = ''
    if (payload.topHeroes.length > 0 || payload.weakHeroes.length > 0) {
      const heroExplanations: string[] = []

      if (payload.topHeroes.length > 0) {
        heroExplanations.push('\n**Their Successful Heroes (with Mechanical Reasons):**')
        payload.topHeroes.slice(0, 5).forEach(h => {
          const synergy = getHeroMapSynergy(h.hero, mapName)
          heroExplanations.push(`- ${h.hero}: ${h.winRate.toFixed(1)}% (${h.games} games)${synergy ? ` → ${synergy}` : ''}`)
        })
      }

      if (payload.weakHeroes.length > 0) {
        heroExplanations.push('\n**Their Struggling Heroes (with Mechanical Reasons):**')
        payload.weakHeroes.slice(0, 3).forEach(h => {
          const synergy = getHeroMapSynergy(h.hero, mapName)
          heroExplanations.push(`- ${h.hero}: ${h.winRate.toFixed(1)}% (${h.games} games)${synergy ? ` → ${synergy}` : ''}`)
        })
      }

      if (heroExplanations.length > 0) {
        heroSynergySection = `\n${heroExplanations.join('\n')}`
      }
    }

    // Create the OpenAI prompt
    const prompt = `You're coaching a player on how they perform on ${payload.map}. Talk to them directly.

**Their ${payload.map} Stats:**
- Games Played: ${payload.games}
- Record: ${payload.wins}W - ${payload.losses}L
- Win Rate: ${payload.winRate.toFixed(1)}%
- Map Ranking: #${payload.playerContext.mapRank} out of ${payload.playerContext.totalMaps} maps

**Heroes That Work for Them Here:**
${payload.topHeroes.length > 0
  ? payload.topHeroes.map(h => `- ${h.hero}: ${h.winRate.toFixed(1)}% (${h.games} games)`).join('\n')
  : 'Not enough data yet'}

${payload.weakHeroes.length > 0 ? `**Heroes That Don't Work Here:**
${payload.weakHeroes.map(h => `- ${h.hero}: ${h.winRate.toFixed(1)}% (${h.games} games)`).join('\n')}` : ''}

${payload.highPotentialHeroes.length > 0 ? `**Promising Heroes (Need More Games):**
${payload.highPotentialHeroes.map(h => `- ${h.hero}: ${h.winRate.toFixed(1)}% (${h.games} games)`).join('\n')}` : ''}

**Their Overall Stats:**
- Overall Win Rate: ${payload.playerContext.overallWinRate.toFixed(1)}%
- Total Games: ${payload.playerContext.totalGames}
- Best Overall Heroes: ${payload.playerContext.topHeroesOverall.slice(0, 3).map(h => `${h.hero} (${h.winRate.toFixed(1)}%)`).join(', ')}${mapKnowledgeSection}${heroSynergySection}

CRITICAL FORMATTING REQUIREMENTS - You MUST output proper markdown:

WRONG FORMAT (DO NOT DO THIS):
## Your ${payload.map} Performance- Win rate: **${payload.winRate.toFixed(1)}%**

CORRECT FORMAT (DO THIS):
## Your ${payload.map} Performance

- Your **${payload.winRate.toFixed(1)}%** win rate here vs your overall **${payload.playerContext.overallWinRate.toFixed(1)}%**
- You've played **${payload.games}** games here (that's your #${payload.playerContext.mapRank} most played map)
- Quick take: [honest assessment of how they're doing]

## Your Best Picks Here

- Explain WHY each hero succeeds using MAP MECHANICS (objectives, map size, macro requirements)
- Reference specific advantages (e.g., "Nazeebo stacks efficiently on this large map")
- Connect their success to objective strategy
- Provide positioning or rotation tips specific to this map
- Use bold for win rates

## Heroes to Skip

- Explain WHY heroes struggle using MAP MECHANICS (not just "low win rate")
- Reference specific disadvantages (e.g., "Nova struggles on small maps with constant teamfights")
- Suggest alternatives from their hero pool that fit the map better
- Keep it constructive and strategic

## My Coaching Tips for ${payload.map}

- Objective strategy (timing, control, when to fight)
- Macro play (laning, rotations, camp timing)
- Draft priorities based on map requirements and what works for them
- Specific positioning tricks or rotation patterns
- Common mistakes on this map and how to avoid them
- Expert-level tips that most players don't know

MANDATORY RULES:
1. Write ## then the section name, then press ENTER TWICE
2. Then write each bullet point starting with "- " on its own line
3. NEVER write text immediately after ## on the same line
4. Do NOT include the word "markdown" anywhere in your response
5. Always use "you/your" not "the player"
6. Share insider knowledge and pro tips

Talk like their personal coach reviewing their map-specific performance.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach with deep map knowledge. Use "you" and "your" to make it personal.\n\nCRITICAL: Use the MAP KNOWLEDGE provided to explain their performance. Don\'t just restate stats - explain WHY:\n- Why certain heroes excel (map objectives, size, macro requirements)\n- Why others struggle (mechanical disadvantages on this map)\n- Objective strategy and timing (specific to this map)\n- Macro play (camps, rotations, laning patterns)\n- Positioning tricks for this map\'s layout\n\nFocus on STRATEGIC DEPTH and MAP-SPECIFIC REASONING, not generic stat prose.\n\nABSOLUTE REQUIREMENT - YOUR RESPONSE MUST START EXACTLY LIKE THIS:\n\n##<space>Your [Map Name] Performance\n<blank line>\n-<space>Your **X.X%** win rate\n\nNEVER write "Your [Map Name] Performance- Your" - this is WRONG.\nALWAYS write "## Your [Map Name] Performance" then blank line then "- Your **X.X%**" - this is CORRECT.\n\nEach section MUST have ## at the start.\nEach bullet MUST be on its own line starting with "-<space>".\nThere MUST be a blank line between ## headers and bullet points.'
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
    console.error('Error generating map commentary:', error)
    return NextResponse.json(
      { error: 'Failed to generate commentary' },
      { status: 500 }
    )
  }
}
