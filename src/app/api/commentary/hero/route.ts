import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { createHeroPayload } from '@/lib/utils/commentary'
import { formatCommentary } from '@/lib/utils/server-commentary'
import { PlayerData, HeroStats } from '@/types'
import { getHeroKnowledge, hasHeroKnowledge } from '@/lib/data/hero-knowledge'
import { getHeroMapSynergy } from '@/lib/data/map-knowledge'

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

    // Get hero knowledge for mechanical/strategic advice
    const heroKnowledge = getHeroKnowledge(heroName)

    // Build hero knowledge section
    let heroKnowledgeSection = ''
    if (heroKnowledge) {
      heroKnowledgeSection = `

**${heroName} MECHANICAL KNOWLEDGE (Use this to provide specific, non-obvious advice):**

Playstyle: ${heroKnowledge.playstyle}

Core Strengths:
${heroKnowledge.strengths.map(s => `- ${s}`).join('\n')}

Core Weaknesses:
${heroKnowledge.weaknesses.map(w => `- ${w}`).join('\n')}

Counters (Heroes that beat ${heroName}):
${heroKnowledge.counters.join(', ')}

Strong Against (Heroes ${heroName} beats):
${heroKnowledge.strongAgainst.join(', ')}

Map Strategy:
- Best Maps: ${heroKnowledge.bestMaps.join(', ')}
- Worst Maps: ${heroKnowledge.worstMaps.join(', ')}
- Why: ${heroKnowledge.mapStrategy}

Positioning: ${heroKnowledge.positioning}

Objective Play: ${heroKnowledge.objectiveValue}

Draft Strategy: ${heroKnowledge.draftStrategy}
Pick Timing: ${heroKnowledge.pickTiming}

${heroKnowledge.abilities ? `Key Abilities:
${heroKnowledge.abilities.map(a => `- ${a.name} (${a.key}): ${a.description}${a.synergies ? `\n  Synergies: ${a.synergies.join(', ')}` : ''}${a.counters ? `\n  Countered by: ${a.counters.join(', ')}` : ''}`).join('\n')}` : ''}

IMPORTANT: Use this knowledge to explain WHY their stats look the way they do based on MECHANICS, not just restate the stats.`
    }

    // Add map-specific strategy explanations
    let mapStrategySection = ''
    if (heroKnowledge && payload.mapPerformance.length > 0) {
      const mapExplanations = payload.mapPerformance
        .filter(m => m.games >= 3)
        .sort((a, b) => b.winRate - a.winRate)
        .slice(0, 5)
        .map(m => {
          const synergy = getHeroMapSynergy(heroName, m.map)
          return `- ${m.map}: ${m.winRate.toFixed(1)}% (${m.games} games)${synergy ? ` → ${synergy}` : ''}`
        })
        .join('\n')

      if (mapExplanations) {
        mapStrategySection = `

**Their Map Performance with Mechanical Context:**
${mapExplanations}`
      }
    }

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
${payload.playerContext.knownSynergies.slice(0, 3).map(s => `- ${s.hero1} + ${s.hero2}: ${s.winRate.toFixed(1)}% win rate`).join('\n')}` : ''}${heroKnowledgeSection}${mapStrategySection}

CRITICAL FORMATTING REQUIREMENTS - You MUST output proper markdown:

WRONG FORMAT (DO NOT DO THIS):
## How You're Doing with ${payload.hero}- Your **${payload.winRate.toFixed(1)}%** win rate

CORRECT FORMAT (DO THIS):
## How You're Doing with ${payload.hero}

- Your **${payload.winRate.toFixed(1)}%** win rate compares to your overall **${payload.playerContext.overallWinRate.toFixed(1)}%**
- You've played **${payload.games}** games - that's enough to see real patterns
- Quick take: [give them an honest, encouraging assessment in one line]

## What's Working

- Explain their success using MECHANICAL reasons (e.g., "Your high win rate on Cursed Hollow makes sense - ${heroName}'s [specific ability] excels on large maps")
- Connect their map performance to hero mechanics (stacking, mobility, teamfight impact)
- Reference specific strengths they're leveraging
- Be specific and encouraging - explain WHY they're succeeding, not just THAT they are

## Where You Can Improve

- Identify mechanical weaknesses they might be struggling with (e.g., "Your low win rate vs dive comps suggests positioning issues")
- Explain map struggles using hero knowledge (e.g., "Braxis is tough for ${heroName} because...")
- Recommend SPECIFIC mechanical improvements (ability usage, positioning, timing)
- Mention counter-matchups they should be aware of
- Keep it constructive with actionable advice

## My Draft Advice for You

- When to pick ${heroName} based on enemy comp and map (use counter/synergy knowledge)
- Specific maps where you should prioritize this pick (with mechanical reasons why)
- Maps to avoid or be cautious on (explain the mechanical disadvantages)
- Team compositions that enable ${heroName} (mention synergies, peel requirements)
- Insider tips on positioning, ability usage, or combos
- Draft timing (early vs late pick) and why

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
          content: 'You are an expert Heroes of the Storm coach with deep game knowledge. Use "you" and "your" to make it personal.\n\nCRITICAL: Use the MECHANICAL KNOWLEDGE provided to explain their performance. Don\'t just restate stats - explain WHY:\n- Why they succeed on certain maps (hero mechanics match map objectives)\n- Why they struggle on others (mechanical disadvantages)\n- What matchups they should avoid (counters)\n- What synergies they should leverage (ability combos)\n- Specific positioning/mechanical tips\n\nFocus on STRATEGIC DEPTH and MECHANICAL REASONING, not generic stat prose.\n\nABSOLUTE REQUIREMENT - YOUR RESPONSE MUST START EXACTLY LIKE THIS:\n\n##<space>How You\'re Doing with [Hero Name]\n<blank line>\n-<space>Your **X.X%** win rate\n\nNEVER write "How You\'re Doing- Your" - this is WRONG.\nALWAYS write "## How You\'re Doing with [Hero Name]" then blank line then "- Your **X.X%**" - this is CORRECT.\n\nEach section MUST have ## at the start.\nEach bullet MUST be on its own line starting with "-<space>".\nThere MUST be a blank line between ## headers and bullet points.'
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
