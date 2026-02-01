import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { createPowerPickPayload } from '@/lib/utils/commentary'
import { formatCommentary } from '@/lib/utils/server-commentary'
import { PlayerData, PowerPick } from '@/types'

export async function POST(req: Request) {
  try {
    // Parse request body
    const { powerPick, playerData } = await req.json() as {
      powerPick: PowerPick
      playerData: PlayerData
    }

    if (!powerPick || !playerData) {
      return NextResponse.json(
        { error: 'Missing powerPick or playerData' },
        { status: 400 }
      )
    }

    // Create the commentary payload
    const payload = createPowerPickPayload(powerPick, playerData)

    // Create the OpenAI prompt
    const prompt = `You are a professional Heroes of the Storm coach analyzing a player's power pick combination.

**Power Pick Analysis:**
${payload.hero} on ${payload.map}
- Win Rate: ${payload.winRate.toFixed(1)}% (${payload.games} games)
- Record: ${payload.wins}W - ${payload.losses}L

**Comparative Performance:**

Hero Overall Performance:
- ${payload.hero} (${payload.playerContext.heroRole}): ${payload.playerContext.heroOverallStats.winRate.toFixed(1)}% across all maps (${payload.playerContext.heroOverallStats.games} games)
- Performance Boost on ${payload.map}: +${(payload.winRate - payload.playerContext.heroOverallStats.winRate).toFixed(1)} percentage points

Map Overall Performance:
- ${payload.map}: ${payload.playerContext.mapOverallStats.winRate.toFixed(1)}% across all heroes (${payload.playerContext.mapOverallStats.games} games)

Role Performance:
- ${payload.playerContext.heroRole}: ${payload.playerContext.rolePerformance.winRate.toFixed(1)}% (${payload.playerContext.rolePerformance.games} games)

${payload.playerContext.otherPowerPicks.length > 0 ? `**Other Power Picks:**
${payload.playerContext.otherPowerPicks.map(pp => `- ${pp.hero} on ${pp.map}: ${pp.winRate.toFixed(1)}%`).join('\n')}` : ''}

**Player Context:**
- Overall Win Rate: ${payload.playerContext.overallWinRate.toFixed(1)}%

CRITICAL FORMATTING REQUIREMENTS - You MUST output proper markdown:

WRONG FORMAT (DO NOT DO THIS):
## Power Pick Performance- Win rate: **${payload.winRate.toFixed(1)}%**

CORRECT FORMAT (DO THIS):
## Power Pick Performance

- Win rate: **${payload.winRate.toFixed(1)}%** on **${payload.map}** (**${payload.wins}**W-**${payload.losses}**L)
- Compare to hero's overall **${payload.playerContext.heroOverallStats.winRate.toFixed(1)}%** and map's **${payload.playerContext.mapOverallStats.winRate.toFixed(1)}%**
- Games played: **${payload.games}**

## Why This is a Power Pick

- Hero kit synergies with map mechanics (2-3 bullets)
- Specific advantages this hero has on this map
- Objective control or map pressure capabilities

## Strategic Advantages

- Team compositions that amplify this pick
- Timing windows and power spikes
- Common matchups and how to handle them

## Recommendations

- When to pick this in draft
- How to maximize impact on this map
- Red flags or situations to avoid

MANDATORY RULES:
1. Write ## then the section name, then press ENTER TWICE
2. Then write each bullet point starting with "- " on its own line
3. NEVER write text immediately after ## on the same line
4. Do NOT include the word "markdown" anywhere in your response

Bold statistics and use clear sections. Be specific to THIS player's performance patterns.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach. Analyze power picks by connecting hero kits to map mechanics and explaining why specific combinations work for individual players. Be analytical and strategic.\n\nABSOLUTE REQUIREMENT - YOUR RESPONSE MUST START EXACTLY LIKE THIS:\n\n##<space>Power Pick Performance\n<blank line>\n-<space>Win rate:\n\nNEVER write "Power Pick Performance- Win rate" - this is WRONG.\nALWAYS write "## Power Pick Performance" then blank line then "- Win rate" - this is CORRECT.\n\nEach section MUST have ## at the start.\nEach bullet MUST be on its own line starting with "-<space>".\nThere MUST be a blank line between ## headers and bullet points.'
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
    console.error('Error generating power pick commentary:', error)
    return NextResponse.json(
      { error: 'Failed to generate commentary' },
      { status: 500 }
    )
  }
}
