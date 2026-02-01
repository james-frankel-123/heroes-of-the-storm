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
    const prompt = `You're coaching a player about one of their best hero+map combinations. Talk to them directly.

**Their Power Pick:**
${payload.hero} on ${payload.map}
- Win Rate: ${payload.winRate.toFixed(1)}% (${payload.games} games)
- Record: ${payload.wins}W - ${payload.losses}L

**How This Compares:**

Their ${payload.hero} Overall:
- ${payload.hero} (${payload.playerContext.heroRole}): ${payload.playerContext.heroOverallStats.winRate.toFixed(1)}% across all maps (${payload.playerContext.heroOverallStats.games} games)
- Performance Boost on ${payload.map}: **+${(payload.winRate - payload.playerContext.heroOverallStats.winRate).toFixed(1)}** percentage points!

Their ${payload.map} Overall:
- ${payload.map}: ${payload.playerContext.mapOverallStats.winRate.toFixed(1)}% across all heroes (${payload.playerContext.mapOverallStats.games} games)

Their ${payload.playerContext.heroRole} Role:
- ${payload.playerContext.heroRole}: ${payload.playerContext.rolePerformance.winRate.toFixed(1)}% (${payload.playerContext.rolePerformance.games} games)

${payload.playerContext.otherPowerPicks.length > 0 ? `**Their Other Power Picks:**
${payload.playerContext.otherPowerPicks.map(pp => `- ${pp.hero} on ${pp.map}: ${pp.winRate.toFixed(1)}%`).join('\n')}` : ''}

**Their Overall Stats:**
- Overall Win Rate: ${payload.playerContext.overallWinRate.toFixed(1)}%

CRITICAL FORMATTING REQUIREMENTS - You MUST output proper markdown:

WRONG FORMAT (DO NOT DO THIS):
## Why This is Your Power Pick- Win rate: **${payload.winRate.toFixed(1)}%**

CORRECT FORMAT (DO THIS):
## Why This is Your Power Pick

- You're crushing it at **${payload.winRate.toFixed(1)}%** on ${payload.map} (**${payload.wins}**W-**${payload.losses}**L)
- That's **+${(payload.winRate - payload.playerContext.heroOverallStats.winRate).toFixed(1)}** points higher than your usual ${payload.hero} games
- Quick take: [celebrate their success and explain why this combo works]

## What Makes This So Good

- Explain how ${payload.hero}'s kit synergizes with ${payload.map}'s mechanics
- Point out specific advantages they're leveraging
- Mention objective control or map pressure capabilities
- Include insider tips on rotations or positioning

## How to Dominate Even More

- Team comps where this pick becomes unstoppable
- Timing windows and power spikes to abuse
- Common matchups and how to handle them
- Advanced tactics or combos most players miss

## Draft Strategy

- When to instalock this in draft (be specific)
- How to maximize impact based on their playstyle
- Enemy comps to watch out for
- Situations where even this power pick might struggle

MANDATORY RULES:
1. Write ## then the section name, then press ENTER TWICE
2. Then write each bullet point starting with "- " on its own line
3. NEVER write text immediately after ## on the same line
4. Do NOT include the word "markdown" anywhere in your response
5. Always use "you/your" not "the player"
6. Be celebratory about their success while giving tactical depth

This is their signature move - help them master it completely.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm coach celebrating your student\'s power pick. Use "you" and "your" to make it personal. Be enthusiastic about their success while giving them tactical depth to dominate even more. Share advanced techniques and insider knowledge.\n\nABSOLUTE REQUIREMENT - YOUR RESPONSE MUST START EXACTLY LIKE THIS:\n\n##<space>Why This is Your Power Pick\n<blank line>\n-<space>You\'re crushing it\n\nNEVER write "Why This is Your Power Pick- You\'re crushing" - this is WRONG.\nALWAYS write "## Why This is Your Power Pick" then blank line then "- You\'re crushing it" - this is CORRECT.\n\nEach section MUST have ## at the start.\nEach bullet MUST be on its own line starting with "-<space>".\nThere MUST be a blank line between ## headers and bullet points.'
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
