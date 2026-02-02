import { NextResponse } from 'next/server'
import { PlayerData } from '@/types'
import { generateInsights } from '@/lib/data/transform'

/**
 * Generate insights using rule-based system instead of LLM
 * This is faster, cheaper, and provides the same value
 *
 * The rule-based system computes insights from simple statistical patterns:
 * - Best map/hero performance
 * - Role weaknesses
 * - Overall consistency
 *
 * These insights are trivially computable and don't require LLM synthesis.
 */
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

    // Use rule-based insights generation (no LLM needed)
    const insights = generateInsights(playerData)

    return NextResponse.json({ insights })
  } catch (error) {
    console.error('Error generating insights:', error)
    return NextResponse.json(
      { error: 'Failed to generate insights' },
      { status: 500 }
    )
  }
}
