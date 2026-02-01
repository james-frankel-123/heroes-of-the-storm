import { NextResponse } from 'next/server'
import { openai, OPENAI_CONFIG } from '@/lib/api/openai'
import { DraftTurn, DraftTeam, getAvailableHeroes } from '@/lib/draft/draft-sequence'
import { PartyMember } from '@/components/draft/draft-config-modal'
import { calculatePlayerCompetency, PlayerCompetency } from '@/lib/draft/competency-score'
import { matchHeroesToPlayers, formatRecommendationForPrompt } from '@/lib/draft/recommendation-matcher'

interface DraftState {
  selectedMap: string
  yourTeam: DraftTeam
  currentTurn: DraftTurn
  bluePicks: (string | null)[]
  redPicks: (string | null)[]
  blueBans: string[]
  redBans: string[]
}

interface DraftCoachRequest {
  draftState: DraftState
  partyRoster: PartyMember[]
  draftHistory: Array<{
    turn: DraftTurn
    hero: string
    timestamp: number
    battletag?: string
  }>
}

// All 89 HotS heroes
const ALL_HEROES = [
  'Abathur', 'Alarak', 'Alexstrasza', 'Ana', 'Anduin', "Anub'arak",
  'Artanis', 'Arthas', 'Auriel', 'Azmodan', 'Blaze', 'Brightwing',
  'Cassia', 'Chen', 'Cho', 'Chromie', 'D.Va', 'Deckard',
  'Dehaka', 'Diablo', 'E.T.C.', 'Falstad', 'Fenix', 'Gall',
  'Garrosh', 'Gazlowe', 'Genji', 'Greymane', 'Gul\'dan', 'Hanzo',
  'Hogger', 'Illidan', 'Imperius', 'Jaina', 'Johanna', 'Junkrat',
  'Kael\'thas', 'Kel\'Thuzad', 'Kerrigan', 'Kharazim', 'Leoric', 'Li Li',
  'Li-Ming', 'Lt. Morales', 'Lúcio', 'Lunara', 'Maiev', 'Malfurion',
  'Mal\'Ganis', 'Medivh', 'Mei', 'Mephisto', 'Muradin', 'Murky',
  'Nazeebo', 'Nova', 'Orphea', 'Probius', 'Qhira', 'Ragnaros',
  'Raynor', 'Rehgar', 'Rexxar', 'Samuro', 'Sgt. Hammer', 'Sonya',
  'Stitches', 'Stukov', 'Sylvanas', 'Tassadar', 'The Butcher', 'The Lost Vikings',
  'Thrall', 'Tracer', 'Tychus', 'Tyrael', 'Tyrande', 'Uther',
  'Valeera', 'Valla', 'Varian', 'Whitemane', 'Xul', 'Yrel',
  'Zagara', 'Zarya', 'Zeratul', 'Zul\'jin'
].sort()

export async function POST(req: Request) {
  try {
    const { draftState, partyRoster, draftHistory } = await req.json() as DraftCoachRequest

    if (!draftState || !partyRoster) {
      return NextResponse.json(
        { error: 'Missing required data' },
        { status: 400 }
      )
    }

    const { selectedMap, yourTeam, currentTurn, bluePicks, redPicks, blueBans, redBans } = draftState

    // Calculate player competencies
    const playerCompetencies: PlayerCompetency[] = partyRoster
      .filter(member => member.playerStats && !member.playerStats.error)
      .map(member => calculatePlayerCompetency(
        member.battletag,
        member.slot,
        member.playerStats!,
        selectedMap
      ))

    // Get available heroes
    const availableHeroes = getAvailableHeroes(
      ALL_HEROES,
      { blue: blueBans, red: redBans },
      { blue: bluePicks, red: redPicks }
    )

    // Analyze current compositions
    const yourPicks = yourTeam === 'blue' ? bluePicks : redPicks
    const enemyPicks = yourTeam === 'blue' ? redPicks : bluePicks
    const yourFilledPicks = yourPicks.filter(Boolean) as string[]
    const enemyFilledPicks = enemyPicks.filter(Boolean) as string[]

    // Determine role gaps (simplified - could be enhanced with actual role data)
    const analyzeRoles = (picks: string[]) => {
      // This is a simplified version - in production you'd check actual hero roles
      return picks.length
    }

    const yourRoleCount = analyzeRoles(yourFilledPicks)
    const enemyRoleCount = analyzeRoles(enemyFilledPicks)

    // Get recommendations if it's your turn
    let recommendations: any[] = []
    if (currentTurn.team === yourTeam && currentTurn.action === 'pick') {
      // Get role-based hero candidates (simplified - would need actual role data)
      const heroCandidates = availableHeroes.slice(0, 20) // Top candidates

      // Match heroes to players
      const matchResult = matchHeroesToPlayers(
        heroCandidates,
        playerCompetencies,
        [] // Role needs - simplified for now
      )

      recommendations = matchResult.recommendations
        .slice(0, 3)
        .map(rec => formatRecommendationForPrompt(rec))
    }

    // Build draft history narrative
    const historyNarrative = draftHistory.length > 0
      ? draftHistory.map((action, idx) => {
          const turnLabel = `${action.turn.team.toUpperCase()} ${action.turn.action.toUpperCase()} #${action.turn.number}`
          const playerInfo = action.battletag ? ` by ${action.battletag.split('#')[0]}` : ''
          return `${idx + 1}. ${turnLabel}: ${action.hero}${playerInfo}`
        }).join('\n')
      : 'No actions yet (draft just started)'

    // Build the comprehensive prompt
    const isYourTurn = currentTurn.team === yourTeam
    const turnDescription = `${currentTurn.team.toUpperCase()} TEAM ${currentTurn.action.toUpperCase()} #${currentTurn.number}`
    const phaseDescription = currentTurn.phase === 1 ? 'Opening' : currentTurn.phase === 2 ? 'Middle' : 'Final'

    const prompt = `You are an expert Heroes of the Storm draft coach speaking directly to the player through their headset during a live Storm League draft.

**DRAFT STATE:**
- Map: ${selectedMap}
- Your Team: ${yourTeam.toUpperCase()}
- Current Turn: ${turnDescription}
- Phase: ${phaseDescription} (Phase ${currentTurn.phase})

**DRAFT HISTORY:**
${historyNarrative}

**CURRENT COMPOSITION:**
Your Team (${yourTeam === 'blue' ? 'BLUE' : 'RED'}): ${yourFilledPicks.join(', ') || 'None yet'}
Enemy Team (${yourTeam === 'blue' ? 'RED' : 'BLUE'}): ${enemyFilledPicks.join(', ') || 'None yet'}

Your Bans: ${yourTeam === 'blue' ? blueBans.join(', ') : redBans.join(', ')}
Enemy Bans: ${yourTeam === 'blue' ? redBans.join(', ') : blueBans.join(', ')}

**YOUR PARTY ROSTER:**
${partyRoster.map((member, idx) => {
  if (!member.battletag) return `${idx + 1}. [Unknown player] (Slot ${idx + 1})`

  const alreadyPicked = yourPicks[idx]
  if (alreadyPicked) {
    return `${idx + 1}. ${member.battletag.split('#')[0]} (Slot ${idx + 1}) - Already picked: ${alreadyPicked}`
  }

  if (!member.playerStats || member.playerStats.error) {
    return `${idx + 1}. ${member.battletag.split('#')[0]} (Slot ${idx + 1}) - No stats available`
  }

  const comp = playerCompetencies.find(pc => pc.slot === idx)
  if (!comp) {
    return `${idx + 1}. ${member.battletag.split('#')[0]} (Slot ${idx + 1}) - Stats loading...`
  }

  const topHeroes = comp.heroCompetencies
    .slice(0, 3)
    .map(h => `${h.hero} (${h.winRate}% WR, ${h.games}g${h.mapBonus ? ', strong on this map' : ''})`)
    .join(', ')

  return `${idx + 1}. ${member.battletag.split('#')[0]} (Slot ${idx + 1})
   Top Heroes: ${topHeroes}`
}).join('\n\n')}

${recommendations.length > 0 ? `**PLAYER-MATCHED RECOMMENDATIONS:**
${currentTurn.pickSlot !== undefined ? `Next pick is for Slot ${currentTurn.pickSlot + 1} (${partyRoster[currentTurn.pickSlot]?.battletag?.split('#')[0] || 'Unknown'})` : ''}

${recommendations.join('\n\n')}` : ''}

**YOUR TASK:**
${isYourTurn
  ? currentTurn.action === 'ban'
    ? `It's YOUR turn to BAN. Suggest which hero to ban and why. Consider:
- Banning heroes the enemy is strong with
- Removing counters to your intended picks
- Banning OP heroes on this map`
    : `It's YOUR turn to PICK (Slot ${currentTurn.pickSlot! + 1}: ${partyRoster[currentTurn.pickSlot!]?.battletag?.split('#')[0] || 'Unknown player'}).
Recommend the TOP 3 heroes this specific player should pick.
CRITICAL: Only recommend heroes this player can actually play well (see their stats above).
Match recommendations to the player picking in this slot.`
  : `It's the ENEMY'S turn (${turnDescription}).
- Predict what they might do
- Explain how to counter it
- Suggest what YOU should prepare for your next turn`}

**TONE:**
Be direct, tactical, and concise like a coach in their headset. Focus on:
1. What just happened (if applicable)
2. What it means strategically
3. What to do about it
4. Specific hero recommendations (matched to the player picking)

Keep it under 200 words. Be actionable.`

    // Stream the response from OpenAI
    const stream = await openai.chat.completions.create({
      model: OPENAI_CONFIG.model,
      messages: [
        {
          role: 'system',
          content: 'You are an expert Heroes of the Storm draft coach providing real-time tactical advice. You speak directly to the team like a coach in their headset during a live draft. Be concise, tactical, and specific. Only recommend heroes that players can actually execute based on their stats.'
        },
        {
          role: 'user',
          content: prompt
        }
      ],
      max_tokens: 500,
      temperature: 0.7,
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
    console.error('Error generating coach commentary:', error)
    return NextResponse.json(
      { error: 'Failed to generate commentary' },
      { status: 500 }
    )
  }
}
