'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import { Sparkles, Loader2, Radio, ChevronDown, ChevronUp } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { StreamingText } from '@/components/commentary/streaming-text'
import { useCoachCommentary, DraftState, HeroRecommendation, QuickAnalysis } from '@/lib/hooks/use-coach-commentary'
import { PartyMember } from '@/components/draft/draft-config-modal'
import { DraftTurn } from '@/lib/draft/draft-sequence'
import { RoleBalanceIndicator } from '@/components/draft/role-balance-indicator'
import { DraftHistoryTimeline } from '@/components/draft/draft-history-timeline'
import { SynergyIndicator } from '@/components/draft/synergy-indicator'
import { findSynergies } from '@/lib/data/hero-synergies'

interface AICoachPanelProps {
  draftState: DraftState
  partyRoster: PartyMember[]
  draftHistory: Array<{
    turn: DraftTurn
    hero: string
    timestamp: number
    battletag?: string
  }>
}

export function AICoachPanel({
  draftState,
  partyRoster,
  draftHistory
}: AICoachPanelProps) {
  const [showRoster, setShowRoster] = React.useState(false)
  const [showRoleBalance, setShowRoleBalance] = React.useState(true)
  const [showHistory, setShowHistory] = React.useState(false)
  const [showSynergies, setShowSynergies] = React.useState(true)

  const {
    commentary,
    isStreaming,
    error,
    recommendations,
    quickAnalysis
  } = useCoachCommentary(draftState, partyRoster, draftHistory, true)

  const yourPicks = draftState.yourTeam === 'blue' ? draftState.bluePicks : draftState.redPicks
  const enemyPicks = draftState.yourTeam === 'blue' ? draftState.redPicks : draftState.bluePicks

  return (
    <div className="flex flex-col h-full bg-card border-l border-border">
      {/* Header */}
      <div className="border-b border-border p-4 flex items-center justify-between bg-gradient-to-r from-blue-500/10 to-accent-cyan/10">
        <div className="flex items-center gap-2">
          <Radio className="h-5 w-5 text-accent-cyan" />
          <h2 className="font-bold text-lg">AI Draft Coach</h2>
        </div>
        {isStreaming && (
          <Badge variant="outline" className="text-accent-cyan border-accent-cyan/30">
            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            Analyzing...
          </Badge>
        )}
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Error Display */}
        {error && (
          <Card className="border-red-500/30 bg-red-500/5">
            <CardContent className="p-4">
              <p className="text-sm text-red-400">{error}</p>
            </CardContent>
          </Card>
        )}

        {/* Main Commentary */}
        {(commentary || isStreaming) && (
          <div className="rounded-lg border border-accent-cyan/30 bg-accent-cyan/5 p-4">
            <div className="flex items-start gap-2">
              <Sparkles className="h-4 w-4 text-accent-cyan mt-1 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <StreamingText
                  text={commentary}
                  isStreaming={isStreaming}
                  className="text-sm leading-relaxed prose prose-sm max-w-none dark:prose-invert"
                  showCursor={true}
                />
              </div>
            </div>
          </div>
        )}

        {/* No commentary yet */}
        {!commentary && !isStreaming && !error && draftHistory.length === 0 && (
          <div className="rounded-lg border border-dashed border-border p-8 text-center">
            <Radio className="h-8 w-8 text-muted-foreground mx-auto mb-3 opacity-50" />
            <p className="text-sm text-muted-foreground">
              Waiting for draft to begin...
            </p>
            <p className="text-xs text-muted-foreground mt-2">
              The AI coach will provide real-time tactical advice after each action.
            </p>
          </div>
        )}

        {/* Quick Analysis */}
        {quickAnalysis && (
          <Card className="border-primary-500/30 bg-primary-500/5">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Quick Analysis</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div>
                <span className="text-muted-foreground">Your comp: </span>
                <span className="font-medium">{quickAnalysis.yourComp}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Enemy comp: </span>
                <span className="font-medium">{quickAnalysis.enemyComp}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Need: </span>
                <span className="font-bold text-accent-cyan">{quickAnalysis.roleNeed}</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Top Recommendations */}
        {recommendations.length > 0 && (
          <Card className="border-gaming-success/30 bg-gaming-success/5">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Top 3 Recommendations</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {recommendations.map((rec, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="flex items-start gap-3 p-3 rounded-lg border border-gaming-success/30 bg-gaming-success/5"
                >
                  <Badge variant="outline" className="text-gaming-success border-gaming-success/30">
                    #{i + 1}
                  </Badge>
                  <div className="flex-1 min-w-0">
                    <div className="font-bold text-sm">{rec.hero}</div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {rec.player} (Slot {rec.slot + 1}) • {rec.winRate}% WR • {rec.games} games
                    </div>
                    <div className="text-xs text-foreground/80 mt-1">
                      {rec.reason}
                    </div>
                  </div>
                </motion.div>
              ))}
            </CardContent>
          </Card>
        )}

        {/* Role Balance (Collapsible) */}
        {yourPicks.some(p => p !== null) && (
          <Card className="border-border">
            <CardHeader className="pb-3">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowRoleBalance(!showRoleBalance)}
                className="w-full flex items-center justify-between p-0 h-auto hover:bg-transparent"
              >
                <CardTitle className="text-sm">Team Role Balance</CardTitle>
                {showRoleBalance ? (
                  <ChevronUp className="h-4 w-4" />
                ) : (
                  <ChevronDown className="h-4 w-4" />
                )}
              </Button>
            </CardHeader>
            {showRoleBalance && (
              <CardContent className="space-y-3">
                <div>
                  <div className="text-xs font-semibold text-muted-foreground mb-2">
                    Your Team ({draftState.yourTeam === 'blue' ? 'Blue' : 'Red'})
                  </div>
                  <RoleBalanceIndicator
                    picks={yourPicks}
                    teamName={draftState.yourTeam === 'blue' ? 'Blue' : 'Red'}
                    compact={true}
                  />
                </div>
                {enemyPicks.some(p => p !== null) && (
                  <div>
                    <div className="text-xs font-semibold text-muted-foreground mb-2">
                      Enemy Team ({draftState.yourTeam === 'blue' ? 'Red' : 'Blue'})
                    </div>
                    <RoleBalanceIndicator
                      picks={enemyPicks}
                      teamName={draftState.yourTeam === 'blue' ? 'Red' : 'Blue'}
                      compact={true}
                    />
                  </div>
                )}
              </CardContent>
            )}
          </Card>
        )}

        {/* Team Synergies (Collapsible) */}
        {findSynergies(yourPicks).length > 0 && (
          <Card className="border-border">
            <CardHeader className="pb-3">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowSynergies(!showSynergies)}
                className="w-full flex items-center justify-between p-0 h-auto hover:bg-transparent"
              >
                <CardTitle className="text-sm">Team Synergies</CardTitle>
                {showSynergies ? (
                  <ChevronUp className="h-4 w-4" />
                ) : (
                  <ChevronDown className="h-4 w-4" />
                )}
              </Button>
            </CardHeader>
            {showSynergies && (
              <CardContent>
                <SynergyIndicator
                  picks={yourPicks}
                  teamName={draftState.yourTeam === 'blue' ? 'Blue' : 'Red'}
                  compact={false}
                />
              </CardContent>
            )}
          </Card>
        )}

        {/* Draft History (Collapsible) */}
        {draftHistory.length > 0 && (
          <Card className="border-border">
            <CardHeader className="pb-3">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowHistory(!showHistory)}
                className="w-full flex items-center justify-between p-0 h-auto hover:bg-transparent"
              >
                <CardTitle className="text-sm">Draft History ({draftHistory.length} actions)</CardTitle>
                {showHistory ? (
                  <ChevronUp className="h-4 w-4" />
                ) : (
                  <ChevronDown className="h-4 w-4" />
                )}
              </Button>
            </CardHeader>
            {showHistory && (
              <CardContent>
                <DraftHistoryTimeline draftHistory={draftHistory} compact={true} />
              </CardContent>
            )}
          </Card>
        )}

        {/* Party Roster Summary (Collapsible) */}
        {partyRoster.some(m => m.battletag) && (
          <Card className="border-border">
            <CardHeader className="pb-3">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowRoster(!showRoster)}
                className="w-full flex items-center justify-between p-0 h-auto hover:bg-transparent"
              >
                <CardTitle className="text-sm">Your Party Roster</CardTitle>
                {showRoster ? (
                  <ChevronUp className="h-4 w-4" />
                ) : (
                  <ChevronDown className="h-4 w-4" />
                )}
              </Button>
            </CardHeader>
            {showRoster && (
              <CardContent className="space-y-3">
                {partyRoster.map((member, idx) => {
                  if (!member.battletag) {
                    return (
                      <div key={idx} className="text-xs text-muted-foreground">
                        Slot {idx + 1}: Unknown player
                      </div>
                    )
                  }

                  const hasStats = member.playerStats !== null
                  const heroCount = hasStats ? member.playerStats!.heroStats.length : 0

                  return (
                    <div key={idx} className="space-y-1">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">
                          {idx + 1}
                        </Badge>
                        <span className="text-sm font-medium">
                          {member.battletag.split('#')[0]}
                        </span>
                        {member.loading && (
                          <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                        )}
                        {hasStats && (
                          <span className="text-xs text-muted-foreground">
                            ({heroCount} heroes)
                          </span>
                        )}
                      </div>
                      {hasStats && member.playerStats && (
                        <div className="text-xs text-muted-foreground pl-8">
                          Top: {member.playerStats.heroStats
                            .sort((a, b) => b.winRate - a.winRate)
                            .slice(0, 3)
                            .map(heroStat => `${heroStat.hero} (${heroStat.winRate.toFixed(0)}%)`)
                            .join(', ')}
                        </div>
                      )}
                      {!hasStats && !member.loading && (
                        <div className="text-xs text-muted-foreground/50 pl-8">
                          No stats available
                        </div>
                      )}
                    </div>
                  )
                })}
              </CardContent>
            )}
          </Card>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-border p-3 text-xs text-muted-foreground text-center bg-card/50">
        Updates automatically after each action
      </div>
    </div>
  )
}
