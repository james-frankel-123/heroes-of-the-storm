'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import { Sparkles, Zap } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { findSynergies, getSynergyScore, HeroSynergy } from '@/lib/data/hero-synergies'

interface SynergyIndicatorProps {
  picks: (string | null)[]
  teamName?: string
  compact?: boolean
}

export function SynergyIndicator({
  picks,
  teamName,
  compact = false
}: SynergyIndicatorProps) {
  const synergies = findSynergies(picks)
  const score = getSynergyScore(picks)

  if (synergies.length === 0) {
    return null
  }

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <Badge variant="outline" className="text-accent-cyan border-accent-cyan/30 bg-accent-cyan/10">
          <Sparkles className="h-3 w-3 mr-1" />
          {synergies.length} {synergies.length === 1 ? 'Synergy' : 'Synergies'}
        </Badge>
      </div>
    )
  }

  return (
    <Card className="border-accent-cyan/30 bg-accent-cyan/5">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-accent-cyan" />
          Team Synergies{teamName ? ` - ${teamName}` : ''}
          <Badge variant="outline" className="ml-auto">
            Score: {score}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {synergies.map((synergy, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: idx * 0.1 }}
            className={`p-3 rounded-lg border ${
              synergy.strength === 'high'
                ? 'border-accent-cyan/30 bg-accent-cyan/10'
                : 'border-primary-500/30 bg-primary-500/10'
            }`}
          >
            <div className="flex items-start gap-2">
              {synergy.strength === 'high' && (
                <Zap className="h-4 w-4 text-accent-cyan flex-shrink-0 mt-0.5" />
              )}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-bold text-sm">
                    {synergy.heroes[0]}
                  </span>
                  <span className="text-xs text-muted-foreground">+</span>
                  <span className="font-bold text-sm">
                    {synergy.heroes[1]}
                  </span>
                  {synergy.strength === 'high' && (
                    <Badge variant="outline" className="text-xs text-accent-cyan border-accent-cyan/30">
                      High Synergy
                    </Badge>
                  )}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {synergy.reason}
                </p>
              </div>
            </div>
          </motion.div>
        ))}
      </CardContent>
    </Card>
  )
}
