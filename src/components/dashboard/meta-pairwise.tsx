'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { formatPercent, getWinRateColor, formatNumber } from '@/lib/utils'
import type { HeroPairwiseStats } from '@/lib/types'

interface MetaPairwiseProps {
  synergies: HeroPairwiseStats[]
  counters: HeroPairwiseStats[]
}

const INITIAL_COUNT = 10

export function MetaPairwise({ synergies, counters }: MetaPairwiseProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Strongest Synergies</CardTitle>
        </CardHeader>
        <CardContent>
          <PairList pairs={synergies} label="together" />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Strongest Counters</CardTitle>
        </CardHeader>
        <CardContent>
          <PairList pairs={counters} label="vs" />
        </CardContent>
      </Card>
    </div>
  )
}

function PairList({
  pairs,
  label,
}: {
  pairs: HeroPairwiseStats[]
  label: string
}) {
  const [expanded, setExpanded] = useState(false)

  if (pairs.length === 0) {
    return <p className="text-sm text-muted-foreground">No data available</p>
  }

  const visible = expanded ? pairs : pairs.slice(0, INITIAL_COUNT)
  const hasMore = pairs.length > INITIAL_COUNT

  return (
    <div>
      <div className="space-y-2">
        {visible.map((pair, i) => (
          <div
            key={`${pair.heroA}-${pair.heroB}`}
            className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-accent/50 transition-colors"
          >
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-xs text-muted-foreground w-5 text-right">
                {i + 1}
              </span>
              <span className="font-medium text-sm">
                {pair.heroA}{' '}
                <span className="text-muted-foreground font-normal">{label}</span>{' '}
                {pair.heroB}
              </span>
            </div>
            <div className="flex items-center gap-4 text-sm shrink-0">
              <span className="text-muted-foreground">
                {formatNumber(pair.games)} games
              </span>
              <span className={`font-semibold w-14 text-right ${getWinRateColor(pair.winRate)}`}>
                {formatPercent(pair.winRate)}
              </span>
            </div>
          </div>
        ))}
      </div>
      {hasMore && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-3 text-xs font-medium text-primary hover:text-primary/80 transition-colors"
        >
          {expanded
            ? 'Show less'
            : `See all ${pairs.length} pairs`}
        </button>
      )}
    </div>
  )
}
