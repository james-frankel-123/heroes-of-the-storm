'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { formatPercent, getWinRateColor, formatNumber } from '@/lib/utils'
import type { HeroPairwiseStats } from '@/lib/types'

interface MetaPairwiseProps {
  synergies: HeroPairwiseStats[]
  counters: HeroPairwiseStats[]
}

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
  if (pairs.length === 0) {
    return <p className="text-sm text-muted-foreground">No data available</p>
  }

  return (
    <div className="space-y-2">
      {pairs.map((pair, i) => (
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
  )
}
