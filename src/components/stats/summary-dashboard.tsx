'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { StatisticsSummary } from '@/lib/data/statistics'
import { getWinRateColor } from '@/lib/utils'

interface SummaryDashboardProps {
  statistics: StatisticsSummary
  onClick?: () => void
}

export function SummaryDashboard({ statistics, onClick }: SummaryDashboardProps) {
  const {
    last10WinRate,
    last20WinRate,
    last50WinRate,
    consistencyScore,
    overallKDA,
  } = statistics

  const getFormStatus = () => {
    if (last10WinRate >= 60) return { label: 'Hot', color: 'text-gaming-success' }
    if (last10WinRate >= 50) return { label: 'Stable', color: 'text-primary-500' }
    if (last10WinRate >= 40) return { label: 'Cold', color: 'text-yellow-500' }
    return { label: 'Struggling', color: 'text-gaming-danger' }
  }

  const formStatus = getFormStatus()

  return (
    <Card
      className="glass border-primary-500/30 cursor-pointer hover:border-primary-500/50 transition-all"
      onClick={onClick}
    >
      <CardHeader>
        <CardTitle className="text-lg">Performance Summary</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Current Form */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Current Form:</span>
            <Badge className={formStatus.color}>{formStatus.label}</Badge>
          </div>
        </div>

        {/* Recent Win Rates */}
        <div className="space-y-2 pt-2 border-t border-border">
          <h4 className="text-xs font-medium text-muted-foreground uppercase">Recent Performance</h4>
          <div className="grid grid-cols-3 gap-2">
            <div className="text-center p-2 rounded-md bg-background/50">
              <div className={`text-lg font-bold ${getWinRateColor(last10WinRate)}`}>
                {last10WinRate.toFixed(0)}%
              </div>
              <div className="text-xs text-muted-foreground">Last 10</div>
            </div>
            <div className="text-center p-2 rounded-md bg-background/50">
              <div className={`text-lg font-bold ${getWinRateColor(last20WinRate)}`}>
                {last20WinRate.toFixed(0)}%
              </div>
              <div className="text-xs text-muted-foreground">Last 20</div>
            </div>
            <div className="text-center p-2 rounded-md bg-background/50">
              <div className={`text-lg font-bold ${getWinRateColor(last50WinRate)}`}>
                {last50WinRate.toFixed(0)}%
              </div>
              <div className="text-xs text-muted-foreground">Last 50</div>
            </div>
          </div>
        </div>

        {/* KDA & Consistency */}
        <div className="space-y-2 pt-2 border-t border-border">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">KDA Ratio:</span>
            <span className="font-medium">{overallKDA.kda.toFixed(2)}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Consistency:</span>
            <div className="flex items-center gap-2">
              <div className="w-20 h-2 bg-background/50 rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary-500 transition-all"
                  style={{ width: `${consistencyScore}%` }}
                />
              </div>
              <span className="font-medium">{consistencyScore}/100</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
