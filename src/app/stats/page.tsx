'use client'

import * as React from 'react'
import { TrendingUp } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { PasswordGate } from '@/components/auth/password-gate'

export default function StatsPage() {
  return (
    <PasswordGate requiredPassword="ronpaul2012" storageKey="protected_pages_auth">
      <div className="space-y-8">
        <div>
          <h1 className="text-4xl font-bold tracking-tight glow">Statistics</h1>
          <p className="mt-2 text-muted-foreground">
            Advanced statistical analysis and trends
          </p>
        </div>

        <Card className="glass border-primary-500/30">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Advanced Statistics
            </CardTitle>
          </CardHeader>
          <CardContent className="py-12 text-center">
            <p className="text-muted-foreground">
              Advanced statistics page coming soon with trend analysis, time-series charts, and more.
            </p>
          </CardContent>
        </Card>
      </div>
    </PasswordGate>
  )
}
