'use client'

import * as React from 'react'
import { Trophy } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export default function AchievementsPage() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold tracking-tight glow">Achievements</h1>
        <p className="mt-2 text-muted-foreground">
          Track your milestones and unlock badges
        </p>
      </div>

      <Card className="glass border-primary-500/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Trophy className="h-5 w-5" />
            Achievements System
          </CardTitle>
        </CardHeader>
        <CardContent className="py-12 text-center">
          <p className="text-muted-foreground">
            Achievements and badges system coming soon!
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
