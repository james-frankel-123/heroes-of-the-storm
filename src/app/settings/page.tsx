'use client'

import * as React from 'react'
import { Settings } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export default function SettingsPage() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold tracking-tight glow">Settings</h1>
        <p className="mt-2 text-muted-foreground">
          Customize your experience
        </p>
      </div>

      <Card className="glass border-primary-500/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Application Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="py-12 text-center">
          <p className="text-muted-foreground">
            Settings page coming soon with theme preferences, data sources, and more.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
