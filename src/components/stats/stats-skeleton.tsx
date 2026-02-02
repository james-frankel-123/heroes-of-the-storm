'use client'

import { Card } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'

export function StatsSkeleton() {
  return (
    <div className="flex h-screen">
      {/* Main Content */}
      <div className="flex-1 overflow-y-auto pr-6 space-y-8 pb-8">
        <div>
          <Skeleton className="h-12 w-96" />
          <Skeleton className="mt-2 h-6 w-64" />
          <Skeleton className="mt-1 h-4 w-80" />
        </div>

        {/* Summary Cards */}
        <div className="grid gap-6 md:grid-cols-2">
          <Skeleton className="h-64" />
          <Skeleton className="h-64" />
        </div>

        {/* Performance Trends */}
        <Skeleton className="h-96" />

        {/* KDA Statistics */}
        <Skeleton className="h-80" />

        {/* Hero Analytics */}
        <Skeleton className="h-96" />

        {/* Temporal Analysis */}
        <Skeleton className="h-96" />
      </div>

      {/* AI Chat Panel */}
      <div className="w-96 border-l border-border bg-card/30">
        <Skeleton className="h-full" />
      </div>
    </div>
  )
}
