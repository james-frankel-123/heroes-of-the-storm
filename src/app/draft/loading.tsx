import { Skeleton } from '@/components/ui/skeleton'

/** Draft page loading skeleton — matches setup phase layout */
export default function DraftLoading() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <Skeleton className="h-9 w-48" />
        <Skeleton className="h-5 w-72" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-3xl">
        {/* Map selection */}
        <div className="space-y-2">
          <Skeleton className="h-4 w-12" />
          <div className="grid grid-cols-2 gap-1.5">
            {Array.from({ length: 14 }).map((_, i) => (
              <Skeleton key={i} className="h-10 w-full rounded-md" />
            ))}
          </div>
        </div>

        {/* Right column */}
        <div className="space-y-6">
          {/* Tier */}
          <div className="space-y-2">
            <Skeleton className="h-4 w-20" />
            <Skeleton className="h-9 w-full" />
          </div>

          {/* Team side */}
          <div className="space-y-2">
            <Skeleton className="h-4 w-40" />
            <div className="flex gap-2">
              <Skeleton className="h-10 flex-1" />
              <Skeleton className="h-10 flex-1" />
            </div>
          </div>

          {/* Player slots */}
          <div className="space-y-2">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-3 w-64" />
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-9 w-full" />
            ))}
          </div>
        </div>
      </div>

      {/* Start button */}
      <Skeleton className="h-12 w-32 rounded-lg" />
    </div>
  )
}
