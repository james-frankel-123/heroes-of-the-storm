import { Skeleton } from '@/components/ui/skeleton'

/** Dashboard loading skeleton — matches MetaHeroes + MetaPairwise + PersonalInsights layout */
export default function DashboardLoading() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton className="h-9 w-48" />
          <Skeleton className="h-5 w-72" />
        </div>
        <Skeleton className="h-9 w-56" />
      </div>

      {/* Meta Madness section */}
      <section className="space-y-4">
        <Skeleton className="h-7 w-40" />
        {/* Two cards side by side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {[0, 1].map((i) => (
            <div key={i} className="rounded-lg border p-4 space-y-3">
              <Skeleton className="h-6 w-48" />
              {Array.from({ length: 8 }).map((_, j) => (
                <div key={j} className="flex items-center justify-between py-1">
                  <div className="flex items-center gap-3">
                    <Skeleton className="h-4 w-4" />
                    <Skeleton className="h-4 w-28" />
                    <Skeleton className="h-5 w-16 rounded-full" />
                  </div>
                  <div className="flex items-center gap-4">
                    <Skeleton className="h-4 w-20" />
                    <Skeleton className="h-4 w-14" />
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>

        {/* Pairwise section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {[0, 1].map((i) => (
            <div key={i} className="rounded-lg border p-4 space-y-3">
              <Skeleton className="h-6 w-36" />
              {Array.from({ length: 6 }).map((_, j) => (
                <div key={j} className="flex items-center justify-between py-1">
                  <Skeleton className="h-4 w-48" />
                  <Skeleton className="h-4 w-14" />
                </div>
              ))}
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
