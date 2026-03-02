import { Skeleton } from '@/components/ui/skeleton'

/** Heroes page loading skeleton — matches hero table layout */
export default function HeroesLoading() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton className="h-9 w-32" />
          <Skeleton className="h-5 w-64" />
        </div>
        <Skeleton className="h-9 w-56" />
      </div>

      {/* View mode toggle */}
      <div className="flex gap-2">
        <Skeleton className="h-8 w-20" />
        <Skeleton className="h-8 w-24" />
        <Skeleton className="h-8 w-24" />
      </div>

      {/* Table header */}
      <div className="rounded-lg border">
        <div className="flex items-center gap-4 px-4 py-3 border-b">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-16 ml-auto" />
          <Skeleton className="h-4 w-16" />
          <Skeleton className="h-4 w-16" />
          <Skeleton className="h-4 w-16" />
        </div>

        {/* Table rows */}
        {Array.from({ length: 20 }).map((_, i) => (
          <div
            key={i}
            className="flex items-center gap-4 px-4 py-2.5 border-b last:border-0"
          >
            <div className="flex items-center gap-2">
              <Skeleton className="h-4 w-28" />
              <Skeleton className="h-5 w-14 rounded-full" />
            </div>
            <Skeleton className="h-4 w-12 ml-auto" />
            <Skeleton className="h-4 w-14" />
            <Skeleton className="h-4 w-12" />
            <Skeleton className="h-4 w-12" />
          </div>
        ))}
      </div>
    </div>
  )
}
