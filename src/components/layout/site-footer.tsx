import { db } from '@/lib/db'
import { heroStatsAggregate } from '@/lib/db/schema'
import { sql } from 'drizzle-orm'

async function getLastUpdated(): Promise<Date | null> {
  try {
    const result = await db
      .select({ latest: sql<Date>`max(${heroStatsAggregate.updatedAt})` })
      .from(heroStatsAggregate)
      .limit(1)
    return result[0]?.latest ?? null
  } catch {
    return null
  }
}

export async function SiteFooter() {
  const lastUpdated = await getLastUpdated()

  return (
    <footer className="border-t border-border/40 mt-12 py-6">
      {/* HP API terms: at least body-text size, legible, not fine print. */}
      <p className="container mb-3 text-center sm:text-left text-base text-foreground">
        Data provided by{' '}
        <a
          href="https://www.heroesprofile.com/"
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium underline underline-offset-2"
        >
          Heroes Profile
        </a>
      </p>
      <div className="container flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-muted-foreground">
        <span>
          HotS Fever · Not affiliated with Blizzard Entertainment ·{' '}
          <a href="/terms" className="underline">Terms of Use</a>
        </span>
        {lastUpdated && (
          <span>
            Stats updated {new Date(lastUpdated).toLocaleDateString('en-US', {
              month: 'short',
              day: 'numeric',
              year: 'numeric',
            })}
            {process.env.NEXT_PUBLIC_COMMIT_SHA && (
              <span className="ml-2 opacity-60">· v{process.env.NEXT_PUBLIC_COMMIT_SHA}</span>
            )}
          </span>
        )}
      </div>
    </footer>
  )
}
