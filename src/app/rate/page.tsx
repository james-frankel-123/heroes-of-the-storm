import { Suspense } from 'react'
import type { Metadata } from 'next'
import { RateClient } from './rate-client'
import { SURVEY_CLOSED, SURVEY_CLOSED_MESSAGE } from '@/lib/rating/assignment'

export const dynamic = 'force-dynamic'

export const metadata: Metadata = {
  title: 'Draft Rating Study | HotS Fever',
  description: 'Expert draft-rating study',
  robots: { index: false, follow: false },
}

/**
 * Unlisted expert draft-rating study page.
 * Access via /rate?rater=NAME[&slot=N] — intentionally not linked from the nav.
 */
export default function RatePage() {
  // collection ended: every rater and slot sees only the closing note (no items are loaded)
  if (SURVEY_CLOSED) {
    return (
      <div className="mx-auto max-w-xl py-16 text-center">
        <h1 className="text-2xl font-bold">Draft Rating Study</h1>
        <p className="mt-4 text-muted-foreground">{SURVEY_CLOSED_MESSAGE}</p>
      </div>
    )
  }
  return (
    <Suspense>
      <RateClient />
    </Suspense>
  )
}
