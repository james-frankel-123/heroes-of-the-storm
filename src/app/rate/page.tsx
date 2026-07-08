import { Suspense } from 'react'
import type { Metadata } from 'next'
import { RateClient } from './rate-client'

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
  return (
    <Suspense>
      <RateClient />
    </Suspense>
  )
}
