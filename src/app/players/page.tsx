import { Suspense } from 'react'
import type { Metadata } from 'next'
import { PlayersClient } from './players-client'

export const metadata: Metadata = {
  title: 'Player Lookup | HotS Fever',
  description: 'Per-hero skill estimates for any public Heroes Profile account.',
}

export default function PlayersPage() {
  return (
    <Suspense>
      <PlayersClient />
    </Suspense>
  )
}
