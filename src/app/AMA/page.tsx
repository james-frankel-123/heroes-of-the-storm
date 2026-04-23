import type { Metadata } from 'next'
import { AMAClient } from './ama-client'

export const metadata: Metadata = {
  title: 'Coach — HotS Fever',
  description: 'Ask me anything about your draft recommendations.',
}

export default function AMAPage() {
  return <AMAClient />
}
