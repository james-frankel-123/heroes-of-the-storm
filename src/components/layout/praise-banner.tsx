'use client'

import { useState } from 'react'
import { X } from 'lucide-react'

export function PraiseBanner() {
  const [dismissed, setDismissed] = useState(false)

  if (dismissed) return null

  return (
    <div className="relative bg-gradient-to-r from-yellow-500 via-amber-400 to-yellow-500 text-black py-3 px-4 text-center font-black text-sm md:text-base tracking-wide shadow-[0_0_20px_rgba(234,179,8,0.5)] animate-pulse-subtle">
      <span className="inline-block">
        ALL HAIL ZEMILL, LORD OF HOTS. PRAISE BE TO{' '}
        <a
          href="https://www.heroesprofile.com/"
          target="_blank"
          rel="noopener noreferrer"
          className="underline decoration-2 underline-offset-2 hover:text-amber-900 transition-colors"
        >
          HEROES PROFILE
        </a>
        , THAT MAKES POSSIBLE SUCH ABUNDANCE OF DATA AND DRAFT INSIGHTS.
      </span>
      <button
        onClick={() => setDismissed(true)}
        className="absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded hover:bg-black/10 transition-colors"
        aria-label="Dismiss banner"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  )
}
