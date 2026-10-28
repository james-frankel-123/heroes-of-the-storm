'use client'

import { useRef } from 'react'

/**
 * Return keyboard focus to whatever opened a dialog. Our detail dialogs are opened from
 * plain buttons, not Radix triggers, so Radix has no trigger to restore focus to.
 * Spread the result onto DialogContent.
 */
export function useReturnFocus() {
  const opener = useRef<HTMLElement | null>(null)
  return {
    onOpenAutoFocus: () => {
      opener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
    },
    onCloseAutoFocus: (e: Event) => {
      const el = opener.current
      if (el && el.isConnected) {
        e.preventDefault()
        el.focus()
      }
    },
  }
}
