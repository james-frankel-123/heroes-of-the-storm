'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import { usePathname } from 'next/navigation'

/**
 * Thin progress bar that appears at the top of the viewport during navigation.
 *
 * Listens for pathname changes — when a navigation starts (link click),
 * the bar animates from 0% to ~90%, then jumps to 100% and fades out
 * when the new page loads.
 */
export function NavProgress() {
  const pathname = usePathname()
  const [progress, setProgress] = useState(0)
  const [visible, setVisible] = useState(false)
  const prevPathname = useRef(pathname)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const cleanup = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    if (intervalRef.current) clearInterval(intervalRef.current)
  }, [])

  // Start the bar when we detect a navigation is pending.
  // We do this by intercepting clicks on internal links.
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      const anchor = (e.target as HTMLElement).closest('a')
      if (!anchor) return

      const href = anchor.getAttribute('href')
      if (!href || href.startsWith('http') || href.startsWith('#')) return

      // Same page — no loading needed
      if (href === pathname) return

      // Start progress animation
      cleanup()
      setVisible(true)
      setProgress(10)

      // Gradually increase toward 90%
      let current = 10
      intervalRef.current = setInterval(() => {
        current += Math.random() * 15
        if (current > 90) current = 90
        setProgress(current)
      }, 200)
    }

    document.addEventListener('click', handleClick, true)
    return () => {
      document.removeEventListener('click', handleClick, true)
      cleanup()
    }
  }, [pathname, cleanup])

  // Complete the bar when pathname actually changes
  useEffect(() => {
    if (pathname !== prevPathname.current) {
      prevPathname.current = pathname
      cleanup()
      setProgress(100)

      timerRef.current = setTimeout(() => {
        setVisible(false)
        setProgress(0)
      }, 200)
    }
  }, [pathname, cleanup])

  if (!visible && progress === 0) return null

  return (
    <div
      className="fixed top-0 left-0 right-0 z-[100] h-0.5 pointer-events-none"
      role="progressbar"
      aria-valuenow={progress}
    >
      <div
        className="h-full bg-primary transition-all duration-200 ease-out"
        style={{
          width: `${progress}%`,
          opacity: progress >= 100 ? 0 : 1,
          transition: progress >= 100
            ? 'width 150ms ease-out, opacity 200ms ease-out 100ms'
            : 'width 200ms ease-out',
        }}
      />
    </div>
  )
}
