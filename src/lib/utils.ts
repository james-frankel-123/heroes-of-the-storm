import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * Utility function to merge Tailwind CSS classes
 * Combines clsx for conditional classes and tailwind-merge for deduplication
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Format a number with commas (e.g., 1000 -> 1,000)
 */
export function formatNumber(num: number): string {
  return new Intl.NumberFormat('en-US').format(num)
}

/**
 * Format a percentage (e.g., 0.5234 -> 52.3%)
 */
export function formatPercent(value: number, decimals: number = 1): string {
  return `${value.toFixed(decimals)}%`
}

/**
 * Get color class based on win rate
 */
export function getWinRateColor(winRate: number): string {
  if (winRate >= 55) return 'text-gaming-success'
  if (winRate >= 50) return 'text-gaming-warning'
  return 'text-gaming-danger'
}

/**
 * Get role color class
 */
export function getRoleColor(role: string): string {
  const roleMap: Record<string, string> = {
    'Tank': 'bg-gaming-tank',
    'Bruiser': 'bg-gaming-bruiser',
    'Healer': 'bg-gaming-healer',
    'Ranged Assassin': 'bg-gaming-ranged',
    'Melee Assassin': 'bg-gaming-melee',
    'Support': 'bg-gaming-support',
  }
  return roleMap[role] || 'bg-gray-500'
}

/**
 * Debounce function for search/filter inputs
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null
  return function (...args: Parameters<T>) {
    if (timeout) clearTimeout(timeout)
    timeout = setTimeout(() => func(...args), wait)
  }
}

/**
 * Check if user prefers reduced motion
 */
export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined') return false
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

// ---------------------------------------------------------------------------
// Confidence-adjusted win rate
// ---------------------------------------------------------------------------

/**
 * Confidence-adjusted win rate using Bayesian shrinkage toward 50%.
 *
 * For hero-level stats: padded with phantom 50% games up to 30 total.
 *   e.g. 8 games at 75% WR → treated as 8 real wins + 11 phantom (50%)
 *        = (6 + 5.5) / (8 + 11) = 60.5% adjusted
 *
 * For hero+map stats: padded up to 10 total.
 *
 * Once games >= threshold, returns the raw rate unmodified.
 *
 * @param wins      Actual wins
 * @param games     Actual games played
 * @param threshold Minimum games for full confidence (30 for hero, 10 for map)
 */
export function confidenceAdjustedWinRate(
  wins: number,
  games: number,
  threshold: number = 30
): number {
  if (games >= threshold) return (wins / games) * 100
  const phantomGames = threshold - games
  const phantomWins = phantomGames * 0.5
  return ((wins + phantomWins) / (games + phantomGames)) * 100
}

/**
 * Confidence-adjusted MAWP. Same idea: if total games < threshold,
 * we blend the MAWP toward 50% proportionally.
 *
 * @param mawp   Raw momentum-adjusted win percentage (0-100)
 * @param games  Total games on this hero (or hero+map)
 * @param threshold  30 for hero overall, 10 for hero+map
 */
export function confidenceAdjustedMawp(
  mawp: number,
  games: number,
  threshold: number = 30
): number {
  if (games >= threshold) return mawp
  // Blend: weight of real data = games/threshold, rest is 50%
  const weight = games / threshold
  return mawp * weight + 50 * (1 - weight)
}

/**
 * Returns a confidence label for display purposes.
 */
export function confidenceLabel(
  games: number,
  threshold: number = 30
): 'high' | 'medium' | 'low' {
  if (games >= threshold) return 'high'
  if (games >= threshold * 0.5) return 'medium'
  return 'low'
}
