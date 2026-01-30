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
