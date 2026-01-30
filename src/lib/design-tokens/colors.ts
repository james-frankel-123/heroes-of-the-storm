export const colors = {
  // Primary palette - Gaming blue
  primary: {
    50: '#e6f2ff',
    100: '#cce5ff',
    200: '#99cbff',
    300: '#66b0ff',
    400: '#3396ff',
    500: '#4a9eff',
    600: '#0073e6',
    700: '#0059b3',
    800: '#003d80',
    900: '#1e3c72',
  },

  // Accent colors
  accent: {
    cyan: '#4affff',
    purple: '#8a2be2',
  },

  // Semantic colors for win rates
  winRate: {
    high: '#4fffb0',
    mid: '#ffeb3b',
    low: '#ff6b6b',
  },

  // Role-based colors
  role: {
    tank: '#8b5a3c',
    bruiser: '#d4af37',
    healer: '#4fffb0',
    rangedAssassin: '#ff6b6b',
    meleeAssassin: '#ff9500',
    support: '#9b59b6',
    unknown: '#95a5a6',
  },

  // Neutrals
  neutral: {
    50: '#fafafa',
    100: '#f5f5f5',
    200: '#e5e5e5',
    300: '#d4d4d4',
    400: '#a3a3a3',
    500: '#737373',
    600: '#525252',
    700: '#404040',
    800: '#262626',
    900: '#171717',
    950: '#0a0a0a',
  },
} as const

export type ColorToken = typeof colors
