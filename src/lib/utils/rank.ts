/**
 * Format league tier from API to displayable rank string
 * Capitalizes the first letter of each word
 */
export function formatLeagueTier(leagueTier: string | null): string {
  if (!leagueTier) {
    return 'Unranked'
  }

  // Capitalize first letter of each word
  return leagueTier
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ')
}
