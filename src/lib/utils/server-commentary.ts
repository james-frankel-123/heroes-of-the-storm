// Server-side commentary formatting utility
// This fixes malformed markdown from the AI before streaming to the client

/**
 * Format commentary for display - fix malformed markdown from AI
 *
 * The AI outputs: "## Header- bullet1- bullet2- bullet3"
 * We need: "## Header\n\n- bullet1\n- bullet2\n- bullet3"
 *
 * @param text - Raw text from OpenAI with malformed markdown
 * @returns Properly formatted markdown text
 */
export function formatCommentary(text: string): string {
  let formatted = text.trim()

  // Step 1: Fix "## Header- text" to "## Header\n\n- text"
  // More specific: only match when we have a complete pattern ending with lowercase letter or punctuation before the dash
  // This avoids re-matching incomplete headers like "## Header (or"
  formatted = formatted.replace(/(##\s+[^-\n]+[a-z\)\"])-\s+/g, '$1\n\n- ')

  // Step 2: Fix "bullet text- nextbullet" to "bullet text\n- nextbullet"
  // Only match when we have complete words (ending with letter/punctuation) before the dash
  formatted = formatted.replace(/([a-z\)\.\,\!])-\s+/g, '$1\n- ')

  // Step 3: Add blank lines before ## headers (except at start)
  formatted = formatted.replace(/([^\n])(##\s+)/g, '$1\n\n$2')

  return formatted
}
