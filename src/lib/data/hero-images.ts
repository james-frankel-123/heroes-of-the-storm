/**
 * Resolve a hero display name to its portrait path under /public/heroes.
 * Portraits were downloaded via scripts/download-hero-portraits.mjs.
 */
export function heroImageSrc(hero: string): string {
  if (hero === "Cho'gall") return '/heroes/cho.png'
  const slug = hero
    .toLowerCase()
    .normalize('NFD').replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]/g, '')
  return `/heroes/${slug}.png`
}
