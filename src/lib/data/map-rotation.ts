/**
 * Storm League map rotation helpers.
 *
 * The live rotation is data-driven: `getMapRotation()` in queries.ts reads the
 * maps present in map_stats_aggregate, which the nightly sync rebuilds from
 * current-patch games. This list is only the fallback for when that query
 * returns nothing; it does not need to track future rotations.
 *
 * The ML models (src/lib/draft/encoding.ts) keep their own fixed map vocabulary
 * and are unaffected by this file.
 */
export const FALLBACK_MAP_ROTATION: readonly string[] = [
  'Alterac Pass',
  'Battlefield of Eternity',
  'Braxis Holdout',
  'Cursed Hollow',
  'Dragon Shire',
  'Garden of Terror',
  'Haunted Mines',
  'Infernal Shrines',
  'Sky Temple',
  'Tomb of the Spider Queen',
  'Towers of Doom',
  'Volskaya Foundry',
]

/** The given rotation, or the fallback list when it is empty. */
export function resolveRotation(maps: readonly string[]): string[] {
  return maps.length > 0 ? [...maps] : [...FALLBACK_MAP_ROTATION]
}

/** Keep only rows whose `map` is in the rotation. */
export function filterToRotation<T extends { map: string }>(
  rows: readonly T[],
  rotation: readonly string[]
): T[] {
  const set = new Set(rotation)
  return rows.filter((r) => set.has(r.map))
}
