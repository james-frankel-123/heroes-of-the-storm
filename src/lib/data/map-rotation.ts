/**
 * Current Storm League map rotation.
 * Update this list when the rotation changes.
 */
export const CURRENT_MAP_ROTATION: string[] = [
  'Alterac Pass',
  'Battlefield of Eternity',
  'Cursed Hollow',
  'Dragon Shire',
  'Garden of Terror',
  'Hanamura Temple',
  'Infernal Shrines',
  'Sky Temple',
  'Tomb of the Spider Queen',
  'Towers of Doom',
  'Volskaya Foundry',
]

const rotationSet = new Set(CURRENT_MAP_ROTATION)

export function isInRotation(map: string): boolean {
  return rotationSet.has(map)
}
