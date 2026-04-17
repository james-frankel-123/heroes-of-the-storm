/**
 * Resolve a map display name to its image path under /public/maps.
 * Images were downloaded via scripts/download-map-images.mjs.
 */

const MAP_SLUGS: Record<string, string> = {
  'Alterac Pass':              'alterac-pass',
  'Battlefield of Eternity':   'battlefield-of-eternity',
  "Blackheart's Bay":          'blackhearts-bay',
  'Braxis Holdout':            'braxis-holdout',
  'Cursed Hollow':             'cursed-hollow',
  'Dragon Shire':              'dragon-shire',
  'Garden of Terror':          'garden-of-terror',
  'Hanamura Temple':           'hanamura',
  'Infernal Shrines':          'infernal-shrines',
  'Sky Temple':                'sky-temple',
  'Tomb of the Spider Queen':  'tomb-of-the-spider-queen',
  'Towers of Doom':            'towers-of-doom',
  'Volskaya Foundry':          'volskaya-foundry',
  'Warhead Junction':          'warhead-junction',
}

export function mapImageSrc(map: string): string | null {
  const slug = MAP_SLUGS[map]
  return slug ? `/maps/${slug}.jpg` : null
}
