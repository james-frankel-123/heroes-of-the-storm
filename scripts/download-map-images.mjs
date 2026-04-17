#!/usr/bin/env node
/**
 * One-shot script: download map loading-screen images from Nexus Compendium
 * into public/maps/{slug}.jpg. Run with `node scripts/download-map-images.mjs`.
 * Skips files that already exist.
 */
import fs from 'node:fs/promises'
import path from 'node:path'

const OUT = path.join(process.cwd(), 'public', 'maps')
await fs.mkdir(OUT, { recursive: true })

// Map display names → nexuscompendium slug
const MAPS = {
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

const BASE = 'https://nexuscompendium.com/images/battlegrounds'
let downloaded = 0, skipped = 0, failed = []

for (const [name, slug] of Object.entries(MAPS)) {
  const dest = path.join(OUT, `${slug}.jpg`)
  try {
    await fs.access(dest)
    skipped++
    continue
  } catch {}

  const url = `${BASE}/${slug}/main.jpg`
  const res = await fetch(url)
  if (!res.ok) {
    failed.push({ name, url, status: res.status })
    continue
  }
  const buf = Buffer.from(await res.arrayBuffer())
  await fs.writeFile(dest, buf)
  downloaded++
  process.stdout.write('.')
}

console.log(`\nDownloaded ${downloaded}, skipped ${skipped}, failed ${failed.length}`)
if (failed.length) console.log('Failed:', failed)
