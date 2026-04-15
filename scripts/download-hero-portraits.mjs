#!/usr/bin/env node
/**
 * One-shot script: download hero portraits from Heroes Profile into
 * public/heroes/{slug}.png. Run with `node scripts/download-hero-portraits.mjs`.
 * Skips files that already exist.
 */
import fs from 'node:fs/promises'
import path from 'node:path'

const OUT = path.join(process.cwd(), 'public', 'heroes')
await fs.mkdir(OUT, { recursive: true })

// Hero list mirrors src/lib/data/hero-roles.ts (without Cho'gall merged name).
const HEROES = [
  "Abathur","Alarak","Alexstrasza","Ana","Anduin","Anub'arak","Artanis",
  "Arthas","Auriel","Azmodan","Blaze","Brightwing","Cassia","Chen","Cho",
  "Chromie","D.Va","Deathwing","Deckard","Dehaka","Diablo","E.T.C.",
  "Falstad","Fenix","Gall","Garrosh","Gazlowe","Genji","Greymane",
  "Gul'dan","Hanzo","Hogger","Illidan","Imperius","Jaina","Johanna",
  "Junkrat","Kael'thas","Kel'Thuzad","Kerrigan","Kharazim","Leoric",
  "Li Li","Li-Ming","Lt. Morales","Lunara","Lúcio","Maiev","Mal'Ganis",
  "Malfurion","Malthael","Medivh","Mei","Mephisto","Muradin","Murky",
  "Nazeebo","Nova","Orphea","Probius","Qhira","Ragnaros","Raynor",
  "Rehgar","Rexxar","Samuro","Sgt. Hammer","Sonya","Stitches","Stukov",
  "Sylvanas","Tassadar","The Butcher","The Lost Vikings","Thrall","Tracer",
  "Tychus","Tyrael","Tyrande","Uther","Valeera","Valla","Varian",
  "Whitemane","Xul","Yrel","Zagara","Zarya","Zeratul","Zul'jin",
]

function slug(name) {
  return name
    .toLowerCase()
    .normalize('NFD').replace(/[\u0300-\u036f]/g, '') // strip diacritics (Lúcio)
    .replace(/[^a-z0-9]/g, '')                        // strip punctuation/spaces
}

const BASE = 'https://www.heroesprofile.com/images/heroes'
let downloaded = 0, skipped = 0, failed = []

for (const hero of HEROES) {
  const s = slug(hero)
  const dest = path.join(OUT, `${s}.png`)
  try {
    await fs.access(dest)
    skipped++
    continue
  } catch {}

  const url = `${BASE}/${s}.png`
  const res = await fetch(url)
  if (!res.ok) {
    failed.push({ hero, url, status: res.status })
    continue
  }
  const buf = Buffer.from(await res.arrayBuffer())
  await fs.writeFile(dest, buf)
  downloaded++
  process.stdout.write(`.`)
}

console.log(`\nDownloaded ${downloaded}, skipped ${skipped}, failed ${failed.length}`)
if (failed.length) console.log('Failed:', failed)
