import OpenAI from 'openai'
import { NextResponse } from 'next/server'
import { timingSafeEqual } from 'crypto'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

// Canonical vocab the model must map heroes/maps onto (matches the draft engine).
const HEROES = [
  'Abathur', 'Alarak', 'Alexstrasza', 'Ana', 'Anduin', "Anub'arak", 'Artanis',
  'Arthas', 'Auriel', 'Azmodan', 'Blaze', 'Brightwing', 'Cassia', 'Chen', 'Cho',
  'Chromie', 'D.Va', 'Deathwing', 'Deckard', 'Dehaka', 'Diablo', 'E.T.C.',
  'Falstad', 'Fenix', 'Gall', 'Garrosh', 'Gazlowe', 'Genji', 'Greymane',
  "Gul'dan", 'Hanzo', 'Hogger', 'Illidan', 'Imperius', 'Jaina', 'Johanna',
  'Junkrat', "Kael'thas", 'Kel\'Thuzad', 'Kerrigan', 'Kharazim', 'Leoric',
  'Li Li', 'Li-Ming', 'Lt. Morales', 'Lunara', 'Lúcio', 'Maiev', "Mal'Ganis",
  'Malfurion', 'Malthael', 'Medivh', 'Mei', 'Mephisto', 'Muradin', 'Murky',
  'Nazeebo', 'Nova', 'Orphea', 'Probius', 'Qhira', 'Ragnaros', 'Raynor',
  'Rehgar', 'Rexxar', 'Samuro', 'Sgt. Hammer', 'Sonya', 'Stitches', 'Stukov',
  'Sylvanas', 'Tassadar', 'The Butcher', 'The Lost Vikings', 'Thrall', 'Tracer',
  'Tychus', 'Tyrael', 'Tyrande', 'Uther', 'Valeera', 'Valla', 'Varian',
  'Whitemane', 'Xul', 'Yrel', 'Zagara', 'Zarya', 'Zeratul', "Zul'jin",
]
const MAPS = [
  'Alterac Pass', 'Battlefield of Eternity', "Blackheart's Bay", 'Braxis Holdout',
  'Cursed Hollow', 'Dragon Shire', 'Garden of Terror', 'Hanamura Temple',
  'Infernal Shrines', 'Sky Temple', 'Tomb of the Spider Queen', 'Towers of Doom',
  'Volskaya Foundry', 'Warhead Junction',
]

const HERO_SET = new Set(HEROES)

function authorized(req: Request): boolean {
  const expected = process.env.DRAFT_VISION_TOKEN
  if (!expected) return false // fail closed — protects our vision-API spend
  const header = req.headers.get('x-vision-token') ?? ''
  const a = Buffer.from(header)
  const b = Buffer.from(expected)
  return a.length === b.length && timingSafeEqual(a, b)
}

/** Keep only names that exactly match the catalog (drops hallucinations/empties). */
function clean(arr: unknown): string[] {
  if (!Array.isArray(arr)) return []
  return arr.filter((h): h is string => typeof h === 'string' && HERO_SET.has(h))
}

const SHAPE = `Return ONLY compact JSON with this exact shape:
{"map": string|null, "leftTeam": string[], "rightTeam": string[], "bansLeft": string[], "bansRight": string[], "pendingLeft": string[], "pendingRight": string[], "previewHero": string|null}

Rules:
- Use hero names EXACTLY from this list (no other spellings): ${HEROES.join(', ')}.
- Use map names EXACTLY from this list, or null if unclear: ${MAPS.join(', ')}.
- List each team's slots top-to-bottom.
- If you cannot identify a slot's hero confidently, omit it rather than guess.`

// Preferred path: the client sends CROPS of the draft screen, not the whole thing. Which
// team a hero belongs to is then decided by which crop it appeared in — the model never
// has to work out the layout, which is exactly what it kept getting wrong.
const REGION_PROMPT = `You are reading cropped regions of a Heroes of the Storm ranked draft screen.

You get two images:

IMAGE 1 — the two pick columns, side by side under drawn labels.
- LEFT half, under "OUR TEAM": the player's own 5 pick slots, top to bottom.
- RIGHT half, under "ENEMY TEAM": the enemy's 5 pick slots, top to bottom.
- The halves are separated by a black gap. A hero on the left half belongs to OUR TEAM and a
  hero on the right half belongs to the ENEMY. This is fixed by how the image was built —
  never move a hero from one side to the other, and never report a hero on both sides.
- Report OUR TEAM in leftTeam and the ENEMY in rightTeam.

IMAGE 2 — bans and the map, stacked under drawn labels "MAP", "OUR BANS", "ENEMY BANS".
- Report them in map, bansLeft and bansRight respectively.
- These strips are cropped from fixed screen positions, so a strip can legitimately contain
  NOTHING but background art — empty space, starfield, nebula, scenery. THIS IS NORMAL and it
  is not your job to fill it in.
- A ban is a small HEXAGON containing a hero portrait, usually with a small padlock on it.
  If a strip contains no such hexagons, return an EMPTY array for it. Never infer a ban from
  the background, from what heroes are common, or from what would make a plausible draft.
  Returning [] is always better than guessing — a guessed ban corrupts the recommendations.
- Read "map" ONLY from the map NAME printed as large text in the MAP strip. If no map name
  text is legible there, return null. Do NOT identify the map from the background scenery.

READING A PICK SLOT
Each slot is one hexagon. A filled slot has the HERO NAME printed on a banner beside it, with
the player's name in smaller text under it. READ THE PRINTED HERO NAME — trust the text over
the portrait art, because skins change the art but the name is always correct.

SLOT STATE — classify each slot as exactly one of:
- LOCKED: a portrait fills the hexagon, bright and fully coloured, sitting settled in its slot.
  This is the normal state of a filled slot. Report it in leftTeam/rightTeam.
- PENDING: the player has "shown" a hero to their team but has NOT locked it in. It looks
  visibly less finished than the locked slots around it — the portrait is dimmed, greyed or
  desaturated, or the hexagon carries a glowing / animated / highlighted border. Report it in
  pendingLeft/pendingRight instead — never in leftTeam/rightTeam.
  Judge this by COMPARING slots within the same column: pending only means anything relative
  to how the locked slots in that same image look. Note that every slot's name banner is a
  pale grey plate whether or not the hero is locked — the banner colour is NOT the cue.
  If you cannot see a clear difference between this slot and the others, call it LOCKED.
- EMPTY: a dark, empty hexagon with NO portrait art in it. The player's name may still be shown
  beside it — that does NOT make it filled. Report nothing for it.

Compare the slots in a column AGAINST EACH OTHER: locked and pending look obviously different
side by side, and at most one slot per team is pending at a time. A hero can legitimately appear
as LOCKED in one slot and PENDING in another — report each slot as what it is; do not suppress
one because the same hero appears twice.

Most slots are EMPTY early in a draft. Only report a hero you can actually see in a hexagon.
previewHero is always null here — these crops contain no centre splash.

${SHAPE}
- If the images show no draft furniture at all (gameplay, a menu, blank panels), return all
  empty arrays and nulls.`

// Fallback path: whole screen in one image. Only used when the client's crop ROIs don't fit
// the player's resolution.
const FULL_PROMPT = `You are reading a Heroes of the Storm ranked draft screen (a full screenshot).

LAYOUT
- The player's own team is the vertical column of hexagon portraits down the LEFT edge (5 pick slots).
- The enemy team is the column down the RIGHT edge (5 pick slots).
- Bans are the small hexagon slots along the TOP-LEFT and TOP-RIGHT.
- The CENTRE of the screen shows one or two large hero splashes: the hero(es) being picked
  right now. These are NOT slots, and the hero shown there is usually NOT yet on the board.
  Report them only in previewHero.
- The row of portraits along the BOTTOM is the hero pool — heroes available to pick, NOT picked.
  Never report anything from it.

Only report a hero as picked if you can see it in one of the two edge COLUMNS.

SLOT STATE — classify every non-empty slot into exactly one of these:
- LOCKED: bright, fully colored, saturated, opaque portrait, sitting settled in its slot.
- PENDING: the team on the clock has highlighted a hero but has NOT confirmed it yet. The
  portrait is dim / dark / greyed / desaturated / semi-transparent, and its slot is the
  active one (glowing or animated border). The same hero is usually ALSO a large centre splash.
- EMPTY: no hero at all.

Report LOCKED heroes in leftTeam / rightTeam / bansLeft / bansRight.
Report PENDING heroes in pendingLeft / pendingRight. Never put a hero in both.

Early in a draft most slots are EMPTY. Do NOT fill a team out to 5 heroes unless you can
clearly see 5 bright, fully-colored portraits in that column — omit every empty slot.

${SHAPE}
- This is NOT the draft screen if you see gameplay/minimap/health bars — then return all empty arrays and nulls.`

export async function POST(req: Request) {
  if (!authorized(req)) return NextResponse.json({ error: 'unauthorized' }, { status: 401 })

  const apiKey = process.env.OPENAI_API_KEY
  if (!apiKey) return NextResponse.json({ error: 'OPENAI_API_KEY not configured' }, { status: 500 })

  const asUrl = (img: unknown): string => {
    if (typeof img !== 'string' || img.length < 100) throw new Error('bad image')
    return img.startsWith('data:') ? img : `data:image/jpeg;base64,${img}`
  }

  // { regions: [{ label, imageBase64 }] } is the crop path; a bare { imageBase64 } is the
  // full-frame fallback (and what older clients send).
  let images: string[]
  let cropped: boolean
  try {
    const body = await req.json()
    if (Array.isArray(body.regions) && body.regions.length > 0) {
      images = body.regions.map((r: { imageBase64: unknown }) => asUrl(r?.imageBase64))
      cropped = !body.regions.some((r: { label?: unknown }) => r?.label === 'fullscreen')
    } else {
      images = [asUrl(body.imageBase64)]
      cropped = false
    }
  } catch {
    return NextResponse.json(
      { error: 'body must be { regions: [{label, imageBase64}] } or { imageBase64: <data-url or base64> }' },
      { status: 400 },
    )
  }

  const openai = new OpenAI({ apiKey })
  // gpt-4o by default: mini could not reliably tell a dim/previewed portrait from a
  // bright locked one. Costs about the same per call here — mini's cheaper tokens are
  // offset by its much heavier image tokenization. Override with VISION_MODEL.
  const model = process.env.VISION_MODEL || 'gpt-4o'
  try {
    const resp = await openai.chat.completions.create({
      model,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: cropped ? REGION_PROMPT : FULL_PROMPT },
            ...images.map((url) => ({
              type: 'image_url' as const,
              image_url: { url, detail: 'high' as const },
            })),
          ],
        },
      ],
      response_format: { type: 'json_object' },
      max_tokens: 400,
      temperature: 0,
    })

    const raw = resp.choices[0]?.message?.content ?? '{}'
    let parsed: Record<string, unknown>
    try { parsed = JSON.parse(raw) } catch { parsed = {} }

    return NextResponse.json({
      map: typeof parsed.map === 'string' && MAPS.includes(parsed.map) ? parsed.map : null,
      leftTeam: clean(parsed.leftTeam),
      rightTeam: clean(parsed.rightTeam),
      bansLeft: clean(parsed.bansLeft),
      bansRight: clean(parsed.bansRight),
      pendingLeft: clean(parsed.pendingLeft),
      pendingRight: clean(parsed.pendingRight),
      previewHero: typeof parsed.previewHero === 'string' && HERO_SET.has(parsed.previewHero)
        ? parsed.previewHero
        : null,
      // Echoed back so the overlay can log real per-call token usage — the vision
      // models tokenize images very differently, so measuring beats estimating.
      model,
      promptTokens: resp.usage?.prompt_tokens ?? 0,
      completionTokens: resp.usage?.completion_tokens ?? 0,
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    return NextResponse.json({ error: 'vision error: ' + message }, { status: 502 })
  }
}
