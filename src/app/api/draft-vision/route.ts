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

const PROMPT = `You are reading a Heroes of the Storm ranked draft screen (a screenshot).

LAYOUT
- The player's own team is the vertical column of hexagon portraits down the LEFT edge (5 pick slots).
- The enemy team is the column down the RIGHT edge (5 pick slots).
- Bans are the small hexagon slots along the TOP-LEFT and TOP-RIGHT.
- The CENTRE of the screen shows one large portrait: the hero the player currently on the
  clock is previewing. It is not a slot.

SLOT STATE — classify every non-empty slot into exactly one of these:
- LOCKED: bright, fully colored, saturated, opaque portrait, sitting settled in its slot.
- PENDING: the team on the clock has highlighted a hero but has NOT confirmed it yet. The
  portrait is dim / dark / greyed / desaturated / semi-transparent, and its slot is the
  active one (glowing or animated border). The same hero is usually ALSO the large centre portrait.
- EMPTY: no hero at all.

Report LOCKED heroes in leftTeam / rightTeam / bansLeft / bansRight.
Report PENDING heroes in pendingLeft / pendingRight. Never put a hero in both.
Report the large centre portrait's hero in previewHero (null if there isn't one).

At most ONE slot on the whole screen is PENDING at any moment — only one team is on the clock.
If you are unsure whether a slot is locked or pending and that hero matches the centre
portrait, call it PENDING.

Early in a draft most slots are EMPTY. Do NOT fill a team out to 5 heroes unless you can
clearly see 5 bright, fully-colored portraits in that column — omit every empty slot.

Return ONLY compact JSON with this exact shape:
{"map": string|null, "leftTeam": string[], "rightTeam": string[], "bansLeft": string[], "bansRight": string[], "pendingLeft": string[], "pendingRight": string[], "previewHero": string|null}

Rules:
- Use hero names EXACTLY from this list (no other spellings): ${HEROES.join(', ')}.
- Use map names EXACTLY from this list, or null if unclear: ${MAPS.join(', ')}.
- List each team's slots top-to-bottom.
- If you cannot identify a slot's hero confidently, omit it rather than guess.
- This is NOT the draft screen if you see gameplay/minimap/health bars — then return all empty arrays and nulls.`

export async function POST(req: Request) {
  if (!authorized(req)) return NextResponse.json({ error: 'unauthorized' }, { status: 401 })

  const apiKey = process.env.OPENAI_API_KEY
  if (!apiKey) return NextResponse.json({ error: 'OPENAI_API_KEY not configured' }, { status: 500 })

  let imageUrl: string
  try {
    const body = await req.json()
    const img = body.imageBase64
    if (typeof img !== 'string' || img.length < 100) throw new Error('bad image')
    imageUrl = img.startsWith('data:') ? img : `data:image/jpeg;base64,${img}`
  } catch {
    return NextResponse.json({ error: 'body must be { imageBase64: <data-url or base64> }' }, { status: 400 })
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
            { type: 'text', text: PROMPT },
            { type: 'image_url', image_url: { url: imageUrl, detail: 'high' } },
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
