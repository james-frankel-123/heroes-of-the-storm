'use client'

import { heroImageSrc } from '@/lib/data/hero-images'
import { cn } from '@/lib/utils'

/**
 * Static visual prototype of a HotS-inspired draft layout.
 * No real data, no search — just the layout + aesthetic for review.
 */

const TEAM_BLUE: (string | null)[] = ['Malthael', 'Johanna', 'Jaina', 'Tyrande', null]
const TEAM_RED: (string | null)[]  = ['Illidan', 'E.T.C.', 'Kael\'thas', null, null]
const BANS_BLUE = ['Xul', 'Rehgar', 'Gall']
const BANS_RED  = ['Johanna', 'Chromie', null] as (string | null)[]

const PICKER_POOL = [
  'Alarak', 'Artanis', 'Azmodan', 'Brightwing', 'Cassia', 'Dehaka',
  'Falstad', 'Garrosh', 'Gazlowe', 'Greymane', 'Hanzo', 'Imperius',
  'Leoric', 'Li-Ming', 'Lt. Morales', 'Lunara', 'Maiev', 'Mei',
  'Mephisto', 'Muradin', 'Nazeebo', 'Orphea', 'Raynor', 'Rexxar',
  'Samuro', 'Sonya', 'Stitches', 'Stukov', 'Sylvanas', 'Thrall',
  'Tracer', 'Tyrael', 'Uther', 'Valla', 'Whitemane', 'Zeratul',
]

const SEARCH_RECS: { hero: string; delta: number; byline?: string }[] = [
  { hero: 'Malthael', delta: 20.8, byline: 'sirwatsonii +20.8 on Malthael (map)' },
  { hero: 'Samuro',   delta: 13.6 },
  { hero: 'Gazlowe',  delta: 13.0 },
  { hero: 'Falstad',  delta: 11.2 },
  { hero: 'Valla',    delta:  8.4 },
]

// Pointy-top hex: vertical left/right edges, triangle-shaped top and bottom.
const HEX_CLIP = 'polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%)'

// Hex vertex positions (fractional) + outward rotation for the chevron.
// Outward angles are relative to a chevron pointing UP by default.
const HEX_VERTICES: { x: number; y: number; rot: number }[] = [
  { x: 0.5, y: 0.0,  rot:  0     },  // top
  { x: 1.0, y: 0.25, rot:  63.43 },  // upper-right
  { x: 1.0, y: 0.75, rot:  116.57},  // lower-right
  { x: 0.5, y: 1.0,  rot:  180   },  // bottom
  { x: 0.0, y: 0.75, rot: -116.57},  // lower-left
  { x: 0.0, y: 0.25, rot: -63.43 },  // upper-left
]

export default function DraftPreviewPage() {
  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_#1a1f3a_0%,_#0a0d1f_70%)] text-[#e8d8a0]">
      <div className="max-w-[1400px] mx-auto px-6 py-8 space-y-6">

        <div className="text-center">
          <h1 className="text-2xl tracking-[0.3em] text-[#d4b85a] font-light">DRAFT</h1>
          <p className="text-xs text-[#8b9bc8] mt-1">Towers of Doom · Pick Phase</p>
        </div>

        {/* Ban bar */}
        <div className="flex items-center justify-center gap-8 pb-6 border-b border-[#d4b85a]/20">
          <BanGroup label="BLUE BANS" bans={BANS_BLUE} accent="blue" />
          <div className="text-[#d4b85a]/60 text-xs tracking-widest">VS</div>
          <BanGroup label="RED BANS" bans={BANS_RED} accent="red" />
        </div>

        {/* Three-column body */}
        <div className="grid grid-cols-[220px_1fr_220px] gap-8">
          {/* Blue team */}
          <TeamColumn picks={TEAM_BLUE} accent="blue" label="BLUE TEAM" />

          {/* Center — picker + recs */}
          <div className="space-y-6">
            <Recommendations />
            <HeroPickerGrid heroes={PICKER_POOL} />
          </div>

          {/* Red team */}
          <TeamColumn picks={TEAM_RED} accent="red" label="RED TEAM" />
        </div>
      </div>
    </div>
  )
}

function BanGroup({ label, bans, accent }: { label: string; bans: (string | null)[]; accent: 'blue' | 'red' }) {
  const tint = accent === 'blue' ? 'text-[#6b8dd4]' : 'text-[#d46b6b]'
  return (
    <div className="flex flex-col items-center gap-2">
      <div className={cn('text-[10px] tracking-[0.3em] font-light', tint)}>{label}</div>
      <div className="flex gap-3">
        {bans.map((hero, i) => (
          <HexTile key={i} hero={hero} size={56} banned accent={accent} />
        ))}
      </div>
    </div>
  )
}

function TeamColumn({ picks, accent, label }: { picks: (string | null)[]; accent: 'blue' | 'red'; label: string }) {
  const tint = accent === 'blue' ? 'text-[#6b8dd4]' : 'text-[#d46b6b]'
  // Zigzag: alternate picks shift outward. Blue column shifts even picks left,
  // odd picks right (so picks snake toward/away from center). Red mirrors it.
  const zigOffset = 24
  return (
    <div className="flex flex-col items-center gap-2">
      <div className={cn('text-xs tracking-[0.3em] font-light mb-1', tint)}>{label}</div>
      {picks.map((hero, i) => {
        const even = i % 2 === 0
        const shift = accent === 'blue'
          ? (even ? -zigOffset : zigOffset)
          : (even ? zigOffset : -zigOffset)
        return (
          <div key={i} style={{ transform: `translateX(${shift}px)` }}>
            <HexTile
              hero={hero}
              size={128}
              accent={accent}
              showName
              bannerSide={accent === 'blue' ? 'lower-left' : 'lower-right'}
            />
          </div>
        )
      })}
    </div>
  )
}

// Team color palette used across banner body, tip, and chevrons
const TEAM_COLORS = {
  blue: {
    frame: '#6b8dd4',
    tip:   '#4a6fb8',
    body:  'linear-gradient(to bottom, #e8ecf4 0%, #c7d3eb 60%, #a8b9d8 100%)',
  },
  red: {
    frame: '#d46b6b',
    tip:   '#b84a4a',
    body:  'linear-gradient(to bottom, #f4e8e8 0%, #ebc7c7 60%, #d8a8a8 100%)',
  },
} as const

// Brushed-steel gradient used for the hex frame and chevron accents
const METALLIC_FRAME =
  'linear-gradient(135deg, #2d3138 0%, #5c636c 18%, #9ea5ae 42%, #d6dbe0 52%, #858c96 70%, #3e434a 100%)'
const METALLIC_DARK = '#1a1d22'
const TEXT_COLOR = '#2b1d10'

// Upward-pointing chevron (V shape), rotated per vertex to point outward
const CHEVRON_CLIP = 'polygon(50% 0%, 100% 55%, 78% 75%, 50% 35%, 22% 75%, 0% 55%)'

function HexTile({
  hero, size, banned, accent, showName, bannerSide = 'lower-right',
}: {
  hero: string | null
  size: number
  banned?: boolean
  accent: 'blue' | 'red'
  showName?: boolean
  bannerSide?: 'lower-left' | 'lower-right'
}) {
  const team = TEAM_COLORS[accent]
  // Pointy-top hex: lower edges run at ±26.57° from horizontal
  // (atan2(size*0.25, size*0.5)). Edge length = size*sqrt(5)/4 (same).
  // The banner body sits INSIDE the hex along that edge; an extra tail
  // overshoots past the bottom vertex with a gold tip.
  const EDGE_DEG = 26.57
  const EDGE_LEN = size * Math.sqrt(5) / 4
  const TAIL_PX = 14 // how far the banner protrudes past the vertical side
  const BANNER_W = EDGE_LEN + TAIL_PX
  const BANNER_H = 14
  const isRight = bannerSide === 'lower-right'
  // Outer padding so the protruding tail isn't clipped by the bounding box
  const padX = showName ? Math.ceil(TAIL_PX + BANNER_H) : 0
  const padB = 0
  const outerW = size + padX * 2
  const outerH = size
  return (
    <div className="relative" style={{ width: outerW, height: outerH }}>
      <div
        className="absolute"
        style={{ width: size, height: size, left: padX, top: 0 }}
      >
        {/* Hex is centered within the wider outer wrapper; banner tails
            extend into the left/right padding region. */}
        {/* Outer hex frame — metallic gold */}
        <div
          className="absolute inset-0"
          style={{ clipPath: HEX_CLIP, background: METALLIC_FRAME }}
        />
        {/* Thin inner team-color ring for identity */}
        <div
          className="absolute"
          style={{
            inset: 2,
            clipPath: HEX_CLIP,
            background: team.frame,
          }}
        />
        {/* Inner hex inset (portrait area) */}
        <div
          className="absolute inset-[4px] bg-[#0a0d1f] overflow-hidden"
          style={{ clipPath: HEX_CLIP }}
        >
          {hero ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={heroImageSrc(hero)}
              alt={hero}
              className={cn('w-full h-full object-cover', banned && 'grayscale opacity-70')}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-[#d4b85a]/20 text-xs">
              empty
            </div>
          )}
        </div>
        {banned && hero && (
          <div
            className="absolute inset-0 flex items-center justify-center"
            style={{ clipPath: HEX_CLIP }}
          >
            <div className="w-[140%] h-[3px] bg-[#d46b6b] rotate-[20deg] shadow-[0_0_8px_#d46b6b]" />
          </div>
        )}

        {/* Steel chevron accents at each hex vertex (except those covered
            by the name banner). Each chevron points outward from hex center. */}
        {(() => {
          const chevW = Math.max(7, Math.round(size / 12))
          const chevH = Math.max(5, Math.round(size / 16))
          const covered = showName
            ? (isRight ? new Set([2, 3]) : new Set([3, 4]))
            : new Set<number>()
          return HEX_VERTICES.map((v, i) => {
            if (covered.has(i)) return null
            return (
              <div
                key={i}
                className="absolute pointer-events-none"
                style={{
                  left: v.x * size - chevW / 2,
                  top: v.y * size - chevH / 2,
                  width: chevW,
                  height: chevH,
                  background: METALLIC_FRAME,
                  clipPath: CHEVRON_CLIP,
                  transform: `rotate(${v.rot}deg)`,
                  filter: 'drop-shadow(0 0 1.5px rgba(0,0,0,0.6))',
                }}
              />
            )
          })
        })()}

        {/* Shadow banner: same transform as main banner but no tail and no
            text, with a translucent dark fill. Sits BEHIND the main banner
            (rendered first) to add depth under the hero name. */}
        {showName && hero && (
          <div
            className="absolute"
            style={{
              left: isRight ? size - EDGE_LEN : 0,
              top: size * 0.75 - BANNER_H + 4,
              width: EDGE_LEN,
              height: BANNER_H,
              transformOrigin: isRight ? '100% 100%' : '0% 100%',
              transform: isRight
                ? `rotate(${-EDGE_DEG}deg) skewX(${-EDGE_DEG}deg)`
                : `rotate(${EDGE_DEG}deg) skewX(${EDGE_DEG}deg)`,
              background: 'rgba(12, 10, 24, 0.65)',
              boxShadow: '0 1px 2px rgba(0,0,0,0.4) inset',
            }}
          />
        )}

        {/* Name banner — body sits INSIDE the hex along the lower diagonal,
            with a team-colored tip that protrudes past the vertical side. Both
            ends are cut vertically via a matching skew. */}
        {showName && hero && (
          <div
            className="absolute"
            style={{
              left: isRight ? size - BANNER_W + TAIL_PX : -TAIL_PX,
              top: size * 0.75 - BANNER_H,
              width: BANNER_W,
              height: BANNER_H,
              transformOrigin: isRight
                ? `calc(100% - ${TAIL_PX}px) 100%`
                : `${TAIL_PX}px 100%`,
              transform: isRight
                ? `rotate(${-EDGE_DEG}deg) skewX(${-EDGE_DEG}deg)`
                : `rotate(${EDGE_DEG}deg) skewX(${EDGE_DEG}deg)`,
            }}
          >
            <div
              className="w-full h-full flex items-center leading-none"
              style={{
                background: team.body,
                borderTop: `1px solid ${METALLIC_DARK}`,
                borderBottom: `1px solid ${METALLIC_DARK}`,
              }}
            >
              {/* Team-colored tip — protrudes past the vertical side */}
              <div
                style={{
                  width: TAIL_PX,
                  height: '100%',
                  background: `linear-gradient(to ${isRight ? 'left' : 'right'}, ${team.tip} 0%, ${team.tip} 60%, ${team.frame} 100%)`,
                  order: isRight ? 2 : 0,
                  flexShrink: 0,
                }}
              />
              {/* Un-skew the text so it renders straight (still inherits
                  the parent's rotation along the hex edge). */}
              <div
                className="flex-1 flex items-center justify-center px-1 overflow-hidden"
                style={{
                  order: 1,
                  transform: `skewX(${isRight ? EDGE_DEG : -EDGE_DEG}deg)`,
                }}
              >
                <span
                  className="truncate text-[9px] font-semibold tracking-wide whitespace-nowrap"
                  style={{ color: TEXT_COLOR }}
                >
                  {hero.toUpperCase()}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function Recommendations() {
  return (
    <div className="bg-[#0f1430]/60 border border-[#d4b85a]/20 rounded-sm p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm tracking-[0.2em] text-[#d4b85a] font-light">SEARCH RECOMMENDATIONS</h3>
        <span className="text-xs text-[#8b9bc8]">Depth 6</span>
      </div>
      <div className="space-y-2">
        {SEARCH_RECS.map((r, i) => (
          <div
            key={r.hero}
            className={cn(
              'flex items-center gap-3 p-2 rounded border transition-colors cursor-pointer',
              i === 0
                ? 'border-[#d4b85a]/50 bg-[#d4b85a]/5 hover:bg-[#d4b85a]/10'
                : 'border-[#6b8dd4]/20 bg-[#0a0d1f]/40 hover:bg-[#6b8dd4]/10'
            )}
          >
            <div style={{ width: 44, height: 44, clipPath: HEX_CLIP }} className="bg-[#0a0d1f] shrink-0">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={heroImageSrc(r.hero)} alt="" className="w-full h-full object-cover" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-[#e8d8a0]">{r.hero}</span>
                <span className={cn(
                  'text-sm font-bold tabular-nums',
                  r.delta >= 15 ? 'text-[#6fd46f]' : r.delta >= 5 ? 'text-[#d4b85a]' : 'text-[#8b9bc8]'
                )}>
                  +{r.delta.toFixed(1)}
                </span>
              </div>
              {r.byline && (
                <p className="text-[10px] text-[#b48ad4] mt-0.5">{r.byline}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function HeroPickerGrid({ heroes }: { heroes: string[] }) {
  return (
    <div className="bg-[#0f1430]/60 border border-[#d4b85a]/20 rounded-sm p-4">
      <h3 className="text-sm tracking-[0.2em] text-[#d4b85a] font-light mb-3">SELECT A HERO</h3>
      <div className="grid grid-cols-8 gap-2">
        {heroes.map((h) => (
          <button
            key={h}
            className="flex flex-col items-center gap-0.5 group"
          >
            <div
              style={{ clipPath: HEX_CLIP }}
              className="w-12 h-12 bg-[#0a0d1f] border-2 border-transparent group-hover:brightness-125 transition"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={heroImageSrc(h)} alt={h} className="w-full h-full object-cover" />
            </div>
            <span className="text-[9px] text-[#8b9bc8] truncate w-full text-center group-hover:text-[#e8d8a0]">
              {h}
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
