'use client'

import { cn } from '@/lib/utils'
import { heroImageSrc } from '@/lib/data/hero-images'
import {
  HEX_CLIP, HEX_VERTICES, CHEVRON_CLIP, EDGE_DEG,
  METALLIC_FRAME, METALLIC_DARK, BANNER_TEXT,
  TEAM_COLORS, type TeamAccent,
} from './constants'

export type BannerSide = 'lower-left' | 'lower-right'

export interface HexTileProps {
  hero: string | null
  size: number
  accent: TeamAccent
  banned?: boolean
  isCurrent?: boolean
  isSkipped?: boolean
  onClick?: () => void
  /** If provided, renders the hero name banner along the given lower edge */
  bannerSide?: BannerSide
  /** Dim / grayscale display */
  dim?: boolean
  title?: string
}

/**
 * Pointy-top hex portrait with steel frame, chevron corner accents, and an
 * optional team-colored name banner running along one lower edge.
 *
 * All sizing is proportional to `size` (square bounding box of the hex).
 */
export function HexTile({
  hero, size, accent, banned, isCurrent, isSkipped,
  onClick, bannerSide, dim, title,
}: HexTileProps) {
  const team = TEAM_COLORS[accent]
  const showName = !!bannerSide
  const isRight = bannerSide === 'lower-right'

  const EDGE_LEN = size * Math.sqrt(5) / 4
  const TAIL_PX = Math.max(10, Math.round(size / 9))
  const BANNER_W = EDGE_LEN + TAIL_PX
  const BANNER_H = Math.max(10, Math.round(size / 9))

  // Outer padding so the protruding tail isn't clipped
  const padX = showName ? Math.ceil(TAIL_PX + BANNER_H) : 0
  const outerW = size + padX * 2
  const outerH = size

  const interactive = !!onClick
  return (
    <div
      className={cn('relative', interactive && 'cursor-pointer group')}
      style={{ width: outerW, height: outerH }}
      onClick={onClick}
      title={title}
    >
      <div
        className="absolute"
        style={{ width: size, height: size, left: padX, top: 0 }}
      >
        {/* Outer hex frame — brushed steel */}
        <div
          className="absolute inset-0"
          style={{ clipPath: HEX_CLIP, background: METALLIC_FRAME }}
        />
        {/* Thin inner team-color ring for identity */}
        <div
          className="absolute"
          style={{ inset: 2, clipPath: HEX_CLIP, background: team.frame }}
        />
        {/* Portrait area */}
        <div
          className="absolute inset-[4px] bg-[#0a0d1f] overflow-hidden"
          style={{ clipPath: HEX_CLIP }}
        >
          {hero ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={heroImageSrc(hero)}
              alt=""
              loading="lazy"
              className={cn(
                'w-full h-full object-cover transition',
                (banned || dim) && 'grayscale opacity-70',
                interactive && 'group-hover:brightness-110',
              )}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              {isSkipped ? (
                <span className="text-[10px] italic text-yellow-500/60">Missed</span>
              ) : isCurrent ? (
                <span className="text-[10px] text-[#d6dbe0] animate-pulse">Select</span>
              ) : (
                <span className="text-[#d6dbe0]/20 text-xs">—</span>
              )}
            </div>
          )}
        </div>

        {/* Current-turn ring */}
        {isCurrent && !hero && (
          <div
            className="absolute inset-0 animate-pulse"
            style={{
              clipPath: HEX_CLIP,
              boxShadow: `inset 0 0 0 2px ${team.frame}`,
            }}
          />
        )}

        {/* Ban slash overlay */}
        {banned && hero && (
          <div
            className="absolute inset-0 flex items-center justify-center"
            style={{ clipPath: HEX_CLIP }}
          >
            <div className="w-[140%] h-[3px] bg-[#d46b6b] rotate-[20deg] shadow-[0_0_8px_#d46b6b]" />
          </div>
        )}

        {/* Chevron accents at vertices not covered by the banner */}
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

        {/* Shadow banner (dark strip behind the main banner, no tail) */}
        {showName && hero && (
          <div
            className="absolute pointer-events-none"
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

        {/* Main banner with team-color tip that protrudes past the vertical side */}
        {showName && hero && (
          <div
            className="absolute pointer-events-none"
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
              <div
                style={{
                  width: TAIL_PX,
                  height: '100%',
                  background: `linear-gradient(to ${isRight ? 'left' : 'right'}, ${team.tip} 0%, ${team.tip} 60%, ${team.frame} 100%)`,
                  order: isRight ? 2 : 0,
                  flexShrink: 0,
                }}
              />
              <div
                className="flex-1 flex items-center justify-center px-1 overflow-hidden"
                style={{
                  order: 1,
                  transform: `skewX(${isRight ? EDGE_DEG : -EDGE_DEG}deg)`,
                }}
              >
                <span
                  className="truncate font-semibold tracking-wide whitespace-nowrap"
                  style={{
                    color: BANNER_TEXT,
                    fontSize: Math.max(8, Math.round(size / 14)),
                  }}
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
