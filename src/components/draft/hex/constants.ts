/**
 * Shared visual constants for the HotS-inspired draft layout.
 */

// Pointy-top hex: vertical left/right edges, triangle top & bottom
export const HEX_CLIP = 'polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%)'

// Hex vertices (fractional coords) + outward rotation for chevron accents
export const HEX_VERTICES: { x: number; y: number; rot: number }[] = [
  { x: 0.5, y: 0.0,  rot:   0     }, // top
  { x: 1.0, y: 0.25, rot:  63.43  }, // upper-right
  { x: 1.0, y: 0.75, rot: 116.57  }, // lower-right
  { x: 0.5, y: 1.0,  rot: 180     }, // bottom
  { x: 0.0, y: 0.75, rot: -116.57 }, // lower-left
  { x: 0.0, y: 0.25, rot: -63.43  }, // upper-left
]

export const CHEVRON_CLIP = 'polygon(50% 0%, 100% 55%, 78% 75%, 50% 35%, 22% 75%, 0% 55%)'

// Pointy-top hex lower-edge angle: atan(0.25 / 0.5) ≈ 26.57°
export const EDGE_DEG = 26.57

// Brushed-steel gradient for hex frame + chevrons
export const METALLIC_FRAME =
  'linear-gradient(135deg, #2d3138 0%, #5c636c 18%, #9ea5ae 42%, #d6dbe0 52%, #858c96 70%, #3e434a 100%)'
export const METALLIC_DARK = '#1a1d22'
export const BANNER_TEXT = '#2b1d10'

export type TeamAccent = 'blue' | 'red'

export const TEAM_COLORS: Record<TeamAccent, {
  frame: string
  tip: string
  body: string
}> = {
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
}
