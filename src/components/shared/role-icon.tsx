/**
 * Tiny inline SVG role icons for each HotS role class.
 * Original designs inspired by each role's gameplay identity.
 */

import type { HeroRole } from '@/lib/data/hero-roles'

interface RoleIconProps {
  role: HeroRole
  size?: number
  className?: string
}

export function RoleIcon({ role, size = 12, className }: RoleIconProps) {
  const props = {
    width: size,
    height: size,
    viewBox: '0 0 16 16',
    fill: 'currentColor',
    className,
  }

  switch (role) {
    case 'Tank':
      // Shield
      return (
        <svg {...props}>
          <path d="M8 1L2 4v4c0 3.5 2.5 6.2 6 7 3.5-.8 6-3.5 6-7V4L8 1zm0 2.2L12 5.5v2.8c0 2.6-1.8 4.6-4 5.2-2.2-.6-4-2.6-4-5.2V5.5L8 3.2z" />
        </svg>
      )
    case 'Bruiser':
      // Fist
      return (
        <svg {...props}>
          <path d="M5 9V5.5a1 1 0 012 0V3a1 1 0 012 0v-.5a1 1 0 012 0V5a1 1 0 012 0v4c0 2.8-2.2 5-5 5H7c-1.7 0-3-1.3-3-3V9.5a1 1 0 012 0V9z" />
        </svg>
      )
    case 'Ranged Assassin':
      // Bow and arrow
      return (
        <svg {...props}>
          <path d="M13 1l-2 .5L4.5 8 3.3 6.8 1.5 8.5l2 2L2 12l-.5 2.5L4 14l2-1.5 2-2L6.8 9.3 8 8l-1.2-1.2L13 1zM3.5 12.5l-.5-1 .5-.5 1 .5-.5.5-.5.5z" />
          <path d="M12.5 1.5C10 4 8.5 5.5 7 7l1 1c1.5-1.5 3-3 5.5-5.5V1.5z" opacity="0.5" />
        </svg>
      )
    case 'Melee Assassin':
      // Crossed swords
      return (
        <svg {...props}>
          <path d="M2 1l4.5 4.5-1.2 1.2L2 3.4V1zm12 0v2.4L10.7 6.7l-1.2-1.2L14 1zM7.3 8.5L3.7 12.1l1.2 1.2L8.5 9.7 7.3 8.5zm1.4 0l1.2 1.2 3.4-3.4-1.2-1.2L8.7 8.5zM8 7.3L6.7 8l1.3 1.3L9.3 8 8 7.3z" />
        </svg>
      )
    case 'Healer':
      // Medical cross
      return (
        <svg {...props}>
          <path d="M6 2h4v4h4v4h-4v4H6v-4H2V6h4V2z" />
        </svg>
      )
    case 'Support':
      // Gear/cog
      return (
        <svg {...props}>
          <path d="M8 5.5a2.5 2.5 0 100 5 2.5 2.5 0 000-5zM6.8 1l-.5 2.1c-.4.2-.8.4-1.1.7L3.2 3 1.6 5.8l1.7 1.3c0 .3-.1.6-.1.9s0 .6.1.9L1.6 10.2 3.2 13l2-.8c.3.3.7.5 1.1.7L6.8 15h3.4l.5-2.1c.4-.2.8-.4 1.1-.7l2 .8 1.6-2.8-1.7-1.3c0-.3.1-.6.1-.9s0-.6-.1-.9l1.7-1.3L13.8 3l-2 .8c-.3-.3-.7-.5-1.1-.7L10.2 1H6.8z" />
        </svg>
      )
  }
}
