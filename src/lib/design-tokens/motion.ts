export const motion = {
  // Duration
  duration: {
    instant: '0ms',
    fast: '100ms',
    normal: '200ms',
    slow: '300ms',
    slower: '500ms',
    slowest: '800ms',
  },

  // Easing curves
  easing: {
    linear: 'linear',
    easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
    easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
    easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
    spring: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
  },

  // Spring configurations (for framer-motion)
  spring: {
    soft: {
      type: 'spring',
      stiffness: 300,
      damping: 30,
    },
    bouncy: {
      type: 'spring',
      stiffness: 400,
      damping: 20,
    },
    stiff: {
      type: 'spring',
      stiffness: 500,
      damping: 40,
    },
  },

  // Transitions
  transition: {
    default: {
      duration: 0.2,
      ease: [0.4, 0, 0.2, 1],
    },
    fast: {
      duration: 0.1,
      ease: [0.4, 0, 0.2, 1],
    },
    slow: {
      duration: 0.3,
      ease: [0.4, 0, 0.2, 1],
    },
  },
} as const

export type MotionToken = typeof motion
