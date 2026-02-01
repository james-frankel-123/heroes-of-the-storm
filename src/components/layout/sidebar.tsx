'use client'

import * as React from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  LayoutDashboard,
  Users,
  Map,
  Trophy,
  TrendingUp,
  Settings,
  Sparkles,
  Lock,
} from 'lucide-react'
import { cn } from '@/lib/utils'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Heroes', href: '/heroes', icon: Users },
  { name: 'Maps', href: '/maps', icon: Map },
  { name: 'Teams', href: '/teams', icon: Users },
  { name: 'Draft Assistant', href: '/draft', icon: Sparkles, locked: true },
  { name: 'Statistics', href: '/stats', icon: Trophy, locked: true },
  { name: 'Settings', href: '/settings', icon: Settings },
]

export function Sidebar() {
  const pathname = usePathname()
  const [hasAnimated, setHasAnimated] = React.useState(false)

  // Trigger animation on mount
  React.useEffect(() => {
    const timer = setTimeout(() => {
      setHasAnimated(true)
    }, 200)
    return () => clearTimeout(timer)
  }, [])

  // Heartbeat animation sequence
  const heartbeatVariants = {
    rest: {
      scale: 1,
      filter: 'drop-shadow(0 0 8px rgba(255, 107, 107, 0.4))',
    },
    pulse: {
      scale: [1, 1.08, 1, 1.08, 1, 1.08, 1],
      filter: [
        'drop-shadow(0 0 8px rgba(255, 107, 107, 0.4))',
        'drop-shadow(0 0 20px rgba(255, 107, 107, 0.8))',
        'drop-shadow(0 0 8px rgba(255, 107, 107, 0.4))',
        'drop-shadow(0 0 20px rgba(255, 107, 107, 0.8))',
        'drop-shadow(0 0 8px rgba(255, 107, 107, 0.4))',
        'drop-shadow(0 0 20px rgba(255, 107, 107, 0.8))',
        'drop-shadow(0 0 8px rgba(255, 107, 107, 0.4))',
      ],
      transition: {
        duration: 2.5,
        times: [0, 0.12, 0.24, 0.44, 0.56, 0.76, 1],
        ease: 'easeInOut',
      },
    },
    hover: {
      scale: 1.05,
      filter: 'drop-shadow(0 0 24px rgba(255, 107, 107, 0.9))',
      transition: {
        duration: 0.3,
      },
    },
  }

  return (
    <div className="flex h-full w-64 flex-col border-r border-border bg-card/50 backdrop-blur-xl">
      {/* Logo */}
      <div className="flex h-16 items-center gap-3 border-b border-border px-6">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary-600 to-primary-500 shadow-lg shadow-primary-500/50">
          <Trophy className="h-6 w-6 text-white" />
        </div>
        <div>
          <motion.h1
            className="text-lg font-bold"
            initial="rest"
            animate={hasAnimated ? 'pulse' : 'rest'}
            whileHover="hover"
          >
            <span className="glow">Hots </span>
            <motion.span
              className="fever-text fever-shimmer inline-block"
              variants={heartbeatVariants}
            >
              fever
            </motion.span>
          </motion.h1>
          <p className="text-xs text-muted-foreground">Storm League Tracker</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 p-4">
        {navigation.map((item) => {
          const isActive = pathname === item.href
          const Icon = item.icon

          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                'group relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all',
                isActive
                  ? 'bg-primary-500/10 text-primary-500'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground'
              )}
            >
              {isActive && (
                <motion.div
                  className="absolute left-0 h-8 w-1 rounded-r-full bg-primary-500"
                  layoutId="sidebar-indicator"
                  transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                />
              )}
              <Icon className="h-5 w-5" />
              <span>{item.name}</span>
              {item.locked && (
                <Lock className="ml-auto h-4 w-4 text-muted-foreground" />
              )}
            </Link>
          )
        })}
      </nav>
    </div>
  )
}
