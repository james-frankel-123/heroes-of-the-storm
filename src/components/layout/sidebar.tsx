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
} from 'lucide-react'
import { cn } from '@/lib/utils'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Draft Assistant', href: '/draft', icon: Sparkles },
  { name: 'Heroes', href: '/heroes', icon: Users },
  { name: 'Maps', href: '/maps', icon: Map },
  { name: 'Teams', href: '/teams', icon: Users },
  { name: 'Insights', href: '/insights', icon: TrendingUp },
  { name: 'Statistics', href: '/stats', icon: Trophy },
  { name: 'Settings', href: '/settings', icon: Settings },
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <div className="flex h-full w-64 flex-col border-r border-border bg-card/50 backdrop-blur-xl">
      {/* Logo */}
      <div className="flex h-16 items-center gap-3 border-b border-border px-6">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary-600 to-primary-500 shadow-lg shadow-primary-500/50">
          <Trophy className="h-6 w-6 text-white" />
        </div>
        <div>
          <h1 className="text-lg font-bold glow">HotS Analytics</h1>
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
            </Link>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="border-t border-border p-4">
        <div className="rounded-lg bg-gradient-to-br from-primary-500/10 to-accent-cyan/10 p-4">
          <p className="text-sm font-medium">Need Help?</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Check out our guide to improve your gameplay
          </p>
          <button className="mt-2 text-xs font-medium text-primary-500 hover:text-primary-600">
            Learn More →
          </button>
        </div>
      </div>
    </div>
  )
}
