'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import { Card } from './card'

interface StatCardProps extends React.HTMLAttributes<HTMLDivElement> {
  label: string
  value: string | number
  icon?: React.ReactNode
  trend?: {
    value: number
    label: string
  }
  animated?: boolean
  valueColor?: string
}

export function StatCard({
  label,
  value,
  icon,
  trend,
  animated = true,
  valueColor,
  className,
  ...props
}: StatCardProps) {
  const [isHovered, setIsHovered] = React.useState(false)

  const cardVariants = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    hover: { y: -5, transition: { duration: 0.2 } },
  }

  const Content = (
    <div
      className={cn(
        'group relative overflow-hidden rounded-xl border-2 border-primary-500/30 bg-gradient-to-br from-card to-card/50 p-6 transition-all duration-300',
        'hover:border-primary-500/60 hover:shadow-lg hover:shadow-primary-500/20',
        className
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      {...props}
    >
      {/* Background glow effect */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary-500/10 to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100" />

      <div className="relative z-10 flex flex-col space-y-3">
        {/* Label and Icon */}
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
            {label}
          </span>
          {icon && (
            <div className="text-primary-500/50 transition-colors duration-300 group-hover:text-primary-500">
              {icon}
            </div>
          )}
        </div>

        {/* Value */}
        <div className={cn('text-4xl font-bold tracking-tight', valueColor)}>
          {value}
        </div>

        {/* Trend */}
        {trend && (
          <div className="flex items-center gap-2 text-sm">
            <span
              className={cn(
                'font-semibold',
                trend.value > 0 ? 'text-gaming-success' : 'text-gaming-danger'
              )}
            >
              {trend.value > 0 ? '+' : ''}
              {trend.value}%
            </span>
            <span className="text-muted-foreground">{trend.label}</span>
          </div>
        )}
      </div>
    </div>
  )

  if (animated) {
    return (
      <motion.div
        variants={cardVariants}
        initial="initial"
        animate="animate"
        whileHover="hover"
      >
        {Content}
      </motion.div>
    )
  }

  return Content
}
