'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import { Shield, Sword, Swords, Target, Heart, Sparkles } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { calculateRoleBalance, analyzeRoleNeeds, RoleBalance } from '@/lib/data/hero-roles'

interface RoleBalanceIndicatorProps {
  picks: (string | null)[]
  teamName: string
  compact?: boolean
}

export function RoleBalanceIndicator({
  picks,
  teamName,
  compact = false
}: RoleBalanceIndicatorProps) {
  const balance = calculateRoleBalance(picks)
  const needs = analyzeRoleNeeds(balance)

  const roles = [
    { name: 'Tank', count: balance.tank, icon: Shield, color: 'text-blue-400' },
    { name: 'Bruiser', count: balance.bruiser, icon: Sword, color: 'text-purple-400' },
    { name: 'Melee', count: balance.meleeAssassin, icon: Swords, color: 'text-red-400' },
    { name: 'Ranged', count: balance.rangedAssassin, icon: Target, color: 'text-orange-400' },
    { name: 'Healer', count: balance.healer, icon: Heart, color: 'text-green-400' },
    { name: 'Support', count: balance.support, icon: Sparkles, color: 'text-cyan-400' },
  ]

  if (compact) {
    return (
      <div className="flex items-center gap-2 flex-wrap">
        {roles.map(role => {
          const Icon = role.icon
          const isNeeded = needs.some(n => n.role === role.name || (n.role === 'Damage' && (role.name === 'Melee' || role.name === 'Ranged')))
          const priority = needs.find(n => n.role === role.name || (n.role === 'Damage' && (role.name === 'Melee' || role.name === 'Ranged')))?.priority

          return (
            <div
              key={role.name}
              className={`flex items-center gap-1 px-2 py-1 rounded-md border ${
                isNeeded && priority === 'critical'
                  ? 'border-red-500/30 bg-red-500/10'
                  : isNeeded && priority === 'important'
                  ? 'border-yellow-500/30 bg-yellow-500/10'
                  : role.count > 0
                  ? 'border-primary-500/30 bg-primary-500/10'
                  : 'border-border bg-background/50 opacity-50'
              }`}
            >
              <Icon className={`h-3 w-3 ${role.color}`} />
              <span className="text-xs font-medium">{role.count}</span>
            </div>
          )
        })}
      </div>
    )
  }

  return (
    <Card className="border-border">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">Role Balance - {teamName}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Role Counts */}
        <div className="grid grid-cols-3 gap-2">
          {roles.map(role => {
            const Icon = role.icon
            const isNeeded = needs.some(n => n.role === role.name || (n.role === 'Damage' && (role.name === 'Melee' || role.name === 'Ranged')))
            const priority = needs.find(n => n.role === role.name || (n.role === 'Damage' && (role.name === 'Melee' || role.name === 'Ranged')))?.priority

            return (
              <motion.div
                key={role.name}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: roles.indexOf(role) * 0.05 }}
                className={`flex items-center gap-2 p-2 rounded-lg border ${
                  isNeeded && priority === 'critical'
                    ? 'border-red-500/30 bg-red-500/10'
                    : isNeeded && priority === 'important'
                    ? 'border-yellow-500/30 bg-yellow-500/10'
                    : role.count > 0
                    ? 'border-primary-500/30 bg-primary-500/10'
                    : 'border-dashed border-border bg-background/50'
                }`}
              >
                <Icon className={`h-4 w-4 ${role.color}`} />
                <div className="flex-1">
                  <div className="text-xs font-medium">{role.name}</div>
                  <div className="text-lg font-bold">{role.count}</div>
                </div>
              </motion.div>
            )
          })}
        </div>

        {/* Role Needs */}
        {needs.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs font-semibold text-muted-foreground">Priority Needs:</div>
            <div className="space-y-1">
              {needs.map((need, idx) => (
                <Badge
                  key={idx}
                  variant="outline"
                  className={`w-full justify-start ${
                    need.priority === 'critical'
                      ? 'border-red-500/30 bg-red-500/10 text-red-400'
                      : need.priority === 'important'
                      ? 'border-yellow-500/30 bg-yellow-500/10 text-yellow-400'
                      : 'border-blue-500/30 bg-blue-500/10 text-blue-400'
                  }`}
                >
                  {need.role} ({need.priority})
                </Badge>
              ))}
            </div>
          </div>
        )}

        {needs.length === 0 && (
          <div className="text-xs text-center text-muted-foreground py-2">
            All core roles covered
          </div>
        )}
      </CardContent>
    </Card>
  )
}
