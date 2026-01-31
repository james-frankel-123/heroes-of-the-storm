'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import {
  Trophy,
  Target,
  TrendingUp,
  Users,
  ChevronRight,
} from 'lucide-react'
import { StatCard } from '@/components/ui/stat-card'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { HeroTable } from '@/components/dashboard/hero-table'
import { MapPerformance } from '@/components/dashboard/map-performance'
import { RoleDistribution } from '@/components/dashboard/role-distribution'
import { PowerPicks } from '@/components/dashboard/power-picks'
import { InsightsPanel } from '@/components/dashboard/insights-panel'
import { formatNumber, getWinRateColor } from '@/lib/utils'
import { usePlayerData } from '@/lib/hooks/use-data'
import { generatePowerPicks, generateInsights } from '@/lib/data/transform'

export default function DashboardPage() {
  const { data, isLoading, error } = usePlayerData()

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 },
  }

  if (isLoading) {
    return (
      <div className="space-y-8">
        <div>
          <Skeleton className="h-12 w-96" />
          <Skeleton className="mt-2 h-6 w-64" />
        </div>
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="flex h-full items-center justify-center">
        <Card className="glass border-gaming-danger/30 p-8">
          <p className="text-gaming-danger">Failed to load data. Please try again.</p>
        </Card>
      </div>
    )
  }

  const topHeroes = data.heroStats.slice(0, 5)
  const powerPicks = generatePowerPicks(data)
  const insights = generateInsights(data)

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-4xl font-bold tracking-tight glow">
          Welcome back, {data.playerName.split('#')[0]}
        </h1>
        <p className="mt-2 text-muted-foreground">
          Here&apos;s your performance overview for Storm League
        </p>
      </motion.div>

      {/* Stats Grid */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="show"
        className="grid gap-6 md:grid-cols-2 lg:grid-cols-4"
      >
        <StatCard
          label="Total Games"
          value={formatNumber(data.totalGames)}
          icon={<Trophy className="h-6 w-6" />}
          animated
        />
        <StatCard
          label="Win Rate"
          value={`${data.overallWinRate.toFixed(1)}%`}
          valueColor={getWinRateColor(data.overallWinRate)}
          icon={<Target className="h-6 w-6" />}
          animated
        />
        <StatCard
          label="Total Wins"
          value={formatNumber(data.totalWins)}
          valueColor="text-gaming-success"
          icon={<TrendingUp className="h-6 w-6" />}
          animated
        />
        <StatCard
          label="Heroes Mastered"
          value={data.heroStats.length}
          icon={<Users className="h-6 w-6" />}
          animated
        />
      </motion.div>

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Hero Performance - Takes 2 columns */}
        <motion.div
          variants={itemVariants}
          initial="hidden"
          animate="show"
          className="lg:col-span-2"
        >
          <Card className="glass border-primary-500/30">
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Trophy className="h-5 w-5 text-primary-500" />
                Top Heroes
              </CardTitle>
              <Button variant="ghost" size="sm">
                View All <ChevronRight className="ml-1 h-4 w-4" />
              </Button>
            </CardHeader>
            <CardContent>
              <HeroTable heroes={topHeroes} />
            </CardContent>
          </Card>
        </motion.div>

        {/* Insights Panel */}
        <motion.div
          variants={itemVariants}
          initial="hidden"
          animate="show"
          className="lg:col-span-1"
        >
          <InsightsPanel insights={insights} />
        </motion.div>
      </div>

      {/* Map Performance & Role Distribution */}
      <div className="grid gap-6 lg:grid-cols-2">
        <motion.div variants={itemVariants}>
          <MapPerformance mapData={data.mapStats.slice(0, 5)} playerData={data} />
        </motion.div>
        <motion.div variants={itemVariants}>
          <RoleDistribution roleData={data.roleStats} />
        </motion.div>
      </div>

      {/* Power Picks */}
      {powerPicks.length > 0 && (
        <motion.div variants={itemVariants}>
          <PowerPicks powerPicks={powerPicks.slice(0, 6)} playerData={data} />
        </motion.div>
      )}
    </div>
  )
}
