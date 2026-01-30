'use client'

import * as React from 'react'
import * as d3 from 'd3'
import { useTheme } from 'next-themes'
import { cn } from '@/lib/utils'

interface HeatmapData {
  hero: string
  map: string
  winRate: number
  games: number
}

interface HeatmapProps {
  data: HeatmapData[]
  className?: string
}

export function Heatmap({ data, className }: HeatmapProps) {
  const svgRef = React.useRef<SVGSVGElement>(null)
  const { theme } = useTheme()
  const [tooltipData, setTooltipData] = React.useState<HeatmapData | null>(null)
  const [tooltipPos, setTooltipPos] = React.useState({ x: 0, y: 0 })

  React.useEffect(() => {
    if (!svgRef.current || !data.length) return

    const margin = { top: 80, right: 40, bottom: 120, left: 120 }
    const width = 800 - margin.left - margin.right
    const height = 600 - margin.top - margin.bottom

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3
      .select(svgRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Get unique heroes and maps
    const heroes = Array.from(new Set(data.map((d) => d.hero)))
    const maps = Array.from(new Set(data.map((d) => d.map)))

    // Build X scales and axis
    const x = d3.scaleBand().range([0, width]).domain(heroes).padding(0.05)

    svg
      .append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x).tickSize(0))
      .selectAll('text')
      .style('text-anchor', 'end')
      .style('fill', theme === 'dark' ? '#a1a1aa' : '#52525b')
      .style('font-size', '11px')
      .attr('dx', '-.8em')
      .attr('dy', '.15em')
      .attr('transform', 'rotate(-45)')

    // Build Y scales and axis
    const y = d3.scaleBand().range([height, 0]).domain(maps).padding(0.05)

    svg
      .append('g')
      .call(d3.axisLeft(y).tickSize(0))
      .selectAll('text')
      .style('fill', theme === 'dark' ? '#a1a1aa' : '#52525b')
      .style('font-size', '11px')

    // Build color scale
    const colorScale = d3
      .scaleSequential()
      .interpolator(d3.interpolateRgb('#ff6b6b', '#4fffb0'))
      .domain([30, 70])

    // Create cells
    svg
      .selectAll()
      .data(data)
      .enter()
      .append('rect')
      .attr('x', (d) => x(d.hero) || 0)
      .attr('y', (d) => y(d.map) || 0)
      .attr('width', x.bandwidth())
      .attr('height', y.bandwidth())
      .style('fill', (d) => colorScale(d.winRate))
      .style('opacity', 0.8)
      .style('cursor', 'pointer')
      .on('mouseenter', function (event, d) {
        d3.select(this).style('opacity', 1).style('stroke', '#4a9eff').style('stroke-width', 2)
        setTooltipData(d)
        setTooltipPos({ x: event.pageX, y: event.pageY })
      })
      .on('mouseleave', function () {
        d3.select(this).style('opacity', 0.8).style('stroke', 'none')
        setTooltipData(null)
      })

    // Add text labels for win rates
    svg
      .selectAll()
      .data(data)
      .enter()
      .append('text')
      .text((d) => `${d.winRate.toFixed(0)}%`)
      .attr('x', (d) => (x(d.hero) || 0) + x.bandwidth() / 2)
      .attr('y', (d) => (y(d.map) || 0) + y.bandwidth() / 2)
      .style('fill', (d) => (d.winRate > 50 ? '#000' : '#fff'))
      .style('font-size', '10px')
      .style('font-weight', 'bold')
      .style('text-anchor', 'middle')
      .style('dominant-baseline', 'middle')
      .style('pointer-events', 'none')
  }, [data, theme])

  return (
    <div className={cn('relative', className)}>
      <svg ref={svgRef} className="w-full" />
      {tooltipData && (
        <div
          className="pointer-events-none fixed z-50 rounded-lg border border-border bg-card p-3 shadow-xl"
          style={{
            left: tooltipPos.x + 10,
            top: tooltipPos.y + 10,
          }}
        >
          <p className="text-sm font-semibold">{tooltipData.hero}</p>
          <p className="text-xs text-muted-foreground">{tooltipData.map}</p>
          <div className="mt-2 flex items-center gap-3">
            <span className="text-lg font-bold text-primary-500">
              {tooltipData.winRate.toFixed(1)}%
            </span>
            <span className="text-xs text-muted-foreground">
              {tooltipData.games} games
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
