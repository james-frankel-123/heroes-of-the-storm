'use client'

import { cn } from '@/lib/utils'

interface SortHeaderProps<F extends string> {
  field: F
  children: React.ReactNode
  className?: string
  sortField: F
  sortAsc: boolean
  onSort: (field: F) => void
}

/**
 * Sortable table header: a real button (keyboard operable), aria-sort on the active column.
 * Module scope on purpose: a header component declared inside a table's render would be a new
 * type every render, remounting the button and dropping keyboard focus after each sort.
 */
export function SortHeader<F extends string>({ field, children, className, sortField, sortAsc, onSort }: SortHeaderProps<F>) {
  return (
    <th
      className={cn('px-3 py-2 text-xs font-medium text-muted-foreground select-none whitespace-nowrap', className)}
      aria-sort={sortField === field ? (sortAsc ? 'ascending' : 'descending') : 'none'}
    >
      <button type="button" onClick={() => onSort(field)} className="inline-flex items-center gap-1 hover:text-foreground">
        {children}
        {sortField === field && (
          <span className="text-primary" aria-hidden="true">
            {sortAsc ? '▲' : '▼'}
          </span>
        )}
      </button>
    </th>
  )
}
