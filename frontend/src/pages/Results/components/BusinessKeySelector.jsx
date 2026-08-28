import { useEffect, useMemo, useRef, useState } from 'react'
import { Check, ChevronDown, Search, X } from 'lucide-react'
import { cn } from '../../../utils/cn'
import DropdownPortal from '../../../components/ui/DropdownPortal'

// One filter-bar field, whatever the key looks like: the key columns are
// split *inside* the dropdown, so the bar keeps its four-column layout for
// a single-series run and for a six-column key alike.
//
// Three things keep the panel usable as the column count grows:
//   * search, which scales to any number of columns where per-column
//     dropdowns do not,
//   * a per-column filter row that wraps and is itself capped and
//     scrollable, so it can never crowd out the list beneath it,
//   * a panel wider than the trigger, since the filter controls truncate
//     badly at filter-bar field width.

const PANEL_MIN_WIDTH = 320
// Roughly two rows of filters; beyond that the row scrolls rather than
// pushing the key list off the bottom of the panel.
const FILTER_ROW_MAX_HEIGHT = 76

function keyColumns(options) {
  for (const option of options) {
    const names = Object.keys(option.key_values || {})
    if (names.length) return names
  }
  return []
}

function uniqueValues(options, column) {
  const seen = new Set()
  const out = []
  for (const option of options) {
    const value = (option.key_values || {})[column]
    if (value !== undefined && !seen.has(value)) {
      seen.add(value)
      out.push(value)
    }
  }
  return out
}

export default function BusinessKeySelector({ value, options = [], onChange }) {
  const [open, setOpen] = useState(false)
  const [filters, setFilters] = useState({})
  const [query, setQuery] = useState('')
  const anchorRef = useRef(null)

  const columns = useMemo(() => keyColumns(options), [options])
  const selected = options.find((o) => (o.group_id ?? o.value) === value)

  useEffect(() => {
    function onPointerDown(event) {
      const inAnchor = anchorRef.current?.contains(event.target)
      const inPanel = event.target.closest?.('[data-dropdown-portal]')
      if (!inAnchor && !inPanel) setOpen(false)
    }
    document.addEventListener('mousedown', onPointerDown)
    return () => document.removeEventListener('mousedown', onPointerDown)
  }, [])

  // A new run brings its own keys; stale narrowing would hide all of them.
  useEffect(() => {
    setFilters({})
    setQuery('')
  }, [options])

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return options.filter((option) => {
      const matchesFilters = Object.entries(filters).every(
        ([column, val]) => !val || (option.key_values || {})[column] === val
      )
      if (!matchesFilters) return false
      return !needle || (option.label || '').toLowerCase().includes(needle)
    })
  }, [options, filters, query])

  const hasNarrowing = query.trim() !== '' || Object.values(filters).some(Boolean)

  return (
    <div className="min-w-0">
      <label className="mb-1.5 block truncate text-xs font-semibold uppercase tracking-wide text-slate-400">
        Business Key
      </label>
      <div className="relative" ref={anchorRef}>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className={cn(
            'flex w-full items-center justify-between gap-2 rounded-lg border border-slate-200 bg-white/80 px-3 py-2 text-left text-sm outline-none transition-all focus:border-brand-400 focus:ring-2 focus:ring-brand-100',
            'dark:border-slate-700 dark:bg-slate-800/60 dark:focus:ring-brand-900/40',
            selected ? 'text-slate-700 dark:text-slate-200' : 'text-slate-400'
          )}
        >
          <span className="truncate">{selected?.label || 'Select key'}</span>
          <ChevronDown size={15} className="shrink-0 text-slate-400" />
        </button>

        <DropdownPortal anchorRef={anchorRef} open={open} minWidth={PANEL_MIN_WIDTH} className="z-50">
          <div className="rounded-lg border border-slate-200 bg-white shadow-lg dark:border-slate-700 dark:bg-slate-800">
            <div className="sticky top-0 z-10 space-y-2 border-b border-slate-100 bg-white p-2 dark:border-slate-700 dark:bg-slate-800">
              <div className="relative">
                <Search
                  size={13}
                  className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400"
                />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search keys"
                  className="w-full rounded border border-slate-200 bg-white py-1.5 pl-7 pr-7 text-xs text-slate-700 outline-none placeholder:text-slate-400 focus:border-brand-400 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-200"
                />
                {query && (
                  <button
                    type="button"
                    onClick={() => setQuery('')}
                    aria-label="Clear search"
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                  >
                    <X size={13} />
                  </button>
                )}
              </div>

              {columns.length > 0 && (
                <div
                  className="flex flex-wrap gap-1.5 overflow-y-auto"
                  style={{ maxHeight: FILTER_ROW_MAX_HEIGHT }}
                >
                  {columns.map((column) => (
                    <select
                      key={column}
                      value={filters[column] ?? ''}
                      onChange={(e) =>
                        setFilters((prev) => ({ ...prev, [column]: e.target.value }))
                      }
                      title={column}
                      className="min-w-0 flex-1 basis-[45%] rounded border border-slate-200 bg-white px-1.5 py-1 text-xs text-slate-600 outline-none focus:border-brand-400 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-300"
                    >
                      <option value="">All {column}</option>
                      {uniqueValues(options, column).map((val) => (
                        <option key={val} value={val}>
                          {column}={val}
                        </option>
                      ))}
                    </select>
                  ))}
                </div>
              )}

              <div className="flex items-center justify-between px-0.5 text-[11px] text-slate-400">
                <span>
                  {visible.length} of {options.length}
                </span>
                {hasNarrowing && (
                  <button
                    type="button"
                    onClick={() => {
                      setFilters({})
                      setQuery('')
                    }}
                    className="font-medium text-brand-600 hover:underline dark:text-brand-400"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>

            <ul className="py-1">
              {visible.length === 0 && (
                <li className="px-3 py-3 text-center text-xs text-slate-400">No matching key.</li>
              )}
              {visible.map((option) => {
                const id = option.group_id ?? option.value
                const isSelected = id === value
                return (
                  <li key={id}>
                    <button
                      type="button"
                      onClick={() => {
                        onChange?.(id)
                        setOpen(false)
                      }}
                      className={cn(
                        'flex w-full items-center justify-between gap-2 px-3 py-1.5 text-left text-sm transition-colors',
                        isSelected
                          ? 'bg-brand-50 font-medium text-brand-700 dark:bg-brand-900/30 dark:text-brand-300'
                          : 'text-slate-600 hover:bg-slate-50 dark:text-slate-300 dark:hover:bg-slate-700/50'
                      )}
                    >
                      <span className="truncate">{option.label}</span>
                      {isSelected && <Check size={14} className="shrink-0" />}
                    </button>
                  </li>
                )
              })}
            </ul>
          </div>
        </DropdownPortal>
      </div>
    </div>
  )
}
