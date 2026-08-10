import { useEffect, useState } from 'react'
import { Database, ChevronLeft, ChevronRight } from 'lucide-react'
import Loader from '../../../components/ui/Loader'
import { fetchDatasetPreview } from '../../../services'

const PAGE_SIZE = 50

/**
 * The curated dataset — cleaned, deduplicated, aggregated to the platform's
 * forecasting grain — shown beside the decision made from it. Deliberately
 * not the raw upload: curated is what every model actually trained on, and
 * it is far smaller, since a daily upload across many keys collapses to one
 * row per key per month.
 *
 * Paged in fixed 50-row windows (page 1 = rows 1-50, page 2 = 51-100, …)
 * rather than an accumulating "load more" — each click replaces the table
 * instead of growing it, so the page stays exactly one page's worth of DOM
 * regardless of how far a user navigates.
 */
export default function DatasetPreviewPanel({ runId }) {
  const [page, setPage] = useState(1)
  const [state, setState] = useState({ loading: true, data: null, error: null })

  useEffect(() => {
    let cancelled = false
    setState((s) => ({ loading: true, data: s.data, error: null }))
    fetchDatasetPreview(runId, page, PAGE_SIZE)
      .then((data) => !cancelled && setState({ loading: false, data, error: null }))
      .catch((e) => !cancelled && setState({ loading: false, data: null, error: String(e.message || e) }))
    return () => {
      cancelled = true
    }
  }, [runId, page])

  // Reset to page 1 whenever the run changes, so switching runs never shows
  // "page 4" of a dataset that has one page.
  useEffect(() => {
    setPage(1)
  }, [runId])

  if (state.error || (!state.loading && !state.data?.available)) {
    return (
      <p className="px-4 py-3 text-xs text-slate-500 dark:text-slate-400">
        {state.error || state.data?.status || 'No curated dataset available for this run.'}
      </p>
    )
  }

  const data = state.data
  const columns = data?.columns || []
  const rows = data?.rows || []
  const totalRows = data?.total_rows ?? 0
  const totalPages = data?.total_pages ?? 1
  const rangeStart = totalRows === 0 ? 0 : (page - 1) * PAGE_SIZE + 1
  const rangeEnd = Math.min(page * PAGE_SIZE, totalRows)

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-4 py-2.5 text-xs text-slate-500 dark:text-slate-400">
        <span className="flex items-center gap-1.5 font-medium text-slate-700 dark:text-slate-200">
          <Database size={13} />
          Curated dataset
        </span>
        <span>
          {totalRows === 0 ? 'empty' : `rows ${rangeStart}–${rangeEnd} of ${totalRows.toLocaleString()}`}
        </span>
      </div>

      {/* Its own horizontal scroll: a wide dataset must never make the page
          scroll sideways. Fixed height, not max-height — a page is always
          exactly PAGE_SIZE rows (except the last), so the panel does not
          resize as the user pages through. */}
      <div className="relative h-80 overflow-auto border-t border-slate-100 dark:border-slate-800">
        {state.loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-white/60 dark:bg-slate-900/60">
            <Loader label="" />
          </div>
        )}
        <table className="w-full text-left">
          <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800">
            <tr className="text-[10px] uppercase tracking-wide text-slate-400">
              <th className="px-3 py-2 font-medium">#</th>
              {columns.map((c) => (
                <th key={c} className="whitespace-nowrap px-3 py-2 font-medium">{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-b border-slate-50 dark:border-slate-800/60">
                <td className="px-3 py-1.5 text-[11px] tabular-nums text-slate-300">
                  {(page - 1) * PAGE_SIZE + i + 1}
                </td>
                {row.map((cell, j) => (
                  <td key={j} className="whitespace-nowrap px-3 py-1.5 text-xs tabular-nums text-slate-600 dark:text-slate-300">
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between border-t border-slate-100 px-4 py-2 dark:border-slate-800">
        <button
          type="button"
          onClick={() => setPage((p) => Math.max(1, p - 1))}
          disabled={page <= 1 || state.loading}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-40 disabled:hover:bg-transparent dark:text-slate-300 dark:hover:bg-slate-800"
        >
          <ChevronLeft size={13} /> Previous
        </button>
        <span className="text-xs tabular-nums text-slate-400">
          Page {page} of {totalPages}
        </span>
        <button
          type="button"
          onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
          disabled={page >= totalPages || state.loading}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-40 disabled:hover:bg-transparent dark:text-slate-300 dark:hover:bg-slate-800"
        >
          Next <ChevronRight size={13} />
        </button>
      </div>
    </div>
  )
}
