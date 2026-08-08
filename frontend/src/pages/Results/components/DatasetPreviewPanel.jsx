import { useEffect, useState } from 'react'
import { FileSpreadsheet, Cloud, HardDrive } from 'lucide-react'
import Loader from '../../../components/ui/Loader'
import { fetchDatasetPreview } from '../../../services'

/**
 * The uploaded file, shown beside the decision made from it.
 *
 * Loaded lazily — only when the section is expanded — so the Results page
 * never pays for a storage read a user did not ask for. The backend caps the
 * download by byte range, so this stays cheap even for a 17MB source file.
 */
export default function DatasetPreviewPanel({ runId }) {
  const [state, setState] = useState({ loading: true, data: null, error: null })

  useEffect(() => {
    let cancelled = false
    setState({ loading: true, data: null, error: null })
    fetchDatasetPreview(runId)
      .then((data) => !cancelled && setState({ loading: false, data, error: null }))
      .catch((e) => !cancelled && setState({ loading: false, data: null, error: String(e.message || e) }))
    return () => {
      cancelled = true
    }
  }, [runId])

  if (state.loading) return <div className="p-4"><Loader label="Reading dataset…" /></div>

  if (state.error || !state.data?.available) {
    return (
      <p className="px-4 py-3 text-xs text-slate-500 dark:text-slate-400">
        {state.error || state.data?.status || 'No preview available for this run.'}
      </p>
    )
  }

  const { dataset_name, source, columns, rows, preview_row_count, truncated } = state.data
  const fromCloud = (source || '').toLowerCase().includes('azure')
  const SourceIcon = fromCloud ? Cloud : HardDrive

  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-4 py-2.5 text-xs text-slate-500 dark:text-slate-400">
        <span className="flex items-center gap-1.5 font-medium text-slate-700 dark:text-slate-200">
          <FileSpreadsheet size={13} />
          {dataset_name}
        </span>
        <span className="flex items-center gap-1">
          <SourceIcon size={12} />
          {source}
        </span>
        <span>
          first {preview_row_count} rows{truncated ? ' · file continues' : ''}
        </span>
      </div>

      {/* Its own horizontal scroll: a wide dataset must never make the page
          scroll sideways. */}
      <div className="max-h-80 overflow-auto border-t border-slate-100 dark:border-slate-800">
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
                <td className="px-3 py-1.5 text-[11px] tabular-nums text-slate-300">{i + 1}</td>
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
    </div>
  )
}
