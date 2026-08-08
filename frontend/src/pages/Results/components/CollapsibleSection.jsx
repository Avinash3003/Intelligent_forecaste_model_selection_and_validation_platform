import { useState } from 'react'
import { ChevronDown } from 'lucide-react'

/**
 * A section that stays shut until asked for.
 *
 * Used for everything that is evidence rather than decision — drift metrics,
 * SHAP drivers, MLflow, debug. Collapsed by default so the page opens as a
 * dashboard rather than a report; the content is unchanged, only deferred.
 */
export default function CollapsibleSection({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-4 py-3 text-left transition hover:bg-slate-50 dark:hover:bg-slate-800/50"
      >
        <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">{title}</span>
        <ChevronDown
          size={15}
          className={`text-slate-400 transition ${open ? 'rotate-180' : ''}`}
        />
      </button>
      {open && <div className="border-t border-slate-100 dark:border-slate-800">{children}</div>}
    </div>
  )
}
