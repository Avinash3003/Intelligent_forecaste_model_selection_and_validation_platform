import { useState } from 'react'
import { Sparkles, Info, ChevronDown } from 'lucide-react'

/**
 * The narrative, compressed to a scannable card.
 *
 * The backend caps summary/key_reason/caveat by word count before they reach
 * here, so this component renders short fields rather than defending against
 * long ones. The full narrative stays reachable behind "Full explanation" —
 * hidden by default, because the dashboard's job is the decision, not the
 * audit trail.
 */
export default function AiInsightCard({ narrative, confidence }) {
  const [open, setOpen] = useState(false)
  const insight = narrative?.insight || {}
  const hasInsight = narrative?.available && insight.summary

  if (!hasInsight) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
        <div className="flex items-center gap-2">
          <Sparkles size={15} className="text-slate-400" />
          <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200">AI Insight</h3>
        </div>
        <div className="mt-3 flex items-start gap-2 text-xs text-slate-500 dark:text-slate-400">
          <Info size={14} className="mt-0.5 shrink-0" />
          <span>{narrative?.status || 'No narrative was generated for this run.'}</span>
        </div>
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-brand-200 bg-brand-50/40 p-4 dark:border-brand-900/60 dark:bg-brand-900/10">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Sparkles size={15} className="text-brand-600 dark:text-brand-400" />
          <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">AI Insight</h3>
        </div>
        {insight.key_reason && (
          <span className="rounded-full bg-brand-100 px-2 py-0.5 text-[11px] font-medium text-brand-700 dark:bg-brand-900/40 dark:text-brand-300">
            {insight.key_reason}
          </span>
        )}
      </div>

      <p className="mt-2.5 text-sm leading-relaxed text-slate-700 dark:text-slate-200">
        {insight.summary}
      </p>

      {insight.caveat && (
        <p className="mt-2 border-l-2 border-amber-300 pl-2.5 text-xs leading-relaxed text-amber-700 dark:border-amber-700 dark:text-amber-400">
          {insight.caveat}
        </p>
      )}

      <div className="mt-3 flex items-center justify-between border-t border-brand-100 pt-2.5 dark:border-brand-900/40">
        <span className="text-xs text-slate-500 dark:text-slate-400">
          Confidence{' '}
          <span className="font-semibold tabular-nums text-slate-800 dark:text-slate-100">
            {confidence == null ? 'N/A' : `${Math.round(confidence * 100)}%`}
          </span>
        </span>
        {narrative.paragraphs?.length > 0 && (
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="flex items-center gap-1 text-xs font-medium text-brand-600 hover:underline dark:text-brand-400"
          >
            {open ? 'Hide' : 'Full explanation'}
            <ChevronDown size={13} className={open ? 'rotate-180 transition' : 'transition'} />
          </button>
        )}
      </div>

      {/* Bounded and scrollable: the full narrative is several hundred words,
          and letting it expand freely stretched this card into a column far
          taller than the chart beside it. It now scrolls in place, so opening
          it never changes the page layout. */}
      {open && (
        <div className="mt-3 max-h-64 space-y-3 overflow-y-auto border-t border-brand-100 pt-3 pr-1 dark:border-brand-900/40">
          {narrative.paragraphs.map((paragraph, idx) => (
            <div key={idx}>
              {/* The engine returns this key's decision first, then run-level
                  context covering every group. Labelling them stops the second
                  block from reading as more detail about this one key. */}
              <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                {idx === 0 ? 'This key' : 'Run overview'}
              </p>
              <p className="whitespace-pre-line text-xs leading-relaxed text-slate-600 dark:text-slate-300">
                {paragraph}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
