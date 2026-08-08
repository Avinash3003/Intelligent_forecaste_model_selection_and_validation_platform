import { Sparkles, Info } from 'lucide-react'
import SectionContainer from '../../../components/layout/SectionContainer'

export default function ExplainabilityCard({ narrative, shapDrivers = [] }) {
  return (
    <SectionContainer
      title="Explainability"
      subtitle="LLM-generated narrative, grounded entirely in this run's quantitative metrics"
    >
      {/* An unconfigured or unreachable LLM is reported honestly rather
          than papered over with placeholder prose. */}
      {narrative.available ? (
        <>
          <div className="flex items-start gap-3 rounded-xl border border-brand-100 bg-brand-50/50 p-4 dark:border-brand-900/50 dark:bg-brand-900/10">
            <Sparkles size={18} className="mt-0.5 shrink-0 text-brand-600 dark:text-brand-400" />
            <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">
              {narrative.keyModelHeadline}
            </p>
          </div>

          <div className="mt-4 space-y-3">
            {narrative.paragraphs.map((paragraph, idx) => (
              <p
                key={idx}
                className="whitespace-pre-line text-sm leading-relaxed text-slate-600 dark:text-slate-300"
              >
                {paragraph}
              </p>
            ))}
          </div>
        </>
      ) : (
        <div className="flex items-start gap-3 rounded-xl border border-slate-200 bg-slate-50/60 p-4 dark:border-slate-700 dark:bg-slate-800/40">
          <Info size={18} className="mt-0.5 shrink-0 text-slate-400" />
          <div>
            <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
              No narrative was generated for this run.
            </p>
            {narrative.status && <p className="mt-0.5 text-xs text-slate-400">{narrative.status}</p>}
          </div>
        </div>
      )}

      {shapDrivers.length > 0 && (
        <div className="mt-6">
          <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
            Top SHAP feature drivers
          </p>
          <div className="space-y-2">
            {shapDrivers.map((driver) => (
              <div key={driver.feature} className="flex items-center gap-3">
                <span className="w-40 shrink-0 truncate text-sm text-slate-600 dark:text-slate-300">
                  {driver.feature}
                </span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                  <div
                    className="h-full rounded-full bg-brand-600"
                    style={{ width: `${Math.min(100, Math.abs(driver.importance) * 100)}%` }}
                  />
                </div>
                <span className="w-16 shrink-0 text-right text-xs tabular-nums text-slate-400">
                  {driver.importance}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </SectionContainer>
  )
}
