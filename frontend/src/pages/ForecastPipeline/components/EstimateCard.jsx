import { Clock, Wallet, Cloud, Cpu, Sparkles, History, Gauge } from 'lucide-react'
import Loader from '../../../components/ui/Loader'

/**
 * What this run will take, shown before the user commits compute.
 *
 * Runtime and total cost get the visual weight — they are the whole
 * reason this step exists. Databricks and LLM cost are broken out
 * separately underneath (Section 8.5): they scale on entirely different
 * axes — one per cluster-minute, the other per key/token — so folding them
 * into one blended number would hide which one actually drives the total.
 * Every figure here is computed from the real uploaded dataset and the
 * real selected configuration; nothing is a fixed placeholder range.
 */
export default function EstimateCard({ estimate, loading, error }) {
  if (loading) {
    return (
      <div className="flex items-center justify-center rounded-xl border border-slate-200 bg-white px-4 py-8 dark:border-slate-800 dark:bg-slate-900">
        <Loader label="Estimating…" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-xs text-slate-400 dark:border-slate-800 dark:bg-slate-900">
        {error}
      </div>
    )
  }

  if (!estimate) return null

  const isCloud = estimate.execution_backend === 'databricks'
  const { dataset, workload, cost } = estimate
  const totalCostLabel = cost.total_cost_available
    ? `${cost.currency} ${cost.total_cost_low.toFixed(2)} – ${cost.total_cost_high.toFixed(2)}`
    : 'Not configured'

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
      <div className="grid grid-cols-1 divide-y divide-slate-100 sm:grid-cols-2 sm:divide-x sm:divide-y-0 dark:divide-slate-800">
        <Figure
          icon={Clock}
          label="Estimated runtime"
          value={estimate.estimated_duration_label}
          tone="text-brand-600 dark:text-brand-400"
        />
        <Figure
          icon={Wallet}
          label="Estimated total cost"
          value={totalCostLabel}
          tone={cost.total_cost_available ? 'text-emerald-600 dark:text-emerald-400' : 'text-slate-300 dark:text-slate-600'}
        />
      </div>

      {/* Cost split — shown only once either side has a configured rate,
          so an all-local, all-unpriced deployment doesn't render an empty
          row of dashes. */}
      {(cost.databricks_cost_available || cost.llm_cost_available) && (
        <div className="grid grid-cols-2 divide-x divide-slate-100 border-t border-slate-100 dark:divide-slate-800 dark:border-slate-800">
          <CostLine
            label="Databricks compute"
            available={cost.databricks_cost_available}
            low={cost.databricks_cost_low}
            high={cost.databricks_cost_high}
            currency={cost.currency}
          />
          <CostLine
            label="LLM explanations"
            available={cost.llm_cost_available}
            low={cost.llm_cost_low}
            high={cost.llm_cost_high}
            currency={cost.currency}
          />
        </div>
      )}

      <div className="flex flex-wrap items-center gap-1.5 border-t border-slate-100 px-4 py-2.5 dark:border-slate-800">
        <span className="flex items-center gap-1 rounded-md bg-slate-50 px-2 py-1 text-[11px] font-medium text-slate-500 dark:bg-slate-800 dark:text-slate-400">
          {isCloud ? <Cloud size={11} /> : <Cpu size={11} />}
          {isCloud ? 'Azure Databricks' : 'Local execution'}
        </span>
        {estimate.breakdown.map((item) => (
          <span
            key={item.label}
            className="rounded-md bg-slate-50 px-2 py-1 text-[11px] text-slate-500 dark:bg-slate-800 dark:text-slate-400"
          >
            <span className="text-slate-400 dark:text-slate-500">{item.label}: </span>
            {item.detail}
          </span>
        ))}
      </div>

      {/* Dataset facts the estimate was actually computed from — the
          "why" behind the numbers above, per-key history included since
          that is what silently determines whether TFT/tuning even run. */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 border-t border-slate-100 px-4 py-3 text-[11px] text-slate-500 dark:border-slate-800 dark:text-slate-400 sm:grid-cols-3">
        <Fact label="Date grain" value={dataset.date_grain} />
        <Fact label="Per-key history" value={`${dataset.history_length_periods} mo (longest key)`} />
        <Fact label="Unique keys" value={dataset.unique_keys.toLocaleString()} />
        <Fact
          label="Missing target values"
          value={dataset.missingness_pct == null ? 'unknown' : `${dataset.missingness_pct}%`}
        />
        <Fact label="Tuning-eligible pairs" value={`${workload.tuning_eligible_pairs} of ${workload.model_evaluations}`} />
        <Fact label="SHAP computations" value={`up to ${workload.shap_computations}`} />
      </div>

      <div className="flex items-center gap-1.5 border-t border-slate-100 px-4 py-2 text-[11px] text-slate-400 dark:border-slate-800 dark:text-slate-500">
        <History size={11} className="shrink-0" />
        Calibrated using {estimate.calibration_basis}.
      </div>
    </div>
  )
}

function Figure({ icon: Icon, label, value, tone }) {
  return (
    <div className="px-4 py-4">
      <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-slate-400">
        <Icon size={12} />
        {label}
      </p>
      <p className={`mt-1 text-xl font-bold tracking-tight ${tone}`}>{value}</p>
    </div>
  )
}

function CostLine({ label, available, low, high, currency }) {
  const Icon = label.startsWith('Databricks') ? Gauge : Sparkles
  return (
    <div className="px-4 py-2.5">
      <p className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wide text-slate-400">
        <Icon size={11} />
        {label}
      </p>
      <p
        className={`mt-0.5 text-sm font-semibold ${
          available ? 'text-slate-700 dark:text-slate-200' : 'text-slate-300 dark:text-slate-600'
        }`}
      >
        {available ? `${currency} ${low.toFixed(2)} – ${high.toFixed(2)}` : 'Not configured'}
      </p>
    </div>
  )
}

function Fact({ label, value }) {
  return (
    <div className="flex items-baseline justify-between gap-2 sm:block">
      <span className="text-slate-400 dark:text-slate-500">{label}</span>
      <span className="font-medium text-slate-600 dark:text-slate-300 sm:ml-1">{value}</span>
    </div>
  )
}
