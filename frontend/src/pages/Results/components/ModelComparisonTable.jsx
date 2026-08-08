import { Fragment, useState } from 'react'
import { ChevronRight } from 'lucide-react'

/**
 * Every evaluated model as one scannable row: status, why, and both ranks.
 *
 * The previous card rendered each model's full training/validation/ranking
 * detail inline, which is why the page read as documentation. Here a row is a
 * verdict; the evidence behind it is one click away, per model, so the default
 * view stays a comparison rather than a report.
 */

function StatusChip({ outcome }) {
  const map = {
    Selected: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
    Eliminated: 'bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300',
    'Failed to train': 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400',
  }
  // The backend's selection_outcome carries its reason inline
  // ("Eliminated — Flat Forecast, Excessive Smoothing, …"). Only the verdict
  // belongs in a chip; the reason has its own column, so keeping both here
  // would widen the table and say the same thing twice.
  const label = (outcome || '—').split('—')[0].trim()
  const cls = map[label] || 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300'
  return (
    <span className={`inline-block whitespace-nowrap rounded-full px-2 py-0.5 text-[11px] font-medium ${cls}`}>
      {label}
    </span>
  )
}

// One short phrase, not the full rule list — the reason a reader scans for.
function reasonFor(row) {
  if ((row.selectionOutcome || '').startsWith('Selected')) return 'Passed all validation gates'
  if (row.trainingStatus === 'Unavailable') return row.trainingError || 'Library not installed'
  if (row.forwardValidationReasons?.length) return row.forwardValidationReasons.join(', ')
  if (row.drift?.evaluated && row.drift.passed === false) return 'Failed drift validation'
  if (row.ranking?.finalRank) return 'Outranked by the selected model'
  return '—'
}

function DetailGrid({ row }) {
  const items = [
    ['WMAPE', row.backtest?.wmape != null ? `${row.backtest.wmape.toFixed(2)}%` : '—'],
    ['RMSE', row.backtest?.rmse != null ? row.backtest.rmse.toFixed(1) : '—'],
    ['MAE', row.backtest?.mae != null ? row.backtest.mae.toFixed(1) : '—'],
    ['Backtest windows', row.backtest?.windowCount ?? '—'],
    ['Composite score', row.ranking?.compositeScore != null ? `${Math.round(row.ranking.compositeScore * 100)}%` : '—'],
    ['Drift statistic', row.drift?.statistic != null ? row.drift.statistic.toFixed(4) : '—'],
    ['Drift threshold', row.drift?.thresholdValue != null ? row.drift.thresholdValue.toFixed(4) : '—'],
    ['Training', row.trainingStatus],
  ]
  const failed = (row.forwardValidationRules || []).filter((r) => !r.passed)
  const passedCount = (row.forwardValidationRules || []).length - failed.length

  return (
    <div className="bg-slate-50/70 px-4 py-3 dark:bg-slate-800/40">
      <div className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-4">
        {items.map(([label, value]) => (
          <div key={label}>
            <div className="text-[10px] uppercase tracking-wide text-slate-400">{label}</div>
            <div className="text-xs font-medium tabular-nums text-slate-700 dark:text-slate-200">{value}</div>
          </div>
        ))}
      </div>

      {row.forwardValidationRules?.length > 0 && (
        <div className="mt-3 border-t border-slate-200 pt-2.5 dark:border-slate-700">
          <div className="text-[10px] uppercase tracking-wide text-slate-400">Forward validation</div>
          {/* Passed rules collapse to a count; only failures are itemized,
              since those are the ones that changed the outcome. */}
          <div className="mt-1 text-xs text-emerald-600 dark:text-emerald-400">
            ✓ {passedCount} of {row.forwardValidationRules.length} rules passed
          </div>
          {failed.map((rule) => (
            <div key={rule.ruleName} className="mt-1 text-xs text-rose-600 dark:text-rose-400">
              ✗ {rule.ruleName} — {rule.detail}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function ModelComparisonTable({ decision, evaluatedModels = [] }) {
  const [openRow, setOpenRow] = useState(null)

  // Selected first, then the models that at least reached ranking, then the rest —
  // so the row order matches the order a reader cares about.
  const rows = [...evaluatedModels].sort((a, b) => {
    if (a.model === decision.selectedModel) return -1
    if (b.model === decision.selectedModel) return 1
    return (a.ranking?.finalRank || 99) - (b.ranking?.finalRank || 99)
  })

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3 dark:border-slate-800">
        <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">Model decision</h3>
        <span className="text-[11px] text-slate-400">{rows.length} evaluated</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] text-left">
          <thead>
            <tr className="border-b border-slate-100 text-[10px] uppercase tracking-wide text-slate-400 dark:border-slate-800">
              <th className="px-4 py-2 font-medium">Model</th>
              <th className="px-4 py-2 font-medium">Status</th>
              <th className="px-4 py-2 font-medium">Reason</th>
              <th className="px-4 py-2 text-right font-medium">Backtest rank</th>
              <th className="px-4 py-2 text-right font-medium">Final rank</th>
              <th className="w-8" />
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const isOpen = openRow === row.model
              const isWinner = row.model === decision.selectedModel
              return (
                <Fragment key={row.model}>
                  <tr
                    onClick={() => setOpenRow(isOpen ? null : row.model)}
                    className={`cursor-pointer border-b border-slate-50 text-sm transition hover:bg-slate-50 dark:border-slate-800/60 dark:hover:bg-slate-800/40 ${
                      isWinner ? 'bg-emerald-50/40 dark:bg-emerald-900/10' : ''
                    }`}
                  >
                    <td className="px-4 py-2.5 font-medium text-slate-800 dark:text-slate-100">{row.model}</td>
                    <td className="px-4 py-2.5">
                      <StatusChip outcome={row.selectionOutcome} />
                    </td>
                    <td className="max-w-[280px] truncate px-4 py-2.5 text-xs text-slate-500 dark:text-slate-400">
                      {reasonFor(row)}
                    </td>
                    <td className="px-4 py-2.5 text-right text-xs tabular-nums text-slate-500">
                      {row.ranking?.originalBacktestRank ?? '—'}
                    </td>
                    <td className="px-4 py-2.5 text-right text-xs tabular-nums text-slate-500">
                      {row.ranking?.finalRank ?? '—'}
                    </td>
                    <td className="pr-3 text-slate-300">
                      <ChevronRight size={14} className={isOpen ? 'rotate-90 transition' : 'transition'} />
                    </td>
                  </tr>
                  {isOpen && (
                    <tr>
                      <td colSpan={6} className="p-0">
                        <DetailGrid row={row} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
