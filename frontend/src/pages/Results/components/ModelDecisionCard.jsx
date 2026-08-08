import { useState } from 'react'
import { ChevronDown, HelpCircle, ShieldAlert, Trophy } from 'lucide-react'
import SectionContainer from '../../../components/layout/SectionContainer'
import Badge from '../../../components/ui/Badge'
import { cn } from '../../../utils/cn'

function outcomeTone(outcome) {
  if (outcome === 'Selected' || outcome === 'Fallback Used') return 'Selected'
  if (outcome.startsWith('Rejected') || outcome.startsWith('Failed') || outcome.startsWith('Eliminated')) {
    return 'Rejected'
  }
  return 'neutral'
}

function formatPercent(value) {
  return typeof value === 'number' ? `${Math.round(value * 100)}%` : 'N/A'
}

function formatNumber(value, digits = 4) {
  return typeof value === 'number' ? value.toPrecision(digits).replace(/\.?0+$/, '') : '—'
}

// Confidence is shown with its formula and justification directly beside
// it — never a bare percentage a user has to take on faith.
function ConfidenceStat({ confidence, breakdown }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="relative sm:text-right">
      <div className="flex items-center gap-1 sm:justify-end">
        <p className="text-xs text-slate-400">Confidence</p>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
          aria-label="Explain confidence"
        >
          <HelpCircle size={12} />
        </button>
      </div>
      <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">{formatPercent(confidence)}</p>

      {open && (
        <div className="absolute right-0 top-full z-20 mt-2 w-72 rounded-lg border border-slate-200 bg-white p-3.5 text-left shadow-glass dark:border-slate-700 dark:bg-slate-900">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Formula</p>
          <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">{breakdown.formula}</p>
          <p className="mt-3 text-xs font-semibold uppercase tracking-wide text-slate-400">Why this number</p>
          <p className="mt-1 text-xs leading-relaxed text-slate-600 dark:text-slate-300">{breakdown.explanation}</p>
          <div className="mt-3 grid grid-cols-2 gap-2 border-t border-slate-100 pt-2.5 dark:border-slate-800">
            <div>
              <p className="text-[11px] text-slate-400">Backtest accuracy</p>
              <p className="text-xs font-medium text-slate-700 dark:text-slate-200">
                {formatPercent(breakdown.backtestAccuracy)}
              </p>
            </div>
            <div>
              <p className="text-[11px] text-slate-400">Drift margin</p>
              <p className="text-xs font-medium text-slate-700 dark:text-slate-200">
                {formatPercent(breakdown.driftMargin)}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div>
      <p className="text-[11px] uppercase tracking-wide text-slate-400">{label}</p>
      <p className="mt-0.5 text-sm font-medium text-slate-700 dark:text-slate-200">{value}</p>
    </div>
  )
}

// Every rule that PASSED is compressed into one line — only failures are
// itemized, since those are the only ones a reader needs to inspect.
function ForwardValidationSummary({ rules }) {
  if (rules.length === 0) return <p className="text-sm text-slate-400">No rule detail recorded.</p>
  const failed = rules.filter((rule) => !rule.passed)
  const passedCount = rules.length - failed.length

  return (
    <div className="space-y-1.5">
      {passedCount > 0 && (
        <p className="text-sm text-emerald-600 dark:text-emerald-400">
          ✓ {passedCount} of {rules.length} rule{rules.length === 1 ? '' : 's'} passed
        </p>
      )}
      {failed.map((rule) => (
        <p key={rule.ruleName} className="text-sm text-rose-600 dark:text-rose-400">
          ✗ {rule.ruleName}
          {rule.detail && <span className="text-slate-400"> — {rule.detail}</span>}
        </p>
      ))}
    </div>
  )
}

// One model's full training -> selection trail, collapsed by default so
// the panel stays scannable — every number is one click from view, not
// dumped in the summary row.
function EvaluatedModelRow({ model, isWinner }) {
  const [open, setOpen] = useState(isWinner)
  const backtest = model.backtest
  const ranking = model.ranking
  const drift = model.drift

  return (
    <>
      <tr
        className={cn(
          'cursor-pointer border-b border-slate-50 last:border-0 hover:bg-slate-50/60 dark:border-slate-800/60 dark:hover:bg-slate-800/40',
          isWinner && 'bg-emerald-50/40 dark:bg-emerald-900/10'
        )}
        onClick={() => setOpen((o) => !o)}
      >
        <td className="w-8 px-3 py-3">
          <ChevronDown size={14} className={cn('text-slate-400 transition-transform', open && 'rotate-180')} />
        </td>
        <td className="px-3 py-3 text-sm font-medium text-slate-700 dark:text-slate-200">{model.model}</td>
        <td className="px-3 py-3 text-sm text-slate-500 dark:text-slate-400">
          {backtest?.wmape != null ? `${backtest.wmape.toFixed(2)}%` : '—'}
        </td>
        <td className="px-3 py-3 text-sm text-slate-500 dark:text-slate-400">{model.forwardValidationStatus}</td>
        <td className="px-3 py-3 text-sm text-slate-500 dark:text-slate-400">
          {drift?.evaluated ? (drift.passed ? 'Passed' : 'Failed') : drift ? 'Not applicable' : 'Not reached'}
        </td>
        <td className="px-3 py-3">
          <Badge status={outcomeTone(model.selectionOutcome)}>{model.selectionOutcome.split(' — ')[0]}</Badge>
        </td>
      </tr>

      {open && (
        <tr className="border-b border-slate-50 last:border-0 dark:border-slate-800/60">
          <td colSpan={6} className="bg-slate-50/40 px-6 py-4 dark:bg-slate-900/30">
            <div className="grid grid-cols-1 gap-x-8 gap-y-4 sm:grid-cols-3">
              <div className="space-y-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Training & backtest</p>
                <Stat label="Training status" value={model.trainingStatus} />
                {model.trainingError && (
                  <p className="text-xs text-rose-600 dark:text-rose-400">{model.trainingError}</p>
                )}
                {backtest ? (
                  <div className="grid grid-cols-2 gap-3">
                    <Stat label="RMSE" value={formatNumber(backtest.rmse)} />
                    <Stat label="MAE" value={formatNumber(backtest.mae)} />
                    <Stat label="Backtest windows" value={backtest.windowCount} />
                  </div>
                ) : (
                  <p className="text-xs text-slate-400">No backtest metrics recorded.</p>
                )}
              </div>

              <div className="space-y-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                  Forward validation & drift
                </p>
                <ForwardValidationSummary rules={model.forwardValidationRules} />
                {model.forwardValidationReasons.length > 0 && (
                  <p className="text-xs text-rose-600 dark:text-rose-400">
                    Eliminated for: {model.forwardValidationReasons.join(', ')}
                  </p>
                )}
                <div className="border-t border-slate-100 pt-3 dark:border-slate-800">
                  {drift?.evaluated ? (
                    <div className="grid grid-cols-2 gap-3">
                      <Stat label="Algorithm" value={drift.algorithm ?? '—'} />
                      <Stat label="Result" value={drift.passed ? 'Passed' : 'Failed'} />
                      <Stat label="Statistic" value={formatNumber(drift.statistic)} />
                      <Stat label="Threshold" value={formatNumber(drift.thresholdValue)} />
                    </div>
                  ) : (
                    <p className="text-xs text-slate-400">{drift?.detail || 'Never reached drift validation.'}</p>
                  )}
                </div>
              </div>

              <div className="space-y-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Ranking & outcome</p>
                {ranking ? (
                  <>
                    <div className="grid grid-cols-2 gap-3">
                      <Stat label="Composite score" value={formatPercent(ranking.compositeScore)} />
                      <Stat label="Final rank" value={ranking.finalRank ?? '—'} />
                      <Stat label="Backtest component" value={formatPercent(ranking.backtestScore)} />
                      <Stat label="Stability component" value={formatPercent(ranking.stabilityScore)} />
                    </div>
                    <p className="text-[11px] leading-relaxed text-slate-400">
                      Relative to this group's other candidates — see Confidence above for an absolute measure.
                    </p>
                  </>
                ) : (
                  <p className="text-xs text-slate-400">Not ranked — eliminated before reaching this stage.</p>
                )}
                <p className="border-t border-slate-100 pt-3 text-sm text-slate-600 dark:border-slate-800 dark:text-slate-300">
                  {model.selectionOutcome}
                </p>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

export default function ModelDecisionCard({ decision, evaluatedModels }) {
  // Winner first, everyone else in original backtest order.
  const rows = [...evaluatedModels].sort((a, b) => {
    const aWin = a.model === decision.selectedModel ? 0 : 1
    const bWin = b.model === decision.selectedModel ? 0 : 1
    if (aWin !== bWin) return aWin - bWin
    return (a.ranking?.originalBacktestRank ?? 99) - (b.ranking?.originalBacktestRank ?? 99)
  })

  return (
    <SectionContainer
      title="Model decision"
      subtitle="Selected model and the complete training-through-selection trail for every model evaluated on this key"
    >
      {/* A fallback did not win on merit, so it is presented as a caution
          rather than a victory — with the reason every model failed. */}
      <div
        className={
          decision.fallbackUsed
            ? 'mb-6 flex flex-col gap-4 rounded-xl border border-amber-200 bg-amber-50/60 p-5 dark:border-amber-800 dark:bg-amber-900/20 sm:flex-row sm:items-start'
            : 'mb-6 flex flex-col gap-4 rounded-xl border border-emerald-200 bg-emerald-50/60 p-5 dark:border-emerald-800 dark:bg-emerald-900/20 sm:flex-row sm:items-start'
        }
      >
        <div
          className={
            decision.fallbackUsed
              ? 'flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-amber-600 text-white'
              : 'flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-emerald-600 text-white'
          }
        >
          {decision.fallbackUsed ? <ShieldAlert size={20} /> : <Trophy size={20} />}
        </div>
        <div className="flex-1">
          <p className="text-base font-bold text-slate-800 dark:text-slate-100">
            {decision.selectedModel} — {decision.fallbackUsed ? 'Default Model Selected' : 'Selected'}
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">{decision.validationStatus}</p>
          {decision.fallbackUsed && (
            <p className="mt-2 text-sm text-amber-700 dark:text-amber-300">
              {decision.fallbackReason
                ? `${decision.fallbackReason} ${decision.selectedModel} was selected as the configured fallback model.`
                : `All evaluated models failed validation. ${decision.selectedModel} was selected as the configured fallback model.`}
            </p>
          )}
        </div>
        <div className="grid grid-cols-2 gap-x-8 gap-y-1 sm:text-right">
          <ConfidenceStat confidence={decision.confidence} breakdown={decision.confidenceExplanation} />
          <div>
            <p className="text-xs text-slate-400">Rank</p>
            <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">
              {decision.rankingPosition > 0 ? `#${decision.rankingPosition}` : '—'}
            </p>
          </div>
        </div>
      </div>

      <div className="overflow-x-auto rounded-xl border border-slate-100 dark:border-slate-800">
        <table className="w-full min-w-[720px] text-left">
          <thead>
            <tr className="border-b border-slate-100 bg-slate-50/60 dark:border-slate-800 dark:bg-slate-800/40">
              <th className="w-8 px-3 py-3" />
              <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-slate-400">Model</th>
              <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-slate-400">WMAPE</th>
              <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
                Forward Validation
              </th>
              <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-slate-400">Drift</th>
              <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-slate-400">Outcome</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((model) => (
              <EvaluatedModelRow key={model.model} model={model} isWinner={model.model === decision.selectedModel} />
            ))}
          </tbody>
        </table>
      </div>
    </SectionContainer>
  )
}
