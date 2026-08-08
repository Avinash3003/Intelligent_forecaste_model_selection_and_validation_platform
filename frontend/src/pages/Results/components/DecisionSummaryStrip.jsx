import { Trophy, Gauge, ShieldCheck, Activity, AlertTriangle } from 'lucide-react'

/**
 * The four facts a reader needs before anything else: which model shipped,
 * how confident the platform is, and whether it cleared both validation gates.
 * Deliberately four tiles and no prose — everything here is a value, not a
 * sentence, so the row can be read in about a second.
 */
function Tile({ icon: Icon, label, value, sub, tone = 'default' }) {
  const tones = {
    default: 'text-slate-800 dark:text-slate-100',
    good: 'text-emerald-600 dark:text-emerald-400',
    warn: 'text-amber-600 dark:text-amber-400',
    bad: 'text-rose-600 dark:text-rose-400',
  }
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-4 py-3 dark:border-slate-800 dark:bg-slate-900">
      <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-slate-400">
        <Icon size={12} />
        {label}
      </div>
      <div className={`mt-1.5 truncate text-lg font-semibold tabular-nums ${tones[tone]}`}>{value}</div>
      {sub && <div className="mt-0.5 truncate text-[11px] text-slate-400">{sub}</div>}
    </div>
  )
}

export default function DecisionSummaryStrip({ decision, evaluatedModels = [] }) {
  const winner = evaluatedModels.find((m) => m.model === decision.selectedModel)
  const drift = winner?.drift
  const confidence = decision.confidence

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <Tile
        icon={Trophy}
        label="Selected model"
        value={decision.selectedModel}
        sub={
          decision.fallbackUsed
            ? 'Fallback — no candidate passed'
            : decision.rankingPosition
              ? `Rank #${decision.rankingPosition}`
              : undefined
        }
        tone={decision.fallbackUsed ? 'warn' : 'default'}
      />
      <Tile
        icon={Gauge}
        label="Confidence"
        value={confidence == null ? 'N/A' : `${Math.round(confidence * 100)}%`}
        sub={winner?.backtest?.wmape != null ? `WMAPE ${winner.backtest.wmape.toFixed(2)}%` : undefined}
        tone={confidence == null ? 'default' : confidence >= 0.7 ? 'good' : confidence >= 0.5 ? 'warn' : 'bad'}
      />
      <Tile
        icon={ShieldCheck}
        label="Forward validation"
        value={winner?.forwardValidationStatus || '—'}
        sub={decision.validationStatus}
        tone={winner?.forwardValidationStatus === 'Survived' ? 'good' : 'warn'}
      />
      <Tile
        icon={drift?.passed ? Activity : AlertTriangle}
        label="Drift"
        value={drift?.evaluated ? (drift.passed ? 'Passed' : 'Failed') : 'Not reached'}
        sub={drift?.algorithm ? drift.algorithm.replace(/_/g, ' ') : undefined}
        tone={drift?.evaluated ? (drift.passed ? 'good' : 'bad') : 'default'}
      />
    </div>
  )
}
