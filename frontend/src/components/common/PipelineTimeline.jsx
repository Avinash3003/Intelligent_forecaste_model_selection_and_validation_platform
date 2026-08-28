import { CheckCircle2, Loader2, Circle, XCircle } from 'lucide-react'
import { cn } from '../../utils/cn'
import ProgressBar from '../ui/ProgressBar'
import { formatISTTime } from '../../utils/formatDateTime'

const statusConfig = {
  Completed: {
    icon: CheckCircle2,
    dot: 'border-emerald-600 bg-emerald-600 text-white',
    label: 'text-slate-700 dark:text-slate-200',
    line: 'bg-emerald-400',
  },
  Running: {
    icon: Loader2,
    dot: 'border-blue-600 bg-white text-blue-600 ring-4 ring-blue-100 dark:bg-slate-900 dark:ring-blue-900/40',
    label: 'text-slate-800 dark:text-slate-100 font-semibold',
    line: 'bg-slate-200 dark:bg-slate-700',
    row: 'bg-blue-50/60 dark:bg-blue-900/15',
    spin: true,
  },
  Failed: {
    icon: XCircle,
    dot: 'border-rose-600 bg-rose-600 text-white',
    label: 'text-rose-600 dark:text-rose-400 font-semibold',
    line: 'bg-slate-200 dark:bg-slate-700',
    row: 'bg-rose-50/60 dark:bg-rose-900/15',
  },
  Pending: {
    icon: Circle,
    dot: 'border-slate-200 bg-white text-slate-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-600',
    label: 'text-slate-400',
    line: 'bg-slate-200 dark:bg-slate-700',
  },
}

// The infrastructure step that happens BEFORE the phases below: Databricks
// acquiring the compute this run was submitted to. Rendered as its own block
// rather than an eighth row, because the phases are the forecast engine's
// own stages and this is not one of them.
const computeConfig = {
  starting: {
    icon: Loader2,
    spin: true,
    box: 'border-blue-200 bg-blue-50/60 dark:border-blue-900 dark:bg-blue-900/15',
    icon_: 'text-blue-600 dark:text-blue-400',
    title: 'text-blue-700 dark:text-blue-300',
    tone: 'text-blue-600 dark:text-blue-400',
  },
  ready: {
    icon: CheckCircle2,
    box: 'border-emerald-200 bg-emerald-50/60 dark:border-emerald-800 dark:bg-emerald-900/15',
    icon_: 'text-emerald-600 dark:text-emerald-400',
    title: 'text-emerald-700 dark:text-emerald-300',
    tone: 'text-emerald-600 dark:text-emerald-400',
  },
  failed: {
    icon: XCircle,
    box: 'border-rose-200 bg-rose-50/60 dark:border-rose-800 dark:bg-rose-900/15',
    icon_: 'text-rose-600 dark:text-rose-400',
    title: 'text-rose-700 dark:text-rose-300',
    tone: 'text-rose-600 dark:text-rose-400',
  },
}

// What each phase is doing, in the user's terms. Keyed by the backend's own
// phase labels (services/pipeline_stages.py PIPELINE_PHASES); a phase with
// no entry simply shows no description.
const phaseDescriptions = {
  'Load & Prepare': 'Reading, validating and cleaning the uploaded dataset',
  'Build Series': 'Splitting the dataset into forecast groups',
  'Train Models': 'Fitting each selected model for every forecast group',
  'Evaluate Models': 'Backtesting and validating every trained model',
  'Explain Models': 'Measuring feature importance for the surviving models',
  'Rank & Select': 'Scoring candidates and choosing a winner per group',
  'Publish Results': 'Persisting models, exporting forecasts and recording the run',
}

// Wall clock across a phase, shown only once it has actually finished
function elapsedLabel(startedAt, completedAt) {
  if (!startedAt || !completedAt) return null
  const seconds = Math.round((new Date(completedAt) - new Date(startedAt)) / 1000)
  if (!Number.isFinite(seconds) || seconds < 0) return null
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  return seconds % 60 ? `${minutes}m ${seconds % 60}s` : `${minutes}m`
}

// Generic vertical DAG stage timeline — reusable for any staged execution
// (forecasting run today, future async workflows).
//
// `progress`, `runStatus` and `error` are optional: passing them turns the
// bare trail into a progress monitor with a header and a terminal banner.
// Without them it renders exactly as before.
export default function PipelineTimeline({ stages, progress, runStatus, compute, error }) {
  const running = stages.find((stage) => stage.status === 'Running')
  const failed = stages.find((stage) => stage.status === 'Failed')
  const completedCount = stages.filter((stage) => stage.status === 'Completed').length
  const showHeader = typeof progress === 'number'
  const computeCfg = compute ? computeConfig[compute.state] : null

  // Before any phase has begun, the compute step is the only thing actually
  // happening — saying "Pending" there is what made a booting cluster look
  // like a stuck application.
  const headline = failed
    ? { text: 'Failed', tone: 'text-rose-600 dark:text-rose-400' }
    : runStatus === 'Completed'
      ? { text: 'Completed', tone: 'text-emerald-600 dark:text-emerald-400' }
      : running
        ? { text: 'Running', tone: 'text-blue-600 dark:text-blue-400' }
        : computeCfg
          ? { text: compute.label, tone: computeCfg.tone }
          : { text: runStatus || 'Pending', tone: 'text-slate-500 dark:text-slate-400' }

  return (
    <div>
      {showHeader && (
        <div className="mb-6 border-b border-slate-100 pb-5 dark:border-slate-800">
          <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Overall progress
            </p>
            <p className="text-xs text-slate-400 tabular-nums">
              {completedCount} of {stages.length} phases complete
            </p>
          </div>
          <ProgressBar
            value={progress}
            showLabel
            barClassName={failed ? 'bg-rose-500' : runStatus === 'Completed' ? 'bg-emerald-500' : undefined}
          />
          <p className="mt-3 flex flex-wrap items-baseline gap-x-2 text-sm">
            <span className={cn('font-semibold', headline.tone)}>{headline.text}</span>
            {(running || failed) && (
              <span className="text-slate-600 dark:text-slate-300">{(failed || running).label}</span>
            )}
          </p>
        </div>
      )}

      {computeCfg && (
        <div className={cn('mb-6 rounded-xl border px-4 py-3.5', computeCfg.box)}>
          <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            Compute
          </p>
          <div className="flex items-start gap-3">
            <computeCfg.icon
              size={18}
              className={cn('mt-0.5 shrink-0', computeCfg.icon_, computeCfg.spin && 'animate-spin')}
            />
            <div className="min-w-0">
              <p className={cn('text-sm font-semibold', computeCfg.title)}>{compute.label}</p>
              <p className="mt-0.5 text-sm text-slate-600 dark:text-slate-300">{compute.message}</p>
              {compute.detail && (
                <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{compute.detail}</p>
              )}
            </div>
          </div>
        </div>
      )}

      <ol>
        {stages.map((stage, idx) => {
          const cfg = statusConfig[stage.status] || statusConfig.Pending
          const Icon = cfg.icon
          const isLast = idx === stages.length - 1
          const elapsed = elapsedLabel(stage.startedAt, stage.completedAt)
          const description = phaseDescriptions[stage.label]

          return (
            <li
              key={stage.label}
              className={cn(
                'relative flex gap-4 rounded-lg pb-8 last:pb-0',
                cfg.row && '-mx-2 px-2 pt-2'
              )}
            >
              {!isLast && (
                <span
                  className={cn('absolute left-[15px] top-8 h-[calc(100%-1.5rem)] w-0.5', cfg.line, cfg.row && 'left-[23px] top-10')}
                />
              )}
              <span
                className={cn(
                  'z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border-2',
                  cfg.dot
                )}
              >
                <Icon size={16} strokeWidth={2.25} className={cfg.spin ? 'animate-spin' : ''} />
              </span>
              <div className="min-w-0 pt-1">
                <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                  <p className={cn('text-sm', cfg.label)}>{stage.label}</p>
                  {elapsed && (
                    <span className="text-xs text-slate-400 tabular-nums">{elapsed}</span>
                  )}
                </div>
                {description && stage.status !== 'Pending' && (
                  <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{description}</p>
                )}
                <p className="mt-0.5 text-xs text-slate-400">
                  {stage.status}
                  {stage.startedAt && (
                    <>
                      {' · started '}
                      {formatISTTime(stage.startedAt)}
                    </>
                  )}
                </p>
              </div>
            </li>
          )
        })}
      </ol>

      {runStatus === 'Completed' && !failed && (
        <div className="mt-6 flex items-start gap-2.5 rounded-xl border border-emerald-200 bg-emerald-50/70 px-4 py-3 text-sm text-emerald-700 dark:border-emerald-800 dark:bg-emerald-900/20 dark:text-emerald-300">
          <CheckCircle2 size={16} className="mt-0.5 shrink-0" />
          <p className="font-medium">Forecast generated successfully across all phases.</p>
        </div>
      )}

      {failed && (
        <div className="mt-6 flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
          <XCircle size={16} className="mt-0.5 shrink-0" />
          <div className="min-w-0">
            <p className="font-medium">Failed during {failed.label}</p>
            {error && <p className="mt-0.5 whitespace-pre-wrap break-words">{error}</p>}
          </div>
        </div>
      )}
    </div>
  )
}
