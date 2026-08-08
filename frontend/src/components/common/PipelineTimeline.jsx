import { CheckCircle2, Loader2, Circle, XCircle } from 'lucide-react'
import { cn } from '../../utils/cn'
import { formatDuration, formatISTTime, secondsBetween } from '../../utils/formatDateTime'

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
    spin: true,
  },
  Failed: {
    icon: XCircle,
    dot: 'border-rose-600 bg-rose-600 text-white',
    label: 'text-rose-600 dark:text-rose-400 font-semibold',
    line: 'bg-slate-200 dark:bg-slate-700',
  },
  Pending: {
    icon: Circle,
    dot: 'border-slate-200 bg-white text-slate-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-600',
    label: 'text-slate-400',
    line: 'bg-slate-200 dark:bg-slate-700',
  },
}

// Generic vertical DAG stage timeline — reusable for any staged execution
// (forecasting run today, future async workflows).
export default function PipelineTimeline({ stages }) {
  return (
    <ol>
      {stages.map((stage, idx) => {
        const cfg = statusConfig[stage.status] || statusConfig.Pending
        const Icon = cfg.icon
        const isLast = idx === stages.length - 1

        return (
          <li key={stage.label} className="relative flex gap-4 pb-8 last:pb-0">
            {!isLast && (
              <span
                className={cn('absolute left-[15px] top-8 h-[calc(100%-1.5rem)] w-0.5', cfg.line)}
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
            <div className="pt-1">
              <p className={cn('text-sm', cfg.label)}>{stage.label}</p>
              <p className="mt-0.5 text-xs text-slate-400">
                {stage.status}
                {stage.startedAt && (
                  <>
                    {' · started '}
                    {formatISTTime(stage.startedAt)}
                    {stage.completedAt && (
                      <> · took {formatDuration(secondsBetween(stage.startedAt, stage.completedAt))}</>
                    )}
                  </>
                )}
              </p>
            </div>
          </li>
        )
      })}
    </ol>
  )
}
