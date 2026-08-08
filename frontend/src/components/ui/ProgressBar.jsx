import { cn } from '../../utils/cn'

export default function ProgressBar({ value = 0, showLabel = false, className, barClassName }) {
  const clamped = Math.min(100, Math.max(0, value))

  return (
    <div className={cn('flex items-center gap-2.5', className)}>
      <div className="h-1.5 w-full min-w-[64px] overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
        <div
          className={cn('h-full rounded-full bg-brand-600 transition-all duration-500', barClassName)}
          style={{ width: `${clamped}%` }}
        />
      </div>
      {showLabel && (
        <span className="shrink-0 text-xs font-medium text-slate-400 tabular-nums">{clamped}%</span>
      )}
    </div>
  )
}
