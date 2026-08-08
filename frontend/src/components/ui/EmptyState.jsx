import { Inbox } from 'lucide-react'
import { cn } from '../../utils/cn'

export default function EmptyState({
  icon: Icon = Inbox,
  title = 'Nothing here yet',
  description = 'This module will be available in a future phase.',
  action,
  className,
}) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-slate-200 dark:border-slate-800 py-20 text-center',
        className
      )}
    >
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-50 text-brand-600 dark:bg-brand-900/30 dark:text-brand-400">
        <Icon size={26} strokeWidth={1.75} />
      </div>
      <div className="space-y-1">
        <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200">{title}</h3>
        <p className="max-w-sm text-sm text-slate-400">{description}</p>
      </div>
      {action}
    </div>
  )
}
