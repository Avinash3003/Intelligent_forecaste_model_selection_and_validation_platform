import { cn } from '../../utils/cn'

export default function Input({ icon: Icon, className, containerClassName, ...props }) {
  return (
    <div className={cn('relative flex items-center', containerClassName)}>
      {Icon && (
        <Icon
          size={16}
          className="absolute left-3 text-slate-400 pointer-events-none"
          strokeWidth={2}
        />
      )}
      <input
        className={cn(
          'w-full rounded-lg border border-slate-200 bg-white/80 py-2 text-sm text-slate-700 placeholder:text-slate-400 outline-none transition-all focus:border-brand-400 focus:ring-2 focus:ring-brand-100 dark:bg-slate-800/60 dark:border-slate-700 dark:text-slate-200 dark:focus:ring-brand-900/40',
          Icon ? 'pl-9 pr-3' : 'px-3',
          className
        )}
        {...props}
      />
    </div>
  )
}
