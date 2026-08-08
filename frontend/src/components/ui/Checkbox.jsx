import { Check } from 'lucide-react'
import { cn } from '../../utils/cn'

export default function Checkbox({ checked, onChange, label, description, disabled, className }) {
  return (
    <label
      className={cn(
        'flex cursor-pointer items-start gap-3 rounded-xl border border-slate-200 bg-white/60 p-3.5 transition-all duration-200 dark:border-slate-700 dark:bg-slate-800/40',
        checked && 'border-brand-300 bg-brand-50/60 dark:border-brand-700 dark:bg-brand-900/20',
        disabled && 'cursor-not-allowed opacity-50',
        className
      )}
    >
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange?.(e.target.checked)}
        className="sr-only"
      />
      <span
        className={cn(
          'mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md border transition-colors',
          checked
            ? 'border-brand-600 bg-brand-600 text-white'
            : 'border-slate-300 bg-white dark:border-slate-600 dark:bg-slate-900'
        )}
      >
        {checked && <Check size={13} strokeWidth={3} />}
      </span>
      <span className="min-w-0">
        <span className="block text-sm font-medium text-slate-700 dark:text-slate-200">{label}</span>
        {description && (
          <span className="mt-0.5 block text-xs text-slate-400">{description}</span>
        )}
      </span>
    </label>
  )
}
