import { Check } from 'lucide-react'
import { cn } from '../../utils/cn'

// Generic horizontal wizard step indicator — reusable across any
// multi-step flow (pipeline builder today, future onboarding flows, etc.)
export default function StepIndicator({ steps, currentStep, onStepClick, maxReachedStep }) {
  return (
    <ol className="flex items-center gap-2 sm:gap-3">
      {steps.map((step, idx) => {
        const isActive = step.id === currentStep
        const isCompleted = step.id < currentStep
        const isReachable = step.id <= maxReachedStep
        const isLast = idx === steps.length - 1

        return (
          <li key={step.id} className={cn('flex items-center', !isLast && 'flex-1')}>
            <button
              type="button"
              disabled={!isReachable}
              onClick={() => isReachable && onStepClick?.(step.id)}
              className={cn(
                'flex items-center gap-2.5 shrink-0 rounded-xl px-2.5 py-1.5 transition-colors',
                isReachable ? 'cursor-pointer' : 'cursor-not-allowed'
              )}
            >
              <span
                className={cn(
                  'flex h-8 w-8 shrink-0 items-center justify-center rounded-full border text-xs font-semibold transition-all duration-200',
                  isCompleted &&
                    'border-brand-600 bg-brand-600 text-white',
                  isActive &&
                    'border-brand-600 bg-white text-brand-600 ring-4 ring-brand-100 dark:bg-slate-900 dark:ring-brand-900/40',
                  !isActive &&
                    !isCompleted &&
                    'border-slate-200 bg-white text-slate-400 dark:border-slate-700 dark:bg-slate-900'
                )}
              >
                {isCompleted ? <Check size={14} strokeWidth={3} /> : step.id}
              </span>
              <span
                className={cn(
                  'hidden text-sm font-medium sm:block',
                  isActive
                    ? 'text-slate-800 dark:text-slate-100'
                    : isCompleted
                    ? 'text-slate-600 dark:text-slate-300'
                    : 'text-slate-400'
                )}
              >
                {step.label}
              </span>
            </button>
            {!isLast && (
              <span
                className={cn(
                  'mx-1 h-px flex-1 min-w-[16px]',
                  isCompleted ? 'bg-brand-400' : 'bg-slate-200 dark:bg-slate-700'
                )}
              />
            )}
          </li>
        )
      })}
    </ol>
  )
}
