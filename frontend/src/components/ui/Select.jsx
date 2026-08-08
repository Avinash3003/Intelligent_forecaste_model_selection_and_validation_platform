import { useEffect, useRef, useState } from 'react'
import { ChevronDown, Check } from 'lucide-react'
import { cn } from '../../utils/cn'
import DropdownPortal from './DropdownPortal'

function normalize(opt) {
  return typeof opt === 'object' ? opt : { value: opt, label: opt }
}

export default function Select({ value, onChange, options, placeholder = 'Select...', className, error }) {
  const normalized = options.map(normalize)
  const isRich = normalized.some((o) => o.sublabel)

  if (!isRich) {
    return (
      <div className="relative">
        <select
          value={value ?? ''}
          onChange={(e) => onChange?.(e.target.value)}
          className={cn(
            'w-full appearance-none rounded-lg border bg-white/80 py-2 pl-3 pr-9 text-sm text-slate-700 outline-none transition-all focus:ring-2 focus:ring-brand-100 dark:bg-slate-800/60 dark:text-slate-200',
            error
              ? 'border-rose-300 focus:border-rose-400 dark:border-rose-700'
              : 'border-slate-200 focus:border-brand-400 dark:border-slate-700 dark:focus:ring-brand-900/40',
            !value && 'text-slate-400',
            className
          )}
        >
          <option value="" disabled>
            {placeholder}
          </option>
          {normalized.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <ChevronDown
          size={15}
          className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-400"
        />
      </div>
    )
  }

  return <RichSelect value={value} onChange={onChange} options={normalized} placeholder={placeholder} error={error} className={className} />
}

function RichSelect({ value, onChange, options, placeholder, error, className }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)
  const selected = options.find((o) => o.value === value)

  useEffect(() => {
    function handleClickOutside(e) {
      const insideAnchor = ref.current && ref.current.contains(e.target)
      const insidePortal = e.target.closest?.('[data-dropdown-portal]')
      if (!insideAnchor && !insidePortal) setOpen(false)
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={cn(
          'flex w-full items-center justify-between gap-2 rounded-lg border bg-white/80 px-3 py-2 text-left text-sm transition-all focus:ring-2 focus:ring-brand-100 dark:bg-slate-800/60',
          error ? 'border-rose-300 dark:border-rose-700' : 'border-slate-200 dark:border-slate-700',
          className
        )}
      >
        {selected ? (
          <span className="flex min-w-0 flex-col">
            <span className="truncate text-sm font-medium text-slate-700 dark:text-slate-200">
              {selected.label}
            </span>
            <span className="truncate text-xs text-slate-400">{selected.sublabel}</span>
          </span>
        ) : (
          <span className="text-slate-400">{placeholder}</span>
        )}
        <ChevronDown size={15} className="shrink-0 text-slate-400" />
      </button>

      <DropdownPortal anchorRef={ref} open={open} className="z-50">
        <div className="rounded-lg border border-slate-200 bg-white py-1.5 shadow-glass dark:border-slate-700 dark:bg-slate-900">
          {options.map((opt) => {
            const active = opt.value === value
            return (
              <div
                key={opt.value}
                onClick={() => {
                  onChange?.(opt.value)
                  setOpen(false)
                }}
                className="flex cursor-pointer items-center justify-between gap-2 px-3 py-2 hover:bg-slate-50 dark:hover:bg-slate-800"
              >
                <span className="flex min-w-0 flex-col">
                  <span className="truncate text-sm text-slate-700 dark:text-slate-200">{opt.label}</span>
                  {opt.sublabel && <span className="truncate text-xs text-slate-400">{opt.sublabel}</span>}
                </span>
                {active && <Check size={14} className="shrink-0 text-brand-600 dark:text-brand-400" />}
              </div>
            )
          })}
        </div>
      </DropdownPortal>
    </div>
  )
}
