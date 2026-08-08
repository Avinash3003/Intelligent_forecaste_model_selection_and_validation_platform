import { useEffect, useRef, useState } from 'react'
import { ChevronDown, Check, X } from 'lucide-react'
import { cn } from '../../utils/cn'
import DropdownPortal from './DropdownPortal'

function normalize(opt) {
  return typeof opt === 'object' ? opt : { value: opt, label: opt }
}

// `options` may be plain strings or { value, label, sublabel } objects.
// `values` / onChange always operate on the underlying `value`.
export default function MultiSelect({ values = [], onChange, options, placeholder = 'Select...', error }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)
  const normalized = options.map(normalize)
  const labelFor = (val) => normalized.find((o) => o.value === val)?.label ?? val

  useEffect(() => {
    function handleClickOutside(e) {
      const insideAnchor = ref.current && ref.current.contains(e.target)
      const insidePortal = e.target.closest?.('[data-dropdown-portal]')
      if (!insideAnchor && !insidePortal) setOpen(false)
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const toggleValue = (val) => {
    if (values.includes(val)) {
      onChange?.(values.filter((v) => v !== val))
    } else {
      onChange?.([...values, val])
    }
  }

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={cn(
          'flex w-full min-h-[2.5rem] items-center justify-between gap-2 rounded-lg border bg-white/80 px-3 py-1.5 text-left text-sm transition-all focus:ring-2 focus:ring-brand-100 dark:bg-slate-800/60',
          error
            ? 'border-rose-300 dark:border-rose-700'
            : 'border-slate-200 dark:border-slate-700'
        )}
      >
        {values.length === 0 ? (
          <span className="text-slate-400">{placeholder}</span>
        ) : (
          <span className="flex flex-wrap gap-1.5 py-0.5">
            {values.map((val) => (
              <span
                key={val}
                className="inline-flex items-center gap-1 rounded-md bg-brand-50 px-2 py-0.5 text-xs font-medium text-brand-700 dark:bg-brand-900/30 dark:text-brand-300"
              >
                {labelFor(val)}
                <X
                  size={12}
                  className="cursor-pointer"
                  onClick={(e) => {
                    e.stopPropagation()
                    toggleValue(val)
                  }}
                />
              </span>
            ))}
          </span>
        )}
        <ChevronDown size={15} className="shrink-0 text-slate-400" />
      </button>

      <DropdownPortal anchorRef={ref} open={open} className="z-50">
        <div className="rounded-lg border border-slate-200 bg-white py-1.5 shadow-glass dark:border-slate-700 dark:bg-slate-900">
          {normalized.map((opt) => {
            const active = values.includes(opt.value)
            return (
              <div
                key={opt.value}
                onClick={() => toggleValue(opt.value)}
                className="flex cursor-pointer items-center justify-between gap-2 px-3 py-2 hover:bg-slate-50 dark:hover:bg-slate-800"
              >
                <span className="flex min-w-0 flex-col">
                  <span className="truncate text-sm text-slate-600 dark:text-slate-300">{opt.label}</span>
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
