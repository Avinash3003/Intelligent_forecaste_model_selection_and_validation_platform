import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'
import { cn } from '../../utils/cn'

// Generic accordion/disclosure panel — reusable anywhere a section needs
// to be revealed on demand (metrics, raw payloads, advanced settings).
export default function ExpandablePanel({ title, subtitle, defaultOpen = false, children, className }) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className={cn('rounded-xl border border-slate-100 dark:border-slate-800', className)}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-4 px-4 py-3.5 text-left"
      >
        <div>
          <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">{title}</p>
          {subtitle && <p className="mt-0.5 text-xs text-slate-400">{subtitle}</p>}
        </div>
        <ChevronDown
          size={16}
          className={cn('shrink-0 text-slate-400 transition-transform duration-200', open && 'rotate-180')}
        />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            className="overflow-hidden"
          >
            <div className="border-t border-slate-100 px-4 py-4 dark:border-slate-800">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
