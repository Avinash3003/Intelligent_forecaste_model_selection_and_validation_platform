import { ChevronLeft, ChevronRight } from 'lucide-react'
import { cn } from '../../utils/cn'

// Generic, reusable pagination control driven entirely by props —
// works with any client-side or server-side paginated list.
export default function Pagination({ page, totalPages, onPageChange, totalItems, pageSize }) {
  if (totalPages <= 1) return null

  const start = (page - 1) * pageSize + 1
  const end = Math.min(page * pageSize, totalItems)

  return (
    <div className="flex flex-col gap-3 border-t border-slate-100 px-1 pt-4 dark:border-slate-800 sm:flex-row sm:items-center sm:justify-between">
      <p className="text-xs text-slate-400">
        Showing <span className="font-medium text-slate-600 dark:text-slate-300">{start}–{end}</span> of{' '}
        <span className="font-medium text-slate-600 dark:text-slate-300">{totalItems}</span>
      </p>
      <div className="flex items-center gap-1.5">
        <button
          onClick={() => onPageChange(page - 1)}
          disabled={page === 1}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-500 transition-colors hover:bg-slate-100 disabled:opacity-40 disabled:hover:bg-transparent dark:text-slate-400 dark:hover:bg-slate-800"
        >
          <ChevronLeft size={16} />
        </button>
        {Array.from({ length: totalPages }, (_, i) => i + 1).map((p) => (
          <button
            key={p}
            onClick={() => onPageChange(p)}
            className={cn(
              'flex h-8 w-8 items-center justify-center rounded-lg text-xs font-medium transition-colors',
              p === page
                ? 'bg-brand-600 text-white shadow-sm shadow-brand-600/30'
                : 'text-slate-500 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800'
            )}
          >
            {p}
          </button>
        ))}
        <button
          onClick={() => onPageChange(page + 1)}
          disabled={page === totalPages}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-500 transition-colors hover:bg-slate-100 disabled:opacity-40 disabled:hover:bg-transparent dark:text-slate-400 dark:hover:bg-slate-800"
        >
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  )
}
