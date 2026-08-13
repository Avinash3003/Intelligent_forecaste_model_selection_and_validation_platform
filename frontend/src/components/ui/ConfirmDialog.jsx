import { AlertTriangle } from 'lucide-react'
import Button from './Button'

// A small, focused confirmation overlay for a destructive action — used
// today only by "Cancel Run" (deleting a run's generated data is
// irreversible), so it defaults to the danger styling rather than trying
// to be a general-purpose modal.
export default function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = 'Confirm',
  cancelLabel = 'Keep it',
  confirmVariant = 'danger',
  loading = false,
  onConfirm,
  onCancel,
}) {
  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 px-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-dialog-title"
      onClick={onCancel}
    >
      <div
        className="glass-panel w-full max-w-sm rounded-2xl border border-slate-200 bg-white p-5 shadow-xl dark:border-slate-700 dark:bg-slate-900"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start gap-3">
          <div className="rounded-full bg-rose-100 p-2 text-rose-600 dark:bg-rose-500/10 dark:text-rose-400">
            <AlertTriangle size={18} />
          </div>
          <div>
            <h3 id="confirm-dialog-title" className="text-sm font-semibold text-slate-800 dark:text-slate-100">
              {title}
            </h3>
            <p className="mt-1.5 text-sm text-slate-500 dark:text-slate-400">{description}</p>
          </div>
        </div>

        <div className="mt-5 flex justify-end gap-2.5">
          <Button variant="secondary" size="sm" onClick={onCancel} disabled={loading}>
            {cancelLabel}
          </Button>
          <Button variant={confirmVariant} size="sm" onClick={onConfirm} disabled={loading}>
            {loading ? 'Cancelling…' : confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  )
}
