import { useRef, useState } from 'react'
import { UploadCloud, FileCheck2, Loader2, X } from 'lucide-react'
import Button from '../ui/Button'
import { cn } from '../../utils/cn'

// Reusable drag & drop file picker. Reads only local file metadata
// (name/size/type) for display — actual upload/transmission is the
// caller's responsibility via `onFileSelect`.
export default function FileDropzone({ file, onFileSelect, onRemove, accept, loading, className }) {
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef(null)

  const handleFiles = (fileList) => {
    if (loading) return
    const selected = fileList?.[0]
    if (selected) onFileSelect?.(selected)
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setDragActive(true)
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragActive(false)
        handleFiles(e.dataTransfer.files)
      }}
      className={cn(
        'flex flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-14 text-center transition-all duration-200',
        dragActive
          ? 'border-brand-400 bg-brand-50/60 dark:bg-brand-900/10'
          : 'border-slate-200 bg-slate-50/40 dark:border-slate-700 dark:bg-slate-800/30',
        className
      )}
    >
      {loading ? (
        <>
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-50 text-brand-600 dark:bg-brand-900/30 dark:text-brand-400">
            <Loader2 size={26} strokeWidth={1.75} className="animate-spin" />
          </div>
          <p className="mt-4 text-sm font-semibold text-slate-700 dark:text-slate-200">Uploading dataset…</p>
          <p className="mt-1 text-xs text-slate-400">This may take a moment for larger files.</p>
        </>
      ) : !file ? (
        <>
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-50 text-brand-600 dark:bg-brand-900/30 dark:text-brand-400">
            <UploadCloud size={26} strokeWidth={1.75} />
          </div>
          <p className="mt-4 text-sm font-semibold text-slate-700 dark:text-slate-200">
            Drop a CSV or Excel file here
          </p>
          <p className="mt-1 text-xs text-slate-400">or connect a Delta table / ADLS path</p>
          <Button
            variant="secondary"
            size="sm"
            className="mt-5"
            onClick={() => inputRef.current?.click()}
          >
            Browse File
          </Button>
          <input
            ref={inputRef}
            type="file"
            accept={accept}
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
        </>
      ) : (
        <div className="flex w-full max-w-sm items-center justify-between gap-3 rounded-xl border border-emerald-200 bg-emerald-50/70 px-4 py-3 text-left dark:border-emerald-800 dark:bg-emerald-900/20">
          <div className="flex min-w-0 items-center gap-3">
            <FileCheck2 size={20} className="shrink-0 text-emerald-600 dark:text-emerald-400" />
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-slate-700 dark:text-slate-200">{file.name}</p>
              <p className="text-xs text-slate-400">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
            </div>
          </div>
          <button
            onClick={onRemove}
            className="shrink-0 rounded-md p-1 text-slate-400 hover:bg-white hover:text-rose-500 dark:hover:bg-slate-800"
            aria-label="Remove file"
          >
            <X size={16} />
          </button>
        </div>
      )}
    </div>
  )
}
