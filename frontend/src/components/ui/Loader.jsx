import { Loader2 } from 'lucide-react'
import { cn } from '../../utils/cn'

export default function Loader({ label = 'Loading...', className }) {
  return (
    <div className={cn('flex flex-col items-center justify-center gap-3 py-16 text-slate-400', className)}>
      <Loader2 className="animate-spin text-brand-500" size={28} strokeWidth={2} />
      <p className="text-sm font-medium">{label}</p>
    </div>
  )
}
