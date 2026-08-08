import * as Icons from 'lucide-react'
import { ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react'
import Card from '../ui/Card'

const trendIcon = {
  up: ArrowUpRight,
  down: ArrowDownRight,
  neutral: Minus,
}

const trendColor = {
  up: 'text-emerald-600 dark:text-emerald-400',
  down: 'text-rose-600 dark:text-rose-400',
  neutral: 'text-slate-400',
}

export default function StatCard({ label, value, delta, trend = 'neutral', icon }) {
  const Icon = Icons[icon] || Icons.Activity
  const TrendIcon = trendIcon[trend]

  return (
    <Card hover className="p-5">
      <div className="flex items-start justify-between">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-900/30 dark:text-brand-400">
          <Icon size={19} strokeWidth={2} />
        </div>
      </div>
      <p className="mt-4 text-2xl font-bold tracking-tight text-slate-800 dark:text-slate-100">{value}</p>
      <p className="text-sm text-slate-400 mt-0.5">{label}</p>
      <div className={`mt-3 flex items-center gap-1 text-xs font-medium ${trendColor[trend]}`}>
        <TrendIcon size={13} />
        <span>{delta}</span>
      </div>
    </Card>
  )
}
