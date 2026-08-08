import Card from '../../../components/ui/Card'
import ProgressBar from '../../../components/ui/ProgressBar'

function SummaryItem({ label, value }) {
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">{label}</p>
      <p className="mt-1 text-sm font-semibold text-slate-800 dark:text-slate-100">{value}</p>
    </div>
  )
}

export default function RunSummaryBar({ run }) {
  return (
    <Card className="p-5">
      <div className="grid grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-6">
        <SummaryItem label="Run ID" value={run.id} />
        <SummaryItem label="Dataset" value={run.dataset} />
        <SummaryItem label="Started" value={run.startTime} />
        <SummaryItem label="Current Stage" value={run.currentStage} />
        <SummaryItem label="Remaining" value={run.estimatedRemaining} />
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Progress</p>
          <div className="mt-2">
            <ProgressBar value={run.progress} showLabel />
          </div>
        </div>
      </div>
    </Card>
  )
}
