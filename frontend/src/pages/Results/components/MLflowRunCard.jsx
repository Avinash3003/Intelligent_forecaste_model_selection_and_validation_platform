import { FlaskConical } from 'lucide-react'
import SectionContainer from '../../../components/layout/SectionContainer'
import Badge from '../../../components/ui/Badge'

const fields = [
  { key: 'runId', label: 'MLflow Run ID' },
  { key: 'experiment', label: 'Experiment' },
  { key: 'trackingUri', label: 'Tracking URI' },
  { key: 'modelsRegistered', label: 'Models Registered' },
]

// Governance pointer: where this run's parameters, metrics and artifacts
// were logged (Section 6.13).
export default function MLflowRunCard({ run }) {
  return (
    <SectionContainer
      title="MLflow run"
      subtitle="Experiment tracking record for this execution"
      action={run.status ? <Badge status={run.status === 'logged' ? 'Completed' : 'neutral'}>{run.status}</Badge> : null}
    >
      <div className="flex items-start gap-3">
        <FlaskConical size={18} className="mt-0.5 shrink-0 text-brand-600 dark:text-brand-400" />
        <div className="grid flex-1 grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {fields.map((field) => (
            <div key={field.key} className="min-w-0">
              <p className="text-xs text-slate-400">{field.label}</p>
              <p className="mt-1 truncate text-sm font-semibold text-slate-700 dark:text-slate-200">
                {run[field.key] ?? '—'}
              </p>
            </div>
          ))}
        </div>
      </div>
    </SectionContainer>
  )
}
