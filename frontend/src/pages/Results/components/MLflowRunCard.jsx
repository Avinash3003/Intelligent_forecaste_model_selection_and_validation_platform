import { FlaskConical } from 'lucide-react'
import { Link } from 'react-router-dom'
import SectionContainer from '../../../components/layout/SectionContainer'
import Badge from '../../../components/ui/Badge'
import OpenInDatabricksLink from '../../../components/ui/OpenInDatabricksLink'

const fields = [
  { key: 'runId', label: 'MLflow Run ID' },
  { key: 'experiment', label: 'Experiment' },
  { key: 'trackingUri', label: 'Tracking URI' },
  { key: 'modelsRegistered', label: 'Models Registered' },
]

// Governance pointer: where this run's parameters, metrics and artifacts
// were logged (Section 6.13).
//
// Two actions, deliberately distinct:
//
//   "View run details" is the path that always works. It stays inside
//   ForecastIQ, and the backend reads the run through its own configured
//   Databricks credentials — so any authenticated ForecastIQ user sees the
//   run without needing a Databricks account of their own.
//
//   "Open in Databricks" is a convenience for people who already have
//   workspace access; Databricks applies its own sign-in on arrival. It is
//   secondary precisely because it cannot be guaranteed for every user, and
//   it renders only when the backend could build a correct URL.
export default function MLflowRunCard({ run, resultRunId }) {
  const detailsHref = resultRunId
    ? `/mlflow-experiments?run=${encodeURIComponent(resultRunId)}`
    : '/mlflow-experiments'

  return (
    <SectionContainer
      title="MLflow run"
      subtitle="Experiment tracking record for this execution"
      action={
        <div className="flex items-center gap-2">
          <Link
            to={detailsHref}
            title="View this run's MLflow record inside ForecastIQ"
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-2.5 py-1.5 text-xs font-semibold text-slate-600 transition hover:border-brand-300 hover:text-brand-600 dark:border-slate-700 dark:text-slate-300 dark:hover:border-brand-500 dark:hover:text-brand-400"
          >
            View run details
          </Link>
          {/* "Open in Databricks" hidden: it deep-links to the Databricks
              workspace, which applies its own sign-in, and ForecastIQ users
              are not provisioned there — so it reliably shows a permission
              error. "View run details" above covers the same need in-app.
              Restore by uncommenting once workspace access exists.
          <OpenInDatabricksLink url={run.databricksRunUrl} />
          */}
          {run.status ? (
            <Badge status={run.status === 'logged' ? 'Completed' : 'neutral'}>{run.status}</Badge>
          ) : null}
        </div>
      }
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
