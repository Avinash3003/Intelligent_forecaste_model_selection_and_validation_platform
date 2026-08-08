import { useEffect, useState } from 'react'
import { Bug, RefreshCw } from 'lucide-react'
import ExpandablePanel from '../../../components/common/ExpandablePanel'
import Loader from '../../../components/ui/Loader'
import Button from '../../../components/ui/Button'
import Badge from '../../../components/ui/Badge'
import { fetchDebugSummary } from '../../../services'
import { formatDuration, formatISTTime, secondsBetween } from '../../../utils/formatDateTime'

function Field({ label, value }) {
  return (
    <div>
      <p className="text-xs text-slate-400">{label}</p>
      <p className="mt-0.5 text-sm font-medium text-slate-700 dark:text-slate-200">{value ?? '—'}</p>
    </div>
  )
}

function StatusCounts({ title, counts }) {
  if (!counts || counts.length === 0) {
    return (
      <div>
        <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</p>
        <p className="text-xs text-slate-400">No records.</p>
      </div>
    )
  }
  return (
    <div>
      <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</p>
      <div className="flex flex-wrap gap-1.5">
        {counts.map((entry) => (
          <Badge key={entry.status} status={entry.status}>
            {entry.status}: {entry.count}
          </Badge>
        ))}
      </div>
    </div>
  )
}

// Developer debugging mode (item 6): a structured, traceable dump of what
// actually happened during a run — fetched separately from the main
// dashboard payload and only loaded when this panel is opened, since it is
// diagnostic detail, not something every user needs on every page load.
export default function DebugPanel({ runId }) {
  const [open, setOpen] = useState(false)
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showRawJson, setShowRawJson] = useState(false)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      setSummary(await fetchDebugSummary(runId))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (open && !summary && !loading) load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  return (
    <ExpandablePanel
      title="Developer debug summary"
      subtitle="Dataset, groups, training, MLflow and artifact detail for this run — traceable to the raw pipeline output"
    >
      <div className="mb-3 flex items-center justify-between">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="flex items-center gap-1.5 text-xs font-medium text-brand-600 hover:text-brand-700 dark:text-brand-400"
        >
          <Bug size={13} /> {open ? 'Hide' : 'Load'} debug summary
        </button>
        {open && (
          <Button variant="secondary" size="sm" icon={RefreshCw} onClick={load}>
            Refresh
          </Button>
        )}
      </div>

      {!open && (
        <p className="text-xs text-slate-400">
          Click "Load debug summary" for the complete execution internals of this run.
        </p>
      )}

      {open && loading && <Loader label="Loading debug summary…" />}

      {open && error && <p className="text-xs text-rose-600 dark:text-rose-400">{error}</p>}

      {open && summary && !loading && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Field label="Run ID" value={<span className="font-mono text-xs">{summary.run_id}</span>} />
            <Field label="Status" value={<Badge status={summary.job_status}>{summary.job_status}</Badge>} />
            <Field label="Dataset" value={summary.dataset_name} />
            <Field label="Frequency detected" value={summary.frequency} />
            <Field label="Mode" value={summary.mode} />
            <Field label="Aggregation" value={summary.aggregation_method} />
            <Field label="Forecast groups" value={summary.forecast_group_count} />
            <Field label="Series built" value={summary.series_count} />
            <Field label="Selected models" value={summary.selected_models.join(', ') || '—'} />
            <Field label="Configured fallback" value={summary.fallback_model} />
            <Field
              label="Execution duration"
              value={formatDuration(summary.execution_duration_seconds)}
            />
            <Field label="Models registered" value={summary.models_registered} />
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <StatusCounts title="Models trained (by status)" counts={summary.training_summary} />
            <StatusCounts title="Models evaluated (by status)" counts={summary.evaluation_summary} />
          </div>

          <div>
            <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Winner selection by group
            </p>
            {summary.winner_selection.length === 0 ? (
              <p className="text-xs text-slate-400">No groups recorded yet.</p>
            ) : (
              <div className="overflow-x-auto rounded-lg border border-slate-100 dark:border-slate-800">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-100 bg-slate-50/60 dark:border-slate-800 dark:bg-slate-800/40">
                      <th className="px-3 py-2 font-semibold text-slate-500">Group</th>
                      <th className="px-3 py-2 font-semibold text-slate-500">Winner</th>
                      <th className="px-3 py-2 font-semibold text-slate-500">Status</th>
                      <th className="px-3 py-2 font-semibold text-slate-500">Fallback</th>
                    </tr>
                  </thead>
                  <tbody>
                    {summary.winner_selection.map((row) => (
                      <tr key={row.group_id} className="border-b border-slate-50 last:border-0 dark:border-slate-800/60">
                        <td className="px-3 py-2 text-slate-600 dark:text-slate-300">{row.group_id}</td>
                        <td className="px-3 py-2 text-slate-600 dark:text-slate-300">{row.winner_model ?? '—'}</td>
                        <td className="px-3 py-2 text-slate-600 dark:text-slate-300">{row.selection_status}</td>
                        <td className="px-3 py-2 text-slate-600 dark:text-slate-300">{row.fallback_used ? 'Yes' : 'No'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div>
            <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">MLflow</p>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Field label="Run ID" value={<span className="font-mono text-xs">{summary.mlflow_run_id}</span>} />
              <Field label="Experiment" value={summary.mlflow_experiment} />
              <Field label="Tracking URI" value={<span className="break-all">{summary.mlflow_tracking_uri}</span>} />
            </div>
          </div>

          <div>
            <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Artifact locations (within the MLflow run)
            </p>
            <ul className="space-y-1">
              {summary.artifact_locations.map((artifact) => (
                <li key={artifact.path} className="text-xs text-slate-500 dark:text-slate-400">
                  <span className="text-slate-600 dark:text-slate-300">{artifact.name}</span>
                  {' — '}
                  <span className="font-mono">{artifact.path}</span>
                </li>
              ))}
            </ul>
          </div>

          {summary.stages.length > 0 && (
            <div>
              <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">Stage trail</p>
              <ul className="space-y-1">
                {summary.stages.map((stage) => (
                  <li key={stage.name} className="text-xs text-slate-500 dark:text-slate-400">
                    <span className="text-slate-600 dark:text-slate-300">{stage.name}</span> — {stage.status}
                    {stage.started_at && stage.completed_at && (
                      <> ({formatISTTime(stage.started_at)}, took {formatDuration(secondsBetween(stage.started_at, stage.completed_at))})</>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div>
            <button
              type="button"
              onClick={() => setShowRawJson((s) => !s)}
              className="text-xs font-medium text-brand-600 hover:text-brand-700 dark:text-brand-400"
            >
              {showRawJson ? 'Hide' : 'Show'} raw result JSON
            </button>
            {showRawJson && (
              <pre className="mt-2 max-h-96 overflow-auto rounded-lg bg-slate-900 p-3 text-[11px] leading-relaxed text-slate-200">
                {JSON.stringify(summary.raw_result, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}
    </ExpandablePanel>
  )
}
