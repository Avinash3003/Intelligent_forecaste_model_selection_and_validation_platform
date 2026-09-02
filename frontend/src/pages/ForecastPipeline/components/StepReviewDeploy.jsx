import { AlertCircle, CheckCircle2, LineChart, Rocket } from 'lucide-react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import SectionContainer from '../../../components/layout/SectionContainer'
import Badge from '../../../components/ui/Badge'
import Button from '../../../components/ui/Button'
import OpenInDatabricksButton from '../../../components/ui/OpenInDatabricksButton'
import EstimateCard from './EstimateCard'
import { formatMonths } from '../../../utils/formatMonths'
import { forecastModels } from '../../../data/appConfig'

// One configuration value. A compact grid cell rather than a full-width
// row: the whole configuration then fits on one screen without scrolling,
// which is what makes this a review step instead of a reading exercise.
function Field({ label, value }) {
  return (
    <div className="min-w-0">
      <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400">{label}</p>
      <div className="mt-0.5 truncate text-sm font-medium text-slate-700 dark:text-slate-200">{value}</div>
    </div>
  )
}

// One line describing the compute this run will use.
function describeCompute(compute, existingCompute) {
  if (compute.mode === 'existing_compute') {
    return {
      mode: 'Existing compute',
      machine: existingCompute?.nodeTypeId ?? '—',
      runtime: existingCompute?.runtime ?? '—',
      workers: existingCompute?.singleNode ? 'Single node' : `${existingCompute?.numWorkers ?? 0} worker(s)`,
    }
  }
  return {
    mode: 'New job compute',
    machine: compute.nodeTypeId || '—',
    runtime: compute.runtimeKey || '—',
    workers: compute.autoscale
      ? `Autoscale ${compute.minWorkers}–${compute.maxWorkers}`
      : compute.numWorkers === 0
        ? 'Single node'
        : `${compute.numWorkers} worker(s)`,
  }
}

export default function StepReviewDeploy({
  file,
  mapping,
  config,
  compute,
  existingCompute,
  deployed,
  deployError,
  runId,
  databricksRunUrl,
  estimate,
  estimateLoading,
  estimateError,
}) {
  const navigate = useNavigate()

  const computeSummary = compute ? describeCompute(compute, existingCompute) : null
  const selectedModelNames = forecastModels
    .filter((m) => config.selectedModels.includes(m.id))
    .map((m) => m.name)

  if (deployed) {
    return (
      <SectionContainer>
        <motion.div
          initial={{ opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
          className="flex flex-col items-center gap-3 py-14 text-center"
        >
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald-50 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400">
            <CheckCircle2 size={30} strokeWidth={1.75} />
          </div>
          <h3 className="text-base font-semibold text-slate-800 dark:text-slate-100">
            Forecast running
          </h3>
          <p className="max-w-sm text-sm text-slate-400">
            Run {runId} is executing. Results appear here on their own, or watch the
            stages run live in Databricks.
          </p>
          <div className="mt-2 flex flex-wrap items-center justify-center gap-2.5">
            <Button icon={LineChart} onClick={() => navigate(`/results?run=${runId}`)}>
              View Results
            </Button>
            <Button variant="secondary" icon={Rocket} onClick={() => navigate(`/deployments/${runId}`)}>
              Track Progress
            </Button>
            {/* Offered at submission, not at completion: watching a run
                execute is only possible while it is still executing. */}
            <OpenInDatabricksButton url={databricksRunUrl} />
          </div>
        </motion.div>
      </SectionContainer>
    )
  }

  return (
    <div className="space-y-4">
      {deployError && (
        <div className="flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <span>{deployError}</span>
        </div>
      )}

      {/* The estimate leads: it is the one thing on this screen a user has
          not already seen in an earlier step. */}
      {/* Runtime/cost estimate temporarily hidden. It made entering this
          step slow — the call sweeps run history and re-reads summary
          artifacts — and on larger datasets it timed out rather than
          answering. The run itself never depended on it (it was advisory
          only), so nothing else changes by not showing it. Restore by
          uncommenting this and the `loadEstimate()` call in
          ForecastPipeline.jsx.
      <EstimateCard estimate={estimate} loading={estimateLoading} error={estimateError} />
      */}

      <SectionContainer title="Run configuration">
        <div className="grid grid-cols-2 gap-x-6 gap-y-4 py-1 lg:grid-cols-3">
          <Field label="Dataset" value={file?.name ?? '—'} />
          <Field label="Date" value={mapping.dateColumn || '—'} />
          <Field label="Target" value={mapping.targetColumn || '—'} />
          <Field label="Keys" value={mapping.keyColumns.length ? mapping.keyColumns.join(', ') : '—'} />
          <Field
            label="Features"
            value={mapping.featureColumns.length ? mapping.featureColumns.join(', ') : 'None'}
          />
          <Field label="Horizon" value={formatMonths(config.horizon)} />
          <Field
            label="Fallback"
            value={forecastModels.find((m) => m.id === config.fallbackModel)?.name ?? '—'}
          />
          <div className="col-span-2 min-w-0 lg:col-span-3">
            <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400">Models</p>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {selectedModelNames.map((name) => (
                <Badge key={name} status="neutral">
                  {name}
                </Badge>
              ))}
            </div>
          </div>
        </div>
      </SectionContainer>

      {compute && (
        <SectionContainer title="Compute">
          <div className="grid grid-cols-2 gap-x-6 gap-y-4 py-1 lg:grid-cols-4">
            <Field label="Mode" value={computeSummary.mode} />
            <Field label="Machine type" value={computeSummary.machine} />
            <Field label="Runtime" value={computeSummary.runtime} />
            <Field label="Workers" value={computeSummary.workers} />
          </div>
        </SectionContainer>
      )}
    </div>
  )
}
