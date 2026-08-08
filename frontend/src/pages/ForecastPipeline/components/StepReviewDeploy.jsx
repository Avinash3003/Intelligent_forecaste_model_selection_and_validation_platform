import { AlertCircle, CheckCircle2, LineChart, Rocket } from 'lucide-react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import SectionContainer from '../../../components/layout/SectionContainer'
import Badge from '../../../components/ui/Badge'
import Button from '../../../components/ui/Button'
import { formatMonths } from '../../../utils/formatMonths'
import { forecastModels } from '../../../data/appConfig'

function SummaryRow({ label, value }) {
  return (
    <div className="flex items-center justify-between border-b border-slate-50 py-3 last:border-0 dark:border-slate-800/60">
      <span className="text-sm text-slate-400">{label}</span>
      <span className="text-sm font-medium text-slate-700 dark:text-slate-200">{value}</span>
    </div>
  )
}

export default function StepReviewDeploy({ file, mapping, config, deployed, deployError, runId }) {
  const navigate = useNavigate()

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
            Run submitted successfully
          </h3>
          <p className="max-w-sm text-sm text-slate-400">
            Job {runId} is executing. Open Results to watch it finish — the page updates on its own.
          </p>
          <div className="mt-2 flex flex-wrap items-center justify-center gap-2.5">
            <Button icon={LineChart} onClick={() => navigate(`/results?run=${runId}`)}>
              View Results
            </Button>
            <Button variant="secondary" icon={Rocket} onClick={() => navigate(`/deployments/${runId}`)}>
              Track Progress
            </Button>
          </div>
        </motion.div>
      </SectionContainer>
    )
  }

  return (
    <div className="space-y-5">
      {deployError && (
        <div className="flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <span>{deployError}</span>
        </div>
      )}

      <SectionContainer title="Run configuration" subtitle="Confirm before deploying to Azure Databricks">
        <SummaryRow label="Dataset" value={file?.name ?? '—'} />
        <SummaryRow label="Date Column" value={mapping.dateColumn || '—'} />
        <SummaryRow label="Target Column" value={mapping.targetColumn || '—'} />
        <SummaryRow
          label="Key Column(s)"
          value={mapping.keyColumns.length ? mapping.keyColumns.join(', ') : '—'}
        />
        <SummaryRow
          label="Feature Column(s)"
          value={mapping.featureColumns.length ? mapping.featureColumns.join(', ') : 'None'}
        />
        <SummaryRow
          label="Forecast Horizon"
          value={`${formatMonths(config.horizon)} (${config.horizon} months)`}
        />
        <SummaryRow
          label="Default Fallback Model"
          value={
            forecastModels.find((m) => m.id === config.fallbackModel)?.name ?? '—'
          }
        />
        <SummaryRow
          label="Selected Models"
          value={
            <span className="flex flex-wrap justify-end gap-1.5">
              {selectedModelNames.map((name) => (
                <Badge key={name} status="neutral">
                  {name}
                </Badge>
              ))}
            </span>
          }
        />
      </SectionContainer>
    </div>
  )
}
