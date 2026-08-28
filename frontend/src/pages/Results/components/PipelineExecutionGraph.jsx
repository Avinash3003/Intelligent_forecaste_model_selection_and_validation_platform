import { useCallback, useEffect, useRef, useState } from 'react'
import { CheckCircle2, Circle, Loader2, XCircle } from 'lucide-react'
import SectionContainer from '../../../components/layout/SectionContainer'
import Loader from '../../../components/ui/Loader'
import RayParallelTimeline from './RayParallelTimeline'
import { fetchDebugSummary, fetchDeployment, isTerminalStatus } from '../../../services'
import { formatDuration, formatISTTime, secondsBetween } from '../../../utils/formatDateTime'
import { cn } from '../../../utils/cn'

// The application's own execution graph — the seven display phases the
// backend already reports (services/pipeline_stages.py PIPELINE_PHASES),
// drawn as a flow.
//
// Deliberately NOT a Databricks DAG: this pipeline is one Databricks task,
// and these phases are the engine's own internal stages. Presenting them as
// Databricks tasks would misrepresent the architecture. This reads the same
// /deployments/{run_id} payload the Deployments page already renders, so
// the two can never disagree, and it is strictly a view — nothing here can
// influence execution.
const REFRESH_INTERVAL_MS = 3000

// Same status vocabulary and colours as components/common/PipelineTimeline,
// so a phase looks identical wherever it appears in the product.
const STATUS_STYLES = {
  Completed: {
    icon: CheckCircle2,
    node: 'border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20',
    icon_: 'text-emerald-600 dark:text-emerald-400',
  },
  Running: {
    icon: Loader2,
    node: 'border-blue-500 bg-blue-50 ring-2 ring-blue-100 dark:bg-blue-900/20 dark:ring-blue-900/40',
    icon_: 'text-blue-600 dark:text-blue-400',
    spin: true,
  },
  Failed: {
    icon: XCircle,
    node: 'border-rose-500 bg-rose-50 dark:bg-rose-900/20',
    icon_: 'text-rose-600 dark:text-rose-400',
  },
  Pending: {
    icon: Circle,
    node: 'border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900',
    icon_: 'text-slate-300 dark:text-slate-600',
  },
}

// The phase the Ray key-level view belongs under. Matches the backend's own
// phase label exactly (PIPELINE_PHASES); a rename there without one here
// simply means the panel stops rendering, never a crash.
const RAY_PHASE_LABEL = 'Train Models'

function PhaseNode({ stage, keyExecution }) {
  const style = STATUS_STYLES[stage.status] || STATUS_STYLES.Pending
  const Icon = style.icon
  const elapsed = secondsBetween(stage.startedAt, stage.completedAt)

  return (
    <div className="min-w-0 flex-1">
      <div className={cn('rounded-xl border p-3 transition-colors', style.node)}>
        <div className="flex items-start gap-2">
          <Icon
            size={15}
            className={cn('mt-0.5 shrink-0', style.icon_, style.spin && 'animate-spin')}
          />
          <div className="min-w-0">
            <p className="truncate text-xs font-semibold text-slate-800 dark:text-slate-100">
              {stage.label}
            </p>
            <p className="mt-0.5 text-[11px] text-slate-500 dark:text-slate-400">{stage.status}</p>
            {elapsed != null && (
              <p className="mt-0.5 text-[11px] tabular-nums text-slate-400">
                {formatDuration(elapsed)}
              </p>
            )}
            {stage.startedAt && (
              <p className="mt-0.5 text-[10px] text-slate-400">{formatISTTime(stage.startedAt)}</p>
            )}
          </div>
        </div>
      </div>
      {stage.label === RAY_PHASE_LABEL && <RayParallelTimeline keyExecution={keyExecution} />}
    </div>
  )
}

export default function PipelineExecutionGraph({ runId }) {
  const [run, setRun] = useState(null)
  const [keyExecution, setKeyExecution] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const cancelledRef = useRef(false)
  const timeoutRef = useRef(null)

  // Ray telemetry comes from the debug endpoint, which already serves this
  // page and already accepts an unfinished run. Fetched once rather than on
  // the poll loop: key_execution is written when Train Models finishes and
  // does not change afterwards, so re-reading it every 3s would be a cold
  // artifact read per tick for a value that cannot have moved.
  useEffect(() => {
    let cancelled = false
    fetchDebugSummary(runId)
      .then((summary) => {
        if (!cancelled) setKeyExecution(summary?.key_execution ?? null)
      })
      // A missing debug summary only means the parallel view stays hidden;
      // the phase graph above it is unaffected.
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [runId])

  const load = useCallback(async () => {
    try {
      const data = await fetchDeployment(runId)
      if (cancelledRef.current) return null
      setRun(data)
      setError(null)
      return data
    } catch (err) {
      if (!cancelledRef.current) setError(err.message)
      return null
    } finally {
      if (!cancelledRef.current) setLoading(false)
    }
  }, [runId])

  // Same polling shape PipelineDetails already uses: keep refreshing only
  // while the run is active, then leave a finished run alone.
  useEffect(() => {
    cancelledRef.current = false
    setLoading(true)

    async function cycle() {
      const data = await load()
      if (cancelledRef.current || !data) return
      if (!isTerminalStatus(data.status)) {
        timeoutRef.current = setTimeout(cycle, REFRESH_INTERVAL_MS)
      }
    }
    cycle()

    return () => {
      cancelledRef.current = true
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
    }
  }, [load])

  if (loading) return <Loader label="Loading execution graph…" />
  if (error) return null
  if (!run?.stages?.length) return null

  return (
    <SectionContainer
      title="Execution flow"
      subtitle={`${run.stages.filter((s) => s.status === 'Completed').length} of ${run.stages.length} phases complete${
        run.duration ? ` · ${run.duration} total` : ''
      }`}
    >
      <div className="overflow-x-auto">
        <div className="flex min-w-max items-start gap-2 pb-2">
          {run.stages.map((stage, index) => (
            <div key={stage.label} className="flex min-w-[150px] items-start gap-2">
              <PhaseNode stage={stage} keyExecution={keyExecution} />
              {index < run.stages.length - 1 && (
                <span className="mt-6 shrink-0 text-slate-300 dark:text-slate-600">→</span>
              )}
            </div>
          ))}
        </div>
      </div>
    </SectionContainer>
  )
}
