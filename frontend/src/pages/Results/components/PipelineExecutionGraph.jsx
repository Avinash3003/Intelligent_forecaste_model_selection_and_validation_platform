import { useCallback, useEffect, useRef, useState } from 'react'
import { CheckCircle2, Circle, Loader2, XCircle } from 'lucide-react'
import SectionContainer from '../../../components/layout/SectionContainer'
import OpenInDatabricksButton from '../../../components/ui/OpenInDatabricksButton'
import Loader from '../../../components/ui/Loader'
import RayParallelTimeline from './RayParallelTimeline'
import { fetchDeployment, isTerminalStatus } from '../../../services'
import { formatDuration, formatISTTime, secondsBetween } from '../../../utils/formatDateTime'
import { cn } from '../../../utils/cn'

// The application's own execution graph — the seven display phases the
// backend already reports (services/pipeline_stages.py PIPELINE_PHASES),
// drawn as a compact stepper, with the Ray parallel-execution view as its
// own full-width section below.
//
// A view over the same seven phases Databricks itself now runs as real
// tasks (see backend/app/orchestration/databricks_runner.py's TASK_KEYS) —
// not a substitute for opening the run in Databricks, which is what shows
// the actual multi-task DAG with real per-task timing and logs (see the
// "Open in Databricks" link elsewhere on this page). This reads the same
// /deployments/{run_id} payload the Deployments page already renders, so
// the two can never disagree, and it is strictly a view — nothing here can
// influence execution.
//
// The stepper used to be seven expanded cards (icon, label, status,
// duration, timestamp — four stacked lines each) with the Ray view nested
// inside whichever one card matched the Train Models phase, squeezed into that one
// card's ~150px column. Ray parallel execution is the thing actually worth
// studying on this page; the phase list is a status summary. The stepper
// below carries the same information in one line per phase, and the
// reclaimed width goes to Ray having the section's full breadth to render
// in, not a seventh of it.
const REFRESH_INTERVAL_MS = 3000

// Same status vocabulary and colours as components/common/PipelineTimeline,
// so a phase reads identically wherever it appears in the product.
const STATUS_STYLES = {
  Completed: {
    icon: CheckCircle2,
    dot: 'border-emerald-500 bg-emerald-50 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400',
    connector: 'bg-emerald-400 dark:bg-emerald-600',
  },
  Running: {
    icon: Loader2,
    dot: 'border-blue-500 bg-blue-50 text-blue-600 ring-2 ring-blue-100 dark:bg-blue-900/30 dark:text-blue-400 dark:ring-blue-900/40',
    connector: 'bg-slate-200 dark:bg-slate-700',
    spin: true,
  },
  Failed: {
    icon: XCircle,
    dot: 'border-rose-500 bg-rose-50 text-rose-600 dark:bg-rose-900/30 dark:text-rose-400',
    connector: 'bg-slate-200 dark:bg-slate-700',
  },
  Pending: {
    icon: Circle,
    dot: 'border-slate-200 bg-white text-slate-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-600',
    connector: 'bg-slate-200 dark:bg-slate-700',
  },
}

function PhaseStep({ stage, prevStatus, isFirst, isLast }) {
  const style = STATUS_STYLES[stage.status] || STATUS_STYLES.Pending
  const prevStyle = STATUS_STYLES[prevStatus] || STATUS_STYLES.Pending
  const Icon = style.icon
  // The engine's own measured time when it has one (real Ray-parallel
  // work), falling back to driver wall clock only when it does not.
  const elapsed = stage.durationSeconds ?? secondsBetween(stage.startedAt, stage.completedAt)

  return (
    <div className="flex min-w-[104px] flex-1 flex-col items-center text-center">
      <div className="flex w-full items-center">
        {/* The line between two dots is drawn in two flush halves — no gap
            between adjacent PhaseStep boxes (see the parent's gap-0) — so
            each half must carry a real connector color or the halves
            visibly fail to meet in the middle. The left half takes the
            previous step's color (what led into this dot), the right half
            this step's own (what leads out of it), exactly mirroring the
            trailing half drawn by the step before this one. */}
        <span className={cn('h-px flex-1', prevStyle.connector, isFirst && 'invisible')} />
        <div
          className={cn(
            'flex h-7 w-7 shrink-0 items-center justify-center rounded-full border transition-colors',
            style.dot
          )}
          title={`${stage.label} — ${stage.status}${elapsed != null ? ` (${formatDuration(elapsed)})` : ''}${stage.detail ? `: ${stage.detail}` : ''}`}
        >
          <Icon size={13} className={cn(style.spin && 'animate-spin')} />
        </div>
        <span className={cn('h-px flex-1', style.connector, isLast && 'invisible')} />
      </div>
      <p className="mt-1.5 w-full truncate text-[11px] font-medium text-slate-700 dark:text-slate-200">
        {stage.label}
      </p>
      <p className="text-[10px] tabular-nums text-slate-400">
        {elapsed != null ? formatDuration(elapsed) : stage.status}
      </p>
      {stage.parallelTasks && stage.parallelTasks.total > 0 && (
        <p className="text-[9px] tabular-nums text-brand-500 dark:text-brand-400">
          {stage.parallelTasks.completed}/{stage.parallelTasks.total} parallel
        </p>
      )}
    </div>
  )
}

export default function PipelineExecutionGraph({ runId }) {
  const [run, setRun] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const cancelledRef = useRef(false)
  const timeoutRef = useRef(null)

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

  // The Databricks link belongs beside the phases it summarises, not in a
  // collapsed card at the foot of the page: this section is a *view* of the
  // DAG, and the real one — per-task timing, logs, a run still executing —
  // is one click away.
  return (
    <SectionContainer
      title="Execution flow"
      subtitle={`${run.stages.filter((s) => s.status === 'Completed').length} of ${run.stages.length} phases complete${
        run.duration ? ` · ${run.duration} total` : ''
      }`}
      action={<OpenInDatabricksButton url={run.databricksRunUrl} size="sm" />}
    >
      <div className="overflow-x-auto">
        {/* No gap between steps — see PhaseStep's comment on why the
            connector line depends on adjacent boxes sitting flush. */}
        <div className="flex min-w-max items-start pb-1 sm:min-w-0">
          {run.stages.map((stage, index) => (
            <PhaseStep
              key={stage.label}
              stage={stage}
              prevStatus={index > 0 ? run.stages[index - 1].status : undefined}
              isFirst={index === 0}
              isLast={index === run.stages.length - 1}
            />
          ))}
        </div>
      </div>

      {/* Full section width, not one phase's sliver of it — see the
          module comment above for why this moved out of the stepper. One
          timeline per Ray-parallel stage — each is that stage's own real
          fan-out, never one run-wide blob standing in for all of them. */}
      {run.stages
        .filter((stage) => stage.parallelTasks)
        .map((stage) => (
          <RayParallelTimeline key={stage.label} stageLabel={stage.label} parallelTasks={stage.parallelTasks} />
        ))}
    </SectionContainer>
  )
}
