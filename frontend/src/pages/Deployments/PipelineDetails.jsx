import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { AlertCircle, AlertTriangle, ArrowLeft, LineChart, SearchX, XCircle } from 'lucide-react'
import PageContainer from '../../components/common/PageContainer'
import SectionContainer from '../../components/layout/SectionContainer'
import Badge from '../../components/ui/Badge'
import Button from '../../components/ui/Button'
import ConfirmDialog from '../../components/ui/ConfirmDialog'
import EmptyState from '../../components/ui/EmptyState'
import Loader from '../../components/ui/Loader'
import PipelineTimeline from '../../components/common/PipelineTimeline'
import OpenInDatabricksButton from '../../components/ui/OpenInDatabricksButton'
import RunSummaryBar from './components/RunSummaryBar'
import { cancelDeployment, fetchDeployment, isTerminalStatus } from '../../services'

const REFRESH_INTERVAL_MS = 3000
const CANCELLABLE_STATUSES = ['Pending', 'Running']

export default function PipelineDetails() {
  const { runId } = useParams()
  const navigate = useNavigate()

  const [run, setRun] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [notFound, setNotFound] = useState(false)

  const [showCancelConfirm, setShowCancelConfirm] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [cancelCleanupWarning, setCancelCleanupWarning] = useState(null)

  const cancelledRef = useRef(false)
  const timeoutRef = useRef(null)

  const load = useCallback(async () => {
    try {
      const data = await fetchDeployment(runId)
      if (cancelledRef.current) return null
      setRun(data)
      setError(null)
      setNotFound(false)
      return data
    } catch (err) {
      if (cancelledRef.current) return null
      if (err.status === 404) setNotFound(true)
      else setError(err.message)
      return null
    } finally {
      if (!cancelledRef.current) setLoading(false)
    }
  }, [runId])

  // Keeps refreshing only while the run is still active; a finished run is
  // fetched once and then left alone.
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

  const handleConfirmCancel = async () => {
    setCancelling(true)
    try {
      const response = await cancelDeployment(runId)
      // Immediate feedback rather than waiting for the next poll cycle —
      // the poller will also pick this run up as terminal on its own, but
      // there is no reason to make the user wait 3 seconds to see it.
      setRun((current) => (current ? { ...current, status: 'Cancelled' } : current))
      if (response.cleanup_errors?.length) {
        setCancelCleanupWarning(response.cleanup_errors)
      }
      await load()
    } catch (err) {
      setError(err.message)
    } finally {
      setCancelling(false)
      setShowCancelConfirm(false)
    }
  }

  const backButton = (
    <Button variant="secondary" icon={ArrowLeft} onClick={() => navigate('/deployments')}>
      Back to Deployments
    </Button>
  )

  if (loading) {
    return (
      <PageContainer>
        <SectionContainer>
          <Loader label="Loading run details…" />
        </SectionContainer>
      </PageContainer>
    )
  }

  if (notFound) {
    return (
      <PageContainer>
        <SectionContainer>
          <EmptyState
            icon={SearchX}
            title="Forecast run not found"
            description={`No pipeline execution matches "${runId}". Run history is held in memory, so runs submitted before the backend last restarted are no longer listed.`}
            action={backButton}
          />
        </SectionContainer>
      </PageContainer>
    )
  }

  if (error) {
    return (
      <PageContainer>
        <SectionContainer>
          <EmptyState
            icon={AlertCircle}
            title="Could not load this run"
            description={error}
            action={backButton}
          />
        </SectionContainer>
      </PageContainer>
    )
  }

  return (
    <PageContainer>
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <button
            onClick={() => navigate('/deployments')}
            className="mb-2 flex items-center gap-1.5 text-sm font-medium text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
          >
            <ArrowLeft size={14} /> Back to Deployments
          </button>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-slate-800 dark:text-slate-100">
              {run.displayName}
            </h1>
            <Badge status={run.status}>{run.status}</Badge>
          </div>
          <p className="mt-1 text-sm text-slate-400">
            Pipeline execution details for {run.dataset} ·{' '}
            <span className="font-mono text-xs">{run.id}</span>
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          {/* Prominent here on purpose: this is the page a user watches a
              run on, so "open the actual Databricks run" is the action they
              reach for. Renders only when the backend supplied a URL. */}
          <OpenInDatabricksButton url={run.databricksRunUrl} />
          {CANCELLABLE_STATUSES.includes(run.status) && (
            <Button variant="danger" icon={XCircle} onClick={() => setShowCancelConfirm(true)}>
              Cancel Run
            </Button>
          )}
          {run.status === 'Completed' && (
            <Button variant="secondary" icon={LineChart} onClick={() => navigate(`/results?run=${run.id}`)}>
              View Results
            </Button>
          )}
        </div>
      </div>

      {cancelCleanupWarning && (
        <div className="mb-6 flex items-start gap-2.5 rounded-xl border border-amber-200 bg-amber-50/70 px-4 py-3 text-sm text-amber-700 dark:border-amber-800 dark:bg-amber-900/20 dark:text-amber-300">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-medium">Run cancelled, but some data could not be fully cleaned up</p>
            <ul className="mt-1 list-inside list-disc space-y-0.5">
              {cancelCleanupWarning.map((issue) => (
                <li key={issue}>{issue}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {run.error && (
        <div className="mb-6 flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-medium">Forecast execution failed</p>
            <p className="mt-0.5 whitespace-pre-wrap break-words">{run.error}</p>
          </div>
        </div>
      )}

      <div className="mb-6">
        <RunSummaryBar run={run} />
      </div>

      <SectionContainer
        title="Execution workflow"
        subtitle={
          run.stages.length > 0
            ? 'Per-stage execution trail reported by the forecast engine'
            : run.compute
              ? 'Waiting on the Databricks compute this run was submitted to'
              : 'Stage detail is reported once the run finishes'
        }
      >
        {run.stages.length > 0 || run.compute ? (
          <PipelineTimeline
            stages={run.stages}
            progress={run.progress}
            runStatus={run.status}
            compute={run.compute}
            error={run.error}
          />
        ) : (
          <p className="text-sm text-slate-400">
            No stage trail is available for this run.
          </p>
        )}
      </SectionContainer>

      <ConfirmDialog
        open={showCancelConfirm}
        title="Cancel this run?"
        description="Are you sure you want to cancel this run? All generated run data will be deleted."
        confirmLabel="Cancel Run"
        cancelLabel="Keep it"
        loading={cancelling}
        onConfirm={handleConfirmCancel}
        onCancel={() => setShowCancelConfirm(false)}
      />
    </PageContainer>
  )
}
