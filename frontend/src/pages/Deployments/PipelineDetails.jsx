import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { AlertCircle, AlertTriangle, ArrowLeft, LineChart, SearchX } from 'lucide-react'
import PageContainer from '../../components/common/PageContainer'
import SectionContainer from '../../components/layout/SectionContainer'
import Badge from '../../components/ui/Badge'
import Button from '../../components/ui/Button'
import EmptyState from '../../components/ui/EmptyState'
import Loader from '../../components/ui/Loader'
import PipelineTimeline from '../../components/common/PipelineTimeline'
import RunSummaryBar from './components/RunSummaryBar'
import { fetchDeployment, isTerminalStatus } from '../../services'

const REFRESH_INTERVAL_MS = 3000

export default function PipelineDetails() {
  const { runId } = useParams()
  const navigate = useNavigate()

  const [run, setRun] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [notFound, setNotFound] = useState(false)

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

        {run.status === 'Completed' && (
          <Button variant="secondary" icon={LineChart} onClick={() => navigate(`/results?run=${run.id}`)}>
            View Results
          </Button>
        )}
      </div>

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
            : 'Stage detail is reported once the run finishes'
        }
      >
        {run.stages.length > 0 ? (
          <PipelineTimeline stages={run.stages} />
        ) : (
          <p className="text-sm text-slate-400">
            No stage trail is available for this run.
          </p>
        )}
      </SectionContainer>
    </PageContainer>
  )
}
