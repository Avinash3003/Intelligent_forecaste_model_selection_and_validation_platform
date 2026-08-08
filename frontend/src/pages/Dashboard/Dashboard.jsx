import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle, Eye, Plus, Rocket } from 'lucide-react'
import PageContainer from '../../components/common/PageContainer'
import StatCard from '../../components/cards/StatCard'
import DataTable from '../../components/tables/DataTable'
import Badge from '../../components/ui/Badge'
import Button from '../../components/ui/Button'
import Loader from '../../components/ui/Loader'
import EmptyState from '../../components/ui/EmptyState'
import SectionContainer from '../../components/layout/SectionContainer'
import { fetchDeployments, isTerminalStatus } from '../../services'

const RECENT_RUN_LIMIT = 5

// Derives the headline figures from the real run history — no counter is
// tracked separately from the runs themselves.
function buildStats(runs) {
  const completed = runs.filter((run) => run.status === 'Completed').length
  const failed = runs.filter((run) => run.status === 'Failed').length
  const active = runs.filter((run) => !isTerminalStatus(run.status)).length
  const latest = runs[0]

  return [
    {
      id: 'total-runs',
      label: 'Total Runs',
      value: String(runs.length),
      delta: runs.length === 0 ? 'No runs submitted yet' : `${completed} completed`,
      trend: 'neutral',
      icon: 'Play',
    },
    {
      id: 'active-runs',
      label: 'Active Runs',
      value: String(active),
      delta: active > 0 ? 'Currently executing' : 'Nothing running',
      trend: active > 0 ? 'up' : 'neutral',
      icon: 'Workflow',
    },
    {
      id: 'failed-runs',
      label: 'Failed Runs',
      value: String(failed),
      delta: failed > 0 ? 'Needs attention' : 'No failures',
      trend: failed > 0 ? 'down' : 'neutral',
      icon: 'TriangleAlert',
    },
    {
      id: 'last-run',
      label: 'Last Run',
      value: latest ? latest.status : '—',
      delta: latest ? latest.id : 'No runs submitted yet',
      trend: 'neutral',
      icon: 'Rocket',
    },
  ]
}

export default function Dashboard() {
  const navigate = useNavigate()

  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    fetchDeployments()
      .then((data) => !cancelled && setRuns(data))
      .catch((err) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [])

  const stats = useMemo(() => buildStats(runs), [runs])
  const recentRuns = useMemo(() => runs.slice(0, RECENT_RUN_LIMIT), [runs])

  const columns = useMemo(
    () => [
      {
        key: 'id',
        header: 'Run',
        render: (r) => (
          <div>
            <p className="font-medium text-slate-700 dark:text-slate-200">{r.displayName}</p>
            <p className="font-mono text-xs text-slate-400">{r.id}</p>
          </div>
        ),
      },
      { key: 'dataset', header: 'Dataset' },
      { key: 'status', header: 'Status', render: (r) => <Badge status={r.status}>{r.status}</Badge> },
      { key: 'startTime', header: 'Started (IST)' },
      { key: 'duration', header: 'Duration' },
      {
        key: 'action',
        header: 'Action',
        render: (r) => (
          <button
            onClick={() => navigate(`/deployments/${r.id}`)}
            className="flex items-center gap-1.5 text-sm font-medium text-brand-600 hover:text-brand-700 dark:text-brand-400"
          >
            <Eye size={14} /> View
          </button>
        ),
      },
    ],
    [navigate]
  )

  return (
    <PageContainer>
      <div className="flex flex-col gap-1 mb-8 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-800 dark:text-slate-100">
            Forecast IQ
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Here's what's happening across your forecasting pipelines today.
          </p>
        </div>
        <Button icon={Plus} onClick={() => navigate('/forecast-pipeline')}>
          New Forecast Pipeline
        </Button>
      </div>

      {error && (
        <div className="mb-6 flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-medium">Could not load run history</p>
            <p className="mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {loading ? (
        <SectionContainer>
          <Loader label="Loading run history…" />
        </SectionContainer>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4 mb-8">
            {stats.map((stat) => (
              <StatCard key={stat.id} {...stat} />
            ))}
          </div>

          <SectionContainer
            title="Recent Runs"
            subtitle="Latest pipeline executions across all datasets"
          >
            {recentRuns.length > 0 ? (
              <DataTable columns={columns} data={recentRuns} />
            ) : (
              <EmptyState
                icon={Rocket}
                title="No runs yet"
                description="Deploy a forecasting run from the Forecast Pipeline page to see it here."
              />
            )}
          </SectionContainer>
        </>
      )}
    </PageContainer>
  )
}
