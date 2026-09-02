import { useMemo, useState } from 'react'
import { AlertCircle, RefreshCw, Rocket } from 'lucide-react'
import PageContainer from '../../components/common/PageContainer'
import SectionContainer from '../../components/layout/SectionContainer'
import Pagination from '../../components/common/Pagination'
import EmptyState from '../../components/ui/EmptyState'
import Loader from '../../components/ui/Loader'
import Button from '../../components/ui/Button'
import DeploymentsFilterBar from './components/DeploymentsFilterBar'
import DeploymentsTable from './components/DeploymentsTable'
import { useRunHistory } from '../../hooks/useRunHistory'

const PAGE_SIZE = 5

export default function Deployments() {
  // The fetch/retry policy this page pioneered now lives in the hook, so
  // Dashboard and Results get it too — they each fetched once and showed
  // "no runs" through the window where history is still warming.
  const { runs, loading, error, reload } = useRunHistory()

  const [search, setSearch] = useState('')
  const [status, setStatus] = useState('All')
  const [page, setPage] = useState(1)

  const filteredRuns = useMemo(() => {
    const term = search.trim().toLowerCase()
    return runs.filter((run) => {
      const matchesSearch =
        !term ||
        run.id.toLowerCase().includes(term) ||
        (run.dataset ?? '').toLowerCase().includes(term)
      const matchesStatus = status === 'All' || run.status === status
      return matchesSearch && matchesStatus
    })
  }, [runs, search, status])

  const totalPages = Math.max(1, Math.ceil(filteredRuns.length / PAGE_SIZE))
  const pagedRuns = filteredRuns.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  const handleSearchChange = (value) => {
    setSearch(value)
    setPage(1)
  }

  const handleStatusChange = (value) => {
    setStatus(value)
    setPage(1)
  }

  const hasFilters = Boolean(search.trim()) || status !== 'All'

  return (
    <PageContainer>
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-800 dark:text-slate-100">
            Deployments
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Monitor run status across every pipeline execution submitted to the Pipeline Executor.
          </p>
        </div>
        <Button variant="secondary" icon={RefreshCw} onClick={() => reload({ showSpinner: true })}>
          Refresh
        </Button>
      </div>

      <SectionContainer>
        {error && (
          <div className="mb-4 flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <div>
              <p className="font-medium">Could not load deployments</p>
              <p className="mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {loading ? (
          <Loader label="Loading deployments…" />
        ) : (
          <>
            <DeploymentsFilterBar
              search={search}
              onSearchChange={handleSearchChange}
              status={status}
              onStatusChange={handleStatusChange}
            />

            {pagedRuns.length > 0 ? (
              <>
                <DeploymentsTable runs={pagedRuns} />
                <div className="mt-4">
                  <Pagination
                    page={page}
                    totalPages={totalPages}
                    onPageChange={setPage}
                    totalItems={filteredRuns.length}
                    pageSize={PAGE_SIZE}
                  />
                </div>
              </>
            ) : (
              <EmptyState
                icon={Rocket}
                title={hasFilters ? 'No runs match your filters' : 'No runs yet'}
                description={
                  hasFilters
                    ? 'Try adjusting the search term or status filter.'
                    : 'Deploy a forecasting run from the Forecast Pipeline page; it appears here as soon as it is submitted.'
                }
              />
            )}
          </>
        )}
      </SectionContainer>
    </PageContainer>
  )
}
