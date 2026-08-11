import { useEffect, useMemo, useState } from 'react'
import { AlertCircle, Bot } from 'lucide-react'
import PageContainer from '../../components/common/PageContainer'
import SectionContainer from '../../components/layout/SectionContainer'
import EmptyState from '../../components/ui/EmptyState'
import Loader from '../../components/ui/Loader'
import Select from '../../components/ui/Select'
import SearchBox from '../../components/ui/SearchBox'
import Pagination from '../../components/common/Pagination'
import Card from '../../components/ui/Card'
import LLMCallCard from './components/LLMCallCard'
import { fetchDeployments, fetchLlmObservability } from '../../services'
import { formatRunLabel } from '../../utils/formatDateTime'
import { formatCost, formatGroundedness, formatLatency, formatTokens } from '../../utils/formatLlmMetrics'

const PAGE_SIZE = 20

// `Select` renders its own placeholder as a disabled `value=""` option, so
// an "All X" choice must use a distinct sentinel — `""` would collide with
// that placeholder and the select would display "Select..." instead of the
// chosen "All X" label.
const ALL = 'all'

const PROVIDER_OPTIONS = [
  { value: ALL, label: 'All providers' },
  { value: 'azure_openai', label: 'Azure OpenAI' },
  { value: 'azure_openai_fallback', label: 'Azure OpenAI (fallback)' },
  { value: 'template', label: 'Template fallback' },
]

const GROUNDING_OPTIONS = [
  { value: ALL, label: 'Grounded or not' },
  { value: 'grounded', label: 'Grounded' },
  { value: 'ungrounded', label: 'Not grounded' },
]

const STATUS_OPTIONS = [
  { value: ALL, label: 'Success or failed' },
  { value: 'success', label: 'Success' },
  { value: 'failed', label: 'Failed' },
]

function SummaryTile({ label, value, sub }) {
  return (
    <Card className="p-4">
      <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400">{label}</p>
      <p className="mt-1.5 text-xl font-semibold tabular-nums text-slate-800 dark:text-slate-100">{value}</p>
      {sub && <p className="mt-0.5 text-[11px] text-slate-400">{sub}</p>}
    </Card>
  )
}

/**
 * LLMOps Observability — the LLM activity behind one run's business
 * insights: what was called, with what tokens/latency/cost, whether it
 * passed schema validation and grounding, and how many retries it took.
 *
 * Reads `GET /results/{run_id}/llmops`, which reshapes data this platform
 * already produces and stores (`business_insights` + `llm_trace` on the
 * run's own result) — nothing here is a second source of truth, and no
 * request reaches Databricks or MLflow directly from the browser.
 */
export default function LLMOps() {
  const [runs, setRuns] = useState([])
  const [runsLoading, setRunsLoading] = useState(true)
  const [runsError, setRunsError] = useState(null)
  const [run, setRun] = useState('')

  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const [providerFilter, setProviderFilter] = useState(ALL)
  const [deploymentFilter, setDeploymentFilter] = useState(ALL)
  const [groundingFilter, setGroundingFilter] = useState(ALL)
  const [statusFilter, setStatusFilter] = useState(ALL)
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)

  useEffect(() => {
    fetchDeployments()
      .then((rows) => {
        const done = rows.filter((r) => r.status === 'Completed')
        setRuns(done.map((r) => ({ value: r.id, label: formatRunLabel(r.startTimeRaw), sublabel: r.dataset })))
        if (done.length) setRun(done[0].id)
      })
      .catch((e) => setRunsError(String(e.message || e)))
      .finally(() => setRunsLoading(false))
  }, [])

  useEffect(() => {
    if (!run) return
    setLoading(true)
    setError(null)
    setPage(1)
    fetchLlmObservability(run)
      .then(setData)
      .catch((e) => {
        setError(e.message || String(e))
        setData(null)
      })
      .finally(() => setLoading(false))
  }, [run])

  // Reset to page 1 whenever a filter narrows the list, so a user never
  // lands on a now-empty trailing page.
  useEffect(() => {
    setPage(1)
  }, [providerFilter, deploymentFilter, groundingFilter, statusFilter, search])

  // A new run has its own deployment set — an option chosen for the
  // previous run would silently filter the new one down to nothing.
  useEffect(() => {
    setDeploymentFilter(ALL)
  }, [run])

  const deploymentOptions = useMemo(() => {
    const deployments = new Set((data?.calls ?? []).map((c) => c.deployment).filter(Boolean))
    return [{ value: ALL, label: 'All deployments' }, ...[...deployments].sort().map((d) => ({ value: d, label: d }))]
  }, [data])

  const filteredCalls = useMemo(() => {
    if (!data) return []
    return data.calls.filter((call) => {
      if (providerFilter !== ALL && call.provider !== providerFilter) return false
      if (deploymentFilter !== ALL && call.deployment !== deploymentFilter) return false
      if (groundingFilter !== ALL && call.grounding_status !== groundingFilter) return false
      if (statusFilter === 'success' && call.final_status !== 'success') return false
      if (statusFilter === 'failed' && call.final_status === 'success') return false
      if (search && !call.group_id.toLowerCase().includes(search.toLowerCase())) return false
      return true
    })
  }, [data, providerFilter, deploymentFilter, groundingFilter, statusFilter, search])

  const totalPages = Math.max(1, Math.ceil(filteredCalls.length / PAGE_SIZE))
  const pageCalls = filteredCalls.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  if (!runsLoading && !runs.length && !runsError) {
    return (
      <PageContainer>
        <SectionContainer title="LLMOps Observability" subtitle="Per-call LLM activity behind business insights">
          <EmptyState
            icon={Bot}
            title="No completed runs yet"
            description="Deploy a forecast run; its LLM calls, tokens, latency, cost and grounding results will be traceable here once it finishes."
          />
        </SectionContainer>
      </PageContainer>
    )
  }

  const s = data?.summary

  return (
    <PageContainer>
      <div className="mb-5">
        <h1 className="text-2xl font-bold tracking-tight text-slate-800 dark:text-slate-100">
          LLMOps Observability
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          The LLM activity behind this run's business insights — every call, its tokens, latency, cost,
          grounding and schema validation outcome.
        </p>
      </div>

      {runsLoading && <Loader label="Loading runs…" />}

      {runsError && (
        <div className="mb-4 flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <span>{runsError}</span>
        </div>
      )}

      {runs.length > 0 && (
        <div className="mb-5 sm:max-w-md">
          <Select value={run} onChange={setRun} options={runs} placeholder="Select a run" />
        </div>
      )}

      {error && (
        <div className="mb-4 flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-medium">Could not load LLM activity</p>
            <p className="mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {loading && <SectionContainer><Loader label="Loading LLM activity…" /></SectionContainer>}

      {!loading && data && !data.available && (
        <SectionContainer>
          <EmptyState
            icon={Bot}
            title="No LLM activity for this run"
            description="This run either had business insights disabled, or Azure OpenAI was not reachable for it — see the run's Results page for the reason."
          />
        </SectionContainer>
      )}

      {!loading && data && data.available && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <SummaryTile label="LLM calls" value={s.call_count} />
            <SummaryTile
              label="Total tokens"
              value={formatTokens(s.total_tokens)}
              sub={`${formatTokens(s.input_tokens)} in · ${formatTokens(s.output_tokens)} out`}
            />
            <SummaryTile label="Average latency" value={formatLatency(s.average_latency_ms)} />
            <SummaryTile
              label="Estimated cost"
              value={formatCost(s.estimated_cost_usd, s.cost_available)}
            />
            <SummaryTile label="Groundedness" value={formatGroundedness(s.groundedness_rate)} />
            <SummaryTile label="Retries" value={s.retry_count} />
            <SummaryTile
              label="Provider / deployment"
              value={s.deployment || s.provider || '—'}
              sub={s.deployment && s.provider ? s.provider : undefined}
            />
            <SummaryTile label="Prompt version" value={s.prompt_version || '—'} sub={`Run status: ${s.status}`} />
          </div>

          <div>
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-200">
                LLM Call Details
              </h2>
              <span className="text-xs text-slate-400">
                {filteredCalls.length} of {data.calls.length} group{data.calls.length === 1 ? '' : 's'}
              </span>
            </div>

            <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
              <SearchBox
                placeholder="Search Group ID…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
              <Select value={providerFilter} onChange={setProviderFilter} options={PROVIDER_OPTIONS} />
              <Select value={deploymentFilter} onChange={setDeploymentFilter} options={deploymentOptions} />
              <Select value={groundingFilter} onChange={setGroundingFilter} options={GROUNDING_OPTIONS} />
              <Select value={statusFilter} onChange={setStatusFilter} options={STATUS_OPTIONS} />
            </div>

            {filteredCalls.length === 0 ? (
              <EmptyState
                icon={Bot}
                title="No calls match these filters"
                description="Try clearing a filter or searching a different Group ID."
              />
            ) : (
              <div className="space-y-2.5">
                {pageCalls.map((call) => (
                  <LLMCallCard key={call.group_id} call={call} />
                ))}
              </div>
            )}

            <Pagination
              page={page}
              totalPages={totalPages}
              onPageChange={setPage}
              totalItems={filteredCalls.length}
              pageSize={PAGE_SIZE}
            />
          </div>
        </div>
      )}
    </PageContainer>
  )
}
