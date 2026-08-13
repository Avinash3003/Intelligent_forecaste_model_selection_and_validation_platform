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
import LlmEvaluationSection from './components/LlmEvaluationSection'
import { fetchDeployments, fetchLlmObservability, fetchPromptUsage } from '../../services'
import { formatRunLabel } from '../../utils/formatDateTime'
import { formatCost, formatGroundedness, formatLatency, formatTokens } from '../../utils/formatLlmMetrics'
import { cn } from '../../utils/cn'

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

function MetricRow({ label, value }) {
  return (
    <div className="flex items-center justify-between border-b border-slate-100 py-2 text-sm last:border-0 dark:border-slate-800">
      <span className="text-slate-400">{label}</span>
      <span className="font-medium tabular-nums text-slate-700 dark:text-slate-200">{value}</span>
    </div>
  )
}

// A version's numbers, split into the three families the spec calls out:
// usage (what was spent), performance (how fast), quality (how good, and
// how often the system had to compensate via retry/fallback).
function PromptVersionDetail({ v }) {
  return (
    <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
      <div>
        <h4 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">Usage</h4>
        <MetricRow label="Calls" value={v.call_count.toLocaleString('en-US')} />
        <MetricRow label="Input Tokens" value={formatTokens(v.input_tokens)} />
        <MetricRow label="Output Tokens" value={formatTokens(v.output_tokens)} />
        <MetricRow label="Total Tokens" value={formatTokens(v.total_tokens)} />
        <MetricRow label="Estimated Cost" value={formatCost(v.estimated_cost_usd, v.cost_available)} />
      </div>
      <div>
        <h4 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">Performance</h4>
        <MetricRow label="Avg Latency" value={formatLatency(v.average_latency_ms)} />
        <MetricRow label="Runs Included" value={v.runs_included.toLocaleString('en-US')} />
      </div>
      <div>
        <h4 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">Quality</h4>
        <MetricRow label="Groundedness" value={formatGroundedness(v.groundedness_rate)} />
        <MetricRow label="Validation Pass" value={formatGroundedness(v.validation_pass_rate)} />
        <MetricRow label="Retry Rate" value={formatGroundedness(v.retry_rate)} />
        <MetricRow label="Fallback Rate" value={formatGroundedness(v.fallback_rate)} />
      </div>
    </div>
  )
}

// Usage/performance/quality aggregated across every completed run, grouped
// by the prompt version each run actually used (Section 13's per-run trace,
// summed server-side — no second tracking system, see
// `LLMOpsService.get_prompt_usage`). Deliberately its own fetch and its own
// state: this reports across all runs, while the rest of the page reports
// on the one run selected below, and the two must never be conflated.
function PromptUsageSection({ usage, loading, error, selected, onSelect }) {
  if (loading) return <Loader label="Loading prompt usage…" />

  if (error) {
    return (
      <div className="flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
        <AlertCircle size={16} className="mt-0.5 shrink-0" />
        <span>{error}</span>
      </div>
    )
  }

  if (!usage || !usage.versions.length) {
    return (
      <EmptyState
        icon={Bot}
        title="No prompt usage yet"
        description="Once a run completes with LLM activity, its calls will be aggregated here by prompt version."
      />
    )
  }

  const activeVersion = usage.versions.find((v) => v.prompt_version === selected) || usage.versions[0]

  return (
    <div className="space-y-4">
      {/* A pure selector, not a second copy of the numbers — every figure
          it might otherwise repeat (calls, tokens, latency, cost...)
          already lives in the detail card below, once. Only shown at all
          once there is something to switch between. */}
      {usage.versions.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {usage.versions.map((v) => (
            <button
              key={v.prompt_version}
              type="button"
              onClick={() => onSelect(v.prompt_version)}
              className={cn(
                'rounded-full border px-3.5 py-1.5 text-sm font-medium transition-colors',
                v.prompt_version === activeVersion.prompt_version
                  ? 'border-brand-300 bg-brand-50 text-brand-700 dark:border-brand-700 dark:bg-brand-900/30 dark:text-brand-300'
                  : 'border-slate-200 text-slate-600 hover:border-slate-300 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800/50'
              )}
            >
              {v.prompt_version}
              <span className="ml-1.5 text-xs opacity-70">
                {v.call_count.toLocaleString('en-US')} call{v.call_count === 1 ? '' : 's'}
              </span>
            </button>
          ))}
        </div>
      )}

      <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
        <h3 className="mb-3 text-sm font-semibold text-slate-800 dark:text-slate-100">
          Prompt: {activeVersion.prompt_version}
        </h3>
        <PromptVersionDetail v={activeVersion} />
      </div>
    </div>
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

  const [promptUsage, setPromptUsage] = useState(null)
  const [promptUsageLoading, setPromptUsageLoading] = useState(true)
  const [promptUsageError, setPromptUsageError] = useState(null)
  const [selectedPromptVersion, setSelectedPromptVersion] = useState(null)

  useEffect(() => {
    fetchPromptUsage()
      .then((res) => {
        setPromptUsage(res)
        if (res.versions.length) setSelectedPromptVersion(res.versions[0].prompt_version)
      })
      .catch((e) => setPromptUsageError(e.message || String(e)))
      .finally(() => setPromptUsageLoading(false))
  }, [])

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

  // Only the per-run "LLM Call Details" section below actually needs a
  // completed run to show anything — Prompt Usage & Performance aggregates
  // across whatever runs exist (nothing, today) and LLM Evaluation is a
  // standalone regression suite tied to no run at all, so neither should
  // be hidden just because no forecast has been deployed yet.
  const noCompletedRuns = !runsLoading && !runs.length && !runsError

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

      <div className="mb-6">
        <SectionContainer
          title="Prompt Usage & Performance"
          subtitle="Usage, performance and quality aggregated across every completed run, grouped by prompt version"
        >
          <PromptUsageSection
            usage={promptUsage}
            loading={promptUsageLoading}
            error={promptUsageError}
            selected={selectedPromptVersion}
            onSelect={setSelectedPromptVersion}
          />
        </SectionContainer>
      </div>

      <div className="mb-6">
        <SectionContainer
          title="LLM Evaluation"
          subtitle="Regression suite (Section 13.3) — is the explanation still correct and high quality for the current prompt version and model?"
        >
          <LlmEvaluationSection />
        </SectionContainer>
      </div>

      {noCompletedRuns && (
        <div className="mb-6">
          <SectionContainer title="LLM Call Details" subtitle="Per-call LLM activity behind one run's business insights">
            <EmptyState
              icon={Bot}
              title="No completed runs yet"
              description="Deploy a forecast run; its LLM calls, tokens, latency, cost and grounding results will be traceable here once it finishes."
            />
          </SectionContainer>
        </div>
      )}

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
