import { useEffect, useState } from 'react'
import { AlertCircle, CheckCircle2, ClipboardList, XCircle } from 'lucide-react'
import Card from '../../../components/ui/Card'
import Badge from '../../../components/ui/Badge'
import Loader from '../../../components/ui/Loader'
import EmptyState from '../../../components/ui/EmptyState'
import CollapsibleSection from '../../Results/components/CollapsibleSection'
import { fetchLlmEvaluation } from '../../../services'
import { cn } from '../../../utils/cn'

const CHECK_LABELS = {
  schema_validity: 'Schema',
  groundedness: 'Groundedness',
  winner_consistency: 'Winner Consistency',
  rejection_accuracy: 'Rejection Accuracy',
  readability: 'Readability',
}

function formatPct(value) {
  if (value == null) return '—'
  return `${(value * 100).toFixed(1)}%`
}

// The evaluation report's one canonical groundedness value — shown once,
// as a headline, never repeated elsewhere as a second (and possibly
// different) number on this screen.
function groundednessStatus(rate) {
  if (rate == null) return '—'
  if (rate >= 1) return 'Grounded'
  if (rate <= 0) return 'Not Grounded'
  return 'Partially Grounded'
}

function MetricTile({ label, value, tone }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3.5 dark:border-slate-800 dark:bg-slate-900">
      <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400">{label}</p>
      <p
        className={cn(
          'mt-1 text-lg font-semibold tabular-nums',
          tone === 'good' && 'text-emerald-600 dark:text-emerald-400',
          tone === 'bad' && 'text-rose-600 dark:text-rose-400',
          !tone && 'text-slate-800 dark:text-slate-100'
        )}
      >
        {value}
      </p>
    </div>
  )
}

function CaseCheckRow({ name, check }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-slate-100 py-2 text-sm last:border-0 dark:border-slate-800">
      <span className="text-slate-500 dark:text-slate-400">{CHECK_LABELS[name] || name}</span>
      <div className="text-right">
        <span className={cn('font-medium', check.passed ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400')}>
          {check.passed ? 'PASS' : 'FAIL'}
        </span>
        {!check.passed && check.detail && (
          <p className="mt-0.5 max-w-md text-xs text-slate-400">{check.detail}</p>
        )}
      </div>
    </div>
  )
}

function CaseDetail({ result }) {
  const insight = result.insight
  const expected = result.expected || {}
  return (
    <div className="space-y-4 p-4">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <h4 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">Expected</h4>
          <div className="space-y-1 text-sm text-slate-600 dark:text-slate-300">
            <p>
              Winner: <span className="font-medium text-slate-800 dark:text-slate-100">{expected.selected_model}</span>
              {expected.is_fallback && <span className="ml-1 text-xs text-amber-600 dark:text-amber-400">(fallback)</span>}
            </p>
            {expected.wmape != null && <p>WMAPE: {expected.wmape}%</p>}
            {expected.confidence_pct != null && <p>Confidence: {expected.confidence_pct}%</p>}
            {expected.rejected_candidates?.length > 0 && (
              <p>
                Rejected:{' '}
                {expected.rejected_candidates.map((c) => `${c.model_name} (${c.reason})`).join('; ')}
              </p>
            )}
            {expected.is_fallback && expected.fallback_trigger && <p>Fallback trigger: {expected.fallback_trigger}</p>}
          </div>
        </div>
        <div>
          <h4 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">Actual LLM Response</h4>
          {insight ? (
            <div className="space-y-1 text-sm text-slate-600 dark:text-slate-300">
              <p>
                Selected model: <span className="font-medium text-slate-800 dark:text-slate-100">{insight.selected_model}</span>
              </p>
              {insight.rejection_reasons?.length > 0 && (
                <p>Rejection reasons: {insight.rejection_reasons.join('; ')}</p>
              )}
              <p className="italic">&ldquo;{insight.concise_summary}&rdquo;</p>
            </div>
          ) : (
            <p className="text-sm text-rose-600 dark:text-rose-400">
              {result.generation_error || 'No response was produced for this case.'}
            </p>
          )}
        </div>
      </div>

      <div>
        <h4 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">Evaluation Results</h4>
        {Object.entries(result.checks || {}).map(([name, check]) => (
          <CaseCheckRow key={name} name={name} check={check} />
        ))}
      </div>

      {result.failed_checks?.length > 0 && (
        <div className="rounded-lg border border-rose-200 bg-rose-50/70 px-3 py-2 text-xs text-rose-700 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
          Failed checks: {result.failed_checks.map((c) => CHECK_LABELS[c] || c).join(', ')}
        </div>
      )}
    </div>
  )
}

export default function LlmEvaluationSection() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchLlmEvaluation()
      .then(setData)
      .catch((e) => setError(e.message || String(e)))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Loader label="Loading evaluation report…" />

  if (error) {
    return (
      <div className="flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
        <AlertCircle size={16} className="mt-0.5 shrink-0" />
        <span>{error}</span>
      </div>
    )
  }

  if (!data || !data.available) {
    return (
      <EmptyState
        icon={ClipboardList}
        title="No evaluation report yet"
        description={
          data?.unavailable_reason ||
          "Run 'python -m forecast_engine.s11_llm.evaluate' to generate the LLM Evaluation & Regression report."
        }
      />
    )
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-sm text-slate-500 dark:text-slate-400">
          <span>
            Evaluation Dataset: <span className="font-medium text-slate-700 dark:text-slate-200">{data.dataset_version}</span>
          </span>
          <span>
            Generation: <span className="font-medium text-slate-700 dark:text-slate-200">{data.generation_mode}</span>
          </span>
          {data.generated_at && <span>As of {data.generated_at}</span>}
        </div>
        <Badge status={data.regression_passed ? 'Completed' : 'Failed'}>
          Overall: {data.regression_passed ? 'Passed' : 'Failed'}
        </Badge>
      </div>

      {/* The evaluation report's ONE canonical Groundedness value — the
          only place this section shows it, so it can never disagree with
          a second copy elsewhere on this screen. */}
      <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900 sm:max-w-xs">
        <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400">Groundedness</p>
        <p className="mt-1 text-2xl font-semibold tabular-nums text-slate-800 dark:text-slate-100">
          {formatPct(data.groundedness_rate)}
        </p>
        <p className="mt-0.5 text-xs text-slate-400">{groundednessStatus(data.groundedness_rate)}</p>
      </div>

      <div>
        <h3 className="mb-2 text-sm font-semibold text-slate-700 dark:text-slate-200">Key Metrics</h3>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <MetricTile label="Cases Evaluated" value={data.case_count} />
          <MetricTile
            label="Overall Pass Rate"
            value={formatPct(data.overall_pass_rate)}
            tone={data.overall_pass_rate >= 0.9 ? 'good' : data.overall_pass_rate >= 0.7 ? undefined : 'bad'}
          />
          <MetricTile label="Hallucination" value={formatPct(data.hallucination_rate)} tone={data.hallucination_rate > 0 ? undefined : 'good'} />
          <MetricTile label="Winner Consistency" value={formatPct(data.winner_consistency_rate)} />
          <MetricTile label="Rejection Accuracy" value={formatPct(data.rejection_accuracy_rate)} />
          <MetricTile label="Schema Validity" value={formatPct(data.schema_pass_rate)} />
          <MetricTile label="Readability" value={formatPct(data.readability_pass_rate)} />
        </div>
      </div>

      {data.threshold_violations?.length > 0 && (
        <div className="rounded-lg border border-rose-200 bg-rose-50/70 px-3 py-2 text-xs text-rose-700 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
          {data.threshold_violations.map((v) => (
            <p key={v}>{v}</p>
          ))}
        </div>
      )}

      {/* Detail, not the primary presentation — collapsed by default so a
          POC demo opens on the result, not a wall of individual cases.
          Nothing here is removed, only deferred behind one click. */}
      <CollapsibleSection title={`Individual Evaluation Cases (${data.results.length})`}>
        <div className="space-y-2 p-2">
          {data.results.map((result) => (
            <CollapsibleSection
              key={result.case_id}
              title={
                <span className="flex items-center gap-2">
                  {result.overall === 'PASS' ? (
                    <CheckCircle2 size={14} className="text-emerald-500" />
                  ) : (
                    <XCircle size={14} className="text-rose-500" />
                  )}
                  <span className="font-mono text-xs">{result.case_id}</span>
                  {result.scenario && <span className="text-xs text-slate-400">({result.scenario})</span>}
                </span>
              }
            >
              <CaseDetail result={result} />
            </CollapsibleSection>
          ))}
        </div>
      </CollapsibleSection>
    </div>
  )
}
