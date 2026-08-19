import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { FlaskConical, AlertCircle } from 'lucide-react'
import PageContainer from '../../components/common/PageContainer'
import SectionContainer from '../../components/layout/SectionContainer'
import EmptyState from '../../components/ui/EmptyState'
import Loader from '../../components/ui/Loader'
import Select from '../../components/ui/Select'
import Badge from '../../components/ui/Badge'
import DatasetRunFilter from '../../components/common/DatasetRunFilter'
import { useDatasetRunFilter } from '../../hooks/useDatasetRunFilter'
import { fetchDeployments, fetchMLflowRun } from '../../services'
import { fallbackModelOptions, forecastModels } from '../../data/appConfig'
import { cn } from '../../utils/cn'
import OpenInDatabricksLink from '../../components/ui/OpenInDatabricksLink'

// Display names the platform already defines (appConfig.js) rather than a
// second hardcoded label set — `xgboost` -> "XGBoost", `seasonal_naive` ->
// "Seasonal naive", and so on.
const MODEL_DISPLAY_NAMES = Object.fromEntries(
  [...forecastModels, ...fallbackModelOptions].map((m) => [m.id, m.name])
)

function modelDisplayName(id) {
  return MODEL_DISPLAY_NAMES[id] || id
}

// A fixed, small categorical palette for model chips — cycled by model
// name so the same model always gets the same color within a run, and
// never confused with the amber "fallback" treatment (Section 4: fallback
// must be visually distinguishable from a normal selection, not just
// another color in the same rotation).
const MODEL_CHIP_TONES = [
  'bg-brand-50 text-brand-700 border-brand-200 dark:bg-brand-500/10 dark:text-brand-300 dark:border-brand-500/30',
  'bg-sky-50 text-sky-700 border-sky-200 dark:bg-sky-500/10 dark:text-sky-300 dark:border-sky-500/30',
  'bg-violet-50 text-violet-700 border-violet-200 dark:bg-violet-500/10 dark:text-violet-300 dark:border-violet-500/30',
  'bg-teal-50 text-teal-700 border-teal-200 dark:bg-teal-500/10 dark:text-teal-300 dark:border-teal-500/30',
  'bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-500/10 dark:text-rose-300 dark:border-rose-500/30',
]

function modelChipTone(modelName) {
  let hash = 0
  for (let i = 0; i < modelName.length; i++) hash = (hash * 31 + modelName.charCodeAt(i)) % MODEL_CHIP_TONES.length
  return MODEL_CHIP_TONES[hash]
}

// One model's (or the fallback bucket's) badge plus every key it won,
// wrapped rather than truncated — Requirement 2 is explicit that every
// key must be visible, so this never hides keys behind a "+N more".
function ModelKeyGroup({ label, count, keys, tone, isFallback }) {
  return (
    <div className="rounded-lg border border-slate-100 p-3 dark:border-slate-800">
      <div className="mb-2 flex items-center gap-2">
        <span
          className={cn(
            'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold',
            tone
          )}
        >
          {isFallback && <span className="h-1.5 w-1.5 rounded-full bg-current opacity-70" />}
          {label}
        </span>
        <span className="text-[11px] text-slate-400">
          {count} key{count === 1 ? '' : 's'}
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {keys.map((key) => (
          <span
            key={key}
            className="rounded-md bg-slate-50 px-1.5 py-0.5 font-mono text-[11px] text-slate-600 dark:bg-slate-800/60 dark:text-slate-300"
          >
            {key}
          </span>
        ))}
      </div>
    </div>
  )
}

// Requirement 2: which model won for which key, derived entirely from
// `per_key` (already fetched for the "Child runs by key" table below) —
// no separate request, no duplicated or hardcoded key/model data.
// Fallback keys are grouped in their own bucket rather than under
// whichever model actually produced the fallback forecast, so "won on
// merit" and "fell back to" never look the same at a glance.
function ModelSelectionSummary({ perKey }) {
  const { modelGroups, fallbackKeys } = useMemo(() => {
    const groups = new Map()
    const fallback = []
    for (const outcome of perKey) {
      if (outcome.fallback_used) {
        fallback.push(outcome.group_id)
      } else if (outcome.model) {
        const list = groups.get(outcome.model) || []
        list.push(outcome.group_id)
        groups.set(outcome.model, list)
      }
    }
    return {
      modelGroups: [...groups.entries()].sort((a, b) => b[1].length - a[1].length),
      fallbackKeys: fallback,
    }
  }, [perKey])

  if (modelGroups.length === 0 && fallbackKeys.length === 0) {
    return <p className="text-xs text-slate-400">No production model was selected for any key yet.</p>
  }

  return (
    <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
      {modelGroups.map(([model, keys]) => (
        <ModelKeyGroup
          key={model}
          label={modelDisplayName(model)}
          count={keys.length}
          keys={keys}
          tone={modelChipTone(model)}
        />
      ))}
      {fallbackKeys.length > 0 && (
        <ModelKeyGroup
          label="Fallback"
          count={fallbackKeys.length}
          keys={fallbackKeys}
          isFallback
          tone="bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-500/10 dark:text-amber-300 dark:border-amber-500/30"
        />
      )}
    </div>
  )
}

// Status badges reuse Badge's existing statusMap tones (Passed/Failed/
// Warning/neutral) rather than a parallel color system — see LLMOps'
// LLMCallCard for the same convention.
const HYPERPARAM_STATUS_TONE = {
  Winner: 'Passed',
  Rejected: 'neutral',
  Fallback: 'Warning',
  Eliminated: 'neutral',
  Failed: 'Failed',
  Skipped: 'neutral',
  Unavailable: 'neutral',
}

function formatMetric(value, digits = 2, suffix = '') {
  return value == null ? '—' : `${value.toFixed(digits)}${suffix}`
}

// One (key, model) pair's final hyperparameters, tied to the exact score
// they produced — Requirement 6: "these hyperparameters produced this
// model's performance for this specific key" is the header + table
// together, never a bare parameter dump with no result attached.
function HyperparameterDetail({ record }) {
  const paramEntries = Object.entries(record.hyperparameters || {})

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-sm font-semibold text-slate-800 dark:text-slate-100">
              {record.group_id}
            </span>
            <span className="text-sm text-slate-400">·</span>
            <span className="text-sm font-medium text-slate-700 dark:text-slate-200">
              {modelDisplayName(record.model_name)}
            </span>
          </div>
          <div className="mt-1">
            <Badge status={HYPERPARAM_STATUS_TONE[record.status] || 'neutral'}>{record.status}</Badge>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-right text-xs sm:grid-cols-4">
          <div>
            <div className="text-slate-400">WMAPE</div>
            <div className="tabular-nums font-medium text-slate-700 dark:text-slate-200">
              {formatMetric(record.wmape, 2, '%')}
            </div>
          </div>
          <div>
            <div className="text-slate-400">RMSE</div>
            <div className="tabular-nums font-medium text-slate-700 dark:text-slate-200">
              {formatMetric(record.rmse)}
            </div>
          </div>
          <div>
            <div className="text-slate-400">MAE</div>
            <div className="tabular-nums font-medium text-slate-700 dark:text-slate-200">
              {formatMetric(record.mae)}
            </div>
          </div>
          <div>
            <div className="text-slate-400">Rank</div>
            <div className="tabular-nums font-medium text-slate-700 dark:text-slate-200">
              {record.rank ?? '—'}
            </div>
          </div>
        </div>
      </div>

      {record.hyperparameters_unavailable_reason ? (
        <p className="rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-400 dark:bg-slate-800/60">
          {record.hyperparameters_unavailable_reason}
        </p>
      ) : paramEntries.length === 0 ? (
        <p className="text-xs text-slate-400">
          No hyperparameter record for this attempt — it did not reach a trained state.
        </p>
      ) : (
        <div className="overflow-hidden rounded-lg border border-slate-100 dark:border-slate-800">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-100 bg-slate-50/60 dark:border-slate-800 dark:bg-slate-800/40">
                <th className="px-3 py-2 font-semibold text-slate-500">Parameter</th>
                <th className="px-3 py-2 font-semibold text-slate-500">Value</th>
              </tr>
            </thead>
            <tbody>
              {paramEntries.map(([name, value]) => (
                <tr key={name} className="border-b border-slate-50 last:border-0 dark:border-slate-800/60">
                  <td className="px-3 py-1.5 font-mono text-slate-600 dark:text-slate-300">{name}</td>
                  <td className="px-3 py-1.5 tabular-nums text-slate-700 dark:text-slate-200">{String(value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {record.tuning && (
        <p className="mt-3 text-[11px] text-slate-400">
          {record.tuning.tuned ? (
            <>
              Tuned via {record.tuning.strategy} search — {record.tuning.candidates_evaluated} candidate(s) over{' '}
              {record.tuning.cv_splits} CV split(s), best CV MAE {formatMetric(record.tuning.best_score_mae, 3)}.
            </>
          ) : (
            <>Not tuned — {record.tuning.reason}</>
          )}
        </p>
      )}
    </div>
  )
}

// Requirement 5: browse every (key, model) combination without rendering
// them all at once — two dependent dropdowns select exactly one record.
function HyperparametersSection({ records }) {
  const keys = useMemo(() => [...new Set(records.map((r) => r.group_id))].sort(), [records])
  const [selectedKey, setSelectedKey] = useState(keys[0] || '')

  useEffect(() => {
    if (keys.length && !keys.includes(selectedKey)) setSelectedKey(keys[0])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keys])

  const modelsForKey = useMemo(
    () => records.filter((r) => r.group_id === selectedKey),
    [records, selectedKey]
  )
  const [selectedModel, setSelectedModel] = useState('')

  useEffect(() => {
    if (modelsForKey.length && !modelsForKey.some((r) => r.model_name === selectedModel)) {
      setSelectedModel(modelsForKey[0].model_name)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelsForKey])

  if (records.length === 0) {
    return <p className="text-xs text-slate-400">No hyperparameter records for this run yet.</p>
  }

  const modelOptions = modelsForKey.map((r) => ({
    value: r.model_name,
    label: modelDisplayName(r.model_name),
    sublabel: r.status,
  }))
  const record = modelsForKey.find((r) => r.model_name === selectedModel)

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 sm:max-w-lg">
        <Select value={selectedKey} onChange={setSelectedKey} options={keys} placeholder="Key" />
        <Select value={selectedModel} onChange={setSelectedModel} options={modelOptions} placeholder="Model" />
      </div>
      {record && <HyperparameterDetail record={record} />}
    </div>
  )
}

/**
 * MLflow Experiments — the run-level audit view.
 *
 * Deliberately not a second Results page: Results explains one key's decision
 * with its chart and narrative, while this shows what the run as a whole
 * logged to the tracking store and how every key came out. The overlap is
 * limited to naming the winning model, which is the join key between the two
 * views rather than duplicated content.
 */

function StatTile({ label, value, sub, tone = 'default' }) {
  const tones = {
    default: 'text-slate-800 dark:text-slate-100',
    good: 'text-emerald-600 dark:text-emerald-400',
    warn: 'text-amber-600 dark:text-amber-400',
  }
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-4 py-3 dark:border-slate-800 dark:bg-slate-900">
      <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">{label}</div>
      <div className={`mt-1.5 text-xl font-semibold tabular-nums ${tones[tone]}`}>{value}</div>
      {sub && <div className="mt-0.5 text-[11px] text-slate-400">{sub}</div>}
    </div>
  )
}

// A compact bar rather than a number alone: with ten-plus keys the relative
// accuracy across keys is the thing worth seeing at a glance.
function AccuracyBar({ value }) {
  if (value == null) return <span className="text-xs text-slate-400">—</span>
  const tone = value >= 70 ? 'bg-emerald-500' : value >= 50 ? 'bg-amber-500' : 'bg-rose-500'
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-20 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
        <div className={`h-full rounded-full ${tone}`} style={{ width: `${Math.min(100, value)}%` }} />
      </div>
      <span className="w-11 text-right text-xs tabular-nums text-slate-600 dark:text-slate-300">
        {value.toFixed(1)}%
      </span>
    </div>
  )
}

export default function MLflowExperiments() {
  const [deployments, setDeployments] = useState([])
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Dataset -> Run, the same dependent pair the Results page uses. Only
  // completed runs are offered: a run still executing has not written the
  // tracking record this page reads.
  // ?run=<id> arrives from the Results page's MLflow card, so "View run
  // details" lands on that exact run instead of the most recent one.
  const [searchParams] = useSearchParams()
  const { dataset, setDataset, datasetOptions, run, setRun, runOptions } = useDatasetRunFilter(
    deployments,
    { completedOnly: true, initialRun: searchParams.get('run') || '' }
  )
  const runs = runOptions

  useEffect(() => {
    fetchDeployments()
      .then(setDeployments)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }, [])

  const load = useCallback(async (runId) => {
    if (!runId) return
    setLoading(true)
    setError(null)
    try {
      setDetail(await fetchMLflowRun(runId))
    } catch (e) {
      setError(String(e.message || e))
      setDetail(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (run) load(run)
  }, [run, load])

  if (!loading && !runs.length) {
    return (
      <PageContainer>
        <SectionContainer title="MLflow Experiments" subtitle="Experiment tracking and run lineage">
          <EmptyState
            icon={FlaskConical}
            title="No completed runs yet"
            description="Deploy a forecast run; its parameters, metrics and registered models will be tracked here."
          />
        </SectionContainer>
      </PageContainer>
    )
  }

  const s = detail?.summary

  return (
    <PageContainer>
      <div className="mb-5">
        <h1 className="text-2xl font-bold tracking-tight text-slate-800 dark:text-slate-100">
          MLflow Experiments
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Run traceability — what this execution logged to the tracking store, and how every key came out.
        </p>
      </div>

      {datasetOptions.length > 0 && (
        <DatasetRunFilter
          className="mb-5 lg:max-w-3xl"
          dataset={dataset}
          datasetOptions={datasetOptions}
          onDatasetChange={setDataset}
          run={run}
          runOptions={runOptions}
          onRunChange={setRun}
        />
      )}

      {error && (
        <div className="mb-4 flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {loading && <SectionContainer><Loader label="Loading tracking record…" /></SectionContainer>}

      {!loading && detail && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <StatTile label="Keys processed" value={s.keys_processed} />
            <StatTile label="Models trained" value={s.models_trained} />
            <StatTile
              label="Fallback used"
              value={s.fallback_used}
              sub={`${s.keys_processed - s.fallback_used} selected on merit`}
              tone={s.fallback_used > 0 ? 'warn' : 'good'}
            />
            <StatTile
              label="Avg accuracy"
              value={s.average_accuracy == null ? 'N/A' : `${s.average_accuracy}%`}
              sub="100 − WMAPE"
              tone={s.average_accuracy >= 70 ? 'good' : 'warn'}
            />
          </div>

          {/* Tracking identity — the ids an auditor needs to find this run in
              MLflow itself, rather than a restatement of the forecast. The
              Databricks deep link sits here for the same reason, and renders
              only when the backend could build a correct URL. */}
          <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
            {/* "Open in Databricks" hidden — see MLflowRunCard.jsx for why.
            {detail.databricks_run_url ? (
              <div className="mb-3 flex justify-end">
                <OpenInDatabricksLink url={detail.databricks_run_url} />
              </div>
            ) : null}
            */}
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {[
                ['MLflow run', detail.mlflow_run_id],
                ['Experiment', detail.experiment],
                ['Tracking store', detail.tracking_uri],
                ['Status', detail.status],
              ].map(([label, value]) => (
                <div key={label}>
                  <div className="text-[11px] uppercase tracking-wide text-slate-400">{label}</div>
                  <div className="mt-0.5 truncate font-mono text-xs text-slate-700 dark:text-slate-200" title={value || '—'}>
                    {value || '—'}
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-3 flex flex-wrap gap-4 border-t border-slate-100 pt-3 text-xs text-slate-500 dark:border-slate-800">
              <span><b className="tabular-nums text-slate-700 dark:text-slate-200">{s.parameters_logged}</b> parameters</span>
              <span><b className="tabular-nums text-slate-700 dark:text-slate-200">{s.metrics_logged}</b> metrics</span>
              <span><b className="tabular-nums text-slate-700 dark:text-slate-200">{s.models_registered}</b> models registered</span>
            </div>
          </div>

          <div className="grid gap-4 lg:grid-cols-3">
            <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
              <h3 className="mb-3 text-sm font-semibold text-slate-800 dark:text-slate-100">
                Model Selection Summary
              </h3>
              <ModelSelectionSummary perKey={detail.per_key} />

              <h3 className="mb-3 mt-5 border-t border-slate-100 pt-4 text-sm font-semibold text-slate-800 dark:border-slate-800 dark:text-slate-100">
                Parameters logged
              </h3>
              <dl className="space-y-1.5">
                {detail.parameters.map((p) => (
                  <div key={p.name} className="flex justify-between gap-3 text-xs">
                    <dt className="font-mono text-slate-400">{p.name}</dt>
                    <dd className="truncate text-right text-slate-700 dark:text-slate-200" title={p.value}>
                      {p.value}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>

            {/* Child runs by key — the per-key roll-up, matching the reference
                traceability view. Decisions link back to Results for detail. */}
            <div className="overflow-hidden rounded-xl border border-slate-200 bg-white lg:col-span-2 dark:border-slate-800 dark:bg-slate-900">
              <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3 dark:border-slate-800">
                <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">Child runs by key</h3>
                <span className="text-[11px] text-slate-400">{detail.per_key.length} keys</span>
              </div>
              <div className="max-h-96 overflow-auto">
                <table className="w-full min-w-[620px] text-left">
                  <thead className="sticky top-0 bg-white dark:bg-slate-900">
                    <tr className="border-b border-slate-100 text-[10px] uppercase tracking-wide text-slate-400 dark:border-slate-800">
                      <th className="px-4 py-2 font-medium">Key</th>
                      <th className="px-4 py-2 font-medium">Model</th>
                      <th className="px-4 py-2 font-medium">Accuracy</th>
                      <th className="px-4 py-2 font-medium">Drift test</th>
                      <th
                        className="px-4 py-2 text-right font-medium"
                        title="The calculated drift statistic for this key — compared against the threshold to its right to determine the drift result."
                      >
                        Drift Statistic
                      </th>
                      <th
                        className="px-4 py-2 text-right font-medium"
                        title="The threshold the drift statistic is compared against. Statistic ≤ threshold passes."
                      >
                        Threshold
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.per_key.map((k) => (
                      <tr key={k.group_id} className="border-b border-slate-50 text-sm dark:border-slate-800/60">
                        <td className="px-4 py-2 font-mono text-xs text-slate-700 dark:text-slate-200">{k.group_id}</td>
                        <td className="px-4 py-2">
                          <span className="text-xs font-medium text-slate-800 dark:text-slate-100">{k.model || '—'}</span>
                          {k.fallback_used && (
                            <span className="ml-1.5 rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-700 dark:bg-amber-900/40 dark:text-amber-300">
                              fallback
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-2"><AccuracyBar value={k.accuracy} /></td>
                        <td className="px-4 py-2 text-xs text-slate-500">
                          {k.drift_algorithm ? k.drift_algorithm.replace(/_/g, ' ') : '—'}
                        </td>
                        <td className="px-4 py-2 text-right text-xs tabular-nums text-slate-500">
                          {k.drift_statistic != null ? k.drift_statistic.toFixed(4) : '—'}
                        </td>
                        <td className="px-4 py-2 text-right text-xs tabular-nums text-slate-500">
                          {k.threshold_value != null ? k.threshold_value.toFixed(4) : '—'}
                          {k.threshold_method && (
                            <span className="ml-1 text-[10px] text-slate-400">{k.threshold_method}</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Requirement 5/6: hyperparameters, tied to the exact metrics/
              status/rank they produced — its own section so a run with
              hundreds of keys never renders more than one parameter table
              at a time. */}
          <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
            <h3 className="mb-3 text-sm font-semibold text-slate-800 dark:text-slate-100">Hyperparameters</h3>
            <HyperparametersSection records={detail.hyperparameters} />
          </div>
        </div>
      )}
    </PageContainer>
  )
}
