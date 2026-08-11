import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { AlertCircle, AlertTriangle, LineChart, Rocket, SearchX } from 'lucide-react'
import PageContainer from '../../components/common/PageContainer'
import SectionContainer from '../../components/layout/SectionContainer'
import Loader from '../../components/ui/Loader'
import EmptyState from '../../components/ui/EmptyState'
import Button from '../../components/ui/Button'
import Select from '../../components/ui/Select'
import DecisionSummaryStrip from './components/DecisionSummaryStrip'
import ModelComparisonTable from './components/ModelComparisonTable'
import ActualVsForecastChart from './components/ActualVsForecastChart'
import AiInsightCard from './components/AiInsightCard'
import CollapsibleSection from './components/CollapsibleSection'
import DatasetPreviewPanel from './components/DatasetPreviewPanel'
import UnderlyingMetricsPanel from './components/UnderlyingMetricsPanel'
import MLflowRunCard from './components/MLflowRunCard'
import DebugPanel from './components/DebugPanel'
import { useRunStatusPolling } from '../../hooks/useRunStatusPolling'
import { fetchDeployments, fetchResults, isTerminalStatus } from '../../services'

// Maps the backend's snake_case dashboard payload into the shape the
// Results components render. Every value originates from the run's real
// PipelineResult — nothing here is derived or invented.
function normalizeResults(response) {
  return {
    runId: response.run_id,
    groupId: response.group_id,
    groups: response.groups.map((group) => ({ value: group.group_id, label: group.label })),
    horizonPoints: response.horizon_points,
    modelDecision: {
      selectedModel: response.model_decision.selected_model,
      // null when even backtest data is unavailable — never a fabricated
      // percentage; components must render "N/A" for this, not "0%".
      confidence: response.model_decision.confidence,
      confidenceExplanation: {
        backtestAccuracy: response.model_decision.confidence_explanation.backtest_accuracy,
        forecastStability: response.model_decision.confidence_explanation.forecast_stability,
        driftMargin: response.model_decision.confidence_explanation.drift_margin,
        formula: response.model_decision.confidence_explanation.formula,
        explanation: response.model_decision.confidence_explanation.explanation,
      },
      rankingPosition: response.model_decision.ranking_position,
      validationStatus: response.model_decision.validation_status,
      fallbackUsed: response.model_decision.fallback_used,
      fallbackReason: response.model_decision.fallback_reason,
      originalCandidates: response.model_decision.original_candidates,
    },
    evaluatedModels: response.evaluated_models.map((row) => ({
      model: row.model,
      trainingStatus: row.training_status,
      trainingError: row.training_error,
      backtest: row.backtest && {
        wmape: row.backtest.wmape,
        rmse: row.backtest.rmse,
        mae: row.backtest.mae,
        mape: row.backtest.mape,
        smape: row.backtest.smape,
        windowCount: row.backtest.window_count,
      },
      forwardValidationStatus: row.forward_validation_status,
      forwardValidationReasons: row.forward_validation_reasons,
      forwardValidationRules: row.forward_validation_rules.map((rule) => ({
        ruleName: rule.rule_name,
        passed: rule.passed,
        detail: rule.detail,
      })),
      ranking: row.ranking && {
        compositeScore: row.ranking.composite_score,
        backtestScore: row.ranking.backtest_score,
        stabilityScore: row.ranking.stability_score,
        shapScore: row.ranking.shap_score,
        shapMethod: row.ranking.shap_method,
        originalBacktestRank: row.ranking.original_backtest_rank,
        finalRank: row.ranking.final_rank,
      },
      drift: row.drift && {
        evaluated: row.drift.evaluated,
        algorithm: row.drift.algorithm,
        statistic: row.drift.statistic,
        thresholdValue: row.drift.threshold_value,
        thresholdMethod: row.drift.threshold_method,
        passed: row.drift.passed,
        detail: row.drift.detail,
      },
      selectionOutcome: row.selection_outcome,
    })),
    actualVsForecast: response.actual_vs_forecast,
    explainability: {
      keyModelHeadline: response.explainability.key_model_headline,
      paragraphs: response.explainability.paragraphs,
      available: response.explainability.available,
      status: response.explainability.status,
      insight: response.explainability.insight || {},
    },
    shapDrivers: response.shap_drivers,
    underlyingMetrics: {
      driftTest: response.underlying_metrics.drift_test,
      thresholdMethod: response.underlying_metrics.threshold_method,
      thresholdValue: response.underlying_metrics.threshold_value,
      driftScore: response.underlying_metrics.drift_score,
      wmape: response.underlying_metrics.wmape,
      rmse: response.underlying_metrics.rmse,
      mae: response.underlying_metrics.mae,
      validationResult: response.underlying_metrics.validation_result,
    },
    mlflowRun: {
      runId: response.mlflow_run.run_id,
      experiment: response.mlflow_run.experiment,
      status: response.mlflow_run.status,
      trackingUri: response.mlflow_run.tracking_uri,
      modelsRegistered: response.mlflow_run.models_registered,
    },
  }
}

export default function Results() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()

  // A run may be selected by ?run=..., which is how the pipeline wizard
  // hands off straight after deploying.
  const runFromUrl = searchParams.get('run') || ''

  const [runs, setRuns] = useState([])
  const [runsLoading, setRunsLoading] = useState(true)
  const [run, setRun] = useState(runFromUrl)
  const [groupId, setGroupId] = useState('')
  const [horizon, setHorizon] = useState('')

  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [notFound, setNotFound] = useState(false)
  // A client-side abort because the response took too long — not the same
  // as a real failure. A large run's first, cold-cache load can genuinely
  // take a while (see `LARGE_RESULT_TIMEOUT_MS`), so this gets its own
  // friendlier message and a retry instead of "Could not load results".
  const [timedOut, setTimedOut] = useState(false)

  // Live status for the selected run. Stops on its own once terminal.
  const { status, notFound: statusNotFound, error: statusError } = useRunStatusPolling(run)

  // Every submitted run, so a user can switch between them. Unlike before,
  // in-progress runs are listed too — they are what polling is for.
  useEffect(() => {
    let cancelled = false
    fetchDeployments()
      .then((data) => {
        if (cancelled) return
        setRuns(data.map((item) => ({ value: item.id, label: `${item.displayName} · ${item.dataset}` })))
        // Default to the newest run only when the URL did not name one.
        setRun((current) => current || (data.length > 0 ? data[0].id : ''))
      })
      .catch((err) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setRunsLoading(false))
    return () => {
      cancelled = true
    }
  }, [])

  // Keep the URL in step with the selected run so the page is linkable and
  // survives a refresh.
  useEffect(() => {
    if (run && run !== runFromUrl) setSearchParams({ run }, { replace: true })
  }, [run, runFromUrl, setSearchParams])

  const loadResults = useCallback(
    async (targetRun, targetGroup, { signal } = {}) => {
      setLoading(true)
      setError(null)
      setNotFound(false)
      setTimedOut(false)
      try {
        const response = await fetchResults(targetRun, targetGroup || undefined)
        if (signal?.aborted) return
        const normalized = normalizeResults(response)
        setResults(normalized)
        setGroupId(normalized.groupId || '')
        setHorizon((current) =>
          normalized.horizonPoints.includes(current) ? current : normalized.horizonPoints[0] || ''
        )
      } catch (err) {
        if (signal?.aborted) return
        setResults(null)
        if (err.status === 404) setNotFound(true)
        // 409 means "still running" — the poller is already handling that,
        // so it is not surfaced as an error.
        else if (err.status !== 409) {
          if (err.isTimeout) setTimedOut(true)
          else setError(err.message)
        }
      } finally {
        if (!signal?.aborted) setLoading(false)
      }
    },
    []
  )

  // Results are fetched only once the run has actually completed — that is
  // the whole point of polling status first.
  useEffect(() => {
    if (!run || status !== 'Completed') {
      if (status && status !== 'Completed') setResults(null)
      return undefined
    }
    const controller = new AbortController()
    loadResults(run, groupId, { signal: controller.signal })
    return () => controller.abort()
  }, [run, groupId, status, loadResults])

  const filters = useMemo(
    () => ({
      run,
      runOptions: runs,
      onRunChange: (value) => {
        // A different run has its own groups, so drop the current key.
        setRun(value)
        setGroupId('')
        setResults(null)
      },
      businessKey: groupId,
      businessKeyOptions: results?.groups ?? [],
      onBusinessKeyChange: setGroupId,
      horizon,
      horizonOptions: results?.horizonPoints ?? [],
      onHorizonChange: setHorizon,
    }),
    [run, runs, groupId, results, horizon]
  )

  const goToDeployments = (
    <Button variant="secondary" icon={Rocket} onClick={() => navigate('/deployments')}>
      Back to Deployments
    </Button>
  )

  function renderBody() {
    // No run submitted on this backend at all.
    if (!runsLoading && runs.length === 0) {
      return (
        <SectionContainer>
          <EmptyState
            icon={LineChart}
            title="No runs yet"
            description="Deploy a forecasting run from the Forecast Pipeline page; its results appear here once it finishes."
            action={
              <Button icon={Rocket} onClick={() => navigate('/forecast-pipeline')}>
                New Forecast Pipeline
              </Button>
            }
          />
        </SectionContainer>
      )
    }

    if (runsLoading) {
      return (
        <SectionContainer>
          <Loader label="Loading runs…" />
        </SectionContainer>
      )
    }

    // 404 — the run id is not known to this backend.
    if (statusNotFound || notFound) {
      return (
        <SectionContainer>
          <EmptyState
            icon={SearchX}
            title="Forecast run not found"
            description={`No forecast run matches "${run}". Run history is held in memory, so runs submitted before the backend last restarted are no longer available.`}
            action={goToDeployments}
          />
        </SectionContainer>
      )
    }

    // FAILED / CANCELLED — a terminal state with no results to render.
    if (status === 'Failed' || status === 'Cancelled') {
      const failed = status === 'Failed'
      return (
        <SectionContainer>
          <div className="flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
            <AlertTriangle size={16} className="mt-0.5 shrink-0" />
            <div>
              <p className="font-medium">
                {failed ? 'Forecast execution failed' : 'Forecast execution was cancelled'}
              </p>
              <p className="mt-0.5">
                {failed
                  ? 'This run did not produce any results. Open it in Deployments to see the failure reported by the forecast engine.'
                  : 'This run was cancelled before it produced any results.'}
              </p>
              <div className="mt-3">
                <Button variant="secondary" onClick={() => navigate(`/deployments/${run}`)}>
                  View run details
                </Button>
              </div>
            </div>
          </div>
        </SectionContainer>
      )
    }

    // 409 / still executing — keep polling, show progress instead of an error.
    if (status && !isTerminalStatus(status)) {
      return (
        <SectionContainer>
          <div className="py-6">
            <Loader label={`Forecast is still running — status: ${status}. This page updates automatically.`} />
          </div>
        </SectionContainer>
      )
    }

    if (statusError) {
      return (
        <SectionContainer>
          <div className="flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <div>
              <p className="font-medium">Could not check run status</p>
              <p className="mt-0.5">{statusError}</p>
            </div>
          </div>
        </SectionContainer>
      )
    }

    // A large run's first, cold-cache load can genuinely take longer than
    // usual — this is not a failure, so it gets its own message and a
    // manual retry rather than the red "Could not load results" box below.
    if (timedOut) {
      return (
        <SectionContainer>
          <div className="flex items-start gap-2.5 rounded-xl border border-amber-200 bg-amber-50/70 px-4 py-3 text-sm text-amber-700 dark:border-amber-800 dark:bg-amber-900/20 dark:text-amber-300">
            <AlertTriangle size={16} className="mt-0.5 shrink-0" />
            <div>
              <p className="font-medium">This run is taking longer than usual to load</p>
              <p className="mt-0.5">
                Large runs with many forecast keys can take a little longer the first time their results are
                opened. This is expected — try again in a moment.
              </p>
              <div className="mt-3">
                <Button variant="secondary" onClick={() => loadResults(run, groupId)}>
                  Try again
                </Button>
              </div>
            </div>
          </div>
        </SectionContainer>
      )
    }

    if (error) {
      return (
        <SectionContainer>
          <div className="flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <div>
              <p className="font-medium">Could not load results</p>
              <p className="mt-0.5">{error}</p>
            </div>
          </div>
        </SectionContainer>
      )
    }

    if (loading || !results) {
      return (
        <SectionContainer>
          <Loader label="Loading forecast results…" />
        </SectionContainer>
      )
    }

    return (
      <div className="space-y-4">
        {/* The decision, as four values. Read before anything else. */}
        <DecisionSummaryStrip
          decision={results.modelDecision}
          evaluatedModels={results.evaluatedModels}
        />

        {/* Chart dominant, insight beside it — the forecast is the thing a
            reader judges first, and the narrative only annotates it. */}
        <div className="grid gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <ActualVsForecastChart filters={filters} points={results.actualVsForecast} />
          </div>
          <div className="lg:col-span-1">
            <AiInsightCard
              narrative={results.explainability}
              confidence={results.modelDecision.confidence}
            />
          </div>
        </div>

        <ModelComparisonTable
          decision={results.modelDecision}
          evaluatedModels={results.evaluatedModels}
        />

        {/* Everything below is evidence, not the decision — collapsed so the
            default page stays scannable (progressive disclosure). */}
        {/* The curated (post-preprocessing) data, so a user can confirm the
            decision was made from what they expect — not the raw upload,
            which can be far larger and still carry duplicates/bad rows.
            Lazy — the fetch only happens when this section is opened. */}
        <CollapsibleSection title="Curated dataset">
          <DatasetPreviewPanel runId={run} />
        </CollapsibleSection>

        <CollapsibleSection title="Validation & drift metrics">
          <UnderlyingMetricsPanel metrics={results.underlyingMetrics} />
        </CollapsibleSection>

        {results.shapDrivers.length > 0 && (
          <CollapsibleSection title="Top SHAP feature drivers">
            <div className="space-y-2 px-4 pb-4">
              {results.shapDrivers.map((driver) => (
                <div key={driver.feature} className="flex items-center gap-3">
                  <span className="w-40 shrink-0 truncate text-sm text-slate-600 dark:text-slate-300">
                    {driver.feature}
                  </span>
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                    <div
                      className="h-full rounded-full bg-brand-600"
                      style={{ width: `${Math.min(100, Math.abs(driver.importance) * 100)}%` }}
                    />
                  </div>
                  <span className="w-16 shrink-0 text-right text-xs tabular-nums text-slate-400">
                    {driver.importance}
                  </span>
                </div>
              ))}
            </div>
          </CollapsibleSection>
        )}

        <CollapsibleSection title="MLflow run">
          <MLflowRunCard run={results.mlflowRun} />
        </CollapsibleSection>

        <CollapsibleSection title="Developer debug">
          <DebugPanel runId={run} />
        </CollapsibleSection>
      </div>
    )
  }

  return (
    <PageContainer>
      <div className="mb-6">
        <h1 className="text-2xl font-bold tracking-tight text-slate-800 dark:text-slate-100">
          Results
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Forecast Insights Dashboard — model decision, chart and explainability by key and horizon.
        </p>
      </div>

      {/* The run selector stays available in every state, so a user can
          always switch away from a failed or missing run. */}
      {runs.length > 0 && (
        <div className="mb-5 sm:max-w-md">
          <Select
            value={run}
            onChange={filters.onRunChange}
            options={runs}
            placeholder="Select a run"
          />
        </div>
      )}

      {renderBody()}
    </PageContainer>
  )
}
