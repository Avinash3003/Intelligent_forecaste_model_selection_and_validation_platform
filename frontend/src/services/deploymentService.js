import { apiClient } from './apiClient'
import { formatIST, formatRunLabel } from '../utils/formatDateTime'

// The backend's JobStatus vocabulary, verbatim
// (app/orchestration/schemas.py). Kept here rather than in UI data so the
// status filter can never drift from what the API actually returns.
export const JOB_STATUSES = ['Pending', 'Running', 'Completed', 'Failed', 'Cancelled']

// Polling stops as soon as a run reaches one of these.
export const TERMINAL_STATUSES = ['Completed', 'Failed', 'Cancelled']

export function isTerminalStatus(status) {
  return TERMINAL_STATUSES.includes(status)
}

// Submits a run configuration. Returns { run_id, status, message }.
export function deployRun({
  fileId,
  datasetName,
  mapping,
  selectedModels,
  fallbackModel,
  horizon,
  aggregationMethod,
}) {
  const metadata = {
    date_column: mapping.dateColumn,
    target_column: mapping.targetColumn,
    key_columns: mapping.keyColumns,
    feature_columns: mapping.featureColumns,
  }
  // Applied by preprocessing only when the detected grain is finer than
  // monthly. The backend field has no null/None variant — it's a required
  // enum with its own default — so the key is omitted rather than sent as
  // null when aggregation isn't required, letting that default apply.
  if (aggregationMethod) {
    metadata.aggregation_method = aggregationMethod
  }

  return apiClient.post('/deploy', {
    file_id: fileId,
    dataset_name: datasetName,
    metadata,
    selected_models: selectedModels,
    fallback_model: fallbackModel,
    horizon,
  })
}

// Maps one backend DeploymentStatus into the camelCase shape the
// Deployments table and detail view render. `payload.start_time` arrives
// as raw ISO 8601 UTC — every timestamp on it is formatted to IST exactly
// once, here, so no two screens can ever disagree on timezone/format.
function normalizeDeployment(payload) {
  return {
    id: payload.id,
    displayName: formatRunLabel(payload.start_time),
    dataset: payload.dataset,
    status: payload.status,
    startTime: formatIST(payload.start_time),
    startTimeRaw: payload.start_time,
    duration: payload.duration,
    progress: payload.progress,
    currentStage: payload.current_stage,
    estimatedRemaining: payload.estimated_remaining,
    stages: (payload.stages ?? []).map((stage) => ({
      label: stage.label,
      status: stage.status,
      startedAt: stage.started_at,
      completedAt: stage.completed_at,
    })),
    error: payload.error ?? null,
  }
}

// Every run the active Runner knows about, most recent first.
export async function fetchDeployments() {
  const payload = await apiClient.get('/deployments')
  return payload.map(normalizeDeployment)
}

// One run's status detail. Throws ApiError with status 404 if unknown.
export async function fetchDeployment(runId) {
  const payload = await apiClient.get(`/deployments/${encodeURIComponent(runId)}`)
  return normalizeDeployment(payload)
}

// Lightweight status poll — the endpoint the Results page watches while a
// run executes. Returns { run_id, job_status }.
export function fetchExecutionStatus(runId) {
  return apiClient.get(`/execution/${encodeURIComponent(runId)}/status`)
}
