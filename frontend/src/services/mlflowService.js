import { apiClient } from './apiClient'

/**
 * One run's MLflow tracking record for the Experiments page.
 *
 * Returned as-is (snake_case): unlike the Results payload this is a flat
 * governance read with no derived fields, so normalizing it would only add a
 * mapping layer to maintain without changing what renders.
 */
export function fetchMLflowRun(runId) {
  return apiClient.get(`/mlflow/runs/${encodeURIComponent(runId)}`)
}
