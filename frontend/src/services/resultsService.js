import { apiClient } from './apiClient'

// Fetches the real Forecast Insights Dashboard payload for one completed
// run. `groupId` selects the business key; omitted, the backend picks the
// first group.
//
// Throws ApiError with `status` 404 (unknown run) or 409 (run has not
// finished), which the Results page distinguishes rather than collapsing
// into one generic error.
export function fetchResults(runId, groupId) {
  const query = groupId ? `?group_id=${encodeURIComponent(groupId)}` : ''
  return apiClient.get(`/results/${encodeURIComponent(runId)}${query}`)
}

// Developer debugging mode — structured execution internals for a run.
// Never refuses on status (a RUNNING or FAILED run is exactly what a
// developer wants to inspect), so no 404/409 branching is needed here.
export function fetchDebugSummary(runId) {
  return apiClient.get(`/results/${encodeURIComponent(runId)}/debug`)
}

/**
 * The head of the file a run was built from.
 *
 * Returned as-is: the payload is a flat table with no derived fields, so a
 * normalizer would add a layer without changing what renders.
 */
export function fetchDatasetPreview(runId) {
  return apiClient.get(`/results/${encodeURIComponent(runId)}/dataset-preview`)
}
