import { apiClient } from './apiClient'
import { LARGE_RESULT_TIMEOUT_MS } from './apiConfig'

// Fetches the real Forecast Insights Dashboard payload for one completed
// run. `groupId` selects the business key; omitted, the backend picks the
// first group.
//
// Throws ApiError with `status` 404 (unknown run) or 409 (run has not
// finished), which the Results page distinguishes rather than collapsing
// into one generic error. Uses the extended timeout — see
// `LARGE_RESULT_TIMEOUT_MS` — because this run's `summary.json` may still
// need a cold download on the backend the first time it is read.
export function fetchResults(runId, groupId) {
  const query = groupId ? `?group_id=${encodeURIComponent(groupId)}` : ''
  return apiClient.get(`/results/${encodeURIComponent(runId)}${query}`, { timeoutMs: LARGE_RESULT_TIMEOUT_MS })
}

// Developer debugging mode — structured execution internals for a run.
// Never refuses on status (a RUNNING or FAILED run is exactly what a
// developer wants to inspect), so no 404/409 branching is needed here.
// Reads the same potentially-cold-cached run result `fetchResults` does,
// so it gets the same extended timeout.
export function fetchDebugSummary(runId) {
  return apiClient.get(`/results/${encodeURIComponent(runId)}/debug`, { timeoutMs: LARGE_RESULT_TIMEOUT_MS })
}

/**
 * One page of the curated dataset a run trained on (not the raw upload —
 * curated is the cleaned, post-preprocessing data, and much smaller).
 *
 * `page` is 1-indexed: page 1 is rows 1-`pageSize`, page 2 is the next
 * `pageSize`, and so on. Returned as-is: the payload is a flat table with no
 * derived fields, so a normalizer would add a layer without changing what
 * renders.
 */
export function fetchDatasetPreview(runId, page = 1, pageSize = 50) {
  return apiClient.get(
    `/results/${encodeURIComponent(runId)}/dataset-preview?page=${page}&page_size=${pageSize}`
  )
}
