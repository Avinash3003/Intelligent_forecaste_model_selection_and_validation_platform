import { apiClient } from './apiClient'

// LLMOps observability — one run's LLM activity: the run-level summary
// plus one entry per forecast group (its final outcome and, when it
// retried, every attempt behind that outcome).
//
// Never refuses on status, like `fetchDebugSummary` — a developer
// inspecting LLM activity on a still-running or failed run is the normal
// case, not an error, so no 404/409 branching is needed beyond a genuinely
// unknown run_id (404).
export function fetchLlmObservability(runId) {
  return apiClient.get(`/results/${encodeURIComponent(runId)}/llmops`)
}
