import { apiClient } from './apiClient'
import { LARGE_RESULT_TIMEOUT_MS } from './apiConfig'

// LLMOps observability — one run's LLM activity: the run-level summary
// plus one entry per forecast group (its final outcome and, when it
// retried, every attempt behind that outcome).
//
// Never refuses on status, like `fetchDebugSummary` — a developer
// inspecting LLM activity on a still-running or failed run is the normal
// case, not an error, so no 404/409 branching is needed beyond a genuinely
// unknown run_id (404). Reads the same potentially-cold-cached run result
// `fetchResults` does, so it gets the same extended timeout — see
// `LARGE_RESULT_TIMEOUT_MS`.
export function fetchLlmObservability(runId) {
  return apiClient.get(`/results/${encodeURIComponent(runId)}/llmops`, { timeoutMs: LARGE_RESULT_TIMEOUT_MS })
}

// Usage/performance/quality aggregated across every completed run, grouped
// by the prompt version each run actually used — built from the same
// per-run trace this file already reads, summed across runs server-side.
export function fetchPromptUsage() {
  return apiClient.get('/results/llmops/prompt-usage')
}

// The latest LLM Evaluation & Regression report (Section 13.3) — read-only:
// this never triggers an evaluation run, only reads back whatever
// `python -m forecast_engine.s11_llm.evaluate` last wrote. `available:
// false` (not an error) means no report has been generated yet.
export function fetchLlmEvaluation() {
  return apiClient.get('/results/llmops/evaluation')
}
