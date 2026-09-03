// Single source of truth for the backend base URL. Every API service reads
// from here instead of embedding a URL directly, so switching environments
// (local -> staging -> prod) only ever touches VITE_API_BASE_URL.
//
// The localhost fallback exists purely so `npm run dev` works out of the
// box without requiring a local .env file — it is never relied on outside
// local development. See .env.example.
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export const REQUEST_TIMEOUT_MS = 30000

// Endpoint-specific, not a global bump. `/results/{run_id}` (and its
// per-run siblings, debug/LLMOps) can — on its *first* read after a
// backend restart or for a run nobody has opened yet — require a cold,
// uncached download of that run's `summary.json` from Databricks/MLflow.
// That artifact scales with the run's group/model count, so for a large
// run (hundreds of groups) it can legitimately take longer than 30s on a
// slow connection, even though the backend is working correctly and every
// later request for the same run answers from its in-memory cache in
// milliseconds. Every other endpoint (deploy, status polling, dataset
// preview, uploads) has a response size that does not grow with run size,
// so the default timeout above still applies to those.
export const LARGE_RESULT_TIMEOUT_MS = 120000

// How long an upload may make NO progress before it is abandoned. Not a
// limit on how long an upload may take: that is the file size over the
// user's upstream bandwidth, which this code cannot know. See apiClient's
// `upload`.
export const UPLOAD_STALL_TIMEOUT_MS = 60000
