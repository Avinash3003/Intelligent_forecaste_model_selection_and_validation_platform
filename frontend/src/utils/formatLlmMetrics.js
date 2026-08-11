// Formatting for LLMOps observability figures — tokens, latency, cost.
// Mirrors `formatDateTime.js`'s convention: one small pure function per
// concern, honest about missing data as "—" rather than a fabricated 0.

// "980" — thousands-separated token count, or "—" when not captured
// (e.g. the template fallback path, which never calls the LLM at all).
export function formatTokens(count) {
  if (count == null) return '—'
  return count.toLocaleString('en-US')
}

// "2.2 s" under a second shows as "820 ms"; a second or more shows in
// seconds — matches how `formatDuration` scales stage timings, just at
// LLM-call granularity instead of pipeline-stage granularity.
export function formatLatency(ms) {
  if (ms == null) return '—'
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

// "$0.0501" when pricing is configured; "Unavailable" when it is not —
// never "$0.00", which would misreport an unpriced call as a free one.
// `available` is the engine's own `cost_available` flag (Section 13.2):
// it is false whenever `AZURE_OPENAI_PRICE_*` was never set, independent
// of whether `usd` happens to be present.
export function formatCost(usd, available) {
  if (!available || usd == null) return 'Unavailable'
  return `$${usd.toFixed(usd < 0.01 ? 4 : 2)}`
}

// "100%" / "0%" / "—" when nothing was graded (e.g. every call used the
// template fallback, which skips grounding entirely).
export function formatGroundedness(rate) {
  if (rate == null) return '—'
  return `${Math.round(rate * 100)}%`
}
