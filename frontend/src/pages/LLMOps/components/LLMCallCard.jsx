import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'
import Badge from '../../../components/ui/Badge'
import { cn } from '../../../utils/cn'
import { formatISTTime } from '../../../utils/formatDateTime'
import { formatCost, formatLatency, formatTokens } from '../../../utils/formatLlmMetrics'

// Grounding/validation/provider statuses aren't in Badge's statusMap (those
// are pipeline-run statuses) — mapped here to the same visual language
// (emerald=good, rose=bad, amber=caveat, slate=neutral) rather than adding
// a parallel badge component.
const GROUNDING_LABEL = { grounded: 'Grounded', ungrounded: 'Not grounded', skipped: 'Grounding skipped', not_attempted: 'No grounding check' }
const GROUNDING_TONE = { grounded: 'Passed', ungrounded: 'Failed', skipped: 'neutral', not_attempted: 'neutral' }
const VALIDATION_TONE = { passed: 'Passed', failed: 'Failed', not_attempted: 'neutral' }
const PROVIDER_LABEL = { azure_openai: 'Azure OpenAI', azure_openai_fallback: 'Azure OpenAI (fallback)', template: 'Template fallback', none: 'No provider' }

function Field({ label, value }) {
  return (
    <div>
      <p className="text-[11px] text-slate-400">{label}</p>
      <p className="mt-0.5 text-sm font-medium text-slate-700 dark:text-slate-200">{value ?? '—'}</p>
    </div>
  )
}

// One forecast group's LLM outcome — a compact header (always visible) and
// an expandable body with the generated content, validation/grounding
// detail, and every retry attempt. No raw prompt or response text is shown
// because none is stored upstream (Section 13.4's trace deliberately keeps
// only structured/derived fields, never the raw request/response body) —
// `concise_summary` etc. below is the full extent of what exists to show.
export default function LLMCallCard({ call }) {
  const [open, setOpen] = useState(false)
  const usedFallbackProvider = call.provider === 'azure_openai_fallback'
  const usedTemplate = call.provider === 'template'

  return (
    <div className="rounded-xl border border-slate-100 dark:border-slate-800">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full flex-wrap items-center gap-x-5 gap-y-2 px-4 py-3 text-left"
      >
        <span className="min-w-[110px] font-mono text-xs font-medium text-slate-700 dark:text-slate-200">
          {call.group_id}
        </span>
        <span className="text-xs text-slate-400">
          {call.forecast_model ?? '—'} · {PROVIDER_LABEL[call.provider] ?? call.provider}
          {call.deployment ? ` · ${call.deployment}` : ''}
        </span>

        <span className="ml-auto flex flex-wrap items-center gap-1.5">
          {usedFallbackProvider && <Badge status="Warning">fallback provider</Badge>}
          {usedTemplate && <Badge status="neutral">template</Badge>}
          {call.retry_count > 0 && <Badge status="Warning">{call.retry_count} retr{call.retry_count === 1 ? 'y' : 'ies'}</Badge>}
          <Badge status={VALIDATION_TONE[call.validation_status] ?? 'neutral'}>{call.validation_status}</Badge>
          <Badge status={GROUNDING_TONE[call.grounding_status] ?? 'neutral'}>
            {GROUNDING_LABEL[call.grounding_status] ?? call.grounding_status}
          </Badge>
        </span>

        <span className="hidden shrink-0 items-center gap-3 text-xs tabular-nums text-slate-500 dark:text-slate-400 sm:flex">
          <span>{formatTokens(call.total_tokens)} tok</span>
          <span>{formatLatency(call.latency_ms)}</span>
          <span>{formatCost(call.estimated_cost_usd, call.estimated_cost_usd != null)}</span>
        </span>

        <ChevronDown
          size={15}
          className={cn('shrink-0 text-slate-400 transition-transform duration-200', open && 'rotate-180')}
        />
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            className="overflow-hidden"
          >
            <div className="space-y-4 border-t border-slate-100 px-4 py-4 dark:border-slate-800">
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Field label="Timestamp" value={call.timestamp ? formatISTTime(call.timestamp) : '—'} />
                <Field label="Input tokens" value={formatTokens(call.input_tokens)} />
                <Field label="Output tokens" value={formatTokens(call.output_tokens)} />
                <Field label="Total tokens" value={formatTokens(call.total_tokens)} />
                <Field label="Latency" value={formatLatency(call.latency_ms)} />
                <Field label="Estimated cost" value={formatCost(call.estimated_cost_usd, call.estimated_cost_usd != null)} />
                <Field label="Final status" value={call.final_status} />
              </div>

              {call.error && (
                <div className="rounded-lg border border-rose-200 bg-rose-50/70 px-3 py-2 text-xs text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
                  {call.error}
                </div>
              )}

              {call.concise_summary && (
                <div>
                  <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Generated explanation
                  </p>
                  <p className="text-sm text-slate-700 dark:text-slate-200">{call.concise_summary}</p>
                  {call.confidence != null && (
                    <p className="mt-1 text-xs text-slate-400">Confidence: {call.confidence.toFixed(1)}%</p>
                  )}
                </div>
              )}

              {call.rejection_reasons?.length > 0 && (
                <div>
                  <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Rejection reasons
                  </p>
                  <ul className="list-inside list-disc space-y-0.5 text-xs text-slate-600 dark:text-slate-300">
                    {call.rejection_reasons.map((reason) => (
                      <li key={reason}>{reason}</li>
                    ))}
                  </ul>
                </div>
              )}

              {call.caveats?.length > 0 && (
                <div>
                  <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">Caveats</p>
                  <ul className="list-inside list-disc space-y-0.5 text-xs text-slate-600 dark:text-slate-300">
                    {call.caveats.map((caveat) => (
                      <li key={caveat}>{caveat}</li>
                    ))}
                  </ul>
                </div>
              )}

              {(call.validation_errors?.length > 0 || call.grounding_issues?.length > 0) && (
                <div className="grid gap-3 sm:grid-cols-2">
                  {call.validation_errors?.length > 0 && (
                    <div>
                      <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
                        Validation errors
                      </p>
                      <ul className="list-inside list-disc space-y-0.5 text-xs text-rose-600 dark:text-rose-300">
                        {call.validation_errors.map((e) => (
                          <li key={e}>{e}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {call.grounding_issues?.length > 0 && (
                    <div>
                      <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
                        Grounding issues
                      </p>
                      <ul className="list-inside list-disc space-y-0.5 text-xs text-rose-600 dark:text-rose-300">
                        {call.grounding_issues.map((i) => (
                          <li key={i}>{i}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {call.attempts?.length > 0 && (
                <div>
                  <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Attempts ({call.attempts.length})
                  </p>
                  <div className="overflow-x-auto rounded-lg border border-slate-100 dark:border-slate-800">
                    <table className="w-full text-left text-xs">
                      <thead>
                        <tr className="border-b border-slate-100 bg-slate-50/60 dark:border-slate-800 dark:bg-slate-800/40">
                          <th className="px-3 py-1.5 font-semibold text-slate-500">#</th>
                          <th className="px-3 py-1.5 font-semibold text-slate-500">Status</th>
                          <th className="px-3 py-1.5 font-semibold text-slate-500">Deployment</th>
                          <th className="px-3 py-1.5 font-semibold text-slate-500">Tokens</th>
                          <th className="px-3 py-1.5 font-semibold text-slate-500">Latency</th>
                          <th className="px-3 py-1.5 font-semibold text-slate-500">Error</th>
                        </tr>
                      </thead>
                      <tbody>
                        {call.attempts.map((attempt) => (
                          <tr
                            key={attempt.attempt_number}
                            className="border-b border-slate-50 last:border-0 dark:border-slate-800/60"
                          >
                            <td className="px-3 py-1.5 text-slate-600 dark:text-slate-300">{attempt.attempt_number}</td>
                            <td className="px-3 py-1.5 text-slate-600 dark:text-slate-300">{attempt.final_status}</td>
                            <td className="px-3 py-1.5 text-slate-600 dark:text-slate-300">{attempt.deployment ?? '—'}</td>
                            <td className="px-3 py-1.5 tabular-nums text-slate-600 dark:text-slate-300">
                              {formatTokens(attempt.total_tokens)}
                            </td>
                            <td className="px-3 py-1.5 tabular-nums text-slate-600 dark:text-slate-300">
                              {formatLatency(attempt.latency_ms)}
                            </td>
                            <td className="max-w-[200px] truncate px-3 py-1.5 text-rose-500" title={attempt.error}>
                              {attempt.error ?? '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
