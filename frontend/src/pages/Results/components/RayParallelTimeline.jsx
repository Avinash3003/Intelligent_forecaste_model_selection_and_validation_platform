import { useState } from 'react'
import { ChevronDown, ChevronRight, Cpu } from 'lucide-react'
import { cn } from '../../../utils/cn'

// One lane per Ray worker, one bar per forecast key, positioned by the real
// start/end offsets the executor recorded (key_execution.key_spans).
//
// Deliberately plain CSS: seven-ish lanes of absolutely-positioned divs is
// not a charting problem, and a chart dependency would be several hundred
// kilobytes for something percentage widths already express exactly.
//
// This is the one view that *proves* key-level parallelism rather than
// asserting it: bars overlapping across different worker lanes is what
// parallel execution looks like, and a single lane with sequential bars is
// what its absence looks like. Neither can be faked by the layout.

// Distinct-but-calm lane colours, cycled. Identity here is the worker, and
// a worker is arbitrary — so this carries no meaning beyond telling one
// bar apart from its neighbour.
const BAR_TONES = [
  'bg-brand-500/80',
  'bg-emerald-500/80',
  'bg-amber-500/80',
  'bg-sky-500/80',
  'bg-violet-500/80',
  'bg-rose-500/80',
]

function shortId(id) {
  return typeof id === 'string' && id.length > 8 ? id.slice(0, 8) : id
}

function Stat({ label, value }) {
  return (
    <div>
      <p className="text-[11px] uppercase tracking-wide text-slate-400">{label}</p>
      <p className="mt-0.5 text-sm font-medium text-slate-700 dark:text-slate-200">{value}</p>
    </div>
  )
}

export default function RayParallelTimeline({ keyExecution }) {
  const [open, setOpen] = useState(false)

  // Three separate "nothing to show" cases, all legitimate, none an error:
  // no telemetry at all (a run predating it), a sequential run (no
  // parallelism to draw), and a Ray run whose spans are missing.
  if (!keyExecution) return null
  if (keyExecution.executor !== 'ray') return null
  const spans = keyExecution.key_spans
  if (!Array.isArray(spans) || spans.length === 0) return null

  // Lanes are derived from the data, never assumed: the worker count is
  // whatever Ray actually used, which is not necessarily ray_cpus.
  const workers = [...new Set(spans.map((s) => s.worker_id))]
  // The visual scale ends at the last key to finish, not at wall_seconds —
  // wall_seconds includes collection/merge after the final task returns, and
  // stretching the axis to it would render a misleading trailing gap.
  const scaleEnd = Math.max(...spans.map((s) => s.end), 0.001)

  return (
    <div className="mt-3 rounded-lg border border-slate-200 dark:border-slate-700">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs font-medium text-brand-600 hover:text-brand-700 dark:text-brand-400"
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <Cpu size={13} />
        Ray parallel execution — {spans.length} keys across {workers.length}{' '}
        worker{workers.length === 1 ? '' : 's'}
      </button>

      {open && (
        <div className="border-t border-slate-100 px-3 py-3 dark:border-slate-800">
          <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label="Executor" value="Ray" />
            <Stat label="Ray CPUs" value={keyExecution.ray_cpus ?? '—'} />
            <Stat label="Max concurrent keys" value={keyExecution.max_concurrent_keys ?? '—'} />
            <Stat
              label="Wall time"
              value={keyExecution.wall_seconds != null ? `${keyExecution.wall_seconds}s` : '—'}
            />
          </div>

          <div className="space-y-1.5">
            {workers.map((workerId, laneIndex) => {
              const laneSpans = spans.filter((s) => s.worker_id === workerId)
              return (
                <div key={workerId} className="flex items-center gap-2">
                  <span
                    className="w-24 shrink-0 truncate font-mono text-[10px] text-slate-400"
                    title={workerId}
                  >
                    {shortId(workerId)}
                  </span>
                  <div className="relative h-6 flex-1 rounded bg-slate-100 dark:bg-slate-800">
                    {laneSpans.map((span) => {
                      const left = (span.start / scaleEnd) * 100
                      // A key faster than the scale's resolution still has to
                      // be visible, so bars have a floor width.
                      const width = Math.max(((span.end - span.start) / scaleEnd) * 100, 0.75)
                      return (
                        <div
                          key={span.group_id}
                          className={cn(
                            'absolute top-1 flex h-4 items-center overflow-hidden rounded-sm px-1',
                            BAR_TONES[laneIndex % BAR_TONES.length]
                          )}
                          style={{ left: `${left}%`, width: `${width}%` }}
                          title={`${span.group_id} — ${span.start}s to ${span.end}s (${(span.end - span.start).toFixed(2)}s)`}
                        >
                          <span className="truncate text-[9px] font-medium text-white">
                            {span.group_id}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>

          <div className="mt-2 flex justify-between pl-26 text-[10px] text-slate-400">
            <span>0s</span>
            <span>{scaleEnd.toFixed(1)}s</span>
          </div>

          {keyExecution.keys_failed?.length > 0 && (
            <p className="mt-3 text-xs text-rose-600 dark:text-rose-400">
              {keyExecution.keys_failed.length} key
              {keyExecution.keys_failed.length === 1 ? '' : 's'} failed and{' '}
              {keyExecution.keys_failed.length === 1 ? 'is' : 'are'} not drawn above:{' '}
              {keyExecution.keys_failed.join(', ')}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
