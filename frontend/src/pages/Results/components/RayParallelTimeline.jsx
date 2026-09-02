import { useMemo, useState } from 'react'
import { ChevronDown, ChevronRight, Cpu, LayoutList, GanttChartSquare } from 'lucide-react'
import { cn } from '../../../utils/cn'

// One lane per Ray worker, one bar per forecast key, positioned by the real
// start/end offsets the stage's own executor recorded (stage.parallelTasks).
//
// Deliberately plain CSS: a handful of lanes of positioned divs is not a
// charting problem, and a chart dependency would be several hundred
// kilobytes for something percentage widths already express exactly.
//
// This is the one view that *proves* key-level parallelism rather than
// asserting it: bars overlapping across different worker lanes is what
// parallel execution looks like, and a single lane with sequential bars is
// what its absence looks like. Neither can be faked by the layout.
//
// v5. Lanes are identified by the Ray worker's own id rather than an
// invented ordinal, each lane carries its own numbers (keys run, busy time,
// share of the window), the time grid is drawn once behind every lane
// instead of repeated inside each, and the table paginates.
//
// The axis spans the execution window rather than the stage: a stage can
// idle most of its wall time before Ray schedules the first task, and
// drawing from zero spent the whole width on that emptiness. The wait is
// reported as its own figure instead.

// Fixed categorical order — never cycled, never assigned by rank. Validated
// with the dataviz palette checker against both surfaces: worst adjacent
// pair Delta-E 23.4 protan / 11.7 tritan, normal-vision 30.6, all six above
// 3:1 contrast. Indigo steps up to 500 in dark mode, where 600 falls to
// 2.84:1 against the card.
const LANE_COLORS = [
  { bar: 'bg-sky-600', dot: 'bg-sky-600' },
  { bar: 'bg-amber-600', dot: 'bg-amber-600' },
  { bar: 'bg-brand-600 dark:bg-brand-500', dot: 'bg-brand-600 dark:bg-brand-500' },
  { bar: 'bg-emerald-600', dot: 'bg-emerald-600' },
  { bar: 'bg-violet-600', dot: 'bg-violet-600' },
  { bar: 'bg-rose-600', dot: 'bg-rose-600' },
]
// A seventh worker and beyond share one neutral slot. Generating a hue past
// the validated order would put two lanes a colourblind reader cannot tell
// apart next to each other.
const OVERFLOW_LANE = { bar: 'bg-slate-500', dot: 'bg-slate-500' }

const laneColor = (index) => LANE_COLORS[index] ?? OVERFLOW_LANE

const LABEL_COL = 132 // px — the fixed left column lanes, grid and axis align to
const TABLE_PAGE_SIZE = 50

// A Ray worker id is 56 hex characters. The leading digits identify it as
// well as a short commit hash identifies a commit; the full value is on the
// element's title and in the table.
const shortHash = (id) => (id ? String(id).slice(0, 10) : '—')

function formatSeconds(value) {
  if (value == null) return '—'
  return `${value.toFixed(value < 10 ? 2 : 1)}s`
}

// A figure and its name. Proportional digits: these are read, not aligned.
function Figure({ value, label, tone = 'default' }) {
  return (
    <div className="min-w-0">
      <p
        className={cn(
          'truncate text-[15px] font-semibold leading-none',
          tone === 'muted' ? 'text-slate-400 dark:text-slate-500' : 'text-slate-800 dark:text-slate-100'
        )}
      >
        {value}
      </p>
      <p className="mt-1 truncate text-[11px] leading-none text-slate-400 dark:text-slate-500">{label}</p>
    </div>
  )
}

function TaskTable({ rows }) {
  const [page, setPage] = useState(0)
  const pages = Math.max(1, Math.ceil(rows.length / TABLE_PAGE_SIZE))
  const start = page * TABLE_PAGE_SIZE
  const visible = rows.slice(start, start + TABLE_PAGE_SIZE)

  return (
    <div>
      {/* Capped and scrollable: a run with hundreds of keys must not push
          everything below it off the page. */}
      <div className="max-h-[420px] overflow-auto rounded-lg border border-slate-200 dark:border-slate-700">
        <table className="w-full min-w-[520px] text-left text-xs">
          <thead className="sticky top-0 z-10 bg-white dark:bg-slate-900">
            <tr className="border-b border-slate-200 text-[11px] uppercase tracking-wide text-slate-400 dark:border-slate-700 dark:text-slate-500">
              <th className="px-3 py-2 font-medium">Key</th>
              <th className="px-3 py-2 font-medium">Worker</th>
              <th className="px-3 py-2 text-right font-medium">Start</th>
              <th className="px-3 py-2 text-right font-medium">End</th>
              <th className="px-3 py-2 text-right font-medium">Duration</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
            {visible.map((row) => (
              <tr key={row.groupId} className="text-slate-600 dark:text-slate-300">
                <td className="px-3 py-1.5">
                  <span className="inline-flex items-center gap-1.5">
                    <span className={cn('h-2 w-2 shrink-0 rounded-full', laneColor(row.laneIndex).dot)} />
                    <span className="truncate font-medium text-slate-700 dark:text-slate-200">{row.groupId}</span>
                  </span>
                </td>
                <td className="px-3 py-1.5">
                  <span className="font-mono text-[11px]" title={row.workerId}>
                    {shortHash(row.workerId)}
                  </span>
                </td>
                {/* tabular-nums here and nowhere else: these align down a column. */}
                <td className="px-3 py-1.5 text-right tabular-nums">{formatSeconds(row.start)}</td>
                <td className="px-3 py-1.5 text-right tabular-nums">{formatSeconds(row.end)}</td>
                <td className="px-3 py-1.5 text-right tabular-nums">{formatSeconds(row.end - row.start)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {pages > 1 && (
        <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400">
          <span className="tabular-nums">
            {start + 1}–{Math.min(start + TABLE_PAGE_SIZE, rows.length)} of {rows.length}
          </span>
          <span className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="rounded-md border border-slate-200 px-2 py-1 font-medium transition enabled:hover:border-brand-300 enabled:hover:text-brand-600 disabled:opacity-40 dark:border-slate-700"
            >
              Previous
            </button>
            <button
              type="button"
              onClick={() => setPage((p) => Math.min(pages - 1, p + 1))}
              disabled={page >= pages - 1}
              className="rounded-md border border-slate-200 px-2 py-1 font-medium transition enabled:hover:border-brand-300 enabled:hover:text-brand-600 disabled:opacity-40 dark:border-slate-700"
            >
              Next
            </button>
          </span>
        </div>
      )}
    </div>
  )
}

// stageLabel: the phase name ("Train Models", ...) this fan-out belongs to.
// parallelTasks: one stage's real telemetry — { executor, total, completed,
// failed, running, maxConcurrent, tasks: [{groupId, status, durationSeconds,
// workerId, nodeId, start, end}] }.
export default function RayParallelTimeline({ stageLabel, parallelTasks }) {
  const [open, setOpen] = useState(false)
  const [view, setView] = useState('timeline')

  const tasks = parallelTasks?.tasks
  const completedTasks = useMemo(
    () => (tasks ?? []).filter((t) => t.status === 'Completed' && t.start != null),
    [tasks]
  )
  const failedTasks = useMemo(() => (tasks ?? []).filter((t) => t.status !== 'Completed'), [tasks])
  const hasTasks = parallelTasks?.executor === 'ray' && Array.isArray(tasks) && tasks.length > 0

  // Sorted by the raw id, so a given worker keeps its lane across renders.
  const workers = useMemo(() => {
    if (!hasTasks) return []
    return [...new Set(completedTasks.map((t) => t.workerId).filter(Boolean))].sort()
  }, [hasTasks, completedTasks])

  const laneOf = useMemo(() => new Map(workers.map((id, index) => [id, index])), [workers])

  // The window work actually happened in.
  const { domainStart, domainEnd, leadIn } = useMemo(() => {
    if (!completedTasks.length) return { domainStart: 0, domainEnd: 0.001, leadIn: 0 }
    const first = Math.min(...completedTasks.map((t) => t.start))
    const last = Math.max(...completedTasks.map((t) => t.end))
    return { domainStart: first, domainEnd: Math.max(last, first + 0.001), leadIn: first }
  }, [completedTasks])

  const domainSpan = Math.max(domainEnd - domainStart, 0.001)

  const peak = useMemo(() => {
    if (!completedTasks.length) return 0
    const edges = completedTasks.flatMap((t) => [
      { at: t.start, delta: 1 },
      { at: t.end, delta: -1 },
    ])
    edges.sort((a, b) => a.at - b.at || b.delta - a.delta)
    let active = 0
    let highest = 0
    for (const edge of edges) {
      active += edge.delta
      highest = Math.max(highest, active)
    }
    return highest
  }, [completedTasks])

  // Per lane: its tasks, how long it was busy, and its share of the window.
  const lanes = useMemo(
    () =>
      workers.map((workerId, laneIndex) => {
        const laneTasks = completedTasks
          .filter((t) => t.workerId === workerId)
          .sort((a, b) => a.start - b.start)
        const busy = laneTasks.reduce((total, t) => total + (t.end - t.start), 0)
        return {
          workerId,
          laneIndex,
          tasks: laneTasks,
          busy,
          utilisation: Math.min(busy / domainSpan, 1),
        }
      }),
    [workers, completedTasks, domainSpan]
  )

  const tableRows = useMemo(
    () =>
      [...completedTasks]
        .sort((a, b) => a.start - b.start)
        .map((task) => ({ ...task, laneIndex: laneOf.get(task.workerId) ?? 0 })),
    [completedTasks, laneOf]
  )

  if (!hasTasks) return null

  const ticks = [0, 0.25, 0.5, 0.75, 1]

  return (
    <div className="mt-3 overflow-hidden rounded-xl border border-slate-200 dark:border-slate-700">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 bg-slate-50/60 px-3.5 py-2.5 text-left transition hover:bg-slate-50 dark:bg-slate-800/40 dark:hover:bg-slate-800/70"
      >
        {open ? (
          <ChevronDown size={14} className="shrink-0 text-slate-400" />
        ) : (
          <ChevronRight size={14} className="shrink-0 text-slate-400" />
        )}
        <Cpu size={13} className="shrink-0 text-brand-600 dark:text-brand-400" />
        <span className="truncate text-xs font-semibold text-slate-700 dark:text-slate-200">{stageLabel}</span>
        <span className="truncate text-xs text-slate-400 dark:text-slate-500">
          {parallelTasks.total} key{parallelTasks.total === 1 ? '' : 's'} · {workers.length} worker
          {workers.length === 1 ? '' : 's'} · peak {peak} in flight
        </span>
      </button>

      {open && (
        <div className="border-t border-slate-100 px-3.5 py-4 dark:border-slate-800">
          <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
            <div className="flex flex-wrap items-end gap-x-8 gap-y-3">
              <Figure value={`${parallelTasks.completed} / ${parallelTasks.total}`} label="Keys completed" />
              <Figure value={peak || '—'} label="Peak in flight" />
              <Figure value={formatSeconds(domainSpan)} label="Execution window" />
              {/* Reported, not drawn: the stage's own set-up time is real,
                  but it is one number, not a chart's worth of blank space. */}
              <Figure value={formatSeconds(leadIn)} label="Before first key" tone="muted" />
            </div>

            <div className="flex items-center gap-0.5 rounded-lg border border-slate-200 p-0.5 dark:border-slate-700">
              {[
                { id: 'timeline', icon: GanttChartSquare, label: 'Timeline' },
                { id: 'table', icon: LayoutList, label: 'Table' },
              ].map((option) => (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => setView(option.id)}
                  aria-pressed={view === option.id}
                  className={cn(
                    'inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-medium transition',
                    view === option.id
                      ? 'bg-brand-50 text-brand-700 dark:bg-brand-900/40 dark:text-brand-300'
                      : 'text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200'
                  )}
                >
                  <option.icon size={12} />
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          {view === 'table' ? (
            <TaskTable rows={tableRows} />
          ) : (
            <div className="relative">
              {/* One time grid behind every lane, not a copy inside each. */}
              <div
                className="pointer-events-none absolute inset-y-0 right-0"
                style={{ left: `${LABEL_COL}px` }}
                aria-hidden="true"
              >
                {ticks.map((tick) => (
                  <span
                    key={tick}
                    className="absolute inset-y-0 w-px bg-slate-200/70 dark:bg-slate-700/50"
                    style={{ left: `${tick * 100}%` }}
                  />
                ))}
              </div>

              <div className="relative space-y-1">
                {lanes.map((lane) => {
                  const color = laneColor(lane.laneIndex)
                  return (
                    <div
                      key={lane.workerId}
                      className="grid items-center gap-3 rounded-lg py-1.5 transition-colors hover:bg-slate-50/70 dark:hover:bg-slate-800/40"
                      style={{ gridTemplateColumns: `${LABEL_COL}px 1fr` }}
                    >
                      <div className="min-w-0 pl-1">
                        <span className="flex items-center gap-1.5" title={`Ray worker ${lane.workerId}`}>
                          <span className={cn('h-2 w-2 shrink-0 rounded-full', color.dot)} />
                          <span className="truncate font-mono text-[11px] font-medium text-slate-700 dark:text-slate-200">
                            {shortHash(lane.workerId)}
                          </span>
                        </span>
                        <span className="mt-0.5 block truncate pl-3.5 text-[10px] tabular-nums text-slate-400 dark:text-slate-500">
                          {lane.tasks.length} {lane.tasks.length === 1 ? 'key' : 'keys'} ·{' '}
                          {formatSeconds(lane.busy)} · {Math.round(lane.utilisation * 100)}%
                        </span>
                      </div>

                      <div className="relative h-8">
                        {lane.tasks.map((task) => {
                          const left = ((task.start - domainStart) / domainSpan) * 100
                          // A key faster than the scale's resolution still
                          // has to be visible, so bars have a floor width.
                          const width = Math.max(((task.end - task.start) / domainSpan) * 100, 0.8)
                          // Only label a bar wide enough to hold the text
                          // whole — a clipped label is worse than none, and
                          // the table view carries every value regardless.
                          const fits = width >= 16 && String(task.groupId).length <= 9

                          return (
                            <div
                              key={task.groupId}
                              // The hit area is the full lane height, so a
                              // sliver-width bar is still reachable.
                              className="group absolute inset-y-0 flex items-center"
                              style={{ left: `${left}%`, width: `${width}%` }}
                            >
                              {/* 2px surface gap between adjacent fills,
                                  drawn as a ring rather than a border. */}
                              <div
                                className={cn(
                                  'flex h-5 w-full items-center overflow-hidden rounded-[4px] px-1.5',
                                  'ring-2 ring-white transition-[filter] group-hover:brightness-110 dark:ring-slate-900',
                                  color.bar
                                )}
                              >
                                {fits && (
                                  <span className="truncate text-[10px] font-medium leading-none text-white">
                                    {task.groupId}
                                  </span>
                                )}
                              </div>

                              <div
                                className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-1.5 hidden w-max max-w-[280px] -translate-x-1/2 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-left shadow-lg group-hover:block dark:border-slate-700 dark:bg-slate-800"
                                role="tooltip"
                              >
                                <p className="truncate text-[11px] font-semibold text-slate-800 dark:text-slate-100">
                                  {task.groupId}
                                </p>
                                <p className="mt-0.5 text-[10px] tabular-nums text-slate-500 dark:text-slate-400">
                                  {formatSeconds(task.start)}–{formatSeconds(task.end)} ·{' '}
                                  {formatSeconds(task.end - task.start)}
                                </p>
                                <p className="mt-0.5 break-all font-mono text-[9px] text-slate-400 dark:text-slate-500">
                                  {lane.workerId}
                                </p>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )
                })}

                <div className="grid gap-3" style={{ gridTemplateColumns: `${LABEL_COL}px 1fr` }}>
                  <span />
                  <div className="relative h-4 text-[10px] tabular-nums text-slate-400 dark:text-slate-500">
                    {ticks.map((tick) => (
                      <span
                        key={tick}
                        className={cn(
                          'absolute top-0',
                          tick === 0 && 'left-0',
                          tick === 1 && 'right-0',
                          tick !== 0 && tick !== 1 && '-translate-x-1/2'
                        )}
                        style={tick !== 0 && tick !== 1 ? { left: `${tick * 100}%` } : undefined}
                      >
                        {formatSeconds(domainStart + tick * domainSpan)}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {failedTasks.length > 0 && (
            <p className="mt-3 text-xs text-rose-600 dark:text-rose-400">
              {failedTasks.length} key{failedTasks.length === 1 ? '' : 's'} failed and{' '}
              {failedTasks.length === 1 ? 'is' : 'are'} not drawn above:{' '}
              {failedTasks.map((t) => t.groupId).join(', ')}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
