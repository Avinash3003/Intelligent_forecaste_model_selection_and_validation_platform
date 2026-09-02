import { useMemo, useState } from 'react'
import { ChevronDown, ChevronRight, Cpu, LayoutList, GanttChartSquare } from 'lucide-react'
import { cn } from '../../../utils/cn'

// One lane per Ray worker, one bar per forecast key, positioned by the real
// start/end offsets the stage's own executor recorded (stage.parallelTasks).
//
// Deliberately plain CSS/SVG: a handful of lanes of positioned divs is not a
// charting problem, and a chart dependency would be several hundred
// kilobytes for something percentage widths already express exactly.
//
// This is the one view that *proves* key-level parallelism rather than
// asserting it: bars overlapping across different worker lanes is what
// parallel execution looks like, and a single lane with sequential bars is
// what its absence looks like. Neither can be faked by the layout.
//
// v4 rebuild. What the previous version got wrong, and what replaced it:
//
//   * The x-axis ran from the stage's start, but a stage can spend almost
//     all of its wall time before the first task is scheduled — Rank &
//     Select spent 8.7s of 8.89s that way. Every bar collapsed into a
//     sliver against ~97% empty width. The axis now spans the execution
//     window (first start -> last end) with both ends labelled in real
//     seconds, and the time before it is reported as its own figure rather
//     than drawn as dead pixels. No broken axis, nothing hidden.
//   * Hues were cycled by lane index. They are now a fixed order, and a
//     seventh worker folds into one shared slot rather than reusing a hue
//     another lane already owns.
//   * Bar labels were clipped by `overflow-hidden` inside 6%-wide bars.
//     A label is rendered only when the bar can hold it whole.
//   * Every timing was reachable only by hovering. There is now a table
//     view carrying the same numbers.
//   * A concurrency band across the top: how many keys were in flight at
//     each moment, which is the claim this component exists to support.

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

const LABEL_COL = 92 // px — the fixed left column lanes, band and axis align to

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
          tone === 'muted'
            ? 'text-slate-400 dark:text-slate-500'
            : 'text-slate-800 dark:text-slate-100'
        )}
      >
        {value}
      </p>
      <p className="mt-1 truncate text-[11px] leading-none text-slate-400 dark:text-slate-500">{label}</p>
    </div>
  )
}

// Active-task count over the execution window, as a step area. This is the
// parallelism claim in one shape: a flat band at 1 is sequential work, a
// band that climbs to N is N keys genuinely in flight.
function ConcurrencyBand({ tasks, domainStart, domainSpan, peak }) {
  const points = useMemo(() => {
    const edges = []
    for (const task of tasks) {
      edges.push({ at: task.start, delta: 1 })
      edges.push({ at: task.end, delta: -1 })
    }
    edges.sort((a, b) => a.at - b.at || b.delta - a.delta)

    const steps = []
    let active = 0
    for (const edge of edges) {
      active += edge.delta
      const x = ((edge.at - domainStart) / domainSpan) * 100
      steps.push({ x: Math.min(Math.max(x, 0), 100), active })
    }
    return steps
  }, [tasks, domainStart, domainSpan])

  if (!points.length || peak < 1) return null

  // A step path: hold each level until the next edge, then jump.
  const height = 28
  const y = (active) => height - (active / peak) * (height - 3)
  let d = `M 0 ${height} L 0 ${y(0)}`
  let previous = 0
  for (const point of points) {
    d += ` L ${point.x} ${y(previous)} L ${point.x} ${y(point.active)}`
    previous = point.active
  }
  d += ` L 100 ${y(previous)} L 100 ${height} Z`

  return (
    <div className="grid items-center gap-3" style={{ gridTemplateColumns: `${LABEL_COL}px 1fr` }}>
      <span className="truncate text-[11px] text-slate-400 dark:text-slate-500">In flight</span>
      <div className="relative">
        <svg
          viewBox={`0 0 100 ${height}`}
          preserveAspectRatio="none"
          className="h-7 w-full overflow-visible"
          role="img"
          aria-label={`Peak ${peak} keys running at once`}
        >
          <path d={d} className="fill-brand-500/15" />
          <path
            d={d}
            className="stroke-brand-500/70"
            fill="none"
            strokeWidth="1.5"
            vectorEffect="non-scaling-stroke"
          />
        </svg>
        <span className="pointer-events-none absolute right-0 top-0 text-[10px] font-medium text-brand-600 dark:text-brand-400">
          peak {peak}
        </span>
      </div>
    </div>
  )
}

function TaskTable({ rows }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[420px] text-left text-xs">
        <thead>
          <tr className="border-b border-slate-200 text-[11px] uppercase tracking-wide text-slate-400 dark:border-slate-700 dark:text-slate-500">
            <th className="py-1.5 pr-3 font-medium">Key</th>
            <th className="py-1.5 pr-3 font-medium">Worker</th>
            <th className="py-1.5 pr-3 text-right font-medium">Start</th>
            <th className="py-1.5 pr-3 text-right font-medium">End</th>
            <th className="py-1.5 text-right font-medium">Duration</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
          {rows.map((row) => (
            <tr key={row.groupId} className="text-slate-600 dark:text-slate-300">
              <td className="py-1.5 pr-3">
                <span className="inline-flex items-center gap-1.5">
                  <span className={cn('h-2 w-2 shrink-0 rounded-full', laneColor(row.laneIndex).dot)} />
                  <span className="truncate font-medium text-slate-700 dark:text-slate-200">{row.groupId}</span>
                </span>
              </td>
              <td className="py-1.5 pr-3">Worker {row.laneIndex + 1}</td>
              {/* tabular-nums here and nowhere else: these align down a column. */}
              <td className="py-1.5 pr-3 text-right tabular-nums">{formatSeconds(row.start)}</td>
              <td className="py-1.5 pr-3 text-right tabular-nums">{formatSeconds(row.end)}</td>
              <td className="py-1.5 text-right tabular-nums">{formatSeconds(row.end - row.start)}</td>
            </tr>
          ))}
        </tbody>
      </table>
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

  // Stable ordinal numbering: sorted by the raw id rather than order of
  // appearance, so "Worker 1" refers to the same lane on every render.
  const workers = useMemo(() => {
    if (!hasTasks) return []
    return [...new Set(completedTasks.map((t) => t.workerId).filter(Boolean))].sort()
  }, [hasTasks, completedTasks])

  const laneOf = useMemo(() => new Map(workers.map((id, index) => [id, index])), [workers])

  // The window work actually happened in. A stage can idle for most of its
  // wall time before Ray schedules the first task; drawing from zero spends
  // the whole width on that emptiness. Both ends are labelled with real
  // seconds, so this reads as a window, never as "the stage started here".
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
          <div className="mb-4 flex flex-wrap items-end justify-between gap-4">
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
            <>
              <div className="space-y-2">
                <ConcurrencyBand
                  tasks={completedTasks}
                  domainStart={domainStart}
                  domainSpan={domainSpan}
                  peak={peak}
                />

                {workers.map((workerId, laneIndex) => {
                  const laneTasks = completedTasks
                    .filter((t) => t.workerId === workerId)
                    .sort((a, b) => a.start - b.start)
                  const color = laneColor(laneIndex)

                  return (
                    <div
                      key={workerId}
                      className="grid items-center gap-3"
                      style={{ gridTemplateColumns: `${LABEL_COL}px 1fr` }}
                    >
                      <span className="flex min-w-0 items-center gap-1.5" title={`Ray worker id ${workerId}`}>
                        {/* Identity is never colour alone: the swatch names
                            the lane the bars below belong to. */}
                        <span className={cn('h-2 w-2 shrink-0 rounded-full', color.dot)} />
                        <span className="truncate text-xs font-medium text-slate-600 dark:text-slate-300">
                          Worker {laneIndex + 1}
                        </span>
                      </span>

                      <div className="relative h-7 rounded-md bg-slate-100/70 dark:bg-slate-800/60">
                        {/* Recessive solid hairlines, one shade off the
                            surface — never dashed. */}
                        {ticks.map((tick) => (
                          <span
                            key={tick}
                            className="pointer-events-none absolute inset-y-0 w-px bg-slate-200/70 dark:bg-slate-700/50"
                            style={{ left: `${tick * 100}%` }}
                          />
                        ))}

                        {laneTasks.map((task) => {
                          const left = ((task.start - domainStart) / domainSpan) * 100
                          // A key faster than the scale's resolution still
                          // has to be visible, so bars have a floor width.
                          const width = Math.max(((task.end - task.start) / domainSpan) * 100, 0.8)
                          // Only label a bar wide enough to hold the text
                          // whole — a clipped label is worse than none, and
                          // the table view carries every value regardless.
                          const fits = width >= 14 && String(task.groupId).length <= 8

                          return (
                            <div
                              key={task.groupId}
                              // The hit area is the full lane height, so a
                              // sliver-width bar is still reachable; `group`
                              // scopes the tooltip by CSS alone.
                              className="group absolute inset-y-0 flex items-center"
                              style={{ left: `${left}%`, width: `${width}%` }}
                            >
                              {/* 2px surface gap between adjacent fills,
                                  drawn as inset margin rather than a border
                                  around the mark. */}
                              <div
                                className={cn(
                                  'h-3.5 w-full rounded-[4px] px-1.5',
                                  'ring-2 ring-white transition-[filter] group-hover:brightness-110 dark:ring-slate-900',
                                  color.bar
                                )}
                              >
                                {fits && (
                                  <span className="block truncate text-[9px] font-medium leading-[14px] text-white">
                                    {task.groupId}
                                  </span>
                                )}
                              </div>

                              <div
                                className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-1.5 hidden w-max max-w-[260px] -translate-x-1/2 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-left shadow-lg group-hover:block dark:border-slate-700 dark:bg-slate-800"
                                role="tooltip"
                              >
                                <p className="truncate text-[11px] font-semibold text-slate-800 dark:text-slate-100">
                                  {task.groupId}
                                </p>
                                <p className="mt-0.5 text-[10px] text-slate-500 dark:text-slate-400">
                                  Worker {laneIndex + 1} · {formatSeconds(task.start)}–{formatSeconds(task.end)} ·{' '}
                                  {formatSeconds(task.end - task.start)}
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
            </>
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
